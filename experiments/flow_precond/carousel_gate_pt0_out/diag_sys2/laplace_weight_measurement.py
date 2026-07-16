"""MEASUREMENT (sampling-free): predicted Laplace-equilibrium weight of system-2's
two posterior modes (compact vs sharp, split on z[37] = planes/3/light/1/beta).

Plain (archive-matched, non-adaptive) renderer, per the pinned protocol:
    S2.AdaptiveImageData = ImageData   (BEFORE build())

Method:
  1. Reproduce the established best-of-3000-subsample representative in each pool
     (same rng(0) draw order as diag_sys2/map_vs_modes_lp.py -- sanity-checked
     against the previously-reported peaks: sharp max=-521043.07,
     compact max=-520956.07).
  2. Locally optimize (L-BFGS-B via scipy, jax grad) from each representative;
     track the z[37] trajectory and report whether it crosses THR.
  3. jax.hessian (float64) of lp at each converged mode; eigen-spectrum,
     negative-definiteness check.
  4. Laplace log-evidence-ratio: peak-gap term + volume term (slogdet), ->
     predicted w_compact.
  5. Laplace sd along z[37] at each mode (sanity check on the "compact is
     broader" 1-D impression) + total log-volume per mode.
  6. Plot: sorted log|eigenvalue| spectra, both modes, shared axes.

Writes:
  diag_sys2/laplace_weight_results.json  (all numbers)
  diag_sys2/laplace_weight_run.log       (full stdout, same as console)
  diag_sys2/laplace_eigenspectra.png
"""
import json
import sys
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import carousel_model_s2 as S2
from gigalens.jax.scene_prob_model import ImageData

OUT_DIR = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out/diag_sys2"
THR = -3.0920
J37 = 37


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()
    log("=" * 78)
    log("SYSTEM-2 LAPLACE WEIGHT MEASUREMENT (sampling-free)")
    log("=" * 78)

    # ---- build plain (archive-matched) renderer ---------------------------------
    S2.AdaptiveImageData = ImageData
    _, pm = S2.build()
    renderer = type(pm.datasets[0]).__name__
    assert renderer == "ImageData", renderer
    log(f"renderer                 = {renderer}")

    names = list(pm.z_param_names)
    log(f"z_param_names[{J37}]         = {names[J37]!r}")
    assert names[J37] == "planes/3/light/1/beta", names[J37]
    log(f"THR (2-means boundary)   = {THR}")

    # ---- load MCLMC archive (read-only) ------------------------------------------
    Z = np.load(S2.MCLMC_NPZ_S2)["samples_z"].reshape(-1, 46)
    log(f"samples_z flat shape     = {Z.shape}")

    def lp_of(A, chunk=64):
        out = []
        for i in range(0, len(A), chunk):
            v, _ = pm.log_prob(jnp.asarray(A[i:i + chunk]))
            out.append(np.asarray(v, dtype=np.float64))
        return np.concatenate(out)

    # ---- reproduce established best-of-3000 representatives ---------------------
    # SAME rng object, SAME loop order (sharp then compact) as
    # diag_sys2/map_vs_modes_lp.py, to exactly reproduce the previously-reported
    # peaks as a sanity check before doing anything new.
    rng = np.random.default_rng(0)
    reps = {}
    log("\n--- reproducing established best-of-3000-subsample representatives ---")
    for tag, pool in (("sharp", Z[Z[:, J37] > THR]), ("compact", Z[Z[:, J37] < THR])):
        sub = pool[rng.choice(len(pool), 3000, replace=False)]
        lp = lp_of(sub)
        i_best = int(np.argmax(lp))
        z0 = sub[i_best].astype(np.float64)
        reps[tag] = dict(z0=z0, lp0=float(lp[i_best]), n_pool=int(len(pool)))
        log(f"{tag:8s}: n_pool={len(pool):6d}  lp0(best-of-3000)={lp[i_best]:.3f}  "
            f"z0[37]={z0[J37]:+.4f}")

    log(f"\ncompact_best0 - sharp_best0 = "
        f"{reps['compact']['lp0'] - reps['sharp']['lp0']:+.2f} nats "
        "(should reproduce the reported +87.00 nats)")

    # ---- differentiable scalar objective ------------------------------------------
    @jax.jit
    def neg_lp_and_grad(z):
        def f(z):
            lp, _ = pm.log_prob(z)
            return -lp
        return jax.value_and_grad(f)(z)

    @jax.jit
    def lp_scalar(z):
        lp, _ = pm.log_prob(z)
        return lp

    def scipy_fun(z_np):
        val, grad = neg_lp_and_grad(jnp.asarray(z_np, dtype=jnp.float64))
        return float(val), np.asarray(grad, dtype=np.float64)

    grad_neg_lp = jax.jit(jax.grad(lambda z: -lp_scalar(z)))
    grad_lp = jax.jit(jax.grad(lp_scalar))

    @jax.jit
    def hvp_neg(z, p):
        _, hv = jax.jvp(grad_neg_lp, (z,), (p,))
        return hv

    @jax.jit
    def hvp_lp(z, v):
        _, hv = jax.jvp(grad_lp, (z,), (v,))
        return hv

    def hessian_by_columns(z_jnp, dim=46):
        # NOTE: jax.hessian(...) (= jacfwd(jacrev(f))) vmaps the forward-mode sweep
        # over all 46 standard-basis tangents at once, which OOM'd the GPU on this
        # model (tried twice, with two different XLA-side workarounds -- disabling
        # CUDA command buffers and lowering XLA_CLIENT_MEM_FRACTION -- both still
        # OOM'd on the actual fused kernel, i.e. a real memory ceiling from the
        # batched sweep, not a command-buffer/driver quirk: batching 46 tangents
        # multiplies the per-tangent intermediate activations (300x300x4-band
        # renders) by 46x simultaneously). Fix: build H one column at a time via
        # grad-then-jvp (exact Hessian-vector product, same math as jax.hessian,
        # just not vmapped) -- O(46x) more dispatches but O(1/46x) peak memory.
        cols = []
        for i in range(dim):
            e_i = jnp.zeros(dim, dtype=jnp.float64).at[i].set(1.0)
            cols.append(np.asarray(hvp_lp(z_jnp, e_i), dtype=np.float64))
        return np.stack(cols, axis=1)  # column i = H @ e_i

    def scipy_hessp(z_np, p_np):
        z = jnp.asarray(z_np, dtype=jnp.float64)
        p = jnp.asarray(p_np, dtype=jnp.float64)
        return np.asarray(hvp_neg(z, p), dtype=np.float64)

    # ---- local optimization within each basin -------------------------------------
    # Stage 1: L-BFGS-B (cheap, gets near the basin). Stage 2: trust-ncg with an
    # EXACT Hessian-vector product (jax jvp-of-grad, no full Hessian materialized)
    # for fast local (quasi-Newton -> Newton) convergence on the curved/
    # ill-conditioned directions this system is known to have (docs/logs
    # carousel-mclmc-sampling.md C-5).
    log("\n--- local optimization: stage 1 L-BFGS-B, stage 2 trust-ncg (exact hessp) ---")
    modes = {}
    for tag in ("sharp", "compact"):
        z0 = reps[tag]["z0"]
        side0 = z0[J37] > THR  # True == sharp side
        path_z37 = [float(z0[J37])]

        def callback(xk, _path=path_z37):
            _path.append(float(xk[J37]))

        res1 = minimize(scipy_fun, z0, jac=True, method="L-BFGS-B",
                         callback=callback,
                         options=dict(maxiter=2000, ftol=1e-15, gtol=1e-10))
        log(f"{tag:8s} [stage1 L-BFGS-B]: converged={res1.success}  nit={res1.nit}  "
            f"|grad|_inf={np.max(np.abs(res1.jac)):.3e}  lp={-res1.fun:.4f}  "
            f"msg={res1.message!r}")

        res2 = minimize(scipy_fun, res1.x, jac=True, method="trust-ncg",
                         hessp=scipy_hessp, callback=callback,
                         options=dict(maxiter=2000, gtol=1e-10))
        log(f"{tag:8s} [stage2 trust-ncg]: raw scipy message={res2.message!r}")
        z_star = np.asarray(res2.x, dtype=np.float64)
        lp_star = float(lp_scalar(jnp.asarray(z_star)))
        grad_inf = float(np.max(np.abs(np.asarray(
            neg_lp_and_grad(jnp.asarray(z_star, dtype=jnp.float64))[1]))))

        crossed = any((v > THR) != side0 for v in path_z37)
        log(f"{tag:8s} [stage2 trust-ncg]: converged={res2.success}  nit={res2.nit}  "
            f"|grad|_inf={grad_inf:.3e}")
        log(f"          lp0={reps[tag]['lp0']:.4f} -> lp*={lp_star:.4f}  "
            f"(delta={lp_star - reps[tag]['lp0']:+.4f})")
        log(f"          z[37]: z0={z0[J37]:+.4f} -> z*={z_star[J37]:+.4f}   "
            f"path min/max=({min(path_z37):+.4f},{max(path_z37):+.4f})  "
            f"CROSSED_THR={crossed}")
        if grad_inf > 1e-2:
            log(f"          *** FINDING: |grad|_inf={grad_inf:.3e} at the reported "
                f"'converged' point for '{tag}' is NOT small -- optimization has "
                "NOT reached a stationary point. Treat lp*/z* and any downstream "
                "Hessian at this point as an approximate, not exact, mode "
                "location. ***")
        if crossed:
            log(f"          *** FINDING: optimization path for '{tag}' crossed "
                f"THR={THR} -- the basin is NOT separated at z[37] along this "
                "trajectory. Laplace estimate at this mode is suspect. ***")

        # ---- stage 3: damped Newton using the EXACT Hessian --------------------
        # Both quasi-Newton (L-BFGS) and trust-region-CG stalled at |grad|~26-29
        # (a genuinely different failure than stage1/2's own internal issues --
        # both independent optimizer families plateaued at a similar residual,
        # which is itself evidence of a real ill-conditioned landscape, not an
        # optimizer bug). This is not "try a third knob": it uses the EXACT
        # Hessian (already needed for the Laplace estimate) directly via a
        # classical damped-Newton step with backtracking, which stages 1/2 only
        # approximate (L-BFGS: low-rank; trust-ncg: truncated CG on the exact
        # hessp). If this also stalls at a similar gradient residual, that is a
        # strong, differently-derived confirmation that the point is genuinely
        # hard to converge further with any local 2nd-order method.
        log(f"{tag:8s} [stage3 damped-Newton, exact H]:")
        for newton_it in range(8):
            H_it = hessian_by_columns(jnp.asarray(z_star, dtype=jnp.float64))
            Hsym_it = 0.5 * (H_it + H_it.T)
            g_it = np.asarray(grad_lp(jnp.asarray(z_star, dtype=jnp.float64)), dtype=np.float64)
            gnorm_it = float(np.max(np.abs(g_it)))
            lp_it = float(lp_scalar(jnp.asarray(z_star, dtype=jnp.float64)))
            log(f"          it={newton_it}  lp={lp_it:.4f}  |grad|_inf={gnorm_it:.3e}  "
                f"z[37]={z_star[J37]:+.4f}")
            if gnorm_it < 1e-3:
                log("          Newton: gradient below 1e-3 -- treating as converged.")
                break
            try:
                d_it = np.linalg.solve(Hsym_it, -g_it)
            except np.linalg.LinAlgError:
                log("          Newton: Hessian solve failed (singular) -- stopping.")
                break
            step_scale = 1.0
            accepted = False
            for _bt in range(15):
                z_try = z_star + step_scale * d_it
                lp_try = float(lp_scalar(jnp.asarray(z_try, dtype=jnp.float64)))
                if np.isfinite(lp_try) and lp_try > lp_it:
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                log("          Newton: no improving step found along the Newton "
                    f"direction after {_bt + 1} halvings -- stopping "
                    "(this IS informative: the exact-Hessian quadratic model "
                    "does not predict an improving direction here either).")
                break
            path_z37.append(float(z_try[J37]))
            z_star = z_try
        lp_star = float(lp_scalar(jnp.asarray(z_star, dtype=jnp.float64)))
        grad_inf = float(np.max(np.abs(np.asarray(
            grad_lp(jnp.asarray(z_star, dtype=jnp.float64)), dtype=np.float64))))
        crossed = any((v > THR) != side0 for v in path_z37)
        log(f"          [stage3 FINAL] lp*={lp_star:.4f}  |grad|_inf={grad_inf:.3e}  "
            f"z[37]*={z_star[J37]:+.4f}  CROSSED_THR={crossed}")
        if grad_inf > 1e-2:
            log(f"          *** FINDING (post-Newton): |grad|_inf={grad_inf:.3e} "
                f"still NOT small at '{tag}' -- the point is NOT a true stationary "
                "point even after an exact-Hessian damped-Newton refinement. The "
                "Laplace estimate below is computed AT THIS POINT ANYWAY (per "
                "instructions, to report raw numbers), but should be read as an "
                "approximate/lower-confidence estimate, not an exact result. ***")
        if crossed:
            log(f"          *** FINDING (post-Newton): the refined path for "
                f"'{tag}' crossed THR={THR}. ***")

        modes[tag] = dict(z0=z0.tolist(), z_star=z_star.tolist(),
                           lp0=reps[tag]["lp0"], lp_star=lp_star,
                           converged=bool(res2.success),
                           nit_stage1=int(res1.nit), nit_stage2=int(res2.nit),
                           grad_inf_norm=grad_inf,
                           path_z37=path_z37, crossed_thr=bool(crossed))

    # ---- Hessian at each converged (post-Newton) mode ------------------------------
    # Uses hessian_by_columns (defined above, alongside the stage-3 Newton refinement)
    # -- an exact Hessian-vector-product sweep done ONE COLUMN AT A TIME rather than
    # via jax.hessian's vmapped forward-over-reverse sweep, which OOM'd the shared
    # GPU on this model (tried twice, with two different XLA-side workarounds --
    # disabling CUDA command buffers and lowering XLA_CLIENT_MEM_FRACTION -- both
    # still OOM'd on the actual fused kernel: batching all 46 tangents multiplies the
    # per-tangent intermediate activations, 300x300x4-band renders, by 46x at once).
    log("\n--- Hessian via sequential exact Hessian-vector products, float64 ---")
    for tag in ("sharp", "compact"):
        z_star = jnp.asarray(modes[tag]["z_star"], dtype=jnp.float64)
        t_h = time.time()
        H = hessian_by_columns(z_star)
        log(f"{tag:8s}: Hessian computed in {time.time() - t_h:.1f}s, shape={H.shape}")

        asym = np.max(np.abs(H - H.T))
        log(f"          max|H - H^T| (asymmetry) = {asym:.3e}")
        Hsym = 0.5 * (H + H.T)

        eigvals = np.linalg.eigvalsh(Hsym)
        n_nonneg = int(np.sum(eigvals >= 0))
        log(f"          eigenvalues: min={eigvals.min():.6e}  max={eigvals.max():.6e}")
        log(f"          eigenvalue count >= 0 (should be 0 at a true max) = "
            f"{n_nonneg} / {len(eigvals)}")
        if n_nonneg > 0:
            log(f"          *** FINDING: Hessian at '{tag}' is NOT negative-definite "
                f"({n_nonneg} non-negative eigenvalue(s)) -- this point is a saddle "
                "or the optimizer has not converged to a true maximum; the Laplace "
                "estimate at this mode is INVALID as stated. ***")

        neg_H = -Hsym
        sign, logdet = np.linalg.slogdet(neg_H)
        log(f"          slogdet(-H): sign={sign}  logdet={logdet:.4f}")
        cond = np.linalg.cond(neg_H)
        log(f"          cond(-H) = {cond:.3e}"
            + ("   *** ill-conditioned (>1e10) ***" if cond > 1e10 else ""))

        # sd along z[37] via inverse Hessian
        try:
            invH = np.linalg.inv(neg_H)
            sd37 = float(np.sqrt(invH[J37, J37])) if invH[J37, J37] > 0 else float("nan")
        except np.linalg.LinAlgError:
            invH = None
            sd37 = float("nan")
        log(f"          Laplace sd along z[37] = {sd37!r}")

        d = H.shape[0]
        log_vol = 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet
        log(f"          per-mode log-volume [d/2 log(2pi) - 0.5 logdet(-H)] = "
            f"{log_vol:.4f}")

        modes[tag].update(dict(
            H_asymmetry=float(asym),
            eigvals=eigvals.tolist(),
            n_nonneg_eigvals=n_nonneg,
            slogdet_sign=float(sign),
            logdet_negH=float(logdet),
            cond_negH=float(cond),
            sd_z37=sd37,
            log_volume=float(log_vol),
        ))

    # ---- Laplace weight prediction --------------------------------------------------
    log("\n--- Laplace weight prediction ---")
    peak_gap = modes["compact"]["lp_star"] - modes["sharp"]["lp_star"]
    vol_term = 0.5 * (modes["sharp"]["logdet_negH"] - modes["compact"]["logdet_negH"])
    log_ratio = peak_gap + vol_term
    w_compact = 1.0 / (1.0 + np.exp(-log_ratio))
    log(f"peak-gap term   [lp_compact* - lp_sharp*]                = {peak_gap:+.4f} nats")
    log(f"volume term     [0.5*(logdet(-H_sharp)-logdet(-H_compact))] = {vol_term:+.4f} nats")
    log(f"total log(w_compact/w_sharp)                              = {log_ratio:+.4f} nats")
    log(f"predicted w_compact = 1/(1+exp(-log_ratio))                = {w_compact:.6g}")
    log(f"predicted w_sharp                                          = {1 - w_compact:.6g}")
    log(f"\n(sampled/stopwatch weight for comparison: w_compact=0.2401, w_sharp=0.7599)")

    both_negdef = (modes["sharp"]["n_nonneg_eigvals"] == 0
                   and modes["compact"]["n_nonneg_eigvals"] == 0)
    any_crossed = modes["sharp"]["crossed_thr"] or modes["compact"]["crossed_thr"]
    log(f"\nboth Hessians negative-definite = {both_negdef}")
    log(f"any optimization path crossed THR = {any_crossed}")

    if not both_negdef or any_crossed:
        log("\n*** VALIDITY WARNING: at least one precondition for the Laplace "
            "estimate above is violated (see FINDING(s) above). Treat the "
            "log_ratio/weight number as UNRELIABLE, not as a certified result. ***")

    log("\n--- 'compact is broader' sanity check (z[37] sd + total log-volume) ---")
    log(f"sharp   : sd_z37={modes['sharp']['sd_z37']:.4f}   log_volume={modes['sharp']['log_volume']:.4f}")
    log(f"compact : sd_z37={modes['compact']['sd_z37']:.4f}   log_volume={modes['compact']['log_volume']:.4f}")
    if modes["compact"]["sd_z37"] > modes["sharp"]["sd_z37"]:
        log("=> along z[37], 'compact' IS the wider mode -- consistent with the "
            "step-0 1-D impression.")
    else:
        log("=> along z[37], 'compact' is NOT the wider mode -- the step-0 1-D "
            "impression is CONTRADICTED by the local Gaussian approx.")
    if modes["compact"]["log_volume"] > modes["sharp"]["log_volume"]:
        log("=> in TOTAL 46-d log-volume, 'compact' is the larger-volume basin.")
    else:
        log("=> in TOTAL 46-d log-volume, 'sharp' is the larger-volume basin "
            "(volume in the other 45 dims can reverse a 1-D impression).")

    # ---- caveats ---------------------------------------------------------------------
    max_cond = max(modes["sharp"]["cond_negH"], modes["compact"]["cond_negH"])
    for tag in ("sharp", "compact"):
        ev_abs = np.abs(np.array(modes[tag]["eigvals"]))
        ev_abs_nz = ev_abs[ev_abs > 0]
        span = float(ev_abs_nz.max() / ev_abs_nz.min()) if len(ev_abs_nz) else float("nan")
        log(f"eigenvalue |lambda| dynamic range ({tag}) = max/min = {span:.3e} "
            f"({np.log10(span):.1f} orders of magnitude)" if span == span else
            f"eigenvalue |lambda| dynamic range ({tag}) = undefined (all-zero spectrum)")
    log(f"\nCAVEAT: max cond(-H) across modes = {max_cond:.3e}. Laplace is a local "
        "Gaussian approx; with 164 profiled lstsq amplitudes the profile "
        "likelihood may not be smooth, and a wide/ill-conditioned spectrum means "
        "the volume term above should be read as an order-of-magnitude statement, "
        "not a precise number.")

    # ---- plot: sorted log|eigenvalue| spectra ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tag, color in (("sharp", "tab:blue"), ("compact", "tab:orange")):
        ev = np.array(modes[tag]["eigvals"])
        log_abs_sorted = np.sort(np.log10(np.abs(ev) + 1e-300))[::-1]
        n_pos = int(np.sum(ev < 0))
        ax.plot(np.arange(len(ev)), log_abs_sorted, "o-", ms=3, color=color,
                label=f"{tag} (n_neg={n_pos}/{len(ev)})")
    ax.set_xlabel("eigenvalue rank (sorted desc. by |lambda|)")
    ax.set_ylabel(r"$\log_{10}|\lambda|$")
    ax.set_title("Hessian eigenvalue spectra at each converged mode\n"
                  "(system-2, plain renderer; all should be negative at a max)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = f"{OUT_DIR}/laplace_eigenspectra.png"
    fig.savefig(plot_path, dpi=130)
    log(f"\nplot saved: {plot_path}")

    # ---- save everything ----------------------------------------------------------
    results = dict(
        renderer=renderer,
        z37_name=names[J37],
        THR=THR,
        peak_gap_nats=float(peak_gap),
        volume_term_nats=float(vol_term),
        log_ratio_nats=float(log_ratio),
        predicted_w_compact=float(w_compact),
        predicted_w_sharp=float(1 - w_compact),
        sampled_w_compact=0.2401,
        sampled_w_sharp=0.7599,
        both_hessians_negdef=bool(both_negdef),
        any_optimization_crossed_thr=bool(any_crossed),
        modes={k: {kk: vv for kk, vv in v.items() if kk not in ("eigvals",)}
               for k, v in modes.items()},
        eigvals={k: v["eigvals"] for k, v in modes.items()},
        wall_time_s=time.time() - t0,
    )
    json_path = f"{OUT_DIR}/laplace_weight_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"results json saved: {json_path}")
    log(f"\nTOTAL WALL TIME: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
