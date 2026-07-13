"""D4 / D5 synthetic-target diagnostics for the MCLMC controller.

Runs the *exact* baseline ``full_mclmc_with_adapt_sharded`` (commit 71357bd)
against synthetic log-densities, mirroring the ``MCLMC_JIT`` wiring but with
NO lensing model in the loop.  Purpose: decide whether the adaptation collapse
seen in Phase 0 (vela shapelets, dim=17) is reproducible from the controller
alone (D4) and, if so, what input pathology triggers it (D5).

The lensing model is replaced by a synthetic ``log_prob``.  The qz object is a
plain ``MultivariateNormalTriL`` so that ``qz.sample``, ``qz.mean`` and
``qz.covariance()`` match the contract the sampler expects.  Geometry matches
Phase 0: dim=17, 8 chains, nb=2000, nr=500, desired_energy_var=5e-4,
frac_tune 0.2/0.6/0.2, seed 0, svi_mass_matrix_weight=10*n_chains,
init L=sqrt(17), init eps=sqrt(17)*0.25.

CPU ONLY.  Set JAX_PLATFORMS=cpu / CUDA_VISIBLE_DEVICES="" before launch.

Experiments (selected via --experiment / --target / --qz-init / --pathology):

  D4 grid:
    target cond in {1e2, 1e4, 1e8}
    qz-init in {honest, diag1e-8, diag1e-6}

  D5 ladder (base: cond=1e4 target, diag1e-8 qz-init):
    pathology = ripple_value   delta in {0.0092, 0.092, 0.92}  (value noise)
    pathology = ripple_grad    delta in {0.0092, 0.092, 0.92}  (gradient noise)
    pathology = nan_shell      (clean logp + NaN slab the chains must cross)

Summary stats and MM window-eigenvalue tables are computed identically to
``repro_vela_shapelets.py`` so D4/D5 are directly comparable to Phase 0
run_{a,b,c}/summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Geometry constants (Phase 0 frozen)
# ---------------------------------------------------------------------------
DIM = 17
DESIRED_ENERGY_VAR = 5e-4
# xi=1 energy target: |dE| = sqrt(dim * desired_energy_var)
DE_TARGET = math.sqrt(DIM * DESIRED_ENERGY_VAR)  # ~0.0922
# Typical posterior std scale (vela z-space): ~1e-2..1e-1.  Pick 5e-2.
STD_SCALE = 5e-2


def _build_target_cov(cond, dim=DIM, std_scale=STD_SCALE, seed=12345):
    """Zero-mean Gaussian covariance with log-spaced eigenvalues spanning
    condition number `cond`, geometric-mean std == std_scale, random rotation.
    Returns (cov, true_logdens_precision_chol_stuff) -- we return cov and a
    callable log_prob built from its precision.
    """
    rng = np.random.default_rng(seed)
    # log-spaced eigenvalues (variances) with geo-mean = std_scale**2
    log_lo = -0.5 * math.log10(cond)
    log_hi = +0.5 * math.log10(cond)
    eig_var = (std_scale ** 2) * np.logspace(log_lo, log_hi, dim)
    # random orthonormal rotation
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    cov = (Q * eig_var) @ Q.T
    cov = 0.5 * (cov + cov.T)
    return cov.astype(np.float64)


def main():
    p = argparse.ArgumentParser(description="D4/D5 synthetic controller diagnostics")
    p.add_argument("--experiment", required=True, help="tag for the run subdir")
    p.add_argument("--cond", type=float, default=1e4,
                   help="condition number of the target Gaussian covariance")
    p.add_argument("--qz-init", choices=["honest", "diag1e-8", "diag1e-6"],
                   default="diag1e-8",
                   help="initial qz covariance: true cov, 1e-8*I, or 1e-6*I")
    p.add_argument("--pathology", choices=["none", "ripple_value", "ripple_grad", "nan_shell"],
                   default="none")
    p.add_argument("--delta", type=float, default=0.0,
                   help="ripple amplitude (D5 ripple pathologies)")
    p.add_argument("--n-chains", type=int, default=8)
    p.add_argument("--num-burnin-steps", type=int, default=2000)
    p.add_argument("--num-results", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--regularize-mm", action="store_true",
                   help="F4: Stan-style shrinkage + PSD floor on every mass-matrix window.")
    args = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    import blackjax

    from gigalens.jax.experimental.mclmc import full_mclmc_with_adapt_sharded
    from gigalens_research.inference.blackjax_updated_utils import (
        _build_kernel_shardmap, init_multi, isokinetic_mclachlan_smart,
    )
    tfd = tfp.distributions

    print(f"[d4d5] JAX {jax.__version__} devices {jax.devices()}", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_out = args.out_dir or os.path.join(script_dir, "diagnosis_2026-06", "d4_d5")
    out_dir = os.path.join(base_out, args.experiment)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Build target -----------------------------------------------------
    cov_np = _build_target_cov(args.cond)
    prec_np = np.linalg.inv(cov_np)
    prec = jnp.asarray(prec_np, dtype=jnp.float32)

    # exact std scales (for step_norm comparison)
    true_std = np.sqrt(np.diag(cov_np))

    def base_log_prob(z):
        # zero-mean Gaussian: -0.5 z^T P z
        return -0.5 * jnp.dot(z, prec @ z)

    # ---- Pathology injection ---------------------------------------------
    # Ripple: high-frequency sinusoid in z.  Wavelength chosen << typical
    # eps*velocity step.  Velocity ~ chol(M^-1) ~ std_scale per unit eps; a
    # single integrator substep moves ~ eps * std_scale ~ (sqrt17*0.25)*5e-2
    # ~ 0.05 in z early, far smaller once eps shrinks.  Pick omega so the
    # ripple wavelength (2pi/|omega|) is ~1e-3 in z -> |omega| ~ 6e3 per
    # component (summed over dim).  This guarantees sub-step aliasing/noise.
    rng = np.random.default_rng(777)
    omega_np = rng.standard_normal(DIM)
    omega_np = omega_np / np.linalg.norm(omega_np) * 6.0e3
    omega = jnp.asarray(omega_np, dtype=jnp.float32)
    delta = jnp.float32(args.delta)

    pathology = args.pathology

    if pathology == "ripple_value":
        def log_prob(z):
            return base_log_prob(z) + delta * jnp.sin(jnp.dot(omega, z))
    elif pathology == "ripple_grad":
        # logp value stays smooth, but the *gradient* gets a ripple added.
        # Implemented via custom_jvp so value_and_grad sees a polluted grad
        # while the value itself is the clean Gaussian.
        from jax import custom_jvp

        @custom_jvp
        def log_prob(z):
            return base_log_prob(z)

        @log_prob.defjvp
        def log_prob_jvp(primals, tangents):
            (z,) = primals
            (dz,) = tangents
            val = base_log_prob(z)
            clean_grad = -(prec @ z)
            # additive ripple on the gradient (cos of same phase, scaled)
            ripple_grad = delta * jnp.cos(jnp.dot(omega, z)) * omega
            g = clean_grad + ripple_grad
            return val, jnp.dot(g, dz)
    elif pathology == "nan_shell":
        # clean Gaussian everywhere except a thin slab in the first coordinate
        # that the chains must occasionally cross.  Slab centred ~1 std out
        # along axis 0, thickness ~0.2 std.
        c0 = jnp.float32(true_std[0])
        shell_center = jnp.float32(0.6) * c0
        shell_halfwidth = jnp.float32(0.05) * c0

        def log_prob(z):
            base = base_log_prob(z)
            in_shell = jnp.abs(z[0] - shell_center) < shell_halfwidth
            return jnp.where(in_shell, jnp.float32(jnp.nan), base)
    else:
        log_prob = base_log_prob

    # ---- qz object (matches MCLMC_JIT contract) ---------------------------
    if args.qz_init == "honest":
        qz_cov = cov_np
    elif args.qz_init == "diag1e-8":
        qz_cov = np.eye(DIM) * 1e-8
    elif args.qz_init == "diag1e-6":
        qz_cov = np.eye(DIM) * 1e-6
    qz_scale_tril = np.linalg.cholesky(qz_cov).astype(np.float32)
    qz = tfd.MultivariateNormalTriL(
        loc=jnp.zeros(DIM, dtype=jnp.float32),
        scale_tril=jnp.asarray(qz_scale_tril),
    )

    # ---- Mirror MCLMC_JIT wiring -----------------------------------------
    n_chains = args.n_chains
    integrator = isokinetic_mclachlan_smart
    kernel = lambda inverse_mass_matrix: _build_kernel_shardmap(
        logdensity_fn=log_prob,
        integrator=integrator,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    rng_key = jax.random.key(args.seed)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)

    state_multi = init_multi(qz.sample((n_chains,), seed=init_key), init_key, log_prob)
    dim = state_multi.position.shape[-1]
    assert dim == DIM, dim

    init_L = jnp.sqrt(dim)
    init_step_size = jnp.sqrt(dim) * 0.25
    starting_adapt_state = blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState(
        L=init_L, step_size=init_step_size, inverse_mass_matrix=qz.covariance()
    )

    print(f"[d4d5] exp={args.experiment} cond={args.cond:.0e} qz_init={args.qz_init} "
          f"pathology={pathology} delta={args.delta:g}", flush=True)
    print(f"[d4d5] DE_TARGET(xi=1)={DE_TARGET:.4g}  true_std[min,med,max]="
          f"[{true_std.min():.3g},{np.median(true_std):.3g},{true_std.max():.3g}]", flush=True)

    t1 = time.perf_counter()
    hist, params_final = full_mclmc_with_adapt_sharded(
        kernel=kernel,
        num_burnin_steps=args.num_burnin_steps,
        num_results=args.num_results,
        state_init=state_multi,
        params_init=starting_adapt_state,
        svi_mean=qz.mean(),
        rng_key=tune_key,
        frac_tune1=0.2,
        frac_tune2=0.6,
        frac_tune3=0.2,
        desired_energy_var=DESIRED_ENERGY_VAR,
        num_chains=n_chains,
        num_effective_samples=100,
        svi_mass_matrix_weight=10.0 * n_chains,
        step_size_adapt_use_psmile=False,
        windowed_mass_matrix=True,
        regularize_mass_matrix=args.regularize_mm,
        progress_bar=False,
    )
    wall = time.perf_counter() - t1
    print(f"[d4d5] MCLMC done in {wall:.1f}s", flush=True)

    # ---- Save raw hist ----------------------------------------------------
    fields = {
        "position": hist.position, "step_size": hist.step_size, "L": hist.L,
        "nonan": hist.nonan, "xi": hist.xi,
        "energy_change_raw": hist.energy_change_raw,
        "kernel_nonan": hist.kernel_nonan, "step_norm": hist.step_norm,
        "inverse_mass_matrix": hist.inverse_mass_matrix,
    }
    for name, arr in fields.items():
        np.save(os.path.join(out_dir, f"hist_{name}.npy"), np.asarray(arr))

    total_steps = args.num_burnin_steps + args.num_results
    n_steps1 = round(args.num_burnin_steps * 0.2)
    n_steps2 = round(args.num_burnin_steps * 0.6)
    n_steps3 = round(args.num_burnin_steps * 0.2)

    config = {
        "experiment": args.experiment, "cond": args.cond, "qz_init": args.qz_init,
        "pathology": pathology, "delta": args.delta, "n_chains": n_chains,
        "num_burnin_steps": args.num_burnin_steps, "num_results": args.num_results,
        "seed": args.seed, "dim": dim, "desired_energy_var": DESIRED_ENERGY_VAR,
        "de_target_xi1": DE_TARGET, "std_scale": STD_SCALE,
        "true_std_min": float(true_std.min()), "true_std_med": float(np.median(true_std)),
        "true_std_max": float(true_std.max()),
        "total_steps": total_steps, "tune1_steps": n_steps1,
        "tune2_steps": n_steps2, "tune3_steps": n_steps3,
        "jax_version": jax.__version__, "wall_time_s": wall,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Per-phase summary (mirror repro_vela_shapelets) ------------------
    tune1_end = n_steps1
    tune2_end = n_steps1 + n_steps2
    tune3_end = n_steps1 + n_steps2 + n_steps3

    def phase_stats(name, start, end):
        sl = slice(start, end)
        ss = np.asarray(hist.step_size[:, sl])
        xi = np.asarray(hist.xi[:, sl])
        knonan = np.asarray(hist.kernel_nonan[:, sl])
        snorm = np.asarray(hist.step_norm[:, sl])
        ecraw = np.asarray(hist.energy_change_raw[:, sl])
        n = end - start
        if n <= 0:
            return {"name": name, "steps": 0}
        return {
            "name": name, "steps": n,
            "step_size_init": float(ss[:, 0].mean()),
            "step_size_final": float(ss[:, -1].mean()),
            "step_size_min": float(ss.min()), "step_size_max": float(ss.max()),
            "xi_q10": float(np.nanpercentile(xi, 10)),
            "xi_q50": float(np.nanpercentile(xi, 50)),
            "xi_q90": float(np.nanpercentile(xi, 90)),
            "kernel_nonan_frac": float(knonan.mean()),
            "kernel_rejected_frac": float((~knonan).mean()),
            "step_norm_median": float(np.median(snorm)),
            "step_norm_p10": float(np.percentile(snorm, 10)),
            "energy_change_raw_rms": float(np.sqrt(np.nanmean(ecraw ** 2))),
            "energy_change_raw_nan_frac": float(np.mean(~np.isfinite(ecraw))),
        }

    phases = [
        phase_stats("tune1", 0, tune1_end),
        phase_stats("tune2", tune1_end, tune2_end),
        phase_stats("tune3", tune2_end, tune3_end),
        phase_stats("sampling", tune3_end, total_steps),
    ]

    # MM window eigenvalues
    mm_start = n_steps1
    n_mm_steps = round(0.67 * n_steps2)
    ratios = [1, 2, 4]
    w_sizes = [max(1, round(n_mm_steps * r / sum(ratios))) for r in ratios]
    w_sizes[-1] = n_mm_steps - sum(w_sizes[:-1])
    pos, window_ends = mm_start, []
    for ws in w_sizes:
        pos += ws
        window_ends.append(pos - 1)

    mm_eig_stats = []
    imm = np.asarray(hist.inverse_mass_matrix)
    for we in window_ends:
        if 0 <= we < total_steps:
            m = imm[0, we]
            try:
                eigs = np.linalg.eigvalsh(m)
                posv = eigs[eigs > 0]
                mm_eig_stats.append({
                    "window_end_step": int(we),
                    "eig_min": float(eigs.min()), "eig_max": float(eigs.max()),
                    "eig_min_pos": float(posv.min()) if posv.size else None,
                    "cond": float(eigs.max() / max(posv.min(), 1e-300)) if posv.size else None,
                    "n_negative_eigs": int(np.sum(eigs < 0)),
                    "any_nan": bool(np.any(~np.isfinite(m))),
                })
            except Exception as ex:
                mm_eig_stats.append({"window_end_step": int(we), "error": str(ex)})

    summary = {"config": config, "phases": phases,
               "mass_matrix_window_eigenvalues": mm_eig_stats}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    for ph in phases:
        if ph.get("steps", 0) == 0:
            continue
        print(f"  {ph['name']:10s}: eps {ph['step_size_init']:.4g} -> {ph['step_size_final']:.4g} "
              f"(min {ph['step_size_min']:.3g}) xi[10,50,90]="
              f"[{ph['xi_q10']:.3g},{ph['xi_q50']:.3g},{ph['xi_q90']:.3g}] "
              f"kern_rej={ph['kernel_rejected_frac']:.3f} "
              f"snorm_med={ph['step_norm_median']:.3g} "
              f"ecraw_rms={ph['energy_change_raw_rms']:.3g} "
              f"ecraw_nan={ph['energy_change_raw_nan_frac']:.3f}", flush=True)
    for e in mm_eig_stats:
        if "error" in e:
            print(f"  MM step={e['window_end_step']}: ERROR {e['error']}", flush=True)
        else:
            print(f"  MM step={e['window_end_step']:4d}: eig[{e['eig_min']:.3g},{e['eig_max']:.3g}] "
                  f"cond={e['cond'] if e['cond'] is None else f'{e['cond']:.3g}'} "
                  f"n_neg={e['n_negative_eigs']} nan={e['any_nan']}", flush=True)
    print(f"\n[d4d5] outputs -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
