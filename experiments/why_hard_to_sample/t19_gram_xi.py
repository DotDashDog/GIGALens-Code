"""T19 (Phase C) -- is the shared MCLMC energy-error (xi) spike carried by
near-degeneracy of the lstsq linear-amplitude layer (Gram conditioning)?

Pre-registered test on STORED chains only (no sampling). Every threshold is a
REGISTERED module constant below; NONE is tuned here. This script PROPOSES
numbers/plots marked UNCERTIFIED and does not adjudicate -- a grader inspects the
artifacts (the operating card, rule 5).

WHAT IS MEASURED, per selected posterior draw z (14,)
-----------------------------------------------------
  * G = B_w^T B_w  (73x73), the *weighted* Gram of the lstsq design matrix. B_w is
    the EXACT production design matrix: the 73 lensed+PSF-convolved+pooled shapelet
    basis images, multiplied by the pixel mask (float32, as production casts it) and
    the per-pixel weight W=1/error_map, reshaped to (n_pixels, 73). This is the
    matrix production forms as ``Xt @ X`` inside
    ``SceneSimulator.lstsq_simulate`` (gigalens/jax/scene_simulator.py). We form B_w
    via the production render (``return_stacked=True``) and replicate ONLY the two
    weighting lines (mask*W, reshape) that immediately follow it -- byte-identical to
    production's own X. G is the UN-regularized normal matrix (the task's definition);
    production adds a tiny diagonal jitter inside the solve, which we deliberately do
    NOT add here so cond(G)/lmin(G) reflect the raw linear layer.
      -> log10 cond(G) = log10(lmax/lmin), log10 lmin(G)  via numpy eigh (float64).

  * lambda1_GN = largest eigenvalue of J^T W J (Gauss-Newton curvature of the pixel
    chi^2 in z-space), where J = d r(z)/dz is the jacobian of the BEST-FIT rendered
    image r(z) (amplitudes RE-SOLVED at each z: we differentiate THROUGH the lstsq
    solve via jax.jacfwd of the default ``lstsq_simulate`` output -- 14 forward
    passes), and W = diag(mask/sigma^2). J columns are standardized by the pooled
    per-coordinate std of the arm (scale column j by std_z[j]) so lambda1 is in
    per-posterior-sigma units.
      -> log10 lambda1_GN via numpy eigh (float64) on the 14x14 scaled J^T W J.

DESIGN-MATRIX CODE PATH (file:line, all reused, none reimplemented)
-------------------------------------------------------------------
  gigalens/jax/scene_prob_model.py
    ProbModel._model_image (lstsq) -> simulator.lstsq_simulate(params, image, err, mask)
    ProbModel.log_like: x=bij.forward(list(z.T)); params=model.to_params(x)
  gigalens/jax/scene_simulator.py
    SceneSimulator.lstsq_simulate: renders each component's basis (_render_light),
      PSF-convolves (_psf_convolve/_convolve_components), pools; with
      return_stacked=True returns ret=(bs,h,w,ncomp). The weighting it then applies is
        W = (1/err_map)[...,None]; X = reshape(ret*mask[None,...,None]*W,(bs,-1,depth))
      and coeffs = _solve_normal_eq_with_fallback(Xt@X, Xt@Y).  <-- Gram = Xt@X = B_w^T B_w
  gigalens/jax/simulator.py:90  _solve_normal_eq_with_fallback (LU solve; the solve we
    differentiate through for the GN jacobian).

Runs on a GPU node inside the Shifter container (JAX_ENABLE_X64=1); numpy-only parts
(alignment gate, stratification, eigh) are smoke-tested on the login node.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import assert_x64, load_target

# ---------------------------------------------------------------------------
# REGISTERED constants (pre-registered; DO NOT TUNE)
# ---------------------------------------------------------------------------
XI_SPIKE_THRESH = 10.0        # xi>10 defines a "spike" (alignment gate + fraction)
P_T19C_ALIGN_TOL = 0.002      # gate: |measured frac(xi>10) - known| must be <= this
KNOWN_FRAC = {"old": 0.1235, "new": 0.1037}  # published post-burn-in frac(xi>10)

SEED = 20260703               # fixed stratified-selection seed
N_TOP = 128                   # top-decile-xi points
N_BOT = 128                   # bottom-decile-xi points
N_UNIF = 256                  # uniform-random points
DECILE_HI = 90.0              # top-decile threshold percentile
DECILE_LO = 10.0              # bottom-decile threshold percentile

OLD_EXCL_CHAIN = 0            # old-arm variant: exclude chain 0, t in [0, TMAX)
OLD_EXCL_TMAX = 5000          # the burn-in-escape "basin" transient

P_T19A_RATIO = 2.0            # P_T19a meets: median cond(G) top-decile >= 2x bottom
F_T19A_RATIO = 1.3            # F_T19a fires if cond ratio < 1.3 AND ...
F_T19A_RHO = 0.1             #   ... pooled Spearman rho(log xi, log cond G) < 0.1 in BOTH arms
P_T19B_FLAT = 1.3            # P_T19b: cond ratio "flat" if < 1.3
P_T19B_LAMBDA_RATIO = 3.0    # P_T19b: curvature carries xi if lambda1 ratio >= 3

N_BURN = 10000               # burn-in steps that precede the results phase in xi
DIM = 14
N_BASIS = 73

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests"
ARMS = {
    "old": dict(
        data_dir=os.path.join(_HERE, "systems", "carousel_min_old"),
        run_dir=os.path.join(_SIM, "minimal_case_oldbij", "mclmc"),
    ),
    "new": dict(
        data_dir=os.path.join(_HERE, "systems", "carousel_min_new"),
        run_dir=os.path.join(_SIM, "minimal_case_newbij", "mclmc"),
    ),
}

STRATA = ("top", "bottom", "uniform")
_STRAT_COLOR = {"top": "#d62728", "bottom": "#1f77b4", "uniform": "#7f7f7f"}


# ---------------------------------------------------------------------------
# numpy-only helpers (import-safe without jax; smoke-tested on login node)
# ---------------------------------------------------------------------------
def load_arm_arrays(run_dir):
    """Return (samples_z (8,10000,14), xi_post (8,10000)) with the documented
    alignment xi_post = xi[:, -10000:]  (xi[:, -10000:][c,t] <-> samples_z[c,t])."""
    a = np.load(os.path.join(run_dir, "arrays.npz"))
    d = np.load(os.path.join(run_dir, "diagnostics.npz"))
    samples_z = np.asarray(a["samples_z"], dtype=np.float64)
    xi = np.asarray(d["xi"], dtype=np.float64)
    if samples_z.ndim != 3 or samples_z.shape[2] != DIM:
        raise ValueError(f"samples_z shape {samples_z.shape} unexpected")
    n_res = samples_z.shape[1]
    if xi.shape[0] != samples_z.shape[0] or xi.shape[1] < n_res:
        raise ValueError(f"xi shape {xi.shape} incompatible with samples {samples_z.shape}")
    xi_post = xi[:, -n_res:]
    return samples_z, xi_post


def alignment_frac(xi_post):
    """frac(xi>threshold) over the full post-burn-in xi -- the P_T19c gate quantity."""
    return float(np.mean(xi_post > XI_SPIKE_THRESH))


def eligible_mask(shape, variant):
    """Boolean (8,10000) mask of eligible pooled points for a variant."""
    m = np.ones(shape, dtype=bool)
    if variant == "excl":
        m[OLD_EXCL_CHAIN, :OLD_EXCL_TMAX] = False
    return m


def stratify(xi_post, elig, rng):
    """Return (rows, unique_pts).

    rows: list of (chain, t, stratum) of length N_TOP+N_BOT+N_UNIF=512, drawn from the
    eligible pooled (chain,t) indices; deciles are computed over the eligible xi.
    unique_pts: sorted list of unique (chain,t) actually referenced (rendered once)."""
    n_res = xi_post.shape[1]
    flat_elig = np.flatnonzero(elig.ravel())
    xf = xi_post.ravel()[flat_elig]
    p_hi = np.percentile(xf, DECILE_HI)
    p_lo = np.percentile(xf, DECILE_LO)
    top_pool = flat_elig[xf >= p_hi]
    bot_pool = flat_elig[xf <= p_lo]
    for nm, pool, n in (("top", top_pool, N_TOP), ("bottom", bot_pool, N_BOT)):
        if pool.size < n:
            raise ValueError(f"stratum {nm} pool has {pool.size} < {n} eligible points")

    sel = {
        "top": rng.choice(top_pool, N_TOP, replace=False),
        "bottom": rng.choice(bot_pool, N_BOT, replace=False),
        "uniform": rng.choice(flat_elig, N_UNIF, replace=False),
    }
    rows = []
    for strat in STRATA:
        for f in sel[strat]:
            c, t = divmod(int(f), n_res)
            rows.append((c, t, strat))
    unique_pts = sorted({(c, t) for c, t, _ in rows})
    return rows, unique_pts


def spd_cond_lmin(G):
    """(log10 cond, log10 lmin) of a symmetric matrix via numpy eigh (float64).

    Records a non-positive-lmin occurrence via a returned flag rather than silently
    clipping: a rank-deficient weighted Gram is itself a finding, not a nuisance."""
    w = np.linalg.eigvalsh(np.asarray(G, dtype=np.float64))
    lmin, lmax = float(w[0]), float(w[-1])
    ok = lmin > 0
    lmin_eff = lmin if ok else float(np.finfo(np.float64).tiny)
    log10_cond = float(np.log10(lmax) - np.log10(lmin_eff))
    log10_lmin = float(np.log10(lmin_eff))
    return log10_cond, log10_lmin, ok


def top_eig(M):
    """log10 of the largest eigenvalue of a symmetric matrix (float64 eigh)."""
    w = np.linalg.eigvalsh(np.asarray(M, dtype=np.float64))
    return float(np.log10(float(w[-1])))


# ---------------------------------------------------------------------------
# jax device kernels: G(z) = B_w^T B_w and M(z) = J^T W J (unscaled)
# ---------------------------------------------------------------------------
def make_kernels(prob_model):
    """Build two jitted per-point functions using the EXACT production render path.

    gram_G(z)  -> (73,73)  B_w^T B_w with mask+weight, un-regularized.
    gn_M(z)    -> (14,14)  J^T diag(mask/sigma^2) J, J = d r(z)/dz through the solve.
    """
    import jax
    import jax.numpy as jnp

    sim = prob_model.simulators[0]
    ds = prob_model.datasets[0]
    image = ds.image
    err = ds.error_map
    mask = ds.mask
    depth = sim.depth
    if depth != N_BASIS:
        raise ValueError(f"expected {N_BASIS} lstsq bases, simulator.depth={depth}")

    maskf = mask.astype(jnp.float32)       # production casts mask to float32
    W = (1.0 / err)[..., jnp.newaxis]      # (h,w,1); production weight
    # per-pixel GN weight sqrt(mask/sigma^2) = mask/sigma (mask in {0,1})
    w_pix = jnp.reshape(maskf * jnp.squeeze(W), (-1,))

    def _params(z):
        x = prob_model.bij.forward(list(z[jnp.newaxis, :].T))
        return prob_model.model.to_params(x)

    def gram_G(z):
        params = _params(z)
        ret = sim.lstsq_simulate(params, image, err, mask, True)  # return_stacked
        Xw = jnp.reshape(ret * maskf[jnp.newaxis, ..., jnp.newaxis] * W,
                         (ret.shape[0], -1, depth))
        Bw = Xw[0]
        return Bw.T @ Bw

    def render_flat(z):
        params = _params(z)
        img = sim.lstsq_simulate(params, image, err, mask)  # best-fit (h,w)
        return jnp.reshape(img, (-1,))

    def gn_M(z):
        J = jax.jacfwd(render_flat)(z)     # (n_pixels, 14), 14 forward passes
        Jw = J * w_pix[:, jnp.newaxis]
        return Jw.T @ Jw

    return jax.jit(gram_G), jax.jit(gn_M)


# ---------------------------------------------------------------------------
# per-arm driver
# ---------------------------------------------------------------------------
def run_arm(arm, out_dir):
    import jax.numpy as jnp

    cfg = ARMS[arm]
    samples_z, xi_post = load_arm_arrays(cfg["run_dir"])

    # --- P_T19c alignment gate (hard) --------------------------------------
    frac = alignment_frac(xi_post)
    known = KNOWN_FRAC[arm]
    print(f"[{arm}] alignment gate: measured frac(xi>{XI_SPIKE_THRESH:g}) = {frac:.6f} "
          f"(known {known}), |delta| = {abs(frac - known):.6f}")
    if abs(frac - known) > P_T19C_ALIGN_TOL:
        raise RuntimeError(
            f"P_T19c ALIGNMENT GATE FAILED for arm {arm}: measured frac(xi>{XI_SPIKE_THRESH})"
            f"={frac:.6f} differs from known {known} by {abs(frac-known):.6f} > "
            f"{P_T19C_ALIGN_TOL}. xi/samples alignment is WRONG -- aborting before any "
            "downstream computation.")

    # per-coordinate pooled std for GN column standardization (full arm)
    std_z = samples_z.reshape(-1, DIM).std(axis=0)
    if not np.all(std_z > 0):
        raise ValueError(f"[{arm}] non-positive per-coord std: {std_z}")
    std_outer = np.outer(std_z, std_z)

    # variants: old gets pooled + excl(chain0 t<5000); new only pooled
    variants = ["pooled", "excl"] if arm == "old" else ["pooled"]

    # selections per variant (fixed seed) + union of unique points to render
    rng_master = np.random.default_rng(SEED)
    sel_rows, sel_unique = {}, {}
    union = set()
    for v in variants:
        elig = eligible_mask(xi_post.shape, "excl" if v == "excl" else "pooled")
        # independent, reproducible rng per variant derived from the master seed
        rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
        rows, uniq = stratify(xi_post, elig, rng)
        sel_rows[v] = rows
        sel_unique[v] = uniq
        union.update(uniq)
    union = sorted(union)
    print(f"[{arm}] variants={variants}  unique points to render = {len(union)}")

    # --- render each unique point once -------------------------------------
    gram_G, gn_M = make_kernels(load_target(cfg["data_dir"])[0].prob_model)
    per_pt = {}
    n_nonpd = 0
    for i, (c, t) in enumerate(union):
        z = jnp.asarray(samples_z[c, t])
        G = np.asarray(gram_G(z), dtype=np.float64)
        M = np.asarray(gn_M(z), dtype=np.float64) * std_outer  # standardize columns
        log10_cond, log10_lmin, pd_ok = spd_cond_lmin(G)
        log10_lam1 = top_eig(M)
        if not pd_ok:
            n_nonpd += 1
        per_pt[(c, t)] = dict(
            log10_cond_G=log10_cond, log10_lmin_G=log10_lmin,
            log10_lambda1_GN=log10_lam1, log10_xi=float(np.log10(xi_post[c, t])),
        )
        if (i + 1) % 64 == 0 or (i + 1) == len(union):
            print(f"[{arm}] rendered {i+1}/{len(union)} points "
                  f"(last log10condG={log10_cond:.2f} log10lam1={log10_lam1:.2f})")
    if n_nonpd:
        print(f"[{arm}] WARNING: {n_nonpd} points had non-positive lmin(G) "
              "(rank-deficient weighted Gram) -- lmin floored for log only.")

    # --- assemble per-variant arrays + registered metrics ------------------
    from scipy.stats import spearmanr

    arm_out = {}
    for v in variants:
        rows = sel_rows[v]
        chain = np.array([r[0] for r in rows], dtype=np.int32)
        tstep = np.array([r[1] for r in rows], dtype=np.int32)
        strat = np.array([r[2] for r in rows])
        lc = np.array([per_pt[(r[0], r[1])]["log10_cond_G"] for r in rows])
        ll = np.array([per_pt[(r[0], r[1])]["log10_lmin_G"] for r in rows])
        lg = np.array([per_pt[(r[0], r[1])]["log10_lambda1_GN"] for r in rows])
        lx = np.array([per_pt[(r[0], r[1])]["log10_xi"] for r in rows])

        top = strat == "top"
        bot = strat == "bottom"
        # cond(G) medians in LINEAR cond units (median of 10**log10)
        med_cond_top = float(np.median(10.0 ** lc[top]))
        med_cond_bot = float(np.median(10.0 ** lc[bot]))
        cond_ratio = med_cond_top / med_cond_bot
        med_lam_top = float(np.median(10.0 ** lg[top]))
        med_lam_bot = float(np.median(10.0 ** lg[bot]))
        lam_ratio = med_lam_top / med_lam_bot

        rho_cond = float(spearmanr(lx, lc).correlation)
        rho_lmin = float(spearmanr(lx, ll).correlation)
        rho_lam = float(spearmanr(lx, lg).correlation)

        p_t19a_meets = cond_ratio >= P_T19A_RATIO
        p_t19b_branch = (cond_ratio < P_T19B_FLAT) and (lam_ratio >= P_T19B_LAMBDA_RATIO)

        print(f"\n[{arm}/{v}] cond(G) ratio (top/bottom) = {cond_ratio:.3f} "
              f"(P_T19a meets>= {P_T19A_RATIO}: {p_t19a_meets})")
        print(f"[{arm}/{v}] lambda1_GN ratio (top/bottom) = {lam_ratio:.3f}")
        print(f"[{arm}/{v}] Spearman rho log xi vs: log condG={rho_cond:+.3f} "
              f"log lminG={rho_lmin:+.3f} log lambda1={rho_lam:+.3f}")
        if cond_ratio < P_T19B_FLAT:
            if lam_ratio >= P_T19B_LAMBDA_RATIO:
                print(f"[{arm}/{v}] P_T19b VERDICT BRANCH: Gram ratio flat "
                      f"({cond_ratio:.2f}<{P_T19B_FLAT}) but lambda1 ratio "
                      f"{lam_ratio:.2f}>={P_T19B_LAMBDA_RATIO} -> CURVATURE spikes carry "
                      "xi, NOT the linear (lstsq) layer.")
            else:
                print(f"[{arm}/{v}] P_T19b VERDICT BRANCH: Gram ratio flat AND lambda1 "
                      f"ratio {lam_ratio:.2f}<{P_T19B_LAMBDA_RATIO} -> neither the linear "
                      "layer nor GN curvature (as measured) carries the xi spikes.")
        else:
            print(f"[{arm}/{v}] P_T19b VERDICT BRANCH: Gram ratio NOT flat "
                  f"({cond_ratio:.2f}>={P_T19B_FLAT}) -> linear-layer conditioning tracks "
                  "the spikes; P_T19b (curvature-not-linear) not triggered.")

        arm_out[v] = dict(
            chain=chain, tstep=tstep, stratum=strat,
            log10_cond_G=lc, log10_lmin_G=ll, log10_lambda1_GN=lg, log10_xi=lx,
            cond_ratio=cond_ratio, lam_ratio=lam_ratio,
            med_cond_top=med_cond_top, med_cond_bot=med_cond_bot,
            med_lam_top=med_lam_top, med_lam_bot=med_lam_bot,
            rho_cond=rho_cond, rho_lmin=rho_lmin, rho_lam=rho_lam,
            p_t19a_meets=bool(p_t19a_meets), p_t19b_branch=bool(p_t19b_branch),
        )
        _plot_variant(arm, v, arm_out[v], xi_post, out_dir)

    return dict(
        frac_measured=frac, frac_known=known, std_z=std_z, n_nonpd=int(n_nonpd),
        variants=variants, per_variant=arm_out,
    )


# ---------------------------------------------------------------------------
# plotting (matplotlib only): 2x2 panel per arm/variant
# ---------------------------------------------------------------------------
def _plot_variant(arm, v, d, xi_post, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strat = d["stratum"]
    lx, lc, lg = d["log10_xi"], d["log10_cond_G"], d["log10_lambda1_GN"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) log xi vs log cond G, colored by stratum
    for s in STRATA:
        m = strat == s
        ax[0, 0].scatter(lx[m], lc[m], s=14, alpha=0.6, c=_STRAT_COLOR[s], label=s)
    ax[0, 0].set_xlabel("log10 xi"); ax[0, 0].set_ylabel("log10 cond(G)")
    ax[0, 0].set_title(f"{arm}/{v}: xi vs Gram cond (rho={d['rho_cond']:+.2f})")
    ax[0, 0].legend(fontsize=8)

    # (0,1) log xi vs log lambda1_GN
    for s in STRATA:
        m = strat == s
        ax[0, 1].scatter(lx[m], lg[m], s=14, alpha=0.6, c=_STRAT_COLOR[s], label=s)
    ax[0, 1].set_xlabel("log10 xi"); ax[0, 1].set_ylabel("log10 lambda1_GN")
    ax[0, 1].set_title(f"{arm}/{v}: xi vs GN curvature (rho={d['rho_lam']:+.2f})")
    ax[0, 1].legend(fontsize=8)

    # (1,0) violin of log cond G by stratum
    data = [lc[strat == s] for s in STRATA]
    parts = ax[1, 0].violinplot(data, showmedians=True)
    ax[1, 0].set_xticks([1, 2, 3]); ax[1, 0].set_xticklabels(list(STRATA))
    ax[1, 0].set_ylabel("log10 cond(G)")
    ax[1, 0].set_title(f"cond ratio top/bottom = {d['cond_ratio']:.2f} "
                       f"(P_T19a>= {P_T19A_RATIO})")

    # (1,1) xi trace of one chain (chain 0) with selected points marked
    c0 = 0
    ax[1, 1].plot(np.log10(xi_post[c0]), lw=0.4, color="0.6")
    ax[1, 1].axhline(np.log10(XI_SPIKE_THRESH), color="k", ls=":", lw=0.8)
    for s in STRATA:
        m = (strat == s) & (d["chain"] == c0)
        ax[1, 1].scatter(d["tstep"][m], lx[m], s=22, c=_STRAT_COLOR[s],
                         edgecolor="k", linewidth=0.3, label=s, zorder=3)
    ax[1, 1].set_xlabel("t (post-burn-in)"); ax[1, 1].set_ylabel("log10 xi")
    ax[1, 1].set_title(f"chain {c0} xi trace + selected points")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"T19 Gram/xi -- arm={arm} variant={v} (PROPOSED, UNCERTIFIED)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(out_dir, f"t19_gram_xi_{arm}_{v}.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"[{arm}/{v}] wrote {png}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="T19 Phase C: Gram conditioning vs xi spikes")
    ap.add_argument("--arm", choices=["old", "new", "both"], default="both")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results_carousel", "phaseC"))
    args = ap.parse_args()

    assert_x64()
    os.makedirs(args.out_dir, exist_ok=True)
    arms = ["old", "new"] if args.arm == "both" else [args.arm]

    results = {a: run_arm(a, args.out_dir) for a in arms}

    # --- F_T19a: fires if cond ratio<1.3 AND pooled rho(log xi,log condG)<0.1 in BOTH
    f_t19a = None
    if all(a in results for a in ("old", "new")):
        old_p = results["old"]["per_variant"]["pooled"]
        new_p = results["new"]["per_variant"]["pooled"]
        cond_flat_both = (old_p["cond_ratio"] < F_T19A_RATIO) and (new_p["cond_ratio"] < F_T19A_RATIO)
        rho_low_both = (old_p["rho_cond"] < F_T19A_RHO) and (new_p["rho_cond"] < F_T19A_RHO)
        f_t19a = bool(cond_flat_both and rho_low_both)
        print(f"\n[F_T19a] fires = {f_t19a}  (cond<{F_T19A_RATIO} both: {cond_flat_both}; "
              f"rho<{F_T19A_RHO} both: {rho_low_both})  "
              "=> linear-layer conditioning does NOT explain the xi spikes")

    # --- save per-point arrays (npz) + summary (json) ----------------------
    npz_payload = {}
    for a, r in results.items():
        for v, d in r["per_variant"].items():
            pre = f"{a}_{v}_"
            npz_payload[pre + "chain"] = d["chain"]
            npz_payload[pre + "tstep"] = d["tstep"]
            npz_payload[pre + "stratum"] = d["stratum"]
            npz_payload[pre + "log10_cond_G"] = d["log10_cond_G"]
            npz_payload[pre + "log10_lmin_G"] = d["log10_lmin_G"]
            npz_payload[pre + "log10_lambda1_GN"] = d["log10_lambda1_GN"]
            npz_payload[pre + "log10_xi"] = d["log10_xi"]
        npz_payload[f"{a}_std_z"] = r["std_z"]
    np.savez(os.path.join(args.out_dir, "t19_gram_xi.npz"), **npz_payload)

    def _reg(d):
        return {k: d[k] for k in (
            "cond_ratio", "lam_ratio", "med_cond_top", "med_cond_bot",
            "med_lam_top", "med_lam_bot", "rho_cond", "rho_lmin", "rho_lam",
            "p_t19a_meets", "p_t19b_branch")}

    summary = {
        "status": "proposed (UNCERTIFIED)",
        "registered_constants": dict(
            XI_SPIKE_THRESH=XI_SPIKE_THRESH, P_T19C_ALIGN_TOL=P_T19C_ALIGN_TOL,
            KNOWN_FRAC=KNOWN_FRAC, SEED=SEED, N_TOP=N_TOP, N_BOT=N_BOT, N_UNIF=N_UNIF,
            DECILE_HI=DECILE_HI, DECILE_LO=DECILE_LO, OLD_EXCL_CHAIN=OLD_EXCL_CHAIN,
            OLD_EXCL_TMAX=OLD_EXCL_TMAX, P_T19A_RATIO=P_T19A_RATIO,
            F_T19A_RATIO=F_T19A_RATIO, F_T19A_RHO=F_T19A_RHO, P_T19B_FLAT=P_T19B_FLAT,
            P_T19B_LAMBDA_RATIO=P_T19B_LAMBDA_RATIO),
        "P_T19c_gate": {a: dict(frac_measured=r["frac_measured"],
                                frac_known=r["frac_known"],
                                passed=abs(r["frac_measured"] - r["frac_known"]) <= P_T19C_ALIGN_TOL,
                                n_nonpd=r["n_nonpd"])
                        for a, r in results.items()},
        "per_arm": {a: {v: _reg(d) for v, d in r["per_variant"].items()}
                    for a, r in results.items()},
        "F_T19a_fires": f_t19a,
    }
    sj = os.path.join(args.out_dir, "t19_summary.json")
    with open(sj, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 't19_gram_xi.npz')}")
    print(f"wrote {sj}")


if __name__ == "__main__":
    main()
