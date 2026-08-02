"""T20 -- MICRO-TEXTURE probe of the carousel minimal-case log-posterior.

Complements the t3 transect scan (run via t20_run_t3.py) by looking INSIDE actual
sampler steps: for adjacent post-burn-in MCLMC draws (c, t-1)->(c, t) it asks
whether logp along the straight z-segment is a smooth quadratic bowl or carries
micro-texture (value roughness), and it contrasts segments whose ENDPOINT xi is a
spike ("top") against calm segments. It also runs two carousel-specific
micro-transect D2 ladders (the escape direction and the z9 axis) that t3's
generic direction set cannot cover.

This script PRODUCES artifacts + a PROPOSED (UNCERTIFIED) summary; it does NOT
adjudicate H1. All thresholds are PRE-REGISTERED constants below and are NEVER
tuned.

Alignment (verified in T19, reproduced by the gate here):
  diagnostics.npz["xi"] is (chains, 20000) spanning burnin+results;
  arrays.npz["samples_z"] is (chains, 10000, dim) post-burn-in; the alignment is
  xi[:, -10000:][c, t] <-> samples_z[c, t]. frac(xi_aligned > 10) must reproduce
  0.1235 (old) / 0.1037 (new) within 0.002 or the run ABORTS.

jax is imported ONLY inside functions so the pure-numpy machinery (gate, pair
selection, quadratic-fit deviation) is importable/testable under plain numpy.
`--dry-run` exercises exactly that path with no jax.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import t3_transects as t3   # reuse second_difference / empirical_eps_f / roundoff / sigma_along

# ===========================================================================
# PRE-REGISTERED constants (NEVER tune)
# ===========================================================================
SEED = 20260703

# Alignment gate (T19).
XI_SPIKE_THRESH = 10.0
ALIGN_FRAC = {"old": 0.1235, "new": 0.1037}
ALIGN_TOL = 0.002

# Pair selection.
N_PAIRS = 32                 # per group per arm
TOP_PCT = 0.01               # "top" = endpoint xi in top 1%
BASIN_CHAIN = 0              # old-arm chain-0 basin transient
BASIN_TSTEP = 5000           #   exclude t < 5000 (old arm only)
N_SEG_POINTS = 33            # evenly spaced points on the straight z-segment

# Registered verdict thresholds (P-T20d + falsifier F-T20d).
DEV_THRESH_NAT = 1.0         # a segment "deviates" if max|logp - quad fit| >= 1 nat
P_T20D_TOPFRAC = 0.5         # MEETS: >= 0.5 of top pairs deviate
P_T20D_CALMFRAC = 0.1        # MEETS: < 0.1 of calm pairs deviate
F_T20D_MARGIN = 0.05         # FIRES: top_frac <= calm_frac + 0.05

# Micro-transect D2 ladder (task 2.4), old arm only.
Z9_INDEX = 9                 # planes/1/light/0/center_x
H_LADDER_LO = 1e-4           # in units of sigma_dir
H_LADDER_HI = 3e-1
N_H = 13
N_PROBE = 16                 # for empirical eps_f floor (t3.empirical_eps_f)

# ===========================================================================
# paths
# ===========================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests"
REF_MCLMC = {
    "old": os.path.join(_SIM, "minimal_case_oldbij", "mclmc"),
    "new": os.path.join(_SIM, "minimal_case_newbij", "mclmc"),
}
SYSTEM_DIR = {
    "old": os.path.join(_HERE, "systems", "carousel_min_old"),
    "new": os.path.join(_HERE, "systems", "carousel_min_new"),
}
CLONE = {
    "old": os.path.join(_HERE, "results_carousel", "old", "t1", "clone.npz"),
    "new": os.path.join(_HERE, "results_carousel", "new", "t1", "clone.npz"),
}


def _now():
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# pure-numpy machinery (offline-testable)
# ===========================================================================

def alignment_gate(arm, xi):
    """Reproduce T19's frac(xi>10) gate on the aligned (post-burn-in) window.
    Returns (frac, ok). Aborts loudly in the caller if not ok."""
    xa = np.asarray(xi)[:, -10000:]
    frac = float(np.mean(xa > XI_SPIKE_THRESH))
    exp = ALIGN_FRAC[arm]
    return frac, bool(abs(frac - exp) <= ALIGN_TOL)


def candidate_pool(arm, samples_z, xi):
    """Adjacent-draw candidates (c, t) with t in [1, T-1]. Endpoint-xi labeling
    (convention A): the segment (c, t-1)->(c, t) is labeled by aligned xi[c, t].
    Old-arm chain-0 basin transient (t < BASIN_TSTEP) excluded.

    Returns (c_idx, t_idx, xi_end) flat arrays over the valid pool."""
    xa = np.asarray(xi)[:, -10000:]
    C, T, _ = samples_z.shape
    cc, tt = np.meshgrid(np.arange(C), np.arange(1, T), indexing="ij")
    cc = cc.ravel(); tt = tt.ravel()
    if arm == "old":
        keep = ~((cc == BASIN_CHAIN) & (tt < BASIN_TSTEP))
        cc, tt = cc[keep], tt[keep]
    xi_end = xa[cc, tt]
    return cc, tt, xi_end


def select_pairs(cc, tt, xi_end, rng):
    """top = N_PAIRS random pairs with endpoint xi in the top TOP_PCT;
    calm = N_PAIRS random pairs with endpoint xi < median. Thresholds computed on
    the (already basin-excluded) pool. Returns dict of index arrays into cc/tt."""
    p_hi = np.quantile(xi_end, 1.0 - TOP_PCT)
    med = np.median(xi_end)
    top_pool = np.where(xi_end >= p_hi)[0]
    calm_pool = np.where(xi_end < med)[0]
    if len(top_pool) < N_PAIRS or len(calm_pool) < N_PAIRS:
        raise RuntimeError(f"insufficient pool: top={len(top_pool)} "
                           f"calm={len(calm_pool)} need {N_PAIRS}")
    top = rng.choice(top_pool, N_PAIRS, replace=False)
    calm = rng.choice(calm_pool, N_PAIRS, replace=False)
    return {"top": top, "calm": calm, "p_hi": float(p_hi), "median": float(med)}


def quad_fit_maxdev(fvals):
    """max |f - quadratic least-squares fit| over evenly spaced samples on [0,1].
    Also returns the fit residual array. This is the segment micro-texture metric
    (nats): 0 for a perfect quadratic bowl, > 0 for kinks/roughness."""
    f = np.asarray(fvals, dtype=np.float64)
    n = f.shape[-1]
    tt = np.linspace(0.0, 1.0, n)
    A = np.vstack([tt * tt, tt, np.ones_like(tt)]).T
    coef, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    resid = f - A @ coef
    return float(np.max(np.abs(resid))), resid


def seg_length_pooled_sigma(z0, z1, pooled_sigma):
    """||(z1 - z0) / pooled_sigma||_2 : diagonal-whitened segment length."""
    return float(np.linalg.norm((np.asarray(z1) - np.asarray(z0)) /
                                np.asarray(pooled_sigma)))


# ===========================================================================
# jax-backed logp evaluator (lazy)
# ===========================================================================

def make_logp_eval(prob_model):
    """Return f(Z (N,dim)) -> np.ndarray logp, one batched jit call per shape."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def _lp(Z):
        return prob_model.log_prob(Z)[0]

    def f(Z, _chunk=16):
        # chunked: the carousel lstsq render is ~52MB/point of intermediates;
        # a 1056-point batch produced a 51.7GiB Gram matmul (OOM on 40GB).
        Z = np.asarray(Z, dtype=np.float64)
        out = np.empty(Z.shape[0], dtype=np.float64)
        for s in range(0, Z.shape[0], _chunk):
            e = min(s + _chunk, Z.shape[0])
            out[s:e] = np.asarray(_lp(jnp.asarray(Z[s:e]))).reshape(-1)
        return out
    return f


# ===========================================================================
# segment evaluation
# ===========================================================================

def segment_points(z_a, z_b, n=N_SEG_POINTS):
    """n evenly spaced z points from z_a (t=0) to z_b (t=1) inclusive."""
    t = np.linspace(0.0, 1.0, n)
    return z_a[None, :] + t[:, None] * (z_b - z_a)[None, :]


def eval_group(f_eval, samples_z, cc, tt, sel_idx, seg_key="A"):
    """Evaluate the N_SEG_POINTS-point logp curve for each selected pair.
      seg_key "A": segment (c, t-1) -> (c, t).
      seg_key "B": segment (c, t)   -> (c, t+1)  (the OTHER alignment convention;
                   only valid where t+1 <= T-1).
    Batches all points across the group into ONE f_eval call.
    Returns dict with segf (M,33), maxdev (M,), resid (M,33), dlogp (M,), valid (M,)."""
    C, T, D = samples_z.shape
    Zblocks = []
    valid = np.ones(len(sel_idx), dtype=bool)
    for k, i in enumerate(sel_idx):
        c, t = int(cc[i]), int(tt[i])
        if seg_key == "A":
            za, zb = samples_z[c, t - 1], samples_z[c, t]
        else:
            if t + 1 > T - 1:
                valid[k] = False
                za, zb = samples_z[c, t], samples_z[c, t]   # placeholder
            else:
                za, zb = samples_z[c, t], samples_z[c, t + 1]
        Zblocks.append(segment_points(za, zb))
    Zall = np.concatenate(Zblocks, axis=0)          # (M*33, D)
    fall = f_eval(Zall).reshape(len(sel_idx), N_SEG_POINTS)
    maxdev = np.empty(len(sel_idx))
    resid = np.empty((len(sel_idx), N_SEG_POINTS))
    for k in range(len(sel_idx)):
        maxdev[k], resid[k] = quad_fit_maxdev(fall[k])
    dlogp = fall[:, -1] - fall[:, 0]
    return {"segf": fall, "maxdev": maxdev, "resid": resid,
            "dlogp": dlogp, "valid": valid}


# ===========================================================================
# micro-transect D2 ladders (old arm)
# ===========================================================================

def micro_transect(f_eval, z0, e, cov, eps_mach):
    """D2 second-difference ladder along unit direction e, centered at z0.
    h = ladder * sigma_dir, ladder = geomspace(H_LADDER_LO, H_LADDER_HI, N_H).
    Mirrors t3.second_difference / t3.roundoff_model_D2 / t3.empirical_eps_f.
    Returns a JSON-able dict incl. raw f values so noise floors are recomputable."""
    e = t3._unit(e)
    sigma_dir = t3.sigma_along(cov, e)
    if sigma_dir <= 0:
        raise ValueError("sigma_dir <= 0")
    ladder = np.geomspace(H_LADDER_LO, H_LADDER_HI, N_H)   # ascending in sigma units
    h = ladder * sigma_dir

    # batch: [z0] + [z0+h e] + [z0-h e] + probe(j*h_min*e, j=1..N_PROBE)
    h_min = float(h.min())
    probe_t = np.arange(1, N_PROBE + 1) * h_min
    offs = np.concatenate([[0.0], h, -h, probe_t])
    Z = z0[None, :] + offs[:, None] * e[None, :]
    fv = f_eval(Z)
    f0 = float(fv[0])
    f_plus = fv[1:1 + N_H]
    f_minus = fv[1 + N_H:1 + 2 * N_H]
    f_probe = fv[1 + 2 * N_H:]

    D2 = t3.second_difference(f_plus, f_minus, f0, h)
    roundoff = t3.roundoff_model_D2(h, f0, eps_mach)       # theory floor 4*eps_f/h^2
    eps_emp = t3.empirical_eps_f(np.concatenate([[f0], f_probe]))
    emp_floor = 4.0 * eps_emp["observed_quantum"] / (h * h)

    # roughness read: |D2| relative to the (theory & empirical) roundoff floors.
    ratio_theory = np.abs(D2) / np.where(roundoff > 0, roundoff, np.nan)
    ratio_emp = np.abs(D2) / np.where(emp_floor > 0, emp_floor, np.nan)

    def _safe_nanmax(a):
        a = np.asarray(a, dtype=np.float64)
        return float(np.nanmax(a)) if np.any(np.isfinite(a)) else float("nan")
    return {
        "sigma_dir": float(sigma_dir),
        "ladder_sigma_units": ladder.tolist(),
        "h": h.tolist(),
        "f0": f0,
        "f_plus": f_plus.tolist(),
        "f_minus": f_minus.tolist(),
        "D2": D2.tolist(),
        "roundoff_model_D2": roundoff.tolist(),
        "empirical_floor_D2": emp_floor.tolist(),
        "eps_f_empirical": eps_emp,
        "eps_f_theory": abs(f0) * float(eps_mach),
        "ratio_D2_to_theory_floor": ratio_theory.tolist(),
        "ratio_D2_to_empirical_floor": ratio_emp.tolist(),
        "max_ratio_theory": _safe_nanmax(ratio_theory),
        "max_ratio_empirical": _safe_nanmax(ratio_emp),
    }


# ===========================================================================
# plots
# ===========================================================================

def plot_dev_profiles(per_arm, out_path):
    """Two-panel overlay of quad-fit residual profiles: top vs calm (one panel
    per group), all arms/pairs faint, centered on segment parameter t in [0,1]."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    tt = np.linspace(0, 1, N_SEG_POINTS)
    colors = {"old": "C0", "new": "C1"}
    for gi, grp in enumerate(["top", "calm"]):
        ax = axes[gi]
        for arm, d in per_arm.items():
            R = d[grp]["resid"]
            for k in range(R.shape[0]):
                ax.plot(tt, R[k], color=colors[arm], lw=0.6, alpha=0.35)
            ax.plot([], [], color=colors[arm], label=f"{arm}")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axhline(DEV_THRESH_NAT, color="crimson", lw=0.8, ls="--")
        ax.axhline(-DEV_THRESH_NAT, color="crimson", lw=0.8, ls="--")
        ax.set_title(f"{grp} segments (residual = logp - quad fit)")
        ax.set_xlabel("segment parameter t (0=draw t-1, 1=draw t)")
        if gi == 0:
            ax.set_ylabel("logp residual (nats)")
        ax.legend(fontsize=8)
    fig.suptitle("T20 in-step logp micro-texture: quad-fit residual profiles "
                 "(dashed = +/-1 nat registered threshold)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_maxdev_hist(per_arm, out_path):
    """Histogram of per-pair max-deviation by group/arm (log x)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ai, arm in enumerate(per_arm):
        ax = axes[ai]
        d = per_arm[arm]
        bins = np.geomspace(1e-6, max(1.0, d["top"]["maxdev"].max() + 1e-9,
                                      d["calm"]["maxdev"].max() + 1e-9), 30)
        ax.hist(np.clip(d["top"]["maxdev"], 1e-6, None), bins=bins, alpha=0.6,
                label="top", color="C3")
        ax.hist(np.clip(d["calm"]["maxdev"], 1e-6, None), bins=bins, alpha=0.6,
                label="calm", color="C2")
        ax.axvline(DEV_THRESH_NAT, color="crimson", lw=1, ls="--",
                   label=f"{DEV_THRESH_NAT} nat")
        ax.set_xscale("log")
        ax.set_title(f"{arm} arm")
        ax.set_xlabel("max |logp - quad fit| per segment (nats)")
        if ai == 0:
            ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("T20 per-segment max deviation (top vs calm)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_micro_ladders(micro, out_path):
    """D2 vs h log-log for each (center, direction) with roundoff overlays."""
    items = list(micro.items())
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    axes = axes[0]
    for k, (name, r) in enumerate(items):
        ax = axes[k]
        h = np.array(r["h"]); D2 = np.array(r["D2"])
        ax.loglog(h, np.abs(D2), ".-", ms=4, lw=1, label="|D2| measured")
        ax.loglog(h, np.array(r["roundoff_model_D2"]), "--", color="crimson",
                  lw=1, label="roundoff 4*eps_f/h^2 (theory)")
        emp = np.array(r["empirical_floor_D2"])
        if np.any(emp > 0):
            ax.loglog(h, emp, ":", color="orange", lw=1, label="empirical floor")
        ax.set_title(f"{name}\nsigma_dir={r['sigma_dir']:.2g} "
                     f"maxR_emp={r['max_ratio_empirical']:.1f}", fontsize=8)
        ax.set_xlabel("h (z-move along e)")
        if k == 0:
            ax.set_ylabel("|D2|"); ax.legend(fontsize=7)
    fig.suptitle("T20 micro-transect D2 ladders (old arm): escape dir & z9 axis, "
                 "centered at z_best and a bulk draw", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# per-arm driver
# ===========================================================================

def run_arm(arm, f_eval_factory, out_dir, rng, do_micro=False, z_center=None):
    ref = REF_MCLMC[arm]
    samples_z = np.load(os.path.join(ref, "arrays.npz"))["samples_z"].astype(np.float64)
    xi = np.load(os.path.join(ref, "diagnostics.npz"))["xi"]

    frac, ok = alignment_gate(arm, xi)
    print(f"[t20:{arm}] alignment gate frac(xi>10)={frac:.4f} "
          f"expected={ALIGN_FRAC[arm]} -> {'OK' if ok else 'ABORT'}")
    if not ok:
        raise RuntimeError(f"[t20:{arm}] alignment gate FAILED: {frac:.4f} vs "
                           f"{ALIGN_FRAC[arm]} (tol {ALIGN_TOL}). ABORT.")

    cc, tt, xi_end = candidate_pool(arm, samples_z, xi)
    sel = select_pairs(cc, tt, xi_end, rng)
    print(f"[t20:{arm}] pool={len(cc)} p_hi(xi)={sel['p_hi']:.2f} "
          f"median(xi)={sel['median']:.4f}")

    flat = samples_z.reshape(-1, samples_z.shape[-1])
    pooled_sigma = flat.std(axis=0)

    f_eval = f_eval_factory()

    out = {"frac_xi_gt10": frac, "align_ok": ok,
           "p_hi_xi": sel["p_hi"], "median_xi": sel["median"],
           "pooled_sigma": pooled_sigma.tolist()}
    group_arrays = {}
    for grp in ("top", "calm"):
        idx = sel[grp]
        res = eval_group(f_eval, samples_z, cc, tt, idx, seg_key="A")
        c_arr = cc[idx]; t_arr = tt[idx]
        seglen = np.array([seg_length_pooled_sigma(
            samples_z[int(c_arr[k]), int(t_arr[k]) - 1],
            samples_z[int(c_arr[k]), int(t_arr[k])], pooled_sigma)
            for k in range(len(idx))])
        res["seglen_sigma"] = seglen
        res["c"] = c_arr; res["t"] = t_arr; res["xi_end"] = xi_end[idx]
        group_arrays[grp] = res
        frac_dev = float(np.mean(res["maxdev"] >= DEV_THRESH_NAT))
        out[f"{grp}_frac_deviate"] = frac_dev
        out[f"{grp}_maxdev_median"] = float(np.median(res["maxdev"]))
        out[f"{grp}_maxdev_max"] = float(np.max(res["maxdev"]))
        out[f"{grp}_seglen_median"] = float(np.median(seglen))
        print(f"[t20:{arm}] {grp}: frac(maxdev>={DEV_THRESH_NAT})={frac_dev:.3f} "
              f"median_maxdev={out[f'{grp}_maxdev_median']:.3g} "
              f"median_seglen(sigma)={out[f'{grp}_seglen_median']:.3g}")

    # OTHER alignment convention on the TOP pairs: segment (t, t+1)
    resB = eval_group(f_eval, samples_z, cc, tt, sel["top"], seg_key="B")
    group_arrays["topB"] = resB
    validB = resB["valid"]
    if validB.any():
        fracB = float(np.mean(resB["maxdev"][validB] >= DEV_THRESH_NAT))
    else:
        fracB = float("nan")
    out["topB_frac_deviate"] = fracB
    out["topB_n_valid"] = int(validB.sum())

    # registered verdict (P-T20d + falsifier F-T20d)
    top_frac = out["top_frac_deviate"]; calm_frac = out["calm_frac_deviate"]
    meets_top = top_frac >= P_T20D_TOPFRAC
    meets_calm = calm_frac < P_T20D_CALMFRAC
    fires = top_frac <= (calm_frac + F_T20D_MARGIN)
    out["P_T20d_meets_top"] = bool(meets_top)
    out["P_T20d_meets_calm"] = bool(meets_calm)
    out["P_T20d_meets"] = bool(meets_top and meets_calm)
    out["F_T20d_fires"] = bool(fires)
    # convention disagreement check (convention A vs B on the P-T20d top clause)
    convA = top_frac >= P_T20D_TOPFRAC
    convB = (fracB >= P_T20D_TOPFRAC) if np.isfinite(fracB) else convA
    out["convention_disagreement"] = bool(convA != convB)
    print(f"[t20:{arm}] P-T20d meets={out['P_T20d_meets']} "
          f"(top>={P_T20D_TOPFRAC}:{meets_top} calm<{P_T20D_CALMFRAC}:{meets_calm}) "
          f"F-T20d fires={fires} | convA_top={top_frac:.3f} convB_top={fracB:.3f} "
          f"disagree={out['convention_disagreement']}")
    if out["convention_disagreement"]:
        print(f"[t20:{arm}] *** LOUD: alignment conventions DISAGREE on P-T20d "
              f"top clause (A={top_frac:.3f} vs B={fracB:.3f}). Do NOT pick one.")

    # micro-transects (old arm only)
    micro = {}
    if do_micro:
        cov = np.load(CLONE[arm])["cov"].astype(np.float64)
        bulk_mean = flat.mean(axis=0)
        if z_center is None:
            raise ValueError("z_center required for micro-transects")
        z_best = np.asarray(z_center, dtype=np.float64)
        # bulk draw excluding basin (old): random valid candidate row
        draw_i = int(rng.choice(len(cc)))
        z_bulk = samples_z[int(cc[draw_i]), int(tt[draw_i])]
        escape = bulk_mean - z_best
        e9 = np.zeros(samples_z.shape[-1]); e9[Z9_INDEX] = 1.0
        eps_mach = 2.2e-16
        dirs = {"escape": escape, "z9axis": e9}
        centers = {"zbest": z_best, "bulk": z_bulk}
        for cname, z0 in centers.items():
            for dname, e in dirs.items():
                micro[f"{cname}|{dname}"] = micro_transect(f_eval, z0, e, cov, eps_mach)
                r = micro[f"{cname}|{dname}"]
                print(f"[t20:{arm}] micro {cname}|{dname}: sigma_dir="
                      f"{r['sigma_dir']:.3g} max|D2|/emp_floor="
                      f"{r['max_ratio_empirical']:.2f} "
                      f"max|D2|/theory={r['max_ratio_theory']:.2f}")
        out["micro_bulk_draw"] = {"flat_pool_index": draw_i,
                                  "chain": int(cc[draw_i]), "step": int(tt[draw_i])}
    return out, group_arrays, micro


# ===========================================================================
# main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T20 in-step micro-texture probe")
    p.add_argument("--arm", choices=["old", "new", "both"], default="both")
    p.add_argument("--out-dir",
                   default=os.path.join(_HERE, "results_carousel", "phaseC", "t20"))
    p.add_argument("--dry-run", action="store_true",
                   help="numpy-only: gate + pair selection counts, NO jax/logp")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    arms = ["old", "new"] if args.arm == "both" else [args.arm]
    rng = np.random.default_rng(SEED)

    if args.dry_run:
        for arm in arms:
            ref = REF_MCLMC[arm]
            samples_z = np.load(os.path.join(ref, "arrays.npz"))["samples_z"]
            xi = np.load(os.path.join(ref, "diagnostics.npz"))["xi"]
            frac, ok = alignment_gate(arm, xi)
            cc, tt, xi_end = candidate_pool(arm, samples_z, xi)
            sel = select_pairs(cc, tt, xi_end, np.random.default_rng(SEED))
            print(f"[dry-run:{arm}] frac(xi>10)={frac:.4f} ok={ok} pool={len(cc)} "
                  f"p_hi={sel['p_hi']:.2f} med={sel['median']:.4f} "
                  f"top={len(sel['top'])} calm={len(sel['calm'])}")
        return

    # jax precision guard (science = float64; env control skips it)
    import jax
    f32_control = os.environ.get("WHTS_FLOAT32_CONTROL") == "1"
    if not f32_control:
        from common import assert_x64
        assert_x64()
        print("[t20] float64 asserted")
    else:
        print("[t20] WHTS_FLOAT32_CONTROL=1: x64 assert SKIPPED (instrument arm)")

    from common import load_target

    all_summ = {}
    all_arrays = {}
    all_micro = {}
    for arm in arms:
        prob_model, qz, z_center, dim, param_names = load_target(SYSTEM_DIR[arm])
        do_micro = (arm == "old")
        summ, garr, micro = run_arm(
            arm, lambda pm=prob_model: make_logp_eval(pm), args.out_dir, rng,
            do_micro=do_micro, z_center=z_center)
        summ["dim"] = int(dim)
        summ["param_names"] = param_names
        all_summ[arm] = summ
        all_arrays[arm] = garr
        if micro:
            all_micro.update({f"{arm}:{k}": v for k, v in micro.items()})

    # ---- npz (all per-pair/per-point arrays) ----
    npz = {}
    for arm, garr in all_arrays.items():
        for grp, res in garr.items():
            for key in ("segf", "maxdev", "resid", "dlogp", "valid"):
                if key in res:
                    npz[f"{arm}_{grp}_{key}"] = np.asarray(res[key])
            for key in ("seglen_sigma", "c", "t", "xi_end"):
                if key in res:
                    npz[f"{arm}_{grp}_{key}"] = np.asarray(res[key])
    for name, r in all_micro.items():
        tag = name.replace(":", "_").replace("|", "_")
        for key in ("h", "D2", "f_plus", "f_minus", "roundoff_model_D2",
                    "empirical_floor_D2", "ratio_D2_to_theory_floor",
                    "ratio_D2_to_empirical_floor", "ladder_sigma_units"):
            npz[f"micro_{tag}_{key}"] = np.asarray(r[key])
    npz_path = os.path.join(args.out_dir, "t20_segments.npz")
    np.savez(npz_path, **npz)
    print(f"[t20] wrote {npz_path}")

    # ---- plots ----
    per_arm_plot = {arm: all_arrays[arm] for arm in arms}
    plot_dev_profiles(per_arm_plot,
                      os.path.join(args.out_dir, "t20_dev_profiles.png"))
    plot_maxdev_hist(per_arm_plot,
                     os.path.join(args.out_dir, "t20_maxdev_hist.png"))
    if all_micro:
        plot_micro_ladders(all_micro,
                           os.path.join(args.out_dir, "t20_micro_ladders.png"))

    # ---- summary json ----
    doc = {
        "experiment": "T20 in-step log-posterior micro-texture probe (carousel)",
        "status": "proposed (UNCERTIFIED) -- grader/orchestrator adjudicates",
        "timestamp_utc": _now(),
        "seed": SEED,
        "registered_constants": {
            "XI_SPIKE_THRESH": XI_SPIKE_THRESH, "ALIGN_FRAC": ALIGN_FRAC,
            "ALIGN_TOL": ALIGN_TOL, "N_PAIRS": N_PAIRS, "TOP_PCT": TOP_PCT,
            "BASIN_CHAIN": BASIN_CHAIN, "BASIN_TSTEP": BASIN_TSTEP,
            "N_SEG_POINTS": N_SEG_POINTS, "DEV_THRESH_NAT": DEV_THRESH_NAT,
            "P_T20D_TOPFRAC": P_T20D_TOPFRAC, "P_T20D_CALMFRAC": P_T20D_CALMFRAC,
            "F_T20D_MARGIN": F_T20D_MARGIN, "Z9_INDEX": Z9_INDEX,
            "H_LADDER_LO": H_LADDER_LO, "H_LADDER_HI": H_LADDER_HI, "N_H": N_H,
        },
        "arms": all_summ,
        "micro_transects": all_micro,
        "notes": ("P-T20d MEETS iff top_frac_deviate>=0.5 AND calm_frac_deviate<0.1; "
                  "F-T20d FIRES iff top_frac<=calm_frac+0.05. Micro-transect verdict "
                  "(roughness above roundoff floor) is left to the orchestrator using "
                  "these ratios + the t3 run. Both alignment conventions reported; if "
                  "convention_disagreement is true the P-T20d top clause is convention-"
                  "sensitive and must NOT be collapsed to one."),
    }
    json_path = os.path.join(args.out_dir, "t20_summary.json")
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[t20] wrote {json_path}")
    print("[t20] done (PROPOSED / UNCERTIFIED).")


if __name__ == "__main__":
    main()
