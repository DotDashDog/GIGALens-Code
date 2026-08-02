"""T21 -- "typical-set init" arm for the two carousel minimal-case posteriors.

Question (docs/logs/why-hard-to-sample.md, T21): every prior arm started MCLMC
from the tight MAP-centered ball (qz = 1e-2/1e-3 diag on z_best). If the real
posteriors are hard to sample *only* because the MAP init sits off the typical
set (T18: MAP loglike ~6 nats below the bulk median), then STARTING chains inside
the typical set should make the real runs as easy as their Gaussian clones. If
the real runs stay much harder than their matched clones even from a typical-set
init, the hardness is intrinsic to the density's fine structure, not the init.

Design (per arm):
  1. z_init selection (fixed seed): subsample 512 reference post-burn-in draws,
     evaluate joint logp per draw (CHUNKED, <=16 rows -- carousel renders are
     ~52MB/pt of intermediates and larger batches OOM 40GB), and pick the draw
     whose logp is CLOSEST TO THE MEDIAN of the 512 ("maximally typical"). The
     best-logp draw is recorded for the ledger but is NOT used as init.
  2. Real runs: qz' = MVNDiag(loc=z_init, scale_diag=1e-3). Seeds 1,2,3 ->
     STANDARD MCLMC. Diagnostics + displaced-chain census (T16) + xi tail stats.
  3. Matched clone control: the arm's existing Gaussian clone (build_clone.py fit
     to the real run's z-space second moments), run for 1 seed with the SAME qz'
     recentered at z_init. Same sampler, same init geometry, smooth-Gaussian
     target -- the "how easy is this init supposed to be" yardstick.

Registered verdicts are constants below (meets/fires printed, never tuned).
Everything is marked "proposed (UNCERTIFIED)".

Reuse:
  * _displaced_census      <- t18_map_arm  (T16 rule, verbatim; not duplicated)
  * build_clone_target     <- run_t1_clone (the Gaussian-clone stub whose
                              .prob_model.log_prob(z)->(logpdf,()) MCLMC consumes;
                              run_t1_clone.py:28. We reuse ONLY that stub builder
                              and supply our own typical-set qz', exactly as the
                              T21 spec requires -- run_t1_clone.py is unchanged.)
  * run_standard_mclmc / compute_diagnostics / load_target <- common

The system module reads WHTS_CONV_PRECISION from the env (set by slurm/run_t21.sh
to float64); this driver does NOT set it in python.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HARNESS = os.path.dirname(os.path.abspath(__file__))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

from common import (  # noqa: E402
    assert_x64, compute_diagnostics, git_commit, load_target,
    print_diagnostics, run_standard_mclmc,
)
from exp_config import STANDARD  # noqa: E402
from t18_map_arm import _displaced_census, CENSUS_SIGMA  # noqa: E402
from run_t1_clone import build_clone_target  # noqa: E402

# ---------------------------------------------------------------------------
# fixed inputs / paths
# ---------------------------------------------------------------------------
OLD_SYS = os.path.join(HARNESS, "systems", "carousel_min_old")
NEW_SYS = os.path.join(HARNESS, "systems", "carousel_min_new")
_SIM = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests"
REF_OLD_SAMPLES = os.path.join(_SIM, "minimal_case_oldbij", "mclmc", "arrays.npz")
REF_NEW_SAMPLES = os.path.join(_SIM, "minimal_case_newbij", "mclmc", "arrays.npz")
CLONE_OLD = os.path.join(HARNESS, "results_carousel", "old", "t1", "clone.npz")
CLONE_NEW = os.path.join(HARNESS, "results_carousel", "new", "t1", "clone.npz")

# ---------------------------------------------------------------------------
# selection / run constants (pre-registered; do NOT tune)
# ---------------------------------------------------------------------------
Z_INIT_SEED = 20260703          # fixed seed for the 512-draw subsample
N_SUB = 512                     # draws subsampled for the typical-set pick
LOGP_CHUNK = 16                 # HARD memory cap: <=16 rows/logp call (52MB/pt)
QZ_SCALE = 1e-3                 # qz' isotropic scale (tight ball at z_init)
SEEDS = [1, 2, 3]               # real-run seeds
CLONE_SEED = 1                  # single clone-control seed
XI_THRESH = 10.0                # xi-tail threshold (frac xi > 10)
NUM_RESULTS = STANDARD.num_results   # post-burn-in half length (2000); xi slice [:, -NR:]

# ---------------------------------------------------------------------------
# REGISTERED verdicts (constants; meets/fires printed, never tuned)
# ---------------------------------------------------------------------------
P_T21A_BAND = (30.0, 400.0)     # old-arm min-ESS band
F_T21A_EASY_FLOOR = 1000.0      # F_T21a fires (old is "easy") if min(min-ESS) >= this
P_T21A_CENSUS_MAX = 0           # old arm: predicted 0/3 seeds with persistent displaced chains
P_T21B_BAND = (30.0, 300.0)     # new-arm min-ESS band
CLONE_GAP_FACTOR = {"old": 5.0, "new": 3.0}   # real >= Nx below its matched clone
P_T21C_XI_TOL = 0.03            # |frac(xi>10) - reference| <= this for the real runs
REF_XI_FRAC = {"old": 0.1235, "new": 0.1037}  # reference frac(xi>10), post-burn-in
XI_GONE_THRESH = 0.01           # tail deemed "GONE" (loud contradiction) below this
# T22 accounting: conv precision contributes xi ~ 1e-6, so an f64-conv run that
# ERASES the reference xi tail contradicts "the tail is intrinsic, not conv noise".

ARMS = {
    "old": {"sys": OLD_SYS, "ref": REF_OLD_SAMPLES, "clone": CLONE_OLD,
            "chains": list(range(1, 8)),   # chain 0 has a burn-in transient -> excluded
            "band": P_T21A_BAND},
    "new": {"sys": NEW_SYS, "ref": REF_NEW_SAMPLES, "clone": CLONE_NEW,
            "chains": list(range(0, 8)),   # all chains
            "band": P_T21B_BAND},
}


# ===========================================================================
# z_init selection (pure-numpy pool bookkeeping + chunked jax logp)
# ===========================================================================

def load_ref_pool(ref_path, chains):
    """Load reference samples_z (8, 10000, dim), keep `chains`, flatten to a
    (N, dim) pool, and return per-pool-row (orig_chain, t) index arrays so the
    ledger can name the median/best draw by its ORIGINAL (chain, t)."""
    d = np.load(ref_path)
    if "samples_z" not in d.files:
        raise KeyError(f"{ref_path} has no 'samples_z'; keys={d.files}")
    sz = np.asarray(d["samples_z"], dtype=np.float64)      # (chains, T, dim)
    if sz.ndim != 3:
        raise ValueError(f"samples_z must be 3-D; got {sz.shape}")
    bad = [c for c in chains if c < 0 or c >= sz.shape[0]]
    if bad:
        raise ValueError(f"chain indices {bad} out of range for {sz.shape[0]} chains")
    sub = sz[list(chains)]                                  # (C, T, dim)
    C, T, dim = sub.shape
    pool = sub.reshape(-1, dim)                             # (C*T, dim)
    orig_chain = np.repeat(np.asarray(chains, dtype=int), T)
    t_idx = np.tile(np.arange(T, dtype=int), C)
    return pool, orig_chain, t_idx, dim


def logp_chunked(prob_model, z, chunk=LOGP_CHUNK):
    """Joint logp (prob_model.log_prob(z)[0]) per row of z, in chunks of <=`chunk`
    rows. HARD memory requirement (carousel renders ~52MB/pt of intermediates;
    larger batches OOM the 40GB card)."""
    import jax.numpy as jnp
    out = np.empty(z.shape[0], dtype=np.float64)
    for s in range(0, z.shape[0], chunk):
        e = min(s + chunk, z.shape[0])
        lp, _ = prob_model.log_prob(jnp.asarray(z[s:e]))
        out[s:e] = np.asarray(lp, dtype=np.float64).reshape(-1)
    return out


def select_typical_init(prob_model, ref_path, chains, seed=Z_INIT_SEED):
    """Subsample N_SUB draws (fixed seed), evaluate chunked logp, and return the
    draw whose logp is CLOSEST TO THE MEDIAN of the subsample ("maximally
    typical"). Records the median AND best draws (chain,t,logp) for the ledger;
    the best draw is NOT used as init."""
    pool, orig_chain, t_idx, dim = load_ref_pool(ref_path, chains)
    rng = np.random.default_rng(seed)
    if N_SUB > pool.shape[0]:
        raise ValueError(f"N_SUB {N_SUB} exceeds pool size {pool.shape[0]}")
    sub_idx = rng.choice(pool.shape[0], N_SUB, replace=False)
    z512 = pool[sub_idx]
    lp512 = logp_chunked(prob_model, z512)                  # (N_SUB,)
    med = float(np.median(lp512))
    j_med = int(np.argmin(np.abs(lp512 - med)))             # closest to median
    j_best = int(np.argmax(lp512))
    z_init = np.asarray(z512[j_med], dtype=np.float64).copy()

    def _draw(j):
        r = int(sub_idx[j])
        return {"chain": int(orig_chain[r]), "t": int(t_idx[r]),
                "logp": float(lp512[j])}

    ledger = {
        "z_init_selection_seed": int(seed),
        "n_sub": int(N_SUB), "logp_chunk": int(LOGP_CHUNK),
        "median_logp_of_subsample": med,
        "subsample_logp_min": float(lp512.min()),
        "subsample_logp_max": float(lp512.max()),
        "median_draw_used_as_init": _draw(j_med),          # <- z_init
        "best_draw_NOT_used": _draw(j_best),               # recorded, not init
        "abs_logp_gap_median_draw_to_median": float(abs(lp512[j_med] - med)),
        "ref_samples": os.path.abspath(ref_path),
        "chains_used": list(chains),
    }
    return z_init, ledger, dim


# ===========================================================================
# xi tail stats on a saved run npz (position/xi span burnin+results)
# ===========================================================================

def xi_stats(npz_path, thresh=XI_THRESH, num_results=NUM_RESULTS):
    """frac(xi>thresh) + p50/p90/p99/max over the POST-BURN-IN half.

    run_standard_mclmc saves xi with shape (chains, burnin+results) -- the same
    (chains, steps) layout as `position` (common.py: xi = np.asarray(hist.xi),
    and mclmc.py moveaxis(0,1) puts chains first). So the post-burn-in half is
    xi[:, -num_results:] (matches compute_diagnostics' position[:, -nr:, :])."""
    d = np.load(npz_path)
    xi = np.asarray(d["xi"], dtype=np.float64)
    if xi.ndim != 2:
        raise ValueError(f"xi must be (chains, steps); got {xi.shape} in {npz_path}")
    post = xi[:, -num_results:]
    flat = post.reshape(-1)
    return {
        "frac_gt10": float(np.mean(flat > thresh)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
        "n_post": int(flat.size),
        "num_results_used": int(post.shape[1]),
    }


# ===========================================================================
# per-arm run
# ===========================================================================

def run_arm(arm, stage, out_root):
    cfg = ARMS[arm]
    out_dir = os.path.join(out_root, arm)
    os.makedirs(out_dir, exist_ok=True)

    assert_x64()
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    prob_model, _ref_qz, z_center, dim, param_names = load_target(cfg["sys"])

    # --- z_init: maximally typical draw (closest-to-median logp) ---
    z_init, ledger, sel_dim = select_typical_init(
        prob_model, cfg["ref"], cfg["chains"], Z_INIT_SEED)
    if sel_dim != dim:
        raise ValueError(f"selection dim {sel_dim} != target dim {dim}")
    dev = z_init - np.asarray(z_center, dtype=np.float64)
    print(f"\n[T21 {arm}] z_init = maximally-typical draw "
          f"(chain {ledger['median_draw_used_as_init']['chain']}, "
          f"t {ledger['median_draw_used_as_init']['t']}, "
          f"logp {ledger['median_draw_used_as_init']['logp']:.3f}); "
          f"subsample median logp {ledger['median_logp_of_subsample']:.3f}")
    print(f"[T21 {arm}] best draw (NOT init): chain "
          f"{ledger['best_draw_NOT_used']['chain']}, t "
          f"{ledger['best_draw_NOT_used']['t']}, logp "
          f"{ledger['best_draw_NOT_used']['logp']:.3f}")
    print(f"[T21 {arm}] max|z_init - MAP center| = {np.max(np.abs(dev)):.4g} "
          f"(qz' scale_diag = {QZ_SCALE})")

    qz_prime = tfd.MultivariateNormalDiag(
        loc=jnp.asarray(z_init), scale_diag=jnp.full(dim, QZ_SCALE))

    arm_result = {
        "arm": arm, "dim": dim, "qz_scale": QZ_SCALE,
        "z_init_ledger": ledger,
        "max_abs_dev_from_map_center": float(np.max(np.abs(dev))),
        "census_sigma": CENSUS_SIGMA,
    }

    # --- real runs (seeds 1,2,3) ---
    if stage in ("real", "both"):
        per_seed = []
        for seed in SEEDS:
            out_npz = os.path.join(out_dir, f"t0_seed{seed}.npz")
            pos = run_standard_mclmc(
                prob_model, qz_prime, STANDARD, seed, out_npz,
                target_desc=f"carousel_min_{arm}, typical-set init (T21)",
                provenance={"system": cfg["sys"], "arm": arm,
                            "qz": f"MVNDiag(loc=z_init, scale_diag={QZ_SCALE})",
                            "z_init_median_draw": ledger["median_draw_used_as_init"]},
            )
            diag = compute_diagnostics(pos, STANDARD.num_results, param_names)
            print_diagnostics(diag, header=f"T21 {arm} real seed {seed}")
            census = _displaced_census(pos, STANDARD.num_results)
            xis = xi_stats(out_npz)
            print(f"[T21 {arm}] seed {seed} displaced-chain census: persistent "
                  f"chains={census['persistent_chains']} (n={census['n_persistent']}); "
                  f"frac(xi>10)={xis['frac_gt10']:.4f} "
                  f"(p99={xis['p99']:.3g} max={xis['max']:.3g})")
            with open(os.path.splitext(out_npz)[0] + ".census.json", "w") as f:
                json.dump(census, f, indent=2)
            per_seed.append({
                "seed": seed, "npz": os.path.abspath(out_npz),
                "min_ess": diag["min_ess"], "min_ess_param": diag["min_ess_param"],
                "max_rhat": diag["max_rhat"], "max_rhat_param": diag["max_rhat_param"],
                "n_persistent": census["n_persistent"],
                "persistent_chains": census["persistent_chains"],
                "xi": xis,
            })
        arm_result["real"] = {"per_seed": per_seed}

    # --- matched clone control (seed 1, SAME qz') ---
    if stage in ("clone", "both"):
        clone_stub, clone_dim = build_clone_target(cfg["clone"])
        if clone_dim != dim:
            raise ValueError(f"clone dim {clone_dim} != target dim {dim}")
        out_npz = os.path.join(out_dir, f"t1_clone_typical_seed{CLONE_SEED}.npz")
        pos = run_standard_mclmc(
            clone_stub, qz_prime, STANDARD, CLONE_SEED, out_npz,
            target_desc=(f"Gaussian clone of carousel_min_{arm} real run "
                         "(full-cov z-space fit), typical-set init (T21)"),
            provenance={"clone_npz": os.path.abspath(cfg["clone"]), "arm": arm,
                        "qz": f"SAME qz' as real: MVNDiag(loc=z_init, scale_diag={QZ_SCALE})",
                        "z_init_median_draw": ledger["median_draw_used_as_init"]},
        )
        diag = compute_diagnostics(pos, STANDARD.num_results, param_names)
        print_diagnostics(diag, header=f"T21 {arm} clone (typical init) seed {CLONE_SEED}")
        xis = xi_stats(out_npz)
        print(f"[T21 {arm}] clone frac(xi>10)={xis['frac_gt10']:.4f} "
              f"(p99={xis['p99']:.3g} max={xis['max']:.3g})")
        arm_result["clone"] = {
            "seed": CLONE_SEED, "npz": os.path.abspath(out_npz),
            "min_ess": diag["min_ess"], "min_ess_param": diag["min_ess_param"],
            "max_rhat": diag["max_rhat"], "max_rhat_param": diag["max_rhat_param"],
            "xi": xis,
        }

    return arm_result


# ===========================================================================
# registered verdicts + interpretation matrix
# ===========================================================================

def arm_verdicts(arm, arm_result):
    cfg = ARMS[arm]
    real = arm_result.get("real")
    clone = arm_result.get("clone")
    v = {"arm": arm}

    if real:
        min_ess_seeds = [s["min_ess"] for s in real["per_seed"]]
        min_min_ess = float(min(min_ess_seeds))
        lo, hi = cfg["band"]
        seeds_persistent = [s["seed"] for s in real["per_seed"] if s["n_persistent"] >= 1]
        fracs = [s["xi"]["frac_gt10"] for s in real["per_seed"]]
        mean_frac = float(np.mean(fracs))
        ref_frac = REF_XI_FRAC[arm]
        xi_dev = abs(mean_frac - ref_frac)
        v.update({
            "min_ess_per_seed": min_ess_seeds,
            "min_min_ess": min_min_ess,
            "band": [lo, hi], "in_band": bool(lo <= min_min_ess <= hi),
            "seeds_with_persistent_displaced": seeds_persistent,
            "n_seeds_persistent": len(seeds_persistent),
            "census_meets_0of3": bool(len(seeds_persistent) == P_T21A_CENSUS_MAX),
            "xi_frac_per_seed": fracs, "xi_frac_mean": mean_frac,
            "ref_xi_frac": ref_frac, "xi_abs_dev_from_ref": xi_dev,
            "P_T21C_xi_tail_same": bool(xi_dev <= P_T21C_XI_TOL),
            "xi_tail_gone": bool(mean_frac < XI_GONE_THRESH),
        })
        if arm == "old":
            v["F_T21a_easy_fires"] = bool(min_min_ess >= F_T21A_EASY_FLOOR)

    if clone:
        v["clone_min_ess"] = float(clone["min_ess"])

    if real and clone:
        factor = CLONE_GAP_FACTOR[arm]
        gap = (float(clone["min_ess"]) / min_min_ess) if min_min_ess > 0 else float("inf")
        v.update({
            "clone_gap_ratio": gap, "clone_gap_factor_required": factor,
            "clone_gap_fires_real_much_harder": bool(gap >= factor),
        })

    # interpretation matrix line
    if real and not v["P_T21C_xi_tail_same"]:
        interp = "CONTRADICTION: tail changed under f64 conv"
    elif real and clone:
        interp = ("hardness co-occurs with tail (causality open)"
                  if v["clone_gap_fires_real_much_harder"]
                  else "tail inconsequential")
    else:
        interp = "(pending: need both real and clone stages)"
    v["interpretation"] = interp
    return v


def print_verdicts(v):
    arm = v["arm"]
    print(f"\n----- T21 {arm} registered verdicts -----")
    if "min_min_ess" in v:
        band = v["band"]
        print(f"  min-ESS band {tuple(band)}: min(min-ESS)={v['min_min_ess']:.4g} -> "
              f"{'IN band' if v['in_band'] else 'OUTSIDE band'}")
        print(f"  census 0/{len(SEEDS)}: seeds with persistent displaced chains="
              f"{v['seeds_with_persistent_displaced']} -> "
              f"{'MEETS (0)' if v['census_meets_0of3'] else 'DOES NOT MEET'}")
        if arm == "old":
            print(f"  F_T21a easy floor (min-ESS >= {F_T21A_EASY_FLOOR}): "
                  f"{'FIRES (easy)' if v['F_T21a_easy_fires'] else 'does not fire'}")
        print(f"  P_T21c xi tail: frac(xi>10) mean={v['xi_frac_mean']:.4f} vs "
              f"ref {v['ref_xi_frac']:.4f} (|dev|={v['xi_abs_dev_from_ref']:.4f}, "
              f"tol {P_T21C_XI_TOL}) -> "
              f"{'SAME' if v['P_T21C_xi_tail_same'] else 'CHANGED'}")
        if v["xi_tail_gone"]:
            print(f"  ***** LOUD CONTRADICTION: xi tail is GONE under f64 conv "
                  f"(frac(xi>10)={v['xi_frac_mean']:.4f} < {XI_GONE_THRESH}); T22 "
                  f"accounting says conv contributes xi~1e-6 -> the reference tail "
                  f"was NOT conv noise, yet it vanished. INVESTIGATE. *****")
    if "clone_gap_ratio" in v:
        print(f"  clone gap: clone min-ESS={v['clone_min_ess']:.4g} / real min(min-ESS)="
              f"{v['min_min_ess']:.4g} = {v['clone_gap_ratio']:.2f}x "
              f"(need >= {v['clone_gap_factor_required']}x) -> "
              f"{'FIRES (real much harder)' if v['clone_gap_fires_real_much_harder'] else 'does not fire'}")
    elif "clone_min_ess" in v:
        print(f"  clone min-ESS={v['clone_min_ess']:.4g} (real stage absent; gap N/A)")
    print(f"  INTERPRETATION [{arm}]: {v['interpretation']}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="T21 typical-set init arm (carousel).")
    ap.add_argument("--arm", choices=["old", "new", "both"], default="both")
    ap.add_argument("--stage", choices=["real", "clone", "both"], default="both")
    ap.add_argument("--out-dir",
                    default=os.path.join(HARNESS, "results_carousel", "phaseC", "t21"))
    args = ap.parse_args()

    arms = ["old", "new"] if args.arm == "both" else [args.arm]
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    verdicts = {}
    for arm in arms:
        arm_result = run_arm(arm, args.stage, args.out_dir)
        v = arm_verdicts(arm, arm_result)
        print_verdicts(v)
        results[arm] = arm_result
        verdicts[arm] = v

    summary = {
        "experiment": "T21_typical_init",
        "status": "proposed (UNCERTIFIED)",
        "stage": args.stage, "arms": arms,
        "config": STANDARD.to_dict(),
        "constants": {
            "Z_INIT_SEED": Z_INIT_SEED, "N_SUB": N_SUB, "LOGP_CHUNK": LOGP_CHUNK,
            "QZ_SCALE": QZ_SCALE, "SEEDS": SEEDS, "CLONE_SEED": CLONE_SEED,
            "XI_THRESH": XI_THRESH, "P_T21A_BAND": list(P_T21A_BAND),
            "F_T21A_EASY_FLOOR": F_T21A_EASY_FLOOR, "P_T21A_CENSUS_MAX": P_T21A_CENSUS_MAX,
            "P_T21B_BAND": list(P_T21B_BAND), "CLONE_GAP_FACTOR": CLONE_GAP_FACTOR,
            "P_T21C_XI_TOL": P_T21C_XI_TOL, "REF_XI_FRAC": REF_XI_FRAC,
            "XI_GONE_THRESH": XI_GONE_THRESH,
        },
        "results": results, "verdicts": verdicts,
        "conv_precision_env": os.environ.get("WHTS_CONV_PRECISION", "float32(default)"),
        "git_commit": git_commit(),
    }
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[T21] wrote {out_json}  (status: proposed (UNCERTIFIED))")


if __name__ == "__main__":
    main()
