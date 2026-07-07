"""T18 -- MAP-quality arm for the carousel minimal-case OLD system.

Question (docs/logs/why-hard-to-sample.md, T18): the T15-T17 line concluded the
old run's apparent "second mode" is just its UNDER-CONVERGED MAP init region
(loglike at the reference OLD z_best is -119479.5, i.e. 6.7 nats BELOW the bulk
sample median -119472.8; sample max -119466.8). T18 tests whether a BETTER MAP
(same optimizer, same seeded init, 10x the steps) removes that phenomenology.

Three stages (argparse --stage):

  map      -- replicate the reference MAP EXACTLY but with num_steps=5000 (vs
              500). Reuses gigalens/jax/inference.py:ModellingSequence.MAP (the
              same routine MAPStage wraps) and pipeline._default_map_optimizer
              (the same optimizer construction MAPStage uses by default), so the
              ONLY intentional difference from the reference run is num_steps.
              Saves z_best/lp_hist/chisq_hist + manifest with the loglike/logprior
              split, and prints the best-so-far trajectory at the reference's
              horizon (500) and beyond for a direct 500-vs-5000 comparison.

  quality  -- run the reusable D1/D2/D3 diagnostic (t18_map_quality) on THREE
              MAPs: reference old (500), reference new (500), improved t18 (5000).
              Prints a combined table and a CALIBRATION line: the trio MUST flag
              both reference 500-step MAPs (>=1 FAIL each) and pass the improved
              MAP (0 FAIL).

  seeds    -- for seeds 1,2,3: qz = MVNDiag(loc=improved z_best, scale_diag=1e-3)
              -- the SAME diag-1e-3 recipe as the reference runs, only re-centered
              on the improved MAP. Run the STANDARD MCLMC (8 chains, 2000/2000)
              per seed, compute min-ESS/max-Rhat, and run the T16 displaced-chain
              census. Tests whether the "second mode" recurs off the better MAP.

Registered predictions (encoded as constants; meets/fires printed, never tuned).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HARNESS = os.path.dirname(os.path.abspath(__file__))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

from common import (  # noqa: E402
    assert_x64, compute_diagnostics, git_commit, load_target,
    print_diagnostics, run_standard_mclmc,
)
from exp_config import STANDARD  # noqa: E402
import t18_map_quality as mapq  # noqa: E402

# ---------------------------------------------------------------------------
# fixed inputs / paths
# ---------------------------------------------------------------------------
OLD_SYS = os.path.join(HARNESS, "systems", "carousel_min_old")
NEW_SYS = os.path.join(HARNESS, "systems", "carousel_min_new")
_SIM = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests"
REF_OLD_MAP = os.path.join(_SIM, "minimal_case_oldbij", "map", "arrays.npz")
REF_NEW_MAP = os.path.join(_SIM, "minimal_case_newbij", "map", "arrays.npz")
REF_OLD_SAMPLES = os.path.join(_SIM, "minimal_case_oldbij", "mclmc", "arrays.npz")
REF_NEW_SAMPLES = os.path.join(_SIM, "minimal_case_newbij", "mclmc", "arrays.npz")

OUT_DIR = os.path.join(HARNESS, "results_carousel", "old", "t18")
IMPROVED_MAP = os.path.join(OUT_DIR, "map_arrays.npz")
IMPROVED_MANIFEST = os.path.join(OUT_DIR, "map_manifest.json")

# ---------------------------------------------------------------------------
# MAP replication constants (match the reference run except num_steps)
# ---------------------------------------------------------------------------
MAP_NUM_STEPS = 5000            # 10x the reference 500
MAP_N_SAMPLES = 64             # reference n_samples
MAP_SEED = 42                 # reference seed
REF_OPTIMIZER_ID = "adabelief_1e-2_b1_0.95_b2_0.99_nesterov"  # reference id
MAP_TRAJECTORY_STEPS = [100, 250, 500, 1000, 2000, 3500, 5000]
REF_OLD_BEST_LP = -119514.83136782396   # reference OLD 500-step MAP best_lp
REF_OLD_LOGLIKE_ZBEST = -119479.5       # established (T15b); for context only

# ---------------------------------------------------------------------------
# REGISTERED predictions (pre-registered; do NOT tune)
# ---------------------------------------------------------------------------
P_T18A_LOGLIKE_MIN = -119470.0          # improved-MAP loglike >= this
P_T18C_ESS_BAND = (25.0, 250.0)         # predicted min-ESS band
F_T18C_ESS_FLOOR = 1000.0               # F_T18c fires if min(min-ESS) >= this
CENSUS_SIGMA = 2.0                      # T16 displaced-chain threshold
SEEDS = [1, 2, 3]
QZ_SCALE = 1e-3                         # SAME diag recipe as reference runs


# ===========================================================================
# stage: map
# ===========================================================================

def stage_map():
    assert_x64()
    import jax
    # Reuse the EXACT optimizer construction MAPStage uses by default. Assert its
    # id equals the reference id -- if optax dropped nesterov support the built
    # optimizer would silently differ from the reference, so fail loudly.
    from gigalens_research.inference_utils.pipeline import (
        _default_map_optimizer, _DEFAULT_MAP_OPTIMIZER_ID)
    if _DEFAULT_MAP_OPTIMIZER_ID != REF_OPTIMIZER_ID:
        raise RuntimeError(
            f"optimizer id mismatch: pipeline default {_DEFAULT_MAP_OPTIMIZER_ID!r} "
            f"!= reference {REF_OPTIMIZER_ID!r}. The optax build differs from the "
            "one that produced the reference MAP; the optimizer would not match.")

    model_seq, _qz, _zc, dim, param_names = load_target(OLD_SYS)
    prob_model = model_seq.prob_model
    dev_cnt = len(jax.devices())
    eff_n = (MAP_N_SAMPLES // dev_cnt) * dev_cnt  # MAP's own rounding
    print(f"[T18 map] dim={dim} devices={dev_cnt} n_samples={MAP_N_SAMPLES}->{eff_n} "
          f"num_steps={MAP_NUM_STEPS} seed={MAP_SEED} optimizer_id={REF_OPTIMIZER_ID}")
    if eff_n != MAP_N_SAMPLES:
        print(f"[T18 map] WARNING: n_samples rounded {MAP_N_SAMPLES}->{eff_n} for "
              f"{dev_cnt} devices; reference used 64. Run on 1 GPU to match exactly.")

    t0 = time.perf_counter()
    # SAME call MAPStage.run makes (pipeline.py:1370), only num_steps changes.
    map_samples, map_lps, map_chisqs = model_seq.MAP(
        optimizer=_default_map_optimizer(),
        n_samples=MAP_N_SAMPLES,
        num_steps=MAP_NUM_STEPS,
        seed=MAP_SEED,
        output_type="best_step",
    )
    wall = time.perf_counter() - t0

    lp_hist = np.asarray(map_lps, dtype=np.float64).reshape(-1)
    chisq_hist = np.asarray(map_chisqs, dtype=np.float64).reshape(-1)
    samples_np = np.asarray(map_samples, dtype=np.float64)
    best = int(np.nanargmax(lp_hist))          # MAPStage best-selection (pipeline.py:1384)
    z_best = samples_np[best]
    if z_best.shape != (dim,):
        raise ValueError(
            f"z_best shape {z_best.shape} != ({dim},); MAP output_type='best_step' "
            "returned an unexpected layout — do not proceed to qz build.")
    best_lp = float(lp_hist[best])
    best_chisq = float(chisq_hist[best])

    # loglike/logprior split at z_best (the T18 headline: is the mode really high-loglike?)
    import jax.numpy as jnp
    zb = jnp.asarray(z_best)[None, :]
    loglike = float(np.asarray(prob_model.log_like(zb)[0]).reshape(-1)[0])
    logprior = float(np.asarray(prob_model.log_prior(zb)).reshape(-1)[0])

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(IMPROVED_MAP, z_best=z_best, lp_hist=lp_hist, chisq_hist=chisq_hist)

    # best-so-far trajectory for direct 500-vs-5000 comparison
    bsf = np.maximum.accumulate(lp_hist)
    traj = {}
    print("[T18 map] best-so-far logp trajectory (500 = reference horizon):")
    for s in MAP_TRAJECTORY_STEPS:
        if s <= lp_hist.size:
            traj[str(s)] = float(bsf[s - 1])
            tag = "  <- reference stopped here" if s == 500 else ""
            print(f"    step {s:>5}: best-so-far logp = {bsf[s-1]:.4f}{tag}")
    gain_500_to_end = float(bsf[-1] - bsf[min(499, lp_hist.size - 1)])
    print(f"[T18 map] logp gained after step 500: {gain_500_to_end:+.4f} nats "
          f"(reference best_lp={REF_OLD_BEST_LP:.4f})")

    meets_a = loglike >= P_T18A_LOGLIKE_MIN
    print(f"[T18 map] loglike(z_best) = {loglike:.4f} ; logprior = {logprior:.4f} ; "
          f"logp = {best_lp:.4f}")
    print(f"[T18 map] P_T18a (loglike >= {P_T18A_LOGLIKE_MIN}) -> "
          f"{'MEETS' if meets_a else 'FAILS'}  "
          f"(reference loglike ~{REF_OLD_LOGLIKE_ZBEST})")

    manifest = {
        "experiment": "T18_map_arm",
        "stage": "map",
        "status": "proposed (UNCERTIFIED)",
        "config": {
            "num_steps": MAP_NUM_STEPS, "n_samples": MAP_N_SAMPLES,
            "n_samples_effective": eff_n, "seed": MAP_SEED,
            "optimizer_id": REF_OPTIMIZER_ID, "output_type": "best_step",
            "system": OLD_SYS, "devices": dev_cnt,
        },
        "best_lp": best_lp, "best_chisq": best_chisq, "best_step": best,
        "loglike_z_best": loglike, "logprior_z_best": logprior,
        "wall_time_s": wall,
        "best_so_far_trajectory": traj,
        "logp_gain_after_step_500": gain_500_to_end,
        "reference_old_best_lp": REF_OLD_BEST_LP,
        "P_T18a_loglike_min": P_T18A_LOGLIKE_MIN,
        "P_T18a_meets": bool(meets_a),
        "arrays_npz": os.path.abspath(IMPROVED_MAP),
        "git_commit": git_commit(),
        "reuse": ("gigalens/jax/inference.py:ModellingSequence.MAP (output_type="
                  "'best_step'); optimizer via pipeline._default_map_optimizer; "
                  "best-selection np.nanargmax(lp_hist) as pipeline.MAPStage.run:1384"),
    }
    with open(IMPROVED_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[T18 map] wrote {IMPROVED_MAP}\n[T18 map] wrote {IMPROVED_MANIFEST}"
          f"\n[T18 map] wall {wall:.1f}s")


# ===========================================================================
# stage: quality  (D1/D2/D3 on all three MAPs + calibration)
# ===========================================================================

def stage_quality(old_map, new_map, improved_map,
                  old_samples, new_samples, improved_samples):
    assert_x64()
    os.makedirs(OUT_DIR, exist_ok=True)

    # old system serves BOTH the old reference MAP and the improved MAP.
    old_ms, _q, _zc, old_dim, _n = load_target(OLD_SYS)
    new_ms, _q, _zc, new_dim, _n = load_target(NEW_SYS)

    def _load_map(p):
        a = np.load(p)
        return (np.asarray(a["z_best"], dtype=np.float64),
                np.asarray(a["lp_hist"], dtype=np.float64))

    specs = [
        ("ref_old_500", old_ms.prob_model, old_map, old_samples, old_dim),
        ("ref_new_500", new_ms.prob_model, new_map, new_samples, new_dim),
        ("improved_5000", old_ms.prob_model, improved_map, improved_samples, old_dim),
    ]
    results = {}
    for label, pm, mp, sp, dim in specs:
        if not os.path.isfile(mp):
            print(f"[T18 quality] SKIP {label}: map arrays missing ({mp})")
            continue
        z_best, lp_hist = _load_map(mp)
        samples_z = mapq._load_samples_z(sp)
        results[label] = mapq.run_quality(pm, z_best, lp_hist, samples_z, dim,
                                          label, out_dir=OUT_DIR)

    # combined table
    print("\n===== T18 MAP-quality combined table =====")
    print(f"  {'label':16s} {'D1':>6} {'D2':>6} {'D3':>6}  {'FAILs':>5}")
    for label in ("ref_old_500", "ref_new_500", "improved_5000"):
        r = results.get(label)
        if r is None:
            print(f"  {label:16s}   (missing)")
            continue
        print(f"  {label:16s} {r['D1']['status']:>6} {r['D2']['status']:>6} "
              f"{r['D3']['status']:>6}  {r['n_fail']:>5}")

    # CALIBRATION: both reference 500-step MAPs must FAIL (>=1); improved must PASS (0)
    have_all = all(k in results for k in ("ref_old_500", "ref_new_500", "improved_5000"))
    if have_all:
        old_f = results["ref_old_500"]["n_fail"] >= 1
        new_f = results["ref_new_500"]["n_fail"] >= 1
        imp_ok = results["improved_5000"]["n_fail"] == 0
        met = old_f and new_f and imp_ok
        print(f"\nCALIBRATION: {'MET' if met else 'NOT-MET'}  "
              f"(ref_old FAILs>=1: {old_f}; ref_new FAILs>=1: {new_f}; "
              f"improved FAILs==0: {imp_ok})")
        summary = {"calibration_met": bool(met), "ref_old_fail": bool(old_f),
                   "ref_new_fail": bool(new_f), "improved_pass": bool(imp_ok)}
    else:
        print("\nCALIBRATION: N/A (need all three MAPs present)")
        summary = {"calibration_met": None}
    summary["results"] = {k: results[k]["n_fail"] for k in results}
    summary["status"] = "proposed (UNCERTIFIED)"
    with open(os.path.join(OUT_DIR, "quality_calibration.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ===========================================================================
# stage: seeds  (MCLMC off the improved MAP + T16 displaced-chain census)
# ===========================================================================

def _displaced_census(position, num_results, sigma=CENSUS_SIGMA):
    """T16 rule: post-burn-in draws split into halves [0:1000],[1000:2000]. For
    each chain and each half, flag if max_param |half_mean - grand_mean|/pooled_sd
    >= sigma. A chain 'persistently displaced' iff BOTH halves are flagged.

    pooled_sd : per-param std over ALL chains' post-burn-in draws pooled.
    grand_mean: per-param mean over ALL chains' post-burn-in draws pooled.
    Returns dict with per-half flags, persistent-chain list, and hit count."""
    post = np.asarray(position)[:, -num_results:, :]     # (chains, R, dim)
    n_chains, R, dim = post.shape
    half = R // 2
    grand_mean = post.reshape(-1, dim).mean(axis=0)      # (dim,)
    pooled_sd = post.reshape(-1, dim).std(axis=0, ddof=0)
    pooled_sd = np.where(pooled_sd > 0, pooled_sd, np.inf)  # avoid /0

    halves = [(0, half), (half, R)]
    per_chain = []
    persistent = []
    for c in range(n_chains):
        flags = []
        maxdev = []
        for (a, b) in halves:
            hm = post[c, a:b, :].mean(axis=0)
            dev = np.abs(hm - grand_mean) / pooled_sd
            md = float(np.max(dev))
            flags.append(bool(md >= sigma))
            maxdev.append(md)
        both = flags[0] and flags[1]
        per_chain.append({"chain": c, "half_flags": flags,
                          "half_max_dev_sigma": maxdev, "persistent": both})
        if both:
            persistent.append(c)
    return {
        "n_chains": int(n_chains), "num_results": int(R), "half": int(half),
        "sigma": sigma, "per_chain": per_chain,
        "persistent_chains": persistent,
        "n_persistent": len(persistent),
        "any_half_flagged": int(sum(1 for pc in per_chain if any(pc["half_flags"]))),
    }


def stage_seeds():
    assert_x64()
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    if not os.path.isfile(IMPROVED_MAP):
        raise FileNotFoundError(
            f"improved MAP not found: {IMPROVED_MAP}. Run --stage map first.")
    z_best = np.asarray(np.load(IMPROVED_MAP)["z_best"], dtype=np.float64)

    model_seq, ref_qz, ref_center, dim, param_names = load_target(OLD_SYS)
    # log the deviation of the improved MAP from the reference qz center
    dev = z_best - np.asarray(ref_center, dtype=np.float64)
    print(f"[T18 seeds] improved z_best deviation from reference qz center: "
          f"max|dz|={np.max(np.abs(dev)):.4g} (per-param sd of ref qz = {QZ_SCALE})")
    print(f"[T18 seeds] qz = MVNDiag(loc=improved z_best, scale_diag=full({dim}, "
          f"{QZ_SCALE}))  [SAME recipe as reference; re-centered]")

    qz = tfd.MultivariateNormalDiag(
        loc=jnp.asarray(z_best), scale_diag=jnp.full(dim, QZ_SCALE))

    os.makedirs(OUT_DIR, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        out_npz = os.path.join(OUT_DIR, f"t0_seed{seed}.npz")
        pos = run_standard_mclmc(
            model_seq, qz, STANDARD, seed, out_npz,
            target_desc="carousel_min_old, qz re-centered on improved T18 MAP",
            provenance={"system": OLD_SYS, "improved_map": os.path.abspath(IMPROVED_MAP),
                        "qz": f"MVNDiag(loc=improved z_best, scale_diag={QZ_SCALE})",
                        "max_abs_dev_from_ref_center": float(np.max(np.abs(dev)))},
        )
        diag = compute_diagnostics(pos, STANDARD.num_results, param_names)
        print_diagnostics(diag, header=f"T18 seed {seed}")
        census = _displaced_census(pos, STANDARD.num_results)
        print(f"[T18 seeds] seed {seed} displaced-chain census: "
              f"persistent(both-halves) chains = {census['persistent_chains']} "
              f"(n={census['n_persistent']}); any-half-flagged chains = "
              f"{census['any_half_flagged']}")
        man_path = os.path.splitext(out_npz)[0] + ".census.json"
        with open(man_path, "w") as f:
            json.dump(census, f, indent=2)
        per_seed.append({"seed": seed, "min_ess": diag["min_ess"],
                         "min_ess_param": diag["min_ess_param"],
                         "max_rhat": diag["max_rhat"],
                         "max_rhat_param": diag["max_rhat_param"],
                         "n_persistent": census["n_persistent"],
                         "persistent_chains": census["persistent_chains"],
                         "npz": os.path.abspath(out_npz)})

    # ---- FULL SET table (like run_t0_seed_variance.py) ----
    print("\n===== T18 SEEDS FULL SET (report this, not a summary) =====")
    for d in per_seed:
        print(f"  seed {d['seed']:>3}: min-ESS={d['min_ess']:.4g} "
              f"({d['min_ess_param']})  max-Rhat={d['max_rhat']:.4f} "
              f"({d['max_rhat_param']})  persistent-chains={d['persistent_chains']}")

    min_ess_vals = np.array([d["min_ess"] for d in per_seed], dtype=float)
    seeds_with_persistent = [d["seed"] for d in per_seed if d["n_persistent"] >= 1]
    n_seeds_hit = len(seeds_with_persistent)
    min_min_ess = float(min_ess_vals.min())

    # ---- registered verdicts ----
    fires_b = n_seeds_hit >= 1
    meets_b = n_seeds_hit == 0
    in_band = P_T18C_ESS_BAND[0] <= min_min_ess <= P_T18C_ESS_BAND[1]
    fires_c = min_min_ess >= F_T18C_ESS_FLOOR
    print("\n----- T18 registered predictions -----")
    print(f"  P_T18b displaced-chain hits: {n_seeds_hit}/3 seeds "
          f"(seeds {seeds_with_persistent}) -> "
          f"{'MEETS (0/3)' if meets_b else 'DOES NOT MEET'}; "
          f"F_T18b {'FIRES' if fires_b else 'does not fire'}")
    print(f"  P_T18c min-ESS band {P_T18C_ESS_BAND}: min(min-ESS)={min_min_ess:.4g} -> "
          f"{'in band' if in_band else 'OUTSIDE band'}; "
          f"F_T18c (>= {F_T18C_ESS_FLOOR}) {'FIRES' if fires_c else 'does not fire'}")

    summary = {
        "experiment": "T18_seeds", "status": "proposed (UNCERTIFIED)",
        "config": STANDARD.to_dict(), "seeds": SEEDS, "qz_scale": QZ_SCALE,
        "per_seed": per_seed,
        "min_min_ess": min_min_ess,
        "P_T18b_hits_seeds": seeds_with_persistent,
        "P_T18b_meets": bool(meets_b), "F_T18b_fires": bool(fires_b),
        "P_T18c_band": list(P_T18C_ESS_BAND), "P_T18c_in_band": bool(in_band),
        "F_T18c_floor": F_T18C_ESS_FLOOR, "F_T18c_fires": bool(fires_c),
        "git_commit": git_commit(),
    }
    with open(os.path.join(OUT_DIR, "seeds_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'seeds_summary.json')}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="T18 MAP-quality arm (carousel OLD).")
    ap.add_argument("--stage", required=True, choices=["map", "quality", "seeds"])
    # quality-stage path overrides (defaults are the registered inputs)
    ap.add_argument("--old-map", default=REF_OLD_MAP)
    ap.add_argument("--new-map", default=REF_NEW_MAP)
    ap.add_argument("--improved-map", default=IMPROVED_MAP)
    ap.add_argument("--old-samples", default=REF_OLD_SAMPLES)
    ap.add_argument("--new-samples", default=REF_NEW_SAMPLES)
    ap.add_argument("--improved-samples", default=None,
                    help="optional; usually absent at quality time (D3 -> N/A)")
    args = ap.parse_args()

    if args.stage == "map":
        stage_map()
    elif args.stage == "quality":
        stage_quality(args.old_map, args.new_map, args.improved_map,
                      args.old_samples, args.new_samples, args.improved_samples)
    elif args.stage == "seeds":
        stage_seeds()


if __name__ == "__main__":
    main()
