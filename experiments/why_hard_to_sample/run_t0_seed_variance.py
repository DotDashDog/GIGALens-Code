"""T0 -- Seed-variance calibration (docs/logs/why-hard-to-sample.md, section T0).

Hypothesis-free prerequisite: run the standard MCLMC config on ONE
already-characterized system (the carousel minimal case) for N>=4 seeds,
identical in every other respect, and report the spread of min bulk-ESS and
max rank-R-hat across seeds. This spread is the "T0 band" that T1's decision
rule thresholds on -- it may NOT be invented, T0 must run first.

Reports the FULL per-seed set, not a summary (T0 spec), plus the derived band
(min/max min-ESS and their ratio) clearly labeled as the T0 threshold input.

Usage:
  python run_t0_seed_variance.py \
      --data-dir /global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag \
      --seeds 1,2,3,4 \
      --out-dir ./t0_out
"""
import argparse
import json
import os

import numpy as np

from common import (
    assert_x64, compute_diagnostics, git_commit, load_target,
    print_diagnostics, run_standard_mclmc,
)
from exp_config import STANDARD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dir with z_best.npy (MAIN checkout _h1h2_diag)")
    ap.add_argument("--seeds", required=True,
                    help="comma-separated seeds, N>=--min-seeds (T0 spec default 4)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-seeds", type=int, default=4,
                    help="minimum seed count (default 4, the original T0 spec; the "
                         "carousel B1 pre-registration sets 3). Only lowers the guard "
                         "when passed explicitly; sys60/vela behavior is unchanged.")
    args = ap.parse_args()

    assert_x64()  # fail loudly before touching the model if not float64

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    if len(seeds) < args.min_seeds:
        raise ValueError(
            f"T0 requires N>={args.min_seeds} seeds; got {len(seeds)}: {seeds}")
    os.makedirs(args.out_dir, exist_ok=True)

    model_seq, qz, z_best, dim, param_names = load_target(args.data_dir)
    print(f"[T0] target dim={dim}, seeds={seeds}")
    print(f"[T0] config = {STANDARD.to_dict()}")

    per_seed = []
    for seed in seeds:
        out_npz = os.path.join(args.out_dir, f"t0_seed{seed}.npz")
        pos = run_standard_mclmc(
            model_seq, qz, STANDARD, seed, out_npz,
            target_desc="carousel _h1h2_diag under-mode (real lensing posterior)",
            provenance={"data_dir": os.path.abspath(args.data_dir),
                        "z_best": os.path.join(os.path.abspath(args.data_dir), "z_best.npy"),
                        "qz": "MultivariateNormalDiag(loc=z_best, scale_diag=1e-2)"},
        )
        diag = compute_diagnostics(pos, STANDARD.num_results, param_names)
        print_diagnostics(diag, header=f"T0 seed {seed}")
        per_seed.append({
            "seed": seed,
            "min_ess": diag["min_ess"],
            "min_ess_param": diag["min_ess_param"],
            "max_rhat": diag["max_rhat"],
            "max_rhat_param": diag["max_rhat_param"],
            "table": diag["table"],
            "npz": os.path.abspath(out_npz),
        })

    # --- T0-derived band ---
    min_ess_vals = np.array([d["min_ess"] for d in per_seed], dtype=float)
    max_rhat_vals = np.array([d["max_rhat"] for d in per_seed], dtype=float)
    band = {
        "min_ess_across_seeds_min": float(min_ess_vals.min()),
        "min_ess_across_seeds_max": float(min_ess_vals.max()),
        "min_ess_ratio": float(min_ess_vals.max() / min_ess_vals.min())
        if min_ess_vals.min() > 0 else float("inf"),
        "max_rhat_across_seeds_min": float(max_rhat_vals.min()),
        "max_rhat_across_seeds_max": float(max_rhat_vals.max()),
    }

    order_of_magnitude = band["min_ess_ratio"] >= 10.0

    result = {
        "experiment": "T0_seed_variance",
        "status": "proposed (UNCERTIFIED)",
        "config": STANDARD.to_dict(),
        "seeds": seeds,
        "per_seed": per_seed,
        "T0_derived_band": band,
        "band_spans_order_of_magnitude": bool(order_of_magnitude),
        "note": (
            "T0-DERIVED THRESHOLD INPUT FOR T1: the min-ESS band (min..max, ratio "
            f"{band['min_ess_ratio']:.2f}) is the seed-variance envelope T1 uses. "
            + ("WARNING: band spans >= 1 order of magnitude -- per T0 spec, ESS on "
               "this posterior is not stable enough to threshold on; STOP and surface "
               "to grader before running T1."
               if order_of_magnitude else
               "Band is within a modest factor; usable as T1 threshold.")
        ),
        "git_commit": git_commit(),
    }

    json_path = os.path.join(args.out_dir, "t0_summary.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # CSV of the full per-seed summary (the headline numbers).
    csv_path = os.path.join(args.out_dir, "t0_summary.csv")
    with open(csv_path, "w") as f:
        f.write("seed,min_ess,min_ess_param,max_rhat,max_rhat_param\n")
        for d in per_seed:
            f.write(f"{d['seed']},{d['min_ess']:.6g},{d['min_ess_param']},"
                    f"{d['max_rhat']:.6g},{d['max_rhat_param']}\n")

    print("\n===== T0 FULL SET (report this, not a summary) =====")
    for d in per_seed:
        print(f"  seed {d['seed']:>4}: min-ESS={d['min_ess']:.4g} "
              f"({d['min_ess_param']})  max-Rhat={d['max_rhat']:.4f} "
              f"({d['max_rhat_param']})")
    print("----- T0-derived band (threshold input for T1) -----")
    print(f"  min-ESS across seeds: {band['min_ess_across_seeds_min']:.4g} .. "
          f"{band['min_ess_across_seeds_max']:.4g}  (ratio {band['min_ess_ratio']:.2f})")
    print(f"  max-Rhat across seeds: {band['max_rhat_across_seeds_min']:.4f} .. "
          f"{band['max_rhat_across_seeds_max']:.4f}")
    if order_of_magnitude:
        print("  *** WARNING: band spans >= 1 order of magnitude -- STOP, surface to grader.")
    print(f"\nwrote {json_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
