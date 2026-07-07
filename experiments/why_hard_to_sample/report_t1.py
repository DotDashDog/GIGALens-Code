"""T1 report -- real vs clone comparison (docs/logs/why-hard-to-sample.md, T1).

Produces the pre-registered comparison:
  - side-by-side per-parameter bulk-ESS table (real vs clone) + rank-R-hat,
  - min-ESS of each with the T0 seed-variance band,
  - trace plots side by side (worst-ESS parameters first, <=4x4 panels/figure),
  - a verdict phrased per the log's decision rule, marked UNCERTIFIED.

Decision rule (from T1 spec):
  clone min-ESS within the T0 band of the real run's min-ESS -> consistent with
    H2 being SUFFICIENT (clone just as slow; conditioning explains it).
  clone min-ESS exceeds real by MORE than the T0 band (clone much easier) ->
    H2 INSUFFICIENT; hardness is in the true density's fine structure (H1) or
    non-Gaussian z-shape (H3).

Pure numpy + matplotlib (no jax). Post-processing only.

Usage:
  python report_t1.py --real-run t0_out/t0_seed1.npz \
      --clone-run t1_out/t1_clone_seed1.npz \
      --t0-table t0_out/t0_summary.json --out-dir ./t1_report [--data-dir <dir>]
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import compute_diagnostics, git_commit, load_param_names


def _load_run(path):
    d = np.load(path)
    pos = np.asarray(d["position"])
    nr = int(d["nr"]) if "nr" in d else pos.shape[1]
    return pos, nr


def _trace_figures(real_pos, clone_pos, nr, order_idx, names, out_dir, tag):
    """Side-by-side traces: real (left col) vs clone (right col) for the
    worst-ESS parameters. Repo hard rule: never more than 4x4 panels/figure.
    Here each row is one parameter -> 4 params per figure (4 rows x 2 cols)."""
    real_post = real_pos[:, -nr:, :]
    clone_post = clone_pos[:, -nr:, :]
    per_fig = 4  # 4 params (rows) x 2 cols (real, clone) = 4x2 <= 4x4
    figs = []
    for start in range(0, len(order_idx), per_fig):
        chunk = order_idx[start:start + per_fig]
        nrows = len(chunk)
        fig, axes = plt.subplots(nrows, 2, figsize=(11, 2.4 * nrows), squeeze=False)
        for r, pi in enumerate(chunk):
            for c, (post, lbl) in enumerate([(real_post, "real"), (clone_post, "clone")]):
                ax = axes[r][c]
                for ch in range(post.shape[0]):
                    ax.plot(post[ch, :, pi], lw=0.5, alpha=0.7)
                ax.set_title(f"{lbl}: {names[pi]}", fontsize=9)
                ax.set_xlabel("step"); ax.set_ylabel("z")
        fig.suptitle(f"T1 traces (worst-ESS block {start//per_fig + 1}) -- {tag}",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fpath = os.path.join(out_dir, f"t1_traces_{tag}_block{start//per_fig + 1}.png")
        fig.savefig(fpath, dpi=110)
        plt.close(fig)
        figs.append(fpath)
    return figs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-run", required=True)
    ap.add_argument("--clone-run", required=True)
    ap.add_argument("--t0-table", required=True, help="t0_summary.json from T0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default=None, help="optional, for names.npy labels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    real_pos, real_nr = _load_run(args.real_run)
    clone_pos, clone_nr = _load_run(args.clone_run)
    dim = real_pos.shape[-1]
    if clone_pos.shape[-1] != dim:
        raise ValueError(f"real dim {dim} != clone dim {clone_pos.shape[-1]}")

    names = load_param_names(args.data_dir, dim) or [str(i) for i in range(dim)]

    real_diag = compute_diagnostics(real_pos, real_nr, names)
    clone_diag = compute_diagnostics(clone_pos, clone_nr, names)

    with open(args.t0_table) as f:
        t0 = json.load(f)
    band = t0["T0_derived_band"]
    band_lo = band["min_ess_across_seeds_min"]
    band_hi = band["min_ess_across_seeds_max"]
    band_ratio = band["min_ess_ratio"]

    real_min = real_diag["min_ess"]
    clone_min = clone_diag["min_ess"]

    # Decision rule. "Within the T0 band of the real run's min-ESS": the clone's
    # min-ESS is at most a factor `band_ratio` above the real run's (same order of
    # slowness). Exceeding that by more -> clone much easier -> H2 insufficient.
    if band_ratio <= 0 or not np.isfinite(band_ratio):
        verdict = ("INDETERMINATE: T0 band ratio is degenerate; cannot threshold. "
                   "Re-examine T0 before concluding.")
    elif clone_min <= real_min * band_ratio:
        verdict = (f"CONSISTENT WITH H2-SUFFICIENCY: clone min-ESS ({clone_min:.4g}) is "
                   f"within the T0 seed band (x{band_ratio:.2f}) of the real run's "
                   f"min-ESS ({real_min:.4g}) -- the clone is as slow as the real run, "
                   f"so z-space conditioning/parameterization alone can account for the "
                   f"slowness. Program shifts toward preconditioning/reparameterization "
                   f"(NFW first); T3/T4 deprioritized.")
    else:
        verdict = (f"H2 INSUFFICIENT: clone min-ESS ({clone_min:.4g}) exceeds the real "
                   f"run's ({real_min:.4g}) by more than the T0 band (x{band_ratio:.2f}) "
                   f"-- the clone is much EASIER than the real posterior. Conditioning "
                   f"alone cannot explain the slowness; the hardness lives in the true "
                   f"log-density's fine structure (H1) or its non-Gaussian z-shape (H3). "
                   f"Next: run T3 (and T2 to separate H3).")

    # order parameters by real-run ESS (worst first) for the tables and traces
    order_idx = list(np.argsort(real_diag["ess"]))

    # --- side-by-side per-parameter table ---
    csv_path = os.path.join(args.out_dir, "t1_ess_table.csv")
    with open(csv_path, "w") as f:
        f.write("param,real_ess,clone_ess,real_rhat,clone_rhat\n")
        for pi in order_idx:
            f.write(f"{names[pi]},{real_diag['ess'][pi]:.6g},{clone_diag['ess'][pi]:.6g},"
                    f"{real_diag['rhat'][pi]:.6g},{clone_diag['rhat'][pi]:.6g}\n")

    print("\n=== T1 per-parameter ESS (worst real-ESS first) ===")
    print(f"{'param':<32} {'real_ess':>10} {'clone_ess':>10} "
          f"{'real_rhat':>10} {'clone_rhat':>10}")
    for pi in order_idx:
        print(f"{names[pi]:<32} {real_diag['ess'][pi]:>10.4g} "
              f"{clone_diag['ess'][pi]:>10.4g} {real_diag['rhat'][pi]:>10.4f} "
              f"{clone_diag['rhat'][pi]:>10.4f}")

    # --- trace plots ---
    trace_files = _trace_figures(real_pos, clone_pos, min(real_nr, clone_nr),
                                 order_idx, names, args.out_dir, tag="realVSclone")

    summary = {
        "experiment": "T1_report",
        "status": "proposed (UNCERTIFIED)",
        "real_run": os.path.abspath(args.real_run),
        "clone_run": os.path.abspath(args.clone_run),
        "real_min_bulk_ess": real_min,
        "real_min_ess_param": real_diag["min_ess_param"],
        "real_max_rank_rhat": real_diag["max_rhat"],
        "clone_min_bulk_ess": clone_min,
        "clone_min_ess_param": clone_diag["min_ess_param"],
        "clone_max_rank_rhat": clone_diag["max_rhat"],
        "T0_band": {"min_ess_lo": band_lo, "min_ess_hi": band_hi,
                    "ratio": band_ratio},
        "verdict": verdict,
        "verdict_status": "proposed (UNCERTIFIED)",
        "ess_table_csv": os.path.abspath(csv_path),
        "trace_figures": [os.path.abspath(p) for p in trace_files],
        "git_commit": git_commit(),
    }
    json_path = os.path.join(args.out_dir, "t1_report.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== T1 VERDICT (proposed, UNCERTIFIED) =====")
    print(f"  real  min-ESS = {real_min:.4g} ({real_diag['min_ess_param']}), "
          f"max-Rhat = {real_diag['max_rhat']:.4f}")
    print(f"  clone min-ESS = {clone_min:.4g} ({clone_diag['min_ess_param']}), "
          f"max-Rhat = {clone_diag['max_rhat']:.4f}")
    print(f"  T0 band: min-ESS {band_lo:.4g}..{band_hi:.4g} (ratio {band_ratio:.2f})")
    print(f"  {verdict}")
    print(f"\nwrote {json_path}\nwrote {csv_path}")
    for p in trace_files:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
