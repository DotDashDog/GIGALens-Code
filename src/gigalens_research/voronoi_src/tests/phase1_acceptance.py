from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from os.path import expanduser

HOME = expanduser("~/")
for p in (
    os.path.join(HOME, "gigalens", "src"),
    os.path.join(HOME, "GIGALens-Code"),
    os.path.join(HOME, "GIGALens-Code", "source_modeling"),
):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 acceptance on vela04 cam12 rep00.")
    p.add_argument("--run-scan", action="store_true", help="Launch vela_truth_pinned_lambda_scan before checking.")
    p.add_argument("--chi2-max", type=float, default=1.10, help="chi2_mean ceiling at evidence-optimal lambda.")
    p.add_argument("--alternating-threshold", type=float, default=-0.01)
    p.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Path to vela_truth_pinned_lambda_scan.json (default: latest under results dir).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    script = os.path.join(
        HOME,
        "GIGALens-Code",
        "source_modeling",
        "voronoi_src",
        "tests",
        "vela_truth_pinned_lambda_scan.py",
    )
    if args.run_scan:
        cmd = [
            sys.executable,
            script,
            "--sim-num",
            "04",
            "--rep",
            "0",
            "--cam",
            "12",
            "--reg-variants",
            "constant_gradient",
            "distance_weighted_gradient",
            "curvature",
            "--adaptive-weight-scheme",
            "brightness_times_invmag",
            "--alternating-threshold",
            str(args.alternating_threshold),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    summary_path = args.summary_json
    if not summary_path:
        base = os.path.join(
            HOME,
            "GIGALens-Code",
            "source_modeling",
            "voronoi_src",
            "truth_pinned_lambda_scan_vela",
        )
        candidates = []
        for root, _, files in os.walk(base):
            if "vela_truth_pinned_lambda_scan.json" in files:
                candidates.append(os.path.join(root, "vela_truth_pinned_lambda_scan.json"))
        if not candidates:
            raise FileNotFoundError("No vela_truth_pinned_lambda_scan.json found; pass --summary-json or --run-scan")
        summary_path = max(candidates, key=os.path.getmtime)

    with open(summary_path) as f:
        summary = json.load(f)

    checks = []
    for variant in summary.get("variant_summaries", []):
        name = variant["reg_kind"]
        chi2 = variant["evidence_optimal_chi2_mean"]
        checks.append(
            {
                "reg_kind": name,
                "chi2_ok": chi2 <= args.chi2_max,
                "alternating_ok": variant.get("phase1_pass_alternating", False),
                "peak_ok": variant.get("phase1_pass_peak", False),
                "evidence_optimal_lambda": variant["evidence_optimal_lambda"],
                "chi2_mean": chi2,
                "alternating_score": variant["evidence_optimal_alternating_score"],
                "peak_offset_arcsec": variant["peak_offset_arcsec"],
            }
        )

    phase1_pass = any(
        c["chi2_ok"] and c["alternating_ok"] and c["peak_ok"] for c in checks
    ) or any(c["chi2_ok"] and c["peak_ok"] for c in checks)

    report = {
        "summary_json": summary_path,
        "checks": checks,
        "phase1_pass": phase1_pass,
        "chi2_max": args.chi2_max,
        "alternating_threshold": args.alternating_threshold,
    }
    print(json.dumps(report, indent=2))
    if not phase1_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
