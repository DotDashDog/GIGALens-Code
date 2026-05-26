from __future__ import annotations

import argparse
import json
import os
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
    p = argparse.ArgumentParser()
    p.add_argument("--gradient-json", type=str, required=True)
    p.add_argument("--sourceplane-summary", type=str, required=True)
    p.add_argument("--imageplane-summary", type=str, required=True)
    p.add_argument("--chi2-tolerance", type=float, default=0.05)
    p.add_argument("--alt-tolerance", type=float, default=0.02)
    p.add_argument("--degeneracy-max", type=float, default=0.05)
    return p.parse_args()


def _best_variant(summary):
    variants = summary.get("variant_summaries", [])
    if not variants:
        return None
    return min(variants, key=lambda v: v["evidence_optimal_chi2_mean"])


def main():
    args = parse_args()
    with open(args.gradient_json) as f:
        grad = json.load(f)
    with open(args.sourceplane_summary) as f:
        sp = json.load(f)
    with open(args.imageplane_summary) as f:
        ip = json.load(f)

    b_sp = _best_variant(sp)
    b_ip = _best_variant(ip)
    chi2_ok = abs(b_sp["evidence_optimal_chi2_mean"] - b_ip["evidence_optimal_chi2_mean"]) <= args.chi2_tolerance
    alt_ok = (
        abs(b_sp["evidence_optimal_alternating_score"] - b_ip["evidence_optimal_alternating_score"])
        <= args.alt_tolerance
    )
    deg_ok = grad["degeneracy"]["degenerate_subpix_fraction"] <= args.degeneracy_max

    report = {
        "phase2_pass_gradient": grad.get("phase2_pass_gradient", False),
        "phase2_pass_finite_perturbations": grad.get("phase2_pass_finite_perturbations", False),
        "phase2_pass_quality_match": chi2_ok and alt_ok,
        "phase2_pass_degeneracy": deg_ok,
        "sourceplane_best": b_sp,
        "imageplane_best": b_ip,
    }
    report["phase2_pass"] = all(
        report[k]
        for k in (
            "phase2_pass_gradient",
            "phase2_pass_finite_perturbations",
            "phase2_pass_quality_match",
            "phase2_pass_degeneracy",
        )
    )
    print(json.dumps(report, indent=2))
    if not report["phase2_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
