from __future__ import annotations

"""Re-run Step 1.2 comparison for source-plane vs image-plane connectivity."""

import argparse
import subprocess
import sys
from os.path import expanduser

HOME = expanduser("~/")
SCRIPT = f"{HOME}/GIGALens-Code/source_modeling/voronoi_src/tests/vela_truth_pinned_lambda_scan.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def main():
    args = parse_args()
    base = [
        args.python,
        SCRIPT,
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
        "--adaptive-target",
        "mclmc_shapelets",
    ]
    for connectivity in ("sourceplane", "imageplane"):
        cmd = base + ["--mesh-connectivity", connectivity]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
