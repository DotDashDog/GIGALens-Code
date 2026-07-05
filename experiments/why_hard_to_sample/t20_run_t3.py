"""T20 -- thin, NON-EDITING wrapper to run t3_transects.py on the carousel arms.

WHY THIS WRAPPER EXISTS (do NOT edit t3_transects.py):
t3.build_directions(..., reduced=False) unconditionally builds coordinate-axis
transects for t3's module constant WORST_ESS_PARAMS =
  ["planes/1/light/0/center_y", "planes/0/mass/0/gamma"]
These are the sys60 worst-ESS axes. On the carousel arms "planes/0/mass/0/gamma"
does NOT exist (the mass profile is NFW_ELLIPSE, not a power-law with gamma), so
build_directions raises AssertionError and the float64 t3 run CRASHES. (Verified
offline: carousel sorted param names have no .../gamma; index 9 =
planes/1/light/0/center_x, matching the task fact.)

t3 exposes NO CLI hook to inject directions, and we must not edit t3. The
smallest possible adapter is to override the module GLOBAL from OUTSIDE before
calling t3.main() -- t3.build_directions reads WORST_ESS_PARAMS at call time, so
setting t3.WORST_ESS_PARAMS here is sufficient and touches no line of t3.

We point the two axis transects at carousel-relevant coordinate axes:
  planes/1/light/0/center_x  (z9; a REQUIRED carousel-specific direction, task
                              1b -- covered here as a bonus AND in t20_step_segments)
  planes/0/mass/0/Rs         (the Rs valley that dominates the sample covariance)
These are DIRECTION COVERAGE only; the "worst-ESS" label in t3's output is
cosmetic here (no carousel T0 ESS ranking was used). The z_best->bulk-mean escape
direction (task 1a) is NOT an axis and cannot be injected via this mechanism; it
is covered by t20_step_segments.py's micro-transects.

For the float32 control (--allow-float32) t3 uses reduced=True and never touches
WORST_ESS_PARAMS, so the override is harmless there.

Usage: identical CLI to t3_transects.py -- all args are forwarded verbatim.
  python3 t20_run_t3.py --data-dir ... --samples ... --clone ... \
      --ref-diagnostics ... --out-dir ... --seed 20260703 [--allow-float32]
"""
from __future__ import annotations

import sys

import t3_transects as t3

# Carousel axis transects (replace sys60 worst-ESS axes; direction coverage only).
CAROUSEL_AXES = ["planes/1/light/0/center_x", "planes/0/mass/0/Rs"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    t3.WORST_ESS_PARAMS = list(CAROUSEL_AXES)   # override module global (no edit)
    print(f"[t20_run_t3] overrode WORST_ESS_PARAMS -> {t3.WORST_ESS_PARAMS} "
          f"(carousel axis coverage; sys60 axes would crash: no .../gamma)")
    t3.main(argv)


if __name__ == "__main__":
    main()
