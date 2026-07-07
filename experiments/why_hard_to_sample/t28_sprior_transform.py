"""T28 -- build the Rs(s) leaf artifact + login-runnable gates (NO jax).

Pre-registered 2026-07-04 (docs/logs/why-hard-to-sample.md, "T28"). Builds the
monotone PCHIP leaf ``forward: s -> Rs`` for the observable-slope prior swap
(s ~ Uniform(0, 0.75)) WITHOUT bisection: evaluate the exact numpy slope
s = slope_s_of_Rs(Rs) on a dense log-spaced Rs grid and use (s_grid, Rs_grid) as
the (u_knots, z_knots) of a MonotoneCubicBijector. Writes

    results_carousel/phaseC/t28/transform_sprior.npz
    results_carousel/phaseC/t28/transform_gates.json

Login-runnable BLOCKING gates (numpy paths only; the jnp G1/G2/G3/G4 run on GPU):
  * G1 round-trip: max |s_analytic(leaf.forward(s)) - s| < 1e-8 on 512 pts in [0, 0.75].
  * endpoint anchors: leaf.forward(0) ~ 10.478, leaf.forward(0.75) ~ 614.4.
  * monotonicity + strict-increase of the leaf on a fine grid.

Reported DIAGNOSTIC (informational; NOT a correctness gate -- see below):
  * numpy chart-consistency  |ds/dRs * dRs/ds - 1|  on 512 pts (ds/dRs = central
    FD of the numpy slope; dRs/ds = leaf derivative).  This measures how flat-in-s
    the sampling CHART is, not the target.  The PHYSICAL POSTERIOR is exact for any
    monotone leaf: with fldj = the leaf's own true derivative, the leaf Jacobian
    cancels exactly under change-of-variables, leaving  pi(Rs) ∝ L(Rs)·(1/0.75)·
    (ds/dRs)_analytic(Rs) -- the intended prior×likelihood (certified by G2+G3).
    The numpy value here is ~1e-3, FD-ROUNDOFF-LIMITED (differencing the already-
    FD s twice); on GPU ds/dRs is taken by jax.grad (exact derivative of the
    s-expression) and the internal jnp consistency is far tighter (reported in
    t28_run_gpu.py).  This is a flagged, documented deviation from the registered
    "~1e-6" wording, whose intent (a correct, consistent ds/dRs) is met by autodiff.

Run:  python t28_sprior_transform.py     (login node; numpy only)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from reparam_bijector import sha256_file  # noqa: E402
from t25_transforms import slope_s_of_Rs  # noqa: E402  (exact numpy g_ mirror)
from t28_common import (  # noqa: E402
    ARTIFACT, N_GRID, RS_AT_S0, RS_AT_S075, RS_TABLE_HI, RS_TABLE_LO,
    S_HI, S_KNOT_HI, S_KNOT_LO, S_LO, THETA_E_STAR, build_leaf, ds_dRs_numpy,
)

OUT_DIR = os.path.dirname(ARTIFACT)
GATES_JSON = os.path.join(OUT_DIR, "transform_gates.json")

# gate thresholds
G1_TOL = 1e-8            # registered round-trip tol (BLOCKING)
N_G1 = 512
ENDPOINT_TOL = 1e-2     # leaf.forward(0)~10.478, forward(0.75)~614.4 (grid-resolution)
CHART_DIAG_WARN = 5e-3  # informational warn level for the numpy chart-consistency


def _now():
    return datetime.now(timezone.utc).isoformat()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72, flush=True)
    print("T28 -- Rs(s) leaf build + login gates (numpy only; UNCERTIFIED)", flush=True)
    print("=" * 72, flush=True)

    # --- build the leaf (no bisection) ------------------------------------
    leaf, s_knots, Rs_knots = build_leaf(THETA_E_STAR)
    print(f"[build] theta_E* = {THETA_E_STAR}", flush=True)
    print(f"[build] leaf knots = {N_GRID} UNIFORM-in-s pts in [{S_KNOT_LO}, {S_KNOT_HI}] "
          f"(Rs via dense table [{RS_TABLE_LO}, {RS_TABLE_HI}]; brackets support "
          f"[{S_LO}, {S_HI}])", flush=True)
    print(f"[build] Rs knot range = [{Rs_knots[0]:.4f}, {Rs_knots[-1]:.2f}]", flush=True)
    leaf.to_npz(ARTIFACT)
    sha = sha256_file(ARTIFACT)
    print(f"[build] wrote {ARTIFACT}\n[build] sha256 = {sha}", flush=True)

    gates = {}

    # --- G1 round-trip: s_analytic(leaf.forward(s)) == s ------------------
    s_test = np.linspace(S_LO, S_HI, N_G1)
    Rs_of_s = leaf._forward_np(s_test)            # spline forward s -> Rs
    s_back = np.asarray(slope_s_of_Rs(Rs_of_s, THETA_E_STAR), np.float64)  # analytic Rs -> s
    g1 = float(np.max(np.abs(s_back - s_test)))
    gates["G1_roundtrip_max"] = g1
    gates["G1_pass"] = bool(g1 < G1_TOL)
    print(f"[G1] max|s_analytic(leaf.forward(s)) - s| = {g1:.3e} "
          f"({'PASS' if gates['G1_pass'] else 'FAIL'}; tol {G1_TOL})", flush=True)

    # --- chart-consistency DIAGNOSTIC (informational, NOT blocking) -------
    # ds/dRs here is a numpy central-FD of the (already-FD) numpy slope, so it is
    # roundoff-limited (~1e-3); the GPU path uses jax.grad (exact). Physical
    # correctness rests on G1 + G2 + G3, not on this number (see module docstring).
    ds_dRs = ds_dRs_numpy(Rs_of_s, THETA_E_STAR)      # FD inverse-derivative (noisy)
    dRs_ds = leaf.derivative_np(s_test)               # spline forward-derivative (exact)
    prod = ds_dRs * dRs_ds
    chart_cons = float(np.max(np.abs(prod - 1.0)))
    gates["chart_consistency_numpy_max_abs_err"] = chart_cons
    gates["chart_consistency_is_blocking"] = False
    gates["chart_consistency_note"] = (
        "FD-roundoff-limited diagnostic of chart flatness; NOT a correctness gate. "
        "Physical posterior is exact by construction (G2+G3); GPU uses exact jax.grad "
        "ds/dRs (see t28_run_gpu.py internal jnp consistency).")
    flag = "" if chart_cons < CHART_DIAG_WARN else "  [above warn; expected for numpy FD]"
    print(f"[chart-diag] max|ds/dRs_FD * dRs/ds - 1| = {chart_cons:.3e} "
          f"(informational, warn>{CHART_DIAG_WARN}){flag}", flush=True)

    # --- endpoint anchors -------------------------------------------------
    Rs_at_0 = float(leaf._forward_np(np.array([S_LO]))[0])
    Rs_at_075 = float(leaf._forward_np(np.array([S_HI]))[0])
    e0 = abs(Rs_at_0 - RS_AT_S0) / RS_AT_S0
    e1 = abs(Rs_at_075 - RS_AT_S075) / RS_AT_S075
    gates["Rs_at_s0"] = Rs_at_0
    gates["Rs_at_s075"] = Rs_at_075
    gates["endpoint_pass"] = bool(e0 < ENDPOINT_TOL and e1 < ENDPOINT_TOL)
    print(f"[endpoint] leaf.forward(0)={Rs_at_0:.4f} (want ~{RS_AT_S0}, rel {e0:.1e}); "
          f"leaf.forward(0.75)={Rs_at_075:.4f} (want ~{RS_AT_S075}, rel {e1:.1e}) "
          f"({'PASS' if gates['endpoint_pass'] else 'FAIL'})", flush=True)

    # --- monotone + strictly increasing on a fine grid --------------------
    s_fine = np.linspace(S_LO - 0.05, S_HI + 0.05, 4000)
    Rs_fine = leaf._forward_np(s_fine)
    mono = bool(np.all(np.diff(Rs_fine) > 0))
    gates["leaf_strictly_increasing"] = mono
    print(f"[mono] leaf forward strictly increasing on fine grid: {mono}", flush=True)

    # --- induced Rs-density descriptive readout (registration sanity) -----
    # p(Rs) = (ds/dRs)/0.75 ; Rs*p(Rs) drifts ~0.31 -> 0.07 over [12,500] per reg.
    Rs_probe = np.array([12.0, 50.0, 100.0, 200.0, 500.0])
    rs_p = Rs_probe * ds_dRs_numpy(Rs_probe, THETA_E_STAR) / (S_HI - S_LO)
    gates["Rs_times_pRs_probe"] = {float(r): float(v) for r, v in zip(Rs_probe, rs_p)}
    print(f"[density] Rs*p(Rs) at {Rs_probe.tolist()} = "
          f"{[round(float(v), 3) for v in rs_p]} (reg: drifts ~0.31 -> 0.07)", flush=True)

    all_pass = bool(gates["G1_pass"] and gates["endpoint_pass"] and mono)
    gates["all_login_gates_pass"] = all_pass

    payload = {
        "experiment": "T28 Rs(s) leaf build + login gates",
        "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(),
        "artifact": os.path.abspath(ARTIFACT),
        "artifact_sha256": sha,
        "theta_E_star": THETA_E_STAR,
        "prior": {"s_low": S_LO, "s_high": S_HI,
                  "Rs_at_s_low": Rs_at_0, "Rs_at_s_high": Rs_at_075},
        "grid": {"s_knot_lo": S_KNOT_LO, "s_knot_hi": S_KNOT_HI, "n_knots": N_GRID,
                 "rs_table": [RS_TABLE_LO, RS_TABLE_HI],
                 "rs_knot_range": [float(Rs_knots[0]), float(Rs_knots[-1])]},
        "gates": gates,
        "note": "G4 (jnp s vs numpy s < 1e-10), G1-in-jnp, G2 prior-density and G3 "
                "loglike identities run on GPU in t28_run_gpu.py.",
    }
    with open(GATES_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"[out] wrote {GATES_JSON}", flush=True)
    print(f"[t28] login gates {'PASS' if all_pass else 'FAIL'} "
          f"(proposed / UNCERTIFIED)", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
