#!/usr/bin/env python
"""GATE PT-2 scorer: pinned formulas from the rd-2-certified checkpoint (@6a4a96f).

W-M1/W-M2 (D1/D2 adaptive-metric arms): RT floor, occupancy band, EEVPD band, and
the SCORED gen-eig clause vs the COMPOSITION-MATCHED reference Sigma_ref(w-hat)
(blind-spot ix out-of-band rule included). W-E (D3/D4): pinned floors 94/94,
pinned windows. Login-node numpy only; constants transcribed, never re-derived.
"""
import json

import numpy as np

OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
MAMS = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/sim_carousel/messy_tests/dpie/mams/arrays.npz"
BAND = (0.32, 0.49)
EEVPD_BAND = (1e-4, 2e-3)
GENEIG_BAND = (1.0 / 3.0, 3.0)
RT_FLOOR_ADAPT = 175          # D1/D2 (all-main basis; op-7 scaled by rounds/1500)
RT_FLOOR_E = 94               # D3/D4 (balanced basis, pinned rd-1 B5)
RT_SECONDARY = 139            # F-M1 noise-banded secondary (117 + 2*sqrt(117))
DRIFT_ENV = 0.05
W_CENTER = 0.42               # blind-spot ix out-of-band reference composition
POCKET_COL, POCKET_THR = 6, -22.35

# per-basin pools (POSITIONS ONLY — standing directive)
draws = np.load(MAMS)["samples_z"].reshape(-1, 33).astype(np.float64)
pk = draws[:, POCKET_COL] > POCKET_THR
cov_P, cov_M = np.cov(draws[pk].T), np.cov(draws[~pk].T)
dmu = draws[pk].mean(0) - draws[~pk].mean(0)


def sigma_ref(w):
    return w * cov_P + (1 - w) * cov_M + w * (1 - w) * np.outer(dmu, dmu)


def geneig(adapted, ref):
    L = np.linalg.cholesky(ref)
    X = np.linalg.solve(L, np.linalg.solve(L, adapted).T)
    return np.sort(np.linalg.eigvalsh((X + X.T) / 2.0))


res = {}
for name, tag, floor, kind in (("D1", "D1_D1pt2", RT_FLOOR_ADAPT, "adapt"),
                               ("D2", "D2_D2pt2", RT_FLOOR_ADAPT, "adapt"),
                               ("D3", "B1_D3pt2", RT_FLOOR_E, "e750"),
                               ("D4", "B1_D4pt2", RT_FLOOR_E, "e8sys")):
    d = np.load(f"{OUT}/arrays_{tag}.npz")
    rounds = int(d["rounds_done"])
    nominal = 750 if kind == "e750" else 1500
    if rounds < nominal:
        print(f"*** {name} INCOMPLETE {rounds}/{nominal} — op-7 floor scaling ***")
    cold = d["cold_ind"][:rounds]
    if kind == "e750":
        win = cold[250:750] if rounds >= 750 else cold[250:]
    else:
        win = cold[-500:]
    nsys = cold.shape[1]
    sysm = win.mean(axis=0)
    m, sd = float(sysm.mean()), float(sysm.std(ddof=1))
    se = sd / np.sqrt(nsys)
    fl = floor * min(1.0, rounds / nominal)
    rtp = int(d["round_trips_pocket"].sum())
    ev = d["eevpd"][:rounds]
    ev_med = np.median(ev[-min(500, rounds):], axis=0)
    ev_ok = all(EEVPD_BAND[0] <= v <= EEVPD_BAND[1] for v in ev_med)
    in_band = BAND[0] <= m <= BAND[1]
    near = min(abs(m - BAND[0]), abs(m - BAND[1])) < DRIFT_ENV
    arm = dict(rounds=rounds, m=m, sd=sd, se=se, rt_pocket=rtp,
               rt_total=int(d["round_trips"].sum()), floor_eff=fl,
               ev_med=ev_med.tolist(), n_revert=int(d["n_revert"]), nsys=nsys)
    print(f"\n[{name}] rounds={rounds} m={m:.4f}±{se:.4f} RTpocket={rtp} "
          f"(floor {fl:.0f}) RTtotal={arm['rt_total']} "
          f"eevpd_med={['%.1e' % v for v in ev_med]} NaN={arm['n_revert']}")
    clauses = dict(rt=rtp >= fl, occ=in_band, eevpd=ev_ok,
                   nan=arm["n_revert"] == 0)
    if near:
        print(f"  *** NEAR-EDGE (±{DRIFT_ENV}): band reading REQUIRES RT+plot "
              f"corroboration ***")
    if kind == "adapt":
        # scored gen-eig clause vs Sigma_ref(w-hat)
        assert list(d["metric_windows"]) == [100, 250, 500], "non-production windows!"
        assert "_smoke" not in tag
        mf = d["metric_frozen"]           # (R, D, D); frozen (rounds>=500 asserted)
        assert rounds >= 500, "metric not frozen — gen-eig clause unscoreable"
        g = geneig(mf[-1], sigma_ref(m))  # cold rung vs composition-matched ref
        n_out = int(((g < GENEIG_BAND[0]) | (g > GENEIG_BAND[1])).sum())
        n_soft = int(((g > GENEIG_BAND[1]) & (g <= 10)).sum())
        clauses["geneig"] = n_out == 0
        arm["geneig_minmax"] = [float(g[0]), float(g[-1])]
        arm["geneig_n_out"] = n_out
        print(f"  gen-eig vs Sigma_ref(w-hat={m:.3f}): [{g[0]:.3f}, {g[-1]:.3f}] "
              f"in [1/3,3] all axes: {n_out == 0} (out: {n_out}, of which (3,10]: "
              f"{n_soft})")
        if not in_band:
            g42 = geneig(mf[-1], sigma_ref(W_CENTER))
            arm["geneig_vs_042"] = [float(g42[0]), float(g42[-1])]
            print(f"  blind-spot ix (w-hat OUT of band): gen-eig vs "
                  f"Sigma_ref(0.42): [{g42[0]:.3f}, {g42[-1]:.3f}] — REQUIRED "
                  f"before any mechanism attribution")
        if n_out > 0 and n_soft == n_out and n_out <= 3 and clauses["rt"]:
            print("  ZONE: adaptation PARTIAL — axes in (3,10], transport passing "
                  "(named-axes report, next-gate decision)")
        if not clauses["rt"]:
            if rtp > RT_SECONDARY:
                print(f"  RT {rtp} in (139, floor): above the F-M1 secondary — "
                      f"flux-limited zone (adaptation helped; ratio vs PT-1's 117 "
                      f"= {rtp/117:.2f}x)")
            elif rtp <= RT_SECONDARY:
                print(f"  F-M1 SECONDARY: RT {rtp} <= 139 (frozen-SVI-class) — "
                      f"gen-eig primary decides mechanism verdict")
    if kind == "e750" and 78 <= rtp < 94:
        print("  ZONE [78,94): spin-up-adjusted marginal — report")
    verdict = all(clauses.values())
    print(f"  clauses: {clauses} -> {'PASS' if verdict else 'not-pass (see zones)'}")
    arm["clauses"] = {k: bool(v) for k, v in clauses.items()}
    res[name] = arm

json.dump(res, open(f"{OUT}/pt2_score.json", "w"), indent=1, default=str)
print(f"\n-> {OUT}/pt2_score.json")
