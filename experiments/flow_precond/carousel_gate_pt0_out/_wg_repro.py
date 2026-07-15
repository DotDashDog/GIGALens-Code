#!/usr/bin/env python
"""Standalone, read-only reproduction of the scored W-G / gen-eig quantity
for PT-4 (G1-G4) and PR (PR1-PR3) arms, using the EXACT formulas transcribed
from pt4_score.py and pt5a_r2_score.py. No new runs; arrays + reference are
read from existing archives. Prints a comparison table.

Formula provenance (quoted, not re-derived):
  Sigma_ref(w) construction: pt4_score.py lines 88-99 == pt5a_r2_score.py
    lines 400-411 (identical: MAMS positions-only pools, cov_P/cov_M/dmu,
    sigma_ref(w) = w*cov_P + (1-w)*cov_M + w*(1-w)*outer(dmu,dmu)).
  gen-eig reduction: pt4_score.py lines 173-179 (geneig()/inline) ==
    pt5a_r2_score.py lines 224-226 (geneig_vs()): both do
      L = cholesky(Sigma_ref(m)); X = solve(L, solve(L, A).T); X=(X+X.T)/2
      g, V = eigh(X); sort ascending.
    A = d["metric_frozen"][-1]  (the COLD rung, i.e. index -1 of the 6-rung
    axis; "post-freeze" in the sense that metric_frozen is the frozen-500
    within-estimator covariance already recorded at that rung -- there is
    no separate freeze/final choice inside this array, metric_frozen IS the
    frozen quantity per the freeze-500 estimator).
  occupancy mean m: pt4_score.py lines 150-157 == pt5a_r2_score.py lines
    557,575,593-598 (identical: cold=d["cold_ind"][:rounds];
    lo=max(500,rounds-500); win=cold[lo:]; sysm=win.mean(0); m=sysm.mean()).
  PT-4's own gate (wg_ok, pt4_score.py lines 367-369): n_hard_hi==0 (NO axis
    > 10) AND n_hard_lo==0 (no axis < 0.1) AND n_soft_hi<=4 (<=4 axes in
    (3,10]). PT-4 has NO explicit "max<=30" cap anywhere in its own gating.
  PR's own W-G gate (pt5a_r2_score.py lines 619-620): wg_pass = g[-1]<=30
    AND n_gt10<=1 (n_gt10 = count of axes with g>10, WG_SOFT=10.0). THIS is
    the clause carrying the "~30" / "<=1 axis>10" threshold the user is
    asking about; it exists ONLY in the PR scorer, not in pt4_score.py.
  For direct comparability, this script ALSO applies the PR-style
  max<=30/<=1-axis>10 test to the PT-4 arms (labeled explicitly), while
  separately reporting PT-4's own native gate (wg_ok) result -- these are
  NOT claimed to be the same clause, just evaluated side by side.
"""
import numpy as np

OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
MAMS = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/sim_carousel/messy_tests/dpie/mams/arrays.npz"
POCKET_COL, POCKET_THR = 6, -22.35
SOFT_HI = 10.0        # WG_SOFT / SOFT_HI -- identical constant in both scorers
LOW_FLOOR = 0.1
GENEIG_BAND = (1.0 / 3.0, 3.0)
WG_MAX = 30.0         # PR-scorer-only cap

# ---- reference (identical construction in both scorers) -------------------
draws = np.load(MAMS)["samples_z"].reshape(-1, 33).astype(np.float64)
pk = draws[:, POCKET_COL] > POCKET_THR
cov_P, cov_M = np.cov(draws[pk].T), np.cov(draws[~pk].T)
dmu = draws[pk].mean(0) - draws[~pk].mean(0)


def sigma_ref(w):
    return w * cov_P + (1 - w) * cov_M + w * (1 - w) * np.outer(dmu, dmu)


def geneig(adapted, L):
    X = np.linalg.solve(L, np.linalg.solve(L, adapted).T)
    X = (X + X.T) / 2.0
    return np.linalg.eigh(X)


def occ_mean(d, rounds):
    cold = d["cold_ind"][:rounds]
    lo = max(500, rounds - 500)
    win = cold[lo:]
    sysm = win.mean(axis=0)
    return float(sysm.mean()), lo, cold


def score_arm(label, path, key_rounds="rounds_done"):
    d = np.load(path)
    rounds = int(d[key_rounds])
    m, lo, cold = occ_mean(d, rounds)
    ref = sigma_ref(m)
    L = np.linalg.cholesky(ref)
    A = d["metric_frozen"][-1]   # cold rung
    g, V = geneig(A, L)
    order = np.argsort(g)
    g = g[order]
    n_gt10 = int((g > SOFT_HI).sum())          # == n_hard_hi in pt4_score.py
    n_soft_hi = int(((g > GENEIG_BAND[1]) & (g <= SOFT_HI)).sum())
    n_hard_lo = int((g < LOW_FLOOR).sum())
    n_soft_lo = int(((g >= LOW_FLOOR) & (g < GENEIG_BAND[0])).sum())
    pt4_native_pass = (n_gt10 == 0) and (n_hard_lo == 0) and (n_soft_hi <= 4)
    pr_style_pass = (g[-1] <= WG_MAX) and (n_gt10 <= 1)
    print(f"[{label}] rounds={rounds} lo={lo} m(occ)={m:.4f} "
          f"gen-eig range [{g[0]:.3f}, {g[-1]:.3f}]  n_axes>10={n_gt10}  "
          f"n_soft_hi(3,10]={n_soft_hi}  n_hard_lo(<0.1)={n_hard_lo}  "
          f"n_soft_lo[0.1,1/3)={n_soft_lo}")
    return dict(label=label, rounds=rounds, m=m, g=g, max_g=float(g[-1]),
                n_gt10=n_gt10, n_soft_hi=n_soft_hi, n_hard_lo=n_hard_lo,
                n_soft_lo=n_soft_lo, pt4_native_pass=pt4_native_pass,
                pr_style_pass=pr_style_pass)


print("=" * 100)
print("PT-4 arms (pt4_score.py construction: A = metric_frozen[-1], "
      "Sigma_ref(m) with m = per-arm occ mean)")
print("=" * 100)
pt4_files = dict(G1="arrays_D2_G1pt4.npz", G2="arrays_D2_G2pt4.npz",
                  G3="arrays_D2_G3pt4.npz", G4="arrays_D1_G4pt4.npz")
pt4_res = {}
for name, fn in pt4_files.items():
    pt4_res[name] = score_arm(name, f"{OUT}/{fn}")

print()
print("=" * 100)
print("PR arms (pt5a_r2_score.py construction: same A/Sigma_ref recipe, "
      "production npz, W-G clause)")
print("=" * 100)
pr_files = dict(PR1="arrays_PR_PR1pt5ar2.npz", PR2="arrays_PR_PR2pt5ar2.npz",
                 PR3="arrays_PR_PR3pt5ar2.npz")
pr_res = {}
for name, fn in pr_files.items():
    pr_res[name] = score_arm(name, f"{OUT}/{fn}")

print()
print("=" * 100)
print("TABLE: scored gen-eig quantity vs each scorer's OWN gate, "
      "+ PR-style max<=30/<=1-axis>10 applied uniformly for direct compare")
print("=" * 100)
hdr = (f"{'arm':5s} {'max g':>8s} {'#>10':>5s} {'PR-style(<=30,<=1)':>20s} "
       f"{'PT4-native(0 axes>10,<=4 in(3,10])':>36s}")
print(hdr)
for name, r in pt4_res.items():
    print(f"{name:5s} {r['max_g']:8.2f} {r['n_gt10']:5d} "
          f"{'PASS' if r['pr_style_pass'] else 'FAIL':>20s} "
          f"{'PASS' if r['pt4_native_pass'] else 'FAIL':>36s}")
for name, r in pr_res.items():
    print(f"{name:5s} {r['max_g']:8.2f} {r['n_gt10']:5d} "
          f"{'PASS' if r['pr_style_pass'] else 'FAIL':>20s} "
          f"{'n/a (PT-4-only gate)':>36s}")
