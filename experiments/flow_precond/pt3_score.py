#!/usr/bin/env python
"""GATE PT-3 scorer: pinned formulas from the rd-2-certified checkpoint.

E1/E2 (NSYS 16) + E3/E4 (NSYS 8), all MAP-entry adaptive arms, windows
250/500/1000 (asserted), scoring window = post-freeze rounds 1000-1500.
F-S convention PINNED per the rd-2 condition: offending eigendirections are
mapped to z-space as x = L^-T u (B-Cholesky of Sigma_ref) before the |cos|
test against the stored pt3_fs_reference vector (whitened-space |cos| ~ 0.095
would silently disable F-S). Login-node numpy only.
"""
import json

import numpy as np


def split_rhat_binary(ch):
    x = ch.astype(float); n2 = x.shape[1] // 2
    halves = np.concatenate([x[:, :n2], x[:, n2:2 * n2]], axis=0)
    W = halves.var(axis=1, ddof=1).mean()
    B = n2 * halves.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n2 - 1) / n2 * W + B / n2) / W)) if W > 0 else float("inf")


OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
MAMS = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/sim_carousel/messy_tests/dpie/mams/arrays.npz"
BAND = (0.32, 0.49)
EEVPD_BAND = (1e-4, 2e-3)
GENEIG_BAND = (1.0 / 3.0, 3.0)
DRIFT_ENV = 0.05
W_CENTER = 0.42
POCKET_COL, POCKET_THR = 6, -22.35
WINDOWS = [250, 500, 1000]
FLOORS = dict(E1=175, E2=175, E3=88, E4=88)   # all-main basis, op-7-scaled below
FS_COS = 0.8

draws = np.load(MAMS)["samples_z"].reshape(-1, 33).astype(np.float64)
pk = draws[:, POCKET_COL] > POCKET_THR
cov_P, cov_M = np.cov(draws[pk].T), np.cov(draws[~pk].T)
dmu = draws[pk].mean(0) - draws[~pk].mean(0)
fs_ref = np.load(f"{OUT}/pt3_fs_reference.npz")["eigvec_z"]


def sigma_ref(w):
    return w * cov_P + (1 - w) * cov_M + w * (1 - w) * np.outer(dmu, dmu)


res = {}
for name in ("E1", "E2", "E3", "E4"):
    tag = f"D2_{name}pt3"
    try:
        d = np.load(f"{OUT}/arrays_{tag}.npz")
    except FileNotFoundError:
        print(f"\n[{name}] MISSING — skipped"); continue
    rounds = int(d["rounds_done"])
    assert float(d["betas"][-1]) == 1.0
    assert list(d["metric_windows"]) == WINDOWS, "non-pinned windows!"
    assert d["cold_ind"].shape[0] >= rounds, "truncated buffer (shape guard)"
    if rounds < 1500:
        print(f"*** {name} INCOMPLETE {rounds}/1500 — op-7 scaling ***")
    cold = d["cold_ind"][:rounds]
    lo = max(WINDOWS[-1], rounds - 500)
    if rounds <= lo:
        print(f"\n[{name}] no post-freeze rounds — skipped"); continue
    win = cold[lo:]
    nsys = cold.shape[1]
    sysm = win.mean(axis=0)
    m, sd = float(sysm.mean()), float(sysm.std(ddof=1))
    se = sd / np.sqrt(nsys)
    fl = FLOORS[name] * min(1.0, rounds / 1500.0)
    rtp = int(d["round_trips_pocket"].sum())
    ev = d["eevpd"][:rounds]
    ev_med = np.median(ev[lo:], axis=0)          # post-freeze only (audit A2)
    ev_ok = all(EEVPD_BAND[0] <= v <= EEVPD_BAND[1] for v in ev_med)
    tail_rung = (ev[lo:] > 2e-3).mean(axis=0)
    att, acc = d["swap_attempts"].astype(float), d["swap_accepts"].astype(float)
    pacc = acc.sum(1) / np.maximum(att.sum(1), 1)
    pacc_ok = bool(np.all((pacc >= 0.25) & (pacc <= 0.65)))
    rhat = split_rhat_binary(cold[lo:].T)        # post-freeze only (audit A1)
    in_band = BAND[0] <= m <= BAND[1]
    near = min(abs(m - BAND[0]), abs(m - BAND[1])) < DRIFT_ENV
    # scored gen-eig vs Sigma_ref(w-hat)
    ref = sigma_ref(m)
    L = np.linalg.cholesky(ref)
    A = d["metric_frozen"][-1]
    X = np.linalg.solve(L, np.linalg.solve(L, A).T); X = (X + X.T) / 2
    g, V = np.linalg.eigh(X)
    order = np.argsort(g); g, V = g[order], V[:, order]
    n_out = int(((g < GENEIG_BAND[0]) | (g > GENEIG_BAND[1])).sum())
    n_soft_hi = int(((g > 3) & (g <= 10)).sum())
    # F-S: low-side exits mapped to z-space x = L^-T u (PINNED convention)
    fs_fired = False
    for i in np.where(g < GENEIG_BAND[0])[0]:
        x = np.linalg.solve(L.T, V[:, i]); x /= np.linalg.norm(x)
        cosr = abs(float(x @ fs_ref))
        fs_arm = name in ("E1", "E2")   # F-S pinned to E1/E2 (audit A4)
        print(f"  [F-S check {name}] low-side axis g={g[i]:.3f}: |cos(stored)| = {cosr:.3f}"
              + (" -> F-S FIRES (systematic)" if (cosr >= FS_COS and fs_arm)
                 else (" (matches stored dir; non-E1/E2 arm — report only)" if cosr >= FS_COS
                       else " (distinct direction)")))
        if cosr >= FS_COS and fs_arm:
            fs_fired = True
    clauses = dict(rt=rtp >= fl, occ=in_band, eevpd=ev_ok,
                   pair_acc=pacc_ok, nan=int(d["n_revert"]) == 0,
                   geneig=n_out == 0)
    arm = dict(rounds=rounds, m=m, sd=sd, se=se, rt_pocket=rtp, floor_eff=fl,
               ev_med=ev_med.tolist(), tail_rung=tail_rung.tolist(),
               pair_acc=pacc.tolist(), split_rhat=rhat, nsys=nsys,
               geneig_full=[float(v) for v in g], n_out=n_out,
               n_soft_hi=n_soft_hi, fs_fired=fs_fired,
               n_revert=int(d["n_revert"]))
    print(f"\n[{name}] rounds={rounds} m={m:.4f}±{se:.4f} RTpocket={rtp} (floor {fl:.0f}) "
          f"geneig [{g[0]:.3f}, {g[-1]:.3f}] out={n_out} soft-hi={n_soft_hi} "
          f"split-Rhat={rhat:.4f} NaN={arm['n_revert']}")
    print(f"  eevpd_med={['%.1e' % v for v in ev_med]}; tail>2e-3 per rung "
          f"{[round(v, 3) for v in tail_rung]} (B3 report); pair acc "
          f"{np.round(pacc, 3).tolist()}")
    if near:
        print(f"  *** NEAR-EDGE (±{DRIFT_ENV}): corroboration rule applies ***")
    if not in_band:
        L42 = np.linalg.cholesky(sigma_ref(W_CENTER))
        Y = np.linalg.solve(L42, np.linalg.solve(L42, A).T)
        g42 = np.sort(np.linalg.eigvalsh((Y + Y.T) / 2.0))
        arm["geneig_vs_042"] = [float(v) for v in g42]
        print(f"  blind-spot ix: gen-eig vs Sigma_ref(0.42) [{g42[0]:.3f}, {g42[-1]:.3f}]")
    if float(g[-1]) > 10:
        print("  *** F-R: axis > 10 post-freeze — freeze-timing fix insufficient ***")
    elif (n_out > 0 and n_soft_hi == n_out and n_out <= 4 and clauses["rt"]
          and in_band and ev_ok and pacc_ok and clauses["nan"]):
        print("  ZONE (3,10] <=4 axes with transport+occ+health passing: "
              "IMPROVED-PARTIAL — certifiable with recorded inflation (B3 caveats attach)")
    if clauses["geneig"] and m < BAND[0] - DRIFT_ENV:
        print("  A1 ZONE candidate: metric fixed, occ SHORT beyond near-edge — check "
              "coldocc trace rising before reading budget-limited")
    elif clauses["geneig"] and m > BAND[1] + DRIFT_ENV:
        print("  occ HIGH beyond near-edge: UNROUTED -> report-to-human")
    if 1.05 < rhat <= 1.2 and clauses["rt"]:
        print(f"  split-Rhat {rhat:.3f} in (1.05,1.2] with transport: budget-limited "
              f"reading (report-only diagnostic this gate)")
    print(f"  clauses: {clauses}")
    arm["clauses"] = {k: bool(v) for k, v in clauses.items()}
    res[name] = arm

# W-s seed pairs + W-p assembly
def pair(a, b):
    if a in res and b in res:
        da, db = res[a], res[b]
        lim = 2 * float(np.hypot(da["se"], db["se"]))
        ok = abs(da["m"] - db["m"]) <= lim
        print(f"\nW-s [{a}/{b}]: |{da['m']:.4f}-{db['m']:.4f}|={abs(da['m']-db['m']):.4f} "
              f"<= {lim:.4f}: {ok}")
        return ok
    return None

ws16, ws8 = pair("E1", "E2"), pair("E3", "E4")
if all(n in res for n in ("E1", "E2", "E3", "E4")):
    def wg_ok(n):
        r = res[n]
        return r["n_out"] == 0 or (r["n_soft_hi"] == r["n_out"] and r["n_out"] <= 4)
    def health_ok(n):
        c = res[n]["clauses"]
        return c["eevpd"] and c["pair_acc"] and c["nan"]
    cand = all(res[n]["rt_pocket"] >= res[n]["floor_eff"]
               and BAND[0] <= res[n]["m"] <= BAND[1]
               and wg_ok(n) and health_ok(n) for n in ("E3", "E4"))
    guard16 = all((res[n]["rt_pocket"] >= res[n]["floor_eff"]) and
                  (BAND[0] - DRIFT_ENV <= res[n]["m"] <= BAND[1] + DRIFT_ENV)
                  for n in ("E1", "E2"))
    ws_all = bool(ws16) and bool(ws8)   # strict reading: "W-s holds" = both pairs
    wp = cand and ws_all and guard16
    verdict = ("PROPOSED DELIVERABLE (UNCERTIFIED; 1e-6*I ratification pending)"
               if wp else ("SUSPENDED pending explanation (NSYS-16 guard)" if cand and ws_all
                           else "NOT assembled (see zones)"))
    print(f"\nW-p (certification): NSYS-8 clauses+geneig+health {cand}; "
          f"W-s both pairs {ws_all} (16: {ws16}, 8: {ws8}); NSYS-16 guard {guard16} "
          f"-> {verdict}")
json.dump(res, open(f"{OUT}/pt3_score.json", "w"), indent=1, default=str)
print(f"\n-> {OUT}/pt3_score.json")
