#!/usr/bin/env python
"""GATE PT-0b scoring: applies the checkpoint's PINNED formulas verbatim.

Reads arrays_B1_P{1,2}pt0b.npz (balanced) and arrays_B4_P{3,4}pt0b.npz (all-main).
Outputs every W-b clause + routed zone verdict as PROPOSED/UNCERTIFIED numbers.
Login-node safe (numpy only). No thresholds are recomputed here -- all constants
are transcribed from the checkpoint (commit 79cdccd + rd-2 numeric fix).
"""
import json

import numpy as np

OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
LAST = 500
RT_POINT = 300.0          # per-arm point prediction
POCKET_FLOOR = 10         # W-b1 floor per balanced arm
EEVPD_BAND = (1e-4, 2e-3)
ACC_BAND = (0.25, 0.65)
POWER_SE = 0.06           # W-b2 POWER clause
CANDS = (0.10, 0.35)      # pocket-weight candidate readings

arms = {}
for p, tag in [("P1", "B1_P1pt0b"), ("P2", "B1_P2pt0b"),
               ("P3", "B4_P3pt0b"), ("P4", "B4_P4pt0b")]:
    d = np.load(f"{OUT}/arrays_{tag}.npz")
    rounds = int(d["rounds_done"])
    cold = d["cold_ind"][:rounds]
    last = min(LAST, rounds)
    sysm = cold[-last:].mean(axis=0)                     # per-system last-500 means
    att, acc = d["swap_attempts"].astype(float), d["swap_accepts"].astype(float)
    pacc = acc.sum(1) / np.maximum(att.sum(1), 1)
    ev = d["eevpd"][:rounds] if "eevpd" in d else None
    ev_med = np.median(ev[-last:], axis=0) if ev is not None else None
    arms[p] = dict(
        rounds=rounds, sysm=sysm, m=float(sysm.mean()),
        sd=float(sysm.std(ddof=1)), se=float(sysm.std(ddof=1) / 4.0),
        rt=int(d["round_trips"].sum()),
        rt_pocket=int(d["round_trips_pocket"].sum()),
        rt_main=int(d["round_trips_main"].sum()),
        pocket_ca=int(d["pocket_cold_arrivals"].sum()),
        pair_acc=pacc.tolist(), ev_med=None if ev_med is None else ev_med.tolist(),
        n_revert=int(d["n_revert"]), init=float(d["init_cold_occ"]),
        flips=int(np.abs(np.diff(cold.astype(int), axis=0)).sum()),
    )
    a = arms[p]
    print(f"[{p}] rounds={a['rounds']} m={a['m']:.4f} sd={a['sd']:.4f} se={a['se']:.4f} "
          f"RT={a['rt']} RTpocket={a['rt_pocket']} RTmain={a['rt_main']} "
          f"pocketCA={a['pocket_ca']} flips_total={a['flips']} reverts={a['n_revert']}")
    print(f"     pair_acc={np.round(pacc,3).tolist()}")
    if ev_med is not None:
        print(f"     eevpd_med(last{last})={['%.1e'%v for v in ev_med]}")

# pooled pairs (32 systems each)
bal = np.concatenate([arms["P1"]["sysm"], arms["P2"]["sysm"]])
alm = np.concatenate([arms["P3"]["sysm"], arms["P4"]["sysm"]])
m_bal, m_alm = bal.mean(), alm.mean()
se_bal = bal.std(ddof=1) / np.sqrt(32)
se_alm = alm.std(ddof=1) / np.sqrt(32)
se_comb = float(np.hypot(se_bal, se_alm))
diff = float(m_bal - m_alm)
pooled = float((m_bal / se_bal**2 + m_alm / se_alm**2) / (1 / se_bal**2 + 1 / se_alm**2))
pooled_se = float(1 / np.sqrt(1 / se_bal**2 + 1 / se_alm**2))
ci = (pooled - 2 * se_comb, pooled + 2 * se_comb)

print("\n===== W-b scoring (PROPOSED / UNCERTIFIED) =====")
for p in ("P1", "P2"):
    a = arms[p]
    print(f"W-b1[{p}]: pocket RT {a['rt_pocket']} >= {POCKET_FLOOR}: "
          f"{a['rt_pocket'] >= POCKET_FLOOR}; total {a['rt']} in x/4 of {RT_POINT:.0f} "
          f"[{RT_POINT/4:.0f},{RT_POINT*4:.0f}]: {RT_POINT/4 <= a['rt'] <= RT_POINT*4}; "
          f"annulus(x4,x10]: {RT_POINT*4 < a['rt'] <= RT_POINT*10 or RT_POINT/10 <= a['rt'] < RT_POINT/4}; "
          f"F-b4(out x10): {a['rt'] > RT_POINT*10 or a['rt'] < RT_POINT/10}")
print(f"W-b2 agreement: m_bal={m_bal:.4f}(se {se_bal:.4f}) m_allmain={m_alm:.4f}(se {se_alm:.4f}) "
      f"|diff|={abs(diff):.4f} vs 2*se_comb={2*se_comb:.4f} -> "
      f"{'AGREE' if abs(diff) <= 2*se_comb else ('2-3sigma zone' if abs(diff) <= 3*se_comb else 'DISAGREE>3sigma')}")
mv_bal = abs(m_bal - 0.5) / (bal.std(ddof=1) / np.sqrt(32))
mv_alm = abs(m_alm - 0.0) / (alm.std(ddof=1) / np.sqrt(32))
print(f"W-b2 movement: balanced |m-0.5|/se = {mv_bal:.1f} (>=3: {mv_bal >= 3}); "
      f"all-main |m-0.0|/se = {mv_alm:.1f} (>=3: {mv_alm >= 3})")
print(f"W-b2 POWER: se_comb={se_comb:.4f} <= {POWER_SE}: {se_comb <= POWER_SE}")
print(f"W-b2 adjudication: pooled={pooled:.4f}+-{pooled_se:.4f}, CI(2*se_comb)=({ci[0]:.4f},{ci[1]:.4f}); "
      f"excludes 0.10: {not (ci[0] <= CANDS[0] <= ci[1])}; excludes 0.35: {not (ci[0] <= CANDS[1] <= ci[1])}")
for p, a in arms.items():
    if a["ev_med"] is not None:
        inband = [EEVPD_BAND[0] <= v <= EEVPD_BAND[1] for v in a["ev_med"]]
        print(f"W-b3[{p}]: eevpd in band per rung {inband}; "
              f"pair acc in {ACC_BAND}: {[ACC_BAND[0] <= v <= ACC_BAND[1] for v in a['pair_acc']]}; "
              f"NaN reverts {a['n_revert']} == 0: {a['n_revert'] == 0}")
for pa, pb in (("P1", "P2"), ("P3", "P4")):
    a, b = arms[pa], arms[pb]
    lim = 2 * np.hypot(a["se"], b["se"])
    print(f"W-b4[{pa}vs{pb}]: |{a['m']:.4f}-{b['m']:.4f}|={abs(a['m']-b['m']):.4f} <= {lim:.4f}: "
          f"{abs(a['m']-b['m']) <= lim}")

def split_rhat_binary(ch):
    x = ch.astype(float); n2 = x.shape[1] // 2
    halves = np.concatenate([x[:, :n2], x[:, n2:2*n2]], axis=0)
    W = halves.var(axis=1, ddof=1).mean()
    B = n2 * halves.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n2 - 1) / n2 * W + B / n2) / W)) if W > 0 else float("inf")

for p, tag in [("P1", "B1_P1pt0b"), ("P2", "B1_P2pt0b"), ("P3", "B4_P3pt0b"), ("P4", "B4_P4pt0b")]:
    d = np.load(f"{OUT}/arrays_{tag}.npz")
    rounds = int(d["rounds_done"]); rl = min(1000, rounds)
    r = split_rhat_binary(d["cold_ind"][:rounds][-rl:].T)
    zone = "PASS" if r <= 1.05 else ("budget-limited zone (1.05,1.2]" if r <= 1.2 else "FAIL")
    print(f"W-b5[{p}]: cold split-Rhat(last {rl}) = {r:.4f} -> {zone}")

json.dump({p: {k: v for k, v in a.items() if k != "sysm"} for p, a in arms.items()}
          | dict(m_bal=float(m_bal), m_allmain=float(m_alm), se_comb=se_comb,
                 pooled=pooled, pooled_se=pooled_se, ci=list(ci)),
          open(f"{OUT}/pt0b_score.json", "w"), indent=1, default=str)
print(f"\n-> {OUT}/pt0b_score.json")
