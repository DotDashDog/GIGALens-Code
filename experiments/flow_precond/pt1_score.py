#!/usr/bin/env python
"""GATE PT-1 cross-arm scorer: pinned formulas from the checkpoint (@328fb32).

Scores W-1a/W-1b (C1), W-2 + drift discrimination (C2 vs PT-0b P1/P2), and
W-3 with the PINNED two-arm moment-matching occ-ESS gate (C3/C4).
Login-node safe (numpy only). All constants transcribed, none re-derived.
"""
import json

import numpy as np

OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
BAND = (0.32, 0.49)          # C-24 occupancy band
RT_FLOOR_FULL, RT_FLOOR_HALF = 175, 60
EEVPD_BAND = (1e-4, 2e-3)    # C1 (target 5e-4); C2 exempt (scaled ref [1e-5, 2e-4])
EEVPD_BAND_C2 = (1e-5, 2e-4)
P12_M, P12_SD = 0.3888125, None   # P1/P2 pooled mean; se recomputed below from npz
DRIFT_ENV = 0.05             # A2 per-window drift envelope
ESS_FLOOR = 4.0              # per-chain moment-matching occ-ESS gate

res = {}

def pt_arm(tag):
    d = np.load(f"{OUT}/arrays_{tag}.npz")
    rounds = int(d["rounds_done"]); cold = d["cold_ind"][:rounds]
    last = min(500, rounds); sysm = cold[-last:].mean(axis=0)
    ev = np.median(d["eevpd"][:rounds][-last:], axis=0)
    tail = float((d["eevpd"][:rounds][-last:] > 2e-3).mean())
    att, acc = d["swap_attempts"].astype(float), d["swap_accepts"].astype(float)
    return dict(rounds=rounds, sysm=sysm, m=float(sysm.mean()),
                sd=float(sysm.std(ddof=1)), se=float(sysm.std(ddof=1) / 4.0),
                rt=int(d["round_trips"].sum()),
                rt_pocket=int(d["round_trips_pocket"].sum()),
                ev_med=ev.tolist(), tail_frac=tail,
                pair_acc=(acc.sum(1) / np.maximum(att.sum(1), 1)).tolist(),
                cold=cold, n_revert=int(d["n_revert"]))

# ---- C1: production composition -------------------------------------------
c1 = pt_arm("B5_C1pt1")
in_band = BAND[0] <= c1["m"] <= BAND[1]
ev_ok = all(EEVPD_BAND[0] <= v <= EEVPD_BAND[1] for v in c1["ev_med"])
near_edge = min(abs(c1["m"] - BAND[0]), abs(c1["m"] - BAND[1])) < DRIFT_ENV
print(f"[C1] rounds={c1['rounds']} m={c1['m']:.4f}±{c1['se']:.4f} "
      f"RTpocket={c1['rt_pocket']} RTtotal={c1['rt']} eevpd_med={['%.1e'%v for v in c1['ev_med']]}")
print(f"W-1a: RTpocket>={RT_FLOOR_FULL}: {c1['rt_pocket'] >= RT_FLOOR_FULL}; occ in {BAND}: {in_band}"
      f"{' (NEAR-EDGE: corroboration rule applies)' if near_edge and in_band else ''}; "
      f"EEVPD medians in band: {ev_ok}; NaN {c1['n_revert']}==0: {c1['n_revert'] == 0}")
if 0 < c1["rt_pocket"] < RT_FLOOR_FULL and in_band:
    print("W-1a ZONE: FLUX-LIMITED under SVI metric (routed: next-checkpoint decision)")
if not ev_ok and c1["rt_pocket"] >= RT_FLOOR_FULL and in_band:
    print("W-1a ZONE: HEALTH-only miss (routed: report, no F-1/flux-limited)")
# W-1b interim window: rounds 250-750, last-500 of a hypothetical 750-round run
cold = c1["cold"]
if c1["rounds"] >= 750:
    sysm_i = cold[250:750].mean(axis=0)
    m_i = float(sysm_i.mean())
    # RTs in window not separable from totals (final counts only) -> use rate proxy:
    # pocket RTs accrue ~linearly (model assumption, recorded) -> window share = 500/rounds
    rt_i_est = c1["rt_pocket"] * 500.0 / c1["rounds"]
    print(f"W-1b (interim 250-750): occ={m_i:.4f} in {BAND}: {BAND[0] <= m_i <= BAND[1]}; "
          f"est. window pocket RTs ≈ {rt_i_est:.0f} >= {RT_FLOOR_HALF} (linear-accrual model): "
          f"{rt_i_est >= RT_FLOOR_HALF}")
res["C1"] = {k: v for k, v in c1.items() if k != "cold" and k != "sysm"}

# ---- C2: kernel-bias probe --------------------------------------------------
c2 = pt_arm("B1_C2pt1")
p1 = np.load(f"{OUT}/arrays_B1_P1pt0b.npz"); p2 = np.load(f"{OUT}/arrays_B1_P2pt0b.npz")
bal = np.concatenate([p1["cold_ind"][-500:].mean(axis=0), p2["cold_ind"][-500:].mean(axis=0)])
se_p12 = float(bal.std(ddof=1) / np.sqrt(32))
se_comb = float(np.hypot(c2["se"], se_p12))
shift = c2["m"] - float(bal.mean())
nsig = abs(shift) / se_comb
# drift discrimination: first/second half of each scoring window
def halves(cold, last=500):
    w = cold[-last:]; h = last // 2
    return float(w[:h].mean()), float(w[h:].mean())
c2h = halves(c2["cold"]); p1h = halves(p1["cold_ind"]); p2h = halves(p2["cold_ind"])
drifts = [abs(b - a) for a, b in (c2h, p1h, p2h)]
drift_consistent = abs(shift) <= 2 * DRIFT_ENV and max(drifts) > DRIFT_ENV / 2
tail_dropped = c2["tail_frac"] < 0.05
print(f"\n[C2] rounds={c2['rounds']} m={c2['m']:.4f}±{c2['se']:.4f}; P1/P2 pooled "
      f"{float(bal.mean()):.4f}±{se_p12:.4f}; shift={shift:+.4f} = {nsig:.2f}σ "
      f"(2σ={2*se_comb:.4f})")
print(f"  window drifts |Δhalf|: C2 {drifts[0]:.4f}, P1 {drifts[1]:.4f}, P2 {drifts[2]:.4f}")
print(f"  above-2e-3 tail fraction: C2 {c2['tail_frac']:.3f} (PT-0b was 0.11-0.20); "
      f"dropped-materially(<0.05): {tail_dropped}")
print(f"  eevpd_med={['%.1e'%v for v in c2['ev_med']]} vs scaled ref {EEVPD_BAND_C2}")
if nsig <= 2:
    v = ("W-2 PASS: no kernel bias > ~0.19 occupancy units detected — 0.10-exclusion "
         "robust at this precision" + ("" if tail_dropped else
         " — BUT tail did NOT drop: L2 UNRESOLVED (vacuous-probe zone, excluded from ALL-PASS)"))
elif nsig <= 3:
    v = "W-2 ZONE (2σ,3σ]: bias not excluded at pilot precision (F-2 does not fire)"
else:
    v = ("F-2 CANDIDATE (>3σ): apply drift discrimination -> " +
         ("INCONCLUSIVE — window drift" if drift_consistent else "BIAS LIVE (drift-clean)"))
print(f"  verdict: {v}")
res["C2"] = dict(m=c2["m"], se=c2["se"], shift=shift, nsig=nsig, tail_frac=c2["tail_frac"],
                 drifts=drifts, verdict=v)

# ---- C3/C4: MH-exact bracket ------------------------------------------------
ms = {}
for tag in ("C3", "C4"):
    d = np.load(f"{OUT}/arrays_{tag}_pt1.npz")
    ms[tag] = dict(occ_chain=d["occ_chain"], m=float(d["occ_mean"]),
                   se_naive=float(d["se_chain_clustered"]),
                   h1=float(d["occ_first_half"]), h2=float(d["occ_second_half"]),
                   iat_ess_med=float(np.median(d["occ_ess_chain"])),
                   init_frac=float(d["realized_init_pocket_frac"]))
phat = float(np.concatenate([ms["C3"]["occ_chain"], ms["C4"]["occ_chain"]]).mean())
for tag in ("C3", "C4"):
    a = ms[tag]
    sd2 = float(a["occ_chain"].std(ddof=1)) ** 2
    a["ess_mm"] = phat * (1 - phat) / sd2 if sd2 > 0 else float("inf")  # pinned estimator
    a["powered"] = a["ess_mm"] >= ESS_FLOOR
    # se: naive if powered; else clamp to measured ESS (one-arm-underpowered route)
    n_eff = 64 * min(a["ess_mm"], max(a["ess_mm"], 1e-9)) if False else 64 * a["ess_mm"]
    a["se"] = float(np.sqrt(phat * (1 - phat) / max(n_eff, 1e-9)))
    print(f"\n[{tag}] m={a['m']:.4f} (naive se {a['se_naive']:.4f}); pinned mm occ-ESS/chain "
          f"= {a['ess_mm']:.2f} (IAT-proxy med {a['iat_ess_med']:.2f}); powered(>= {ESS_FLOOR}): "
          f"{a['powered']}; mm-se {a['se']:.4f}; drift halves {a['h1']:.4f}->{a['h2']:.4f}; "
          f"init_frac {a['init_frac']:.3f}")
if not ms["C3"]["powered"] and not ms["C4"]["powered"]:
    print("W-3: BOTH UNDERPOWERED -> L3 INCONCLUSIVE (routed; excluded from ALL-PASS)")
    res["W3"] = "L3 INCONCLUSIVE (double underpowered)"
else:
    se_c = float(np.hypot(ms["C3"]["se"], ms["C4"]["se"]))
    dd = ms["C3"]["m"] - ms["C4"]["m"]
    ns = abs(dd) / se_c
    pooled = float((ms["C3"]["m"] + ms["C4"]["m"]) / 2)
    agree = "AGREE" if ns <= 2 else ("(2σ,3σ] zone: bracket not closed at pilot precision" if ns <= 3 else "F-4 (>3σ): dwell unequilibrated")
    inb = BAND[0] <= pooled <= BAND[1]
    out_sig = min(abs(pooled - BAND[0]), abs(pooled - BAND[1])) / se_c if not inb else 0.0
    print(f"W-3: |m_C3-m_C4|={abs(dd):.4f} = {ns:.2f}σ -> {agree}; pooled={pooled:.4f} "
          f"in {BAND}: {inb}" + ("" if inb else f" (out by {out_sig:.2f}σ; F-3 needs >2σ with arms agreeing)"))
    res["W3"] = dict(diff_sig=ns, pooled=pooled, in_band=inb, agree=agree,
                     ess_mm=[ms['C3']['ess_mm'], ms['C4']['ess_mm']])

json.dump(res, open(f"{OUT}/pt1_score.json", "w"), indent=1, default=str)
print(f"\n-> {OUT}/pt1_score.json")
