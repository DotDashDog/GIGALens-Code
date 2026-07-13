#!/usr/bin/env python
"""GATE PT-4 link L pre-registered validation of ladder_recipe.py.

L1  : port reproduces the archived 21-knot 1.0-nat full-range design
      (ladder_design_power.json) from arrays_A_power.npz: max |dknot| < 1e-9,
      total_cost match 1e-9-relative.
L1b : certified-R6 end-to-end -- design_ladder with beta_min from the pinned
      ln(100) beta_min_rule and TARGET = 1.0 nat/pair (PT-0b convention; the
      certified 0.894 realized nats/pair comes out of the re-knot arithmetic)
      must reproduce [0.3594, 0.4388, 0.5373, 0.6598, 0.8116, 1.0]: 1e-9 vs an
      in-script full-precision recompute from the json's betas_measured/sd_u,
      AND 4-decimal equality with the pinned knots.
L3  : nearest-mode classifier vs the pinned z[6] > -22.35 indicator on the
      MAMS64 position pool; disagreement reported (criterion <= 0.045 is
      applied by the scorer, not here).
L2  : needs the fresh seed-54 probe -- point --l2 at its arrays npz.

Exit status: nonzero iff L1 or L1b fails. Login-node safe (numpy + stdlib).
"""
import json
import math
import sys

import numpy as np

from ladder_recipe import (beta_min_rule, classify_nearest_mode, cost_curve,
                           design_ladder, knot_se, measure_sd_u,
                           target_from_acceptance)

OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
MAMS = ("/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/"
        "experiments/sim_carousel/messy_tests/dpie/mams/arrays.npz")
DISCARD = 1500          # pt0_ladder_design.py retained-window convention
TARGET = 1.0            # PT-0b desired-spacing input, nats/pair
PINNED_R6 = [0.3594, 0.4388, 0.5373, 0.6598, 0.8116, 1.0]
NSYS, BUDGET_STEPS, PROBE_STEPS = 16, 16500, 1500

# Arm A per-beta basin-class leakage fractions per 1500 retained probe steps.
# MACHINE-READABLE archive: carousel_gate_pt0_out/summary_A_power.json,
# per_beta[k]["leak_Minit"/"leak_Pinit"] (writer: carousel_gate_pt0.py lines
# 494-495/635-636); corroborated by the stdout archive full_A_power.log lines
# 42-51 ("leak M/P=...") and by the log's quoted 25.6/62.6% at beta=0.3594,
# 17.6/36.7% at beta=0.6 (docs/logs/carousel-mclmc-sampling.md, PT-0b entry).
# Loaded (not transcribed) below; conservative (smaller) direction per beta.


def load_leakage():
    s = json.load(open(f"{OUT}/summary_A_power.json"))
    betas = np.array([r["beta"] for r in s["per_beta"]])
    leak = np.array([min(r["leak_Minit"], r["leak_Pinit"]) for r in s["per_beta"]])
    return betas, leak


def check_L1():
    j = json.load(open(f"{OUT}/ladder_design_power.json"))
    d = np.load(f"{OUT}/arrays_A_power.npz")
    betas = np.asarray(d["betas"], np.float64)
    sd, se = measure_sd_u(np.asarray(d["u"]), DISCARD)
    sd_dev = np.abs(sd - np.array(j["sd_u"])).max()
    des = design_ladder(betas, sd, betas[0], TARGET)
    kdev = np.abs(des["knots"] - np.array(j["ladder"])).max()
    tdev = abs(des["total_cost"] - j["total_cost_nats"]) / j["total_cost_nats"]
    ok = kdev < 1e-9 and tdev < 1e-9
    print(f"[L1] n_rungs={des['n_rungs']} (archived {len(j['ladder'])}); "
          f"max|d sd|={sd_dev:.3e}  max|d knot|={kdev:.3e}  rel|d total|={tdev:.3e}")
    print(f"[L1] {'PASS' if ok else 'FAIL'}: archived 21-knot design reproduced "
          f"(max knot deviation {kdev:.3e} < 1e-9: {kdev < 1e-9}; "
          f"total rel {tdev:.3e} < 1e-9: {tdev < 1e-9})")
    return ok, (betas, sd, se)


def check_L1b():
    # erfc-model round trips (pinned in the module spec)
    rt1 = abs(math.erfc(target_from_acceptance(0.527) / 2.0) - 0.527)
    rt2 = abs(target_from_acceptance(math.erfc(0.5)) - 1.0)
    erf_ok = rt1 < 1e-12 and rt2 < 1e-10
    print(f"[L1b] erfc round-trips: |erfc(s/2)-0.527|={rt1:.2e}, "
          f"|target(erfc(0.5))-1|={rt2:.2e} -> {'ok' if erf_ok else 'FAIL'}")

    lb, leak = load_leakage()
    print("[L1b] NOTE: leakage table LOADED machine-readable from "
          "summary_A_power.json (per_beta leak_Minit/leak_Pinit; conservative "
          "= smaller direction per beta) -- no transcription needed.")
    bmin = beta_min_rule(lb, leak, NSYS, BUDGET_STEPS, PROBE_STEPS, level=0.99)
    # Transparency: candidate-own-rate reading would return the colder 0.5995
    # (16*0.176*11 = 31 >= ln(100)); the pinned rule's conservative
    # next-colder-neighbor estimate (PT-0b grader advisory 7) yields 0.3594.
    own = [b for b, p in zip(lb, leak) if NSYS * p * (BUDGET_STEPS / PROBE_STEPS)
           >= math.log(100.0)]
    print(f"[L1b] beta_min_rule -> {bmin:.10g} (own-rate reading would give "
          f"{max(own):.10g}; conservative-neighbor form is the pinned one)")
    bmin_ok = round(bmin, 4) == 0.3594

    j = json.load(open(f"{OUT}/ladder_design_power.json"))
    betas = np.array(j["betas_measured"])
    sd = np.array(j["sd_u"])
    des = design_ladder(betas, sd, bmin, TARGET)

    # In-script full-precision reference recompute (independent arithmetic;
    # the log records the certified knots to 4 decimals only).
    bgrid = np.geomspace(betas[0], 1.0, 4000)
    sdg = np.exp(np.interp(np.log(bgrid), np.log(betas), np.log(np.maximum(sd, 1e-12))))
    cost = np.concatenate([[0.0], np.cumsum(0.5 * (sdg[1:] + sdg[:-1]) * np.diff(bgrid))])
    c0 = float(np.interp(bmin, bgrid, cost))
    total_r = float(cost[-1] - c0)
    n_r = int(np.ceil(total_r / TARGET)) + 1
    kref = np.interp(np.linspace(c0, cost[-1], n_r), cost, bgrid)
    kref[0], kref[-1] = bmin, 1.0
    npp_ref = total_r / (n_r - 1)

    kdev = np.abs(des["knots"] - kref).max() if des["n_rungs"] == n_r else np.inf
    npp_dev = abs(des["nats_per_pair"] - npp_ref)
    r4 = [round(float(k), 4) for k in des["knots"]]
    pin_ok = r4 == PINNED_R6
    npp_ok = round(des["nats_per_pair"], 3) == 0.894
    ok = (erf_ok and bmin_ok and des["n_rungs"] == 6 and kdev < 1e-9
          and npp_dev < 1e-9 and pin_ok and npp_ok)
    print(f"[L1b] R6 design: n_rungs={des['n_rungs']} total={des['total_cost']:.6f} "
          f"nats realized={des['nats_per_pair']:.6f} nats/pair "
          f"(log quotes 0.894; full precision {npp_ref!r})")
    print(f"[L1b] knots={r4} vs pinned {PINNED_R6}: {'match' if pin_ok else 'MISMATCH'}; "
          f"max|d knot| vs full-precision reference = {kdev:.3e}")
    print(f"[L1b] {'PASS' if ok else 'FAIL'}: beta_min 0.3594 emerges={bmin_ok}, "
          f"knots 1e-9={kdev < 1e-9}, pinned-4dp={pin_ok}, nats/pair 0.894={npp_ok}")
    return ok


def check_L3():
    d = np.load(MAMS)
    z = np.asarray(d["samples_z"], np.float64).reshape(-1, 33)
    pocket = z[:, 6] > -22.35                      # pinned indicator
    centers = np.stack([z[~pocket].mean(0), z[pocket].mean(0)])  # [main, pocket]
    cov = np.cov(z, rowvar=False)                  # pooled cov of ALL draws
    pred = classify_nearest_mode(z, centers, cov) == 1
    n = len(z)
    tp = int(np.sum(pred & pocket)); tn = int(np.sum(~pred & ~pocket))
    fp = int(np.sum(pred & ~pocket)); fn = int(np.sum(~pred & pocket))
    dis = (fp + fn) / n
    print(f"[L3] n={n} pocket(indicator)={int(pocket.sum())}; confusion "
          f"[pred-pocket&pocket={tp}, pred-main&main={tn}, "
          f"pred-pocket&main={fp}, pred-main&pocket={fn}]")
    print(f"[L3] disagreement fraction = {dis:.6f} "
          f"(pre-registered criterion <= 0.045: "
          f"{'met' if dis <= 0.045 else 'NOT met'}; applied HERE — audit A2-2)")
    return dis


def check_L2(npz_path, betas_ref, sd_ref, se_ref):
    """Fresh-probe mode: same rung count AND every knot within 3x the
    propagated knot-position se (delta method through the cost integral)."""
    lb, leak = load_leakage()
    bmin = beta_min_rule(lb, leak, NSYS, BUDGET_STEPS, PROBE_STEPS)
    ref = design_ladder(betas_ref, sd_ref, bmin, TARGET)
    se_ref_knots = knot_se(betas_ref, sd_ref, se_ref, bmin, TARGET)
    d = np.load(npz_path)
    betas = np.asarray(d["betas"], np.float64)
    sd, se = measure_sd_u(np.asarray(d["u"]), DISCARD)
    fresh = design_ladder(betas, sd, bmin, TARGET)
    se_fresh = knot_se(betas, sd, se, bmin, TARGET)
    print(f"[L2] fresh probe {npz_path}: n_rungs {fresh['n_rungs']} vs ref {ref['n_rungs']}")
    if fresh["n_rungs"] != ref["n_rungs"]:
        print("[L2] FAIL: rung count differs")
        return False
    se_comb = np.sqrt(se_ref_knots ** 2 + se_fresh ** 2)
    dk = np.abs(fresh["knots"] - ref["knots"])
    ok = bool(np.all(dk <= 3.0 * np.maximum(se_comb, 1e-15)))
    for i, (a, b, s) in enumerate(zip(ref["knots"], fresh["knots"], se_comb)):
        print(f"  knot {i}: ref={a:.6f} fresh={b:.6f} |d|={abs(a - b):.2e} 3se={3 * s:.2e}")
    print(f"[L2] {'PASS' if ok else 'FAIL'}: all knots within 3x propagated se")
    return ok


def main():
    l2_npz = None
    if "--l2" in sys.argv:
        l2_npz = sys.argv[sys.argv.index("--l2") + 1]
    ok1, (betas_a, sd_a, se_a) = check_L1()
    ok1b = check_L1b()
    dis = check_L3()
    ok3 = dis <= 0.045
    print(f"[SUMMARY] L1  {'PASS' if ok1 else 'FAIL'}")
    print(f"[SUMMARY] L1b {'PASS' if ok1b else 'FAIL (PORT-BUG class: fix + re-audit; rd-2 condition 3)'}")
    print(f"[SUMMARY] L3  {'PASS' if ok3 else 'FAIL'} (disagreement={dis:.6f} <= 0.045)")
    ok2 = None
    if l2_npz is not None:
        ok2 = check_L2(l2_npz, betas_a, sd_a, se_a)
        print(f"[SUMMARY] L2  {'PASS' if ok2 else 'FAIL'}")
    # audit A2-2: W-L = L1 & L1b & L2 & L3 is ASSEMBLED HERE, not by pt4_score.py
    if ok2 is None:
        print(f"[SUMMARY] W-L: L1&L1b&L3 {'PASS' if (ok1 and ok1b and ok3) else 'FAIL'}; "
              f"L2 PENDING (fresh probe not yet run) -> W-L PENDING")
    else:
        print(f"[SUMMARY] W-L = L1 & L1b & L2 & L3 -> "
              f"{'PASS' if (ok1 and ok1b and ok2 and ok3) else 'FAIL'}")
    sys.exit(0 if (ok1 and ok1b and ok3 and (ok2 is None or ok2)) else 1)


if __name__ == "__main__":
    main()
