#!/usr/bin/env python
"""GATE PT-5a-r2 scorer: pinned formulas from the rd-2-CERTIFY-RECOMMENDED
design checkpoint (docs/logs/carousel-mclmc-sampling.md, FIRST "carousel
GATE PT-5a-r2" checkpoint entry, incl. rd-1/rd-2 amendments, @aec56b6).

LINEAGE: adapted from experiments/flow_precond/pt5a_score.py (the
audit-certified pt5a scorer). This scheme REPLACES pt5a's in-run
u-stationarity trigger with a DEDICATED broad-init cheap-probe phase (Phase
P) whose readiness signal is a genuine convergence check (ladder + beta_min
re-derived on a growing trailing window, ready when they stop moving across
a 1.5x window-growth comparison) feeding an UNCHANGED C-24 production phase
(Phase Q, MAP + 1e-6*I entry, K knots, NSYS=16, freeze-500 within-estimator,
ROUNDS 1000 nominal). Arms PR1/PR2/PR3, seeds 60/61/62.

DROPPED vs pt5a: the in-run trigger + W-S u-stationarity clause (no in-run
retuning here -- tuning is entirely in the discarded probe phase). ADDED:
W-R (probe readiness fired within the pinned cap). KEPT (same clause
semantics + pinned constants as pt5a): W-T ladder-repro (BUT beta_min is now
an EXACT grid-point check -- no B2' pooled-rescue branch, since this scheme's
own readiness self-consistency, not a B2'-style counting rule, is what
determines beta_min); W-P production transport (RT floor, pooled occupancy,
W-s pairs); W-H production health; W-G regression guard (report-only zone
labels, gating only via max<=30 / <=1 axis>10).

Clauses (transcribed from the design checkpoint, never re-derived):
  W-R  per-arm pr_readiness_fired == True AND pr_t_ready_steps <= 2000 (the
       pinned cap, rd-2 item 2). Evaluated from the PRODUCTION npz's own
       pr_readiness_fired/pr_t_ready_steps (the copy that actually gated
       whether the run proceeded); cross-checked (report-only) against the
       probe companion's own readiness_fired/t_ready_steps. If a probe-only
       arm shows readiness_fired == False (production npz correctly absent)
       -> F-R, scored on probe data alone (graceful, A4-style).
  W-T  pr_knots has 6 rungs AND max|pr_knots - certified_knots| <= 3x the
       COMBINED knot se (probe's own in-run knot_se, from ladder_recipe.
       knot_se on the probe's OWN sd_u/leak/beta_min, combined in quadrature
       with the reference's knot_se -- ladder_recipe.knot_se again, on the
       ARCHIVED pt5a probe inputs, L1b/L2 pattern) AND pr_beta_min == 0.3594
       EXACTLY (grid point; no B2' pooled-rescue branch here -- unlike pt5a,
       this scheme's beta_min comes from the probe's OWN convergence check,
       not a rate-based discovery rule with sparse-count edge cases pooled
       across arms; a miss is a straight F-T).
  W-P  (supporting-but-gating, B7-style) per-arm RT_pocket >= 117 (=175*
       1000/1500 op-7, scaled further if clipped); pooled 3-arm occupancy
       over the 48 last-500 per-system means in (0.32, 0.49) with the A6
       near-edge rules (+ PT-4 A3-2 clipped-arm plot-rule warning); W-s all
       three pairs |dm| <= 2*sqrt(se_i^2+se_j^2). ANY per-arm RT<floor / W-s
       fail / beyond-near-edge pooled occ => W-a NOT assembled (routed as
       F-P, the checkpoint's Falsifiers-section label).
  W-H  per-arm EEVPD medians (post-lo, lo=max(500,rounds-500)) in
       [1e-4,2e-3]; pair acc in [0.25,0.65]; NaN reverts == 0.
  W-G  (regression guard) production post-freeze cold-rung gen-eig vs
       Sigma_ref(w-hat) (pt4 construction): max <= 30 AND <= 1 axis > 10;
       exceeding => F-H (HANDOFF/config finding, explicitly NOT metric-menu
       adjudication); {19,2,3,20}-family axis attribution + PT-4 zone labels
       reported (report-only, C-28 metric menu remains open).
  W-a  W-R ^ W-T ^ W-P ^ W-H ^ W-G on ALL arms => the pinned PROPOSED
       verdict (with zone/counting qualifiers + the rd-2 blind-spot note
       reaching the verdict string -- pt4 B3-1 lesson); else the routed
       F-R/F-T/F-H/F-P branch (checkpoint Falsifiers-section text).

NPZ KEY CONTRACT (anti-B2-bug pin -- EXACT names, no alias fallback, fail
LOUD with the full key inventory on any miss; never silently default):
  production  (arrays_PR_<tag>.npz): standard pt4-class keys (cold_ind,
              eevpd, swap_attempts, swap_accepts, round_trips_pocket,
              metric_frozen, metric_within_covs, metric_between_covs,
              rounds_done, n_revert, metric_windows, metric_estimator,
              betas) PLUS pr_t_ready_steps (int), pr_beta_min (float),
              pr_knots (array), pr_readiness_trace (array of (t, beta_min,
              max_knot_dev)), pr_readiness_fired (bool), pr_crossing_counts
              (array; DEMOTED to a report-only sanity floor here, N_x>=~12,
              per rd-1 B1/B2 -- not part of any gating clause).
  probe companion (arrays_PR_<tag>_probe.npz): betas_grid, sd_u, leak,
              knots, beta_min, knot_se, readiness_fired, t_ready_steps,
              readiness_trace.

Login-node numpy only. --selftest: synthetic 3-arm verdict-branch data
(all-pass; F-R one arm; F-T one arm; F-P pooled-occ-fail) plus direct unit
checks of the pure wr_check/wt_ladder_check helpers, asserting verdict
strings. Missing arms => pt5a-style skip; W-a requires all three.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_recipe import beta_min_rule, design_ladder, knot_se, measure_sd_u

OUT = os.environ.get(
    "GATE_PT5A_R2_OUT",
    "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/"
    "experiments/flow_precond/carousel_gate_pt0_out")  # env override: smoke fixtures only
MAMS = ("/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/"
        "experiments/sim_carousel/messy_tests/dpie/mams/arrays.npz")
ARMS = ("PR1", "PR2", "PR3")
TAGS = dict(PR1="PR1pt5ar2", PR2="PR2pt5ar2", PR3="PR3pt5ar2")
SEEDS = dict(PR1=60, PR2=61, PR3=62)
GRID = np.geomspace(0.01, 1.0, 10)      # Phase-P probe grid (pinned)
EXPECTED_IDX = 7                        # grid index of beta_min = 0.3594
NSYS = 16
ROUNDS_NOM = 1000
WINDOWS = [100, 250, 500]
RT_FLOOR = 117                           # = 175 * 1000/1500, op-7 scaled below
BAND = (0.32, 0.49)
DRIFT_ENV = 0.05
EEVPD_BAND = (1e-4, 2e-3)
PACC_BAND = (0.25, 0.65)
WG_MAX, WG_SOFT = 30.0, 10.0            # B5-style: max <= 30 AND <= 1 axis > 10
GENEIG_BAND = (1.0 / 3.0, 3.0)          # PT-4 zone labels (report-only here)
LOW_FLOOR = 0.1
POCKET_COL, POCKET_THR = 6, -22.35
FS_COS = 0.8
TARGET_NATS = 1.0                       # PT-0b convention (L1b)
PINNED_R6 = [0.3594, 0.4388, 0.5373, 0.6598, 0.8116, 1.0]
BM_NSYS, BM_BUDGET, BM_PROBE, BM_LEVEL = 16, 16500, 1500, 0.99  # certified beta_min_rule inputs
INFLATED_FAMILY = (19, 2, 3, 20)        # C-28 axis family (report-only)
READY_CAP_STEPS = 2000                  # rd-2 item 2: readiness cap (200 rounds)
CROSS_FLOOR = 12                        # report-only sanity floor (N_x), demoted per rd-1
PASS_VERDICT = ("DEDICATED-PROBE POINT-AND-GO PIPELINE PROPOSED AS PT-5 "
                "TUNING LAYER (UNCERTIFIED; carousel-only, F-P-guarded; "
                "1e-6*I pending; metric menu open; generalization untested "
                "— PT-5)")
BLIND_SPOT_NOTE = (
    " BLIND-SPOT NOTE (rd-2): readiness self-consistency NECESSARY; "
    "F-P-vs-known-answer is what makes it SUFFICIENT here; "
    "metastable-plateau + stably-wrong-too-cold modes documented, weaken "
    "at PT-5.")

STD_KEYS = ["cold_ind", "eevpd", "swap_attempts", "swap_accepts",
            "round_trips_pocket", "metric_frozen", "metric_within_covs",
            "metric_between_covs", "rounds_done", "n_revert",
            "metric_windows", "metric_estimator", "betas"]
PR_KEYS = ["pr_t_ready_steps", "pr_beta_min", "pr_knots",
           "pr_readiness_trace", "pr_readiness_fired", "pr_crossing_counts"]
PROBE_KEYS = ["betas_grid", "sd_u", "leak", "knots", "beta_min", "knot_se",
              "readiness_fired", "t_ready_steps", "readiness_trace"]

FR_ROUTING = ("F-R: readiness does NOT fire by the generous cap => "
              "broad-init occupancy not equilibrating as the arm-A finding "
              "predicted => report (contradicts the finding — "
              "investigate before PT-5)")
FT_ROUTING = ("F-T: fresh probe ladder != certified within tol => "
              "probe/recipe bug OR the fresh realization genuinely differs "
              "(recompute; the arm-A subsampling says it should reproduce) "
              "=> report")
FH_ROUTING = ("W-H (health) or W-G (regression guard) failed — W-G "
              "exceeding is a HANDOFF/config finding, explicitly NOT metric-"
              "menu adjudication; W-H failure is a production-transport "
              "health finding; mechanism split before any code-class "
              "presumption")
FP_ROUTING = ("F-P: production does not transport (RT < floor) or "
              "occupancy out of band beyond near-edge => the END-TO-END "
              "pipeline fails on the KNOWN answer => the scheme is broken, "
              "report-to-human (a hard negative — the pieces validated "
              "separately don't compose)")


# --------------------------------------------------------------------------
# pinned clause functions (pure; exercised by --selftest)
# --------------------------------------------------------------------------
def require_keys(d, keys, what):
    """Fail LOUD with the full key inventory on ANY missing key (anti-B2-bug
    pin: exact-name match only, no alias fallback, never silently default)."""
    missing = [k for k in keys if k not in d.files]
    if missing:
        raise KeyError(f"{what}: missing required key(s) {missing}; "
                       f"available: {sorted(d.files)}")


def wr_check(ready_fired, t_ready, cap=READY_CAP_STEPS):
    """W-R: readiness fired AND within the pinned cap."""
    return bool(ready_fired) and (int(t_ready) <= int(cap))


def wt_ladder_check(pr_knots, pr_beta_min, se_probe, ref_knots, ref_se,
                    bmin_ref):
    """W-T: rung count == 6 AND max|dev| <= 3x combined se (elementwise)
    AND beta_min is the EXACT certified grid point (no B2' rescue -- this
    scheme's beta_min comes from probe self-consistency, not a pooled
    counting rule). Returns a dict with the per-clause booleans + arrays."""
    pr_knots = np.asarray(pr_knots, np.float64)
    ref_knots = np.asarray(ref_knots, np.float64)
    count_ok = bool(pr_knots.shape[0] == 6)
    bmin_ok = bool(abs(float(pr_beta_min) - float(bmin_ref)) < 1e-9)
    if count_ok:
        se_probe = np.asarray(se_probe, np.float64)
        ref_se = np.asarray(ref_se, np.float64)
        if se_probe.shape != (6,) or ref_se.shape != (6,):
            raise ValueError(
                f"wt_ladder_check: se shape mismatch (probe {se_probe.shape}, "
                f"ref {ref_se.shape}) at pr_knots length 6 — anomaly, "
                f"refusing to silently broadcast")
        se_comb = np.sqrt(se_probe ** 2 + ref_se ** 2)
        dk = np.abs(pr_knots - ref_knots)
        knots_ok = bool(np.all(dk <= 3.0 * np.maximum(se_comb, 1e-15)))
    else:
        se_comb, dk, knots_ok = None, None, False
    return dict(count_ok=count_ok, bmin_ok=bmin_ok, knots_ok=knots_ok,
                dk=dk, se_comb=se_comb,
                pass_=bool(count_ok and bmin_ok and knots_ok))


def occ_a6(m, recent, prev):
    """A6-pinned occupancy reading (pt4_score.py/pt5a_score.py convention).
    LOW-side near-edge corroborated by a RISING trace only; HIGH-side needs
    flat-or-falling toward band. Returns (ok, description)."""
    in_b = BAND[0] <= m <= BAND[1]
    near_lo = abs(m - BAND[0]) < DRIFT_ENV
    near_hi = abs(m - BAND[1]) < DRIFT_ENV
    rising = recent > prev
    if in_b and not (near_lo or near_hi):
        return True, "in band"
    if near_lo:
        return rising, (f"near LOW edge +-{DRIFT_ENV} (trace {prev:.3f}->{recent:.3f} "
                        f"{'RISING: corroborated' if rising else 'not rising: NOT corroborated'})")
    if near_hi:
        return (not rising), (f"near HIGH edge +-{DRIFT_ENV} (trace {prev:.3f}->{recent:.3f} "
                              f"{'flat-or-falling toward band: corroborated' if not rising else 'RISING: NOT corroborated (A6: rising corroborates LOW side only)'})")
    return False, "OUT of band beyond near-edge"


def geneig_vs(adapted, L):
    X = np.linalg.solve(L, np.linalg.solve(L, adapted).T)
    return np.linalg.eigh((X + X.T) / 2.0)


def assemble_verdict(arms, wo_ok, wo_reading, ws_pairs_ok, missing):
    """W-a assembly with the checkpoint's routing text. `arms` = list of
    per-arm clause dicts (see keys below, _mk_arm gives the schema)."""
    if missing:
        return f"NOT assembled (W-a requires all three arms; missing: {missing})"
    quals = []
    for a in arms:
        quals.extend(a.get("zone_notes", []))
    qual_str = (" — QUALIFIERS (pt4 B3-1: must reach the verdict string): "
                + "; ".join(quals)) if quals else ""
    fr = [a["name"] for a in arms if a.get("f_r")]
    if fr:
        return (f"F-R — readiness did not fire within the cap (or a "
                f"fired-but-missing anomaly) on arm(s) {fr}; {FR_ROUTING}"
                + qual_str)
    ft = [a["name"] for a in arms if not a.get("wt_pass", False)]
    if ft:
        return (f"F-T — ladder reproduction failed on arm(s) {ft}; "
                f"{FT_ROUTING}" + qual_str)
    fh = [a["name"] for a in arms
          if not (a.get("wh_pass", False) and a.get("wg_pass", False))]
    if fh:
        return (f"F-H — production transport/health or W-G regression "
                f"on arm(s) {fh} ({FH_ROUTING})" + qual_str)
    rt_bad = [a["name"] for a in arms if not a.get("rt_ok", False)]
    wp_ok = (not rt_bad) and wo_ok and ws_pairs_ok
    if not wp_ok:
        why = []
        if rt_bad:
            why.append(f"per-arm RT < floor on {rt_bad}")
        if not wo_ok:
            why.append(f"pooled occupancy: {wo_reading}")
        if not ws_pairs_ok:
            why.append("W-s pair(s) > 2 sigma")
        return (f"F-P — W-P supporting-but-gating clause failed "
                f"({'; '.join(why)}); {FP_ROUTING}" + qual_str)
    return PASS_VERDICT + qual_str + BLIND_SPOT_NOTE


# --------------------------------------------------------------------------
# --selftest
# --------------------------------------------------------------------------
def _mk_arm(name, **kw):
    a = dict(name=name, f_r=False, wt_pass=True, wh_pass=True, wg_pass=True,
             rt_ok=True, zone_notes=[])
    a.update(kw)
    return a


def _selftest():
    # ---- pure-function unit checks -----------------------------------
    assert wr_check(True, 1200) is True
    assert wr_check(True, 2000) is True          # boundary: cap is inclusive
    assert wr_check(True, 2001) is False         # boundary: just over cap
    assert wr_check(False, 1200) is False
    print("[selftest] wr_check: fires-in-budget True, cap-boundary "
          "2000->True/2001->False, never-fired->False -- OK")

    ref_knots = np.array(PINNED_R6, np.float64)
    ref_se = np.array([0.0, 0.004, 0.005, 0.006, 0.007, 0.0])   # endpoints fixed -> se=0
    probe_se = np.array([0.0, 0.003, 0.004, 0.005, 0.006, 0.0])
    se_comb = np.sqrt(ref_se ** 2 + probe_se ** 2)
    # (a) exact reproduction -> pass
    r = wt_ladder_check(ref_knots, 0.3594, probe_se, ref_knots, ref_se, 0.3594)
    assert r["pass_"] and r["count_ok"] and r["bmin_ok"] and r["knots_ok"], r
    # (b) knot 2 nudged to exactly the 3se boundary -> still pass ( <= )
    knots_b = ref_knots.copy()
    knots_b[2] += 3.0 * se_comb[2]
    r = wt_ladder_check(knots_b, 0.3594, probe_se, ref_knots, ref_se, 0.3594)
    assert r["pass_"], r
    # (c) knot 2 nudged just past the boundary -> F-T (knots_ok False)
    knots_c = ref_knots.copy()
    knots_c[2] += 3.0 * se_comb[2] + 1e-6
    r = wt_ladder_check(knots_c, 0.3594, probe_se, ref_knots, ref_se, 0.3594)
    assert not r["pass_"] and not r["knots_ok"] and r["bmin_ok"], r
    # (d) beta_min a one-grid-point-off miss -> F-T (no rescue branch here)
    r = wt_ladder_check(ref_knots, GRID[EXPECTED_IDX - 1], probe_se,
                        ref_knots, ref_se, 0.3594)
    assert not r["pass_"] and not r["bmin_ok"] and r["knots_ok"], r
    # (e) wrong rung count -> F-T, knot comparison moot
    r = wt_ladder_check(ref_knots[:5], 0.3594, probe_se[:5], ref_knots,
                        ref_se, 0.3594)
    assert not r["pass_"] and not r["count_ok"] and r["dk"] is None, r
    print("[selftest] wt_ladder_check: exact pass, 3se boundary pass, "
          "past-boundary F-T, one-grid-point beta_min miss F-T (no rescue), "
          "wrong rung count F-T -- OK")

    # ---- (i) all-pass -------------------------------------------------
    arms = [_mk_arm("PR1"), _mk_arm("PR2"), _mk_arm("PR3")]
    v = assemble_verdict(arms, True, "in band", True, [])
    assert v.startswith(PASS_VERDICT), v
    assert "BLIND-SPOT NOTE (rd-2)" in v and "readiness self-consistency NECESSARY" in v, v
    print(f"[selftest i] all-pass verdict carries the PASS string + "
          f"blind-spot note:\n  {v}")

    # ---- (ii) F-R one arm ----------------------------------------------
    arms_fr = [_mk_arm("PR1"), _mk_arm("PR2", f_r=True), _mk_arm("PR3")]
    v = assemble_verdict(arms_fr, True, "in band", True, [])
    assert v.startswith("F-R"), v
    assert "PR2" in v and "readiness does NOT fire" in v, v
    print(f"[selftest ii] F-R (one arm) routes correctly:\n  {v}")

    # ---- (iii) F-T one arm ----------------------------------------------
    arms_ft = [_mk_arm("PR1"), _mk_arm("PR2"), _mk_arm("PR3", wt_pass=False)]
    v = assemble_verdict(arms_ft, True, "in band", True, [])
    assert v.startswith("F-T"), v
    assert "PR3" in v and "ladder reproduction failed" in v, v
    print(f"[selftest iii] F-T (one arm) routes correctly:\n  {v}")

    # ---- (iv) F-P pooled-occupancy-fail --------------------------------
    arms_ok = [_mk_arm("PR1"), _mk_arm("PR2"), _mk_arm("PR3")]
    v = assemble_verdict(arms_ok, False, "OUT of band beyond near-edge", True, [])
    assert v.startswith("F-P"), v
    assert "OUT of band beyond near-edge" in v, v
    print(f"[selftest iv] F-P (pooled occupancy out of band) routes "
          f"correctly:\n  {v}")

    # ---- bonus: remaining branches (missing / F-H / rt<floor / W-s-fail) --
    v = assemble_verdict([_mk_arm("PR1")], True, "", True, ["PR2", "PR3"])
    assert v.startswith("NOT assembled (W-a requires all three arms"), v
    v = assemble_verdict([_mk_arm("PR1", wg_pass=False), _mk_arm("PR2"),
                          _mk_arm("PR3")], True, "in band", True, [])
    assert v.startswith("F-H"), v
    v = assemble_verdict([_mk_arm("PR1", rt_ok=False), _mk_arm("PR2"),
                          _mk_arm("PR3")], True, "in band", True, [])
    assert v.startswith("F-P") and "RT < floor" in v, v
    v = assemble_verdict(arms_ok, True, "in band", False, [])
    assert v.startswith("F-P") and "W-s pair(s)" in v, v
    print("[selftest] bonus branches (missing / F-H / rt<floor / W-s-fail) OK")
    print("[selftest] ALL PASS")


if "--selftest" in sys.argv:
    _selftest()
    sys.exit(0)


# --------------------------------------------------------------------------
# reference artifacts (run in all live-scoring modes, incl. empty-state:
# validates the reference machinery against the archives before any arm is
# read). Re-derives the CERTIFIED carousel ladder from the ARCHIVED pt5a
# probe/production inputs -- same L1b/L2 pattern as pt4_recipe_validate.py
# and pt5a_score.py; this is the fixed target the fresh dedicated probes are
# judged against, NOT re-measured live here (the live re-measurement is what
# each arm's OWN probe does, scored by W-T above).
# --------------------------------------------------------------------------
print("=== GATE PT-5a-r2 scorer: reference artifacts ===")
jd = json.load(open(f"{OUT}/ladder_design_power.json"))
betas_A = np.asarray(jd["betas_measured"], np.float64)
sd_A_json = np.asarray(jd["sd_u"], np.float64)
assert np.allclose(betas_A, GRID, rtol=1e-12, atol=0), "archived grid mismatch"
dA = np.load(f"{OUT}/arrays_A_power.npz")
assert np.allclose(np.asarray(dA["betas"], np.float64), GRID, rtol=1e-12, atol=0)
sd_A_npz, se_A = measure_sd_u(np.asarray(dA["u"]), 1500)
dev_sd = float(np.abs(sd_A_npz - sd_A_json).max())
assert dev_sd < 1e-9, f"arrays_A_power sd(u) != archived json sd_u ({dev_sd:.3e})"
sA = json.load(open(f"{OUT}/summary_A_power.json"))
leak_A = np.array([min(r["leak_Minit"], r["leak_Pinit"]) for r in sA["per_beta"]])
assert np.allclose([r["beta"] for r in sA["per_beta"]], GRID, rtol=1e-12, atol=0)
BMIN_REF = beta_min_rule(GRID, leak_A, BM_NSYS, BM_BUDGET, BM_PROBE, level=BM_LEVEL)
assert round(BMIN_REF, 4) == 0.3594 and abs(BMIN_REF - betas_A[EXPECTED_IDX]) < 1e-12
ref_des = design_ladder(betas_A, sd_A_json, BMIN_REF, TARGET_NATS)
REF_KNOTS = np.asarray(ref_des["knots"], np.float64)
assert ref_des["n_rungs"] == 6 and [round(float(k), 4) for k in REF_KNOTS] == PINNED_R6, \
    f"reference recompute does not reproduce the certified R6: {REF_KNOTS}"
REF_KNOT_SE = knot_se(betas_A, sd_A_json, se_A, BMIN_REF, TARGET_NATS)
print(f"reference ladder (recomputed, full precision): "
      f"{[f'{k:.6f}' for k in REF_KNOTS]}")
print(f"  beta_min_rule -> {BMIN_REF:.10g}; knot se (delta method) = "
      f"{[f'{s:.2e}' for s in REF_KNOT_SE]}; max|sd_npz - sd_json| = {dev_sd:.1e}")

# MAMS pools (POSITIONS ONLY -- standing directive) for Sigma_ref / occupancy
draws = np.load(MAMS)["samples_z"].reshape(-1, 33).astype(np.float64)
pk = draws[:, POCKET_COL] > POCKET_THR
cov_P, cov_M = np.cov(draws[pk].T), np.cov(draws[~pk].T)
dmu = draws[pk].mean(0) - draws[~pk].mean(0)
cov_pool = np.cov(draws.T)              # fixed pooled diagnostic ref (W/B echo, unused here)
L_POOL = np.linalg.cholesky(cov_pool)
fs_ref = np.load(f"{OUT}/pt3_fs_reference.npz")["eigvec_z"]


def sigma_ref(w):
    return w * cov_P + (1 - w) * cov_M + w * (1 - w) * np.outer(dmu, dmu)


# --------------------------------------------------------------------------
# arm loop (single pass: no cross-arm dependency for W-T here, unlike pt5a's
# B2' pooled-rescue -- beta_min is a straight per-arm grid-point check)
# --------------------------------------------------------------------------
res = {}
for name in ARMS:
    tag = TAGS[name]
    prod_path = f"{OUT}/arrays_PR_{tag}.npz"
    probe_path = f"{OUT}/arrays_PR_{tag}_probe.npz"
    have_prod, have_probe = os.path.exists(prod_path), os.path.exists(probe_path)
    if not have_prod and not have_probe:
        print(f"\n[{name}] MISSING — skipped")
        continue
    if have_prod and not have_probe:
        raise FileNotFoundError(
            f"[{name}] production npz present but probe companion {probe_path} "
            f"missing — refusing to score (the probe companion is the "
            f"ladder-reproduction reference basis for W-T; anti-B2-bug pin)")

    dp = np.load(probe_path)
    require_keys(dp, PROBE_KEYS, f"[{name}] probe companion {probe_path}")
    betas_grid = np.asarray(dp["betas_grid"], np.float64)
    assert betas_grid.shape == (10,) and np.allclose(betas_grid, GRID, rtol=1e-12, atol=0), \
        f"[{name}] probe betas_grid != geomspace(0.01,1,10): {betas_grid}"
    sd_u_probe = np.asarray(dp["sd_u"], np.float64)
    leak_probe = np.asarray(dp["leak"], np.float64)
    knots_probe = np.asarray(dp["knots"], np.float64)
    beta_min_probe = float(np.asarray(dp["beta_min"]).item())
    knot_se_probe = np.asarray(dp["knot_se"], np.float64)
    ready_fired_probe = bool(np.asarray(dp["readiness_fired"]).item())
    t_ready_probe = int(np.asarray(dp["t_ready_steps"]).item())
    trace_probe = np.asarray(dp["readiness_trace"], np.float64)
    print(f"\n[{name}] probe companion: readiness_fired={ready_fired_probe} "
          f"t_ready_steps={t_ready_probe} (cap {READY_CAP_STEPS}); "
          f"beta_min={beta_min_probe:.6g}; knots={np.round(knots_probe, 4).tolist()}")
    if trace_probe.size:
        tr = trace_probe.reshape(-1, 3) if trace_probe.ndim == 1 else trace_probe
        print(f"  [readiness trace] {tr.shape[0]} rows; first {tr[0].tolist()}; "
              f"last {tr[-1].tolist()}")
        fire_rows = tr[np.abs(tr[:, 0] - t_ready_probe) < 0.5]
        if fire_rows.size:
            print(f"  [readiness trace] FIRE row (t={t_ready_probe}): "
                  f"{fire_rows[0].tolist()} (t, beta_min, max_knot_dev)")
    else:
        print("  [readiness trace] EMPTY array archived")

    if not have_prod:
        # A4-style graceful F-R: readiness never fired (or, unexpectedly,
        # fired but production is still missing) -- scored on probe data
        # alone, nothing downstream evaluable.
        anomaly = ready_fired_probe and (t_ready_probe <= READY_CAP_STEPS)
        if anomaly:
            print(f"*** [{name}] WARNING: probe indicates readiness FIRED "
                  f"within the cap but no production npz exists — "
                  f"UNEXPECTED (production should have run); treating as "
                  f"F-R pending investigation, not the graceful A4 case ***")
        else:
            print(f"  [{name}] readiness did not fire within the cap "
                  f"(fired={ready_fired_probe}, t_ready={t_ready_probe}) "
                  f"— production npz correctly absent (graceful F-R)")
        res[name] = dict(name=name, f_r=True, anomaly=bool(anomaly),
                          probe_ready_fired=ready_fired_probe,
                          probe_t_ready=t_ready_probe,
                          wt_pass=False, wh_pass=False, wg_pass=False,
                          rt_ok=False, zone_notes=[])
        continue

    dP = np.load(prod_path)
    require_keys(dP, STD_KEYS + PR_KEYS, f"[{name}] production {prod_path}")
    pr_ready_fired = bool(np.asarray(dP["pr_readiness_fired"]).item())
    pr_t_ready = int(np.asarray(dP["pr_t_ready_steps"]).item())
    pr_beta_min = float(np.asarray(dP["pr_beta_min"]).item())
    pr_knots = np.asarray(dP["pr_knots"], np.float64)
    pr_trace = np.asarray(dP["pr_readiness_trace"], np.float64)
    pr_cross = np.asarray(dP["pr_crossing_counts"], np.float64)
    rounds = int(dP["rounds_done"])
    betas1 = np.asarray(dP["betas"], np.float64)

    # ---- internal consistency: production copy vs probe companion (report)
    if pr_ready_fired != ready_fired_probe or pr_t_ready != t_ready_probe:
        print(f"*** [{name}] WARNING: production pr_readiness_fired/"
              f"pr_t_ready_steps ({pr_ready_fired}/{pr_t_ready}) != probe "
              f"companion readiness_fired/t_ready_steps "
              f"({ready_fired_probe}/{t_ready_probe}) — investigate "
              f"before trusting the handoff ***")
    if not np.array_equal(betas1, pr_knots):
        raise ValueError(
            f"[{name}] production 'betas' != 'pr_knots' — the "
            f"handoff contract requires production to run on the probe's "
            f"OWN knots verbatim: {betas1} vs {pr_knots}")
    if not np.allclose(pr_knots, knots_probe, rtol=1e-9, atol=1e-12):
        print(f"*** [{name}] WARNING: production pr_knots != probe "
              f"companion's own 'knots' — handoff carry-over anomaly "
              f"(max|dev|={np.abs(pr_knots - knots_probe).max():.2e}) ***")

    # ---- W-R (production's own copy is authoritative) --------------------
    f_r = not wr_check(pr_ready_fired, pr_t_ready, READY_CAP_STEPS)
    print(f"  [W-R {name}] pr_readiness_fired={pr_ready_fired} "
          f"pr_t_ready_steps={pr_t_ready} vs cap {READY_CAP_STEPS} -> "
          f"{'FAIL (F-R)' if f_r else 'PASS'}")
    print(f"  [crossing-counts {name}] pr_crossing_counts="
          f"{np.round(pr_cross, 1).tolist()} (report-only sanity floor "
          f"N_x>={CROSS_FLOOR}, demoted per rd-1 B1/B2, NOT gating)")
    below = [i for i, c in enumerate(pr_cross) if c < CROSS_FLOOR]
    if below:
        print(f"    *** rung index(es) {below} below the {CROSS_FLOOR}-count "
              f"sanity floor (report-only) ***")

    # ---- W-T --------------------------------------------------------------
    wt = wt_ladder_check(pr_knots, pr_beta_min, knot_se_probe, REF_KNOTS,
                        REF_KNOT_SE, BMIN_REF)
    print(f"  [W-T {name}] rung count {pr_knots.shape[0]} == 6: "
          f"{wt['count_ok']}; beta_min {pr_beta_min:.6g} == "
          f"{BMIN_REF:.6g} (exact grid point, no rescue): {wt['bmin_ok']}")
    if wt["count_ok"]:
        for i in range(6):
            print(f"    knot {i}: ref={REF_KNOTS[i]:.6f} probe="
                  f"{pr_knots[i]:.6f} |d|={wt['dk'][i]:.2e} "
                  f"3se_comb={3 * wt['se_comb'][i]:.2e} "
                  f"{'ok' if wt['dk'][i] <= 3 * max(wt['se_comb'][i], 1e-15) else '*** EXCEEDS ***'}")
    print(f"  [W-T {name}] -> {'PASS' if wt['pass_'] else 'FAIL (F-T)'}")
    # internal consistency: probe's OWN design_ladder recompute vs its
    # archived knots (catches recipe-input assembly / carry-over bugs)
    des_check = design_ladder(betas_grid, sd_u_probe, beta_min_probe, TARGET_NATS)
    dc_knots = np.asarray(des_check["knots"], np.float64)
    if (des_check["n_rungs"] != len(knots_probe)
            or np.abs(dc_knots - knots_probe).max() > 1e-6):
        print(f"*** [{name}] WARNING: probe's own design_ladder recompute "
              f"does not reproduce its archived 'knots' (recipe-input "
              f"assembly suspect — F-T investigation input) ***")

    if rounds > ROUNDS_NOM:
        raise AssertionError(f"[{name}] rounds_done {rounds} > nominal {ROUNDS_NOM}")
    metric_windows = list(np.asarray(dP["metric_windows"]).tolist())
    assert metric_windows == WINDOWS, f"[{name}] non-pinned metric_windows! {metric_windows}"
    assert str(dP["metric_estimator"]) == "within", \
        f"[{name}] estimator is not 'within'!"
    assert dP["cold_ind"].shape[1] == NSYS, \
        f"[{name}] NSYS {dP['cold_ind'].shape[1]} != pinned 16"
    assert dP["cold_ind"].shape[0] >= rounds, f"[{name}] truncated buffer"
    if rounds < ROUNDS_NOM:
        print(f"*** {name} INCOMPLETE {rounds}/{ROUNDS_NOM} — op-7 scaling ***")

    lo = max(500, rounds - 500)
    clipped_out = rounds <= lo
    arm = dict(name=name, f_r=bool(f_r), anomaly=False, rounds=rounds,
              pr_t_ready_steps=pr_t_ready, pr_beta_min=pr_beta_min,
              pr_knots=[float(v) for v in pr_knots],
              pr_crossing_counts=[float(v) for v in pr_cross],
              readiness_trace=trace_probe.tolist(),
              wt_pass=bool(wt["pass_"]), wt_count_ok=wt["count_ok"],
              wt_bmin_ok=wt["bmin_ok"], wt_knots_ok=wt["knots_ok"],
              knot_dev=(None if wt["dk"] is None else [float(v) for v in wt["dk"]]),
              zone_notes=[])
    if clipped_out:
        print(f"*** [{name}] no post-freeze rounds (rounds={rounds} <= "
              f"lo={lo}) — W-H/W-P/W-G unevaluable, marked FAIL ***")
        arm.update(wh_pass=False, wg_pass=False, rt_ok=False)
        res[name] = arm
        continue

    cold = dP["cold_ind"][:rounds]
    ev = dP["eevpd"][:rounds]

    # ---- W-H ---------------------------------------------------------
    ev_med = np.median(ev[lo:], axis=0)
    ev_ok = bool(np.all((ev_med >= EEVPD_BAND[0]) & (ev_med <= EEVPD_BAND[1])))
    att = dP["swap_attempts"].astype(float)
    acc = dP["swap_accepts"].astype(float)
    pacc = acc.sum(1) / np.maximum(att.sum(1), 1)
    pacc_ok = bool(np.all((pacc >= PACC_BAND[0]) & (pacc <= PACC_BAND[1])))
    n_rev = int(dP["n_revert"])
    wh_pass = bool(ev_ok and pacc_ok and n_rev == 0)
    print(f"  [W-H {name}] eevpd_med(post-lo)={['%.1e' % v for v in ev_med]} "
          f"in [1e-4,2e-3]: {ev_ok}; pair acc {np.round(pacc, 3).tolist()} "
          f"in [0.25,0.65]: {pacc_ok}; NaN reverts={n_rev} -> "
          f"{'PASS' if wh_pass else 'FAIL'}")

    # ---- occupancy / W-P per-arm ---------------------------------------
    win = cold[lo:]
    sysm = win.mean(axis=0)
    m = float(sysm.mean())
    sd_m = float(sysm.std(ddof=1))
    se_m = sd_m / math.sqrt(NSYS)
    occ_recent = float(win.mean())
    occ_prev = float(cold[max(0, lo - 500):lo].mean())
    rtp = int(dP["round_trips_pocket"].sum())
    floor_eff = RT_FLOOR * min(1.0, rounds / float(ROUNDS_NOM))
    rt_ok = bool(rtp >= floor_eff)
    print(f"  [W-P {name}] RT_pocket={rtp} vs floor {floor_eff:.0f} "
          f"(117 op-7-scaled): {'PASS' if rt_ok else 'FAIL'}; per-arm "
          f"occ m={m:.4f}+-{se_m:.4f} (report; pooled is primary)")
    if not rt_ok:
        rising = occ_recent > occ_prev
        print(f"  *** [{name}] RT below floor — "
              + ("rising coldocc trace: budget-limited zone (report)"
                 if rising else "trace not rising: report") + " ***")

    # ---- W-G (regression guard) -----------------------------------------
    ref = sigma_ref(m)
    Lr = np.linalg.cholesky(ref)
    A = dP["metric_frozen"][-1]                  # cold rung
    g, V = geneig_vs(A, Lr)
    order = np.argsort(g)
    g, V = g[order], V[:, order]
    n_gt10 = int((g > WG_SOFT).sum())
    wg_pass = bool(g[-1] <= WG_MAX and n_gt10 <= 1)
    n_soft_hi = int(((g > GENEIG_BAND[1]) & (g <= WG_SOFT)).sum())
    n_soft_lo = int(((g >= LOW_FLOOR) & (g < GENEIG_BAND[0])).sum())
    n_hard_lo = int((g < LOW_FLOOR).sum())
    print(f"  [W-G {name}] cold-rung gen-eig vs Sigma_ref(w={m:.3f}): "
          f"[{g[0]:.3f}, {g[-1]:.3f}]; axes>10: {n_gt10} (<=1 required), "
          f"max<={WG_MAX}: {g[-1] <= WG_MAX} -> {'PASS' if wg_pass else 'F-H FIRES'}")
    print(f"    PT-4 zone labels (REPORT-ONLY — metric menu open, C-28 "
          f"class rides along): (3,10] on {n_soft_hi} axes; [0.1,1/3) on "
          f"{n_soft_lo}; <0.1 on {n_hard_lo}")
    if not wg_pass:
        print(f"  *** [{name}] F-H: gen-eig EXCEEDS the C-28 measured class "
              f"(max {g[-1]:.1f} vs 30 / {n_gt10} axes > 10 vs 1) — a "
              f"HANDOFF finding, explicitly NOT metric-menu adjudication ***")
    fam_notes = []
    for i in np.where(g > GENEIG_BAND[1])[0]:
        x = np.linalg.solve(Lr.T, V[:, i])
        ax = int(np.argmax(np.abs(x / np.linalg.norm(x))))
        fam_notes.append(f"g={g[i]:.1f}->z[{ax}]"
                         + ("(fam)" if ax in INFLATED_FAMILY else ""))
    if fam_notes:
        print(f"    inflated-axis attribution vs {{19,2,3,20}} family "
              f"(report): {', '.join(fam_notes)}")
    fs_aligned = False
    for i in np.where(g < GENEIG_BAND[0])[0]:
        x = np.linalg.solve(Lr.T, V[:, i])
        x /= np.linalg.norm(x)
        cosr = abs(float(x @ fs_ref))
        print(f"    [F-S-style check] low-side axis g={g[i]:.3f}: "
              f"|cos(stored pt3 dir)| = {cosr:.3f}"
              + (" -> aligned (report)" if cosr >= FS_COS else " (distinct)"))
        if cosr >= FS_COS:
            fs_aligned = True
    if n_soft_hi > 0:
        arm["zone_notes"] = arm["zone_notes"] + [
            f"{name}: {n_soft_hi} axis(es) in (3,10] (PT-4 zone label, "
            f"report-only this gate)"]
    if n_soft_lo > 0 or n_hard_lo > 0:
        arm["zone_notes"] = arm["zone_notes"] + [
            f"{name}: {n_soft_lo + n_hard_lo} low-side axis(es) "
            f"(< 1/3{'; <0.1 present' if n_hard_lo else ''}, report-only)"]
    if fs_aligned:
        arm["zone_notes"] = arm["zone_notes"] + [
            f"{name}: F-S-style aligned low-side direction (recorded)"]

    arm.update(wh_pass=wh_pass, ev_med=ev_med.tolist(), pair_acc=pacc.tolist(),
              n_revert=n_rev, wg_pass=wg_pass,
              geneig_full=[float(v) for v in g], n_gt10=n_gt10,
              n_soft_hi=n_soft_hi, n_soft_lo=n_soft_lo,
              fs_aligned=bool(fs_aligned), rt_pocket=rtp,
              floor_eff=floor_eff, rt_ok=rt_ok, m=m, sd=sd_m, se=se_m,
              per_system_means=[float(v) for v in sysm],
              occ_recent=occ_recent, occ_prev=occ_prev)
    res[name] = arm

# --------------------------------------------------------------------------
# pooled W-P: occupancy (48 systems), W-s pairs
# --------------------------------------------------------------------------
pooled = {}
wo_ok, wo_reading = False, "unavailable"
phase1_arms = [n for n in ARMS if n in res and not res[n]["f_r"]
              and "m" in res[n]]
if len(phase1_arms) == 3:
    pool_means = np.concatenate([np.asarray(res[n]["per_system_means"])
                                 for n in ARMS])
    m_pool = float(pool_means.mean())
    se_pool = float(pool_means.std(ddof=1) / math.sqrt(pool_means.size))
    recent = float(np.mean([res[n]["occ_recent"] for n in ARMS]))
    prev = float(np.mean([res[n]["occ_prev"] for n in ARMS]))
    wo_ok, wo_reading = occ_a6(m_pool, recent, prev)
    arm_ms = [res[n]["m"] for n in ARMS]
    var_between = float(np.var(arm_ms, ddof=1))
    var_within = float(np.mean([res[n]["sd"] ** 2 / NSYS for n in ARMS]))
    pooled = dict(n=int(pool_means.size), m_pool=m_pool, se_pool=se_pool,
                 arm_means=arm_ms, var_between_arms=var_between,
                 var_within_over_nsys=var_within, trace_prev=prev,
                 trace_recent=recent, wo=bool(wo_ok), reading=wo_reading)
    print(f"\nW-P pooled occupancy: m_pool={m_pool:.4f}+-{se_pool:.4f} "
          f"(n={pool_means.size}) in ({BAND[0]}, {BAND[1]}): "
          f"{BAND[0] <= m_pool <= BAND[1]} — {wo_reading}")
    print(f"  between-arm var {var_between:.3e} vs within/nsys "
          f"{var_within:.3e} (report)")
    clipped = [n for n in ARMS if res[n].get("rounds", ROUNDS_NOM) < ROUNDS_NOM]
    if clipped:
        print(f"  *** PT-4 A3-2 clipped-arm plot rule (WARNING): arm(s) "
              f"{clipped} < 1000 rounds — consult the pre-committed "
              f"coldocc overlay plot before accepting any near-edge read ***")
else:
    print(f"\nW-P pooled occupancy: requires all three phase-1 arms — "
          f"unavailable (phase-1 arms present: {phase1_arms})")


def pair(a, b):
    if a in res and b in res and "m" in res[a] and "m" in res[b]:
        da, db = res[a], res[b]
        s = float(np.hypot(da["se"], db["se"]))
        dm = abs(da["m"] - db["m"])
        ok = bool(dm <= 2 * s)
        print(f"W-s [{a}/{b}]: |{da['m']:.4f}-{db['m']:.4f}|={dm:.4f} "
              f"<= {2 * s:.4f}: {ok}")
        if dm > 3 * s:
            print(f"  *** F-eq: pair > 3 sigma ({3 * s:.4f}) — seed "
                  f"pooling invalid at this budget (report, no auto-lever) ***")
        return ok
    return None


print()
ws_pairs = dict(P1P2=pair("PR1", "PR2"), P1P3=pair("PR1", "PR3"),
               P2P3=pair("PR2", "PR3"))
ws_pairs_ok = all(bool(v) for v in ws_pairs.values()) \
    if all(v is not None for v in ws_pairs.values()) else False

# --------------------------------------------------------------------------
# assembly + dump
# --------------------------------------------------------------------------
missing = [n for n in ARMS if n not in res]
verdict = assemble_verdict([res[n] for n in ARMS if n in res],
                          wo_ok, wo_reading, ws_pairs_ok, missing)
print(f"\nW-a verdict: {verdict}")
print("\nReminder: gen-eig is REPORTED with PT-4 zone labels, NOT a gating "
      "clause here beyond the max<=30/<=1-axis>10 W-G threshold (layering "
      "pin) — the PT-4 human menu (bounded metric vs accept-(3,28] vs "
      "both) remains OPEN.")

out = dict(arms=res, pooled_occ=pooled, ws_pairs=ws_pairs,
          reference=dict(knots=[float(v) for v in REF_KNOTS],
                         knot_se=[float(v) for v in REF_KNOT_SE],
                         beta_min=float(BMIN_REF)),
          verdict=verdict)
json.dump(out, open(f"{OUT}/pt5a_r2_score.json", "w"), indent=1, default=str)
print(f"\n-> {OUT}/pt5a_r2_score.json")
