#!/usr/bin/env python
"""GATE PT-5a scorer: pinned formulas from the rd-3-certified design checkpoint
(docs/logs/carousel-mclmc-sampling.md, "carousel GATE PT-5a" checkpoint).

Arms ST1/ST2/ST3 (seeds 55/56/57), two-phase self-tuning runner mode:
phase 0 = preset geomspace(0.01, 1, 10) grid, SWAPS OFF (B1), stationarity
trigger in [200, 400], re-space at R* = max(trigger, 300); phase 1 = fresh run
on the recipe ladder, ROUNDS 1000 nominal, freeze-500 windows 100/250/500,
scoring on rounds 500-1000 (lo = max(500, rounds-500) lineage).

Clauses (transcribed, never re-derived):
  W-S  trigger in [200,400]; per-rung ratio = st_sd_u/probe_sd within
       +-3*combined rel se on >= 8/10 rungs; WEIGHTED STOUFFER (rd-2 B4')
       z_i = (ratio_i-1)/se_i, w_i = 1/se_i, Z = sum(w z)/sqrt(sum w^2),
       |Z| < 2; Z <= -2 => F-EARLY (under-dispersion, transit signature);
       Z >= +2 => over-dispersion, report-to-human. in-run rel se =
       1/sqrt(2 N_eff_in), N_eff_in = 16*100/tau_round (st_tau_rounds);
       probe rel se = 1/sqrt(2*(ess_proxy_Minit + ess_proxy_Pinit)) from
       summary_A_power_probe54.json (B4 pin: in-script computation).
  W-T  rung count == 6 AND every knot within 3x COMBINED knot se of the
       certified reference (reference recomputed in-script from the ARCHIVED
       ladder_design_power.json inputs via ladder_recipe -- L1b pattern, full
       precision; reference knot se via ladder_recipe.knot_se with se(sd)
       from measure_sd_u on arrays_A_power.npz, the L2 pattern) AND beta_min
       leg: exact grid 0.3594, OR the rd-2 B2' pre-registered pooled rescue
       (one-grid-point miss AND binding-rung count < 5 AND B2' rule
       (min_count 2) on the POOLED 3-arm counts returns 0.3594) =>
       pass-WITH-COUNTING-CAVEAT; anything else = F-T with the rd-3 pin-2
       investigation order (counting anomaly vs printed flip counts BEFORE
       code-class presumption).
  W-H  phase-1 EEVPD post-lo medians in [1e-4, 2e-3] every rung; pair acc in
       [0.25, 0.65]; NaN reverts == 0.
  W-G  (B5 regression guard) phase-1 post-freeze cold-rung gen-eig vs
       Sigma_ref(w-hat), pt4 construction: max <= 30 AND <= 1 axis > 10;
       exceeding => F-H (HANDOFF finding, explicitly NOT metric-menu
       adjudication); within-class inflation reported with PT-4 zone labels.
  W-P  (supporting, B7 pin) per-arm RT_pocket >= 117 (op-7 scaled if
       clipped); pooled occupancy over 48 per-system last-500 means in
       (0.32, 0.49) with the A6 near-edge rules (+ PT-4 A3-2 clipped-arm plot
       warning); W-s all three pairs <= 2 sigma. Any failure => W-a NOT
       assembled (tuning-layer-only adjudication note).
  W-a  W-S ^ W-T ^ W-H ^ W-G on ALL arms ^ W-P => the pinned PROPOSED
       verdict; counting-caveat and zone qualifiers MUST reach the verdict
       string (pt4 audit B3-1 lesson).

Exposure (rd-3 pin 1, the SHARED denominator): (R*-100) rounds x K=10 steps
x NSYS=16 chains kernel steps; per-1500-step units = that / 1500 -- RECOMPUTED
here, the stored normalization is never trusted.

Also reported (unscored): hot-rung flip presence (F-flip if the hottest rung
shows 0 flips post-100), per-rung IAT (K design data), st_interp_geneig echo
(A1 leg a), phase-1 u-level re-equilibration vs phase-0 stationary levels
(A1 leg b), in-run leakage vs the probe's 0.2565/0.1756/0.008 with Poisson
se's, W-only/pooled W/B echo (pt4 convention, report-only), split-Rhat.

F-NEVER handling (A4): an arm with only the phase-0 npz is scored for W-S
from phase-0 data (sd/tau recomputed from the final 100 recorded rounds),
marked F-NEVER; W-a not assemblable. Missing arms => pt4-style skip.

Login-node numpy only. --selftest: synthetic 3-arm data exercising (i) the
Stouffer statistic against known-Z cases, (ii) the pooled beta_min rescue
branch, (iii) the F-T branch (pooled denies), asserting verdict strings.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_recipe import (beta_min_rule, design_ladder, knot_se,
                           measure_sd_u)

OUT = os.environ.get(
    "GATE_PT5A_OUT",
    "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/"
    "experiments/flow_precond/carousel_gate_pt0_out")  # env override: smoke fixtures only
MAMS = ("/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/"
        "experiments/sim_carousel/messy_tests/dpie/mams/arrays.npz")
ARMS = ("ST1", "ST2", "ST3")
TAGS = dict(ST1="ST_ST1pt5a", ST2="ST_ST2pt5a", ST3="ST_ST3pt5a")
SEEDS = dict(ST1=55, ST2=56, ST3=57)
GRID = np.geomspace(0.01, 1.0, 10)      # phase-0 preset grid (pinned)
EXPECTED_IDX = 7                        # grid index of beta_min = 0.3594
NSYS, K_STEPS = 16, 10
ROUNDS_NOM = 1000
TRIG_LO, TRIG_HI = 200, 400
WINDOWS = [100, 250, 500]
T_WIN = [100, 150, 250]                 # window LENGTHS (pt4 pinned)
RT_FLOOR = 117                          # = 175 * 1000/1500, op-7 scaled below
BAND = (0.32, 0.49)
DRIFT_ENV = 0.05
EEVPD_BAND = (1e-4, 2e-3)
PACC_BAND = (0.25, 0.65)
WG_MAX, WG_SOFT = 30.0, 10.0            # B5: max <= 30 AND <= 1 axis > 10
GENEIG_BAND = (1.0 / 3.0, 3.0)          # PT-4 zone labels (report-only here)
LOW_FLOOR = 0.1
POCKET_COL, POCKET_THR = 6, -22.35
FS_COS = 0.8
TARGET_NATS = 1.0                       # PT-0b convention (L1b)
PINNED_R6 = [0.3594, 0.4388, 0.5373, 0.6598, 0.8116, 1.0]
# B2' rule constants (certified beta_min_rule inputs; scale = 16*16500/1500)
BM_NSYS, BM_BUDGET, BM_PROBE, BM_LEVEL = 16, 16500, 1500, 0.99
INFLATED_FAMILY = (19, 2, 3, 20)        # C-28 axis family (report-only)
PASS_VERDICT = ("SELF-TUNING SCHEME PROPOSED AS PT-5 TUNING LAYER "
                "(UNCERTIFIED; carousel-only; beta_min-rule circularity "
                "carries (B1); metric menu open; 1e-6*I ratification pending)")


# --------------------------------------------------------------------------
# pinned clause functions (pure; exercised by --selftest)
# --------------------------------------------------------------------------
def stouffer(ratios, rel_se):
    """rd-2 B4' weighted Stouffer over all 10 rungs + 3se coverage count.
    Returns (n_covered, Z, ws_ratio_pass, f_early, over_dispersed)."""
    ratios = np.asarray(ratios, float)
    rel_se = np.asarray(rel_se, float)
    z = (ratios - 1.0) / rel_se
    w = 1.0 / rel_se
    Z = float((w * z).sum() / math.sqrt(float((w * w).sum())))
    cov = int((np.abs(ratios - 1.0) <= 3.0 * rel_se).sum())
    return cov, Z, bool(cov >= 8 and abs(Z) < 2.0), bool(Z <= -2.0), bool(Z >= 2.0)


def b2prime_beta_min(counts, expo_units, min_count=2, grid=GRID):
    """rd-2 B2' rule: raw neighbor-conservative ln(100) discovery rule
    (ladder_recipe.beta_min_rule arithmetic: p_hat = min(own, next-colder
    rate), coldest uses own) + a k >= min_count guard on the flip COUNT at
    the binding (min-rate) rung. counts aligned with ascending grid;
    expo_units = exposure in per-1500-kernel-step units. Returns the grid
    INDEX of beta_min, or None if no rung is admitted."""
    counts = np.asarray(counts, float)
    p = counts / float(expo_units)
    need = math.log(1.0 / (1.0 - BM_LEVEL))
    scale = BM_NSYS * (BM_BUDGET / BM_PROBE)
    best = None
    n = len(grid)
    for i in range(n):
        j = i + 1 if (i < n - 1 and p[i + 1] <= p[i]) else i   # binding rung
        if scale * p[j] >= need and counts[j] >= min_count:
            best = i                    # ascending scan keeps the coldest
    return best


def wt_beta_min_leg(name, bmin_idx, counts_by_arm, expo_by_arm):
    """Scorer-side beta_min adjudication (rd-2 B2' + rd-3 pin 2).
    Returns (status in {'exact','caveat','F-T'}, note)."""
    if bmin_idx == EXPECTED_IDX:
        return "exact", "beta_min == 0.3594 grid point (exact)"
    if abs(bmin_idx - EXPECTED_IDX) != 1:
        return "F-T", (f"beta_min at grid[{bmin_idx}] is a "
                       f"{abs(bmin_idx - EXPECTED_IDX)}-grid-point miss "
                       f"(> 1) — outside the pre-registered pooled routing")
    binding = max(bmin_idx, EXPECTED_IDX) + 1
    k_bind = int(counts_by_arm[name][binding])
    if k_bind >= 5:
        return "F-T", (f"one-grid-point miss but binding-rung "
                       f"(grid[{binding}]={GRID[binding]:.4f}) count "
                       f"{k_bind} >= 5 — pooled rescue not applicable")
    if len(counts_by_arm) < len(ARMS):
        return "F-T", ("one-grid-point miss, binding count "
                       f"{k_bind} < 5, but pooled 3-arm counts unavailable "
                       f"(arms present: {sorted(counts_by_arm)}) — "
                       "rescue DENIED (fail-closed)")
    pooled_counts = np.sum([counts_by_arm[a] for a in counts_by_arm], axis=0)
    pooled_expo = float(sum(expo_by_arm.values()))
    pooled_idx = b2prime_beta_min(pooled_counts, pooled_expo, min_count=2)
    if pooled_idx == EXPECTED_IDX:
        return "caveat", (f"one-grid-point miss (grid[{bmin_idx}]="
                          f"{GRID[bmin_idx]:.4f}), binding-rung count "
                          f"{k_bind} < 5, POOLED 3-arm B2' (counts summed, "
                          f"exposures summed = {pooled_expo:.1f} units) "
                          f"returns 0.3594 — pass-WITH-COUNTING-CAVEAT "
                          f"(pre-registered)")
    return "F-T", (f"one-grid-point miss, binding count {k_bind} < 5, but "
                   f"pooled 3-arm B2' returns "
                   f"grid[{pooled_idx}] != 0.3594 — rescue DENIED "
                   f"(the fail-closed 9.4% pooled branch)")


FT_ROUTING = ("rd-3 pin-2 investigation order: check the PRINTED FLIP COUNTS "
              "for a COUNTING ANOMALY (the fail-closed 9.4% pooled branch) "
              "BEFORE any code-class presumption; genuine recipe-input "
              "assembly or carry-over bugs get fix + re-audit")
FH_ROUTING = ("HANDOFF finding, explicitly NOT metric-menu adjudication — "
              "mechanism split per the A1 discriminating pair: (a) offline "
              "interpolation-quality gen-eig at handoff (st_interp_geneig, "
              "echoed above); (b) per-rung phase-1 u-level re-equilibration "
              "vs phase-0 stationary levels; EEVPD re-entry corroborates but "
              "does not discriminate alone => report + handoff redesign")


def assemble_verdict(arms, wo_ok, wo_reading, ws_pairs_ok, missing):
    """W-a assembly with the checkpoint's routing text. `arms` = list of
    per-arm clause dicts (see keys below)."""
    if missing:
        return f"NOT assembled (W-a requires all three arms; missing: {missing})"
    quals = []
    for a in arms:
        if a.get("wt_bmin_status") == "caveat":
            quals.append(f"{a['name']}: {a['wt_bmin_note']}")
        quals.extend(a.get("zone_notes", []))
    qual_str = (" — QUALIFIERS (pt4 B3-1: must reach the verdict string): "
                + "; ".join(quals)) if quals else ""
    fn = [a["name"] for a in arms if a["f_never"]]
    if fn:
        return ("F-NEVER (phase-0-only; W-S adjudicated on phase-0 data) — "
                f"arm(s) {fn} never triggered by round {TRIG_HI} (A4 exit: "
                "run ended after the phase-0 final save; no re-space, no "
                "phase 1); W-a not assemblable" + qual_str)
    fe = [a["name"] for a in arms if a["f_early"]]
    if fe:
        return ("F-EARLY — stationarity gate insufficient (report; redesign "
                "candidate: 2 consecutive windows) — arm(s) "
                f"{fe} Z <= -2 (systematic under-dispersion, the transit "
                "signature); do NOT re-space-and-hope" + qual_str)
    ws_all = all(a["ws_pass"] and a["trigger_ok"] for a in arms)
    ft = [a["name"] for a in arms
          if a["wt_bmin_status"] == "F-T"
          or not (a["wt_count_ok"] and a["wt_knots_ok"])]
    if ws_all and ft:
        return (f"F-T — ladder reproduction failed beyond the B2' counting "
                f"routing on arm(s) {ft}; {FT_ROUTING}" + qual_str)
    if not ws_all:
        over = [a["name"] for a in arms if a["ws_over"]]
        extra = (f"; Z >= +2 over-dispersion on {over} — unexpected "
                 "direction, report-to-human") if over else ""
        return f"NOT assembled (see clauses; W-S failed{extra})" + qual_str
    if ft:
        return "NOT assembled (see clauses; W-T failed with W-S)" + qual_str
    bad_h = [a["name"] for a in arms if not (a["wh_pass"] and a["wg_pass"])]
    if bad_h:
        return (f"F-H — phase-1 transport/health or W-G regression on "
                f"arm(s) {bad_h} ({FH_ROUTING})" + qual_str)
    rt_bad = [a["name"] for a in arms if not a["rt_ok"]]
    wp_ok = (not rt_bad) and wo_ok and ws_pairs_ok
    if not wp_ok:
        why = []
        if rt_bad:
            why.append(f"per-arm RT < floor on {rt_bad}")
        if not wo_ok:
            why.append(f"pooled occupancy: {wo_reading}")
        if not ws_pairs_ok:
            why.append("W-s pair(s) > 2 sigma")
        return ("NOT assembled — W-P supporting clause failed "
                f"({'; '.join(why)}); tuning-layer-only adjudication note: "
                "W-S/W-T/W-H/W-G pass on all arms, the scheme is adjudicated "
                "as a tuning-layer-only finding (B7 pin: W-a requires W-P)"
                + qual_str)
    return PASS_VERDICT + qual_str


# --------------------------------------------------------------------------
# --selftest
# --------------------------------------------------------------------------
def _mk_arm(name, **kw):
    a = dict(name=name, f_never=False, f_early=False, trigger_ok=True,
             ws_pass=True, ws_over=False, wt_count_ok=True, wt_knots_ok=True,
             wt_bmin_status="exact", wt_bmin_note="", wh_pass=True,
             wg_pass=True, rt_ok=True, zone_notes=[])
    a.update(kw)
    return a


def _selftest():
    # ---- (i) weighted Stouffer against known-Z cases
    se = np.full(10, 0.05)
    cov, Z, ok, fe, ov = stouffer(1.0 + se, se)          # z_i = 1 each
    assert abs(Z - math.sqrt(10.0)) < 1e-12, Z
    assert cov == 10 and not ok and not fe and ov        # Z = 3.162 >= 2
    cov, Z, ok, fe, ov = stouffer(np.ones(10), se)
    assert Z == 0.0 and cov == 10 and ok and not fe and not ov
    se2 = np.array([0.02, 0.05])
    _, Z2, _, _, _ = stouffer(np.array([1.04, 0.95]), se2)   # z = [2, -1]
    assert abs(Z2 - 80.0 / math.sqrt(2900.0)) < 1e-12, Z2
    cov, Z, ok, fe, ov = stouffer(1.0 - 2.5 * se, se)    # z_i = -2.5 each
    assert fe and not ok and not ov
    print(f"[selftest i] Stouffer OK: equal-w Z=sqrt(10)={math.sqrt(10):.6f}, "
          f"mixed Z={Z2:.6f}=80/sqrt(2900), under-dispersed => F-EARLY flag")

    # ---- B2' cross-check vs ladder_recipe.beta_min_rule on the ARCHIVED
    #      arm-A conservative leakage table (loaded, not transcribed)
    s = json.load(open(f"{OUT}/summary_A_power.json"))
    lb = np.array([r["beta"] for r in s["per_beta"]])
    leak = np.array([min(r["leak_Minit"], r["leak_Pinit"])
                     for r in s["per_beta"]])
    assert np.allclose(lb, GRID, rtol=1e-12, atol=0)
    bmr = beta_min_rule(lb, leak, BM_NSYS, BM_BUDGET, BM_PROBE, level=BM_LEVEL)
    expo = (300 - 100) * K_STEPS * NSYS / 1500.0         # 21.33 units
    idx0 = b2prime_beta_min(leak * expo, expo, min_count=0)
    assert abs(GRID[idx0] - bmr) < 1e-12, (idx0, bmr)
    assert idx0 == EXPECTED_IDX and round(bmr, 4) == 0.3594
    print(f"[selftest] B2' (guard off) == beta_min_rule == "
          f"{bmr:.10g} on the archived table; exposure units at R*=300 = "
          f"{expo:.4f} (rd-3 pin-1 recompute)")

    # ---- synthetic 3-arm flip counts (Poisson means from the archived
    #      table, rounded): clean arm admits 0.3594 with k >= 2 guard
    clean = np.round(leak * expo)                        # rung 8 -> 4, rung 9 -> 0
    assert clean[8] == 4 and clean[9] == 0
    assert b2prime_beta_min(clean, expo) == EXPECTED_IDX
    miss = clean.copy()
    miss[8] = 1.0            # k <= 1 at 0.5995 => guard denies 0.3594
    idx_miss = b2prime_beta_min(miss, expo)
    assert idx_miss == EXPECTED_IDX - 1, idx_miss        # one-grid-point HOTTER
    print(f"[selftest] clean counts -> grid[7]; k=1 at grid[8] -> false-cold "
          f"miss to grid[{idx_miss}] (the ~11%/arm branch)")

    # ---- (ii) pooled rescue branch: ST2 misses, pooled counts rescue
    counts = dict(ST1=clean.copy(), ST2=miss.copy(), ST3=clean.copy())
    expos = dict(ST1=expo, ST2=expo, ST3=expo)
    st, note = wt_beta_min_leg("ST2", idx_miss, counts, expos)
    assert st == "caveat", (st, note)
    for nm in ("ST1", "ST3"):
        st_c, _ = wt_beta_min_leg(nm, b2prime_beta_min(counts[nm], expo),
                                  counts, expos)
        assert st_c == "exact"
    arms = [_mk_arm("ST1"),
            _mk_arm("ST2", wt_bmin_status="caveat", wt_bmin_note=note),
            _mk_arm("ST3")]
    v = assemble_verdict(arms, True, "in band", True, [])
    assert v.startswith(PASS_VERDICT), v
    assert "COUNTING-CAVEAT" in v, v
    print(f"[selftest ii] pooled rescue -> caveat; verdict carries the "
          f"caveat qualifier:\n  {v}")

    # ---- (iii) F-T branch: pooled denies (other arms flip at beta = 1.0)
    c1, c3 = clean.copy(), clean.copy()
    c1[9] = 1.0
    c3[9] = 1.0              # per-arm still exact (guard k=1 < 2 at rung 9)
    assert b2prime_beta_min(c1, expo) == EXPECTED_IDX
    counts_ft = dict(ST1=c1, ST2=miss.copy(), ST3=c3)
    st, note = wt_beta_min_leg("ST2", idx_miss, counts_ft, expos)
    assert st == "F-T", (st, note)   # pooled k(1.0)=2 -> pooled admits 0.5995
    arms_ft = [_mk_arm("ST1"),
               _mk_arm("ST2", wt_bmin_status="F-T", wt_bmin_note=note),
               _mk_arm("ST3")]
    v_ft = assemble_verdict(arms_ft, True, "in band", True, [])
    assert v_ft.startswith("F-T"), v_ft
    assert "COUNTING ANOMALY" in v_ft, v_ft
    print(f"[selftest iii] pooled denies -> F-T with pin-2 routing:\n  {v_ft}")

    # ---- remaining verdict branches
    v = assemble_verdict([_mk_arm("ST1")], True, "", True, ["ST2", "ST3"])
    assert v.startswith("NOT assembled (W-a requires all three arms"), v
    v = assemble_verdict([_mk_arm("ST1"), _mk_arm("ST2"),
                          _mk_arm("ST3", f_never=True, trigger_ok=False)],
                         True, "", True, [])
    assert v.startswith("F-NEVER (phase-0-only; W-S adjudicated on phase-0 data)"), v
    v = assemble_verdict([_mk_arm("ST1", f_early=True, ws_pass=False),
                          _mk_arm("ST2"), _mk_arm("ST3")], True, "", True, [])
    assert v.startswith("F-EARLY — stationarity gate insufficient (report; "
                        "redesign candidate: 2 consecutive windows)"), v
    v = assemble_verdict([_mk_arm("ST1", wh_pass=False), _mk_arm("ST2"),
                          _mk_arm("ST3")], True, "", True, [])
    assert v.startswith("F-H"), v
    v = assemble_verdict([_mk_arm("ST1", rt_ok=False), _mk_arm("ST2"),
                          _mk_arm("ST3")], True, "in band", True, [])
    assert "W-P supporting clause failed" in v and "tuning-layer-only" in v, v
    v = assemble_verdict([_mk_arm("ST1"), _mk_arm("ST2"), _mk_arm("ST3")],
                         False, "OUT of band beyond near-edge", True, [])
    assert "W-P supporting clause failed" in v, v
    print("[selftest] verdict branches (missing / F-NEVER / F-EARLY / F-H / "
          "W-P-fail) OK")
    print("[selftest] ALL PASS")


if "--selftest" in sys.argv:
    _selftest()
    sys.exit(0)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def npz_get(sources, names, what):
    """Fetch the first present key among `names` from (npz, label) sources.
    Fail loud with the full available-key inventory."""
    for d, lab in sources:
        if d is None:
            continue
        for n in names:
            if n in d.files:
                return d[n], f"{lab}:{n}"
    inv = {lab: sorted(d.files) for d, lab in sources if d is not None}
    raise KeyError(f"{what}: none of {names} found; available: {inv}")


def orient3(a, n_rung, what, n_chain=NSYS):
    """Reorder a 3-d array to (n_rung, rounds, n_chain); prefers the runner
    convention (rounds, rung, chain) when sizes are ambiguous."""
    a = np.asarray(a)
    if a.ndim != 3:
        raise ValueError(f"{what}: expected 3-d, got shape {a.shape}")
    cand_r = [i for i in range(3) if a.shape[i] == n_rung]
    if not cand_r:
        raise ValueError(f"{what}: no axis of length {n_rung} in {a.shape}")
    ir = 1 if 1 in cand_r else cand_r[0]
    cand_c = [i for i in range(3) if a.shape[i] == n_chain and i != ir]
    if not cand_c:
        raise ValueError(f"{what}: no chain axis of length {n_chain} in {a.shape}")
    ic = 2 if 2 in cand_c else cand_c[-1]
    it = ({0, 1, 2} - {ir, ic}).pop()
    return np.transpose(a, (ir, it, ic))


def tau_rounds_bm(w):
    """Effective per-rung u IAT in rounds from a (T, nchain) window by batch
    means (ladder_recipe.measure_sd_u arithmetic): tau_eff such that
    N_eff = nchain*T/tau_eff."""
    T = w.shape[0]
    s = w.T
    b = max(int(math.sqrt(T)), 2)
    nbat = T // b
    bm = s[:, :nbat * b].reshape(-1, nbat, b).mean(axis=2)
    iat = b * bm.var(axis=1, ddof=1) / np.maximum(s.var(axis=1, ddof=1), 1e-300)
    neff = float((T / np.maximum(iat, 1.0)).sum())
    return w.size / neff


def split_rhat_binary(ch):
    x = ch.astype(float)
    n2 = x.shape[1] // 2
    halves = np.concatenate([x[:, :n2], x[:, n2:2 * n2]], axis=0)
    W = halves.var(axis=1, ddof=1).mean()
    B = n2 * halves.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n2 - 1) / n2 * W + B / n2) / W)) if W > 0 else float("inf")


def occ_a6(m, recent, prev):
    """A6-pinned occupancy reading (pt4_score.py convention). LOW-side
    near-edge corroborated by a RISING trace only; HIGH-side needs
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


# --------------------------------------------------------------------------
# reference artifacts (run in all modes, incl. empty-state: validates the
# reference machinery against the archives before any arm is read)
# --------------------------------------------------------------------------
print("=== GATE PT-5a scorer: reference artifacts ===")
# probe (seed-54): per-beta equilibrium sd(u) + ess for the probe-side se
d54 = np.load(f"{OUT}/arrays_A_power_probe54.npz")
betas54 = np.asarray(d54["betas"], np.float64)
assert np.allclose(betas54, GRID, rtol=1e-12, atol=0), \
    f"probe grid != geomspace(0.01,1,10): {betas54}"
probe_sd, probe_se_bm = measure_sd_u(np.asarray(d54["u"]), 1500)
s54 = json.load(open(f"{OUT}/summary_A_power_probe54.json"))
pb = s54["per_beta"]
assert np.allclose([r["beta"] for r in pb], GRID, rtol=1e-12, atol=0)
# probe-side N_eff: the summary's per-beta ess (both init groups) — B4 pin
probe_neff = np.array([r["ess_proxy_Minit"] + r["ess_proxy_Pinit"] for r in pb])
probe_rel_se = 1.0 / np.sqrt(2.0 * probe_neff)
probe_iat = np.array([0.5 * (r["iat_Minit"] + r["iat_Pinit"]) for r in pb])
print("probe (seed 54, discard 1500): per-beta sd(u), ess -> rel se "
      "(batch-means rel se cross-check in parens):")
for i, b in enumerate(GRID):
    print(f"  beta={b:8.4f}  sd={probe_sd[i]:10.3f}  ess(M+P)={probe_neff[i]:7.1f}"
          f"  rel_se={probe_rel_se[i]*100:5.2f}%  (bm {probe_se_bm[i]/probe_sd[i]*100:5.2f}%)"
          f"  iat~{probe_iat[i]:6.1f} steps")

# certified reference ladder: recomputed from the ARCHIVED inputs (L1b
# pattern, full precision) + reference knot se (L2 pattern: se(sd) from
# measure_sd_u on arrays_A_power.npz, asserted consistent with the json)
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
# archived probe leakage at the three named betas (for the report clause)
LEAK_REF3 = {7: leak_A[7], 8: leak_A[8], 9: leak_A[9]}
print(f"  archived conservative leakage at beta 0.3594/0.5995/1.0 = "
      f"{leak_A[7]:.4f}/{leak_A[8]:.4f}/{leak_A[9]:.4f}")

# MAMS pools (POSITIONS ONLY — standing directive) for Sigma_ref / occupancy
draws = np.load(MAMS)["samples_z"].reshape(-1, 33).astype(np.float64)
pk = draws[:, POCKET_COL] > POCKET_THR
cov_P, cov_M = np.cov(draws[pk].T), np.cov(draws[~pk].T)
dmu = draws[pk].mean(0) - draws[~pk].mean(0)
cov_pool = np.cov(draws.T)              # fixed pooled diagnostic ref (W/B echo)
L_POOL = np.linalg.cholesky(cov_pool)
fs_ref = np.load(f"{OUT}/pt3_fs_reference.npz")["eigvec_z"]


def sigma_ref(w):
    return w * cov_P + (1 - w) * cov_M + w * (1 - w) * np.outer(dmu, dmu)


# --------------------------------------------------------------------------
# arm loop (pass 1: load + per-arm quantities; beta_min legs need pooled
# counts, resolved in pass 2)
# --------------------------------------------------------------------------
res = {}
counts_by_arm, expo_by_arm = {}, {}
for name in ARMS:
    tag = TAGS[name]
    p1 = f"{OUT}/arrays_{tag}.npz"
    p0 = f"{OUT}/arrays_{tag}_phase0.npz"
    have1, have0 = os.path.exists(p1), os.path.exists(p0)
    if not have1 and not have0:
        print(f"\n[{name}] MISSING — skipped")
        continue
    d1 = np.load(p1) if have1 else None
    d0 = np.load(p0) if have0 else None
    if have1 and not have0:
        raise FileNotFoundError(f"[{name}] production npz present but phase-0 "
                                f"companion {p0} missing — refusing to score "
                                f"(phase-0 data required for W-S recompute + "
                                f"A1 leg (b) baseline)")
    src1 = [(d1, "arrays"), (d0, "phase0")]
    src0 = [(d0, "phase0"), (d1, "arrays")]
    card = None
    card_path = f"{OUT}/summary_{tag}.json"
    if os.path.exists(card_path):
        card = json.load(open(card_path)).get("model_card", {})

    # ---- pinned asserts (fail loud) -------------------------------------
    grid0, gsrc = npz_get(src0, ["betas", "st_grid", "betas_phase0"],
                          f"[{name}] phase-0 grid")
    grid0 = np.asarray(grid0, np.float64)
    if have1 and grid0.shape[0] != 10:      # production npz betas = knots
        grid0, gsrc = npz_get([(d0, "phase0")], ["betas", "st_grid"],
                              f"[{name}] phase-0 grid")
        grid0 = np.asarray(grid0, np.float64)
    assert grid0.shape == (10,) and np.allclose(grid0, GRID, rtol=1e-12, atol=0), \
        f"[{name}] phase-0 grid != geomspace(0.01,1,10): {grid0} ({gsrc})"
    # swaps-off flag (B1): npz flag, else model card
    swaps_off = None
    for d, lab in src0:
        if d is not None and "swaps_off_phase0" in d.files:
            swaps_off = bool(np.asarray(d["swaps_off_phase0"]).item())
            break
    if swaps_off is None and card is not None:
        swaps_off = bool(card.get("swaps_off_phase0"))
    assert swaps_off is True, f"[{name}] swaps_off_phase0 flag missing/False (B1)!"
    if card is not None and "seed" in card:
        assert int(card["seed"]) == SEEDS[name], \
            f"[{name}] seed {card['seed']} != pinned {SEEDS[name]}"
    else:
        print(f"[{name}] NOTE: no model card seed found — seed unverified "
              f"(pinned {SEEDS[name]})")

    f_never = not have1
    # phase-0 u (W-S recompute basis + A1 leg-(b) baseline; the ONLY W-S
    # source for F-NEVER arms)
    u0_raw, _ = npz_get([(d0, "phase0")], ["u", "st_u", "u_phase0", "st_u0"],
                        f"[{name}] phase-0 u")
    u0 = orient3(u0_raw, 10, f"[{name}] phase-0 u")
    R0 = u0.shape[1]

    if not f_never:
        trig_a, _ = npz_get(src1, ["st_trigger"], f"[{name}] st_trigger")
        trigger = int(np.asarray(trig_a).item())
        assert TRIG_LO <= trigger <= TRIG_HI, \
            f"[{name}] st_trigger {trigger} outside [{TRIG_LO}, {TRIG_HI}]"
        trigger_ok = True
        rs_a, _ = npz_get(src1, ["st_R_star", "st_rstar"], f"[{name}] st_R_star")
        R_star = int(np.asarray(rs_a).item())
        assert R_star == max(trigger, 300), \
            f"[{name}] st_R_star {R_star} != max(trigger, 300) = {max(trigger, 300)}"
        assert R0 >= R_star, f"[{name}] phase-0 u has {R0} rounds < R* = {R_star}"
        R_end = R_star
    else:
        # A4 exit: no trigger by the deadline; run ended after the phase-0
        # final save. W-S adjudicated on phase-0 data (last 100 rounds).
        trigger, R_star, trigger_ok = None, None, False
        for d, lab in src0:
            if d is not None and "st_trigger" in d.files:
                print(f"*** [{name}] WARNING: st_trigger present ({lab}) but "
                      f"no production npz — inconsistent A4 state ***")
        if R0 < TRIG_HI:
            print(f"*** [{name}] WARNING: phase-0 has only {R0} rounds "
                  f"< deadline {TRIG_HI} — truncated F-NEVER arm ***")
        R_end = R0
    # rd-3 pin 1: SHARED exposure denominator, RECOMPUTED (F-NEVER arms:
    # (R0-100), a scorer resolution — no R* exists)
    expo_steps = (R_end - 100) * K_STEPS * NSYS
    expo_units = expo_steps / 1500.0
    for d, lab in src1:
        if d is None:
            continue
        for kk in ("st_exposure", "st_exposure_steps", "st_leak_exposure"):
            if kk in d.files:
                stored = float(np.asarray(d[kk]).item())
                assert (abs(stored - expo_steps) < 0.5
                        or abs(stored - expo_units) < 1e-9), \
                    (f"[{name}] stored exposure {stored} matches neither "
                     f"{expo_steps} kernel steps nor {expo_units:.6f} "
                     f"per-1500 units (rd-3 pin-1 recompute)")

    # sd(u) window: rounds (R_end-100, R_end], pooled chains, NO mean removal
    w0 = u0[:, R_end - 100:R_end, :]
    sd_rec = np.array([np.std(w0[r].reshape(-1)) for r in range(10)])
    tau_rec = np.array([tau_rounds_bm(w0[r]) for r in range(10)])
    lvl0 = w0.mean(axis=(1, 2))                  # per-rung stationary u mean

    if not f_never:
        sd_in, _ = npz_get(src1, ["st_sd_u"], f"[{name}] st_sd_u")
        sd_in = np.asarray(sd_in, np.float64)
        assert sd_in.shape == (10,), f"[{name}] st_sd_u shape {sd_in.shape}"
        dev = float(np.max(np.abs(sd_rec - sd_in) / sd_in))
        if dev > 1e-6:
            print(f"*** [{name}] WARNING: recomputed phase-0 sd(u) over "
                  f"(R*-100, R*] deviates from st_sd_u by up to {dev:.2e} "
                  f"(relative) — internal-consistency anomaly ***")
        tau_in, _ = npz_get(src1, ["st_tau_rounds", "st_tau"],
                            f"[{name}] st_tau_rounds")
        tau_in = np.maximum(np.asarray(tau_in, np.float64), 1.0)
        assert tau_in.shape == (10,)
        flips, fsrc = npz_get(src1, ["st_flip_counts", "st_flips",
                                     "flip_counts", "st_leak_counts"],
                              f"[{name}] flip counts")
        flips = np.asarray(flips, np.float64)
        if flips.ndim == 2:                 # (rung, chain) -> per-rung totals
            flips = flips.sum(axis=1)
        assert flips.shape == (10,), f"[{name}] flip counts shape {flips.shape}"
        bmin_a, _ = npz_get(src1, ["st_beta_min"], f"[{name}] st_beta_min")
        bmin_arm = float(np.asarray(bmin_a).item())
        bmin_idx = int(np.argmin(np.abs(GRID - bmin_arm)))
        assert abs(GRID[bmin_idx] - bmin_arm) < 1e-9, \
            f"[{name}] st_beta_min {bmin_arm} is not a grid point"
        knots_a, _ = npz_get(src1, ["st_knots", "st_ladder"], f"[{name}] st_knots")
        st_knots = np.asarray(knots_a, np.float64)
    else:
        sd_in, tau_in = sd_rec, np.maximum(tau_rec, 1.0)   # phase-0-only basis
        for d, lab in src0:                 # prefer stored values if archived
            if d is not None and "st_sd_u" in d.files:
                sd_in = np.asarray(d["st_sd_u"], np.float64)
            if d is not None and "st_tau_rounds" in d.files:
                tau_in = np.maximum(np.asarray(d["st_tau_rounds"], np.float64), 1.0)
        flips, fsrc = None, "n/a"
        for d, lab in src0:
            if d is not None:
                for kk in ("st_flip_counts", "st_flips", "flip_counts",
                           "st_leak_counts"):
                    if kk in d.files:
                        flips = np.asarray(d[kk], np.float64)
                        if flips.ndim == 2:
                            flips = flips.sum(axis=1)
                        fsrc = f"{lab}:{kk}"
                        break
            if flips is not None:
                break
        bmin_arm, bmin_idx, st_knots = None, None, None
    if flips is not None:
        counts_by_arm[name] = flips
        expo_by_arm[name] = expo_units

    print(f"\n[{name}] trigger={trigger} R*={R_star} exposure="
          f"{expo_steps} kernel steps = {expo_units:.2f} per-1500 units "
          f"(recomputed at R_end={R_end}); beta_min="
          f"{'n/a' if bmin_arm is None else f'{bmin_arm:.6g} (grid[{bmin_idx}])'}; "
          f"{'PHASE-0 ONLY (F-NEVER)' if f_never else 'phase 1 present'}")

    # ---- W-S ------------------------------------------------------------
    rel_in = 1.0 / np.sqrt(2.0 * (NSYS * 100.0 / tau_in))
    rel_comb = np.sqrt(rel_in ** 2 + probe_rel_se ** 2)
    ratios = sd_in / probe_sd
    cov_n, Z, ws_ratio_ok, f_early, ws_over = stouffer(ratios, rel_comb)
    ws_pass = bool(trigger_ok and ws_ratio_ok)
    print(f"  [W-S {name}] per-rung in-run vs probe sd(u) "
          f"(N_eff_in = 16*100/tau; tau from st_tau_rounds, recompute in parens):")
    for i, b in enumerate(GRID):
        covd = abs(ratios[i] - 1.0) <= 3.0 * rel_comb[i]
        print(f"    beta={b:8.4f} sd_in={sd_in[i]:10.3f} sd_probe={probe_sd[i]:10.3f} "
              f"ratio={ratios[i]:.4f} se_in={rel_in[i]*100:4.1f}% "
              f"se_probe={probe_rel_se[i]*100:4.1f}% se_comb={rel_comb[i]*100:4.1f}% "
              f"z={((ratios[i]-1)/rel_comb[i]):+6.2f} w={1/rel_comb[i]:6.1f} "
              f"tau={tau_in[i]:6.2f} ({tau_rec[i]:6.2f}) "
              f"{'ok' if covd else '*** OUTSIDE 3se ***'}")
    print(f"  [W-S {name}] coverage {cov_n}/10 (>= 8 required); weighted "
          f"Stouffer Z = {Z:+.3f} (|Z| < 2 required); trigger clause "
          f"{'ok' if trigger_ok else 'FAILED (no trigger in [200,400])'} -> "
          f"{'PASS' if ws_pass else 'FAIL'}")
    if f_early:
        print(f"  *** [{name}] F-EARLY: Z = {Z:+.3f} <= -2 — systematic "
              f"UNDER-dispersion, the transit signature; stationarity gate "
              f"insufficient ***")
    if ws_over:
        print(f"  *** [{name}] Z = {Z:+.3f} >= +2 — OVER-dispersion, "
              f"unexpected direction: report-to-human ***")

    # ---- phase-0 leakage report (probe comparison + hot-rung monitor) ----
    hot_flips = None
    if flips is not None:
        rates = flips / expo_units
        rate_se = np.sqrt(np.maximum(flips, 0.0)) / expo_units
        print(f"  [leak {name}] per-rung flip counts (rounds (100, R_end], {fsrc}) "
              f"-> rate per 1500 steps (Poisson se); probe archived at "
              f"0.3594/0.5995/1.0 = {LEAK_REF3[7]:.4f}/{LEAK_REF3[8]:.4f}/{LEAK_REF3[9]:.4f}:")
        for i, b in enumerate(GRID):
            ref_s = f"  [probe {LEAK_REF3[i]:.4f}]" if i in LEAK_REF3 else ""
            print(f"    beta={b:8.4f} k={int(flips[i]):4d} rate={rates[i]:.4f} "
                  f"+-{rate_se[i]:.4f}{ref_s}")
        hot_flips = int(flips[0])
        if hot_flips == 0:
            print(f"  *** [{name}] F-flip: ZERO hot-rung (beta={GRID[0]:.3g}) "
                  f"flips post-100 on a posterior known to cross — in-run "
                  f"leakage counting broken (code class) ***")
    else:
        print(f"  [leak {name}] no flip counts archived (F-NEVER arm) — "
              f"leakage report unavailable")
        rates = None
    # raw-indicator recompute if the runner archived it (report-only:
    # round-granularity transitions can undercount vs in-run step counting)
    ind0 = None
    for d, lab in [(d0, "phase0"), (d1, "arrays")]:
        if d is not None:
            for kk in ("ind", "st_ind", "basin_ind"):
                if kk in d.files:
                    ind0 = orient3(d[kk], 10, f"[{name}] phase-0 ind")
                    break
        if ind0 is not None:
            break
    if ind0 is not None and ind0.shape[1] >= R_end:
        rec = (np.abs(np.diff(ind0[:, 100:R_end, :].astype(np.int8),
                              axis=1)) > 0).sum(axis=(1, 2))
        print(f"    round-boundary recount from raw indicator: "
              f"{[int(v) for v in rec]} (report-only)")
    # runner-side B2' recompute (internal consistency)
    if flips is not None and bmin_idx is not None:
        idx_rec = b2prime_beta_min(flips, expo_units, min_count=2)
        if idx_rec != bmin_idx:
            print(f"*** [{name}] WARNING: scorer B2' recompute on stored counts "
                  f"-> grid[{idx_rec}] but st_beta_min = grid[{bmin_idx}] — "
                  f"runner/scorer rule mismatch (investigate BEFORE trusting "
                  f"the beta_min leg) ***")

    # per-rung IAT: K design data (unscored, PT-5 input)
    print(f"  [K design data {name}] st_tau_rounds (u IAT, rounds; K=10 "
          f"steps/round): {[f'{v:.2f}' for v in tau_in]} (unscored)")

    arm = dict(name=name, f_never=bool(f_never), f_early=bool(f_early),
               trigger_ok=bool(trigger_ok), trigger=trigger, R_star=R_star,
               expo_units=expo_units, ws_pass=bool(ws_pass),
               ws_cov=cov_n, ws_Z=Z, ws_over=bool(ws_over),
               ratios=[float(v) for v in ratios],
               rel_se_comb=[float(v) for v in rel_comb],
               sd_in=[float(v) for v in sd_in],
               tau_rounds=[float(v) for v in tau_in],
               flip_counts=(None if flips is None else [int(v) for v in flips]),
               leak_rates=(None if rates is None else [float(v) for v in rates]),
               hot_rung_flips=hot_flips,
               f_flip=bool(hot_flips == 0) if hot_flips is not None else None,
               bmin=bmin_arm, bmin_idx=bmin_idx,
               st_knots=(None if st_knots is None
                         else [float(v) for v in st_knots]),
               zone_notes=[])

    if f_never:
        print(f"  [{name}] F-NEVER: no production npz — W-S adjudicated on "
              f"phase-0 data above; no phase 1, nothing downstream (A4)")
        arm.update(wt_count_ok=False, wt_knots_ok=False,
                   wt_bmin_status="F-NEVER", wt_bmin_note="no phase 1",
                   wh_pass=False, wg_pass=False, rt_ok=False)
        res[name] = arm
        continue

    # ---- phase-1 asserts + clauses ---------------------------------------
    rounds = int(d1["rounds_done"])
    betas1 = np.asarray(d1["betas"], np.float64)
    assert np.array_equal(betas1, st_knots), \
        f"[{name}] phase-1 betas != st_knots EXACTLY: {betas1} vs {st_knots}"
    assert float(betas1[-1]) == 1.0, f"[{name}] cold rung is not beta=1"
    assert list(d1["metric_windows"]) == WINDOWS, f"[{name}] non-pinned windows!"
    assert str(d1["metric_estimator"]) == "within", \
        f"[{name}] estimator is not 'within'!"
    assert d1["cold_ind"].shape[1] == NSYS, \
        f"[{name}] NSYS {d1['cold_ind'].shape[1]} != pinned 16"
    assert rounds <= ROUNDS_NOM, \
        f"[{name}] rounds_done {rounds} > nominal {ROUNDS_NOM}"
    assert d1["cold_ind"].shape[0] >= rounds, f"[{name}] truncated buffer"
    if rounds < ROUNDS_NOM:
        print(f"*** {name} INCOMPLETE {rounds}/{ROUNDS_NOM} — op-7 scaling ***")
    # A5 carry-over map: nearest-log-beta rule on (grid, ACTUAL knots)
    exp_src = np.array([int(np.argmin(np.abs(np.log(k) - np.log(GRID))))
                        for k in st_knots])
    cmap, csrc = npz_get(src1, ["st_carryover_map", "st_carryover"],
                         f"[{name}] carry-over map")
    cmap = np.asarray(cmap)
    if cmap.ndim == 2 and cmap.shape[1] == 2:
        cmap = cmap[:, 1]                       # (new, old) pairs -> old
    if np.issubdtype(cmap.dtype, np.integer):
        assert np.array_equal(cmap.astype(int), exp_src), \
            f"[{name}] carry-over map {cmap} != nearest-log-beta {exp_src}"
    else:
        assert np.allclose(np.asarray(cmap, float), GRID[exp_src],
                           rtol=1e-9, atol=0), \
            f"[{name}] carry-over map {cmap} != nearest-log-beta {GRID[exp_src]}"
    print(f"  [{name}] carry-over map OK ({csrc}): "
          f"{[f'{k:.4f}<-{GRID[j]:.4f}' for k, j in zip(st_knots, exp_src)]}")

    nrung1 = len(st_knots)
    lo = max(500, rounds - 500)
    clipped_out = rounds <= lo
    cold = d1["cold_ind"][:rounds]
    ev = d1["eevpd"][:rounds]

    # ---- W-H -------------------------------------------------------------
    if clipped_out:
        print(f"*** [{name}] no post-freeze rounds (rounds={rounds} <= "
              f"lo={lo}) — phase-1 clauses unevaluable, marked FAIL ***")
        arm.update(rounds=rounds, wh_pass=False, wg_pass=False, rt_ok=False,
                   wt_count_ok=False, wt_knots_ok=False,
                   wt_bmin_status="F-T",
                   wt_bmin_note="no post-freeze scoring window")
        res[name] = arm
        continue
    ev_med = np.median(ev[lo:], axis=0)
    ev_ok = bool(np.all((ev_med >= EEVPD_BAND[0]) & (ev_med <= EEVPD_BAND[1])))
    att = d1["swap_attempts"].astype(float)
    acc = d1["swap_accepts"].astype(float)
    pacc = acc.sum(1) / np.maximum(att.sum(1), 1)
    pacc_ok = bool(np.all((pacc >= PACC_BAND[0]) & (pacc <= PACC_BAND[1])))
    n_rev = int(d1["n_revert"])
    wh_pass = bool(ev_ok and pacc_ok and n_rev == 0)
    rhat = split_rhat_binary(cold[lo:].T)       # report-only
    print(f"  [W-H {name}] eevpd_med(post-lo)={['%.1e' % v for v in ev_med]} "
          f"in [1e-4,2e-3]: {ev_ok}; pair acc {np.round(pacc, 3).tolist()} "
          f"in [0.25,0.65]: {pacc_ok}; NaN reverts={n_rev} -> "
          f"{'PASS' if wh_pass else 'FAIL'}; split-Rhat={rhat:.4f} (report)")

    # ---- occupancy / W-P per-arm -----------------------------------------
    win = cold[lo:]
    sysm = win.mean(axis=0)
    m = float(sysm.mean())
    sd_m = float(sysm.std(ddof=1))
    se_m = sd_m / math.sqrt(NSYS)
    occ_recent = float(win.mean())
    occ_prev = float(cold[max(0, lo - 500):lo].mean())
    rtp = int(d1["round_trips_pocket"].sum())
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

    # ---- W-G (B5 regression guard) ----------------------------------------
    ref = sigma_ref(m)
    Lr = np.linalg.cholesky(ref)
    A = d1["metric_frozen"][-1]                  # cold rung
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
    # {19,2,3,20} attribution + F-S-style low-side check (report-only)
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
        arm["zone_notes"] = arm.get("zone_notes", []) + [
            f"{name}: {n_soft_hi} axis(es) in (3,10] (PT-4 zone label, "
            f"report-only this gate)"]
    if n_soft_lo > 0 or n_hard_lo > 0:
        arm["zone_notes"] = arm.get("zone_notes", []) + [
            f"{name}: {n_soft_lo + n_hard_lo} low-side axis(es) "
            f"(< 1/3{'; <0.1 present' if n_hard_lo else ''}, report-only)"]
    if fs_aligned:
        arm["zone_notes"] = arm.get("zone_notes", []) + [
            f"{name}: F-S-style aligned low-side direction (recorded)"]

    # ---- W/B echo (pt4 convention, report-only here) ----------------------
    Wc, Bc = d1["metric_within_covs"], d1["metric_between_covs"]
    if Wc.ndim == 4 and Wc.shape[0] > 0:
        for b in range(Wc.shape[0]):
            Tb = T_WIN[b] if b < len(T_WIN) else T_WIN[-1]
            pool_rec = (((NSYS - 1) * Tb * Wc[b, -1]
                         + NSYS * (Tb - 1) * Bc[b, -1]) / (Tb * NSYS - 1))
            gw = np.sort(geneig_vs(Wc[b, -1], L_POOL)[0])
            gp = np.sort(geneig_vs(pool_rec, L_POOL)[0])
            print(f"    [W/B echo b{b} T={Tb}] cold-rung max gen-eig vs fixed "
                  f"pooled ref: W-only {gw[-1]:.1f}, reconstructed-pooled "
                  f"{gp[-1]:.1f} (report-only, unscored this gate)")
    else:
        print(f"    [W/B echo] within/between covs absent or empty — "
              f"skipped (unscored this gate)")

    # ---- handoff reports (A1 discriminating pair) --------------------------
    ig = None
    for d, lab in src1:
        if d is not None and "st_interp_geneig" in d.files:
            ig = d["st_interp_geneig"]
            break
    if ig is not None:
        print(f"  [A1 leg (a) {name}] st_interp_geneig echo: "
              f"{np.array2string(np.asarray(ig), precision=3)}")
    else:
        print(f"*** [{name}] st_interp_geneig MISSING — A1 leg (a) "
              f"unevaluable (offline interpolation check not archived) ***")
    u1 = None
    for d, lab in [(d1, "arrays")]:
        for kk in ("u", "st_u", "u_phase1", "st_u_phase1"):
            if kk in d.files and np.asarray(d[kk]).ndim == 3:
                try:
                    u1 = orient3(d[kk], nrung1, f"[{name}] phase-1 u")
                except ValueError:
                    u1 = None
                break
    reeq = None
    if u1 is not None and u1.shape[1] >= 100:
        m1 = u1[:, :100, :].mean(axis=(1, 2))    # first-100-round u mean
        lgrid, lknot = np.log(GRID), np.log(st_knots)
        lvl_interp = np.interp(lknot, lgrid, lvl0)
        sd_interp = np.exp(np.interp(lknot, lgrid, np.log(sd_in)))
        reeq = []
        print(f"  [A1 leg (b) {name}] phase-1 first-100-round u vs phase-0 "
              f"stationary level, in units of phase-0 sd(u):")
        for r in range(nrung1):
            s_r = exp_src[r]
            d_src = (m1[r] - lvl0[s_r]) / sd_in[s_r]
            d_int = (m1[r] - lvl_interp[r]) / sd_interp[r]
            reeq.append(dict(beta=float(st_knots[r]), d_src=float(d_src),
                             d_interp=float(d_int)))
            print(f"    beta={st_knots[r]:.4f}: vs source rung "
                  f"(beta={GRID[s_r]:.4f}) {d_src:+.2f} sd; vs log-beta "
                  f"interp {d_int:+.2f} sd")
    else:
        print(f"*** [{name}] phase-1 u trace not found/short — A1 leg (b) "
              f"re-equilibration report unavailable ***")

    arm.update(rounds=rounds, wh_pass=wh_pass, ev_med=ev_med.tolist(),
               pair_acc=pacc.tolist(), n_revert=n_rev, split_rhat=rhat,
               wg_pass=wg_pass, geneig_full=[float(v) for v in g],
               n_gt10=n_gt10, n_soft_hi=n_soft_hi, n_soft_lo=n_soft_lo,
               fs_aligned=bool(fs_aligned), rt_pocket=rtp,
               floor_eff=floor_eff, rt_ok=rt_ok, m=m, sd=sd_m, se=se_m,
               per_system_means=[float(v) for v in sysm],
               occ_recent=occ_recent, occ_prev=occ_prev,
               reequilibration=reeq,
               interp_geneig=(np.asarray(ig, float).tolist()
                              if ig is not None else None))
    res[name] = arm

# --------------------------------------------------------------------------
# pass 2: beta_min legs (need pooled counts) + W-T design comparison
# --------------------------------------------------------------------------
for name, arm in res.items():
    if arm["f_never"]:
        print(f"\n[W-T {name}] F-NEVER — no phase 1, W-T not adjudicated")
        continue
    status, note = wt_beta_min_leg(name, arm["bmin_idx"], counts_by_arm,
                                   expo_by_arm)
    arm["wt_bmin_status"], arm["wt_bmin_note"] = status, note
    # count/knot legs: scorer recompute from (grid, st_sd_u) at the
    # adjudicated beta_min (the pooled-rescued 0.3594 under the caveat
    # branch — the T-claim tests measurements + recipe, see docstring)
    bmin_used = BMIN_REF if status in ("exact", "caveat") else arm["bmin"]
    sd_in = np.asarray(arm["sd_in"])
    des = design_ladder(GRID, sd_in, bmin_used, TARGET_NATS)
    count_ok = bool(des["n_rungs"] == 6)
    rel_in = 1.0 / np.sqrt(2.0 * (NSYS * 100.0 / np.asarray(arm["tau_rounds"])))
    se_in_knots = knot_se(GRID, sd_in, sd_in * rel_in, bmin_used, TARGET_NATS)
    print(f"\n[W-T {name}] beta_min leg: {status.upper()} — {note}")
    if status == "caveat":
        print(f"  counting-caveat branch: count/knot legs evaluated on the "
              f"scorer recompute at the POOLED-rescued beta_min "
              f"{BMIN_REF:.6g}; the arm RAN st_knots="
              f"{[round(k, 4) for k in arm['st_knots']]} — phase-1 clauses "
              f"carry the arm's actual ladder")
    if count_ok:
        se_comb = np.sqrt(REF_KNOT_SE ** 2 + se_in_knots ** 2)
        dk = np.abs(np.asarray(des["knots"]) - REF_KNOTS)
        knots_ok = bool(np.all(dk <= 3.0 * np.maximum(se_comb, 1e-15)))
        for i in range(6):
            print(f"  knot {i}: ref={REF_KNOTS[i]:.6f} in-run="
                  f"{des['knots'][i]:.6f} |d|={dk[i]:.2e} "
                  f"3se_comb={3 * se_comb[i]:.2e} "
                  f"{'ok' if dk[i] <= 3 * max(se_comb[i], 1e-15) else '*** EXCEEDS ***'}")
        arm["knot_dev"] = [float(v) for v in dk]
        arm["knot_se_comb"] = [float(v) for v in se_comb]
    else:
        knots_ok = False
        print(f"  rung count {des['n_rungs']} != 6 — knot comparison moot")
    # internal consistency: runner's own design vs its stored knots
    des_own = design_ladder(GRID, sd_in, arm["bmin"], TARGET_NATS)
    if (des_own["n_rungs"] != len(arm["st_knots"])
            or np.abs(np.asarray(des_own["knots"])
                      - np.asarray(arm["st_knots"])).max() > 1e-9):
        print(f"*** [{name}] WARNING: scorer recompute at the ARM's beta_min "
              f"does not reproduce st_knots (recipe-input assembly suspect — "
              f"F-T investigation input) ***")
    arm["wt_count_ok"], arm["wt_knots_ok"] = count_ok, knots_ok
    wt_pass = count_ok and knots_ok and status in ("exact", "caveat")
    print(f"  [W-T {name}] count==6: {count_ok}; knots<=3se: {knots_ok}; "
          f"beta_min leg: {status} -> {'PASS' if wt_pass else 'FAIL'}")
    if not wt_pass and status == "F-T":
        print(f"  *** [{name}] F-T on the beta_min leg — {FT_ROUTING} ***")

# --------------------------------------------------------------------------
# pooled W-P: occupancy (48 systems), W-s pairs
# --------------------------------------------------------------------------
pooled = {}
wo_ok, wo_reading = False, "unavailable"
phase1_arms = [n for n in ARMS if n in res and not res[n]["f_never"]
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
              f"{clipped} < 1000 rounds — consult the pre-committed coldocc "
              f"overlay plot before accepting any near-edge read ***")
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
            print(f"  *** F-eq: pair > 3 sigma ({3 * s:.4f}) — seed pooling "
                  f"invalid at this budget (report, no auto-lever) ***")
        return ok
    return None


print()
ws_pairs = dict(S1S2=pair("ST1", "ST2"), S1S3=pair("ST1", "ST3"),
                S2S3=pair("ST2", "ST3"))
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
      "clause here (layering pin) — the PT-4 human menu (bounded metric vs "
      "accept-(3,28] vs both) remains OPEN; K = 10 preset, per-rung IAT "
      "reported above as PT-5 design data.")

out = dict(arms=res, pooled_occ=pooled, ws_pairs=ws_pairs,
           reference=dict(knots=[float(v) for v in REF_KNOTS],
                          knot_se=[float(v) for v in REF_KNOT_SE],
                          beta_min=float(BMIN_REF),
                          probe_sd=[float(v) for v in probe_sd],
                          probe_rel_se=[float(v) for v in probe_rel_se]),
           verdict=verdict)
json.dump(out, open(f"{OUT}/pt5a_score.json", "w"), indent=1, default=str)
print(f"\n-> {OUT}/pt5a_score.json")
