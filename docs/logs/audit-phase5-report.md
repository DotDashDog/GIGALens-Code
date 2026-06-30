# Adversarial audit — Phase-5 LAPS validation report

Target: `docs/logs/laps_validation_report.md`. Checked against raw artifacts in
`experiments/laps_validation/results/{core,mscale,ksweep,initmode,schedC}/`.
All `b2_floor` columns = 1/M; `b2_floor2` = 2/M. Verdicts below cite specific rows.

## Per-claim verdicts

### Claim 1 — Unbiasedness: b² ∝ 1/M, no plateau, "pure finite-M noise"
**Verdict: SUPPORTED as a trend, OVER-CLAIMED in certainty.**
- b²_avg / (1/M) seed-means (3 seeds):
  - T_iso: M512 = 1.45, M2048 = 1.55, M8192 = 0.92
  - T_ill: M512 = 1.39, M2048 = 1.03, M8192 = 1.31
- Neither bed's ratio *rises* monotonically with M, so there is **no plateau signature** — good. But within-M seed spread is ~2× (individual seeds 0.36×–2.25×), so the SE on each ratio is large. With only **3 seeds, 2 beds, M_max = 8192** (floor still ~1.2e-4), a residual constant bias b₀² up to ~1e-4 would be invisible. The data is *consistent with* no plateau; it cannot *establish* "pure finite-M noise / unbiased." Report should scope to "no detectable plateau out to M=8192."
- Inconsistency: §1 says "b²_avg sits 1.0–1.4× the floor throughout" — actual seed-means reach **1.55** (T_iso M2048); §1 says "b²_max ~3–4×" — actual b²_max/floor reaches **10.9×** (mscale T_iso M512 seed0 = 0.02132/0.00195), ~6.9× (T_ill M512 seed0), and the cited `summary.png` plainly shows b²_max ~6× the floor at M=512, not 3–4×. The "3–4×" is the *core* warm number, mis-applied to the *mscale* dataset that is the actual evidence.
- Scope creep: mscale only covers **T_iso and T_ill**. T_corr and T_curve are N=1, M=512, single seed (core only). The principled unbiasedness test was **never run on the banana (T_curve)**; extending "unbiased" to all four beds is unsupported.

### Claim 2 — "Switch decisive; schedule/C second-order; Phase-2 corrects Phase-1"
**Verdict: switch-decisiveness PARTIALLY SUPPORTED (leans on out-of-band tests); "no efficiency separation" UNSUPPORTED — contradicted by the report's own plot.**
- "steps_to_floor 76–125, within noise" is **false**. `schedC/summary.png` is titled *cost-to-equilibrate* and shows four cleanly separated, tiny-error-bar clusters: paper/C=0.1 ≈75, emaus/C=0.025 ≈78, paper/C=0.025 ≈105, emaus/C=0.1 ≈125. (CSV min = 68, not 76.) Schedule×C **clearly** separates equilibration cost; the claim "no clean efficiency separation" is contradicted by the very figure cited. This is a plots-before-metrics violation.
- Bias second-order: roughly OK but glossed. Paper/C=0.1 fails b2_success in **2/3 seeds** (b2_max 0.01185, 0.01191) and b2_avg mean = 1.49× floor vs paper/C=0.025 = 0.86× floor (~1.7× higher). "ALL give unbiased final b²" overstates; by the report's own floor criterion C=0.1 is borderline.
- Switch decisiveness on the grid is **weak**: in warm core E-A, the never-firing emaus switch (switched=False@300) gives T_iso sched-emaus/sw-emaus b2_avg = **0.00027** (the *best* run) and only ~2–3× worse b2_avg in the paper-schedule cells. The dramatic "mostly fails to converge" symptom comes from unit/smoke tests **not in these CSVs**; the grid CSVs show only a factor ~2–3 on easy warm beds.
- "Phase-2 corrects Phase-1 bias" is **weakly evidenced, not isolated**. b2_avg_at_switch → final drops when Phase-1 is biased (e.g. emaus/C=0.1 seed1: 0.0245 → 0.0031) but rises in others (paper/C=0.025 seed1: 0.00126 → 0.00224). There is **no Phase-2-off control run**, so the mechanism is asserted via correlation.

### Claim 3 — k=1.5 justification
**Verdict: "k=1.0 too strict" SUPPORTED; k=1.5 specifics mildly OVER-CLAIMED; k∈[1.5,3] honestly admitted unseparated.**
- k=1.0 (cold T_ill): switch_index = 325, 175, **400 (seed2 never fires, switched=False)**. So k=1.0 fires far after the floor (steps_to_floor=105) or never → "too strict" holds. But §4's "switch@300" hides that seed1 fired at 175 (prompt) and seed2 never fired.
- k=1.5/2/3 are **window-guard-bound, not k-bound**: k=2.0 and k=3.0 rows are **byte-identical per seed** (switch_index 150,150,150; identical b² to 16 digits) because the 150-step window-eligibility guard binds first. k=1.5 fires at 150–175. So "k=1.5 ripe + prompt" overstates — the **window guard sets the timing**, not k, and k=1.5 actually fires *later* (175) than k=2/3 (150) for 2 seeds. Choice of 1.5 over 2/3 is unsupported by data (report admits this in §5).

### Claim 4 — Off-qz safety (E-C): switch after equilibration AND lands at floor, all 3 seeds
**Verdict: OVER-CLAIMED (cherry-pick). Timing holds; "lands at floor" is FALSE for 1/3 seeds.**
- Per-seed off-qz (initmode, warm/qz-off): all three switch at index **150 > steps_to_floor** (132, 146, 91) → "switches after equilibration" holds, but timing is set by the 150 window guard, and seed1's equilibration (146) beat it by only **4 steps**.
- "Lands at floor across all 3 seeds" is **false**. Seed1 off-qz: b2_max = **0.02510**, b2_avg = 0.00505 (2.6× floor), **b2_success = False**, max_var_rel_err = 0.223 — the **worst b2_max and worst variance error in the entire dataset**. The report cites E-C as proof the self-calibrated switch "makes warm-start robust" and "off-qz unbiased," while hiding this 1-in-3 failure. This is the report's most serious oversell.

### Claim 5 — Ignored data
- **off-qz seed1** (above): worst run, omitted.
- **Acceptance out of band.** §1/§4 claim Phase-2 accept → target within ±3% / "bisection lands 0.68–0.72." Actual accept_final spans **0.654 → 0.738**: e.g. core T_ill sched-emaus/sw-paper = **0.7382** (+5.5%), T_iso sched-emaus/sw-paper = 0.7319; mscale T_ill M2048 seed0 = **0.6538** (−6.6%). Several runs violate the ±3% / 0.68–0.72 claim.
- **T_corr / T_curve** barely tested (N=1, M512, single seed) yet folded into "confirmed across all four beds."
- **k=1.0 seed2** never switches — a qualitative failure averaged away.
- **seconds = 0.0** on several rows and byte-identical k=2/k=3 rows suggest caching; not a science issue but worth noting the sweep above k=1.5 is degenerate (no independent information).
- NaNs: none — nan_free=True everywhere (clean).

### Claim 6 — Internal consistency
Multiple quoted numbers don't match the CSVs/plots: b²_max "3–4×" (actual up to 10.9× / plot ~6×); accept "0.68–0.72" (actual 0.654–0.738); steps_to_floor "76–125 within noise" (actual 68–125, cleanly clustered per the cited plot); "switch@300" for k=1.0 (actual 175/325/never).

## Single most important oversell
**Off-qz (E-C) robustness.** It is the headline empirical justification for the `self_calibrated` switch-mode default, yet 1 of 3 seeds (seed1) fails badly — b2_max 0.0251, b2_success False, worst variance error in the dataset — equilibrating only 4 steps before the window-guard-driven switch. The report presents off-qz as "unbiased / robust" and omits this. Secondary: the schedC "no efficiency separation / within noise" claim is contradicted by the report's own cost-to-equilibrate figure.

## Bottom line
Core qualitative conclusions (x_i² switch is the load-bearing fix; warm easy beds track the 1/M floor; no plateau out to M=8192) are directionally credible. But the report **oversells certainty and robustness**: off-qz is not uniformly safe (1/3 failure), "unbiased" is a 3-seed/2-bed/M≤8192 trend not a proof, schedule/C *do* separate on equilibration cost, and several quoted ranges (b²_max, acceptance, steps_to_floor) are inconsistent with the artifacts. Claims 1, 2(efficiency), and 4 need to be walked back or backed by more seeds before the lensing handoff.
