# Lab Notebook — Lensed point-source SBC campaign

SBC validation of the point-source observables pipeline (positions + flux + time-delay)
on 100 lenstronomy-simulated EPL+shear quads: `experiments/hundred_point_sources/campaign.yaml`,
plugin `simtests/experiments/lenstronomy_point_source.py`,
results `simtests_results/hundred_point_sources_v1/`.

**Last updated:** 2026-07-28

---

## Current state

As of 2026-07-24 night: **all 100 systems converged** (94 campaign + 6 intensive
reruns with the source-plane anchor and the fixed MAP), and the quick-view SBC over
100/100 passes 0/13 Holm rejections (UNCERTIFIED — mixed budgets/likelihood
variants). Two root causes found and fixed this cycle: the MAP off-by-one pairing
bug (C-6; PR seanxuseanxu/gigalens#80) and the saturation-shelf gradient flat-spot
(P-4 source-plane anchor, local only). sys_58's wrong-mode capture traced to the MAP
bug; PT-MCLMC with carousel defaults failed on it (negative result). Remaining for
certification: campaign v2 (anchor on, uniform budget, fresh SBC) → C-5 scope
upgrade; watch item O-5 (src center_y). Artifacts:
`simtests_results/hundred_point_sources_v1/{aggregate,diagnostics}/`.

2026-07-24 update: campaign v2 is deliberately ON HOLD — Linus is updating the
priors to be more observation-aligned, and the campaign will move to
GPU-batched inference first (phase A done; see `batched-point-source.md`).

2026-07-25 update: v2 ran as a THREE-ARM design (quad / triple / double; P-6),
priors set by Linus (H0 Uniform(20, 100), frac_flux 0.005), batched pipeline on
4x 2080 Ti (~2.6 h for 300 systems at 3x budget). **Verdicts: quad FULL PASS
(0/13 Holm, 1% attrition, O-5 dead) — calibration claim proposed; double 1/13
(gamma) rejected — SELECTION effect (truth-tilt vs untruncated prior, no
non-detection likelihood term), not pipeline — **VERIFIED by the P-7 filtered
re-rank (gamma p 0.0024 -> 0.56, 0/13 Holm after filtering; ~30% of double
posterior mass lies outside the N=2 region)**; triple TAINTED (28% attrition,
merging-cusp-pair class).** simtests_results now lives on scratch (home has a
~20 GB quota), symlinked from the old path. Follow-ups: any2plus arm /
non-detection term / triple-budget rerun. See the 2026-07-25 night log entry.

2026-07-27 update: the non-detection term now has a VALIDATED operator — the
differentiable smoothed image count N_eff (topology-free ray-shooting
integral; P-8): 97-98% agreement with corrected lenstronomy labels on both
double and naked-cusp draws, residuals understood (flux-threshold band +
deliberate caustic blur). Side discovery: the campaign's lenstronomy selection
operator (search_window=6) misses outer images of big-theta_E lenses and
merging-pair members near cusps — P-7 unaffected (same operator on both
sides), but future generators should use window >= 12. Next: wire N_eff into
the likelihood as an opt-in annealed term and rerun the double arm.

---

## Claims register

### C-1 — Gate-failing systems split into two classes: borderline mixers vs MAP-initialization failures
- **Status:** `proposed (UNCERTIFIED)`
- **Claim:** Among the 85 finished runs, all convergence-gate failures with max R̂ > 1.5
  (sys_58, sys_68, sys_76) share a single upstream cause: **the MAP stage never found the
  high-likelihood basin** (MAP best reduced χ² = 22.6 / 26.6 / 28.6 vs campaign median
  0.26); SVI and MCLMC inherited a bad start and had to do the basin-finding themselves.
  Failures with R̂ ∈ (1.05, 1.1) (sys_66, sys_87) have healthy MAP (red χ² 0.23 / 0.78)
  and are plain not-long-enough chains.
- **Evidence / artifact:** `diagnostics/flagged_systems_report.json` (per-chain ll +
  solver health), MAP manifests (`runs/*/default/map/manifest.json`), correlation table
  in the 2026-07-24 log entry below. Traces/corners: `diagnostics/sys_*_{trace,corner,chainhist,map}.png`.
- **Doubt report:** (i) `chisq_hist` was initially misread as raw χ² — it is REDUCED
  (verified against sys_66: MAP 0.22 matches an overfit 15-dof/12-param system; and
  against direct re-evaluation of z_best). (ii) Correlation ≠ cause: MAP failure and
  sampling failure could share a hidden third cause (hard lensing configuration). Partial
  control: sys_68/76 chains DID find the good basin (red χ² ≈ 0.9) once past init, so the
  likelihood surface is samplable when entered — the missing ingredient was the start
  point. (iii) n is small (3 catastrophic failures). Pre-registered test: sys_91 (see
  checkpoint P-1).
- **Proposed by / on:** Claude (agent) · 2026-07-24 · **Grader:** _pending_

### C-2 — sys_76: two chains frozen on a "phantom shelf" created by the saturated honesty charge; MAP initialized them there
- **Status:** `proposed (UNCERTIFIED)`
- **Claim:** sys_76's R̂ = 2.06 and 25% unconverged solves come from chains 1 and 4
  being frozen for all 10k draws in a region where **all 4 images are unreproducible**
  (solver src residual > 1e-4″ on 100% of their draws; position χ² ≈ 421 ≈ 4 images ×
  the ~100-unit saturated honesty cap; ll ≈ −170 vs +47 in the good basin). The shelf is
  a *designed* feature of the saturated charge (bounded height AND bounded stiffness ⇒
  locally flat), and the init pipeline delivered chains onto it: MAP's returned z_best
  has an unconverged solve (src_res 3.5e-2″), and 95.9% of draws from the SVI surrogate
  built around it are unconverged — i.e. ~8/8 chains started on the shelf and 6 escaped
  during sampling. Barrier from shelf to good basin along the straight z-path: Δll ≈ 341
  (`diagnostics/sys_76_scan.png`) — unhoppable for MCLMC.
- **Evidence / artifact:** `diagnostics/sys_76_{trace,scan}.png`,
  `diagnostics/barrier_scan.json`, per-chain table in `flagged_systems_report.json`,
  MAP/SVI re-evaluation in the 2026-07-24 log entry.
- **Doubt report:** (i) "Frozen" could be step-size collapse rather than a flat shelf —
  not distinguished; per-chain MCLMC step sizes were not persisted (open question O-1).
  Either way the chains start on the shelf and never leave it. (ii) The 6 escaped chains
  show the shelf is not inescapable in tuning/burn-in; freezing may depend on where on
  the shelf a chain lands. (iii) frac_unconverged 25% ≈ 2/8 chains exactly, consistent
  with the two frozen chains being the sole contributors.
- **Proposed by / on:** Claude (agent) · 2026-07-24 · **Grader:** _pending_

### C-3 — sys_58: posterior missed a higher-density basin containing the truth (bimodal likelihood; all init in the wrong mode)
- **Status:** `proposed (UNCERTIFIED)`
- **Claim:** All 8 chains of sys_58 sampled one basin (red χ² ≈ 3.7–5.5, solver fully
  converged) while the truth sits in a second basin with **higher** likelihood
  (ll 47.1 vs best sampled draw 43.0; red χ² 1.11 vs 1.66) and comparable z-space prior
  (−2.7 vs −1.6). Zero of 80k draws entered it; the truth's log-likelihood exceeds every
  posterior draw's (loglik_rank 512/512). A Δll ≈ 11–15 dip separates the best sampled
  draw from the truth along the straight z-path with the solver converged throughout
  (`sys_58_scan.png`) — a genuine second mode, not a solver artifact. The misfit the
  sampled mode pays is concentrated in the time-delay channel (χ²_td 12.4 vs 2.8 at
  truth), and it drags H0: posterior ≈ 65.5 ± few vs truth 76.5. MAP (best red χ² 22.6)
  and SVI never saw the truth mode either.
- **Evidence / artifact:** `diagnostics/sys_58_{trace,corner,scan,chainhist}.png`,
  `diagnostics/barrier_scan.json`, channel attribution in the 2026-07-24 log entry.
- **Doubt report:** (i) A calibrated posterior may legitimately put little mass on the
  truth mode if its VOLUME is small — peak density (e³ higher at truth) is not mass.
  <1/80000 mass would require the truth basin to be ~e⁻¹⁴ smaller in volume; not
  impossible for the stiff td channel, and NOT ruled out here. Decisive test proposed as
  checkpoint P-2 (truth-anchored chain). (ii) The straight z-line dip could overstate
  the barrier (curved valleys); it cannot understate it to zero, and chains show no hops
  in 10k draws. (iii) Under exact calibration a loglik rank of 512/512 occurs with
  p = 1/513 per system; one among ~84 is unremarkable ALONE — this claim rests on the
  scan + per-chain evidence, not the rank.
- **Proposed by / on:** Claude (agent) · 2026-07-24 · **Grader:** _pending_

### C-4 — sys_68: slow mixing on a ridge against the γ = 3 prior boundary; posterior covers truth; benign given longer chains
- **Status:** `proposed (UNCERTIFIED)`
- **Claim:** sys_68 (truth γ = 2.62, steep EPL) mixes slowly along a
  γ–e2–H0 ridge with chains repeatedly pinned at the TruncNormal γ ≤ 3 ceiling
  (trace: `sys_68_trace.png`); solver 100% converged, all chains in one basin around the
  truth, truth ll (35.5) INSIDE the sampled ll range. MAP also failed here (red χ² 26.6,
  still descending at step 1500 of 1500), but chains recovered; the residual failure is
  mixing speed, not location. Longer chains (or more of them) should pass the gates
  without bias.
- **Evidence / artifact:** `diagnostics/sys_68_{trace,corner,map}.png`,
  `flagged_systems_report.json`.
- **Doubt report:** chain 7 wanders to e2 ≈ +0.06 while others sit ≈ −0.05 — could be a
  second shallow mode rather than a ridge; per-chain ll ranges overlap fully, so if it is
  a mode it is likelihood-equivalent. Not resolved; would show up in a longer run's R̂.
- **Proposed by / on:** Claude (agent) · 2026-07-24 · **Grader:** _pending_

### C-6 — MAP off-by-one pairing bug: returned z_best was one optimizer step AHEAD of its recorded score (FIXED)
- **Status:** `proposed (UNCERTIFIED)` — fix merged locally in ~/gigalens inference.py
- **Claim:** ``ModellingSequence.MAP``'s scan recorded the post-update particle with the
  pre-update (lp, chi2): ``z_best`` returned by MAPStage was NOT the particle that
  achieved ``best_lp``/``best_chisq`` but that particle one adam step later. Smooth
  landscapes hide it (sys_45: drift 0.004 lp); the stiff point-source likelihood does
  not: sys_58 campaign z_best was 11 lp units off its record; anchored sys_76's
  recorded best was lp +6.9 / red chi2 5.96 (an excellent fit — the anchor's MAP run
  actually SUCCEEDED) while the returned particle scored lp −277 / red chi2 43.8,
  catapulted across a caustic wall by the final step. This resolves O-6: the
  z_best-vs-best_chisq gap was the bug, not (only) the lp-vs-chi2 criterion.
- **Fix:** record before update (trajectory unchanged; records now self-consistent).
  Verified: fresh MAP on anchored sys_76 gives recorded == re-evaluated triple to
  float noise (|d_lp| 2e-6 on lp ≈ −477); gigalens test_batch_contract 7/7,
  point-source suites 37/37.
- **Evidence:** numeric audit in the 2026-07-24 log entry; code comment at the fix
  site cites the sys_76 measurement.
- **Doubt report:** the audit re-evaluated stored z_best against stored (lp, chi2)
  under an identical float64 model — mismatch can only come from record inconsistency
  or code-path float noise; the observed 11–284-unit gaps are 7 orders above the
  measured 2e-6 noise floor, and the code reading (update before record) predicts
  exactly this signature. Affects every past MAPStage z_best in this repo (mildly on
  smooth systems); campaign v1's MAP artifacts remain as-run history.
- **Proposed by / on:** Claude (agent) · 2026-07-24 · **Grader:** _pending_

### C-5 — The point-source pipeline is calibrated (SBC uniform) over the 94% of systems passing convergence gates
- **Status:** `proposed (UNCERTIFIED)`
- **Claim:** With 100/100 systems run, 94 pass the gates (R̂ ≤ 1.05, ESS ≥ 100,
  solver-unconverged ≤ 5%). Over those 94, ALL 13 SBC uniformity tests pass
  (12 parameter marginals + the joint log-likelihood rank; Holm-corrected KS at 0.05).
  Weakest: source center_y raw p = 0.013 (mean PIT 0.574, +2.5σ; its ECDF grazes the
  95% simultaneous band near u ≈ 0.55) and lens center_x raw p = 0.067 — neither
  rejected. H0 is comfortably uniform (p = 0.74) — the time-delay channel is calibrated
  where chains converge.
- **Criterion (pre-registered in campaign.yaml):** per-parameter exact-PIT KS with Holm
  at 0.05 over gate-passing runs; attrition reported, not hidden.
- **Evidence / artifact:** `aggregate/sbc_report.json`, `aggregate/sbc_ecdf_default.png`
  (inspected: 12/13 curves well inside the band; src center_y grazes it),
  `aggregate/sbc_hist_default.png`.
- **Doubt report:** (i) **Attrition is 6%** (6 systems), just above the yaml's ~5%
  caution line, and it is NOT parameter-uniform: excluded truths lean steep-γ
  (γ z-scores +2.5/+2.1/+1.6 for sys_58/sys_87/sys_68; 3 of 6 in the top-γ tail)
  — so the calibration claim's scope is effectively "systems whose γ the pipeline can
  converge on", and the γ panel's pass does not cover the excluded steep tail.
  (ii) One-of-13 panels grazing a 95% simultaneous band has null probability ≈ 50%
  (1 − 0.95¹³), so src center_y is statistically legal — but it was ALSO the
  worst-trending parameter in mid-campaign checks (mean PIT 0.617 at n = 40), so it is
  a watch item (O-5), not noise to forget. (iii) Ranks use deterministic hash-seeded
  PIT jitter; thinning to L = 1024 leaves residual autocorrelation for the ESS ≈ 100–300
  runs, which inflates rank noise at the extremes — this pushes TOWARD false alarms,
  not false passes. (iv) The excluded 6% are exactly the systems diagnosed in C-1..C-4;
  their exclusion is diagnostics-based (R̂/ESS/solver), never truth-based, so no
  selection-on-truth enters the retained set beyond the γ-hardness correlation in (i).
- **Proposed by / on:** Claude (agent) · 2026-07-24 · **Grader:** _pending_

---

## Design checkpoints (criteria awaiting approval)

- **P-1 (pre-registered PREDICTION, no run needed).** Hypothesis: C-1 (MAP-init failure
  causes catastrophic R̂). **sys_91's** MAP manifest (written before its sampling
  finished; best red χ² = 24.8, in the failure class) predicts sys_91 will fail the
  convergence gates: **max R̂ > 1.5** (order of magnitude: like sys_58/68/76, i.e.
  R̂ ∈ 1.5–3, min ESS ~10–30). Falsifier: sys_91 completes with max R̂ < 1.1. Logged
  2026-07-24 ~13:00 PT while sys_91 was still sampling on n0028.es1. Blind spot: a
  single system; a pass would falsify the *determinism* of the link, not the trend.
  Cost: none. **Status: RESOLVED 2026-07-24 ~13:20 PT — observed max R̂ = 1.229,
  min ESS = 25.3, frac_unconverged 0. Direction CORRECT (fails both gates: R̂ > 1.05,
  ESS < 100; ESS landed inside the predicted 10–30), magnitude MISSED (R̂ 1.23, below
  the predicted 1.5–3). Honest reading: MAP-failure systems reliably fail convergence
  gates, but MAP red χ² > 20 does not by itself set the R̂ scale — sys_91's chains
  evidently recovered the basin better than sys_58/68/76's did. C-1 survives as a
  class link (MAP failure ⇒ gate failure), weakened as a severity predictor.**
- **P-2 — truth-anchored chain for sys_58** (decides "missed mass" vs "legitimately
  negligible mode"). Run 4 MCLMC chains initialized AT z_truth (plus the standard 8-chain
  run for comparison), same budgets. Hypothesis: the truth mode carries non-negligible
  posterior mass that the standard init missed. Prediction: truth-anchored chains STAY in
  the truth basin (their draws' ll stays within ~[35, 55], red χ² ≈ 1.1 ± 0.5) rather
  than migrating to the sampled basin within 10k draws. Falsifier: chains drain out of
  the truth basin (final draws indistinguishable from the standard run's basin, ll < 30) —
  then the mode is a thin spike, the standard posterior is approximately right, and
  sys_58's ranks are legal tail events. Metric: fraction of post-burn-in draws with
  ll > 40 (threshold = midpoint of the two basins' observed ll ranges — separations are
  ~10 units, well beyond in-basin spread ~5). Blind spot: cannot measure the mass RATIO,
  only occupancy/stability. Expected plot: ll trace per chain, bimodal or single-basin.
  Cost: ~5 min single shard (one system, sampling only). **Status: APPROVED by user and
  RESOLVED 2026-07-24 — prediction CONFIRMED. 4 chains, qz = MVN(z_truth, campaign SVI
  tril), same MCLMC budgets, seed 1: pooled frac(ll > 40) = 99.4% over 10k post-burn-in
  draws; per-chain ll_med 48.7–48.8; validity guard passed (chains genuinely mix:
  z-step RMS ≈ 0.07, in-chain ll spread 17–20; trace plot flat-stationary around
  ll ≈ 49, no drain, brief dips to ~34 that recover). The truth mode is stable and its
  TYPICAL draws sit ~25 log-like units above the wrong basin's typical draws
  (ll_med 48.8 vs ~24) — so the campaign posterior for sys_58 sampled a SUBDOMINANT
  mode and missed what is almost certainly the dominant mass. Mass ratio still not
  directly measured (pre-registered blind spot; PTMCLMCStage could measure it).
  Artifacts: `diagnostics/sys_58_truth_anchored/{result.json,ll_traces.png,mclmc/}`.
  → C-3 upgraded from "missed a basin" to "missed the dominant mode"; sys_58's
  pinned SBC ranks are a real sampler-robustness failure, not a legal tail event.**
- **P-4 — source-plane anchor term in the point-source likelihood** (user-approved
  2026-07-24; implementation + sys_76 rerun authorized). **Hypothesis:** the sys_76-class
  phantom shelf exists because the only failure-sensing term (the saturated honesty
  charge) is bounded, and any bounded charge must go flat — MAP/chains on the shelf get
  no gradient. Adding a per-image source-plane anchor
  ``chi2_anchor = sum_i |beta(theta_hat_i) - beta_s|^2 / sigma_beta^2`` (sigma_beta =
  sigma_ast; zero at convergence, gradient GROWS with failure distance, no A^-1 so no
  stiffness) restores gradient flow without touching the converged-draw likelihood
  (contribution at solver tolerance ~6e-4 chi2/image). **Prediction:** sys_76 rerun with
  the anchor: MAP returns a point with a CONVERGED solve (src_res < 1e-4") and red chi2
  < 3 (vs 28.6 / 3.5e-2" unconverged today); all 8 chains land in the good basin
  (0 frozen; max R-hat < 1.1 vs 2.06); healthy-system spot-checks (2 systems) shift
  posterior means by < 0.05 sigma per parameter and their SBC-relevant marginals are
  unchanged. **Falsifiers:** (i) MAP still returns an unconverged-solve point → the
  intervening flux-channel ridge blocks gradient escape; MAP-side converged-solve
  selection (P-3a) becomes necessary. (ii) healthy-system posteriors shift ≥ 0.05 sigma
  → the anchor is not inert; sigma_beta choice must be re-derived. **Threshold
  derivation:** 0.05 sigma = well below the ~1/sqrt(2·ESS~1000) ≈ 0.02 MC error floor
  times a safety factor; red chi2 < 3 = the campaign's pathological/imperfect boundary;
  1e-4" = the solver-convergence tolerance already used by ps_solver_health. **Blind
  spot:** two spot-check systems cannot detect a subtle global calibration shift — a
  full campaign v2 SBC is required before certifying the anchored likelihood. **Expected
  plot:** sys_76 ll traces: 8/8 chains at ll ≈ +47, no flat-liners. **Cost:** code +
  tests, then 3 pipeline runs (~15 min on the free interactive node).
  **Status: RESOLVED 2026-07-24 (implementation merged locally in ~/gigalens;
  37/37 tests pass, 5 new). Scored against the pre-registered predictions:**
  - **Shelf mechanism CONFIRMED at the unit level:** deep-shelf scan (sigma 4 mas,
    source 0.5" outside caustic): unanchored |dchi2/dbeta_y| collapses 6.6e3 → 24
    (chi2 flat to ~50 units over 0.2") while anchored grows 2.2e4 → 1.9e5 — an
    ~8000x tilt. Also measured: the shelf needs DEEP saturation (|s| >> cap); at
    sigma = 0.04" the charge still has slope and the anchor adds little
    (complementary-regimes picture confirmed). Pinned in
    `test_src_anchor_tilts_the_saturation_shelf`.
  - **sys_76 rerun (anchor = 0.004): the pathology is GONE.** MAP's returned point
    has a CONVERGED solve (src_res 2.8e-9" vs 3.5e-2"); all 8 chains in the good
    basin (ll_med ≈ 48, red chi2 ≈ 0.9), 0 frozen, 0.0% unconverged draws
    (vs 25%). Posterior center_y moved +0.93 sigma toward the truth — the frozen
    chains' contamination removed. Magnitude misses, scored honestly: MAP best red
    chi2 5.96 (predicted < 3; vs 28.6 before) and z_best red chi2 43.8 — the
    best-lp z_best selection still returns a poor-fit (though now converged-solve)
    point (→ O-6); and max R-hat 1.302 (predicted < 1.1; vs 2.057) — the
    catastrophic mode-trap is gone, the residual is ordinary slow mixing that
    longer chains should clear. Direction confirmed everywhere; both magnitude
    targets missed.
  - **Inertness falsifier (ii) FIRED at the letter on sys_45** (max shift 0.0599
    sigma, H0; sys_00: 0.0440 sigma — under threshold). Post-hoc noise analysis
    (recorded, not silently absorbed): normalized by ESS-based two-run errors the
    worst shift is z = 3.3, BUT the same statistic on a WITHIN-run chain split of
    the unmodified campaign run already reaches z = 2.3 — the ESS floor
    understates chain-level variability, so the 0.05-sigma threshold was
    under-derived (producer error in the checkpoint, not evidence of
    non-inertness). Structural bound: both spot-check runs have 100% converged
    solves, where the anchor changes log-like by < 1e-2 units — a real 0.05-sigma
    posterior shift cannot arise from that; the observed shift must be
    trajectory-level MC noise (the tiny anchor term chaotically decorrelates the
    two runs, making them independent realizations). **UNCERTIFIED — the decisive
    inertness test is the campaign-v2 SBC (all 100 systems, anchor on), not
    two spot checks.** Artifacts: `diagnostics/{sys_76,sys_45,sys_00}_anchored/p4_report.json`.
- **P-5 — intensive reruns of the 6 gate-failing systems + quick-view SBC over all
  100** (user-requested 2026-07-24). Config: anchor 0.004, fixed-MAP code (C-6),
  MCLMC 8 chains x 15000 burn-in / 30000 results (3x campaign), MAP/SVI budgets
  unchanged; sys_58 gets an additional PT-MCLMC arm (PTMCLMCStage, beta_min 0.36,
  defaults, seeded from campaign z_best) because longer plain chains cannot hop its
  Δll ≈ 11–15 mode barrier. **Predictions:** sys_66/87/91 converge (R̂ < 1.05,
  ESS > 100) with high confidence; sys_76 converges (was 1.30 at 1x with a bad init;
  now 3x + the true lp +6.9 MAP point); sys_68 likely but least certain (z-space R̂
  2.9 at 1x). sys_58 plain arm: either stays unconverged OR "converges" into the
  subdominant mode with pinned ranks (loglik rank ≈ L) — flagged either way, NOT
  quick-view-eligible without the mode check; PT arm: cold chain reaches the dominant
  mode (frac ll > 40 majority). **Falsifiers:** any of 66/87/91 still failing gates
  (would break the borderline-class diagnosis); PT cold chain never leaves the
  subdominant mode (would contradict P-2's mass reading). **Quick view:** merge the
  94 campaign rank sets with the 6 rerun rank sets (same thinning L ≈ 1024, same
  hashlib PIT jitter, Holm KS) — labeled UNCERTIFIED: mixed budgets and likelihood
  variants across systems; the calibration-grade answer remains campaign v2.
  Blind spot: reruns use the anchor while the 94 don't (identical at converged draws;
  that identity is exactly what v2 must certify). Cost: ~1 h on n0028.
  **Status: waves 1–2 DONE 2026-07-24 evening; scored:**
  - **sys_58 RESOLVED, beyond prediction:** with the C-6 fixed MAP (best red chi2
    16.0, converged solve) the plain 3x run landed ALL 8 chains in the DOMINANT
    (truth) mode — R̂ 1.004, ESS 1736, all chains ll_med ≈ 48.6, loglik_rank 119/512
    (was pinned 512/512), posterior moved 5.0 sigma in src center_x vs campaign.
    The campaign's subdominant-mode capture was downstream of the MAP pairing bug.
    Residual scope caveat: single-mode chains ignore the subdominant mode's mass —
    negligible per the ~25-unit typical-ll gap (P-2), not directly measured.
  - **PT arm FALSIFIER FIRED** (negative result, logged): PTMCLMCStage with
    carousel-validated defaults (beta_min 0.36, 8 walkers, 2000x10) ended in a
    region WORSE than both modes (cold-rung ll_med ≈ −8, 0% dominant-mode
    occupancy, R̂ 1.43, loglik_rank 512/512). The carousel knobs do not transfer to
    this 12-dim point-source posterior; not retried (superseded by the plain-run
    success). Not to be reused without a dedicated ladder/round tuning pass.
  - sys_66 (R̂ 1.005), sys_87 (1.022), sys_68 (1.044, ESS 168 — from z-space 2.9)
    PASS. sys_76 (1.100, ESS 176) and sys_91 (1.103, ESS 98.5) improved but
    marginally FAIL; v3 extensions (30k/60k) running. sys_66's rerun doubles as an
    inertness spot-check: max shift vs campaign 0.027 sigma (clean).
  - Fixed-MAP quality across reruns: every z_best now solver-converged and
    self-consistent; sys_76 MAP best red 5.96, sys_68 7.78 (vs 28.6/26.6 buggy).
  - **v3 extensions (30k/60k): sys_76 R̂ 1.049 / ESS 208, sys_91 R̂ 1.006 /
    ESS 1239 — ALL 100 SYSTEMS NOW PASS the strict z-space gates.**
  - **QUICK-VIEW SBC over 100/100 (UNCERTIFIED, mixed budgets + likelihood
    variants): 0/13 parameters Holm-rejected.** Worst: src center_y p = 0.0155
    (mean PIT 0.569 — the SAME signature as the 94-system aggregate, shared data,
    not independent evidence; O-5 stands), loglik p = 0.088, all others p > 0.10.
    ECDF plot inspected: src center_y touches the 95% simultaneous band at
    u ≈ 0.55, everything else comfortably inside. Artifacts:
    `diagnostics/quickview_sbc/{quickview_report.json,quickview_ecdf.png}`.
    Certification still requires campaign v2 (uniform likelihood + budget).
- **P-6 — campaign v2: tri-arm SBC (quad / triple / double) with the corrected
  loss, widened H0 prior, tightened flux noise** (designed 2026-07-25; dataset
  generation authorized by Linus; **RUNS awaiting his go — do not launch**).
  Configs: `experiments/hundred_point_sources/campaign_v2_{quad,triple,double}.yaml`
  (seeds 2/4/3; fresh truths, deliberately not shared with v1). Shared v2 changes:
  C-6 fixed MAP + P-4 anchor (src_anchor_sigma = sigma_ast = 0.004"), H0 prior
  Uniform(20, 100) (was 60–80), frac_flux 0.005 (was 0.05), MCLMC intensive budget
  15k/30k uniform (P-5 showed the borderline class needs it). Generator gained
  `multiplicity: triple` (n_img == 3; selection on the solver-found count, like the
  other modes). **Claim type:** distributional — SBC rank uniformity of the
  corrected pipeline, per arm. This run tests the calibration link (prior==truth
  draw ⇒ uniform ranks under the anchored likelihood at uniform budget); it does
  NOT test robustness to prior misspecification (untested) or real-data validity
  (out of scope by design). **Cause hypothesis:** the v1-era failures were (i) the
  MAP pairing bug (C-6, fixed) and (ii) the bounded-charge shelf (P-4, anchored);
  with both corrected and budget uniform, no miscalibration mechanism remains at
  any multiplicity — including doubles (prior-dominated posteriors are still
  calibrated posteriors) and triples (naked-cusp region; strongest selection).
  **Prediction:** per arm, 0/13 parameters Holm-rejected at alpha 0.05 (the
  quick-view already showed 0/13 on mixed budgets); gate attrition < 5%
  (P-5: 100/100 passed at intensive budget; doubles/triples have smoother
  landscapes than quads' fold caustics — expect attrition at or below the quad
  arm's); src center_y mean-PIT offset (O-5, 0.569 on shared v1 data) NOT
  reproduced on fresh truths if it was a fluctuation. **Falsifiers:** any
  Holm-adjusted KS p < 0.05 (per arm) — if it is src center_y again on
  independent data, O-5 is real structure, anchor/parameterization suspect;
  attrition >= 5% in an arm taints that arm's rank set (partial-failure mode:
  tightened flux at 0.5% sharpens the likelihood ~10x — if this manufactures a
  NEW hard class, it shows up as attrition, not silent rank distortion, because
  gates run first). **Threshold derivation:** Holm at alpha 0.05 over 13
  KS tests, n = 100 trials/arm, L = 1024 ranks — same machinery as the
  quick-view; 5% attrition = the yaml's sbc gate, above which the excluded-truth
  region visibly truncates the rank set. **Blind spots:** (i) marginal ranks
  miss joint miscalibration — loglik_rank covers the joint direction; (ii)
  multiplicity selection truncates the prior to the n-image region while the fit
  prior is untruncated (yaml note 3) — benign-by-conditioning argument, residual
  distortion possible, strongest for the triple arm (lowest acceptance);
  a rank failure CONSISTENT across arms would implicate the pipeline, one
  ISOLATED to triples would implicate selection; (iii) the n_img == 3 class
  mixes true naked cusps with solver-missed-image configs — irrelevant to SBC
  (selection is on the observed count) but recorded for physical interpretation.
  **Expected appearance:** all three arms' ECDF-difference plots inside the 95%
  simultaneous envelope; if falsified, the offending parameter's ECDF exits the
  band (v1 quick-view: src center_y touching at u ~ 0.55 is the shape to watch).
  **Cost:** generation ~5–20 CPU-min/arm (probe-measured acceptance 0.33 quad /
  0.10 triple / 0.40 double; probe artifact: job-scratch `mult_probe.log`,
  N = 3000, seed 99; realized: 0.323 / 0.088 / 0.417); runs (NOT authorized yet): batched GPU pipeline at 3x
  budget, ~1–2 h/arm on 4x 2080 Ti, vs ~60 CPU-h/arm solo — the batched path is
  certified (see `batched-point-source.md` phase B). **Status: RUNS APPROVED
  by Linus 2026-07-25 ("go ahead with v2 using the batched pipeline") after
  datasets were generated + verified. Prior sub-question resolved first: the
  honesty charge's plain-Newton scoring step (vs the solve's Levenberg
  schedule) confirmed deliberate and correct — the charge measures first-order
  distance-to-root; LM damping would under-report it worst near critical
  curves. Launched 2026-07-25 on n0027.es1 (4x 2080 Ti, salloc 24248279) via
  the phase-C batched campaign runner
  (`simtests/experiments/batched_campaign_run.py`): per-GPU 25-system slices,
  13-wide waves, arms sequential (quad -> double -> triple), seeds mirroring
  solo Pipeline semantics (campaign seed for every stage). Runner
  smoke-certified on CPU first: solo `run.py` RESUMES from batched artifacts
  (all three stages "loaded", 0.0 s wall) — hash-chain parity; aggregate /
  sbc_uniformity ingest the layout.**
- **P-3 — MAP robustness fix candidates** (from C-1/C-2; pick after P-1/P-2 grade).
  Candidates, each calibration-safe (init/optimizer choices don't alter the posterior):
  (a) select MAP output by best CHI² particle with a CONVERGED solve (src_res < 1e-4″)
  instead of best-lp-at-best-step — rejects phantom-shelf outputs outright;
  (b) multi-start MAP survival: keep top-K distinct basins (cluster particles), start
  chains from several basins (also addresses sys_58-class bimodality);
  (c) anneal channels: fit positions-only first, then +flux, then +td (positions-only
  landscape has no delay-driven secondary structure; cheap because the solve dominates).
  Each needs its own checkpoint with derived thresholds before running; the natural test
  campaign is a rerun of the 4 failed systems + sys_91 with the fix toggled.
  **Status: sketch only — not runnable as written.**
- **P-10 — double-arm rerun with the in-likelihood multiplicity constraint**
  (config `experiments/hundred_point_sources/campaign_v2_double_mc.yaml`; runs land in
  `runs/<sid>/mc1`, P-7's `runs/<sid>/default` untouched; same dataset, seed 3, no
  regeneration). **Claim type:** distributional (SBC calibration of the constrained
  posterior over 100 systems) plus a per-system basin claim (stochastic-estimator:
  chain occupancy of the count-2 region). This run tests ONE link of the chain: *the
  soft in-likelihood tilt reproduces the calibration that P-7's post-hoc hard count==2
  filtering achieved, while eliminating the zero-kept class*. Untested links left open:
  the soft-vs-hard selection mismatch at eps=0.1" (this run bounds it empirically, it
  does not derive it), and transfer to the triple arm. **Cause hypothesis:** the
  position likelihood is deliberately LOCAL (no global image search), so it carries no
  image-count information; 12 systems' MAP+posterior settled in phantom-image basins
  (P-7 kept fraction ~0). Adding the validated selection tilt −lam (N_eff − 2)^2
  (P-8 operator, P-9 capture check: all phantom modes within d ≤ 0.87" of the caustic,
  reachable from eps_init 0.3") supplies exactly the missing global information.
  **Implementation under test:** gigalens `multiplicity_constraint` (mu_min 0.1,
  eps 0.1, lam 10, grid 384^2, window 4·theta_E, fp32 tiled+remat quadrature) with
  staging MAP(off) → eps-anneal refine (0.3"→0.15"→0.1", lam ramped, top-64 particles,
  300 steps/rung) → SVI(off, preconditioner) → MCLMC(on, fixed target). 40 regression
  tests green; solo==batched exact; term-off bit-identical to v2 code; penalty matches
  the standalone-validated operator to 1.7e-5. Pilot (job 24291987, sys_05/sys_63/
  sys_80, 1/5 MCLMC budget): frac(N_eff≈2) = 1.000 / 0.947 / 1.000 vs P-7 kept
  0.000 / 0.000 / ~1; healthy control N_eff median 1.9997 (term inert). **Predictions
  (direction + magnitude):** (1) all 12 zero-kept systems recover frac(N_eff≈2) > 0.9
  (pilot worst 0.947); (2) SBC over 100 systems, UNFILTERED draws: ≤ 2/13 Holm
  rejections (P-7's filtered-run level), and specifically the gamma rank skew that
  P-7 saw pre-filtering disappears; (3) healthy systems statistically unchanged vs
  P-7 filtered: median |Δ posterior mean|/posterior sd < 0.3 per parameter.
  **Falsifiers:** (a) any zero-kept system with frac(N_eff≈2) < 0.5 → the anneal
  fails to escape its phantom basin (hypothesis wrong or ladder inadequate);
  (b) > 2/13 Holm rejections concentrated in gamma/theta_E → the soft-vs-hard
  selection mismatch at eps 0.1" biases the posterior (remedy pre-declared: tighten
  eps to 0.05, grid 768, rerun — do NOT reinterpret post hoc); (c) healthy-system
  shifts ≥ 0.3 sd → the term is not inert where it must be (lam or quadrature
  artifact). **Threshold derivation:** 0.9/0.5 fracs bracket the pilot's observed
  bimodality — recovered systems sit at 0.95–1.00, phantom-basin systems at ~0.0;
  0.5 is an order-of-magnitude separator, not a tuned number. 2/13 Holm = the
  measured P-7 filtered outcome (its run-to-run variance envelope). 0.3 sd = the
  z-score scatter observed between P-5 intensive reruns of the same systems;
  below it a shift is indistinguishable from rerun noise. **Blind spots:** the
  frac(N_eff≈2) metric shares its operator with the constraint — a common quadrature
  artifact would self-confirm; mitigation IN the analysis plan: lenstronomy
  LensEquationSolver cross-check (window 12, min_distance ladder) on 500 thinned
  final draws for each of the 12 recovered systems. SBC ranks pool over the prior —
  a bias confined to the small caustic-proximal sub-population could hide inside
  uniform pooled ranks. **Expected plots:** gamma SBC rank histogram flat without
  any post-hoc filtering (P-7's unfiltered version was visibly skewed); per-system
  frac(N_eff≈2) bar chart with the zero-kept class jumping 0 → ≈1. If (b) fires:
  U-shape/slope in gamma ranks. **Cost (measured, not guessed):** 158 ms/step at
  24 lanes/384^2 on one 2080 Ti → 25-system wave (200 lanes) ≈ 1.3 s/step × 45k
  MCLMC steps ≈ 16 h + ~2 h MAP/anneal/SVI/metrics; plan: 4 offset slices × 25
  systems on 4× es0 2080 Ti, wall ≈ 18 h (overnight+). Seeds: campaign seed 3
  throughout (runner passes it to map/svi/mclmc). Code: gigalens linusu-dev-merge +
  multiplicity changes, GIGALens-Code batched pipeline changes — both working-tree,
  to be committed before launch so run.json provenance hashes are reproducible.
  **Status: APPROVED by user 2026-07-28 ("Yes, can you commit and go ahead with
  the P-10 launch? I'd like it to run in under 12 hours, so feel free to submit
  2 jobs to make that happen."). Launch plan adjusted for the <12 h wall-time
  request: 8 offset slices × 12–13 systems on 8× 2080 Ti (2 full es0 nodes,
  4 GPUs each — 2 Slurm jobs, 4 runner processes per job pinned via
  CUDA_VISIBLE_DEVICES), ~104-lane MCLMC waves ≈ 0.7 s/step × 45k ≈ 9 h +
  ~1 h other stages ≈ 10 h expected wall. Same target, seeds, and budgets as
  registered — only the slicing changed. Code committed before launch:
  gigalens d321d3c (linusu-dev-merge), GIGALens-Code e82e7e8 (main).**

  **Launch 1 (job array 24307807) FAILED ~15 min in — all 8 slices, GPU OOM in
  the anneal phase, no posterior draws taken.** The anneal descended all 64
  particles x 13 systems = 832 lanes through the multiplicity-term gradient at
  the final 384^2 rung in one op (9.4 GiB working set; the 3-system pilot's 192
  lanes fit, which is why this wasn't caught). Fix: `batched_map_anneal` now
  Adam-descends particles in sequential blocks of `mc_anneal_block` (default
  16) per system via `lax.map`, and scores the refined pool the same way —
  peak memory / 4 at identical total FLOPs. Semantics note: global-norm
  clipping now acts per 16-particle block instead of across all 64; this only
  touches the optimization stage (basin selection), not the measured MCLMC
  target, so the pre-registered predictions stand unchanged. No run dirs were
  written (run.json only lands on success), so launch 2 is a clean rerun of
  the same plan.

  **RESOLVED 2026-07-31 — see the log entry of that date. Summary: predictions
  2 and 3 PASS; prediction 1 passes for 11/12 systems; falsifier (a) fired for
  sys_57 with a diagnosed operator-accuracy cause (coarse N_eff undercounts a
  genuine 3rd image), for which the pre-declared eps 0.05 / grid 768 remedy
  applies. Plus a new dataset defect: 5 systems mislabeled as doubles by the
  window-6 generator count.**

- **P-11 — fine-operator remedy rerun (eps 0.05" / grid 768) on the 7 affected
  systems + unpenalized loglik re-rank + window-12 truth recount**
  (config `experiments/hundred_point_sources/campaign_v2_double_mc_fine.yaml`,
  sweep `{mc: 2}` → runs land in `runs/<sid>/mc2`; job
  `experiments/hundred_point_sources/p11_remedy.sbatch`; analysis
  `experiments/hundred_point_sources/p11_analysis.py`). **Claim type:**
  per-system stochastic-estimator behaviour (basin occupancy of the penalized
  MCLMC posterior under the corrected operator) plus one distributional claim
  (unpenalized loglik PIT uniformity over the existing 100 mc1 runs) and two
  deterministic side-measurements (operator-convergence rung; lenstronomy truth
  recount at window 12). This tests the REMEDY link of P-10's falsifier-(a)
  chain; untested links left open: population-level SBC calibration under the
  fine config (needs a full 100-system arm) and transfer to triples.

  **Pre-run truth-level measurement (2026-08-03, login CPU, pinned code) that
  SPLITS the hypothesis.** N_eff at each system's truth under three operator
  rungs (coarse 384/eps 0.1 = in-likelihood mc1; fine 768/eps 0.05 = remedy;
  2x 1536/eps 0.025), lenstronomy truth count = 2 for all seven:

  | sid | coarse(t) | fine(t) | 2x(t) | pen_fine(t) | class |
  |---|---|---|---|---|---|
  | sys_35 | 2.292 | 2.203 | 2.013 | −0.41 | A: resolution artifact |
  | sys_67 | 2.040 | 2.037 | 2.000 | −0.01 | A |
  | sys_69 | 2.526 | 2.174 | 2.006 | −0.30 | A |
  | sys_57 | 2.176 | 2.338 | 2.334 | −1.14 | B: intrinsic excess (converged) |
  | sys_86 | 2.398 | 2.661 | 2.799 | −4.38 | B (rising with resolution) |
  | sys_99 | 2.720 | 2.876 | 2.929 | −7.68 | B |
  | sys_38 | 2.965 | 2.993 | 2.923 | −9.32 | B |

  **Refined cause hypothesis (two mechanisms):** (H-a) coarse-quadrature
  undercount — a pure resolution artifact, fully removed at 768/eps 0.05
  (Class A: truth-level operator converges to 2.0); (H-b) intrinsic
  near-caustic smoothed-count excess — the Gaussian×|det A|×w quadrature
  assigns fractional image weight to near-critical source-plane area even when
  only 2 discrete images exist, at ANY resolution (Class B: truth-level
  N_eff 2.33–2.99 across rungs). For sys_57 the mc1 failure is still H-a at
  the POSTERIOR level: its phantom-basin draws read fine N_eff ≈ 2.86
  (pen ≈ −7.4) vs ≈ −1.1 in the truth basin, a ≈ +6 log-like differential
  favouring truth that the coarse operator erased (coarse read ≈ 2.04
  everywhere, penalty engaged nowhere). For sys_38/86/99 the truth itself pays
  pen_fine −4.4 to −9.3: the constraint disfavours these truths BY DESIGN at
  any resolution, so "recovery" is not predicted — they measure the
  displacement cost of the constraint on near-caustic doubles, the number that
  decides whether this constraint can carry to triples.

  **Predictions (direction + magnitude):**
  (1) Class A (sys_35/67/69): lenstronomy frac(count==2) ≥ 0.9 AND fine-audit
      frac(|N_eff−2| < 0.35) ≥ 0.9 (mc1 fine-audit values 0.21/0.83/0.67 →
      the recovered band 0.94–1.00). Note sys_67 starts at 0.83, so its
      confirmatory weight is low; sys_35/69 carry the test.
  (2) sys_57: lenstronomy frac(count==2) ≥ 0.9 (mc1: 0.000, hist {3: 500});
      fine-audit MEDIAN N_eff in [2.0, 2.45] (mc1: 2.86). frac(|N_eff−2|<0.35)
      > 0.9 is deliberately NOT predicted — truth reads 2.34, at the tolerance
      edge (declared now, not reinterpreted later).
  (3) sys_38/86/99: displacement, not recovery — per-system UNPENALIZED
      loglik PIT ≤ 0.1 persists in mc2 (truth in the disfavoured tail), and
      the population scan over all 100 predicts N(pen_fine(truth) < −0.5) ≈
      7–12: the 4 Class-B systems + the 5 known mislabeled multi-image systems
      + at most ~3 undiscovered.
  (4) Unpenalized loglik re-rank of all 100 mc1 posteriors: PIT uniform
      (KS p > 0.05) and Spearman(pen_coarse(truth), PIT) collapses from the
      measured +0.512 (p = 5e−8) to |ρ| ≤ 0.2.
  (5) Truth recount (all 100, window 12, ladder {0.05, 0.025}): confirms the
      5 known mislabels (sys_02/26/64/81 = 3 images, sys_19 = 4); ≤ 5
      additional among the 86 not yet recounted.
  (6) Operator convergence: fine vs 2x on 500 mc2 sys_57 draws agree to
      frac(|Δ| < 0.05) ≥ 0.9.

  **Falsifiers:** (a) any Class-A system with lenstronomy frac(count==2) < 0.9
  → resolution was not the mechanism even where the operator converges at
  truth; anneal/staging or lam inadequacy — remedy path invalid. (b) sys_57
  lenstronomy frac(count==2) < 0.5 → the +6 differential fails to move
  occupancy; the constraint cannot handle near-caustic doubles at any
  resolution — method redesign (discrete-count constraint or excess-aware
  target) required before any triple arm. (c) re-rank KS p < 0.05 or |ρ| >
  0.3 → the loglik Holm rejection was NOT (only) the penalty-at-truth
  artifact; the P-10 "prediction 2 PASS" interpretation is compromised —
  investigate before new arms. (d) convergence-rung agreement < 0.9 → the
  audit operator is unconverged and every N_eff-based number here is suspect.
  (e) > 5 additional mislabels in the recount → generator defect broader than
  the window; regenerate the dataset wholesale before ANY further arm.

  **Threshold derivation:** 0.9 = below P-10's recovered band (fine-audit
  0.939–1.000; lenstronomy 0.844–1.000 with sys_63's 0.844 the one straddler)
  and above the affected band (≤ 0.83). sys_57 median band [2.0, 2.45] =
  truth-level converged value 2.334 + 0.11 (the q50→q90 half-spread of fine
  N_eff observed on P-10 recovered systems, e.g. sys_98). KS α = 0.05: single
  pre-registered test (not a 13-way family), n = 100 gives power against
  |ΔF| ≳ 0.135 — the mc1 penalized PIT deficit is ~3x that. |ρ| ≤ 0.2 =
  2× the null sd of Spearman at n = 100 (1/√99 ≈ 0.10). Rung tolerance 0.05 =
  7× below the 0.35 decision tolerance, so residual quadrature error at that
  level cannot flip any classification; the 0.9 fraction allows a tail of
  near-fold draws where quadrature convergence is intrinsically slow.
  Threshold for prediction (3)'s upper count (12) is NOT derivable beyond the
  9 known members — the +3 allowance is a guess, flagged as such; only the
  falsifier-(e) bound (> 5 new mislabels) is decision-relevant.

  **Blind spots:** the fine-audit N_eff now shares BOTH kernel and resolution
  with the in-likelihood operator (P-10's audit at least differed in
  resolution) — a common operator artifact would fully self-confirm;
  mitigation: the lenstronomy discrete-count cross-check is PRIMARY for every
  basin claim, and the 2x rung bounds quadrature error. A 7-system rerun has
  no SBC power: calibration under the fine config is NOT tested here (next
  full arm). The lenstronomy count itself can miss unconverged roots — the
  min_distance ladder stability check guards it.

  **Expected appearance:** paired mc1→mc2 bars of lenstronomy frac(count==2):
  Class A and sys_57 jump to ≈ 1.0; sys_38/86/99 land wherever displacement
  puts them (recorded, not gated). sys_57 fine-N_eff histogram: mode 2.86 →
  2.0–2.4. Re-rank PIT deciles: flat (the penalized version piles up in the
  lowest deciles). If (b) fires: sys_57 lenstronomy hist stays {3: 500} and
  the N_eff mode stays ≥ 2.6.

  **Cost (measured basis):** P-10 production = 12 systems/GPU in 13.9 h at
  chunk 7 (56 lanes, coarse). P-11 runs chunk 1 (8 lanes) × 4× tiles →
  per-step ≈ 0.57× the 56-lane coarse step → ≈ 4 h/system; worst GPU carries
  2 systems ≈ 8 h, + ≈ 1 h audit → 16 h wall requested (2× margin on the
  tail). One es0 node, 4× GRTX2080TI, ≤ 64 GPU-h. **Memory (the P-10 OOM
  lesson, pre-verified):** peak_gpu_bytes = 1.35 GB on ALL 100 mc1 systems at
  chunk 7; the fine config keeps per-tile quadrature size identical
  (768²/64 = 384²/16 = 9216 points) and runs 1/7 the lanes → strictly below
  1.35 GB on 11 GB cards. Anneal uses the same mc_anneal_block=16 path that
  fixed the P-10 OOM. Host: MaxRSS 12.4 GB at 4 concurrent slices < 90 GB.
  **Seeds/config/code:** campaign seed 3 all stages (matches mc1); yaml
  differs from mc1's ONLY in the constraint knobs (eps 0.1→0.05, grid
  384→768, tiles 16→64, anneal rungs [0.3, 0.15] → [0.3, 0.15, 0.1]); code
  PINNED at the P-10 commits via worktrees — gigalens d321d3c
  (`~/gigalens-worktree-p10`), GIGALens-Code 4358ebb
  (`~/GIGALens-Code-worktree-p10`) — because both live checkouts have since
  taken the ModellingSequence migration, and the remedy comparison is only
  controlled if nothing but the operator resolution differs from mc1.
  Smoke-tested on the login node: imports, truth→z mapping, unpenalized
  log_like, all three operator rungs, recount worker.
  **Status: APPROVED by user 2026-08-03 ("Okay - go ahead!") after review of
  the checkpoint. Files committed as GIGALens-Code 0535116 before launch;
  submitted as job 24529614 (es0, es_normal, 4x GRTX2080TI, 16 h).**

---

## Log (newest first)

- **2026-07-31 (P-10 RESOLVED: double-arm rerun with the in-likelihood
  multiplicity constraint — core claim substantiated; one operator-accuracy
  failure; dataset mislabeling discovered).** Campaign completed 100/100 in
  `runs/<sid>/mc1` after a walltime loss and two gap-fills (24307807 OOM →
  24308191 16h-TIMEOUT with 4 systems persisted → 24350103_0 systems 0–51 →
  24382995 systems 52–99, chunk 7, 24h). Mid-campaign the live gigalens
  checkout advanced past d321d3c (PR #94 removed `ModellingSequence`), so the
  second gap-fill ran against a pinned worktree `~/gigalens-worktree-p10` at
  d321d3c — all 100 systems used byte-identical code. Analysis: job 24390199
  + `experiments/hundred_point_sources/p10_analysis.py`; artifacts in
  `diagnostics/p10_analysis/` (JSON + per-system N_eff/count arrays) and
  `aggregate/` (rank ECDFs, sbc_report.json). Observed vs predicted:
  - **Prediction 2 PASS.** Unfiltered SBC over 100 systems: **1/13 Holm
    rejections** (predicted ≤ 2/13); gamma is clean unfiltered — P-7's
    unfiltered gamma rejection is gone (the `default` sweep still shows it
    under identical gates). Falsifier (b) does not fire (no gamma/theta_E
    rejection). The one rejection is `loglik` (KS p = 1e-4, mean PIT 0.386,
    skewed low, uniform across systems, absent in P-7). Mechanism quantified:
    the rank scores truth under the PENALIZED likelihood; draws equilibrate
    where the coarse-operator penalty ≈ 0, while truth pays the coarse
    operator's error at its (genuinely count-2) parameters. Spearman(pen_coarse
    (truth), PIT) = 0.51 (p = 5e-8); the 26 systems with PIT < 0.1 have median
    pen(truth) = −4.1 vs −0.05 for the rest. An operator-accuracy artifact at
    truth, not parameter miscalibration (parameter ranks uniform).
  - **Prediction 1: 11/12 PASS, falsifier (a) fired for sys_57.** The 11:
    frac(N_eff≈2) = 0.939–1.000 (predicted > 0.9; P-7 kept ≈ 0), confirmed by
    the pre-registered lenstronomy cross-check (500 thinned draws each, window
    12, min_distance ladder): frac(count==2) ≥ 0.844 (9 systems at ≥ 0.998).
    **sys_57 = 0.000**: its constrained posterior sits ENTIRELY in a genuine
    3-image region — lenstronomy finds 3 real images on all 500 draws (radii
    0.79–1.40", inside the 2·theta_E = 2.63" window half-width), fine-config
    N_eff reads 2.86, yet the in-likelihood coarse operator reads 2.04 on the
    same draws (direct measurement, 32 draws), so the lam = 10 penalty never
    engaged. Diagnosis: NOT an anneal/capture failure — the coarse quadrature
    (eps 0.1, grid 384, fp32) undercounts the third image by ≈ 0.8 of an
    image. This is exactly the registered blind spot ("the frac metric shares
    its operator with the constraint"), caught by its registered mitigation.
    Partially affected genuine doubles (fine frac < 0.9 from the same cause):
    sys_86 (0.12), sys_35 (0.21), sys_99 (0.40), sys_38 (0.65), sys_69
    (0.67), sys_67 (0.83). The pre-declared remedy — eps 0.05, grid 768 —
    targets precisely this failure; a P-11 checkpoint is required before any
    rerun (window 4·theta_E at grid 768 gives step 0.0069 ≤ 0.05·√0.1 ✓,
    ~4× quadrature cost).
  - **Prediction 3 PASS.** Healthy systems (P-7 kept ≥ 0.5, n = 70) vs P-7
    count==2-filtered draws: per-parameter median |Δmean|/sd = 0.06–0.17
    (predicted < 0.3 for all 13; worst is light center_y at 0.172). The term
    is inert where it must be.
  - **NEW dataset defect (double arm):** window-12 truth recount of every
    system with a P-7 filter matrix shows 5 systems mislabeled as doubles —
    truth has 3 images (sys_02, sys_26, sys_64, sys_81) or 4 (sys_19); the
    generator counted inside search_window 6 and missed wide images. Their
    mc1 posteriors legitimately track the extra-image structure (fine frac
    0.00–0.83); P-7 kept_frac for them ranged 0.02–0.99 (the window-6 filter
    was equally blind). Confirms the standing rule: generators and filters
    must use lenstronomy search_window ≥ 12. `truth_counts_w12.json` in
    diagnostics/p10_analysis.
  - **Verdict:** the core claim — the soft in-likelihood tilt reproduces
    P-7's filtered calibration while eliminating the zero-kept class —
    holds at the registered thresholds for calibration (1/13) and inertness
    (< 0.3 sd), and for 11/12 recovery targets. The registered failure mode
    that fired is operator accuracy at eps 0.1/grid 384, not the hypothesis
    (the position likelihood lacking image-count information) and not the
    annealing design. Follow-ups requiring their own checkpoints: (P-11)
    fine-config (eps 0.05 / grid 768) rerun of the 7 affected genuine
    doubles + loglik re-rank under the unpenalized likelihood; dataset
    regeneration with window ≥ 12 before any new arm.

- **2026-07-28 (P-9: capture-radius + sampling-cost micro-check on the
  zero-kept systems — annealed eps_init=0.3" reaches 100% of phantom-mode
  mass; fp32 quadrature validated).** Question: can MCLMC/MAP actually
  *sample* with the N_eff multiplicity penalty when whole posteriors sit in
  the 4-image region (the P-7 zero-kept systems)? Since N_eps = N ⊛ g_eps,
  the penalty gradient decays ~exp(-d²/2eps²) with source-plane caustic
  distance d, so eps_init must reach the phantom modes. Method
  (`capture_check.py`, job 24290841, 1×2080Ti; artifacts
  `simtests_results/smooth_count_check/capture_check/`): (A) re-derived
  zero-kept list from the P-7 double_filter labels → 11 systems at exactly
  0.000 kept + sys_65 at 0.016; phantom draws are counts 3/4/5. (B) per-draw
  d_caustic, topology-free: det A sign changes on a 512² grid over a
  6·theta_E window, edge midpoints mapped through beta, min distance to the
  source; 12,272 phantom draws. Result: **phantom modes hug the caustic** —
  d q10/50/90 = 0.029/0.111/0.244", max 0.874" (per-system medians
  0.023–0.312"); frac(d<0.9") = 1.000. Expected in hindsight: the local
  position likelihood creates phantoms by dragging the source just past the
  caustic, not deep inside. (C) gradient reach of P=(N_eff−2)², mu_min=0.1,
  jacfwd on 1536 subsampled phantom draws, eps ladder 0.5→0.05 with matched
  grids (step ≤ eps·√mu_min, 12" window): **zero dead gradients (<1e-10) at
  every eps**; at eps=0.3, med|∇P| = 5–14 across ALL d bins. eps=0.5 is too
  blurry (median N_eff 2.20 — the blur mixes in the N=1 region beyond the
  radial caustic), so eps_init=0.3 is the recommendation. Surprise: at
  eps=0.05 the far bins (d 0.3–1.0") keep med|∇P| ~ 0.6–2 instead of
  Gaussian-tail collapse — interpretation (plausible, not proven): the
  mu_min weight makes N_eff smoothly magnification-dependent, so draws with
  an image near the flux floor retain O(1) gradients through mu-drift even
  on the plateau; flux-awareness softens the plateau problem itself.
  Control (2 healthiest systems, count-2 draws): term inert at eps=0.1
  (med|N_eff−2| ≤ 0.022, |∇P| q50 ≤ 3e-2). (D) sampling-shaped cost,
  128-chain batched, per eval: fp64 value 31/69/122 ms at 128²/192²/256²,
  grad (remat) 194 ms at 128², OOM ≥192²; fp32 value 11/25/45 ms, grad
  78/150 ms at 128²/192², OOM at 256². **fp32 N_eff agrees with fp64 to
  max|dN| = 1.7e-6** on 512 phantom draws (both 256²/eps=0.1 and
  128²/eps=0.3) — fp32 quadrature is safe and ~3× faster on 2080 Ti.
  Implementation requirement: grad memory scales chains×grid → the
  production term needs grid-tile accumulation (scan over tiles with remat)
  to go past 128² at 128 chains; extrapolated final-stage cost ~0.1–0.3
  s/gradient step (fp32, theta_E-scaled window). Design settled: MAP anneals
  eps 0.3→0.05 (lambda ramped), VI+MCLMC at fixed final (eps, lambda), any
  MCLMC tempering confined to discarded warmup. Next: implement
  `multiplicity_constraint: {n_obs, mu_min, eps}` with tile-accumulated
  fp32 quadrature and rerun the double arm.

- **2026-07-27 (P-8: smoothed-count non-detection operator — standalone check
  PASSED; two lenstronomy solver failure modes found).** Motivation: a
  non-detection likelihood term must work for arbitrary caustic topology
  (naked cusps, future multi-profile lenses), ruling out curve-based
  caustic-distance constructions. Candidate operator: the smoothed image
  count `N_eps(beta_src, theta) = integral g_eps(beta_src - beta(x))
  |det A(x)| d^2x` — differentiable ray-shooting; EXACTLY the true integer
  multiplicity map convolved with a Gaussian of width eps in the source
  plane; detectability-weighted variant N_eff multiplies the integrand by
  `1/(1+(mu_min |det A|)^4)` (a soft |mu| >= mu_min floor). Implementation:
  gigalens jax EPL(niter=18)+Shear deriv, fwd-mode jvp for det A, fp64,
  batched on 2080 Ti. Validated against lenstronomy LensEquationSolver
  labels on ~102k v2_double posterior draws (all 100 systems), 3.1k
  naked-cusp v2_triple draws (6 systems), and a 10-system adversarial
  subset (chosen central-image/count-3/5-rich) relabeled with per-image
  magnifications. **Results: triple (naked-cusp) arm N_eff at mu_min 0.1
  agrees 97.2% overall / 97.9% off-boundary (1.6% of draws within the
  deliberate eps-blur of a caustic); adversarial double subset, scored
  against corrected window-12 labels: 93.9% / 95.7%, with the
  geometrically-defined even classes (N=2, N=4) at 98.4% recall each and
  the residual concentrated in odd classes = images AT the mu floor (a
  soft-vs-hard threshold comparison artifact, not operator error). Gradient
  (jacfwd) matches finite differences to ~1e-5 at converged resolution.**
  Discoveries about the SELECTION-DEFINING operator itself: (i) lenstronomy
  `image_position_from_source` with the campaign's search_window=6.0
  systematically MISSES the 4th bright image of big-theta_E draws (24/24
  sampled "count-3" big lenses became quads at window 12) — its candidate
  grid spans only +-3", and images whose basins lie outside are found only
  when Newton refinement escapes the window by luck; (ii) at native
  min_distance=0.01 it drops one member of merging pairs near cusps
  (sys_05: 125/129 "doubles" gained an image at min_distance/2), while
  N_eff = 2.000 for all 129 (the extra image is sub-threshold |mu| < 0.1)
  — the flux-aware count is the only operationally STABLE multiplicity
  definition near merging configurations; raw counts are ill-conditioned
  there for BOTH root-finders and grids. P-7 is NOT invalidated (generator
  and filter used the identical operator, which is all SBC consistency
  requires), but future campaign generators should use search_window >= 12
  and the production term must be flux-aware. Resolution requirements
  measured: window must cover all images (>= 10-12" for this theta_E
  prior; the 6" window was the sole cause of an apparent double-arm
  residual), grid step <~ eps*sqrt(mu_min) (0.0065" at eps=0.02",
  mu_min=0.1; coarser steps under-resolve moderate-mu image preimages).
  Cost at full resolution: ~35 ms/draw-eval on one 2080 Ti (fp64) — fine
  for post-hoc reweighting/SBC filtering at scale, needs adaptive gridding
  and/or coarse-eps annealing stages (grid requirement scales with 1/eps)
  before living inside an MCLMC likelihood. Also noted: the term's window
  must scale with the theta_E prior support, and gradcheck rel-err degrades
  to ~1e-2 when the grid under-resolves (quadrature noise, a useful
  built-in diagnostic). Scripts + labels + sweep JSONs:
  `simtests_results/smooth_count_check/` (neps_check.py / neff_check.py /
  label_mags.py, phases 1-5). Next: implement as an opt-in likelihood
  component (`multiplicity_constraint: {n_obs, mu_min, eps}`) with annealed
  eps, and rerun the double arm — prediction: zero-kept systems vanish and
  gamma stays uniform WITHOUT post-hoc filtering.

- **2026-07-25 late night (P-7: double-arm selection attribution VERIFIED by
  filtered re-rank).** Prediction registered before running (user-approved):
  since the likelihood is unchanged, the truncated-prior posterior is the
  untruncated posterior restricted to the N=2 region, so filtering the
  existing posterior draws by the SELECTION-DEFINING operator (lenstronomy
  image count, generator settings, + min-sep cut) and re-ranking must
  restore uniformity — gamma's rejection should evaporate, PIT -> ~0.5.
  **CONFIRMED: gamma KS p 0.0024 -> 0.558, mean PIT 0.572 -> 0.494; Holm
  rejections after filtering: NONE (0/13).** Mechanics: 95 gate-passers x
  1024 thinned draws (identical thinning/jitter conventions to sbc_ranks),
  ~97k lenstronomy solves on one lr6 node (38 workers, ~13 min); ranks
  recomputed per system among kept draws only, fresh deterministic jitter.
  Quantified: mean in-region posterior mass 0.70 — ~30% of double-arm
  posterior mass sits OUTSIDE the double region on average, the measured
  size of the missing non-detection term. Caveats (disclosed): (i) 14/95
  systems had < 50 in-region draws and were dropped from the re-rank (ranked
  n = 81) — 10 of them have ZERO in-region draws (posterior confidently in
  the quad region while the truth is a double); exact truncated-prior SBC
  for those would need re-SAMPLING under the truncated prior, so the re-rank
  set itself carries a mild residual selection; (ii) solver bake-off:
  gigalens LensSolver agreed 40/40 with lenstronomy but needed an explicit
  deflection_ratio=1.0 (auto-derivation silently resolved to 0 on standalone
  structured params — solve_images returned the source position as "1
  image"; flagged as a quiet failure mode worth fixing) and was only 1.4x
  faster on CPU (host-side per-point loop; GPU would not help) — lenstronomy
  chosen as the selection-defining operator. Artifacts:
  `diagnostics/double_filtered_sbc.json`, per-system counts in
  `diagnostics/double_filter/`. **Conclusion: the double-arm gamma anomaly
  is fully explained by multiplicity selection; the pipeline itself shows no
  miscalibration in any arm where it can be cleanly tested.**
- **2026-07-25 night (P-6 COMPLETE — all three arms run and scored; runs
  300/300 status ok; allocation released).** Compute: ~2.6 h fleet time on
  4x 2080 Ti for 300 systems at the 3x MCLMC budget (~126 s/system per GPU;
  solo-CPU equivalent ~190 h). Per-arm verdicts vs the pre-registered
  criteria:
  - **QUAD: FULL PASS.** 0/13 Holm rejections (worst raw p 0.078), attrition
    1/100, O-5 dead (src center_y p 0.27 / PIT 0.520 on fresh truths), H0
    calibrated across Uniform(20,100) (p 0.85). Proposed upgrade: the
    anchored likelihood + fixed MAP + intensive budget is CALIBRATED for
    quad point-source SBC (grader to certify; artifacts
    `hundred_point_sources_v2_quad/aggregate/`).
  - **DOUBLE: falsifier fired as predicted possible — 1/13 rejected (gamma,
    p 0.0024, PIT 0.572) + attrition exactly 5.0%.** Attributed to SELECTION,
    not pipeline (see entry below: +4.5-sem truth-gamma tilt, no
    non-detection term in the likelihood, bias anti-correlated with mixing
    difficulty, quad data-dominated arm clean).
  - **TRIPLE: TAINTED — attrition 28/100, far over the 5% boundary; not a
    calibration-grade rank set.** Attrition is geometric: excluded systems'
    median min image sep 0.415" vs 1.099" for passers; 60% attrition in the
    tightest-sep quartile vs 17% elsewhere — merging cusp pairs near the
    critical curve are the hard class. Registered gamma prediction scored
    honestly: direction CONFIRMED (mean PIT 0.450 < 0.5), magnitude MISSED
    (p 0.068, not rejected — data richer than doubles + taint muddies it).
    The one Holm rejection is src center_x (p 0.0006, PIT 0.619), consistent
    with the far stronger SOURCE-position selection in naked-cusp configs,
    but attribution beyond "selection-consistent" is blocked by the taint.
  - Follow-ups (not started): (a) an `any2plus` arm — no multiplicity
    selection, the cleanest SBC of the pipeline itself (generator already
    supports it); (b) a non-detection/multiplicity term in the likelihood if
    selective arms must be strictly calibrated; (c) triple rerun at higher
    budget (or PT) for the tight-cusp class before any triple-arm claims.
- **2026-07-25 evening (P-6 in progress: quad PASS, double falsifier fired —
  attributed to SELECTION, prediction registered for triple BEFORE its
  aggregate)** — Quad arm: 100/100 ok, 1% attrition (sys_88), **0/13 Holm
  rejections** (worst raw p 0.078 gamma2); **O-5 NOT reproduced on fresh
  truths** (src center_y p 0.27, mean PIT 0.520) — v1's signal was a
  fluctuation; H0 under Uniform(20,100) calibrated (p 0.85, PIT 0.503).
  Double arm: 100/100 ok, attrition 5/100 = exactly the 5% taint boundary
  (sys_05/08/50/54/59); **1/13 Holm-rejected: EPL gamma, p 0.0024, mean PIT
  0.572** — truths rank HIGH. Diagnosis (post-hoc, labeled): the bias
  ANTI-correlates with mixing difficulty (PIT 0.662 in the better-mixed half
  vs 0.481; strongest where the posterior is most prior-dominated) — not a
  sampler artifact. Root cause measured in the DATASETS: accepted truths are
  prior draws conditioned on multiplicity, and truth gamma is tilted
  +0.196 (+4.5 sem) in doubles, +0.148 in quads (invisible: data-dominated),
  **-0.419 (-9.5 sem) in triples** (naked cusps need shallow slopes). The
  fitted prior is untruncated and the likelihood has NO non-detection term
  ("no additional images seen"), so where data are weak the posterior sits at
  the prior while truths sit tilted — the pre-registered blind-spot (ii)
  mechanism, isolated to selective arms, implicating selection not pipeline.
  **Registered PREDICTION before the triple aggregate exists (triple arm
  still sampling): triple gamma mean PIT < 0.5 (truths rank LOW), plausibly
  Holm-rejected given the 2x-larger tilt; falsifier: triple gamma PIT >= 0.5
  or uniform.** Quota incident logged: home has an enforced ~20 GB VAST quota
  (invisible to `quota`); double arm died mid-persist on EDQUOT, recovered by
  moving simtests_results to scratch (symlinked) and re-running doubles.

- **2026-07-25 (v2 tri-arm datasets generated; runs on hold)** — Per Linus:
  v2 priors set (H0 Uniform(20, 100), frac_flux 0.005) and the campaign split
  into three multiplicity arms (quad / triple / double, same priors). Generator
  gained `multiplicity: triple` (n_img == 3). Pre-generation rate probe
  (N = 3000 prior draws, seed 99): quad 33.0% / double 39.7% / triple 10.0%
  (naked-cusp region well populated; also seen: 4.2% five-image draws where
  the central image is recovered, 12.0% single-image, 1.1% zero-image, and
  no min-sep rejections in 3000 draws). P-6 design checkpoint written
  (see Design checkpoints). Datasets generated (seeds 2/3/4, 100 systems each;
  realized acceptance 32.3% / 41.7% / 8.8%, zero solver errors, zero min-sep
  rejections) and verified: correct n_images everywhere; H0 truths span
  [20.0, 99.8]; sigma_flux == 0.005 x flux_true exactly; td_obs[0] == 0;
  campaign builder (anchor 0.004) gives finite lp and truth red chi2
  1.52 / 2.60 / 0.67 on arm-respective spot systems; batched-pipeline row
  extraction works at all three multiplicities. Note for the triple arm:
  closest image pair 0.055" (just above the 0.05" cut) — cusp pairs are
  tight; astrometric channel will be near-degenerate there. **Runs NOT
  launched — awaiting Linus's approval (P-6).**
- **2026-07-24 (evening: C-6 MAP bug + P-5 reruns)** — Audited MAP per user request:
  found and fixed the off-by-one pairing bug (→ C-6; PR
  seanxuseanxu/gigalens#80 onto linusu-dev-merge, MAP fix only). P-5 intensive
  reruns: sys_58 resolved into the dominant mode by the fixed MAP alone (the
  campaign's wrong-mode posterior was downstream of the bug); PT-MCLMC arm failed
  its falsifier (carousel knobs don't transfer — negative result); 4/6 systems pass
  gates at 3x budget, sys_76/91 at R̂ ≈ 1.10 extended to 6x (v3). C-3's cause chain
  updated: bimodality is real, but capture-of-the-wrong-mode required the MAP bug.
- **2026-07-24 (P-4: source-plane anchor)** — Implemented ``src_anchor_sigma`` in
  gigalens ``point_source_position.py`` (user-approved): per-image
  ``|beta(theta_hat)-beta_src|^2/sigma_a^2`` added to the scored chi2, default OFF,
  zero at convergence, no event_size/normalization change; threaded through
  ``PointSourceObsData`` and the simtests builder. 37/37 tests (5 new, incl. the
  k=0 closed-form identity extended with the anchor and the deep-shelf tilt pin).
  sys_76 rerun: phantom shelf eliminated (0 frozen chains, 0% unconverged, MAP
  solve converged); R-hat 2.06 → 1.30 (residual = plain slow mixing). Inertness
  spot-checks ambiguous at the letter (see P-4 resolution — threshold was
  under-derived; structural bound favors inertness). Next decisive step: campaign
  v2 with the anchor on, full SBC. New O-6: MAP z_best selection discards
  good-chi2 particles.
- **2026-07-24 (P-2 run)** — Truth-anchored MCLMC for sys_58 (user-approved): chains
  initialized at z_truth with campaign-SVI preconditioning STAY in the truth basin
  (99.4% of draws ll > 40, mixing healthily) and their typical ll exceeds the
  campaign posterior's by ~25 units. Verdict: the campaign run sampled a subdominant
  mode. See P-2 resolution and C-3. Also noted for the fix path: `PTMCLMCStage`
  (parallel-tempered MCLMC, validated on the 33-dim carousel two-basin system,
  gigalens PR #66) is purpose-built for exactly this class and could both sample
  sys_58 correctly and measure the mode mass ratio.
- **2026-07-24 (later)** — **Campaign complete: 100/100 run, 94 pass gates, all 13
  Holm-corrected SBC tests pass** (→ C-5). Excluded: sys_58/66/68/76/87/91 (6%
  attrition, mildly clustered at steep γ). ECDF plot inspected and agrees with the
  numbers except src center_y grazing the band (raw p = 0.013 — watch item O-5).
  P-1 resolved (sys_91: direction confirmed, magnitude missed — see checkpoint).
  Artifacts: `aggregate/{sbc_report.json,sbc_ecdf_default.png,sbc_hist_default.png}`.
- **2026-07-24** — Diagnosed the 4 gate-failing systems of the 84 finished (skill:
  diagnose-sampling; no code/config changes). Artifacts:
  `simtests_results/hundred_point_sources_v1/diagnostics/`.
  - Step 0: model card clean (float64 likelihood, event_size 15, wCDM + sampled H0;
    one info advisory: mixed float32/float64 free-param dtypes — unexamined, see O-2).
  - Numbers (worst param, z-space tfp R̂/ESS over 8×10k):
    sys_58 R̂ 1.57 (γ1_shear) ESS 20 (e1); sys_66 R̂ 1.08 (H0) ESS 181;
    sys_68 R̂ 2.92 (γ) ESS 13; sys_76 R̂ 5.30 (center_x) ESS 10.
    (Campaign-reported max R̂ 1.80/1.06/1.69/2.06 — different space/estimator, both bad.)
  - **Key numeric correction:** MAP `chisq_hist`/`best_chisq` are REDUCED χ². Campaign
    MAP best red χ²: median 0.26, p90 1.11 (n = 92 incl. in-progress); the three
    catastrophic systems are 22.6/26.6/28.6 — MAP never fit them. All R̂ > 1.5 failures
    have MAP red χ² > 20; both R̂ ∈ (1.05, 1.1) failures (sys_66, sys_87 — the latter
    from the resumed batch) have MAP red χ² < 1. sys_91 (in progress) has MAP red χ²
    24.8 → prediction P-1.
  - sys_76: chains 1+4 frozen on unconverged-solve shelf (100% src_res > 1e-4″,
    ll ≈ −170); MAP z_best itself unconverged (src_res 3.5e-2″), SVI surrogate 95.9%
    unconverged draws ⇒ init delivered chains onto the shelf; 6/8 escaped. Δll ≈ 341
    barrier shelf→basin (straight z-line). → C-2.
  - sys_58: single wrong basin, solver converged everywhere, truth basin higher-ll
    (47.1 vs 43.0 best draw) never visited; Δll ≈ 11–15 straight-line barrier; misfit
    concentrated in td channel (χ²_td 12.4 vs 2.8), H0 posterior 65.5 vs truth 76.5.
    → C-3, decisive test P-2.
  - sys_68: ridge + γ prior-ceiling pinning, truth covered, MAP unplateaued. → C-4.
  - sys_66: H0-only R̂ 1.08, all else < 1.02; healthy everywhere. Plain short-chain.
  - Negative results (checks that came back clean): no NaNs in any chain (nan_rate 0
    all four); solver 100% converged across posterior for sys_58/66/68 (the 25% on
    sys_76 is entirely its two frozen chains); truth z-prior NOT suppressed for sys_58
    (−2.7 vs −1.6 typical) — an apparent −46 was a normalization artifact of comparing
    `prob.log_prob` output against `log_like` (resolved by coordinate-swap attribution).
  - Method traps hit and resolved (for the next agent): coordinate-wise posterior
    means/medians are NOT valid evaluation points in this stiff landscape (the z-mean
    scored red χ² 15 where draws score 4; chain-median 85 where draws score 1) — use
    actual draws; and `prob.log_prob` vs `log_like` outputs are not on directly
    comparable scales — derive prior contributions by coordinate swaps, not subtraction.
- **2026-07-23/24** — Campaign v1: 100/100 quads generated (lenstronomy, 32%
  multiplicity acceptance, 0 solver errors); 84/100 systems ran before walltime
  (~13 min/system/shard); resumed on n0028.es1. 80/84 passed gates; ~4.8% attrition.

---

## Open questions

- **O-1:** Were sys_76's frozen chains victims of per-chain step-size collapse (MCLMC
  tuning on the shelf) or of the shelf's designed flatness? Requires persisting tuned
  step size / L per chain — currently not saved by MCLMCStage.
- **O-2:** Model card advisory "mixed float32/float64 free-param dtypes" — benign?
  Which parameters are float32, and does the promotion affect the solve path?
- **O-3:** Is sys_58-style td-channel bimodality common at higher σ_td / different H0
  draws (it pulls H0 by ~10)? The σ_td sweep already sketched in campaign.yaml would
  double as a probe.
- **O-4:** sys_87 (resumed batch) fails gates at R̂ 1.087 with healthy MAP — recheck
  after aggregate; expected to be sys_66-class (longer chains fix).
- **O-6:** MAPStage's returned ``z_best`` (best-lp-at-best-step) scores far worse than
  the best chi2 the optimizer visited, on hard systems — anchored sys_76: z_best red
  chi2 43.8 (converged solve) vs best-seen 5.96; unanchored sys_58/76 showed the same
  gap. Either the per-step "best particle" selection or the lp-vs-chi2 criterion is
  discarding good particles. Orthogonal to the anchor; relevant to P-3.
- **O-5:** src center_y is the consistently worst-calibrated marginal (mean PIT 0.617
  at n = 40 mid-campaign, 0.574 at n = 94 final, raw KS p = 0.013, ECDF grazing the 95%
  band). Statistically legal, but if a rerun/extension shows the same sign, investigate
  (candidate mechanisms: quad-selection truncation of the source-position prior, or the
  solver's seed geometry breaking y-symmetry).
