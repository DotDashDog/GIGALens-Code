# LAPS transferability battery (laps_transfer_v1)

**Area:** does the certified LAPS operating point (see `laps_prior_init_investigation.md`
DC-7.3/7.4 and `laps_validation_report.md`) transfer beyond the single demo lens it was
certified on? First battery: systems 0–9 of `data/simulated_systems/100SystemsStandard80px.npz`
(GL2 EPL+Shear+Sérsic² standard sims, known truth in `100SystemsStandardParams.yaml`).
System 60 (the known-hard `why_hard_to_sample` case) is explicitly out of scope here.
Requested by the user 2026-07-08 after LAPS failed on a separate hard test system of theirs.

**Harness:** `map_svi_hmc_laps` pipeline (MAP → SVI → HMC → laps_warm → laps_cold, one
shared cached front end per system), campaign `experiments/laps_transfer/campaign.yaml`,
offline analysis `experiments/laps_transfer/analyze.py` (Tier 1 internals / Tier 2 truth
coverage / Tier 3 vs-HMC, thresholds below). Tests:
`simtests/tests/test_laps_pipeline.py` (23, CPU).

---

## Certified claims

_None yet._

---

## Design checkpoints (criteria awaiting approval)

### DC-T1 — Transferability battery: certified LAPS presets on GL2 systems 0–9

- **Status:** APPROVED FOR LAUNCH (grader pass 2, 2026-07-08) — CERTIFY-RECOMMENDED, launch
  only. Scope: battery execution per the pre-registered design; NO transferability claim
  until results are graded with corners viewed (light-corner review gating for cold-arm
  passes). Caveats carried: single seed per system (a clean pass is single-seed evidence);
  shared-basin blindness of the warm-seeded HMC reference; pooled coverage canary-only.
  Grader independently re-executed the box-core dilution case (rms 1.0 / box 0.9 / FAIL) and
  a healthy-ensemble false-FAIL check (box 1.0, no flags) against the live `_tier3_metrics`;
  full pytest file not independently re-run by grader (no jax env without installs) —
  coordinator-reported 24/24 accepted with that note.
- **Claim under test + classification:** distributional claim, per system: samples from the
  certified `laps_warm` (W128) and `laps_cold` (R128a, mid-P2 straggler resampling) presets
  match the same system's warm-started HMC posterior within MC error, and recover known truth
  at nominal coverage. Chain of links: (a) warm-init validity class transfers; (b) the
  prior-init machinery transfers — survivor-cut separation, resample timing (chunk 13), guard
  (24 @ 128 chains), Phase-1 EEVPD tuning; (c) the HMC reference itself is converged and
  in-basin on these systems. This battery tests (a) and (b) **conditional on** (c); (c) is
  gated per system by HMC's own R-hat/ESS and by truth coverage (an HMC arm that misses truth
  disqualifies that system as a reference, it does not count against LAPS). The shared-basin
  caveat from certification carries over unchanged (warm-seeded HMC cannot rule out a missed
  global mode; truth anchoring partially mitigates).
- **Cause hypothesis:** the certified mechanisms are not lens-specific: the core/straggler
  logp separation (demo lens: core p5 = 152 vs straggler max = 147 at d=22), the chunk-13
  plateau-escape timing, and the d-dependent survivor cut logp > max − (d/2 + 4√(d/2)) were
  derived from properties of the posterior geometry class (single dominant basin, d≈22 EPL+
  Shear+Sérsic²), not from the specific demo image. These 10 systems are in the same model
  class at the same dimension, so the presets should transfer as-is. Named failure
  mechanisms if wrong: (i) cut fails to separate (survivors contaminated or over-pruned),
  (ii) stragglers arrive after chunk 13 (resample fires too early → post-resample straggler
  recurrence), (iii) guard=24 miscalibrated for these posteriors, (iv) EEVPD Phase-1 tuning
  destabilised by different noise draws, (v) genuine secondary modes (light-swap degeneracies)
  that resampling would actively prune.
- **Prediction (direction + magnitude):** warm arm passes 10/10 (core_fraction = 1.0, per-param
  offsets ≤ 0.3σ_HMC, width ratios within [0.8, 1.25] — demo-lens certified values were
  ≤ 0.115σ and [0.83, 1.10]). Cold arm passes ≥ 8/10 outright with n_survivors in ~[30, 70]
  of 128 (demo-lens certified seeds: 31/47/51; pre-resample core fraction 39–46% → expected
  survivors ≈ 0.25–0.55 × 128). Pooled 1σ truth coverage in [0.58, 0.78] for all three arms
  (n ≈ 220 params per arm). Failures, if any, concentrated in Tier-3 cold-arm metrics with
  Tier-1 canaries firing first (low survivors / skip), not silent.
- **Falsifier:** any cold-arm system with core_fraction < 1.0 or post-resample chains below the
  logp threshold (mixture returned), or a resample skip (guard trip), or offsets > 1.0σ /
  width ratios outside [0.5, 2.0] ⇒ the certified cold preset does **not** transfer as-is
  and the failing mechanism must be identified from Tier 1 before any retune. Any warm-arm
  failure ⇒ the validity class itself doesn't transfer (more serious; reopens certification
  scope, not just cold-start machinery). All-arms-fail on a system with HMC also missing truth
  ⇒ reference/system problem, battery inconclusive for that system, not a LAPS result.
- **Metrics + derived thresholds** (implemented in `analyze.py`; z = unconstrained space):
  - Offset |mean_LAPS − mean_HMC|/σ_HMC: WARN > 0.3, FAIL > 1.0. Derived: certified maxima
    are 0.109σ (warm, W128) and 0.207σ (cold, R128b — the larger of the two arms this battery
    centrally tests); 0.3 ≈ certified cold max + ~2× the SE of a mean from ~500 effective
    samples (≈ 0.045σ). NOTE (grader correction): the cold arm may therefore legitimately
    WARN in the 0.2–0.3σ range without that constituting a transfer failure — WARN is a
    canary; FAIL at 1.0σ (unambiguous disagreement) is the verdict level.
  - Width ratio σ_LAPS/σ_HMC: WARN outside [0.8, 1.25], FAIL outside [0.5, 2.0]. Derived:
    certified arm-to-arm envelope [0.83, 1.10] ± ~3× the ~5–7% SE of a std ratio at
    n_eff ≈ 200–400.
  - Core, TWO verdict-bearing masks per chain (both FAIL if fraction < 1.0), computed on the
    chain's kept-state mean in HMC-whitened z: (i) L2 rms < 6 (catches bulk displacement;
    demo-lens stragglers sat at rms p50 ≈ 29, ~5× the cut — but DILUTED for low-dimensional
    excursions: 10σ in 4 of 22 params gives rms ≈ 4.3 < 6); (ii) per-param L∞ box, all
    |z_i| < 6 — the certification-style box core (certified as per-sample box on mass params;
    here per-chain-mean over ALL params), which is what catches light-swap/label-switching
    modes (mechanism v) the rms and the equal-logp gate are both blind to. (Grader DC-T1
    correction: an earlier draft had only the rms core and mislabelled it as the certified
    definition.)
  - Per-chain final logp < median(HMC logp) − (d/2 + 4√(d/2)) (= the sampler's own survivor-cut
    form; ≈ 24.3 at d = 22, ~7× the posterior's own logp std √(d/2) ≈ 3.3 below the median →
    negligible false-positive rate; demo-lens stragglers were 10s–100s below). FAIL if any
    such chain on cold post-resample; WARN on warm.
  - Tier-1 survivor canary: FAIL on resample skip (guard trip); WARN if n_survivors < 31
    (the certified demo-lens minimum). NOTE: an earlier draft had FAIL < 36, which would have
    failed a *certified* demo-lens seed (31 survivors) — corrected before any run; recorded
    here as a threshold-calibration lesson.
  - Pooled truth coverage: WARN outside 1σ ∈ [0.58, 0.78] or 2σ < 0.90. Derived: binomial SE
    at n ≈ 220 is ≈ 0.031; band = 0.68 ± ~3 SE. Treated as canary (params within a system are
    correlated, so effective n < 220; a hard threshold is **not derivable** without a
    per-system correlation estimate — flagged, not verdict-bearing).
- **Metric blind spots (one sentence each):** Tier 3 is blind to a mode that both warm-HMC and
  LAPS miss together (shared-basin caveat; truth coverage partially covers it); pooled Tier-2
  coverage can hide compensating per-system miscalibrations (per-system corners cover it);
  ensemble mean/std offsets are mixture-blind (the three-strikes lesson — covered explicitly
  by per-chain core fraction and per-chain logp, which is why those are verdict-bearing).
- **Pre-committed expected appearance (GATING, not decorative):** the cold-arm verdict for a
  system is not "pass" until the light-group corner has been visually reviewed for satellite
  blobs (light-swap modes, mechanism v) — the automated flags are read second, the corners
  first. Per-system corner overlays (mass + light groups):
  three nested/overlapping contour sets (HMC black, warm, cold) with truth crosshairs inside
  the bulk. If the falsifier fires: cold contours inflated or multi-blob with satellite
  scatter far outside HMC (the pre-fix demo-lens appearance), or truth crosshairs outside all
  contours on a system where HMC also misses (reference problem, not LAPS).
- **Cost estimate:** ≤ ~5 node-hours total on one 4×A100 node: per system ~20–30 min
  (MAP ~2–3 + SVI ~4 + HMC ~10 + 2 × LAPS ~4, plus model build), × 10 systems, sequential;
  1 smoke system first, then the rest via resume (front-end stages cached). Interactive QOS
  (4 h) may need two sallocs; fallback regular QOS sbatch overnight.
- **Seeds / config / code:** campaign seed 0 (pipeline-wide; stage seeds derived);
  `experiments/laps_transfer/campaign.yaml`; code = worktree-laps @ 1f910e8 + harness commit
  (SHA recorded at launch). keep = 4 per chain (`p2_keep_per_chain`, matches the validated
  R128k collection arm; R-hat(n=4) honest-caveat from certification noted).

---

## Log (newest first)

- **2026-07-08 (later)** — Grader pass 1 on DC-T1: NEEDS-MORE, two findings, both fixed with
  zero GPU: (1) rms-only core was mixture-blind to low-dim light-swap excursions and
  mislabelled as the certified definition → added per-param L∞ box core as a second
  verdict-bearing FAIL (+ regression test proving the 10σ-in-4-params case passes rms but
  fails box); (2) offset derivation cited the warm arm's 0.115σ where the certified cold max
  is 0.207σ (R128b) → derivation corrected, cold-arm 0.2–0.3σ WARNs pre-declared as
  non-verdicts. Light-corner review promoted to a gating step for the cold-arm verdict.
  Grader confirmed: presets correctly wired; survivor set {31,47,51} matches artifacts;
  analyzer FAIL thresholds do not fail the certified demo run.
- **2026-07-08** — Harness built (Sonnet subagent; orchestrator-reviewed): `map_svi_hmc_laps`
  builder, `system_ids` subset support in `gl2_existing`, campaign yaml, `analyze.py`,
  23 CPU tests green (plus full simtests+inference suites: 67 passed, 1 pre-existing
  unrelated failure `smoke_test.py::test_system_io`, reproduced on the untouched home
  checkout). Survivor-count FAIL threshold corrected against certification artifacts
  (`diag_resample128`: n_survivors 31/47/51) before any run. DC-T1 written; awaiting grader.

## Open questions

- Does chunk-13 resample timing generalise, or does straggler arrival time vary by system?
  (Post-resample logp recurrence check in Tier 3 is the probe; a per-system arrival-time
  measurement would need track_chains trajectories — kept for a follow-up if Tier 3 fires.)
- Multi-seed robustness: this battery is single-seed per system (seed 0); 3-seed follow-up
  planned only for systems that fail or sit near thresholds (re-seed via `laps_seed` without
  invalidating the cached front end).
