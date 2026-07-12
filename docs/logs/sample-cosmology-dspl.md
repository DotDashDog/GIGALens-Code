# Lab log — sample_cosmology / DSPL cosmology sampling (new API)

Area: `experiments/sample_cosmology/` (ported from
`GIGALens-Code-paper/experiments/sample_cosmology/dspl(1).ipynb`).
System: single EPL deflector at z_lens=0.5, two independent Sersic sources at
z=1.0 (reference plane) and z=1.5, each its own noisy band; free cosmology
(Om0, w0), H0=70/k=0/wa=0 fixed; NormalCDF event-space bijectors on Om0, w0.
Goal: get MCLMC to sample the (Om0, w0) posterior correctly.

---

## Claims register

### C-1 — MCLMC (std config) truncates the DSPL cosmology posterior at Om0≈0.15, omitting ~10% of mass; the omission is a sampler artifact
- **Status:** `proposed (UNCERTIFIED)` — awaiting grader inspection of `def_ratio_grid_overlay.png` + `mech_traces.png`
- **Criterion:** density is exactly constant along r2 contours (uniform priors; cosmology enters likelihood only via r2), so an empty segment of the visited contour is an artifact by construction. Mass at stake: 0.103 below Om0=0.146 (σ-independent over σ_r ∈ [6.7e-4, 1.3e-3]).
- **Evidence:** `results/sample_cosmology/dspl_cosmology_newapi/def_ratio_grid_overlay.png`, `mech_traces.png`, `def_ratio_grid.npz`; code `experiments/sample_cosmology/def_ratio_grid.py`, `dspl_mclmc_mechanism.py`.
- **Doubt report:** (a) *expected shape wrong for this system* — eliminated by regenerating the grid at the notebook's z_lens=0.5 (original script used 0.4); (b) *arm genuinely low-density after nuisance marginalization* — eliminated structurally by the constancy argument; caveat: "cosmology enters only via r2" was code-verified in the OLD API (`gigalens-old/.../simulator.py`), asserted-by-design in the new scene API, and will be independently checked by Route A's equivalence gate; (c) *frozen/under-burned chains* — eliminated: within-segment mixing is healthy, ~64 independent approaches over 10k steps; (d) *NaN rejections* — eliminated: 0/160000 flagged. Residual doubts: σ_r,eff measured from the truncated run (cross-ridge width should be truncation-insensitive, but is not independently verified); R̂/ESS computed on the results segment only.
- **Proposed by / on:** producer agent session · 2026-07-07 · **Grader:** _pending_

---

## Design checkpoints (criteria awaiting approval)

- **Run A: free-r2 reparameterization + analytic (Om0, w0) reconstruction.**
  Hypothesis: the (Om0,w0)+NormalCDF sampling coordinates are the disease (rotating, semi-infinite thin ridge in z-space); the likelihood's only cosmology dependence is the scalar r2, so sampling plane-2 geometry as a free `deflection_ratio` (prior UniformBij over [1.2645, 1.3432], the exact range of r2 over the prior box; plane 1 fixed at 1.0) removes the pathological geometry entirely.
  Prediction: r2 becomes an unremarkable parameter — R̂(r2)<1.01, bulk-ESS(r2) within 2× of the median nuisance ESS (vs. cosmology worst-in-run at ≈200/80k today); r2 posterior ≈ N(1.32417, ~6.7e-4); reconstructed p(Om0,w0) ∝ ĥ(r2(Om0,w0)) contains the full arm with mass(Om0<0.146) = 0.103 ± 0.02 (grid-derived, σ-insensitive).
  Falsifier: R̂(r2)≥1.01 or ESS(r2) < half the median nuisance ESS (residual hard geometry in r2 ⇒ coordinate-disease hypothesis wrong); or the **pre-run equivalence gate fails** — log-prob of the r2-model vs. the cosmology-model at ~64 matched prior draws (r2 set to r2(Om0,w0)) differing by rel > 1e-8 (float64) ⇒ the structural claim is wrong in the new API and everything upstream is re-opened.
  Metric + derived threshold: R̂ 1.01 = repo standard; ESS factor 2 covers the seed-band scatter seen in repo history (~1.4×); mass tolerance ±0.02 is generous vs. the <0.001 σ_r-sensitivity measured on the grid.
  Blind spot: errors common to both parameterizations (e.g. a wrong deflection_ratio formula) are invisible to both the gate and the reconstruction-vs-grid comparison; r2-metrics say nothing about the nuisance block (also report worst nuisance R̂/ESS).
  Expected plot: per-chain r2 traces stationary and overlapping; reconstructed 2-D contours reproduce `def_ratio_grid.png` bands including the arm to Om0=0. If falsified: r2 traces wander/split, or gate mismatch.
  Cost: one 8-chain 10k+10k MCLMC run on one interactive GPU node (≈ the completed notebook run) + CPU-cheap gate and reconstruction. Seeds: MAP 1, MCLMC 10 (match the baseline run). Status: **approved by grader (user) 2026-07-07; launched**.
  - *2026-07-07 update:* pre-run **equivalence gate PASSED** — max relative log-likelihood diff 1.49e-14 over 64 matched prior draws (threshold 1e-8); `results/sample_cosmology/dspl_free_r2/equivalence_gate.json`, code `experiments/sample_cosmology/dspl_r2_equivalence_gate.py`. The "cosmology enters only via r2" link is now verified in the NEW API at float64 round-off (closes the corresponding caveat in C-1's doubt report). Implementation ready: `dspl_free_r2.py` (import-safe, `--run`-gated), `dspl_r2_reconstruct.py` (dry-run smoke-tested), `run_dspl_free_r2.sh` (launcher; account/queue flags copied from `why_hard_to_sample/slurm/run_t28.sh`, unconfirmed). Free r2 = z-index 12, `planes/2/geometry/deflection_ratio`, 20 free params total. Noted discrepancy found during implementation: the baseline notebook's MAP cell has `optimizer_factory` commented out, so its actual completed run used the nesterov-enabled default despite the markdown's claim — Run A uses the explicit no-nesterov factory per design; minor numerical difference if MAP outputs are compared.

- **Run B: arm-initialized frozen-metric MCLMC (mechanism falsification).**
  Hypothesis: the truncation is a soft reflection barrier caused by the frozen global metric mis-tracking the rotating ridge past the crest (tangent-metric mismatch 2.9°→34.5° bulk→turnaround); crossing is inefficient from BOTH sides under that metric.
  Design: profile-MAP the 19 nuisances with cosmology FIXED on-contour at Om0=0.05 (w0 from `def_ratio_grid.npz`, ≈−1.2); start 8 chains in a 1e-3 ball there; sample 10k steps with adaptation OFF, importing the baseline run's final inverse mass matrix, step_size=0.1528, L=41.44 (duck-typed qz wrapper: tiny-ball `.sample()`, baseline-metric `.covariance()`).
  Prediction: few-to-zero crest crossings — per-approach crossing prob < 4.6% (95% binomial bound from 0 crossings in ~64 bulk-side approaches in the baseline run); majority-arm occupancy; any chains that do cross stay in the bulk (z-space density gradient is bulk-ward).
  Falsifier: free bidirectional mixing — mean ≥ 3 crossings per chain in 10k steps (baseline bulk-side chains made ~8 approaches each with 0 crossings; ≥3 implies per-approach crossing prob ≳40%, incompatible with a barrier).
  Metric + derived threshold: crossing = passage between Om0<0.163 and Om0>0.25 with ≥50-step dwell (de-jitters the count; 0.163/0.25 are the measured turnaround and excursion thresholds from T2).
  Blind spot: tests the baseline's frozen metric only — silent on whether arm-local adaptation would cross; a failure to cross could also reflect an undiscovered genuine density defect in the new-API arm (mitigated by Run A's gate + reconstruction running in the same campaign).
  Expected plot: Om0 traces resident below ~0.15 with bounces (hypothesis) vs. free oscillation across 0.05–0.55 (falsifier).
  Cost: small profile-MAP + one 8-chain 10k-result MCLMC run (~half the baseline) on the same interactive GPU node. Seed: 10. Status: **approved by grader (user) 2026-07-07 (incl. the bare-scan frozen-metric mechanics amendment); launched**.
  - *2026-07-07 update:* implementation ready: `experiments/sample_cosmology/dspl_arm_init.py` (+ `_analysis.py`, launcher). Arm init point resolved: **w0_arm = −1.011009** at Om0=0.05 (on-contour to r2 residual ~0; local grid density 1.0e-4 ≫ the 99.7% level 1.2e-6 — sits on the near-maximal-density contour as T1 predicts; contour runs (0, −1.264) → crest (~0.2, −0.94), so −1.011 at 0.05 is where it should be). Design amendment (mechanics, not science): `MCLMC_JIT` with all `frac_tune=0` is broken as suspected — `L_adaptation_step=0` fires `calc_new_L` at i=0 on an empty ESS buffer (confirmed empirically: IndexError on a CPU toy; `mclmc.py` ~lines 279/429-437) — so Stage 3 drives `_build_kernel_shardmap` directly in a bare `jax.lax.scan` with step_size/L/inverse-mass-matrix frozen (no adaptation state exists, so no collectives needed). Toy validation (3-D normal, 4 chains, 200 steps): no NaNs, chains move, energy errors sane. Real stages are double-gated (`--run` + `--confirm-run-b-approved`).

- **Run C: ratio-coordinates grouped prior — single-run full forward model.** *(status: awaiting approval)*
  Claim class: distributional (sampler correctness/efficiency on a fixed target). Link tested:
  "the (Om0, w0)+NormalCDF sampling coordinates are the sole obstacle in the FULL forward
  model, and making the data-stiff scalar a sampling coordinate removes it" — the
  likelihood-equivalence link is already closed by the pre-run gate (below); generalization to
  ≥2 deflection ratios is explicitly NOT tested by this run.
  Hypothesis: replacing the cosmology Component's scalar priors with the grouped tuple-key
  ratio-coordinates prior (`gigalens_research.priors.ratio_coords`; triangular map z1→Om0,
  z2→u=r2 squashed into its conditional range at that Om0, w0 by bracketed solve; prior
  density unchanged = uniform box) removes the rotating, semi-infinite thin-ridge geometry:
  the stiff direction becomes the single coordinate z2, so the frozen global metric suffices
  and the full arm is visited — Run A's sampleability, without deleting cosmology from the
  model.
  Prediction: rank-R̂(Om0), rank-R̂(w0) < 1.01; bulk-ESS of BOTH cosmology z-columns within
  2× of the median nuisance ESS (Run A observed 1.02×; order of magnitude ~3k/80k, ~15× the
  baseline's ≈200); mass(Om0<0.146) measured DIRECTLY from the Om0 marginal = 0.103 ± 0.02
  (grid-derived; binomial error at ESS≈2k is ≈0.007, well inside); 0 nonfinite-flagged steps
  (baseline: 0/160000); MAP χ²/ν ≈ 1 (Run B profile-MAP: 0.998).
  Falsifier: either cosmology z-column with bulk-ESS < half the median nuisance ESS or
  rank-R̂ ≥ 1.01 ⇒ residual hard geometry in the new coordinates (conditional-bracket drift /
  tangency tail is itself a disease) — hypothesis wrong. Healthy R̂/ESS but
  mass(Om0<0.146) outside 0.103±0.02 ⇒ distribution wrong despite mixing ⇒ Jacobian-level
  implementation error — would contradict gate 3 and reopen everything upstream.
  Metric + derived threshold: R̂ 1.01 = repo standard; ESS factor 2 covers the ~1.4× seed-band
  scatter (repo history, same derivation as Run A); mass tolerance ±0.02 vs <0.001 measured
  σ_r-sensitivity and ≈0.007 sampling error.
  Blind spots: (a) R̂/ESS see only visited regions (T1's lesson) — the PRIMARY phenomenon
  check is the plot: 2-D (Om0, w0) sample overlay on `def_ratio_grid.png` must show the full
  arm to Om0=0; (b) errors common to BOTH parameterizations (e.g. a wrong deflection_ratio
  formula) are invisible, as in Run A; (c) the measured w0-degeneracy sliver (genuine shallow
  dr2/dw0 inversion, worst −4.128e-5, confined to Om0 ≥ 0.785, ≥15.5 σ_r,eff from the data
  contour, posterior mass ~1e-53; log-det has an integrable singularity on the dr2/dw0
  zero-crossing curve, float64-capped at ~46 nats vs ≥~120-nat likelihood penalty to reach
  it) is quantified for THIS dataset only — monitored via nonfinite-step count and max
  sampled Om0, not certified for other datasets; (d) single seed, single (seed-0) noise
  realization — same realization as Runs A/B, NOT the baseline notebook's unseeded one;
  (e) k=1 only: the multi-ratio u_fn combination path exists in code but is untested.
  Expected plot: all 8 Om0 traces oscillating over ~0–0.57 with free crossings of the former
  0.146–0.163 truncation edge; overlay reproduces the full grid band arc. If falsified:
  bounce at ~0.15 persists, or a NEW truncation appears (watch the w0=−2 tangency end,
  Om0≈0.54, where the band tail stretches in z2).
  Cost: one 8-chain 10k+10k MCLMC + 4000-step MAP on one interactive GPU node (≈ Run A).
  Seeds: data 0, MAP 1, MCLMC 10, pipeline 42 (all = Run A). Code: branch
  `ratio-coords-prior` — `src/gigalens_research/priors/ratio_coords.py` (+11 passing unit
  tests), `experiments/sample_cosmology/dspl_ratio_coords.py` (import-safe; `--run` requires
  `--confirm-run-c-approved` AND a passing gate JSON). Status: **approved by grader (user)
  2026-07-11 ("can you test this on the double source-plane system"); launched** via
  `run_dspl_ratio_coords.sh` (Run A launcher idiom, 1 interactive hbm80g GPU).
  **COMPLETED 2026-07-11 — predictions P1–P3 FAILED (P4/P5 hit): crest cured but mirror
  truncation at Om0≈0.385; see the "Run C outcome" entry below. Checkpoint cleared.**
  - *2026-07-11 update:* pre-run **equivalence gate PASSED** —
    `results/sample_cosmology/dspl_ratio_coords/ratio_coords_gate.json`
    (`dspl_ratio_coords_gate.py`, 64 matched prior draws, CPU): (1) matched-θ log-likelihood
    baseline-vs-grouped max rel diff **0.0** (bitwise; threshold 1e-8); (2) matched-θ prior
    log-density diff 0.0; (3) grouped-bijector FLDJ vs numeric slogdet max abs diff 1.33e-15
    (threshold 1e-8); (4) (Om0, w0) round-trip through the 80-iteration bisection exact to
    0.0 (threshold 1e-9). Validator report (autodiff of the actual u_fn, 201² grid):
    dominant sign +1, worst signed dr2/dw0 −4.128e-5 @ (Om0=0.88, w0=−2), 408/40401 grid
    flips ALL within du_dw_atol=1e-4 (derived: 2.4× measured worst, 100× below median
    |dr2/dw0|≈1e-2), max interior excursion 8.785e-7 within excursion_atol=3e-6 (3.4×
    measured; ≈1.3e-3 σ_r,eff). Flat-z-migration note: `dspl_arm_init.make_prob_model`
    predates the library's Dataset→ImageData rename and no longer runs; Run C carries its own
    migrated `make_prob_model` (used for BOTH models in the gate). Full-model gradient smoke:
    finite grad of log_prob through pixel likelihood + implicit solve, 29 ms/grad on CPU
    (solver overhead negligible).

- **Run D (DRAFT, awaiting approval): u-first ratio coordinates — global squash of the
  stiff scalar.**
  Hypothesis (from Run C's T3 mechanism below): Run C's residual disease is the ROTATION of
  the likelihood band in (z1, z2) (−8°→−84°), caused by squashing u into a bracket that
  drifts with Om0. Swapping the conditioning order — z1 → u squashed over its GLOBAL range
  [min u, max u] over the box; z2 → Om0 squashed into the u-contour's Om0-interval
  [Om0_min(u), Om0_max(u)] (endpoints by root-solve, implicit gradients); w0 = solve as in
  Run C — makes the likelihood a function of z1 ALONE: the band is an axis-aligned slab,
  zero rotation by construction, at every point of the posterior.
  Prediction: rank-R̂(Om0), R̂(w0) < 1.01; BOTH cosmology z-columns' bulk-ESS within 2× of
  the median nuisance ESS; mass(Om0 < 0.146) = 0.104 ± 0.02 (u*-corrected grid value, see
  Run C outcome); full arc visited INCLUDING both edges (per-chain min Om0 < 0.05 AND max
  Om0 > 0.50); 0 nonfinite steps.
  Falsifier: any cosmology z-column ESS < half the median nuisance ESS, or a visited-range
  edge short of [0.05, 0.50] ⇒ slab geometry is not sufficient ⇒ the rotation mechanism
  (T3) is not the (only) residual disease — stop and diagnose, do not re-fix.
  Threshold derivation: same standards as Runs A/C (R̂ 1.01 repo standard, ESS factor 2 =
  seed band, mass ±0.02 ⊇ binomial+σ_r sensitivity).
  Blind spots: prior-pushforward structure along z2 (density along the contour) is assumed
  mild — checked post-hoc from the samples; Om0-interval endpoints introduce two more
  root-solves per eval (validated by the same gate battery before running); same
  single-seed / single-realization limits as Run C.
  Cost: one 8-chain 10k+10k run + gate battery, ≈ Run C (~13 min wall on 1 interactive GPU).
  Status: **approved by grader (user) 2026-07-11 ("Go ahead with Run D!"); implemented;
  launched** via `run_dspl_ratio_ufirst.sh` after the amendment + gate below.
  **COMPLETED 2026-07-11 — truncation cured in both directions (P1/P3/P4/P5/P6 hit);
  P2's along-arc column technically fired the falsifier (efficiency residual, not
  correctness) — see the "Run D outcome" entry below. Checkpoint cleared.**
  - *2026-07-11 amendment (BANDED SUPPORT — science-level, quantified-immaterial):* the
    full-box u-first map is topologically impossible for the real u_fn — u has interior
    critical points (u_a(Om0)=r2(Om0,−2) dips to a minimum at Om0≈0.02, corner slope −4.7,
    before rising; u_b(Om0)=r2(Om0,−⅓) wiggles, 110/201 grid slopes negative), caught by the
    strict validator, so level sets change topology and "u as a global coordinate" cannot
    exist on the whole box. Adaptation: z1 squashes u into the BAND (u_a(0), min u_b) =
    (1.2867134, 1.3398900), inside which u_a is single-crossing (min signed slope above the
    band floor +0.0250, strict curve_atol=0 passes) and the u_b constraint never binds. The
    prior becomes an unnormalized TRUNCATED uniform: 11.1% of PRIOR box volume is excluded,
    but the band edges sit 55.5σ and 23.8σ from the data (u*=1.3239203, σ_r,eff=6.7e-4) —
    excluded posterior mass < 1e-125. Explicit support change, not a silent approximation.
  - *2026-07-11 mechanics fix (found by the new FLDJ-vs-jacrev unit test):* the om-root
    solver's implicit-vjp rule emitted ~1e15 garbage gradients when a constraint was
    INACTIVE (bisection clamps to the box edge; `max/min` then selected the clamped root
    over the constant). Fixed twice over: interval bounds are `jnp.where`-gated on concrete
    activity tests (inactive constraint → constant bound, no gradient path), and the shared
    solver's bwd rule zeroes itself at bracket-edge clamps. Analytic FLDJ was verified
    correct against finite differences (1e-10) before the fix — the bug was in autodiff
    gradients only. 19/19 unit tests (11 om-first + 8 u-first).
  - *2026-07-11 pre-run gate PASSED* —
    `results/sample_cosmology/dspl_ratio_ufirst/ratio_ufirst_gate.json`
    (`dspl_ratio_ufirst_gate.py`, 96 draws, 88 in-band = 8.3% dropped vs 11.1% grid
    estimate): (1) matched-θ log-likelihood bitwise equal; (2) prior log-density equal;
    (3) FLDJ vs numeric slogdet 2.2e-15; (4) round-trip exact; (5) NEW zero-rotation check —
    u(forward(z)) invariant under z2 shifts to 1.1e-15 (the construction claim, verified
    numerically on the real u_fn).

---

## 2026-07-11 — Run D outcome: banded u-first ratio coordinates — truncation CURED in both directions; P2 falsifier technically fired on the along-arc column (proposed UNCERTIFIED)

Evidence: `results/sample_cosmology/dspl_ratio_ufirst/ratio_ufirst_run_summary.json`,
`ratio_ufirst_traces.png`, `ratio_ufirst_overlay.png` (NOTE: this run's overlay PNG carries
an inherited "Run C" title/legend — labels fixed in the script afterward; the data is Run D),
`mclmc/`, `map/`; code `dspl_ratio_ufirst.py`, `_gate.py`, `_analysis.py`, launcher
`run_dspl_ratio_ufirst.sh` (wall: 811 s run + 48 s analysis on 1 GPU).

Observed vs predicted:

| Quantity | Predicted | Observed | Verdict |
|---|---|---|---|
| rank-R̂ Om0 / w0 (physical) | < 1.01 | **1.0087 / 1.0052** | hit |
| bulk-ESS cosmo z-cols vs median nuisance (3364) | ≥ half | u-col (z1) **3206 (0.95×)**; along-arc col (z2) **949 (0.28×)** | **z2 miss** |
| mass(Om0 < 0.146) | 0.103 ± 0.02 | **0.0899** | hit |
| full arc per chain (min < 0.05 AND max > 0.50) | all 8 | min 0.001–0.011, max 0.540–0.552 | hit |
| nonfinite-flagged steps | 0 | 0 / 160000 | hit |
| MAP χ²/ν | ≈ 1 | 0.9975 | hit |

**Phenomenon (plot-confirmed):** the truncation is gone in BOTH directions. All 8 chains
traverse the full arc dozens of times (Om0 0.001→0.552, w0 −0.93→−2.0); the overlay traces
the entire grid band including the Om0=0 edge and the w0=−2 plunge to the tangency. This is
the full DSPL cosmology posterior from ONE run of the FULL forward model — the campaign's
goal state. Also note nuisance ESS roughly doubled vs Run C (median 3364 vs 1830): the
rotating-band coupling was degrading the nuisance block too.

**P2 falsifier discipline:** as pre-registered, ESS(z2)=949 < half the nuisance median
technically fires the falsifier "slab geometry is not sufficient — stop and diagnose".
Reading: the slab construction fixed exploration/correctness (P3+P6+plots), and z1 — the
formerly pathological stiff direction — now mixes AT the nuisance median (0.95×); what
remains is ~3.5×-slower mixing ALONG the likelihood-flat arc direction, whose z2-marginal
carries the prior-pushforward density structure (visible in the traces as ~100-step Om0
sweeps). This is a mild efficiency residual, not a correctness defect, and it is the
diagnosable leftover the falsifier was written to catch. Candidate cheap diagnostic (T4,
not run): metric/L vs the z2-marginal shape from existing `mclmc/diagnostics.npz`.

**Proposed claim C-5 (UNCERTIFIED):** the banded u-first ratio-coordinates grouped prior
samples the full DSPL cosmology posterior correctly in a single full-forward-model MCLMC
run (all mass checks pass; both former truncations absent; 0 nonfinite steps), with
cosmology stiff-direction ESS at the nuisance median and a residual ~3.5× ESS deficit in
the along-arc coordinate only. Doubt report: (a) single seed / single noise realization
(shared with Runs A–C); (b) the banded-support amendment excludes 11.1% of PRIOR volume
(< 1e-125 posterior mass — quantified in the checkpoint amendment); (c) P3's grid
reference assumes the T1 constancy argument transfers across the ~1σ noise-realization
shift (Run A's reconstruction supports this); (d) z2-ESS deficit mechanism is untested
(correlational reading above). Grader: pending (inspect the two PNGs + summary JSON).

**T5 — transverse-width verification (grader question: samples off-center, almost all
inside the plotted 68% band, no outer spill — bijector bias?).** Verified and resolved as
a PLOT artifact, not a sampler/bijector defect (code
`dspl_ratio_ufirst_transverse_check.py`, `dspl_ratio_ufirst_overlay_fair.py`; evidence
`ratio_ufirst_overlay_fair.png`):
- Run D's transverse marginal (u, analytic from z1 — no solves) is Gaussian to the 2.3%/97.7%
  quantiles: standardized quantiles [−1.98, −1.00, 0.00, 0.99, 2.00]; mean 1.3236072,
  std 5.23e-4 over all 80k samples.
- **Cross-parameterization check (the decisive one):** Run A sampled the SAME seed-0 dataset
  with free r2 (no cosmology block, different coordinates entirely): mean 1.3235776,
  std 5.56e-4 — Run D agrees to **+3.0e-5 = 0.04 σ** in the mean; widths within 6%
  (≈2–3× the naive ESS-based scatter — borderline; single-seed caveat).
- The original overlay's contours use the grid DISPLAY σ = 1.32e-3 (σ_frac=0.001), 2.5× the
  actual per-dataset width ~5.2–5.6e-4 (same mismatch Run A's outcome noted), and this
  realization's ML sits −5.58e-4 = −0.42 σ_display below r2_truth. Quantified against the
  plotted 68% band: 93.0% of samples inside, 6.9% spill INNER side, 0.03% outer — exactly
  the observed asymmetry. Against bands recomputed at Run A's independent (mean, σ)
  (`ratio_ufirst_overlay_fair.png`) the cloud is centered and fills the 68% band
  symmetrically along the whole arc.
- Note for future overlays: the earlier σ_r,eff = 6.7e-4 (from the truncated baseline run's
  cross-ridge width) is also ~20% wider than the per-dataset widths measured here.

---

## 2026-07-11 — Run C outcome: ratio-coordinates grouped prior — crest cured, but predictions FAILED; mirror truncation at the w0=−2 arm (proposed UNCERTIFIED)

Evidence: `results/sample_cosmology/dspl_ratio_coords/ratio_coords_run_summary.json`,
`ratio_coords_traces.png`, `ratio_coords_overlay.png`, `mclmc/`, `map/`; code
`dspl_ratio_coords.py`, `dspl_ratio_coords_analysis.py`, launcher
`run_dspl_ratio_coords.sh` (wall: 734 s run + 37 s analysis on 1 GPU).

Observed vs predicted:

| Quantity | Predicted | Observed | Verdict |
|---|---|---|---|
| rank-R̂ Om0 / w0 (physical) | < 1.01 | **1.0156 / 1.0158** | **miss** |
| bulk-ESS cosmo z-cols vs median nuisance (1830) | ≥ half, ~within 2× | **499 / 491 (0.27×)** | **miss** |
| mass(Om0 < 0.146) | 0.103 ± 0.02 | **0.282** | **miss** |
| nonfinite-flagged steps | 0 | 0 / 160000 | hit |
| MAP χ²/ν | ≈ 1 | 0.9972 | hit |

**Phenomenon (plot-confirmed, traces + overlay):** the old disease is CURED — all 8 chains
cross the baseline's 0.146–0.163 truncation edge freely (dozens of round trips; per-chain
min Om0 0.032–0.040; the arm Run B showed to be one-way is now routine) — but a MIRROR
truncation appears at the other end: per-chain max Om0 0.358–0.385 across all 8 chains,
while the posterior's right arm (the w0 plunge to the w0=−2 tangency at Om0≈0.52) is
unvisited; w0 never goes below −1.10.

**Mass accounting (kills a density-bug explanation):** recomputing the grid posterior at
the run's own u* = r2(MAP) = 1.3239203 (σ_r,eff = 6.7e-4): true mass(Om0 < 0.146) = 0.104,
true mass(Om0 > 0.385) = **0.627**, and the conditional mass(< 0.146 | Om0 ≤ 0.385) =
**0.280 vs the run's measured 0.282**. The sampler reproduces the restricted density to
3 parts in 1000 — the map, Jacobian, and solver are exact in practice (consistent with the
bitwise gate); the failure is pure mobility: ~63% of the posterior, a |Δz| ≈ 1 walk away,
is never entered in 8×20k steps.

**T3 — mechanism (UNCERTIFIED, from cheap contour arithmetic, `band_geom` check):** in
(z1, z2) the likelihood band's tangent rotates continuously from −8° (Om0=0.03) through
−60° (Om0≈0.385, the observed truncation edge) to −84° (Om0=0.515), because the
conditional bracket [u(Om0,−2), u(Om0,−⅓)] drifts with Om0 (t = position of u* in the
bracket falls 0.76→0.04); cross-band width simultaneously grows 15× (0.028→0.43). MAP sat
at Om0=0.111 (near-horizontal band, −8°..−18°), so the burn-in metric adapted to that
orientation — the same frozen-global-metric-vs-rotating-ridge disease as T2/C-3
(playbook (iv)), relocated: the reparameterization removed the crest kink and the
semi-infinite left tail but NOT the band rotation. Doubts: (a) rotation-vs-truncation
colocation is correlational — no arm-init falsification run performed (a Run-B-style
mirror test, init at Om0=0.48, would discriminate); (b) single seed; (c) metric history
not yet inspected (`mclmc/diagnostics.npz` has the full inverse-mass-matrix history —
a T2-style alignment analysis is the cheap next diagnostic).

**Proposed claim C-4 (UNCERTIFIED):** the ratio-coordinates grouped prior samples the DSPL
posterior with exactly correct restricted density and cures the crest/left-arm truncation,
but under a single frozen global metric it truncates the w0-plunge arm (63% of mass) —
the conditional-bracket construction leaves a rotating band; a global (non-conditional)
squash of the stiff scalar (Run D draft above) is the structural candidate fix.
Grader: pending (inspect the two PNGs + summary JSON).

Mechanics note (fixed during launch): under the sharded MCLMC/MAP kernels the bisection's
`fori_loop` carry must inherit the inputs' sharding/varying type — seeding it from plain
constants fails inside `shard_map` (`ratio_coords.py::_bisect`, fixed + unit-tested; first
GPU launch aborted in MAP with the carry-type error, no artifacts written).

---

## 2026-07-07 — T1: grid-search overlay proves low-Om0 arm is a sampler failure (UNCERTIFIED cause, CONFIRMED phenomenon)

**Structural fact (code-verified, both API worlds):** cosmology enters the pixel
likelihood only through the scalar r2 = deflection_ratio(z_source2); source 1
sits at the reference plane (ratio ≡ 1). With uniform priors on Om0, w0, the
exact marginal 2-D posterior p(Om0, w0) ∝ h(r2(Om0, w0)) — **constant along
level contours of r2**. Any density difference along one contour in a sampled
posterior is a sampler artifact, with no nuisance-marginalization caveat.

**T1 (regenerated grid at the correct z_lens=0.5 + overlay of the actual MCLMC
run):** script `experiments/sample_cosmology/def_ratio_grid.py`; evidence
`results/sample_cosmology/dspl_cosmology_newapi/def_ratio_grid_overlay.png`,
`def_ratio_grid.png`, `def_ratio_grid.npz`. (The original
`def_ratio_likelihood.py` hardcodes z_lens=0.4 — its plot is for a different
system; superseded for this area by `def_ratio_grid.py`.)

Findings:
- r2(truth) = 1.3241652; r2 spans only [1.2645, 1.3432] over the whole prior
  box (~6% — deflection-ratio likelihood is highly informative).
- The r2(truth) contour is one continuous arc from the Om0=0 edge (w0≈−1.27)
  over a high-curvature **crest** near (Om0≈0.2, w0≈−0.9) down to tangency with
  the w0=−2 edge at Om0≈0.54. The grid posterior bands trace it fully.
- All 8 MCLMC chains sit tightly on the contour (sampled r2 ∈ [1.3218, 1.3271])
  but cover only Om0 ∈ [0.146, 0.574]. **The segment from the crest down to
  Om0=0 is unvisited by every chain**; per-chain truncation edges agree to
  ~0.02 (0.146–0.163).
- By the constancy argument, the missing arm has the same posterior density as
  the sampled segment ⇒ **sampler failure, phenomenon confirmed**.
- **Metric blind spot (negative result worth keeping):** rank-R̂ Om0=1.038,
  w0=1.046, bulk-ESS≈200/80k — below the 1.1 alarm while an entire posterior
  arm is missing, because all chains truncate at the same place (chains were
  initialized from a 1e-3 ball at the single MAP point). R̂ cannot detect a
  region no chain visits; cf. playbook §3.14 truncation bias.

**Proposed causes (UNCERTIFIED, per playbook disease catalog):** curved-valley
parameterization (iv) and/or marginal-vs-conditional neck + tuner suppression
(v) at the crest, where the ridge tangent rotates fastest and the NormalCDF
Jacobian starts stretching z_Om0; init trap (ii) contributes exposure (single
common start point). Mechanism analysis from existing artifacts (traces, xi vs
position, metric-vs-ridge geometry) in progress:
`experiments/sample_cosmology/dspl_mclmc_mechanism.py` (bounce-at-neck vs
slow-diffusion discrimination).

**Not yet done:** pre-registered fix. Leading structural candidate if a
neck/curved valley is confirmed: exact reparameterization of cosmology to
(r2, arc-coordinate) — analytic analog of the playbook Route-A transform —
to be design-checkpointed before any run.

---

## 2026-07-07 — Run A outcome: free-r2 reparameterization — ALL pre-registered predictions HIT (proposed UNCERTIFIED)

Checkpoint A cleared. Observed vs predicted (evidence:
`results/sample_cosmology/dspl_free_r2/r2_reconstruction.png`,
`r2_reconstruction_summary.json`, run log; code `dspl_free_r2.py`,
`dspl_r2_reconstruct.py`):

| Quantity | Predicted | Observed | Verdict |
|---|---|---|---|
| rank-R̂(r2) | < 1.01 | **1.0015** | hit |
| bulk-ESS(r2) | within 2× of median nuisance | **3027 vs 3080 (1.02×)** | hit (15× the baseline cosmology ESS≈200) |
| mass(Om0<0.146) reconstructed | 0.103 ± 0.02 | **0.1062** (gap +0.003) | hit |
| worst nuisance | — | R̂ 1.0022, ESS 2833 | healthy |

Plot inspected (producer): all 8 r2 traces overlapping/stationary; reconstructed
p(Om0,w0) contains the FULL arm to Om0=0. Benign observed details: sampled r2
centers ~1σ below r2(truth) (seeded noise realization differs from the
baseline's unseeded one — ML shift within σ_r); reconstruction bands narrower
than the grid plot (σ_r,eff 6.7e-4 vs the grid's display default 1.3e-3).

**Proposed claim C-2 (UNCERTIFIED):** sampling plane-2 deflection_ratio
directly makes the DSPL cosmology posterior trivially sampleable, and the
analytic reconstruction p(Om0,w0) ∝ ĥ(r2(Om0,w0)) recovers the full posterior
including the ~10% arm the (Om0,w0)-parameterized run truncates. Doubt report:
(a) reconstruction assumes "only via r2" — independently verified by the
equivalence gate (1.5e-14); (b) different noise realization than baseline —
affects posterior center (~1σ), not the sampleability comparison, which is a
geometry property; (c) single seed — ESS numbers carry seed-band uncertainty
(~1.4× repo experience), far from the 15× margin observed. Grader: pending
(inspect the PNG + summary JSON).

---

## 2026-07-07 — Run B outcome: arm-init frozen-metric — falsifier NOT triggered; mechanism CONFIRMED with an amendment (proposed UNCERTIFIED)

Checkpoint B cleared. Evidence:
`results/sample_cosmology/dspl_arm_init/arm_init_traces.png`, `samples_z.npz`,
run log; code `dspl_arm_init.py`, `dspl_arm_init_analysis.py`.

Observed vs predicted:
- Crossings (pre-registered: few-to-zero; falsifier mean ≥ 3/chain): per-chain
  [1,3,1,1,1,1,1,0], mean **1.125** — falsifier not triggered. Chain 1's "3"
  is boundary jitter at the 0.163/0.25 lines, not deep round trips.
- "Crossers stay in the bulk": HIT — after escape, no chain re-enters below
  ~0.16 (the baseline's own truncation edge reappears from the other side).
- Arm occupancy 0.76 — but see amendment: this is transient, not equilibrium.
- Bonus likelihood-level check: profile-MAP at (Om0=0.05, w0=−1.011) reached
  **χ²/ν = 0.998** — the arm fits the data exactly as well as the bulk,
  independently corroborating C-1's constancy argument.

**Amendment (plot beats table):** the traces show the dominant phenomenon is
not bounce-at-a-wall but **arm-wide mobility collapse with a systematic
bulk-ward drift**: every chain crawls smoothly up the arm (~100× slower
per-step progress than bulk dynamics), escapes once at a staggered time
(chain 2 @ ~4.2k steps … chains 5/6 @ ~9.4k), then mixes vigorously in the
bulk and never returns. Chain 7 (deepest init, Om0 0.043–0.066 for all 10k
steps) never escaped — the arm tip, where T2's geometry found the true width
minimum, is the extreme of the same pathology. Under the frozen bulk-tuned
metric the arm is one-way and effectively unlivable in equilibrium — a
*stronger* version of the pre-registered soft-barrier picture, same causal
story (global metric mis-tracks the rotating, z-stretched ridge; playbook
disease (iv) + (v)).

**Proposed claim C-3 (UNCERTIFIED):** the baseline truncation is caused by the
single frozen bulk-tuned global metric: under it, the arm below Om0≈0.15 is a
low-mobility, net-outflow region (arm→bulk evacuation, no re-entry in 8×10k
steps), so the sampler's stationary distribution effectively excludes the arm.
Doubt report: (a) tests the baseline's metric only — an arm-tuned or adaptive
metric might traverse freely (untested); (b) one seed; (c) the slow crawl's
direction could include an init-relaxation component (profile-MAP nuisances at
fixed cosmology may sit slightly off the full-model ridge) — but the crawl
persists for thousands of steps and its endpoint (bulk residence, no return)
matches the baseline's equilibrium, so relaxation alone cannot explain it.
Grader: pending.

---

## 2026-07-07 — T2: mechanism from existing artifacts — soft reflection barrier from metric-direction mismatch (UNCERTIFIED)

Script `experiments/sample_cosmology/dspl_mclmc_mechanism.py`; evidence
`results/.../mech_traces.png`, `mech_xi_vs_om0.png`, `mech_ridge_geometry.png`.
All from the existing run's `diagnostics.npz` (per-step 8×20000 histories of
step_size/L/xi/nonan incl. burn-in; single shared mass-matrix history) — no new
sampling.

**Bounce, not diffusion.** All 8 chains traverse the visited arc
(Om0 0.15–0.57) many times in 10k steps — within-segment mixing is healthy —
and are turned back at a tightly consistent Om0 ≈ 0.146–0.163 across ~64
independent approaches (5–12 excursions below Om0=0.25 per chain, durations
~60–320 steps, immediate reversals at the floor). Inconsistent with
"hasn't diffused there yet".

**Mechanism evidence (soft barrier):**
- Final shared global metric (frozen after burn-in; step_size 0.1528, L jumped
  4.583→41.44 at the burn-in→results handoff): its cosmo-block long axis is
  aligned with the ridge tangent to **2.9° in the bulk**, but mis-tracks by
  **28° at the crest (Om0≈0.2)** and **34.5° at the turnaround (0.163)** — the
  turnaround sits exactly where the misalignment grows.
- Ridge tangent rotates 54.1° (z-space, unwrapped) from Om0=0.35→0.05.
- Contrary/nuancing evidence: z-space ridge *width* narrows only ~20–25%
  (crest/bulk 0.805) — no order-of-magnitude pinch at the turnaround; the
  geometric minimum width sits deeper (Om0≈0.02–0.03), inside the unvisited
  region. xi (energy error) is elevated at BOTH ends of the visited range
  (V-shape vs Om0; 99th-pct ratio low/high only 1.09) — generic turning-point
  signature, not uniquely low-Om0. 0/160000 NaN-flagged steps.
- Structural aggravator: the r2(truth) contour terminates ON the Om0=0 prior
  edge (w0≈−1.27), so in NormalCDF-unconstrained space the arm is a
  semi-infinite, progressively stretching, rotating tail (z_Om0→−∞). A single
  frozen global Gaussian metric cannot cover it at any tuning — this marks the
  (Om0, w0) sampling coordinates themselves as the disease class
  (playbook (iv), with a mild (v) component), not the tuner.

**Mass at stake:** the unvisited arm carries **~10–12%** of the grid posterior
(10.3% below Om0=0.146; 11.7% below 0.163; from `def_ratio_grid.npz`,
σ_frac=0.001). The run's cosmology posterior is missing that mass entirely;
its Om0 marginal is biased high and w0∈(−1.27,−0.9) left-arm support is absent.

**Minor anomaly (unexplained, logged):** chain 0 goes quiescent for the last
~2000 result steps (slow directed w0 crawl ≈ −1.65→−1.5) — visible in
`mech_traces.png`; not investigated.

**Proposed next steps (design-checkpoint drafts, need grader approval):**
- **A (structural fix, science route):** sample plane-2 geometry as a free
  `deflection_ratio` parameter (new API supports it natively; prior = Uniform
  over [1.2645, 1.3432] or the pushforward of Uniform(Om0,w0)); reconstruct
  p(Om0, w0) analytically on a grid via
  p ∝ posterior_r(r2(Om0,w0)) / prior_r(r2(Om0,w0)). Prediction: r2 samples
  like any bounded scalar (R̂<1.01, ESS≳5k); reconstructed posterior contains
  the full arm by construction and matches the grid-search shape with
  σ_r,eff≈6.7e-4. Falsifier: reconstructed posterior fails to be constant
  along r2 contours ⇒ the "cosmology enters only via r2" claim is wrong.
- **B (mechanism falsification, sampler-research route):** arm-initialized run
  — profile-MAP nuisances at fixed (Om0,w0)≈(0.05, −1.21) on-contour, start
  chains there, standard config. Prediction if soft-barrier picture is right:
  chains evacuate over the crest to the bulk within ~2k steps and do not
  return. If they stay and mix locally: two-domain metastability, mechanism
  claim needs revision.
