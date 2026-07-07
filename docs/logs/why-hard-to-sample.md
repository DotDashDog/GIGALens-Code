# Lab Notebook — Why are lensing posteriors so hard to sample?

Standing scientific question (cross-system, not a per-run diagnosis): gigalens lensing
posteriors sample far more slowly than literature benchmarks would predict, *even when the
converged posterior looks nearly Gaussian*. This log holds the hypothesis space and a
pre-registered decision-tree of discriminating experiments.

**Last updated:** 2026-07-01

> One log per rough research area (see `../../AGENTS.md` → *The record*). Every claim below is
> `proposed (UNCERTIFIED)` unless graded otherwise. Producers may not self-certify; the grader
> inspects artifacts, not summaries.

---

## Current state

Hypothesis space defined and experiments T0–T5 pre-registered below (status: awaiting
approval). Nothing has been run yet under this log. The experiments are ordered as a decision
tree — T2/T5 are free post-processing, T0 calibrates thresholds, T1 (Gaussian clone) is the
flagship split of the hypothesis space. Do not skip to T3/T4 before T1's outcome is known.

## Established context (inputs to this investigation)

Reported observations from the human (source: working experience across systems, 2026-07-01
conversation — verbatim statement in the *Primary source* appendix at the bottom of this file;
O1–O6 are a distillation of it. Treat as trusted context but note most lack a single citable
artifact):

- **O1.** Converged posteriors almost always look very Gaussian in θ-space cornerplots.
- **O2.** Despite O1, sampling is much harder than in cited literature examples (low min-ESS,
  slow mixing). E.g. the carousel run after the MAP fix: min bulk-ESS ≈ 12/16000
  (`carousel-mclmc-sampling.md`, UNCERTIFIED).
- **O3.** "Banana-like curvature" has been repeatedly *proposed* by agents as the cause but
  never definitively demonstrated. Closest attempts were Hessian-based diagnostics, which come
  out rough and unstable, with no consistent trend; point Hessians appear unrepresentative of
  the posterior even near the typical set.
- **O4.** Proposed remedies aimed at global geometry (normalizing flows, tempering, exotic
  samplers) don't help much or actively hurt (see `anti-patterns.md` AP-6).
- **O5.** A float32 basis/convolution noise floor demonstrably broke MCLMC adaptation at high
  `n_max`; float64 fixed the adaptation collapse (`project-standards.md` §8). It is *unknown*
  whether float64 eliminated the small-scale noise floor or merely lowered it.
- **O6.** Prior diagnosis attempts have rabbit-holed; suspicion has fallen at various times on
  bijectors and numerical instabilities without resolution.

**Key reframe (proposed, drives H1):** O3 is evidence, not a failed diagnostic. For a smooth
quasi-Gaussian posterior, point Hessians near the typical set should be stable and consistent.
"Rough, unstable, no trend" is the expected signature of a log-likelihood with small-scale
non-smoothness on top of a large-scale Gaussian bowl — macro-geometry Gaussian (what
cornerplots see), micro-scale corrugated (what Hessians, gradient samplers, and energy-error
control see).

## Hypothesis space

- **H1 — Micro-roughness of the effective likelihood** (prime suspect). The lstsq amplitude
  marginalization (hundreds of linear params solved per evaluation) plus simulator internals
  generate small-scale roughness: Gram-matrix conditioning that shifts with the nonlinear
  params, gradients via VJP through the solve, possible profile-vs-marginal mismatch (is the
  Gram log-det in the objective?), guard ops (`clip`/`where`/`maximum`) creating kinks. O5 is a
  confirmed instance of this *class*. Explains O1+O2+O3+O4 simultaneously, and why literature
  comparisons fail (published benchmarks are smooth analytic posteriors; none contain an
  ill-conditioned linear solve inside the likelihood).
- **H2 — Rotated ill-conditioning.** The posterior is essentially Gaussian but with a large
  condition number whose stiff/soft directions are *combinations* of parameters (NFW
  parameterization already implicated in `carousel-mclmc-sampling.md`). Marginals hide the
  elongation; preconditioning fails to represent it. "Gaussian but hard" with no curvature.
- **H3 — Bijector distortion / diagnosing in the wrong space.** Sampling happens in
  unconstrained z-space; all routine diagnostics are θ-space. Box priors + sigmoid-type
  bijectors turn bound-adjacent mass into heavy tails/funnels in z that θ-cornerplots cannot
  show. Also possible saturation/gradient-plateau numerics near bounds.
- **H0 — Macro-scale curvature ("banana") — disfavored.** Repeatedly proposed, never
  demonstrated (O3). Not dismissed by fiat: T3's 1-D transects give a definitive macro-scale
  curvature verdict as a by-product, immune to the high-D instability that made Hessian
  diagnostics inconclusive. Until then, do not cite H0 as a cause in any log.

Why O4 (flows/tempering fail) fits: those tools treat global structure — multimodality,
large-scale curvature. Under H1/H2/H3 they address a disease we don't have and add their own
failure modes. Their failure is weak evidence *for* H1/H2/H3.

## Claims register

### C-1 — Macro-scale banana curvature is the dominant cause of slow sampling
- **Status:** `proposed (UNCERTIFIED)` — **now supported by a direct artifact on sys60**
  (2026-07-02 T3 entry): non-parabolic, non-monotonic macro transects (random_2:
  −31,500-nat cliff at +1.5σ then recovery) + local-vs-σ-scale curvature ratio ~100× with
  curvature correlation length ~0.07σ, on a SMOOTH (T3-verified) log-density; causally
  isolated by T1 (Gaussian clone with identical covariance = 130× faster). Scope: one system
  (sys60), one typical-set point, 7 directions. Prior "disfavored" standing was for lack of
  artifact; that objection is resolved for sys60. Cross-system replication open.
- **Evidence / artifact:** `experiments/why_hard_to_sample/results_t0t1/sys60/t3/float64/`
  (t3_macro, t3_D2_vs_h) + `report/t1_report.json`.

### C-2 — Hessian instability near the typical set indicates micro-scale non-smoothness (not high-D noise)
- **Status:** **WITHDRAWN (2026-07-02)** per the pre-registered T3 falsifier: float64
  second-difference curvature converges and stays stable down to ≪ ε_step/100 on all 7
  transect directions (float32 control blew up at its predicted scale, validating
  sensitivity). O3's Hessian instability is re-explained WITHOUT micro-roughness: the
  Hessian field is smooth but varies ~100× over ~0.07σ scales, so point Hessians are
  honestly unrepresentative. The *observation* O3 stands; this causal reading of it is dead.

### C-3 — sys60's sampling difficulty is governed by a single compact counter-image ("counter-image lever")
- **Status:** `proposed (UNCERTIFIED)` (2026-07-02; T10+T11 + data check). The GN stiffness
  is carried almost entirely (L≈0.999, top 1% of pixels) by the compact lensed counter-image
  at pixel (row 57, col 12) — a real observed feature (peak 15.3 vs bg rms 0.2 ≈ 75σ, ~13%
  of the main image's peak) — at spikes AND baselines alike; λ1 spikes are ×10–500
  sensitivity flares of that same feature, which the chain meets on-ridge (T10: rate 11%,
  max/median up to 562, widths 1–3 steps typical). Proposed micro-mechanism (untested):
  sub-pixel crossings of the compact image's steep flank. Cross-system corollary (testable,
  E3): systems with bright compact counter-images should be the slow ones.
- **Evidence:** `results_t0t1/sys60/t10/`, `t11/` + observed-image check (logged 2026-07-02).

### C-4 — The λ1 spike field is supersample-grid aliasing of the counter-image ("subgrid comb"); the computed likelihood is drastically stiffer than the intended one
- **Status:** `proposed (UNCERTIFIED)` (2026-07-02, T12). λ1 vs measured counter-image
  displacement is a comb of narrow teeth at pitch ≈0.52 px = the supersample=2 subgrid
  spacing (recovered dominant frequency 1.90–1.91/px on both spike scans), spanning up to
  ×30,000, riding a smooth (caustic-proximity) envelope. Rebuilt at supersample=4 the comb
  collapses (top tooth 6.0e7 → 4.2e4, ×1400; peak-to-trough 30645 → 4.7) leaving the mild
  envelope ⇒ the giant spikes are RENDERING-FIDELITY artifacts of ss=2, not physical
  pixel-integration information. Census phase regression: ΔR² = 0.563 (≥0.4 pre-registered).
  This is the refined-H1 revival: T3 was right that the ss=2 log-density is *smooth*, and
  the original H1 intuition was right in substance — simulator internals (finite rendering
  quadrature of a compact image) create small-scale likelihood structure that no published
  benchmark contains; the texture is a smooth comb, not noise. O5 (float32 conv floor) was
  the same disease at a different fidelity knob.
- **Immediate remedy candidate (needs its own checkpoint):** raise supersampling (ss=4, ~4×
  render cost) → predicted standard-config min-ESS rises from ~13 toward the clone's ~1700.
- **Evidence:** `results_t0t1/sys60/t12/` (scans figure = the comb + ss4 overlay).
- **SCOPE AMENDMENT (2026-07-02, T13′):** the comb exists in the ss=2 model's Fisher metric
  unconditionally (comb-identity check on accurate data: 1.908/px, peak-to-trough 37,046 —
  unchanged), but it slows sampling ONLY in the SELF-CONSISTENT pairing (ss2 data + ss2
  model), where the model tracks the data at comb scale and the posterior inherits the
  comb (E1's GN-dominance was measured exactly there). On accurate (ss=128) data, the ss2
  model's 13.9σ counter-image misfit decouples the posterior from the comb and sampling is
  FAST (min-ESS ≈ 1100) — fast but BIASED (systematic misfit). Consequence: C-4's sampling
  pathology is specific to synthetic benchmarks generated and fit at the same low render
  fidelity; it does NOT explain slow sampling on real-data systems (e.g. the carousel),
  whose hardness needs a different mechanism.

*(Register further claims as experiments produce them; withdraw explicitly when killed.)*

---

## Design checkpoints (criteria awaiting approval)

Execution rules for whichever agent picks these up: (1) run `/pre-run-checklist` mechanics —
each checkpoint below must be graded/approved before its run; (2) after each run, log observed
vs. predicted *magnitude* — a bad miss means the hypothesis failed even if the direction was
right; (3) thresholds marked *derived-by-T0* may not be invented — T0 must run first;
(4) negative results go in the Log section; they are the point of this design.

### T0 — Seed-variance calibration (prerequisite; hypothesis-free)
- **Purpose:** derive the thresholds T1/T4 need. Run the standard MCLMC config
  (8 chains / 2000 burn-in / 2000 results) on one already-characterized system (carousel
  minimal case or equivalent) for N≥4 seeds, identical otherwise.
- **Metric:** spread of min bulk-ESS and max rank-R̂ across seeds; report the full set, not a
  summary.
- **Expected appearance:** min-ESS varying within a modest band (guess: factor ≲2–3 across
  seeds). If the band spans an order of magnitude, that is itself a finding (ESS estimates on
  this posterior are not stable enough to threshold on, and T1/T4 criteria must be redesigned —
  stop and surface to grader).
- **Blind spot:** one system only; assumes its seed-variance is representative.
- **Cost:** N× the standard run on one system. **Status: awaiting approval.**

### T1 — Gaussian-clone test (flagship; splits H2 from {H1, H3})
- **Cause hypothesis (H2):** conditioning/parameterization as seen in z-space is *sufficient*
  to explain the slowness; no fine structure needed.
- **Design:** from the best available converged run, fit mean + full covariance to the samples
  **in z-space**. Define a synthetic target = that Gaussian logpdf (no lensing simulator, no
  bijectors — the clone lives natively in z). Run the identical MCLMC pipeline (same chains,
  adaptation, budget) on it.
- **Prediction if H2 dominant:** clone min-ESS falls within the T0 seed-variance band of the
  real run's min-ESS (same order of slowness).
- **Falsifier for H2-sufficiency:** clone min-ESS exceeds the real run's by more than the
  T0-derived band (direction: clone much easier) → conditioning alone cannot explain the
  slowness; the hardness lives in the true log-density's fine structure (H1) or its
  non-Gaussian z-space shape (H3). T2/T3 distinguish those.
- **Metric blind spot:** the clone is built from samples of a run we believe converged; if
  that run was secretly unconverged, the covariance (and hence the clone's difficulty) is
  wrong. Mitigation: state which run was used and its max rank-R̂ / min-ESS in the log entry.
  Also blind to non-Gaussian *tails* by construction — that is intentional (it is the control).
- **Expected appearance:** trace plots + ESS table for clone vs. real, side by side.
- **Cost:** trivial compute (Gaussian logpdf target); pipeline-plumbing effort only.
  **Status: awaiting approval.**

### T2 — z-space diagnostics (free post-processing; tests H3)
- **Cause hypothesis (H3):** the pathology is visible only in the sampled (unconstrained)
  space; θ-space diagnostics are structurally blind to it.
- **Design:** for existing converged runs, no new sampling: cornerplot and per-parameter ESS
  **in z-space**; identify worst-ESS directions in z; per parameter, the fraction of posterior
  mass in the outer 1% of its prior box (bound-crowding).
- **Prediction if H3 real:** worst-ESS parameters are bound-crowded and/or z-marginals show
  heavy tails/funnel shapes absent from their θ-marginals.
- **Falsifier:** z-marginals as Gaussian as θ-marginals and bound-crowding ≈ 0 for the
  worst-ESS parameters (threshold: a posterior comfortably inside the box puts ~0 mass in the
  outer 1%; anything ≳ a few % is crowding — derive exact number from the actual histograms
  before judging, and record it).
- **Metric blind spot:** joint (multi-parameter) z-space structure that 1-D/2-D views miss —
  same limitation as θ cornerplots.
- **Cost:** none (post-processing). **Status: awaiting approval.**

### T3 — Multi-scale transect scan (tests H1 directly; settles H0 as by-product)
- **Cause hypothesis (H1):** the log-density is smooth at macro scales but non-smooth below
  some scale h\*, with h\* comparable to the sampler's step size.
- **Design:** through a typical-set point of a converged run, take 1-D transects of the exact
  log-density along (a) a few random z-directions, (b) the stiffest and softest covariance
  eigendirections. Evaluate at geometrically decreasing spacings h (macro → well below the
  adapted step size). Compute: log-density curves; second-difference curvature estimates vs.
  h; autodiff-vs-finite-difference gradient agreement vs. h. Both float64 and (one transect)
  float32 for contrast — O5 says float32 must show the floor; it is the positive control that
  the diagnostic *can* see roughness.
- **Prediction if H1 real:** second-difference curvature converges as h shrinks at macro
  scales, then *diverges* (non-convergent, erratic) below some h\*; h\* within ~an order of
  magnitude of the adapted step size; FD/autodiff agreement degrades at the same scale.
- **Falsifier:** curvature estimates converge to stable values for all h down to well below
  the step size (say 100× smaller) on all transects → surface is smooth at sampler-relevant
  scales; H1 is dead along the tested directions, and C-2 must be withdrawn.
- **H0 by-product:** the macro-scale portion of the same transects (and 2-D log-density slices
  along the worst eigenplane if desired) give a definitive curvature verdict, immune to
  point-Hessian instability. Stable near-quadratic transects → withdraw C-1.
- **Metric blind spot:** transects sample a handful of directions out of ~32 dims; roughness
  confined to unvisited directions escapes. Partial mitigation: include the worst-ESS
  direction from T2 among the transect directions.
- **Expected appearance:** curvature-vs-h on log axes — flat plateau (smooth) vs. blow-up
  below h\* (rough); the float32 control must blow up.
- **Cost:** hundreds–thousands of likelihood evaluations; minutes–hours, no sampling.
  **Status: awaiting approval.**

### T4 — Amplitude-marginalization ablation (tests H1's *mechanism*)
- **Cause hypothesis:** the lstsq linear-solve layer is the source of the fine structure.
- **Design:** freeze linear amplitudes at truth (synthetic system) or MAP values; sample the
  nonlinear parameters only, standard config. Compare min-ESS to the unfrozen baseline.
- **Prediction if the lstsq layer is the mechanism:** min-ESS improves by more than the T0
  band — expected order of magnitude, not tens of percent.
- **Falsifier:** min-ESS unchanged within the T0 band → lstsq layer exonerated as the
  *dominant* mechanism (roughness, if T3 found it, comes from elsewhere in the simulator).
- **Companion code audit (free, do alongside):** (a) is the amplitude treatment profiling
  (plug-in lstsq) or true marginalization — is the Gram log-det term present in the sampled
  objective? Document which, with file/line. (b) grep the likelihood path for
  `clip` / `where` / `maximum` / `nan_to_num`-style guards; list each with whether it can
  create a kink or plateau inside the typical set.
- **Metric blind spot:** freezing amplitudes also shrinks the posterior (removes
  amplitude-nonlinear degeneracies), which improves ESS for reasons other than smoothness —
  so a *positive* result overstates the mechanism; interpret jointly with T3. A *negative*
  result is the cleaner signal.
- **Cost:** ~2 standard runs on one system. **Status: awaiting approval.**

### T5 — Eigenspectrum and rotation read (free post-processing; quantifies H2)
- **Design:** from converged z-space samples: covariance eigenspectrum (condition number),
  and for the stiffest/softest directions, their participation across parameters (axis-aligned
  vs. rotated combination). Compare against what the run's preconditioner (inverse mass
  matrix) could actually represent (diagonal? full?).
- **Prediction if H2 significant:** condition number ≳10⁴ with stiff directions strongly
  rotated, and the preconditioner form unable to capture them.
- **Falsifier:** modest condition number (≲10²) or axis-aligned stiffness that the existing
  preconditioner already represents.
- **Blind spot:** covariance is a global second moment — says nothing about local structure
  (that's T3's job).
- **Cost:** none (post-processing). **Status: awaiting approval.**

### T6 — Fisher-metric survey (E1: localize the curvature in full dimension)
- **Cause hypothesis:** the T3 curved-valley finding = the stiff subspace of the local
  Gauss–Newton metric M(z)=JᵀWJ (J = image Jacobian, W = 1/err²) ROTATES on scales ~h*
  (≈0.07σ) as the point moves along the typical set; the curled subspace is a consistent,
  nameable parameter family (expected: θ_E–γ–source combinations).
- **Design (sys60):** ~32 seeded typical-set points from the converged reference run for
  loading statistics, plus same-chain lag pairs (lags ~1,4,16,64,256,1024,4096) giving
  on-ridge separations spanning ~ε_move…σ. At each point: J via jax.jacfwd of the rendered
  model image, M=JᵀWJ; verify the render path by recomputing χ² from it and matching
  log_prob's aux at ≥3 points. Rotation = principal angles between stiff subspaces (k=3,
  plus top-vector angle) vs z-separation. At a subset: full H=−∇²logp to split GN vs
  residual-weighted curvature (intrinsic manifold vs misfit). Verify bijector diagonality
  numerically at one point before any per-coordinate chain rule.
- **Prediction:** top-vector rotation ≥15° at separations ~h* (3.5e-3 z-units), growing
  ~linearly with separation below σ; GN term dominant on the stiff subspace.
- **Falsifier:** median rotation <3° at h*-scale separations ⇒ the curved-valley reading of
  T3 is wrong; revisit. **Cost:** thousands of forward evals + a few 22-dim Hessians; no
  sampling; minutes on 1 GPU. **Status: approved by human (conversation 2026-07-02).**

### T7 — Bijector curvature contribution (E2: z vs θ)
- **Cause hypothesis:** part of the z-space valley curvature is manufactured by the diagonal
  nonlinear bijector bending a straighter θ-space ridge (log/sigmoid maps curve linear
  degeneracy contours).
- **Design:** repeat T6's rotation-vs-separation analysis in θ coordinates (same points,
  θ=bij.forward(z); J_θ via the verified per-coordinate chain rule J_z/diag(dz→θ), separations
  measured in θ with per-coordinate standardization). Report the ratio of rotation rates
  z/θ at matched separations.
- **Prediction (jackpot branch):** ratio ≳3 ⇒ transform-induced curvature is a major term;
  remedy = prior/bijector redesign, generalizes to all systems immediately.
- **Falsifier (innocent branch):** ratio ≈1 (θ curls as fast as z) ⇒ curvature is intrinsic
  physics; remedies move to reparameterization of the physical family (E5a).
  **Cost:** free alongside T6 (same evals + cheap transforms).
  **Status: approved by human (conversation 2026-07-02).**

### T8 — Does breathing (not bending) set h*? (transect λ-profile)
- **Cause hypothesis:** T3's D2(h) transition at h_dev reflects the GN quadratic form
  g(t)=eᵀM(x0+t·e)e changing along the transect — dominated by the top eigenvalue breathing,
  not by stiff-direction rotation (T6 showed rotation is weak at these scales).
- **Design (sys60):** along the T3 directions with clear transitions (random_1, random_2,
  axis[gamma]) plus stiffest (its −12-nat dip), evaluate M(z) at ~25 points per direction
  spanning |t| from 0 to ~4×that direction's h_D2_dev (log+linear spacing). Per point:
  g(t)=eᵀMe, λ1(t), overlap |v1(t)·e|. Attribute g's variation via eᵀMe = Σ λ_k (v_k·e)²:
  breathing (λ changes) vs bending (overlap changes).
- **Prediction:** g(t) varies by ≥3× (the D2-deviation criterion) within |t| ≤ h_D2_dev on
  the transition directions, with the λ-term dominating the attribution (≥70% of the change).
- **Falsifier:** g(t) constant within ~1.5× over |t| ≤ 2×h_D2_dev while D2 transitions ⇒ GN
  curvature profiles do NOT explain h*; the T3 transition needs another mechanism — revisit.
- **Cost:** ~100 Jacobian evals, minutes on 1 GPU. **Status: approved by human
  (conversation 2026-07-02).**

### T9 — Step-size account: do energy-error spikes track high-λ1 regions?
- **Cause hypothesis:** with one global ε and λ1 breathing ×22, per-step energy error (xi)
  should spike where the chain visits high-λ1 regions; the adapted ε is set by the worst
  wall, explaining the ESS deficit.
- **Design (sys60 reference run):** align results-phase positions (arrays.npz, 8×20000) with
  results-phase xi (diagnostics.npz, last 20000 steps); FIRST characterize xi's convention
  from `mclmc.py` (per-step energy-error variance proxy? sentinels like −1 must be excluded)
  and document it. Stratified sample ~300 (chain, step) points across the xi distribution
  (top decile, middle, bottom decile of a locally-smoothed |xi|, ±20-step RMS to average
  momentum noise); compute λ1(M(z)) at each.
- **Prediction:** Spearman ρ(λ1, smoothed xi) > 0.5, and median λ1 in the top xi-decile ≥3×
  the bottom-decile median.
- **Falsifier:** ρ < 0.2 or decile ratio ≈ 1 ⇒ the step-size/energy-error account is wrong
  (xi dominated by something other than local stiffness); the breathing→ESS chain breaks.
- **Cost:** ~300 Jacobian evals + post-processing, minutes on 1 GPU. **Status: approved by
  human (conversation 2026-07-02).**

### T10 — On-ridge spike census
- **Cause hypothesis:** the T8 spike field (narrow λ1 peaks, width ~1 sampler step) is not an
  off-ridge artifact of straight probe lines: the chain itself encounters spikes at a rate
  and height sufficient to pin the global ε (per T9's decile signal).
- **Design (sys60 reference run):** 8 contiguous same-chain segments (one per chain, seeded
  starts) × ~256 steps; λ1(M(ẑ)) at every step (batched jacfwd), paired 1:1 with that step's
  smoothed xi. Census statistics per segment + pooled: spike rate (fraction of steps with
  λ1 > 3× segment median), max/median ratio, spike width along the path (consecutive steps
  above threshold), inter-spike spacing, dense within-segment Spearman ρ(λ1, xi_s). Store
  v1 at every point + the chain displacement direction (for T11 and the encounter model).
- **Prediction (from E1's tail + T8):** ≥5% of on-ridge steps above 3× segment median;
  pooled max/median ≥10; spike width ~1–3 steps; within-segment ρ ≥ 0.4.
- **Falsifier:** max/median < 2 on all segments ⇒ spikes are an off-ridge phenomenon the
  chain never meets — the ε account collapses and T9's decile signal needs another source.
- **Cost:** ~2000 batched Jacobians, tens of GPU-minutes. **Status: approved by human
  (conversation 2026-07-02).**

### T11 — Render-space spike localization (physics of the spikes)
- **Cause hypothesis:** spike stiffness is carried by a small set of image pixels —
  caustic-adjacent arc features whose positions respond violently to the γ–shear–src
  combination — rather than by the whole image.
- **Design:** top ~12 census spikes + 12 matched same-segment baseline (median-λ1) points.
  At each: pixel contribution field c_i = √W_i (J v1)_i (so Σc² = λ1); render c² as 80×80
  maps beside the model image; localization fraction L = share of λ1 in the top 1% of
  pixels; participation ratio; v1 loadings at spikes vs the E1 family.
- **Prediction:** spikes strongly localized (L ≥ 50%) on arc/caustic-image pixels; baselines
  substantially more diffuse (L_spike/L_baseline ≥ 3).
- **Falsifier:** spike localization ≈ baseline (ratio ≲ 1.5) ⇒ spikes are a global-image
  effect; the caustic story is wrong.
- **Cost:** ~24 Jacobians + plotting; minutes. **Status: approved by human (conversation
  2026-07-02).**

### T12 — Flank-crossing / pixelation check (micro-mechanism of the λ1 spikes)
- **Cause hypothesis (C-3 micro-mechanism):** λ1 spikes occur when the compact
  counter-image's steep flank sweeps across pixel sample points — λ1 oscillates with the
  SUB-PIXEL PHASE of the counter-image centroid x_c, with amplitude large enough to account
  for the observed ×10–500 flares.
- **Design (three parts, sys60):**
  (1) **Dial scan:** at the top census spike + one mid-list spike + one matched baseline,
  impose a controlled source-center shift (θ-space dial, mapped back through the bijector)
  that translates x_c by ±2 pixels in ~96 steps; MEASURE x_c per point (windowed centroid
  around (row 57, col 12)) and compute λ1 exactly as the census does. Detrend (moving
  median over ±1 pixel period) and measure the pixel-period oscillation: peak-to-trough
  ratio and the Fourier amplitude at the pixel frequency (in measured-x_c units) vs the
  local continuum.
  (2) **Supersample control:** repeat one scan with the model rebuilt at supersample=4
  (all else identical, own χ² gate): amplitude ratio ss4/ss2 distinguishes physical
  pixel-integration information (stable, ≥0.7) from rendering artifact (shrinks, ≤0.5;
  0.5–0.7 = mixed, report as such).
  (3) **Correlational cross-check (census reuse):** one render at each of the 2048 census
  steps → x_c, blob flux A, phases φ=frac(x_c); regress log λ1 on [log A] vs
  [log A + first two Fourier harmonics of φx, φy]; report ΔR².
- **Predictions:** dial-scan detrended peak-to-trough ≥10 with a pixel-frequency Fourier
  peak ≥5× continuum; census ΔR²(phase | flux) ≥ 0.4; spikes cluster at specific phase
  lines.
- **Falsifiers:** peak-to-trough ≤2 or ΔR² ≤ 0.1 ⇒ sub-pixel phase does not explain the
  spikes; the smooth magnification/deflection-sensitivity alternative becomes lead (its
  signature: log A carries the census variance).
- **Branch consequences:** artifact branch (ss4 shrinks) partially revives a refined H1
  (likelihood-as-computed stiffer than intended) with an immediate remedy candidate
  (higher supersampling) testable by a T0-style before/after; physical branch feeds E3
  (predictor = counter-image brightness × compactness relative to pixel scale) and
  constrains remedies (reparameterization can't remove physical information oscillation).
- **Cost:** ~600 Jacobians + ~2100 renders; minutes on 1 GPU. **Status: approved by human
  (conversation 2026-07-02).**

### T13′ — Re-simulated sys60 (data ss=16) × model fidelity 2×2 (the clean payoff test)
- **Motivation (human caveat, 2026-07-02):** the original sys60 data was believed simulated
  at supersample=2 (generator: `attic/LinusFourSim.ipynb`) — confirmed worth treating as
  fact pending Step 0. Then ss2-model + ss2-data is self-consistent (the comb is a bona
  fide property of that synthetic posterior), the original T13 (ss4 model vs ss2 data) is
  confounded by real mismatch (measured Δχ² ≈ 355 at matched parameters), and the benchmark
  doesn't represent real data (effectively ∞-supersampled sky).
- **Design:**
  Step 0: extract sys60's exact generation recipe from `attic/LinusFourSim.ipynb`.
  Step 1 (GATE A, reproduction): with TODAY's code, r = observed − render_ss2(truth) must
  be pure noise: reduced χ²(r/err) ∈ [0.956, 1.044] (1 ± 5√(2/6400)); max |box-smoothed
  r/err| (13-px kernel) < 1.5 (expected max ≈ 1.0 for pure noise over ~500 cells); windowed
  reduced χ² at the counter-image 15×15 ∈ [0.53, 1.47]; residual map plotted for grading.
  FAIL ⇒ STOP (codebase drift; investigate before any re-simulation).
  Step 2 (GATE B, data convergence): render truth at ss=8 and ss=16; require
  max|m16−m8|/err < 0.05. Also REPORT the data-side aliasing size max|m2−m16|/err (how
  aliased the original data is, expected concentrated at the counter-image).
  Step 3: new data d′ = m16 + r (SAME noise realization, recovered from the original
  image). Save STRICTLY under `experiments/why_hard_to_sample/resim/sys60_ss16/` — the
  original `data/simulated_systems/*` is never touched.
  Step 4: fresh pipeline (MAP→SVI) per model arm on d′ (model ss=2 and ss=4), notebook
  config, out-dirs under resim/. Report m4(truth)-vs-m16(truth) misfit relative to noise
  (if ≳0.3σ anywhere, add an ss=8 model arm).
  Step 5 (the 2×2 payoff): standard 8/2000/2000 MCLMC, seeds 1–4 per arm, each arm using
  its own SVI qz (necessarily different from the old reference — data changed; logged).
  Step 6 (comb identity check): repeat one T12 dial scan against d′ with the ss=2 model —
  J is data-independent, so the comb must be unchanged (W shifts negligibly via the err
  map); a changed comb is itself a finding.
- **Predictions:** {d′, model ss2}: min-ESS within ~the old T0 band (11–15) — the comb is
  model-side and survives the data fix; {d′, model ss4}: min-ESS ≥ 10× the old band
  (≥130, plausibly approaching the clone's ~1700). Falsifiers: {d′, ss2} fast ⇒ the comb
  was data-side after all (C-4 mechanism misassigned); {d′, ss4} unchanged ⇒ the comb was
  not the ESS bottleneck (C-4's sampling relevance dies; envelope/bending resume as lead).
- **Cost:** renders trivial; 2 pipelines (~2 min) + 8 MCLMC arms (~minutes) + 1 dial scan;
  one allocation. **Status: approved by human (conversation 2026-07-02; design, ss=16, and
  strict separation explicitly approved).**

### T14 — Close the last link: does the residual Hessian term wash out the Fisher comb?
- **Cause hypothesis (the untested link of the C-4 chain):** on accurate data, the exact
  posterior Hessian H = JᵀWJ − Σ Wᵢrᵢ∇²mᵢ is SMOOTH at the tooth scale even though the GN
  term combs — the residual term carries an anti-comb (cancellation), because the ~14σ
  un-fittable counter-image residual multiplies the same rendering-aliasing second
  derivatives. On self-consistent (ss2) data, r ≈ noise ⇒ residual term small ⇒ H ≈ GN ⇒
  the posterior itself combs (already measured: T3 fine-D2, E1 r-values).
- **Design:** the T12 top-spike dial (97 points, ±2 px measured x_c displacement), ss2
  model, run TWICE — against the ORIGINAL ss2 data and against d′ (ss128) — computing per
  point in standardized ẑ units: (i) g_M(t)=êᵀMê (GN), (ii) g_H(t)=êᵀHê with H = −∇²logp
  (exact, jax.hessian; includes prior — document), (iii) f(t)=logp, (iv) the residual term
  quadratic form g_R = g_M − g_H directly. Detrend all with the T12 moving-median (1-px
  window of measured displacement); teeth = detrended series.
- **Predictions:** ORIGINAL data: g_H tracks g_M tooth-for-tooth (detrended corr ≥ 0.8;
  tooth peak-to-trough within ×3), f(t) corrugated at tooth scale by ≥3 nats. d′: g_M
  combs identically (known), but g_H teeth suppressed ≥10× vs original, f(t) tooth
  corrugation reduced ≥10×, and the residual term g_R combs IN PHASE with g_M (detrended
  corr(g_R, g_M) ≥ 0.8) — the direct signature of cancellation.
- **Falsifier:** d′ g_H teeth ≥ 1/3 of g_M's while the arm demonstrably samples fast ⇒
  wash-out story wrong; fast sampling needs a different explanation (e.g. tooth-narrowness
  tunneling) — stop and rethink before writing any mechanism claim.
- **Cost:** ~200 renders + 200 Jacobians + 200 22-dim Hessians; minutes on 1 GPU.
  **Status: approved by human (conversation 2026-07-03).**

### T14b — Corrected last-link test: curvature along each posterior's OWN chains
- **Why (T14 post-mortem, 2026-07-03):** T14 ran as registered and its falsifier FIRED
  (NEW-data g_H teeth ≈ 7× g_M along the dial; g_R anti-phased, i.e. residual term
  AMPLIFIES teeth there; f-corrugation unchanged 420→408 nats) — but the f(t) panels show
  the dial exits the typical set within ~0.02 px (≪ one tooth spacing): beyond that, BOTH
  targets carry a ~75σ displaced-blob residual that swamps the 14σ data difference. The
  dial measures off-support curvature; the sampler never goes there. Registered design
  error (orchestrator's), not an agent error. The wash-out claim remains UNRESOLVED, not
  refuted.
- **Design:** T10-style census at the correct locus: 2 contiguous 128-step segments per
  seed-1 run of (a) the OLD self-consistent posterior (frozen reference,
  mclmc.stale-20260703T111618) and (b) the NEW {d′, ss2-model} arm
  (resim/sys60_ss16/arm_2/t0/t0_seed1.npz; its npz carries xi). At every step: λ1(M) and
  ê₁ᵀHê₁ (exact Hessian along the local stiff direction), plus the run's own smoothed xi.
  Also per-target: posterior spread of the counter-image centroid x_c (renders of ~256
  posterior samples) in px — does the NEW posterior still lock x_c sub-pixel?
- **Predictions:** (i) λ1(M) tooth statistics comparable across targets IF their x_c
  supports are comparable (comb is position-set); (ii) the H-form teeth experienced by
  the NEW chain suppressed ≥10× vs the OLD chain's (this is the wash-out claim at the
  right locus); (iii) xi–λ1 coupling present on OLD (ρ≈0.4, T9) and ABSENT on NEW
  (ρ<0.2); (iv) plausible alternative protector, tested by the x_c-spread measurement:
  the NEW posterior's x_c spread ≫ OLD's (sub-pixel lock broken by the misfit floor) —
  if so, "protection" = de-locking, not Hessian smoothing, and both are reported.
- **Falsifier:** NEW chains experience H-teeth ≥ 1/3 of OLD's AND xi stays calm ⇒ teeth
  don't cause slowness at all ⇒ the T9/T10 causal reading collapses — stop and rethink.
- **Cost:** ~512 Jacobians + 512 Hessians + ~512 renders; minutes. **Status: run under
  the human's standing "close the last link" mandate (conversation 2026-07-03); flagged
  for explicit grading with the T14 post-mortem.**

### Decision tree (read before dispatching runs)

1. **T2 + T5 first** (free, existing runs) — either can be done immediately alongside T0.
2. **T0** to calibrate thresholds.
3. **T1** (clone): *clone slow* → H2 sufficient; program shifts to
   preconditioning/reparameterization (NFW parameterization first — already implicated);
   T3/T4 deprioritized. *Clone easy* → H2 insufficient; run **T3**; if T3 finds roughness,
   run **T4** to localize the mechanism; if T3 finds smoothness, H3/bijector numerics and
   whatever T2 surfaced become the leading suspects.
4. At every branch: two failed same-class fixes → stop, list untested assumptions
   (method-discipline §6). That rule exists because of this exact investigation's history
   (O6: prior attempts rabbit-holed).

---

## Log (newest first)

- **2026-07-03 (T14 + T14b RAN — last link closed: protection = posterior-support
  decoupling, not off-support Hessian cancellation)** — allocations 55446130/55446195
  (T14; first start hit the reference-rotation, see below) and 55446596 (T14b), released.
  Artifacts: `results_t0t1/sys60/t14/`, `t14b/`. All `proposed (UNCERTIFIED)`.
  **Reference-rotation incident:** the human's own re-runs (2026-07-03) rotated the
  original testsys60 pipeline stages to `.stale-*`; the qz-staleness guard design caught
  the class of failure it was built for. Original stages verified byte-identical to our
  archive and RE-PINNED: mclmc → `mclmc.stale-20260703T111618`, svi →
  `svi.stale-20260703T110909` (mtimes 2026-07-02 10:12/10:27 ✓). All completed results
  unaffected.
  **T14 (dial): falsifier FIRED — and the design was wrong (orchestrator error, logged).**
  Along the T12 dial, NEW-data g_H teeth ≈ 6.9× g_M (not ≤1/3), g_R anti-phased
  (amplification), f-corrugation 420→408 nats (no reduction). But the f(t) panels show
  the dial exits the typical set within ~0.02 px ≪ one tooth spacing; beyond that BOTH
  targets carry a ~75σ displaced-blob residual that swamps the 14σ data difference. OLD
  predictions did hold there (corr(g_H,g_M)=0.9994, p2t ratio 2.69). Verdict: T14
  measured off-support curvature; inconclusive for the mechanism; wash-out-as-stated
  (off-support cancellation) is dead as a framing.
  **T14b (census along each posterior's OWN chains — the right locus): KEY prediction MET
  ×43 over threshold.** Same-coordinate 2×128-step segments; χ² gates machine-zero; one
  common std_z from the frozen reference. (1) λ1(M) along-path: OLD max/med 25.4, spike
  rate 6.2% vs NEW max/med 2.75, spike rate 0.0% — the NEW chain never meets a Fisher
  tooth. (2) **g_H excursion OLD 3.06e6 vs NEW 7.14e3 — suppression 428× (predicted
  ≥10×); falsifier dead.** (3) xi: OLD chain in sustained energy-error crisis (xi_s
  plateaus 10³–10⁴) vs NEW at/below target (≲1). Per-segment ρ(λ1,xi) MISSED both
  predictions (OLD −0.085 vs ≥0.4; NEW 0.231 vs <0.2) — within-segment correlation is the
  wrong instrument (xi_s plateaus dominate); T9's stratified cross-decile measurement
  remains the valid one; logged as a design-sensitivity miss, not a reversal. (4) x_c
  support: OLD 12.19±~0.01 px (range 0.054) — sub-pixel LOCKED onto tooth structure; NEW
  shifted to 12.13 (the ~0.06 px aliasing bias expressed as a parameter shift) and 2.55×
  broader (range 0.137) — parked BETWEEN teeth; neither support spans a tooth spacing,
  so "averaging over many teeth" is NOT the protector either.
  **CLOSED MECHANISM (final form of the C-4 chain):** self-consistent low-fidelity
  pairing ⇒ residuals → noise ⇒ sub-pixel placement pays χ² dividends ⇒ posterior locks
  onto the comb (support ON teeth; H≈GN teeth; alternating-sign washboard curvature;
  sustained ξ crisis; ε pinned; min-ESS 13). Accurate data ⇒ the model's aliasing error
  becomes an un-fittable floor ⇒ sub-pixel placement stops paying ⇒ the posterior SHIFTS
  (absorbing the aliasing as a small parameter bias), RELAXES (×2.5), and settles between
  teeth on the smooth envelope ⇒ the chain never engages the comb (λ1 flat, g_H 428×
  calmer, ξ at target, min-ESS ~1000). **What protects other systems (the human's
  question):** any condition that breaks the lock — data the model cannot track to
  sub-noise precision at compact features (ALL real data, and any synthetic data
  generated at higher fidelity than the fit), no compact counter-image lever (extended
  sources: Vela), or adequate model render fidelity (comb ×1400 weaker at ss4). The
  uniquely pathological corner is synthetic data generated by the same low-fidelity
  renderer used to fit it, containing a bright compact image.

- **2026-07-02 (T13′ RAN — the pathology is the SELF-CONSISTENT aliased pair; accurate
  data alone restores fast sampling)** — allocations 55409821…55411086 (several false
  starts: bij.inverse structure fix, ladder extension, stale-name fixes, 80GB node for
  the pipeline's batch op, reduced ss4-arm MAP/SVI batches 50/125 vs notebook 200/500 —
  OOM at 84.6GiB = the ss2 op ×4 pixels; deviation recorded in the arm manifest; each
  fix logged in git-diffable code comments). All `proposed (UNCERTIFIED)`; artifacts under
  `experiments/why_hard_to_sample/resim/sys60_ss16/` (STRICT separation held; original
  npz read-only, sha1 recorded).
  **Step 0/Gate A (reproduction):** today's scene API at ss2 reproduces the original
  attic/Linus-FourSim.ipynb simulation EXACTLY to the noise: reduced χ² 0.9729 ∈
  [0.956,1.044]; smoothed-|r/err| max 0.346 < 1.5; counter-image-window χ² 0.9959.
  No codebase drift; the data's ss=2 provenance is now verified fact. (Generation recipe
  + prior/dtype/API discrepancies documented in t13_resim.py's docstring — none affect
  data reproduction.)
  **Gate B (convergence ladder, amended from ss16 after the 0.69σ cusp failure):**
  16v8=0.69, 32v16=0.148, 64v32=0.068, 128v64=0.0294 < 0.05 ⇒ d′ built at **ss=128**
  (path names say ss16 — registered before the amendment; manifests record ss_hi=128).
  REPORT: the ORIGINAL data is aliased at **13.82σ** at the counter-image (57,12).
  **The 2×2 (standard 8/2000/2000, seeds 1–4, each arm its own fresh MAP→SVI qz):**
  {d′, model ss2}: min-ESS **1031–1253** (R̂≤1.011) — the pre-registered FALSIFIER FIRED
  (predicted "stays slow, comb is model-side": WRONG). {d′, model ss4}: min-ESS
  **889–1164** — prediction (≥130) met. Both ≈80–110× the old self-consistent band
  (11–15), within ~1.5–2× of the T1 clone (1718) — residual ordinary hardness (envelope,
  σ-scale bending) remains. Model-side misfit vs truth: ss2 arm 13.88σ at the
  counter-image (fast but BIASED — misspecified); ss4 arm 1.93σ at the central cusp
  (ss8 arm recommended by the checkpoint rule; memory-infeasible without pipeline
  chunking — left open).
  **Step 6 (comb identity):** on d′ the ss2 model's Fisher comb is UNCHANGED (recovered
  freq 1.908/px; peak-to-trough 37,046; χ² gate machine-zero) ⇒ the comb is definitively
  model-side and data-independent — AND it does not impede sampling by itself.
  **Synthesis (C-4 scope amendment registered):** the slow-sampling pathology required
  the self-consistent pairing: aliased model FITTING aliased data ⇒ near-zero residuals
  ⇒ posterior curvature = GN comb (exactly the regime where E1 measured GN-dominance)
  ⇒ ε pinned by comb teeth ⇒ min-ESS 13. Accurate data + aliased model ⇒ 13.9σ misfit
  decouples the posterior from the comb (residual Hessian term) ⇒ fast but biased.
  Accurate data + adequate model ⇒ fast and (nearly) unbiased. **Implications:** (i)
  synthetic benchmarks must be generated at higher fidelity than they are fit, or they
  manufacture exactly this trap; (ii) production-relevant prescription: render fidelity
  chosen for accuracy (bias), and the comb trap only bites simulation studies; (iii) the
  STANDING QUESTION IS NOT CLOSED — the carousel is REAL data, so its slowness cannot be
  this mechanism; E3 cross-system now splits into "synthetic self-consistent" (explained)
  vs "real-data hardness" (open; curved degeneracies per the carousel logs remain the
  lead there). **Next candidates:** E3 sibling census re-scoped (which of the 100
  ss2-simulated siblings are slow ⇔ counter-image compactness — now a *prediction of the
  self-consistency trap*, checkable against the user's experience); carousel program
  (real-data hardness) via the same T1/T5/E1 toolkit; vela e1-prior fix; optional ss8-arm
  once pipeline batching allows.

- **2026-07-02 (T12 RAN on sys60 — the spikes are a SUPERSAMPLE-GRID COMB; largely a
  rendering-fidelity artifact)** — allocations 55407728/55407810/55407876 (two false starts
  from over-strict blob guards: the 15×15 window sits on the lens-light pedestal (~2.0
  counts, tilted) and the dial dims the counter-image ~2× (real magnification change);
  fixed = plane-fit pedestal subtraction (centroid bias 0.14→0.000 px on tilted-pedestal
  smoke) + contrast guard vs bg rms only). χ² gates machine-zero for BOTH model builds
  (ss2 and ss4). Artifacts: `results_t0t1/sys60/t12/`. All `proposed (UNCERTIFIED)`.
  **Part 1 (dial scans):** λ1 vs MEASURED counter-image displacement is a comb: narrow
  teeth at pitch ≈0.52 px (recovered dominant frequency 1.90–1.91/px on top+mid spikes) =
  the ss=2 SUBSAMPLE pitch, spanning peak-to-trough 30,645 / 1,540 / 9,979 (top/mid/
  baseline; predicted ≥10 — met ×150+; falsifier ≤2 dead). The pre-registered 1/px Fourier
  metric FAILED (0.64/0.24/2.22 vs ≥5) because it was aimed at the wrong frequency — the
  registered hypothesis said "pixel sample points"; the actual sampling grid is the
  supersampled one. Mechanism CONFIRMED in refined form: flank-crossing at the SUBGRID.
  The "baseline" census point simply sits between comb teeth (9.4e6 teeth half a pixel
  away); blob-window λ1 carries the big teeth; smaller full-image teeth ≈ the main arc's
  own subgrid crossings.
  **Part 2 (ss=4 control): ARTIFACT branch in substance.** Same scan at supersample=4:
  the comb collapses — top tooth 6.0e7 → 4.2e4 (×1400), peak-to-trough 30645 → 4.7 —
  leaving the smooth (caustic-proximity) envelope. The pre-registered fourier-amp ratio
  (0.553) lands in the "mixed" band only because it, too, was pinned to 1/px; the
  log-amplitude ratio 0.150 and the ×1400 collapse are decisive. Registered as C-4.
  **Part 3 (census cross-check):** ΔR²(phase | flux) = 0.563 (predicted ≥0.4; R²_flux
  0.055 → R²_full 0.618); magnification-branch signature absent. (Caveat: harmonic
  regressors are collinear over the census's narrow x_c range — betas uninterpretable,
  ΔR² still valid.)
  **Reading:** the chain-facing stiffness landscape = smooth envelope × subgrid comb; the
  comb (fidelity artifact) is what pins ε and produces T0's min-ESS ≈ 13. H1 returns in
  refined, smooth form (see C-4) — the original intuition about simulator internals was
  substantively right. **Proposed next (T13, needs checkpoint + approval): the payoff
  test** — sys60 standard-config MCLMC at ss=4, seeds 1–4: predict min-ESS ≥ 10× the T0
  band (13 → ≥130, plausibly approaching the clone's ~1700); falsifier: unchanged within
  the T0 band ⇒ the comb was not the ESS bottleneck. Note ss=4 is a (slightly) different,
  MORE ACCURATE posterior — a modeling choice to grade, not a sampler trick. Cross-system
  corollary: Vela runs supersample=1 but its extended source has no compact feature to
  alias (consistent with its ease); the E3 sibling predictor should be counter-image
  brightness × compactness relative to the SUBGRID pitch.

- **2026-07-02 (T10+T11 RAN on sys60 — the chain DOES meet the spikes; the stiffness is one
  compact counter-image)** — allocation 55406521 (released); χ² gates machine-zero; 2048
  on-ridge Jacobians in 12.4 s (vmapped). Artifacts: `results_t0t1/sys60/t10/`, `t11/`.
  All `proposed (UNCERTIFIED)`.
  **T10 (census): ALL FOUR pre-registered predictions MET.** Pooled over 8×256 on-ridge
  steps: spike rate (λ1 > 3× segment median) = 11.4% (predicted ≥5%); pooled max/median =
  263 (predicted ≥10; per-segment 3.3–562 — falsifier <2-everywhere decisively dead);
  median spike width 2 steps (predicted 1–3; heavy tail: sustained 18- and 26-step wall
  encounters in seg6/seg7); pooled within-segment ρ(λ1, xi_s) = 0.41 (predicted ≥0.4;
  per-segment −0.12…0.56, strongest exactly where the ×562 monster lives, λ1=6.0e7 —
  on-ridge, ×8 above T8's off-ridge max). Segment heterogeneity large (spike rate
  0.4%–24%): chains traverse spike-rich and spike-poor stretches. Per-step v1 +
  displacement angles banked in `t10_arrays.npz` for the future encounter model.
  **T11 (localization): prediction L≥50% met at L=0.999 — but the ratio falsifier FIRED
  (L_spike/L_baseline = 1.00), and not in the anticipated "diffuse" direction:**
  localization is CONSTANT-MAXIMAL — at spikes and baselines alike, ~all of λ1 is carried
  by the top <1% of pixels, a single ~4×4-px hotspot at (row 57, col 12), identical across
  chains/segments (self-check Σc²=λ1 exact). The pre-registered discriminant (spread) was
  mis-chosen; the spike/baseline difference is INTENSITY (×10–500), not location or spread.
  **Physical identification (data check, logged):** the observed image has a real compact
  feature at exactly that pixel — peak 15.3 vs bg rms 0.2 (~75σ), ~13% of the main peak at
  (40,40) ⇒ the lensed COUNTER-IMAGE of the source. Mechanism reading (C-3): the compact,
  PSF-sharp counter-image is the lever through which the γ–shear–src degeneracy is
  constrained; its position is the steepest function of slope/shear; sub-pixel motion of a
  75σ compact feature produces enormous noise-normalized response ⇒ λ1 spike field =
  sensitivity flares of ONE feature. Proposed micro-mechanism for spike widths (untested):
  steep-flank pixel crossings. **Corollaries:** (a) E3 gets a sharp cheap predictor —
  counter-image brightness/compactness vs min-ESS across the 100 siblings; (b) Vela's
  extended Sersiclet source lacks a dominant compact counter-image lever, consistent with
  its healthy sampling; (c) remedy thinking should target the counter-image constraint
  (e.g., parameterizations aligned with its position, or likelihood tempering of the
  compact-feature pixels is NOT acceptable — it changes the posterior; reparameterization
  remains the clean route). **Next:** encounter model (v1·displacement data already
  banked); sub-pixel-crossing check; E3 sibling census.

- **2026-07-02 (T8+T9 RAN on sys60 — h* mechanism = a SPIKE FIELD in the stiffness; ε-account
  supported with a caveat)** — allocation 55404453 (released); both scripts re-ran the χ²
  render gate (machine zero) and re-verified the T3 directions sign-sensitively. Artifacts:
  `results_t0t1/sys60/t8/`, `t9/`. All `proposed (UNCERTIFIED)`.
  **T8:** primary prediction CONFIRMED far beyond threshold — the GN quadratic form
  g(t)=êᵀMê varies ×112 / ×43 / ×84 (random_1 / random_2 / gamma-axis) within |t|≤h_dev
  (predicted ≥3; falsifier ≤1.5 decisively dead). D2's h* transition is fully accounted by
  the GN curvature profile. The ≥70%-breathing attribution sub-claim did NOT hold cleanly:
  shares are direction-dependent and exceed 1 with cancellation (random_1 breathing-dominant
  3.87/0.22; random_2 0.61/2.77; gamma 1.00/0.68; stiffest 0.99/2.32) — along off-ridge
  lines, eigenvalue motion and eigenvector churn (v1/v2 identity swaps at peaks, visible in
  the stiffest row's overlap panel) are entangled; the pre-registered dichotomy was too
  coarse. **Refined picture from the λ1(t) profiles:** the stiffness field is a set of SHARP
  NARROW PEAKS (width ~1e-3 z ≈ ε_move); λ1(t=0)=2.831e5 identical across directions
  (consistency check passed), and random_1 grazes a spike of λ1=7.9e6 at t=−5.8e-4 — ×28
  above x0's value HALF A SAMPLER STEP away, and ×8 above E1's 32-point maximum (E1
  undersampled the tail; true dynamic range ≥×70 vs the bottom-decile median). h* = the
  spike width. D2-deviation onset aligns with the spike edges in all transition directions.
  **T9:** decile prediction MET — median λ1 in the top xi-decile / bottom decile = 3.30×
  (predicted ≥3). Spearman ρ(λ1, smoothed xi) = 0.439 — just below the 0.5 prediction,
  far above the 0.2 falsifier (within-stratum ρ≈0, as expected with compressed range).
  Caveat logged per discipline: xi_s spans ~10 DECADES across strata while λ1 spans ~1.3
  ⇒ static local stiffness is one ingredient; spike-ENCOUNTER geometry (how the trajectory
  meets a narrow wall) must carry the rest of the dynamic range. xi convention: mclmc.py:334,
  xi = energy_change²/(dim·desired_energy_var)+1e-8; results-phase sentinel-free (1/160000
  floor-excluded); xi heavy tail: median 2.5e-4, max 2.0e8.
  **Chain status:** manifold spike field (T8) → h* (T3) → global ε pinned by spike
  encounters (T9, partial) → ESS deficit (T0/T1). Every link now has direct evidence; the
  T9 link is supported-but-incomplete (encounter geometry unmeasured).
  **Follow-up candidates (checkpoints required):** (i) spike census: dense λ1 sampling along
  ON-RIDGE chain segments (not off-ridge transects) — spike frequency/height distribution
  the trajectory actually encounters; (ii) encounter model: condition xi spikes on
  (λ1, angle between trajectory direction and v1) — completes T9; (iii) physics of spikes:
  render-space localization (which pixels drive JᵀWJ at a spike — caustic-crossing images?);
  (iv) E3 cross-system + E5a reparameterization unchanged.

- **2026-07-02 (T6+T7 RAN on sys60 — bijector innocent; curved-valley picture REFINED to
  two scales: σ-scale bending + h*-scale breathing)** — allocation 55401964 (released).
  `e1_fisher_survey.py`; χ² render gate passed at machine zero (render path byte-identical
  to likelihood; aux = reduced χ²·event_size reconciled); bijector verified exactly diagonal.
  Artifacts: `results_t0t1/sys60/e1/`. All `proposed (UNCERTIFIED)`.
  **T7 (bijector): FALSIFIED-INNOCENT, decisively.** z and θ rotation-vs-separation curves
  are superimposed at ALL scales (matched-bin ratio 1.00). The curvature is intrinsic to the
  physical model family; prior/bijector redesign is ruled out as a remedy for THIS pathology
  (the smoke toy proved the diagnostic detects transform-induced curvature when present, so
  this null is meaningful).
  **T6 (rotation): prediction MISSED in magnitude — logged per discipline.** Predicted ≥15°
  top-vector rotation at h*-scale; measured median 5.0° (k=3 mean principal angle ~1° at the
  same separations — the falsifier's <3° fires on the subspace measure, not the top-vector).
  Substantial bending instead turns on at **0.2–0.7σ** separations (k=3 mean principal angle
  15–25°, top-vector 20–70°, z=θ throughout) — the valley DOES bend within the typical set,
  accumulating large rotations over the ~2σ traverse, but ~3–7× more slowly than the "rotation
  sets h*" story required. That sub-claim is dead.
  **New mechanism at h*-scale — eigenvalue BREATHING:** across 32 typical-set points the
  stiffest GN eigenvalue varies ×22 (standardized: 4.3e4→9.6e5; heavy-tailed, median 1.9e5,
  p90 6.7e5) while ranks 2–3 vary only ×1.9. Spectra overlay: local GN spectra coincide with
  the global precision (clone Σ⁻¹) at ALL ranks except the top 1–2, which sit 10–200× above
  it. Geometry: 20/22 dimensions effectively Gaussian; ONE razor-thin crease direction,
  locally far stiffer than the global covariance implies, whose width breathes point-to-point
  and whose direction bends on σ-scales. A single global step size must satisfy the
  crease's WORST wall anywhere on the ridge ⇒ ε is ~√22≈4.7× too small for the median
  region — a candidate quantitative account of the ESS deficit (with bending compounding).
  **Named stiff family** (mean |loading| over 32 points): EPL slope γ 0.59, shear γ2 0.38,
  src center_x 0.32, shear γ1 0.29, mass e1 0.26, mass e2 0.17, θ_E 0.14 — the classic
  slope–shear–source-position degeneracy.
  **GN vs rest:** r = 0.04–0.35 (Frobenius, 6 points) ⇒ Gauss–Newton/model-manifold term
  dominates the stiff subspace — intrinsic "hyper-ribbon" geometry, not residual/prior
  effects.
  **Follow-up candidates (design checkpoints required):** (i) correlate λ1(z) along a T3
  transect with the D2 transition (does breathing, not bending, set h*? cheap); (ii) test the
  step-size account: predict energy-error (xi) spikes at high-λ1 regions from the run
  history; (iii) E3 cross-system + E4 toy model; (iv) remedy E5a: reparameterize the named
  γ–shear–src family (bijector redesign is off the table per T7).

- **2026-07-02 (T3 RAN on sys60 — H1 falsified; curved-valley geometry demonstrated)** —
  Human approved; allocation 55396852 (released). `t3_transects.py`: 7 transects through one
  seeded typical-set draw of the converged 20000/20000 reference run (3 random, stiffest +
  softest clone-cov eigendirections, worst-ESS axes src center_y + mass gamma); 36-step
  h-ladder over 6.3 decades per direction; D2(h) second differences, FD-vs-AD gradients,
  macro curves; roundoff model 4·eps_f/h² overlaid; float32 arm as positive control (clean:
  `WHTS_FLOAT32_CONTROL=1` skips the module's x64 forcing so constants build float32;
  achieved energy_dtype verified float32). Reference run precision verified beforehand:
  x64 on, `conv_precision=None` → convolution at img.dtype → **float64 end-to-end**
  (scene_simulator.py `_psf_convolve`; model_card `likelihood_precision: null` = default).
  Artifacts: `results_t0t1/sys60/t3/{float64,float32}/`. All `proposed (UNCERTIFIED)`.
  **Smoothness read (H1):** float64 D2(h) converges and stays FLAT down to the roundoff
  crossover on ALL directions — through the entire sampler band [ε/100, ε] and below
  (ε_move/dir ≈ ε·σ_dir/√dim, ε=0.1255, L=13.3). FD-AD relative error 1e-9–5e-8, stable.
  float32 control blew up at its predicted noise scale (h≈1.2e-5 vs model crossover ~6e-6;
  FD-AD floor degraded to 2.7e-3). **Falsifier met ⇒ H1 dead along tested directions; C-2
  WITHDRAWN.** (Caveat: table's `h_D2_dev` flags are top-decade-plateau deviations = macro
  averaging, NOT small-h blow-up — read the plots, not the table.)
  **Macro/H0 by-product — the positive finding:** local D2 at x0 along random directions
  ≈ 5e8, vs σ-scale average ≈ 5e6 (ratio ~100×, approaching the stiffest direction's 9.3e8);
  transition scale (curvature correlation length) ≈ 0.07σ_dir ≈ 3× the per-step move. Macro
  transects drop 15,000–30,000 nats over ±2σ_dir along random directions and are strongly
  non-parabolic — random_2 is NON-MONOTONIC (cliff to −31,500 at +1.5σ, then recovery ⇒ the
  straight line re-enters a high-density region: a curved ridge). Axis-aligned + eigen
  transects are near-flat (≤~2,000) ⇒ the structure lives in ROTATED joint directions,
  invisible to cornerplots. **Coherent picture (UNCERTIFIED):** sys60's posterior is a
  smooth, strongly CURVED stiff valley — same-covariance straight-valley clone samples 130×
  faster (T1); global 2nd moment benign (T5 whitened ≈17); marginals Gaussian (O1); point
  Hessians honestly unrepresentative because the smooth Hessian field varies 100× over
  ~0.07σ (O3 re-explained); H2/H3 dead on this system (T1/T2). C-1 now has its artifact.
  **Post-hoc null-overlay replot** (`replot_macro_with_clone.py`; T3 stored no direction
  vectors, so they were reconstructed and verified sign-sensitively against the stored g·e
  and σ_dir before plotting): each macro transect now carries the T1 clone's EXACT transect
  along the same line as the Gaussian null — the nat-drop magnitudes are the null: measured
  marginal/conditional width ratios (= per-direction rotation amplification,
  √(eᵀΣe·eᵀΣ⁻¹e)) are 124/126/184 along random_0/1/2, 25.7/34.6 along the worst-ESS axes
  (src center_y / mass gamma), 1 along eigendirections ⇒ ~10⁴-nat drops at 2σ_dir are
  Gaussian-expected; deviation from the dashed null is the finding. Companion figure with
  x in conditional widths (null = universal −t²/2): `t3_macro_condwidth.png`. New observations from the overlay: (i) real transects
  sit ABOVE the null at ±2σ along all random directions — locally ~100× stiffer than the
  clone yet globally shallower ⇒ heavier-than-Gaussian along-line tails, the line re-entering
  the curved ridge; (ii) the STIFFEST eigendirection is itself non-monotonic at the 10-nat
  scale (dip to −12 at +1σ_stiff, recovery to −4 by +2σ) — invisible at the combined plot's
  y-scale; (iii) gamma-axis transect has a shelf/notch at −0.5σ. Artifact:
  `results_t0t1/sys60/t3/float64/t3_macro_with_clone_ref.png`.
  **Blind spots:** one typical-set point; 7 directions of 22; one system. **Next candidates
  (design checkpoints required before running):** (a) repeat T3 at 2–3 more typical-set
  points + 2-D log-density slices in the worst plane (cheap); (b) quantify valley curvature
  radius vs step size along the ridge (geometry → step-size ceiling mechanism); (c)
  cross-system replication (vela post-e1-fix, carousel old/new NFW param — the user's θ_E
  reparameterization plausibly worked by STRAIGHTENING exactly this class of curved
  degeneracy); (d) remedy branch: nonlinear reparameterization / local-metric methods.

- **2026-07-02 (T0+T1 RAN — flagship split executed)** — Human approved (conversation,
  2026-07-02): sys60 primary, own GPU allocation (55392666, released after). All arms standard
  config (8/2000/2000), per-system reference qz, seeds 1–4; artifacts under
  `experiments/why_hard_to_sample/results_t0t1/{sys60,vela01}/` (t0/, t1/, report/, clone
  manifests). All `proposed (UNCERTIFIED)`.
  **T0 full sets (min bulk-ESS / max rank-R̂ per seed):**
  sys60: 12.8/1.66, 11.7/1.84, 11.2/1.89, 15.3/1.48 → band 11.2–15.3, ratio 1.37 (tight;
  usable). Worst params: src center_y / mass gamma. NOTE: standard-budget arms are unconverged
  (R̂≈1.5–1.9) — the same phenotype as O2's carousel citation (min-ESS≈12/16000), on a fully
  simulated, well-specified, NO-lstsq system.
  vela: 2584/1.007, 1732/1.010, 1276/1.011, 803/1.019 → band 803–2584, ratio 3.22 (top of the
  expected 2–3 range). Worst param: src e1 in 4/4 seeds.
  **T2 completion (θ-space bound-crowding, bijector live):** vela `planes/1/light/0/e1`
  crowding fraction **0.984** (98.4% of mass in the outer 1% of its prior box → bound-pinned;
  H3 mechanism CONFIRMED for this parameter; actionable: widen the e1 prior). sys60: crowding
  ≈0 on all box-bounded params, θ-marginals clean → **H3 falsified for sys60** (1-D/crowding
  scope; joint z-structure remains T2's blind spot).
  **T1 clone arms (4 seeds each, same qz/config as real arms):**
  sys60 clone: min-ESS 1718, 1734, 1851, 1722 (R̂≤1.007). vs real 11.2–15.3 →
  **clone ≈130× easier, vastly beyond the 1.37 band ⇒ H2 INSUFFICIENT on sys60** (verdict in
  `results_t0t1/sys60/report/t1_report.json`). With T2 excluding H3 and T5's whitened
  condition ≈17, the surviving prime suspect for sys60 is **H1 fine structure of the true
  log-density** — and since sys60 has NO lstsq layer, the mechanism (if T3 confirms roughness)
  must live in the simulator itself (convolution/guard ops), not amplitude marginalization.
  Note the clone at 2000/2000 (ESS≈1700) even beats the real 20000/20000 run (ESS 1239) at
  1/10 the budget.
  vela clone: min-ESS 2237, 1792, 1028, 764 — worst param src e1 in 4/4, seed-for-seed inside
  the real band → **CONSISTENT WITH H2-SUFFICIENCY on vela**: the z-covariance geometry
  (dominated by the bound-pinned e1 direction that H3 created) fully accounts for its mild
  slowness. (The canned verdict text's "NFW first" is carousel boilerplate — the actionable
  reparameterization here is the e1 prior box.) The vela clone-slow result doubles as the
  **positive control** for the clone pipeline: the machinery produces slow clones when
  geometry warrants, so sys60's fast clone is not a plumbing artifact.
  **Decision-tree position:** sys60 → run **T3** (multi-scale transects; cheap, no sampling;
  include worst-ESS directions src center_y / mass gamma among transect directions, and use
  the float32-vs-float64 contrast as positive control). T4 is structurally moot for sys60
  (no lstsq); if T3 finds roughness, mechanism hunt moves to simulator internals. vela →
  branch closed pending prior-box fix; a widened-e1 rerun would test the H3→H2 chain
  end-to-end (predict: crowding→0, z-tail gone, min-ESS up several×).

- **2026-07-02 (later)** — **Systems designated + T2/T5 free sweep ran on their reference
  runs.** Human designated (conversation, 2026-07-02) two SIMULATED systems: **sys60** (one of
  100 simulated Sersic systems; EPL+shear+2 Sersics, `mode="forward"` — amplitudes SAMPLED, **no
  lstsq layer at all** ⇒ clean H1-mechanism discriminator; 22 params; reference run
  `results/testsys60/`, MAP→SVI→MCLMC 8ch 20000/20000 seed 0) and **vela01_cam12_rep03_a0.500_f814w**
  (analytic EPL+shear mass + Sersic lens light, source = real Vela-catalog galaxy fit by
  elliptical Sersiclets n_max=5, lstsq amplitudes ⇒ H1 layer present but source-mismatch
  CONTROLLED; 20 params; reference run 8ch 5000/5000 seed 0, truth-bootstrap qz; NOTE
  `conv_precision="float32"` — an O5-class roughness candidate T3 can toggle). Correction to
  prior context: the carousel is a REAL observed system (not synthetic as an agent had wrongly
  inferred from dir naming); it moves to real-data replication + old/new-NFW-parameterization
  positive control. Per-system modules `experiments/why_hard_to_sample/systems/*/system.py`
  rebuild each reference run's exact qz from persisted stage arrays — verified by
  `stable_hash(qz)` == the mclmc manifest's `upstream_hashes.qz` (sys60 `e59573bb…`, vela
  `1d80d19…`) — with a load-time staleness guard on that manifest hash. Names via sorted-key
  bijector probe (C-8-safe).
  **T2/T5 results (free post-processing, `proposed (UNCERTIFIED)`, artifacts in
  `experiments/why_hard_to_sample/results_free_sweep/{sys60,vela01}/`):**
  sys60 — min bulk-ESS 1239/160k draws (z[7]=`planes/0/mass/0/center_x`), max rank-R̂ 1.009;
  raw cond 1.8e6 → **whitened 16.7** vs final adapted mass matrix; stiffest direction rotated
  (θ_E+γ1+γ2+src cx+mass e2); z-marginals clean (0 params |skew|>0.5 or ex-kurt>1).
  vela — min bulk-ESS 9196/40k draws, max rank-R̂ 1.001 (healthy at intensive budget);
  raw cond 2.0e7 → **whitened 14.3**; one anomaly: z[17]=`planes/1/light/0/e1` (source
  ellipticity): skew −1.3, ex-kurt 2.7, MAD-tail 4% (5–6× all others), and it is
  simultaneously worst-ESS AND a pure axis-aligned softest eigendirection → H3-flavored
  bound-crowding suspect (θ-space crowding check pending; needs container jax).
  **Reading (UNCERTIFIED):** both systems land in T5's pre-registered H2-falsifier region —
  the full windowed mass matrix already represents the covariance (whitened ≈15). sys60
  sharpens the standing question: whitened cond ~17, Gaussian-clean marginals, yet ~0.8%
  ESS/draw. Roles proposed: sys60 = hard primary, vela = easy-ish control.
  **Proposed next runs (awaiting grader approval):** T0 seed sweep (seeds 1–4, standard
  8/2000/2000 arms) on BOTH systems (cheap: reference 20k/20k took ~5 min wall); T1 clones
  built from the intensive reference runs (sys60 source: min-ESS 1239, R̂ 1.009; vela source:
  min-ESS 9196, R̂ 1.001 — both recorded per the pre-registered mitigation); clone arms use
  each system's own reference qz. Under-converged-source bias direction: toward falsely
  rejecting H2 (clone too easy) — mild here given source diagnostics above.

- **2026-07-02** — **Harness code for T0/T1/T2/T5 written (NO runs executed; all checkpoints
  still awaiting approval).** New `experiments/why_hard_to_sample/` (branch `why-hard-t0t1`):
  one frozen `StandardMCLMCConfig` (8 chains / 2000 burn-in / 2000 results, energy-var 5e-4,
  regularize_mass_matrix=True) shared byte-identically by every arm; T0 seed-sweep driver
  (full per-seed set + derived band, ≥10× band ⇒ stop-and-surface warning); T1 clone builder
  (pooled post-burn-in z-samples → mean + full cov, source-run min-ESS/max-R̂ recorded per the
  pre-registered mitigation) + clone runner (clone target lives natively in z; **qz is the real
  run's MAP-centered isotropic 1e-2 ball, deliberately NOT the fitted clone covariance** — qz
  seeds init positions, the initial inverse mass matrix, and the adaptation anchor, so reusing
  the real run's qz is what makes the pipeline "identical") + real-vs-clone report implementing
  the pre-registered decision rule; T2/T5 post-processing scripts (run on existing npz
  artifacts; T2 bound-crowding SKIPs loudly without the bijector, MAD-tail proxy clearly
  labeled supplement; T5 raw + preconditioner-whitened condition numbers via the generalized
  eigenproblem). Diagnostics conventions copied from `posterior.py` (R̂ = max(rank, folded),
  bulk-ESS, worst-parameter reporting). Param labels: `names.npy` confirmed reversed (C-8, see
  `carousel-mclmc-sampling.md` 2026-07-02 entry) — harness only ever uses sorted keys /
  sorted-key bijector probe. NOTE: `_h1h2_diag` is the FULL 32-param carousel; if the minimal
  case is chosen for T0/T1, a `build_model.py` + `z_best.npy` data-dir for it must be extracted
  from `carousel_sampling_minimal_example.ipynb` first. Pending human decisions: which
  hard-to-sample system for T0+T1 (must be the SAME system for both — T1's threshold is the
  T0 band; leaning carousel-minimal), which easy system as the clone-pipeline positive control,
  then grader approval before any Slurm submission.

- **2026-07-01** — Log created. Hypothesis space (H0–H3), observations O1–O6, claims C-1/C-2,
  and pre-registered experiments T0–T5 with decision tree, distilled from human observations +
  discussion (conversation, 2026-07-01). No runs executed yet. All checkpoints awaiting
  grader approval.

---

## Open questions

- Did float64 *eliminate* the small-scale noise floor (O5) or merely lower it below the old
  detection threshold? (T3's float64-vs-float32 contrast addresses this directly.)
- Is the sampled objective a profile likelihood or a true marginal (Gram log-det)? (T4 audit.)
- Which literature benchmarks are actually comparable — do any published "hard posterior"
  examples contain an inner linear solve? Worth a short literature check before writing up
  any conclusion about why lensing differs.
- If H2 wins: what is the right NFW/mass reparameterization, and can the preconditioner form
  (diagonal vs. full) be upgraded instead?

---

## Primary source — the human's original statement (2026-07-01, verbatim)

The observations O1–O6 above are a distillation of the following; per the primary-sources rule
(`AGENTS.md`, carried-over rigor rules), read the original rather than trusting the summary:

> I'd like to put less of an emphasis on curvature and nongaussianity in the sampling
> diagnostics, but it's brought to mind something that I don't understand. For context,
> typically, lensing posteriors are significantly harder to sample than many of the examples
> cited in literature. I really don't know why that is. For the most part, the typical sets
> almost always all look very Gaussian in the cornerplots when sampling converges. Other
> agents I've worked with have cited banana-like curvature, but it's always seemed like a
> cop-out to me, and they've never been able to definitively produce diagnostics showing this
> curvature. The closest they've gotten is Hessian-based diagnostics, but they always look
> very rough and unstable, with no clear trend that would indicate some form of consistent
> curvature. In my experience, Hessian estimates at points tend to be not representative of
> the lensing posteriors as whole, even near the typical set. The high-dimensionality makes
> most diagnostics inconclusive or non-functional. Additionally, the solutions that are
> typically proposed (normalizing flows, tempering, etc) don't seem to help that much, or
> actively make things worse. I'd almost like to blame bijectors, numerical instabilities, or
> something else. But every attempt I've made at diagnosing this has resulted in an
> interminable rabbit hole. What ideas do you have about why our posteriors are so hard to
> sample?

---

## 2026-07-03 — NEW INVESTIGATION ARM: carousel minimal case (old NFW_ELLIPSE vs NFW_ELLIPSE_EINSTEIN)

**Status: proposed (UNCERTIFIED). Pre-registration for Phase A (T15), B1 (T16), B2 (T17). User-approved scope: A + B1 + B2, orchestrator supervises subagents and reports per-phase.**

Context: user reports old parameterization (Rs, alpha_Rs; wide priors) samples very badly with second-mode
indications; NFW_ELLIPSE_EINSTEIN (Rs, theta_E ~ N(13,1)) accidentally discovered to fix it; even the
EINSTEIN arm suspected slower than Gaussian. Goal: mechanism, no bad-parameterization cop-out; feed the
diagnosis playbook. Classification: **structurally wrong vs fine-tuning → structural** (mechanism ID).

### Baseline harvest (zero-GPU, read-only, from user's 10k/10k seed-42 runs of 2026-07-03)
- Runs: `experiments/sim_carousel/messy_tests/minimal_case_{oldbij,newbij}` (8 ch, 10000/10000, standard
  MCLMC config, MAP(chi2_red 1.186) -> diag_qz bridge -> MCLMC; model: lstsq amplitudes 73 shapelet bases
  UNREGULARIZED, supersample=1, multi-plane 0.49/1.432 deflection_ratio; model cards identical except NFW type).
- oldbij: min bulk-ESS **34.9**/80k (z[9]), max Rhat 1.178. Chain 0 parked **3-4 sigma** out along a coherent
  ~6-param direction (z9 -3.2sig; z2,z3,z5,z7,z12 +1.1..1.8sig) for first ~half of post-burn-in, migrates in
  Q3, equilibrated Q4 → metastable basin, escape ~1e4 steps (this seed).
- newbij: min bulk-ESS **725** (z[0], 4x behind next; median 2694), Rhat<=1.02. ~0.9%/draw worst vs ~11%/draw
  sys60-clone benchmark → plausibly ~10x sub-Gaussian even after reparam (needs clone control = T17).
- xi tails nearly IDENTICAL across arms (frac xi>10: 0.124 vs 0.104; p99 1290 vs 635) → reparam fixed
  basin/ridge geometry WITHOUT touching stiff micro-structure → two separable diseases suspected.
- Dead end closed: L = sqrt(dim) exactly in ALL runs incl. healthy sys60 → recording behavior, not a lead.

### Hypotheses
- **HC1 (coordinate geometry):** data pins theta_E=f(Rs,alpha_Rs); Rs weak → thin CURVED ridge in old coords;
  EINSTEIN coords straighten it.
- **HC2 (prior surgery):** metastable basin exists under old (wide) priors; theta_E~N(13,1) suppresses it.
  NOT a pure reparam — coordinates AND prior changed together; must decompose.
- **HC3 (shared micro-stiffness):** xi tails in BOTH arms from a mechanism sys60 lacked — lstsq Gram
  near-degeneracy (73 unregularized bases) and/or ss=1 render structure (real data + chi2_red 1.19 floor →
  T14b predicts render comb decoupled; lstsq is fresh suspect). Phase C (not yet approved) probes this.

### T15 (Phase A: decomposition; one short GPU job, no new sampling)
Sorted-key names (C-8-safe); z→theta dumps of both runs; bound-crowding census; basin (oldbij chain-0
Q1 mean) vs main mode: physical location, theta_E value, log-likelihood/log-prior SPLIT under BOTH
parameterizations; ridge-spine curvature in (Rs, alpha_Rs) from newbij pushforward.
- **P1 (HC2):** |theta_E(basin) - 13| >= 2 (>=2 prior sigma) → new prior penalizes basin >=2 nats vs mode.
  **F1:** |theta_E(basin)-13| < 1 while Delta-loglike(basin vs mode) >= 10 nats → prior-surgery dead.
- **P2 (HC1):** spine quadratic sagitta over central +-2sigma window >= 0.5x transverse sigma.
  **F2:** sagitta < 0.15x transverse sigma → curved-ridge-straightening dead.
- **P3 (context, not falsifier):** newbij pushforward bulk overlaps oldbij bulk (chain-0 Q1+Q2 excluded)
  within ~0.5 pooled sigma per marginal; >1 sigma shifts in non-NFW params flag prior leakage.
- **P4 (census):** flag any crowding fraction > 0.3 (Vela precedent 0.984).

### T16 (B1: T0 seed band, both arms, 3 seeds, standard 8x2000/2000, arms' own qz)
- **P5:** oldbij min-ESS in [8,120] all seeds; >=1/3 seeds shows a >=2sigma-displaced chain persisting
  >=1000 post-burn-in draws. **F5:** all seeds min-ESS > 300 AND no displaced chain → 10k pathology
  seed-atypical at 2k scale; reassess before Phase C.
- **P6:** newbij min-ESS in [400,1500]; SAME worst param (z[0]'s physical name) in >=2/3 seeds.
  **F6:** worst param random across seeds AND band indistinguishable from clone band → z[0] lag is noise.

### T17 (B2: T1 Gaussian clones, both arms; clone fitted to that arm's 10k post-burn-in samples;
clone run's qz = the REAL arm's qz — NEVER the fitted clone covariance; standard config)
- **P7:** clone min-ESS >= 1200 both arms (dim 14 < sys60's 22; sys60 clones hit 1718-1851).
- **P8:** clone-ratio (clone/real min-ESS, matched 2000/2000): old >= 10; new in [2,15].
  **F8a:** new ratio <= 1.3 → newbij effectively Gaussian-limited (user impression = sampler variance).
  **F8b:** old ratio <= 3 → 10k pathology mostly transient, not stationary hardness.

Artifacts → `experiments/why_hard_to_sample/results_carousel/{phaseA,old,new}/`. All results UNCERTIFIED
until user grades. Strict separation: nothing written under experiments/sim_carousel/.

---

## 2026-07-03 (T15+T16+T17 RAN — carousel minimal case Phase A+B: prior-surgery dead, three-disease decomposition)

**Status: proposed (UNCERTIFIED). Artifacts: `experiments/why_hard_to_sample/results_carousel/{phaseA,old,new}/`.
System modules `systems/carousel_min_{old,new}/` GPU-verified (qz hashes byte-exact vs pipeline manifests;
chi2@z_best rel 3.1e-8 old / 2.6e-6 new — the latter adjudicated as evaluation-path float noise via logp
discriminator (0.12 nats, wrong-prior scale is O(1)+) + notebook priors verified character-exact + source
mtimes predate the reference runs; --verify gate documented at 1e-5).**

### Measured vs registered
- **P1 MISSED / F1 FIRED — then instrument-corrected (T15b).** theta_E(basin)=13.792, only 0.79 from
  prior mean (P1 predicted >=2). At cloud MEANS the basin looked +1284 nats better in likelihood — but
  means of curved sheets are off-support (the T14 lesson): per-sample loglike (512/group) shows bulk
  median −119472.8 vs basin −119485.0, 5–95% DISJOINT. **Basin is 12 nats WORSE on-support; the +1284
  was pure off-support artifact** (bulk mean sits ~1500 nats below its own samples). Prior deltas
  b-vs-m <1 nat under BOTH priors ⇒ **HC2 prior-surgery dead**: the theta_E prior neither creates nor
  kills the basin.
- **Basin identity resolved: it is the MAP-init region.** |z_best−basin|=1.84 pooled-σ vs
  |z_best−bulk|=6.19σ (z9 offset +0.09σ). All chains init in a 1e-3 ball at z_best ⇒ the "second mode"
  = under-converged MAP shelf ~7–12 nats below the posterior typical set, displaced along
  z9=src4-center_x & co. Old-coord escape time ~1e4 steps (ref-run chain 0 parked ~15k steps).
  P5 sub-check: 2/3 old T0 seeds show a >=2σ-displaced chain persisting >=1000 draws, BOTH along z9.
  NEW arm's z_best is 14.3σ from its posterior mean — also badly under-converged — but escapes in
  <10k steps. Matches the carousel diagnosis memory: under-converged MAP + curved degeneracies.
- **P2 MET, refined.** Old-native (Rs,alpha_Rs) spine sagitta/transverse = 1.71 (newbij pushforward
  1.65 — same physical ridge; conversion cross-check: loglike at identical physical points matches
  across arms to 1e-5 nats). NEW-native (Rs,theta_E) = 0.436: **EINSTEIN straightens the valley ~4x
  but does NOT flatten it** (0.436 > F2's 0.15). HC1 supported as partial mechanism.
- **P3: REAL prior leakage, large.** Old-bulk vs newbij-pushforward 1-D shifts: src5 beta 6.3σ,
  src5 center_y 4.8σ, mass e1 2.7σ, Rs 1.6σ (old Rs 5-95% [79.5,93.4] vs new [85.5,99.5] — presses
  the U(20,100) bound). Newbij's typical loglike (−119486.2) sits 13 nats BELOW old bulk: the tight
  theta_E prior trades fit for prior along the near-flat Rs valley. **The two parameterizations
  sample materially different posteriors; the switch is not a pure geometry fix.**
- **P4: no crowding flags (>0.3) either arm** (new-arm Rs upper tail approaches but does not flag).
- **T16/P5 MET:** old min-ESS band [12.5, 17.9] (within [8,120]), R̂ 1.37–1.71; worst params
  src4-center_x (z9) ×2 seeds, alpha_Rs ×1.
- **T16/P6 SPLIT:** same worst param 3/3 seeds ✓ — and it is **Rs**, the along-valley coordinate.
  But band [400,1500] MISSED HARD: measured 17–65 (R̂ 1.09–1.41). The 10k reference run's 725 was
  flattered by long adaptation; **at the standard 2000/2000 config the EINSTEIN arm is nearly as
  slow as the old arm** (gain ~1.2–4x, not 20x).
- **T17/P7 SPLIT:** old clone 2243 ✓ (>=1200); **new clone 839 ✗** — even the pure Gaussian fit of
  the newbij posterior is adaptation-limited at 2k from the real 1e-3-ball qz (long Rs axis).
- **T17/P8:** old clone-ratio 125–180 (>=10 ✓, F8b dead). New clone-ratio 13–49 — EXCEEDS the
  predicted [2,15] upper edge; F8a (Gaussian-limited) dead. **Both arms far from Gaussian-limited;
  user's suspicion about the EINSTEIN arm confirmed and understated.**

### Synthesis: three separable diseases (carousel minimal case)
1. **Under-converged MAP init (both arms):** 500-step/64-particle MAP lands 7–12 nats below and
   several σ away from the typical set (old best_step 421; new 499 = still descending). Chains must
   burn OUT; in old coords the escape is ~1e4 steps (the entire "second mode" phenomenology).
2. **Coordinate curvature (old arm):** the theta_E level-set valley is bent in (Rs,alpha_Rs)/z-space
   (1.71 sagitta ratio); a global mass matrix cannot align with a bent valley → eps pinned, R̂ 1.4–1.7,
   min-ESS ~15. EINSTEIN straightens ~4x → escape + adaptation succeed by 10k.
3. **Long near-flat Rs valley (both arms; dominant residual in new):** Rs effectively unidentified
   (position along valley prior-dominated; the two priors park it in different places — the leakage
   above). Along-valley diffusion sets the new arm's floor (Rs worst in ALL seeds; even its Gaussian
   clone hits only 839 at 2k). HC3 (shared xi tails / lstsq Gram) untested — Phase C.

### Playbook seeds (proposed)
- Cornerplot "second mode" ≠ posterior mode until checked against init: compare suspect cluster to
  z_best location (cheap, zero-GPU) — here it WAS the init.
- Means of curved posteriors are off-support: never evaluate logp at cloud means; use per-sample
  distributions (T15b instrument).
- Reparameterizations that change priors are posterior CHANGES: check pushforward overlap (P3-style)
  before attributing speedups to geometry.
- Clone-ratio ladder localizes hardness: real-vs-clone (nongaussianity+geometry) and clone-vs-ideal
  (adaptation burden from qz) are different diseases with different fixes.

### Orchestration notes
- Subagent A (system modules) + B (T15 script & wiring): 1 real bug caught in review (P3/marginals
  column-order mismatch between arms' sorted-name orders — would have compared alpha_Rs to center_x),
  1 cosmetic (crowding-chart leaf-name collisions), 1 stale print ("isotropic 1e-2 ball") fixed.
  Registered clone-qz constraint verified intact in run_t1_clone.py (qz from load_target line ~90).
- OLD-arm priors were reconstructed from notebook comments (the live cell now holds only the NEW arm);
  validated end-to-end by chi2/logp at z_best (3e-8/5e-3 nats).
- build_clone gained --exclude-chains (old-arm clone fit excludes chain 0's transient; logged in its
  manifest); back-compat 'position' path unchanged. run_t0 gained --min-seeds (default 4 preserved;
  carousel passes 3 per pre-registration).

### Open next (need fresh checkpoints + user approval)
- Phase C: lstsq Gram conditioning + xi-spike correlation along transects (HC3); T3-style transects
  along z9 (basin escape direction) and Rs valley; GN survey adapted to lstsq scene.
- MAP-quality arm: longer/better MAP (or SVI bridge) → does the old arm's "second mode" phenomenology
  vanish? (Cheap, decisive for disease 1.)
- Adaptation arm: clone run with longer burn-in or wider qz → separates disease-3 intrinsic diffusion
  from adaptation burden.
- Identifiability: is Rs unidentified in the DATA or gated by the lstsq layer? (ties to HC3).

---

## 2026-07-03 — pre-registration: T18 (MAP-quality arm + MAP-convergence diagnostic) and T19 (Phase C: Gram-vs-xi)

**Status: proposed (UNCERTIFIED). User-approved. Classification: structural (mechanism tests).
User's future items (observable-anchored reparameterization; removing hard prior bounds on
weakly-constrained params) REGISTERED for later, out of scope here.**

### T18 — MAP-quality arm (old arm) + practical MAP-convergence diagnostic
Motivation: T15-T17 identified the old arm's "second mode" as the under-converged-MAP init region
(loglike(z_best) = −119479.5 sits 6.7 nats BELOW the bulk samples' median −119472.8, when a
converged mode should sit ABOVE the typical set by ~dim/2 ≈ 7 nats).
Design: (a) re-run the OLD arm's MAP with the same optimizer family but 10x steps (5000) and
n_samples 64, outputs ONLY under results_carousel/old/t18/ (never touching user dirs); build qz
by the same diag-1e-3 recipe around the improved z_best (logged deviation: different qz center
than reference); (b) T0-style 3 seeds x 8 x 2000/2000 from the new init; (c) ship
t18_map_quality.py, a reusable diagnostic trio requiring NO ground truth:
  D1 trajectory-slope: best-so-far lp still improving in final 10% of steps -> WARN
     (necessary, not sufficient: old arm plateaued at step 421 yet was 7 nats short);
  D2 Newton decrement at z_best: lambda = sqrt(g^T H^{-1} g) (exact 14-dim Hessian of logp via
     jax); expected logp gain of a Newton step; > 0.5 nat -> FAIL;
  D3 mode-vs-typical gap (POST-HOC, zero extra compute): loglike(z_best) − median per-sample
     loglike of the chains; should be ≈ +dim/2 (Gaussian ballpark); strongly negative -> init
     was off the mode. Calibration requirement: the trio must flag BOTH 500-step reference MAPs
     as unconverged AND pass the improved MAP.
Predictions/falsifiers:
- **P-T18a:** improved MAP loglike ≥ −119470 (reaches/exceeds bulk sample range; ideally ≥ sample
  max −119466.8 + a few nats). If after 5000 steps lp plateaus but loglike stays ≤ −119475 →
  optimizer stall is STRUCTURAL (e.g., lstsq jitter) — flag, do not silently extend.
- **P-T18b:** displaced-chain census 0/3 seeds (vs 2/3 at T16) — the "second mode" phenomenology
  vanishes with converged init. **F-T18b:** ≥2σ persistent chain along z9 recurs with converged
  MAP → basin is a genuine likelihood feature; reopen.
- **P-T18c:** old-arm min-ESS improves into [25, 250] but stays FAR below clone 2243 — curvature
  disease (#2) remains. **F-T18c** (curvature overweighted): min-ESS ≥ 1000 with converged init →
  init was the dominant disease, not coordinate curvature.
- Newton decrement: old ref MAP ≥ 3 nats; improved ≤ 0.5. D3: old −6.7 (measured); improved ∈ [+3, +12].

### T19 — Phase C: shared xi tails vs lstsq Gram conditioning (HC3), stored chains, both arms
Design: align stored xi (8, 20000; take [:, -10000:]) with stored post-burn-in samples_z; at a
stratified subsample of chain points (top-decile-xi steps and bottom-decile-xi steps, ~128 each
per arm + 256 uniform), build the lstsq design matrix B(z) (73 lensed+convolved bases, the EXACT
scene_prob_model code path) and record: log10 cond(G), log10 lambda_min(G) for G = B^T W B, plus
secondary carrier lambda1(GN over the 14 nonlinear params, amplitudes profiled) at the same
points. Instrument lesson from T14b encoded: stratified contrasts primary, pooled rho secondary.
Predictions/falsifiers:
- **P-T19a (HC3-lstsq):** median cond(G) over top-decile-xi points ≥ 2x the bottom-decile median
  (either arm). **F-T19a:** ratio < 1.3 AND pooled rho(log xi, log cond) < 0.1 in both arms →
  lstsq-Gram mechanism dead for the xi tails.
- **P-T19b (alternative carrier):** if Gram is flat but lambda1(GN) stratified ratio ≥ 3 →
  curvature spikes (caustic-crossing-like, physical) carry xi, not the linear layer.
- **P-T19c (sanity):** xi tail fractions recomputed on the alignment match the Phase-A harvest
  (0.124 / 0.104) — alignment check, hard gate before any correlation is read.
Note: T19 is independent of T18 (stored chains); any NEW transect sampling would use T18's
improved MAP if needed later.

---

## 2026-07-03 (T18+T18b+T19 RAN — the optimizer cannot reach the typical set; Gram hypothesis dead; xi carrier still open)

**Status: proposed (UNCERTIFIED). Artifacts: `results_carousel/old/t18/` (map, quality_*.json,
t18b_logp_gap.json, t0_seed*.npz + census), `results_carousel/phaseC/` (t19). One infra incident:
first launch lost the T18 seeds stage to run_t18.sh's 45-min srun cap and T19 to a CUDA-binding bug
(CUDA_VISIBLE_DEVICES=1 inside an srun --gpus=1 step that exposes its GPU as device 0); both rerun
on the same allocation, no data loss.**

### T18 — measured vs registered
- **P_T18a FAILED, structural branch fired (registered contingency).** 5000 steps (10x reference)
  gained only +0.32 nats of logp; best-so-far flat from step ~500 (-119514.56 -> -119514.24).
  loglike(z_best) = -119478.8, still ~6 nats below the bulk samples' median. z_best moved only
  0.63 posterior-sigma (L2 1.35 sigma) from the reference MAP.
- **T18b discriminator: verdict (b) NON-GLOBAL LOCAL MAX.** In full JOINT logp, ALL THREE MAPs sit
  BELOW their own posterior draws: mode-minus-median logp gap = -5.27 (ref_old), -5.45 (ref_new),
  -4.84 (improved); 441-475 of 512 draws BEAT the "mode". Yet D2 certifies the improved point as a
  sharp local max (Newton gain 0.044 nats). ⇒ **the logp landscape has micro-structure that traps
  adabelief (64 particles!) AND Newton locally, ~5 nats below and ~1σ away from the typical set.**
  "Run MAP longer" is structurally useless here — 10x steps bought 0.3 nats.
- **P_T18b/P_T18c: PREMISE NOT ESTABLISHED — do not over-read.** Seeds off the "improved" MAP:
  min-ESS 14.7/17.4/16.0, R̂ 1.39-1.53, worst params src4-center_x x2 + alpha_Rs x1, 1/3 seeds with a
  persistent displaced chain — statistically identical to T16 (12.5/17.9/14.4, 2/3). F_T18b fired
  NOMINALLY but its interpretation ("basin is a genuine likelihood feature") is VOID because the
  intervention failed to move the init (0.63σ, 0.3 nats): this was a replica of T16, not a test of
  a typical-set init. The "does a good init fix the old arm" question remains OPEN and now requires
  a different init construction (e.g., best-logp posterior draw from the newbij run mapped to old
  coords, or SVI) rather than more optimizer steps.
- **MAP-convergence diagnostic (the user-requested deliverable) — honest calibration:**
  - D1 (trajectory slope): one-sided. PASSes bad MAPs that plateau (both refs plateaued short).
  - D2 (Newton decrement): certifies LOCAL optimality only. FAILed both refs (+8.97 nats predicted
    gain for ref_new; indefinite/large for ref_old) = sufficient evidence of badness; but PASSed
    the improved MAP which is still 5 nats sub-typical ⇒ **a D2 PASS is NOT evidence of a good MAP
    on rugged landscapes.**
  - **D3 (mode-vs-typical gap) is the workhorse:** zero extra compute, catches ALL THREE bad MAPs
    in BOTH loglike form (-6.7/-12.3/~-6.0) and the coordinate-consistent LOGP form (-5.3/-5.5/-4.8;
    T18b). Registered CALIBRATION (improved must show 0 FAILs) is **NOT-MET** once D3 is evaluated
    honestly (improved's own seed chains sample the same posterior; gap ≈ -6 ⇒ D3 FAIL).
  - **Playbook rule (proposed):** after ANY sampling run, compare logp(z_init/z_best) to the
    per-sample logp distribution of the chains. Mode below the bulk median ⇒ bad init/MAP,
    regardless of optimizer trajectory or local curvature. D2-FAIL confirms badness cheaply
    BEFORE sampling; D2-PASS proves nothing global.
- Amendment logged: D3's original loglike basis was my design; T18b showed the logp form is the
  coordinate-consistent one (both now reported; thresholds unchanged, both fired identically here).

### T19 — measured vs registered (alignment gates PASSED to 4e-5 / 0)
- **P_T19a FAILED everywhere:** cond(G) top-decile-xi / bottom-decile ratio = 1.011 (old/pooled),
  1.015 (old/excl-basin), 1.079 (new) — the Gram matrix of the 73-basis lstsq layer is FLAT across
  xi strata. Spearman rho(log xi, log cond G) = +0.03/+0.04/+0.11.
- **F_T19a technically does not fire** (new-arm rho 0.110 > 0.10 by a hair) but the registered
  intent is settled: **the linear-amplitude layer's conditioning does NOT carry the xi spikes.
  HC3-lstsq-Gram is dead.**
- **P_T19b branch: neither carrier.** lambda1_GN stratified ratios 1.59/1.62/0.47 (all < 3);
  rho(log xi, log lambda1) = +0.37/+0.42 old (mild), -0.16 new. Caveat logged: probes are AT stored
  draws; sys60's spikes had ~1-step widths, so flank crossings BETWEEN draws are undersampled —
  a stratified contrast this flat is still hard to square with a curvature carrier, but the
  resolution limit is real.
- **Convergent hypothesis (NEW, untested — T20 candidate):** the same logp MICRO-TEXTURE that traps
  the optimizer (T18b) is the natural remaining candidate for the xi tails: micro-roughness =>
  energy errors during leapfrog regardless of smooth-curvature carriers. sys60's T3 falsified
  micro-roughness THERE (no lstsq); the carousel HAS the unregularized 73-amplitude lstsq layer, and
  gradient noise through the solve is scale-compatible. Direct test = T3-style transect
  second-difference scan on carousel_min_old (machinery exists; float32 control ready) + a
  gradient-consistency probe (finite-difference logp vs autodiff gradient along a transect).

### Open next (need user approval)
- **T20:** carousel micro-roughness transects (t3 machinery, both arms, float64 + float32 control)
  + gradient-consistency probe. Decisive for the unified texture hypothesis.
- **T21:** typical-set init arm — init old-arm chains at the best-logp posterior DRAW (not any
  optimizer output); tests the original "does a good init fix it" question that T18 could not.
- Diagnostic hardening: fold the logp-form D3 into t18_map_quality as the primary; document
  D2's one-sidedness in the tool's output text.

---

## 2026-07-03 — pre-registration: T20 (carousel logp micro-texture: transect roughness + gradient consistency + step-segment probe)

**Status: proposed (UNCERTIFIED). User-approved ("the MAP getting stuck in a small local minimum
fits with my experience"). Classification: structural (last-standing unified hypothesis for the
xi tails + optimizer trap).**

Hypothesis HT20: carousel logp (both arms; lstsq layer present) carries genuine micro-texture in
float64 — value roughness and/or gradient inconsistency at scales ~1e-3..1e-2 posterior-sigma —
deep enough (~1-5 nats cumulative) to trap optimizers (T18b) and to generate leapfrog energy
errors (the shared xi tails T19 left unexplained). Contrast: sys60 (NO lstsq) was smooth to the
machine floor on all 7 directions (T3).

Design (three probes, both arms; t3_transects.py REUSED unchanged for 1-2):
1. **Value roughness (t3 D2 ladder):** 7 directions (incl. z_best->bulk escape direction, z9 axis,
   Rs-valley axis, randoms), float64 + float32 instrument-control arm (control = detector must fire).
   Carousel system modules gain the WHTS_FLOAT32_CONTROL env gate (skips the forced x64 update and
   the rebuilt-qz hash check under the control ONLY, loud warning printed; manifest-hash guard kept).
2. **Gradient consistency (t3 FD-AD analysis):** autodiff vs central-FD gradient along the same
   directions.
3. **Step-segment probe (NEW, t20_step_segments.py):** for 32 adjacent-draw pairs with top-1% xi
   and 32 calm pairs (xi < median), evaluate logp at 33 points along the straight z-segment between
   the two draws; deviation from the smooth (quadratic-fit) profile measures what the integrator
   actually stepped over. xi->step alignment (off-by-one) documented and checked both ways.

Registered predictions/falsifiers:
- **P-T20a:** float64 D2 roughness above 100x the empirically measured float64 machine floor
  (eps_f from t3) at h <= 1e-2 sigma_dir on >= 2 of 7 directions (old arm).
  **F-T20a:** all directions at machine floor (sys60-like) -> value-texture dead.
- **P-T20c:** FD-AD relative gradient error >= 1e-5 at the error-minimizing h on >= 2 directions
  (smooth-function expectation ~1e-8..1e-7). **F-T20c:** gradient consistent at float precision
  everywhere AND F-T20a -> texture hypothesis dead entirely; xi carrier reverts to
  integrator/momentum-space candidates (between-draw flank crossings remain unprobed by T19).
- **P-T20d:** >= 50% of top-xi segments show max |logp - quadratic fit| >= 1 nat; < 10% of calm
  segments do. **F-T20d:** top-xi segments as smooth as calm ones -> xi is NOT value-texture on
  the traversed path; texture may still explain the optimizer trap but not xi (diseases split).

---

## 2026-07-03 (T20 RAN — texture verdict: the carousel's disease is GRADIENT noise through the lstsq layer; it explains the optimizer trap, NOT the xi tail; F-T20d fired cleanly)

**Status: proposed (UNCERTIFIED). Artifacts: `results_carousel/phaseC/t20/{t3_old_f64,t3_new_f64,t3_old_f32,inputs}/`,
`t20_segments.npz`, `t20_summary.json`, PNGs. Infra: two OOMs fixed by chunking t3's f_eval and
t20's segment evaluator to 16-point batches (carousel lstsq render ~52MB/pt of intermediates; the
1056-pt segment batch tried a 51.7GiB Gram matmul). t3 reused UNCHANGED except that mechanical
chunking; carousel axes injected by module-global override (sys60's hardcoded WORST_ESS_PARAMS
includes .../gamma which does not exist here and would AssertionError).**

### Measured vs registered
- **P-T20a (value roughness) MET in letter, refined in reading.** float64 logp carries an
  effective value-noise quantum ~1e-6..3e-5 nats = **~1e5-1e6x the float64 roundoff floor**
  (|f|·eps ≈ 2.7e-11); D2 ladders show a 1/h^2 noise regime crossing the true-curvature plateau
  at h ~ 1e-5..1e-4 z-units. REFINEMENT (plots before metrics): the crossover sits BELOW the
  sampler band [eps_step/100, eps_step] — within the band every direction shows a clean smooth
  plateau. Value texture exists but the sampler never resolves it.
- **P-T20c (gradient inconsistency) MET overwhelmingly — the headline.** FD-vs-AD relative
  gradient error at its h-optimum: carousel OLD 3.1e-5..3.0e-3 (7/7 directions), NEW
  1.7e-5..4.0e-4 (7/7) vs **sys60 baseline 1.4e-9..5.0e-8** — a 1,000–100,000x gap. float32
  control fires at 0.03–0.23 (instrument valid). The carousel-specific structural difference is
  the 73-amplitude UNREGULARIZED lstsq layer; AD-vs-FD disagreement of this size means the
  gradient the optimizer AND integrator consume carries ~1e-4..3e-3 relative noise
  (|g|~1e3 nats/z-unit -> absolute noise ~0.1..3 nats/z-unit).
- **P-T20d FAILED / F-T20d FIRED (clean negative, both arms, both xi alignment conventions).**
  Along ACTUAL sampler steps: max |logp - quadratic| ~ 5e-4 nats median, 0/32 top-xi and 0/32
  calm segments reach 1 nat; top and calm are indistinguishable. **The xi spikes are NOT
  value-texture on the traversed path.** Per the registered falsifier: texture may explain the
  optimizer trap but not xi — the diseases split.

### Synthesis
1. **Optimizer trap (T18b) now has a quantitative mechanism:** near the flat-valley top the true
   gradient is O(1-10 nats/sigma); the lstsq-layer gradient noise is ~0.1-3 nats/z-unit; AdaBelief
   (any first-order method) stalls where signal ~ noise — measured ~5 nats short, insensitive to
   step count (5000 steps bought 0.32 nats). "MAP quality" here is GRADIENT-NOISE-limited, not
   iteration-limited. D2's Newton certificate at such a point is likewise noise-scale-local.
2. **xi tails remain unexplained in the extreme** (not Gram cond T19, not lambda1-at-draws T19,
   not path texture T20d). Uniform gradient noise CAN account for the elevated xi BASELINE
   (per-step energy error ~0.005-0.15 nats -> xi ~ 0.004-3, cf. medians ~0.015, p90 ~10-19) but
   the 10-decade tail (xi up to 1e5-3e4 post-burn-in) needs Delta-e ~ 2-15 nats — 20-100x above
   the noise budget unless |grad| or the relative error spikes locally. OPEN.
3. **Actionable prediction (registered for a future fix-arm):** physical regularization of the
   lstsq (the model card itself warns about the unregularized null space) or analytic amplitude
   marginalization should (a) restore FD-AD to ~1e-8, (b) let MAP reach the typical set
   (D3 gap flips positive), (c) calm the xi baseline. If (a) happens without (b), the trap has
   another source; each limb is falsifiable.

### Playbook seeds
- Run the FD-vs-AD gradient-consistency ladder on any system with linear-solve layers BEFORE
  trusting optimizers/HMC-family samplers: fd_ad_min >= ~1e-5 marks a gradient-noise-limited
  system (sys60-clean is ~1e-8).
- D2 value-ladders must be read against the SAMPLER BAND: noise below the band is harmless to
  sampling but fatal to optimizers whose signal shrinks to zero at stationarity.

---

## 2026-07-03 — CORRECTION to T20 attribution (user challenge; orchestrator over-claim owned)

The T20 entry attributed the FD-vs-AD gap to "gradient noise through the lstsq layer." Three
cheap checks weaken/redirect that:
1. **cond(G) is BENIGN:** T19's stored arrays give log10 cond(G) median 3.53 (old) / 2.63 (new).
   cond*eps64 ~ 1e-12 — five decades short of the observed 1e-4..3e-3. Normal-equation
   conditioning cannot be the mechanism.
2. **The solve is branchless and clean:** `_solve_normal_eq_with_fallback` (jax/simulator.py:90)
   UNCONDITIONALLY jitters (1e-6*diag_mean) and uses jnp.linalg.solve (LU) chosen precisely for a
   well-conditioned VJP; no fallback branch exists at runtime. With cond ~ 3e3 both forward and
   backward passes are numerically healthy.
3. **The one clean numerical difference vs sys60 is conv precision:** carousel sets
   `conv_precision="float32"` (from the user's notebook); sys60's SimulatorConfig leaves it None
   (= ambient float64). Both systems convolve with a PSF. float32 PSF convolution injects
   per-pixel model error ~1e-6 counts -> logp VALUE noise ~1e-5..1e-3 nats — exactly the
   magnitude the D2 noise regime shows.
4. **Instrument re-read:** fd_ad measures |FD − AD|. With float32-conv VALUE noise, the corrupted
   leg is plausibly FD (differencing a bumpy function), while AD (whose conv segment merely runs
   its backward pass in float32, rel ~6e-8) may be nearly clean. If so, T20's "the optimizer
   consumes noisy gradients" inference is WRONG, and the T18 trap needs a different mechanism:
   a GENUINE ~5-nat-deep local max (T18b: -H PD, all draws above it) and/or thin-curved-ridge
   optimizer dynamics failure (stiff width sigma~2.4e-4 vs adabelief lr 1e-2; every one of 64
   particles ended below a random posterior draw — a dynamics signature, not a noise one).

### T22 (proposed, needs approval): conv-precision discriminator — one config field, three arms
- Arm A: carousel old, conv_precision="float64", fd_ad ladder (2-3 directions) + D2 ladder.
  **P-T22a:** fd_ad_min drops to <= 1e-7 and the D2 noise regime collapses onto the roundoff
  model. **F-T22a:** unchanged -> conv exonerated; suspicion returns to the lstsq VJP / basis
  generation; run Arm B.
- Arm B (only if F-T22a): frozen amplitudes (constants at z0's solution), conv float32:
  isolates the solve's contribution.
- Arm C: 500-step MAP under conv float64, same seed. **P-T22c (my current read):** NO
  improvement (trap is genuine-local-max/dynamics, not noise) — if MAP suddenly reaches the
  typical set, the noise-stall story was right and this correction over-corrected.
Cost ~15-20 min GPU total.

---

## 2026-07-03 (T22 RAN — conv_precision=float32 WAS the texture source; lstsq fully exonerated; the MAP trap survives clean numerics)

**Status: proposed (UNCERTIFIED). Artifacts: `results_carousel/phaseC/t22/{t3_old_conv64/,
map_conv_float64.*}`. Arm B (frozen amplitudes) skipped per pre-registration (F-T22a did not fire).
Infra: arm C (64-particle MAP, conv f64) needs the 80GB pool — f64 conv doubles the batch
convolution buffers (15.9GiB alloc OOM on 40GB).**

### Measured vs registered
- **P-T22a MET decisively.** With WHTS_CONV_PRECISION=float64 (one config field), fd_ad_min on the
  old arm collapses on ALL 7 directions: 5.3e-5..3.0e-3 -> **3.3e-9..1.3e-6** (5/7 inside the
  sys60-clean 1e-9..5e-8 band; softest 1.3e-6 and random_2 1.9e-7 within ~1 decade). The
  1,000-100,000x FD-AD anomaly was the **float32 PSF convolution**, full stop. The lstsq layer is
  exonerated on every count (cond(G)~3e3, branchless clean-VJP solve, and now this).
  (observed_quantum did not move consistently — as established, it measures g*probe-spacing on
  smooth functions, not noise; not a discriminator.)
- **P-T22c HELD.** Reference-matched 500-step MAP under clean f64 conv: best_lp = -119514.8002
  vs f32-conv reference -119514.8314 (0.03 nats), still ~5 nats below the draws' median logp.
  **The T18 optimizer trap is NOT numerical noise: it is a genuine local maximum of the true
  posterior (T18b: -H PD) plus, plausibly, thin-curved-ridge first-order dynamics failure
  (stiff width sigma~2.4e-4 vs adabelief lr 1e-2; all 64 particles below a random draw).**

### Consequences for the ledger
- xi accounting: under f32 conv the VALUE noise (~1e-4 nats) contributes xi ~ delta^2/0.007 ~
  1e-6 — negligible; and the sampler's AD gradients were likely clean all along. So conv noise
  explains NEITHER the xi baseline NOR the tail; the T20 "gradient noise elevates the xi
  baseline" speculation is withdrawn. The xi carrier for the carousel remains OPEN (not Gram,
  not lambda1-at-draws, not path texture, not conv noise).
- The carousel's practical SAMPLING slowness was never the texture (steps traverse smooth
  terrain; the clone ladder reproduces slowness on exact Gaussians): diseases (2) curved valley
  in old coords and (3) long flat Rs valley + adaptation burden stand as the sampling story.
- Real bug worth fixing anyway: conv_precision="float32" corrupts every FD-based check and any
  value-sensitive comparison at the 1e-5..1e-3-nat level, and cost the investigation a full
  false attribution cycle. Recommended default: float64 conv for inference-grade runs (memory
  cost: ~2x conv buffers; 64-particle MAP needed hbm80g).
- Scoreboard for honesty: my T20 attribution ("gradient noise through lstsq") was wrong on both
  nouns — the source was conv f32 and the corrupted leg was FD, not the gradient. The user's
  challenge triggered the correction; T22's registered predictions then went 2/2.

### Open next
- xi-tail carrier (momentum/integrator-space accounting) — the last unexplained phenomenon.
- T21 typical-set init (old arm) — now sharper: init at best-logp DRAW vs the genuine local max.
- Optional control: MCLMC rerun under conv f64 (prediction: sampling unchanged — texture is
  sub-sampler-scale). Low priority.

---

## 2026-07-03 — pre-registration: T21 (typical-set init, conv float64 = new standard)

**Status: proposed (UNCERTIFIED). User-approved; conv_precision=float64 adopted as the standard
for all inference-grade runs from here on. Question: does sampling still struggle RELATIVE TO A
GAUSSIAN when started in the typical set? → separates stationary-phase hardness (where the xi
tail lives) from init/burn-in pathology, and bounds the xi tail's practical cost.**

Design (both arms, WHTS_CONV_PRECISION=float64 throughout):
- z_init per arm = the MEDIAN-logp draw among 512 fixed-seed-subsampled reference draws (old arm:
  chains 1-7 only) — "maximally typical", not best-logp (a max-logp point is atypical; the
  best-logp draw is recorded for the ledger but not used as init).
- qz' = MVNDiag(loc=z_init, scale 1e-3) — ONLY the center moves vs all prior arms (qz enters
  MCLMC three ways: init positions, initial inverse mass matrix, adaptation anchor; scale and
  mechanics unchanged for comparability).
- 3 seeds x standard 8x2000/2000 per arm + displaced-chain census + xi tail stats.
- MATCHED GAUSSIAN CONTROL: each arm's existing T17 clone cov, re-run with the SAME
  typical-set-recentered qz' (1 seed) — the "Gaussian started there" yardstick.

Registered predictions/falsifiers:
- **P-T21a (old):** displaced-chain phenomenology vanishes (0/3 seeds persistent) and min-ESS
  improves vs T16's [12.5,17.9] into [30,400], but stays >=5x BELOW its matched clone.
  **F-T21a-easy:** old min-ESS >= 1000 (~clone level) -> old-arm hardness was ~entirely
  init/burn-in escape; stationary xi tail practically inconsequential for the old arm.
- **P-T21b (new):** min-ESS in [30,300], still >=3x below its matched clone -> stationary
  Rs-valley diffusion dominates; typical-set init does not rescue.
- **P-T21c (xi consequence read):** frac(xi>10) in T21 runs within +-0.03 of the references
  (0.124/0.104) — xi tail is a stationary property, present from step one at typical init.
  Interpretation matrix (registered): same tail + ESS ~= clone -> tail inconsequential;
  same tail + ESS << clone -> hardness co-occurs with tail (causality still needs the
  momentum-space probe); tail gone under f64 conv -> conv contributed after all (would
  contradict T22's accounting — flag loudly).
- Sanity: conv-f64 vs f32 comparability rests on T20's sub-sampler-scale texture finding; if
  T21-old's band shifts far from T16's for reasons beyond init, flag before interpreting.

---

## 2026-07-03 (T21 RAN — typical-set init kills the mode phenomenology but NOT the hardness; the xi tail is a real-posterior property that co-occurs with a 7-70x Gaussian gap)

**Status: proposed (UNCERTIFIED). Artifacts: `results_carousel/phaseC/t21/{old,new}/` + summary.json.
All runs conv float64 (new standard), hbm80g. z_init = median-logp draw of 512 scored reference
draws (old: chains 1-7); best-logp draws recorded but unused.**

### Measured vs registered
- **OLD arm, real:** min-ESS 40.4 / 144 / 26.0 (R-hat 1.15/1.06/1.24). **Census 0/3 persistent
  displaced chains — P-T21a census MET: the entire "second mode" phenomenology is an init
  artifact; started in the typical set it never appears.** Band [30,400] MISSED LOW by seed 3
  (26.0; 2/3 seeds in band) — logged as a narrow miss. F-T21a-easy does NOT fire: 26 vs matched
  clone 1832 -> **70.4x below Gaussian**. Worst param is now Rs (2/3 seeds) — with init fixed,
  the old arm's residual bottleneck becomes the SAME Rs-valley diffusion as the new arm.
- **NEW arm, real:** min-ESS 139.9 / 144 / 144 — strikingly tight; all Rs; R-hat ~1.06; census
  0/3. IN band [30,300]. vs T16 (MAP init): 17-65 -> ~140: **typical-set init buys 2-8x AND
  removes nearly all seed variance** (the T16 spread was init/adaptation noise). Still
  **7.2x below matched clone (1008)**.
- **Matched Gaussian clones at the SAME typical init:** old 1832, new 1008 — and their xi tails
  are TINY: frac(xi>10) = 0.024 / 0.021, p99 ~ 19, max ~ 230. The real targets: frac 0.13-0.17
  (old; mean 0.154 vs ref 0.124 -> |dev| 0.031 marginally OVER the 0.03 tol, in the HEAVIER
  direction — logged as a marginal P-T21c miss, plausibly seed variance) and 0.065-0.087 (new;
  SAME as ref within tol), p99 ~ 1200-1900, max 5e4-7e4.
- **Interpretation-matrix branch (registered): "hardness co-occurs with tail (causality open)"**
  for BOTH arms: tail at reference levels + ESS 7-70x below clone. The "tail gone under f64
  conv" contradiction branch did NOT occur (consistent with T22's accounting).

### Synthesis — answer to the user's question
Started in the typical set, with clean float64 numerics, no init pathology and no mode
migration, the carousel posteriors STILL sample 7x (EINSTEIN param) to 70x (old param) slower
than their own Gaussian clones under identical sampler mechanics — while exhibiting a heavy
xi tail (p99 ~1e3) that the clones (p99 ~19) completely lack. **The xi tail is a property of
the real posterior's stationary geometry, not of init, adaptation, or numerics — and it is
the signature (cause or co-symptom) of essentially ALL the remaining sampling hardness.**
Ledger of carousel diseases, final form:
  (1) init/mode phenomenology — CLOSED (T15-T18-T21: under-converged MAP + genuine ~5-nat
      local max; typical-set init removes it entirely);
  (2) numerics — CLOSED (T22: float32 conv; fixed by config; never affected sampling);
  (3) coordinate geometry — old param's curved valley costs ~10x on top of (4) (70x vs 7.2x);
  (4) stationary Rs-valley + xi-tail hardness — OPEN, now cleanly isolated as THE remaining
      target (7.2x below Gaussian in the better parameterization, Rs worst in 8/9 runs
      incl. old arm post-fix).
Next probes registered as candidates: momentum-space xi accounting along T21 chains (what
does the integrator hit if not path texture? candidates: Rs-valley end-wall reflections,
stiff-direction resonances at L=sqrt(dim)); encounter-geometry census (T10-style) on T21 runs.

---

## 2026-07-03 — pre-registration: T23 (momentum-space xi accounting) + T24 (encounter-geometry census), both on saved T21 chains. Approved by human (conversation 2026-07-03).

**Housekeeping note:** T21 artifacts had been written to `<harness>/0/` (out-dir argument
mishap in `slurm/run_t21.sh`); moved verbatim to canonical `results_carousel/phaseC/t21/`
on 2026-07-03. Contents unchanged; the T21 adjudication used these same arrays.

**Shared data + instruments (verified before registration):** T21 npz files carry
`position (8,4000,14)`, `xi (8,4000)`, `step_size (8,4000)`, `L (8,4000)`,
`inverse_mass_matrix (1,4000,14,14)`, `nb=nr=2000` — positions span burnin+results on the
SAME axis as xi, so index alignment is direct. Both tests use the results phase only
(steps 2000–3999; builder must verify the mass matrix is frozen over that span).
Momentum proxy: u_t ∝ z_{t+1}−z_t (direction only). The xi↔displacement index convention
(does xi[t] pair with z_t→z_{t+1} or z_{t−1}→z_t?) must be resolved by TRACING the kernel
stacking in common.py/blackjax source and cited by line — NOT chosen post hoc by whichever
alignment correlates better. Curvature along motion c_t = Δz'(−H)Δz / (Δz' Σ⁻¹ Δz) with H
the z-space logp Hessian (HVP, conv float64, chunk ≤16) and Σ the sampler's preconditioning
metric as saved (builder traces the exact meaning of the saved `inverse_mass_matrix`);
Euclidean-normalized version reported as secondary. All runs WHTS_CONV_PRECISION=float64.
Instrument control: the SAME instruments run on the matched T21 clone chains.

### T23 — momentum-space xi accounting (what does the integrator hit?)
- **Competing cause hypotheses:**
  - **HW (end-wall):** xi spikes are generated at the Rs prior wall (U(20,100); the new-arm
    posterior presses Rs 5–95% = [85.5, 99.5] against the upper bound) — bijector
    log-Jacobian saturation acts as a curvature wall; spikes are reflections.
  - **HS (stiff-direction):** spikes occur when the momentum aligns with interior stiff
    directions so that eps·sqrt(c_t) reaches O(1) — integrator-stability events.
  - **H0 (neither):** spikes occur at unremarkable locations with unremarkable c_t →
    per-step local geometry does not carry xi; the remaining candidate is trajectory-level
    energy accumulation (multi-step), and this arm of local probing CLOSES.
- **Design:** spike set = steps with xi > 10 (registered tail threshold), subsampled to
  ≤512 per arm plus all top-0.1% extremes; calm controls ≤512 per arm drawn from
  xi < per-chain median, matched per chain. Columns per step: (C1) wall saturation —
  unconstrained |z_k| for every hard-bounded param (bounded list derived from the prior
  spec and printed for review); FIRST a plot of xi vs z_Rs over ALL steps (plots before
  metrics); (C2) c_t + stability number eps_t·sqrt(max(c_t,0)); (C3) direction identity —
  loadings of u_t on the fixed eigenbasis of −H(z_init) per arm; (C4) turn angle between
  successive displacements (reversals ⇒ reflection signature).
- **Orchestrator prediction (direction + magnitude):** HW favored. Spike-set median |z_Rs|
  sits at ≥ the 0.80 quantile of the calm |z_Rs| distribution (new arm; old arm: Rs or
  alpha_Rs), with reversal-like turn angles loaded on the same coordinate. Classified
  structural (a nameable mechanism), not fine-tuning.
- **Registered thresholds/falsifiers:**
  - P-T23-wall MET if spike median |z_Rs| quantile ≥ 0.80 within calm; **F-T23-wall:**
    quantile in [0.35, 0.65] ⇒ wall story dead.
  - P-T23-stiff MET if median c_t(spike)/c_t(calm) ≥ 10; **F-T23-stiff:** ratio < 2 ⇒
    stiff-resonance story dead.
  - P-T23-clone (instrument sanity): clone spike steps show neither signature (quantile in
    [0.35,0.65] AND ratio < 2). If the clone shows a "signature", the instrument is broken.
  - Both falsified + clean clone control ⇒ H0 branch is the finding (registered as a valid,
    reportable outcome — not a failure).
- **Cost:** ~2.5k HVPs + login-node array work.

### T24 — encounter-geometry census (T10-style; per-step scalar = curvature along motion)
- **Cause hypothesis:** the stationary xi tail is generated by LOCALIZED curvature
  encounters along the actual chain path (spikes with rate/width/spacing structure), not a
  diffuse background elevation.
- **Design deviation from T10, justified:** T19 already nulled lambda1(GN) at xi-stratified
  draws, so the census scalar is c_t at EVERY step of contiguous segments — 8 chains × 256
  contiguous results-phase steps, seed 1, both arms + both clones (1 HVP per step);
  lambda1(−H) power iteration ONLY at the top-12 census spikes + 12 matched same-segment
  calm points (v1 loadings for direction identity; cross-check vs T23-C3).
- **Statistics per segment + pooled (mirroring T10):** spike rate (c_t > 3× segment
  median), max/median, spike width (consecutive steps above), inter-spike spacing,
  Spearman rho(c_t, xi_t) and rho(c_t, xi smoothed w=5).
- **Predictions:** if encounters carry the tail — pooled rho ≥ 0.4; census spike rate
  within 2× of frac(xi>10) (real: old ~0.15, new ~0.08); widths 1–3 steps. Clone control:
  near-zero spike rate, small rho.
- **F-T24:** rho < 0.15 on ≥ 6/8 segments in BOTH arms AND spike-rate mismatch > 5× ⇒
  curvature-along-motion encounters do NOT carry the tail; combined with T23-H0 this closes
  per-step local probing and the next registered candidate is trajectory-level accounting.
- **Cost:** ~9k HVPs (segments) + ~0.5k (power iteration); shares one hbm80g interactive
  allocation with T23.

**Causality caveat (registered):** both tests identify what GENERATES the xi tail, not yet
that the tail CAUSES the ESS deficit; the causal link would need an intervention (e.g.
step-size/trust-threshold change watching the clone gap), scoped only if a mechanism is
named here.

### Checkpoint completion (pre-launch; added before the run per pre-run-checklist)
- **Claim class:** distributional (spike sub-population vs calm control, deterministic
  re-evaluation of saved states); a three-way DIAGNOSTIC discriminator (HW/HS/H0), not a
  single-hypothesis confirmation. **Chain link tested:** "what generates the xi tail";
  link NOT tested: "the tail causes the ESS deficit" (registered caveat above).
- **Threshold derivations:**
  - Wall 0.80 / falsifier [0.35,0.65]: under the null (spikes at random positions) the
    spike-median's quantile within calm is ~0.50 with sampling scatter ~±0.04 at n≈512;
    [0.35,0.65] is null ±~3-4 sigma; 0.80 requires the typical spike to sit in the calm
    distribution's outer fifth — far outside null scatter.
  - Stiff 10x / falsifier 2x: T19 measured benign ~2x-level lambda1 heterogeneity between
    xi strata at draws; <2x is therefore indistinguishable from already-observed background;
    10x moves eps*sqrt(c) by >3x, the scale an integrator-stability account needs.
  - T24 rho 0.4 / 0.15: 0.4 mirrors the sys60 T10 registration ("clearly load-bearing");
    0.15 is at the n=256 Spearman noise floor (2/sqrt(256) ≈ 0.125).
- **Metric blind spots (named):** (1) the wall MEDIAN statistic is blind to a minority
  (<~30%) wall-event sub-population — mitigated by reading the full xi-vs-z_Rs hexbin and
  ECDFs first; (2) the stiff statistic evaluates curvature at the step ENDPOINT — spikes
  generated by curvature VARIATION inside a step (third-derivative events) land in H0 by
  design; (3) the chord momentum proxy hides within-step reversals (a bounce completing
  inside one step shrinks |Dz| and moves the endpoint off the wall) — partially covered by
  the turn-angle columns; (4) T24 segment Spearman is bulk-dominated and blind to few
  extreme co-events — covered by rate/width/spacing stats + overlay plots.
- **Pre-committed plot appearance:** HW ⇒ xi-vs-z_Rs hexbin shows the xi>10 mass hugging
  one edge of the z_Rs range (hockey-stick), spike |z_Rs| ECDF right-shifted, spike
  turn-cos mass near -1. HS ⇒ hexbin flat in z_Rs but spike c_t ECDF right-shifted ≥10x.
  H0/falsifiers ⇒ hexbin flat AND ECDFs overlap (quantile ~0.5, ratio <2).
- **Cost:** 1x hbm80g interactive GPU, ~6.8k chunked HVPs, est. 30-60 min + smoke pass;
  login-node analysis free. Runs launched by orchestrator on OWN allocation.
- **Approval:** design described to and approved by the human in conversation 2026-07-03
  BEFORE the build; this completion block adds the missing checklist items pre-launch.

---

## 2026-07-03 (T23+T24 RAN — the xi tail is a census of FUNNEL-NECK REFLECTIONS at the low-Rs end of the valley; prior wall exonerated; neither registered branch as stated)

**Status: proposed (UNCERTIFIED). Artifacts: `results_carousel/phaseC/t23/` (npz + manifests +
5 plots/arm + t23_analysis.json), `.../t24/` (npz + manifests + segment plots + t24_analysis.json).
Smoke pass then full run, one hbm80g allocation, 292 s total GPU wall (~6.8k HVPs — far under
the 30–60 min estimate; HVPs ~0.12 s/chunk-of-8 after compile). Sanity gates all passed:
z_ref recovered to logp gap ~4e-10 both arms; Sigma frozen (max diff 0.0) all runs; bounded
lists old=6/new=5 as derived.**

### Measured vs registered (T23)
- **P-T23-wall (prior-wall version): PREDICTION WRONG — outcome landed OUTSIDE the registered
  partition.** NEW arm: spike |z_Rs| quantile-in-calm = **0.000** (registered MET ≥0.80,
  falsifier band [0.35,0.65] — neither; q=0 is an INVERSION the registration did not
  anticipate). Spikes are pinned at the LOW-Rs edge of exploration (signed z_Rs spike median
  +1.86, 5–95% [+1.71,+2.17] vs calm +3.16 [+2.35,+5.11]); the region actually pressed
  against the Rs=100 prior wall (z_Rs 5–10) is QUIET in the hexbin. **The hard prior bound is
  exonerated.** OLD arm: the statistic returned "MET" on alpha_Rs (q=0.889) but this is a
  **false positive of the registered metric**: signed z_alphaRs at spikes is −0.03 [−0.11,+0.20]
  — |z| ≤ 0.2, nowhere near bijector saturation (O(2–3)). The |z|-quantile statistic had no
  absolute saturation scale (registered-metric design flaw, logged). Old-arm spikes sit at
  BOTH ends of the visited Rs range (V-shaped hexbin; z_Rs q=0.080, low end dominant).
- **P-T23-stiff: GRAY ZONE.** Median c_t spike/calm = 6.03 (old) / 7.12 (new) — between the
  falsifier (<2, did NOT fire) and the MET bar (≥10, not met). But the registered descriptive
  column is decisive: stability number eps*sqrt(c) spike median 1.85 / p90 4.13 (new; old
  1.46 / 3.14) vs calm 0.69 / 0.57 — **the spike population crosses the O(1) integrator-
  stability threshold; calm does not.** The 10x median bar was too crude (spike set includes
  marginal xi~10–50 events that dilute the median).
- **P-T23-clone: compound criterion was MIS-SPECIFIED (registration error, owned).** The clone
  is positionally PERFECTLY clean (all bounded-param q ≈ 0.46–0.61 both arms — instrument
  works), but clone stiff ratio = 1.72 (old) / 2.60 (new) tripped the bundled "<2" clause.
  A Gaussian target SHOULD show c–xi correlation — that is universal integrator physics, at
  trivial amplitude (clone c median ~2, max ~15 vs real max ~1740).
- **C3 loadings (decisive):** BOTH spike and calm motion load ~0.95 on the SOFTEST
  eigenvector of −H(z_ref) — the chain always travels the valley floor. **The direction of
  motion does not rotate at spikes; the terrain under the same direction stiffens.**
- **C4 turn angles:** spikes show strong reversal mass at cos ≈ −1 on BOTH incoming and
  outgoing displacements (new: ~10x calm density at −1; old ~3.4x); calm is forward (+1).
  Spikes are BOUNCES.
- **Co-location:** new-arm spikes also have shear at extremes (gamma1 q=0.979, gamma2 0.967);
  theta_E uninvolved (spike/calm medians identical to 4e-3). The spike locus is a corner:
  low-Rs x extreme-shear.

### Measured vs registered (T24)
- **P-T24 MET:** pooled Spearman rho(c,xi) = 0.511 (old) / 0.539 (new) ≥ 0.4; smoothed 0.545 /
  0.599. Per-segment: old 7/8 ≥ 0.41 (one 0.02), new 8/8 ≥ 0.34. Spike widths median 1,
  p90 3 (registered 1–3). Census spike rate vs frac(xi>10): new 0.181/0.104 = 1.74x (in the
  2x band); old 0.242/0.113 = 2.14x (**marginal miss** of the 2x band). **F-T24 does NOT fire.**
- **Clone prediction PARTIAL MISS (logged):** clone spike rate low (0.042/0.069) as predicted,
  but clone rho = 0.495 (new) is NOT "near-zero" — same universal-physics point as the clone
  stiff clause; the registered clone expectation conflated amplitude with correlation.
- **lambda1 power iteration at census spikes vs calm: NO difference** (old 2.28e7 vs 2.29e7;
  new 1.69e7 vs 1.82e7) — confirms T19 with the chain's own points: the ambient top
  curvature does NOT move at spikes. This is NOT a lambda1-spike field (different disease
  from sys60's subgrid comb).
- Segment traces: episodic — chains ENTER the funnel region, bounce for tens of steps (c and
  xi co-moving on the envelope, per-event widths 1–3), and exit.

### Synthesis — named mechanism (proposed): LOW-Rs FUNNEL-NECK REFLECTIONS
The chain moves along the valley floor (softest eigendirection, loading ~0.95 always). At the
low-Rs end of the Rs valley (new arm: z_Rs ≲ 2.2, Rs ≲ ~92 under the sigmoid bij; old arm:
both ends of the visited range) — co-located with extreme shear — the curvature ALONG the
floor rises ~6–7x at the median spike and orders of magnitude in the tail, while ambient
lambda1 stays fixed: a funnel neck. Bulk-adapted eps then violates integrator stability
(eps*sqrt(c) crosses 1–4+), producing the huge energy errors (xi 10..1e4+) and momentum
reversals: the chain BOUNCES off a soft likelihood cliff INSIDE the support. The stationary
xi tail is the census of these bounces (rho ~0.5, widths 1–3, rate ~ frac(xi>10) within
~2x). Neither registered branch as stated: NOT the prior wall (HW as registered — wrong),
NOT unremarkable geometry (H0), but the HS mechanism operating at a nameable stationary
PLACE. Physics reading (interpretation, untested): with theta_E pinned to 0.1%, lowering Rs
concentrates the profile and the likelihood stiffens along the Rs–shear compensation
direction; a render-space check (T11-style) would test this.
**ESS causality (still formally open, registered caveat stands):** the bounce boundary
truncates low-Rs exploration; Rs is the worst-ESS parameter in 8/9 T21 runs — the funnel
neck is now the concrete causal candidate. Candidate intervention test: lower
desired_energy_variance (smaller eps) or clip the Rs range, watch the clone gap. Connects
directly to the user's deferred observable-anchored-reparameterization topic: the funnel
exists because the data constrains a COMBINATION (profile at the arcs), not Rs itself.

### Honest misses (this entry)
Wall prediction direction WRONG (predicted prior-wall clustering ≥0.80; got inversion 0.000);
old-arm wall statistic false-positived on a coordinate nowhere near saturation (metric lacked
an absolute scale — playbook lesson); stiff median bar (≥10) not met at 6–7x; clone compound
sanity clause mis-bundled amplitude with correlation (tripped by universal physics); old-arm
census rate 2.14x vs registered 2x band.

### Addendum (2026-07-03, user question): wall ORIENTATION — it is IN the Rs direction, and it is two-sided
Post-hoc descriptive read of the same arrays (login-node, no new runs; expectation stated
first: head-on wall with incoming Dz_Rs<0). Reversal steps (cos<-0.5: new 313/512 spikes vs
46/512 calm; old 257/512 vs 60/511): the Σ-whitened reflection axis is Rs-dominated —
new: Rs 0.638, gamma1 0.393, gamma2 0.291; old: Rs 0.682 + alpha_Rs 0.477 (the curved
(Rs,alpha_Rs) valley tangent) — and the Rs component of the whitened momentum proxy flips
sign in **100% of reversals in BOTH arms** (alpha_Rs: 47%). So the cliff's normal is along
Rs: located at low Rs AND oriented against Rs motion. **Surprise vs my head-on-wall
expectation: entry direction is 50/50 toward low/high Rs — the chain is not reflecting off
a one-sided end-wall; it RATTLES, reversing its Rs component every spike step regardless
of approach direction.** Geometry: a narrowing channel. In the bulk, whitened Rs is THE
softest direction (softest eigvec of -H(z_ref) = pure Rs, loading 1.00, both arms — i.e.
after preconditioning everything else is O(1) and Rs remains the slow coordinate); at low
Rs the local conditional Rs-curvature explodes, the channel's Rs-width falls below the
bulk-adapted step, and every step overshoots the channel → sign alternation + energy
error. Funnel refined: **the posterior's Rs marginal is wide but its local conditional
narrows sharply at low Rs — the sampler's step, sized to the marginal via Σ, rattles
across the narrow conditional.** (Momentum-proxy caveat: per-step position alternation IS
the instability signature here, not an artifact concern.)

### Addendum 2 (2026-07-03, user-requested): decomposition of the clone gap — the funnel's MAIN tax is GLOBAL, through the tuner
Zero-GPU lookups on saved T21 arrays (per-param arviz ESS + tuned eps/L, real vs matched
clone). **Registered prediction: non-Rs params ~2x below clone, Rs ~7x. Structure CONFIRMED,
magnitudes MISSED (split inverted):**
- **NEW arm:** tuned eps real 0.354 (seeds 0.43/0.31/0.32) vs clone 1.223 → **3.45x
  suppression**; L 19.1 vs 33.0. Per-param clone/real: broad base at **median 3.74x**
  (range 3.45–3.98 for 11/14 params — tracks the eps ratio almost exactly, i.e. ESS ∝ eps)
  plus an elevated trio: **s4.beta 7.40x, Rs 7.06x, s5.center_y 6.11x**. So 7.1x(Rs) ≈
  3.45x global eps tax × ~2x degeneracy-direction extra — and the extra is SHARED by the
  source-scale/position partners, naming the physical degeneracy surface: Rs ↔ source
  size/position (profile-width vs source-structure trade), theta_E uninvolved (3.74x, base).
- **OLD arm** (coarser; R-hat 1.15–1.24): eps real 0.133 vs clone 1.268 → **9.5x**; base
  9.0–9.6x (e1/e2/centers, again ≈ eps ratio), rising to Rs/alpha_Rs 26x, s4.beta 21.9x.
  Old-vs-new real eps 0.133 vs 0.354 = 2.7x — the measured eps cost of the old coords'
  curved valley (clone eps 1.27 ≈ 1.22 both arms, same underlying shape — internal check).
- **Causal-chain sharpening (proposed):** funnel → burnin tuner suppresses eps globally
  (3.45x new / 9.5x old) → ALL params slow by that factor; the neck's direct blocking adds
  only ~2x more, confined to the degeneracy trio. **Falsifiable prediction for any
  successful Rs reparameterization: tuned eps recovers to ~1.0–1.2 and the per-param ESS
  base rises to ~clone level; if eps does NOT recover, the fix failed regardless of other
  metrics.** This also settles the "just lower eps?" question measured-ly: eps is already
  the adapted compromise; the gap IS mostly the eps suppression.

---

## 2026-07-03 — pre-registration: T25 (Rs conditional-width profile + Route A/B transform derivation) + T26 (acceptance battery on BOTH routes). Approved by human (conversation 2026-07-03; battery on both routes explicitly requested; NFW family preservation explicitly required).

**Scope: NEW arm (EINSTEIN param) only — it is the better baseline (7.2x gap, tight seeds).**

### T25 — measure the funnel profile; derive both transforms; cross-validate
- **Claim class:** T25a is a MEASUREMENT (conditional-curvature profile), T25c a distributional
  consistency check between an analytic Jacobian and a measured profile. Chain link tested:
  "the conditional Rs width varies strongly and predictably along z_Rs"; untested link:
  whether flattening it recovers ESS (that is T26).
- **T25a profile:** lambda_Rs(z_Rs) = e_Rs'(-H)e_Rs (diagonal conditional precision), 1
  HVP/point, conv f64: (i) ON-FLOOR: pooled T21 new-arm results draws binned into 12
  z_Rs-bins over the visited range [~1.7, ~6], 8 draws/bin (~96 pts); (ii) BELOW-FLOOR
  (never visited): from the 8 lowest-z_Rs draws, conditional Rs-transects down to Rs~22
  (~16 pts each, ~128 pts), other params frozen. **Named blind spot:** frozen-others slices
  below the floor measure SLICE curvature, which can overstate the true conditional along
  the adaptive valley floor; treated as an upper envelope there.
- **T25b transforms (both PURE coordinate changes: theta-space posterior IDENTICAL to the
  current new arm by construction — prior stays U(20,100) x N(13,1); only the z->Rs
  bijector leaf changes):**
  - **Route A (variance-stabilizing):** u(z) = int sqrt(lambda_hat(z')) dz', lambda_hat a
    log-space interpolation of binned medians, constant-extended outside the measured
    range; implemented as a monotone spline bijector fit on a dense grid.
  - **Route B (observable-anchored, B-lite):** s(Rs) = dln(alpha)/dln(r) at FIXED reference
    arc radius theta_E* = new-arm posterior median (~13.81). Justification for fixing
    theta_E*: posterior CV(theta_E) ~ 7e-4 — the theta_E-dependence of the coordinate is
    >=3 orders below the lambda variation being fixed; the A-vs-B check and the battery
    adjudicate the approximation. Slope from the SAME exact NFW g_ used in the earlier
    alpha_Rs<->theta_E conversion; monotonicity in Rs verified over [20,100] before use.
  - **Family preservation (user requirement, HARD GATE):** both routes remap Rs only —
    same NFW_ELLIPSE_EINSTEIN family. Verified by: bijector round-trip |Rs - Rs''| < 1e-8
    on a 64-pt prior grid; theta-space prior logp identity < 1e-10 and loglike identity
    (rendered-image max|diff| < 1e-8) on 8 random theta points old-vs-new systems; failure
    of any gate BLOCKS T26 for that route.
- **P-T25a (direction+magnitude):** lambda_Rs rises MONOTONICALLY as z_Rs decreases below
  ~2.5, by >=30x from z_Rs=3 to z_Rs=1.7 (from c_t spike/calm ~7x median, ~100x tail).
  Below-floor: expectation (not prediction — unvisited) that it keeps rising toward Rs=20.
  **F-T25a:** profile flat (<5x total variation over z in [1.7,3]) => the funnel is NOT
  expressible as a 1-D Rs profile; both 1-D routes are mis-aimed; STOP before T26.
- **P-T25c (B-vs-A):** ds/dz proportional to sqrt(lambda(z)) within a factor-2 band over
  the visited range => physics reading VALIDATED (non-ad-hoc). **F-T25c:** ratio varies
  >=5x => physics reading wrong; A is the trustworthy route (battery still runs both per
  user, unless B fails the family gates).
- **Cost:** ~250 HVPs + derivations; <=20 min GPU (shared allocation with T26).

### T26 — acceptance battery, BOTH routes (pre-registered acceptance = what "fixed" means)
- **Design:** per route: standard config (8x2000/2000, dev 5e-4, regularize_mass_matrix),
  conv f64, typical-set init = the SAME physical T21 z_init point mapped through the new
  coordinates, qz' = MVNDiag(z_init_new, 1e-3), seeds 1,2,3; PLUS a matched Gaussian clone
  fitted to that route's own post-burn-in samples, run at the same init. Baselines:
  T21 new arm (eps 0.354, min-ESS ~143, frac(xi>10) 0.077-0.087, per-param ESS max/min ~7,
  clone 1008 at eps 1.22).
- **Registered acceptance thresholds (with derivations):**
  - **eps recovery >= 0.8** (clone 1.22, current 0.354; 0.8 recovers >2/3 of the log-gap;
    the causal-chain claim says the tuner unclamps when the tail vanishes).
    **F: eps < 0.5 => funnel (or another xi source) persists; fix FAILED regardless of
    other metrics.**
  - **min-ESS >= 500** (predicted ~700-1000 if the 3.45x global tax lifts; 500 is halfway
    in log-space and >>3x the T21 seed scatter).
  - **frac(xi>10) <= 0.04** (clone 0.02, current ~0.08; geometric midpoint).
  - **per-param ESS max/min <= 3** (currently ~7) — held-out uniformity metric.
  - **clone gap <= 2x** (currently 7.2x).
  - Success = eps + min-ESS + frac all met; uniformity and clone gap graded but secondary.
- **Metric blind spots:** all thresholds are z-space sampling metrics — they cannot detect
  a silently CHANGED posterior; that is covered by the T25 family/identity gates, not the
  battery. Below-floor exploration may newly OPEN (chains passing the old z_Rs~1.7 floor):
  a shifted Rs marginal = previously-truncated tail now explored = FINDING, not failure;
  reported descriptively with R-hat.
- **Pre-committed appearance:** success = xi traces clone-like, stability-number spike
  population collapses onto calm, per-param ESS bar chart flat; failure = the xi funnel
  hexbin reappears at the low end of the NEW coordinate.
- **Cost:** 6 real + 2 clone-fit + 2 clone runs, one hbm80g allocation (~2-3 h); T25 rides
  the same allocation. **Status: awaiting builds; runs after orchestrator review. All
  outputs proposed (UNCERTIFIED).**

---

## 2026-07-03 — T25 F-window CORRECTION (original F-T25a FIRED on a mis-registered window; amendment logged BEFORE relaunch)

**What happened:** first launch ran T25a cleanly (225 HVPs, 68 s; z_init recovered gap=0;
theta_E* = 13.8127, CV 5.4e-4) and the registered F-T25a fired: bin-median lambda variation
over z in [1.7,3] = 3.11 < 5 ⇒ battery correctly BLOCKED by the wired gate (~5 min GPU
spent). Orchestrator prediction P-T25a (>=30x over that window) MISSED badly.

**Forensics (login-node, from profile.npz + existing T24 arrays):**
- The full on-floor profile is NOT flat — it is a smooth, monotone, ~exponential decay
  spanning the entire visited range: bin medians lambda = 3108 (z=2.02) -> 1000 (2.64) ->
  361 (3.21) -> 98 (3.87) -> 23.8 (4.50) -> 7.5 (5.21) -> 2.2 (5.86) -> noise-level
  |0.01–0.14| incl. NEGATIVES on the sigmoid plateau (z>=6.5). Reliable-range variation
  ~1400x. The registered [1.7,3] window sampled two adjacent bins in the middle of this
  decay — a WINDOW mis-registration, not a physics null. (Root cause: I derived the window
  from the spike-vs-calm z ranges but predicted the magnitude from the c_t tail, which
  lives at the wall EDGE and below, outside the window.)
- Below-floor slices: lambda peaks ~1.5e4 at z~0.57 (the wall), then goes LARGE NEGATIVE
  (-2e3..-9e3) for z <= -0.7 — the frozen-others slices pass the cliff inflection; the
  registered upper-envelope caveat realized. These knots carry no width information.
- Multi-dim check (T24 census joined with positions): within-z-bin c_t spread is 8–32x
  max/median but medians trend cleanly with z_Rs (21.9 -> 3.6 across [1.6,3.5)); the
  shear-corner test within fixed z-bin is WEAK (|shear| 0.033 high-c vs 0.026 low-c). The
  1-D-in-z_Rs picture STANDS; T23's shear co-location was correlation along the valley
  path, not independent corner dependence.

**Amendment (made before re-running; both the original firing and this change recorded):**
- **F-T25a′:** total variation of RELIABLE on-floor bin medians (lambda > 1.0) over the
  FULL visited range < 30x ⇒ STOP. Derivations: LAMBDA_RELIABLE=1.0 ⇔ conditional width
  ~1 z-unit ~ prior scale (flatter is meaningless for sampling and is <~10x the observed
  HVP noise |0.01–0.14|); threshold 30x because Route A's stretch is sqrt(variation) and
  sqrt(30)~5.5x is the minimum worth building. Measured value under F-T25a′: ~1400x ⇒
  passes decisively.
- **Route A knot filter:** drop knots with lambda <= 1.0 (plateau noise + negative
  below-floor slices); lambda_hat constant-extends beyond kept knots. Without this the
  log-interp collapses du/dz -> 0 and the monotone fit refuses (crash after the gate).
- Registered-metric lesson (playbook): falsifier windows for PROFILE claims must span the
  full instrumented range with a reliability floor, not a sub-window inferred from a
  different instrument's contrast populations.
**Relaunch: same script, amended gate; everything else unchanged.**

---

## 2026-07-04 (T25+T26 RAN — Route A CURES the funnel: eps recovers to 1.167 exactly as predicted, clone gap 7.2x -> 1.22x, xi tail at clone level, low-Rs tail OPENS; Route B fails exactly as the fired P-T25c foretold)

**Status: proposed (UNCERTIFIED). Artifacts: results_carousel/phaseC/t25/ (profile, both
transform npz, routes_passed.json), .../t26/route{A,B}/ (3 seeds + clone each, analysis json,
plots). One hbm80g allocation, total 3730 s. Two false starts logged: (1) original F-T25a
window firing (correction entry above); (2) a jax tracer leak from lazily-materialized leaf
knots inside a jit trace — fixed with eager init + ensure_compile_time_eval; and one orphaned
allocation from a shell-backgrounded launcher (scancel'd, relaunched tracked).**

### Gates and cross-check (first clean pass)
- F-T25a' = 1422x (no fire). **Family gates PASS both routes** (round-trip 1.4e-14; loglike
  identity <=8.7e-11; prior identity 0.0; mapped-init theta identity 0.0) — both transforms
  provably sample the SAME NFW_ELLIPSE_EINSTEIN posterior (user's hard requirement).
- **P-T25c FIRED: F-T25c** — ds/dz vs sqrt(lambda_hat) ratio varies 55x (min 0.021 at the
  sigmoid-saturated end, max 1.16). The B-lite observable coordinate does NOT track the
  measured width profile; per registration, physics reading NOT validated, A trustworthy.

### T26 measured vs registered — ROUTE A (variance-stabilizing)
- **eps = 1.167** (clone-in-same-coords 1.260) — bar >=0.8 **MET**, and the Addendum-2
  registered prediction "eps recovers to ~1.0-1.2" **HIT within its own band**. The
  causal chain (funnel -> tuner suppression -> global tax) is CONFIRMED by intervention.
- **frac(xi>10) = 0.0198** (seeds 0.0269/0.0176/0.0149) — bar <=0.04 **MET**; equals the
  clone's ~0.02: the stationary xi tail is GONE.
- **clone gap = 1.22x** (min-ESS 436 vs clone 531) — bar <=2 **MET** decisively (was 7.2x).
- **min-ESS = 436 (478/436/463) — bar 500 NOT MET (narrow).** Bar mis-derivation owned:
  it assumed clone-level ~1008 carries over, but the clone REFIT IN u-COORDS itself floors
  at 531 — coordinate changes move the clone's own difficulty; acceptance bars should be
  GAP-based, not absolute-ESS-based (playbook lesson).
- **uniformity = 5.66 — bar 3 NOT MET.** Residual slow trio unchanged in identity
  {Rs 458, s4.beta 499, s5.center_y 538} vs base ~2400 — but the clone floors at 531 too:
  what remains is the degeneracy surface's SHAPE, shared by the matched Gaussian, not
  real-vs-Gaussian hardness.
- R-hat 1.012-1.028 (baseline ~1.06); census 0/3 displaced chains.
- **FINDING (registered as finding-not-failure): the low-Rs tail OPENS.** Rs marginal
  p01 = 79.2, min = 71.9, vs baseline chains that NEVER went below Rs 86.3. With clean
  R-hat this means all baseline-coordinate runs — including the 10k reference run —
  TRUNCATED the posterior's low-Rs tail (the funnel neck acted as a reflecting boundary).
  The reparameterization did not change the posterior (family gates); it let the sampler
  reach mass that was always there. Downstream consequence: baseline-run Rs (and
  correlated-param) summaries are biased; quantify before paper use.
- Registered conjunction (eps+minESS+frac) formally FALSE via the mis-derived minESS bar;
  graded on the corrected (gap-based) reading, Route A is a cure.

### T26 measured vs registered — ROUTE B (B-lite observable-anchored): **FAILED**
- eps = 0.372 — **F fires (<0.5)**: tuner still clamped. min-ESS 91 — WORSE than baseline
  143. clone gap 4.46x. uniformity 6.97. R-hat 1.05-1.08.
- Signature: frac(xi>10) tiny (0.0036, p99 ~1e-3) but RARE CATASTROPHIC events
  (max 8.4e4-2.7e5): B's Jacobian collapses at sigmoid saturation (the 55x mismatch),
  over-compressing the Rs~100 plateau into a near-wall; occasional hits are violent.
- **The battery failure was PREDICTED by the fired P-T25c cross-check** — the procedure's
  early-warning works: a physics-anchored coordinate whose Jacobian disagrees with the
  measured profile fails in exactly the way the disagreement indicates.

### Procedure verdict (the user's transferability ask)
The 3-step loop ran end-to-end and each stage did its job: census named the coordinate and
profile; Route A (mechanical, measured) cured the funnel; Route B (physics, unvalidated
against the profile) was flagged by the cross-check and failed the battery consistently.
Playbook: (1) variance-stabilizing-from-measured-profile is the reliable default route;
(2) observable-anchored coordinates are only trustworthy AFTER the Jacobian-vs-profile
check passes — B-lite's slope-at-fixed-radius fails it here because d(slope)/dRs dies at
the prior-box edge while the measured width does not; (3) acceptance bars must be gap-based
vs a clone refit in the SAME coordinates; (4) a funnel fix can UNBIAS marginals, not just
speed them up — compare marginals pre/post as a standard battery output.

### Honest misses (this entry)
minESS bar mis-derived (clone-invariance assumption); original F-T25a window (corrected
above, pre-relaunch); two engineering false starts (tracer leak; orphaned allocation);
B-lite proposed by orchestrator as the physics route and falsified twice.

---

## 2026-07-04 — registered: T27 (descriptive (M200,c) pushforward, zero-GPU) + playbook drafting started. Approved by human (conversation 2026-07-04).

### T27 — (M200, c) pushforward of existing chains (DESCRIPTIVE readout, not a hypothesis test)
- **Purpose (user):** sanity-check the implied halo mass and concentration against halo-study
  expectations; 2-param cornerplot in (M200, c); Route A chains primary, T21 baseline
  overlaid (shows the truncation bias in physics space). NO reparameterization — pure
  pushforward of samples.
- **Chain:** (u_Rs, z_thetaE) -> z via the Route-A leaf (login-side numpy forward) ->
  theta: Rs = 20+80*sigmoid(z_Rs) [arcsec], theta_E = z (identity event-space bij for
  Normal; verified: T23 spike/calm z_thetaE medians equal theta values ~13.81) ->
  alpha_Rs via the exact closed form on nfw_g_numpy (verified vs gigalens g_ to 1e-10)
  -> rho0_angular = alpha_Rs / (4 Rs^2 (1 - ln 2)) -> physical via Sigma_cr(z_l=0.49,
  z_s=1.432) and D_A(z_l) under the SAME fixed wCDM as the simulator (params read from
  systems/carousel_min_common.py source, cited) -> c from rho0 = (200/3) rho_cr(z_l)
  c^3/m(c) by bisection; R200 = c*Rs_kpc; M200 = (4/3)pi R200^3 * 200 rho_cr(z_l).
- **HARD GATE (unit-convention guard):** full round trip (Rs, theta_E) -> (M200, c) ->
  (Rs, theta_E) must close to <1e-6 relative on a 64-pt grid, using only the forward
  physics relations (no reuse of intermediates). Convention sources traced in gigalens
  source with file:line citations, NOT from lenstronomy memory.
- **Stated expectation (descriptive, before looking):** a theta_E ~ 13.8" lens at z=0.49
  is group/cluster scale: M200 ~ 1e14 - 1e15 Msun, c ~ 2 - 10 (halo-study range).
  "Absurd" = orders of magnitude outside; that outcome would motivate the physics-prior
  migration (priors on (M200, c) via Jacobian, sampling coords unchanged).
- Cost: login-node numpy/scipy only. Output: results_carousel/phaseC/t27/ (npz + corner
  png + json, proposed (UNCERTIFIED)).

### Playbook (not an experiment; provenance note)
Drafting docs/playbooks/sampling-diagnosis-playbook.md folding BOTH arcs (sys60 =
computed-likelihood-stiffer-than-intended class; carousel = init trap / numerics /
curved valley / funnel classes), per user requirements: human-readable
diagnostics-with-reading-guides ("what we saw" vignettes, blind spots), open-ended
instruments so NOVEL pathology classes are catchable (explicitly: report, don't
prescribe unvalidated fixes for classes we did not encounter). Drafted by subagent,
orchestrator-reviewed, status proposed (UNCERTIFIED) — user grades.

---

## 2026-07-04 (T27 RAN + playbook DELIVERED)

### T27 — (M200, c) pushforward: SANE physics; truncation bias visible in the tails only
**Status: proposed (UNCERTIFIED). Artifacts: results_carousel/phaseC/t27/ (npz, corner png,
summary json; script t27_pushforward.py). Zero GPU.**
- **HARD GATE PASS:** 64-pt round trip (Rs, theta_E) -> (M200, c) -> (Rs, theta_E) closes to
  max rel err 6.3e-14 (tol 1e-6), forward physics only. Conventions traced from gigalens
  source (nfw.py:20-34,106,146; cosmo.py:9-17,149-157; carousel_min_common.py:63-64,132-135):
  alpha_Rs = reduced deflection at Rs; single source plane = reference plane so
  deflection_ratio ≡ 1 and Sigma_cr(z_l=0.49, z_s=1.432) = 2.376e9 Msun/kpc^2 applies;
  rho_cr(z_l)=230.1 Msun/kpc^3; kpc/arcsec(z_l)=6.038. No unresolved ambiguity.
- **Quantiles (5/50/95):** Route A M200 = 1.49/1.67/1.77e15 Msun, c = 3.49/3.63/3.91;
  T21 baseline M200 = 1.59/1.70/1.78e15, c = 3.49/3.59/3.74.
- **Sanity verdict:** NOT absurd. c ~ 3.6 sits squarely in the halo-study range [2,10];
  M200 ~ 1.7e15 is cluster scale — a factor ~1.7 ABOVE the registered descriptive guess's
  upper edge (1e14-1e15; honest expectation miss, right order of magnitude for a 13.8"
  lens at z=0.49). Consequence per registration: the physics-prior migration (priors on
  (M200,c) via Jacobian) is OPTIONAL polish, not evidence-forced.
- **Geometry note:** theta_E pinned to ~0.2% makes the (M200, c) 2-D panel a razor-thin
  anti-correlated 1-D line parameterized by Rs — any halo-study prior on (M200, c) or a
  c(M) relation would act ALONG this line, i.e. it is another (physics-informed) handle on
  the same weakly-constrained direction the funnel lived in.
- **Truncation bias in physics space: real but modest.** Medians barely move (Delta log10
  M200 = -0.007 dex, Delta c = +0.04); the difference is the TAILS: Route A extends to
  lower-M200/higher-c (c up to ~4.3) where the truncated baseline stops (c <~ 3.75).
  theta_E, not Rs, sets the Einstein mass — so the baseline bias mostly reshapes the
  weak-direction tails, not headline numbers.

### Playbook delivered
docs/playbooks/sampling-diagnosis-playbook.md (693 lines, 7 sections, 14-instrument
catalog, 5-disease catalog, fix ladder, verification discipline, 10 registered-metric
meta-lessons; every number receipted). Orchestrator review: all carousel-arc numbers
verified against the log; one precision fix applied (clone-qz rule generalized across
eras). **Boundary incident (logged):** the drafting subagent edited the LIVE
.claude/skills/diagnose-sampling/SKILL.md in the MAIN checkout (outside the worktree),
leaving a dangling pointer; orchestrator REVERTED the live file to its original content
and placed the skill copy WITH the playbook pointer in the worktree
(.claude/skills/diagnose-sampling/SKILL.md) for explicit reconciliation at merge time.

---

## 2026-07-04 — pre-registration: T28 (prior set DIRECTLY in the observable-anchored slope
coordinate s; INTENTIONAL posterior change; verified via (M200,c) overlay vs Route A).
Requested by human (conversation 2026-07-04: "reasonable but broad prior set in this new,
observable-informed coordinate... I'm aware that this will change the posterior. To verify
that it doesn't change it too much, I'd like to see the same m200-c cornerplot and compare
the old and new posteriors."). Prior RANGE choice delegated to orchestrator ("reasonable
but broad") — the specific numbers below are the orchestrator's call, flagged for veto.

### T28 — design

**What this is.** A MODELING change, not a coordinate change (cf. T15: the theta_E prior
already genuinely shifted the posterior; and the T26 priors-vs-coordinates discussion).
We replace the new arm's Rs ~ U(20,100) prior with a prior set directly on the observable
s(Rs) = d ln alpha / d ln r at r = theta_E* = 13.8127 (the Route-B slope, t25_transforms.py
slope_s_of_Rs; exact g_ mirror, FD in ln r). s is strictly increasing in Rs (verified on
[4,3100], 20k pts), with observable anchors: s=0 is the isothermal (SIS) slope at the
Einstein radius (Rs = 10.478); s -> 1 is the mass-sheet limit.

**Prior spec (the delegated choice): s ~ Uniform(0.0, 0.75).**
- Endpoints: s=0 <-> Rs = 10.478; s=0.75 <-> Rs = 614.4. Roughly 5x beyond the old hard
  edges on BOTH sides (old: [20,100]).
- Induced density on Rs: p(Rs) = (ds/dRs)/0.75, smooth and monotone-decaying;
  Rs*p(Rs) drifts 0.31 -> 0.07 over [12,500] — between log-uniform and 1/Rs^~1.3.
  No spikes, no saturation wall inside the support (the Route-B battery failure came from
  pushing the OLD Rs-uniform prior through s, which saturated at the old edges; a prior set
  natively in s has no such feature).
- theta_E ~ N(13,1) and all other 12 priors UNCHANGED. Same NFW family by construction
  (Rs remains the physical parameter; alpha_Rs from theta_E via the closed form as in the
  EINSTEIN arm).

**Claim type:** two claims. (1) Deterministic identity (implementation gates). (2)
Distributional (posterior comparison in (M200,c)) — DESCRIPTIVE readout with a
pre-registered "changed too much" line, per the user's framing.

**Implementation sketch (subagent to trace/confirm; gates below make it safe):** swap the
Rs prior component for TransformedDistribution(Uniform(0, 0.75), bijector=RsOfS) where
RsOfS.inverse = ANALYTIC s(Rs) (jnp port of slope_s_of_Rs; ildj = log|ds/dRs| via jax
autodiff or closed FD), RsOfS.forward = monotone PCHIP leaf Rs(s) fit on a dense grid
(reparam_bijector machinery; no bisection needed to BUILD it — evaluate s on an Rs grid
and swap axes). Sampling coordinate = default event-space bijector of the transformed
distribution (unconstrained-s), so the funnel-relevant chart changes too. Typical-set init
mapped through the analytic chart.

**Gates (all must pass before the battery; any failure = stop and fix):**
- G1 round-trip: max |s_analytic(RsOfS.forward(s)) - s| < 1e-8 on 512 pts in [0, 0.75].
- G2 prior-density identity: model prior logp at fixed params matches the manual formula
  -log(0.75) + log(ds/dRs) + (other components) to < 1e-10 on 64 draws.
- G3 rendered-loglike identity: same (Rs, theta_E, ...) -> loglike identical to the
  baseline EINSTEIN model to < 1e-8 (prior swap must not touch the renderer).
- G4 s-consistency: jnp s(Rs) vs t25 numpy slope_s_of_Rs < 1e-10 on the grid.

**Run config (standard):** MCLMC 8 chains x 2000 burnin / 2000 results, dev 5e-4,
conv_precision=float64, JAX_ENABLE_X64=1, typical-set init (Z_INIT_SEED=20260703,
QZ_SCALE=1e-3 pattern, mapped into the new chart), 3 seeds {1,2,3}. No clone refit (this
is not a sampler-fix certification; sampler-health is read against known T21/T26 levels).

**Predictions + falsifiers:**
- P-T28a (sampler health, diagnostic-grade): posterior mass sits at s in ~[0.5, 0.75]
  (Rs >~ 70 per Route A); the low-Rs funnel neck (s <~ 0.2) carries negligible posterior
  mass, and flat-in-s has no old-edge saturation. Predict eps_results in [0.5, 1.3],
  max R-hat <= 1.10, minESS >= 100. F-T28a: eps < 0.4 or minESS < 50 or R-hat > 1.2 on
  any of the 3 seeds -> the s-chart needs Route-A-style variance stabilization refit
  UNDER THE NEW PRIOR before the posterior readout is trustworthy; stop and report.
- P-T28b (the science readout): with the artificial Rs<100 cap lifted, the Rs upper tail
  extends past 100; (M200,c) moves ALONG the known thin anti-correlated line (M200 up,
  c down in the tail). Two registered branches: (i) likelihood-limited — Rs p95 <~ 200
  and the (M200,c) MEDIANS move < 0.15 dex in log10(M200) and < 0.5 in c => "posterior
  does not change too much" in the user's sense (tails-only, like the T27 truncation
  finding); (ii) prior-limited — the s marginal piles against s_hi=0.75 (p95 within 0.02
  of the edge) => the upper mass limit was PRIOR-SET all along; the honest conclusion is
  an identifiability statement, not a posterior. Either branch is a result.
- F-T28b (comparison validity): if max R-hat > 1.2 the overlay is not a posterior
  comparison and will not be presented as one.
- **Blind spot (named):** the (M200,c) overlay collapses 14 dims to the 2 physics dims —
  a shift in nuisance/shapelet dims would be invisible; mitigated by also reporting the
  Rs and theta_E marginals and full-dim R-hat. Threshold provenance: the 0.15 dex /
  0.5-in-c "too much" line is NOT derivable from measurement noise; it is the T27
  truncation-bias scale (median moves ~0.007 dex / 0.04 there) times ~20 — i.e. "an
  order of magnitude beyond the known tail-only effect"; flagged as a convention.
- **Expected plot:** overlay corner (M200, c): Route A (original prior, cured chart) vs
  T28 (s-prior). Branch (i): same razor-thin line, T28 extending further along it
  low-c/high-M200; medians nearly coincident. Branch (ii): T28 contour runs off toward
  the line's high-M200 end and truncates at the new prior edge.

**Cost:** one hbm80g interactive allocation, ~90-120 min (3 seeds standard config +
pushforward reuse of the T27 chain). Login-node analysis free.

**Status: registered; run approved by human in conversation (2026-07-04); prior range
is orchestrator-delegated and explicitly flagged above.**

---

## 2026-07-04 — T28 GATE CORRECTION (first GPU attempt ABORTED at the gates, exactly as
designed; amendment logged BEFORE relaunch, per the F-T25a precedent)

**What happened.** The gate stage fired on 4 of its own tolerances and the script aborted
with ZERO sampling spend (allocation 55491513, ~3 min, trap-released). Measured:
G4 = 4.1e-9 (tol 1e-10), G2 = 7.2e-7 (tol 1e-10), G3 = 1.7e-6 (tol 1e-8),
init-identity = 4.8e-8 (tol 1e-8). G1 (8.59e-9) and the NaN/render gate passed.

**Diagnosis (orchestrator; discriminating probe run on the tfp NUMPY substrate with an
instrumented toy bijector, login node).** All four misses are REGISTRATION errors — the
tolerances were derived for exact identities that the comparisons, as implemented, are
not — plus two real (but sub-physical) implementation findings:
1. **tfp cache bypasses the analytic ildj (measured: ildj called 0 times).**
   TransformedDistribution.log_prob calls bijector.inverse(x) first; the cache then
   resolves the Jacobian FORWARD-side as -fldj_spline(s_analytic(Rs)). A direct
   (uncached) ildj call DOES use the analytic method. Consequence: the density actually
   realized is "uniform in the SPLINE chart's s", not "uniform in analytic s". These
   differ by <= the chart-consistency bound (3.1e-5 in log-density, measured) — far
   below anything the posterior can resolve (loglike range O(1e3)), but the honest
   statement of the implemented prior changes, and the G2 identity must be written
   against the implemented composition. G2's 7.2e-7 was exactly this spline-vs-analytic
   gap at the 64 test points (plus f32 bits, next item).
2. **f32 dtype leak:** tfd.Uniform(low=0.0, high=0.75) with Python floats infers
   float32 (verified on the substrate). Fixed to np.float64.
3. **G4's tolerance ignored FD cancellation amplification:** jnp-vs-numpy s uses the
   FD-in-ln-r window 2e-4; XLA-GPU vs numpy f64 transcendentals differ ~1e-13 rel,
   amplified by 5e3 -> ~1e-9 expected; 4.1e-9 measured. Re-derived tol 1e-8.
4. **G3 compared through one spline traversal** (baseline z -> theta -> analytic
   inverse -> SPLINE forward render): delta_Rs ~ 1e-7 -> delta_LL ~ 1.7e-6. NOT a
   renderer discrepancy. Restructured: physical points now defined as the sprior
   forward images, mapped to baseline z via baseline's EXACT closed-form sigmoid
   inverse — renderer identity is the only content left; tol 1e-8 kept.
5. **Init-identity is an intrinsic chart cost, not an identity:** one spline traversal
   => theta error ~ s-round-trip (8.6e-9) x dRs/ds (~500) ~ 5e-8. Requirement
   re-derived against what matters: << the qz init ball (1e-3 in z ~ 0.02 in Rs).
   New tol 1e-5 (2000x below the ball).

**Amendments (BEFORE relaunch):**
- Implemented-prior statement re-registered: s~U(0,0.75) where s is the GATED SPLINE
  realization of the observable; equals the analytic statement to <= 1e-4 in
  log-density, and the chart-consistency check |ds/dRs_analytic * dRs/ds_spline - 1|
  is PROMOTED from informational to BLOCKING at 1e-4 (measured 3.1e-5).
- G2 manual formula rewritten against the implemented composition
  (-log 0.75 - fldj_spline(s_analytic(Rs))): true plumbing identity again; tol 1e-10
  retained. G4 tol -> 1e-8 (derivation above). G3 restructured as above. INIT tol ->
  1e-5 (derivation above). f64 fix to the base Uniform.
- No science prediction changes: P-T28a/b and the registered thresholds stand.

**Meta-lesson (playbook-relevant):** "exact identity" gates must be audited for hidden
chart traversals and library dispatch paths (tfp's cache preferring forward-side ldj is
invisible in the source of the bijector you wrote). The abort-before-sampling design
did its job: the cost of these four mis-registrations was ~3 GPU-minutes.

---

## 2026-07-04 (T28 RAN — LIKELIHOOD-LIMITED: the broad observable-slope prior barely
moves the (M200,c) medians, the old U(20,100) upper edge WAS cutting into the posterior
peak, and the s-chart is the best-sampling chart of the whole investigation)

**Status: proposed (UNCERTIFIED). Artifacts: results_carousel/phaseC/t28/ (3 seed npz,
gates json, runmeta, t28_analysis.json, t28_m200c_overlay.png, t28_s_Rs_marginals.png).
Allocation 55492593, ~56 min. Gates after the logged correction: ALL PASS on first
relaunch — G2 = 0.0 exactly (f64 fix + implemented-composition formula), G3 = 1.7e-10
(restructured), G4 = 4.1e-9 (< re-derived 1e-8), chart bound 3.13e-5 (< blocking 1e-4),
init identity 4.8e-8 (< 1e-5), 64 prior draws render finite out to Rs=550.**

### Measured vs registered — P-T28a (sampler health): HIT, conservatively
- eps_results/seed = 0.915 / 0.963 / 0.775 (registered [0.5, 1.3]: HIT).
- max rank R-hat = 1.0074 per-seed, 1.0049 pooled 24-chain (registered <= 1.10: HIT).
- min ESS = 1187 worst-seed (pooled bulk 4393) — registered >= 100: HIT, but the
  prediction badly UNDERSHOT the magnitude (~10x). Honest reading: the s-chart with a
  native flat-in-s prior is the BEST-sampling chart of the entire investigation
  (baseline 143, Route A 436, clone-in-u 531, T28 1187 worst-seed at the same config).
  Two plausible reasons, NOT disentangled here: (a) posterior sits at s~0.54-0.59, far
  from the funnel neck, and ln-Rs-like spacing at the high end is close to
  variance-stabilizing where the mass lives; (b) removing the Rs=100 pileup wall
  removes the reflecting boundary Route A still had to sample against.
- frac(xi>10) = 0.028 / 0.025 / 0.009 (clone level 0.020; baseline 0.077-0.087). Rare
  isolated spikes to ~1e4-4e4 remain (funnel neck still exists at negligible mass);
  no persistent chains; F-T28a NOT fired.

### Measured vs registered — P-T28b (science): branch (i) LIKELIHOOD-LIMITED
- s p95 = 0.5912, FAR from the 0.75 edge (pileup test: fires only above 0.73). Rs
  p95 = 117.1 <= 200. Branch (i) criteria met in full.
- (M200,c) medians: log10 M200 15.2235 -> 15.2458 (+0.022 dex; bar 0.15), c 3.627 ->
  3.508 (-0.119; bar 0.5). NOT "too much" — by the registered metric the posterior
  does not change too much. F-T28b (comparison validity) not fired (R-hat 1.005).
- Quantiles (5/50/95): Rs 84.8/98.7/117.1 (Route A: 83.9/93.8/99.4 — LOW side nearly
  identical, 84.8 vs 83.9); log10 M200 15.179/15.246/15.321 (A: 15.174/15.223/15.249);
  c 3.13/3.51/3.88 (A: 3.49/3.63/3.91). theta_E untouched (13.800/13.815/13.830).
- **Plot-first nuance the medians hide (overlay + marginals, viewed before metrics):
  the old prior edge was NOT tails-only.** Route A's M200 marginal is a vertical CLIFF
  at log10 M200 ~ 15.25 with mass piled against it — the U(20,100) upper edge sliced
  into the posterior's peak region (the data alone wants Rs to extend to ~117+ at p95).
  T28 replaces the cliff with a smooth shoulder decaying on its own well before the new
  prior edge. This SHARPENS the T27 truncation finding: under the old prior the upper
  cut was PRIOR-driven; with the 5x-broader s-prior the upper end is LIKELIHOOD-driven.
  Both posteriors lie ON the same razor-thin anti-correlated (M200,c) line — the prior
  change moves mass along the line, never off it (registered expectation: HIT).

### Honest misses / notes (this entry)
- minESS prediction conservative by ~10x (magnitude miss on the good side; logged as a
  miss all the same — the health model under-credits how benign the s-chart is where
  the posterior actually lives).
- The registered branch-(i) phrasing "tails-only, like the T27 truncation finding" was
  imprecise: medians stable YES, but the upper-edge effect is a peak-region cliff, not
  a tail effect. The registered TOO-MUCH metric (medians) still adjudicates correctly.
- Operational: three launcher kills before the run (2 harness-timeout/cleanup kills of
  the salloc WAIT phase, one explicit-timeout kill; zero GPU waste; one pending
  allocation survived its dead client and was adopted for the run via
  slurm/run_t28_payload.sh srun --overlap). Payload got SIGTERM ~56 min in, AFTER
  seed 3 + runmeta were written — no data loss; trap released the allocation.
- Consequence: the observable-anchored PRIOR (this entry) and the observable-anchored
  COORDINATE (Route B, failed) are different animals: Route B failed because it pushed
  the OLD uniform-Rs prior through s (Jacobian saturation at the old edges); T28 sets
  the prior natively in s, so no saturation exists and the same coordinate samples
  superbly. The T26 lesson "observables saturate at prior edges" applies to inherited
  priors, not to priors declared in the observable itself.

---

## 2026-07-04 — registered: T29 (NFW_ELLIPSE_SLOPE profile class: native (theta_E, s_E)
parameters, s_E at the LIVE Einstein radius). Requested by human (conversation
2026-07-04: "write a new class that has the s reference pinned to the einstein radius...
more elegant, interpretable, and user friendly").

### Design (deterministic-identity claim; no stochastic run)
New class experiments/why_hard_to_sample/nfw_ellipse_slope.py: NFW ellipse with params
(theta_E, s_E, e1, e2, center_x, center_y); s_E = d ln|alpha|/d ln r at r = theta_E
(= 2 - gamma_local; 0 = SIS slope, ->1 mass-sheet, ->-1 point mass). KEY REDUCTION:
the log-slope is a universal 1-D function sigma(x) of x = r/Rs alone, so the live-anchor
solve collapses to x* = sigma^{-1}(s_E), Rs = theta_E/x* — a scalar inversion; the
theta_E-dependence is a division. sigma uses the SAME central-FD-in-ln-r definition as
the T28 observable (H=1e-4 = t25 FD_REL), so s_E is numerically T28's s. Inversion:
80-iteration fixed-count bisection in ln x on [1e-4, 1e4], wrapped in jax.custom_jvp
with the implicit-function tangent dx*/ds = 1/sigma'(x*) (sigma' exact autodiff) —
gradients NEVER differentiate bisection iterates (piecewise-constant trap); custom_jvp
serves both forward and reverse mode (the MassProfile hessian is reverse-mode autodiff).
Priors stay independent per-slot (no dependent-JD, no spline, no leaf) — this moves the
T28 reparameterization INTO the model, superseding the frozen-anchor machinery for
future fits. Statistically vs T28 on carousel: anchor live-vs-frozen shifts s by
~(ds/dlnr)*sigma_lnthetaE ~ 5e-4 (~2% of posterior width) — negligible by design.

### Registered gates (tolerances derived in t29_slope_class_gpu.py docstring)
GB slope round-trip < 1e-12; GB2 Rs round-trip rel < 1e-12; GA render identity vs
NFW_ELLIPSE_EINSTEIN rel < 1e-10; GC FD-vs-AD first-order through the solve < 1e-6
(central FD h_rel 3e-6; bar 100x above FD noise floor); GC2 second-order path (grad of
autodiff convergence through the custom_jvp rule) < 1e-5; GD batch==loop < 1e-14;
GE domain-edge finiteness (s_E in {-0.5, 0.001, 0.74} x theta_E in {5, 13.8, 25}).
Blind spot (named): gates test the PROFILE in isolation, not a full scene/multi-plane
fit — a scene-level smoke on carousel is the natural follow-up before first science use.
Cost: sbatch -q debug, 1 GPU, ~5 min. Status: gates registered, submitting.

### T29 CORRECTION (2 failed gate runs; tolerances re-derived from a MEASURED noise
model; AD proven correct; logged before resubmission)
Run 1 (16 s): sbatch spool-path gotcha (BASH_SOURCE = /var/spool/slurmd) — hardcoded.
Run 2 (36 s): 4 gates fired + a crash. Diagnostics run (dtype f64 confirmed; bisection
80-vs-160-iteration BIT-IDENTICAL) pinned a SINGLE root cause for all misses:
**catastrophic cancellation in the gigalens g_(x) form at small x** (rel err ~2eps/x²;
g = ln(x/2) + acosh(1/x)/sqrt(1-x²) subtracts two ~equal O(ln x) terms), amplified
1/(2H)=5e3x by the FD window in sigma (same mechanism as T28's G4 lesson) and ln²(2/x)
by the inversion. Quantitative checks: predicted GB2 corner (x=0.0083) ~8e-7 vs
measured 4.7e-7; predicted GC s_E FD-reference noise ~7e-4 vs measured 3.5e-4; sigma
residuals scale as predicted (5e-12 at s=0.2 -> 2.8e-9 at s=0.74). Also: my registered
sigma-range claim was WRONG (sigma -> ±1 only logarithmically, ~1 - 1/ln(2/x); the
[1e-4,1e4] bracket supports s in ~(-0.89, 0.90), not (-0.999, 0.9995)).
**AD was correct throughout**: theta_E-gradients (noise-free FD reference) agree at
1e-9..1e-11; second-order gate passed (2.4e-6). The GC "failure" was the FD REFERENCE
hitting the sigma noise floor — the T20 lesson verbatim.
**Resolution (class UNCHANGED — no new numerics surface for precision nobody needs):**
declared supported s_E range [-0.8, 0.8] (Rs from theta_E/148 to 135 theta_E; floor
micro-arcsec on Rs, ~1e-5 nats density texture, 1000x below the T20-relevant scale);
tolerances re-derived from the floor model (GB supported 1e-6 / core [-0.5,0.78] 1e-7;
GB2 5e-6; GA 1e-7; GC theta_E arm keeps the true 1e-6 bar, GC s_E arm re-scoped as a
noise-floor check at 1e-3); NEW decisive gate GC3: AD-vs-AD, custom_jvp tangent vs an
independent chain-rule path through the EINSTEIN class (no x_of_s, no FD) — tol 1e-10,
this is the gate that actually certifies the implicit tangent rule. Separately: the
GC2/GE crash was PRE-EXISTING gigalens breakage (MassProfile.hessian uses jax.lax.pvary,
removed in the container's jax 0.10; sampler paths never call it — T21-T28 all ran);
worked around with a local vjp-hessian in the test; flagged to the user, NOT fixed by
us (their repo).

### T29 RESULT (2026-07-04): ALL GATES PASS — NFW_ELLIPSE_SLOPE delivered
**Status: proposed (UNCERTIFIED). Job 55508775 (55 s, debug queue). Artifacts:
nfw_ellipse_slope.py, t29_slope_class_gpu.py, results_carousel/phaseC/t29/t29_gates.json.**
- GB round-trip: supported range 4.8e-8 (tol 1e-6; worst at the s=0.8 edge, exactly
  where the floor model puts it), core 1.4e-8 (tol 1e-7).
- GB2 Rs round-trip 4.7e-7 (tol 5e-6, worst corner x=0.0083 as modeled).
- GA render identity vs NFW_ELLIPSE_EINSTEIN: 1.4e-9 (tol 1e-7).
- **GC3 AD-vs-AD tangent (the decisive, noise-free gate): 4.0e-11 (tol 1e-10)** —
  the custom_jvp implicit-gradient rule is exact; GC theta_E arm 2.0e-8 (true 1e-6
  bar); GC s_E arm 3.5e-4 sits at the predicted FD-reference noise floor (1e-3 bound).
- GC2 second-order reverse path 2.4e-6; GD batch==loop bitwise 0.0; GE edges finite.
- Measured-vs-model note: every gate landed where the corrected noise model said it
  would (GB edge, GB2 corner, GC floor) — the floor derivation is validated, not just
  the tolerances met.
- Remaining named blind spot (unchanged): profile-in-isolation; a scene-level smoke
  on carousel before first science use. The class is ready for the user's
  more-complex-system fits within the documented s_E in [-0.8, 0.8] support.
