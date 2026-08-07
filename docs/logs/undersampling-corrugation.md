# Lab Notebook — Undersampling corrugation testbed

Reproduce and characterize quadrature-aliasing "corrugation" of the log-likelihood on the
simplest possible system (a single **unlensed** Sersic), as the testbed for the
"quadrature error in the noise model" mitigation. Rationale: the main thing lensing changes
is *where* the undersampled points sit; if corrugation does not reproduce unlensed, the
lensed choppiness has a different cause (factor-map boundaries, solver, interpolation) and
noise-inflation would treat the wrong disease.

**Last updated:** 2026-08-07

---

## Current state

run1 (pilot) and run2 (DC-2, definitive for now) complete; claims C-1..C-5 registered
below, all UNCERTIFIED pending grader inspection of
`$PSCRATCH/gigalens/results/undersampling_corrugation/run2/plots/`. Harness:
`experiments/undersampling_corrugation/` (branch `corrugation-testbed`). Headlines:
corrugation = cusp-power quadrature aliasing (comb at f=ss, collapses under core
softening); A(ss) is NON-monotone pre-asymptotically (ss=4 can be worse than ss=1);
frozen σ_render noise inflation de-biases the mode and kills teeth *inside its
footprint* but is not a global smoother; n=8 cusps defeat ss_ref=64 certification.
Open next steps: lensed reproduction; model-dependent σ_render(θ) variant; archive run2
to CFS if kept.

---

## Claims register

All claims below are from **run2** (`$PSCRATCH/gigalens/results/undersampling_corrugation/run2/`,
harness commit `acce3e3`, gigalens `linusu-dev-merge` @ 923cefd) unless noted. Producer:
Claude (agent session, 2026-08-07). All **proposed (UNCERTIFIED)** — grader has not inspected.

### C-1 — Likelihood corrugation on an unlensed Sersic is subgrid quadrature aliasing
- **Status:** `proposed (UNCERTIFIED)`
- **Evidence:** comb locked at f = ss cyc/pix wherever the comb dominates the spectrum
  (f_peak/ss = 0.999–1.000 on the cuspy lanes; `plots/spectrum_*.png`); pipeline validated
  by injection (P1c: 6.1% amplitude error, freq within a bin); amplitude ordering correct in
  n and R_e; control relative amplitude ≤ 2×10⁻³ of the cuspiest lane at every ss;
  **softened-core arm**: at ss ≥ 2, A collapses monotonically with r_c by 1–3 orders of
  magnitude (e.g. ss=4: 4101 → 47 → 22 → 4.6 logL for r_c = 0, 0.25, 0.5, 1 pix), the
  discriminating signature of cusp-power aliasing (`plots/soft_arm.png`).
- **Doubt report:** (a) P1a's argmax criterion *failed* on low-amplitude configs — the
  global argmax lands on 0.75–1.5 cyc/pix detrending leftovers; the spectra still show the
  comb at f=ss as a secondary peak, so the criterion (argmax), not the mechanism, is what
  failed — but this is a producer's reading of plots, not a passing pre-registered test.
  (b) P1b slope = 0.90 (outside 1±0.05), dragged by the same ss=1 argmax misdetection.
  (c) At ss=1 the soft arm is non-monotone (A rises at r_c=0.25): softening only bites when
  r_c ≳ subpixel scale — consistent with band-limiting but NOT predicted in DC-2, which
  wrongly pinned the criterion to ss=1.

### C-2 — Pre-asymptotic aliasing: raising ss can *worsen* corrugation (STRUCTURAL)
- **Status:** `proposed (UNCERTIFIED)` — falsifies the naive "A monotone ↓ in ss" prediction
- **Evidence:** on reference-**certified** lanes (cert_gap < 0.03σ), A(ss) peaks interior:
  n4_re0.5_nopsf: 814 → 2241 → 4101 → 297 logL across ss = 1,2,4,8; same shape for
  n4_re1, n4_re3. Pre-declared as structural in DC-2 §7 and it fired.
- **Interpretation (producer's):** the error is dominated by the subgrid cell nearest the
  near-singular profile peak; its node-to-peak distance is a phase/resonance function of ss
  and the fixed truth sub-pixel offset, so the pre-asymptotic amplitude is non-monotone.
  Practical: moderate supersampling (2–4×) can make sampling *worse* on deeply
  undersampled cusps; only ss far into the asymptotic regime helps.

### C-3 — Aliasing bias = quantified "sub-pixel games"
- **Status:** `proposed (UNCERTIFIED)`
- **Evidence:** mode displacement up to 6.6 σ_x0 on a certified lane (n4_re0.5_nopsf,
  ss=1) and 2.3 σ at ss=2, vs ≤ 0.06 σ for the control; displacement oscillates in sign and
  magnitude with the truth sub-pixel phase (P6b′ passed; caveat: the two extra phase lanes
  are reference-limited, so magnitudes there are indicative only). Corrugation within ±1σ
  of the optimum reaches 211 logL (P6 passed) — far beyond posterior-relevant.

### C-4 — Stage 2, idea #3 (frozen noise inflation): a *local* de-biaser, not a global smoother
- **Status:** `proposed (UNCERTIFIED)` — pre-registered criterion (ii) FAILED, criterion (i) passed
- **Evidence:** lane n4_re0.5_nopsf. Mode bias 6.6σ → 0.87σ (ss=1) and 2.26σ → 0.05σ
  (ss=2), width cost 1.8× / 1.3× — criterion (i) pass. Global full-scan suppression only
  1.6× / 1.0× (criterion (ii) relevance_gain 0.9/0.8 < 3 — FAIL). The stage-2 plot shows
  why: corrugation teeth are visibly annihilated *within* the frozen σ_render map's spatial
  footprint (|Δx| ≲ 0.7 pix) and untouched beyond it; the full-scan metric integrates
  mostly uncovered territory (±2 pix ≫ σ_x0 ≈ 0.002–0.02 pix).
- **Practical reading:** the fit→σ_render(MAP)→refit/sample workflow should work — the
  posterior bulk sits deep inside the footprint — but the frozen map cannot protect
  far-from-MAP exploration. A global version needs model-dependent inflation
  σ_render(θ) = |m_ss(θ) − m_2ss(θ)| (≈2× render cost per likelihood call). F_pred
  (map-derived suppression forecast) failed in both runs; withdrawn as a planning tool.

### C-5 — n=8 cusps defeat supersampling as its own referee
- **Status:** `proposed (UNCERTIFIED)`
- **Evidence:** every n=8 lane is reference-limited even at ss_ref=64 without a PSF
  (cert_gap 0.15–1.7σ at peak-SNR-50 scaling); softened lanes certify at ~0.004–0.01σ.
  Supersampling alone cannot produce a trustworthy reference for such profiles —
  singularity handling (core split / pre-filtering) is required, not more factor.

### Sub-findings / harness notes
- P3′ absolute part failed as specified: control A(ss=2)=1.55 > 0.5 over the full ±2 pix
  span, but A_within_1σ = 0.009 ≪ 0.5 — full-span peak-to-trough is a scan-span-dependent
  quantity and was the wrong absolute observable; the posterior-relevant control amplitude
  is tiny. (Criterion error, recorded; relative criterion passed at all ss.)
- P4 harness check: grad/pred ratios in [0.36, 4.96] — outside the factor-3 band on
  sawtooth-like configs (high crest factor); expected for non-sinusoidal combs.
- P5 (descriptive): PSF is **not** an anti-aliasing filter here — A_nopsf/A_psf straddles 1
  (0.53–4.8): convolution happens after subgrid sampling, so committed aliasing is not
  removed (grader's caution confirmed).
- PSF-lane references converge far worse than no-PSF (cert_gap up to 0.66σ at ss_ref=32):
  much of the "reference error" lives in the PSF convention path (native vs subgrid kernel
  + fused path), corroborating the undersampling diagnostic's convention-gap finding.

---

## Design checkpoints (criteria awaiting approval)

### DC-1 — Corrugation reproduction scan (unlensed Sersic, centroid phase scan)

**Cause hypothesis (H1).** Choppiness of the posterior under coarse supersampling is
midpoint-quadrature aliasing of profile power above the subgrid Nyquist. Translating the
profile centroid changes only the sub-pixel phase of the cusp relative to the quadrature
grid, so the noiseless log-likelihood along the centroid must corrugate **periodically with
period exactly `delta_pix / ss`** (the subgrid spacing).

**Setup.** Single unlensed Sersic; data = noiseless model image rendered at reference
supersampling ss_ref=16 (convention-matched: same pipeline, only the factor changes);
Gaussian background-only noise map (σ = const), flux scaled to peak-pixel SNR = 50;
Gaussian PSF FWHM = 2.5 pix; truth centroid at a generic sub-pixel phase (+0.30, +0.15) pix
off a pixel center. Scan model `center_x` over truth ± 1 pix, 801 points; everything else
frozen at truth. Record logL(x0) and d logL/dx0 (jax.grad). Grid: n ∈ {1,4,8} ×
R_e ∈ {0.5,1,3} pix × ss ∈ {1,2,4,8}; plus a no-PSF variant of (n=4, R_e=0.5) and a
well-resolved negative control (n=1, R_e=10 pix, ss=1). Noiseless data by design — no
seeds; the corrugation is deterministic and this isolates it. float64 throughout
(`jax_enable_x64` + `likelihood_precision="float64"`).

**Predictions (direction + magnitude) and falsifiers:**

- **P1 (period — the sharp falsifier).** Dominant DFT peak of the detrended logL(x0) scan
  at spatial frequency `ss/delta_pix` (± 1 frequency bin), peak power > 10× the median
  detrended power. The ±1-bin window is derived from the scan length (2 pix span ⇒ bin
  width 0.5 cyc/pix); the 10× margin is a pre-registered detection convention, not derived.
  *Falsifier: comb absent, or at any other frequency ⇒ H1 wrong or harness broken.*
- **P2 (amplitude ordering).** Peak-to-trough amplitude A increases with n, increases as
  R_e shrinks, decreases monotonically with ss. Direction only; magnitudes recorded, not
  predicted (no honest derivation available for a cusp under midpoint rule).
- **P3 (negative control).** The resolved control (n=1, R_e=10 pix, ss=1) shows **no** comb
  by P1's criterion. *If the control shows a comb, the criterion or the reference render is
  broken (e.g. ss_ref itself aliased) — stop, do not interpret the grid.*
- **P4 (gradient amplification).** Corrugation amplitude of the gradient ≈ (2π·ss/delta_pix)·A,
  within a factor of 3 (exact for a pure sinusoid; harmonics soften it). This is the
  sampler-relevant quantity.
- **P5 (PSF).** PSF convolution reduces A but leaves the period unchanged (aliasing is
  committed at binning; convolution is linear and smooth).
- **P6 (posterior relevance).** In at least one cuspy config (expected: n=8, R_e=0.5 pix,
  ss≤2), A > 1 logL unit within ±1σ of the optimum, σ derived from a quadratic fit to the
  smooth ss_ref scan. Threshold derivation: 1 logL unit ≈ the scale that displaces a 1σ
  contour; below that, corrugation is measurable but posterior-irrelevant at this SNR.
- **P6b (mode displacement — the "sub-pixel games" metric).** The corrugation wavelength
  (≥ 1/ss pix) likely exceeds the single-parameter posterior width at SNR 50, in which case
  aliasing acts as a *bias of the mode*, not roughness: predict |argmax logL_ss −
  argmax logL_ref| > 1 σ_x0 in the cuspy configs at ss ≤ 2, shrinking with ss and with
  resolution (control: ≪ 1 σ_x0). This is the quantitative form of the optimizer
  "situating images on sub-pixel features". (Added pre-run, before any scan was executed.)
- **Structural-vs-fine-tuning classification, pre-declared:** P1 or P3 failing = structural
  (mechanism story wrong / harness broken — stop and diagnose). P2/P4/P5 failing in
  magnitude but not direction = fine-tuning of the harness (investigate, but H1 survives).

**Stage 2 (only if P1+P3 pass) — noise-inflation mitigation (H2).** Freeze
σ_render(x) = |m_ss − m_2ss| at truth params; inflate σ_eff² = σ² + σ_render²; re-run the
worst config's scan. Pre-run, compute a predicted suppression factor F from the maps alone
(per-pixel weights (σ/σ_eff)² applied to the corrugation residual map at scan extremes).
*Prediction: measured whitened-amplitude suppression within a factor 2 of F, and smooth
posterior width (quadratic-fit σ) inflated by < 3×. Falsifier: suppression < 2× in the
flagged config, or width inflation > 3× (cure worse than disease).*

**Cost.** Login-node, CPU/GPU-trivial (48² images × ~40 scans × 801 points), minutes.
**Approval:** design approved in conversation by Linus 2026-08-07 (predictions stated
verbatim in chat before this checkpoint; run is cheap and non-Slurm).

---

### DC-2 — run2: amended harness (grader amendments + pilot post-mortem)

Pre-registered **before** run2; run1 is demoted to an uncertified pilot (its evidence is
recorded below in the Log). Amendments implemented (grader verdict: APPROVE-WITH-AMENDMENTS;
pilot failures independently corroborated amendments 1, 3, 5):

1. **No-PSF series is primary for H1.** The PSF operator differs by ss (native kernel at
   ss=1, `subgrid_kernel` at ss≥2, fused spectral path at ss≥3), so the PSF grid is
   demoted to descriptive/P5. No-PSF lanes have one operator (render + bin) at every ss.
   Deviation from grader amendment 2 (pin `_FUSE_CONV_POOL=False`): not patched — H1 no
   longer rides on any convolved lane, and the PSF lanes are meant to show the production
   path as shipped.
2. **Reference certification per lane.** cert_gap = max|m(ss_ref) − m(ss_ref/2)|/σ at
   truth; ss_ref=64 (no-PSF) / 32 (PSF). Lanes with cert_gap > 0.1 are **reference-limited**:
   period claims stand, amplitude/stage-2 claims do not. (Pilot: n8_re0.5 was
   reference-limited at ss_ref=16 — residual comb sat at the *reference's* 16 cyc/pix.)
3. **P1′ (co-primary, replaces P1).** (a) f_peak within ±1 DFT bin of ss per lane;
   (b) log–log regression of f_peak on ss across {1,2,4,8}: slope 1 ± 0.05 per cuspy
   primary lane; (c) **injection test**: a planted sinusoid (amplitude A_inj, 3.3 cyc/pix)
   on the reference scan must be recovered by the detrend+spectrum pipeline within 15% in
   amplitude and ±1 bin in frequency, else no null is interpretable. Scan span widened to
   ±2 pix (ss=1 gets 4 periods).
4. **P3′ (control, absolute scale — replaces P3).** Pilot post-mortem: the scale-free
   "peak > 10× median" criterion **cannot fail** on noiseless deterministic scans (the
   spectral background is roundoff leakage), and the physical prediction "no comb" was
   wrong — every Sersic has a central cusp, so the control corrugates *small*, not zero
   (measured: A=0.99 logL at ss=1, 0.0019 at ss=8). New criterion: A_control(ss)/A_cuspiest(ss)
   < 10⁻² at every ss, and A_control < 0.5 logL (the 1σ-contour unit, declared convention)
   for ss ≥ 2. Amplitudes below 10× the float64 floor (≈10⁻¹³·max|logL|·scan) are
   reported "unresolved", not zero.
5. **Softened-core arm (discriminates cusp-aliasing from any other comb source, and
   previews the pre-filter mitigation).** `SoftenedSersic`: r → √(r² + r_c²), lane
   (n=4, R_e=0.5 pix, no-PSF), r_c ∈ {0.25, 0.5, 1.0} pix, gated by r_c=0 ≡ stock Sersic
   to float64 roundoff (max rel diff < 10⁻¹²). Prediction: A monotone ↓ in r_c;
   A(r_c=1 pix)/A(0) < 0.1 at ss=1 (order-of-magnitude expectation, declared as such).
   Falsifier: A insensitive to r_c or non-monotone ⇒ the comb is not cusp-power aliasing.
6. **P6b′ (phase arm).** Mode displacement must oscillate with the truth sub-pixel phase:
   cuspy no-PSF lane at truth phases (0,0), (0.30,0.15), (0.47,0.31) — displacement
   varying in sign/magnitude with phase distinguishes aliasing bias from any smooth
   systematic (which would displace identically).
7. **Stage-2′ (dimensionless, replaces the F_pred primary).** Pilot post-mortem: the
   map-derived suppression prediction F_pred **failed** (predicted 76×/47×, measured
   15×/2.3× — the frozen truth-point map under-covers scan-displaced error patterns);
   F_pred is demoted to a recorded secondary. New pre-registered pass criteria, on a
   reference-certified lane: (i) post-inflation mode bias |x̂ − x̂_ref|/σ_new < 1;
   (ii) corrugation *relevance* ρ = A/0.5 falls by > 3× more than the posterior width
   ratio grows, i.e. (ρ_old/ρ_new)/(σ_new/σ_old) > 3; width inflation recorded, no longer
   a hard bound (pilot: 9.5× at the extreme lane — arguably honest uncertainty, recorded
   as a cost). P2's A-vs-ss reversal on a reference-**certified** lane is reclassified
   **structural** (grader amendment); P4 is relabeled a harness check (near-identity under
   jax.grad), not H1 evidence.

## Log

- **2026-08-07** — DC-1 written; harness under construction on branch
  `corrugation-testbed` (worktree).
- **2026-08-07** — **run1 (pilot, UNCERTIFIED)** at
  `$PSCRATCH/gigalens/results/undersampling_corrugation/run1/`. Mechanism strongly
  supported where the harness is valid: combs at f = ss cyc/pix to ≲½ bin in every
  non-reference-limited lane; amplitude ordering correct (control 0.99 logL vs cuspy
  10³–10⁴ at ss=1); gradient amplification within 1.0–1.5× of 2πf·A except on the
  reference-limited lane; mode displacement up to 86 σ_x0 (n=8, R_e=0.5 pix, ss=1) vs
  0.04 σ in the control — the "sub-pixel games" bias, quantified. Harness failures (why
  pilot, not result): P3 criterion unfalsifiable as designed; n8_re0.5 reference-limited
  at ss_ref=16 (comb at 16 cyc/pix, non-monotone A, grad ratio 26×); stage-2 F_pred
  missed by ≫2×. All three fed DC-2. Plots inspected (scan_*, spectrum_*, maps_*):
  combs visibly locked to subgrid-period gridlines; cusp-tooth harmonics at high n;
  sawtooth gradient residuals.
