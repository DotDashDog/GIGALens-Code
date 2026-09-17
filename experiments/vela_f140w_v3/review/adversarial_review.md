# Adversarial review — is the "Vela F140W v3" set a good test of *source structure*?

Referee pass, 2026-09-17. Scope: **source-side** validity only. Everything below is measured on the
delivered set (`/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset`), the pristine
sources (`/pscratch/sd/l/linusu/gigalens/vela_sources_pristine`), the HLSP FITS headers, and the
generator at
`/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/vela-generator-v2/src/gigalens_research/simtests/experiments/vela_simulated.py`.
Scripts and figures are under `/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/`.
Nothing in the repo was modified.

Labels: **[M]** measured here, **[I]** inferred from a measurement, **[R-UNCERTIFIED]** recalled
literature, not checked.

---

## Ranked summary

| # | Severity | Kind | Finding | Key number |
|---|---|---|---|---|
| 1 | **Blocking** | structurally wrong | The VELA "pristine" maps are not surface-brightness fields: at the pixel scale they are a shot-noise / discreteness realisation, and that noise enters the delivered truth images at a level comparable to the observational noise. | A 3×3 median removes **4.8–29.3 %** of the source flux; smoothing the source by **one simulation cell** changes the noiseless image by Δχ² = **22–6365** (0.08–1.24 σ rms over detected pixels) |
| 2 | **Serious** | structurally wrong | What the data constrain is a single compact blob, not clumpy morphology; the source models are far more flexible than the data. | **10–100** independent resolution elements per system vs **66–496** shapelet coefficients in the n_max sweep; ≤1 peak above 20 % of max at the effective source-plane resolution in **6/10** systems |
| 3 | **Serious** | structurally wrong | The set has only one effective axis of variation: source size, magnification, brightness, S/N and information content are locked together, so per-system performance cannot be attributed to morphology. | ρ(R50, μ) = **−0.76** (p=0.011); ρ(R50, m_unlensed) = **−0.90**; ρ(R50, S/N_arc) = **+0.71** |
| 4 | **Serious** | fine-tuning with a structural cause | The peak-SB anchor applied to VELA's flat, extended profiles yields implausible source luminosities; this is a *source-structure* problem, not a brightness-draw problem. | Unlensed **M_AB = −21.2 … −25.1** (median −22.8); total arc flux = **15–110 ×** the flux in the 9 calibration pixels |
| 5 | **Serious** | structurally wrong *for a source test* | Every real limitation on source reconstruction is absent or exact: lens light is exactly the fitter's Sérsic, PSF is handed over exactly, noise is white. | **7–48 %** (median 27 %) of detected arc flux lies where the lens light is brighter than the arc; lens-light Poisson inflates the effective arc noise by **9–29 %** |
| 6 | Minor | structurally wrong (small) | Hidden heterogeneity in the truth: 2 of the 10 kept sources are rendered at **half** the intrinsic resolution; morphology space is one suite / one snapshot / one camera / hand-picked. | PIXKPC 0.125 vs 0.0625 kpc/px, `linear_fov` 100 vs 50 kpc (vela21, vela26) |
| 7 | Minor | bookkeeping | The "source" panel of the group figures is the **transpose** (mirror) of the source that was actually lensed. | verified numerically: `img[3,1]` renders at (x=+1, y=−1) |
| 8 | — | checks that came out **negative** | units/zeropoint, frame-edge truncation, robustness of the peak-SB anchor to spikes, supersample adequacy, lens-light amplitude realism. | see §8 |

---

## 1. Blocking — the source is a noise realisation at the pixel level

### What a VELA "pristine" image is
Header facts **[M]** (`review/hdr.py`, `review/ext.py`): `CODE = 'Sunrise (Jonsson 06)'`, 800×800,
`linear_fov = 50 kpc` (100 kpc for vela07/21/26), `PIXKPC = 0.0625`, `PIXSIZE = 0.00727"`,
`IMUNIT = nanoJanskies`, `SKYSIG = 0.0`. For **F140W the tarballs contain only two HDUs,
`IMAGE_PSF` and `IMAGE_PRISTINE`** — there is **no `IMAGE_PRISTINE_NONSCATTER`** (the module
docstring's verified layout is from an f814w file). So the `source_variant:
IMAGE_PRISTINE_NONSCATTER` escape hatch does not exist for this campaign, and the scattered-light
(Monte-Carlo) term cannot be switched off.

### Evidence that the map is shot-noise / discreteness dominated
All **[M]**, `review/mcnoise.py`, `review/quant.py`, `review/shotnoise.py`,
figures `fig_mcnoise.png`, `fig_shotnoise.png`:

* **A 3×3 median filter removes 4.8 % (vela22), 7.3 % (vela21), 16.7 % (vela02), 18.9 % (vela04),
  29.3 % (vela25) of the total source flux.** A smooth galaxy profile loses <1 %. The flux lives in
  isolated single pixels.
* Pixels with `img > 5 × median3` hold **3.0–20.7 %** of the total flux.
* In a fixed annulus (r = 1.0–1.5″): 0.2–17.7 % of pixels are **exactly zero**; mean/median up to
  **190**; p99/median up to **2810**.
* The brightest isolated peaks of vela25 and vela04 drop to **0.3–1.8 % of the peak one pixel away** —
  literal delta functions at 0.0625 kpc. (vela25's five brightest isolated peaks are scattered over
  the whole 5.8″ frame, i.e. in the halo, not in star-forming regions.)
* `fig_shotnoise.png`: Var(residual)/mean rises ∝ mean, i.e. the scatter is **constant-fractional**,
  not fixed-weight Poisson — σ/mean ≈ **2–6 at faint SB and still 0.34–1.05 in the brightest 0.5 %
  of the map**. **[I]** this is Sunrise MC photon-packet noise with a wide weight distribution and/or
  star-particle discreteness; either way it is not a physical 60-pc surface-brightness field.

### It reaches the delivered images
`review/render_speckle.py`, `review/render_speckle2.py`, figures `fig_speckle.png`,
`fig_speckle2.png`. Re-rendering the lensed source (same truth, same PSF, supersample 16 — validated
against the on-disk ss32 render to 0.002–0.016 σ rms) from a source Gaussian-smoothed by **one
source pixel** (σ = 0.00727″ = 0.0625 kpc = 1 simulation cell; flux preserved to 0.07–0.5 %):

| system | max\|Δ\|/σ | rms/σ over detected px | Δχ² |
|---|---|---|---|
| vela22 | 1.4 | 0.077 | 22 |
| vela04 | 2.3 | 0.221 | 33 |
| vela02 | 2.0 | 0.250 | 183 |
| vela25 | 6.1 | 0.492 | 1077 |
| vela21 | 22.9 | 1.244 | 6365 |

At 2 cells (still ≲ the tangential source-plane resolution at μ≈10, and ≪ the radial one) the numbers
are 242 / 437 / 973 / 2586 / 51205.

**[I] Consequence for a method comparison.** The truth carries pixel-scale power of order the
observational noise that (a) no smooth source model can reproduce and (b) *flexible* source models
can partly absorb. Any ranking by χ², evidence, residual rms or z-score therefore rewards source
freedom for a reason that is numerical, not astrophysical — exactly the axis along which shapelets
(n_max), Voronoi/pixelated (pixel size, regularisation) and neural fields differ. It also means the
effective noise on the arcs is larger than the noise the likelihood is told about, so posterior
widths and mass-parameter z-scores are biased.

**Caveat, stated honestly [I]:** part of the 1-cell difference is genuine compact structure, because
at μ_t ≈ 10 a 1-cell scale maps to ≈ 0.07″ in the tangential direction, comparable to the PSF σ. For
vela21 (a numerically sharp nucleus, profile 1 / 0.69 / 0.32 / 0.21 at r = 0,1,2,3 px) the 6365 is
probably cusp, not noise. The *shot-noise* interpretation rests on the median-filter flux loss, the
exact zeros, the constant-fractional scatter and the delta-function peaks, which are independent of
the render test.

**Cheapest falsifier.** Regenerate the set from sources pre-smoothed by 1–2 source pixels (total flux
changes <0.5 %, and the group's own supersample-32 decision already shows the pipeline is converged
at that scale). Then: (i) each method's best-fit χ² should drop by the Δχ² above — if it does, the
current set is charging methods for simulation noise; (ii) re-run the method comparison on both
versions — **if the ranking changes, the current set is measuring MC noise, not source
reconstruction.** A second, independent check: take the brightest 20 isolated single-pixel peaks per
source, zero them, and confirm the lensed image changes by <0.1 σ; it will not for vela25/04.

---

## 2. Serious — the data constrain far less structure than the models have freedom

`review/imgsn.py`, `review/obs_src.py`, figure `fig_obs_src.png`. All **[M]**.

* Independent resolution elements above 3σ (area above 3σ ÷ PSF effective area 1/Σp² = 36.0 px):
  **vela04 10.3, vela23 19.9, vela03 37.9, vela08 50.3, vela02 61.9, vela09 63.8, vela22 73.9,
  vela21 88.5, vela26 93.7, vela25 99.9.**
* The sweep axis is `n_max ∈ {10,15,20,30}` = **66 / 136 / 231 / 496** shapelet amplitudes. For every
  system except perhaps vela25, n_max ≥ 15 already gives more free source amplitudes than the image
  has resolution elements. **[I]** the sweep then measures the prior/regulariser, not the data, and
  differences between methods will be dominated by how each regularises an under-determined problem.
* Blurring each source to the isotropic-equivalent source-plane resolution PSF/√μ (0.037–0.096″ =
  0.31–0.82 kpc) and counting peaks above 20 % of the maximum: **1 peak in 6/10 systems**, 2 in
  vela03/vela09, 3 in vela02, 7 in vela25. Fraction of flux in structure above 2× a smooth baseline:
  **0.00–0.42, median 0.13**. `fig_obs_src.png` bottom row (linear stretch, i.e. the stretch χ² sees)
  shows a single centrally-peaked blob for 7 of 10 sources; the clumps only appear in asinh at <5 %
  of the peak.
* Positive note **[M]**: the clumps are *not* parked at low magnification — the six brightest clumps
  per system sit at μ = 1.1–98, typically within a factor ~2 of μ_eff, and 26–99 % (median 88 %) of
  the source flux is at μ>3 (`review/lensgeom.py`, validated: my μ_eff reproduces the recorded
  `magnification_cutout` to 1–14 %). So the geometry is not the problem; the resolution and contrast
  budget is.

**Cheapest falsifier.** Fit each system with a *single elliptical Sérsic source* and record Δχ² against
the best flexible model. If that Δχ² is small compared with the spread between methods, the set is not
discriminating source-structure models at all. (Also worth reporting per system: the set then has a
built-in "difficulty" scale you can quote.)

---

## 3. Serious — the design is one-dimensional; morphology is confounded with magnification

`review/confound.py`, all **[M]**, n=10, Spearman:

* ρ(R50_source, μ) = **−0.76 (p = 0.011)** — bigger sources get systematically *lower* magnification
  (partly geometry, partly the cut-off redraw loop, which rejects on `flux_outside` and
  `border_sb_sigma`, both of which scale with source extent × θ_E).
* ρ(R50, unlensed AB mag) = **−0.90 (p < 0.001)** — near-deterministic: peak-SB calibration means
  total flux ∝ emitting area.
* ρ(R50, total arc S/N) = **+0.71 (p = 0.022)**; ρ(R50, N_res) = **+0.62 (p = 0.054)**.
* Ranges: R50 0.16–0.72″, μ 3.2–22.2, unlensed mag 20.1–24.0, arc S/N 72–702, N_res 10–100.

**[I]** "Ten different morphologies" is in practice a single difficulty ladder ordered by source size,
with magnification running the other way. When method A beats method B on vela22 and loses on vela25,
you cannot say whether that is morphology (compact bulge vs extended irregular), magnification (22 vs
3.2) or S/N (225 vs 702). For a paper claim of the form "method X recovers clumpy sources better",
this design cannot support the claim.

**Cheapest fix / falsifier.** Cross the design: lens every source through the **same** 2–3 lens
configurations (θ_E ∈ {1.0, 1.5, 2.2}, one or two source positions), instead of one private lens draw
per source. Cost 2–3×; it removes the confound entirely and makes the per-source differences
interpretable. Falsifier for the concern: if per-system method rankings are the same at fixed source
across θ_E, the confound is harmless.

---

## 4. Serious — the implied source luminosities, and why it is a *structure* problem

`review/conc.py`, `review/units.py`, all **[M]** except the LF comparison.

* Unlensed AB magnitudes from the peak-SB route: 20.12 … 24.03. With the header distance modulus
  45.2226 that is **M_AB(rest ≈ 5550 Å) = −25.11 (vela25), −24.08 (vela21), −23.10 (vela09) …
  −21.19 (vela04)**.
* **[R-UNCERTIFIED]** M*_V at z ≈ 1.5 is around −22.3; −25.1 is then ≈ 15–20 L*, far out on the
  luminosity function even after magnification bias, which anyway favours intrinsically *faint*
  sources.
* The Foundry-derived lower limits recorded in the lab log are 26.5–29.0 AB unlensed (mean 27.7).
  The set sits **3–8 mag brighter**. Part of that gap is legitimate (an isophotal magnitude over a
  5–33 px contour is a lower limit on the arc's flux), but a factor 40–1500 is not.
* The mechanism is structural, and that is the point: **the arcs carry 15–110× the flux in the 9
  pixels used for the calibration** (vela25 110×, vela22 81×, vela03 65×; vela04 only 15×). A real
  arc whose isophotal contour is 9 px does not have 110× more flux outside it. VELA's SB profiles are
  too flat/extended relative to their peak, so *any* peak-anchored calibration puts too much light in
  the envelope.
* Corollary **[M]**: source-to-lens flux ratio in the cutout is **0.07–1.22, median 0.33** — in
  vela25 the arcs are *brighter than the deflector*. (Whether that is unrealistic is not settled by
  the one real cutout on disk; see §8.)

This overlaps the tabled brightness question, so I flag only the part I think is worse than recorded:
it is **not fixable by changing the brightness draw**. The peak-SB anchor and the magnification route
disagree by 3–8 mag *because of the source profile shape*; whichever anchor you pick, one of "arc peak
SB" and "source total luminosity" will be wrong for these sources. The honest options are (a) accept
it and never quote source photometry from this set, or (b) truncate/steepen the source envelopes,
which the "no crop" decision deliberately forbids.

**Cheapest falsifier.** Measure total-to-isophotal flux ratios for a handful of real F140W arcs (the
Foundry team has the mosaics; even 3 systems suffice). If real arcs also have F_tot/F_9px ≈ 50–100,
this finding collapses.

---

## 5. Serious — for a *source* test, every real limitation has been removed

**[M]** `review/confound.py`, `review/lensprof.py`:

* **7–48 % (median 27 %) of the detected arc flux** lies in pixels where the lens light is brighter
  than the arc (vela03 48 %, vela04 31 %, vela21 35 %, vela23 31 %). The lens light there is *exactly*
  the SersicEllipse the fitter uses, with no isophote twists, boxiness, colour gradient or PSF
  mismatch.
* The lens-light Poisson term inflates the effective per-pixel noise on the arc by **9–29 %** — that
  part is realistic and correctly modelled.
* The fitter is handed the **exact** PSF kernel; noise is white (drizzle correlation tabled); no
  cosmic rays, bad pixels or neighbours.

**[I]** In real reconstructions of HST arcs, lens-light residuals and PSF error are the dominant
systematics on the recovered source, and they hit methods unequally (a flexible pixelated source
absorbs a lens-light residual as a fake central source component; a shapelet source with low n_max
cannot). A ranking obtained on this set is a ranking in the *absence* of the effects that separate the
methods in practice. This does not invalidate the set, but it does bound the claim it can support to
"in the idealised limit".

**Cheapest falsifier.** Re-fit one system twice more: once with the PSF perturbed (e.g. 3 % FWHM
error, or the neighbouring STDPSF grid point), once with a 2 % lens-light model mismatch (fit an
n-fixed Sérsic to an n-free truth). If the ranking survives both, the concern is minor.

---

## 6. Minor — hidden heterogeneity and narrow morphology space

**[M]** `review/src_stats.py`, metadata + headers:

* **vela21 and vela26 (kept) and vela07 (dropped) come from 100-kpc-FOV renders**: `PIXKPC = 0.125`,
  `PIXSIZE = 0.01454″`, `SBFACTOR = 3.193` versus 0.0625 kpc / 0.00727″ / 0.798 for the other seven.
  The pipeline handles the scale correctly, but the *truth* for those two systems is at half the
  intrinsic resolution and twice the angular field. For a test whose whole point is small-scale source
  structure, that is an uncontrolled factor — and vela21 is the system with the largest 1-cell
  smoothing response (Δχ² 6365).
* Sample: 10 galaxies, **one suite, one snapshot (a = 0.400), one camera (cam12), chosen by eye**.
  Measured R50 = 0.163–0.717″ (1.40–6.16 kpc), native VELA total mags 21.2–24.2. All are star-forming
  clumpy/irregular rest-frame-V morphologies. There is no smooth quiescent spheroid source, no clean
  edge-on disc with a dust lane (vela23 is the nearest), no wide interacting pair, no source with an
  AGN point component. **[R-UNCERTIFIED]** the VELA suite is known to produce overly compact,
  clump-rich, high-SB galaxies; and at fixed camera the same galaxy at other viewing angles would give
  very different apparent clumpiness — with one camera you sample none of that variance.
* One camera also means **projection is perfectly correlated with morphology**: vela23's edge-on
  appearance is a viewing choice, not a property of the sample.

---

## 7. Minor — the truth panel is the mirror of the lensed source

**[M]** `review/orient.py`: `ImageBasedLight` builds
`RegularGridInterpolator((coords_1d, coords_1d), image)` and queries `stack([x, y])`, so **axis 0 of
the image is x**. A test array with a single non-zero pixel at `img[3,1]` renders at
(x = +1, y = −1) — i.e. the lensed morphology is the **transpose** of what
`imshow(img, origin="lower")` shows. `experiments/vela_plot_dataset.py:53-73` loads the source and
displays it with exactly that `imshow`, so the "source" panel of `dataset_gallery.png` is mirrored
across the diagonal relative to the arcs in the same figure. Physically harmless (a reflection of a
galaxy is a galaxy), but it will mislead any visual truth-vs-reconstruction comparison and would
silently corrupt a quantitative source-plane comparison written the same way. Either set
`transpose_image: true` or transpose in the plotting/scoring code — and say which in the manifest.

Related, smaller: `ImageBasedLight` casts the source to **float32** while the rest of the pipeline runs
float64, and the truth source is defined by **bilinear** interpolation on a 0.00727″ lattice. Neither
matters at these resolutions **[I]**, but it means "the true source" is only defined once you state a
comparison resolution — worth fixing in the scoring protocol before any method comparison is scored.

---

## 8. Checks that came out negative (reported so they are not re-run)

* **Units / zeropoint [M]** (`review/units.py`): the stored `image_sum_nJy` and the header `MAG` agree
  to ≤0.07 mag for all ten sources; the derived ZP 26.4535 is consistent with `PHOTFNU = 9.52e-8`.
  Note that the `FLUX_NJY` consistency assertion in `_extract_pristine` **did not fire** for these
  files (`pristine_flux_nJy: null` in every metadata.json) — the check above is the substitute.
* **Frame-edge truncation after recentring [M]** (`review/recenter_edge.py`, `review/edge_render.py`):
  recentring shifts the frame by up to 1.12″ × 1.72″ (vela03), leaving the nearest zero-fill edge at
  1.19″ (vela03) to 5.75″ (vela26). Rendering with the outer 0.3″ of the source frame zeroed changes
  the arc by coherent S/N **1.9 for vela25** and **<0.1 for vela03/09/02**. By the group's own
  standard (the 1.5″ crop edge that summed to S/N 16 was rejected) this is acceptable, but vela25's
  edge is the one worth quoting: azimuthally-averaged SB at the truncation radius is 0.44 σ per output
  pixel.
* **Is the peak-SB anchor set by a numerical spike? [M]** (`review/peakcal.py`): re-deriving the
  9-brightest-pixel SB from a median-filtered source shifts it by only **0.05–0.18 mag** — the PSF
  spreads a single-pixel spike over ~36 px, so the anchor is on the arc, not on a spike. Good.
* **Supersample [M]**: my ss16 renders match the delivered ss32 lensed-source images to 0.002–0.016 σ
  rms (max 0.06–0.32 σ). The ss32 decision is fine.
* **Lens-light realism [M]** (`review/lensprof.py`): the simulated lens-only median S/N at
  r = 1.0/1.5/2.0/2.5″ is 22–39 / 9.5–20 / 5.3–12 / 3.3–8.2, against 39 / 26 / 12 / 5.7 for the real
  DESI-238 cutout. The deflector light level is realistic; it is the *arcs* that are bright relative
  to that one real system.
* **Are the clumps hidden at low magnification? [M]** — no, see §2.

---

## What I could not check

* Whether the pixel-scale scatter is Sunrise MC photon noise specifically, versus star-particle
  discreteness: the decisive comparison (`IMAGE_PRISTINE` vs `IMAGE_PRISTINE_NONSCATTER`) is
  impossible because the F140W HLSP files ship only `IMAGE_PSF` + `IMAGE_PRISTINE`. A second
  realisation of the same snapshot, or the same galaxy from an adjacent camera, would settle it.
  The consequence for the test set is the same either way.
* Stellar masses / SFRs of the ten galaxies: not in the FITS headers (only MAG, ABSMAG, distances,
  cosmology). Any statement about where they sit on the z≈1.5 mass–size or main-sequence relations
  would be recall, not measurement.
* A quantitative comparison with the *real* lensed-source population (sizes, clumpiness, arc
  total-to-isophotal flux ratios): only one real cutout (DESI-238) is on disk, and its arc cannot be
  separated from the deflector without a lens model.
* Whether the ranking of shapelets / Voronoi / neural-field sources actually changes under any of the
  above — that needs the fits, which I did not run.
