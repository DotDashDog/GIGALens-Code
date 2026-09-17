# Lab notebook — Vela simulated lens systems (generation)

Area: simulating lensed Vela galaxies as test systems for source-model
systematics campaigns. Companion code: `src/gigalens_research/simtests/experiments/vela_simulated.py`,
campaign `experiments/vela_revised_v2/campaign.yaml`.

## Current state (2026-09-13)

- **Generation 1** (`data/vela_sim_systems`, 60 systems, a0.500 = z 1.0) came from the
  notebook `experiments/vela_sim_systems/lens_vela_system.ipynb`: notebook cell-4 prior,
  Gaussian 0.13" PSF, `background_rms=0.002`, `exp_time=2000`, 200 px at 0.03", supersample 4.
  Known problems: unrealistic parameter combinations (lens-dominated images, arcs leaving the
  cutout), an ad-hoc noise level, and a Gaussian stand-in for the ACS PSF.
- **Generation 2 / v1** (`vela_revised_v1`, WIP commit e7f80b4): the `vela_simulated`
  generator moved the recipe into simtests with the five science knobs required-but-uncalibrated.
  A 12-system test set at a0.400 (z 1.5) exists on `$PSCRATCH/gigalens/simtests_results/vela_revised_v1`
  (Gaussian PSF 0.13", `background_rms=0.005`, `exp_time=2000`, `source_flux_scale=1`).
  The generator used the removed old API (`gigalens.jax.physical_model`, `LensSimulator`) and
  no longer imported.
- **v2 (this entry):** generator ported to the scene API (`LensModel` / `SceneSimulator`),
  `ImageBasedLight` given the `amp` parameter the scene API requires, and four mechanisms added:
  YAML-tunable truth prior, instrument PSF (STScI ePSF) + instrument-derived noise, source-to-lens
  flux calibration, and a cut-off rejection check. All choices are recorded in the manifest and
  per-system `generation.json`.

## Measurements on the v1 test set (basis for the v2 defaults)

Re-rendered with the scene API (lens light only, lensed source only) for three systems:

| system | theta_E | lens light (cps, cutout) | lensed source (cps) | source/lens | source flux outside 6" cutout |
|---|---|---|---|---|---|
| vela02 | 1.77 | 944 | 123 | 0.13 | 1.4% |
| vela07 | 0.86 | 1976 | 96 | 0.05 | 13.6% |
| vela22 | 2.85 | 1435 | 66 | 0.05 | 20.6% |

Across all 12: unlensed Vela sources at z=1.5 are 2–47 cps (AB 21.8–25.3), magnifications 2–16,
lens light 400–2900 cps (Sérsic with Ie≈20, Re≈1.6" is ≈18.4 AB total; SLACS-like). Border max
of the *observed* image reached 4–11 sigma_bkg. Sources vela09 / vela13 / vela20 are doubles or
very extended (candidate_sources.png in `experiments/vela_revised_v1`).

PSF: the STScI empirical ACS/WFC F814W ePSF (`STDPSF_ACSWFC_F814W_SM4.fits`, 4x oversampled,
native 0.05") reads with `photutils.psf.STDPSFGrid`; measured FWHM 0.081", EE(<0.13") = 0.81,
EE(<0.5") = 0.997. VELA's own TinyTim PSF (`TinyTim_IllustrisPSFs/F814W_rebin.fits`, header
APROXPSF 0.13") is not distributed in the HLSP.

Noise (ACS F814W, ZP 25.94 from PHOTFNU, read noise 4.5 e, dark 0.0075 e/s): sigma_bkg per
0.03" pixel = 0.0058 cps (1200 s, 2 reads), 0.0041 (2400 s, 4 reads), 0.0029 (4800 s, 8 reads)
for sky 22.3 AB/arcsec²; ±0.4 mag of sky moves these by ~10%.

## v2 design (proposed to the group; every value is a campaign key)

| Choice | Proposal | Basis |
|---|---|---|
| Redshift | a0.400 (z = 1.5) | HLSP snapshot; already used in v1 |
| Sources | v1 list minus vela09 (11 ids) | doubles rejected by eye |
| Source crop / recenter | 1.5" radius, recenter on flux centroid | companions/outskirts at 1–2" drove the v1 cut-offs |
| PSF | STScI ePSF F814W SM4, chip centre, 33 px | measured FWHM 0.081"; approximation recorded (native pixel response kept, no drizzle broadening) |
| Noise | instrument: sky 22.3, 2400 s, 4 reads, RN 4.5, dark 0.0075, native 0.05" | → 0.0041 cps/px; v1's 0.005 was ~1 orbit |
| Source brightness | `source_to_lens_flux_ratio: 0.5` in the cutout | user target; v1 sat at 0.05–0.13; sources scaled 4–10x (recorded as `source_ab_mag_unlensed`) |
| Cut-off | ≤1% lensed-source flux outside; border SB ≤ 1 sigma_bkg in outer 3 px; redraw ≤ 50 | border ≤ 1σ = no detectable truncation; 1% is a global guard |
| theta_E prior | LogNormal(median 1.1, σ 0.2) (was 1.25 / 0.25) | keeps arcs inside 6" at typical source offsets; rejection handles the tail |
| Everything else | notebook cell-4 baseline | unchanged; recorded in manifest |

Open decisions for the group: scale the source vs. dim the lens to reach 0.5 (v2 scales the
source, keeping the ~18.4 AB lens); 0.03" drizzled vs. 0.05" native pixels (the ePSF is exact for
native); the source list (vela09 kept in the review set so the group can judge it); `n_reps`.

**Open question (user, 2026-09-13): how to treat dim sources in general.** Forcing every source
to a fixed source-to-lens ratio implicitly folds in the selection effects of how lenses are
detected and chosen for modelling (bright arcs are what gets found and modelled). Whether to
mimic that selection, sample a brightness distribution, or reject compact/dim sources is a
population-design decision, parked until the user has talked to the group.

**Display standard (user, 2026-09-13):** `inferno` colormap with a square-root stretch floored
at 0 (`PowerNorm(gamma=0.5, vmin=0)`) for every image of these systems; `experiments/vela_plot_dataset.py` (moved from `vela_revised_v2/plot_dataset.py`) follows it.

## Design checkpoint — v2 review set generation (UNCERTIFIED, awaiting grader)

- **Hypothesis:** with calibration + cut-off + crop, every generated system has
  source/lens = 0.50 ± 1e-6 in the cutout and ≤ 1% lensed flux outside, without exceeding 50 redraws.
- **Prediction:** amplitudes 4–10x (source AB 21–23 unlensed), magnifications 3–15, 0–5 redraws
  per system with the theta_E prior at 1.1/0.2.
- **Falsifier:** any system needing > 10 redraws, or an accepted image whose observed border shows
  arc structure above 3σ (plot, not metric), means the thresholds/prior are mis-set, not "fine-tuning".
- **Check before use:** open `experiments/vela_revised_v2/dataset_gallery.png` — arcs inside the
  frame, arcs visibly ~half the lens brightness, PSF core narrower than v1.

## v2 review-set test (2026-09-13, 2 systems, CPU, seed 0) — UNCERTIFIED

`vela_revised_v2/campaign.yaml` with `vela_ids: ["02", "22"]`; STScI ePSF resampled to 0.03"
measures FWHM 0.084" (native 0.081"), EE(<0.25") = 0.95 (v1 Gaussian: 0.130", EE 1.00);
derived `background_rms` = 0.00412 cps/px at 2400 s.

| system | theta_E | amp | src/lens (cutout) | mu | outside | border | redraws | source AB unlensed | lens light (cps) |
|---|---|---|---|---|---|---|---|---|---|
| vela02 | 1.04 | 4.97 | 0.500 | 9.9 | 0.00% | 0.01σ | 0 | 22.06 | 705 |
| vela22 | 0.93 | 35.4 | 0.500 | 3.4 | 0.00% | 0.17σ | 0 | 20.01 | 1605 |

Gallery: `experiments/vela_revised_v2/dataset_gallery.png` (arcs inside the frame; ring +
clumps visible for vela02; compact arc + counter-image for vela22). Against the checkpoint
prediction (amp 4–10x): vela22 needs **35x** (a compact, low-magnification source behind a
bright lens) → its unlensed source is AB 20.0, brighter than a real z=1.5 galaxy. This is the
"scale the source vs. dim the lens" decision made concrete: the ratio target is met, but the
implied source luminosity is unphysical for compact sources unless the lens light (Ie prior) or
the ratio target is lowered, or high-|amp| systems are rejected (a `max_amp` knob would be a
one-line addition). Flagged for the group; not changed here.

Runtime note: the login node's shared A100 is visible to JAX; a run that landed on it died with
`CUDA_ERROR_OUT_OF_MEMORY`. Use `JAX_PLATFORMS=cpu` for review-set generation on the login
node (~minutes), or the GPU allocation for the full campaign.

## v2 review set (2026-09-13, 12 systems = v1 list incl. vela09, CPU, seed 0) — UNCERTIFIED

Dataset: `$PSCRATCH/gigalens/simtests_results/vela_revised_v2/dataset`. Figures:
`experiments/vela_revised_v2/dataset_grid.png` (observed, 4x3) and `dataset_gallery.png`.

| system | theta_E | amp | mu | outside | border | redraws | source AB unlensed | lens (cps) |
|---|---|---|---|---|---|---|---|---|
| vela02 | 1.04 | 4.97 | 9.9 | 0.00% | 0.01σ | 0 | 22.06 | 705 |
| vela03 | 1.16 | 11.0 | 11.1 | 0.00% | 0.01σ | 1 | 21.85 | 963 |
| vela04 | 1.19 | 25.6 | 10.2 | 0.10% | 0.23σ | 0 | 21.88 | 863 |
| vela07 | 1.35 | 4.58 | 4.1 | 0.01% | 0.35σ | 1 | 20.90 | 847 |
| vela08 | 1.05 | 9.56 | 4.8 | 0.00% | 0.00σ | 0 | 21.26 | 714 |
| vela09 | 1.12 | 2.37 | 4.6 | 0.00% | 0.01σ | 1 | 21.61 | 494 |
| vela10 | 0.83 | 31.8 | 4.9 | 0.00% | 0.00σ | 0 | 20.70 | 1230 |
| vela21 | 1.45 | 8.02 | 4.6 | 0.00% | 0.08σ | 2 | 20.52 | 1377 |
| vela22 | 1.07 | 10.1 | 5.8 | 0.00% | 0.00σ | 0 | 21.37 | 778 |
| vela23 | 0.91 | 31.4 | 3.5 | 0.00% | 0.00σ | 0 | 20.77 | 815 |
| vela25 | 1.46 | 7.46 | 6.8 | 0.09% | 0.36σ | 0 | 21.64 | 721 |
| vela26 | 1.21 | 1.97 | 7.6 | 0.00% | 0.01σ | 1 | 22.75 | 289 |

All ratio 0.500 (to 1e-6). Against the design checkpoint: redraws 0–2 (predicted 0–5, OK);
amplitudes 2.0–31.8 (predicted 4–10: **3 of 12 exceed**, the compact/low-mu sources vela04, vela10,
vela23 → unlensed AB 20.7–21.9); no accepted image shows arc structure at the border (checked on
the grid, sqrt stretch). Plot reading: vela09's double nucleus produces a ring plus three bright
knots (judge-able by eye, as the user wanted); extended sources (vela07, vela21, vela26) meet the
integrated-flux ratio but look lens-dominated because their arcs are spread thin — the ratio target
is an integrated-flux statement, not a surface-brightness one. (The 2-system test earlier drew a
different vela22 truth because the seed fold index follows list position.)

## Claims register

| claim | status | scope |
|---|---|---|
| v1 test set is lens-dominated (source/lens 0.05–0.13) with arcs cut off (up to 21% outside) | proposed | 3 re-rendered systems + border stats on all 12 |
| STScI ePSF resampled to 0.03" has FWHM ≈ 0.08–0.09" | proposed | unit test on synthetic ePSF + measured library PSF |
| v2 generator reproduces the requested flux ratio to 1e-6 | proposed | end-to-end test on a synthetic source (`vela_simulated_test.py`, 18 tests pass) + 2 real systems (ratio 0.500) |
| reaching source/lens = 0.5 by scaling the source requires 25–35x (unlensed AB ~20–22) for compact, low-mu sources | proposed | 3 of 12 review-set systems (vela04, vela10, vela23) + vela22 in the 2-system test |
| with crop 1.5" + theta_E LogNormal(1.1, 0.2) + cut-off, all 12 review systems keep ≥ 99.9% of lensed flux inside the 6" cutout with ≤ 2 redraws | proposed | 12-system review set, seed 0 |

## Group decisions (2026-09-14) → v3 configuration `experiments/vela_f140w_v3/`

Decisions relayed by the user after the group meeting (verbatim intent):
WFC3/IR **F140W**; drizzle to **0.065"/px** to match DESI Strong Lens Foundry V;
**HST PSF from Jay Anderson** (STDPSF library); **exposure 1200 s**; use the
**Foundry V photometry + magnifications to get the unlensed brightness**;
**120×120 px**; **θ_E LogNormal(median 1.5", σ 0.25)** (corrected from 1.25 mid-turn).

### What the papers actually contain (checked in the arXiv LaTeX sources)

* Foundry I (arXiv:2502.03455) `hst-observations.tex`: 3 × 399.23 s = **1197.7 s**,
  no CR-split, native 0.13" drizzled to **0.065"**. `photometry-single-arc.tex`
  Table 1: **F140W isophotal magnitude of the brightest source image** and the
  contour area, for all 51 systems. Contour areas are 0.02–0.14 arcsec² =
  **5–33 drizzled pixels**, i.e. the brightest few pixels of one image, not the
  arc flux. Aperture photometry of lenses and arcs was done ("preliminary",
  commented out) but never published.
* Foundry V (arXiv:2512.07823): **no photometry table**. Per system: mass +
  light parameters and **magnifications per image and total** (three methods;
  I use method 3 = median lenstronomy). Empirical PSF from field stars via
  `photutils.EPSFBuilder`, 27–33 px; cutouts 64–120 px.

### Derived numbers (Foundry I Table 1 × Foundry V magnifications), F140W AB

| system | z_d | z_s | θ_E | m_iso (brightest img) | contour px | peak SB [mag/"²] | μ(img) | μ(tot) | m_unlensed via μ(img) | via μ(tot) |
|---|---|---|---|---|---|---|---|---|---|---|
| J154.6972−01.3590 | 0.388 | 1.430 | 2.90 | 23.24 | 7 | 19.42 | 80.8 | 100.8 | 28.01 | 28.25 |
| J165.4754−06.0423 | 0.483 | 1.675 | 2.63 | 23.87 | 33 | 21.73 | 57.1 | 170.0 | 28.26 | 29.45 |
| J094.5639+50.3059 | 0.552 | 3.333 | 2.29 | 24.88 | 5 | 20.69 | 11.7 | 14.7 | 27.55 | 27.80 |
| J234.4783+14.7232 | 0.731 | 2.478 | 1.55 | 25.12 | 6 | 21.13 | 34.7 | 62.3 | 28.97 | 29.61 |
| J257.4348+31.9046 | 0.746 | 2.120 | 1.99 | 24.13 | 9 | 20.58 | 20.4 | 26.2 | 27.40 | 27.68 |
| J238.5690+04.7276 | 0.777 | 1.721 | 1.48 | 24.24 | 32 | 22.07 | 8.1 | 21.5 | 26.51 | 27.57 |
| J246.0062+01.4836 | 1.092 | 2.369 | 2.70 | 24.97 | 19 | 22.23 | 6.2 | 8.7 | 26.95 | 27.32 |

peak SB: mean **21.12**, sd 0.99, median contour **9 px**. m_unlensed via μ(img): mean **27.67**, sd 0.83.

**Interpretation (UNCERTIFIED, but arithmetic):** because the isophotal
magnitude covers only the brightest 5–33 px of one image, "m_iso + 2.5 log μ"
is a **lower limit on the source brightness** (the true arcs carry far more
flux than the contour). At the v3 noise level (1σ per 0.065" px = 24.8
mag/"²) a 27.7-mag source magnified ×10–30 gives arcs of total 24–25 mag
spread over hundreds of pixels — invisible. The real arcs are bright (their
peak pixels are 19.4–22.2 mag/"², i.e. 10–100σ per pixel). So the
magnification route cannot be executed from the published numbers; what the
published photometry *does* pin down is the **peak surface brightness of the
arcs**, which lensing conserves. Hence two calibration modes were added:

* `calibration: {peak_sb: {sb_mag_arcsec2, n_brightest_pix}}` — mean SB of the
  N brightest pixels of the PSF-convolved lensed source (noiseless, in the
  cutout) set to the target. **Used for the v3 review set** with
  Normal(21.1, 1.0) and N = 9.
* `calibration: {unlensed_ab_mag: m | {dist}}` — the group's route, for when
  real arc photometry is available (the Foundry team has the mosaics). A
  comparison set was generated with Normal(27.7, 0.85) to show what the
  published lower limits imply (`campaign_unlensed_mag.yaml`).

Both modes record `lens_ab_mag_cutout`, `source_ab_mag_lensed_cutout`,
`source_ab_mag_unlensed`, `peak_sb_mag_arcsec2` (+ `peak_sb_n_pix`), and the
sampled `calibration_target` per system, so any mode can be judged against the
Foundry numbers.

### Other v3 choices (generator_version 3)

* `delta_pix: 0.065` overrides the mock's TPIX (the WFC3/IR VELA mocks are
  0.06"); provenance recorded (`delta_pix_source`).
* PSF `STDPSF_WFC3IR_F140W.fits` (3×3 grid, 101² at 4× over 0.13" px; no era
  suffix in the WFC3 libraries — `era: null`), chip centre [507, 507], 33 px.
  Approximation as before: native 0.13" pixel response kept; no drizzle term.
* Noise: 1197.7 s, 3 ramps × 15 e⁻ effective read noise (ETC default;
  ASSUMPTION), dark 0.048 e⁻/s (IHB 5.7), sky **1.2 e⁻/s per native px =
  21.8 mag/"²** (ASSUMPTION: "typical" broad-band IR total; IHB 7.9 quotes
  0.3–1.0 for zodi alone) → background_rms **0.0196 cps per 0.065" px**.
  Sensitivity: σ ∝ √(sky t + dark t + 3 RN²) = √(1466 + 57 + 675) e⁻; halving
  the sky lowers σ by 17 %.
* Lens light `Ie`: **falsified prediction, corrected.** I first assumed Ie is
  cps per output pixel (the simulator average-pools) and set LogNormal(100, 0.3)
  expecting ≈18.9 AB; the first v3 run gave lens light **17.3–17.6 AB** in all
  12 systems (5× the v2 flux). Direct probe (`ie_probe.py`, one Sérsic R_e 1.6",
  n 4, Ie 20, PSF 0.1"): sum = 812 cps at 0.03"/200 px, 887 at 0.065"/120 px,
  1053 at 0.065"/260 px, 991 at 0.03"/400 px → the integrated flux is
  independent of delta_pix (differences are field of view); Ie acts as a surface
  brightness per arcsec². Regenerated with **LogNormal(30, 0.3)** → ≈18.7 AB in
  F140W for the baseline R_e/n prior (ASSUMPTION: LRG lens at z≈0.5–0.8; v2 used
  20 → ≈18.4 AB in F814W). The lens brightness is a group decision.
* Cut-off thresholds unchanged (1 % outside, 1σ border, canvas ×2 = 15.6").
* $HOME is at 99 % of its 40 GiB quota; the F140W tarballs (~0.5 GB per sim)
  and extracted sources live on `$PSCRATCH/gigalens/vela_{downloads,sources_pristine}`
  (`datadir` / `source_root` keys).

### Design checkpoint — v3 review set (UNCERTIFIED)

Cause hypothesis: with peak-SB calibration the arcs should look like the
Foundry cutouts (bright, ~30σ peak pixels) and the source/lens flux ratio will
*vary* (it is no longer imposed). Prediction: ratio spread ~0.1–1 across the 12
sources; unlensed source AB ≈ 23–25 (VELA F140W unlensed ≈ 22.9 for vela02 at
amp = 1, so amps ≈ 0.2–1 for compact sources). Falsifier: if amps land ≫ 10
again, the peak-SB target is too bright for these sources (or N too small).
Structural-vs-tuning: the calibration mode is structural; every number is
tuning and YAML-only.

### v3 review set results (2026-09-14, 12 systems, CPU, seed 0, Ie median 30) — UNCERTIFIED

> Lens column superseded: Ie median lowered to 20 later the same day (see the
> boundary-treatments section below); every other column is unchanged.

Figures: `experiments/vela_f140w_v3/dataset_{grid,gallery}.png`; comparison set in
`experiments/vela_f140w_v3/unlensed_mag/`. Datasets on
`$PSCRATCH/gigalens/simtests_results/vela_f140w_v3{,_unlensed_mag}/dataset`.
PSF: resampled ePSF FWHM 0.172", EE(<0.25") 0.80, 33 px. Noise: σ_bkg 0.0196 cps/px, ZP 26.453.

| system | θ_E | amp | src/lens | μ | outside | border | redraws | src AB unl | arcs AB | lens AB | peak SB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vela02 | 1.39 | 1.09 | 0.336 | 15.0 | 0.00% | 0.00σ | 0 | 22.93 | 19.98 | 18.80 | 20.52 |
| vela03 | 1.21 | 0.94 | 0.055 | 4.3 | 0.00% | 0.00σ | 0 | 22.60 | 21.03 | 17.88 | 21.08 |
| vela04 | 1.65 | 1.03 | 0.077 | 13.9 | 0.02% | 0.04σ | 0 | 24.21 | 21.35 | 18.58 | 20.92 |
| vela07 | 1.95 | 0.54 | 0.353 | 6.2 | 0.02% | 0.09σ | 1 | 21.70 | 19.73 | 18.60 | 20.75 |
| vela08 | 1.42 | 1.04 | 0.220 | 6.6 | 0.00% | 0.00σ | 0 | 22.50 | 20.45 | 18.80 | 21.04 |
| vela09 | 1.54 | 1.65 | 1.534 | 7.2 | 0.00% | 0.02σ | 1 | 20.85 | 18.72 | 19.18 | 19.48 |
| vela10 | 1.05 | 2.68 | 0.167 | 6.3 | 0.00% | 0.00σ | 0 | 22.14 | 20.15 | 18.21 | 19.97 |
| vela21 | 1.51 | 0.59 | 0.412 | 8.5 | 0.00% | 0.02σ | 1 | 21.99 | 19.66 | 18.70 | 20.55 |
| vela22 | 1.45 | 0.19 | 0.076 | 7.8 | 0.00% | 0.00σ | 0 | 23.72 | 21.50 | 18.70 | 21.82 |
| vela23 | 1.18 | 2.05 | 0.140 | 4.1 | 0.00% | 0.00σ | 0 | 22.28 | 20.75 | 18.61 | 20.39 |
| vela25 | 1.64 | 1.56 | 0.925 | 10.3 | 0.00% | 0.02σ | 1 | 21.53 | 18.99 | 18.91 | 21.48 |
| vela26 | 1.79 | 0.23 | 0.153 | 14.0 | 0.01% | 0.04σ | 0 | 23.78 | 20.91 | 18.87 | 21.78 |

Checkpoint verdict: prediction held — amps 0.19–2.68 (median 1.03; the VELA
galaxies at their native F140W brightness), ratio spread 0.05–1.53 (median
0.19), lens 17.9–19.2 AB (median 18.7). Plots inspected before the table: all
arcs inside the frame; vela09 (ring + knots, ratio 1.5) and vela25 (bright
extended ring, 0.9) dominate their lens; vela03/22/26 are the faint end
(0.05–0.15), visible but low contrast.

**Comparison set** (`campaign_unlensed_mag.yaml`, unlensed AB ~ Normal(27.7,
0.85), same seeds): amps 0.002–0.05, src/lens 0.001–0.010, arcs 24.2–26.1 AB
total, peak SB 24.2–27.1 mag/"² vs 24.8 per-pixel noise → **no arc visible in
any panel** (inspected). Confirms that m_iso + 2.5 log μ from the published
tables is a lower limit, not a brightness. Note the cut-off check is
brightness-dependent: with invisible arcs vela07/21/25 accepted draws with
θ_E 2.1–2.5" that the peak-SB set had rejected (border SB below 1σ trivially).

### Claims register additions

| claim | status | evidence |
|---|---|---|
| Foundry V has no arc photometry; Foundry I Table 1 is brightest-image isophotal over 5–33 px | VERIFIED | arXiv LaTeX sources, `photometry-single-arc.tex`, per-system `desi*.tex` |
| m_iso + 2.5 log μ = 26.5–29.0 AB is a lower limit on source brightness; such sources are invisible at the v3 noise | VERIFIED (simulation) | comparison set, 12/12 panels without arcs |
| Sérsic `Ie` acts as SB per arcsec²: integrated flux independent of delta_pix | VERIFIED | `ie_probe.py` sums 812/887/1053/991 cps (FOV-limited) |
| peak-SB calibration reproduces the Foundry peak SB distribution with amps ≈ 1 | VERIFIED (by construction) + plots | table above |
| sky 21.8 mag/"², RN 3×15 e⁻, lens 18.7 AB | ASSUMPTIONS for the group | — |

## Source-plane boundary artefacts and the lens brightness check (2026-09-14, user question)

User: the 1.5" circular crop is visible in the lensed images; removing it would
move the edge to the VELA cutout boundary; proposes rejecting sources with
significant flux near the source-plane boundary. Peak-SB calibration approved.

**Measured (`edge_sb.py`, v3 amps; 1σ per 0.065" px = 4.64 cps/"² = 24.79 mag/"²;
lensing conserves SB so a source-plane SB in σ units is its image-plane
visibility before PSF blur):**

| source | amp | VELA cutout | flux > 1.5" | SB at 1.5" mean / max | SB at cutout edge mean / max | flux in outer 0.3" |
|---|---|---|---|---|---|---|
| vela02 | 1.09 | 5.8" | 6.8% | 0.05σ / 39σ | 0.00σ / 0.96σ | 0.05% |
| vela03 | 0.94 | 5.8" | 9.4% | 0.06σ / 83σ | 0.02σ / 78σ | 1.98% |
| vela04 | 1.03 | 5.8" | 5.7% | 0.01σ / 1.6σ | 0.00σ / 1.8σ | 1.81% |
| vela07 | 0.54 | 11.6" | 38.5% | 1.40σ / 117σ | 0.00σ / 1.3σ | 0.26% |
| vela08 | 1.04 | 5.8" | 4.3% | 0.05σ / 5.3σ | 0.00σ / 1.6σ | 0.06% |
| vela09 | 1.65 | 5.8" | 17.7% | 0.88σ / 115σ | 0.12σ / 21σ | 2.03% |
| vela10 | 2.68 | 5.8" | 15.6% | 0.03σ / 5.5σ | 0.08σ / 11σ | 10.9% |
| vela21 | 0.59 | 11.6" | 16.7% | 0.63σ / 13σ | 0.00σ / 2.0σ | 0.19% |
| vela22 | 0.19 | 5.8" | 2.6% | 0.01σ / 0.28σ | 0.00σ / 0.22σ | 0.34% |
| vela23 | 2.05 | 5.8" | 1.1% | 0.02σ / 1.5σ | 0.00σ / 0.92σ | 0.14% |
| vela25 | 1.56 | 5.8" | 12.6% | 0.51σ / 91σ | 0.02σ / 79σ | 0.92% |
| vela26 | 0.23 | 11.6" | 2.8% | 0.01σ / 0.30σ | 0.00σ / 0.06σ | 0.03% |

Reading: the crop circle's *mean* SB is below the noise everywhere, but it
cuts through compact clumps/companions (max 13–117σ) in six sources — the
visible artefact. At the VELA frame edge four sources (03, 09, 10, 25) have
bright objects touching the boundary; eight are ≤ 2σ there. So (i) a
"boundary SB below noise" criterion (the user's proposal, made quantitative)
would reject 03/09/10/25 without any crop; (ii) a hard circle at a fixed
radius is the wrong tool for the others because it is the max, not the mean,
that matters; a smooth taper still slices clumps. Recommendation given to the
user: measure boundary SB in σ units (recorded), reject above a threshold,
and where cropping is needed use a *segmentation-based* mask (keep the
connected component of the main galaxy above an SB threshold, dilated and
tapered) instead of a circle, so no edge crosses a clump. Not implemented
(assessment only). Gallery now has a 4th column: lensed source only with an
independent realisation of the same noise model (`add_noise` in the plot script).

**Lens brightness vs Foundry V fits (`foundry_lens_mag.py`, same
`SersicEllipse` profile, 0.065"/px, Gaussian 0.17" PSF; only J234 and J246
publish I_e):** J234 (z_d 0.731): 19.25 AB in 120 px (19.49 in 64 px, 18.92
total; comps 20.34 + 19.74). J246 (z_d 1.092): 19.04 AB in 120 px (19.29 in
64 px, 18.67 total). Ours: 17.9–19.2, median 18.7 in 120 px → ~0.5 mag
brighter than the two highest-z_d Foundry lenses; lower-z_d lenses would be
brighter. Caveats: conventions (I_e per arcsec², ε definition) assumed
identical to the paper's GIGA-Lens version; two systems only. Ie median 20
would put ours at ≈19.1.

## Ie retune and source-boundary treatments in the image plane (2026-09-14, user follow-up)

User: lower the Ie median to 20 (yes); is rejecting on the σ of objects touching
the boundary the best way? In vela21, where the circle is most visible, everything
near the boundary is below 2σ.

**Ie retune.** `campaign.yaml` and `campaign_unlensed_mag.yaml`: Ie LogNormal
median 30 → 20 (comment updated). Both sets regenerated with `--force` (30 s
each, sources cached). Prediction: lens light −0.44 mag, everything else
unchanged (Ie has its own key in the joint prior; calibration is lens-independent).
Held: θ_E, amps, peak SB, redraws identical; lens AB now 18.32–19.62
(median 19.14; was 17.9–19.2, median 18.7); src/lens 0.08–2.30
(median 0.29). Per system: vela02 19.24, vela03 18.32, vela04 19.02, vela07 19.04, vela08 19.24, vela09 19.62, vela10 18.65, vela21 19.14, vela22 19.14, vela23 19.05, vela25 19.35, vela26 19.31. Figures re-rendered
(`dataset_{grid,gallery}.png` in both dirs).

**Boundary treatments (`experiments/vela_f140w_v3/boundary_treatments.py`,
`boundary_edges.py`, `boundary_fig.py` → `boundary_treatments.png`).**
Hypothesis: the circle in vela21 is visible not because any pixel is significant
but because a ~1σ edge is coherent over thousands of pixels; predicted per-pixel
peak 1–3σ and integrated S/N of the removed light 20–50; falsifier: integrated
S/N < 5 (then the visibility would be a display effect of the noiseless column).
Method: each system re-lensed through its own truth (same amp, PSF) with three
sources — hard 1.5" crop (as the set), raised-cosine taper 1 inside 1.2" / 0
beyond 1.8" (½ at 1.5"), uncropped VELA frame — plus the lensed outer 0.15" band
of the VELA frame. Removed light = full − treated, in units of the per-pixel
noise σ = sqrt(σ_bkg² + (lens + source)/t). The hard-crop render reproduces
`noiseless − lens_light_only` to ≤ 1e-6 (geometry certified). Edge statistic:
high-pass (img − G(2 px)) S/N over the lensed image of the 1.4–1.6" ring.

| source | flux removed (hard) | hard peak/px | hard S/N | taper peak/px | taper S/N | frame-edge peak/px | frame-edge S/N | edge S/N hard / taper / none | src S/N |
|---|---|---|---|---|---|---|---|---|---|
| vela02 | 6.8% | 2.74σ | 13.4 | 2.36σ | 12.0 | 0.00σ | 0.0 | 1.6 / 1.3 / 3.3 | 360 |
| vela03 | 9.3% | 4.68σ | 14.5 | 4.68σ | 14.4 | 0.31σ | 1.4 | 1.2 / 1.1 / 1.2 | 187 |
| vela04 | 5.7% | 0.12σ | 0.6 | 0.11σ | 0.6 | 0.00σ | 0.0 | 0.1 / 0.1 / 0.2 | 157 |
| vela07 | 38.6% | 9.15σ | 56.4 | 8.84σ | 54.9 | 0.00σ | 0.0 | 14.4 / 11.9 / 18.4 | 338 |
| vela08 | 4.3% | 0.97σ | 4.8 | 0.97σ | 4.8 | 0.00σ | 0.0 | 2.9 / 2.9 / 3.0 | 254 |
| vela09 | 17.9% | 4.06σ | 47.2 | 3.46σ | 45.1 | 0.28σ | 3.2 | 14.5 / 14.3 / 15.6 | 844 |
| vela10 | 15.5% | 11.35σ | 41.2 | 11.35σ | 41.2 | 2.04σ | 7.5 | 1.6 / 1.6 / 1.6 | 369 |
| vela21 | 16.8% | 1.40σ | 16.2 | 1.15σ | 16.2 | 0.00σ | 0.0 | 5.2 / 4.2 / 5.1 | 396 |
| vela22 | 2.6% | 0.01σ | 0.3 | 0.01σ | 0.3 | 0.00σ | 0.0 | 0.3 / 0.3 / 0.3 | 120 |
| vela23 | 1.1% | 0.05σ | 0.5 | 0.05σ | 0.5 | 0.00σ | 0.0 | 5.4 / 5.4 / 5.4 | 258 |
| vela25 | 12.6% | 1.01σ | 19.5 | 0.86σ | 19.1 | 0.09σ | 0.2 | 4.4 / 3.5 / 3.9 | 564 |
| vela26 | 2.8% | 0.04σ | 0.4 | 0.05σ | 0.5 | 0.00σ | 0.0 | 1.0 / 1.0 / 1.0 | 139 |

Result: hypothesis held in direction, magnitude at the low end (vela21: peak
1.40σ/px, S/N 16.2; vela25: 1.01σ/px, S/N 19.5; vela07 9σ/px, S/N 56). The
user is right that nothing at vela21's boundary exceeds 2σ per pixel, and that
is exactly why a per-pixel criterion cannot be the right one. Plots (figure):
hard crop → sharp outer ring edge in vela21/25/07 and a cut companion in
vela09; taper → soft outskirt, indistinguishable by eye from the uncropped
frame; uncropped → same, plus companions (vela02 blob, vela09 second galaxy).
The taper removes the same light (S/N within 10% of the hard crop) and only
removes the discontinuity. The high-pass edge statistic FAILED to discriminate
(hard ≈ taper ≈ none, e.g. vela21 5.2/4.2/5.1): after the 0.17" PSF a ~1σ step
has no high-pass power above the real structure. So the evidence for the taper
is the figure, not a number — recorded as such.

Conclusions: (i) rejection by boundary σ (per pixel) is not the right
criterion; the integrated S/N of the light a boundary removes is what predicts
visibility. (ii) By that criterion no rejection is needed for the crop at all:
replace the hard edge with the taper. (iii) The VELA frame edge does not bite
here: with the taper at zero by 1.8" no source reaches its frame (half-width
≥ 2.9"); even uncropped only vela10 (S/N 7.5) and vela09 (3.2) lens frame-edge
light into the cutout. (iv) Segmentation masks are no longer recommended: a
faded clump is a smooth object; the earlier "max SB on the circle" numbers
were per source pixel and overstated the per-image-pixel contrast.

Implemented (opt-in, default unchanged): `source_crop_taper_arcsec` (width w;
weight 1 inside r−w/2, ½ at r, 0 beyond r+w/2; `crop_weight()`), recorded in the
manifest; `test_preprocess_source_taper`. The review set is still the hard crop;
adopting the taper is `source_crop_taper_arcsec: 0.6` in the YAML plus a
regeneration.

Claims register additions:

| claim | status | evidence |
|---|---|---|
| Ie median 20 → lens 18.3–19.6 AB (median 19.1), all other draws unchanged | VERIFIED | regenerated manifests, per-system list above |
| vela21's crop edge is < 2σ per pixel everywhere but S/N 16 integrated; per-pixel boundary σ does not predict visibility | VERIFIED (simulation) | table above, figure |
| a raised-cosine taper 1.2–1.8" removes the visible edge | VERIFIED by inspection only (edge statistic non-discriminating) | `boundary_treatments.png` |
| frame-edge light is negligible in the cutout for this set (≤ S/N 7.5 uncropped, 0 with taper) | VERIFIED (simulation) | table above |

### No crop at all? (2026-09-15, user question; `experiments/vela_f140w_v3/nocrop_check.py`)

User: the VELA frame boundary itself cuts off little; could it replace the taper?
Measured on the 12 sources (v3 amps) and the current truths, uncropped source
through the 2× canvas with the existing cut-off thresholds (outside ≤ 1%, border < 1σ):

| src | frame | flux in outer 0.15" / 0.3" | max SB at frame edge per 0.065" px | uncropped: outside / border / passes cut-off |
|---|---|---|---|---|
| vela02 | 5.8" | 0.02% / 0.05% | 0.0σ | 0.04% / 0.07σ / yes |
| vela03 | 5.8" | 0.79% / 1.98% | 1.4σ | 0.26% / 0.64σ / yes |
| vela04 | 5.8" | 0.71% / 1.81% | 0.3σ | 1.00% / 0.06σ / marginal |
| vela07 | 11.6" | 0.10% / 0.26% | 0.3σ | 4.80% / 2.86σ / **NO** |
| vela08 | 5.8" | 0.02% / 0.06% | 0.0σ | 0.00% / 0.00σ / yes |
| vela09 | 5.8" | 0.92% / 2.03% | 1.0σ | 0.22% / 0.50σ / yes |
| vela10 | 5.8" | 2.52% / 10.86% | 7.4σ | 0.03% / 0.06σ / yes (truncated companion enters the cutout at S/N 7.5) |
| vela21 | 11.6" | 0.15% / 0.19% | 0.9σ | 0.41% / 0.91σ / marginal |
| vela22 | 5.8" | 0.19% / 0.34% | 0.1σ | 0.04% / 0.01σ / yes |
| vela23 | 5.8" | 0.05% / 0.14% | 0.0σ | 0.01% / 0.06σ / yes |
| vela25 | 5.8" | 0.41% / 0.92% | 1.4σ | 0.52% / 0.96σ / marginal |
| vela26 | 11.6" | 0.02% / 0.03% | 0.0σ | 0.27% / 0.04σ / yes |

Correction to the 2026-09-14 edge table: its "max SB at the cutout edge" (up
to 79σ) was per 0.0073" source pixel; binned to the 0.065" image pixel the
frame-edge contrast is ≤ 1.4σ for every source except vela10 (7.4σ, a
companion cut by the frame). So the user is right that the VELA boundary cuts
off little: ≤ 1% of the flux in the outer 0.15" for 11/12. Without any crop the
cut-off check becomes the only thing keeping arcs inside the 120 px frame:
vela07 fails for its current truth (4.8% outside, 2.9σ border; its 11.6" frame
holds outskirts that lens beyond 3.9"), and vela04/21/25 sit at the thresholds
→ higher redraw rate, i.e. a selection toward smaller θ_E for the extended
sources, which the 1.5" taper avoids. Assessment only; nothing changed.


## No-crop review set (2026-09-15, user decision) — UNCERTIFIED

User: keeping arcs inside the 120 px frame is not a reason to modify the source
morphology (that is what the systems test); keep VELA sources as delivered;
drop vela10; a higher redraw rate is fine; show what happens with no crop.

Config: `source_crop_radius_arcsec: null`, vela10 removed, everything else as
before (both YAMLs). Prediction: no visible edges; redraw rate up for the
extended sources; vela07 impossible at outside <= 1% (12% of its flux beyond 3").
Result: vela07 rejected 51/51 (last: theta_E 1.32, 4.3% outside, 1.3 sigma border) ->
excluded from the main YAML and run alone in `campaign_vela07_nocrop.yaml`
(max_flux_outside 0.10, border rule unchanged): accepted on draw 2 with
4.8% outside / 0.68 sigma border (first draw theta_E 1.39, 5.9%, 1.6 sigma);
a full clumpy Einstein ring, theta_E 1.54, src/lens 0.58, mu 3.5.
Main set, 10 sources, 8 redraws total (vela21: 7 — its rejections were 0.2–8.7%
outside / 0.7–4.2 sigma border at theta_E 1.54–2.74; vela08: 1):

| system | θ_E | amp | src/lens | μ | outside | border | redraws | src AB unl | arcs AB | lens AB | peak SB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vela02 | 1.39 | 1.63 | 0.540 | 10.0 | 0.04% | 0.10σ | 0 | 22.41 | 19.91 | 19.24 | 20.52 |
| vela03 | 1.21 | 0.94 | 0.088 | 4.1 | 0.26% | 0.63σ | 0 | 22.49 | 20.96 | 18.32 | 21.08 |
| vela04 | 1.65 | 1.14 | 0.099 | 10.1 | 1.00% | 0.06σ | 0 | 24.03 | 21.53 | 19.02 | 20.92 |
| vela08 | 1.95 | 1.32 | 0.471 | 8.6 | 0.18% | 0.56σ | 1 | 22.19 | 19.85 | 19.04 | 20.75 |
| vela09 | 1.42 | 0.39 | 0.447 | 6.8 | 0.02% | 0.08σ | 0 | 22.20 | 20.12 | 19.24 | 21.04 |
| vela21 | 1.45 | 0.59 | 0.225 | 3.4 | 0.30% | 0.32σ | 7 | 21.78 | 20.44 | 18.82 | 21.04 |
| vela22 | 1.05 | 1.15 | 0.362 | 6.2 | 0.01% | 0.02σ | 0 | 21.73 | 19.75 | 18.65 | 19.97 |
| vela23 | 2.22 | 0.50 | 0.384 | 29.2 | 0.08% | 0.02σ | 0 | 23.79 | 20.13 | 19.09 | 21.46 |
| vela25 | 1.45 | 1.23 | 0.540 | 5.4 | 0.12% | 0.16σ | 0 | 21.64 | 19.81 | 19.14 | 21.82 |
| vela26 | 1.18 | 1.55 | 0.298 | 3.4 | 0.16% | 0.02σ | 0 | 21.68 | 20.37 | 19.05 | 20.39 |

Plots inspected before the table (`dataset_{grid,gallery}.png`,
`vela07_nocrop/`): no source-plane edge anywhere; vela02's companion lenses to
a separate blob, vela09's second nucleus is a knot on the ring, vela25/26 show
broad diffuse haloes; all arcs inside the frame (border <= 0.63 sigma, outside <=
1.0%). Caveats: (i) removing sources shifts `seed_fold_index`, so the lens draws
differ from the 12-source set (vela23 now theta_E 2.22", mu 29 — a giant ring);
folding by source id would make sets comparable and is a small generator
change if wanted. (ii) vela03's galaxy sits (1.1, 1.7)" off its VELA frame
centre; recentering brings the frame edge to ~1.2" on one side — no edge is
visible (border 0.63 sigma, the highest of the set), but it is the source to watch
with other lens draws. (iii) The `source_crop_taper_arcsec` knob stays
available but unused. Gallery source panel now shows the 5.8" half-frame and
its title no longer says "cropped".

Claims register additions:

| claim | status | evidence |
|---|---|---|
| uncropped sources show no source-plane edge in the lensed images (10 sources) | VERIFIED by inspection | `dataset_gallery.png` |
| vela07 cannot satisfy outside <= 1% uncropped at any theta_E in the prior | VERIFIED (51/51 rejected; 12% of flux beyond 3") | generation log, radial table above |
| no-crop redraw rate: 8 redraws / 10 systems (vela21 7) | MEASURED (n=1 rep) | generation.json |


### vela07 admitted, outside limit 10% for the set (2026-09-15, user decision)

User: keep vela07 ("we're just really losing clumps"); seeds need not match;
wants one clean artifact with the current configuration and the open questions.
Config: `max_flux_outside: 0.10` in both YAMLs (border rule < 1σ unchanged),
vela07 back in `vela_ids`; `campaign_vela07_nocrop.yaml` and its figures
removed. Both sets regenerated (`--force`), 11 systems, 5 redraws total
(vela07 4, vela09 1); outside ≤ 4.2% (vela07), border ≤ 0.63σ (vela03). vela07
drew a faint peak-SB target (22.20 mag/□", amp 0.08) → low-contrast ring; that
is the per-system draw working as designed. Plots inspected: no edges, all arcs
inside. Comparison set (unlensed-mag route) regenerated alongside, still no
visible arcs.

| system | θ_E | amp | src/lens | μ | outside | border | redraws | src AB unl | arcs AB | lens AB | peak SB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vela02 | 1.39 | 1.63 | 0.540 | 10.0 | 0.04% | 0.10σ | 0 | 22.41 | 19.91 | 19.24 | 20.52 |
| vela03 | 1.21 | 0.94 | 0.088 | 4.1 | 0.26% | 0.63σ | 0 | 22.49 | 20.96 | 18.32 | 21.08 |
| vela04 | 1.65 | 1.14 | 0.099 | 10.1 | 1.00% | 0.06σ | 0 | 24.03 | 21.53 | 19.02 | 20.92 |
| vela07 | 1.33 | 0.08 | 0.126 | 5.9 | 4.21% | 0.57σ | 4 | 23.21 | 21.28 | 19.03 | 22.20 |
| vela08 | 1.42 | 1.07 | 0.319 | 5.9 | 0.00% | 0.00σ | 0 | 22.42 | 20.48 | 19.24 | 21.04 |
| vela09 | 1.54 | 1.76 | 1.924 | 4.6 | 0.22% | 0.51σ | 1 | 20.57 | 18.91 | 19.62 | 19.48 |
| vela21 | 1.05 | 1.06 | 0.505 | 5.0 | 0.18% | 0.08σ | 0 | 21.15 | 19.39 | 18.65 | 19.97 |
| vela22 | 2.22 | 0.20 | 0.344 | 22.2 | 0.27% | 0.08σ | 0 | 23.61 | 20.25 | 19.09 | 21.46 |
| vela23 | 1.45 | 0.54 | 0.120 | 8.2 | 0.01% | 0.01σ | 0 | 23.72 | 21.44 | 19.14 | 21.82 |
| vela25 | 1.18 | 4.87 | 1.183 | 3.3 | 0.02% | 0.27σ | 0 | 20.15 | 18.87 | 19.05 | 20.39 |
| vela26 | 2.13 | 0.76 | 0.689 | 13.7 | 0.15% | 0.03σ | 0 | 22.45 | 19.61 | 19.20 | 20.36 |

New page "Vela F140W Lens Set" (configuration + basis tags + open questions,
no iteration history) replaces the design-review page as the group-facing
document; the old page stays as the record of the iterations.

## Config refinement: DESI-238 noise, supersample 32, tabled questions (2026-09-15, user answers)

User answers to the five open questions of the "Vela F140W Lens Set" page, plus
one new request (supersample 32 on the simulation "to be absolutely sure there
are no finite-resolution artifacts in my observed data"):

1. Sky / read noise: take them from the DESI-238 (Foundry V) data on disk; no
   drizzle modelling now, note it as a possible extension.
2. Lens light: no redshift dependence (the lens redshift is not set anywhere;
   it is implicit in theta_E). Keep the single LogNormal.
3. Sources / n_reps: tabled. First modelling passes use one or two simple
   systems picked by eye from the review set to work out each source method.
4. Per-system brightness draw: keep. Log the possibility that source structure
   and overall brightness are correlated in real arcs (the independent draw
   ignores it); tabled.
5. Cut-off thresholds / selection against large theta_E: fine for now; no
   change to the redraw procedure.

### DESI-238 noise (measured; `data/desi238/cutout238b.npy`)

There is no FITS file for DESI-238 anywhere under $HOME or $PSCRATCH (searched
by name and by extension); the data on disk are the 120 x 120 float32 cutout
`data/desi238/cutout238b.npy`, `psf94.npy` (27 px), and the notebook
`experiments/real_systems/desi238/MCLMC_238.ipynb`, which records
`background_rms = 0.007616` (photutils.background) and `exp_time = 1197.699`
(header). User confirmed "it might just be a numpy cutout".

| quantity | value | how |
|---|---|---|
| background rms, clipped, lens-light gradient removed (9 px median filter) | 0.0070 cps/px | r > 35/45/55 px, 3-sigma clipped; `tmp/cut238_noise.py` |
| photutils value used by the real-data fit | 0.0076 cps/px | notebook |
| noise autocorrelation lag (0,1) / (1,0) / (1,1) / (0,2) / (0,3) | 0.60 / 0.52 / 0.30 / 0.10 / -0.02 | residual after the median filter, r > 45 px, clipped mask |
| white-noise-equivalent sigma (sum of the ACF over lags ~ 4.8) | ~0.017 cps/px | 0.0076 x sqrt(4.8) |
| physical estimate (sky 21.8 + 3 x 15 e- RN + dark, `kind: instrument`) | 0.0196 cps/px | previous config |
| total flux in the 7.8" cutout | 18.5 AB | ZP 26.453 |

So the earlier sky/read-noise assumption was not far from the total variance;
the factor 2.6 between it and the per-pixel rms is the drizzle correlation.
Decision (user directive "reference the DESI-238 data"): `noise: {kind:
explicit, background_rms: 0.0076, exp_time: 1197.7}` in both YAMLs — the number
the real-data fit uses, so simulated and real per-pixel S/N match as the fitter
sees them. Caveat, UNCERTIFIED as to its size for inference: white noise at
0.0076 carries ~4.8x more information per area than the correlated real noise;
the drizzle extension (generate at 0.13" and drizzle, or colour the noise to
the measured ACF) is logged on the page as tabled question 4. The 9 px median
filter slightly biases the ACF and the clipped rms (0.0070 vs 0.0076); the
photutils value is used.

### Supersample 32 (probe; `experiments/vela_f140w_v3/supersample_check.{py,png,json}`)

Hypothesis: the VELA pristine sources have 0.0073" pixels, so the supersample-4
render grid (0.016" sub-pixels) under-samples them. Prediction: |img(4) -
img(32)| after PSF exceeds 0.1 sigma_238 along the arcs and is negligible in the
lens light. Falsifier: < 0.1 sigma everywhere -> 32 is insurance only.
Structural (a grid that cannot resolve the source), not fine-tuning.

vela02 and vela22 rendered with their stored truths at 4/8/16/32 on the 120 px
frame (the 4x render reproduces the stored noiseless image to 5e-7):

| system | (4 − 32) max / rms / N > 0.1σ | (8 − 32) max | (16 − 32) max | 32x: time, peak RSS |
|---|---|---|---|---|
| vela02 (clumpy) | 0.74σ / 0.050σ / 525 px | 0.29σ | 0.06σ | 5–8 s, 3.5 GB |
| vela22 (smooth) | 0.18σ / 0.008σ / 17 px | 0.51σ (one pixel at the lens centre) | 0.07σ | 5–9 s, 4.6 GB |

Plot inspected first: the vela02 residual is a speckle pattern that traces the
clumpy arcs (source-pixel aliasing, as predicted); vela22's arcs are fine at 4
but the 8x grid puts 0.5 sigma in the central lens pixel — the Sersic cusp
sampled at different sub-grid positions, the second finite-resolution effect.
Both are < 0.08 sigma at 16 vs 32. Prediction held for the clumpy source; the
cusp effect was not predicted. Flux differences are < 3e-4.

Implementation: `supersample` is now the truth render only; new
`inference_supersample` (default: same as `supersample`, so older configs are
unchanged) is what `System.supersample`/meta.json carry for the fitter. Both
YAMLs: `supersample: 32`, `inference_supersample: 4`, `cutoff.canvas_supersample:
8` (numerics of the check). Test `test_end_to_end_synthetic_source` asserts the
split. 22 tests pass; silent-defaults lint clean.

### Regenerated review set (2026-09-15, noise 0.0076, supersample 32) — UNCERTIFIED

Both sets regenerated (`--force`; ~10 min each on the login node CPU, ~5 GB).
Comparison set (unlensed-mag route): 11 systems, 0 redraws, still no visible
arcs. Main set: 11 systems, 14 redraws (vela07 11, vela09 2, vela03 1); outside
≤ 6.7% (vela07), border ≤ 0.97σ (vela09). Plots inspected first
(`dataset_{grid,gallery}.png`): arcs sharper at the lower noise, no
source-plane edge, all arcs inside the frame; vela07 is a lens with an
invisible source (see below).

| system | θ_E | amp | src/lens | μ | outside | border | redraws | src AB unl | arcs AB | lens AB | peak SB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vela02 | 1.39 | 1.63 | 0.542 | 10.0 | 0.04% | 0.25σ | 0 | 22.41 | 19.91 | 19.24 | 20.52 |
| vela03 | 1.61 | 0.50 | 0.177 | 9.6 | 0.05% | 0.14σ | 1 | 23.19 | 20.73 | 18.85 | 21.72 |
| vela04 | 1.65 | 1.14 | 0.099 | 10.1 | 1.01% | 0.16σ | 0 | 24.03 | 21.53 | 19.02 | 20.92 |
| vela07 | 1.71 | 0.02 | 0.024 | 4.5 | 6.69% | 0.25σ | 11 | 24.79 | 23.15 | 19.10 | 24.19 |
| vela08 | 1.42 | 1.09 | 0.323 | 5.9 | 0.00% | 0.01σ | 0 | 22.40 | 20.47 | 19.24 | 21.04 |
| vela09 | 1.68 | 0.42 | 0.416 | 5.9 | 1.91% | 0.97σ | 2 | 22.12 | 20.20 | 19.25 | 20.83 |
| vela21 | 1.05 | 1.06 | 0.505 | 5.0 | 0.18% | 0.21σ | 0 | 21.15 | 19.39 | 18.65 | 19.97 |
| vela22 | 2.22 | 0.20 | 0.343 | 22.2 | 0.27% | 0.21σ | 0 | 23.61 | 20.25 | 19.09 | 21.46 |
| vela23 | 1.45 | 0.54 | 0.120 | 8.2 | 0.01% | 0.01σ | 0 | 23.72 | 21.44 | 19.14 | 21.82 |
| vela25 | 1.18 | 5.02 | 1.219 | 3.2 | 0.02% | 0.40σ | 0 | 20.12 | 18.84 | 19.05 | 20.39 |
| vela26 | 2.13 | 0.76 | 0.689 | 13.7 | 0.15% | 0.07σ | 0 | 22.45 | 19.61 | 19.20 | 20.36 |

Same seeds as the previous set, so the systems with 0 redraws have the same
truths; only the border rule changed (σ_bkg 0.0196 -> 0.0076 makes "border <
1σ" 2.6x stricter), which is why vela03/07/09 moved.

**Finding — the border rule now selects on brightness, not only on θ_E.**
vela07's 11 rejections were all on the border rule (1.4σ–25σ at θ_E
1.31–2.47); the accepted draw has θ_E 1.71, the same as rejected attempt 2
(5.0σ), and passes at 0.25σ only because its peak-SB target landed at 24.19
mag/□", a 3.1σ tail of Normal(21.1, 1.0) (20x fainter = 3.3 mag, matching the
5.0σ -> 0.25σ ratio). For an extended source the loop therefore keeps waiting
for a faint draw. vela07 is a lens-only image at src/lens 0.02. UNCERTIFIED as
a general statement (one source, n=1 rep); the rejection records do not store
the peak-SB target (a diagnostic-only generator change if wanted). Not acted
on: the user has tabled the cut-off procedure and will start with one or two
simple systems; the page's tabled question 3 now carries this. Options when it
is revisited: a larger frame for extended sources, a border rule in absolute
SB rather than σ, or dropping vela07 after all.

Claims register additions:

| claim | status | evidence |
|---|---|---|
| DESI-238 per-pixel background rms 0.0070–0.0076 cps/px; lag-1 noise autocorrelation 0.5–0.6 | MEASURED (one cutout) | `tmp/cut238_noise.py`, notebook value |
| supersample 4 leaves up to 0.74 σ_238 of source-pixel aliasing on clumpy arcs; 16 vs 32 < 0.08σ | MEASURED (vela02, vela22) | `supersample_check.{png,json}` |
| regenerated set: no edges, all arcs inside, 14 redraws / 11 systems | VERIFIED by inspection / MEASURED | `dataset_grid.png`, generation.json |
| border rule at σ 0.0076 selects faint draws for extended sources (vela07 accepted at a 3.1σ-faint target) | UNCERTIFIED (n=1) | vela07 generation.json rejections |

### vela07 dropped (2026-09-15, user decision)

User: "Let's just drop vela07 for now" — no regeneration needed yet, log it.
Both YAMLs: `vela_ids` without "07" (10 sources) with the reason in a comment.
The datasets on scratch (`vela_f140w_v3{,_unlensed_mag}/dataset`) still hold
the 11-system set generated at 16:00 (vela07 included) until the next
`generate --force`; because seeds fold by list index, that regeneration will
change the lens draws for every source after vela04 (user: acceptable). A
stale `vela10_cam12_a0.400_rep00` directory from the 12-source run also sits in
the main dataset's `systems/` folder outside the manifest (not removed).

### Is supersample 32 enough? Lens-cusp ladder (2026-09-15, user question)

User: "I'd like to make sure we don't need higher than supersample=32."

Step 1 (`supersample_check_64.{png,json}`, vela02/vela22/vela08 at 16/32/64):
sources converged (vela02 |32−64| 0.008σ max), but the lens centre is not:
vela08 (n 5.78) |32−64| 0.44σ in 12 px, |16−64| 6σ; vela22 (n 3.97) 0.05σ. The
residual is one PSF-shaped spot at the lens centre. Cause: gigalens evaluates
the pure Sérsic at sub-pixel centres (no core softening); for n≈6 the profile
has a ~1e-6" spike at the centre and a steep core, so the central pixel's
midpoint-rule value depends on where the nearest sample lands. The frame
origin is a pixel corner (LensWCS: pixel centres at (j − 59.5)δ) and lens
centres are drawn within ±0.01" of it, so the spike sits at the shared corner
of the four central pixels.

Step 2 (`cusp_ladder.{py,png,json}`): exact per-pixel reference (midpoint at
4096², self-check vs 2048² ≤ 0.004σ) on the central 8×8 px, with gigalens'
own `SersicEllipse.light` on the LensWCS sub-grid; error patch convolved with
the system PSF; σ = 0.0076. Max |error| after PSF, in σ:

| lens | n | R_e | s=4 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| vela08 | 5.78 | 1.35 | 6.5 | 5.9 | 0.48 | 0.042 | 0.042 | 0.002 | 0.002 |
| vela09 | 5.30 | 1.36 | 1.6 | 0.68 | 0.23 | 0.030 | 0.022 | 0.002 | 0.003 |
| vela23 | 4.26 | 1.46 | 0.68 | 2.2 | 0.083 | 0.020 | 0.003 | 0.000 | 0.001 |
| vela22 | 3.97 | 1.64 | 0.16 | 0.13 | 0.056 | 0.003 | 0.004 | 0.000 | 0.000 |
| n=6, R_e 1", centre at corner | 6 | 1.0 | 11.6 | 1.5 | 0.51 | 0.16 | 0.050 | 0.015 | 0.004 |
| n=6, R_e 1", centre at pixel centre | 6 | 1.0 | 16.2 | 2.2 | 0.72 | 0.23 | 0.071 | 0.021 | 0.006 |

Reading: 32 is converged for the VELA sources (0.0073" pixels) but leaves
0.06–0.5σ at the lens centre for the current set and 0.5–0.7σ for an n=6 lens
(the prior is n ~ U(1, 6)); 64 brings the current set under 0.05σ but an n=6
lens only to 0.2σ; 128 → 0.07σ; 256 → 0.02σ. Below 32 the error is erratic
(vela23 is worse at 16 than at 4): it is the placement of the nearest sample,
not a smooth quadrature error. Above 64 the n=6 case falls ~3.3× per doubling.
The fitter's own ss=4 model has 1.6–16σ of central error against the exact
lens light; that is an inference-side choice (`inference_supersample`) and
not the subject here, but it is the same cusp.

Cost of brute force (measured, CPU, 120 px, per render): 32 → 5–9 s, 3.5–4.6
GB; 64 → 22–25 s, 10–18 GB; 128 → ~100 s, ~40–70 GB (extrapolated); 256 is
impractical. Cheap alternative: keep 32 everywhere and replace the central
8×8 px of the *unconvolved* lens light by the exact patch (1024² midpoints:
≤ 0.006σ even for n=6; ~2 s), then add PSF ⊛ (exact − rendered) to the image
(linear, PSF-convention-consistent for bin-first, and the patch is 50 px from
any edge). Not implemented — awaiting the user's choice. Config stays at 32.

Claims register additions:

| claim | status | evidence |
|---|---|---|
| supersample 32 converges the VELA source render (|32−64| ≤ 0.008σ on vela02) | MEASURED | `supersample_check_64.json` |
| lens-cusp error after PSF at 32: 0.06–0.48σ (current set), 0.5–0.7σ (n=6, R_e 1"); 64: ≤ 0.04σ / 0.16–0.23σ; 128: ≤ 0.07σ all | MEASURED vs exact reference (self-check ≤ 0.004σ) | `cusp_ladder.json` |
| below 32 the cusp error is erratic in s (sample placement), not monotone | MEASURED | `cusp_ladder.png` |

### Correction: the cusp error against the TOTAL noise (2026-09-15, user question)

User: does "in units of the DESI-238 noise" include the Poisson component? It
did not — the ladder above is in units of the background rms alone (0.0076
cps/px). At the lens centre the pixels are 3.8–6.0 cps/px, so the Poisson term
(sqrt(img/exp_time) = 0.057–0.071) makes the total per-pixel σ 7.5–9.3x the
background rms there. Re-run of `cusp_ladder.py` with σ_tot² = bkg² +
noiseless/exp_time per pixel (stored noiseless image for the real systems; a
full-frame ss=32 lens render + PSF for the synthetic ones). Max |error| after
PSF in units of σ_tot:

| lens | n | s=4 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| vela08 | 5.78 | 0.70 | 0.64 | 0.051 | 0.005 | 0.004 |
| vela09 | 5.30 | 0.22 | 0.078 | 0.026 | 0.003 | 0.002 |
| vela23 | 4.26 | 0.089 | 0.27 | 0.010 | 0.002 | 0.000 |
| vela22 | 3.97 | 0.026 | 0.018 | 0.007 | 0.000 | 0.000 |
| n=6, R_e 1", corner | 6 | 1.5 | 0.20 | 0.066 | 0.021 | 0.006 |
| n=6, R_e 1", pixel centre | 6 | 2.0 | 0.26 | 0.088 | 0.028 | 0.009 |

Conclusion (supersedes the reading above): **32 is adequate**. Against the
noise the pixel actually carries, the lens-centre error at 32 is ≤ 0.05σ for
the current lenses and ≤ 0.09σ for an n=6, R_e=1" lens anywhere in the prior;
the χ² damage (Σ(err/σ_tot)²) at 32 is < 0.01 for every case. 64 is ≤ 0.03σ.
The exact-central-patch route stays a possible refinement, not a need. The
background-only numbers remain the right ones for the ARCS (their pixels are
near the background), and there 32 was already converged (0.008σ).
Inference-side note: at the fitter's ss=4 the lens-centre model error is
0.7σ_tot (vela08) to 2σ_tot (n=6) — a reason to raise `inference_supersample`
or use the adaptive simulator when modelling high-n lenses; not a truth-data
issue.

| claim | status | evidence |
|---|---|---|
| lens-cusp error at supersample 32, against total per-pixel noise: ≤ 0.05σ (current lenses), ≤ 0.09σ (n=6, R_e 1"); χ² damage < 0.01 | MEASURED vs exact reference | `cusp_ladder.json` (`postpsf_max_sigma_total`, `postpsf_chi2`) |

## Adversarial review of the simulation procedure (2026-09-17, Opus subagent, user request)

Brief: an honest referee looking for reasons the set is not a representative or
good test of *source structure*; tabled questions off-limits unless they hide
something worse. Full report with scripts, figures and JSON preserved in
`experiments/vela_f140w_v3/review/adversarial_review.md` (measured [M] /
inferred [I] / recalled-uncertified [R] are marked there). Ranked findings:

| # | severity | finding | key number |
|---|---|---|---|
| 1 | BLOCKING, structural | VELA "pristine" maps are pixel-scale shot-noise / discreteness realisations, not SB fields, and the speckle reaches the truth images at ~the observational noise | 3×3 median removes 4.8–29.3% of source flux; 26–50% of pixels exactly zero; 1-cell smoothing changes the noiseless image by Δχ² 22–6365 (0.08–1.24σ rms on detected px). No NONSCATTER extension exists for F140W. |
| 2 | serious, structural | the data constrain one compact blob, not clumpy morphology; source models have far more freedom than resolution elements | 10–100 elements above 3σ vs 66–496 shapelet coefficients; 1 peak above 20% of max at PSF/√μ resolution in 6/10 |
| 3 | serious, structural | one-dimensional design: source size, μ, brightness, S/N locked together by the private lens draw + cut-off loop | ρ(R50, μ) = −0.76, ρ(R50, m_unl) = −0.90, ρ(R50, S/N) = +0.71 |
| 4 | serious | peak-SB anchor × VELA's flat envelopes → M_AB −21 to −25 and arcs carrying 15–110× the calibration-pixel flux; not fixable by the brightness draw | 3–8 mag brighter than the Foundry lower limits |
| 5 | serious for a source test | lens light exactly the fitter's Sérsic, exact PSF, white noise | 7–48% (median 27%) of detected arc flux under lens light brighter than the arc |
| 6 | minor | vela21/26 (and dropped vela07) are 100-kpc-FOV renders at half the resolution (0.125 kpc/px); one suite/snapshot/camera, hand-picked | PIXKPC 0.125 vs 0.0625 |
| 7 | minor, bookkeeping | gallery "source" panel was the transpose of what was lensed (ImageBasedLight axis 0 = x) | fixed in `vela_plot_dataset.py` this date |

Negative checks (do not re-run): units/ZP ≤ 0.07 mag; peak-SB anchor robust to
spikes (0.05–0.18 mag); recentred frame-edge truncation coherent S/N 1.9
(vela25), < 0.1 elsewhere; ss16 ≈ ss32; lens-light S/N profile matches DESI-238.

My own spot check of #1 (`tmp/spot_speckle.py`, plot inspected): vela25's
brightest pixel (4.58 nJy) has neighbours ≤ 0.045 nJy and sits in the diffuse
halo; 3×3 median removes 29.3% (vela25) / 16.7% (vela02) of the flux; 26% /
50% of pixels are exactly zero. Confirmed. #7 confirmed from the
`ImageBasedLight` docstring ("first axis is x").

Implications I draw (UNCERTIFIED until tested): the "out of the box" source is
itself a Monte-Carlo realisation at 0.06 kpc, below VELA's physical resolution;
a 1–2-cell smoothing restores a surface-brightness field rather than altering
morphology, but it is a change to the source and needs the user's decision.
The reviewer's falsifier (regenerate from 1-cell-smoothed sources; each
method's χ² should drop by the Δχ² above; check whether the method ranking
moves) is cheap. #3 is addressable for the campaign by lensing every source
through the same 2–3 lens configurations. #2 and #5 bound the claims the set
can support rather than invalidating it.

### Where the speckle comes from (2026-09-17, user question)

Not added on purpose. The `IMAGE_PRISTINE` extension is the raw output of the
Sunrise Monte-Carlo radiative-transfer code (header `CODE = 'Sunrise (Jonsson
06)'`, HLSP Simons et al. 2019; `SKYSIG = 0`; no ray count or run parameters
survive in the HLSP header — only `HISTORY 't r5i2n5.'`). Measured on the raw
maps (`tmp/vela_quant.py`, `tmp/vela_pop.py`; image / SBFACTOR):

* Exactly-zero pixels: 25.6% of vela25's map (39% beyond r = 2"), 50% of
  vela02's, 54% of vela22's — pixels no photon packet reached.
* The non-zero values are bimodal in log: a population near 1e-6 nJy/px (far
  below any physical SB) and the galaxy at 1e-3 to 1 nJy/px, separated by a
  ~3-dex gap (vela25: modes 10^-5.9 / 10^-2.4, gap at 10^-3.3; vela22: 10^-6.7 /
  10^-2.6). The faint population is 37% (vela25) / 19% (vela22) of the map but
  holds 3e-4 / 2e-4 of the flux — low-weight scattered packets.
* 9784 isolated single-pixel spikes (> 20x all eight neighbours) hold 2.5% of
  vela25's flux; the brightest (4.58 nJy, in the halo) is M_AB = -15.5, ~1e8
  L_sun(V) — too luminous for one 1e3–1e4 M_sun VELA star particle
  [R-UNCERTIFIED for the particle mass], so a sampled packet weight rather than
  a physical cluster.

So the speckle is Monte-Carlo sampling noise (finite photon packets, peel-off
weights) on top of the discreteness of the emitting star particles; the two
cannot be separated without the Sunrise run parameters or a second
realisation (F140W ships no NONSCATTER image). [I]: the VELA team's "observed"
products convolve this map with the HST PSF at the native 0.06" pixel, where
~70 source pixels per output pixel and ~500 under the PSF average the speckle
far below the noise; lensing at mu ~ 5–20 stretches the map so fewer source
pixels fall under each PSF footprint, and the peak-SB calibration then scales
it to arc brightness. That is why it surfaces here and not in the HLSP mocks.

### Smoothing scan for the speckle (2026-09-17, `review/smooth_scan.py`)

Gaussian smoothing of the raw source map, MC signature measured as the flux a
3x3 median removes and the fractional pixel scatter about a 2.5-cell local
mean in the brightest 0.5% of pixels (which at sigma >= 2 also includes real
structure, so it is an upper bound on the noise):

| sigma (cells) | vela25 median loss / bright scatter | vela02 | vela22 | sigma in " / kpc |
|---|---|---|---|---|
| 0 | 29.3% / 0.47 | 16.7% / 0.46 | 4.8% / 0.08 | 0 |
| 1 | 4.7% / 0.13 | 2.0% / 0.17 | 0.6% / 0.05 | 0.0073 / 0.062 |
| 2 | 0.6% / 0.04 | 0.4% / 0.10 | 0.1% / 0.03 | 0.0145 / 0.125 |
| 3 | 0.1% / 0.02 | 0.2% / 0.07 | 0.1% / 0.02 | 0.0218 / 0.188 |

Tangential source-plane PSF sigma (0.073"/sqrt(mu)) is 5.8 / 4.5 / 3.2 / 2.1
cells at mu = 3 / 5 / 10 / 22, i.e. sigma = 2 cells is at or below the finest
tangential resolution any system in the set has, and it removes the MC
signature (median loss < 0.6%, bright scatter <= 0.10). Recommendation to the
user: fixed Gaussian sigma = 2 cells (0.125 kpc) as a recorded preprocessing
knob, adaptive (packet-density-targeted) smoothing as the better follow-up;
verify with the reviewer's chi^2 test at the modelling stage.

## Source smoothing implemented, sigma = 2 cells (2026-09-17, user decision) — UNCERTIFIED

User: "I like this idea. Implement it with the default 2-cell sigma", plus
residual images of the sources and normalised residuals of the lenses between
the smoothed and unsmoothed versions.

Implementation: `preprocess_source(..., smooth_sigma_pix=)` applies a Gaussian
(scipy, zero-padded) BEFORE crop/recentre; YAML key `source_smooth_sigma_pix`
(both campaigns: 2.0 = 0.125 kpc = 0.0145"); code default None (raw map, the
v2/v3 behaviour) with the physics-default-ok annotation; manifest records
`smooth_sigma_pix` and per source `smooth_sigma_arcsec` and
`smooth_flux_change_frac` (measured: -1e-5 to -3e-4, frame-edge leakage).
Test `test_preprocess_source_smooth` (flux preserved, spikes spread, flat
interior untouched, order before crop/recentre, ValueError on sigma <= 0);
the end-to-end test asserts the manifest fields. 23 tests pass; lint clean.
The gallery source panel shows the smoothed (as-lensed) source.

Both sets regenerated (`--force`, 10 sources; the harness killed the first
launch on a spurious low-memory signal — relaunched with `setsid nohup`, which
survives it; log this for next time). Main set: 12 redraws (vela21 7, vela08 4,
vela03 1); outside <= 1.0%; border <= 0.97sigma (vela21). Note: with 10 sources
the seed folding changed, so lens draws differ from the 11-source set for every
source after vela04 (vela23 now theta_E 2.22", mu 29).

| system | θ_E | amp | src/lens | μ | outside | border | redraws | src AB unl | arcs AB | lens AB | peak SB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vela02 | 1.39 | 1.68 | 0.561 | 10.1 | 0.04% | 0.26σ | 0 | 22.37 | 19.87 | 19.24 | 20.52 |
| vela03 | 1.61 | 0.51 | 0.181 | 9.6 | 0.05% | 0.15σ | 1 | 23.16 | 20.70 | 18.85 | 21.72 |
| vela04 | 1.65 | 1.21 | 0.105 | 10.1 | 1.00% | 0.17σ | 0 | 23.97 | 21.46 | 19.02 | 20.92 |
| vela08 | 1.33 | 0.24 | 0.138 | 13.8 | 0.09% | 0.69σ | 4 | 24.04 | 21.19 | 19.03 | 22.20 |
| vela09 | 1.42 | 0.40 | 0.463 | 6.8 | 0.02% | 0.20σ | 0 | 22.17 | 20.08 | 19.24 | 21.04 |
| vela21 | 1.45 | 0.68 | 0.258 | 3.4 | 0.30% | 0.97σ | 7 | 21.63 | 20.29 | 18.82 | 21.04 |
| vela22 | 1.05 | 1.18 | 0.374 | 6.2 | 0.01% | 0.04σ | 0 | 21.69 | 19.72 | 18.65 | 19.97 |
| vela23 | 2.22 | 0.51 | 0.391 | 29.4 | 0.09% | 0.04σ | 0 | 23.78 | 20.11 | 19.09 | 21.46 |
| vela25 | 1.45 | 1.27 | 0.555 | 5.4 | 0.12% | 0.47σ | 0 | 21.61 | 19.78 | 19.14 | 21.82 |
| vela26 | 1.18 | 1.67 | 0.322 | 3.4 | 0.16% | 0.05σ | 0 | 21.60 | 19.76 | 19.05 | 20.39 |

### Raw vs smoothed at the same truth (`smoothing_residuals.{py,png,json}`)

For each system the noiseless image was re-rendered at the stored truth from
the smoothed source (reproduces `noiseless_image.npy` to <= 1.3e-6) and from
the raw source; residual normalised by sigma_tot^2 = bkg^2 + image/exp_time.
Plot inspected first: source residuals are pixel-scale speckle in the bright
regions plus isolated packet spikes; lensed residuals are sub-sigma speckle
along the arcs, with the largest values on the images of compact nuclei
(vela21, vela26), where the smoothing broadens a numerically sharp cusp.

| system | Δχ² (raw vs smoothed, whole frame) | max |res| | px > 1σ | px > 3σ |
|---|---|---|---|---|
| vela02 | 151 | 1.3σ | 12 | 0 |
| vela03 | 65 | 0.9σ | 0 | 0 |
| vela04 | 51 | 2.4σ | 10 | 0 |
| vela08 | 27 | 1.5σ | 4 | 0 |
| vela09 | 177 | 2.2σ | 43 | 0 |
| vela21 | 243 | 5.8σ | 28 | 7 |
| vela22 | 33 | 1.8σ | 7 | 0 |
| vela23 | 82 | 1.1σ | 2 | 0 |
| vela25 | 103 | 1.4σ | 8 | 0 |
| vela26 | 113 | 4.7σ | 8 | 2 |

Reading: the raw set would have carried a χ² floor of 27–243 per system
(14400 px) that no smooth model could reach; these are the numbers the
reviewer's falsifier predicts each method's best-fit χ² should drop by. They
are far below the reviewer's 22–6365 because those were per sigma_bkg (which
is 8–9x smaller than sigma_tot at the bright pixels) and at 1-cell smoothing.
Caveat [I]: at the nuclei of vela21/vela26 part of the 5–6σ difference is
compact structure at the smoothing scale, not noise; whether VELA's central
cusps are physical is not decidable from the mocks. Both truths are "the
simulation" at a stated resolution; the smoothed one is the recorded truth.

Claims register additions:

| claim | status | evidence |
|---|---|---|
| 2-cell smoothing changes the source flux by < 0.03% and the lensed noiseless image by Δχ² 27–243 per system (σ_tot), max 5.8σ at a nucleus | MEASURED | `smoothing_residuals.json` |
| smoothed-set generation: 12 redraws / 10 systems, outside <= 1.0%, border <= 0.97σ | MEASURED | generation log |

### Does the 2-cell smoothing over-spread vela02's bright points? (2026-09-17, user question)

User: vela02's scattered bright points became larger diffuse blobs; worried the
smoothing spreads real structure too far. Measured (`review/vela02_points.py`,
`vela02_smoothing_check.png`, plot inspected first):

* Local maxima above 5% of the peak within 1.5": 228. Only 11 are single-pixel
  spikes; the median maximum has its brightest neighbour at 0.66 of its peak
  and holds 11% of its 5x5 flux, i.e. the bright points are few-cell clumps
  (0.1-0.3 kpc), not packet spikes. Their 5x5 boxes hold 52% of the galaxy flux.
* But the pixel-to-pixel fractional scatter inside the brightest 1% of the map
  is 0.63: the clumps' internal pixel structure is Monte-Carlo noise, so their
  shapes at the 1-cell scale were never trustworthy (the sigma = 1 panel is
  still grainy).
* At vela02's data resolution (tangential PSF sigma / sqrt(mu) = 3.16 cells =
  0.023"), raw and 2-cell-smoothed maps convolved to that scale differ by at
  most 10% of the peak (at the nucleus; ~5% at the two bright clumps), rms 1%
  of the peak over pixels above 5% of the peak. The effective resolution
  broadening for compact features is sqrt(3.16^2 + 2^2) / 3.16 = 1.18 (18%).
  In the lensed image the direct measurement is max 1.3 sigma, dchi2 151.

Assessment: the "diffuse blobs" are how few-cell clumps look at sigma = 2 when
displayed at the native 0.007" pixel; at the resolution the data have, raw and
smoothed are the same picture to ~10% at the nucleus and less elsewhere. The
smoothing does broaden real 0.1-0.3 kpc clumps (area roughly doubles for a
1.5-cell clump), which costs 18% of effective source-plane resolution at
mu = 10 and would cost ~45% at mu ~ 30 (vela23), where 2 cells exceeds the
data's 1.9-cell resolution. Config kept at 2 (uniform, flux-conserving, simple);
alternatives if the high-mu systems are modelled: sigma = 1.5 (bright scatter
0.13 instead of 0.10), or define the comparison truth at the data resolution.
Making sigma depend on the lens draw is not recommended (the source truth
would then depend on the lens).

| claim | status | evidence |
|---|---|---|
| vela02's bright points are few-cell clumps (11/228 single-pixel), internally 63% noisy; raw vs 2-cell differ by <= 10% of peak at the data's source-plane resolution | MEASURED | `review/vela02_points.py` |

### Finding 3: how large a frame removes the cut-off selection? (2026-09-17, user question)

User: "How much larger would I have to make the cutouts to effectively remove the
magnification confound?" Measured with `frame_size_scan.py` (100 prior draws per
source, lensed source rendered once on a 360 px canvas at supersample 2, the
campaign's cut-off rule evaluated for frames of 120-320 px from that render;
plot inspected first: `frame_size_scan.png`, numbers in `frame_size_scan.json`).

Hypothesis: the reviewer's rho(R50, mu) = -0.76 is the border rule rejecting
large-theta_E draws for extended sources; prediction: acceptance -> 1 and the
R50-theta_E correlation -> 0 once the frame holds the 1-sigma isophote of the
largest arcs (~2 x (theta_E,95% + R90) = 2 x (2.3 + 1.9)" = 8.4" -> ~130 px
half-width would be needed for the worst case... i.e. ~200 px). Falsifier: a
residual R50-mu correlation among unselected draws would show the confound is
geometry, not selection.

| frame | min acceptance (source) | max E[redraws] | rho(R50, median theta_E | accepted) | rho(R50, median mu | accepted) |
|---|---|---|---|---|
| 120 px 7.8" | 0.19 (vela25) | 4.3 | -0.78 | -0.84 |
| 140 px 9.1" | 0.49 | 1.0 | -0.73 | -0.88 |
| 160 px 10.4" | 0.72 | 0.4 | -0.32 | -0.89 |
| 180 px 11.7" | 0.84 | 0.2 | +0.04 | -0.89 |
| 200 px 13.0" | 0.97 | 0.03 | +0.05 | -0.88 |
| 240-320 px | >= 0.98 | 0 | +0.08 | -0.88 |
| no selection (all draws) | 1 | 0 | -0.01 | -0.88 |

* The border rule is the only rule that rejects (flux-outside fails 6 of 303
  rejections). At 120 px it caps theta_E at ~1.9" for vela21/vela25 (prior 95%
  point 2.2-2.3") and costs vela25 81% of its draws; acceptance ordering follows
  R90, not R50 (vela04, R50 0.17" but R90 0.81", loses 25%).
* Selection on theta_E is gone at 180-200 px (13"): acceptance >= 0.97 for every
  source, rho(R50, theta_E) = +0.05. 160 px removes about 60% of it.
* The magnification correlation does NOT go away: with no selection at all the
  per-source median flux-weighted magnification is 6.0 (vela21, R50 0.72") to
  9.9 (vela22, R50 0.25"), rho(R50, median mu) = -0.88. Selection at 120 px only
  moves the extended sources' medians from 6.0-6.7 to 5.5. So the reviewer's
  -0.76 is mostly geometry: at a fixed lens an extended source averages over
  low-magnification sky. No frame size removes it; it is intrinsic to lensing
  the same prior with sources of different size, and it is what the crossed
  design makes interpretable rather than removes.
* Relaxing the border threshold instead of enlarging the frame does not work at
  120 px (rho stays -0.7 to -0.8 even at 5 sigma: the arcs really reach the
  edge); at 160 px a 2-sigma border gives rho +0.03 with min acceptance 0.82.
  So the two equivalent choices are 200 px with the 1-sigma rule, or 160 px with
  a 2-sigma rule (visible arc-edge truncation in a few systems).
* Cost of 200 px: 2.8x the pixels for the truth render and for every
  likelihood evaluation (inference sub-grid 800^2 at supersample 4).

| claim | status | evidence |
|---|---|---|
| cut-off selection on theta_E vanishes at 180-200 px (acceptance >= 0.97, rho(R50, theta_E) +0.05); 120 px caps theta_E at ~1.9" for the extended sources | MEASURED (supersample 2, 100 draws/source) | `frame_size_scan.json` |
| without any selection rho(R50, median mu) = -0.88 (median mu 6.0-9.9): the size-magnification relation is geometric and no frame size removes it | MEASURED | `frame_size_scan.json` |

### Decision: 160 px frame with a 2-sigma border rule (2026-09-17, user)

User keeps one private lens draw per system ("I want to test a wide variety of
lens configurations") and chose 160 px + `max_border_sb_sigma: 2.0` over 200 px
at 1 sigma (frame-size scan: rho(R50, median theta_E) +0.03, min acceptance
0.82). Both campaign files updated (num_pix 160, canvas 20.8", comments); NOT
regenerated yet, pending the brightness-anchor decision below so the set is
regenerated once.

### Brightness anchors compared on the current sources (2026-09-17, user question)

User: with the private lens draws kept, what are the options for the
size-brightness correlation (reviewer finding 3, second mechanism)? Measured
from the two v3 manifests (peak-SB set A and the unlensed-mag set B) plus the
recorded amplitudes, which give the VELA native photometry (amp = 1):

| source | R50 | A unlensed AB | A peak SB | B unlensed AB (27.7 target) | B peak SB | native unlensed AB | native peak SB at A's lens | native M_AB |
|---|---|---|---|---|---|---|---|---|
| vela02 | 0.31 | 22.37 | 20.52 | 27.20 | 25.33 | 22.94 | 21.08 | -22.29 |
| vela03 | 0.36 | 23.16 | 21.72 | 27.68 | 26.27 | 22.43 | 20.99 | -22.79 |
| vela04 | 0.17 | 23.97 | 20.92 | 27.55 | 24.46 | 24.18 | 21.13 | -21.05 |
| vela08 | 0.34 | 24.04 | 22.20 | 26.80 | 25.22 | 22.49 | 20.66 | -22.73 |
| vela09 | 0.56 | 22.17 | 21.04 | 27.65 | 26.51 | 21.18 | 20.05 | -24.04 |
| vela21 | 0.72 | 21.63 | 21.04 | 28.50 | 27.42 | 21.21 | 20.61 | -24.01 |
| vela22 | 0.25 | 21.69 | 19.97 | 26.74 | 25.00 | 21.88 | 20.15 | -23.34 |
| vela23 | 0.20 | 23.78 | 21.46 | 28.01 | 25.70 | 23.05 | 20.73 | -22.18 |
| vela25 | 0.61 | 21.61 | 21.82 | 28.32 | 28.53 | 21.87 | 22.08 | -23.35 |
| vela26 | 0.31 | 21.60 | 20.39 | 27.10 | 25.87 | 22.16 | 20.95 | -23.07 |

Spearman with R50: A unlensed mag -0.48 (the 1-mag target scatter dilutes the
reviewer's -0.90), A peak SB +0.43 (the draw); B unlensed +0.52, B peak SB
+0.87 (the correlation moves into per-pixel S/N, and at the 27.7 target the
arcs sit at 24.5-28.5 mag/arcsec^2, i.e. below 1 sigma per pixel: undetectable);
native unlensed -0.81 (VELA's own size-luminosity relation), native peak SB -0.18.

Key fact [M]: the peak-SB calibration returns amplitudes 0.24-1.68, i.e. the
VELA galaxies' OWN F140W photometry at z = 1.5 already puts their arcs at
20.1-22.1 mag/arcsec^2 at these lenses, inside the Foundry band 21.1 +- 1. The
Foundry anchor was a proxy for a brightness the simulation already has. The
size-brightness correlation is therefore the simulation's (bigger VELA
galaxies are more massive, M_AB -21 to -24), not an artefact of the anchor;
finding 4's "over-luminous" sources are a VELA sample property (massive
galaxies), not a calibration error.

Options given to the user (no decision yet):
1. native photometry (`source_flux_scale: 1.0`, existing mode): physical
   correlation kept and declared, no per-system draw, tabled question 4 closed;
2. keep peak-SB anchor: per-pixel S/N matched, flux ~ area by construction;
3. total-flux anchor: kills the flux correlation, creates a per-pixel S/N
   correlation (+0.87) and pushes both size ends outside the Foundry SB band;
4. matched lensed flux / arc S/N: same trade as 3, luminosity ~ 1/mu;
5. brightness as a second factor at the same lens draw (x2 systems): the only
   design that tests S/N dependence at fixed morphology.

## Regenerated: 160 px, 2-sigma border, native VELA photometry (2026-09-17, user decisions) — UNCERTIFIED

User: "I'm good just keeping the native Vela photometry. Can you regenerate
everything with these new modifications?" Config: `num_pix: 160`,
`cutoff.max_border_sb_sigma: 2.0` (canvas 320 px = 20.8"), `calibration:
{source_flux_scale: 1.0}`; `campaign_unlensed_mag.yaml` marked SUPERSEDED (not
regenerated). Same seed 0; a draw that the old rule rejected is now accepted
at attempt 0, so the lenses of vela03/vela08/vela21 differ from the previous
set (vela08: theta_E 2.47" giant ring). 1 redraw in total (vela21). Grid and
gallery regenerated and inspected first (no arc near the frame edge; vela21 and
vela09 now outshine their lens). 23 tests pass.

| system | θ_E | amp | src/lens | μ | outside | border | redraws | src AB unl | arcs AB | lens AB | peak SB (7 px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vela02 | 1.39 | 1 | 0.309 | 10.1 | 0.00% | 0.00σ | 0 | 22.94 | 20.43 | 19.15 | 21.06 |
| vela03 | 1.21 | 1 | 0.084 | 4.1 | 0.00% | 0.00σ | 0 | 22.43 | 20.89 | 18.20 | 21.02 |
| vela04 | 1.65 | 1 | 0.081 | 10.2 | 0.04% | 0.52σ | 0 | 24.18 | 21.66 | 18.93 | 21.09 |
| vela08 | 2.47 | 1 | 0.487 | 12.8 | 0.00% | 0.00σ | 0 | 22.49 | 19.73 | 18.95 | 20.91 |
| vela09 | 1.42 | 1 | 1.070 | 6.8 | 0.00% | 0.00σ | 0 | 21.18 | 19.09 | 19.17 | 20.04 |
| vela21 | 1.54 | 1 | 1.431 | 6.6 | 0.06% | 0.13σ | 1 | 21.21 | 19.17 | 19.56 | 20.35 |
| vela22 | 1.05 | 1 | 0.294 | 6.2 | 0.00% | 0.00σ | 0 | 21.88 | 19.90 | 18.57 | 20.14 |
| vela23 | 2.22 | 1 | 0.706 | 29.4 | 0.01% | 0.10σ | 0 | 23.05 | 19.38 | 19.00 | 20.73 |
| vela25 | 1.45 | 1 | 0.407 | 5.4 | 0.00% | 0.00σ | 0 | 21.87 | 20.04 | 19.06 | 22.08 |
| vela26 | 1.18 | 1 | 0.178 | 3.4 | 0.02% | 0.03σ | 0 | 22.16 | 20.84 | 18.96 | 20.93 |

Peak SB is now recorded over the generator's 7-px default for non-peak-SB
modes (was the 9-px target); 0.05 mag difference, not changed.

### Raw vs smoothed at the new truths (`smoothing_residuals.{png,json}`, re-run)

| system | Δχ² (σ_tot) | max |res| | px > 1σ | px > 3σ |
|---|---|---|---|---|
| vela02 | 68 | 0.95σ | 0 | 0 |
| vela03 | 27 | 0.71σ | 0 | 0 |
| vela04 | 39 | 2.1σ | 9 | 0 |
| vela08 | 314 | 3.7σ | 44 | 4 |
| vela09 | 554 | 3.7σ | 124 | 4 |
| vela21 | 3070 | 10.5σ | 205 | 72 |
| vela22 | 26 | 1.6σ | 6 | 0 |
| vela23 | 202 | 1.7σ | 9 | 0 |
| vela25 | 72 | 1.2σ | 2 | 0 |
| vela26 | 58 | 3.5σ | 8 | 1 |

vela21 [M]: the raw-vs-smoothed difference at its nucleus is now 10.5σ over 72
pixels above 3σ (was 5.8σ / 7 px): the source is 1.5x brighter than under the
peak-SB draw and its new lens (theta_E 1.54", mu 6.6 vs 3.4) magnifies the
nucleus more. This is the compact-nucleus caveat of the smoothing decision,
now at a level the data would see: whether VELA's central cusp is physical or
a Sunrise sampling artefact is not decidable from the mocks, and the recorded
truth is the smoothed source. Flagged to the user; a modelling method that
fits the smoothed truth cannot be judged on the raw one at vela21's centre.

Page (artifact version 7): pills "native VELA photometry", "2 tabled questions"
(brightness-structure correlation and cut-off thresholds resolved); steps,
instrument / calibration / cut-off rows, stats, table, choices, files updated.
README row updated. Memory updated.

| claim | status | evidence |
|---|---|---|
| regenerated set: 1 redraw / 10 systems, outside <= 0.06%, border <= 0.52σ (2σ rule), amp = 1, peak SB 20.0–22.1 | MEASURED | generation log, manifest |
| raw-vs-smoothed Δχ² 26–3070, max 10.5σ at vela21's nucleus | MEASURED | `smoothing_residuals.json` |
