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
