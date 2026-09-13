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
native); the 12th source; `n_reps`.

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

## Claims register

| claim | status | scope |
|---|---|---|
| v1 test set is lens-dominated (source/lens 0.05–0.13) with arcs cut off (up to 21% outside) | proposed | 3 re-rendered systems + border stats on all 12 |
| STScI ePSF resampled to 0.03" has FWHM ≈ 0.08–0.09" | proposed | unit test on synthetic ePSF + measured library PSF |
| v2 generator reproduces the requested flux ratio to 1e-6 | proposed | end-to-end test on a synthetic source (`vela_simulated_test.py`, 18 tests pass) + 2 real systems (ratio 0.500) |
| reaching source/lens = 0.5 by scaling the source can require 35x (unphysical source AB ~20) for compact sources | proposed | 1 of 2 review-set systems (vela22) |
