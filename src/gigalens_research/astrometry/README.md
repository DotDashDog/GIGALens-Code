# `astrometry` — point-source positions and their covariance, measured from pixels

gigalens does not render point sources (`PointSource.renders = False`), so the
image positions consumed by `gigalens.jax.point_source_position.PointSourcePositionData`
are *data*, measured elsewhere. This package is that elsewhere: a lenstronomy
forward model of the cutout that reports positions and the full `(2n, 2n)`
astrometric covariance that the position likelihood now accepts as `cov_img`
(gigalens PR #112).

---

## Contents

- [Quick start](#quick-start)
- [What you need to supply](#what-you-need-to-supply)
- [What validation has to be run](#what-validation-has-to-be-run)
- [Design decisions worth knowing about](#design-decisions-worth-knowing-about)
- [Known limits](#known-limits)

---

## Quick start

```python
from gigalens_research.astrometry import (
    Frame, PSFSpec, NoiseSpec, SystematicsBudget, measure_astrometry)

result = measure_astrometry(
    cutout,
    frame=Frame(transform_pix2angle=wcs_matrix, ra_at_xy_0=..., dec_at_xy_0=...),
    psf=PSFSpec(kernel=psf_kernel, supersampling_factor=3),
    noise=NoiseSpec(noise_map=sigma_map),
    init_ra=[0.60, -0.55, 0.10, -0.20],      # by-eye positions are fine
    init_dec=[0.15, 0.20, -0.62, 0.58],
    search_radius=0.15,
    lens_light_model_list=["SERSIC_ELLIPSE"],
    kwargs_lens_light_params=[init, sigma, fixed, lower, upper],
    systematics=SystematicsBudget(sigma_translation=1.5e-3),
)
print(result.summary())
```

and the handover into gigalens:

```python
from gigalens.jax.point_source_position import PointSourcePositionData

data = PointSourcePositionData(source_component, **result.to_gigalens_kwargs())
```

`to_gigalens_kwargs()` emits `cov_img` already in gigalens' interleaved
`[x0, y0, x1, y1, ...]` order. Do **not** pass it through `interleave_xy_cov`;
that helper is for a raw lenstronomy chain, and applying it here would permute
the matrix back into blocked order — silently, since both orderings are
`(2n, 2n)` and symmetric.

A runnable end-to-end example, measurement through validation, is in
`experiments/astrometry/demo_measure_and_validate.py`.

---

## What you need to supply

| Input | Type | Notes |
|---|---|---|
| `image` | `(npix, npix)` array | The cutout, in the units the noise model describes. |
| `frame` | `Frame` | `transform_pix2angle` + the angular coordinate of pixel `(0,0)`. **Must be the same angular frame the gigalens lens model uses.** |
| `psf` | `PSFSpec` | Odd-sided kernel, plus its supersampling factor if oversampled. |
| `noise` | `NoiseSpec` | Either a per-pixel `noise_map`, or `background_rms` (+ `exposure_time`). |
| `init_ra`, `init_dec` | length-`n` | Starting positions. By-eye is fine; their length fixes `n`. |
| `search_radius` | float | Box half-width, arcsec. Must be under half the closest image separation. |
| `lens_light_model_list` + `kwargs_lens_light_params` | lenstronomy model list + `[init, sigma, fixed, lower, upper]` | Optional in the API, **not optional in practice** for a real lens — see below. |
| `mask` | `(npix, npix)` 0/1 | Optional. |
| `systematics` | `SystematicsBudget` | Optional, defaults to zero (statistical only). Calibrate it, don't guess it. |

Three of these deserve more than a table row.

**The frame is the input most likely to be wrong and the only one with no
internal check.** Every number reported is a coordinate in this frame, and
gigalens will read those numbers in whatever frame its own grid defines. If the
two disagree — a flipped RA axis, a half-pixel origin offset, a transposed
matrix — the fit still converges, the covariance is still finite and positive
definite, every pull test still passes, and the lens model is quietly wrong.
Build it from the same WCS as the gigalens grid and check
`frame.corner_coordinates(npix)` against that grid's corners.

**The noise model sets the overall scale of the covariance.** Getting it wrong
by a factor rescales every error bar by that factor and nothing inside a single
fit will notice. The pull test will.

**Lens light is not optional for a real lens.** Unmodelled deflector light under
the images pulls the fitted positions toward the galaxy centre. That bias is
*common-mode*, so it lands in exactly the direction the lens model is most
sensitive to, and it is not reduced by having more images or deeper data.

---

## What validation has to be run

A covariance is a claim about a repeated experiment. Nothing inside one fit can
check it, and the failure mode is unpleasant: an astrometric error bar that is
too small gives a lens posterior that is too tight and centred in the wrong
place, while every downstream diagnostic — R-hat, ESS, corner plots — looks
perfectly healthy. Four things need running, and they catch different faults.

### 1. `pull_test` — statistical calibration

Inject a known configuration at ~200 noise realizations, refit, and compare the
scatter with the reported covariance.

```python
from gigalens_research.astrometry.validate import (
    TruthScene, pull_test, plot_pull_diagnostics)

scene = TruthScene(num_pix=60, frame=frame, psf=psf, noise=noise,
                   ra=truth_ra, dec=truth_dec, amp=truth_amp)
res = pull_test(scene, n_realizations=200, measure_kwargs=dict(search_radius=0.15))
plot_pull_diagnostics(res, path="pulls.png")
print(res.report())
```

Checks, in increasing order of what they can detect:

- per-coordinate pull mean ≈ 0 and width ≈ 1;
- `chi2` over the whitened residual against `chi2(2n)`;
- the **whitened empirical covariance** against the identity, judged against the
  Marchenko–Pastur band for the realization count.

Only the last two see the off-diagonals. Per-coordinate pulls are *identical*
for a diagonal matrix and a correlated one with the same variances, so a test
that stops at pull widths certifies nothing about the correlations — which are
the entire reason `cov_img` exists.

Sizing: the sampling error on the pull width is `1/sqrt(2Nk)`. For a quad
(`k=8`), 200 realizations measure it to ~2.5%, 60 to ~4.5%, and 20 only to ~8%.
Each realization is a full optimisation, so run this on a compute node.

### 2. `psf_systematics_scan` — the systematic that actually matters

The pull test is *structurally blind* to PSF error, because it simulates and
fits with the same kernel: the dominant real-world systematic is assumed away by
construction. This holds the data fixed and varies the PSF over a family
representing what is genuinely uncertain about it — different stars, epochs,
reconstruction settings:

```python
scan = psf_systematics_scan(scene, [("nominal", psf_a), ("star B", psf_b), ...],
                            image=real_cutout)
print(scan.report(statistical_sigma=np.sqrt(np.diag(res.reported_cov))))
budget = scan.budget()        # feed straight back into measure_astrometry
```

Shifts are decomposed onto translation / rotation / scale rather than reported
per image, because a PSF error moves every image together. The output is a
calibrated `SystematicsBudget`. It is only as good as the PSF family is
representative: a scan over three near-identical PSFs reports a small budget and
proves nothing.

### 3. `frame_roundtrip_check` — the cheapest test, the worst failure

Recover sources injected at known angular coordinates. Passing shows `Frame` is
self-consistent with lenstronomy's convention; it does **not** show the frame
matches the gigalens grid. Compare corner coordinates for that.

### 4. `model_perturbation_check` — light-model systematics

Refit with a component dropped, added, or re-centred. Same currency as the PSF
scan, and it belongs in the same budget.

### Read the plots first

`plot_pull_diagnostics` draws the pull histograms, the χ² distribution against
its theoretical density, the whitened empirical covariance, and the per-image
residual scatter against the reported error ellipses. A covariance that is wrong
in an interesting way — a tail, one biased image, a correlation of the wrong
sign — is obvious there and invisible in a summary statistic. A pull width of
1.00 reached by averaging an over-estimate against an under-estimate is still a
broken covariance.

---

## Design decisions worth knowing about

**No lens model is used in the measurement.** The point sources are `UNLENSED`:
`2n` free coordinates with no lens equation relating them. Folding a lens model
in would make the data a function of the thing being inferred, and the gigalens
posterior would be tightened by its own prior. The cost is real — a lens model
is genuinely informative — but a joint fit is a different analysis, not a better
version of this one.

**The covariance is marginal, not conditional.** Amplitudes and lens-light
parameters are free and non-linear (`linear_solver=False`), the Hessian is taken
over all of them, and the position block is cut out of the *inverse*:
`Sigma = (H^-1)[pos, pos]`, never `(H[pos, pos])^-1`. The latter is the
covariance at fixed nuisances and is too small wherever position trades off
against flux or against the deflector light — i.e. the regime of a real lens.

**Errors come from the likelihood Hessian, not MCMC.** `emcee` is not installed
in the pinned env, and for well-detected point sources the position likelihood
is close enough to Gaussian that a Laplace covariance is a good approximation —
which the pull test then verifies rather than assumes. The Hessian is
central-difference, and every result carries a `hessian_step_stability`
diagnostic comparing the covariance at the chosen step against twice it; if that
is not small, the step is wrong and the number is not trustworthy.

**A PSO alone under-reports the errors.** A swarm that stops slightly short of
the optimum scatters the position from fit to fit, and that scatter is invisible
to the Hessian, which describes the curvature wherever the optimiser stopped —
not the distance from there to the true minimum. Measured here: pull width 1.12
with PSO alone, 1.06 after adding a Nelder–Mead polish. Hence `polish_iterations`
defaults to non-zero; setting it to zero silently shrinks the error bars.

**Ordering is built from named indices.** lenstronomy's parameter vector is
blocked `[ra..., dec...]`; gigalens wants interleaved `[x0, y0, ...]`. Rather
than extract-then-permute, positions are looked up by name from
`param_class.num_param()` and the interleaved order is constructed directly, so
the result does not depend on lenstronomy's internal layout. The convention is
pinned against gigalens' own `interleave_xy_cov` in `tests/test_astrometry.py`.

---

## Known limits

- **Statistical correlations between well-separated images are tiny.** In the
  demo (four unblended images, no lens light) the statistical correlation peaks
  at ~0.004, and χ² using only the diagonal is indistinguishable from the full
  form. The correlations that matter come from the *systematic* common mode, not
  from the fit. Two consequences: the case for `cov_img` rests on
  `SystematicsBudget` being calibrated, and for blended images or a bright
  deflector the statistical block should be re-checked rather than assumed small.
- **`SystematicsBudget` models translation, rotation and isotropic scale.** A
  skew/shear term is not included; a PSF error that stretches one axis relative
  to the other lands partly in `sigma_independent` instead, where it is treated
  as uncorrelated.
- **Amplitudes and their covariance are reported but not validated.** The pull
  test covers positions only. Flux and time-delay channels of
  `PointSourceObsData` remain diagonal in gigalens regardless.
- **Single band.** Multi-band joint fits are not wired up.
- **Runtime.** ~7 s per fit for a quad on a 60×60 cutout, dominated by the two
  Hessians (`O(p^2)` likelihood evaluations). A 200-realization pull test is
  ~25 min single-core; parallelise over realizations on a compute node.
