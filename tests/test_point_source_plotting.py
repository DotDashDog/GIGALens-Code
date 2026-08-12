"""Point-source prediction plumbing and plotting.

The load-bearing checks are the ones that would let a wrong figure look right:

- the recomputed chi2 decomposition **sums back to the term's own scored chi2**
  (otherwise the breakdown describes something the likelihood does not compute);
- the batched draw path agrees with the single-point path at identical parameters
  (otherwise the predictive cloud is a different model from the marked prediction);
- pulls carry the package's ``observed - predicted`` sign (a sign flip is invisible
  in chi2 and inverts every bar on the pull chart);
- image-plane accessors refuse a point-source dataset, and dataset indices map to
  the right simulator when the two kinds are mixed.

Everything runs on one real EPL+shear quad built by solving the lens equation, so
the numbers are a genuine lensing configuration rather than invented arrays.
"""

import matplotlib
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

matplotlib.use("Agg")

import jax

jax.config.update("jax_enable_x64", True)

from gigalens.jax.analysis.lens_solver import ImageSolverConfig, LensSolver
from gigalens.jax.point_source_position import (
    PointSourceObsData,
    PointSourcePositionData,
)
from gigalens.jax.profiles.light.point_source import PointSourcePosition
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.scene import Component, LensModel, Plane
from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.simulator import SimulatorConfig

from gigalens_research.inference_utils.datasets import (
    KIND_IMAGE,
    KIND_POINT_SOURCE,
    KIND_POINT_SOURCE_LOSS,
    covariance_ellipse,
    dataset_kind,
    image_position_covariances,
    mahalanobis_sq,
    whiten,
)
from gigalens_research.inference_utils.point_source import point_source_term
from gigalens_research.inference_utils.posterior import SamplerPosterior
from gigalens_research.plotting.reports import PosteriorReport

tfd = tfp.distributions

TRUTH = {
    "planes/0/mass/0/theta_E": 1.4, "planes/0/mass/0/gamma": 2.05,
    "planes/0/mass/0/e1": 0.15, "planes/0/mass/0/e2": -0.10,
    "planes/0/mass/1/gamma1": 0.05, "planes/0/mass/1/gamma2": -0.03,
    "planes/1/light/0/center_x": 0.05, "planes/1/light/0/center_y": 0.03,
}
SIGMA_AST = 0.005
N_DRAWS = 120


class _Ctx:
    """The whole contract Posterior needs from an inference context."""

    def __init__(self, prob_model, sim_config=None):
        self.prob_model = prob_model
        self.sim_config = sim_config


def _mass_components():
    epl_c = Component(EPL(), dict(
        theta_E=tfd.Normal(1.4, 0.1), gamma=tfd.Normal(2.0, 0.1),
        e1=tfd.Normal(0.0, 0.2), e2=tfd.Normal(0.0, 0.2),
        center_x=0.0, center_y=0.0))
    sh_c = Component(Shear(), dict(gamma1=tfd.Normal(0.0, 0.05),
                                   gamma2=tfd.Normal(0.0, 0.05)))
    return epl_c, sh_c


def _build(with_amp=False):
    epl_c, sh_c = _mass_components()
    prior = dict(center_x=tfd.Normal(0.0, 0.2), center_y=tfd.Normal(0.0, 0.2))
    if with_amp:
        prior["amp"] = tfd.LogNormal(0.0, 0.5)
    ps_c = Component(PointSourcePosition(absolute=True, with_amp=with_amp), prior)
    model = LensModel([Plane(mass=[epl_c, sh_c]),
                       Plane(deflection_ratio=1.0, light=[ps_c])])
    return model, ps_c


@pytest.fixture(scope="module")
def quad():
    """True image positions of a real EPL+shear quad, plus noisy 'observations'."""
    model, ps_c = _build()
    seed_data = PointSourcePositionData(ps_c, [1.0, -1.0], [0.0, 0.1], SIGMA_AST)
    solver = LensSolver(ProbModel(model, seed_data),
                        ImageSolverConfig(search_window=8.0, min_distance=0.02))
    xy = np.asarray(solver.solve_images(
        model.to_params(dict(TRUTH)),
        (TRUTH["planes/1/light/0/center_x"], TRUTH["planes/1/light/0/center_y"])))
    assert xy.shape[1] == 4, f"expected a quad, got {xy.shape[1]} images"
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, SIGMA_AST, size=xy.shape)
    return xy[0], xy[1], xy[0] + noise[0], xy[1] + noise[1], noise


def _posterior(quad, *, scatter=0.0015, sigma=SIGMA_AST, with_amp=False,
               flux=None, n_steps=60):
    _, _, x_obs, y_obs, _ = quad
    model, ps_c = _build(with_amp=with_amp)
    truth = dict(TRUTH)
    if with_amp:
        truth["planes/1/light/0/amp"] = 1.0
        data = PointSourceObsData(ps_c, x_obs, y_obs, sigma,
                                  flux_obs=flux, sigma_flux=0.05 * np.asarray(flux))
    else:
        data = PointSourcePositionData(ps_c, x_obs, y_obs, sigma,
                                       src_anchor_sigma=SIGMA_AST)
    prob = ProbModel(model, data)
    z_truth = np.asarray(prob.bij.inverse({k: np.float64(v) for k, v in truth.items()}))
    rng = np.random.default_rng(5)
    z = z_truth[None, None, :] + scatter * rng.normal(size=(2, n_steps, z_truth.size))
    return SamplerPosterior(_Ctx(prob), z), z_truth


# ---------------------------------------------------------------------------
# Dataset kinds
# ---------------------------------------------------------------------------


def test_dataset_kind_separates_the_two_point_source_modules(quad):
    """The calibrated position module and the three-term loss are NOT one kind."""
    _, _, x_obs, y_obs, _ = quad
    _, ps_c = _build()
    assert dataset_kind(PointSourcePositionData(ps_c, x_obs, y_obs, SIGMA_AST)) \
        == KIND_POINT_SOURCE

    from gigalens.jax.point_source import PointSourceData
    assert PointSourceData.is_pointsource is True
    # Duck-typed path: the marker is what separates them without construction.
    class _ThreeTerm:
        is_pointsource = True
        x_img = y_img = np.zeros(2)
    assert dataset_kind(_ThreeTerm()) == KIND_POINT_SOURCE_LOSS


def test_point_source_term_rejects_the_three_term_loss(quad):
    """Its chi2 is a weighted loss, so pulls drawn from it would claim a
    calibration that module does not offer."""
    post, _ = _posterior(quad)

    class _Loss:
        is_pointsource = True
        x_img = y_img = np.zeros(2)

    post.ctx.prob_model.datasets = [_Loss()]
    with pytest.raises(TypeError, match="three-term hand-weighted loss"):
        point_source_term(post, 0)


# ---------------------------------------------------------------------------
# Uncertainty normalization (incl. the full-covariance form)
# ---------------------------------------------------------------------------


def test_covariance_forms_agree_where_they_overlap(quad):
    """Scalar, per-image and per-coordinate sigmas are all the same covariance."""
    _, _, x_obs, y_obs, _ = quad
    _, ps_c = _build()
    n = x_obs.size
    scalar = image_position_covariances(
        PointSourcePositionData(ps_c, x_obs, y_obs, SIGMA_AST))
    vector = image_position_covariances(
        PointSourcePositionData(ps_c, x_obs, y_obs, np.full(n, SIGMA_AST)))
    per_coord = image_position_covariances(
        PointSourcePositionData(ps_c, x_obs, y_obs, np.full((n, 2), SIGMA_AST)))
    assert scalar.shape == (n, 2, 2)
    np.testing.assert_allclose(scalar, vector)
    np.testing.assert_allclose(scalar, per_coord)
    np.testing.assert_allclose(scalar[0], np.eye(2) * SIGMA_AST ** 2)


def test_full_covariance_is_read_from_cov_img(quad):
    """A dataset carrying a full covariance overrides the sigma form.

    Forward-compatible with the upstream full-covariance change: whichever way the
    dataset ends up exposing it, everything downstream is written against (n, 2, 2).
    """
    _, _, x_obs, y_obs, _ = quad
    _, ps_c = _build()
    n = x_obs.size
    ds = PointSourcePositionData(ps_c, x_obs, y_obs, SIGMA_AST)
    rho = 0.6
    c = SIGMA_AST ** 2 * np.array([[1.0, rho], [rho, 1.0]])
    ds.cov_img = np.repeat(c[None], n, axis=0)
    got = image_position_covariances(ds)
    np.testing.assert_allclose(got, ds.cov_img)


@pytest.mark.parametrize("bad, match", [
    (np.tile(np.array([[1e-4, 1e-5], [2e-5, 1e-4]]), (4, 1, 1)), "not symmetric"),
    (np.tile(np.array([[1e-4, 1e-3], [1e-3, 1e-4]]), (4, 1, 1)), "non-positive-definite"),
    (np.zeros((4, 3)), "none of the recognized forms"),
])
def test_bad_covariances_raise(quad, bad, match):
    """A Cholesky factor or a precision matrix must not be silently whitened as if
    it were a covariance."""
    _, _, x_obs, y_obs, _ = quad
    _, ps_c = _build()
    ds = PointSourcePositionData(ps_c, x_obs, y_obs, SIGMA_AST)
    ds.cov_img = bad
    with pytest.raises(ValueError, match=match):
        image_position_covariances(ds)


def test_whiten_reduces_to_sigma_division_for_diagonal_covariance():
    res = np.array([[0.01, -0.02], [0.03, 0.004]])
    sig = np.array([[0.005, 0.010], [0.002, 0.004]])
    cov = np.einsum("ij,jk->ijk", sig ** 2, np.eye(2))
    np.testing.assert_allclose(whiten(res, cov), res / sig, rtol=1e-12)


def test_mahalanobis_matches_explicit_inverse_for_correlated_covariance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 2, 2))
    cov = np.einsum("nij,nkj->nik", a, a) + np.eye(2) * 1e-3
    res = rng.normal(size=(3, 2))
    want = np.array([r @ np.linalg.inv(c) @ r for r, c in zip(res, cov)])
    np.testing.assert_allclose(mahalanobis_sq(res, cov), want, rtol=1e-10)


def test_covariance_ellipse_axes_and_angle():
    cov = np.diag([4.0, 1.0])                       # sigma 2 along x, 1 along y
    w, h, angle = covariance_ellipse(cov, n_sigma=1.0)
    assert (w, h) == pytest.approx((4.0, 2.0))      # full axis lengths
    assert angle % 180 == pytest.approx(0.0)
    w2, _, _ = covariance_ellipse(cov, n_sigma=2.0)
    assert w2 == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# The predictions themselves
# ---------------------------------------------------------------------------


def test_chi2_decomposition_sums_to_the_terms_own_chi2(quad):
    """The decomposition must describe the chi2 the likelihood actually scores."""
    post, _ = _posterior(quad)
    pred = post.point_source_prediction(dataset=0)
    assert pred.chi2_closes
    assert pred.chi2_parts_sum == pytest.approx(pred.chi2_total, rel=1e-9)
    assert pred.chi2_anchor is not None          # src_anchor_sigma was set
    assert pred.red_chi2 == pytest.approx(pred.chi2_total / pred.event_size)


def test_chi2_decomposition_closes_over_perturbed_draws(quad):
    """Including well away from convergence, where the honesty charge is live."""
    post, _ = _posterior(quad, scatter=0.02)
    draws = post.point_source_draws(dataset=0, n_draws=N_DRAWS)
    parts = draws.chi2_displacement.sum(axis=0) + draws.chi2_honesty.sum(axis=0)
    if draws.chi2_anchor is not None:
        parts = parts + draws.chi2_anchor.sum(axis=0)
    rel = np.abs(parts - draws.chi2) / np.maximum(np.abs(draws.chi2), 1.0)
    assert rel.max() < 1e-9
    # This perturbation is wide enough that the honesty charge is genuinely doing
    # work; if it were not, the test would pass vacuously.
    assert draws.honesty_fraction.max() > 0


def test_pulls_are_observed_minus_predicted(quad):
    """At the truth the solve returns the true images, so the pull IS the injected
    astrometric noise over sigma — sign included."""
    x_true, y_true, _, _, noise = quad
    post, z_truth = _posterior(quad, scatter=0.0, n_steps=4)
    pred = post.point_source_prediction(dataset=0)

    np.testing.assert_allclose(pred.x_pred, x_true, atol=1e-8)
    np.testing.assert_allclose(pred.y_pred, y_true, atol=1e-8)
    expected = np.stack([noise[0], noise[1]], axis=1) / SIGMA_AST
    np.testing.assert_allclose(pred.pulls, expected, atol=1e-5)


def test_batched_draws_reproduce_the_single_point_path(quad):
    """A predictive cloud drawn from a different computation than the marked
    prediction would misplace one relative to the other."""
    post, _ = _posterior(quad, scatter=0.0, n_steps=20)
    pred = post.point_source_prediction(dataset=0)
    draws = post.point_source_draws(dataset=0, n_draws=20)
    np.testing.assert_allclose(draws.x_pred[:, 0], pred.x_pred, atol=1e-12)
    np.testing.assert_allclose(draws.y_pred[:, 0], pred.y_pred, atol=1e-12)
    np.testing.assert_allclose(draws.chi2[0], pred.chi2_total, rtol=1e-12)
    np.testing.assert_allclose(draws.mu[:, 0], pred.mu, rtol=1e-12)
    np.testing.assert_allclose(draws.chi2_displacement[:, 0], pred.chi2_displacement,
                               rtol=1e-10)


def test_solver_health_responds_to_draw_quality(quad):
    """A diagnostic that reports the same thing regardless of the input is not a
    diagnostic: tight draws converge, wide ones pin against the trust region."""
    tight, _ = _posterior(quad, scatter=0.0)
    wide, _ = _posterior(quad, scatter=0.05)
    d_tight = tight.point_source_draws(dataset=0, n_draws=N_DRAWS)
    d_wide = wide.point_source_draws(dataset=0, n_draws=N_DRAWS)
    assert d_tight.frac_unconverged == 0.0
    assert d_wide.frac_unconverged > 0.5
    assert d_tight.max_src_residual.max() < 1e-6
    assert np.median(d_wide.trust_frac) > np.median(d_tight.trust_frac)


def test_draws_are_thinned_not_truncated(quad):
    """Thinning must span the chains, not return the first k samples."""
    post, _ = _posterior(quad, n_steps=200)
    draws = post.point_source_draws(dataset=0, n_draws=40)
    assert 0 < draws.n_draws <= 2 * 200
    assert draws.x_pred.shape == (4, draws.n_draws)


def test_point_estimate_has_no_draws(quad):
    """A MAP point has no distribution to draw a cloud from; the panel must be
    told that rather than shown a contour built from one point."""
    from gigalens_research.inference_utils.posterior import PointEstimate
    post, z_truth = _posterior(quad)
    pe = PointEstimate(post.ctx, np.asarray(z_truth))
    assert pe.point_source_draws(dataset=0) is None
    pred = pe.point_source_prediction(dataset=0)
    assert pred.chi2_closes


# ---------------------------------------------------------------------------
# Kind dispatch on the Posterior
# ---------------------------------------------------------------------------


def test_imaging_accessors_refuse_a_point_source_dataset(quad):
    post, _ = _posterior(quad)
    assert post.dataset_kinds() == [KIND_POINT_SOURCE]
    assert post.imaging_datasets() == []
    for call in (post.observed_for, post._error_for, post.mask_for):
        with pytest.raises(TypeError, match="imaging dataset"):
            call(0)


def test_predict_dispatches_on_kind(quad):
    post, _ = _posterior(quad)
    out = post.predict(dataset=0)
    assert out.__class__.__name__ == "PointSourcePrediction"
    assert out.n_images == 4


@pytest.fixture(scope="module")
def joint(quad):
    """A model fit against BOTH a point source and an image, point source first.

    This ordering is the one that breaks naive indexing: ``ProbModel.simulators`` is
    compacted to imaging terms, so the image is dataset 1 but simulator 0.
    """
    _, _, x_obs, y_obs, _ = quad
    epl_c, sh_c = _mass_components()
    ps_c = Component(PointSourcePosition(absolute=True),
                     dict(center_x=tfd.Normal(0.0, 0.2), center_y=tfd.Normal(0.0, 0.2)))
    host = Component(SersicEllipse(use_lstsq=True), dict(
        R_sersic=tfd.Normal(0.3, 0.05), n_sersic=tfd.Normal(1.0, 0.2),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1)))
    model = LensModel([Plane(mass=[epl_c, sh_c]),
                       Plane(deflection_ratio=1.0, light=[ps_c, host])])
    sc = SimulatorConfig(delta_pix=0.05, num_pix=48)
    image = np.random.default_rng(0).normal(0.0, 0.1, (48, 48))
    prob = ProbModel(model, [
        PointSourcePositionData(ps_c, x_obs, y_obs, SIGMA_AST),
        ImageData(image, sc, background_rms=0.1, exp_time=100.0, sees=[host]),
    ])
    z = np.random.default_rng(1).normal(size=(2, 20, model.num_free_params)) * 0.01
    return SamplerPosterior(_Ctx(prob, sim_config=sc), z)


def test_mixed_datasets_map_to_the_right_simulator(joint):
    assert joint.dataset_kinds() == [KIND_POINT_SOURCE, KIND_IMAGE]
    assert joint.imaging_datasets() == [1]
    assert len(joint.ctx.prob_model.simulators) == 1
    # Dataset 1 is simulator 0: indexing simulators by dataset position would raise
    # IndexError here, which is exactly the bug this mapping exists to prevent.
    sim = joint._sim_for(1)
    assert sim is joint.ctx.prob_model.simulators[0]
    assert np.asarray(joint.observed_for(1)).shape == (48, 48)


def test_source_plane_views_skip_non_imaging_datasets(joint):
    """A source plane is reconstructed from pixels; a point source has none."""
    views = joint.source_plane_views()
    assert [d for d, _, _ in views] == [1]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_point_source_panel_renders(quad):
    post, _ = _posterior(quad)
    fig = PosteriorReport(post, prefix="test").point_source_panel(
        dataset=0, n_draws=N_DRAWS)
    # 4 top + 4 zooms + 2 health + 3 channel slots
    assert len(fig.axes) == 13
    assert "Point source" in fig._suptitle.get_text()


def test_panel_renders_with_flux_channel_and_full_covariance(quad):
    x_true, y_true, x_obs, y_obs, _ = quad
    n = x_obs.size
    # Anisotropic, per-image astrometry alongside the flux channel.
    sigma = np.stack([np.full(n, SIGMA_AST),
                      SIGMA_AST * np.array([1.0, 3.0, 1.0, 3.0])], axis=1)
    post, _ = _posterior(quad, sigma=sigma, with_amp=True,
                         flux=np.array([15.0, 11.5, 5.8, 7.6]))
    pred = post.point_source_prediction(dataset=0)
    assert pred.inv_flux_obs is not None and pred.chi2_flux is not None
    assert pred.chi2_closes
    fig = PosteriorReport(post).point_source_panel(dataset=0, n_draws=N_DRAWS)
    assert len(fig.axes) == 13


def test_full_report_omits_the_image_panel_for_a_point_source_only_model(quad):
    post, _ = _posterior(quad)
    figs = PosteriorReport(post).full_report()
    assert "point_source" in figs
    assert "image" not in figs and "source" not in figs


def test_image_panel_raises_a_useful_error_without_imaging_data(quad):
    post, _ = _posterior(quad)
    with pytest.raises(TypeError, match="point_source_panel"):
        PosteriorReport(post).image_panel()
