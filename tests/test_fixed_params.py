"""FixedParams: forward-mode rendering, and non-drift against the fitted path.

The load-bearing test here is :func:`test_fixed_params_render_matches_point_estimate`.
``FixedParams`` and ``PointEstimate`` reach the renderer by different routes — one
holds structured params, the other unpacks a ``z`` vector through the bijector — and
if those two routes ever disagree, every forward-mode figure silently stops
describing the model that would actually be fit. Pinning them bit-identical is what
makes the second entry point safe to add.
"""
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.scene import Component, LensModel, Plane
from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.simulator import SimulatorConfig
from gigalens_research.inference_utils.posterior import (
    FixedParams,
    PointEstimate,
    SceneContext,
)
from gigalens_research.plotting.reports import PosteriorReport, _per_view, plot_scene

tfd = tfp.distributions

NUM_PIX, DELTA_PIX = 40, 0.1


def _kernel(n=5):
    yy, xx = np.mgrid[-(n // 2):n // 2 + 1, -(n // 2):n // 2 + 1]
    k = np.exp(-(xx ** 2 + yy ** 2) / 2.0)
    return k / k.sum()


def _cfg(num_pix=NUM_PIX, delta_pix=DELTA_PIX):
    return SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, kernel=_kernel(),
                           supersample=1, likelihood_precision="float32")


def _two_source_model():
    """One deflector, two source planes at different redshifts.

    ``theta_E`` is free so the model has a non-degenerate ``z`` — a fully-fixed model
    has ``num_free_params == 0``, which would make the PointEstimate half of the
    comparison vacuous.
    """
    lens = Component(EPL(50), dict(theta_E=tfd.LogNormal(np.log(1.4), 0.1), gamma=2.0,
                                   e1=0.0, e2=0.0, center_x=0.0, center_y=0.0))
    s1 = Component(SersicEllipse(use_lstsq=False),
                   dict(R_sersic=0.3, n_sersic=2.0, e1=0.0, e2=0.0,
                        center_x=0.05, center_y=0.0, Ie=12.0))
    s2 = Component(SersicEllipse(use_lstsq=False),
                   dict(R_sersic=0.2, n_sersic=1.5, e1=0.0, e2=0.0,
                        center_x=-0.1, center_y=0.1, Ie=8.0))
    return LensModel([
        Plane(deflection_ratio=None, mass=[lens], name="lens"),
        Plane(deflection_ratio=0.7, light=[s1], name="src_a"),
        Plane(deflection_ratio=1.0, light=[s2], name="src_b"),
    ])


def _sims(model, cfgs=None):
    """One simulator per source plane, each seeing only that plane's light."""
    planes = [i for i, p in enumerate(model.planes) if p.has_light]
    cfgs = cfgs or [_cfg() for _ in planes]
    return [SceneSimulator(model, c, sees=model.planes[i].light)
            for i, c in zip(planes, cfgs)]


def _fitted_counterpart(model, sims, params):
    """A PointEstimate over the same scene, at the same parameters.

    Renders each band first so the ImageData carries a plausible image; the images
    only matter for constructing a valid ProbModel (forward mode never reads them
    while rendering).
    """
    datasets = [
        ImageData(np.asarray(s.simulate(params)), s.sim_config,
                  background_rms=0.01, exp_time=100.0, sees=model.planes[i].light)
        for s, i in zip(sims, [i for i, p in enumerate(model.planes) if p.has_light])
    ]
    prob = ProbModel(model, datasets, mode="forward")
    ctx = SimpleNamespace(prob_model=prob, sim_config=sims[0].sim_config)
    # unconstrained() takes the STRUCTURED params and calls to_unique itself.
    z_best = np.asarray(model.unconstrained(params)).reshape(-1)
    return PointEstimate(ctx, z_best), prob


@pytest.fixture(scope="module")
def scene():
    model = _two_source_model()
    sims = _sims(model)
    truth = {n: 1.4 for n in model.z_param_names}
    params = model.to_params({n: np.asarray(v) for n, v in truth.items()})
    return model, sims, params


# ---------------------------------------------------------------------------
# The drift test
# ---------------------------------------------------------------------------


def test_fixed_params_render_matches_point_estimate(scene):
    """The two entry points must render the same pixels at the same parameters.

    FixedParams overrides ``params_at`` directly; PointEstimate goes z -> bijector ->
    to_params. Same scene, same values, so the images must agree exactly — any
    divergence means forward-mode figures describe a different model than the fit.
    """
    model, sims, params = scene
    fitted, prob = _fitted_counterpart(model, sims, params)
    forward = FixedParams(model, prob.simulators, fitted.params_at("best"))

    assert forward.n_datasets() == fitted.n_datasets() == len(sims)
    for d in range(len(sims)):
        a = np.asarray(fitted.simulate(point="best", dataset=d))
        b = np.asarray(forward.simulate(dataset=d))
        np.testing.assert_array_equal(
            a, b, err_msg=f"band {d}: FixedParams and PointEstimate renders differ")


def test_source_plane_views_match_point_estimate(scene):
    """Per-plane geometry (dataset, plane, deflection ratio) must agree too."""
    model, sims, params = scene
    fitted, prob = _fitted_counterpart(model, sims, params)
    forward = FixedParams(model, prob.simulators, fitted.params_at("best"))
    assert forward.source_plane_views() == fitted.source_plane_views(point="best")


# ---------------------------------------------------------------------------
# Forward mode stands alone
# ---------------------------------------------------------------------------


def test_renders_without_any_prob_model(scene):
    """The whole point: no ProbModel, no data, still renders."""
    model, sims, params = scene
    fp = FixedParams(model, sims, params)
    assert fp.is_backward is False           # nothing to solve amplitudes against
    assert fp.n_datasets() == len(sims)
    assert len(fp.source_plane_views()) == 2  # one per source plane
    img = np.asarray(fp.simulate())
    assert img.shape[-2:] == (NUM_PIX, NUM_PIX)
    assert np.isfinite(img).all()


def test_data_dependent_paths_raise_pointed_error(scene):
    """A missing-data failure must name the alternative, not surface as AttributeError."""
    model, sims, params = scene
    fp = FixedParams(model, sims, params)
    for call in (lambda: fp.observed_for(0), lambda: fp.err_map_at(np.zeros((4, 4)))):
        with pytest.raises(TypeError, match="forward-mode scene with no ProbModel"):
            call()


def test_point_name_is_ignored(scene):
    """Every representative point is the same one; the argument survives for
    compatibility with PosteriorReport, which threads ``point=`` unconditionally."""
    model, sims, params = scene
    fp = FixedParams(model, sims, params)
    a = fp.params_at("median")
    for name in ("mean", "best", "anything-at-all"):
        assert fp.params_at(name) is a


def test_no_z_vector_is_offered(scene):
    """_point_z must fail loudly rather than fabricate a z for a fixed model."""
    model, sims, params = scene
    fp = FixedParams(model, sims, params)
    with pytest.raises(TypeError, match="no z vector"):
        fp._point_z("median")


def test_scene_context_requires_a_simulator():
    model = _two_source_model()
    with pytest.raises(ValueError, match="at least one SceneSimulator"):
        SceneContext(model, [])


# ---------------------------------------------------------------------------
# Per-band extent and per-view framing
# ---------------------------------------------------------------------------


def test_band_extent_is_per_band_not_figure_wide():
    """Bands with different grids must get different extents.

    The field of view must actually differ between the two configs, or this passes
    without testing anything (40 x 0.1 and 20 x 0.2 are both 4 arcsec).
    """
    model = _two_source_model()
    sims = _sims(model, cfgs=[_cfg(40, 0.1), _cfg(20, 0.1)])
    rep = PosteriorReport(FixedParams(model, sims, None))
    assert rep._band_extent(0) == (-2.0, 2.0, -2.0, 2.0)
    assert rep._band_extent(1) == (-1.0, 1.0, -1.0, 1.0)


@pytest.mark.parametrize("spec,plane,dataset,expected", [
    (8.0, 3, 2, 8.0),                       # scalar applies everywhere
    ({3: 4.0}, 3, 2, 4.0),                  # keyed by plane
    ({3: 4.0}, 5, 2, None),                 # unlisted -> auto-frame
    ({(2, 3): 9.0, 3: 4.0}, 3, 2, 9.0),     # (dataset, plane) wins
    ((1.0, 2.0), 3, 2, (1.0, 2.0)),         # tuple is a value, not a mapping
    (None, 3, 2, None),
])
def test_per_view_resolution(spec, plane, dataset, expected):
    assert _per_view(spec, plane, dataset) == expected


# ---------------------------------------------------------------------------
# The convenience wrapper
# ---------------------------------------------------------------------------


def test_plot_scene_returns_both_panels(scene):
    model, sims, params = scene
    figs = plot_scene(model, sims, params, grid_pix=32)
    assert set(figs) == {"model", "source"}
    # source_panel puts one row per source plane
    assert len(figs["source"].axes) >= 2


def test_plot_scene_accepts_per_view_framing(scene):
    model, sims, params = scene
    figs = plot_scene(model, sims, params, grid_pix=32,
                      fov_arcsec={1: 2.0}, center={2: (0.0, 0.0)})
    assert set(figs) == {"model", "source"}
