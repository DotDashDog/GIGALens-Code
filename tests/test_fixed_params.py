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


def _cfg(num_pix=NUM_PIX, delta_pix=DELTA_PIX, *, with_psf=True):
    """``with_psf=False`` is not an edge case — it is the natural first thing to try
    when building a mock, before you have a real PSF to hand."""
    return SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix,
                           kernel=_kernel() if with_psf else None,
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


def _clump_model(centers=((0.0, 0.0),), *, r_sersic=0.2, ie=10.0, lens_light=False):
    """One deflector and one source plane carrying a clump per entry in ``centers``.

    Fully fixed (``num_free_params == 0``) — these are framing tests, and a compact
    source in a wide cutout is exactly the case auto-framing exists for.
    """
    lens = Component(EPL(50), dict(theta_E=1.4, gamma=2.0, e1=0.0, e2=0.0,
                                   center_x=0.0, center_y=0.0))
    light = [Component(SersicEllipse(use_lstsq=False),
                       dict(R_sersic=r_sersic, n_sersic=1.0, e1=0.0, e2=0.0,
                            center_x=cx, center_y=cy, Ie=ie))
             for cx, cy in centers]
    halo_light = [Component(SersicEllipse(use_lstsq=False),
                            dict(R_sersic=2.0, n_sersic=4.0, e1=0.0, e2=0.0,
                                 center_x=0.0, center_y=0.0, Ie=3.0))] if lens_light else []
    return LensModel([
        Plane(deflection_ratio=None, mass=[lens], light=halo_light, name="lens"),
        Plane(deflection_ratio=1.0, light=light, name="src"),
    ])


#: A wide cutout — 40 x 0.5" = 20" across, the cluster-field regime where a
#: cutout-framed source plane degenerates into a few pixels.
def _wide_cfg():
    return SimulatorConfig(delta_pix=0.5, num_pix=40, kernel=None, supersample=1,
                           likelihood_precision="float32")


def _wide_scene(model):
    sims = [SceneSimulator(model, _wide_cfg(), sees=p.light)
            for p in model.planes if p.has_light]
    return FixedParams(model, sims, model.to_params({}))


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


@pytest.mark.parametrize("with_psf", [True, False])
def test_fixed_params_render_matches_point_estimate(with_psf):
    """The two entry points must render the same pixels at the same parameters.

    FixedParams overrides ``params_at`` directly; PointEstimate goes z -> bijector ->
    to_params. Same scene, same values, so the images must agree exactly — any
    divergence means forward-mode figures describe a different model than the fit.

    Run with and without a PSF: the two paths pick their cast dtype differently when
    there is no kernel (fitted has an observed image to read it from, forward does
    not), so a PSF-less scene is where they would most plausibly drift apart.
    """
    model = _two_source_model()
    sims = _sims(model, cfgs=[_cfg(with_psf=with_psf) for _ in range(2)])
    params = model.to_params({n: np.asarray(1.4) for n in model.z_param_names})
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


@pytest.mark.parametrize("with_psf", [True, False])
def test_renders_without_psf_kernel(with_psf):
    """A forward scene must render with OR without a PSF.

    Regression: ``simulate`` picked its cast dtype from the PSF kernel and fell back
    to the *observed image* when there was none — so a PSF-less forward scene raised
    "needs observed data" from a path that needs no data at all. Every other test
    here supplies a kernel, which is precisely why the gap survived; a mock is
    usually built PSF-first-missing, so this is the common case, not the exotic one.
    """
    model = _two_source_model()
    sims = _sims(model, cfgs=[_cfg(with_psf=with_psf) for _ in range(2)])
    assert (sims[0].flat_kernel is not None) is with_psf
    fp = FixedParams(model, sims, model.to_params({n: np.asarray(1.4)
                                                   for n in model.z_param_names}))
    img = np.asarray(fp.simulate())
    assert np.isfinite(img).all()
    figs = plot_scene(model, sims, fp.params_at(), grid_pix=24)
    assert set(figs) == {"scene"}


@pytest.mark.parametrize("with_psf", [True, False])
def test_plain_float_params_render_like_array_params(with_psf):
    """A params dict may hold Python floats, and must render identically to arrays.

    Regression: with a PSF, ``simulate`` casts every floating leaf to the kernel's
    dtype via ``.astype`` — which a Python ``float`` does not have. ``to_params``
    returns arrays, so the tests never saw it, but a hand-written or partly-literal
    dict is a perfectly reasonable thing to hand ``FixedParams``, and that is what
    crashed.

    Asserting equality, not merely that it runs, is the real claim: a weakly-typed
    Python float adopts whatever dtype it meets, while the same number as an array
    gets cast here, so the two spellings could otherwise render different pixels.
    """
    model = _clump_model()
    cfg = SimulatorConfig(delta_pix=0.5, num_pix=40,
                          kernel=_kernel() if with_psf else None,
                          supersample=1, likelihood_precision="float64")
    sims = [SceneSimulator(model, cfg, sees=p.light)
            for p in model.planes if p.has_light]
    arrays = model.to_params({})
    floats = jax.tree_util.tree_map(
        lambda a: float(a) if np.ndim(a) == 0 else a, arrays)
    assert any(isinstance(v, float) for v in jax.tree_util.tree_leaves(floats))

    np.testing.assert_array_equal(
        np.asarray(FixedParams(model, sims, arrays).simulate()),
        np.asarray(FixedParams(model, sims, floats).simulate()),
        err_msg="literal and array params rendered differently")
    # The path the failure actually came through.
    assert set(plot_scene(model, sims, floats, grid_pix=24)) == {"scene"}


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
# Auto-framing the source plane
# ---------------------------------------------------------------------------


def _half(extent):
    return (extent[1] - extent[0]) / 2.0


def test_autoframe_zooms_onto_a_compact_source():
    """The default window must be sized by the source, not by the cutout.

    A 0.2" source in a 20" cutout is 1% of the frame — the case that made source
    panels unreadable. The lower bound matters as much as the upper: a frame that
    collapsed onto the peak pixel would also 'zoom in' while showing nothing.
    """
    fp = _wide_scene(_clump_model())
    half = _half(fp.source_plane()[1])
    assert 0.3 < half < 2.0, half        # cutout half is 10.0


def test_autoframe_contains_every_clump_on_the_plane():
    """All of a plane's sources must land inside its window.

    Framing on the FIRST component (the old behaviour) puts a second clump 4" away
    outside a source-sized window — visibly cropping half the source and looking
    like the model, not the frame, is wrong.
    """
    centers = ((-2.0, 0.0), (2.0, 0.6))
    x0, x1, y0, y1 = _wide_scene(_clump_model(centers)).source_plane()[1]
    for cx, cy in centers:
        assert x0 < cx < x1 and y0 < cy < y1, (cx, cy, (x0, x1, y0, y1))
    assert (x1 - x0) < 12.0               # and still tighter than the cutout


def test_autoframe_searches_wide_enough_for_clumps_near_the_cutout_edge():
    """The search window must cover every clump, not one cutout centered on the first.

    Separate from the test above, which the search window is NOT load-bearing for:
    two clumps 4" apart both fall inside a 20" window wherever it is centered, so
    only widely-separated clumps can tell a mis-centered scan from a correct one.
    Here the flux at +8" lies outside a cutout-sized window centered on the clump at
    -8", and gets framed out of the figure entirely.
    """
    centers = ((-8.0, 0.0), (8.0, 1.0))
    x0, x1, y0, y1 = _wide_scene(_clump_model(centers)).source_plane()[1]
    for cx, cy in centers:
        assert x0 < cx < x1 and y0 < cy < y1, (cx, cy, (x0, x1, y0, y1))


def test_autoframe_weighs_absolute_flux_not_signed():
    """A negative source frames exactly like its positive twin.

    Some sources here are genuinely negative (a slightly-negative Sersic under
    brighter shapelets, and one plane negative outright). Summing signed flux would
    give a cumulative that decreases, making every quantile edge meaningless.
    """
    pos = _wide_scene(_clump_model(((1.0, -0.5),), ie=5.0)).source_plane()[1]
    neg = _wide_scene(_clump_model(((1.0, -0.5),), ie=-5.0)).source_plane()[1]
    np.testing.assert_allclose(pos, neg)


def test_flat_plane_falls_back_to_the_cutout():
    """No flux to frame on -> the full window, not a degenerate box."""
    ext = _wide_scene(_clump_model(ie=0.0)).source_plane()[1]
    assert _half(ext) == pytest.approx(10.0)


def test_fov_full_restores_the_cutout_window():
    ext = _wide_scene(_clump_model()).source_plane(fov_arcsec="full")[1]
    assert ext == pytest.approx((-10.0, 10.0, -10.0, 10.0))


def test_explicit_fov_and_center_are_honoured():
    ext = _wide_scene(_clump_model()).source_plane(fov_arcsec=3.0, center=(1.0, -1.0))[1]
    assert ext == pytest.approx((-0.5, 2.5, -2.5, 0.5))


def test_locked_center_still_gets_an_auto_width():
    """An explicit center must not disable auto-sizing: the width still adapts, and
    must reach far enough from the given center to include the (offset) source."""
    ext = _wide_scene(_clump_model(((0.8, 0.0),))).source_plane(center=(0.0, 0.0))[1]
    assert ext[0] + ext[1] == pytest.approx(0.0)   # center kept exactly
    assert 0.8 < ext[1] < 10.0                     # source inside, still zoomed


def test_unknown_fov_keyword_is_rejected():
    with pytest.raises(ValueError, match="'full'"):
        _wide_scene(_clump_model()).source_plane(fov_arcsec="tight")


# ---------------------------------------------------------------------------
# The combined scene panel
# ---------------------------------------------------------------------------


def _titles(fig):
    return [ax.get_title() for ax in fig.axes if ax.get_title()]


def test_scene_panel_pairs_each_plane_with_its_band(scene):
    """One row per source plane, image plane and source plane side by side."""
    model, sims, params = scene
    fig = PosteriorReport(FixedParams(model, sims, params)).scene_panel(grid_pix=32)
    titles = _titles(fig)
    assert sum(t.startswith("Model") for t in titles) == 2
    assert sum(t.startswith("Source plane") for t in titles) == 2


def test_fixed_scene_titles_do_not_claim_a_median(scene):
    """``point`` defaults to "median" everywhere, and a fixed scene has no median.

    Rendering it into the title labels explicit construction parameters as a summary
    statistic of a posterior that was never sampled — the kind of caption that
    survives into a paper figure.
    """
    model, sims, params = scene
    rep = PosteriorReport(FixedParams(model, sims, params))
    for fig in (rep.scene_panel(grid_pix=24), rep.model_panel()):
        titles = _titles(fig)
        assert any("(fixed)" in t for t in titles), titles
        assert not any("median" in t for t in titles), titles


def test_scene_panel_renders_a_shared_band_once(scene):
    """A band seeing two planes contributes two rows but is simulated once.

    Rows are per plane, not per band, so the naive loop would re-render (and, on a
    fitted model, re-solve) the same image for every plane it carries.
    """
    model, _, params = scene
    fp = FixedParams(model, [SceneSimulator(model, _cfg())], params)  # sees everything
    calls = []
    inner = fp.simulate
    fp.simulate = lambda **kw: (calls.append(kw.get("dataset", 0)), inner(**kw))[1]
    fig = PosteriorReport(fp).scene_panel(grid_pix=24)
    assert sum(t.startswith("Model") for t in _titles(fig)) == 2
    assert calls == [0]


def test_scene_panel_keeps_a_band_with_no_lensed_source():
    """A lens-light-only band has no source plane, and must still get a row.

    Iterating source planes alone would drop that observation from the figure with
    no trace — the failure mode is silence, so the blank cell is the point.
    """
    model = _clump_model(lens_light=True)
    src = model.planes[1].light
    halo = model.planes[0].light
    sims = [SceneSimulator(model, _wide_cfg(), sees=src),
            SceneSimulator(model, _wide_cfg(), sees=halo)]
    fp = FixedParams(model, sims, model.to_params({}))
    titles = _titles(PosteriorReport(fp).scene_panel(grid_pix=24))
    assert sum(t.startswith("Model") for t in titles) == 2
    assert any("no lensed source light in band 1" in t for t in titles)


# ---------------------------------------------------------------------------
# The convenience wrapper
# ---------------------------------------------------------------------------


def test_plot_scene_returns_one_combined_figure(scene):
    model, sims, params = scene
    figs = plot_scene(model, sims, params, grid_pix=32)
    assert set(figs) == {"scene"}
    titles = _titles(figs["scene"])
    assert sum(t.startswith("Source plane") for t in titles) == 2


def test_plot_scene_can_still_split_the_panels(scene):
    model, sims, params = scene
    figs = plot_scene(model, sims, params, grid_pix=32, combined=False)
    assert set(figs) == {"model", "source"}
    # source_panel puts one row per source plane
    assert len(figs["source"].axes) >= 2


def test_plot_scene_accepts_per_view_framing(scene):
    model, sims, params = scene
    figs = plot_scene(model, sims, params, grid_pix=32,
                      fov_arcsec={1: 2.0}, center={2: (0.0, 0.0)})
    assert set(figs) == {"scene"}


def test_panel_titles_name_the_plane():
    """Panels identify the plane by name, not only by its index in redshift order.

    A source is a numbered object in a catalogue; "plane 4" is an accident of how the
    model happens to be sorted. The index stays because the per-view framing dicts
    key on it.
    """
    model = _clump_model()
    model.planes[1].name = "src9_z1.506"
    sims = [SceneSimulator(model, _wide_cfg(), sees=p.light)
            for p in model.planes if p.has_light]
    titles = _titles(PosteriorReport(FixedParams(model, sims,
                                                 model.to_params({}))).scene_panel(
        grid_pix=24))
    assert any("src9_z1.506" in t for t in titles), titles
    assert any("Source plane 1" in t for t in titles), titles
