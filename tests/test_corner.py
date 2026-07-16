"""Corner-plot column resolution and rendering.

Uses a stub posterior — a scene plus prior draws — so these run without an
inference context. What's under test is the selection / ordering / labelling /
truth alignment, all of which live above the sampler.
"""

import matplotlib
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

matplotlib.use("Agg")

from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.scene import Component, LensModel, Plane

from gigalens_research.param_index import param_sites
from gigalens_research.plotting.corner import _resolve_columns, plot_corner

tfd = tfp.distributions

Z_LENS, Z_SOURCE = 0.5, 2.0
N_SAMPLES = 200


def _c(profile, **priors):
    return Component(profile, priors)


class StubPosterior:
    """The corner layer's whole contract: a scene, and flat physical samples."""

    def __init__(self, scene, n=N_SAMPLES, seed=0):
        self.scene = scene
        rng = np.random.default_rng(seed)
        self.flat_x = {
            s.ukey: rng.normal(size=n) for s in param_sites(scene)
        }


@pytest.fixture(scope="module")
def scene():
    p0 = Plane(
        redshift=Z_LENS,
        mass=[
            _c(epl.EPL(), theta_E=tfd.Normal(1.5, 0.1), gamma=tfd.Normal(2.0, 0.1),
               e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
               center_x=0.0, center_y=0.0),
            _c(shear.Shear(), gamma1=tfd.Normal(0.0, 0.05),
               gamma2=tfd.Normal(0.0, 0.05)),
        ],
        light=[],
    )
    p1 = Plane(
        redshift=Z_SOURCE,
        mass=[],
        light=[
            _c(sersic.SersicEllipse(), R_sersic=tfd.Normal(0.3, 0.05),
               n_sersic=tfd.Normal(1.0, 0.2), Ie=tfd.Normal(2.0, 0.2),
               e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
               center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1)),
        ],
    )
    cosmo = _c(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE),
               H0=tfd.Uniform(60.0, 80.0), Om0=tfd.Uniform(0.1, 0.5),
               k=0.0, w0=tfd.Uniform(-1.5, -0.5))
    return LensModel(planes=[p0, p1], cosmo=cosmo)


@pytest.fixture(scope="module")
def post(scene):
    return StubPosterior(scene)


# --- column resolution (the backend seam) ----------------------------------


def test_default_resolves_every_parameter(post, scene):
    samples, labels, sites = _resolve_columns(post)
    assert samples.shape == (N_SAMPLES, scene.num_free_params)
    assert len(labels) == len(sites) == scene.num_free_params


def test_columns_are_ordered_cosmology_mass_light(post):
    _, _, sites = _resolve_columns(post)
    assert [s.kind for s in sites][:3] == ["cosmology"] * 3
    kinds = [s.kind for s in sites]
    assert kinds.index("mass") < kinds.index("light")


def test_kind_filter_narrows_columns(post):
    samples, labels, sites = _resolve_columns(post, kind="cosmology")
    assert samples.shape == (N_SAMPLES, 3)
    assert labels == [r"$H_0$", r"$\Omega_m$", r"$w_0$"]


def test_plane_filter_narrows_columns(post):
    _, _, sites = _resolve_columns(post, plane=1)
    assert {s.plane for s in sites} == {1}


def test_filters_and(post):
    _, _, sites = _resolve_columns(post, kind="mass", component=("mass", 1))
    assert {s.param for s in sites} == {"gamma1", "gamma2"}


def test_samples_follow_their_column(post):
    """Guards the failure that matters most: right labels, wrong data."""
    _, _, all_sites = _resolve_columns(post)
    _, _, cosmo_sites = _resolve_columns(post, kind="cosmology")
    samples, _, _ = _resolve_columns(post, kind="cosmology")
    for i, s in enumerate(cosmo_sites):
        np.testing.assert_allclose(samples[:, i], post.flat_x[s.ukey])


def test_plot_params_takes_explicit_keys_in_order(post):
    keys = ["planes/1/light/0/Ie", "cosmo/H0"]
    samples, _, sites = _resolve_columns(post, plot_params=keys)
    assert [s.key for s in sites] == keys
    np.testing.assert_allclose(samples[:, 0], post.flat_x["planes/1/light/0/Ie"])


def test_plot_params_rejects_unknown_key(post):
    with pytest.raises(KeyError, match="does not have"):
        _resolve_columns(post, plot_params=["src_R_sersic"])  # a retired label


def test_plot_params_and_filters_are_mutually_exclusive(post):
    with pytest.raises(ValueError, match="not both"):
        _resolve_columns(post, plot_params=["cosmo/H0"], kind="mass")


# --- rendering -------------------------------------------------------------


def test_plot_corner_draws_default(post, scene):
    fig = plot_corner(post)
    n = scene.num_free_params
    assert len(fig.axes) == n * n
    matplotlib.pyplot.close(fig)


def test_plot_corner_draws_filtered(post):
    fig = plot_corner(post, kind="cosmology")
    assert len(fig.axes) == 3 * 3
    matplotlib.pyplot.close(fig)


def test_plot_corner_with_scene_nested_truth(post):
    truth = {"cosmo": {"H0": 70.0, "Om0": 0.3, "w0": -1.0}}
    fig = plot_corner(post, kind="cosmology", truth=truth)
    assert len(fig.axes) == 9
    matplotlib.pyplot.close(fig)


def test_plot_corner_with_partial_truth_warns_but_draws(post):
    with pytest.warns(UserWarning, match="does not define"):
        fig = plot_corner(post, kind="cosmology", truth={"cosmo/H0": 70.0})
    matplotlib.pyplot.close(fig)
