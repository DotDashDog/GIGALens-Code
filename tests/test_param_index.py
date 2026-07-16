"""Tests for the path-space parameter index.

The fixture scene is deliberately awkward: three planes, two mass components on
one of them, a free redshift, a cosmology, and a shared parameter — i.e. every
structure the retired 3-group label space could not represent.
"""

import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.scene import Component, LensModel, Plane, shared

from gigalens_research.param_index import (
    KINDS,
    kind_of_key,
    param_of_key,
    param_sites,
    path_of_key,
    select_sites,
    site_labels,
    sites_to_matrix,
    truth_row,
)

tfd = tfp.distributions

Z_LENS, Z_SOURCE = 0.5, 2.0


def _c(profile, **priors):
    return Component(profile, priors)


@pytest.fixture(scope="module")
def scene():
    """Plane 0: EPL + shear + sersic light. Plane 1: free redshift, one EPL.
    Plane 2: sersic light (lensed). Plus a wCDM cosmology."""
    p0 = Plane(
        redshift=Z_LENS,
        mass=[
            _c(epl.EPL(), theta_E=tfd.Normal(1.5, 0.1), gamma=tfd.Normal(2.0, 0.1),
               e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
               center_x=0.0, center_y=0.0),
            _c(shear.Shear(), gamma1=tfd.Normal(0.0, 0.05),
               gamma2=tfd.Normal(0.0, 0.05)),
        ],
        light=[
            _c(sersic.SersicEllipse(), R_sersic=tfd.Normal(1.0, 0.1),
               n_sersic=tfd.Normal(4.0, 0.5), Ie=tfd.Normal(1.0, 0.1),
               e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
               center_x=0.0, center_y=0.0),
        ],
    )
    p1 = Plane(
        redshift=tfd.Uniform(0.8, 1.2),  # free -> a geometry parameter
        mass=[
            _c(epl.EPL(), theta_E=tfd.Normal(0.5, 0.1), gamma=tfd.Normal(2.0, 0.1),
               e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
               center_x=0.0, center_y=0.0),
        ],
        light=[],
    )
    p2 = Plane(
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
    return LensModel(planes=[p0, p1, p2], cosmo=cosmo)


@pytest.fixture(scope="module")
def sites(scene):
    return param_sites(scene)


# --- the index itself ------------------------------------------------------


def test_one_record_per_free_parameter(scene, sites):
    # The scene counts a shared param once; so must we. Any mismatch means we
    # either dropped a column or fanned one out into duplicates.
    assert len(sites) == scene.num_free_params


def test_constants_are_not_parameters(sites):
    # Plane 0's light center_x/center_y are fixed at 0.0 in the fixture.
    keys = {s.key for s in sites}
    assert "planes/0/light/0/center_x" not in keys
    assert "planes/2/light/0/center_x" in keys  # free on plane 2


def test_kinds_are_in_panel_order(sites):
    """Cosmology, then geometry, then mass, then light — the ordering contract."""
    ranks = [KINDS.index(s.kind) for s in sites]
    assert ranks == sorted(ranks)
    assert [s.key for s in sites if s.kind == "cosmology"] == [
        "cosmo/H0", "cosmo/Om0", "cosmo/w0",
    ]


def test_sorted_by_plane_then_component_within_kind(sites):
    mass = [s for s in sites if s.kind == "mass"]
    assert [(s.plane, s.component) for s in mass] == sorted(
        (s.plane, s.component) for s in mass
    )


def test_records_locate_each_parameter(sites):
    by_key = {s.key: s for s in sites}
    theta = by_key["planes/1/mass/0/theta_E"]
    assert (theta.kind, theta.plane, theta.component, theta.param) == (
        "mass", 1, 0, "theta_E",
    )
    z = by_key["planes/1/geometry/redshift"]
    assert (z.kind, z.plane, z.param) == ("geometry", 1, "redshift")
    h0 = by_key["cosmo/H0"]
    assert (h0.kind, h0.plane, h0.param) == ("cosmology", None, "H0")


def test_plane_and_component_survive_multiplane(sites):
    """The point of the migration: theta_E on plane 0 and on plane 1 stay
    distinguishable. The old space collapsed both to a global running index."""
    thetas = [s for s in sites if s.param == "theta_E"]
    assert {s.plane for s in thetas} == {0, 1}
    assert len({s.key for s in thetas}) == 2


# --- key helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "key,kind",
    [
        ("cosmo/H0", "cosmology"),
        ("planes/1/geometry/redshift", "geometry"),
        ("planes/0/mass/1/gamma1", "mass"),
        ("planes/2/light/0/R_sersic", "light"),
        ("shared_7", None),  # synthetic key, not a site path
    ],
)
def test_kind_of_key(key, kind):
    assert kind_of_key(key) == kind


def test_path_of_key_roundtrips(sites):
    for s in sites:
        assert path_of_key(s.key) == s.paths[0]


def test_param_of_key():
    assert param_of_key("planes/0/mass/0/theta_E") == "theta_E"
    assert param_of_key("theta_E") == "theta_E"


# --- selection -------------------------------------------------------------


def test_default_selects_everything(sites):
    assert select_sites(sites) == list(sites)


def test_select_by_kind(sites):
    assert [s.key for s in select_sites(sites, kind="cosmology")] == [
        "cosmo/H0", "cosmo/Om0", "cosmo/w0",
    ]
    both = select_sites(sites, kind=["cosmology", "mass"])
    assert {s.kind for s in both} == {"cosmology", "mass"}


def test_select_by_plane(sites):
    got = select_sites(sites, kind="mass", plane=1)
    assert {s.key for s in got} == {
        "planes/1/mass/0/theta_E", "planes/1/mass/0/gamma",
        "planes/1/mass/0/e1", "planes/1/mass/0/e2",
    }


def test_select_by_component_role_pair(sites):
    # The shear is plane 0's mass component 1.
    got = select_sites(sites, component=("mass", 1))
    assert {s.param for s in got} == {"gamma1", "gamma2"}


def test_select_filters_are_anded(sites):
    got = select_sites(sites, kind="light", plane=2)
    assert {s.plane for s in got} == {2}
    assert {s.kind for s in got} == {"light"}


def test_select_callable_escape_hatch(sites):
    got = select_sites(sites, select=lambda s: s.param.startswith("e"))
    assert {s.param for s in got} == {"e1", "e2"}


def test_empty_selection_raises_rather_than_drawing_nothing(sites):
    with pytest.raises(ValueError, match="no parameters matched"):
        select_sites(sites, kind="mass", plane=2)  # plane 2 has no mass


def test_unknown_kind_raises(sites):
    with pytest.raises(ValueError, match="unknown kind"):
        select_sites(sites, kind="lens_light")  # a retired group name


# --- labels ----------------------------------------------------------------


def test_labels_are_unique(sites):
    assert len(set(site_labels(sites))) == len(sites)


def test_unique_param_renders_bare(sites):
    """center_x is free only on plane 2, so nothing collides with it."""
    labels = dict(zip([s.key for s in sites], site_labels(sites)))
    assert labels["planes/2/light/0/center_x"] == r"$x$"
    assert labels["cosmo/H0"] == r"$H_0$"
    assert labels["planes/0/mass/1/gamma1"] == r"$\gamma_{\rm ext,1}$"


def test_colliding_param_gains_plane_and_component(sites):
    labels = dict(zip([s.key for s in sites], site_labels(sites)))
    assert labels["planes/0/mass/0/theta_E"] == r"$\theta_E^{(0,m0)}$"
    assert labels["planes/1/mass/0/theta_E"] == r"$\theta_E^{(1,m0)}$"


def test_suffix_carries_role_because_mass_and_light_index_separately(sites):
    """Plane 0's mass[0] and light[0] are both "component 0"; e1 exists on both.
    A bare (plane, component) tag would render them identically."""
    labels = dict(zip([s.key for s in sites], site_labels(sites)))
    assert labels["planes/0/mass/0/e1"] == r"$\epsilon_1^{(0,m0)}$"
    assert labels["planes/0/light/0/e1"] == r"$\epsilon_1^{(0,l0)}$"


def test_labels_depend_on_selection(sites):
    """Plotting one plane's mass drops the collision, so the label goes bare."""
    one = select_sites(sites, kind="mass", plane=1)
    labels = dict(zip([s.key for s in one], site_labels(one)))
    assert labels["planes/1/mass/0/theta_E"] == r"$\theta_E$"


def test_plain_labels_when_latex_off(sites):
    one = select_sites(sites, kind="cosmology")
    assert site_labels(one, latex=False) == ["H0", "Om0", "w0"]


# --- shared parameters -----------------------------------------------------


@pytest.fixture(scope="module")
def shared_scene():
    """One EPL slope shared across two planes: ONE free parameter, two sites."""
    g = shared(tfd.Normal(2.0, 0.1))
    p0 = Plane(
        deflection_ratio=None,
        mass=[_c(epl.EPL(), theta_E=tfd.Normal(1.5, 0.1), gamma=g,
                 e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)],
    )
    p1 = Plane(
        deflection_ratio=1.0,
        mass=[_c(epl.EPL(), theta_E=tfd.Normal(0.5, 0.1), gamma=g,
                 e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)],
        light=[],
    )
    return LensModel(planes=[p0, p1])


def test_shared_param_is_one_column_not_two(shared_scene):
    sites = param_sites(shared_scene)
    assert len(sites) == shared_scene.num_free_params
    gammas = [s for s in sites if s.param == "gamma"]
    assert len(gammas) == 1, "a shared param must not fan out into duplicate columns"
    g = gammas[0]
    assert g.shared and g.planes == {0, 1} and g.plane is None


def test_shared_param_matches_either_plane(shared_scene):
    sites = param_sites(shared_scene)
    for plane in (0, 1):
        assert "gamma" in {s.param for s in select_sites(sites, plane=plane)}


# --- pulling values --------------------------------------------------------


def _fake_flat(sites, value=1.0):
    return {s.ukey: np.full((3,), value) for s in sites}


def test_sites_to_matrix_shape_and_order(sites):
    x = {s.ukey: np.arange(3, dtype=float) + i for i, s in enumerate(sites)}
    m = sites_to_matrix(sites, x)
    assert m.shape == (3, len(sites))
    np.testing.assert_allclose(m[:, 5], np.arange(3, dtype=float) + 5)


def test_sites_to_matrix_reports_a_foreign_scene(sites):
    with pytest.raises(KeyError, match="same scene"):
        sites_to_matrix(sites, {"nonsense": np.zeros(3)})


# --- truth alignment -------------------------------------------------------


def test_truth_row_from_scene_nested(sites):
    truth = {
        "planes": {1: {"mass": {0: {"theta_E": 0.55}}}},
        "cosmo": {"H0": 70.0},
    }
    row = truth_row(sites, truth, warn=False)
    by_key = {s.key: i for i, s in enumerate(sites)}
    assert row[by_key["cosmo/H0"]] == 70.0
    assert row[by_key["planes/1/mass/0/theta_E"]] == 0.55


def test_truth_row_from_path_keys(sites):
    truth = {"cosmo/H0": 70.0, "planes/0/mass/0/theta_E": 1.51}
    row = truth_row(sites, truth, warn=False)
    by_key = {s.key: i for i, s in enumerate(sites)}
    assert row[by_key["cosmo/H0"]] == 70.0
    assert row[by_key["planes/0/mass/0/theta_E"]] == 1.51


def test_truth_row_nan_fills_what_the_truth_omits(sites):
    row = truth_row(sites, {"cosmo/H0": 70.0}, warn=False)
    assert np.isfinite(row).sum() == 1
    assert np.isnan(row).sum() == len(sites) - 1


def test_truth_row_warns_about_omissions(sites):
    with pytest.warns(UserWarning, match="does not define"):
        truth_row(sites, {"cosmo/H0": 70.0})


def test_truth_row_tolerates_stringified_keys(sites):
    truth = {"planes": {"1": {"mass": {"0": {"theta_E": 0.55}}}}}
    row = truth_row(sites, truth, warn=False)
    by_key = {s.key: i for i, s in enumerate(sites)}
    assert row[by_key["planes/1/mass/0/theta_E"]] == 0.55


def test_legacy_3group_truth_is_rejected_with_guidance(sites):
    """A silently-misaligned truth overlay is worse than a loud failure: the old
    3-group form cannot say which plane a parameter was on."""
    legacy = {"lens_mass": {"0": {"theta_E": 1.5}}, "lens_light": {},
              "source_light": {}}
    with pytest.raises(ValueError, match="retired 3-group label space"):
        truth_row(sites, legacy)
