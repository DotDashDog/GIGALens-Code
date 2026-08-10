"""Params-tree keys: reading a scene params dict, and building one.

A scene params tree is keyed by NAME where the scene names a plane/component and by
``str(index)`` where it does not (gigalens ``scene.py`` §names, ``cc5a078``). Callers
hold integer positions — ``enumerate(model.planes)``, the simulator's ``_light`` list,
a point-source term's ``lens_i`` — so every one of them needs a translation, and
:mod:`gigalens_research.inference_utils.params` owns it.

The build side is what these tests exist for. A *reader* that uses the wrong key form
raises ``KeyError`` and is found immediately; a *builder* that writes ``0`` where the
tree already holds ``"0"`` raises nothing at all. It produces a dict carrying both,
which JAX then refuses to flatten — far from the line that caused it — and, against a
named scene, silently orphans the block from the name it was supposed to fill.
"""

import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.scene import Component, LensModel, Plane

from gigalens_research.inference_utils.params import (
    component_key,
    component_params,
    mass_params_list,
    plane_key,
    plane_params,
    truth_x_to_scene_params,
)

tfd = tfp.distributions


def _epl():
    return dict(theta_E=tfd.Normal(1.5, 0.1), gamma=tfd.Normal(2.0, 0.1),
                e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)


def _ser():
    return dict(R_sersic=tfd.Normal(1.0, 0.1), n_sersic=tfd.Normal(2.0, 0.1),
                Ie=tfd.Normal(1.0, 0.1), e1=0.0, e2=0.0,
                center_x=0.0, center_y=0.0)


def _build(named: bool) -> LensModel:
    """Lens plane: EPL + shear, plus its own light. Source plane: one Sersic."""
    n = (lambda s: s) if named else (lambda s: None)
    return LensModel([
        Plane(
            name=n("lens"),
            deflection_ratio=None,
            mass=[
                Component(epl.EPL(20), _epl(), name=n("host")),
                Component(shear.Shear(),
                          dict(gamma1=tfd.Normal(0.0, 0.05),
                               gamma2=tfd.Normal(0.0, 0.05)),
                          name=n("ext")),
            ],
            light=[Component(sersic.SersicEllipse(), _ser(), name=n("halo"))],
        ),
        Plane(
            name=n("src"),
            deflection_ratio=1.0,
            light=[Component(sersic.SersicEllipse(), _ser(), name=n("arc"))],
        ),
    ])


@pytest.fixture(params=[False, True], ids=["unnamed", "named"])
def model(request):
    return _build(request.param)


def _sample_params(model):
    return model.to_params(model.prior.sample(seed=jax.random.PRNGKey(0)))


# --- reading ---------------------------------------------------------------------


def test_keys_are_the_scene_s_own(model):
    """``plane_key``/``component_key`` agree with the tree the scene actually emits."""
    params = _sample_params(model)
    assert set(params["planes"]) == {plane_key(model, i)
                                     for i in range(len(model.planes))}
    lens = params["planes"][plane_key(model, 0)]
    assert set(lens["mass"]) == {component_key(model, 0, "mass", j) for j in range(2)}


def test_names_appear_in_the_keys_when_named():
    """The name IS the key; without one the key is the index as a string."""
    named, unnamed = _sample_params(_build(True)), _sample_params(_build(False))
    assert set(named["planes"]) == {"lens", "src"}
    assert set(named["planes"]["lens"]["mass"]) == {"host", "ext"}
    assert set(unnamed["planes"]) == {"0", "1"}
    assert set(unnamed["planes"]["0"]["mass"]) == {"0", "1"}


def test_accessors_reach_the_same_leaf(model):
    params = _sample_params(model)
    direct = params["planes"][plane_key(model, 0)]["mass"][
        component_key(model, 0, "mass", 0)]
    assert component_params(model, params, 0, "mass", 0) is direct
    assert plane_params(model, params, 0) is params["planes"][plane_key(model, 0)]


def test_mass_params_list_is_ordered_by_position(model):
    """gigalens' point-source helpers pair ``mass_params[j]`` with
    ``mass_profiles[j]`` positionally, so this must follow the scene's component
    ORDER — not the sort order of the keys, which naming makes arbitrary."""
    params = _sample_params(model)
    lst = mass_params_list(model, params, 0, 2)
    assert [id(p) for p in lst] == [
        id(component_params(model, params, 0, "mass", j)) for j in range(2)
    ]
    # The EPL is component 0 and the shear component 1; their params tell them apart.
    assert "theta_E" in lst[0] and "gamma1" in lst[1]


# --- building --------------------------------------------------------------------


def _legacy_truth():
    """A persisted OLD 3-group truth: the form ``truth_x_to_scene_params`` adapts."""
    return {
        "lens_mass": {
            "0": dict(theta_E=1.4, gamma=2.05, e1=0.0, e2=0.0,
                      center_x=0.0, center_y=0.0),
            "1": dict(gamma1=0.01, gamma2=-0.02),
        },
        "lens_light": {
            "0": dict(R_sersic=0.9, n_sersic=3.0, Ie=1.1, e1=0.0, e2=0.0,
                      center_x=0.0, center_y=0.0),
        },
        "source_light": {
            "0": dict(R_sersic=0.3, n_sersic=1.2, Ie=2.0, e1=0.0, e2=0.0,
                      center_x=0.05, center_y=-0.05),
        },
    }


def test_adapted_truth_uses_the_model_s_keys(model):
    out = truth_x_to_scene_params(_legacy_truth(), model)
    assert set(out["planes"]) >= {plane_key(model, i)
                                  for i in range(len(model.planes))}
    lens = out["planes"][plane_key(model, 0)]
    assert set(lens["mass"]) == {component_key(model, 0, "mass", j) for j in range(2)}
    assert set(lens["light"]) == {component_key(model, 0, "light", 0)}


def test_adapted_truth_is_flattenable(model):
    """The regression guard for the silent failure.

    Writing an integer key onto a string-keyed tree does not raise where it happens.
    It leaves ``0`` sitting beside ``"0"``, and JAX cannot even sort the keys of such
    a dict — so the model blows up at the first ``tree_map``, with nothing pointing
    back at the adapter.
    """
    out = truth_x_to_scene_params(_legacy_truth(), model)
    leaves = jax.tree_util.tree_leaves(out)
    assert leaves, "expected parameter leaves"
    for node in [out["planes"]] + list(out["planes"].values()):
        kinds = {type(k) for k in node}
        assert len(kinds) <= 1, f"mixed key types {kinds} — JAX cannot flatten this"


def test_adapted_truth_routes_values_by_role_and_index(model):
    """Source light must land on the source plane, lens light on the lens plane —
    the mapping the adapter exists to perform."""
    out = truth_x_to_scene_params(_legacy_truth(), model)
    lens_light = out["planes"][plane_key(model, 0)]["light"][
        component_key(model, 0, "light", 0)]
    src_light = out["planes"][plane_key(model, 1)]["light"][
        component_key(model, 1, "light", 0)]
    assert float(lens_light["R_sersic"]) == pytest.approx(0.9)
    assert float(src_light["R_sersic"]) == pytest.approx(0.3)
    mass0 = out["planes"][plane_key(model, 0)]["mass"][
        component_key(model, 0, "mass", 0)]
    assert float(mass0["theta_E"]) == pytest.approx(1.4)


def test_adapted_truth_feeds_fix_to(model):
    """End to end: the adapted dict is consumed by the API it is built for."""
    out = truth_x_to_scene_params(_legacy_truth(), model)
    fixed = model.fix_to(out, free=[model.planes[1].light[0]])
    params = fixed.to_params(fixed.prior.sample(seed=jax.random.PRNGKey(0)))
    mass0 = params["planes"][plane_key(fixed, 0)]["mass"][
        component_key(fixed, 0, "mass", 0)]
    assert float(jnp.squeeze(mass0["theta_E"])) == pytest.approx(1.4)
