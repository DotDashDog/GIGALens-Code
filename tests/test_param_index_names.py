"""Scene Component/Plane names reaching the keys and the display labels.

Names ARE the params-tree keys (gigalens ``scene.py`` §names, ``cc5a078``): a named
plane keys as ``planes/lens/...`` where an unnamed one keys as ``planes/0/...``. What
naming must *not* do is move a parameter — gigalens orders flat-z columns by
derivation rather than by key text precisely so that a rename changes column names
without permuting the vector, and this index inherits that guarantee.

So the contract is pinned from three sides:

- an unnamed scene labels exactly as it did before names existed;
- a named one substitutes the name into both the key and the tag, while keeping every
  disambiguation the positional tag provided;
- naming changes the keys and *only* the keys — same parameters, same order, same
  planes and components, same kinds.
"""

import matplotlib

matplotlib.use("Agg")
import pytest
import tensorflow_probability.substrates.jax as tfp
from matplotlib import mathtext

from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.scene import Component, LensModel, Plane, shared

from gigalens_research.param_index import param_sites, select_sites, site_labels

tfd = tfp.distributions


def _epl():
    return dict(
        theta_E=tfd.Normal(1.5, 0.1),
        gamma=tfd.Uniform(1.5, 2.5),
        e1=tfd.Normal(0.0, 0.1),
        e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.1),
        center_y=tfd.Normal(0.0, 0.1),
    )


def _ser():
    return dict(
        R_sersic=tfd.Uniform(0.1, 1.0),
        n_sersic=tfd.Uniform(0.5, 4.0),
        e1=tfd.Normal(0.0, 0.1),
        e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.1),
        center_y=tfd.Normal(0.0, 0.1),
    )


def _build(named: bool):
    """The same scene twice. The lens plane carries a mass AND a light both called
    ``host`` — one galaxy's two aspects, which the (plane, kind) uniqueness rule
    exists to allow."""
    n = (lambda s: s) if named else (lambda s: None)
    return LensModel(
        [
            Plane(
                name=n("lens"),
                mass=[
                    Component(epl.EPL(20), _epl(), name=n("host")),
                    Component(
                        shear.Shear(),
                        dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05)),
                        name=n("ext"),
                    ),
                ],
                light=[
                    Component(sersic.SersicEllipse(use_lstsq=True), _ser(), name=n("host"))
                ],
            ),
            Plane(
                name=n("src"),
                light=[
                    Component(sersic.SersicEllipse(use_lstsq=True), _ser(), name=n("arc"))
                ],
            ),
        ]
    )


def _labels(model, latex=False):
    sites = param_sites(model)
    return dict(zip((s.key for s in sites), site_labels(sites, latex=latex)))


def test_unnamed_scene_labels_are_unchanged():
    """The positional tag is what a scene without names has always rendered."""
    lab = _labels(_build(False))
    assert lab["planes/0/mass/0/center_x"] == "center_x (0,m0)"
    assert lab["planes/0/light/0/center_x"] == "center_x (0,l0)"
    assert lab["planes/1/light/0/center_x"] == "center_x (1,l0)"
    # gamma1 is unique to the shear, so it collides with nothing and stays bare.
    assert lab["planes/0/mass/1/gamma1"] == "gamma1"


def test_names_replace_indices_in_the_key():
    """The name IS the key segment, in place of the positional fallback."""
    named = {s.key for s in param_sites(_build(True))}
    unnamed = {s.key for s in param_sites(_build(False))}
    assert "planes/lens/mass/host/center_x" in named
    assert "planes/src/light/arc/center_x" in named
    assert "planes/0/mass/0/center_x" in unnamed
    assert not any(k.startswith("planes/0/") for k in named)


def test_names_replace_indices_in_the_tag():
    lab = _labels(_build(True))
    assert lab["planes/lens/mass/host/center_x"] == "center_x (lens,m:host)"
    assert lab["planes/src/light/arc/center_x"] == "center_x (src,l:arc)"


def test_mass_and_light_sharing_a_name_stay_separable():
    """The role letter is load-bearing: names are unique only per (plane, kind).

    Note the two collide in the KEY space too — ``planes/lens/mass/host`` and
    ``planes/lens/light/host`` differ only in the kind segment, which is why gigalens
    always renders ``kind`` and why the tag keeps the ``m``/``l`` letter.
    """
    lab = _labels(_build(True))
    assert lab["planes/lens/mass/host/center_x"] == "center_x (lens,m:host)"
    assert lab["planes/lens/light/host/center_x"] == "center_x (lens,l:host)"
    assert len(set(lab.values())) == len(lab), "labels must stay unique"


def test_naming_changes_the_keys_and_only_the_keys():
    """A rename must not move a parameter.

    gigalens orders flat-z columns by derivation rather than by sorting key text, so
    naming may rewrite every key without permuting a single column. This is the
    property that makes names safe to add to a live model — and the one that would
    silently invalidate every saved chain if it ever regressed.
    """
    a, b = param_sites(_build(False)), param_sites(_build(True))
    assert [s.kind for s in a] == [s.kind for s in b]
    assert [s.param for s in a] == [s.param for s in b]
    assert [s.planes for s in a] == [s.planes for s in b]
    assert [s.components for s in a] == [s.components for s in b]
    # The keys, and only the keys, differ.
    assert [s.key for s in a] != [s.key for s in b]


def test_names_are_selectable_by_name_and_by_index():
    """A named plane/component stays reachable positionally: the name is an added
    identity, not a replacement for the index."""
    sites = param_sites(_build(True))
    by_name = select_sites(sites, plane="lens")
    by_index = select_sites(sites, plane=0)
    assert [s.key for s in by_name] == [s.key for s in by_index]
    host_mass = select_sites(sites, component=("mass", "host"))
    assert {s.key for s in host_mass} == {
        s.key for s in select_sites(sites, kind="mass", plane="lens", component=0)
    }


def test_uncollided_labels_still_render_bare():
    """Naming must not start decorating parameters that need no disambiguation."""
    lab = _labels(_build(True))
    assert lab["planes/lens/mass/host/theta_E"] == "theta_E"


# -- the mathtext trap ----------------------------------------------------------------
def test_underscored_names_are_escaped_for_math_mode():
    """An unescaped ``_`` is the subscript operator, so ``host_light`` would render as
    ``host`` subscript ``light`` — silently, without raising. Names like
    ``main_deflector`` are the norm, so this is the common case."""
    m = LensModel(
        [
            Plane(name="lens_plane", mass=[Component(epl.EPL(20), _epl(), name="main_deflector")]),
            Plane(
                name="src_plane",
                light=[Component(sersic.SersicEllipse(use_lstsq=True), _ser(), name="host_galaxy")],
            ),
        ]
    )
    latex = [l for l in _labels(m, latex=True).values() if "^" in l]
    assert latex, "expected at least one decorated label"
    for l in latex:
        assert "\\_" in l, f"underscore not escaped: {l}"
        assert "_{" not in l.split("^")[1], f"name subscripted: {l}"
        # and it is still valid mathtext
        mathtext.MathTextParser("agg").parse(l, dpi=72, prop=None)


def test_shared_param_name_is_recorded():
    te = shared(tfd.Normal(1.5, 0.1), name="einstein_radius")
    p1, p2 = _epl(), _epl()
    p1["theta_E"] = te
    p2["theta_E"] = te
    m = LensModel(
        [
            Plane(
                name="lens",
                mass=[
                    Component(epl.EPL(20), p1, name="a"),
                    Component(epl.EPL(20), p2, name="b"),
                ],
            )
        ]
    )
    # A NAMED handle keys as ``shared/<name>``; only an unnamed one is ``shared_0007``.
    shared_sites = [s for s in param_sites(m)
                    if s.ukey.startswith(("shared_", "shared/"))]
    assert len(shared_sites) == 1
    assert shared_sites[0].ukey == "shared/einstein_radius"
    assert shared_sites[0].group_name == "einstein_radius"
