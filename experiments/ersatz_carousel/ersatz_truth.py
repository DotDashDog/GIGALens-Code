#!/usr/bin/env python
"""Simulate the carousel's ersatz data straight from a new-API params dump.

This is the replacement for the ``translate_old_params.build`` +
``real_datasets.real_simulators_for_model`` pair the earlier notebooks used. That pair
existed to get an OLD-API MAP JSON into new-API shape, and most of its bulk is the
translation: inverting ``CombinedProfile`` suffixes, re-deriving shared parameters,
mapping profile classes by list position. ``improved_sersic_carousel.json`` is already
a new-API ``model.to_params(...)`` dump -- planes keyed by name, Components keyed by
name, every parameter under its own key -- so none of that work is left. What remains
is only:

* read the dump,
* recognise each Component's profile from the parameter names it carries,
* hand each source plane its own real cutout's PSF, background RMS and exposure time,
* render, and noise each render from its OWN flux.

Everything here is a fully-fixed (zero free parameter) model: the "truth" the mock is
drawn from. The model you FIT with stays where it was -- ``ersatz_carousel_prior_*.py``
-- and the two are deliberately different objects. In particular the fitting model's
Sersics are ``use_lstsq=True`` and so carry no ``Ie`` at all (linear amplitudes are
solved, not sampled), while the truth model's must carry one, since a simulation with
no amplitude has no flux. That is why :func:`truth_params` keeps ``Ie`` and the fitting
side drops it (the notebook's ``remove_key(p, 'Ie')``).

Two things are checked rather than assumed, both of them silent failures otherwise:

**Profile identity.** A Component is matched to a profile class by its exact set of
parameter names, against the classes' own ``.params`` -- never by position in a list.
An unmatched or ambiguous set raises.

**Plane / cutout pairing.** Source planes are ordered by redshift and the cutouts are
named by source number, and the two orders agree for only three of nine (see
``translate_old_params.cutout_extensions``). Every cutout here is looked up through
``source_id(plane redshift)``, and each plane's own NAME is asserted to be the one that
redshift implies -- which is what catches a source6/source9-style relabel. Get this
wrong and all nine planes still render, six of them with another source's seeing and
noise model, and nothing anywhere complains.

Usage::

    from ersatz_truth import truth_params, build_truth_model, simulate_ersatz

    p = truth_params("improved_sersic_carousel.json")
    truth = build_truth_model(p)
    obs = simulate_ersatz(truth, p, "real_cutouts", seed=0, supersample=16)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import jax
import numpy as np

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.cosmo import w0waCDM_Cosmo
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE, NFW_ELLIPSE_EINSTEIN
from gigalens.jax.profiles.mass.nfw_ellipse_slope import NFW_ELLIPSE_SLOPE
from gigalens.jax.profiles.mass.piemd import DPIE
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse

from translate_old_params import STRUCTURE, source_id
from real_datasets import load_real_cutout

#: Reduced-deflection normalisation theta_E is quoted against (lenstronomy's
#: z_source_convention). Not recoverable from a params dump -- it is a property of the
#: model that produced it -- so it comes from STRUCTURE, the same place the fitting
#: priors get it.
Z_SOURCE_REF = STRUCTURE["z_source_ref"]

#: Profile classes a dumped Component may be. Matched by parameter-name set only, so
#: adding a class here is enough to support it -- provided its set stays unique, which
#: :func:`_profile_table` asserts.
_MASS_CANDIDATES = (NFW_ELLIPSE_EINSTEIN, NFW_ELLIPSE_SLOPE, NFW_ELLIPSE, DPIE, Shear, EPL)
#: ``use_lstsq=False``: the truth model samples nothing, so its amplitudes are given,
#: not solved. See the module docstring.
_LIGHT_CANDIDATES = ((SersicEllipse, dict(use_lstsq=False)),)


def _profile_table(candidates):
    """{frozenset(param names): (cls, kwargs)}, raising if two classes collide."""
    table = {}
    for entry in candidates:
        cls, kwargs = entry if isinstance(entry, tuple) else (entry, {})
        key = frozenset(cls(**kwargs).params)
        if key in table:
            raise ValueError(
                f"{cls.__name__} and {table[key][0].__name__} have identical parameter "
                "sets; they cannot be told apart from a params dump. Match on something "
                "other than parameter names, or drop one from the candidate list.")
        table[key] = (cls, kwargs)
    return table


_MASS_BY_PARAMS = _profile_table(_MASS_CANDIDATES)
_LIGHT_BY_PARAMS = _profile_table(_LIGHT_CANDIDATES)


def _component(name: str, values: Dict[str, float], table, kind: str) -> Component:
    """One fully-fixed Component, its profile identified by ``values``' key set."""
    key = frozenset(values)
    match = table.get(key)
    if match is None:
        known = "\n  ".join(
            f"{cls.__name__}: {sorted(k)}" for k, (cls, _) in table.items())
        raise ValueError(
            f"{kind} Component {name!r} has parameters {sorted(key)}, which match no "
            f"known profile. Candidates:\n  {known}\n"
            "Add the profile to the candidate list rather than guessing.")
    cls, kwargs = match
    return Component(cls(**kwargs), {k: float(v) for k, v in values.items()}, name=name)


# --------------------------------------------------------------------------------
def truth_params(path: str) -> Dict[str, Any]:
    """Read the cleaned params dump and check its plane names against its redshifts.

    The name check is the load-bearing one: a plane whose name disagrees with its
    redshift pairs that source with another's cutout downstream, invisibly.
    """
    with open(path) as f:
        params = json.load(f)

    for expected in ("cosmo", "planes"):
        if expected not in params:
            raise ValueError(f"{path}: no {expected!r} key; this is not a to_params dump")

    for name, plane in params["planes"].items():
        z = plane["geometry"]["redshift"]
        if "light" not in plane:            # the cluster (mass-only) plane
            continue
        sid = source_id(z)                  # raises on an unknown redshift
        if name != f"source{sid}":
            raise ValueError(
                f"{path}: plane {name!r} sits at z = {z:g}, which is canonically source "
                f"{sid} (translate_old_params.STRUCTURE['source_ids']) -- expected the "
                f"plane to be named 'source{sid}'. A name/redshift swap here hands this "
                "plane another source's cutout, PSF and noise model, and every panel "
                "still renders.")
    return params


def build_truth_model(params: Dict[str, Any], *, z_source_ref: float = Z_SOURCE_REF,
                      unconstrain: str = "gaussian") -> LensModel:
    """A fully-fixed ``LensModel`` whose ``to_params({})`` reproduces ``params``.

    Every value becomes a constant, so the model has no free parameters and no priors:
    it exists to render one specific scene, not to be sampled. ``z_lens`` for the
    cosmology is taken from the mass-bearing plane's own redshift.
    """
    planes, z_lens = [], None
    for name, entry in sorted(params["planes"].items(),
                              key=lambda kv: kv[1]["geometry"]["redshift"]):
        z = float(entry["geometry"]["redshift"])
        if "mass" in entry:
            if z_lens is not None:
                raise ValueError(
                    "more than one mass-bearing plane; z_lens for the cosmology is "
                    "ambiguous. Pass the cosmology in explicitly instead.")
            z_lens = z
            planes.append(Plane(
                redshift=z, name=name,
                mass=[_component(cn, cv, _MASS_BY_PARAMS, "mass")
                      for cn, cv in entry["mass"].items()]))
        else:
            planes.append(Plane(
                redshift=z, name=name,
                light=[_component(cn, cv, _LIGHT_BY_PARAMS, "light")
                       for cn, cv in entry["light"].items()]))
    if z_lens is None:
        raise ValueError("no mass-bearing plane in params['planes']")

    cosmo = Component(
        w0waCDM_Cosmo(z_lens=z_lens, z_source_ref=z_source_ref),
        {k: float(v) for k, v in params["cosmo"].items()},
    )
    return LensModel(planes, cosmo=cosmo, unconstrain=unconstrain)


@dataclass(frozen=True)
class Ersatz:
    """One source plane's simulated observation, with the provenance to trace it back.

    ``clean`` is the noiseless render; ``image`` is ``clean`` noised from its own flux
    (never from the real cutout's stored error map, which is the noise level for the
    REAL pixels and would hand a recovery test a sigma no observation could supply in
    advance -- see ``real_datasets``' module docstring).
    """

    plane: str
    ext: str
    redshift: float
    clean: Any
    image: Any
    exp_time: float
    background_rms: float
    sim_config: Any
    light: Sequence[Any]
    model: Any

    def simulator(self, *, supersample: int = 1):
        """This plane's ``SceneSimulator``, with its real PSF -- for ``plot_scene`` and
        anything else that re-renders rather than reusing :attr:`clean`.

        Rebuilt on demand rather than kept, because ``SceneSimulator.__init__`` stores
        the whole supersampled coordinate grid (``img_X``/``img_Y``). At the render's
        ``supersample=16`` that is 4800 x 4800 x float64 = 184 MB per array, ~370 MB per
        simulator; holding all nine would pin ~3.3 GB for the lifetime of the notebook,
        which is what runs a login-node GPU out of memory. Building one costs nothing.

        ``supersample`` defaults to 1, not the render's 16, because this simulator is for
        LOOKING at the scene: the PSF is applied either way, and supersampling only
        refines the sub-pixel integration underneath it. Pass the render's value if you
        specifically want to reproduce :attr:`clean` exactly.
        """
        import copy as _copy
        from gigalens.jax.scene_simulator import SceneSimulator

        cfg = _copy.deepcopy(self.sim_config)
        cfg.supersample = supersample
        return SceneSimulator(self.model, cfg, sees=self.light)


def simulate_ersatz(model: LensModel, params: Dict[str, Any], cutout_dir: str, *,
                    seed: int = 0, delta_pix: float = 0.2, supersample: int = 16,
                    likelihood_precision: str = "float64",
                    conv_precision: str = "float64") -> List[Ersatz]:
    """Render + noise one mock per source plane, each with its own real cutout's PSF.

    ``supersample`` applies to the RENDER only. The truth image is what a fine sub-grid
    says it is; matching the coarser grid a fit will later use would bake that fit's own
    discretisation error into the data it is scored against.

    One PRNG key per plane, split from ``seed`` -- reusing one key would correlate the
    planes' noise realisations.
    """
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.jax.utils.noise import add_noise

    source_planes = [pl for pl in model.planes if pl.has_light]
    keys = jax.random.split(jax.random.PRNGKey(seed), len(source_planes))

    out = []
    for plane, key in zip(source_planes, keys):
        sid = source_id(plane.redshift)
        ext = STRUCTURE["cutout_ext"].get(sid, sid.replace("_", "-"))
        if plane.name != f"source{sid}":
            raise ValueError(
                f"plane {plane.name!r} at z = {plane.redshift:g} is canonically source "
                f"{sid}; refusing to hand it source{ext}.fits.")
        cutout = load_real_cutout(cutout_dir, ext)

        cfg = SimulatorConfig(
            delta_pix=delta_pix, num_pix=cutout.image.shape, supersample=supersample,
            kernel=cutout.psf, likelihood_precision=likelihood_precision,
            conv_precision=conv_precision)
        clean = SceneSimulator(model, cfg, sees=plane.light).simulate(params)
        image = add_noise(key, clean, cutout.exp_time, cutout.background_rms)

        out.append(Ersatz(
            plane=plane.name, ext=ext, redshift=plane.redshift, clean=clean, image=image,
            exp_time=cutout.exp_time, background_rms=cutout.background_rms,
            sim_config=cfg, light=plane.light, model=model))
    return out


# --------------------------------------------------------------------------------
def main() -> int:
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=os.path.join(here, "improved_sersic_carousel.json"))
    ap.add_argument("--cutouts", default=os.path.join(here, "real_cutouts"))
    ap.add_argument("--supersample", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p = truth_params(args.json)
    model = build_truth_model(p)

    # The round trip is the strongest check available that the dump and the rebuilt
    # model agree: same planes, same Components, same profiles, same values.
    model.validate_params(p)
    rebuilt = model.to_params({})
    bad = [f"{pl}/{cn}/{k}"
           for pl, pv in rebuilt["planes"].items()
           for grp in ("mass", "light") if grp in pv
           for cn, cv in pv[grp].items()
           for k, v in cv.items()
           if not np.isclose(float(v), float(p["planes"][pl][grp][cn][k]))]
    if bad:
        raise SystemExit(f"round trip changed {len(bad)} value(s): {bad[:5]}")
    print(f"round trip OK: {len(model.planes)} planes, "
          f"{len(model.z_param_names)} free parameters (expected 0)")

    for e in simulate_ersatz(model, p, args.cutouts, seed=args.seed,
                             supersample=args.supersample):
        snr = np.asarray(e.clean) / e.background_rms
        print(f"  {e.plane:<12} z = {e.redshift:<6.3f} src{e.ext:<6} "
              f"{tuple(e.image.shape)}  bkg_rms {e.background_rms:.3f}  "
              f"peak clean {float(np.max(e.clean)):9.3f}  peak S/N {float(np.max(snr)):8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
