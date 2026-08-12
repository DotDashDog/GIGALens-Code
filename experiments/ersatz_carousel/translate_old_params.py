#!/usr/bin/env python
"""Translate an old-API cluster lens model dump into a new-API ``LensModel``.

The input is any MAP dump in the old API's flat layout -- the coworker's fit
(``MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json``) and the plain-Sersic simulated
version of the same model (``sersic-simulated.json``) both parse the same way::

    {"cosmo":       {"H0", "Om0", "k", "w0", "wa"},
     "lens_mass":   {"0": {...}, ... "4": {...}},
     "lens_light":  {},
     "source_light":{"0": {...}, ... "8": {...}}}

and the output is ``gigalens.jax.scene.LensModel`` with one mass plane and nine
source planes, plus the matching truth-parameter dict for ``SceneSimulator``. The two
JSONs share the same five lens_mass components and the same nine source redshifts
(:data:`STRUCTURE` applies unchanged); they differ only in what each source_light
entry combines -- see point 2 below.

What the JSON alone cannot tell you
-----------------------------------
The old API stored *values*, not *structure*: which profile class each index was,
the shapelet order, and the two cosmology redshifts all lived in the driver script
(``boiler(2).py``), not the dump. Those are pinned in :data:`STRUCTURE` below, read
off that script. Everything else is derived from the JSON itself.

The three structural rewrites
-----------------------------
1. **Geometry moves from the profile to the plane.** Old sources carried ``z_source``
   and ``deflection_ratio`` as light-profile parameters (``cosmo_sample=True``). New
   API: each source is a ``Plane(redshift=...)`` and the ratio is *derived* from the
   cosmology. The ``deflection_ratio`` values in the JSON are therefore outputs, not
   inputs — they are not passed in, they are used by ``main`` to check the translation.

2. **``CombinedProfile`` is gone.** The new API has no ``combined_profile`` module; a
   plane simply holds a list of light ``Component``s. The old API built one up by
   suffixing every non-shared parameter of sub-profile ``i`` with ``_{i}`` and writing
   a ``shared_params`` parameter (default ``center_x``/``center_y``/``e1``/``e2``)
   bare, once, for the whole combine (see ``combined_profile.py``). That nests: a
   source can be several **clumps** at different centres (e.g. the two physical
   sources behind one grouped "1_2" plane) and, within a clump, several profiles
   **sharing** one centre -- a Sersic+Shapelets pair in the coworker's fit
   (``SersicShapelets(n_max)``), or a bulge+disk Sersic+Sersic pair in the simulated
   version. :func:`_expand_components` undoes both by peeling one suffix layer at a
   time and does not care which combination, or how deep, produced the keys; a plain,
   never-combined source just recurses zero times. (The coworker's fit also lists
   ``e1``/``e2`` in ``shared_params``, but ``Shapelets._params`` is only
   ``beta``/``center_x``/``center_y``, so the ellipticity was never actually consumed
   by the shapelet part -- and, empirically, every key in that dump is suffixed
   regardless, so nothing there is actually written bare.)

3. **Amplitudes become explicit.** The fit ran ``use_lstsq=True``, so ``Ie``/``amp*``
   were solved per band and stored. To *simulate* there is no data to solve against,
   so this script builds the model with ``use_lstsq=False`` and feeds the stored
   amplitudes back as fixed parameters. That is exact, not approximate: both profiles
   return the identical basis and simply multiply by the amplitude
   (``sersic.py``: ``ret[jnp.newaxis, ...] if use_lstsq else Ie * ret``).

   This is also where the translation stops being mechanical. The old lstsq solve is
   an *unconstrained* weighted least squares (``simulator.lstsq_simulate``: normal
   equations, no non-negativity), and lstsq-solved amplitudes are exempt from the
   physicality layer because they never enter ``profile.params``. Making them explicit
   removes that exemption, and three sersic amplitudes in this dump are negative --
   see :func:`negative_amplitudes`. ``LensModel`` construction therefore *raises* by
   default, which is correct: a negative ``Ie`` is a negative-surface-brightness
   source. Use ``--zero-negative-amplitudes`` to proceed deliberately.

Where the model itself comes from
---------------------------------
This module no longer builds a model. It loads the one in
``ersatz_carousel_prior_new_api.py`` (:data:`PRIOR_FILE`) -- the *fitting* model, priors
and all -- and pins every parameter to this dump's value with ``LensModel.fix_to``. The
JSON supplies numbers; the prior file supplies names, component order, plane order,
profile classes and the ``shared()`` links.

That used to be two independent models with two sets of component names kept in step by
hand. It stopped being a matter of tidiness when gigalens put names INTO the parameter
keys (``planes/source3/light/source3_main/R_sersic``): two vocabularies now mean a truth
dict and a chain from the fitting model key differently, and every downstream join --
corner overlays, chain persistence, SBC -- silently fails to find anything rather than
erroring. One scene, defined once, is the fix.

:func:`align` is where the two sides meet, and it checks rather than assumes: plane
count and redshifts in order, one old-API *leaf* per prior light Component, profile
class equality at every position, and full parameter coverage. A dump that is not the
same scene raises there. Two consequences worth knowing:

* A dump whose sources are ``SersicShapelets`` (the coworker's ``MAP_best_*.json``) has
  light the prior file has no Component for, so it CANNOT be pinned this way -- pinning
  it would quietly drop the shapelet half. :func:`truth_scene` refuses. ``--audit-only``
  still parses it fine.
* ``sersic-simulated-full.json`` puts three clumps behind source 1_2 where the prior
  file has two (its ``sersic_0_0`` is commented out), so it raises too, by design.

Parameter *values* transfer 1:1 — the profile classes used here (``NFW_ELLIPSE_SLOPE``,
``DPIE``, ``SHEAR``, ``SERSIC_ELLIPSE``, ``SHAPELETS``) have identical ``_params`` in
both APIs, and the only numerical change in the shared code is a DPIE NaN-guard that
agrees exactly whenever ``r_core < r_cut`` (true for all three DPIEs here).

Usage
-----
    PYTHONPATH=/global/u1/l/linusu/gigalens/src \
        ~/.conda/envs/gigalens_oldapi/bin/python translate_old_params.py

    # sersic-simulated.json is the default: it is the dump whose structure matches the
    # prior file. It has one negative Ie, so building it needs the flag.
    python translate_old_params.py --zero-negative-amplitudes

    # Parses and audits any dump, including the shapelet one, without gigalens:
    python translate_old_params.py --json MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json \
        --audit-only

From a notebook::

    from translate_old_params import build, truth_scene, prior_model, parse

    # the fitting model's scene, pinned to the dump: same names, same parameter keys
    model, spec = build("sersic-simulated.json", zero_negative_amplitudes=True)
    params = model.to_params({})           # ready for SceneSimulator.simulate

    # just the values, keyed the way the fitting model keys them
    truth = truth_scene(parse(json.load(open("sersic-simulated.json"))))
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

# --------------------------------------------------------------------------------
# Structure that lives in boiler(2).py, not in the JSON
# --------------------------------------------------------------------------------
STRUCTURE = {
    # w0waCDM_Cosmo(0.49, 1.432) in boiler(2).py. z_lens is the cluster plane; every
    # mass component sits on it. z_source_ref is the reduced-deflection normalisation
    # (lenstronomy's z_source_convention) that theta_E is quoted against.
    "z_lens": 0.49,
    "z_source_ref": 1.432,
    # Profile class per lens_mass index, in the JSON's own key order. Matches the
    # `lenses=[...]` list in boiler(2).py's prior_from_sources(); cross-checked against
    # the fixed centres in that file (e.g. index 0 center_x 5.2895684 == 5.28956836933847).
    # `upper_right_halo` (EPL) is commented out there and absent from the JSON.
    "lens_profiles": ["NFW_ELLIPSE_SLOPE", "DPIE", "SHEAR", "DPIE", "DPIE"],
    # Human labels, same order. OFFLINE FALLBACK ONLY: the authoritative names are the
    # Component names in PRIOR_FILE, and :func:`align` asserts the two agree, so a
    # rename there fails loudly here instead of silently producing two vocabularies.
    # This list exists solely so --audit-only can label mass components without
    # importing gigalens.
    "lens_names": ["halo", "Ld", "ext_shear", "Le", "Lf"],
    # Shapelet order for every SersicShapelets in the model: SersicShapelets(8) in
    # boiler(2).py. Also implied by the 45 stored amplitudes ((8+1)(8+2)/2 == 45), and
    # this script asserts the two agree rather than trusting either alone.
    "n_max": 8,
    # Canonical source ID per plane redshift. A plane is one numbered source, or a
    # group of them sharing a redshift ("4_5" = sources 4 and 5, one plane).
    #
    # This is the ONLY place the two vocabularies meet. The JSON knows redshifts and
    # nothing else; the cutouts, the finding charts and the collaboration all speak
    # source numbers. Note the two orderings DISAGREE -- sorted by redshift the
    # sources run 1_2, 3, 4_5, 9, 7, 6, 12_13, 8, 11, which is not sorted by number.
    # Anything pairing a per-source input (a cutout, a PSF, a mask) with a plane by
    # LIST POSITION is therefore wrong for six of the nine planes. Iterate
    # ``cutout_extensions`` instead; it is ordered by plane.
    "source_ids": {
        0.962: "1_2", 1.166: "3", 1.432: "4_5", 1.506: "9", 1.627: "7",
        1.656: "6", 3.086: "12_13", 3.549: "8", 4.090: "11",
    },
    # Cutout file label per source ID, where the file is not just the ID with "-" for
    # "_". Group 1_2 is stored as source1.fits, NOT source1-2.fits, while group 12_13
    # is source12-13.fits -- verified against real_cutouts/, not inferred from the
    # pattern, because the two groups are named inconsistently.
    "cutout_ext": {"1_2": "1"},
}


def source_id(redshift: float) -> str:
    """Canonical source ID for a plane redshift, e.g. ``0.962 -> "1_2"``.

    Matched with a tolerance far tighter than the closest pair of redshifts in the
    model (1.627 vs 1.656), so a near miss is a typo rather than an ambiguity.

    Raises on an unknown redshift instead of falling back to the number itself: a
    plane quietly named after an ID nobody recognises is exactly how a mislabelled
    panel ends up in a figure, and this is the last place that can catch it.
    """
    for z, sid in STRUCTURE["source_ids"].items():
        if abs(z - redshift) < 5e-4:
            return sid
    known = ", ".join(f"{z:g}" for z in sorted(STRUCTURE["source_ids"]))
    raise ValueError(
        f"no canonical source ID for redshift {redshift:g}; STRUCTURE['source_ids'] "
        f"knows {known}. Add it there -- do not guess from position.")


def cutout_extensions(spec: Dict[str, Any]) -> List[str]:
    """Per-source cutout labels in **plane order**: ``["1", "3", "4-5", "9", ...]``.

    Build per-observation inputs by iterating THIS, not a hand-written list of source
    numbers. Planes are ordered by redshift and sources are numbered by something
    else, so the two orders agree only for the first three of nine; zipping a
    number-sorted list against the planes hands six of them another source's cutout,
    PSF and mask. Every panel still renders, and all six are wrong.

        for ext in cutout_extensions(spec):
            ...  # opens source{ext}.fits, in the order the sims must be built
    """
    out = []
    for source in spec["sources"]:
        sid = source_id(source["redshift"])
        out.append(STRUCTURE["cutout_ext"].get(sid, sid.replace("_", "-")))
    return out

# Observation-side settings that are NOT part of the lens model but are needed before
# anything can actually be rendered. Values from boiler(2).py where it has them.
OBSERVATION = {
    "numPix": 300,           # boiler(2).py
    "deltaPix": 0.2,         # boiler(2).py
    "exp_time": 9920,        # boiler(2).py
    "psf": None,             # COMMENTED OUT in boiler(2).py (Gaussian, FWHM 0.7178 arcsec)
    "background_rms": None,  # not present anywhere
}


# --------------------------------------------------------------------------------
# JSON -> intermediate spec (no gigalens import needed; safe for --audit-only)
# --------------------------------------------------------------------------------
_SUFFIX = re.compile(r"^(.*)_(\d+)$")


def _n_max_from_amp_count(n_amps: int) -> int:
    """Invert n_layers = (n+1)(n+2)/2. Raises if `n_amps` is not a triangular count."""
    n = 0
    while (n + 1) * (n + 2) // 2 < n_amps:
        n += 1
    if (n + 1) * (n + 2) // 2 != n_amps:
        raise ValueError(
            f"{n_amps} shapelet amplitudes is not (n+1)(n+2)/2 for any n_max")
    return n


def _split_one_level(params: Dict[str, float]) -> Dict[str, Dict[str, float]] | None:
    """Undo one layer of ``CombinedProfile`` suffixing.

    A ``CombinedProfile`` of N sub-profiles gives every sub-profile's own parameters a
    trailing ``_{i}``, *except* a parameter in ``shared_params`` (default
    ``center_x``/``center_y``/``e1``/``e2``), which is instead written bare, once, for
    the whole combine (``combined_profile.py``: ``if param not in shared_params: append
    param + f"_{i}"``). Splitting on the *rightmost* numeric suffix and re-attaching
    every bare key to every suffixed group inverts exactly that.

    Returns ``None`` when no key carries a suffix at all: ``params`` is already one
    profile's own parameters and there is nothing left to split.
    """
    grouped: Dict[str, Dict[str, float]] = {}
    shared: Dict[str, float] = {}
    for key, value in params.items():
        match = _SUFFIX.match(key)
        if not match:
            shared[key] = value
        else:
            base, idx = match.groups()
            grouped.setdefault(idx, {})[base] = value
    if not grouped:
        return None
    for idx in grouped:
        for key, value in shared.items():
            grouped[idx].setdefault(key, value)
    return dict(sorted(grouped.items(), key=lambda kv: int(kv[0])))


def _expand_components(
    params: Dict[str, float], path: Tuple[str, ...] = ()
) -> List[Tuple[Tuple[str, ...], Dict[str, float]]]:
    """Recursively split one source_light entry into leaf-profile params.

    A source plane can combine profiles two ways, and both go through the same
    suffix/shared-param mechanism: several **clumps** at different centres (e.g. the
    two physical sources behind one grouped "1_2" plane), and, within one clump,
    several profiles **sharing** a centre (a bulge+disk pair, or the old API's
    Sersic+Shapelets). Peeling one suffix layer at a time with :func:`_split_one_level`
    -- re-broadcasting whatever comes back bare into every group it produces -- and
    recursing until a group has no suffix left handles either nesting, and any depth of
    it, uniformly. A plain, never-combined source (bare keys throughout) recurses zero
    times and returns itself as the only leaf.

    ``path`` names a leaf by its index at each layer, outermost (clump) first, e.g.
    ``("0", "1")`` for the second profile of the first clump; ``()`` for a source that
    was never combined at all.
    """
    level = _split_one_level(params)
    if level is None:
        return [(path, params)]
    leaves: List[Tuple[Tuple[str, ...], Dict[str, float]]] = []
    for idx, sub in level.items():
        leaves.extend(_expand_components(sub, path + (idx,)))
    return leaves


def parse(data: Dict[str, Any]) -> Dict[str, Any]:
    """Old-API JSON -> a plain-data spec of the new-API scene.

    Returns ``{"cosmo": {...}, "lens": [...], "sources": [...]}`` where every component
    is ``{"profile": <class name>, "name": str, "params": {...}, "n_max": int|None}``.
    Source planes are returned sorted by redshift, which the new API requires
    (planes must be ordered observer->source).
    """
    spec: Dict[str, Any] = {"cosmo": dict(data["cosmo"]), "lens": [], "sources": []}

    # ---- mass plane -------------------------------------------------------------
    lens_keys = sorted(data["lens_mass"], key=int)
    if len(lens_keys) != len(STRUCTURE["lens_profiles"]):
        raise ValueError(
            f"JSON has {len(lens_keys)} lens_mass entries but STRUCTURE['lens_profiles'] "
            f"names {len(STRUCTURE['lens_profiles'])}. Re-read boiler(2).py's lenses=[...] "
            "list; the two must agree or every mass component is mislabelled.")
    for key, profile, name in zip(lens_keys, STRUCTURE["lens_profiles"],
                                  STRUCTURE["lens_names"]):
        spec["lens"].append({
            "profile": profile, "name": name, "n_max": None,
            "params": dict(data["lens_mass"][key]),
        })

    # ---- source planes ----------------------------------------------------------
    for key in sorted(data["source_light"], key=int):
        entry = dict(data["source_light"][key])
        redshift = entry.pop("z_source")
        # Derived from the cosmology in the new API; kept only for the check in main().
        stored_ratio = entry.pop("deflection_ratio", None)

        components: List[Dict[str, Any]] = []
        for path, params in _expand_components(entry):
            amps = {k: v for k, v in params.items() if k.startswith("amp")}
            rest = {k: v for k, v in params.items() if not k.startswith("amp")}
            tag = "_" + "_".join(path) if path else "_0"

            if amps:
                # Old SersicShapelets: a SersicEllipse + a Shapelets sharing the centre.
                n_max = _n_max_from_amp_count(len(amps))
                if n_max != STRUCTURE["n_max"]:
                    raise ValueError(
                        f"source {key}{tag}: {len(amps)} amplitudes imply n_max={n_max}, "
                        f"but STRUCTURE['n_max'] is {STRUCTURE['n_max']}.")
                width = len(str(len(amps)))
                components.append({
                    "profile": "SERSIC_ELLIPSE", "name": f"sersic{tag}",
                    "n_max": None, "leaf": tag,
                    "params": {k: v for k, v in rest.items() if k != "beta"},
                })
                components.append({
                    "profile": "SHAPELETS", "name": f"shapelets{tag}",
                    "n_max": n_max, "leaf": tag,
                    "params": {
                        "beta": rest["beta"],
                        # Shared with the sersic above -- this is the old
                        # SersicShapelets.shared_params link, made explicit.
                        "center_x": rest["center_x"], "center_y": rest["center_y"],
                        **{f"amp{i:0{width}d}": amps[f"amp{i:0{width}d}"]
                           for i in range(len(amps))},
                    },
                })
            else:
                # A plain Sersic (bare params, path == ()) or one profile of a
                # multi-Sersic clump (bulge+disk sharing a centre, no shapelets) --
                # _expand_components has already merged the shared centre/ellipticity
                # in either case, so this is just the profile's own parameters.
                components.append({
                    "profile": "SERSIC_ELLIPSE", "name": f"sersic{tag}",
                    "n_max": None, "leaf": tag, "params": rest,
                })

        spec["sources"].append({
            "index": int(key), "redshift": redshift,
            "stored_deflection_ratio": stored_ratio, "components": components,
        })

    spec["sources"].sort(key=lambda s: s["redshift"])
    return spec


# --------------------------------------------------------------------------------
# The prior file is the single source of NAMES and STRUCTURE; the JSON supplies VALUES
# --------------------------------------------------------------------------------
# This module used to assemble a second, independent LensModel and name its Components
# itself, so the fitting model (PRIOR_FILE) and the truth model (here) carried two
# separate vocabularies that had to be kept in step by hand. Since gigalens put names
# INTO the parameter keys (``planes/<plane>/light/<component>/<param>``; commit
# "Put names in parameter keys"), that is no longer merely untidy -- a truth dict and a
# chain from the fitting model would key differently, so nothing downstream (corner
# plots, chain persistence, truth overlays) would line up, and the mismatch would show
# up as absent parameters rather than as an error.
#
# So the scene is now defined ONCE, in PRIOR_FILE, and this module pins it with
# ``LensModel.fix_to``. Names, component order, plane order, sharing, and profile
# classes all come from there; the JSON contributes only numbers. :func:`align` refuses
# to proceed unless the dump really is the same scene, checked component by component.
PRIOR_FILE = "ersatz_carousel_prior_new_api.py"

_PRIOR_CACHE: Dict[str, Any] = {}


def prior_model(path: str | None = None):
    """The assembled ``LensModel`` from :data:`PRIOR_FILE` -- the *fitting* model.

    Loaded by file path (it is a script beside this one, not an installed module) and
    cached per path: ``fix_to`` matches its ``free=`` argument by OBJECT IDENTITY, so
    handing callers two separately-imported copies of the same file would silently stop
    that matching.
    """
    import importlib.util

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PRIOR_FILE)
    path = os.path.abspath(path)
    if path not in _PRIOR_CACHE:
        module_spec = importlib.util.spec_from_file_location("_ersatz_prior", path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load the prior file at {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        if not hasattr(module, "model"):
            raise AttributeError(
                f"{path} defines no `model`; this needs the assembled LensModel.")
        _PRIOR_CACHE[path] = module.model
    return _PRIOR_CACHE[path]


#: new-API profile class -> the JSON/STRUCTURE vocabulary :func:`parse` speaks.
_PROFILE_ALIAS = {
    "NFW_ELLIPSE_SLOPE": "NFW_ELLIPSE_SLOPE", "DPIE": "DPIE", "Shear": "SHEAR",
    "SersicEllipse": "SERSIC_ELLIPSE", "Shapelets": "SHAPELETS",
}


def _profile_kind(profile) -> str:
    return _PROFILE_ALIAS.get(type(profile).__name__, type(profile).__name__)


def _leaf_groups(components: List[Dict[str, Any]]
                 ) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Regroup a plane's parsed components into old-API *leaf profiles*.

    ``parse`` emits one component per leaf, except an old ``SersicShapelets`` leaf which
    becomes two (a Sersic and a Shapelets sharing one centre). Collecting the consecutive
    entries carrying the same ``leaf`` tag undoes that, so what gets matched 1:1 against
    the prior model's light Components is a LEAF, not a component -- otherwise a
    two-profile clump would swallow its neighbour's slot and every later component in the
    plane would shift by one.
    """
    groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    for entry in components:
        if groups and groups[-1][0] == entry["leaf"]:
            groups[-1][1].append(entry)
        else:
            groups.append((entry["leaf"], [entry]))
    return groups


def _prior_param_names(comp) -> List[str]:
    """Every parameter a Component's prior dict covers, tuple (grouped) keys expanded."""
    names: List[str] = []
    for key in comp.priors:
        names.extend(key if isinstance(key, tuple) else (key,))
    return names


def align(spec: Dict[str, Any], model=None):
    """Bind each parsed JSON component to the prior-model Component that owns it.

    Writes two fields onto every spec entry, in place, and returns the model:

      ``name``  the prior model's Component name -- replacing ``parse``'s structural
                ``sersic_0`` placeholder. That placeholder is the JSON's vocabulary;
                this is the model's, and only one of them may reach a parameter key.
      ``key``   the params-tree path ``(plane_key, kind, component_key)`` that
                :func:`truth_scene` and ``fix_to`` address values by.

    The correspondence is ultimately positional, and positional matching is precisely how
    components get mixed up, so every position is checked before it is used:

      * plane count, and each plane's redshift matched IN ORDER against the model's --
        both sides are sorted by redshift, but this asserts it instead of trusting it
        (STRUCTURE's own warning: source NUMBER order and redshift order disagree for six
        of the nine planes);
      * mass -- one component per index, profile class equal, and the model's name equal
        to ``STRUCTURE["lens_names"]`` so the offline fallback cannot drift;
      * light -- one old-API LEAF per prior light Component, profile class equal;
      * parameters -- every parameter the prior Component prices must be in the dump.

    Anything that does not match raises. A dump with a different structure (a shapelet
    leaf, an extra clump) is a different scene and is not bent to fit this one.
    """
    model = prior_model() if model is None else model
    planes = model.planes

    if len(planes) != 1 + len(spec["sources"]):
        raise ValueError(
            f"the prior model has {len(planes)} planes (1 cluster + "
            f"{len(planes) - 1} sources) but the dump has {len(spec['sources'])} "
            "sources; these are not the same scene.")

    # ---- cluster plane / mass ---------------------------------------------------
    if abs(float(planes[0].redshift) - STRUCTURE["z_lens"]) > 5e-4:
        raise ValueError(
            f"prior plane 0 is at z={float(planes[0].redshift)} but STRUCTURE['z_lens'] "
            f"is {STRUCTURE['z_lens']}; plane 0 must be the cluster.")
    plane_key = model.plane_key(0)
    if len(planes[0].mass) != len(spec["lens"]):
        raise ValueError(
            f"prior cluster plane has {len(planes[0].mass)} mass components but the dump "
            f"has {len(spec['lens'])}.")
    for j, (comp, entry) in enumerate(zip(planes[0].mass, spec["lens"])):
        kind, want = _profile_kind(comp.profile), entry["profile"]
        if kind != want:
            raise ValueError(
                f"mass component {j}: the prior model has {kind} but the dump's "
                f"lens_mass[{j}] is {want} (STRUCTURE['lens_profiles']). One of the two "
                "orders is wrong -- do not reorder blindly, re-read boiler(2).py.")
        comp_key = model.component_key(0, "mass", j)
        if comp_key != STRUCTURE["lens_names"][j]:
            raise ValueError(
                f"mass component {j} is named {comp_key!r} in {PRIOR_FILE} but "
                f"{STRUCTURE['lens_names'][j]!r} in STRUCTURE['lens_names']. The prior "
                "file is authoritative -- update STRUCTURE['lens_names'] to match.")
        entry["name"] = comp_key
        entry["key"] = (plane_key, "mass", comp_key)
        _check_covered(comp, entry, f"{plane_key}/mass/{comp_key}")

    # ---- source planes ----------------------------------------------------------
    for i, source in enumerate(spec["sources"], start=1):
        plane = planes[i]
        if abs(float(plane.redshift) - source["redshift"]) > 5e-4:
            raise ValueError(
                f"plane {i}: the prior model is at z={float(plane.redshift):g} but the "
                f"dump's source {source['index']} is at z={source['redshift']:g}. Both "
                "sides must be sorted by redshift; a mismatch here means every "
                "component after it would be attached to the wrong source.")
        plane_key = model.plane_key(i)
        source["plane_key"] = plane_key
        source["source_id"] = source_id(source["redshift"])

        groups = _leaf_groups(source["components"])
        if len(groups) != len(plane.light):
            raise ValueError(
                f"plane {plane_key!r} (z={source['redshift']:g}, source "
                f"{source['source_id']}): the prior model has {len(plane.light)} light "
                f"components {[c.name for c in plane.light]} but the dump expands to "
                f"{len(groups)} leaf profiles {[g[0] for g in groups]}. Same scene, "
                "different clumps -- fix the prior file or use a matching dump.")
        for j, ((leaf, entries), comp) in enumerate(zip(groups, plane.light)):
            comp_key = model.component_key(i, "light", j)
            kind = _profile_kind(comp.profile)
            main = [e for e in entries if e["profile"] == kind]
            if len(main) != 1:
                raise ValueError(
                    f"{plane_key}/light/{comp_key}: the prior model is {kind} but leaf "
                    f"{leaf!r} of the dump is {[e['profile'] for e in entries]}.")
            main[0]["name"] = comp_key
            main[0]["key"] = (plane_key, "light", comp_key)
            _check_covered(comp, main[0], f"{plane_key}/light/{comp_key}")
            # An old SersicShapelets leaf also carries a Shapelets half, which the prior
            # file has no Component for. Name it after the Sersic it shares a centre with
            # and mark it: truth_scene refuses to build a truth that the prior model
            # cannot represent, rather than dropping light on the floor.
            for extra in entries:
                if extra is main[0]:
                    continue
                extra["name"] = f"{comp_key}_{extra['profile'].lower()}"
                extra["key"] = (plane_key, "light", extra["name"])
                extra["unrepresented"] = True
    return model


def _check_covered(comp, entry: Dict[str, Any], where: str):
    """Every parameter the prior Component prices must be present in the dump."""
    missing = [n for n in _prior_param_names(comp) if n not in entry["params"]]
    if missing:
        raise ValueError(
            f"{where}: the dump has no value for {sorted(missing)}, which "
            f"{PRIOR_FILE} treats as parameters of this component. fix_to needs a truth "
            "for every one of them.")


def truth_scene(spec: Dict[str, Any], model=None) -> Dict[str, Any]:
    """The dump's values as a structured truth dict keyed by the PRIOR MODEL's names.

    This is the §5 ``planes``/``cosmo`` layout ``fix_to`` consumes, and -- because the
    keys are the prior model's own -- also the dict to hand a truth overlay, an SBC
    check, or anything else that has to line a chain up against the input values.

    Amplitudes (``Ie``, ``amp*``) are included even though ``fix_to`` ignores them: they
    are not parameters of an ``use_lstsq=True`` Component, but
    :func:`_with_explicit_amplitudes` needs them right after.
    """
    model = align(spec, model)

    unrepresented = [f"{s['plane_key']}/{e['name']}" for s in spec["sources"]
                     for e in s["components"] if e.get("unrepresented")]
    if unrepresented:
        raise ValueError(
            f"this dump has {len(unrepresented)} light component(s) with no counterpart "
            f"in {PRIOR_FILE}: {unrepresented}. They are the Shapelets halves of old "
            "SersicShapelets profiles; the prior file models those sources as plain "
            "Sersics. Pinning the prior model to this dump would silently DROP that "
            "light. Add the shapelet Components to the prior file, or translate a dump "
            "whose sources are plain Sersics.")

    truth: Dict[str, Any] = {"cosmo": dict(spec["cosmo"]), "planes": {}}

    def put(entry: Dict[str, Any]):
        plane_key, kind, comp_key = entry["key"]
        plane = truth["planes"].setdefault(plane_key, {})
        plane.setdefault(kind, {})[comp_key] = dict(entry["params"])

    cluster_key = model.plane_key(0)
    truth["planes"].setdefault(cluster_key, {})["geometry"] = {
        "redshift": STRUCTURE["z_lens"]}
    for entry in spec["lens"]:
        put(entry)
    for source in spec["sources"]:
        truth["planes"].setdefault(source["plane_key"], {})["geometry"] = {
            "redshift": source["redshift"]}
        for entry in source["components"]:
            put(entry)

    _check_shared_consistency(model, truth)
    return truth


def _check_shared_consistency(model, truth: Dict[str, Any]):
    """Linked sites in the prior model must hold ONE value in the dump.

    A ``shared()`` handle means several sites draw a single value (the prior file shares
    source3's centre and ellipticity across its two Sersics). The old API stored such a
    parameter once, bare, and ``_split_one_level`` re-broadcasts it to every sub-profile,
    so the sites SHOULD agree exactly. If they ever did not, ``fix_to`` would pin them to
    different numbers without complaint and produce a "truth" the fitting model cannot
    represent -- a truth no fit could ever recover.

    Checked through the model's own gather/scatter rather than by hand: ``to_unique``
    keeps one value per unique parameter and ``to_params`` writes it back to every site
    it feeds, so any disagreement shows up as a changed value.
    """
    try:
        round_trip = model.to_params(model.to_unique(truth))
    except Exception as exc:                      # pragma: no cover - defensive
        raise ValueError(
            f"could not round-trip the truth through the prior model: {exc}") from exc

    problems: List[str] = []
    for plane_key, kinds in truth["planes"].items():
        for kind in ("mass", "light"):
            for comp_key, params in kinds.get(kind, {}).items():
                got = round_trip["planes"][plane_key][kind][comp_key]
                for name, value in params.items():
                    if name not in got:
                        continue          # lstsq amplitude: not a model parameter
                    if abs(float(got[name]) - float(value)) > 1e-9:
                        problems.append(
                            f"{plane_key}/{kind}/{comp_key}/{name}: dump {value!r} vs "
                            f"{float(got[name])!r} after the model's own sharing")
    if problems:
        raise ValueError(
            "the dump's values disagree across parameters that " + PRIOR_FILE +
            " links with shared(): \n  " + "\n  ".join(problems) +
            "\nThe prior model cannot represent this truth. Either the sharing in the "
            "prior file is wrong, or the two components really are independent in the "
            "dump.")


def _with_explicit_amplitudes(model, truth: Dict[str, Any]):
    """Re-render every lstsq light Component with its stored amplitude made explicit.

    ``fix_to`` pins parameter VALUES but keeps each Component's profile instance, and the
    prior file's Sersics are ``use_lstsq=True`` -- their ``Ie`` is not a parameter at all,
    it is solved per evaluation against data. Simulating has no data to solve against, so
    the stored amplitude must become a fixed parameter, which takes a new profile instance
    with ``use_lstsq=False``. That is exact, not approximate: both modes return the same
    basis and differ only by the multiply (``sersic.py``: ``ret[jnp.newaxis, ...] if
    use_lstsq else Ie * ret``).
    """
    from gigalens.jax.scene import Component, Plane, LensModel

    new_planes = []
    for i, plane in enumerate(model.planes):
        plane_key = model.plane_key(i)
        light = []
        for j, comp in enumerate(plane.light):
            if not getattr(comp.profile, "use_lstsq", False):
                light.append(comp)
                continue
            comp_key = model.component_key(i, "light", j)
            stored = truth["planes"][plane_key]["light"][comp_key]
            amps = {k: v for k, v in stored.items()
                    if k == "Ie" or k.startswith("amp")}
            if not amps:
                raise ValueError(
                    f"{plane_key}/light/{comp_key} is an lstsq component but the dump "
                    "stores no amplitude for it, so it cannot be made explicit.")
            try:
                profile = type(comp.profile)(use_lstsq=False)
            except TypeError as exc:
                raise TypeError(
                    f"cannot rebuild {type(comp.profile).__name__} with use_lstsq=False "
                    f"for {plane_key}/light/{comp_key}") from exc
            light.append(Component(profile, {**comp.priors, **amps}, name=comp.name))
        new_planes.append(Plane(redshift=plane.redshift,
                                deflection_ratio=plane.deflection_ratio,
                                mass=list(plane.mass), light=light, name=plane.name))
    return LensModel(new_planes, cosmo=model.cosmo)


def build(json_path: str, use_lstsq: bool = False,
          zero_negative_amplitudes: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """The prior model, pinned to this dump's values. Returns ``(model, spec)``.

    Same contract as before -- with ``use_lstsq=False`` (the default, and what you want
    for simulating) every parameter including the amplitudes is fixed, so
    ``model.num_free_params == 0`` and ``model.to_params({})`` yields the full structured
    params dict for ``SceneSimulator.simulate``; with ``use_lstsq=True`` the amplitudes
    are left to be solved against data, matching how the fit was run -- but the model is
    now the one from :data:`PRIOR_FILE` with its parameters fixed, not a second model
    built here, so its component names and parameter keys ARE the fitting model's.

    ``zero_negative_amplitudes`` sets any negative sersic ``Ie`` to 0.0, which silences
    the physicality raise by *deleting* those components' light. It changes the model; it
    is not a fix. It also only matters when the amplitudes are explicit: with
    ``use_lstsq=True`` they are not parameters and never reach the physicality layer.
    """
    model = prior_model()

    with open(json_path) as handle:
        spec = parse(json.load(handle))

    # Before anything reports a component by name: bind the dump to the prior model, so
    # every message below names components the way the fitting model does.
    align(spec, model)

    if zero_negative_amplitudes:
        for where, name, value in negative_amplitudes(spec):
            print(f"  zeroing {where}/{name}: Ie {value:g} -> 0.0 (component removed)")
        for source in spec["sources"]:
            for entry in source["components"]:
                if entry["params"].get("Ie", 0.0) < 0:
                    entry["params"]["Ie"] = 0.0

    truth = truth_scene(spec, model)
    fixed = model.fix_to(truth)
    if not use_lstsq:
        fixed = _with_explicit_amplitudes(fixed, truth)
    return fixed, spec


# --------------------------------------------------------------------------------
# Audit: what is present, what is missing
# --------------------------------------------------------------------------------
#: Parameters each profile needs, duplicated here so --audit-only runs without gigalens.
_EXPECTED = {
    "NFW_ELLIPSE_SLOPE": ["theta_E", "s_E", "e1", "e2", "center_x", "center_y"],
    "DPIE": ["theta_E", "r_core", "r_cut", "center_x", "center_y", "e1", "e2"],
    "SHEAR": ["gamma1", "gamma2"],
    "SERSIC_ELLIPSE": ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y", "Ie"],
    "SHAPELETS": ["beta", "center_x", "center_y"],  # + amp00..ampNN, checked separately
    "COSMO": ["H0", "Om0", "k", "w0", "wa"],
}


def audit(spec: Dict[str, Any]) -> List[str]:
    """Return a list of problems. Empty list == every profile parameter is supplied."""
    problems: List[str] = []

    missing = set(_EXPECTED["COSMO"]) - set(spec["cosmo"])
    if missing:
        problems.append(f"cosmo: missing {sorted(missing)}")

    def check(where: str, entry: Dict[str, Any]):
        expected = set(_EXPECTED[entry["profile"]])
        got = set(entry["params"])
        if entry["profile"] == "SHAPELETS":
            n_layers = (entry["n_max"] + 1) * (entry["n_max"] + 2) // 2
            width = len(str(n_layers))
            expected |= {f"amp{i:0{width}d}" for i in range(n_layers)}
        if expected - got:
            problems.append(f"{where} ({entry['profile']}): missing "
                            f"{sorted(expected - got)}")
        if got - expected:
            problems.append(f"{where} ({entry['profile']}): unexpected "
                            f"{sorted(got - expected)}")

    for entry in spec["lens"]:
        check(f"lens/{entry['name']}", entry)
    for source in spec["sources"]:
        for entry in source["components"]:
            check(f"src{source_id(source['redshift'])}/{entry['name']}", entry)
    return problems


def negative_amplitudes(spec: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    """Sersic components whose stored ``Ie`` is negative: ``(where, name, value)``.

    These are what make ``LensModel`` construction raise once the amplitudes stop being
    lstsq-exempt. They are a real property of the coworker's fit, not a translation
    error: the old solve is unconstrained, so a component with little support in the
    data (e.g. a heavily masked source) can come back negative.
    """
    found = []
    for source in spec["sources"]:
        for entry in source["components"]:
            value = entry["params"].get("Ie")
            if value is not None and value < 0:
                found.append((f"src{source_id(source['redshift'])} "
                              f"z={source['redshift']:g}",
                              entry["name"], value))
    return found


def structural_gaps() -> List[str]:
    """Things the JSON does not carry at all. Not bugs -- just what you must supply."""
    gaps = [
        f"z_lens = {STRUCTURE['z_lens']} and z_source_ref = "
        f"{STRUCTURE['z_source_ref']}: absent from the JSON's `cosmo` block; taken from "
        "w0waCDM_Cosmo(0.49, 1.432) in boiler(2).py. The new API needs both.",
        f"shapelet n_max = {STRUCTURE['n_max']}: absent from the JSON; taken from "
        "boiler(2).py and confirmed against the 45 stored amplitudes.",
        "profile classes per lens_mass index: absent from the JSON; taken from "
        "boiler(2).py's lenses=[...] order (see STRUCTURE['lens_profiles']).",
        "lens_light is {} -- the cluster/BCG starlight was never modelled. Fine if you "
        "are simulating source light only; otherwise it must be added.",
    ]
    if OBSERVATION["psf"] is None:
        gaps.append(
            "PSF: commented out in boiler(2).py (Gaussian, FWHM 0.717821594376891 "
            "arcsec = 3.589 pix at 0.2\"/pix, 15x15 kernel). Needed by SimulatorConfig "
            "before anything can be rendered -- this is the one real blocker.")
    if OBSERVATION["background_rms"] is None:
        gaps.append(
            "background_rms / sigma_bkd: not present anywhere. Needed by add_noise and "
            "by ImageData if you refit the mock (exp_time = 9920 is in boiler(2).py).")
    gaps.append(
        "per-source masks: boiler(2).py blanks pixels [150:250, 150:250] for the "
        "z=1.627 and z=4.090 sources. Not in the JSON; re-apply if you refit.")
    return gaps


# --------------------------------------------------------------------------------
def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The default is the dump the prior file's structure matches. MAP_best_*.json is a
    # SersicShapelets fit; it still parses with --audit-only, but it cannot be pinned
    # onto the prior model (see the module docstring), so it is no longer the default.
    parser.add_argument("--json", default=os.path.join(here, "sersic-simulated.json"))
    parser.add_argument("--audit-only", action="store_true",
                        help="parse and audit without importing gigalens")
    parser.add_argument("--lstsq", action="store_true",
                        help="drop amplitudes and solve them against data instead")
    parser.add_argument("--zero-negative-amplitudes", action="store_true",
                        help="set negative sersic Ie to 0.0 (changes the model)")
    args = parser.parse_args()

    if args.audit_only:
        with open(args.json) as handle:
            spec = parse(json.load(handle))
        model = None
    else:
        model, spec = build(args.json, use_lstsq=args.lstsq,
                            zero_negative_amplitudes=args.zero_negative_amplitudes)

    n_comp = sum(len(s["components"]) for s in spec["sources"])
    print(f"cluster plane  z = {STRUCTURE['z_lens']}  "
          f"({len(spec['lens'])} mass components: "
          f"{', '.join(e['profile'] for e in spec['lens'])})")
    print(f"source planes  {len(spec['sources'])}  ({n_comp} light components)")
    for source in spec["sources"]:
        kinds = ", ".join(e["profile"].lower() for e in source["components"])
        sid = source_id(source["redshift"])
        print(f"  src{sid:<6s} z = {source['redshift']:<6.3f} "
              f"cutout {STRUCTURE['cutout_ext'].get(sid, sid.replace('_', '-')):<5s} "
              f"{kinds}")

    if model is not None:
        # The whole correspondence, printed. Everything downstream keys off these names,
        # so this is the table to read when a component looks like it landed in the
        # wrong place.
        print(f"\n--- component map: dump -> {PRIOR_FILE} ---")
        for j, entry in enumerate(spec["lens"]):
            print(f"  lens_mass[{j}]           {entry['profile']:<18} -> "
                  f"{'/'.join(entry['key'])}")
        for source in spec["sources"]:
            for leaf, entries in _leaf_groups(source["components"]):
                for entry in entries:
                    print(f"  source_light[{source['index']}]{leaf:<9} "
                          f"{entry['profile']:<18} -> {'/'.join(entry['key'])}")

    print("\n--- parameter audit ---")
    problems = audit(spec)
    if problems:
        for problem in problems:
            print(f"  PROBLEM  {problem}")
    else:
        print("  every profile parameter required by the new API is present in the JSON.")

    negatives = negative_amplitudes(spec)
    if negatives:
        print("\n--- negative sersic amplitudes (LensModel raises on these) ---")
        for where, name, value in negatives:
            print(f"  {where:<22} {name:<12} Ie = {value:10.4f}")
        print("  The old lstsq solve is unconstrained and its amplitudes were exempt "
              "from the\n  physicality layer; fixing them for simulation removes that "
              "exemption. Pass\n  --zero-negative-amplitudes to drop these components, "
              "or refit them non-negative.")

    print("\n--- not carried by the JSON (supplied from boiler(2).py, or still missing) ---")
    for gap in structural_gaps():
        print(f"  * {gap}")

    if model is None:
        return 1 if problems else 0

    # The JSON's stored deflection_ratio is an OUTPUT in the new API (cosmology +
    # redshifts determine it). Reproducing it is an end-to-end check that z_lens,
    # z_source_ref and the cosmology parameters were all read correctly -- but not
    # every old-API dump stores it (only the fit the coworker actually ran did), so a
    # dump without it just skips the check rather than crashing on `None - float`.
    print("\n--- check: derived vs stored deflection_ratio ---")
    print(f"  trace mode: {model.trace_mode}, free params: {model.num_free_params}")
    params = model.to_params({})
    worst = 0.0
    checked = False
    for i, source in enumerate(spec["sources"], start=1):
        stored = source["stored_deflection_ratio"]
        if stored is None:
            print(f"  z = {source['redshift']:<6.3f}  (no stored deflection_ratio "
                  "in this JSON -- skipped)")
            continue
        checked = True
        derived = float(model.plane_deflection_ratio(params, i))
        delta = abs(derived - stored)
        worst = max(worst, delta)
        print(f"  z = {source['redshift']:<6.3f}  derived {derived:.6f}  "
              f"stored {stored:.6f}  diff {delta:.2e}")
    if not checked:
        print("  nothing to check.")
        return 1 if problems else 0
    print(f"  worst |diff| = {worst:.2e}")
    if worst > 1e-5:
        print("  MISMATCH -- z_lens / z_source_ref / cosmology do not reproduce the fit.")
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
