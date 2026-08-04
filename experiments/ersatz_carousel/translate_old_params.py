#!/usr/bin/env python
"""Translate the coworker's old-API cluster lens model into a new-API ``LensModel``.

The input is a MAP dump in the old API's flat layout
(``MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json``)::

    {"cosmo":       {"H0", "Om0", "k", "w0", "wa"},
     "lens_mass":   {"0": {...}, ... "4": {...}},
     "lens_light":  {},
     "source_light":{"0": {...}, ... "8": {...}}}

and the output is ``gigalens.jax.scene.LensModel`` with one mass plane and nine
source planes, plus the matching truth-parameter dict for ``SceneSimulator``.

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

2. **``CombinedProfile`` / ``SersicShapelets`` are gone.** The new API has no
   ``combined_profile`` or ``sersic_shapelets`` module; a plane simply holds a list of
   light ``Component``s. Old ``SersicShapelets(n_max)`` was a ``SersicEllipse`` plus a
   ``Shapelets(n_max)`` sharing ``center_x``/``center_y``, so it expands into two
   Components that repeat the same centre. (Old ``shared_params`` also listed
   ``e1``/``e2``, but ``Shapelets._params`` is only ``beta``/``center_x``/``center_y``,
   so the ellipticity was never actually consumed by the shapelet part.)

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

Parameter *values* transfer 1:1 — the profile classes used here (``NFW_ELLIPSE_SLOPE``,
``DPIE``, ``SHEAR``, ``SERSIC_ELLIPSE``, ``SHAPELETS``) have identical ``_params`` in
both APIs, and the only numerical change in the shared code is a DPIE NaN-guard that
agrees exactly whenever ``r_core < r_cut`` (true for all three DPIEs here).

Usage
-----
    PYTHONPATH=/global/u1/l/linusu/gigalens/src \
        ~/.conda/envs/gigalens_oldapi/bin/python translate_old_params.py

    python translate_old_params.py --json path/to/MAP.json --audit-only

From a notebook::

    from translate_old_params import build
    model, truth = build("MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json")
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
    # Human labels, same order.
    "lens_names": ["cluster_halo", "bcg_dpie", "ext_shear", "le_dpie", "group_dpie"],
    # Shapelet order for every SersicShapelets in the model: SersicShapelets(8) in
    # boiler(2).py. Also implied by the 45 stored amplitudes ((8+1)(8+2)/2 == 45), and
    # this script asserts the two agree rather than trusting either alone.
    "n_max": 8,
}

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


def _split_subprofiles(entry: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Split one old source_light entry into its sub-profiles.

    A plain profile has bare keys (``center_x``); a CombinedProfile suffixes every key
    with its sub-profile index (``center_x_0``, ``center_x_1``). ``z_source`` and
    ``deflection_ratio`` are plane-level and are stripped by the caller.
    """
    if "center_x" in entry:
        return {"": dict(entry)}
    subs: Dict[str, Dict[str, float]] = {}
    for key, value in entry.items():
        match = _SUFFIX.match(key)
        if not match:
            raise ValueError(f"unsuffixed key {key!r} in a combined-profile entry")
        base, idx = match.groups()
        subs.setdefault(idx, {})[base] = value
    return dict(sorted(subs.items()))


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
        for sub_idx, params in _split_subprofiles(entry).items():
            amps = {k: v for k, v in params.items() if k.startswith("amp")}
            rest = {k: v for k, v in params.items() if not k.startswith("amp")}
            tag = f"_{sub_idx}" if sub_idx != "" else ""

            if amps:
                # Old SersicShapelets: a SersicEllipse + a Shapelets sharing the centre.
                n_max = _n_max_from_amp_count(len(amps))
                if n_max != STRUCTURE["n_max"]:
                    raise ValueError(
                        f"source {key}{tag}: {len(amps)} amplitudes imply n_max={n_max}, "
                        f"but STRUCTURE['n_max'] is {STRUCTURE['n_max']}.")
                width = len(str(len(amps)))
                components.append({
                    "profile": "SERSIC_ELLIPSE", "name": f"sersic{tag or '_0'}",
                    "n_max": None,
                    "params": {k: v for k, v in rest.items() if k != "beta"},
                })
                components.append({
                    "profile": "SHAPELETS", "name": f"shapelets{tag or '_0'}",
                    "n_max": n_max,
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
                components.append({
                    "profile": "SERSIC_ELLIPSE", "name": f"sersic{tag or '_0'}",
                    "n_max": None, "params": rest,
                })

        spec["sources"].append({
            "index": int(key), "redshift": redshift,
            "stored_deflection_ratio": stored_ratio, "components": components,
        })

    spec["sources"].sort(key=lambda s: s["redshift"])
    return spec


# --------------------------------------------------------------------------------
# Spec -> LensModel
# --------------------------------------------------------------------------------
def _profile(name: str, n_max=None, use_lstsq: bool = False):
    from gigalens.jax.profiles.mass.nfw_ellipse_slope import NFW_ELLIPSE_SLOPE
    from gigalens.jax.profiles.mass.piemd import DPIE
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.light.shapelets import Shapelets

    if name == "NFW_ELLIPSE_SLOPE":
        return NFW_ELLIPSE_SLOPE()
    if name == "DPIE":
        return DPIE()
    if name == "SHEAR":
        return Shear()
    if name == "SERSIC_ELLIPSE":
        return SersicEllipse(use_lstsq=use_lstsq)
    if name == "SHAPELETS":
        return Shapelets(n_max, use_lstsq=use_lstsq)
    raise ValueError(f"no new-API profile mapped for {name!r}")


def build(json_path: str, use_lstsq: bool = False,
          zero_negative_amplitudes: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """Build the new-API model. Returns ``(model, spec)``.

    With ``use_lstsq=False`` (the default, and what you want for simulating) every
    parameter including the amplitudes is fixed, so ``model.num_free_params == 0`` and
    ``model.to_params({})`` yields the full structured params dict for
    ``SceneSimulator.simulate``. With ``use_lstsq=True`` the amplitudes are dropped and
    solved against data instead, matching how the coworker's fit was run.

    ``zero_negative_amplitudes`` sets any negative sersic ``Ie`` to 0.0, which silences
    the physicality raise by *deleting* those components' light. It changes the model;
    it is not a fix. Shapelet amplitudes are left alone -- they are sign-indefinite
    basis coefficients by construction and the physicality layer exempts them.
    """
    from gigalens.jax.cosmo import w0waCDM_Cosmo
    from gigalens.jax.scene import Component, Plane, LensModel

    with open(json_path) as handle:
        spec = parse(json.load(handle))

    if zero_negative_amplitudes:
        for where, name, value in negative_amplitudes(spec):
            print(f"  zeroing {where}/{name}: Ie {value:g} -> 0.0 (component removed)")
        for source in spec["sources"]:
            for entry in source["components"]:
                if entry["params"].get("Ie", 0.0) < 0:
                    entry["params"]["Ie"] = 0.0

    cosmo = Component(
        w0waCDM_Cosmo(z_lens=STRUCTURE["z_lens"],
                      z_source_ref=STRUCTURE["z_source_ref"]),
        spec["cosmo"], name="w0waCDM")

    def to_component(entry, is_light):
        params = dict(entry["params"])
        if is_light and use_lstsq:
            params = {k: v for k, v in params.items()
                      if k != "Ie" and not k.startswith("amp")}
        return Component(_profile(entry["profile"], entry["n_max"],
                                  use_lstsq=use_lstsq and is_light),
                         params, name=entry["name"])

    planes = [Plane(redshift=STRUCTURE["z_lens"], name="cluster",
                    mass=[to_component(e, False) for e in spec["lens"]])]
    for source in spec["sources"]:
        planes.append(Plane(
            redshift=source["redshift"],
            name=f"src{source['index']}_z{source['redshift']:g}",
            light=[to_component(e, True) for e in source["components"]]))

    return LensModel(planes, cosmo=cosmo), spec


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
            check(f"src{source['index']}/{entry['name']}", entry)
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
                found.append((f"src{source['index']} z={source['redshift']:g}",
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
    parser.add_argument(
        "--json", default=os.path.join(here, "MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json"))
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
        print(f"  src{source['index']}  z = {source['redshift']:<6.3f}  {kinds}")

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
    # z_source_ref and the cosmology parameters were all read correctly.
    print("\n--- check: derived vs stored deflection_ratio ---")
    print(f"  trace mode: {model.trace_mode}, free params: {model.num_free_params}")
    params = model.to_params({})
    worst = 0.0
    for i, source in enumerate(spec["sources"], start=1):
        derived = float(model.plane_deflection_ratio(params, i))
        stored = source["stored_deflection_ratio"]
        delta = abs(derived - stored)
        worst = max(worst, delta)
        print(f"  z = {source['redshift']:<6.3f}  derived {derived:.6f}  "
              f"stored {stored:.6f}  diff {delta:.2e}")
    print(f"  worst |diff| = {worst:.2e}")
    if worst > 1e-5:
        print("  MISMATCH -- z_lens / z_source_ref / cosmology do not reproduce the fit.")
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
