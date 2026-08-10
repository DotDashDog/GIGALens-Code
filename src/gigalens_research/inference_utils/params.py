"""Parameter-structure helpers for the new gigalens (dev refactor) API.

The refactored gigalens simulator/prob_model keys parameters by component name,
``{'lens_mass': {'0': {..}, '1': {..}}, 'lens_light': {'0': {..}}, 'source_light':
{'0': {..}}}``, rather than the legacy 3-list ``[lens, lens_light, source]``.
Truth params persisted before the migration (vela ``true_params`` pickles, older
``truth_x.pkl``, GL2 YAML extraction, hand-built fixtures) are still in the list
form, so consumers that feed params into a gigalens ``simulate`` / ``lstsq_simulate``
call must normalise first.
"""
from __future__ import annotations

from typing import Any, Dict

# Canonical [lens, lens_light, source] component order, keyed as the new
# gigalens prior/simulator expect.
_COMPONENT_KEYS = ("lens_mass", "lens_light", "source_light")


# ---------------------------------------------------------------------------
# Scene params-tree keys
# ---------------------------------------------------------------------------
# A scene params tree is keyed by NAME where the scene names a plane/component and by
# ``str(index)`` where it does not (gigalens ``scene.py`` §names, ``cc5a078``) --
# ``params["planes"]["lens"]["mass"]["host"]["theta_E"]``. Keys are always ``str``,
# never a bare ``int``: JAX cannot flatten a dict mixing ``int`` and ``str`` keys at
# all, so an int-keyed tree is not merely inconsistent, it is unusable.
#
# Callers hold integer positions (``enumerate(model.planes)``, the simulator's
# ``_light`` list), so every one of them needs this translation. The model owns it;
# these wrappers just tolerate a model old enough to predate the accessors.


def plane_key(scene_model: Any, i: int):
    """The params-tree key of plane ``i``.

    Falls back to the bare ``int`` — not ``str(i)`` — for a model predating the
    accessors, because such a model also predates string keys and its tree really is
    int-keyed. ``str(i)`` would be wrong there in the quietest possible way: a
    ``KeyError`` on a tree that plainly contains the plane.
    """
    fn = getattr(scene_model, "plane_key", None)
    return str(fn(i)) if fn is not None else i


def component_key(scene_model: Any, i: int, kind: str, j: int):
    """The params-tree key of component ``j`` in plane ``i``'s ``kind`` list."""
    fn = getattr(scene_model, "component_key", None)
    return str(fn(i, kind, j)) if fn is not None else j


def mass_params_list(scene_model: Any, params: Dict[str, Any], i: int,
                     n: int) -> list:
    """Plane ``i``'s first ``n`` mass components as an ordered LIST.

    gigalens' point-source physics helpers (``_total_deflection`` and friends) pair
    ``mass_params[j]`` with ``mass_profiles[j]`` positionally, while the params tree
    keys them by name. Handing those helpers the raw ``["mass"]`` dict happened to
    work only while the keys were integers; it is a ``KeyError`` now, and would have
    been a silent mispairing had the keys stayed sortable. This is the bridge.
    """
    return [component_params(scene_model, params, i, "mass", j) for j in range(n)]


def plane_params(scene_model: Any, params: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Plane ``i``'s sub-dict of a scene params tree."""
    return params["planes"][plane_key(scene_model, i)]


def component_params(scene_model: Any, params: Dict[str, Any], i: int, kind: str,
                     j: int) -> Dict[str, Any]:
    """One component's parameter dict out of a scene params tree.

    ``kind`` is ``"mass"`` or ``"light"``. Resolves the naming so callers never have
    to know whether the component was named.
    """
    return plane_params(scene_model, params, i)[kind][
        component_key(scene_model, i, kind, j)]


def to_dict_params(params: Any) -> Dict[str, Dict[str, Any]]:
    """Normalise params to the dict-keyed structure the new gigalens API uses.

    Accepts either the dict form (already-migrated ``prior.sample`` output) or
    the legacy 3-list form ``[lens_list, lens_light_list, source_list]`` and
    returns ``{'lens_mass': {'0': {..}, ..}, 'lens_light': {..}, 'source_light':
    {..}}``.  A dict is returned unchanged, so this is safe to apply defensively.
    """
    if isinstance(params, dict):
        return params
    keyed: Dict[str, Dict[str, Any]] = {}
    for comp_list, key in zip(params, _COMPONENT_KEYS):
        keyed[key] = {str(i): p for i, p in enumerate(comp_list)}
    return keyed


def truth_x_to_scene_params(truth_x: Any, scene_model: Any) -> Dict[str, Any]:
    """Adapt a persisted OLD 3-group truth (G1 D2) to a SCENE structured-params dict.

    The old truth is ``{lens_mass:{i:{param:val}}, lens_light:{j:..}, source_light:{k:..}}``
    (via :func:`to_dict_params`); the scene API consumes
    ``{planes:{p:{geometry, mass:{m:..}, light:{l:..}}}, cosmo:..}``. This adapter maps by
    ROLE + index onto the scene model's actual structure:

      - mass Components (in plane order) <- ``lens_mass[0,1,...]``
      - source-plane light (``LensModel.source_plane_light``) <- ``source_light[...]``
      - the remaining (lens) light <- ``lens_light[...]``
      - geometry (deflection_ratio / redshift) is taken from the model's own constants
        (it is not part of the persisted light/mass truth).

    This is a CONTAINED research-side adapter (D2: do NOT re-persist truth on disk). It
    maps only the params present in BOTH the truth and the scene profile: a persisted
    truth may carry EXTRA keys the scene profile does not take (e.g. a Sérsic ``Ie`` lstsq
    amplitude — absent from ``profile.params``) and may OMIT params the scene profile adds
    (e.g. a parametric ``beta``/``n_sersic`` for a source that was generated from a
    pixelized image and only persisted ``center_x``/``center_y``). Omitted params are
    simply not provided here; ``LensModel.fix_to`` reads this dict ONLY for the params it
    FIXES, so any omission must belong to a FREE Component or fix_to will raise loudly at
    that site. lstsq amplitudes are absent from both sides and are neither mapped nor
    required.
    """
    import copy

    import jax.numpy as _jnp

    td = to_dict_params(truth_x)

    def _leaf(d, comp):
        """Map only params present in BOTH the profile and the truth dict ``d``."""
        out = {}
        for pn in comp.profile.params:
            if pn in d:
                out[pn] = _jnp.asarray(float(_jnp.squeeze(_jnp.asarray(d[pn]))))
        return out

    # Start from the model's constants (carries geometry + any fixed params), then
    # overwrite mass/light leaves from the truth by role+index.
    out: Dict[str, Any] = copy.deepcopy(scene_model.constants)
    out.setdefault("planes", {})

    src_ids = {id(c) for c in scene_model.source_plane_light()}
    mass_idx = 0
    lens_light_idx = 0
    source_light_idx = 0

    # Keys must be the model's OWN params-tree keys. Writing the integer position here
    # would not raise: it would add ``0`` alongside the ``"0"`` already carried by
    # ``constants``, leaving a dict with mixed int/str keys that JAX refuses to flatten
    # -- and, where the scene names a plane, silently orphan the block from the name it
    # is supposed to fill.
    for p_i, plane in enumerate(scene_model.planes):
        pblock = out["planes"].setdefault(plane_key(scene_model, p_i), {})
        if plane.mass:
            mblock = pblock.setdefault("mass", {})
            for m_j, comp in enumerate(plane.mass):
                mblock[component_key(scene_model, p_i, "mass", m_j)] = _leaf(
                    td["lens_mass"][str(mass_idx)], comp)
                mass_idx += 1
        if plane.light:
            lblock = pblock.setdefault("light", {})
            for l_j, comp in enumerate(plane.light):
                if id(comp) in src_ids:
                    src = td["source_light"][str(source_light_idx)]
                    source_light_idx += 1
                else:
                    src = td["lens_light"][str(lens_light_idx)]
                    lens_light_idx += 1
                lblock[component_key(scene_model, p_i, "light", l_j)] = _leaf(src, comp)
    return out
