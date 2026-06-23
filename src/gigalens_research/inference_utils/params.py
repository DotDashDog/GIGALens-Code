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

    for p_i, plane in enumerate(scene_model.planes):
        pkey = out["planes"].setdefault(p_i, {})
        if plane.mass:
            mblock = pkey.setdefault("mass", {})
            for m_j, comp in enumerate(plane.mass):
                mblock[m_j] = _leaf(td["lens_mass"][str(mass_idx)], comp)
                mass_idx += 1
        if plane.light:
            lblock = pkey.setdefault("light", {})
            for l_j, comp in enumerate(plane.light):
                if id(comp) in src_ids:
                    src = td["source_light"][str(source_light_idx)]
                    source_light_idx += 1
                else:
                    src = td["lens_light"][str(lens_light_idx)]
                    lens_light_idx += 1
                lblock[l_j] = _leaf(src, comp)
    return out
