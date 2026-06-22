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
