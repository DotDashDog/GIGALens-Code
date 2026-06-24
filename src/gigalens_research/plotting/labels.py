"""Parameter naming and LaTeX label helpers.

GIGALens parameters live in a nested ``[[lens_params], [lens_light_params],
[source_light_params]]`` structure. This module provides:

- :func:`flatten_params` — flatten that structure into ``{label: array}``
  using a consistent naming convention.
- :func:`flatten_param_names` — labels only (cheap; for axis labels etc.).
- :func:`latex_label` — map a label to a LaTeX string for plotting.
- :data:`LATEX_LABELS` — the registry; extend in-place when adding new
  profiles. (Long-term TODO: move LaTeX labels onto the profile classes
  themselves so this dict can be retired.)

Naming convention (deliberately matches gigalens' simulator-side flattener,
``flatten_params_to_labeled_dict_sim``):

- Mass profiles (``params[0]``): no prefix (``theta_E``, ``gamma1``, ...).
- Lens light (``params[1]``): ``lens_`` prefix (``lens_R_sersic``, ...).
- Source light (``params[2]``): ``src_`` prefix (``src_R_sersic``, ...).

If a profile group has more than one component (e.g. two mass profiles), each
component contributes all its keys without further indexing — they will appear
sequentially in the flat dict. If two components share a key name within the
same group, the later one overwrites the earlier one in the flat dict; pass
``_index_collisions=True`` to disambiguate with ``_0`` / ``_1`` suffixes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# LaTeX label registry
# ---------------------------------------------------------------------------


#: Default LaTeX rendering for known GIGALens parameter names. Extend in-place
#: when you add a new profile until a per-profile registry on the gigalens
#: side replaces this.
LATEX_LABELS: Dict[str, str] = {
    # Mass — EPL
    "theta_E": r"$\theta_E$",
    "gamma": r"$\gamma_{\rm epl}$",
    "e1": r"$\epsilon_{\rm epl,1}$",
    "e2": r"$\epsilon_{\rm epl,2}$",
    "center_x": r"$x_{\rm epl}$",
    "center_y": r"$y_{\rm epl}$",
    # Mass — external shear
    "gamma1": r"$\gamma_{\rm ext,1}$",
    "gamma2": r"$\gamma_{\rm ext,2}$",
    # Lens light
    "lens_R_sersic": r"$R_l$",
    "lens_n_sersic": r"$n_l$",
    "lens_e1": r"$\epsilon_{l,1}$",
    "lens_e2": r"$\epsilon_{l,2}$",
    "lens_center_x": r"$x_l$",
    "lens_center_y": r"$y_l$",
    "lens_Ie": r"$I_l$",
    # Source light
    "src_R_sersic": r"$R_s$",
    "src_n_sersic": r"$n_s$",
    "src_e1": r"$\epsilon_{s,1}$",
    "src_e2": r"$\epsilon_{s,2}$",
    "src_center_x": r"$x_s$",
    "src_center_y": r"$y_s$",
    "src_Ie": r"$I_s$",
}


def latex_label(name: str, *, fallback: Optional[str] = None) -> str:
    """Return a LaTeX-formatted version of ``name`` if known, else ``fallback``
    (which defaults to ``name`` itself, useful for unknown shapelet coefficients
    etc.).
    """
    return LATEX_LABELS.get(name, name if fallback is None else fallback)


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


_GROUP_PREFIXES = ("", "lens_", "src_")

# New gigalens params are dict-keyed by component name; map each onto the same
# label prefix the legacy positional ``_GROUP_PREFIXES`` produced.
_GROUP_KEY_PREFIXES = {"lens_mass": "", "lens_light": "lens_", "source_light": "src_"}

# Canonical [mass, lens_light, source] ordering for dict-keyed params, so the
# flat output (and hence corner-plot column order) is stable and intuitive.
_CANONICAL_GROUP_ORDER = ("lens_mass", "lens_light", "source_light")


def _iter_groups(params):
    """Yield ``(prefix, group)`` per component group, for either the dict-keyed
    structure ``{'lens_mass': {..}, ..}`` or the legacy positional list
    ``[[mass..], [lens_light..], [source..]]``."""
    if isinstance(params, dict):
        keys = [k for k in _CANONICAL_GROUP_ORDER if k in params]
        keys += [k for k in params if k not in _CANONICAL_GROUP_ORDER]
        for ckey in keys:
            yield _GROUP_KEY_PREFIXES.get(ckey, f"{ckey}_"), params[ckey]
    else:
        for group_idx, group in enumerate(params):
            prefix = _GROUP_PREFIXES[group_idx] if group_idx < len(_GROUP_PREFIXES) else f"group{group_idx}_"
            yield prefix, group


def _iter_profiles(group):
    """Yield profile param-dicts in a group, for either a dict ``{'0': {..}, ..}``
    keyed by stringified profile index or the legacy list of dicts."""
    if isinstance(group, dict):
        for pkey in sorted(group, key=int):
            yield group[pkey]
    else:
        yield from group


def flatten_params(
    params: Any,
    *,
    _index_collisions: bool = False,
) -> Dict[str, Any]:
    """Flatten a nested parameter structure into a flat ``{label: array}`` dict.

    Accepts either the new dict-keyed gigalens structure
    ``{'lens_mass': {'0': {..}, ..}, 'lens_light': {..}, 'source_light': {..}}``
    or the legacy positional list ``[[mass..], [lens_light..], [source..]]``.
    See the module docstring for the naming convention. ``params`` follows the
    output convention of :meth:`ProbabilisticModel.bij.forward` (and is what
    ``LensSimulator.simulate`` expects as its argument).
    """
    # Scene-API (G1) flat form: the scene bijector returns a flat
    # ``{unique_key: array}`` dict (keys like ``planes/0/mass/0/theta_E`` or
    # ``shared_<uid>``) where every value is a leaf array — there is no group/profile
    # nesting to walk, and each key already IS one parameter column. Detect it (all
    # top-level values are non-dict/non-list) and return it as-is. The legacy 3-group
    # dict has dict values at the top, and the positional form is a list, so neither
    # trips this.
    if isinstance(params, dict) and params and all(
            not isinstance(v, (dict, list)) for v in params.values()):
        return dict(params)

    flat: Dict[str, Any] = {}
    seen: Dict[str, int] = {}
    for prefix, group in _iter_groups(params):
        for profile in _iter_profiles(group):
            for key, value in profile.items():
                label = f"{prefix}{key}"
                if label in flat:
                    if _index_collisions:
                        seen[label] = seen.get(label, 0) + 1
                        flat[f"{label}_{seen[label]}"] = value
                        continue
                    # Otherwise the later value silently overwrites the earlier;
                    # documented behavior. Disambiguate caller-side if needed.
                flat[label] = value
    return flat


def flatten_param_names(params: List[List[Dict[str, Any]]]) -> List[str]:
    """Just the labels, in the order they'd appear from :func:`flatten_params`."""
    return list(flatten_params(params).keys())
