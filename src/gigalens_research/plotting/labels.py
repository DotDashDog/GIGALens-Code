"""LaTeX labels for parameters.

Parameters are identified by their scene path — ``planes/0/mass/0/theta_E``,
``cosmo/H0`` — and :func:`latex_label` renders one for display. The registry and
the parameter records themselves live in :mod:`gigalens_research.param_index`;
this module is the plotting-side entry point.

Superseded label space
----------------------
This module used to own a second, parallel label space: a flattener that
collapsed the scene into three groups (mass / lens light / source light) and
named parameters ``theta_E__1``, ``lens_R_sersic__0``, ``src_Ie``, ``cosmo_H0``.
It is gone, along with ``flatten_params`` / ``flatten_param_names``. It was lossy
in ways that kept biting:

- Its ``__<i>`` suffix was a *global running index* across planes, so
  ``theta_E__1`` could not say which plane it was on.
- Its ``lens_``/``src_`` prefixes encoded a lens-light/source-light distinction
  the model does not have — a ``Plane`` carries only ``mass`` and ``light``, and
  the split was synthesized from "light is source light iff an earlier plane has
  mass".
- Being derived, it silently disagreed with the bijector's own key order, which
  is the recurring column-mislabel hazard documented in :mod:`.diagnostics`.

Path space is the scene's own naming, so none of that translation happens. Use
:func:`gigalens_research.param_index.param_sites` for the parameter records
(kind, plane, component) and
:func:`gigalens_research.param_index.sites_to_matrix` for ordered columns.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence

from ..param_index import LATEX_LABELS, param_of_key
from ..param_index import latex_label as _latex_label_for_param

__all__ = ["LATEX_LABELS", "latex_label", "z_column_labels"]


def latex_label(name: str, *, fallback: Optional[str] = None) -> str:
    """LaTeX for a parameter, given its scene path key or a bare parameter name.

    ``planes/0/mass/0/theta_E``, ``cosmo/H0`` and ``theta_E`` all render as the
    plain symbol. The path locates the parameter, but a lone axis label has no
    room to repeat it; callers labelling several components at once should use
    :func:`gigalens_research.param_index.site_labels`, which adds a
    plane/component superscript exactly where two labels would otherwise
    collide.

    Unknown parameters fall back to ``fallback`` (default: the bare name) rather
    than raising, so an unregistered profile still plots.
    """
    return _latex_label_for_param(param_of_key(name), fallback=fallback)


def z_column_labels(names: Sequence[str]) -> List[str]:
    """Display labels for the sampler's unconstrained (z) columns.

    The z-space peer of :func:`gigalens_research.param_index.site_labels`: a column
    renders as its bare symbol (``$\\theta_E$``) where that symbol appears once, and
    as the scene's own full name (``planes/0/mass/0/theta_E``) where two columns
    would otherwise read identically. The symbol alone is not an identity — a
    multiplane model has a ``theta_E`` per plane, and ``center_x`` / ``e1`` belong to
    mass *and* light profiles alike — so the collision is the common case, not the
    exotic one.

    Unlike ``site_labels`` this cannot disambiguate with a plane/component
    superscript: a z column need not correspond to a single site at all (a
    dimension-reducing grouped prior emits synthetic ``<ukey>#j`` columns), so the
    name the scene itself assigned is the only label guaranteed to be unique.

    ``names`` must come from ``prob_model.z_param_names``, the ONE correct
    column→name map (see ``gigalens.jax.scene._z_param_names``). Names recovered any
    other way — from a bijector output dict's key order, say — are in a *different*
    order, which is the recurring "C-8" mislabel.
    """
    bases = [latex_label(n) for n in names]
    collisions = {b for b, count in Counter(bases).items() if count > 1}
    return [name if b in collisions else b for name, b in zip(names, bases)]
