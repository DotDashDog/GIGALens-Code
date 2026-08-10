"""Corner plots from :class:`Posterior` objects.

Two entry points:

- :func:`plot_corner` — one posterior, optional truth and overlay points.
- :func:`plot_corner_overlay` — multiple posteriors on the same figure,
  with a built-in legend.

Both consume :class:`Posterior` views directly, so callers don't have to know
about array shapes or the bijector.

Selecting what to plot
----------------------
Everything is plotted by default. To narrow it, combine any of ``kind``,
``plane`` and ``component`` — they AND together::

    plot_corner(post)                            # all parameters
    plot_corner(post, kind="cosmology")
    plot_corner(post, kind=["cosmology", "mass"])
    plot_corner(post, plane=0)
    plot_corner(post, kind="mass", plane=1)
    plot_corner(post, component=("mass", 0))
    plot_corner(post, select=lambda s: s.param.startswith("e"))   # escape hatch

Panels are ordered cosmology, geometry, mass, light; then by plane, then by
component. See :mod:`gigalens_research.param_index` for the parameter records
these filters run against.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import corner as _corner_pkg
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..param_index import (
    ParamSite,
    param_sites,
    select_sites,
    site_labels,
    sites_to_matrix,
    truth_row,
)


def _flat_x(posterior) -> Dict[str, Any]:
    """The posterior's physical params as the scene's flat ``{unique_key: array}``
    dict — samples for a sampler, a single point for a point estimate."""
    if hasattr(posterior, "flat_x"):
        return posterior.flat_x
    if hasattr(posterior, "x"):
        return posterior.x
    return posterior.z_to_x(posterior.median_z)


def _resolve_columns(
    posterior,
    *,
    kind: Any = None,
    plane: Any = None,
    component: Any = None,
    select: Optional[Callable[[ParamSite], bool]] = None,
    plot_params: Optional[Sequence[str]] = None,
    latex: bool = True,
) -> Tuple[np.ndarray, List[str], List[ParamSite]]:
    """Everything a corner plot needs, with no plotting in it: the ``(n, p)``
    sample matrix, the display labels, and the :class:`ParamSite` records.

    This is the seam. A faster corner backend replaces only the rendering that
    consumes this — column resolution, selection, ordering and labelling are
    backend-agnostic and stay put.
    """
    sites = param_sites(posterior)

    if plot_params is not None:
        if any(f is not None for f in (kind, plane, component, select)):
            raise ValueError(
                "pass either plot_params (an explicit column list) or the "
                "kind/plane/component/select filters, not both — otherwise it is "
                "ambiguous whether plot_params is a selection or an ordering."
            )
        by_key = {s.key: s for s in sites}
        missing = [k for k in plot_params if k not in by_key]
        if missing:
            raise KeyError(
                f"plot_params names {len(missing)} parameter(s) this model does "
                f"not have: {missing}. Parameters are keyed by scene path, e.g. "
                f"{next(iter(by_key))!r}. Available: {sorted(by_key)}"
            )
        sites = [by_key[k] for k in plot_params]  # caller's order wins
    else:
        sites = select_sites(
            sites, kind=kind, plane=plane, component=component, select=select
        )

    samples = sites_to_matrix(sites, _flat_x(posterior))
    labels = site_labels(sites, latex=latex)
    return samples, labels, sites


def plot_corner(
    posterior,
    *,
    fig: Optional[Figure] = None,
    kind: Any = None,
    plane: Any = None,
    component: Any = None,
    select: Optional[Callable[[ParamSite], bool]] = None,
    plot_params: Optional[Sequence[str]] = None,
    truth=None,
    overplots: Optional[Dict[str, Any]] = None,
    color: str = "black",
    truth_color: str = "black",
    overplot_color: str = "red",
    latex: bool = True,
    **corner_kwargs,
) -> Figure:
    """Corner plot of one posterior.

    Parameters
    ----------
    posterior : Posterior
        Source of samples.
    kind : str or list of str, optional
        Restrict to ``"cosmology"``, ``"geometry"``, ``"mass"`` and/or
        ``"light"``. Default: all.
    plane : int or list of int, optional
        Restrict to one or more planes. Default: all.
    component : int, (role, index), or list, optional
        Restrict to component indices within their ``(plane, role)``. Pass
        ``("mass", 0)`` to pin the role.
    select : callable, optional
        Predicate on a :class:`~gigalens_research.param_index.ParamSite`, for
        selections the keyword filters don't express.
    plot_params : list of str, optional
        Explicit columns by scene path key (``"planes/0/mass/0/theta_E"``), in
        the order given. Mutually exclusive with the filters above.
    truth : optional
        A truth point, either scene-nested
        (``{"planes": {"lens": {"mass": {"host": {...}}}}, "cosmo": {...}}``) or
        path-keyed (``{"planes/lens/mass/host/theta_E": ...}``). Keys are the
        scene's — a component's name where it has one, ``str(index)`` where it does
        not. Drawn as crosshairs. Parameters it doesn't define simply get no marker,
        so a truth built against differently-named components silently draws nothing;
        take the names from the file that defines the model.
    overplots : dict, optional
        Map ``{legend_label: point}`` of extra points to overplot as stars, in
        either truth form. Useful for marking a MAP point on top of HMC samples.
    """
    samples, labels, sites = _resolve_columns(
        posterior, kind=kind, plane=plane, component=component, select=select,
        plot_params=plot_params, latex=latex,
    )
    truths = None if truth is None else truth_row(sites, truth)

    defaults = dict(show_titles=True, title_fmt=".3f", color=color,
                    hist_kwargs={"density": True, "color": color})
    defaults.update(corner_kwargs)

    fig = _corner_pkg.corner(
        samples, fig=fig, truths=truths, truth_color=truth_color,
        labels=labels, **defaults,
    )

    if overplots:
        for name, point in overplots.items():
            row = truth_row(sites, point, what=f"overplot {name!r}")
            _corner_pkg.overplot_points(
                fig, row[None, :], marker="*", markersize=18,
                mfc=overplot_color, mec=overplot_color,
            )
    return fig


def plot_corner_overlay(
    posteriors: Dict[str, Any],
    *,
    kind: Any = None,
    plane: Any = None,
    component: Any = None,
    select: Optional[Callable[[ParamSite], bool]] = None,
    plot_params: Optional[Sequence[str]] = None,
    truth=None,
    overplots: Optional[Dict[str, Any]] = None,
    colors: Optional[Dict[str, str]] = None,
    truth_label: str = "Truth",
    legend_loc: str = "upper right",
    legend_kwargs: Optional[dict] = None,
    latex: bool = True,
    range_pad: float = 0.05,
    range_quantile: float = 0.999,
) -> Figure:
    """Overlay multiple posteriors on a single corner figure.

    ``posteriors`` is an ordered dict ``{legend_label: Posterior}``. All are
    drawn into the same figure with identical axis ranges so the corner
    package's ``fig=`` overlay path works reliably across versions; some
    newer ``corner`` releases silently create a new figure when the per-call
    data range disagrees with the existing axes' range, which can produce
    duplicated single-posterior plots.

    Selection (``kind``/``plane``/``component``/``select``/``plot_params``) works
    as in :func:`plot_corner`. It is resolved once against the first posterior,
    then every overlay is pinned to exactly those columns, so the figure's axes
    mean the same thing for each.

    Parameters
    ----------
    range_quantile : float, default 0.999
        Symmetric quantile bound used to build the shared per-parameter range
        (so e.g. 0.999 → [0.0005, 0.9995] quantiles). Robust to outliers.
    range_pad : float, default 0.05
        Fractional padding applied to each axis range.

    ``overplots`` works as in :func:`plot_corner`. A unified legend is added
    at the end.
    """
    import matplotlib.pyplot as plt

    if not posteriors:
        raise ValueError("plot_corner_overlay needs at least one posterior.")

    default_palette = ["blue", "black", "orange", "purple", "brown"]
    colors = colors or {name: default_palette[i % len(default_palette)]
                        for i, name in enumerate(posteriors)}

    # Resolve the column set once from the first posterior, then pin every other
    # overlay to those exact keys. A posterior from a different scene surfaces
    # here as a missing key rather than silently overlaying mismatched columns.
    first = next(iter(posteriors.values()))
    _, _labels, sites = _resolve_columns(
        first, kind=kind, plane=plane, component=component, select=select,
        plot_params=plot_params, latex=latex,
    )
    keys = [s.key for s in sites]
    name_to_samples = {
        name: _resolve_columns(post, plot_params=keys, latex=latex)[0]
        for name, post in posteriors.items()
    }

    combined = np.vstack(list(name_to_samples.values()))
    lo_q = (1.0 - range_quantile) / 2.0
    hi_q = 1.0 - lo_q
    lo = np.quantile(combined, lo_q, axis=0)
    hi = np.quantile(combined, hi_q, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    lo -= range_pad * span; hi += range_pad * span
    shared_range = list(zip(lo.tolist(), hi.tolist()))

    # Track pre-existing matplotlib figure ids so we can clean up any
    # orphans that strict corner versions may create.
    pre_existing = set(plt.get_fignums())

    fig = None
    for i, (name, post) in enumerate(posteriors.items()):
        fig = plot_corner(
            post, fig=fig, plot_params=keys,
            truth=truth if i == 0 else None,
            overplots=overplots if i == 0 else None,
            color=colors[name], latex=latex, range=shared_range,
        )

    # Close any orphans corner.corner may have created on overlay (a
    # symptom of cross-version disagreement on figure-reuse semantics).
    for fid in list(plt.get_fignums()):
        if fid not in pre_existing and plt.figure(fid) is not fig:
            plt.close(fid)

    handles = []
    for name in posteriors:
        handles.append(Patch(facecolor=colors[name], edgecolor="none",
                             alpha=0.6, label=name))
    if truth is not None and truth_label:
        handles.append(Line2D([0], [0], color="black", lw=1.5, label=truth_label))
    if overplots:
        for label in overplots:
            handles.append(Line2D([0], [0], marker="*", markersize=12,
                                  linestyle="none", markerfacecolor="red",
                                  markeredgecolor="red", label=label))
    legend_kwargs = {"fontsize": 12, **(legend_kwargs or {})}
    fig.legend(handles=handles, loc=legend_loc, frameon=False, **legend_kwargs)
    return fig
