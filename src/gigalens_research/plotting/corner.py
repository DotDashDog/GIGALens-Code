"""Corner plots from :class:`Posterior` objects.

Two entry points:

- :func:`plot_corner` — one posterior, optional truth and overlay points.
- :func:`plot_corner_overlay` — multiple posteriors on the same figure,
  with a built-in legend.

Both consume :class:`Posterior` views directly (they grab ``flat_x`` for
samples, ``median_x`` for the median, etc.), so callers don't have to know
about array shapes or the bijector.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import corner as _corner_pkg
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .labels import flatten_params, flatten_param_names, latex_label


def _samples_to_matrix(
    posterior,
    plot_params: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Pull samples out of a posterior as an ``(n, p)`` matrix and the matching
    parameter labels. For point estimates (no flat_x), a single-row matrix is
    returned for completeness, though corner plots aren't meaningful in that case.
    """
    # ``_index_collisions=True`` (the default) is essential here: without it,
    # two profiles in the same group that share a parameter name (e.g.
    # ``e1``/``center_x`` across several mass profiles, or ``center_x`` across
    # multiple source profiles) collapse to a single column and the rest are
    # silently dropped — a multi-profile / multi-band model would lose ~half its
    # parameters. The ``__<i>`` profile-index suffix keeps every parameter a
    # distinct, attributable column. Passed explicitly to pin the behavior.
    if hasattr(posterior, "flat_x"):
        flat = flatten_params(posterior.flat_x, _index_collisions=True)
    else:
        flat = flatten_params(posterior.x if hasattr(posterior, "x") else
                              posterior.z_to_x(posterior.median_z),
                              _index_collisions=True)
    if plot_params is None:
        plot_params = list(flat.keys())
    cols = [np.asarray(flat[k]).reshape(-1) for k in plot_params]
    samples = np.vstack(cols).T
    return samples, list(plot_params)


def _point_to_row(
    point_x: Any,
    plot_params: Sequence[str],
    *,
    what: str = "truth",
) -> np.ndarray:
    """Convert a single physical-space point (nested-dict structure with batch
    dim 1) into a 1-D array in ``plot_params`` order.

    Parameters the point does not define are filled with ``NaN`` (rather than
    raising). This is the common case when the truth model and the fitted
    model differ in their light parameterization — e.g. approximating a Vela
    ``ImageBasedLight`` truth (``center_x``/``center_y``) with a shapelet
    source (``src_beta``, ``src_n_max``, …). ``corner`` skips non-finite
    truth/overlay entries, so unmatched parameters simply get no marker.
    """
    # Must match _samples_to_matrix's disambiguation so truth columns align
    # with the sample columns when profiles share parameter names.
    flat = flatten_params(point_x, _index_collisions=True)
    missing = [k for k in plot_params if k not in flat]
    if missing:
        warnings.warn(
            f"{what} point is missing {len(missing)} of {len(plot_params)} "
            f"plotted parameters; these will have no marker: {missing}.",
            stacklevel=2,
        )
    return np.array([
        float(np.squeeze(np.asarray(flat[k]))) if k in flat else np.nan
        for k in plot_params
    ])


def plot_corner(
    posterior,
    *,
    fig: Optional[Figure] = None,
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
        Source of samples (uses ``flat_x``).
    plot_params : list of str, optional
        Subset of parameter labels (using the registry from :mod:`.labels`).
        Default: all.
    truth : nested-dict params, optional
        A truth point in the same physical-space structure as
        ``posterior.median_x``. Drawn as crosshairs.
    overplots : dict, optional
        Map ``{legend_label: point_x}`` of additional points to overplot as
        stars. Useful for marking a MAP point on top of HMC samples.
    """
    samples, plot_params = _samples_to_matrix(posterior, plot_params)
    labels = [latex_label(k) for k in plot_params] if latex else list(plot_params)

    truth_row = None if truth is None else _point_to_row(truth, plot_params)

    defaults = dict(show_titles=True, title_fmt=".3f", color=color,
                    hist_kwargs={"density": True, "color": color})
    defaults.update(corner_kwargs)

    fig = _corner_pkg.corner(
        samples, fig=fig, truths=truth_row, truth_color=truth_color,
        labels=labels, **defaults,
    )

    if overplots:
        for _name, pt in overplots.items():
            row = _point_to_row(pt, plot_params, what=f"overplot {_name!r}")
            _corner_pkg.overplot_points(
                fig, row[None, :], marker="*", markersize=18,
                mfc=overplot_color, mec=overplot_color,
            )
    return fig


def plot_corner_overlay(
    posteriors: Dict[str, Any],
    *,
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
    default_palette = ["blue", "black", "orange", "purple", "brown"]
    colors = colors or {name: default_palette[i % len(default_palette)]
                        for i, name in enumerate(posteriors)}

    # Resolve plot_params from the first posterior if not provided, then
    # collect per-posterior sample matrices in one pass so we can compute a
    # shared axis range.
    if plot_params is None:
        first_post = next(iter(posteriors.values()))
        _, plot_params = _samples_to_matrix(first_post)
    name_to_samples = {
        name: _samples_to_matrix(post, plot_params)[0]
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
    for i, (name, _post) in enumerate(posteriors.items()):
        post = posteriors[name]
        fig = plot_corner(
            post, fig=fig, plot_params=plot_params,
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
