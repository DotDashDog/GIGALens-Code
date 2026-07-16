"""Plotting helpers for simulated-system diagnostics (truth-aware).

Composes :mod:`inference_utils.truth_diagnostics` (data layer) with the
existing image primitives. The compound panel built on top lives on
:class:`PosteriorReport` in :mod:`.reports`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..inference_utils.truth_diagnostics import (
    filter_keys_by_kind,
    source_comparison,
    z_scores,
)
from ..param_index import param_sites, site_labels
from .image import plot_image


def plot_z_scores(
    ax: Axes,
    posterior,
    truth_x,
    *,
    kind: Optional[str] = "mass",
    params: Optional[Sequence[str]] = None,
    color_pos: str = "C0",
    color_neg: str = "C3",
    threshold: float = 2.0,
    sort_by_abs: bool = False,
    latex: bool = True,
) -> None:
    """Bar plot of per-parameter z-scores against truth.

    ``z = (x_truth - x_median) / σ`` with asymmetric ±1σ quantiles (see
    :func:`z_scores`). Bars above ``threshold`` are accented (default ±2σ)
    so biased fits stand out at a glance.

    Parameters
    ----------
    kind : str
        ``'mass'`` (default), ``'cosmology'``, ``'geometry'``, ``'light'``,
        ``'all'``, or ``None``. Used only when ``params`` is not given.
    params : list of str, optional
        Explicit subset of scene path keys (``"planes/0/mass/0/theta_E"``),
        in the order given. Overrides ``kind``.
    threshold : float
        Reference horizontal lines at ±``threshold``σ; set ``None`` to skip.
    sort_by_abs : bool
        Sort bars by descending ``|z|`` (puts the most-biased on the left).
    """
    zs = z_scores(posterior, truth_x)
    if params is None:
        params = filter_keys_by_kind(list(zs.keys()), kind)
    params = list(params)

    by_key = {s.key: s for s in param_sites(posterior)}
    # z_scores drops any parameter the truth is silent on, so a key can be
    # absent for two very different reasons. A bare KeyError on `zs` below reads
    # the same for both.
    unknown = [k for k in params if k not in zs]
    if unknown:
        unscored = [k for k in unknown if k in by_key]
        raise KeyError(
            f"cannot plot {len(unknown)} requested parameter(s). "
            + (f"The truth does not define {unscored}. " if unscored else "")
            + (
                f"This model has no {[k for k in unknown if k not in by_key]}; "
                f"parameters are keyed by scene path. "
                if len(unscored) < len(unknown)
                else ""
            )
            + f"Scored: {sorted(zs)}."
        )
    if not params:
        raise ValueError(
            f"nothing to plot: the truth scores no parameter of kind {kind!r}. "
            f"Scored: {sorted(zs)}."
        )
    if sort_by_abs:
        params = sorted(params, key=lambda k: -abs(zs[k]))

    values = np.array([zs[k] for k in params])
    colors = [color_pos if v >= 0 else color_neg for v in values]
    xpos = np.arange(len(params))
    ax.bar(xpos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xpos)
    # Label from the ParamSite records, not the key: a bare label renders only
    # the parameter name, so on a multiplane model every plane's `e1` would read
    # `$\epsilon_1$` and the bars would be unattributable. site_labels adds the
    # plane/component superscript exactly where two labels would collide.
    ax.set_xticklabels(
        site_labels([by_key[k] for k in params], latex=latex),
        rotation=45, ha="right",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    if threshold is not None:
        ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(-threshold, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"$z = (x_{\rm true} - x_{\rm med})/\sigma_{\pm 1}$")
    title_kind = "all" if kind is None else kind
    ax.set_title(f"Truth z-scores ({title_kind})")


def plot_source_comparison(
    fig: Figure,
    posterior,
    truth_source,
    *,
    extent: Optional[Tuple[float, float, float, float]] = None,
    point: str = "median",
    scale: str = "asinh",
    linear_width: Optional[float] = None,
    log_vmin: float = 1e-2,
    titles: Optional[Tuple[str, str, str]] = None,
    grid_pix: Optional[int] = None,
    fov_arcsec: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> Tuple[Axes, Axes, Axes]:
    """1×3 panel: truth source, recovered source, residual (truth − recovered).

    ``truth_source`` is either a pre-rendered 2-D image (with ``extent``) or
    a callable ``f(X, Y) -> image`` (e.g. wrapping a gigalens
    :class:`LightProfile`). See :func:`source_comparison` for full input
    semantics.

    Truth and recovered share one color scale (a common ``vmax`` computed from
    both panels, with ``vmin=0`` since negative pixels are clipped) so their
    brightnesses are directly comparable. The residual uses a centered diverging
    map. ``scale`` controls the normalization (default ``"asinh"``); see
    :func:`plot_image` for the full set of options.

    Returns the three axes so the caller can post-decorate (e.g. overlay
    caustics with :func:`plot_caustics`).
    """
    truth, recovered, residual, extent = source_comparison(
        posterior, truth_source,
        extent=extent, point=point,
        grid_pix=grid_pix, fov_arcsec=fov_arcsec, center=center,
    )
    titles = titles or ("Truth source", f"Recovered ({point})", "Truth − Recovered")

    # Shared color limits: joint vmax across both panels, vmin=0 (negatives
    # clipped for asinh/sqrt) or log_vmin for log scale.
    both = np.concatenate([
        np.clip(np.asarray(truth).ravel(), 0, None),
        np.clip(np.asarray(recovered).ravel(), 0, None),
    ])
    finite = both[np.isfinite(both)]
    shared_vmax = float(finite.max()) if finite.size else 1.0
    shared_vmin = log_vmin if scale == "log" else 0.0

    # For asinh, share the linear_width computed from the joint vmax so both
    # panels have identical knee positions (critical for a fair comparison).
    lw = linear_width
    if scale == "asinh" and lw is None:
        lw = max(shared_vmax / 100.0, 1e-12)

    axs = fig.subplots(1, 3)
    plot_image(axs[0], truth, fig=fig, extent=extent, title=titles[0],
               scale=scale, linear_width=lw, log_vmin=log_vmin,
               vmin=shared_vmin, vmax=shared_vmax, remove_axis=False)
    plot_image(axs[1], recovered, fig=fig, extent=extent, title=titles[1],
               scale=scale, linear_width=lw, log_vmin=log_vmin,
               vmin=shared_vmin, vmax=shared_vmax, remove_axis=False)
    plot_image(axs[2], residual, fig=fig, extent=extent, title=titles[2],
               residual=True, remove_axis=False)
    for ax in axs:
        ax.set_xlabel("x [arcsec]"); ax.set_ylabel("y [arcsec]")
    return tuple(axs)
