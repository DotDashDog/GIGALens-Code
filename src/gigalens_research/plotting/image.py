"""Image-plane rendering primitives.

These are pure plotters: arrays in, matplotlib axes drawn. The compound
panels in :mod:`.reports` orchestrate them; the :class:`Posterior` view in
:mod:`gigalens_research.inference_utils.posterior` provides the arrays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import kstest, norm


def plot_image(
    ax: Axes,
    img: np.ndarray,
    *,
    fig: Optional[Figure] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    title: Optional[str] = None,
    residual: bool = False,
    colorbar: bool = True,
    remove_axis: bool = True,
    log_vmin: float = 1e-2,
    log_norm: bool = True,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> matplotlib.image.AxesImage:
    """Draw a 2-D image with GIGALens' default styling.

    Two presets:

    - ``residual=False`` (default): inferno + ``LogNorm`` (good for lensed
      arcs and broad dynamic range).
    - ``residual=True``: bwr + ``CenteredNorm`` (good for normalized residuals).

    ``vmin`` / ``vmax`` override the auto-computed color limits. Supplying them
    is the way to make several panels share a single color scale (e.g. a
    truth-vs-recovered comparison); see
    :func:`gigalens_research.plotting.truth.plot_source_comparison`.

    Returns the ``AxesImage`` so callers can adjust further.
    """
    if residual:
        norm_ = matplotlib.colors.CenteredNorm()
        cmap_ = cmap or "bwr"
    else:
        arr = np.asarray(img)
        finite_max = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 0.0
        hi = finite_max if vmax is None else float(vmax)
        if log_norm and hi > log_vmin:
            # LogNorm requires strictly positive bounds with vmin < vmax. Fall
            # back to a linear norm when the image is too flat / non-positive
            # (e.g. early-iteration MAP results, or all-zero source planes).
            lo = float(max(np.nanmin(arr), log_vmin)) if vmin is None else float(vmin)
            lo = max(lo, log_vmin)
            lo = min(lo, hi * 0.99)  # ensure vmin < vmax
            norm_ = matplotlib.colors.LogNorm(vmin=lo, vmax=hi, clip=True)
        else:
            norm_ = matplotlib.colors.Normalize(
                vmin=None if vmin is None else float(vmin),
                vmax=None if vmax is None else float(vmax),
            )
        cmap_ = cmap or "inferno"

    im = ax.imshow(img, cmap=cmap_, norm=norm_, extent=extent, origin="lower")
    if colorbar:
        fig = fig or ax.figure
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)
    if title is not None:
        ax.set_title(title)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    if remove_axis:
        ax.set_xticks([]); ax.set_xticks([], minor=True)
        ax.set_yticks([]); ax.set_yticks([], minor=True)
    return im


def normalized_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    err_map: np.ndarray,
) -> np.ndarray:
    """Compute ``(observed - predicted) / err_map``.

    ``err_map`` is the per-pixel noise σ. Use
    :meth:`Posterior.err_map_at(predicted)` to get the right σ for either the
    forward or backward gigalens noise convention.
    """
    return (np.asarray(observed) - np.asarray(predicted)) / np.asarray(err_map)


def plot_residual_histogram(
    ax: Axes,
    residual: np.ndarray,
    *,
    bins: int = 50,
    title: Optional[str] = None,
) -> None:
    """Histogram of pixel residuals with a fitted Gaussian and a KS p-value.

    For a well-calibrated model and noise model, this should be ~N(0, 1)."""
    flat = np.asarray(residual).flatten()
    mu, std = norm.fit(flat)
    p = kstest(flat, norm.cdf).pvalue
    xs = np.linspace(np.min(flat), np.max(flat), 200)
    ax.hist(flat, bins=bins, density=True,
            label=f"μ={mu:.4f}\nσ={std:.4f}\nKS p={p:.4f}")
    ax.plot(xs, norm.pdf(xs, mu, std))
    if title is not None:
        ax.set_title(title)
    ax.legend()
