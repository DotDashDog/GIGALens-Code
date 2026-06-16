"""Image-plane rendering primitives.

These are pure plotters: arrays in, matplotlib axes drawn. The compound
panels in :mod:`.reports` orchestrate them; the :class:`Posterior` view in
:mod:`gigalens_research.inference_utils.posterior` provides the arrays.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import kstest, norm

_SCALE_CHOICES = ("asinh", "log", "sqrt", "linear")
Scale = Literal["asinh", "log", "sqrt", "linear"]


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
    scale: Scale = "asinh",
    linear_width: Optional[float] = None,
    log_vmin: float = 1e-2,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    # Deprecated: use scale="log" instead.
    log_norm: Optional[bool] = None,
) -> matplotlib.image.AxesImage:
    """Draw a 2-D image with GIGALens' default styling.

    Two presets:

    - ``residual=False`` (default): inferno colormap with the color scale
      set by ``scale`` (default ``"asinh"``).
    - ``residual=True``: bwr + ``CenteredNorm``.

    Parameters
    ----------
    scale : {"asinh", "log", "sqrt", "linear"}
        Color-scale normalization for non-residual images.

        - ``"asinh"`` (default): :class:`matplotlib.colors.AsinhNorm`. Handles
          zeros and mildly negative pixels gracefully; shows faint structure
          without a hard floor. The linear-to-log transition is set by
          ``linear_width`` (default: ``vmax / 100``).
        - ``"log"``: :class:`matplotlib.colors.LogNorm`. Requires a positive
          floor; set via ``log_vmin``.
        - ``"sqrt"``: :class:`matplotlib.colors.PowerNorm` with ``gamma=0.5``.
        - ``"linear"``: :class:`matplotlib.colors.Normalize`.

        Pixels below zero are clipped to zero for ``"asinh"`` and ``"sqrt"``.

    linear_width : float, optional
        The ``a`` in ``asinh(x / a)``. Below ``a`` the scale is nearly
        linear; above it is nearly logarithmic. Defaults to ``vmax / 100``
        when ``scale="asinh"``. Ignored for other scales.
    log_vmin : float
        Lower bound for ``"log"`` scale (``LogNorm(vmin=log_vmin)``).
        Unused for other scales.
    vmin, vmax : float, optional
        Explicit color limits. Supplying both is the standard way to share
        a single colorbar across panels (e.g. truth-vs-recovered); see
        :func:`gigalens_research.plotting.truth.plot_source_comparison`.

    Returns the ``AxesImage`` so callers can adjust further.
    """
    # Backward-compat shim: log_norm=True was the old default.
    if log_norm is not None:
        scale = "log" if log_norm else "linear"

    if residual:
        norm_ = matplotlib.colors.CenteredNorm()
        cmap_ = cmap or "bwr"
    else:
        arr = np.asarray(img, dtype=float)
        finite_vals = arr[np.isfinite(arr)]
        finite_max = float(finite_vals.max()) if finite_vals.size else 0.0
        hi = finite_max if vmax is None else float(vmax)

        if scale == "asinh":
            arr_clipped = np.clip(arr, 0.0, None)
            lw = float(linear_width) if linear_width is not None else max(hi / 100.0, 1e-12)
            lo = 0.0 if vmin is None else float(vmin)
            norm_ = matplotlib.colors.AsinhNorm(
                linear_width=lw, vmin=lo, vmax=hi if hi > 0 else 1.0,
            )
            img = arr_clipped

        elif scale == "sqrt":
            arr_clipped = np.clip(arr, 0.0, None)
            lo = 0.0 if vmin is None else float(vmin)
            norm_ = matplotlib.colors.PowerNorm(
                gamma=0.5, vmin=lo, vmax=hi if hi > 0 else 1.0,
            )
            img = arr_clipped

        elif scale == "log":
            # LogNorm requires strictly positive bounds with vmin < vmax.
            # Fall back to linear when the image is too flat / non-positive.
            if hi > log_vmin:
                lo = float(max(np.nanmin(arr), log_vmin)) if vmin is None else float(vmin)
                lo = max(lo, log_vmin)
                lo = min(lo, hi * 0.99)
                norm_ = matplotlib.colors.LogNorm(vmin=lo, vmax=hi, clip=True)
            else:
                norm_ = matplotlib.colors.Normalize(
                    vmin=None if vmin is None else float(vmin),
                    vmax=None if vmax is None else float(vmax),
                )

        else:  # "linear"
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
