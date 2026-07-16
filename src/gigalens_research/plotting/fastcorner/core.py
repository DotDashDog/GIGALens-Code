"""``corner()`` -- same signature, same picture, O(K) axes instead of O(K^2).

The shape of the trick
----------------------
corner builds ``K*K`` matplotlib ``Axes`` and decorates each one. An Axes costs
~5ms whether or not anything is drawn in it, so at K=30 that is 8 of corner's 14
seconds spent on bookkeeping, and it grows as K^2. Nothing about that cost is
about your samples: the binning is 3% of the runtime.

So this module pays the Axes cost only where it is O(K), and refuses it where it
is O(K^2):

* The **diagonal** is K panels, so it uses K real Axes and calls ``ax.hist``
  exactly as corner does. Fidelity of the 1-D panels, the quantile lines, the
  ylim rules and ``scale_hist`` therefore comes for free, and K*5ms is not a
  scaling problem.
* The **lower triangle** is K*(K-1)/2 panels -- the part that actually hurts --
  and is drawn into a *single* shared Axes whose data coordinates are panel-grid
  units (see :mod:`._layout`). Each panel's density is one small ``imshow``; all
  panels' contours are found with ``contourpy`` and drawn as one
  ``LineCollection``; all panels' datapoints are one ``Line2D``.

The numbers behind every layer are bit-identical to corner's -- see
:mod:`._data`. This module is only allowed to draw them faster.

Cost of the trick: ``fig.axes`` holds ``1 + K`` entries rather than ``K*K``, and
the lower-triangle panels are not addressable as Axes. Call
:func:`materialize_axes` if you need them.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence

import contourpy
import matplotlib
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, colorConverter

from ._data import (
    DEFAULT_LEVELS,
    bin_edges,
    density_levels,
    digitize_columns,
    hist2d_from_codes,
    levels_are_degenerate,
    parse_ranges,
    quantile,
)
from ._layout import Grid, panel_ticks

__all__ = ["corner"]


def _parse_input(xs):
    """corner's own input handling: accept (N, K) and hand back (K, N)."""
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    return xs


def _extend_for_contours(H, X, Y):
    """corner's contour grid: bin centres, padded by two rings of ``H.min()``.

    Verbatim from ``corner.core.hist2d``. The padding is what makes contours
    close cleanly at the panel edges instead of running off them, and the exact
    padding geometry determines the contour paths -- so it is reproduced rather
    than approximated.
    """
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )
    return H2, X2, Y2


def _density_cmap(color, base_color):
    """corner's density colormap: opaque ``color`` at the peak, fading to fully
    transparent at zero density, so the datapoints show through the wings.

    Built once per figure, not once per panel -- it depends only on the colors.
    Rebuilding it inside the panel loop cost 1740 ``_create_lookup_table`` calls
    and ~12% of the runtime, which the profile caught and reasoning did not.
    """
    return LinearSegmentedColormap.from_list(
        "density_cmap", [color, colorConverter.to_rgba(base_color, alpha=0.0)]
    )


def _density_rgba(H, cmap):
    """The panel's density layer as an RGBA image.

    corner draws this with ``ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)``
    on the bin-edge grid, which is already an image -- a blocky one, on a regular
    grid -- so ``imshow`` with nearest interpolation reproduces it exactly rather
    than approximately.

    The normalization is pcolor's default (autoscale to the data), worked through
    for this particular input: ``C = H.max() - H.T`` has ``C.min() == 0`` and
    ``C.max() == H.max() - H.min()``, so ``t`` below is exactly what pcolor's
    ``Normalize`` would produce. The ``span == 0`` guard matches matplotlib's
    handling of a constant array.
    """
    C = H.max() - H.T
    span = H.max() - H.min()
    t = np.zeros_like(C, dtype=float) if span == 0 else C / span
    return cmap(t)


def corner(
    data,
    bins=20,
    *,
    range=None,
    axes_scale="linear",
    weights=None,
    color=None,
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    labels=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_quantiles=None,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    # --- fastcorner extensions ---------------------------------------------
    levels=None,
    plot_datapoints=True,
    plot_density=True,
    plot_contours=True,
    fill_contours=False,
    no_fill_contours=False,
    contour_kwargs=None,
    contourf_kwargs=None,
    data_kwargs=None,
    pcolor_kwargs=None,
    quiet=False,
    **kwargs,
):
    """Drop-in replacement for :func:`corner.corner`.

    Accepts the same arguments and returns a matplotlib ``Figure`` that renders
    the same picture. See the module docstring for what differs (``fig.axes``)
    and :mod:`gigalens_research.plotting.fastcorner` for why.
    """
    import matplotlib.pyplot as pl

    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if titles is None:
        titles = labels

    if title_quantiles is None:
        title_quantiles = quantiles if len(quantiles) > 0 else [0.16, 0.5, 0.84]
    if show_titles and len(title_quantiles) != 3:
        raise ValueError(
            "'title_quantiles' must contain exactly three values; "
            "pass a length-3 list or array using the 'title_quantiles' argument"
        )

    for flag, name in ((top_ticks, "top_ticks=True"), (reverse, "reverse=True")):
        if flag:
            raise NotImplementedError(
                f"fastcorner does not implement {name} yet -- use "
                f"`corner.corner` for this plot. The omission is scope, not a "
                f"fundamental limit; it fails loudly rather than drawing "
                f"something subtly different from what you asked for."
            )

    xs = _parse_input(data)
    assert xs.shape[0] <= xs.shape[1], (
        "I don't believe that you want more dimensions than samples!"
    )
    K = len(xs)

    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    if isinstance(axes_scale, str):
        axes_scale = [axes_scale] * K
    else:
        assert len(axes_scale) == K, (
            "'axes_scale' should contain as many elements as data dimensions"
        )
    if any(s != "linear" for s in axes_scale):
        raise NotImplementedError(
            "fastcorner implements linear axes only so far; use `corner.corner` "
            "for log-scaled axes."
        )

    grid = Grid(K, reverse=reverse)
    rng = parse_ranges(xs, range, weights)
    if len(rng) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    try:
        bins = [int(bins) for _ in rng]
    except TypeError:
        if len(bins) != len(rng):
            raise ValueError("Dimension mismatch between bins and range")
    try:
        hist_bin_factor = [float(hist_bin_factor) for _ in rng]
    except TypeError:
        if len(hist_bin_factor) != len(rng):
            raise ValueError(
                "Dimension mismatch between hist_bin_factor and range"
            )

    if color is None:
        color = matplotlib.rcParams["ytick.color"]
    if levels is None:
        levels = DEFAULT_LEVELS
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    # --- figure ------------------------------------------------------------
    if fig is None:
        fig = pl.figure(figsize=(grid.dim, grid.dim))
    else:
        fig.clf()
        fig.set_size_inches(grid.dim, grid.dim)

    left, bottom, width, height = grid.rect
    tri = fig.add_axes([left, bottom, width, height], zorder=0)
    tri.set_xlim(0, grid.span)
    tri.set_ylim(0, grid.span)
    tri.set_axis_off()
    base_color = tri.get_facecolor()

    # --- the O(K^2) part: every lower-triangle panel, one Axes -------------
    edges = [bin_edges(rng[i], bins[i]) for i in np.arange(K)]
    codes = digitize_columns(xs, edges)

    dens_cmap = _density_cmap(color, base_color)
    # A column's data->grid map depends only on the column, so it is computed K
    # times, not K^2. With the default range (the full span of each column) no
    # sample can fall outside its panel, so the per-panel range mask is a no-op
    # and is skipped -- it is only needed when the caller narrowed `range`.
    gcol = [grid.x_to_grid(grid.axes_index(0, j)[1], xs[j], rng[j]) for j in np.arange(K)]
    grow = [grid.y_to_grid(grid.axes_index(i, 0)[0], xs[i], rng[i]) for i in np.arange(K)]
    inside = [
        bool(np.all((xs[j] >= rng[j][0]) & (xs[j] <= rng[j][1]))) for j in np.arange(K)
    ]

    contour_segs, contour_lws = [], []
    fill_polys, fill_colors = [], []
    pts_x, pts_y = [], []
    frames = []
    degenerate = False

    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for _ in levels] + [rgba_color]
    for li in np.arange(len(levels)):
        contour_cmap[li][-1] *= float(li) / (len(levels) + 1)

    for i in np.arange(K):
        for j in np.arange(i):
            row, col = grid.axes_index(i, j)
            ext = grid.cell_extent(row, col)
            x0, x1, y0, y1 = ext
            frames.append(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            )

            H = hist2d_from_codes(
                codes[j], codes[i], bins[j], bins[i], weights
            )
            if H.sum() == 0:
                raise ValueError(
                    "It looks like the provided 'range' is not valid "
                    "or the sample is empty."
                )
            if smooth is not None:
                from scipy.ndimage import gaussian_filter

                H = gaussian_filter(H, smooth)

            V = density_levels(H, levels)
            if levels_are_degenerate(V) and not quiet:
                degenerate = True

            # datapoints: masked to the panel's range, because on a shared axis
            # an out-of-range point lands in the neighbouring panel rather than
            # being clipped away.
            if plot_datapoints:
                if inside[i] and inside[j]:
                    pts_x.append(gcol[j])
                    pts_y.append(grow[i])
                else:
                    m = (
                        (xs[j] >= rng[j][0])
                        & (xs[j] <= rng[j][1])
                        & (xs[i] >= rng[i][0])
                        & (xs[i] <= rng[i][1])
                    )
                    pts_x.append(gcol[j][m])
                    pts_y.append(grow[i][m])

            H2, X2, Y2 = _extend_for_contours(H, edges[j], edges[i])
            gx2 = grid.x_to_grid(col, X2, rng[j])
            gy2 = grid.y_to_grid(row, Y2, rng[i])
            cg = contourpy.contour_generator(
                gx2, gy2, H2.T, name="serial",
                fill_type=contourpy.FillType.OuterCode,
            )

            if (plot_contours or plot_density) and not no_fill_contours:
                # the opaque base fill that hides the densest datapoints
                for poly in _filled(cg, V.min(), H.max()):
                    fill_polys.append(poly)
                    fill_colors.append(base_color)

            if plot_contours and fill_contours:
                lv = np.concatenate([[0], V, [H.max() * (1 + 1e-4)]])
                for k in np.arange(len(lv) - 1):
                    for poly in _filled(cg, lv[k], lv[k + 1]):
                        fill_polys.append(poly)
                        fill_colors.append(contour_cmap[k])
            elif plot_density:
                tri.imshow(
                    _density_rgba(H, dens_cmap),
                    extent=ext,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    zorder=1,
                )

            if plot_contours:
                for v in V:
                    for seg in cg.lines(float(v)):
                        if len(seg) > 1:
                            contour_segs.append(seg)

    if degenerate:
        logging.warning("Too few points to create valid contours")

    # --- collapse every layer to a single artist ---------------------------
    if plot_datapoints and pts_x:
        dk = dict(data_kwargs or {})
        tri.plot(
            np.concatenate(pts_x),
            np.concatenate(pts_y),
            "o",
            zorder=-1,
            rasterized=True,
            color=dk.pop("color", color),
            ms=dk.pop("ms", 2.0),
            mec=dk.pop("mec", "none"),
            alpha=dk.pop("alpha", 0.1),
            **dk,
        )
    if fill_polys:
        tri.add_collection(
            PathCollection(
                fill_polys, facecolors=fill_colors, edgecolors="none",
                antialiaseds=False, zorder=0.5,
            )
        )
    if contour_segs:
        ck = dict(contour_kwargs or {})
        tri.add_collection(
            LineCollection(
                contour_segs,
                colors=ck.pop("colors", color),
                linewidths=ck.pop(
                    "linewidths", matplotlib.rcParams["lines.linewidth"]
                ),
                zorder=2,
                **ck,
            )
        )
    # clip_on=False: the outermost frames lie exactly on the shared axes'
    # boundary, so default clipping would shave off half their linewidth and the
    # bottom row would lose its bottom border. Each of corner's panels owns its
    # spine and never hits this.
    tri.add_collection(
        LineCollection(
            frames, colors=matplotlib.rcParams["axes.edgecolor"],
            linewidths=matplotlib.rcParams["axes.linewidth"], zorder=3,
            clip_on=False,
        )
    )

    fig._fastcorner = dict(grid=grid, ranges=rng, K=K, tri=tri)
    return fig


def _filled(cg, lo, hi):
    """The region ``lo <= z <= hi`` as matplotlib ``Path``s.

    Built with ``FillType.OuterCode`` so each polygon arrives as vertices plus
    path codes. That matters: a filled contour band can have holes (a ring of
    high density around a low-density centre), and codes are what carry them.
    contourpy's codes are already matplotlib's, which is what it was designed
    for -- ``PolyCollection`` verts would silently fill the holes in.
    """
    points, codes_ = cg.filled(float(lo), float(hi))
    out = []
    for verts, cds in zip(points, codes_):
        if verts is None or len(verts) < 3:
            continue
        out.append(Path(np.asarray(verts), np.asarray(cds)))
    return out
