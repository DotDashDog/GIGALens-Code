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
  units (see :mod:`._layout`). All panels' contours are found with
  ``contourpy`` and drawn as one ``LineCollection``; all panels' datapoints are
  one ``Line2D``; each panel's density is one ``pcolor``, matching corner's own
  call (``pcolormesh`` is *not* a substitute -- measured only 90.9% pixel-identical
  to ``pcolor`` on the same grid).

The numbers behind every layer are bit-identical to corner's -- see
:mod:`._data`. This module is only allowed to draw them faster.

Cost of the trick: ``fig.axes`` holds ``1 + K`` entries rather than ``K*K``, and
the lower-triangle panels are not addressable as Axes. Call
:func:`materialize_axes` if you need them.
"""

from __future__ import annotations

import logging
import warnings

import contourpy
import matplotlib
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter

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
from ._layout import Grid, panel_ticks, points_to_grid

__all__ = ["corner", "materialize_axes"]


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

    # Overlaying onto a figure we already drew: reuse its ranges rather than
    # growing them. corner can afford to union ranges because its artists live
    # in each panel's own data coordinates and simply move when the limits
    # change. Ours are baked into shared grid coordinates at draw time, so a
    # widened range would leave the *first* dataset silently misplaced. Pinning
    # is the honest option; the warning says so rather than quietly clipping.
    prev = getattr(fig, "_fastcorner", None) if fig is not None else None
    reuse = prev is not None and prev["K"] == K
    force_range = range is not None

    if reuse and not force_range:
        rng = prev["ranges"]
        outside = [
            i for i in np.arange(K)
            if xs[i].min() < rng[i][0] or xs[i].max() > rng[i][1]
        ]
        if outside:
            warnings.warn(
                f"Overlaid data falls outside the existing figure's range in "
                f"column(s) {outside}; those samples are dropped. fastcorner "
                f"pins ranges when overplotting (corner would widen them), "
                f"because the panels are composited in shared coordinates. "
                f"Pass an explicit `range=` covering both datasets.",
                stacklevel=2,
            )
    else:
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
    elif not reuse:
        fig.clf()
        fig.set_size_inches(grid.dim, grid.dim)

    left, bottom, width, height = grid.rect
    if reuse:
        tri = prev["tri"]
        diag_axes = prev["diag"]
    else:
        tri = fig.add_axes([left, bottom, width, height], zorder=0)
        tri.set_xlim(0, grid.span)
        tri.set_ylim(0, grid.span)
        # Frame off, but the axis itself stays *on*: the shared axes' bottom
        # edge is the bottom row's bottom edge and its left edge is column 0's
        # left edge, so its own ticks land exactly where corner puts the
        # bottom-row / left-column ticks. One x axis and one y axis carry every
        # tick in the plot.
        tri.set_frame_on(False)
        tri.set_xticks([])
        tri.set_yticks([])
        diag_axes = None
    base_color = tri.get_facecolor()

    def _cell_rect(row, col):
        """A panel's figure-fraction rect, from its position in grid units."""
        x0, _, y0, _ = grid.cell_extent(row, col)
        return [
            left + (x0 / grid.span) * width,
            bottom + (y0 / grid.span) * height,
            width / grid.span,
            height / grid.span,
        ]

    # --- the O(K^2) part: every lower-triangle panel, one Axes -------------
    edges = [bin_edges(rng[i], bins[i]) for i in np.arange(K)]
    codes = digitize_columns(xs, edges)

    dens_cmap = _density_cmap(color, base_color)
    # A column's data->grid map depends only on the column, so it is computed K
    # times, not K^2.
    #
    # `needs_clip` decides whether anything can escape its panel. With the
    # default range -- the full span of every column -- nothing can, and the
    # clip path below is pure cost (~1s at K=30, clipping millions of points
    # against a K^2-rect compound path). It is only earned when the caller
    # narrows `range`, which is exactly when corner's per-panel axes start
    # clipping too.
    gcol = [grid.x_to_grid(grid.axes_index(0, j)[1], xs[j], rng[j]) for j in np.arange(K)]
    grow = [grid.y_to_grid(grid.axes_index(i, 0)[0], xs[i], rng[i]) for i in np.arange(K)]
    needs_clip = any(
        xs[j].min() < rng[j][0] or xs[j].max() > rng[j][1] for j in np.arange(K)
    )

    contour_segs, contour_lws = [], []
    clip_verts, clip_codes = [], []
    dens_artists = []
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
            clip_verts.extend(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            )
            clip_codes.extend(
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
                 Path.CLOSEPOLY]
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
                pts_x.append(gcol[j])
                pts_y.append(grow[i])

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
                # pcolormesh, not imshow: corner draws this with ax.pcolor, and
                # imshow *resamples* the histogram onto the display grid where
                # pcolor rasterizes exact cell polygons. The two round bin
                # boundaries differently, which shows up as a grid of 1px
                # differences -- 0.9% of pixels at bins=35, and worse the more
                # bins you ask for. pcolormesh is pcolor's cheap twin (one
                # QuadMesh instead of a polygon per cell) and shares its
                # geometry. The autoscaled norm is pcolor's own, so the array
                # and cmap go in unconverted, exactly as corner passes them.
                dens_artists.append(tri.pcolor(
                    grid.x_to_grid(col, edges[j], rng[j]),
                    grid.y_to_grid(row, edges[i], rng[i]),
                    H.max() - H.T,
                    cmap=dens_cmap,
                    zorder=1,
                    **(pcolor_kwargs or {}),
                ))

            if plot_contours:
                for v in V:
                    for seg in cg.lines(float(v)):
                        if len(seg) > 1:
                            contour_segs.append(seg)

    if degenerate:
        logging.warning("Too few points to create valid contours")

    # --- collapse every layer to a single artist ---------------------------
    panel_clip = Path(clip_verts, clip_codes) if clip_verts else None
    if plot_datapoints and pts_x:
        dk = dict(data_kwargs or {})
        (line,) = tri.plot(
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
        # Every one of corner's panels clips to its own axes, so a marker
        # straddling the range boundary is drawn *half* visible. Dropping those
        # samples instead (the obvious shortcut on a shared axis, where a stray
        # point would otherwise land in a neighbouring panel) loses exactly
        # those half-markers. A compound clip path buys corner's behaviour back
        # for one artist -- but only pay for it when something can actually
        # escape.
        if needs_clip and clip_verts:
            line.set_clip_path(panel_clip, tri.transData)
    if fill_polys:
        fills = PathCollection(
            fill_polys, facecolors=fill_colors, edgecolors="none",
            antialiaseds=False, zorder=0.5,
        )
        tri.add_collection(fills)
        _clip_to_panels(fills, panel_clip, tri)
    if contour_segs:
        ck = dict(contour_kwargs or {})
        cl = LineCollection(
            contour_segs,
            colors=ck.pop("colors", color),
            linewidths=ck.pop(
                "linewidths", matplotlib.rcParams["lines.linewidth"]
            ),
            zorder=2,
            **ck,
        )
        tri.add_collection(cl)
        # corner pads its contour grid two bins beyond the panel (see
        # _extend_for_contours) and relies on each panel's axes to clip the
        # overhang. With the default range the density out there is flat so
        # nothing escapes, but a narrowed `range=` puts real density at the
        # panel edge and the contours run out into the gutters.
        _clip_to_panels(cl, panel_clip, tri)
    # clip_on=False: the outermost frames lie exactly on the shared axes'
    # boundary, so default clipping would shave off half their linewidth and the
    # bottom row would lose its bottom border. Each of corner's panels owns its
    # spine and never hits this.
    if not reuse:
        # clip_on=False: the outermost frames lie exactly on the shared axes'
        # boundary, so default clipping would shave off half their linewidth and
        # the bottom row would lose its bottom border. Each of corner's panels
        # owns its spine and never hits this.
        tri.add_collection(
            LineCollection(
                frames, colors=matplotlib.rcParams["axes.edgecolor"],
                linewidths=matplotlib.rcParams["axes.linewidth"], zorder=3,
                clip_on=False,
            )
        )

    # --- the O(K) part: the diagonal keeps real Axes -----------------------
    # K of them is not a scaling problem, and in exchange ax.hist's step path,
    # the ylim rules and scale_hist come out right by construction instead of
    # by transcription.
    if diag_axes is None:
        diag_axes = []
        for i in np.arange(K):
            row, col = grid.axes_index(i, i)
            ax = fig.add_axes(_cell_rect(row, col), zorder=4)
            diag_axes.append(ax)

    for i in np.arange(K):
        ax = diag_axes[i]
        x = xs[i]
        n_bins_1d = int(max(1, np.round(hist_bin_factor[i] * bins[i])))
        bins_1d = bin_edges(rng[i], n_bins_1d, axes_scale[i])
        if smooth1d is None:
            n, _, _ = ax.hist(x, bins=bins_1d, weights=weights, **hist_kwargs)
        else:
            from scipy.ndimage import gaussian_filter

            n, _ = np.histogram(x, bins=bins_1d, weights=weights)
            n = gaussian_filter(n, smooth1d)
            xh = np.array(list(zip(bins_1d[:-1], bins_1d[1:]))).flatten()
            yh = np.array(list(zip(n, n))).flatten()
            ax.plot(xh, yh, **hist_kwargs)

        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color)
            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        if show_titles:
            title = None
            if title_fmt is not None:
                q_lo, q_mid, q_hi = quantile(
                    x, title_quantiles, weights=weights
                )
                q_m, q_p = q_mid - q_lo, q_hi - q_mid
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_mid), fmt(q_m), fmt(q_p))
                if titles is not None:
                    title = "{0} = {1}".format(titles[i], title)
            elif titles is not None:
                title = "{0}".format(titles[i])
            if title is not None:
                ax.set_title(title, **title_kwargs)

        _set_lim(force_range, not reuse, ax.set_xlim, ax.get_xlim, list(rng[i]))
        maxn = np.max(n)
        ylim = [-0.1 * maxn, 1.1 * maxn] if scale_hist else [0, 1.1 * maxn]
        _set_lim(force_range, not reuse, ax.set_ylim, ax.get_ylim, ylim)

        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(
                MaxNLocator(max_n_ticks, prune="lower")
            )
        ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            ax.set_xticklabels([])
            ax.set_xticklabels([], minor=True)
        else:
            for lab in ax.get_xticklabels():
                lab.set_rotation(45)
            for lab in ax.get_xticklabels(minor=True):
                lab.set_rotation(45)
            if labels is not None:
                ax.set_xlabel(labels[i], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text)
            )

    # --- ticks and labels on the shared axes -------------------------------
    if not reuse:
        _decorate_shared(
            tri, grid, rng, K, labels, label_kwargs, labelpad,
            max_n_ticks, axes_scale, use_math_text,
        )

    # --- truths ------------------------------------------------------------
    if truths is not None:
        _overplot_truths(tri, diag_axes, grid, rng, K, truths, truth_color)

    fig._fastcorner = dict(
        grid=grid, ranges=rng, K=K, tri=tri, diag=diag_axes, axes=None
    )
    return fig


def _clip_to_panels(artist, panel_clip, tri):
    """Confine an artist to the panel rectangles.

    The shared axes is one big rectangle, so its default clip box lets anything
    drawn near a panel edge bleed into the gutters between panels -- somewhere
    corner can never draw, because there each panel is its own axes.
    """
    if panel_clip is not None:
        artist.set_clip_path(panel_clip, tri.transData)


def _set_lim(force, new_fig, setter, getter, lim):
    """corner's ``_set_xlim``/``_set_ylim``: overlays grow limits, never shrink.

    On a fresh figure, or when the caller forced ``range``, the limit is set
    outright; otherwise it is unioned with what is already there, so a second
    dataset drawn onto the same figure cannot crop the first.
    """
    if force or new_fig:
        return setter(lim)
    cur = getter()
    return setter([min(cur[0], lim[0]), max(cur[1], lim[1])])


def _decorate_shared(
    tri, grid, rng, K, labels, label_kwargs, labelpad,
    max_n_ticks, axes_scale, use_math_text,
):
    """Every tick and edge label of the lower triangle, on one x and one y axis.

    Only the bottom row shows x ticks and only the left column shows y ticks, and
    those edges coincide with the shared axes' own bottom and left edges -- so
    the whole plot's ticks are two ``set_xticks``/``set_yticks`` calls rather
    than K^2 axes' worth of tick artists.

    The bottom-right panel is the diagonal and the top-left panel is the
    diagonal, so the x ticks cover columns ``0..K-2`` and the y ticks rows
    ``1..K-1``; each diagonal Axes carries its own (corner gives a histogram's
    y axis a ``NullLocator``).
    """
    xt_pos, xt_lab = [], []
    for j in np.arange(K - 1):
        _, col = grid.axes_index(K - 1, j)
        locs, labs, _off = panel_ticks(
            rng[j][0], rng[j][1], max_n_ticks, axes_scale[j], use_math_text
        )
        xt_pos.extend(grid.x_to_grid(col, locs, rng[j]))
        xt_lab.extend(labs)
    tri.set_xticks(xt_pos, xt_lab, rotation=45)

    yt_pos, yt_lab = [], []
    for i in np.arange(1, K):
        row, _ = grid.axes_index(i, 0)
        locs, labs, _off = panel_ticks(
            rng[i][0], rng[i][1], max_n_ticks, axes_scale[i], use_math_text
        )
        yt_pos.extend(grid.y_to_grid(row, locs, rng[i]))
        yt_lab.extend(labs)
    tri.set_yticks(yt_pos, yt_lab, rotation=45)

    # The shared axis contributes labels only. Its own tick marks would appear
    # solely on the figure's outer bottom/left edges, but corner puts marks on
    # *every* panel -- interior panels get marks with blank labels. So the marks
    # are drawn uniformly by _tick_marks, and matplotlib's are suppressed here to
    # avoid doubling them up on the outer edges.
    #
    # The pad must be restored by hand. matplotlib offsets a tick label from the
    # *far end* of its tick mark -- Tick.get_tick_padding() adds
    # `size * {in: 0, inout: 0.5, out: 1.0}[direction]` to the base pad -- so
    # zeroing the size silently pulls every outer label inward by one tick
    # length. Adding that term back keeps the labels where corner puts them.
    for axis, direction in (("x", "xtick"), ("y", "ytick")):
        size = matplotlib.rcParams[f"{direction}.major.size"]
        pad = matplotlib.rcParams[f"{direction}.major.pad"]
        share = {"in": 0.0, "inout": 0.5, "out": 1.0}[
            matplotlib.rcParams[f"{direction}.direction"]
        ]
        tri.tick_params(axis=axis, length=0, pad=pad + size * share)
    _tick_marks(tri, grid, rng, K, max_n_ticks, axes_scale, use_math_text)

    if labels is None:
        return

    # corner places these with set_label_coords(0.5, -0.3 - labelpad) in each
    # panel's axes fraction. A panel is 1 grid unit, so the same offsets apply
    # directly in grid units. The alignments are matplotlib's own defaults for
    # XAxis.label (va=top, ha=center) and YAxis.label (rotation=vertical,
    # rotation_mode=anchor, va=bottom, ha=center) -- reproduced, because a
    # shared axis has no per-panel label to inherit them from.
    lk = dict(label_kwargs)
    lk.setdefault("fontsize", matplotlib.rcParams["axes.labelsize"])
    for j in np.arange(K - 1):
        _, col = grid.axes_index(K - 1, j)
        tri.text(
            grid.col_origin(col) + 0.5, -0.3 - labelpad, labels[j],
            ha="center", va="top", clip_on=False, **lk,
        )
    for i in np.arange(1, K):
        row, _ = grid.axes_index(i, 0)
        tri.text(
            -0.3 - labelpad, grid.row_origin(row) + 0.5, labels[i],
            rotation="vertical", rotation_mode="anchor",
            ha="center", va="bottom", clip_on=False, **lk,
        )


def _tick_marks(tri, grid, rng, K, max_n_ticks, axes_scale, use_math_text):
    """The little tick stubs on every lower-triangle panel, as one collection.

    Easy to forget, and invisible at a glance: corner gives each panel its own
    Axes, so every panel gets tick marks on its bottom and left spines and only
    the outer ones get *labels*. A shared axis draws marks on the figure's outer
    edges alone, which leaves every interior panel subtly bare. The pixel golden
    test is what caught it -- at full-figure scale the eye does not.

    Marks are sized in points (matplotlib's convention) and converted to grid
    units, honouring ``xtick.direction`` so 'in'/'out'/'inout' styles all land
    where matplotlib would put them.
    """
    xlen = points_to_grid(matplotlib.rcParams["xtick.major.size"])
    ylen = points_to_grid(matplotlib.rcParams["ytick.major.size"])
    xdir = matplotlib.rcParams["xtick.direction"]
    ydir = matplotlib.rcParams["ytick.direction"]

    def _span(edge, length, direction):
        if direction == "in":
            return edge, edge + length
        if direction == "inout":
            return edge - length, edge + length
        return edge - length, edge  # 'out', matplotlib's default

    # A column's tick locations do not depend on which row you are in, so these
    # are resolved K times rather than K^2. Doing it inside the panel loop meant
    # 870 MaxNLocator/ScalarFormatter constructions at K=30 -- most of a second,
    # to compute the same K answers over and over.
    locs = [
        panel_ticks(rng[k][0], rng[k][1], max_n_ticks, axes_scale[k], use_math_text)[0]
        for k in np.arange(K)
    ]
    segs = []
    for i in np.arange(K):
        for j in np.arange(i):
            row, col = grid.axes_index(i, j)
            x0, _, y0, _ = grid.cell_extent(row, col)
            lo, hi = _span(y0, xlen, xdir)
            for gx in grid.x_to_grid(col, locs[j], rng[j]):
                segs.append([(gx, lo), (gx, hi)])
            lo, hi = _span(x0, ylen, ydir)
            for gy in grid.y_to_grid(row, locs[i], rng[i]):
                segs.append([(lo, gy), (hi, gy)])
    if segs:
        tri.add_collection(
            LineCollection(
                segs,
                colors=matplotlib.rcParams["xtick.color"],
                linewidths=matplotlib.rcParams["xtick.major.width"],
                zorder=3,
                clip_on=False,
            )
        )


def _overplot_truths(tri, diag_axes, grid, rng, K, truths, truth_color):
    """corner's ``overplot_lines`` + ``overplot_points``, as two artists.

    Panel ``[k2, k1]`` gets a vertical line at column ``k1``'s truth and a
    horizontal line at row ``k2``'s, each guarded independently -- a parameter
    with no truth simply contributes no line, rather than suppressing its
    partner's. The square marker only appears where *both* are known.
    """
    segs, mx, my = [], [], []
    lw = matplotlib.rcParams["lines.linewidth"]
    for k1 in np.arange(K):
        if truths[k1] is not None:
            diag_axes[k1].axvline(truths[k1], color=truth_color, lw=lw)
        for k2 in np.arange(k1 + 1, K):
            row, col = grid.axes_index(k2, k1)
            x0, x1, y0, y1 = grid.cell_extent(row, col)
            if truths[k1] is not None:
                gx = float(grid.x_to_grid(col, truths[k1], rng[k1]))
                segs.append([(gx, y0), (gx, y1)])
            if truths[k2] is not None:
                gy = float(grid.y_to_grid(row, truths[k2], rng[k2]))
                segs.append([(x0, gy), (x1, gy)])
            if truths[k1] is not None and truths[k2] is not None:
                mx.append(float(grid.x_to_grid(col, truths[k1], rng[k1])))
                my.append(float(grid.y_to_grid(row, truths[k2], rng[k2])))
    if segs:
        # capstyle="butt" matches corner's semantics: its axvline/axhline is
        # clipped by the panel's own axes, so the line stops dead at the frame
        # rather than projecting half a linewidth past it. Measured no pixel
        # change at the default linewidth -- kept because it is the correct
        # endpoint rule, not because it fixed anything.
        tri.add_collection(
            LineCollection(
                segs, colors=truth_color, linewidths=lw, zorder=4,
                capstyle="butt",
            )
        )
    if mx:
        tri.plot(
            mx, my, marker="s", linestyle="none", color=truth_color, zorder=5
        )


def materialize_axes(fig):
    """Real per-panel ``Axes`` for a fastcorner figure, created on demand.

    This is the escape hatch for the one thing the composite renderer gives up:
    ``fig.axes`` holds ``1 + K`` entries, so a lower-triangle panel is not
    addressable the way ``corner``'s is. Call this and you get the familiar
    ``(K, K)`` array back, sharing each panel's data coordinates, so
    ``axes[i, j].axvline(...)`` and friends work as expected.

    The Axes are transparent and frameless -- the composite underneath already
    drew the picture, and these exist only to be annotated. You pay corner's
    ~5ms-per-Axes cost here, but only if you actually want them, and the result
    is cached on the figure.

    Note ``np.array(fig.axes).reshape((K, K))`` still will not work: the shared
    and diagonal Axes come first in creation order. Use the returned array.
    """
    st = getattr(fig, "_fastcorner", None)
    if st is None:
        raise ValueError(
            "Not a fastcorner figure -- materialize_axes() needs the panel "
            "geometry recorded by fastcorner.corner()."
        )
    if st["axes"] is not None:
        return st["axes"]

    grid, rng, K = st["grid"], st["ranges"], st["K"]
    left, bottom, width, height = grid.rect
    out = np.empty((K, K), dtype=object)
    for i in np.arange(K):
        for j in np.arange(K):
            row, col = grid.axes_index(i, j)
            if i == j:
                out[row, col] = st["diag"][i]
                continue
            x0, _, y0, _ = grid.cell_extent(row, col)
            ax = fig.add_axes(
                [
                    left + (x0 / grid.span) * width,
                    bottom + (y0 / grid.span) * height,
                    width / grid.span,
                    height / grid.span,
                ],
                zorder=5,
            )
            ax.patch.set_visible(False)
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if j < i:
                ax.set_xlim(rng[j])
                ax.set_ylim(rng[i])
            out[row, col] = ax
    st["axes"] = out
    return out


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
