"""The numeric layer: ranges, binning, density levels, quantiles.

Everything here is defined to be *bit-identical* to ``corner`` 2.2.3, which is
what makes the fast renderer safe to trust. The renderer draws differently; it
must never draw something different.

Bit-identity is a deliberate choice, not perfectionism. A corner plot is read as
evidence — the 1σ contour is quoted in papers — so a fast backend that is
"visually indistinguishable" is not good enough. If a histogram bin differs by
one count near a contour level, the contour moves, and nothing in the output
says so. Exact agreement is the only claim that can be checked mechanically, so
it is the claim we make.

The binning below is numpy's own ``histogramdd`` algorithm with the per-pair
work hoisted out of the loop. corner calls ``np.histogram2d`` once per panel,
which re-searches the bin edges for every pair: O(D^2 * N log nbins). Digitizing
each column once and then combining pairs of codes is O(D * N log nbins) for the
search plus an O(N) ``bincount`` per pair. Because it is numpy's algorithm
rather than a faster approximation of it, the counts agree exactly rather than
almost — see ``tests/test_fastcorner_data.py``.

The tempting shortcut, ``((x - lo) * nbins / (hi - lo)).astype(int)``, is
*wrong*: ``np.linspace`` edges are not exactly ``lo + k*(hi-lo)/nbins``, so a
sample sitting near a bin boundary can land on either side of it. It agrees with
numpy on most data, which is what makes it dangerous.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_LEVELS",
    "bin_edges",
    "density_levels",
    "digitize_columns",
    "hist1d_from_codes",
    "hist2d_from_codes",
    "parse_ranges",
    "quantile",
]

#: corner's "sigma" contour levels: the 0.5, 1, 1.5 and 2 sigma enclosed-mass
#: fractions of a 2-D Gaussian. Verbatim from ``corner.core.hist2d``.
DEFAULT_LEVELS = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)


def parse_ranges(
    xs: np.ndarray,
    range_: Optional[Sequence] = None,
    weights: Optional[np.ndarray] = None,
) -> List[List[float]]:
    """Resolve corner's ``range`` argument into explicit ``[lo, hi]`` per column.

    ``xs`` is ``(K, N)`` — corner's internal orientation, one row per parameter.

    An entry may be a ``(lo, hi)`` pair or a bare float in ``(0, 1)`` naming an
    equal-tailed fraction of the samples to enclose. ``None`` means "the full
    span of each column", which is corner's default.

    Raises on a column with no dynamic range, matching corner: a zero-width
    range makes every bin edge identical and the histogram meaningless, and the
    failure is far easier to read here than as an empty panel.
    """
    if range_ is None:
        out = [[x.min(), x.max()] for x in xs]
        m = np.array([e[0] == e[1] for e in out], dtype=bool)
        if np.any(m):
            raise ValueError(
                "It looks like the parameter(s) in column(s) {0} have no "
                "dynamic range. Please provide a `range` argument.".format(
                    ", ".join(map("{0}".format, np.arange(len(m))[m]))
                )
            )
        return out

    out = list(range_)
    for i, _ in enumerate(out):
        try:
            emin, emax = out[i]
        except TypeError:
            q = [0.5 - 0.5 * out[i], 0.5 + 0.5 * out[i]]
            out[i] = quantile(xs[i], q, weights=weights)
    if len(out) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")
    return out


def bin_edges(rng: Sequence[float], nbins: int, scale: str = "linear") -> np.ndarray:
    """Bin edges for one column, matching corner's ``linspace``/``logspace``.

    The exact edge *values* matter: they are what :func:`digitize_columns`
    searches, so producing them any other way would silently move counts between
    bins near a boundary.
    """
    lo, hi = min(rng), max(rng)
    if scale == "linear":
        return np.linspace(lo, hi, nbins + 1)
    if scale == "log":
        return np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    raise ValueError(f"Scale {scale!r} not supported. Use 'linear' or 'log'.")


def digitize_columns(
    xs: np.ndarray, edges: Sequence[np.ndarray]
) -> List[np.ndarray]:
    """Bin-code every column once. This is the hoist that makes the rest cheap.

    Returns one integer array per column, coded exactly as ``np.histogramdd``
    codes them: ``0`` is underflow, ``1..nbins`` are the real bins, ``nbins+1``
    is overflow. The off-by-one padding is not an accident of numpy's — it is
    what lets :func:`hist2d_from_codes` drop out-of-range samples with a slice
    instead of a per-pair boolean mask.

    The ``on_edge`` adjustment is numpy's: ``searchsorted(..., "right")`` would
    put a sample sitting exactly on the top edge into overflow, but a histogram's
    last bin is closed on the right, so it belongs in the last bin.
    """
    codes = []
    for x, e in zip(xs, edges):
        idx = np.searchsorted(e, x, side="right")
        idx[x == e[-1]] -= 1
        codes.append(idx)
    return codes


def hist2d_from_codes(
    ci: np.ndarray,
    cj: np.ndarray,
    nbi: int,
    nbj: int,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One 2-D histogram from two digitized columns, as ``H[xbin, ybin]``.

    Equivalent to ``np.histogram2d(x, y, bins=[ei, ej], weights=w)[0]`` for
    codes produced by :func:`digitize_columns`, but without re-searching the
    edges. The padded ``(nbi+2, nbj+2)`` grid absorbs underflow and overflow into
    the border, which the final slice discards.

    Bit-identity extends to weighted sums: this accumulates with ``bincount`` in
    input order, exactly as numpy does, so even the floating-point rounding
    matches.

    Returns float64 even when unweighted. That is not a choice -- ``histogramdd``
    ends with ``hist.astype(float, casting='safe')`` under a comment calling it
    "the (bad) behavior observed in gh-7845", and ``histogram2d`` inherits it.
    Downstream arithmetic (``density_levels`` divides by a cumulative sum in
    place) silently depends on it, so the quirk is part of the contract.
    Note the 1-D path does *not* share it -- see :func:`hist1d_from_codes`.
    """
    flat = ci * (nbj + 2) + cj
    hist = np.bincount(flat, weights=weights, minlength=(nbi + 2) * (nbj + 2))
    hist = hist.reshape(nbi + 2, nbj + 2)[1:-1, 1:-1]
    return hist.astype(float, casting="safe")


def hist1d_from_codes(
    c: np.ndarray, nb: int, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """One 1-D histogram from a digitized column (the diagonal panels).

    Note this cannot always reuse the 2-D codes: corner's ``hist_bin_factor``
    lets the diagonal use a different bin count than the off-diagonal panels, in
    which case the caller must digitize again against the 1-D edges.

    Deliberately *not* cast to float, unlike :func:`hist2d_from_codes`: 1-D
    ``np.histogram`` returns int64 when unweighted, and only ``histogramdd``
    carries the gh-7845 float cast. The asymmetry is numpy's, and matching it is
    the whole point.
    """
    hist = np.bincount(c, weights=weights, minlength=nb + 2)
    return hist[1:-1]


def density_levels(H: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Contour heights enclosing the given fractions of total density.

    Verbatim from ``corner.core.hist2d`` — including the degenerate-level
    nudging, which fires when too few samples make two requested levels land on
    the same histogram height. corner emits a warning there and so should the
    caller; the nudge only keeps ``contour`` from erroring on a non-monotonic
    level list.

    Returns the levels sorted ascending. ``H`` is the (optionally smoothed)
    histogram.
    """
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()
    return V


def levels_are_degenerate(V: np.ndarray) -> bool:
    """Whether :func:`density_levels` had to nudge — i.e. too few points.

    Kept separate so the caller can reproduce corner's "Too few points to create
    valid contours" warning without ``density_levels`` needing to know about
    logging.
    """
    return bool(np.any(np.diff(V) == 0))


def quantile(
    x: np.ndarray,
    q: Sequence[float],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample quantiles, weighted or not. Verbatim from ``corner.core.quantile``.

    Unweighted this is just ``np.percentile``. The weighted branch uses the
    cumulative-weight convention corner uses, which is *not* the same as
    interpolating the weighted empirical CDF — reproducing it exactly is the
    point, since these values are printed in panel titles.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))

    weights = np.atleast_1d(weights)
    if len(x) != len(weights):
        raise ValueError("Dimension mismatch: len(weights) != len(x)")
    idx = np.argsort(x)
    sw = weights[idx]
    cdf = np.cumsum(sw)[:-1]
    cdf /= cdf[-1]
    cdf = np.append(0, cdf)
    return np.interp(q, cdf, x[idx]).tolist()
