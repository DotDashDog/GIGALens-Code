"""Panel geometry, and ticks computed without an Axes to hang them on.

The composite renderer draws every panel into a *single* Axes, so it needs to do
by hand what matplotlib's subplot grid would have done: decide where each panel
sits, and map a panel's data coordinates into the shared space.

The shared space ("grid units") is chosen so one panel is exactly 1 unit wide.
Panel column ``c`` spans ``[c*(1+whspace), c*(1+whspace) + 1]``, so the whole
grid spans ``[0, K + (K-1)*whspace]``. That is not an arbitrary convention --
it is algebraically what ``subplots_adjust(wspace=whspace)`` produces, which is
what lets the output land on the same pixels as corner's.

All the magic numbers here (``factor = 2.0``, ``0.5``/``0.2`` margins,
``whspace = 0.05``) are corner's, reproduced so figures come out the same size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
    ScalarFormatter,
)

__all__ = ["Grid", "panel_ticks"]

_FACTOR = 2.0
_WHSPACE = 0.05


def points_to_grid(points: float) -> float:
    """Convert a length in typographic points to grid units.

    One grid unit is one panel, and a panel is ``_FACTOR`` inches on a side by
    construction: the shared axes maps ``span = K + (K-1)*whspace`` grid units
    onto ``plotdim = _FACTOR*K + _FACTOR*(K-1)*whspace`` inches, so a grid unit
    is exactly ``_FACTOR`` inches at any K.

    Needed because tick marks are sized in points but drawn, here, in grid
    coordinates. This holds at the figure's constructed size; resizing the
    figure afterwards would scale these marks where matplotlib's own would stay
    fixed. corner's layout assumes its figsize too, so a resized corner plot is
    already off-spec.
    """
    return points / 72.0 / _FACTOR


@dataclass(frozen=True)
class Grid:
    """Where every panel lives, in inches and in shared data coordinates.

    ``K`` is the number of parameters. ``reverse`` mirrors corner's option to
    build the plot from the upper-right instead of the lower-left, which only
    swaps the margins and flips the row/column indexing.
    """

    K: int
    reverse: bool = False
    whspace: float = _WHSPACE

    # --- figure sizing (corner's, verbatim) --------------------------------

    @property
    def _lbdim(self) -> float:
        return (0.2 if self.reverse else 0.5) * _FACTOR

    @property
    def _trdim(self) -> float:
        return (0.5 if self.reverse else 0.2) * _FACTOR

    @property
    def _plotdim(self) -> float:
        return _FACTOR * self.K + _FACTOR * (self.K - 1.0) * self.whspace

    @property
    def dim(self) -> float:
        """Figure side length in inches. corner's figures are always square."""
        return self._lbdim + self._plotdim + self._trdim

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        """The panel region as ``(left, bottom, width, height)`` figure fractions.

        Equivalent to the region ``subplots_adjust(left=lb, bottom=lb, right=tr,
        top=tr)`` would carve out. Because the figure is square and corner uses
        the same margin on left and bottom, one fraction serves both axes.
        """
        lb = self._lbdim / self.dim
        tr = (self._lbdim + self._plotdim) / self.dim
        return (lb, lb, tr - lb, tr - lb)

    # --- shared data coordinates -------------------------------------------

    @property
    def span(self) -> float:
        """Width of the whole grid in grid units (one panel == 1 unit)."""
        return self.K + (self.K - 1) * self.whspace

    def col_origin(self, col: int) -> float:
        return col * (1.0 + self.whspace)

    def row_origin(self, row: int) -> float:
        """Bottom edge of ``row`` in grid units. Row 0 is the *top* row, matching
        matplotlib's ``axes[row, col]`` indexing."""
        return (self.K - 1 - row) * (1.0 + self.whspace)

    def x_to_grid(
        self, col: int, v: np.ndarray, rng: Sequence[float]
    ) -> np.ndarray:
        """Map a panel's x data onto the shared axis. Values outside ``rng``
        map outside the cell, so callers must clip or mask before drawing."""
        lo, hi = rng
        return self.col_origin(col) + (np.asarray(v) - lo) / (hi - lo)

    def y_to_grid(
        self, row: int, v: np.ndarray, rng: Sequence[float]
    ) -> np.ndarray:
        lo, hi = rng
        return self.row_origin(row) + (np.asarray(v) - lo) / (hi - lo)

    def cell_extent(
        self, row: int, col: int
    ) -> Tuple[float, float, float, float]:
        """``(x0, x1, y0, y1)`` of a panel in grid units -- an ``imshow`` extent."""
        x0, y0 = self.col_origin(col), self.row_origin(row)
        return (x0, x0 + 1.0, y0, y0 + 1.0)

    def axes_index(self, i: int, j: int) -> Tuple[int, int]:
        """corner's logical ``(i, j)`` -> physical ``(row, col)``.

        Under ``reverse`` the plot is built from the opposite corner, which
        corner implements purely as this index flip -- the ``j > i`` upper-
        triangle test is unchanged.
        """
        if self.reverse:
            return (self.K - i - 1, self.K - j - 1)
        return (i, j)


class _StubAxis:
    """The only thing ``ScalarFormatter`` wants from an axis is its interval.

    Giving it this instead of a real ``Axis`` is what lets us format ticks
    without creating the ~5ms of tick artists we are trying to avoid. Verified
    against a real axes -- positions, labels and the offset text all agree,
    including the math-text cases (``tests/test_fastcorner_layout.py``).
    """

    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = lo, hi

    def get_view_interval(self):
        return self.lo, self.hi

    def get_data_interval(self):
        return self.lo, self.hi

    def get_minpos(self):
        return self.lo

    def get_tick_space(self):
        return 9


def panel_ticks(
    lo: float,
    hi: float,
    max_n_ticks: int = 5,
    scale: str = "linear",
    use_math_text: bool = False,
) -> Tuple[np.ndarray, List[str], str]:
    """Tick positions, labels and offset text for one panel's axis.

    Reproduces corner's setup exactly: ``MaxNLocator(max_n_ticks,
    prune="lower")`` on linear axes, ``LogLocator(numticks=max_n_ticks)`` on log
    ones. ``prune="lower"`` is why the lowest tick is missing from corner plots
    -- it keeps adjacent panels' corner labels from colliding.

    Ticks outside ``[lo, hi]`` are dropped: a real Axes clips them at draw time,
    but we are placing them on a shared axis where an out-of-range tick would
    land in the neighbouring panel.

    ``max_n_ticks == 0`` means no ticks at all (corner's ``NullLocator`` branch).
    """
    if max_n_ticks == 0:
        return np.array([]), [], ""

    if scale == "linear":
        locs = MaxNLocator(max_n_ticks, prune="lower").tick_values(lo, hi)
        fmt = ScalarFormatter(useMathText=use_math_text)
    elif scale == "log":
        locs = LogLocator(numticks=max_n_ticks).tick_values(lo, hi)
        fmt = LogFormatterMathtext()
    else:
        raise ValueError(f"Scale {scale!r} not supported. Use 'linear' or 'log'.")

    locs = np.asarray(locs)
    locs = locs[(locs >= lo) & (locs <= hi)]
    fmt.axis = _StubAxis(lo, hi)
    fmt.set_locs(locs)
    return locs, [fmt(v, i) for i, v in enumerate(locs)], fmt.get_offset()
