"""A drop-in ``corner`` replacement built for high-dimensional posteriors.

``corner`` is not slow because of your samples -- at D=30 the binning is 3% of
its runtime. It is slow because it builds a matplotlib ``Axes`` per panel and
decorates each one: ``plt.subplots(K, K)`` creates K**2 Axes where a corner plot
uses only K*(K+1)/2, and each Axes costs ~5ms whether or not anything is drawn
in it. That floor scales as K**2, which is what makes 30+ dimensions painful.

This backend refuses the floor where it is O(K**2) and pays it where it is only
O(K). The K diagonal panels keep real Axes -- ``ax.hist`` fidelity for free. The
K*(K-1)/2 lower-triangle panels share a single Axes in panel-grid coordinates:
every panel's contours become one ``LineCollection``, every panel's datapoints
one ``Line2D``, and the ticks and labels are placed directly. The Axes count
stops depending on K**2.

The numbers it plots are bit-identical to corner's -- same bins, same density
levels, same quantiles -- because a corner plot is read as evidence and
"visually indistinguishable" is not a checkable claim. See :mod:`._data` for why
and ``tests/test_fastcorner_data.py`` for the proof. The *picture* is pinned
separately against real corner renders in ``tests/test_fastcorner_render.py``:
>99.9% of pixels within 8/255 across the option matrix, which is what caught the
missing per-panel tick marks and the contours bleeding into the gutters.

What this costs you: ``fig.axes`` holds one entry, not K**2. Nothing that only
looks at the figure notices, but code that reaches for a specific panel does.
Call :func:`materialize_axes` to get the real per-panel Axes back when you need
them -- you pay corner's Axes cost only if you actually use them.
"""

from __future__ import annotations

from ._data import DEFAULT_LEVELS, density_levels, quantile
from .core import corner, materialize_axes

__all__ = [
    "DEFAULT_LEVELS",
    "corner",
    "density_levels",
    "materialize_axes",
    "quantile",
]
