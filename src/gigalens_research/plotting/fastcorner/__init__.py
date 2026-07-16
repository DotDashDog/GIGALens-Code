"""A drop-in ``corner`` replacement built for high-dimensional posteriors.

``corner`` is not slow because of your samples -- at D=30 the binning is 3% of
its runtime. It is slow because it builds a matplotlib ``Axes`` per panel and
decorates each one: ``plt.subplots(K, K)`` creates K**2 Axes where a corner plot
uses only K*(K+1)/2, and each Axes costs ~5ms whether or not anything is drawn
in it. That floor scales as K**2, which is what makes 30+ dimensions painful.

This backend refuses the floor. The whole lower triangle is composited into a
single image, the contours of every panel are found with ``contourpy`` and drawn
as one ``LineCollection``, and the decorations are placed directly on the
figure. The artist count stops depending on K.

The numbers it plots are bit-identical to corner's -- same bins, same density
levels, same quantiles -- because a corner plot is read as evidence and
"visually indistinguishable" is not a checkable claim. See
:mod:`._data` for why, and ``tests/test_fastcorner_data.py`` for the proof.

What this costs you: ``fig.axes`` holds one entry, not K**2. Nothing that only
looks at the figure notices, but code that reaches for a specific panel does.
Call :func:`materialize_axes` to get the real per-panel Axes back when you need
them -- you pay corner's Axes cost only if you actually use them.
"""

from __future__ import annotations

from ._data import DEFAULT_LEVELS, density_levels, quantile

__all__ = ["DEFAULT_LEVELS", "density_levels", "quantile"]
