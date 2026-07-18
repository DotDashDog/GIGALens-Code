"""Why the density layer is ``pcolormesh(snap=False)`` and not something else.

corner draws its density with ``ax.pcolor``. We draw it with ``pcolormesh`` and
``snap=False``, which is ~3x cheaper and -- these tests assert -- pixel-for-pixel
identical.

``snap=False`` looks like a cosmetic kwarg and is not. Without it the two agree
exactly whenever a cell happens to span a whole number of pixels and disagree
otherwise, because matplotlib snaps mesh edges to the pixel grid under a
heuristic that ``pcolor`` and ``pcolormesh`` apply differently. That makes the
bug *bin-count dependent*: at ``bins=20`` on a default figure the two are 100%
identical and every test passes, while ``bins=35`` silently grows a grid of 1px
seams at every bin boundary. Exactly the sort of thing that looks fine until
someone changes an unrelated default.

So these tests sweep bin counts on purpose, including the fractional-pixel cases
that expose it. Delete ``snap=False`` and ``test_snap_false_is_what_makes_them_agree``
fails; that is the point of it.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 300px panels. px/cell is 300/bins: whole for 10/15/20/25/30/50, fractional
# for 24 (12.5), 35 (8.57) and 40 (7.5) -- the cases where snapping shows up.
WHOLE_PIXEL_BINS = [10, 15, 20, 25, 30, 50]
FRACTIONAL_PIXEL_BINS = [24, 35, 40]


def _draw(kind, nbins, **kw):
    H = np.random.default_rng(0).poisson(30, size=(nbins, nbins)).astype(float)
    e = np.linspace(0, 1, nbins + 1)
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    if kind == "imshow":
        ax.imshow(H, extent=(0, 1, 0, 1), origin="lower", aspect="auto",
                  interpolation="nearest", cmap="Greys", **kw)
    else:
        getattr(ax, kind)(e, e, H, cmap="Greys", **kw)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.int16)
    plt.close(fig)
    return buf


def _identical(a, b):
    return (np.abs(a - b).max(axis=2) == 0).mean()


@pytest.mark.parametrize("nbins", WHOLE_PIXEL_BINS + FRACTIONAL_PIXEL_BINS)
def test_pcolormesh_snap_false_is_identical_to_pcolor(nbins):
    """The substitution the renderer relies on, at every bin count."""
    got = _identical(_draw("pcolor", nbins), _draw("pcolormesh", nbins, snap=False))
    assert got == 1.0, (
        f"pcolormesh(snap=False) differs from pcolor at bins={nbins} "
        f"({got:.3%} identical). The density layer assumes they are "
        f"interchangeable; if this breaks, corner's pcolor is the fallback."
    )


@pytest.mark.parametrize("nbins", FRACTIONAL_PIXEL_BINS)
def test_snap_false_is_what_makes_them_agree(nbins):
    """Guards the kwarg itself: without it these bin counts disagree.

    If this ever starts passing, matplotlib has changed its snapping and the
    comment explaining ``snap=False`` is stale -- but the kwarg is still
    harmless, so fix the docs, not the code.
    """
    snapped = _identical(_draw("pcolor", nbins), _draw("pcolormesh", nbins))
    assert snapped < 1.0, (
        f"pcolormesh now matches pcolor at bins={nbins} even with snapping on; "
        f"the rationale recorded for snap=False no longer holds."
    )


@pytest.mark.parametrize("nbins", [20, 35])
def test_imshow_would_not_do(nbins):
    """Records the alternative that was rejected, and why.

    imshow *resamples* the histogram onto the display grid instead of
    rasterizing cells, so bin edges land wherever the resampler puts them. It is
    the cheapest option and the wrong one -- worth pinning so the tempting
    'just use imshow, it's an image' change gets a failing test rather than a
    reviewer's opinion.
    """
    got = _identical(_draw("pcolor", nbins), _draw("imshow", nbins))
    assert got < 0.9, (
        f"imshow now matches pcolor at bins={nbins} ({got:.3%}); if that is "
        f"really true it is cheaper still and worth revisiting."
    )
