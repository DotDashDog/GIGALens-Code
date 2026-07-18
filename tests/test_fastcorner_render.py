"""The rendered figure must match corner's pixels, not just its numbers.

The numeric layer is pinned bit-exactly in ``test_fastcorner_data.py``. That is
the stronger guarantee but it cannot see *placement*: a panel drawn in the wrong
cell, a missing tick mark, a label shifted by a pad -- all of those keep the
numbers perfect and the picture wrong. So the two suites are complementary, and
neither is redundant.

These compare against the real installed ``corner`` by rendering both to an Agg
buffer. The tolerance is deliberately two-sided: essentially every pixel must be
within a hair (catching shifts and omissions), while a tiny handful may differ
outright (subpixel antialiasing, where our shared-axis transform and corner's
per-panel transform round a boundary differently). Tightening the second bound to
zero would be pinning matplotlib's rasterizer, not our correctness.

Every threshold below was measured, not guessed. If one of these fails, look at
the diff image before relaxing it -- the per-panel tick marks and the tick-label
pad were both found exactly this way, and both were real bugs.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ref_corner = pytest.importorskip("corner")

from gigalens_research.plotting import fastcorner  # noqa: E402

D, N = 5, 8000
LABELS = [f"$x_{i}$" for i in range(D)]
TRUTHS = [0.4, -0.3, 0.0, 0.5, None]


@pytest.fixture(scope="module")
def data():
    """A correlated posterior -- contours that actually bend, not circles."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(D, D)) / np.sqrt(D)
    cov = A @ A.T + np.eye(D) * 0.1
    return rng.multivariate_normal(np.zeros(D), cov, size=N)


def _render(fn, x, **kw):
    fig = fn(x, **kw)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.int16)
    plt.close(fig)
    return buf


def _compare(data, **kw):
    a = _render(ref_corner.corner, data, **kw)
    b = _render(fastcorner.corner, data, **kw)
    assert a.shape == b.shape, (
        f"figure size differs: corner {a.shape} vs fastcorner {b.shape}. The "
        f"layout constants (factor/lbdim/trdim/whspace) must match exactly."
    )
    d = np.abs(a - b).max(axis=2)
    return (d <= 8).mean(), (d > 64).mean()


# --- the option matrix -----------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({}, id="defaults"),
        pytest.param(dict(plot_datapoints=False), id="no-datapoints"),
        pytest.param(dict(plot_contours=False), id="no-contours"),
        pytest.param(dict(plot_density=False), id="no-density"),
        pytest.param(dict(fill_contours=True), id="fill-contours"),
        pytest.param(dict(no_fill_contours=True), id="no-fill-contours"),
        pytest.param(dict(smooth=1.0), id="smooth"),
        pytest.param(dict(smooth1d=1.0), id="smooth1d"),
        pytest.param(dict(bins=35), id="bins-35"),
        pytest.param(dict(hist_bin_factor=2), id="hist-bin-factor"),
        pytest.param(dict(labels=LABELS), id="labels"),
        pytest.param(dict(labels=LABELS, show_titles=True), id="titles"),
        pytest.param(dict(quantiles=[0.16, 0.5, 0.84]), id="quantiles"),
        pytest.param(dict(truths=TRUTHS), id="truths-with-a-None"),
        pytest.param(dict(color="C1"), id="color"),
        pytest.param(dict(scale_hist=True), id="scale-hist"),
        pytest.param(dict(max_n_ticks=3), id="max-n-ticks"),
        pytest.param(dict(max_n_ticks=0), id="no-ticks"),
        pytest.param(dict(range=[0.9] * D), id="range-as-fraction"),
        pytest.param(dict(labelpad=0.1, labels=LABELS), id="labelpad"),
        pytest.param(
            dict(labels=LABELS, show_titles=True, truths=TRUTHS,
                 quantiles=[0.16, 0.5, 0.84], title_fmt=".3f"),
            id="everything-at-once",
        ),
    ],
)
def test_render_matches_corner(data, kw):
    near, gross = _compare(data, **kw)
    # `gross` is the real guard: anything mislaid, missing or restyled shows up
    # as pixels that differ outright. Measured worst case across this matrix is
    # 0.013%; the bound is a few times that, and it is what caught the absent
    # per-panel tick marks, the tick-label pad, and contours bleeding into the
    # gutters.
    assert gross < 0.0005, (
        f"{gross:.4%} of pixels differ by more than 64/255. Render the diff "
        f"before relaxing this -- every previous failure here was a real bug."
    )
    # `near` is deliberately looser. Most cases land above 99.9%, but the
    # datapoint haze is thousands of alpha=0.1 markers, and once a clip path is
    # active (a narrowed `range=`) they composite a shade differently than
    # corner's per-panel artists do. That moves ~1.7% of pixels by a few grey
    # levels while `gross` stays at 0.01%, so it is a compositing artefact, not
    # a drawing error. Bounding it at all still catches wholesale shifts.
    assert near > 0.98, (
        f"only {near:.4%} of pixels within 8/255 of corner's. That is a "
        f"placement or styling difference, not antialiasing."
    )


def test_weighted_render_matches_corner(data):
    rng = np.random.default_rng(11)
    w = rng.exponential(size=N)
    near, gross = _compare(data, weights=w)
    assert gross < 0.0005 and near > 0.98


# --- structure -------------------------------------------------------------


def test_axes_count_is_linear_in_K(data):
    """The whole point: one shared Axes plus the diagonal, not K**2."""
    fig = fastcorner.corner(data)
    assert len(fig.axes) == 1 + D
    plt.close(fig)


def test_figure_size_matches_corner(data):
    a, b = ref_corner.corner(data), fastcorner.corner(data)
    assert a.get_size_inches().tolist() == b.get_size_inches().tolist()
    plt.close(a)
    plt.close(b)


def test_materialize_axes_gives_addressable_panels(data):
    fig = fastcorner.corner(data)
    axes = fastcorner.materialize_axes(fig)
    assert axes.shape == (D, D)
    # a lower-triangle panel carries its own data coordinates, so annotating it
    # lands where the caller means
    lo, hi = axes[3, 1].get_xlim()
    np.testing.assert_allclose([lo, hi], [data[:, 1].min(), data[:, 1].max()])
    lo, hi = axes[3, 1].get_ylim()
    np.testing.assert_allclose([lo, hi], [data[:, 3].min(), data[:, 3].max()])
    plt.close(fig)


def test_materialize_axes_is_cached(data):
    fig = fastcorner.corner(data)
    assert fastcorner.materialize_axes(fig) is fastcorner.materialize_axes(fig)
    plt.close(fig)


def test_materialize_axes_rejects_a_foreign_figure():
    fig = plt.figure()
    with pytest.raises(ValueError, match="Not a fastcorner figure"):
        fastcorner.materialize_axes(fig)
    plt.close(fig)


# --- overlay ---------------------------------------------------------------


def test_overlay_reuses_axes_and_does_not_clear(data):
    fig = fastcorner.corner(data, color="k")
    n_before = len(fig.axes)
    rng = np.random.default_rng(5)
    fastcorner.corner(data + rng.normal(0, 0.05, data.shape), fig=fig, color="r")
    assert len(fig.axes) == n_before, "overlay must not build a second grid"
    plt.close(fig)


def test_overlay_warns_when_data_exceeds_pinned_range(data):
    """We pin ranges on overlay where corner would widen them.

    corner can widen because its artists live in per-panel data coordinates and
    move with the limits. Ours are baked into shared grid coordinates, so
    widening would silently misplace the *first* dataset. Warning is the honest
    alternative -- and the message must say how to fix it.
    """
    fig = fastcorner.corner(data)
    with pytest.warns(UserWarning, match="pins ranges"):
        fastcorner.corner(data * 3.0, fig=fig)
    plt.close(fig)


# --- loud failures rather than quiet wrong pictures -------------------------


@pytest.mark.parametrize("kw", [dict(reverse=True), dict(top_ticks=True)])
def test_unimplemented_options_raise(data, kw):
    with pytest.raises(NotImplementedError):
        fastcorner.corner(data, **kw)


def test_log_axes_raise(data):
    with pytest.raises(NotImplementedError, match="linear axes only"):
        fastcorner.corner(np.abs(data) + 0.1, axes_scale="log")


def test_zero_dynamic_range_raises_like_corner(data):
    x = data.copy()
    x[:, 2] = 1.0
    with pytest.raises(ValueError, match="no dynamic range"):
        fastcorner.corner(x)
