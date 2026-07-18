"""The fast numeric layer must agree with corner *exactly*, not approximately.

These tests are the load-bearing ones. The renderer is verified by pixels, which
can only catch differences big enough to see; a one-count binning difference
moves a contour by a sub-pixel amount and would pass a pixel test while
silently changing a published 1-sigma bound. So the numbers are pinned against
the real library here, and the renderer is free to be fast.

Everything is checked against actually-installed ``corner`` / ``numpy``, not
against a transcription of their algorithm — a transcription would agree with
itself by construction and prove nothing.
"""

import numpy as np
import pytest

from gigalens_research.plotting.fastcorner._data import (
    DEFAULT_LEVELS,
    bin_edges,
    density_levels,
    digitize_columns,
    hist1d_from_codes,
    hist2d_from_codes,
    parse_ranges,
    quantile,
)


def assert_identical(mine, ref):
    """Same values AND same dtype.

    ``assert_array_equal`` alone compares values only. That is too weak here:
    it happily passed an int64 histogram against numpy's float64 one, and the
    mismatch surfaced much later as an in-place-divide crash in
    ``density_levels``. Downstream code inherits the dtype, so the dtype is part
    of the contract being tested.
    """
    assert mine.dtype == ref.dtype, f"dtype {mine.dtype} != numpy's {ref.dtype}"
    np.testing.assert_array_equal(mine, ref)


def _mine(x, y, ei, ej, w=None):
    ci, cj = digitize_columns(np.array([x, y]), [ei, ej])
    return hist2d_from_codes(ci, cj, len(ei) - 1, len(ej) - 1, w)


def _numpy(x, y, ei, ej, w=None):
    return np.histogram2d(x, y, bins=[ei, ej], weights=w)[0]


# --- binning: bit-identity against numpy -----------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_bins_bit_identical_random(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=2000), rng.normal(size=2000)
    ei = bin_edges([x.min(), x.max()], 20)
    ej = bin_edges([y.min(), y.max()], 20)
    assert_identical(_mine(x, y, ei, ej), _numpy(x, y, ei, ej))


def test_bins_bit_identical_on_exact_edges():
    """The case the naive ``(x-lo)*scale`` digitize gets wrong.

    Samples sitting exactly on bin edges are where float error decides which
    bin they land in, and where a histogram's closed-right last bin is a special
    case. Real posteriors hit this whenever a parameter is bounded and the
    sampler piles up against the bound.
    """
    ei = bin_edges([0.0, 1.0], 20)
    ej = bin_edges([0.0, 1.0], 20)
    x = np.tile(ei, 5)               # every sample exactly on an edge
    y = np.repeat(ej[:5], len(x) // 5)
    assert_identical(_mine(x, y, ei, ej), _numpy(x, y, ei, ej))


def test_bins_bit_identical_with_samples_out_of_range():
    """A narrower `range=` than the data must drop the outliers, not clip them.

    Clipping would silently pile every outlier into the edge bins and inflate
    the outermost contour.
    """
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=5000), rng.normal(size=5000)
    ei, ej = bin_edges([-1.0, 1.0], 20), bin_edges([-1.0, 1.0], 20)
    H = _mine(x, y, ei, ej)
    assert_identical(H, _numpy(x, y, ei, ej))
    assert H.sum() < len(x)          # outliers really were dropped


def test_bins_bit_identical_weighted():
    """Weighted sums must match to the last bit, so the accumulation order must
    match too -- both accumulate via ``bincount`` in input order."""
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=3000), rng.normal(size=3000)
    w = rng.exponential(size=3000)
    ei = bin_edges([x.min(), x.max()], 15)
    ej = bin_edges([y.min(), y.max()], 25)
    assert_identical(_mine(x, y, ei, ej, w), _numpy(x, y, ei, ej, w))


def test_bins_bit_identical_asymmetric_bin_counts():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=2000), rng.normal(size=2000)
    ei = bin_edges([x.min(), x.max()], 7)
    ej = bin_edges([y.min(), y.max()], 31)
    H = _mine(x, y, ei, ej)
    assert H.shape == (7, 31)
    assert_identical(H, _numpy(x, y, ei, ej))


def test_bins_bit_identical_log_scale():
    rng = np.random.default_rng(3)
    x, y = rng.lognormal(size=2000), rng.lognormal(size=2000)
    ei = bin_edges([x.min(), x.max()], 20, scale="log")
    ej = bin_edges([y.min(), y.max()], 20, scale="log")
    assert_identical(_mine(x, y, ei, ej), _numpy(x, y, ei, ej))


def test_bins_bit_identical_integer_valued_data():
    """Discrete data puts many samples on the same edge at once."""
    rng = np.random.default_rng(4)
    x = rng.integers(0, 10, size=3000).astype(float)
    y = rng.integers(0, 10, size=3000).astype(float)
    ei, ej = bin_edges([0.0, 10.0], 10), bin_edges([0.0, 10.0], 10)
    assert_identical(_mine(x, y, ei, ej), _numpy(x, y, ei, ej))


def test_hist1d_bit_identical():
    rng = np.random.default_rng(5)
    x = rng.normal(size=4000)
    e = bin_edges([x.min(), x.max()], 20)
    (c,) = digitize_columns(np.array([x]), [e])
    assert_identical(hist1d_from_codes(c, 20), np.histogram(x, bins=e)[0])


def test_digitize_hoist_matches_per_pair_histogram2d():
    """The actual optimization: digitize once, reuse across every pair.

    Guards the failure mode that a hoisted code array is subtly per-pair
    dependent -- e.g. reusing x's codes against y's edges.
    """
    rng = np.random.default_rng(6)
    xs = rng.normal(size=(5, 1000))
    edges = [bin_edges([x.min(), x.max()], 20) for x in xs]
    codes = digitize_columns(xs, edges)
    for i in range(5):
        for j in range(i):
            assert_identical(
                hist2d_from_codes(codes[i], codes[j], 20, 20),
                _numpy(xs[i], xs[j], edges[i], edges[j]),
            )


# --- levels / quantiles: identity against corner itself --------------------


def test_levels_match_corner_rendered_contours():
    """Check against the levels corner actually draws, not a transcription.

    corner computes its density levels inside ``hist2d`` and never returns
    them, so we read them back off the ContourSet it leaves on the axes. That
    makes this a real external check rather than a restatement of our own code.
    """
    corner = pytest.importorskip("corner")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.contour import ContourSet

    rng = np.random.default_rng(7)
    x, y = rng.normal(size=20000), rng.normal(size=20000)

    fig, ax = plt.subplots()
    corner.hist2d(x, y, ax=ax, bins=20, plot_datapoints=False, new_fig=False)
    drawn = [c for c in ax.findobj(ContourSet)]
    assert drawn, "corner drew no contours; test cannot check anything"
    corner_levels = np.sort(np.concatenate([np.asarray(c.levels) for c in drawn]))
    plt.close(fig)

    ei = bin_edges([x.min(), x.max()], 20)
    ej = bin_edges([y.min(), y.max()], 20)
    H = _mine(x, y, ei, ej)
    V = density_levels(H, DEFAULT_LEVELS)

    # corner draws V on the contour lines plus [V.min(), H.max()] on the base
    # fill, so ours must be a subset of what it rendered.
    for v in V:
        assert np.any(np.isclose(corner_levels, v, rtol=0, atol=0)), (
            f"level {v} not among corner's rendered levels {corner_levels}"
        )


@pytest.mark.parametrize("seed", range(4))
def test_quantile_matches_corner_unweighted(seed):
    corner_core = pytest.importorskip("corner.core")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=1000)
    q = [0.16, 0.5, 0.84]
    np.testing.assert_array_equal(
        np.asarray(quantile(x, q)), np.asarray(corner_core.quantile(x, q))
    )


def test_quantile_matches_corner_weighted():
    corner_core = pytest.importorskip("corner.core")
    rng = np.random.default_rng(8)
    x, w = rng.normal(size=1000), rng.exponential(size=1000)
    q = [0.16, 0.5, 0.84]
    np.testing.assert_array_equal(
        np.asarray(quantile(x, q, weights=w)),
        np.asarray(corner_core.quantile(x, q, weights=w)),
    )


def test_quantile_rejects_out_of_bounds():
    with pytest.raises(ValueError, match="between 0 and 1"):
        quantile(np.arange(10.0), [1.5])


# --- ranges ----------------------------------------------------------------


def test_ranges_default_to_full_span():
    xs = np.array([[0.0, 1.0, 2.0], [-5.0, 0.0, 5.0]])
    assert parse_ranges(xs) == [[0.0, 2.0], [-5.0, 5.0]]


def test_ranges_reject_zero_dynamic_range():
    xs = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="no dynamic range"):
        parse_ranges(xs)


def test_ranges_fraction_becomes_equal_tailed_quantiles():
    rng = np.random.default_rng(9)
    xs = rng.normal(size=(1, 10000))
    (lo, hi), = parse_ranges(xs, range_=[0.5])
    np.testing.assert_allclose([lo, hi], np.percentile(xs[0], [25.0, 75.0]))
