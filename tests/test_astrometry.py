"""Unit tests for :mod:`gigalens_research.astrometry`.

These are fast structural tests: input validation, the coordinate ordering, and
the algebra of the systematics budget. They deliberately do *not* certify that
the reported covariance is correct — that is an ensemble question, it costs
hundreds of fits, and it lives in
:mod:`gigalens_research.astrometry.validate`. A green test suite here means the
plumbing is sound, not that the error bars can be believed.
"""
from __future__ import annotations

import numpy as np
import pytest

from gigalens_research.astrometry.measure import (
    AstrometryResult,
    Frame,
    NoiseSpec,
    PSFSpec,
    SystematicsBudget,
    _position_indices,
    common_mode_jacobian,
)
from gigalens_research.astrometry.validate import decompose_common_mode


def _gauss_kernel(npix=15, sigma_pix=1.5):
    g = np.arange(npix) - (npix - 1) / 2.0
    xx, yy = np.meshgrid(g, g)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma_pix ** 2)
    return k / k.sum()


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


def test_frame_roundtrip_is_exact():
    frame = Frame.from_pixel_scale(0.05, 60)
    ra = np.array([0.3, -0.44, 0.07])
    dec = np.array([-0.11, 0.52, 0.0])
    x, y = frame.angle2pix(ra, dec)
    ra2, dec2 = frame.pix2angle(x, y)
    assert np.allclose(ra2, ra, atol=1e-12)
    assert np.allclose(dec2, dec, atol=1e-12)


def test_frame_centres_the_grid():
    frame = Frame.from_pixel_scale(0.05, 61)
    ra, dec = frame.pix2angle(30, 30)
    assert ra == pytest.approx(0.0, abs=1e-12)
    assert dec == pytest.approx(0.0, abs=1e-12)
    assert frame.pixel_scale == pytest.approx(0.05)


def test_frame_handles_a_rotated_flipped_transform():
    """A real WCS usually flips RA and is rotated; the round trip must survive."""
    theta = 0.37
    m = 0.04 * np.array([[-np.cos(theta), np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    frame = Frame(transform_pix2angle=m, ra_at_xy_0=1.2, dec_at_xy_0=-0.8)
    ra = np.array([0.5, -0.2]); dec = np.array([0.1, 0.9])
    ra2, dec2 = frame.pix2angle(*frame.angle2pix(ra, dec))
    assert np.allclose(ra2, ra, atol=1e-12) and np.allclose(dec2, dec, atol=1e-12)


def test_frame_rejects_singular_transform():
    with pytest.raises(ValueError, match="singular"):
        Frame(transform_pix2angle=np.array([[0.05, 0.05], [0.05, 0.05]]))


# ---------------------------------------------------------------------------
# PSF / noise validation
# ---------------------------------------------------------------------------


def test_psf_kernel_is_normalised():
    spec = PSFSpec(kernel=_gauss_kernel() * 7.3)
    assert spec.kernel.sum() == pytest.approx(1.0)


def test_psf_rejects_even_kernel():
    """An even kernel has no centre pixel, which biases every position."""
    with pytest.raises(ValueError, match="odd side lengths"):
        PSFSpec(kernel=np.ones((16, 16)))


def test_noise_spec_requires_exactly_one_model():
    with pytest.raises(ValueError, match="needs a noise model"):
        NoiseSpec()
    with pytest.raises(ValueError, match="both noise_map and background_rms"):
        NoiseSpec(background_rms=0.01, noise_map=np.ones((4, 4)))


def test_noise_map_must_be_positive():
    with pytest.raises(ValueError, match="finite and strictly positive"):
        NoiseSpec(noise_map=np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# Ordering — the silent failure mode
# ---------------------------------------------------------------------------


class _FakeParamClass:
    """Stands in for lenstronomy's Param with its native blocked layout."""

    def __init__(self, n, extra=("point_amp",)):
        names = ["ra_image"] * n + ["dec_image"] * n
        for e in extra:
            names += [e] * n
        self._names = names

    def num_param(self):
        return len(self._names), self._names


def test_position_indices_interleave_a_blocked_vector():
    order, blocked = _position_indices(_FakeParamClass(4))
    assert list(blocked) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(order) == [0, 4, 1, 5, 2, 6, 3, 7]


def test_position_indices_are_found_by_name_not_by_position():
    """Extra leading parameters must not shift the lookup."""
    fake = _FakeParamClass(3)
    fake._names = ["theta_E", "gamma"] + fake._names
    order, _ = _position_indices(fake)
    assert list(order) == [2, 5, 3, 6, 4, 7]


def test_position_indices_reject_a_mismatched_vector():
    fake = _FakeParamClass(2)
    fake._names = [n for n in fake._names if n != "dec_image"]
    with pytest.raises(RuntimeError, match="matched ra_image/dec_image"):
        _position_indices(fake)


def test_interleaving_matches_gigalens_helper():
    """Pin the convention against gigalens' own ``interleave_xy_cov``.

    The two arrive at the answer differently — this module selects named
    indices out of the full parameter vector, gigalens permutes an
    already-blocked matrix — so agreeing here is a real cross-check of the
    convention rather than a restatement of it.
    """
    gigalens_ps = pytest.importorskip(
        "gigalens.jax.point_source_position",
        reason="gigalens (linusu-dev-merge or later) not importable")
    n = 4
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2 * n, 2 * n))
    blocked_cov = a @ a.T                       # in [ra..., dec...] order

    order, blocked_idx = _position_indices(_FakeParamClass(n))
    # Rebuild the interleaved matrix the way measure_astrometry does: index the
    # full parameter-space covariance directly by interleaved index.
    full = np.zeros((3 * n, 3 * n))
    full[np.ix_(blocked_idx, blocked_idx)] = blocked_cov
    ours = full[np.ix_(order, order)]

    theirs = gigalens_ps.interleave_xy_cov(blocked_cov)
    assert np.allclose(ours, theirs)


# ---------------------------------------------------------------------------
# Systematics budget
# ---------------------------------------------------------------------------


def test_zero_budget_is_zero():
    b = SystematicsBudget()
    assert b.is_zero
    assert np.allclose(b.covariance(np.arange(4.0), np.arange(4.0)), 0.0)


def test_pure_translation_is_perfectly_correlated_across_images():
    """A rigid shift correlates like-axis coordinates at exactly 1, not 0.

    This is the whole reason ``cov_img`` exists: a diagonal matrix would call
    these independent.
    """
    n, s = 4, 3e-3
    x = np.array([0.6, -0.5, 0.1, -0.2])
    y = np.array([0.15, 0.2, -0.6, 0.58])
    cov = SystematicsBudget(sigma_translation=s).covariance(x, y)
    assert cov.shape == (2 * n, 2 * n)
    assert np.allclose(np.diag(cov), s ** 2)
    for i in range(n):
        for j in range(n):
            assert cov[2 * i, 2 * j] == pytest.approx(s ** 2)        # x with x
            assert cov[2 * i + 1, 2 * j + 1] == pytest.approx(s ** 2)  # y with y
            assert cov[2 * i, 2 * j + 1] == pytest.approx(0.0)       # x with y


def test_independent_floor_is_diagonal():
    s = 2e-3
    cov = SystematicsBudget(sigma_independent=s).covariance(
        np.arange(3.0), np.arange(3.0))
    assert np.allclose(cov, np.eye(6) * s ** 2)


def test_budget_covariance_is_positive_semidefinite():
    x = np.array([0.6, -0.5, 0.1, -0.2])
    y = np.array([0.15, 0.2, -0.6, 0.58])
    cov = SystematicsBudget(sigma_translation=1e-3, sigma_rotation=5e-4,
                            sigma_scale=1e-3, sigma_independent=2e-4).covariance(x, y)
    assert np.min(np.linalg.eigvalsh(cov)) >= -1e-18
    assert np.allclose(cov, cov.T)


def test_budget_rejects_negative_terms():
    with pytest.raises(ValueError, match="finite and >= 0"):
        SystematicsBudget(sigma_translation=-1e-3)


# ---------------------------------------------------------------------------
# Common-mode decomposition (round trip against the same Jacobian)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("truth", [
    {"t_x": 1.5e-3, "t_y": -0.7e-3, "rotation": 0.0, "scale": 0.0},
    {"t_x": 0.0, "t_y": 0.0, "rotation": 2e-4, "scale": 0.0},
    {"t_x": 0.0, "t_y": 0.0, "rotation": 0.0, "scale": 3e-4},
    {"t_x": -0.4e-3, "t_y": 1.1e-3, "rotation": -1e-4, "scale": 5e-4},
])
def test_decompose_common_mode_recovers_what_the_jacobian_generated(truth):
    x = np.array([0.6, -0.5, 0.1, -0.2])
    y = np.array([0.15, 0.2, -0.6, 0.58])
    jac = common_mode_jacobian(x, y)
    coef = np.array([truth["t_x"], truth["t_y"], truth["rotation"], truth["scale"]])
    got = decompose_common_mode(x, y, jac @ coef)
    for key in ("t_x", "t_y", "rotation", "scale"):
        assert got[key] == pytest.approx(truth[key], abs=1e-12)
    assert got["residual_rms"] == pytest.approx(0.0, abs=1e-12)


def test_decompose_puts_a_non_common_shift_into_the_leftover():
    """A shift of one image alone is mostly *not* common-mode."""
    x = np.array([0.6, -0.5, 0.1, -0.2])
    y = np.array([0.15, 0.2, -0.6, 0.58])
    shift = np.zeros(8)
    shift[0] = 1e-3
    got = decompose_common_mode(x, y, shift)
    assert got["residual_rms"] > 0.2e-3


def test_common_mode_jacobian_columns_are_what_they_claim():
    x = np.array([1.0, -1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, -1.0])
    jac = common_mode_jacobian(x, y)
    assert np.allclose(jac[:, 0], [1, 0, 1, 0, 1, 0, 1, 0])   # shift in x
    assert np.allclose(jac[:, 1], [0, 1, 0, 1, 0, 1, 0, 1])   # shift in y
    # Rotation is perpendicular to the radius; scale is along it.
    for i in range(4):
        r = np.array([x[i], y[i]])
        rot = jac[2 * i:2 * i + 2, 2]
        sca = jac[2 * i:2 * i + 2, 3]
        assert np.dot(r, rot) == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(sca, r)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


def _result(n=3, corr=0.0):
    rng = np.random.default_rng(1)
    sig = np.linspace(1e-3, 2e-3, 2 * n)
    c = np.full((2 * n, 2 * n), corr) + np.eye(2 * n) * (1 - corr)
    cov = c * np.outer(sig, sig)
    return AstrometryResult(
        x_img=rng.normal(size=n), y_img=rng.normal(size=n), cov_img=cov,
        cov_stat=cov, cov_sys=np.zeros_like(cov), amp=np.ones(n),
        amp_err=np.ones(n) * 0.1)


def test_sigma_img_is_the_marginal_of_cov_img():
    res = _result(n=3, corr=0.4)
    assert res.sigma_img.shape == (3, 2)
    assert np.allclose(res.sigma_img.ravel(), np.sqrt(np.diag(res.cov_img)))


def test_to_gigalens_kwargs_hands_over_cov_not_sigma():
    res = _result(n=4, corr=0.3)
    kw = res.to_gigalens_kwargs()
    assert set(kw) == {"x_img", "y_img", "cov_img"}
    assert kw["cov_img"].shape == (8, 8)
    # A copy, so a downstream mutation cannot reach back into the result.
    kw["cov_img"][0, 0] = 123.0
    assert res.cov_img[0, 0] != 123.0


def test_gigalens_accepts_the_handover():
    """The output must actually construct a PointSourcePositionData."""
    pytest.importorskip("jax")
    gigalens_ps = pytest.importorskip(
        "gigalens.jax.point_source_position",
        reason="gigalens (linusu-dev-merge or later) not importable")
    if not hasattr(gigalens_ps, "interleave_xy_cov"):
        pytest.skip("gigalens predates the cov_img feature (PR #112)")
    import jax
    jax.config.update("jax_enable_x64", True)
    from gigalens.jax.scene import Component
    from gigalens.jax.profiles.light.point_source import PointSourcePosition

    res = _result(n=4, corr=0.25)
    src = Component(PointSourcePosition(), dict(dx=0.0, dy=0.0))
    data = gigalens_ps.PointSourcePositionData(src, **res.to_gigalens_kwargs())
    assert data.has_cov
    assert np.allclose(data.cov_img, res.cov_img)
    assert data.n_images == 4
