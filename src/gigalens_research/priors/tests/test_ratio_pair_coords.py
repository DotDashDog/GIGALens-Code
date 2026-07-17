"""Unit tests for the ratio-PAIR grouped prior (analytic pair fns, CPU).

The pair here is a cheap stand-in with the same qualitative structure as two
deflection ratios at well-separated source redshifts (smooth, injective on the
box, curved non-separable level contours whose crossing angle is O(10 deg));
the real-cosmology wiring is exercised by
``experiments/sample_cosmology/ratio_pair_coverage.py``.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens_research.priors.ratio_pair_coords import (
    RatioPairBijector,
    RatioPairUniform,
    deflection_ratio_pair_fn,
    validate_ratio_pair,
)

OM_BOUNDS = (0.0, 1.0)
W_BOUNDS = (-2.0, -1.0 / 3.0)

# Measured on pair_crossing with the settings below (2026-07-17): round-trip
# 0.0, no degenerate cells at det_atol=1e-6.
DET_ATOL = 1e-6
ROUNDTRIP_ATOL = 1e-9


def pair_crossing(om, w):
    """Injective on the box; contours cross at a modest angle everywhere."""
    return jnp.exp(0.4 * om + 0.15 * w), om - 0.35 * w + 0.05 * om * w


def pair_folded(om, w):
    """Second coordinate nearly parallel to the first with a wiggle: the
    Jacobian determinant changes sign inside the box — must be rejected."""
    r_a = jnp.exp(0.4 * om + 0.15 * w)
    return r_a, r_a + 1e-3 * jnp.sin(6.0 * om)


@pytest.fixture(scope="module")
def bij():
    return RatioPairBijector(
        pair_crossing, OM_BOUNDS, W_BOUNDS,
        n_image_grid=101, init_table_size=15)


@pytest.fixture(scope="module")
def theta_batch():
    rng = np.random.default_rng(0)
    om = OM_BOUNDS[0] + (OM_BOUNDS[1] - OM_BOUNDS[0]) * rng.uniform(
        0.01, 0.99, size=64)
    w = W_BOUNDS[0] + (W_BOUNDS[1] - W_BOUNDS[0]) * rng.uniform(
        0.01, 0.99, size=64)
    return jnp.stack([jnp.asarray(om), jnp.asarray(w)], axis=-1)


def test_roundtrip_through_image(bij, theta_batch):
    theta_rt = bij.forward(bij.inverse(theta_batch))
    np.testing.assert_allclose(
        np.asarray(theta_rt), np.asarray(theta_batch), rtol=0, atol=1e-12)


def test_fldj_matches_autodiff_and_ildj(bij):
    z0 = bij.inverse(jnp.array([0.3, -1.0]))
    fldj = bij.forward_log_det_jacobian(z0, event_ndims=1)
    ildj = bij.inverse_log_det_jacobian(bij.forward(z0), event_ndims=1)
    sld = jnp.linalg.slogdet(jax.jacrev(bij.forward)(z0))[1]
    assert np.isfinite(float(fldj))
    np.testing.assert_allclose(float(fldj), float(sld), rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(fldj + ildj), 0.0, rtol=0, atol=1e-12)


def test_no_preimage_region_rejected(bij):
    # The image fills only part of its bounding box (measured empty fraction
    # ~0.45 for pair_crossing); an extreme box corner has no preimage and the
    # forward log-det must be -inf (zero density), with zero gradients.
    zbad = jnp.array([-6.0, 6.0])
    assert float(bij.forward_log_det_jacobian(zbad, event_ndims=1)) == -np.inf
    g = jax.jacrev(bij.forward)(zbad)
    np.testing.assert_allclose(np.asarray(g), 0.0, rtol=0, atol=0)


def test_validator_passes_crossing_pair():
    report = validate_ratio_pair(
        pair_crossing, OM_BOUNDS, W_BOUNDS,
        det_atol=DET_ATOL, roundtrip_atol=ROUNDTRIP_ATOL,
        n_grid=81, n_roundtrip=21, n_image_grid=101)
    assert report["n_grid_flips_beyond_atol"] == 0
    assert report["max_roundtrip_err_outside_degen"] <= ROUNDTRIP_ATOL
    assert 0.0 <= report["r_box_empty_frac"] < 1.0


def test_validator_rejects_folded_pair():
    with pytest.raises(ValueError, match="folded"):
        validate_ratio_pair(
            pair_folded, OM_BOUNDS, W_BOUNDS,
            det_atol=DET_ATOL, roundtrip_atol=ROUNDTRIP_ATOL,
            n_grid=41, n_roundtrip=9, n_image_grid=41)


def test_uniform_prior_density_is_plain_box():
    prior = RatioPairUniform(
        pair_crossing, OM_BOUNDS, W_BOUNDS, skip_validation=True,
        n_image_grid=41, n_newton=20)
    ref = tfd.Independent(
        tfd.Uniform(low=jnp.asarray([OM_BOUNDS[0], W_BOUNDS[0]]),
                    high=jnp.asarray([OM_BOUNDS[1], W_BOUNDS[1]])),
        reinterpreted_batch_ndims=1)
    theta = jnp.array([0.3, -1.0])
    np.testing.assert_allclose(
        float(prior.log_prob(theta)), float(ref.log_prob(theta)),
        rtol=0, atol=1e-12)
    assert prior._default_event_space_bijector() is prior._esb


def test_deflection_ratio_pair_fn_arg_validation():
    from gigalens.jax.cosmo import w0waCDM_Cosmo

    cosmo = w0waCDM_Cosmo(z_lens=0.49, z_source_ref=0.962)
    fixed = dict(H0=70.0, k=0.0, wa=0.0)
    with pytest.raises(ValueError, match="exactly 2"):
        deflection_ratio_pair_fn(cosmo, (1.166,), fixed=fixed)
    with pytest.raises(ValueError, match="distinct"):
        deflection_ratio_pair_fn(cosmo, (1.166, 1.166), fixed=fixed)
    with pytest.raises(ValueError, match="reference"):
        deflection_ratio_pair_fn(cosmo, (0.962, 4.090), fixed=fixed)
    r_pair_fn = deflection_ratio_pair_fn(cosmo, (1.166, 4.090), fixed=fixed)
    r_a, r_b = r_pair_fn(0.3, -1.0)
    assert np.isfinite(float(r_a)) and np.isfinite(float(r_b))
    assert float(r_a) != float(r_b)
