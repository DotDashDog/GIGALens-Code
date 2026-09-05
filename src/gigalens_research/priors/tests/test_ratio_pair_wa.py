"""Unit tests for the ratio-pair + wa grouped prior (analytic stand-in pair, CPU).

The stand-in has the qualitative structure of two deflection ratios at
well-separated redshifts with a weak, wa-dependent bend: smooth, injective on the
box at every wa, non-separable level contours crossing at O(10 deg). The real
cosmology wiring is exercised in the lab log's pilot
(docs/logs/ersatz-carousel-cosmology.md) and by the validator on the
ersatz-carousel pair.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens_research.priors.ratio_pair_wa import (
    RatioPairWaBijector,
    RatioPairWaUniform,
    validate_ratio_pair_wa,
)

OM = (0.05, 0.95)
W0 = (-2.0, -1.0 / 3.0)
WA = (-3.0, 2.0)
DET_ATOL = 1e-6
RT_ATOL = 1e-9
SMALL = dict(n_image_grid=(21, 21, 11), init_table_size=(9, 9, 7))


def pair_crossing(om, w, wa):
    """Injective in (om, w) at every wa; contours cross at a modest angle."""
    return (jnp.exp(0.4 * om + 0.15 * w + 0.02 * wa),
            om - 0.35 * w + 0.05 * om * w + 0.03 * wa * (1.0 + om))


def pair_folded(om, w, wa):
    """Second coordinate nearly parallel to the first with a wiggle -> fold."""
    a = jnp.exp(0.4 * om + 0.15 * w + 0.02 * wa)
    return a, a + 0.02 * jnp.sin(6.0 * om) * w


@pytest.fixture(scope="module")
def bij():
    return RatioPairWaBijector(pair_crossing, OM, W0, WA, **SMALL)


def _grid(n=5):
    o, w, a = np.meshgrid(np.linspace(0.1, 0.9, n), np.linspace(-1.9, -0.4, n),
                          np.linspace(-2.8, 1.8, n), indexing="ij")
    return jnp.asarray(np.stack([o.ravel(), w.ravel(), a.ravel()], -1))


def _fresh(z):
    """Break TFP's bijector cache: ``forward(inverse(x))`` on the SAME array object
    returns the cached ``x`` without solving. A numpy round-trip is a new object."""
    return jnp.asarray(np.array(z))


def test_roundtrip_exact(bij):
    x = _grid()
    xr = bij.forward(_fresh(bij.inverse(x)))
    err = np.abs(np.asarray(xr - x)) / np.array(
        [OM[1] - OM[0], W0[1] - W0[0], WA[1] - WA[0]])
    assert err.max() < 1e-12


def test_wa_passes_through_unchanged(bij):
    x = _grid()
    z = _fresh(bij.inverse(x))
    assert np.allclose(np.asarray(bij.forward(z))[:, 2], np.asarray(x)[:, 2], atol=1e-14)


def test_forward_log_det_matches_autodiff(bij):
    z = bij.inverse(_grid(4))
    fldj = np.asarray(bij.forward_log_det_jacobian(z, event_ndims=1))
    J = jax.vmap(jax.jacrev(bij.forward))(z)
    ad = np.log(np.abs(np.linalg.det(np.asarray(J))))
    assert np.all(np.isfinite(fldj))
    assert np.abs(fldj - ad).max() < 1e-10


def test_inverse_log_det_is_negative_forward(bij):
    x = _grid(3)
    z = bij.inverse(x)
    assert np.allclose(np.asarray(bij.inverse_log_det_jacobian(x, event_ndims=1)),
                       -np.asarray(bij.forward_log_det_jacobian(z, event_ndims=1)),
                       atol=1e-12)


def test_custom_vjp_gradient_matches_finite_difference(bij):
    z0 = bij.inverse(jnp.array([0.3, -1.0, 0.4]))
    f = lambda q: jnp.sum(bij.forward(q) * jnp.array([1.0, 2.0, 3.0]))
    g = np.asarray(jax.grad(f)(z0))
    h = 1e-6
    fd = np.array([(float(f(z0 + h * e)) - float(f(z0 - h * e))) / (2 * h)
                   for e in np.eye(3)])
    assert np.abs(g - fd).max() < 1e-6 * max(1.0, np.abs(fd).max())


def test_no_preimage_gets_minus_inf(bij):
    # Far corner of the r-box: outside the (thin) image of the theta-box.
    z = jnp.array([[8.0, -8.0, 0.0]])
    assert np.isneginf(np.asarray(bij.forward_log_det_jacobian(z, event_ndims=1)))[0]


def test_validator_accepts_crossing_and_rejects_fold():
    rep = validate_ratio_pair_wa(pair_crossing, OM, W0, WA, det_atol=DET_ATOL,
                                 roundtrip_atol=RT_ATOL, n_grid=(11, 9, 7),
                                 n_roundtrip=(4, 4, 3), n_image_grid=(21, 21, 11))
    assert rep["n_grid_flips_beyond_atol"] == 0
    assert rep["max_roundtrip_err_outside_degen"] <= RT_ATOL
    with pytest.raises(ValueError, match="changes sign"):
        validate_ratio_pair_wa(pair_folded, OM, W0, WA, det_atol=DET_ATOL,
                               roundtrip_atol=RT_ATOL, n_grid=(11, 9, 7),
                               n_roundtrip=(4, 4, 3), n_image_grid=(21, 21, 11))


def test_uniform_density_is_the_plain_box():
    prior = RatioPairWaUniform(pair_crossing, OM, W0, WA, skip_validation=True, **SMALL)
    ref = tfd.Independent(tfd.Uniform(jnp.array([OM[0], W0[0], WA[0]]),
                                      jnp.array([OM[1], W0[1], WA[1]])), 1)
    x = _grid(3)
    assert np.allclose(np.asarray(prior.log_prob(x)), np.asarray(ref.log_prob(x)))
    assert list(prior.event_shape) == [3]
    esb = prior.experimental_default_event_space_bijector()
    assert isinstance(esb, RatioPairWaBijector)
    # z-space density = box log-prob + fldj: finite on the image.
    z = esb.inverse(x)
    lz = np.asarray(prior.log_prob(esb.forward(z))
                    + esb.forward_log_det_jacobian(z, event_ndims=1))
    assert np.all(np.isfinite(lz))
