"""Unit tests for the ratio-coordinates grouped prior (analytic u_fn, CPU).

The u_fn here is a cheap stand-in with the same qualitative structure as the
DSPL deflection ratio (monotone in w0 at fixed Om0, curved non-separable level
contours); the real-cosmology wiring is exercised by
``experiments/sample_cosmology/dspl_ratio_coords_gate.py``.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens_research.priors.ratio_coords import (
    RatioCoordsBijector,
    RatioCoordsUniform,
    validate_ratio_coords,
)

OM_BOUNDS = (0.0, 1.0)
W_BOUNDS = (-2.0, -1.0 / 3.0)


def u_curved(om, w):
    """Monotone-increasing in w at fixed om; non-separable (curved contours)."""
    return (1.0 + om) ** 0.2 * jnp.exp(0.15 * w + 0.04 * om * w)


def u_folded(om, w):
    """Non-monotone in w (a genuine fold) — must be rejected."""
    return u_curved(om, w) + 0.05 * jnp.sin(4.0 * w)


@pytest.fixture(scope="module")
def bij():
    return RatioCoordsBijector(u_curved, OM_BOUNDS, W_BOUNDS)


@pytest.fixture(scope="module")
def z_batch():
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.normal(size=(64, 2)))


def test_forward_in_box_and_roundtrip(bij, z_batch):
    x = bij.forward(z_batch)
    om, w = x[..., 0], x[..., 1]
    assert bool(jnp.all((om > OM_BOUNDS[0]) & (om < OM_BOUNDS[1])))
    assert bool(jnp.all((w >= W_BOUNDS[0]) & (w <= W_BOUNDS[1])))
    z_back = bij.inverse(x)
    np.testing.assert_allclose(np.asarray(z_back), np.asarray(z_batch),
                               rtol=0, atol=1e-9)
    x_back = bij.forward(bij.inverse(x))
    np.testing.assert_allclose(np.asarray(x_back), np.asarray(x),
                               rtol=0, atol=1e-12)


def test_solver_residual(bij, z_batch):
    x = bij.forward(z_batch)
    om, w = x[..., 0], x[..., 1]
    # w solves u_curved(om, w) = u for the u implied by z2: check the residual
    # against the bracket-implied u directly.
    u_a = u_curved(om, jnp.full_like(om, W_BOUNDS[0]))
    u_b = u_curved(om, jnp.full_like(om, W_BOUNDS[1]))
    from jax.scipy.special import ndtr
    u_target = u_a + (u_b - u_a) * ndtr(z_batch[..., 1])
    np.testing.assert_allclose(np.asarray(u_curved(om, w)),
                               np.asarray(u_target), rtol=0, atol=1e-13)


def test_fldj_matches_numeric_jacobian(bij, z_batch):
    fldj = bij.forward_log_det_jacobian(z_batch, event_ndims=1)
    # custom_vjp supports reverse-mode only -> jacrev, per point.
    jac = jax.vmap(jax.jacrev(bij.forward))(z_batch[:16])
    _, logdet = jnp.linalg.slogdet(jac)
    np.testing.assert_allclose(np.asarray(fldj[:16]), np.asarray(logdet),
                               rtol=0, atol=1e-8)


def test_ildj_is_minus_fldj(bij, z_batch):
    x = bij.forward(z_batch)
    fldj = bij.forward_log_det_jacobian(z_batch, event_ndims=1)
    ildj = bij.inverse_log_det_jacobian(x, event_ndims=1)
    np.testing.assert_allclose(np.asarray(ildj), np.asarray(-fldj),
                               rtol=0, atol=1e-8)


def test_gradients_flow_and_are_finite(bij, z_batch):
    # Emulates ProbModel.log_prior's z-gradient: prior log_prob + FLDJ. The FLDJ
    # gradient needs one differentiation THROUGH the solver's implicit-vjp rule.
    dist = RatioCoordsUniform(u_curved, OM_BOUNDS, W_BOUNDS)

    def log_prior(z):
        x = bij.forward(z)
        return jnp.sum(dist.log_prob(x)
                       + bij.forward_log_det_jacobian(z, event_ndims=1))

    g = jax.grad(log_prior)(z_batch)
    assert bool(jnp.all(jnp.isfinite(g)))
    # And a likelihood-like term through the forward map alone.
    g2 = jax.grad(lambda z: jnp.sum(jnp.square(bij.forward(z))))(z_batch)
    assert bool(jnp.all(jnp.isfinite(g2)))


def test_fldj_gradient_matches_finite_difference(bij):
    z0 = jnp.asarray([0.3, -0.7])
    f = lambda z: bij.forward_log_det_jacobian(z, event_ndims=1)
    g = jax.grad(lambda z: jnp.sum(f(z)))(z0)
    eps = 1e-6
    for i in range(2):
        dz = jnp.zeros(2).at[i].set(eps)
        fd = (f(z0 + dz) - f(z0 - dz)) / (2 * eps)
        np.testing.assert_allclose(float(g[i]), float(fd), rtol=1e-5, atol=1e-7)


def test_validator_accepts_monotone_and_rejects_fold():
    report = validate_ratio_coords(u_curved, OM_BOUNDS, W_BOUNDS)
    assert report["n_grid_flips_beyond_atol"] == 0
    assert report["max_interior_excursion"] == 0.0
    with pytest.raises(ValueError, match="not monotone in w0"):
        validate_ratio_coords(u_folded, OM_BOUNDS, W_BOUNDS)


def test_validator_excursion_tolerance():
    # Interior rises above BOTH endpoint values (sin arch vanishing at the two
    # w0 endpoints): a huge du_dw_atol silences the monotonicity check, so the
    # excursion check alone must raise under a strict tolerance.
    w_span = W_BOUNDS[1] - W_BOUNDS[0]

    def u_arch(om, w):
        return w + 1.0 * jnp.sin(jnp.pi * (w - W_BOUNDS[0]) / w_span)

    with pytest.raises(ValueError, match="endpoint bracket"):
        validate_ratio_coords(u_arch, OM_BOUNDS, W_BOUNDS,
                              du_dw_atol=1e9, excursion_atol=0.0)


def test_distribution_is_uniform_box_with_custom_esb():
    dist = RatioCoordsUniform(u_curved, OM_BOUNDS, W_BOUNDS)
    assert list(dist.event_shape) == [2]
    assert isinstance(dist.experimental_default_event_space_bijector(),
                      RatioCoordsBijector)
    area = (OM_BOUNDS[1] - OM_BOUNDS[0]) * (W_BOUNDS[1] - W_BOUNDS[0])
    x_in = jnp.asarray([0.3, -1.0])
    np.testing.assert_allclose(float(dist.log_prob(x_in)), -np.log(area),
                               rtol=0, atol=1e-12)
    samples = dist.sample(512, seed=jax.random.PRNGKey(1))
    assert samples.shape == (512, 2)
    assert bool(jnp.all((samples[:, 0] >= OM_BOUNDS[0])
                        & (samples[:, 0] <= OM_BOUNDS[1])
                        & (samples[:, 1] >= W_BOUNDS[0])
                        & (samples[:, 1] <= W_BOUNDS[1])))
    assert dist.ratio_coords_report is not None


def test_joint_distribution_named_integration():
    # Mirrors LensModel._derive: JointDistributionNamed + joint esb + inverse of
    # a prior sample (shape derivation path in scene.py).
    dist = RatioCoordsUniform(u_curved, OM_BOUNDS, W_BOUNDS,
                              skip_validation=True)
    joint = tfd.JointDistributionNamed(
        {"cosmo/Om0|cosmo/w0": dist, "other": tfd.Normal(0.0, 1.0)})
    esb = joint.experimental_default_event_space_bijector()
    example = joint.sample(seed=jax.random.PRNGKey(2))
    unc = esb.inverse(example)
    back = esb.forward(unc)
    np.testing.assert_allclose(
        np.asarray(back["cosmo/Om0|cosmo/w0"]),
        np.asarray(example["cosmo/Om0|cosmo/w0"]), rtol=0, atol=1e-9)
    lp = joint.log_prob(back)
    assert bool(jnp.isfinite(lp))


def test_pushforward_density_is_uniform_on_box(bij):
    # p_theta(theta) proportional-to N(z1)N(z2)/|det dtheta/dz| must be CONSTANT
    # on the box when z ~ N(0,I) is pushed through... it is NOT (the prior is
    # uniform by log_prob, not by pushforward) — instead check the actual
    # invariant: the z-space density implied by a UNIFORM theta prior,
    # p_z(z) = p_theta(theta(z)) * |det dtheta/dz|, integrates consistently:
    # compare Monte-Carlo box mass under importance weights to 1.
    rng = np.random.default_rng(3)
    z = jnp.asarray(rng.normal(size=(4096, 2)) * 2.0)  # wide proposal
    x = bij.forward(z)
    fldj = bij.forward_log_det_jacobian(z, event_ndims=1)
    area = (OM_BOUNDS[1] - OM_BOUNDS[0]) * (W_BOUNDS[1] - W_BOUNDS[0])
    log_pz = -jnp.log(area) + fldj  # uniform-box density pulled back to z
    # proposal density q(z) = prod N(0, 2)
    log_q = jnp.sum(-0.5 * (z / 2.0) ** 2 - jnp.log(2.0)
                    - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
    w = jnp.exp(log_pz - log_q)
    mass = float(jnp.mean(w))
    assert abs(mass - 1.0) < 0.05, mass


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------------------------
# u-first ordering (Run D). u_curved has u_a and u_b both INCREASING in om —
# deliberately a different sign combination than the DSPL system (u_a up,
# u_b down), so the interval logic's other branch is covered here and the
# DSPL branch by the CPU gate (dspl_ratio_ufirst_gate.py) before any GPU run.
# ------------------------------------------------------------------------------
from gigalens_research.priors.ratio_coords import (  # noqa: E402
    UFirstRatioCoordsBijector,
    UFirstRatioCoordsUniform,
    validate_u_first_ratio_coords,
)


@pytest.fixture(scope="module")
def ubij():
    return UFirstRatioCoordsBijector(u_curved, OM_BOUNDS, W_BOUNDS)


def test_ufirst_forward_in_box_and_roundtrip(ubij, z_batch):
    x = ubij.forward(z_batch)
    om, w = x[..., 0], x[..., 1]
    assert bool(jnp.all((om >= OM_BOUNDS[0]) & (om <= OM_BOUNDS[1])))
    assert bool(jnp.all((w >= W_BOUNDS[0]) & (w <= W_BOUNDS[1])))
    z_back = ubij.inverse(x)
    np.testing.assert_allclose(np.asarray(z_back), np.asarray(z_batch),
                               rtol=0, atol=1e-8)
    x_back = ubij.forward(z_back)
    np.testing.assert_allclose(np.asarray(x_back), np.asarray(x),
                               rtol=0, atol=1e-11)


def test_ufirst_u_depends_on_z1_alone(ubij, z_batch):
    # The construction's whole point: u(theta(z)) is a function of z1 only.
    x = ubij.forward(z_batch)
    u_vals = u_curved(x[..., 0], x[..., 1])
    z_mod = z_batch.at[:, 1].add(1.7)      # move z2 arbitrarily
    x2 = ubij.forward(z_mod)
    u_vals2 = u_curved(x2[..., 0], x2[..., 1])
    np.testing.assert_allclose(np.asarray(u_vals), np.asarray(u_vals2),
                               rtol=0, atol=1e-11)


def test_ufirst_fldj_matches_numeric_jacobian(ubij, z_batch):
    fldj = ubij.forward_log_det_jacobian(z_batch, event_ndims=1)
    jac = jax.vmap(jax.jacrev(ubij.forward))(z_batch[:16])
    _, logdet = jnp.linalg.slogdet(jac)
    np.testing.assert_allclose(np.asarray(fldj[:16]), np.asarray(logdet),
                               rtol=0, atol=1e-7)


def test_ufirst_ildj_is_minus_fldj(ubij, z_batch):
    x = ubij.forward(z_batch)
    fldj = ubij.forward_log_det_jacobian(z_batch, event_ndims=1)
    ildj = ubij.inverse_log_det_jacobian(x, event_ndims=1)
    np.testing.assert_allclose(np.asarray(ildj), np.asarray(-fldj),
                               rtol=0, atol=1e-7)


def test_ufirst_gradients_flow_and_are_finite(ubij, z_batch):
    dist = UFirstRatioCoordsUniform(u_curved, OM_BOUNDS, W_BOUNDS)

    def log_prior(z):
        x = ubij.forward(z)
        return jnp.sum(dist.log_prob(x)
                       + ubij.forward_log_det_jacobian(z, event_ndims=1))

    g = jax.grad(log_prior)(z_batch)
    assert bool(jnp.all(jnp.isfinite(g)))
    g2 = jax.grad(lambda z: jnp.sum(jnp.square(ubij.forward(z))))(z_batch)
    assert bool(jnp.all(jnp.isfinite(g2)))


def test_ufirst_fldj_gradient_matches_finite_difference(ubij):
    z0 = jnp.asarray([0.4, -0.6])
    f = lambda z: ubij.forward_log_det_jacobian(z, event_ndims=1)
    g = jax.grad(lambda z: jnp.sum(f(z)))(z0)
    eps = 1e-6
    for i in range(2):
        dz = jnp.zeros(2).at[i].set(eps)
        fd = (f(z0 + dz) - f(z0 - dz)) / (2 * eps)
        np.testing.assert_allclose(float(g[i]), float(fd), rtol=1e-4, atol=1e-6)


def test_ufirst_validator_accepts_and_rejects():
    report = validate_u_first_ratio_coords(u_curved, OM_BOUNDS, W_BOUNDS)
    assert report["u_a_n_flips_beyond_atol"] == 0
    assert report["u_band_ceiling"] > report["u_band_floor"]
    assert 0.0 <= report["excluded_prior_volume_frac"] < 1.0
    # a fold in w0 fails the shared w0-monotonicity check first
    with pytest.raises(ValueError, match="not monotone in w0"):
        validate_u_first_ratio_coords(u_folded, OM_BOUNDS, W_BOUNDS)
    # a curve that rises then falls in om fails the endpoint-curve check
    def u_om_arch(om, w):
        return u_curved(om, w) + 0.2 * jnp.sin(jnp.pi * om)
    with pytest.raises(ValueError, match="endpoint curve"):
        validate_u_first_ratio_coords(u_om_arch, OM_BOUNDS, W_BOUNDS,
                                      du_dw_atol=1e9, excursion_atol=1e9)


def test_ufirst_distribution_uniform_box_and_jdn():
    dist = UFirstRatioCoordsUniform(u_curved, OM_BOUNDS, W_BOUNDS,
                                    skip_validation=True)
    assert list(dist.event_shape) == [2]
    assert isinstance(dist.experimental_default_event_space_bijector(),
                      UFirstRatioCoordsBijector)
    area = (OM_BOUNDS[1] - OM_BOUNDS[0]) * (W_BOUNDS[1] - W_BOUNDS[0])
    np.testing.assert_allclose(float(dist.log_prob(jnp.asarray([0.3, -1.0]))),
                               -np.log(area), rtol=0, atol=1e-12)
    joint = tfd.JointDistributionNamed(
        {"cosmo/Om0|cosmo/w0": dist, "other": tfd.Normal(0.0, 1.0)})
    esb = joint.experimental_default_event_space_bijector()
    example = joint.sample(seed=jax.random.PRNGKey(4))
    back = esb.forward(esb.inverse(example))
    np.testing.assert_allclose(
        np.asarray(back["cosmo/Om0|cosmo/w0"]),
        np.asarray(example["cosmo/Om0|cosmo/w0"]), rtol=0, atol=1e-8)
