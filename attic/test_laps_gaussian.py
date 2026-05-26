"""Gaussian-target tests for LAPS (parity with BlackJAX MAMS + end-to-end sharded run).

Run (CPU recommended):
  JAX_PLATFORM_NAME=cpu python -m pytest GIGALens-Code/alternate_inference/test_laps_gaussian.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", False)

from blackjax.mcmc import integrators
from blackjax.mcmc.integrators import IntegratorState

from alternate_inference.laps import full_laps_sharded
from alternate_inference.mclmc_alt import _build_kernel_shardmap, init_multi, isokinetic_mclachlan_smart


def _standard_normal_logdensity(dim: int):
    """Log p(x) for N(0, I) up to an additive constant."""

    def logdensity_fn(x):
        return -0.5 * jnp.sum(jnp.square(x))

    return logdensity_fn


@pytest.fixture(scope="module")
def dim():
    return 5


@pytest.fixture(scope="module")
def logdensity_fn(dim):
    return _standard_normal_logdensity(dim)


def test_mams_trajectory_matches_blackjax_fori_loop(logdensity_fn, dim):
    """Deterministic trajectory: our scan + _build_kernel_shardmap vs BlackJAX inner fori_loop."""
    step_size = jnp.float32(0.15)
    num_steps = 10
    L_proposal_factor = jnp.inf * (num_steps * step_size)  # same as build_kernel default path

    stoch_int = integrators.with_isokinetic_maruyama(
        integrators.isokinetic_mclachlan(
            logdensity_fn=logdensity_fn, inverse_mass_matrix=1.0
        )
    )

    def blackjax_fori_loop(state0, rng_key):
        def step(i, carry):
            state, kinetic_energy, key = carry
            key, sub = jax.random.split(key)
            next_state, next_kinetic = stoch_int(
                state, step_size, L_proposal_factor, sub
            )
            return next_state, kinetic_energy + next_kinetic, key

        end_state, kinetic_sum, _ = jax.lax.fori_loop(
            0, num_steps, step, (state0, jnp.float32(0.0), rng_key)
        )
        delta_energy = -state0.logdensity + end_state.logdensity - kinetic_sum
        return end_state, kinetic_sum, delta_energy

    kernel = _build_kernel_shardmap(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=jnp.float32(1.0),
        integrator=integrators.isokinetic_mclachlan,
    )
    L_no_refresh = jnp.float32(1e30)

    def our_scan(state0, rng_key):
        def body(carry, _):
            state, cum_de, key = carry
            key, sub = jax.random.split(key)
            next_state, info = kernel(
                rng_key=sub, state=state, L=L_no_refresh, step_size=step_size
            )
            return (next_state, cum_de + info.energy_change, key), None

        (end_state, total_de, _), _ = jax.lax.scan(
            body, (state0, state0.logdensity * 0.0, rng_key), xs=None, length=num_steps
        )
        return end_state, total_de

    @jax.jit
    def run(seed):
        key0 = jax.random.key(seed)
        key_pos, key_rest = jax.random.split(key0)
        pos = jax.random.normal(key_pos, (dim,), dtype=jnp.float32) * jnp.float32(0.3)
        l, g = jax.value_and_grad(logdensity_fn)(pos)
        k_m, k_i = jax.random.split(key_rest)
        from blackjax.util import generate_unit_vector

        momentum = generate_unit_vector(k_m, pos)
        state0 = IntegratorState(pos, momentum, l, g)
        end_bj, k_sum, delta_bj = blackjax_fori_loop(state0, k_i)
        end_ours, total_de = our_scan(state0, k_i)
        return end_bj, k_sum, delta_bj, end_ours, total_de

    end_bj, k_sum, delta_bj, end_ours, total_de = run(42)

    assert jnp.allclose(end_bj.position, end_ours.position, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(end_bj.logdensity, end_ours.logdensity, rtol=1e-4, atol=1e-4)
    # total_de = -(logd_end - logd_start - K) = -delta_bj  with BlackJAX's delta definition
    assert jnp.allclose(total_de, -delta_bj, rtol=1e-4, atol=1e-4)


def test_full_laps_sharded_gaussian_e2e(logdensity_fn, dim):
    """End-to-end LAPS on N(0,I): samples roughly match target; adjusted acceptance in band."""
    num_devices = jax.device_count()
    num_chains = max(8, 2 * num_devices)
    # Round down to multiple of num_devices (full_laps_sharded does this internally)
    num_chains = (num_chains // num_devices) * num_devices
    if num_chains == 0:
        num_chains = num_devices

    inv_mm = jnp.eye(dim, dtype=jnp.float32)
    svi_mean = jnp.zeros((dim,), dtype=jnp.float32)

    init_key = jax.random.key(0)
    pos_key, run_key = jax.random.split(init_key)
    positions = jax.random.normal(pos_key, (num_chains, dim), dtype=jnp.float32) * jnp.float32(
        0.5
    )
    state_init = init_multi(positions, pos_key, logdensity_fn)

    kernel_builder = lambda imm: _build_kernel_shardmap(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=imm,
        integrator=isokinetic_mclachlan_smart,
    )

    num_unadjusted = 300
    num_adjusted = 120
    num_results = 200

    (unadj_hist, adj_hist, samples), _ = full_laps_sharded(
        kernel_builder=kernel_builder,
        num_unadjusted_steps=num_unadjusted,
        num_adjusted_steps=num_adjusted,
        num_results=num_results,
        state_init=state_init,
        init_step_size=jnp.sqrt(jnp.float32(dim)) * jnp.float32(0.25),
        init_L=jnp.sqrt(jnp.float32(dim)),
        init_inverse_mass_matrix=inv_mm,
        svi_mean=svi_mean,
        rng_key=run_key,
        num_chains=num_chains,
        C=jnp.float32(0.025),
        alpha=jnp.float32(1.0),
        adj_target_accept=jnp.float32(0.65),
        adj_n_steps=10,
        adapt_mass_matrix=False,
        svi_mass_matrix_weight=jnp.float32(10.0 * num_chains),
    )

    # samples.position: (num_chains, num_results, dim) — materialize for stats (avoids sharded matmul)
    x = np.asarray(jax.device_get(samples.position.reshape(-1, dim)))
    mean = jnp.asarray(np.mean(x, axis=0))
    assert float(jnp.linalg.norm(mean)) < 0.35, f"sample mean too far from 0: {mean}"

    xm = x - np.asarray(mean)
    cov = (xm.T @ xm) / float(x.shape[0] - 1)
    assert np.allclose(np.diag(cov), 1.0, rtol=0.35, atol=0.12)

    acc = np.asarray(jax.device_get(adj_hist.acceptance[:, num_adjusted:]))
    mean_acc_last = float(np.mean(acc[:, -50:]))
    assert 0.45 < mean_acc_last < 0.92, f"mean acceptance last segment: {mean_acc_last}"

    # Smoke: unadjusted phase produced finite diagnostics
    assert bool(jnp.all(jnp.isfinite(jax.device_get(unadj_hist.D_tilde))))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
