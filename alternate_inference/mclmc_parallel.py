"""Parallel-friendly MCLMC kernel helpers.

This mirrors the core BlackJAX MCLMC logic but adds light wrappers so the
kernel and init functions can be vmapped/pmapped over an arbitrary number of
chains. Adaptation is intentionally omitted here; these helpers only cover the
core kernel.
"""
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import IntegratorState, with_isokinetic_maruyama
from blackjax.types import ArrayLike, ArrayTree, PRNGKey
from blackjax.util import generate_unit_vector, pytree_size


class MCLMCInfo(NamedTuple):
    """Additional information on the MCLMC transition."""

    logdensity: float
    kinetic_change: float
    energy_change: float


def _single_init(position: ArrayLike, logdensity_fn: Callable, rng_key: PRNGKey):
    if pytree_size(position) < 2:
        raise ValueError(
            "The target distribution must have more than 1 dimension for MCLMC."
        )
    l, g = jax.value_and_grad(logdensity_fn)(position)

    return IntegratorState(
        position=position,
        momentum=generate_unit_vector(rng_key, position),
        logdensity=l,
        logdensity_grad=g,
    )


def _single_kernel(
    logdensity_fn: Callable,
    inverse_mass_matrix: ArrayTree,
    integrator: Callable,
    desired_energy_var_max_ratio=jnp.inf,
    desired_energy_var=5e-4,
):
    step = with_isokinetic_maruyama(
        integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix)
    )

    def kernel(
        rng_key: PRNGKey, state: IntegratorState, L: float, step_size: float
    ) -> tuple[IntegratorState, MCLMCInfo]:
        (position, momentum, logdensity, logdensitygrad), kinetic_change = step(
            state, step_size, L, rng_key
        )

        energy_error = kinetic_change - logdensity + state.logdensity

        eev_max_per_dim = desired_energy_var_max_ratio * desired_energy_var
        ndims = pytree_size(position)

        new_state, new_info = jax.lax.cond(
            jnp.abs(energy_error) > jnp.sqrt(ndims * eev_max_per_dim),
            lambda: (
                state,
                MCLMCInfo(
                    logdensity=state.logdensity,
                    energy_change=0.0,
                    kinetic_change=0.0,
                ),
            ),
            lambda: (
                IntegratorState(position, momentum, logdensity, logdensitygrad),
                MCLMCInfo(
                    logdensity=logdensity,
                    energy_change=energy_error,
                    kinetic_change=kinetic_change,
                ),
            ),
        )

        return new_state, new_info

    return kernel


def _make_mapper(map_factory: Optional[Callable], in_axes):
    if map_factory is not None:
        return map_factory
    return lambda fn: jax.vmap(fn, in_axes=in_axes)


def _maybe_jit(map_factory: Optional[Callable], fn: Callable) -> Callable:
    """Jit when using the default mapper; leave as-is if user supplies a mapper."""
    return fn if map_factory is not None else jax.jit(fn)


def init_multi(
    positions: ArrayLike,
    rng_keys: PRNGKey,
    logdensity_fn: Callable,
    map_factory: Optional[Callable] = None,
):
    """Vectorized initializer for multiple chains.

    `rng_keys` can be a single key (will be split) or an array of keys with
    leading dimension equal to the number of chains.
    """
    mapper = _make_mapper(map_factory, in_axes=(0, 0))
    if rng_keys.ndim == 0:
        rng_keys = jax.random.split(rng_keys, positions.shape[0])
    init_fn = mapper(lambda pos, key: _single_init(pos, logdensity_fn, key))
    init_fn = _maybe_jit(map_factory, init_fn)
    return init_fn(positions, rng_keys)


def build_kernel_multi(
    logdensity_fn: Callable,
    inverse_mass_matrix: ArrayTree,
    integrator: Callable,
    desired_energy_var_max_ratio=jnp.inf,
    desired_energy_var=5e-4,
    map_factory: Optional[Callable] = None,
):
    """Vectorized MCLMC kernel over a leading chain axis.

    The default uses `jax.vmap` with shared `L` and `step_size` across chains.
    To use per-chain hyperparameters or to map across devices, provide a custom
    `map_factory`, e.g. `lambda f: jax.vmap(f, in_axes=(0, 0, 0, 0))` or
    `lambda f: jax.pmap(f, in_axes=(0, 0, None, None), axis_name="chain")`.
    """
    single_kernel = _single_kernel(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inverse_mass_matrix,
        integrator=integrator,
        desired_energy_var_max_ratio=desired_energy_var_max_ratio,
        desired_energy_var=desired_energy_var,
    )
    mapper = _make_mapper(map_factory, in_axes=(0, 0, None, None))
    kernel = mapper(single_kernel)
    return _maybe_jit(map_factory, kernel)


def mclmc_multi(
    logdensity_fn: Callable,
    L: float,
    step_size: float,
    num_chains: int,
    *,
    integrator,
    inverse_mass_matrix: ArrayTree = 1.0,
    desired_energy_var_max_ratio=jnp.inf,
    map_factory: Optional[Callable] = None,
) -> SamplingAlgorithm:
    """Top-level parallel MCLMC API mirroring `blackjax.mcmc.mclmc.mclmc`.

    This returns a ``SamplingAlgorithm`` where both `init_fn` and `step_fn` are
    vectorized over chains using `map_factory` (defaults to `jax.vmap`). All
    chains share `L` and `step_size` unless the provided `map_factory` maps
    those arguments as well.
    """
    kernel = build_kernel_multi(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inverse_mass_matrix,
        integrator=integrator,
        desired_energy_var_max_ratio=desired_energy_var_max_ratio,
        map_factory=map_factory,
    )

    def init_fn(positions: ArrayLike, rng_keys: PRNGKey):
        return init_multi(
            positions=positions,
            rng_keys=rng_keys,
            logdensity_fn=logdensity_fn,
            map_factory=map_factory,
        )

    def update_fn(rng_key, state):
        rng_keys = jax.random.split(rng_key, num_chains)
        return kernel(rng_keys, state, L, step_size)

    return SamplingAlgorithm(init_fn, update_fn)
