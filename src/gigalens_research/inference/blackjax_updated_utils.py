import jax.numpy as jnp
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import io_callback
from fastprogress.fastprogress import progress_bar as fastprogress_bar


import blackjax
from blackjax.mcmc.integrators import GeneralIntegrator, IntegratorState, ArrayTree, Callable, euclidean_position_update_fn
from blackjax.mcmc.integrators import with_isokinetic_maruyama, generalized_two_stage_integrator, format_isokinetic_state_output, ravel_pytree, _normalized_flatten_array
from blackjax.mcmc.integrators import mclachlan_coefficients, velocity_verlet_coefficients, yoshida_coefficients, omelyan_coefficients
from blackjax.adaptation.mass_matrix import welford_algorithm, WelfordAlgorithmState
from blackjax.util import generate_unit_vector, pytree_size
from blackjax.types import ArrayLike, PRNGKey
from blackjax.mcmc.mclmc import MCLMCInfo
from blackjax.adaptation.mclmc_adaptation import pytree_size, MCLMCAdaptationState, handle_nans, incremental_value_update


from typing import Callable, NamedTuple, Optional

from scipy.fftpack import next_fast_len

import gigalens.jax.simulator as sim

import time
from threading import Lock


class KernelExtras(NamedTuple):
    """Extra diagnostic fields returned alongside MCLMCInfo.

    These fields are for instrumentation only and do NOT affect sampler
    behavior.  They are threaded out of the kernel so that per-step
    diagnostics can distinguish:
      - energy_change_raw: the raw integrator energy error BEFORE NaN-zeroing.
        When nan_reject is True, MCLMCInfo.energy_change is zeroed; this field
        retains the original (possibly NaN/Inf) value for diagnostic use.
      - kernel_nonan: True when the step was NOT nan/Inf-rejected by the
        kernel (i.e. ~nan_reject).  Distinct from step_size_adapt's
        `successes` (which may also reflect handle_nans logic downstream).
    """
    energy_change_raw: jnp.ndarray
    kernel_nonan: jnp.ndarray


class MCLMCInfoWithExtras(NamedTuple):
    """Drop-in replacement for MCLMCInfo that carries KernelExtras alongside.

    The four fields (logdensity, energy_change, kinetic_change, nonans) are
    identical in meaning to blackjax.mcmc.mclmc.MCLMCInfo; extras is the
    additional diagnostic payload.  We cannot mutate the upstream namedtuple,
    so we wrap it here.
    """
    logdensity: jnp.ndarray
    energy_change: jnp.ndarray
    kinetic_change: jnp.ndarray
    nonans: jnp.ndarray
    extras: KernelExtras


def _build_kernel_shardmap(logdensity_fn, inverse_mass_matrix, integrator):
    """shard_map-compatible version of blackjax.mcmc.mclmc.build_kernel.
    Replaces jax.lax.cond with jnp.where to avoid varying-annotation mismatch.

    Returns MCLMCInfoWithExtras which carries the standard four MCLMCInfo
    fields plus a KernelExtras bundle for diagnostic instrumentation.
    The extras do NOT influence sampler behavior.
    """
    step = with_isokinetic_maruyama(
        integrator(logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix)
    )

    def kernel(rng_key, state, L, step_size):
        (position, momentum, logdensity, logdensitygrad), kinetic_change = step(
            state, step_size, L, rng_key
        )
        energy_error = kinetic_change - logdensity + state.logdensity

        # Baseline behavior: reject (revert state) on NaN/Inf energy error only.
        nan_reject = jnp.isinf(energy_error) | jnp.isnan(energy_error)
        select = lambda a, b: jax.tree.map(lambda x, y: jnp.where(nan_reject, x, y), a, b)

        new_state = IntegratorState(
            *select(
                (state.position, state.momentum, state.logdensity, state.logdensity_grad),
                (position, momentum, logdensity, logdensitygrad),
            )
        )
        # NaN/Inf: report a zero energy change (the value is meaningless and
        # zeroing keeps it out of the step-size controller).
        energy_change_zeroed = jnp.where(nan_reject, jnp.zeros_like(energy_error), energy_error)
        new_info = MCLMCInfoWithExtras(
            logdensity=jnp.where(nan_reject, state.logdensity, logdensity),
            energy_change=energy_change_zeroed,
            kinetic_change=jnp.where(nan_reject, jnp.zeros_like(kinetic_change), kinetic_change),
            nonans=~nan_reject,
            extras=KernelExtras(
                # Raw energy error BEFORE zeroing — carries the true NaN/Inf
                # value when nan_reject is True.  Useful for distinguishing
                # "zeroed because NaN" from "truly small dE" in diagnostics.
                energy_change_raw=energy_error,
                # kernel_nonan: True iff this step was NOT nan/Inf-rejected.
                kernel_nonan=~nan_reject,
            ),
        )
        return new_state, new_info

    return kernel


class AdjustedKernelExtras(NamedTuple):
    """Diagnostic fields for the MAMS (adjusted MCLMC) kernel; instrumentation only.

    energy_change_raw
        The raw MH log-ratio ``delta = logp_new - logp_old - kinetic_change``
        BEFORE the NaN -> -inf guard. Lets diagnostics distinguish a genuinely
        large (rejected) energy error from a NaN-forced rejection.
    kernel_nonan
        True iff the proposed endpoint (position + energy) was finite.
    """
    energy_change_raw: jnp.ndarray
    kernel_nonan: jnp.ndarray


class AdjustedMCLMCInfo(NamedTuple):
    """Per-transition info for the MAMS kernel (analog of MCLMCInfoWithExtras).

    acceptance_rate
        The Metropolis acceptance probability ``min(1, e^delta)``.
    is_accepted
        Whether the proposal was accepted (and finite).
    energy_change
        The MH log-ratio ``delta`` after the NaN -> -inf guard (so a NaN step
        reads as -inf, i.e. certain rejection).
    num_integration_steps
        Number of deterministic isokinetic steps in the trajectory.
    nonans
        Finite-proposal flag (same as extras.kernel_nonan; kept for parity with
        MCLMCInfoWithExtras.nonans).
    extras
        AdjustedKernelExtras diagnostic bundle.
    """
    acceptance_rate: jnp.ndarray
    is_accepted: jnp.ndarray
    energy_change: jnp.ndarray
    num_integration_steps: jnp.ndarray
    nonans: jnp.ndarray
    extras: AdjustedKernelExtras


def _build_adjusted_kernel_shardmap(logdensity_fn, inverse_mass_matrix, integrator):
    """shard_map-safe MAMS (Metropolis-Adjusted Microcanonical Sampler) kernel.

    Mirrors `_build_kernel_shardmap` (the unadjusted MCLMC kernel) but implements
    the *adjusted*, pure-Hamiltonian variant of Robnik et al. (MAMS):

      * the momentum is FULLY refreshed (unit vector) once per transition,
      * the trajectory is `num_integration_steps` DETERMINISTIC isokinetic steps
        with NO Maruyama/Langevin partial refreshment (the "Hamiltonian, not
        Langevin" variant -- equivalent to blackjax adjusted_mclmc with
        L_proposal_factor=inf),
      * a Metropolis-Hastings step accepts/rejects the trajectory endpoint.

    Like `_build_kernel_shardmap`, all branching is done with `jnp.where`
    (never `jax.lax.cond`) so the kernel passes shard_map's varying-manual-axis
    check. The dense (non-diagonal) inverse mass matrix is handled entirely
    inside `integrator` -- pass `isokinetic_mclachlan_smart`, exactly as MCLMC
    does -- so MAMS inherits the dense-metric support for free.

    The returned kernel signature is
        kernel(rng_key, state, step_size, num_integration_steps)
    -- note `num_integration_steps` replaces MCLMC's momentum-decoherence `L`;
    the driver computes it (shared across chains) from L and the step size.
    """
    # Raw deterministic isokinetic integrator: (state, step_size) -> (state, kinetic_change).
    # No with_isokinetic_maruyama wrapper => no intra-trajectory momentum noise.
    isokinetic_step = integrator(
        logdensity_fn=logdensity_fn, inverse_mass_matrix=inverse_mass_matrix
    )

    def kernel(rng_key, state, step_size, num_integration_steps):
        key_momentum, key_accept = jax.random.split(rng_key, 2)

        # Full momentum refresh. Cast to the position dtype so the carry is a
        # single dtype (generate_unit_vector draws float32 normals even under
        # x64); mirrors _single_init.
        momentum = jax.tree_util.tree_map(
            lambda m, p: m.astype(p.dtype),
            generate_unit_vector(key_momentum, state.position),
            state.position,
        )
        start = IntegratorState(
            state.position, momentum, state.logdensity, state.logdensity_grad
        )

        # Deterministic trajectory: integrate num_integration_steps isokinetic
        # steps, accumulating the kinetic-energy change (the MH "work" term).
        # num_integration_steps is identical across chains (shared Halton), so
        # the fori_loop trip count never varies within a vmapped/sharded batch.
        def body(_, carry):
            st, ke = carry
            new_st, dke = isokinetic_step(st, step_size)
            return new_st, ke + dke

        end_state, kinetic_change = jax.lax.fori_loop(
            0, num_integration_steps,
            body, (start, jnp.zeros_like(state.logdensity)),
        )

        # MH log-ratio, same convention as blackjax.adjusted_mclmc_proposal:
        #   delta = logp_new - logp_old - kinetic_change ;  p_accept = min(1, e^delta).
        # (This equals -energy_error of the unadjusted MCLMC kernel.)
        delta_energy_raw = -state.logdensity + end_state.logdensity - kinetic_change
        finite = jnp.logical_and(
            jnp.all(jnp.isfinite(ravel_pytree(end_state.position)[0])),
            jnp.isfinite(delta_energy_raw),
        )
        # NaN/Inf -> certain rejection (matches the upstream NaN -> -inf guard).
        delta_energy = jnp.where(finite, delta_energy_raw, -jnp.inf)

        p_accept = jnp.minimum(jnp.exp(delta_energy), 1.0)
        do_accept = jnp.logical_and(jax.random.bernoulli(key_accept, p_accept), finite)

        select = lambda a, b: jax.tree.map(
            lambda x, y: jnp.where(do_accept, x, y), a, b
        )
        new_state = IntegratorState(
            *select(
                (end_state.position, end_state.momentum,
                 end_state.logdensity, end_state.logdensity_grad),
                (state.position, state.momentum,
                 state.logdensity, state.logdensity_grad),
            )
        )
        new_info = AdjustedMCLMCInfo(
            acceptance_rate=p_accept,
            is_accepted=do_accept,
            energy_change=delta_energy,
            num_integration_steps=num_integration_steps,
            nonans=finite,
            extras=AdjustedKernelExtras(
                energy_change_raw=delta_energy_raw,
                kernel_nonan=finite,
            ),
        )
        return new_state, new_info

    return kernel


def _gen_scan_fn_one_bar(num_samples, progress_bar, print_rate=None, axis_name=None):
    if not progress_bar:
        return jax.lax.scan

    if print_rate is None:
        print_rate = int(num_samples / 20) if num_samples > 20 else 1

    bar_state = {"bar": None}
    lock = Lock()

    def _update_bar(iter_num):
        iter_num = int(iter_num)
        with lock:
            if bar_state["bar"] is None:
                bar_state["bar"] = fastprogress_bar(range(num_samples))
                bar_state["bar"].update(0)
            bar_state["bar"].update_bar(iter_num + 1)
        return jnp.int32(0)

    def _close_bar(_):
        with lock:
            if bar_state["bar"] is not None:
                bar_state["bar"].on_iter_end()

    def scan_wrap(f, init, *args, **kwargs):
        def wrapped(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x

            should_render = True if axis_name is None else (jax.lax.axis_index(axis_name) == 0)
            should_update = should_render & (
                (iter_num % print_rate == 0) | (iter_num == (num_samples - 1))
            )

            _ = jax.lax.cond(
                should_update,
                lambda _: io_callback(_update_bar, jnp.int32(0), iter_num),
                lambda _: jnp.int32(0),
                operand=None,
            )

            carry, y = f(carry, x)

            _ = jax.lax.cond(
                should_render & (iter_num == num_samples - 1),
                lambda _: io_callback(_close_bar, None, iter_num),
                lambda _: None,
                operand=None,
            )

            return carry, y

        return jax.lax.scan(wrapped, init, *args, **kwargs)

    return scan_wrap


def _ess_shardmap(input_array, chain_axis=0, sample_axis=1):
    """shard_map-compatible effective_sample_size.
    Replaces jax.lax.scan with associative_scan and jax.lax.cond with jnp.where.
    """
    input_shape = input_array.shape
    sample_axis = sample_axis if sample_axis >= 0 else len(input_shape) + sample_axis
    num_chains = input_shape[chain_axis]
    num_samples = input_shape[sample_axis]

    mean_across_chain = input_array.mean(axis=sample_axis, keepdims=True)
    centered_array = input_array - mean_across_chain
    m = next_fast_len(2 * num_samples)
    ifft_ary = jnp.fft.rfft(centered_array, n=m, axis=sample_axis)
    ifft_ary *= jnp.conjugate(ifft_ary)
    autocov_value = jnp.fft.irfft(ifft_ary, n=m, axis=sample_axis)
    autocov_value = (
        jnp.take(autocov_value, jnp.arange(num_samples), axis=sample_axis) / num_samples
    )
    mean_autocov_var = autocov_value.mean(chain_axis, keepdims=True)
    mean_var0 = (
        jnp.take(mean_autocov_var, jnp.array([0]), axis=sample_axis)
        * num_samples / (num_samples - 1.0)
    )
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    weighted_var = jnp.where(
        num_chains > 1,
        weighted_var + mean_across_chain.var(axis=chain_axis, ddof=1, keepdims=True),
        weighted_var,
    )

    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(
        mean_autocov_var, jnp.arange(1, num_samples_even), axis=sample_axis
    )
    rho_hat = jnp.concatenate(
        [
            jnp.ones_like(mean_var0),
            1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,
        ],
        axis=sample_axis,
    )

    rho_hat = jnp.moveaxis(rho_hat, sample_axis, 0)
    rho_hat_even = rho_hat[0::2]
    rho_hat_odd = rho_hat[1::2]

    mask0 = (rho_hat_even + rho_hat_odd) > 0.0

    # Initial positive sequence via prefix-AND (replaces scan)
    cumulative_mask = jax.lax.associative_scan(jnp.logical_and, mask0, axis=0)
    max_t = jnp.sum(cumulative_mask, axis=0).astype(int) - 1
    max_t = jnp.maximum(max_t, 0)

    indices = jnp.indices(max_t.shape)
    indices = tuple([max_t + 1] + [indices[i] for i in range(max_t.ndim)])

    rho_hat_odd = jnp.where(cumulative_mask, rho_hat_odd, jnp.zeros_like(rho_hat_odd))
    mask_even = cumulative_mask.at[indices].set(rho_hat_even[indices] > 0)
    rho_hat_even = jnp.where(mask_even, rho_hat_even, jnp.zeros_like(rho_hat_even))

    # Monotone sequence via prefix-min (replaces scan)
    rho_hat_sum = rho_hat_even + rho_hat_odd
    rho_hat_sum_monotone = jax.lax.associative_scan(jnp.minimum, rho_hat_sum, axis=0)
    update_mask = rho_hat_sum > rho_hat_sum_monotone

    rho_hat_even_final = jnp.where(update_mask, rho_hat_sum_monotone / 2.0, rho_hat_even)
    rho_hat_odd_final = jnp.where(update_mask, rho_hat_sum_monotone / 2.0, rho_hat_odd)

    ess_raw = num_chains * num_samples
    tau_hat = (
        -1.0
        + 2.0 * jnp.sum(rho_hat_even_final + rho_hat_odd_final, axis=0)
        - rho_hat_even_final[indices]
    )
    tau_hat = jnp.maximum(tau_hat, 1 / jnp.log10(ess_raw))
    ess = ess_raw / tau_hat

    return ess.squeeze()

#* ---- INTEGRATORS SUPPORTING NON-DIAGONAL MASS MATRICES ---------------------------------------------
#  Define new isokinetic mclachlan. Only difference is it now takes in 2D, non-diagonal inverse matrix

def generate_isokinetic_integrator_smart(coefficients):
    """Make an isokinetic integrator. Exactly the same as the blackjax version, but works with non-diagonal mass matrices.

    Args:
        coefficients (jnp.array): The coefficents for the integrator
    """
    def isokinetic_integrator(
        logdensity_fn: Callable, inverse_mass_matrix: ArrayTree = 1.0
    ) -> GeneralIntegrator:
        position_update_fn = euclidean_position_update_fn(logdensity_fn)
        one_step = generalized_two_stage_integrator(
            esh_dynamics_momentum_update_one_step_smart(inverse_mass_matrix),
            position_update_fn,
            coefficients,
            format_output_fn=format_isokinetic_state_output,
        )
        return one_step

    return isokinetic_integrator

def esh_dynamics_momentum_update_one_step_smart(inverse_mass_matrix):
    if len(inverse_mass_matrix.shape) != 2:
        raise ValueError("inverse_mass_matrix must have 2 dimensions. If you're trying to just input the diagonal, switch to the unmodified blackjax version.")
    
    chol_inverse_mass_matrix = jnp.linalg.cholesky(inverse_mass_matrix)

    def update(
        momentum: ArrayTree,
        logdensity_grad: ArrayTree,
        step_size: float,
        coef: float,
        previous_kinetic_energy_change=None,
        is_last_call=False,
    ):
        """Momentum update based on Esh dynamics.

        The momentum updating map of the esh dynamics as derived in :cite:p:`steeg2021hamiltonian`
        There are no exponentials e^delta, which prevents overflows when the gradient norm
        is large.
        """
        del is_last_call

        logdensity_grad = logdensity_grad
        flatten_grads, unravel_fn = ravel_pytree(logdensity_grad)
        flatten_grads = chol_inverse_mass_matrix.T @ flatten_grads
        flatten_momentum, _ = ravel_pytree(momentum)
        dims = flatten_momentum.shape[0]
        normalized_gradient, gradient_norm = _normalized_flatten_array(flatten_grads)
        momentum_proj = jnp.dot(flatten_momentum, normalized_gradient)
        delta = step_size * coef * gradient_norm / (dims - 1)
        zeta = jnp.exp(-delta)
        new_momentum_raw = (
            normalized_gradient * (1 - zeta) * (1 + zeta + momentum_proj * (1 - zeta))
            + 2 * zeta * flatten_momentum
        )
        new_momentum_normalized, _ = _normalized_flatten_array(new_momentum_raw)
        gr = unravel_fn(chol_inverse_mass_matrix@new_momentum_normalized)
        next_momentum = unravel_fn(new_momentum_normalized)
        kinetic_energy_change = (
            delta
            - jnp.log(2)
            + jnp.log(1 + momentum_proj + (1 - momentum_proj) * zeta**2)
        ) * (dims - 1)
        if previous_kinetic_energy_change is not None:
            kinetic_energy_change += previous_kinetic_energy_change
        return next_momentum, gr, kinetic_energy_change

    return update

isokinetic_mclachlan_smart = generate_isokinetic_integrator_smart(mclachlan_coefficients)
isokinetic_velocity_verlet_smart = generate_isokinetic_integrator_smart(velocity_verlet_coefficients)
isokinetic_yoshida_smart = generate_isokinetic_integrator_smart(yoshida_coefficients)
isokinetic_omelyan_smart = generate_isokinetic_integrator_smart(omelyan_coefficients)

def _single_init(position: ArrayLike, logdensity_fn: Callable, rng_key: PRNGKey):
    if pytree_size(position) < 2:
        raise ValueError(
            "The target distribution must have more than 1 dimension for MCLMC."
        )
    l, g = jax.value_and_grad(logdensity_fn)(position)

    # generate_unit_vector samples jax.random.normal, which defaults to float32 even under
    # jax_enable_x64; cast the momentum to the position dtype so the initial state is
    # single-dtype. Otherwise it clashes with the kernel's float64 momentum in the
    # lax.select inside handle_nans (and similar branch sites).
    momentum = jax.tree_util.tree_map(
        lambda m, p: m.astype(p.dtype),
        generate_unit_vector(rng_key, position),
        position,
    )
    return IntegratorState(
        position=position,
        momentum=momentum,
        logdensity=l,
        logdensity_grad=g,
    )

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

def _make_mapper(map_factory: Optional[Callable], in_axes):
    if map_factory is not None:
        return lambda fn: map_factory(fn, in_axes=in_axes)
    return lambda fn: jax.vmap(fn, in_axes=in_axes)


def _maybe_jit(map_factory: Optional[Callable], fn: Callable) -> Callable:
    """Jit when using the default mapper; leave as-is if user supplies a mapper."""
    return fn if map_factory is not None else jax.jit(fn)

def welford_combine(wa_state1, wa_state2):
    mean_a, m2_a, sample_size_a = wa_state1
    mean_b, m2_b, sample_size_b = wa_state2

    sample_size = sample_size_a + sample_size_b

    delta = mean_b - mean_a
    mean = mean_a + delta * (sample_size_b/sample_size)
    updated_delta = mean_b - mean
    m2 = m2_a + m2_b + (sample_size_a*sample_size_b/sample_size) * jnp.outer(updated_delta, delta)

    # delta = value - mean
    # mean = mean + delta / sample_size
    # updated_delta = value - mean
    # new_m2 = m2 + jnp.outer(updated_delta, delta)

    return WelfordAlgorithmState(mean, m2, sample_size)