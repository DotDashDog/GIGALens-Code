import functools
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

try:
    _shard_map = jax.shard_map  # type: ignore[attr-defined]
except AttributeError:
    from jax.experimental.shard_map import shard_map as _shard_map

from blackjax.adaptation.mass_matrix import WelfordAlgorithmState
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import (
    IntegratorState,
    _normalized_flatten_array,
    euclidean_position_update_fn,
    format_isokinetic_state_output,
    generalized_two_stage_integrator,
    mclachlan_coefficients,
    omelyan_coefficients,
    ravel_pytree,
    velocity_verlet_coefficients,
    with_isokinetic_maruyama,
)
from blackjax.util import generate_unit_vector


UnadjustedHist = namedtuple(
    "UnadjustedHist",
    [
        "position",
        "step_size",
        "L",
        "D_tilde",
        "EEVPD",
        "EEVPD_wanted",
        "delta_x2",
        "r_avg",
        "mass_matrix_min_eig",
        "mass_matrix_mean_eig",
        "mass_matrix_max_eig",
        "mass_matrix_adapting",
        "active",
    ],
)

AdjustedHist = namedtuple(
    "AdjustedHist",
    [
        "position",
        "step_size",
        "acceptance",
        "acceptance_running_mean",
        "accepted",
        "energy_error",
        "logdensity",
        "invalid_substeps",
        "invalid_trajectory",
        "mass_matrix_min_eig",
        "mass_matrix_mean_eig",
        "mass_matrix_max_eig",
        "mass_matrix_adapting",
        "adapting",
    ],
)

LAPSCarry = namedtuple(
    "LAPSCarry",
    [
        "unadjusted_state",
        "adjusted_state",
        "step_size",
        "L",
        "inverse_mass_matrix",
        "switch_index",
        "switched",
    ],
)


def _tree_where(mask, on_true, on_false):
    return jax.tree.map(lambda a, b: jnp.where(mask, a, b), on_true, on_false)


def _broadcast_local(value, local_batch):
    value = jnp.asarray(value)
    return jnp.broadcast_to(value[jnp.newaxis, ...], (local_batch,) + value.shape)


def _require_full_rank_mass_matrix(matrix, dim):
    matrix = jnp.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(
            "LAPS now requires `init_inverse_mass_matrix` to be a full-rank 2D matrix."
        )
    if matrix.shape != (dim, dim):
        raise ValueError(
            f"`init_inverse_mass_matrix` must have shape ({dim}, {dim}); got {matrix.shape}."
        )
    return matrix


def _esh_dynamics_momentum_update_one_step_fullrank(inverse_mass_matrix):
    inverse_mass_matrix = jnp.asarray(inverse_mass_matrix)
    if inverse_mass_matrix.ndim != 2:
        raise ValueError("Full-rank isokinetic integrator requires a 2D inverse mass matrix.")

    chol_inverse_mass_matrix = jnp.linalg.cholesky(inverse_mass_matrix)

    def update(
        momentum,
        logdensity_grad,
        step_size,
        coef,
        previous_kinetic_energy_change=None,
        is_last_call=False,
    ):
        del is_last_call

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
        gr = unravel_fn(chol_inverse_mass_matrix @ new_momentum_normalized)
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


def _generate_isokinetic_integrator_fullrank(coefficients):
    def isokinetic_integrator(logdensity_fn, inverse_mass_matrix):
        inverse_mass_matrix = jnp.asarray(inverse_mass_matrix)
        position_update_fn = euclidean_position_update_fn(logdensity_fn)
        one_step = generalized_two_stage_integrator(
            _esh_dynamics_momentum_update_one_step_fullrank(inverse_mass_matrix),
            position_update_fn,
            coefficients,
            format_output_fn=format_isokinetic_state_output,
        )
        return one_step

    return isokinetic_integrator


isokinetic_velocity_verlet_fullrank = _generate_isokinetic_integrator_fullrank(
    velocity_verlet_coefficients
)
isokinetic_mclachlan_fullrank = _generate_isokinetic_integrator_fullrank(
    mclachlan_coefficients
)
isokinetic_omelyan_fullrank = _generate_isokinetic_integrator_fullrank(
    omelyan_coefficients
)


def _welford_seed_m2(covariance, sample_size):
    covariance = jnp.asarray(covariance)
    sample_size = jnp.asarray(sample_size, dtype=covariance.dtype)
    return covariance * sample_size


def _welford_covariance(state):
    denom = jnp.maximum(state.sample_size - 1.0, 1.0)
    return state.m2 / denom


def _welford_combine(state_a, state_b):
    mean_a, m2_a, sample_size_a = state_a
    mean_b, m2_b, sample_size_b = state_b
    sample_size = sample_size_a + sample_size_b
    safe_sample_size = jnp.maximum(sample_size, 1.0)
    delta = mean_b - mean_a
    mean = mean_a + delta * (sample_size_b / safe_sample_size)
    updated_delta = mean_b - mean
    cross = (sample_size_a * sample_size_b / safe_sample_size) * jnp.outer(
        updated_delta, delta
    )
    m2 = m2_a + m2_b + cross
    return WelfordAlgorithmState(mean, m2, sample_size)


def _covariance_to_mass_matrix(reference, covariance):
    ref = jnp.asarray(reference)
    covariance = jnp.asarray(covariance)
    if ref.ndim == 2:
        return covariance
    raise ValueError(
        "LAPS now requires full-rank inverse mass matrices; expected a 2D reference matrix."
    )


def _stabilize_mass_matrix(matrix):
    matrix = jnp.asarray(matrix)
    if matrix.ndim == 2:
        sym = 0.5 * (matrix + matrix.T)
        jitter = jnp.maximum(jnp.trace(sym) / sym.shape[0], 1e-8) * 1e-6
        return sym + jitter * jnp.eye(sym.shape[0], dtype=sym.dtype)
    raise ValueError("LAPS now requires full-rank inverse mass matrices.")


def _mass_matrix_eig_summary(matrix):
    matrix = jnp.asarray(matrix)
    if matrix.ndim == 2:
        eigvals = jnp.linalg.eigvalsh(_stabilize_mass_matrix(matrix))
    else:
        raise ValueError("LAPS now requires full-rank inverse mass matrices.")
    eigvals = jnp.real(eigvals)
    return jnp.min(eigvals), jnp.mean(eigvals), jnp.max(eigvals)


def _normalize_flat_batch(batch):
    norms = jnp.linalg.norm(batch, axis=-1, keepdims=True)
    safe_norms = jnp.where(norms > 0.0, norms, 1.0)
    return batch / safe_norms, norms > 0.0


def _initialize_like_laps(state):
    flat_pos = jax.vmap(lambda x: ravel_pytree(x)[0])(state.position)
    flat_grad = jax.vmap(lambda g: ravel_pytree(g)[0])(state.logdensity_grad)
    equi_diag = jnp.mean(-flat_pos * flat_grad, axis=0)
    signs = jnp.where(equi_diag < 1.0, -1.0, 1.0)
    grad_unit, has_grad = _normalize_flat_batch(flat_grad)
    aligned = grad_unit * signs[jnp.newaxis, :]
    momentum_flat = jnp.where(has_grad, aligned, state.momentum)
    return state._replace(momentum=momentum_flat)


def _update_history(new_vals, history):
    return jnp.concatenate((new_vals[jnp.newaxis, :], history[:-1, :]), axis=0)


def _update_history_scalar(new_val, history):
    return jnp.concatenate((jnp.asarray([new_val]), history[:-1]), axis=0)


def _contract_history(theta, weights):
    weight_sum = jnp.maximum(jnp.sum(weights), 1.0)
    normalized_weights = weights / weight_sum
    square_average = jnp.square(jnp.average(theta, axis=0, weights=normalized_weights))
    average_square = jnp.average(
        jnp.square(theta), axis=0, weights=normalized_weights
    )
    r = (average_square - square_average) / jnp.maximum(square_average, 1e-12)
    return jnp.array([jnp.max(r), jnp.average(r)])


def _equipartition_diagonal_loss(eii):
    return jnp.mean(jnp.square(1.0 - eii))


def _integrator_state_is_finite(state, kinetic_change):
    flat_position = ravel_pytree(state.position)[0]
    flat_momentum = ravel_pytree(state.momentum)[0]
    flat_grad = ravel_pytree(state.logdensity_grad)[0]
    return (
        jnp.all(jnp.isfinite(flat_position))
        & jnp.all(jnp.isfinite(flat_momentum))
        & jnp.isfinite(state.logdensity)
        & jnp.all(jnp.isfinite(flat_grad))
        & jnp.isfinite(kinetic_change)
    )


def _bisection_monotonic_update(
    state,
    exp_x,
    acc_rate_new,
    acc_prob_wanted,
    reduce_shift=jnp.log(2.0),
    tolerance=0.03,
):
    bounds, terminated = state
    acc_high = acc_rate_new > acc_prob_wanted
    x = jnp.log(exp_x)

    def on_true(current_bounds):
        lower, upper = current_bounds
        lower = jnp.max(jnp.array([lower, x]))
        return jnp.array([lower, upper]), lower + reduce_shift

    def on_false(current_bounds):
        lower, upper = current_bounds
        upper = jnp.min(jnp.array([upper, x]))
        return jnp.array([lower, upper]), upper - reduce_shift

    bounds_new, x_new = jax.lax.cond(acc_high, on_true, on_false, bounds)
    bracketing = jnp.all(jnp.isfinite(bounds_new))
    x_new = jax.lax.cond(bracketing, lambda b: jnp.average(b), lambda _: x_new, bounds_new)
    step_size = jnp.where(terminated, exp_x, jnp.exp(x_new))
    terminated_new = (jnp.abs(acc_rate_new - acc_prob_wanted) < tolerance) | terminated
    return (bounds_new, terminated_new), step_size


def _adjusted_mclmc_proposal_shardmap(
    base_kernel,
    rng_key,
    state,
    step_size,
    num_integration_steps,
    L_proposal_factor=1.25,
):
    key_momentum, key_integrator, key_accept = jax.random.split(rng_key, 3)
    momentum = generate_unit_vector(key_momentum, state.position)
    integrator_state = IntegratorState(
        state.position, momentum, state.logdensity, state.logdensity_grad
    )
    partial_refresh_L = L_proposal_factor * (num_integration_steps * step_size)
    trajectory_keys = jax.random.split(key_integrator, num_integration_steps)

    def body_fn(carry, key):
        next_state, info = base_kernel(
            rng_key=key,
            state=carry,
            L=partial_refresh_L,
            step_size=step_size,
        )
        return next_state, info.energy_change

    end_state, energy_changes = jax.lax.scan(body_fn, integrator_state, trajectory_keys)
    energy_error = jnp.sum(energy_changes)
    log_p_accept = -energy_error
    log_p_accept = jnp.where(jnp.isnan(log_p_accept), -jnp.inf, log_p_accept)
    p_accept = jnp.minimum(1.0, jnp.exp(log_p_accept))
    do_accept = jax.random.bernoulli(key_accept, p_accept)
    sampled_state = _tree_where(do_accept, end_state, integrator_state)
    next_state = HMCState(
        sampled_state.position,
        sampled_state.logdensity,
        sampled_state.logdensity_grad,
    )
    return next_state, p_accept, do_accept, energy_error, jnp.array(0), jnp.array(False)


def _adjusted_mclmc_proposal_integrator_shardmap(
    integrator_step,
    rng_key,
    state,
    step_size,
    num_integration_steps,
    L_proposal_factor=1.25,
):
    key_momentum, key_integrator, key_accept = jax.random.split(rng_key, 3)
    momentum = generate_unit_vector(key_momentum, state.position)
    integrator_state = IntegratorState(
        state.position, momentum, state.logdensity, state.logdensity_grad
    )
    partial_refresh_L = L_proposal_factor * (num_integration_steps * step_size)
    trajectory_keys = jax.random.split(key_integrator, num_integration_steps)

    def body_fn(carry, key):
        current_state, kinetic_sum, invalid_count, invalid_any = carry
        candidate_state, candidate_kinetic = integrator_step(
            current_state, step_size, partial_refresh_L, key
        )
        valid = _integrator_state_is_finite(candidate_state, candidate_kinetic)
        next_state = _tree_where(valid, candidate_state, current_state)
        next_kinetic = jnp.where(valid, candidate_kinetic, jnp.zeros_like(candidate_kinetic))
        next_invalid_count = invalid_count + (~valid).astype(jnp.int32)
        next_invalid_any = invalid_any | (~valid)
        return (
            next_state,
            kinetic_sum + next_kinetic,
            next_invalid_count,
            next_invalid_any,
        ), None

    (end_state, kinetic_sum, invalid_substeps, invalid_trajectory), _ = jax.lax.scan(
        body_fn,
        (
            integrator_state,
            jax.lax.pvary(jnp.zeros_like(state.logdensity), "device"),
            jax.lax.pvary(jnp.zeros_like(state.logdensity, dtype=jnp.int32), "device"),
            jax.lax.pvary(jnp.zeros_like(state.logdensity, dtype=bool), "device"),
        ),
        trajectory_keys,
    )

    energy_error = kinetic_sum - end_state.logdensity + integrator_state.logdensity
    log_p_accept = -energy_error
    log_p_accept = jnp.where(invalid_trajectory, -jnp.inf, log_p_accept)
    log_p_accept = jnp.where(jnp.isnan(log_p_accept), -jnp.inf, log_p_accept)
    p_accept = jnp.minimum(1.0, jnp.exp(log_p_accept))
    do_accept = jax.random.bernoulli(key_accept, p_accept)
    sampled_state = _tree_where(do_accept, end_state, integrator_state)
    next_state = HMCState(
        sampled_state.position,
        sampled_state.logdensity,
        sampled_state.logdensity_grad,
    )
    return (
        next_state,
        p_accept,
        do_accept,
        energy_error,
        invalid_substeps,
        invalid_trajectory,
    )


def full_laps_sharded(
    kernel_builder,
    num_unadjusted_steps,
    num_adjusted_steps,
    num_results,
    state_init,
    init_step_size,
    init_L,
    init_inverse_mass_matrix,
    svi_mean,
    rng_key,
    num_chains,
    C=jnp.float32(0.1),
    alpha=jnp.float32(1.9),
    switch_threshold=jnp.float32(0.01),
    adj_target_accept=jnp.float32(0.70),
    adj_tolerance=jnp.float32(0.03),
    adj_n_steps=15,
    adj_L_proposal_factor=jnp.float32(1.25),
    adapt_mass_matrix=True,
    mass_matrix_trigger_threshold=jnp.float32(0.05),
    mass_matrix_svi_sample_size=jnp.float32(100.0),
    mass_matrix_min_steps=50,
    svi_mass_matrix_weight=None,
    align_initial_momentum=True,
    save_frac=0.2,
    bias_type=3,
    adjusted_logdensity_fn=None,
    adjusted_integrator=None,
):
    del svi_mass_matrix_weight, init_L

    num_devices = len(jax.devices())
    num_chains = (num_chains // num_devices) * num_devices
    if num_chains == 0:
        raise ValueError(f"num_chains must be >= num_devices ({num_devices})")
    chains_per_device = num_chains // num_devices
    dim = state_init.position.shape[-1]
    norm_factor = jnp.sqrt(jnp.asarray(dim, dtype=state_init.position.dtype))
    save_num = int(jnp.rint(save_frac * num_unadjusted_steps))
    save_num = max(save_num, 1)
    total_adjusted_steps = num_adjusted_steps + num_results
    mesh = jax.make_mesh((num_devices,), ("device",))

    if align_initial_momentum:
        state_init = _initialize_like_laps(state_init)

    # For strongly anisotropic posteriors, starting phase 1 from the supplied
    # preconditioner is often much safer than forcing an identity metric.
    phase1_mass_matrix = _require_full_rank_mass_matrix(init_inverse_mass_matrix, dim)
    svi_covariance = phase1_mass_matrix
    welford_start = WelfordAlgorithmState(
        jnp.asarray(svi_mean),
        _welford_seed_m2(svi_covariance, mass_matrix_svi_sample_size),
        jnp.asarray(mass_matrix_svi_sample_size, dtype=phase1_mass_matrix.dtype),
    )
    keys_unadjusted = jax.random.split(rng_key, (num_chains, num_unadjusted_steps))
    keys_unadjusted = jnp.moveaxis(keys_unadjusted, 0, 1)

    unadjusted_carry_out_specs = (
        P("device"),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
    )

    @jax.jit
    @functools.partial(
        _shard_map,
        mesh=mesh,
        in_specs=((P(None, "device"), None), P("device"), None, None),
        out_specs=(unadjusted_carry_out_specs, P("device")),
    )
    def run_unadjusted_sharded(xs, state0, step_size0, phase1_imm):
        n_total = chains_per_device * jax.lax.axis_size("device")

        def step_fn(carry, xs_step):
            rng_keys_batch, step_idx = xs_step
            (
                states,
                step_size,
                running_L,
                current_mass_matrix,
                step_count,
                eevpd,
                eevpd_wanted,
                history_obs,
                history_weights,
                switched,
                switch_index,
                d_tilde,
                r_max,
                r_avg,
                welford_mean,
                welford_m2,
                welford_sample_size,
                mass_matrix_step_count,
                mass_matrix_adapting,
            ) = carry

            active = jnp.logical_not(switched)
            kernel_fn = kernel_builder(current_mass_matrix)

            next_states, infos = jax.vmap(
                lambda key, st: kernel_fn(
                    rng_key=key,
                    state=st,
                    L=running_L,
                    step_size=step_size,
                )
            )(rng_keys_batch, states)

            flat_pos = jax.vmap(lambda x: ravel_pytree(x)[0])(next_states.position)
            flat_grad = jax.vmap(lambda g: ravel_pytree(g)[0])(next_states.logdensity_grad)
            energy = infos.energy_change

            local_sum_x = jnp.sum(flat_pos, axis=0)
            local_sum_xsq = jnp.sum(jnp.square(flat_pos), axis=0)
            local_sum_e = jnp.sum(energy)
            local_sum_esq = jnp.sum(jnp.square(energy))
            local_sum_equi_diag = jnp.sum(-flat_pos * flat_grad, axis=0)

            mean_x = jax.lax.psum(local_sum_x, axis_name="device") / n_total
            mean_xsq = jax.lax.psum(local_sum_xsq, axis_name="device") / n_total
            mean_e = jax.lax.psum(local_sum_e, axis_name="device") / n_total
            mean_esq = jax.lax.psum(local_sum_esq, axis_name="device") / n_total
            mean_equi_diag = (
                jax.lax.psum(local_sum_equi_diag, axis_name="device") / n_total
            )
            deltas = flat_pos - mean_x[jnp.newaxis, :]
            local_m2 = jnp.einsum("ci,cj->ij", deltas, deltas)
            m2_step = jax.lax.psum(local_m2, axis_name="device")

            variances = jnp.maximum(mean_xsq - jnp.square(mean_x), 1e-12)
            new_L = alpha * jnp.sqrt(jnp.mean(variances)) * norm_factor
            observed_eevpd = jnp.maximum((mean_esq - jnp.square(mean_e)) / dim, 1e-12)
            equi_diag_loss = _equipartition_diagonal_loss(mean_equi_diag)

            history_obs_next = _update_history(mean_x, history_obs)
            history_weights_next = _update_history_scalar(1.0, history_weights)
            fluctuations = _contract_history(history_obs_next, history_weights_next)
            equi_full_loss = equi_diag_loss
            if bias_type == 0:
                bias = fluctuations[0]
            elif bias_type == 1:
                bias = fluctuations[1]
            elif bias_type == 2:
                bias = equi_full_loss
            else:
                bias = equi_diag_loss

            target_eevpd = C * jnp.power(jnp.maximum(bias, 1e-12), 3.0 / 8.0)
            eps_factor = jnp.power(target_eevpd / observed_eevpd, 1.0 / 6.0)
            eps_factor = jnp.clip(eps_factor, 0.3, 3.0)
            next_step_size = step_size * eps_factor
            next_step_count = step_count + 1
            continue_unadjusted_base = (fluctuations[0] > switch_threshold) | (
                next_step_count < save_num
            )
            can_start_mass_matrix = next_step_count >= save_num
            mass_matrix_triggered = can_start_mass_matrix & (
                (fluctuations[0] <= mass_matrix_trigger_threshold)
                | (fluctuations[1] <= mass_matrix_trigger_threshold)
            )
            mass_matrix_adapting_next = mass_matrix_adapting | mass_matrix_triggered
            mass_matrix_step_count_next = jnp.where(
                active & adapt_mass_matrix & mass_matrix_adapting_next,
                mass_matrix_step_count + 1,
                mass_matrix_step_count,
            )
            continue_unadjusted = continue_unadjusted_base | (
                adapt_mass_matrix
                & mass_matrix_adapting_next
                & (mass_matrix_step_count_next < mass_matrix_min_steps)
            )
            welford_update = WelfordAlgorithmState(
                mean_x, m2_step, jnp.asarray(n_total, dtype=step_size.dtype)
            )
            next_welford_state = _welford_combine(
                WelfordAlgorithmState(welford_mean, welford_m2, welford_sample_size),
                welford_update,
            )
            empirical_mass_matrix = _stabilize_mass_matrix(
                _covariance_to_mass_matrix(
                    init_inverse_mass_matrix,
                    _welford_covariance(next_welford_state),
                )
            )
            next_mass_matrix = jax.tree.map(
                lambda new, old: jnp.where(
                    active & adapt_mass_matrix & mass_matrix_adapting_next, new, old
                ),
                empirical_mass_matrix,
                current_mass_matrix,
            )
            welford_mean = jnp.where(
                active & adapt_mass_matrix & mass_matrix_adapting_next,
                next_welford_state.mean,
                welford_mean,
            )
            welford_m2 = jnp.where(
                active & adapt_mass_matrix & mass_matrix_adapting_next,
                next_welford_state.m2,
                welford_m2,
            )
            welford_sample_size = jnp.where(
                active & adapt_mass_matrix & mass_matrix_adapting_next,
                next_welford_state.sample_size,
                welford_sample_size,
            )
            switched_next = jnp.logical_or(switched, jnp.logical_not(continue_unadjusted))
            switch_index_next = jnp.where(
                jnp.logical_and(active, jnp.logical_not(continue_unadjusted)),
                step_idx + 1,
                switch_index,
            )

            states = _tree_where(active, next_states, states)
            step_size = jnp.where(active, next_step_size, step_size)
            running_L = jnp.where(active, new_L, running_L)
            current_mass_matrix = next_mass_matrix
            step_count = jnp.where(active, next_step_count, step_count)
            eevpd = jnp.where(active, observed_eevpd, eevpd)
            eevpd_wanted = jnp.where(active, target_eevpd, eevpd_wanted)
            history_obs = _tree_where(active, history_obs_next, history_obs)
            history_weights = _tree_where(active, history_weights_next, history_weights)
            d_tilde = jnp.where(active, equi_diag_loss, d_tilde)
            r_max = jnp.where(active, fluctuations[0], r_max)
            r_avg = jnp.where(active, fluctuations[1], r_avg)
            mass_matrix_adapting = jnp.where(
                active, mass_matrix_adapting_next, mass_matrix_adapting
            )
            switched = switched_next
            mm_min_eig, mm_mean_eig, mm_max_eig = _mass_matrix_eig_summary(
                current_mass_matrix
            )

            local_batch = states.position.shape[0]
            hist = UnadjustedHist(
                position=states.position,
                step_size=_broadcast_local(step_size, local_batch),
                L=_broadcast_local(running_L, local_batch),
                D_tilde=_broadcast_local(d_tilde, local_batch),
                EEVPD=_broadcast_local(eevpd, local_batch),
                EEVPD_wanted=_broadcast_local(eevpd_wanted, local_batch),
                delta_x2=_broadcast_local(r_max, local_batch),
                r_avg=_broadcast_local(r_avg, local_batch),
                mass_matrix_min_eig=_broadcast_local(mm_min_eig, local_batch),
                mass_matrix_mean_eig=_broadcast_local(mm_mean_eig, local_batch),
                mass_matrix_max_eig=_broadcast_local(mm_max_eig, local_batch),
                mass_matrix_adapting=_broadcast_local(mass_matrix_adapting, local_batch),
                active=_broadcast_local(active, local_batch),
            )
            carry = (
                states,
                step_size,
                running_L,
                current_mass_matrix,
                step_count,
                eevpd,
                eevpd_wanted,
                history_obs,
                history_weights,
                switched,
                switch_index_next,
                d_tilde,
                r_max,
                r_avg,
                welford_mean,
                welford_m2,
                welford_sample_size,
                mass_matrix_step_count_next,
                mass_matrix_adapting,
            )
            return carry, hist

        init_carry = (
            state0,
            jnp.asarray(init_step_size),
            jnp.inf,
            phase1_imm,
            jnp.array(0, dtype=jnp.int32),
            jnp.asarray(1e-3, dtype=state0.position.dtype),
            jnp.asarray(1e-3, dtype=state0.position.dtype),
            jnp.zeros((save_num, dim), dtype=state0.position.dtype),
            jnp.zeros((save_num,), dtype=state0.position.dtype),
            jnp.array(False),
            jnp.array(num_unadjusted_steps, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=state0.position.dtype),
            jnp.asarray(jnp.inf, dtype=state0.position.dtype),
            jnp.asarray(jnp.inf, dtype=state0.position.dtype),
            welford_start.mean,
            welford_start.m2,
            welford_start.sample_size,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
        )
        carry, hist = jax.lax.scan(step_fn, init_carry, xs)
        hist = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), hist)
        return carry, hist

    unadjusted_carry, unadjusted_hist = run_unadjusted_sharded(
        (keys_unadjusted, jnp.arange(num_unadjusted_steps, dtype=jnp.int32)),
        state_init,
        jnp.asarray(init_step_size),
        phase1_mass_matrix,
    )

    (
        unadjusted_state,
        unadjusted_step_size,
        unadjusted_L,
        inverse_mass_matrix_after_unadjusted,
        _,
        _,
        _,
        _,
        _,
        switched,
        switch_index,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = unadjusted_carry

    adjusted_step_size_init = unadjusted_step_size

    adjusted_state_init = HMCState(
        unadjusted_state.position,
        unadjusted_state.logdensity,
        unadjusted_state.logdensity_grad,
    )

    keys_adjusted = jax.random.split(
        jax.random.fold_in(rng_key, 1), (num_chains, total_adjusted_steps)
    )
    keys_adjusted = jnp.moveaxis(keys_adjusted, 0, 1)

    adjusted_carry_out_specs = (
        P("device"),
        P(),
        P(),
        P(),
        P(),
        P(),
        P("device"),
    )

    @jax.jit
    @functools.partial(
        _shard_map,
        mesh=mesh,
        in_specs=((P(None, "device"), None), P("device"), None, None),
        out_specs=(adjusted_carry_out_specs, P("device")),
    )
    def run_adjusted_sharded(xs, state0, step_size0, imm0):
        n_total = chains_per_device * jax.lax.axis_size("device")
        base_kernel = kernel_builder(imm0)
        mm_min_eig, mm_mean_eig, mm_max_eig = _mass_matrix_eig_summary(imm0)
        raw_integrator_step = None
        if adjusted_logdensity_fn is not None and adjusted_integrator is not None:
            raw_integrator_step = with_isokinetic_maruyama(
                adjusted_integrator(
                    logdensity_fn=adjusted_logdensity_fn,
                    inverse_mass_matrix=imm0,
                )
            )

        def step_fn(carry, xs_step):
            rng_keys_batch, step_idx = xs_step
            (
                states,
                step_size,
                bisection_state,
                iteration,
                imm,
                terminated,
                acceptance_sum,
            ) = carry

            if raw_integrator_step is not None:
                (
                    next_states,
                    acceptances,
                    accepted,
                    energy_error,
                    invalid_substeps,
                    invalid_trajectory,
                ) = jax.vmap(
                    lambda key, st: _adjusted_mclmc_proposal_integrator_shardmap(
                        integrator_step=raw_integrator_step,
                        rng_key=key,
                        state=st,
                        step_size=step_size,
                        num_integration_steps=adj_n_steps,
                        L_proposal_factor=adj_L_proposal_factor,
                    )
                )(rng_keys_batch, states)
            else:
                (
                    next_states,
                    acceptances,
                    accepted,
                    energy_error,
                    invalid_substeps,
                    invalid_trajectory,
                ) = jax.vmap(
                    lambda key, st: _adjusted_mclmc_proposal_shardmap(
                        base_kernel=base_kernel,
                        rng_key=key,
                        state=st,
                        step_size=step_size,
                        num_integration_steps=adj_n_steps,
                        L_proposal_factor=adj_L_proposal_factor,
                    )
                )(rng_keys_batch, states)

            local_acceptance_sum = jnp.sum(acceptances)
            mean_acceptance = jax.lax.psum(local_acceptance_sum, axis_name="device") / n_total
            do_adapt = iteration < num_adjusted_steps
            bisection_state_next, proposed_step_size = _bisection_monotonic_update(
                bisection_state,
                step_size,
                mean_acceptance,
                adj_target_accept,
                tolerance=adj_tolerance,
            )
            terminated_next = jnp.where(do_adapt, bisection_state_next[1], terminated)
            step_size_next = jnp.where(do_adapt, proposed_step_size, step_size)
            bisection_state = jax.tree.map(
                lambda new, old: jnp.where(do_adapt, new, old),
                bisection_state_next,
                bisection_state,
            )
            acceptance_sum_next = acceptance_sum + acceptances
            acceptance_running_mean = acceptance_sum_next / (iteration + 1)

            local_batch = states.position.shape[0]
            hist = AdjustedHist(
                position=next_states.position,
                step_size=_broadcast_local(step_size, local_batch),
                acceptance=acceptances,
                acceptance_running_mean=acceptance_running_mean,
                accepted=accepted,
                energy_error=energy_error,
                logdensity=next_states.logdensity,
                invalid_substeps=invalid_substeps,
                invalid_trajectory=invalid_trajectory,
                mass_matrix_min_eig=_broadcast_local(mm_min_eig, local_batch),
                mass_matrix_mean_eig=_broadcast_local(mm_mean_eig, local_batch),
                mass_matrix_max_eig=_broadcast_local(mm_max_eig, local_batch),
                mass_matrix_adapting=_broadcast_local(False, local_batch),
                adapting=_broadcast_local(do_adapt & (~terminated), local_batch),
            )
            carry = (
                next_states,
                step_size_next,
                bisection_state,
                iteration + 1,
                imm,
                terminated_next,
                acceptance_sum_next,
            )
            return carry, hist

        init_carry = (
            state0,
            step_size0,
            (jnp.array([-jnp.inf, jnp.inf]), jnp.array(False)),
            jnp.array(0, dtype=jnp.int32),
            imm0,
            jnp.array(False),
            jax.lax.pvary(
                jnp.zeros((chains_per_device,), dtype=state0.logdensity.dtype),
                "device",
            ),
        )
        carry, hist = jax.lax.scan(step_fn, init_carry, xs)
        hist = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), hist)
        return carry, hist

    adjusted_carry, adjusted_hist = run_adjusted_sharded(
        (keys_adjusted, jnp.arange(total_adjusted_steps, dtype=jnp.int32)),
        adjusted_state_init,
        adjusted_step_size_init,
        inverse_mass_matrix_after_unadjusted,
    )

    adjusted_state, final_step_size, _, _, final_inverse_mass_matrix, _, _ = adjusted_carry
    samples = adjusted_hist._replace(position=adjusted_hist.position[:, -num_results:, :])

    carry = LAPSCarry(
        unadjusted_state=unadjusted_state,
        adjusted_state=adjusted_state,
        step_size=final_step_size,
        L=unadjusted_L,
        inverse_mass_matrix=final_inverse_mass_matrix,
        switch_index=switch_index,
        switched=switched,
    )
    return (unadjusted_hist, adjusted_hist, samples), carry


def LAPS_JIT(
    model_seq,
    qz,
    n_hmc=128,
    num_unadjusted_steps=1000,
    num_adjusted_steps=500,
    num_results=2000,
    init_L=None,
    init_step_size=None,
    init_inverse_mass_matrix=None,
    C=jnp.float32(0.1),
    alpha=jnp.float32(1.9),
    switch_threshold=jnp.float32(0.01),
    adj_target_accept=None,
    adj_tolerance=jnp.float32(0.03),
    adj_n_steps=15,
    adj_L_proposal_factor=jnp.float32(1.25),
    adapt_mass_matrix=True,
    mass_matrix_trigger_threshold=jnp.float32(0.05),
    mass_matrix_svi_sample_size=jnp.float32(100.0),
    mass_matrix_min_steps=50,
    seed=0,
    debug_output=False,
    progress_bar=False,
    use_shard_map=True,
):
    """GIGALens-facing wrapper for shard-mapped LAPS.

    Defaults are chosen to stay close to the BlackJAX LAPS reference where
    possible, while retaining full-rank mass-matrix support and adaptation.

    `init_inverse_mass_matrix` is always treated as a full-rank inverse mass
    matrix and must therefore be a `(dim, dim)` array.
    """
    raise NotImplementedError(
        "LAPS_JIT was not migrated to the scene API; it used the legacy LensSimulator "
        "(and an `alternate_inference` package not present here), removed with the old "
        "gigalens API. Restore from git b82397c if needed.")

    del progress_bar

    if not use_shard_map:
        raise ValueError("LAPS_JIT currently supports only the shard-mapped path.")

    import time

    import gigalens.jax.simulator as sim
    from alternate_inference.mclmc_alt import _build_kernel_shardmap, init_multi

    lens_sim = sim.LensSimulator(
        model_seq.phys_model,
        model_seq.sim_config,
        bs=1,
    )

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    rng_key = jax.random.key(seed)
    init_key, run_key = jax.random.split(rng_key, 2)

    n_chains = n_hmc
    positions = qz.sample((n_chains,), seed=init_key)
    state_multi = init_multi(positions, init_key, log_prob)
    dim = state_multi.position.shape[-1]

    init_L = jnp.sqrt(dim) if init_L is None else init_L
    init_step_size = (
        jnp.sqrt(dim) * jnp.float32(0.01)
        if init_step_size is None
        else init_step_size
    )
    if init_inverse_mass_matrix is None:
        init_inverse_mass_matrix = qz.covariance()
    init_inverse_mass_matrix = _require_full_rank_mass_matrix(
        init_inverse_mass_matrix, dim
    )

    adjusted_integrator = (
        isokinetic_omelyan_fullrank if dim > 200 else isokinetic_mclachlan_fullrank
    )
    if adj_target_accept is None:
        adj_target_accept = jnp.float32(0.9 if dim > 200 else 0.7)

    kernel_builder = lambda inverse_mass_matrix: _build_kernel_shardmap(
        logdensity_fn=log_prob,
        inverse_mass_matrix=inverse_mass_matrix,
        integrator=isokinetic_velocity_verlet_fullrank,
    )

    starttime = time.perf_counter()
    debug_hist, carry = full_laps_sharded(
        kernel_builder=kernel_builder,
        num_unadjusted_steps=num_unadjusted_steps,
        num_adjusted_steps=num_adjusted_steps,
        num_results=num_results,
        state_init=state_multi,
        init_step_size=init_step_size,
        init_L=init_L,
        init_inverse_mass_matrix=init_inverse_mass_matrix,
        svi_mean=qz.mean(),
        rng_key=run_key,
        num_chains=n_chains,
        C=C,
        alpha=alpha,
        switch_threshold=switch_threshold,
        adj_target_accept=adj_target_accept,
        adj_tolerance=adj_tolerance,
        adj_n_steps=adj_n_steps,
        adj_L_proposal_factor=adj_L_proposal_factor,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_trigger_threshold=mass_matrix_trigger_threshold,
        mass_matrix_svi_sample_size=mass_matrix_svi_sample_size,
        mass_matrix_min_steps=mass_matrix_min_steps,
        adjusted_logdensity_fn=log_prob,
        adjusted_integrator=adjusted_integrator,
    )
    total_time = time.perf_counter() - starttime
    print(f"Sampling took {total_time} s")

    if debug_output:
        return debug_hist, carry

    _, _, samples = debug_hist
    return samples.position


def plot_laps_diagnostics(
    debug_hist,
    carry=None,
    chain_idx=0,
    smooth_kernel_size=20,
    figsize=(10, 12),
):
    """Plot phase-wise LAPS diagnostics from `debug_output=True`.

    Parameters
    ----------
    debug_hist
        Tuple `(unadjusted_hist, adjusted_hist, samples)` returned by
        `LAPS_JIT(..., debug_output=True)` or `full_laps_sharded(...)`.
    carry
        Optional `LAPSCarry` returned alongside `debug_hist`. If provided, the
        detected switch step is drawn as a vertical line.
    chain_idx
        Chain index used for the per-chain overlays.
    smooth_kernel_size
        Window size for a simple moving-average smoothing on selected traces.
    figsize
        Matplotlib figure size.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    unadjusted_hist, adjusted_hist, _ = debug_hist

    to_np = lambda x: np.asarray(jax.device_get(x))

    unadj_step = to_np(unadjusted_hist.step_size)
    unadj_L = to_np(unadjusted_hist.L)
    unadj_D = to_np(unadjusted_hist.D_tilde)
    unadj_r_max = to_np(unadjusted_hist.delta_x2)
    unadj_r_avg = to_np(unadjusted_hist.r_avg)
    unadj_eevpd = to_np(unadjusted_hist.EEVPD)
    unadj_eevpd_wanted = to_np(unadjusted_hist.EEVPD_wanted)
    unadj_mm_min = to_np(unadjusted_hist.mass_matrix_min_eig)
    unadj_mm_mean = to_np(unadjusted_hist.mass_matrix_mean_eig)
    unadj_mm_max = to_np(unadjusted_hist.mass_matrix_max_eig)
    unadj_mm_adapting = to_np(unadjusted_hist.mass_matrix_adapting).astype(float)
    unadj_active = to_np(unadjusted_hist.active).astype(float)

    adj_step = to_np(adjusted_hist.step_size)
    adj_acc = to_np(adjusted_hist.acceptance)
    adj_acc_running = to_np(adjusted_hist.acceptance_running_mean)
    adj_logdensity = to_np(adjusted_hist.logdensity)
    adj_energy = to_np(adjusted_hist.energy_error)
    adj_invalid = to_np(adjusted_hist.invalid_substeps)
    adj_invalid_traj = to_np(adjusted_hist.invalid_trajectory).astype(float)
    adj_mm_min = to_np(adjusted_hist.mass_matrix_min_eig)
    adj_mm_mean = to_np(adjusted_hist.mass_matrix_mean_eig)
    adj_mm_max = to_np(adjusted_hist.mass_matrix_max_eig)
    adj_adapting = to_np(adjusted_hist.adapting).astype(float)

    acc_mean = np.mean(adj_acc, axis=0)
    acc_q10 = np.quantile(adj_acc, 0.10, axis=0)
    acc_q90 = np.quantile(adj_acc, 0.90, axis=0)
    acc_running_mean = np.mean(adj_acc_running, axis=0)

    logdensity_mean = np.mean(adj_logdensity, axis=0)
    logdensity_q10 = np.quantile(adj_logdensity, 0.10, axis=0)
    logdensity_q90 = np.quantile(adj_logdensity, 0.90, axis=0)

    energy_cumsum = np.cumsum(adj_energy, axis=1)
    energy_cumsum_mean = np.mean(energy_cumsum, axis=0)
    energy_cumsum_q10 = np.quantile(energy_cumsum, 0.10, axis=0)
    energy_cumsum_q90 = np.quantile(energy_cumsum, 0.90, axis=0)

    invalid_mean = np.mean(adj_invalid, axis=0)
    invalid_any_rate = np.mean(adj_invalid_traj, axis=0)

    chain_idx = int(np.clip(chain_idx, 0, unadj_step.shape[0] - 1))
    num_unadjusted_steps = unadj_step.shape[1]
    total_adjusted_steps = adj_step.shape[1]
    num_adjusted_steps = int(np.sum(adj_adapting[chain_idx]))

    step_unadj = np.arange(num_unadjusted_steps)
    step_adj = np.arange(num_unadjusted_steps, num_unadjusted_steps + total_adjusted_steps)
    kernel = np.ones(smooth_kernel_size, dtype=float) / max(smooth_kernel_size, 1)

    fig, axs = plt.subplots(7, 1, sharex=True, figsize=figsize)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axs

    ax1.plot(step_unadj, unadj_step.T, alpha=0.25, color="tab:blue")
    ax1.plot(step_adj, adj_step.T, alpha=0.25, color="tab:orange")
    ax1.plot(
        step_unadj,
        np.convolve(unadj_step[chain_idx], kernel, mode="same"),
        color="tab:blue",
        lw=2,
        label="Unadjusted",
    )
    ax1.plot(
        step_adj,
        np.convolve(adj_step[chain_idx], kernel, mode="same"),
        color="tab:orange",
        lw=2,
        label="Adjusted",
    )
    ax1.set_title("Chain-Wise Step Size")
    ax1.set_ylabel("Step Size")
    ax1.set_yscale('log')
    ax1.legend(loc="best")

    ax2.plot(step_unadj, unadj_L.T, alpha=0.25, color="tab:green")
    ax2.plot(step_unadj, unadj_L[chain_idx], color="tab:green", lw=2)
    if carry is not None:
        final_L = float(np.asarray(jax.device_get(carry.L)))
        ax2.plot(
            step_adj,
            final_L * np.ones_like(step_adj),
            color="tab:orange",
            lw=2,
        )
    ax2.set_title("Chain-Wise L")
    ax2.set_ylabel("L")

    ax3.plot(step_unadj, unadj_D[chain_idx], color="black", label=r"$\tilde{D}$")
    ax3.plot(step_unadj, unadj_r_max[chain_idx], color="tab:red", label=r"$r_{\max}$")
    ax3.plot(step_unadj, unadj_r_avg[chain_idx], color="tab:blue", label=r"$r_{\mathrm{avg}}$")
    ax3.set_yscale("log")
    ax3.set_title("Unadjusted Bias / Contraction Diagnostics")
    ax3.set_ylabel("Diagnostic")
    ax3.legend(loc="best")

    ax4.plot(step_unadj, unadj_eevpd[chain_idx], color="tab:orange", label="EEVPD")
    ax4.plot(
        step_unadj,
        unadj_eevpd_wanted[chain_idx],
        color="black",
        alpha=0.7,
        label="EEVPD target",
    )
    ax4.set_yscale("log")
    ax4.set_ylabel("EEVPD")
    ax4.set_title("Phase-Specific Hyperparameter Targets")
    ax4.legend(loc="upper left")

    ax4b = ax4.twinx()
    ax4b.plot(step_adj, adj_acc[chain_idx], color="tab:purple", alpha=0.6, label="Acceptance")
    ax4b.plot(
        step_adj,
        np.convolve(adj_acc[chain_idx], kernel, mode="same"),
        color="tab:purple",
        lw=2,
    )
    ax4b.fill_between(
        step_adj,
        acc_q10,
        acc_q90,
        color="tab:purple",
        alpha=0.15,
    )
    ax4b.plot(
        step_adj,
        acc_mean,
        color="tab:purple",
        lw=1.5,
        linestyle="--",
        alpha=0.9,
    )
    ax4b.plot(
        step_adj,
        acc_running_mean,
        color="tab:pink",
        lw=1.5,
        alpha=0.9,
    )
    ax4b.set_ylim(0.0, 1.05)
    ax4b.set_ylabel("Acceptance")

    ax5.plot(
        step_adj,
        adj_logdensity[chain_idx],
        color="tab:brown",
        lw=1.0,
        alpha=0.5,
        label="Adjusted logdensity (chain)",
    )
    ax5.fill_between(
        step_adj,
        logdensity_q10,
        logdensity_q90,
        color="tab:brown",
        alpha=0.15,
    )
    ax5.plot(
        step_adj,
        logdensity_mean,
        color="tab:brown",
        lw=2.0,
        linestyle="--",
        label="Adjusted logdensity (ensemble mean)",
    )
    ax5.set_ylabel("Logdensity")
    ax5.set_title("Adjusted Trajectory Diagnostics")
    ax5.legend(loc="upper left")

    ax5b = ax5.twinx()
    ax5b.fill_between(
        step_adj,
        energy_cumsum_q10,
        energy_cumsum_q90,
        color="tab:gray",
        alpha=0.12,
    )
    ax5b.plot(
        step_adj,
        np.cumsum(adj_energy[chain_idx]),
        color="tab:gray",
        alpha=0.6,
        label="Cumulative energy error (chain)",
    )
    ax5b.plot(
        step_adj,
        energy_cumsum_mean,
        color="tab:gray",
        lw=2.0,
        linestyle="--",
        alpha=0.9,
        label="Cumulative energy error (ensemble mean)",
    )
    ax5b.set_ylabel("Cum. dE")

    ax6.plot(step_unadj, unadj_mm_min[chain_idx], color="tab:blue", label="min eig")
    ax6.plot(step_unadj, unadj_mm_mean[chain_idx], color="black", label="mean eig")
    ax6.plot(step_unadj, unadj_mm_max[chain_idx], color="tab:red", label="max eig")
    ax6.plot(step_adj, adj_mm_min[chain_idx], color="tab:blue", linestyle="--", alpha=0.8)
    ax6.plot(step_adj, adj_mm_mean[chain_idx], color="black", linestyle="--", alpha=0.8)
    ax6.plot(step_adj, adj_mm_max[chain_idx], color="tab:red", linestyle="--", alpha=0.8)
    ax6.set_yscale("log")
    ax6.set_ylabel("Eigenvalue")
    ax6.set_title("Inverse Mass Matrix Eigenvalue Summary")
    ax6.legend(loc="best")

    ax6b = ax6.twinx()
    ax6b.plot(
        step_unadj,
        unadj_mm_adapting[chain_idx],
        color="tab:green",
        alpha=0.5,
        lw=1.5,
        label="Metric adapting",
    )
    ax6b.set_ylim(-0.05, 1.05)
    ax6b.set_ylabel("Metric adapt on/off")

    invalid_map = np.concatenate(
        [np.zeros_like(unadj_active), adj_invalid.astype(float)],
        axis=1,
    )
    ax7.imshow(
        invalid_map,
        aspect="auto",
        interpolation="none",
        cmap="magma",
        vmin=0.0,
        vmax=max(1.0, float(np.max(invalid_map))),
    )
    ax7.set_title("Invalid Substeps Per Proposal")
    ax7.set_ylabel("Chain")
    ax7.set_xlabel("Step")

    ax7b = ax7.twinx()
    ax7b.plot(
        step_adj,
        invalid_mean,
        color="tab:orange",
        lw=1.5,
        label="Mean invalid substeps",
    )
    ax7b.plot(
        step_adj,
        invalid_any_rate,
        color="tab:red",
        lw=1.5,
        linestyle="--",
        label="Frac. invalid trajectories",
    )
    ax7b.set_ylabel("Invalidity summary")
    ax7b.legend(loc="upper right")

    activity_map = np.concatenate([unadj_active, adj_adapting + adj_invalid_traj], axis=1)
    ax7c = ax7.twinx()
    ax7c.imshow(
        activity_map,
        aspect="auto",
        interpolation="none",
        cmap="Greens",
        vmin=0.0,
        vmax=2.0,
        alpha=0.15,
    )
    ax7c.set_yticks([])

    switch_step = None
    if carry is not None:
        switch_step = int(np.asarray(jax.device_get(carry.switch_index)))

    phase2_adapt_end = num_unadjusted_steps + num_adjusted_steps
    for ax in axs:
        ax.axvline(num_unadjusted_steps, color="tab:orange", linestyle="--", alpha=0.8)
        ax.axvline(phase2_adapt_end, color="tab:green", linestyle="--", alpha=0.8)
        if switch_step is not None:
            ax.axvline(switch_step, color="tab:red", linestyle="--", alpha=0.8)

    fig.tight_layout()
    return fig, axs
