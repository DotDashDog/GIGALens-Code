import jax.numpy as jnp
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import io_callback
from fastprogress.fastprogress import progress_bar as fastprogress_bar

try:
    _shard_map = jax.shard_map  # type: ignore[attr-defined]
except AttributeError:
    from jax.experimental.shard_map import shard_map as _shard_map

# MAMS = Metropolis-Adjusted Microcanonical Sampler (Robnik, Cohn-Gordon, Seljak).
# This module is the adjusted sibling of mclmc.py: it shares MCLMC's microcanonical
# (isokinetic) dynamics, dense windowed mass-matrix adaptation, SVI-seeded prior, and
# sharded multi-chain scaffolding, but (1) fully refreshes the momentum once per
# transition and integrates a DETERMINISTIC trajectory of `n` isokinetic steps,
# (2) adds a Metropolis-Hastings accept/reject (so samples are asymptotically
# unbiased), and (3) tunes the step size by dual averaging to a target acceptance
# rate instead of the MCLMC energy-variance setpoint.
#
# Variant implemented: pure Hamiltonian (no Langevin partial refreshment) with a
# DETERMINISTIC, SHARED Halton trajectory length -- n_k is computed once per step
# from the cross-chain-mean step size and the global L, so every chain integrates
# the same number of steps and the batch stays perfectly uniform under shard_map.
# Per the paper, dropping Langevin makes little difference on their benchmarks, and
# the Halton (vs i.i.d.-uniform) schedule preserves the anti-cycling benefit of
# randomized trajectory lengths while remaining shared across chains.

from .blackjax_updated_utils import *
from .blackjax_updated_utils import (
    _build_adjusted_kernel_shardmap,
    _ess_shardmap,
    _gen_scan_fn_one_bar,
    AdjustedKernelExtras,
)

from blackjax.adaptation.step_size import (
    dual_averaging_adaptation,
    DualAveragingAdaptationState,
)
from blackjax.mcmc.dynamic_hmc import halton_trajectory_length

import functools
from typing import Callable, NamedTuple, Optional
from collections import namedtuple

import time
from threading import Lock


def MAMS_JIT(model_seq, qz, n_hmc=16, num_burnin_steps=1000, num_results=2000,
          target_acceptance=0.9, init_L=None, init_step_size=None,
          frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
          progress_bar=False, seed=0, debug_output=False, regularize_mass_matrix=False):
    # Scene-only: the ProbModel owns batch-flexible per-dataset SceneSimulators, so
    # log_prob(z) renders through them directly -- no separately-built simulator.
    def log_prob(z):
        return model_seq.prob_model.log_prob(z)[0]

    n_chains = n_hmc
    integrator = isokinetic_mclachlan_smart

    build_kernel_fn = _build_adjusted_kernel_shardmap
    kernel = lambda inverse_mass_matrix : build_kernel_fn(
        logdensity_fn=log_prob,
        integrator=integrator,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    rng_key = jax.random.key(seed)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)

    state_multi = init_multi(qz.sample((n_chains,), seed=init_key), init_key, log_prob)
    dim=state_multi.position.shape[-1]

    init_L = jnp.sqrt(dim) if init_L is None else init_L
    init_step_size = (jnp.sqrt(dim) * 0.25) if init_step_size is None else init_step_size
    starting_adapt_state = blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState(
        L=init_L, step_size=init_step_size, inverse_mass_matrix=qz.covariance()
    )

    adapt_fn = full_mams_with_adapt_sharded

    starttime = time.perf_counter()
    debug_hist, params = adapt_fn(
        kernel=kernel,
        num_burnin_steps=num_burnin_steps,
        num_results=num_results,
        state_init=state_multi,
        params_init=starting_adapt_state,
        svi_mean=qz.mean(),
        rng_key=tune_key,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        frac_tune3=frac_tune3,
        target_acceptance=target_acceptance,
        num_chains=n_chains,
        num_effective_samples=100,
        svi_mass_matrix_weight=10.0 * n_chains,
        windowed_mass_matrix=True,
        regularize_mass_matrix=regularize_mass_matrix,
        progress_bar=progress_bar,
    )

    total_time = time.perf_counter()-starttime
    print(f"Sampling took {total_time} s")
    if debug_output:
        return debug_hist
    else:
        all_samples=debug_hist.position
        result_samples = all_samples[:, -num_results:, :]
        return result_samples


def full_mams_with_adapt_sharded(
    kernel,
    num_burnin_steps,
    num_results,
    state_init,
    params_init,
    svi_mean,
    rng_key,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    target_acceptance=0.9,
    num_effective_samples=150,
    Lfactor=0.3,
    L_max_ratio=2.0,
    num_chains=8,
    svi_mass_matrix_weight=20.,
    windowed_mass_matrix=True,
    regularize_mass_matrix=False,
    progress_bar=False,
):
    """Sharded MAMS tuning + sampling, mirroring `full_mclmc_with_adapt_sharded`.

    Same multi-chain shard_map scaffolding, expanding-window dense mass-matrix
    adaptation, SVI-seeded Welford prior, and single-scan mode schedule as MCLMC.
    The three MAMS-specific changes:

      * the kernel runs a deterministic `n_k`-step trajectory + MH accept/reject
        (`_build_adjusted_kernel_shardmap`) instead of a single Langevin step;
      * the step size is tuned by dual averaging to `target_acceptance` (the paper
        uses 0.9 for the 2nd-order McLachlan integrator) instead of the MCLMC
        energy-variance controller;
      * the trajectory length L is tuned via the ALBA / ESS relation
        L_new = Lfactor * L * num_steps3 / ESS  (Lfactor=0.3 for the no-Langevin
        variant), the adjusted analog of MCLMC's L = Lfactor * num_steps3 * eps / ESS.
        The inter-sample interval here is one trajectory (~ L in dynamics time),
        not one step (eps), which is why L (not eps) appears on the RHS.

    n_k (number of integration steps) is a SHARED Halton draw computed once per step
    from the cross-chain-mean step size and the global L, so the fori_loop trip count
    is identical across all chains -- no ragged trajectories under shard_map.
    """
    num_devices = len(jax.devices())
    num_chains = (num_chains // num_devices) * num_devices
    if num_chains == 0:
        raise ValueError(f"num_chains must be >= num_devices ({num_devices})")
    chains_per_device = num_chains // num_devices

    dim = state_init.position.shape[-1]
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    welford_init_fn, _, welford_cov = welford_algorithm(is_diagonal_matrix=False)

    # Dual averaging controller for the step size (targets `target_acceptance`).
    da_init, da_update, da_final = dual_averaging_adaptation(target=target_acceptance)

    # F4 (opt-in): Stan-style regularization of every window's sample covariance
    # before it is installed as the inverse mass matrix. Identical to MCLMC.
    def _regularize_cov(cov, n):
        if not regularize_mass_matrix:
            return cov
        cov = 0.5 * (cov + jnp.swapaxes(cov, -1, -2))           # symmetrize
        n = jnp.asarray(n, cov.dtype)
        eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
        shrink = 1e-3 * (5.0 / (n + 5.0))
        reg = (n / (n + 5.0)) * cov + shrink * eye              # Stan window shrinkage
        w, V = jnp.linalg.eigh(reg)                             # PSD floor
        w = jnp.clip(w, shrink, None)
        return (V * w[..., jnp.newaxis, :]) @ jnp.swapaxes(V, -1, -2)

    # Single-dtype sampler: the energy/log-density dtype drives EVERYTHING. Cast the
    # whole initial state and all adaptation params to it (see MCLMC for rationale).
    _canon = jnp.asarray(state_init.logdensity).dtype
    _cast_float = lambda a: (
        jnp.asarray(a).astype(_canon)
        if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating)
        else jnp.asarray(a)
    )
    state_init = jax.tree_util.tree_map(_cast_float, state_init)
    svi_mean = jnp.asarray(svi_mean).astype(_canon)
    params_init = params_init._replace(
        inverse_mass_matrix=jnp.asarray(params_init.inverse_mass_matrix).astype(_canon),
        step_size=jnp.asarray(params_init.step_size).astype(_canon),
        L=jnp.asarray(params_init.L).astype(_canon),
    )

    svi_inverse_mass_matrix = params_init.inverse_mass_matrix

    total_steps = num_burnin_steps + num_results
    num_steps1, num_steps2, num_steps3 = round(num_burnin_steps * frac_tune1), round(num_burnin_steps * frac_tune2), round(num_burnin_steps * frac_tune3)
    tuning_steps = num_steps1 + num_steps2 + num_steps3

    step_size_sync_step = num_steps1 + num_steps2
    L_adaptation_step = tuning_steps

    # --- Per-chain step-size adaptation via dual averaging (replaces MCLMC's
    #     energy-variance controller). Carried adaptive_state = (da_state, step_size_max). ---

    def step_size_adapt(previous_state, next_state, info, params, adaptive_state, nan_key):
        da_state, step_size_max = adaptive_state
        # NaN guard (blackjax MCLMC-adaptation handle_nans): reverts to old state and
        # shrinks step_size_max by 0.8 on a non-finite proposal. The MH step already
        # rejects divergences, so on a clean reject `next_state` is finite and this is
        # a no-op; dual averaging shrinks the step size when acceptance drops.
        success, state, step_size_max, _ = handle_nans(
            previous_state, next_state, params.step_size, step_size_max,
            info.energy_change, nan_key,
        )
        da_state = da_update(da_state, info.acceptance_rate)
        step_size = jnp.exp(da_state.log_step_size)
        step_size = jnp.minimum(step_size, step_size_max)
        da_state = da_state._replace(log_step_size=jnp.log(step_size))
        adaptive_state = (da_state, step_size_max)
        return state, params._replace(step_size=step_size), adaptive_state, success, info.acceptance_rate

    # --- Windowed mass matrix setup (STAN-style expanding windows) -- identical to MCLMC. ---
    if windowed_mass_matrix:
        n_windows = 3
        num_mm_steps = round(0.67 * num_steps2)
        mm_start = num_steps1
        ratios = [2**k for k in range(n_windows)]
        total_ratio = sum(ratios)
        w_sizes = [max(1, round(num_mm_steps * r / total_ratio)) for r in ratios]
        w_sizes[-1] = num_mm_steps - sum(w_sizes[:-1])
        _pos, window_ends = mm_start, []
        for _ws in w_sizes:
            _pos += _ws
            window_ends.append(_pos - 1)
        _mask = [False] * total_steps
        for _we in window_ends:
            if 0 <= _we < total_steps:
                _mask[_we] = True
        window_end_mask = jnp.array(_mask)
        welford_empty = WelfordAlgorithmState(
            jnp.zeros(dim), jnp.zeros((dim, dim)), jnp.array(0.0))

        def _make_adapt_reset(cur):
            # cur = (da_state, step_size_max); reset DA, keep step_size_max.
            da_state, step_size_max = cur
            return (da_init(jnp.exp(da_state.log_step_size)), step_size_max)

    # --- Batched scan body: explicit batch dim, collectives only use 'device'. ---

    Hist = namedtuple("hist", [
        "position", "step_size", "L", "inverse_mass_matrix", "nonan", "acceptance_rate",
        "is_accepted", "num_integration_steps", "energy_change_raw", "kernel_nonan", "step_norm",
    ])

    l_buffer_start = L_adaptation_step - num_steps3

    def step_batched(carry, mode_and_key):
        with jax.named_scope("mams_step_batched"):
            i, mode, rng_keys_batch = mode_and_key

            states, params, step_sizes, adapt_states, welford_state, l_stage_bufs = carry

            do_ssa = jnp.logical_or(mode == 1, mode == 2)
            do_mm_adapt = mode == 2

            key_pairs = jax.vmap(jax.random.split)(rng_keys_batch)
            chain_keys = key_pairs[:, 0]
            nan_keys = key_pairs[:, 1]

            kernel_fn = kernel(params.inverse_mass_matrix)

            # --- Shared (cross-chain) Halton trajectory length for this step. ---
            # avg_n = L / mean(step_size); identical across chains -> uniform fori_loop.
            with jax.named_scope("trajectory_length"):
                local_ss_sum = jnp.sum(step_sizes)
                global_ss_sum = jax.lax.psum(local_ss_sum, axis_name='device')
                mean_ss = global_ss_sum / (chains_per_device * jax.lax.axis_size('device'))
                avg_n = jnp.maximum(params.L / mean_ss, 1.0)
                num_integration_steps = jnp.maximum(
                    halton_trajectory_length(i, avg_n), 1
                ).astype(jnp.int32)

            def per_chain(prev_state, rng_key, nan_key, step_size, adapt_state):
                with jax.named_scope("per_chain_kernel"):
                    new_state, info = kernel_fn(
                        rng_key=rng_key, state=prev_state, step_size=step_size,
                        num_integration_steps=num_integration_steps)

                    energy_change_raw = info.extras.energy_change_raw
                    kernel_nonan_flag = info.extras.kernel_nonan
                    acceptance_rate = info.acceptance_rate
                    is_accepted = info.is_accepted
                    pos_diff = new_state.position.reshape(-1) - prev_state.position.reshape(-1)
                    step_norm_val = jnp.sqrt(jnp.sum(pos_diff ** 2))

                def adapt_one(_):
                    with jax.named_scope("step_size_adapt"):
                        pseudo_params = params._replace(step_size=step_size)
                        a_state, a_params, a_adapt, a_success, a_acc = step_size_adapt(
                            prev_state, new_state, info, pseudo_params, adapt_state, nan_key
                        )
                        return (
                            a_state,
                            a_params.step_size,
                            a_adapt,
                            a_success,
                            a_acc,
                        )

                def skip_adapt(_):
                    success_placeholder = jnp.isfinite(new_state.position.reshape(-1)[0])
                    return (
                        new_state,
                        step_size,
                        adapt_state,
                        success_placeholder,
                        acceptance_rate,
                    )

                result = jax.lax.cond(
                    do_ssa,
                    adapt_one,
                    skip_adapt,
                    operand=None,
                )
                return result + (is_accepted, energy_change_raw, kernel_nonan_flag, step_norm_val)

            (new_states, new_step_sizes, new_adapt_states, successes, accs,
             is_accepteds, energy_changes_raw, kernel_nonans, step_norms) = jax.vmap(per_chain)(
                states, chain_keys, nan_keys, step_sizes, adapt_states
            )

            with jax.named_scope("history_write"):
                def write_l_stage_buffer(buf):
                    buf_index = i - l_buffer_start
                    return buf.at[:, buf_index].set(new_states.position)

                l_stage_bufs = jax.lax.cond(
                    mode == 3,
                    write_l_stage_buffer,
                    lambda buf: buf,
                    l_stage_bufs,
                )

            _sel = lambda c, a, b: jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)

            # Cross-chain mass matrix adaptation (jnp locally, psum across devices) -- identical to MCLMC.
            with jax.named_scope("mass_matrix_adapt"):
                def run_mass_matrix_adapt(_):
                    xs_pos = jax.vmap(lambda s: ravel_pytree(s.position)[0])(new_states)
                    n_dev = jax.lax.axis_size('device')
                    n_total = chains_per_device * n_dev

                    local_sum_x = jnp.sum(xs_pos, axis=0)
                    global_sum_x = jax.lax.psum(local_sum_x, axis_name='device')
                    x_mean = global_sum_x / n_total

                    deltas = xs_pos - x_mean[jnp.newaxis, :]
                    local_m2 = jnp.einsum('ci,cj->ij', deltas, deltas)
                    m2_step = jax.lax.psum(local_m2, axis_name='device')

                    update = WelfordAlgorithmState(x_mean, m2_step, n_total)

                    if windowed_mass_matrix:
                        new_welford = welford_combine(welford_state, update)
                        updated_welford = _sel(do_mm_adapt, new_welford, welford_state)
                        at_boundary = window_end_mask[i]
                        update_mm = jnp.logical_and(do_mm_adapt, at_boundary)
                        sample_cov = _regularize_cov(
                            welford_cov(updated_welford)[0], updated_welford.sample_size)
                        mm_params = params._replace(inverse_mass_matrix=sample_cov)
                        updated_params = _sel(update_mm, mm_params, params)
                        updated_welford = _sel(update_mm, welford_empty, updated_welford)
                        updated_adapt_states = _sel(
                            update_mm, _make_adapt_reset(new_adapt_states), new_adapt_states
                        )
                        return updated_params, updated_welford, updated_adapt_states

                    new_welford = welford_combine(welford_state, update)
                    sample_cov = _regularize_cov(
                        welford_cov(new_welford)[0], new_welford.sample_size)
                    mm_params = params._replace(inverse_mass_matrix=sample_cov)
                    updated_params = _sel(do_mm_adapt, mm_params, params)
                    updated_welford = _sel(do_mm_adapt, new_welford, welford_state)
                    return updated_params, updated_welford, new_adapt_states

                params, welford_state, new_adapt_states = jax.lax.cond(
                    do_mm_adapt,
                    run_mass_matrix_adapt,
                    lambda _: (params, welford_state, new_adapt_states),
                    operand=None,
                )

            # Step size sync (average the per-chain step sizes at the stage boundary).
            with jax.named_scope("step_size_sync"):
                local_ss_sum = jnp.sum(new_step_sizes)
                global_ss_sum = jax.lax.psum(local_ss_sum, axis_name='device')
                synced_ss = global_ss_sum / (chains_per_device * jax.lax.axis_size('device'))
                new_step_sizes = jnp.where(
                    i == step_size_sync_step,
                    jnp.full_like(new_step_sizes, synced_ss),
                    new_step_sizes,
                )

            # L adaptation (ALBA / ESS): L_new = Lfactor * L * num_steps3 / min_ESS,
            # capped at L * L_max_ratio. Cross-chain-min ESS = worst parameter, worst chain.
            def calc_new_L(_):
                with jax.named_scope("L_adaptation"):
                    per_chain_ess = jax.vmap(lambda buf: _ess_shardmap(
                        buf[jnp.newaxis, :, :],
                        chain_axis=0, sample_axis=1,
                    ))(l_stage_bufs)
                    local_min_ess = jnp.min(per_chain_ess)
                    global_min_ess = jax.lax.pmin(local_min_ess, axis_name='device')
                    L_new = Lfactor * params.L * num_steps3 / global_min_ess
                    return jnp.minimum(L_new, params.L * L_max_ratio)

            new_L = jax.lax.cond(
                i == L_adaptation_step,
                calc_new_L,
                lambda _: params.L,
                operand=None,
            )
            params = params._replace(L=new_L)

            h = Hist(
                position=new_states.position,
                step_size=new_step_sizes,
                L=jnp.broadcast_to(params.L, new_step_sizes.shape),
                inverse_mass_matrix=jnp.broadcast_to(params.inverse_mass_matrix[jnp.newaxis], (chains_per_device, dim, dim)),
                nonan=successes,
                acceptance_rate=accs,
                is_accepted=is_accepteds,
                num_integration_steps=jnp.broadcast_to(num_integration_steps, new_step_sizes.shape),
                energy_change_raw=energy_changes_raw,
                kernel_nonan=kernel_nonans,
                step_norm=step_norms,
            )
            return (new_states, params, new_step_sizes, new_adapt_states, welford_state, l_stage_bufs), h

    # --- Setup inputs (identical mode schedule to MCLMC). ---

    mode = jnp.concatenate((
        jnp.ones(num_steps1, dtype=jnp.int32),
        2 * jnp.ones(round(0.67 * num_steps2), dtype=jnp.int32),
        1 * jnp.ones(round(0.33 * num_steps2), dtype=jnp.int32),
        3 * jnp.ones(num_steps3, dtype=jnp.int32),
        jnp.zeros(total_steps - tuning_steps, dtype=jnp.int32),
    ))

    keys = jax.random.split(rng_key, (num_chains, total_steps))
    keys = jnp.moveaxis(keys, 0, 1)  # (total_steps, num_chains, key_shape)

    welford_start = WelfordAlgorithmState(svi_mean, svi_inverse_mass_matrix*svi_mass_matrix_weight, svi_mass_matrix_weight)

    # Per-chain adaptive state = (dual-averaging state, step_size_max).
    adapt_single = (da_init(jnp.asarray(params_init.step_size)), jnp.asarray(jnp.inf, _canon))

    _tile = lambda x: jnp.broadcast_to(jnp.asarray(x)[jnp.newaxis], (num_chains,) + jnp.asarray(x).shape)
    step_sizes_init = jnp.full((num_chains,), params_init.step_size)
    adapt_states_init = jax.tree.map(_tile, adapt_single)
    l_stage_bufs_init = jnp.zeros((num_chains, num_steps3, dim))

    # --- shard_map (no vmap axis_name -- collectives only use 'device'). ---

    mesh = jax.make_mesh((num_devices,), ('device',))

    carry_out_specs = (P('device'), P(), P('device'), P('device'), P(), P('device'))
    samples_out_specs = P('device')

    pbar_scan_fn = _gen_scan_fn_one_bar(total_steps, progress_bar, axis_name='device')

    @jax.jit
    @functools.partial(_shard_map, mesh=mesh,
        in_specs=(
            (None, None, P(None, 'device')),
            P('device'), None, P('device'), P('device'), None, P('device'),
        ),
        out_specs=(carry_out_specs, samples_out_specs))
    def run_sharded(xs, state_init, params_init, step_sizes, adapt_states, welford_start, l_stage_bufs):
        with jax.named_scope("mams_run_sharded"):
            carry, samples = pbar_scan_fn(
                step_batched,
                init=(state_init, params_init, step_sizes, adapt_states, welford_start, l_stage_bufs),
                xs=xs,
            )
            samples = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), samples)
            return carry, samples

    _sharded_chain = NamedSharding(mesh, P('device'))
    _sharded_keys = NamedSharding(mesh, P(None, 'device'))
    _reshard = getattr(jax, 'reshard', jax.device_put)

    xs = (
        jnp.arange(total_steps, dtype=jnp.int32),
        mode,
        _reshard(keys, _sharded_keys),
    )
    state_init = _reshard(state_init, _sharded_chain)
    step_sizes_init = _reshard(step_sizes_init, _sharded_chain)
    adapt_states_init = _reshard(adapt_states_init, _sharded_chain)
    l_stage_bufs_init = _reshard(l_stage_bufs_init, _sharded_chain)

    carry, samples = run_sharded(
        xs,
        state_init, params_init, step_sizes_init, adapt_states_init, welford_start, l_stage_bufs_init,
    )
    _, params_final, _, _, _, _ = carry

    _replicated = NamedSharding(mesh, P())
    samples = jax.tree.map(lambda x: _reshard(x, _replicated), samples)
    params_final = jax.tree.map(lambda x: _reshard(x, _replicated), params_final)

    return samples, params_final


# Short public name, mirroring MCLMC.
MAMS = MAMS_JIT
