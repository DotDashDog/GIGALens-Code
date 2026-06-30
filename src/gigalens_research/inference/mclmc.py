import jax.numpy as jnp
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import io_callback
from fastprogress.fastprogress import progress_bar as fastprogress_bar

try:
    _shard_map = jax.shard_map  # type: ignore[attr-defined]
except AttributeError:
    from jax.experimental.shard_map import shard_map as _shard_map

# import blackjax
# from blackjax.mcmc.integrators import GeneralIntegrator, IntegratorState, ArrayTree, Callable, euclidean_position_update_fn
# from blackjax.mcmc.integrators import with_isokinetic_maruyama, generalized_two_stage_integrator, format_isokinetic_state_output, ravel_pytree, _normalized_flatten_array
# from blackjax.mcmc.integrators import mclachlan_coefficients, velocity_verlet_coefficients, yoshida_coefficients, omelyan_coefficients
# from blackjax.adaptation.mass_matrix import welford_algorithm, WelfordAlgorithmState
# from blackjax.util import generate_unit_vector, pytree_size
# from blackjax.types import ArrayLike, PRNGKey
# from blackjax.mcmc.mclmc import MCLMCInfo
# from blackjax.adaptation.mclmc_adaptation import pytree_size, MCLMCAdaptationState, handle_nans, incremental_value_update

from .blackjax_updated_utils import *
from .blackjax_updated_utils import (
    _build_kernel_shardmap,
    _ess_shardmap,
    _gen_scan_fn_one_bar,
    KernelExtras,
)

import functools
from typing import Callable, NamedTuple, Optional
from collections import namedtuple

import time
from threading import Lock



def MCLMC_JIT(model_seq, qz, n_hmc=16, num_burnin_steps=1000, num_results=2000,
          desired_energy_variance=5e-4, init_L=None, init_step_size=None, frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
          progress_bar=False, seed=0, debug_output=False, regularize_mass_matrix=False):
    # Scene-only: the ProbModel owns batch-flexible per-dataset SceneSimulators, so
    # log_prob(z) renders through them directly -- no separately-built simulator.
    def log_prob(z):
        return model_seq.prob_model.log_prob(z)[0]

    n_chains = n_hmc
    integrator = isokinetic_mclachlan_smart

    build_kernel_fn = _build_kernel_shardmap
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



    adapt_fn = full_mclmc_with_adapt_sharded

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
        desired_energy_var=desired_energy_variance,
        num_chains=n_chains,
        num_effective_samples=100,
        svi_mass_matrix_weight=10.0 * n_chains,
        # mass_matrix_num_effective_samples=mass_matrix_num_effective_samples,
        step_size_adapt_use_psmile=False,
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


def full_mclmc_with_adapt_sharded(
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
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    Lfactor=0.4,
    num_chains=8,
    svi_mass_matrix_weight=20.,
    # mass_matrix_num_effective_samples=1000,
    step_size_adapt_use_psmile=False,
    windowed_mass_matrix=True,
    regularize_mass_matrix=False,
    progress_bar=False,
):
    """Sharded version of full_mclmc_with_adapt. Distributes chains across
    devices via shard_map with explicit batching (no vmap axis_name).
    Cross-chain reductions use jnp ops locally + psum/pmin('device').
    """
    num_devices = len(jax.devices())
    num_chains = (num_chains // num_devices) * num_devices
    if num_chains == 0:
        raise ValueError(f"num_chains must be >= num_devices ({num_devices})")
    chains_per_device = num_chains // num_devices

    dim = state_init.position.shape[-1]
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)
    # decay_rate_mass_matrix = (mass_matrix_num_effective_samples - 1.0) / (mass_matrix_num_effective_samples + 1.0)

    welford_init_fn, _, welford_cov = welford_algorithm(is_diagonal_matrix=False)

    # F4 (opt-in): Stan-style regularization of EVERY window's sample covariance before it is
    # installed as the inverse mass matrix. Baseline only regularizes window 1 (via the SVI
    # prior); windows 2/3 accumulate from an empty Welford with no shrinkage, so a window built
    # from frozen/correlated/multi-modal chains can be rank-deficient or (under float32 Welford)
    # non-PSD -> cholesky NaN -> rejection cascade (diagnosis F3/F4). This mirrors blackjax's
    # mass_matrix_adaptation: scale by n/(n+5), add a 1e-3 shrinkage*I floor on all windows, and
    # lift any roundoff-negative eigenvalues so the downstream cholesky never sees a non-PSD
    # metric. Default False => byte-identical to baseline.
    def _regularize_cov(cov, n):
        if not regularize_mass_matrix:
            return cov
        cov = 0.5 * (cov + jnp.swapaxes(cov, -1, -2))           # symmetrize
        n = jnp.asarray(n, cov.dtype)
        eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
        shrink = 1e-3 * (5.0 / (n + 5.0))
        reg = (n / (n + 5.0)) * cov + shrink * eye              # Stan window shrinkage
        w, V = jnp.linalg.eigh(reg)                             # PSD floor (belt-and-suspenders)
        w = jnp.clip(w, shrink, None)
        return (V * w[..., jnp.newaxis, :]) @ jnp.swapaxes(V, -1, -2)

    # Single-dtype sampler. The log-density / energy dtype drives EVERYTHING: it is float64
    # when the likelihood runs in high precision under jax_enable_x64, float32 otherwise.
    # qz.sample() can yield float32 positions/momentum even when the energy is float64 (and
    # qz.mean()/covariance() may be float32 too), which mixes float32 state with float64
    # energy/step_size and trips lax.select/cond dtype checks (blackjax handle_nans, the
    # mass-matrix cond). Cast the whole initial state AND all adaptation params to the energy
    # dtype so the scan carry is uniformly one dtype. The likelihood forward model stays
    # float32 regardless (see gigalens.jax.model.BackwardProbModel.log_prob).
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

    # --- Per-chain step size adaptation (unchanged, called via vmap without axis_name) ---

    def step_size_adapt(previous_state, next_state, info, params, adaptive_state, nan_key):
        time, x_average, step_size_max = adaptive_state
        success, state, step_size_max, energy_change = handle_nans(
            previous_state, next_state, params.step_size, step_size_max, info.energy_change, nan_key,
        )
        xi = jnp.square(energy_change) / (dim * desired_energy_var) + 1e-8
        weight = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate)))
        weighted_x = weight * (xi / jnp.power(params.step_size, 6.0))
        x_average = decay_rate * x_average + weighted_x
        time = decay_rate * time + weight
        step_size = jnp.power(x_average / time, -1.0 / 6.0)
        step_size = jnp.minimum(step_size, step_size_max)
        adaptive_state = (time, x_average, step_size_max)
        return state, params._replace(step_size=step_size), adaptive_state, success, xi

    def step_size_adapt_psmile_continuous(previous_state, next_state, info, params, adaptive_state, nan_key):
        mu, sigma2, count, step_size_max = adaptive_state
        success, state, step_size_max, energy_change = handle_nans(
            previous_state, next_state, params.step_size, step_size_max, info.energy_change, nan_key,
        )
        xi = jnp.square(energy_change) / (dim * desired_energy_var) + 1e-8
        beta = 1 - decay_rate
        delta = 0.1
        eps = 1e-8
        abs_dE = jnp.abs(energy_change)
        count = count + 1
        mu_next = (1.0 - beta) * mu + beta * abs_dE
        sigma2_next = (1.0 - beta) * sigma2 + beta * jnp.square(abs_dE - mu_next)
        bias_correction = 1.0 - jnp.power(1.0 - beta, count)
        mu_hat = mu_next / bias_correction
        sigma2_hat = sigma2_next / bias_correction
        shape = jnp.square(mu_hat) / (sigma2_hat + eps)
        scale = sigma2_hat / (mu_hat + eps)
        # Wilson-Hilferty normal approximation to gamma CDF
        # (avoids igamma's internal while_loop which triggers VMA mismatches in shard_map)
        x_std = abs_dE / (scale * shape + eps)
        z = (jnp.cbrt(x_std) - (1.0 - 1.0 / (9.0 * shape))) / jnp.sqrt(1.0 / (9.0 * shape + eps))
        cdf_value = jax.scipy.special.ndtr(z)
        step_size = params.step_size * (1 + (0.5 - cdf_value) * delta)
        step_size = jnp.minimum(step_size, step_size_max)
        adaptive_state_new = (mu_next, sigma2_next, count, step_size_max)
        return state, params._replace(step_size=step_size), adaptive_state_new, success, xi

    step_size_adapt_func = step_size_adapt_psmile_continuous if step_size_adapt_use_psmile else step_size_adapt

    # --- Windowed mass matrix setup (STAN-style expanding windows) ---
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
        if step_size_adapt_use_psmile:
            def _make_adapt_reset(cur):
                return (jnp.zeros_like(cur[0]), jnp.zeros_like(cur[1]),
                        jnp.zeros_like(cur[2]), cur[3])
        else:
            def _make_adapt_reset(cur):
                return (jnp.zeros_like(cur[0]), jnp.zeros_like(cur[1]), cur[2])

    # --- Batched scan body: explicit batch dim, collectives only use 'device' ---

    Hist = namedtuple("hist", [
        "position", "step_size", "L", "inverse_mass_matrix", "nonan", "xi",
        "energy_change_raw", "kernel_nonan", "step_norm",
    ])

    l_buffer_start = L_adaptation_step - num_steps3

    def step_batched(carry, mode_and_key):
        with jax.named_scope("mclmc_step_batched"):
            i, mode, rng_keys_batch = mode_and_key

            states, params, step_sizes, adapt_states, welford_state, l_stage_bufs = carry

            do_ssa = jnp.logical_or(mode == 1, mode == 2)
            do_mm_adapt = mode == 2

            key_pairs = jax.vmap(jax.random.split)(rng_keys_batch)
            chain_keys = key_pairs[:, 0]
            nan_keys = key_pairs[:, 1]

            kernel_fn = kernel(params.inverse_mass_matrix)

            def per_chain(prev_state, rng_key, nan_key, step_size, adapt_state):
                with jax.named_scope("per_chain_kernel"):
                    new_state, info = kernel_fn(
                        rng_key=rng_key, state=prev_state, L=params.L, step_size=step_size)

                    # Capture diagnostic fields before NaN-zeroing happens in
                    # step_size_adapt (via handle_nans).  info.energy_change has
                    # already been zeroed for NaN/Inf steps by the kernel; the
                    # raw (pre-zeroed) value is in info.extras.energy_change_raw.
                    energy_change_raw = info.extras.energy_change_raw
                    kernel_nonan_flag = info.extras.kernel_nonan
                    # Step norm: ||x_t - x_{t-1}|| in position space
                    pos_diff = new_state.position.reshape(-1) - prev_state.position.reshape(-1)
                    step_norm_val = jnp.sqrt(jnp.sum(pos_diff ** 2))

                def adapt_one(_):
                    with jax.named_scope("step_size_adapt"):
                        pseudo_params = params._replace(step_size=step_size)
                        a_state, a_params, a_adapt, a_success, a_xi = step_size_adapt_func(
                            prev_state, new_state, info, pseudo_params, adapt_state, nan_key
                        )
                        return (
                            a_state,
                            a_params.step_size,
                            a_adapt,
                            a_success,
                            a_xi,
                        )

                def skip_adapt(_):
                    success_placeholder = jnp.isfinite(new_state.position.reshape(-1)[0])
                    # Log the real energy-error ratio xi even when step-size adaptation
                    # is off (modes 0=results and 3=L-tuning), instead of the old -1
                    # sentinel. Uses the SAME definition as step_size_adapt
                    # (info.energy_change is already NaN-zeroed by the kernel), so the
                    # burn-in and results-phase xi are directly comparable. This value
                    # is logged only (it feeds the Hist diagnostic, never the kernel or
                    # step-size), so it does not change sampling.
                    xi_val = jnp.square(info.energy_change) / (dim * desired_energy_var) + 1e-8
                    return (
                        new_state,
                        step_size,
                        adapt_state,
                        success_placeholder,
                        xi_val.astype(step_size.dtype),
                    )

                result = jax.lax.cond(
                    do_ssa,
                    adapt_one,
                    skip_adapt,
                    operand=None,
                )
                return result + (energy_change_raw, kernel_nonan_flag, step_norm_val)

            (new_states, new_step_sizes, new_adapt_states, successes, xis,
             energy_changes_raw, kernel_nonans, step_norms) = jax.vmap(per_chain)(
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

            # Cross-chain mass matrix adaptation (jnp locally, psum across devices)
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

            # Step size sync
            with jax.named_scope("step_size_sync"):
                local_ss_sum = jnp.sum(new_step_sizes)
                global_ss_sum = jax.lax.psum(local_ss_sum, axis_name='device')
                synced_ss = global_ss_sum / (chains_per_device * jax.lax.axis_size('device'))
                new_step_sizes = jnp.where(
                    i == step_size_sync_step,
                    jnp.full_like(new_step_sizes, synced_ss),
                    new_step_sizes,
                )

            # L adaptation
            def calc_new_L(_):
                with jax.named_scope("L_adaptation"):
                    per_chain_ess = jax.vmap(lambda buf: _ess_shardmap(
                        buf[jnp.newaxis, :, :],
                        chain_axis=0, sample_axis=1,
                    ))(l_stage_bufs)
                    local_min_ess = jnp.min(per_chain_ess)
                    global_min_ess = jax.lax.pmin(local_min_ess, axis_name='device')
                    return Lfactor * num_steps3 * synced_ss / global_min_ess

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
                xi=xis,
                energy_change_raw=energy_changes_raw,
                kernel_nonan=kernel_nonans,
                step_norm=step_norms,
            )
            return (new_states, params, new_step_sizes, new_adapt_states, welford_state, l_stage_bufs), h

    # --- Setup inputs ---

    mode = jnp.concatenate((
        jnp.ones(num_steps1, dtype=jnp.int32),
        2 * jnp.ones(round(0.67 * num_steps2), dtype=jnp.int32),
        1 * jnp.ones(round(0.33 * num_steps2), dtype=jnp.int32),
        3 * jnp.ones(num_steps3, dtype=jnp.int32),
        jnp.zeros(total_steps - tuning_steps, dtype=jnp.int32),
    ))

    keys = jax.random.split(rng_key, (num_chains, total_steps))
    keys = jnp.moveaxis(keys, 0, 1)  # (total_steps, num_chains, key_shape)

    welford_start = WelfordAlgorithmState(svi_mean, svi_inverse_mass_matrix*svi_mass_matrix_weight, svi_mass_matrix_weight)#welford_init_fn(dim)


    if step_size_adapt_use_psmile:
        adapt_single = (jnp.array(0.0), jnp.array(0.0), jnp.array(0, dtype=jnp.int32), jnp.array(jnp.inf))
    else:
        adapt_single = (jnp.array(0.0), jnp.array(0.0), jnp.array(jnp.inf))

    _tile = lambda x: jnp.broadcast_to(jnp.asarray(x)[jnp.newaxis], (num_chains,) + jnp.asarray(x).shape)
    step_sizes_init = jnp.full((num_chains,), params_init.step_size)
    adapt_states_init = jax.tree.map(_tile, adapt_single)
    l_stage_bufs_init = jnp.zeros((num_chains, num_steps3, dim))

    # --- shard_map (no vmap axis_name — collectives only use 'device') ---

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
        with jax.named_scope("mclmc_run_sharded"):
            carry, samples = pbar_scan_fn(
                step_batched,
                init=(state_init, params_init, step_sizes, adapt_states, welford_start, l_stage_bufs),
                xs=xs,
            )
            samples = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), samples)
            return carry, samples

    # JAX 0.10+ no longer auto-reshards shard_map inputs to match in_specs.
    # Pre-shard only the inputs whose in_specs is a PartitionSpec (skipping
    # None-spec inputs, since those bypass the strict check inside shard_map
    # and remain plain replicated arrays — important to avoid the
    # "Closing over inputs sharded on Explicit axes" follow-on error).
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

    # Gather sharded outputs back to fully-replicated arrays. Without this,
    # JAX 0.10's strict gather/sharding rules raise ShardingTypeError on
    # innocuous indexing like `samples.step_size[0, -1]` because the chain
    # axis is sharded across 'device'. Replicating after sampling restores
    # the pre-0.10 UX without changing semantics. No-op on older JAX.
    _replicated = NamedSharding(mesh, P())
    samples = jax.tree.map(lambda x: _reshard(x, _replicated), samples)
    params_final = jax.tree.map(lambda x: _reshard(x, _replicated), params_final)

    return samples, params_final


# Short public name for new code; MCLMC_JIT is kept for compatibility with
# mclmc_alt.py-era scripts/notebooks.
MCLMC = MCLMC_JIT
