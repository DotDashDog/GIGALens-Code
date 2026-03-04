
import blackjax.progress_bar
import jax.numpy as jnp
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import blackjax
from blackjax.mcmc.integrators import GeneralIntegrator, IntegratorState, ArrayTree, Callable, euclidean_position_update_fn
from blackjax.mcmc.integrators import generalized_two_stage_integrator, format_isokinetic_state_output, ravel_pytree, _normalized_flatten_array
from blackjax.mcmc.integrators import mclachlan_coefficients, yoshida_coefficients, omelyan_coefficients
from blackjax.adaptation.mass_matrix import welford_algorithm, WelfordAlgorithmState

from typing import Callable, NamedTuple, Optional
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.integrators import IntegratorState, with_isokinetic_maruyama
from blackjax.types import ArrayLike, PRNGKey
from blackjax.util import generate_unit_vector, pytree_size

import gigalens.jax.simulator as sim

import time



# from mclmc_parallel import init_multi, mclmc_multi

def MCLMC(model_seq, qz, n_hmc=16, num_burnin_steps=1000, num_results=2000, mass_matrix_adapt=True, 
          continuous_adaptation=True, desired_energy_variance=5e-4, init_L=None, init_step_size=None, 
          progress_bar=False, print_adapt_params=False,seed=0):
    
    """GIGALens-like wrapper for MCLMC sampling (modifed from blackjax). 
        Note that because it isn't a method of ModellingSequence, you do need to pass the model_seq

    Returns:
        np.array, shape: (num_chains, num_results, num_params): The final MCLMC chains
    """

    lens_sim = sim.LensSimulator(
        model_seq.phys_model,
        model_seq.sim_config,
        bs=1,
    )

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    rng_key = jax.random.key(seed)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)


    n_chains = n_hmc
    # desired_energy_variance= 5e-4 #* Tuning parameter. Keep as is for now
    transform = lambda state, info: state.position #* For final chain outputs, just output locations

    integrator = isokinetic_mclachlan_smart

    # build the kernel
    kernel = lambda inverse_mass_matrix : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=log_prob,
        integrator=integrator,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    #* Initialize states for burnin from surrogate
    state_multi = init_multi(qz.sample((n_chains,), seed=init_key), init_key, log_prob)
    dim = state_multi.position.shape[-1]

    #* Start hyperparameters at default guesses based on dimensionality
    init_L = jnp.sqrt(dim) if init_L is None else init_L
    init_step_size = (jnp.sqrt(dim) * 0.25) if init_step_size is None else init_step_size
    starting_adapt_state = blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState(
        L=init_L, step_size=init_step_size, inverse_mass_matrix=qz.covariance()
    )

    #* Run burnin, which adapts L, step size, and the mass matrix (if told to)
    starttime = time.perf_counter()
    (
        blackjax_state_after_tuning, #* The final positions (and other state info) of the chains
        blackjax_mclmc_sampler_params, #* The tuned hyperparameters
        _
    ) = mclmc_find_L_and_step_size_smart(
        mclmc_kernel=kernel,
        num_steps=num_burnin_steps,
        state=state_multi,
        rng_key=tune_key,
        frac_tune1=0.1, #* initial step size tuning
        frac_tune2=0.7, #* Used for mass matrix adaptation
        frac_tune3=0.2, #! Tuning L. ~10 effective samples are needed for this to be accurate
        params=starting_adapt_state,
        desired_energy_var=desired_energy_variance,
        multi_chain=True,
        num_chains=n_chains,
        mass_matrix_adapt=mass_matrix_adapt,
        continuous_adaptation=continuous_adaptation,
    )
    total_time = time.perf_counter()-starttime
    print("Burnin Time:", total_time)


    L = blackjax_mclmc_sampler_params.L
    step_size = blackjax_mclmc_sampler_params.step_size
    if print_adapt_params:
        print(f"ADAPTED. L: {L}, step_size: {step_size}, L/step: {L/step_size}")


    sampling_alg = mclmc_multi(
        log_prob,
        L=L,
        step_size=step_size,
        num_chains=n_chains,
        inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
        integrator=integrator,
    )

    starttime = time.perf_counter()
    _, multi_chain_samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_results,
        transform=transform,
        progress_bar=progress_bar,
    )
    total_time = time.perf_counter()-starttime
    print(f"Sampling took {total_time} s")

    #* Transpose to (num_chains, num_results, dim)
    multi_chain_samples = jnp.transpose(multi_chain_samples, axes=(1, 0, 2))

    #* Final shape should be (num_results, num_chains, dim)
    return multi_chain_samples




#* ----- MODIFICATIONS TO MCLMC BASE FUNCTIONS SUPPORTING MULTIPLE PARALLEL CHAINS

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
        return lambda fn: map_factory(fn, in_axes=in_axes)
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
    mapper = _make_mapper(map_factory, in_axes=(0, 0, 0, 0))
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

    if len(jnp.shape(L)) == 0:
        L = jnp.full(num_chains, L)
    if len(jnp.shape(step_size)) == 0:
        step_size = jnp.full(num_chains, step_size)

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

#* Define integrators that work with non-diagonal mass matrices
isokinetic_mclachlan_smart = generate_isokinetic_integrator_smart(mclachlan_coefficients)
isokinetic_yoshida_smart = generate_isokinetic_integrator_smart(yoshida_coefficients)
isokinetic_omelyan_smart = generate_isokinetic_integrator_smart(omelyan_coefficients)


##* ----- MULTI CHAIN MASS MATRIX ADAPTATION FOR MCLMC --------------------------------------

from blackjax.adaptation.mclmc_adaptation import pytree_size, MCLMCAdaptationState, handle_nans, incremental_value_update
from blackjax.mcmc.integrators import generalized_two_stage_integrator, format_isokinetic_state_output, ravel_pytree, _normalized_flatten_array
from blackjax.diagnostics import effective_sample_size

def full_mclmc_with_adapt(
    kernel,
    num_burnin_steps,
    num_results,
    state_init,
    params_init,
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
):
    dim = state_init.position.shape[-1]
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    welford_init, welford_update, welford_cov = welford_algorithm(is_diagonal_matrix=False)

    svi_inverse_mass_matrix = params_init.inverse_mass_matrix

    total_steps = num_burnin_steps + num_results
    num_steps1, num_steps2, num_steps3 = round(num_burnin_steps * frac_tune1), round(num_burnin_steps * frac_tune2), round(num_burnin_steps * frac_tune3)
    tuning_steps = num_steps1 + num_steps2 + num_steps3

    step_size_sync_step = num_steps1 + num_steps2
    L_adaptation_step = tuning_steps
    
    def step_size_adapt(previous_state, next_state, info, params, adaptive_state, nan_key):
        time, x_average, step_size_max = adaptive_state

        # step updating
        success, state, step_size_max, energy_change = handle_nans(
            previous_state,
            next_state,
            params.step_size,
            step_size_max,
            info.energy_change,
            nan_key,
        )


        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (
            jnp.square(energy_change) / (dim * desired_energy_var)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        weighted_x = weight * (xi / jnp.power(params.step_size, 6.0))
        
        x_average = decay_rate * x_average + weighted_x#jax.lax.psum(weighted_x, axis_name='chain')
        
        time = decay_rate * time + weight#jax.lax.psum(weight, axis_name='chain')
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        params_new = params._replace(step_size=step_size)

        adaptive_state = (time, x_average, step_size_max)

        return state, params_new, adaptive_state, success
    
    def mass_matrix_adapt(state, params, welford_state):
        #* Do mass matrix adaptation if in stage 2
        #* Aggregate welford states across chains to get full covariance. 
        x = ravel_pytree(state.position)[0]
        n = jax.lax.axis_size('chain')
        x_mean = jax.lax.pmean(x, axis_name='chain')
        delta = x - x_mean
        m2_step = jax.lax.psum(jnp.outer(delta, delta), axis_name='chain')
        update_state = WelfordAlgorithmState(x_mean, m2_step, n)

        welford_state = welford_combine(welford_state, update_state)

        # #! Right now using a dumb criterion for mass matrix adaptation (start after 50 samples per chain)
        # params_new = jax.lax.cond(welford_state.sample_size > num_chains*50, #! CHANGE IF YOU WANT THIS TO WORK FOR OTHER CASES
        #     lambda : params._replace(inverse_mass_matrix=welford_cov(welford_state)[0]),
        #     lambda : params,
        # )
        sample_cov = welford_cov(welford_state)[0]
        weighted_mean_mat = (sample_cov * welford_state.sample_size + svi_inverse_mass_matrix * svi_mass_matrix_weight)/(welford_state.sample_size+svi_mass_matrix_weight)
        params_new = params._replace(inverse_mass_matrix=weighted_mean_mat)
        return params_new, welford_state

    def step(iteration_state, mode_and_key):
        """does one step of the dynamics and updates the estimate of the optimal step size, continuously updating the mass matrix instead of just tracking"""
        mode, rng_key, i = mode_and_key
        do_step_size_adapt = jnp.logical_or(mode==1, mode==2)
        do_mass_matrix_adapt = mode == 2
        rng_key, nan_key = jax.random.split(rng_key)

        previous_state, params, adaptive_state, welford_state, sample_buffer = iteration_state

        state, info = kernel(params.inverse_mass_matrix)(
            rng_key=rng_key,
            state=previous_state,
            L=params.L,
            step_size=params.step_size,
        )

        sample_buffer = sample_buffer.at[i].set(state.position)

        #* Do step size adaptation if in stage 1 or 2
        state, params, adaptive_state, success = jax.lax.cond(
            do_step_size_adapt,
            (lambda : step_size_adapt(previous_state, state, info, params, adaptive_state, nan_key)),
            (lambda : (state, params, adaptive_state, True)),
        )

        #* Do mass matrix adaptation if in stage 2
        params, welford_state = jax.lax.cond(
            do_mass_matrix_adapt, 
            (lambda : mass_matrix_adapt(state, params, welford_state)),
            (lambda : (params, welford_state)),
        )

        params = jax.lax.cond(
            i == step_size_sync_step,
            lambda : params._replace(step_size=jax.lax.pmean(params.step_size, axis_name='chain')),
            lambda : params
        )

        def calc_new_L():
            ess = effective_sample_size(sample_buffer[jnp.newaxis, L_adaptation_step-num_steps3:L_adaptation_step], chain_axis=0, sample_axis=1)
            return Lfactor* num_steps3 * params.step_size/jax.lax.pmin(jnp.min(ess), axis_name='chain')

        params = jax.lax.cond(
            i == L_adaptation_step,
            lambda : params._replace(L=calc_new_L()),
            lambda : params
        )
        
        
        #! Still need to add ESS-based L adaptation and sync step sizes at end of step size adaptation
        
        return (state, params, adaptive_state, welford_state, sample_buffer), state.position
    

    # step_vmapped = jax.vmap(step, in_axes=(0, (None, 0)), axis_name='chain')

    # pbar_scan_fn = blackjax.progress_bar.gen_scan_fn(total_steps, progress_bar)

    # def tile_params(p):
    #     return jax.tree.map(lambda x: jnp.repeat(jnp.array(x)[jnp.newaxis, ...], num_chains, axis=0), p)

    # run_steps_multi = lambda xs, state_init, params_init : pbar_scan_fn(
    #     step_vmapped,
    #     init=(
    #         state_init,
    #         tile_params(params_init),#! NEED TO TILE RIGHT
    #         tile_params((0.0, 0.0, jnp.inf)),
    #         tile_params(welford_start)
    #     ),
    #     xs=xs,
    # )
    # print(mode.shape)
    # print(keys.shape)
    # print(tile_params(params_init))
    # print(tile_params((0.0, 0.0, jnp.inf)))
    # print(tile_params(welford_start))

    mode = jnp.concatenate((
        jnp.ones(num_steps1, dtype=jnp.int32), 
        2*jnp.ones(round(0.67 * num_steps2), dtype=jnp.int32), 
        1*jnp.ones(round(0.33*num_steps2), dtype=jnp.int32), #! Not doing any step adapt after mass matrix
        3*jnp.ones(num_steps3, dtype=jnp.int32), 
        jnp.zeros(total_steps-tuning_steps, dtype=jnp.int32),
    ))

    keys = jax.random.split(rng_key, (num_chains, total_steps))

    welford_start = welford_init(dim)

    sample_buffer_init = jnp.zeros((total_steps, dim))
    run_steps = lambda xs, state_init, params_init : jax.lax.scan(
        step,
        init=(
            state_init,
            params_init,
            (0.0, 0.0, jnp.inf),
            welford_start,
            sample_buffer_init,
        ),
        xs=xs,
    )
    run_steps_vmap = jax.jit(jax.vmap(run_steps, in_axes=((None, 0, None), 0, None), axis_name='chain'))

    # mesh = jax.make_mesh((len(jax.devices()),), ('device',))
    # in_specs = ((None, P('device'), None), P('device'), None)


    # run_steps_multi = jax.pmap(run_steps_vmap, in_axes=((None, 0, None), 0, None), out_axes=0, axis_name='device')

    # reshape_pmap = lambda x : x.reshape(num_devices, num_chains_per_device, *x.shape[1:])
    # state_init = jax.tree.map(reshape_pmap, state_init)

    carry, samples = run_steps_vmap(
        (mode, keys, jnp.arange(total_steps, dtype=jnp.int32)), state_init, params_init
    )
    state, params, _, welford_state, samples_buffered = carry
    result_samples = samples[:, -num_results:, :]
    return result_samples, params
    

def mclmc_find_L_and_step_size_smart(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    params=None,
    Lfactor=0.4,
    multi_chain=False,
    num_chains=8,
    mass_matrix_adapt=True,
    continuous_adaptation=True,
):
    """
    Modified version of burnin from blackjax that runs over 

    Parameters
    ----------
    mclmc_kernel
        The kernel function used for the MCMC algorithm. FOR VMAPPING REASONS, ALWAYS USE STOCK BLACKJAX KERNEL, NOT PARALLEL KERNEL
    num_steps
        The number of MCMC steps that will subsequently be run, after tuning.
    state
        The initial state of the MCMC algorithm. IF multi_chain IS True, MUST USE STATE GENERATED BY init_multi
    rng_key
        The random number generator key.
    frac_tune1
        The fraction of tuning for the first step of the adaptation.
    frac_tune2
        The fraction of tuning for the second step of the adaptation.
    frac_tune3
        The fraction of tuning for the third step of the adaptation.
    desired_energy_var
        The desired energy variance for the MCMC algorithm.
    trust_in_estimate
        The trust in the estimate of optimal stepsize.
    num_effective_samples
        The number of effective samples for the MCMC algorithm.
    diagonal_preconditioning
        Whether to do diagonal preconditioning (i.e. a mass matrix)
    params
        Initial params to start tuning from (optional)
    Lfactor
        The factor scaling the estimated autocorrelation length to obtain momentum decoherence length L.
    ------------------------------------------ LINUS' MODIFICATIONS -----------------------------------------------
    multi_chain
        Whether to run multiple chains in parallel (uses jax.vmap, no cross-GPU parallelism yet)
    num_chains
        How many chains to run in parallel. If multi_chain is False, not used. 
        Currently there's some sort of error with num_chains = 1, so just use multi_chain=False if you want to one chain
    mass_matrix_adapt
        Whether to adapt the mass matrix based on the covariance of the samples during burnin. 
        Non-diagonal mass matrix adaptation is my addition, and it can cause problems if there aren't enough effective samples to make
        the covariance well-conditioned. Also, the second stage of adaptation at the very end may invalidate the chosen step size in extreme cases
    


    Returns
    -------
    A tuple containing the final state of the MCMC algorithm and the final hyperparameters.

    Example
    -------
    .. code::
        kernel = lambda inverse_mass_matrix : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=integrator,
        inverse_mass_matrix=inverse_mass_matrix,
        )

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            diagonal_preconditioning=preconditioning,
        )
    """
    dim = state.position.shape[-1]
    if params is None:
        raise ValueError("Must specify a starting point for adaptation")
        # params = MCLMCAdaptationState(
        #     jnp.sqrt(dim), jnp.sqrt(dim) * 0.25, inverse_mass_matrix=jnp.ones((dim,))
        # )

    part1_key, part2_key = jax.random.split(rng_key, 2)
    total_num_tuning_integrator_steps = 0

    num_steps1, num_steps2 = round(num_steps * frac_tune1), round(
        num_steps * frac_tune2
    )
    num_steps2 += num_steps2 // 3
    num_steps3 = round(num_steps * frac_tune3)

    state, params = make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        desired_energy_var=desired_energy_var,
        trust_in_estimate=trust_in_estimate,
        num_effective_samples=num_effective_samples,
        multi_chain=multi_chain,
        num_chains=num_chains,
        mass_matrix_adapt=mass_matrix_adapt,
        continuous_adaptation=continuous_adaptation,
    )(state, params, num_steps, part1_key)
    total_num_tuning_integrator_steps += num_steps1 + num_steps2

    if num_steps3 >= 2:  # at least 2 samples for ESS estimation
        state, params = make_adaptation_L(
            mclmc_kernel(params.inverse_mass_matrix), frac=frac_tune3, Lfactor=Lfactor, 
            multi_chain=multi_chain, num_chains=num_chains, mass_matrix_adapt=mass_matrix_adapt
        )(state, params, num_steps, part2_key)
        total_num_tuning_integrator_steps += num_steps3

    return state, params, total_num_tuning_integrator_steps



def make_L_step_size_adaptation(
    kernel,
    dim,
    frac_tune1,
    frac_tune2,
    desired_energy_var=1e-3,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    multi_chain=True,
    num_chains=8,
    mass_matrix_adapt=True,
    continuous_adaptation=False,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for unadjusted MCLMC"""

    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    welford_init, welford_update, welford_cov = welford_algorithm(is_diagonal_matrix=False)

    def predictor(previous_state, params, adaptive_state, rng_key):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
        Designed for the unadjusted MCHMC"""

        time, x_average, step_size_max = adaptive_state

        rng_key, nan_key = jax.random.split(rng_key)

        # dynamics
        next_state, info = kernel(params.inverse_mass_matrix)(
            rng_key=rng_key,
            state=previous_state,
            L=params.L,
            step_size=params.step_size,
        )

        # step updating
        success, state, step_size_max, energy_change = handle_nans(
            previous_state,
            next_state,
            params.step_size,
            step_size_max,
            info.energy_change,
            nan_key,
        )

        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = (
            jnp.square(energy_change) / (dim * desired_energy_var)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        params_new = params._replace(step_size=step_size)

        adaptive_state = (time, x_average, step_size_max)

        return state, params_new, adaptive_state, success

    def step(iteration_state, weight_and_key):
        """does one step of the dynamics and updates the estimate of the optimal step size, tracking updates to the covariance"""

        mask, rng_key, svi_inverse_mass_matrix = weight_and_key
        state, params, adaptive_state, welford_state = iteration_state

        state, params, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )

        x = ravel_pytree(state.position)[0]

        #! Choose how to update mass matrix, right now just tracking
        welford_state = jax.lax.cond(mask, 
            welford_update,
            lambda welford_state, x : welford_state,
            welford_state, x
        )

        return (state, params, adaptive_state, welford_state), state.position

    def step_continuous_mass_matrix_adapt(iteration_state, weight_and_key):
        """does one step of the dynamics and updates the estimate of the optimal step size, continuously updating the mass matrix instead of just tracking"""
        mask, rng_key, svi_inverse_mass_matrix = weight_and_key
        state, params, adaptive_state, welford_state = iteration_state

        state, params, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )

        x = ravel_pytree(state.position)[0]
        
        # welford_state = jax.lax.cond(mask, 
        #     welford_update,
        #     lambda in_state, x : in_state,
        #     welford_state, x
        # )

        # jax.debug.print(str(params))
        # jax.debug.print(str(params._replace(inverse_mass_matrix=welford_cov(welford_state)[0])))

        #* Aggregate welford states across chains to get full covariance. 
        #! Will break things if only 1 chain?
        # welford_aggregate = aggregate_chain_welford(welford_state, chain_axis='chain')

        n = jax.lax.axis_size('chain')
        x_mean = jax.lax.pmean(x, axis_name='chain')
        delta = x - x_mean
        m2_step = jax.lax.psum(jnp.outer(delta, delta), axis_name='chain')
        update_state = WelfordAlgorithmState(x_mean, m2_step, n)

        #! Choose how to update mass matrix, right now just tracking
        welford_state = jax.lax.cond(mask, 
            welford_combine,
            lambda running_state, updating_state : running_state,
            welford_state, update_state
        )

        # #! Right now using a dumb criterion for mass matrix adaptation (start after 100+ adapt samples)
        # params_new = jax.lax.cond(jnp.logical_and(mask, welford_state.sample_size > num_chains*50), #! CHANGE IF YOU WANT THIS TO WORK FOR OTHER CASES
        #     lambda welford_state_in : params._replace(inverse_mass_matrix=welford_cov(welford_state_in)[0]),
        #     lambda welford_state_in : params,
        #     welford_state
        # )


        smp_cov, smp_n, smp_mean = welford_cov(welford_state)

        
        svi_n = 100
        joined_covariance = ((smp_n * smp_cov) + (svi_n* svi_inverse_mass_matrix))/(svi_n+smp_n)
        params_new = params._replace(inverse_mass_matrix=joined_covariance)
        
        return (state, params_new, adaptive_state, welford_state), state.position
        

    def L_step_size_adaptation(state, params, num_steps, rng_key):

        welford_start = welford_init(state.position.shape[-1])

        step_func = step_continuous_mass_matrix_adapt if continuous_adaptation else step
        if continuous_adaptation:
            print("USING CONTINUOUS MASS MATRIX ADAPTATION. This is mostly untested. Could screw up")
        run_steps = lambda xs, state, params: jax.lax.scan(
            step_func,
            init=(
                state,
                params,
                (0.0, 0.0, jnp.inf),
                welford_start,
            ),
            xs=xs,
        )
        num_steps1, num_steps2 = round(num_steps * frac_tune1), round(
            num_steps * frac_tune2
        )

        # we use the last num_steps2 to compute the diagonal preconditioner
        mask = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))
        num_steps_net = num_steps1 + num_steps2

        inverse_mass_matrix_tiled = jnp.tile(params.inverse_mass_matrix, (num_steps1+num_steps2, 1, 1))
        
        # run the steps
        if multi_chain:
            run_key, final_key = jax.random.split(rng_key, 2)
            L_step_size_adaptation_keys = jax.random.split(run_key, (num_chains, num_steps1 + num_steps2))

            run_steps = jax.jit(jax.vmap(run_steps, in_axes=((None, 0, None), 0, None), axis_name='chain'))
        else:
            #* Original behavior for only one chain
            L_step_size_adaptation_keys = jax.random.split(
                rng_key, num_steps1 + num_steps2 + 1
            )
            L_step_size_adaptation_keys, final_key = (
                L_step_size_adaptation_keys[:-1],
                L_step_size_adaptation_keys[-1],
            )

        carry, samples = run_steps(
            (mask, L_step_size_adaptation_keys, inverse_mass_matrix_tiled), state, params
        )
        state, params, _, welford_state = carry

        L = params.L
        # determine L
        if multi_chain:
            if not jnp.all(jnp.isclose(params.inverse_mass_matrix-params.inverse_mass_matrix[0], 0.0)):
                print("Inverse mass matrix somehow changed between chains. Make sure this is intended behavior. Pooling all covariances to generate overall inverse mass matrix")
                wel_state = welford_init(state.position.shape[-1])
                for i in range(num_chains):
                    wel_state = welford_combine(wel_state, WelfordAlgorithmState(welford_state.mean[i], welford_state.m2[i], welford_state.sample_size[i]))
                welford_state = wel_state
                inverse_mass_matrix, _, _ = welford_cov(welford_state)
                # inverse_mass_matrix, _, _ = welford_cov(WelfordAlgorithmState(welford_state.mean[0], welford_state.m2[0], welford_state.sample_size[0]))
                
                
            else:
                #! Should change this in a smart way??
                inverse_mass_matrix = params.inverse_mass_matrix[0]
            params = params._replace(inverse_mass_matrix=inverse_mass_matrix)

            print(params.step_size)
            
            #* Take means of each chain's params
            params = params._replace(
                step_size=jnp.mean(params.step_size), 
                L = jnp.mean(params.L),
            )



        
        if num_steps2 > 1:
            #* Samples should be of shape (num_chains, num_steps, dim)
            
            #* Guess at how to replace old calculation of L
            #* Using eigenvalues instead of elements of diagonal matrix
            # L = jnp.sqrt(jnp.sum(jnp.real(jnp.linalg.eig(inverse_mass_matrix)[0])))

            L = jnp.sqrt(dim)

            if mass_matrix_adapt:
                if multi_chain:
                    #* Merge all samples to take covariance of all of them
                    samples = jnp.transpose(samples, (2, 0, 1)) #* Now of shape (dim, num_chains, num_steps)
                    mask = jnp.tile(mask, (num_chains, 1)) #* Should be of shape (num_chains, num_steps)
                    samples = samples.reshape((dim, num_chains * num_steps_net))
                    mask = mask.reshape((num_chains* num_steps_net,))

                    #! Put back in eventually?
                    # welford_mapped = jax.vmap(welford_cov)
                    # covs, sample_sizes, means = welford_mapped(welford_state)

                    # #* Put together covarainces from each chain
                    # cov_wel, mean = aggregate_covariance(covs, sample_sizes, means)
                    pass
                else:
                    # samples = samples.T
                    pass
                # print(welford_state.mean)
                # print(jnp.average(samples, weights=mask, axis=1))
                
                egval, egvec = jnp.linalg.eig(inverse_mass_matrix)
                print(f"Stage 2 Welford Cov Egval Mean: {jnp.mean(egval)}, Min: {jnp.min(egval)}, Max: {jnp.max(egval)}")
                inverse_mass_matrix_smp = jnp.cov(samples, aweights=mask)
                egval, egvec = jnp.linalg.eig(inverse_mass_matrix_smp)
                print(f"Stage 2 Cov Egval Mean: {jnp.mean(egval)}, Min: {jnp.min(egval)}, Max: {jnp.max(egval)}")
                print(f"Stage 2 Step Size: {params.step_size}, L: {params.L}")
                # if not continuous_adaptation:
                #     params = params._replace(inverse_mass_matrix=inverse_mass_matrix)

            # readjust the stepsize
            steps = round(num_steps2 / 3)  # we do some small number of steps
            if multi_chain:
                keys = jax.random.split(final_key, (num_chains, steps))
            else:
                keys = jax.random.split(final_key, steps)

            #! This also runs mass matrix adaptation. Think about if I really want to do that
            inverse_mass_matrix_tiled = jnp.tile(params.inverse_mass_matrix, (steps, 1, 1))
            state, params, _, welford_state = run_steps(
                (jnp.zeros(steps), keys, inverse_mass_matrix_tiled), state, params 
            )[0]

        return state, MCLMCAdaptationState(L, jnp.mean(params.step_size), inverse_mass_matrix)

    return L_step_size_adaptation

def make_adaptation_L(kernel, frac, Lfactor, multi_chain=True, num_chains=8, mass_matrix_adapt=True):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps_3 = round(num_steps * frac)

        def step(state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                L=params.L,
                step_size=params.step_size,
            )

            return next_state, next_state.position

        if multi_chain:
            adaptation_L_keys = jax.random.split(key, (num_chains, num_steps_3))
            mapped_scan = jax.vmap(lambda state_in, keys : jax.lax.scan(f=step, init=state_in, xs=keys), in_axes=(0, 0))

            state, samples = mapped_scan(state, adaptation_L_keys)

            #* Calculate per-chain ESS, 
            #* ESS fucntion squeezes shapes so the .reshape just ensures that any 1s in the shape are preserved 
            ess = effective_sample_size(samples[None, ...], chain_axis=0, sample_axis=2).reshape((num_chains, samples.shape[-1])) 
            chain_min_ess = jnp.min(ess, axis=0) #* Take mean along chain axis
            if jnp.min(chain_min_ess) < 10:
                print(f"Min ESS: {jnp.min(chain_min_ess)} < 10, L adapt may be iffy")
            L_new = Lfactor * params.step_size * num_steps_3 / jnp.min(chain_min_ess)

            samples_flat = jnp.transpose(samples, (2, 0, 1)).reshape(samples.shape[-1], num_chains*num_steps_3)

        else:
            adaptation_L_keys = jax.random.split(key, num_steps_3)
            state, samples = jax.lax.scan(
                f=step,
                init=state,
                xs=adaptation_L_keys,
            )

        
            ess = effective_sample_size(samples[None, ...])
            print(f"Stage 3 ESS Mean: {jnp.mean(ess)}, Min: {jnp.min(ess)}, Max: {jnp.max(ess)}")
            print("L Before Final Update:", params.L)
            L_new=Lfactor * params.step_size * num_steps_3 / jnp.mean(ess)
        
            samples_flat = samples.T
        

        # if mass_matrix_adapt:
        #     inverse_mass_matrix = jnp.cov(samples_flat)
        #     params = params._replace(inverse_mass_matrix = inverse_mass_matrix)

        return state, params._replace(L=L_new)

    return adaptation_L


# def aggregate_covariance(covs, sample_sizes, means):
#     cov_net = covs[0]
#     mu_net = means[0]
#     n_net = sample_sizes[0]
    
#     for i in range(1, len(covs)):
#         cov = covs[i]
#         mu = means[i]
#         n = sample_sizes[i]

#         delta_mu = mu - mu_net
#         delta_prod=jnp.outer(delta_mu, delta_mu)
        
#         cov_net = 1/(n_net+n-1) * ((n_net-1)*cov_net + (n-1)*cov + (n_net*n)/(n_net+n) * delta_prod)
#         mu_net = (n_net * mu_net + n*mu)/(n_net+n)
#         n_net = n_net + n
#     return cov_net, mu_net

def aggregate_m2(m2s, sample_sizes, means):
    n = jnp.sum(sample_sizes)
    pooled_mean = jnp.sum(sample_sizes[:, None]*means, axis=0)/n

    delta_sum = jnp.sum(sample_sizes[:, None, None]*jnp.einsum('ij,ik->ijk', means, means), axis=0)
    pooled_mean_prod = jnp.outer(pooled_mean, pooled_mean)

    m2_net = jnp.sum(m2s, axis=0) + delta_sum- n*pooled_mean_prod

    return m2_net, pooled_mean

def aggregate_chain_welford(welford_state, chain_axis='chain'):
    """Aggregates welford state across axes"""
    means, m2s, sample_sizes = welford_state
    n = jax.lax.psum(sample_sizes, axis_name=chain_axis)
    pooled_mean = jax.lax.psum(sample_sizes[None]*means, axis_name=chain_axis)/n

    delta_sum = jax.lax.psum(sample_sizes[None, None]*jnp.einsum('j,k->jk', means, means), axis_name=chain_axis)
    pooled_mean_prod = jnp.outer(pooled_mean, pooled_mean)

    m2_net = jax.lax.psum(m2s, axis_name=chain_axis) + delta_sum- n*pooled_mean_prod

    return WelfordAlgorithmState(pooled_mean, m2_net, n)

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

def aggregate_covariance(covs, sample_sizes, means):
    #* Join covariances by algorithm found here:
    #* https://stats.stackexchange.com/questions/655028/combining-mean-and-covariance-matrix-of-two-populations#:~:text=Your%20problem:%20The%20case%20you,%CB%89x2)T%5D.
    n = jnp.sum(sample_sizes)
    m2s = sample_sizes[:, None, None] * covs
    m2_net, pooled_mean = aggregate_m2(m2s, sample_sizes, means)
    
    cov_net = 1/(n-1) * m2_net

    return cov_net, pooled_mean