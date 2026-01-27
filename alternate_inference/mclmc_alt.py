
import jax.numpy as jnp
import jax
import blackjax
from blackjax.mcmc.integrators import GeneralIntegrator, IntegratorState, ArrayTree, Callable, euclidean_position_update_fn
from blackjax.mcmc.integrators import generalized_two_stage_integrator, format_isokinetic_state_output, ravel_pytree, _normalized_flatten_array
from blackjax.mcmc.integrators import mclachlan_coefficients, yoshida_coefficients, omelyan_coefficients

import time
#* Define new isokinetic mclachlan. Only difference is it now takes in 2D, non-diagonal inverse matrix

def generate_isokinetic_integrator_smart(coefficients):
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
        raise ValueError("Your mass matrix isn't a matrix, dumbass.")
    
    chol_inverse_mass_matrix = jnp.linalg.cholesky(inverse_mass_matrix)
    # sqrt_inverse_mass_matrix = jnp.sqrt(inverse_mass_matrix)

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
isokinetic_yoshida_smart = generate_isokinetic_integrator_smart(yoshida_coefficients)
isokinetic_omelyan_smart = generate_isokinetic_integrator_smart(omelyan_coefficients)


#* ----- NON-DIAGONAL MASS MATRIX ADAPTATION FOR MCLMC --------------------------------------

from blackjax.adaptation.mclmc_adaptation import pytree_size, MCLMCAdaptationState, handle_nans, incremental_value_update
from blackjax.mcmc.integrators import generalized_two_stage_integrator, format_isokinetic_state_output, ravel_pytree, _normalized_flatten_array
from blackjax.diagnostics import effective_sample_size

def mclmc_find_L_and_step_size_smart(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    multi_chain=False,
    num_chains=8,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    params=None,
    Lfactor=0.4,
    mass_matrix_adapt=True,
):
    """
    Finds the optimal value of the parameters for the MCLMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function used for the MCMC algorithm.
    num_steps
        The number of MCMC steps that will subsequently be run, after tuning.
    state
        The initial state of the MCMC algorithm.
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
        mass_matrix_adapt=mass_matrix_adapt
    )(state, params, num_steps, part1_key)
    total_num_tuning_integrator_steps += num_steps1 + num_steps2

    if num_steps3 >= 2:  # at least 2 samples for ESS estimation
        state, params = make_adaptation_L(
            mclmc_kernel(params.inverse_mass_matrix), frac=frac_tune3, Lfactor=Lfactor, multi_chain=multi_chain, num_chains=num_chains,
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
    mass_matrix_adapt=True
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for unadjusted MCLMC"""

    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

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
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""

        mask, rng_key = weight_and_key
        state, params, adaptive_state, streaming_avg = iteration_state

        state, params, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )

        x = ravel_pytree(state.position)[0]
        # update the running average of x, x^2
        # streaming_avg = incremental_value_update(
        #     expectation=jnp.array([x, jnp.square(x)]),
        #     incremental_val=streaming_avg,
        #     weight=mask * success * params.step_size,
        # )

        return (state, params, adaptive_state, streaming_avg), state.position

    def L_step_size_adaptation(state, params, num_steps, rng_key):

        run_steps = lambda xs, state, params: jax.lax.scan(
            step,
            init=(
                state,
                params,
                (0.0, 0.0, jnp.inf),
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=xs,
        )
        num_steps1, num_steps2 = round(num_steps * frac_tune1), round(
            num_steps * frac_tune2
        )

        # we use the last num_steps2 to compute the diagonal preconditioner
        mask = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))
        num_steps_net = num_steps1 + num_steps2
        # run the steps
        if multi_chain:
            run_key, final_key = jax.random.split(rng_key, 2)
            L_step_size_adaptation_keys = jax.random.split(run_key, (num_chains, num_steps1 + num_steps2))

            run_steps = jax.jit(jax.vmap(run_steps, in_axes=((None, 0), 0, None)))
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
            (mask, L_step_size_adaptation_keys), state, params
        )
        state, params, _, (_, average) = carry

        L = params.L
        # determine L
        inverse_mass_matrix = params.inverse_mass_matrix
        if num_steps2 > 1:
            #* Samples should be of shape (num_chains, num_steps, dim)
            if multi_chain:
                #* Merge all samples to take covariance of all of them

                samples = jnp.transpose(samples, (2, 0, 1)) #* Now of shape (dim, num_chains, num_steps)
                mask = jnp.tile(mask, (num_chains, 1)) #* Should be of shape (num_chains, num_steps)
                samples = samples.reshape((dim, num_chains * num_steps_net))
                mask = mask.reshape((num_chains* num_steps_net,))

                #* Take means of each chain's params
                params = params._replace(
                    step_size=jnp.mean(params.step_size), 
                    L = jnp.mean(params.L),
                )
            else:
                samples = samples.T
            
            #* Guess at how to replace old calculation of L
            #* Using eigenvalues instead of elements of diagonal matrix
            # L = jnp.sqrt(jnp.sum(jnp.real(jnp.linalg.eig(inverse_mass_matrix)[0])))

            L = jnp.sqrt(dim)

            if mass_matrix_adapt:
                inverse_mass_matrix = jnp.cov(samples, aweights=mask)
                egval, egvec = jnp.linalg.eig(inverse_mass_matrix)
                print(f"Stage 2 Cov Egval Mean: {jnp.mean(egval)}, Min: {jnp.min(egval)}, Max: {jnp.max(egval)}")
                params = params._replace(inverse_mass_matrix=inverse_mass_matrix)

            # readjust the stepsize
            steps = round(num_steps2 / 3)  # we do some small number of steps
            if multi_chain:
                keys = jax.random.split(final_key, (num_chains, steps))
            else:
                keys = jax.random.split(final_key, steps)
            state, params, _, (_, average) = run_steps(
                (jnp.ones(steps), keys), state, params
            )[0]

        return state, MCLMCAdaptationState(L, jnp.mean(params.step_size), inverse_mass_matrix)

    return L_step_size_adaptation

def make_adaptation_L(kernel, frac, Lfactor, multi_chain=True, num_chains=8):
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
            chain_mean_ess = jnp.mean(ess, axis=0) #* Take mean along chain axis
            L_new = Lfactor * params.step_size * jnp.max(num_steps_3 / chain_mean_ess)

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
            L_new=Lfactor * params.step_size * num_steps_3 / jnp.mean(ess) #! Essentially now focusing on the worst parameter
        
            samples_flat = samples.T

        inverse_mass_matrix = jnp.cov(samples_flat)

        return state, params._replace(
            L=L_new,
            #inverse_mass_matrix = inverse_mass_matrix #! My change: recondition mass matrix again
        )

    return adaptation_L


#* ------ GIGALens-like wrapper for MCLMC
from mclmc_parallel import init_multi, build_kernel_multi, mclmc_multi

def MCLMC(qz, log_prob, n_hmc=16, num_burnin_steps=5000, num_results=10000, 
        init_L=None, init_step_size=None, progress_bar=False, print_adapt_params=False,seed=0):
    rng_key = jax.random.key(0)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)


    n_chains = n_hmc
    desired_energy_variance= 5e-4 #* Tuning parameter. Keep as is for now
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

    #* Run burnin, which adapts L, step size, and the mass matrix
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
        num_chains=n_chains
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

    return multi_chain_samples
