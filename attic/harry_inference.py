import functools

import jax.random
import optax
from tensorflow_probability.python.internal import unnest
import tensorflow_probability.substrates.jax as tfp
import time
from jax import jit, pmap
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from tensorflow_probability.substrates.jax import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
from tqdm.auto import trange

import gigalens.inference
import gigalens.jax.simulator as sim
import gigalens.model

import numpy as np
import gc

import warnings

if not jax.distributed.is_initialized():
    warnings.warn('jax.distributed.initialize() was not called. For multinode, please call it before running any JAX functions.')
mesh = jax.make_mesh((jax.device_count(),), ('device',))


class ModellingSequence(gigalens.inference.ModellingSequenceInterface):
    def MAP(
            self,
            optimizer: optax.GradientTransformation,
            start=None,
            n_samples=500,
            num_steps=350,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        n_samples = (n_samples // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_samples // dev_cnt,
        )
        seed = jax.random.PRNGKey(seed)

        start = (
            self.prob_model.prior.sample(n_samples, seed=seed)
            if start is None
            else start
        )
        params = jnp.stack(self.prob_model.bij.inverse(start)).T

        opt_state = optimizer.init(params)

        def loss(z):
            lp, chisq = self.prob_model.log_prob(lens_sim, z)
            return -jnp.mean(lp) / jnp.size(self.prob_model.observed_image), chisq

        loss_and_grad = jax.pmap(jax.value_and_grad(loss, has_aux=True))

        def update(params, opt_state):
            splt_params = jnp.array(jnp.split(params, dev_cnt, axis=0))
            (_, chisq), grads = loss_and_grad(splt_params)
            grads = jnp.concatenate(grads, axis=0)
            chisq = jnp.concatenate(chisq, axis=0)

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return chisq, new_params, opt_state

        with trange(num_steps) as pbar:
            for _ in pbar:
                loss, params, opt_state = update(params, opt_state)
                pbar.set_description(
                    f"Chi-squared: {float(jnp.nanmin(loss)):.3f}"
                )
        return params


    
    def MAP_multi(
            self,
            optimizer: optax.GradientTransformation,
            start=None,
            n_samples=500,
            num_steps=350,
            seed=0,
    ):
        
        dev_cnt = jax.device_count()
        n_samples = (n_samples // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=1,
        )
        jax_seed = jax.random.PRNGKey(seed)

        start = (
            self.prob_model.prior.sample(n_samples, seed=jax_seed)
            if start is None
            else start
        )
        params = jnp.stack(self.prob_model.bij.inverse(start)).T
        sharding = NamedSharding(mesh, P('device'))

        def loss(z):
            lp, chisq = self.prob_model.log_prob(lens_sim, z)
            return chisq

        # loss_and_grad = jit(jax.value_and_grad(loss))
        loss_and_grad = jit(jax.vmap(jax.value_and_grad(loss)))
        
        @functools.partial(jax.jit, static_argnums=1)
        @functools.partial(jax.experimental.shard_map.shard_map, mesh=mesh, in_specs=(P('device'), None), out_specs=P('device'))
        def run_map(params, optimizer):
            opt_state = optimizer.init(params)
            pvary = lambda x: jax.lax.pvary(x, 'device') if isinstance(x, jax.Array) else x
            opt_state = jax.tree_util.tree_map(pvary, opt_state)

            def f(carry, b):
                params, opt_state = carry
                loss, grads = loss_and_grad(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                carry = params, opt_state
                b = (params, loss)
                return carry, b

            
            _, b = jax.lax.scan(f, (params, opt_state), length=num_steps)
            # b = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), b)
            # swap_axes (num_steps, num_samples_per_device) -> (num_samples_per_device, num_steps)
            b = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), b)
            return b


        # first 3 dims are shape (num_devices, num_steps, num_samples_per_device)
        # import time
        # import os
        # prev_time = time.time()
        map_samples, map_losses = run_map(params, optimizer)
        # with open(f"{os.environ['HOME']}/gigalens_personal/map_{len(jax.devices())}", 'w') as f:
        #         f.write(str(time.time() - prev_time))

        # reshape so that first 2 dims are (num_steps, num_samples)
        map_samples = jnp.swapaxes(map_samples, 0, 1)
        map_losses = jnp.swapaxes(map_losses, 0, 1)

        return map_samples, map_losses

    
    def MAP_hist(
            self,
            optimizer: optax.GradientTransformation,
            start=None,
            n_samples=500,
            num_steps=350,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        n_samples = (n_samples // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_samples // dev_cnt,
        )
        seed = jax.random.PRNGKey(seed)

        start = (
            self.prob_model.prior.sample(n_samples, seed=seed)
            if start is None
            else start
        )
        params = jnp.stack(self.prob_model.bij.inverse(start)).T

        opt_state = optimizer.init(params)

        def loss(z):
            lp, chisq = self.prob_model.log_prob(lens_sim, z)
            return -jnp.mean(lp) / jnp.size(self.prob_model.observed_image), chisq

        loss_and_grad = jax.pmap(jax.value_and_grad(loss, has_aux=True))

        def update(params, opt_state):
            splt_params = jnp.array(jnp.split(params, dev_cnt, axis=0))
            (_, chisq), grads = loss_and_grad(splt_params)
            grads = jnp.concatenate(grads, axis=0)
            chisq = jnp.concatenate(chisq, axis=0)

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return chisq, new_params, opt_state

        loss_history = []
        params_history = []
        state_history = []
        min_loss = float('inf')
        best_params = params
        min_loss_idxs = None
        with trange(num_steps) as pbar:
            for i in pbar:
                loss, params, opt_state = update(params, opt_state)
                loss_history.append(loss)
                params_history.append(params)
                state_history.append(opt_state)
                best_idx = jnp.nanargmin(loss)
                loss = float(loss[best_idx])
                if loss < min_loss:
                    best_params = params
                    min_loss_idxs = (i, int(best_idx))
                pbar.set_description(
                    f"Chi-squared: {loss:.4f}"
                )
        return best_params, params_history, loss_history, state_history, min_loss_idxs
    

    # adaptation_start_steps is the number of steps after which the optimization will begin resampling bad points
    # resampling will happen every adaptation_interval
    # we will keep percentile good points
    def MAP_alt(
            self,
            optimizer: optax.GradientTransformation,
            adaptation_start_steps,
            adaptation_interval,
            adaptation_cull_percentile,
            start=None,
            n_samples=500,
            num_steps=350,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        n_samples = (n_samples // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_samples // dev_cnt,
        )
        seed = jax.random.PRNGKey(seed)

        start = (
            self.prob_model.prior.sample(n_samples, seed=seed)
            if start is None
            else start
        )
        params = jnp.stack(self.prob_model.bij.inverse(start)).T

        opt_state = optimizer.init(params)

        def loss(z):
            lp, chisq = self.prob_model.log_prob(lens_sim, z)
            return -jnp.mean(lp) / jnp.size(self.prob_model.observed_image), chisq

        loss_and_grad = jax.pmap(jax.value_and_grad(loss, has_aux=True))

        steps = 0
        def update(params, opt_state):
            splt_params = jnp.array(jnp.split(params, dev_cnt, axis=0))
            (_, chisq), grads = loss_and_grad(splt_params)
            grads = jnp.concatenate(grads, axis=0)
            chisq = jnp.concatenate(chisq, axis=0)                    

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            # NEW SECTION: Resample
            replacement_dist = None
            if steps > adaptation_start_steps:
                if (steps - adaptation_start_steps - 1) % adaptation_interval == 0:
                    
                    replace_mask = chisq > jnp.percentile(chisq, adaptation_cull_percentile)
                    nonlocal seed
                    seed, subkey = jax.random.split(seed)

                    # replacement_dist = tfd.MultivariateNormalDiag(
                    #     loc = jnp.mean(new_params[~replace_mask], axis=0),
                    #     scale_diag = jnp.std(new_params[~replace_mask], axis=0),
                    # )
                    
                    # replacement_samples = replacement_dist.sample(
                    #     jnp.sum(replace_mask), 
                    #     seed=subkey
                    # )
                    prior_samples = self.prob_model.prior.sample(jnp.sum(replace_mask).item(), seed=subkey)
                    replacement_samples = jnp.stack(self.prob_model.bij.inverse(prior_samples)).T
                    
                    new_params.at[replace_mask].set(replacement_samples)
                    jax.clear_caches()
                    
            
            return chisq, new_params, opt_state, replacement_dist

        map_history = []
        replacement_dist_history = []
        with trange(num_steps) as pbar:
            for _ in pbar:
                steps += 1
                loss, params, opt_state, replacement_dist = update(params, opt_state)
                if replacement_dist is not None:
                    replacement_dist_history.append(replacement_dist)
                map_history.append(params)
                pbar.set_description(
                    f"Chi-squared: {float(jnp.nanmin(loss, keepdims=True).item()):.3f}"
                )
        return params, map_history, replacement_dist_history

    def SVI(
            self,
            start,
            optimizer: optax.GradientTransformation,
            n_vi=250,
            init_scales=1e-3,
            num_steps=500,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_vi = (n_vi // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi // dev_cnt,
        )
        scale = (
            jnp.diag(jnp.ones(jnp.size(start))) * init_scales
            if jnp.size(init_scales) == 1
            else init_scales
        )
        cov_bij = tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6)
        qz_params = jnp.concatenate(
            [jnp.squeeze(start), cov_bij.inverse(scale)], axis=0
        )
        replicated_params = jax.tree_util.tree_map(lambda x: jnp.array([x] * dev_cnt), qz_params)

        n_params = jnp.size(start)

        def elbo(qz_params, seed):
            mean = qz_params[:n_params]
            cov = cov_bij.forward(qz_params[n_params:])
            qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
            z = qz.sample(n_vi // dev_cnt, seed=seed)
            lps = qz.log_prob(z)
            return jnp.mean(lps - self.prob_model.log_prob(lens_sim, z)[0])

        elbo_and_grad = jit(jax.value_and_grad(jit(elbo), argnums=(0,)))

        @functools.partial(pmap, axis_name="num_devices")
        def get_update(qz_params, seed):
            val, grad = elbo_and_grad(qz_params, seed)
            return jax.lax.pmean(val, axis_name="num_devices"), jax.lax.pmean(
                grad, axis_name="num_devices"
            )[0]

        opt_state = optimizer.init(replicated_params)
        loss_hist = []
        with trange(num_steps) as pbar:
            for step in pbar:
                # losses = []
                # grads = []
                # for _ in range(3):
                #     loss, grad = get_update(replicated_params, seeds)
                #     seeds = jax.random.split(seeds[0], dev_cnt)
                #     losses.append(loss)
                #     grads.append(grad)
                # loss = jnp.mean(jnp.stack(losses), axis=0)
                # grads = jnp.mean(jnp.stack(grads), axis=0)
                loss, grads = get_update(replicated_params, seeds)
                    
                # loss_1, grads_1 = get_update(replicated_params, seeds)
                # seeds = jax.random.split(seeds[0], dev_cnt)
                # loss_2, grads_2 = get_update(replicated_params, seeds)
                # loss = (loss_1 + loss_2) / 2
                # grads = (grads_1 + grads_2) / 2
                
                loss = float(jnp.mean(loss))
                seeds = jax.random.split(seeds[0], dev_cnt)
                updates, opt_state = optimizer.update(grads, opt_state)
                replicated_params = optax.apply_updates(replicated_params, updates)
                pbar.set_description(f"ELBO: {loss:.3f}")
                loss_hist.append(loss)
        mean = replicated_params[0, :n_params]
        cov = cov_bij.forward(replicated_params[0, n_params:])
        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
        return qz, loss_hist

    def SVI_multi(
            self,
            start,
            optimizer: optax.GradientTransformation,
            n_vi=250,
            init_scales=1e-3,
            num_steps=500,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        jax_seed = jax.random.PRNGKey(seed)
        sharding = NamedSharding(mesh, P('device'))

        n_vi = (n_vi // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi // dev_cnt
        )
        scale = (
            jnp.diag(jnp.ones(jnp.size(start))) * init_scales
            if jnp.size(init_scales) == 1
            else init_scales
        )
        cov_bij = tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6)
        qz_params = jnp.concatenate(
            [jnp.squeeze(start), cov_bij.inverse(scale)], axis=0
        )
        replicated_params = jax.tree_util.tree_map(lambda x: jnp.array(x), qz_params)

        n_params = jnp.size(start)

        def elbo(qz_params, jax_seed):
            mean = qz_params[:n_params]
            cov = cov_bij.forward(qz_params[n_params:])
            qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
            # jax_seed is (1, 2) due to sharding, want shape (2)
            z = qz.sample(n_vi // dev_cnt, seed=jax_seed[0])
            lps = qz.log_prob(z)
            return jnp.mean(lps - self.prob_model.log_prob(lens_sim, z)[0])

        elbo_and_grad = jit(jax.value_and_grad(jit(elbo), argnums=(0,)))

        @jit
        @functools.partial(jax.experimental.shard_map.shard_map, mesh=mesh, in_specs=(None, P('device')), out_specs=P())
        def get_update(qz_params, jax_seed):
            val, grad = elbo_and_grad(qz_params, jax_seed)
            return jax.lax.pmean(val, axis_name="device"), jax.lax.pmean(
                grad, axis_name="device"
            )[0]

        opt_state = optimizer.init(replicated_params)
        loss_hist = []
        min_loss = float('inf')
        best_params = replicated_params
        for step in range(num_steps):
            jax_seed, curr_seed = jax.random.split(jax_seed)
            jax_seeds = jax.random.split(curr_seed, dev_cnt)
            loss, grads = get_update(replicated_params, jax_seeds)
            loss = float(jnp.mean(loss))

            if loss < min_loss:
                best_params = replicated_params
                min_loss = loss
            
            updates, opt_state = optimizer.update(grads, opt_state)
            replicated_params = optax.apply_updates(replicated_params, updates)
            loss_hist.append(loss)
        
        
        mean = best_params[:n_params]
        cov = cov_bij.forward(best_params[n_params:])
        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
        return qz, loss_hist

        
        
        

    def SVI_hist(
            self,
            start,
            optimizer: optax.GradientTransformation,
            n_vi=250,
            init_scales=1e-3,
            num_steps=500,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_vi = (n_vi // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi // dev_cnt,
        )
        scale = (
            jnp.diag(jnp.ones(jnp.size(start))) * init_scales
            if jnp.size(init_scales) == 1
            else init_scales
        )
        cov_bij = tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6)
        qz_params = jnp.concatenate(
            [jnp.squeeze(start), cov_bij.inverse(scale)], axis=0
        )
        replicated_params = jax.tree_util.tree_map(lambda x: jnp.array([x] * dev_cnt), qz_params)

        n_params = jnp.size(start)

        def elbo(qz_params, seed):
            mean = qz_params[:n_params]
            cov = cov_bij.forward(qz_params[n_params:])
            qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
            z = qz.sample(n_vi // dev_cnt, seed=seed)
            lps = qz.log_prob(z)
            return jnp.mean(lps - self.prob_model.log_prob(lens_sim, z)[0])

        elbo_and_grad = jit(jax.value_and_grad(jit(elbo), argnums=(0,)))

        @functools.partial(pmap, axis_name="num_devices")
        def get_update(qz_params, seed):
            val, grad = elbo_and_grad(qz_params, seed)
            return jax.lax.pmean(val, axis_name="num_devices"), jax.lax.pmean(
                grad, axis_name="num_devices"
            )[0]

        opt_state = optimizer.init(replicated_params)
        params_hist = []
        loss_hist = []
        min_loss = float('inf')
        best_params = replicated_params[0]
        with trange(num_steps) as pbar:
            for step in pbar:
                loss, grad = get_update(replicated_params, seeds)
                
                loss = float(jnp.mean(loss))
                curr_params = replicated_params[0]
                if loss < min_loss:
                    # best_replicated_params = jax.tree_util.tree_map(jnp.copy, replicated_params)
                    best_params = curr_params
                    min_loss = loss
                    
                seeds = jax.random.split(seeds[0], dev_cnt)
                updates, opt_state = optimizer.update(grads, opt_state)
                replicated_params = optax.apply_updates(replicated_params, updates)
                pbar.set_description(f"ELBO: {loss:.3f}")
                params_hist.append((curr_params[:n_params], cov_bij.forward(curr_params[n_params:])))
                loss_hist.append(loss)
        mean = best_params[0, :n_params]
        cov = cov_bij.forward(best_params[0, n_params:])
        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
        return qz, loss_hist, params_hist

    def SVI_hist_adaptive(
            self,
            start,
            optimizer: optax.GradientTransformation,
            n_vi=250,
            init_scales=1e-3,
            num_steps=500,
            seed=0,
    ):
        n_vi_progression = [200, 1000, 2000, 3000]
        progression_idx = 0
        patience = 28
        wait_beginning_steps = 50
        steps_since_last_improvement = 0
        n_vi = n_vi_progression[progression_idx]
        progression_min_loss = float('inf')
        
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_vi = (n_vi // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi // dev_cnt,
        )
        scale = (
            jnp.diag(jnp.ones(jnp.size(start))) * init_scales
            if jnp.size(init_scales) == 1
            else init_scales
        )
        cov_bij = tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6)
        qz_params = jnp.concatenate(
            [jnp.squeeze(start), cov_bij.inverse(scale)], axis=0
        )
        replicated_params = jax.tree_util.tree_map(lambda x: jnp.array([x] * dev_cnt), qz_params)

        n_params = jnp.size(start)

        def elbo(qz_params, seed, n_vi, lens_sim):
            mean = qz_params[:n_params]
            cov = cov_bij.forward(qz_params[n_params:])
            qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
            z = qz.sample(n_vi // dev_cnt, seed=seed)
            lps = qz.log_prob(z)
            return jnp.mean(lps - self.prob_model.log_prob(lens_sim, z)[0])

        # elbo_and_grad = jit(jax.value_and_grad(jit(elbo), argnums=(0,)))
        # elbo_and_grad = jit(jax.value_and_grad(jit(lambda qz_params, seed: elbo(qz_params, seed, n_vi, lens_sim)), argnums=(0,)))

        # @functools.partial(pmap, axis_name="num_devices")
        # def get_update(qz_params, seed):
        #     val, grad = elbo_and_grad(qz_params, seed)
        #     return jax.lax.pmean(val, axis_name="num_devices"), jax.lax.pmean(
        #         grad, axis_name="num_devices"
        #     )

        def make_get_update(n_vi, lens_sim):
            elbo_and_grad = jit(jax.value_and_grad(
                lambda qz_params, seed: elbo(qz_params, seed, n_vi, lens_sim), argnums=0
            ))
        
            @functools.partial(pmap, axis_name="num_devices")
            def get_update(qz_params, seed):
                val, grad = elbo_and_grad(qz_params, seed)
                return jax.lax.pmean(val, axis_name="num_devices"), jax.lax.pmean(
                    grad, axis_name="num_devices"
                )
        
            return get_update
        get_update = make_get_update(n_vi, lens_sim)

        opt_state = optimizer.init(replicated_params)
        params_hist = []
        loss_hist = []
        min_loss = float('inf')
        best_params = replicated_params[0]
        with trange(num_steps) as pbar:
            for step in pbar:
                losses = []
                grads = []
                for _ in range(5):
                    loss, grad = get_update(replicated_params, seeds)
                    seeds = jax.random.split(seeds[0], dev_cnt)
                    losses.append(loss)
                    grads.append(grad)
                loss = jnp.mean(jnp.stack(losses), axis=0)
                grads = jnp.mean(jnp.stack(grads), axis=0)
                
                loss = float(jnp.mean(loss))
                # scale mean gradients
                # grads = jax.lax.dynamic_update_slice(grads, grads[:, :n_params] * 100, (0, 0))
                curr_params = replicated_params[0]
                if loss < min_loss:
                    # best_replicated_params = jax.tree_util.tree_map(jnp.copy, replicated_params)
                    best_params = curr_params
                    min_loss = loss
                if loss < progression_min_loss:
                    progression_min_loss = loss
                    steps_since_last_improvement = 0
                else:
                    steps_since_last_improvement += 1
                    if step > wait_beginning_steps and steps_since_last_improvement > patience and progression_idx < len(n_vi_progression) - 1:
                        steps_since_last_improvement = 0
                        progression_min_loss = loss
                        progression_idx += 1
                        n_vi = n_vi_progression[progression_idx]
                        n_vi = (n_vi // dev_cnt) * dev_cnt
                        lens_sim.bs = n_vi // dev_cnt
                        _, _, img_X, img_Y = lens_sim.get_coords(
                            lens_sim.supersample, self.sim_config.num_pix, np.array(lens_sim.transform_pix2angle)
                        )
                        lens_sim.img_X = jnp.repeat(img_X[..., jnp.newaxis], lens_sim.bs, axis=-1)
                        lens_sim.img_Y = jnp.repeat(img_Y[..., jnp.newaxis], lens_sim.bs, axis=-1)
                        # lens_sim.img_X = jnp.repeat(lens_sim.img_X, 2, axis=-1)
                        # lens_sim.img_Y = jnp.repeat(lens_sim.img_Y, 2, axis=-1)
                        # lens_sim = sim.LensSimulator(
                        #     self.phys_model,
                        #     self.sim_config,
                        #     bs=n_vi // dev_cnt,
                        # )
                        get_update = make_get_update(n_vi, lens_sim)
                        print(f'New n_vi: {n_vi}')
                    
                # (mean, cov)
                params_hist.append((curr_params[:n_params], cov_bij.forward(curr_params[n_params:])))
                seeds = jax.random.split(seeds[0], dev_cnt)
                updates, opt_state = optimizer.update(grads, opt_state)
                replicated_params = optax.apply_updates(replicated_params, updates)
                pbar.set_description(f"ELBO: {loss:.3f}")
                loss_hist.append(loss)
        mean = best_params[:n_params]
        cov = cov_bij.forward(best_params[n_params:])
        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
        return qz, loss_hist, params_hist

    def HMC(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            num_burnin_steps=250,
            num_results=750,
            max_leapfrog_steps=30,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc // dev_cnt,
        )
        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

        @pmap
        def run_chain(seed):
            start = q_z.sample(n_hmc // dev_cnt, seed=seed)
            num_adaptation_steps = int(num_burnin_steps * 0.8)
            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l,
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps,
            )
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
            )

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                trace_fn=lambda _, pkr: None,
                seed=seed,
                kernel=mc_kernel,
            )

        start = time.time()
        ret = run_chain(seeds)
        end = time.time()
        print(f"Sampling took {(end - start):.1f}s")
        return ret

    
    def HMC_multi(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            num_burnin_steps=250,
            num_results=750,
            max_leapfrog_steps=30,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        local_dev_cnt = jax.local_device_count()
        # seeds are per process (node)
        seeds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), jax.process_index()), local_dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc // dev_cnt,
        )
        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

        @pmap
        def run_chain(seed):
            start = q_z.sample(n_hmc // dev_cnt, seed=seed)
            num_adaptation_steps = int(num_burnin_steps * 0.8)
            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l,
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps,
            )
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
            )

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                trace_fn=lambda _, pkr: None,
                seed=seed,
                kernel=mc_kernel,
            )

        start = time.time()
        samples = run_chain(seeds)
        end = time.time()
        # aggregate over all devices

        # process_mesh = jax.make_mesh((local_dev_cnt,), ('local_device',))
        # sharding = jax.sharding.NamedSharding(process_mesh, P('local_device', None, None, None)) 
        # print(f'{samples.all_states.shape=}')
        # process_samples = jax.make_array_from_process_local_data(sharding, samples.all_states)
        # print(f'{process_samples.shape=}')
        # all_samples is (num_processes, num_devices_per_process, num_steps, n_hmc_per_device, 22)
        all_samples = jax.experimental.multihost_utils.process_allgather(samples.all_states)
        # print(all_samples.reshape(all_samples.shape[0] * all_samples.shape[1], *all_samples.shape[2:]).shape)
        # reshape to (num_devices, num_steps, n_hmc_per_device, , 22), then swap num_steps, n_hmc_per_device
        return jnp.swapaxes(all_samples.reshape(all_samples.shape[0] * all_samples.shape[1], *all_samples.shape[2:]), 1, 2)

    def HMC_alt_multi(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            n_vi=1000,
            num_burnin_steps=250,
            num_results=750,
            max_leapfrog_steps=30,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        local_dev_cnt = jax.local_device_count()
        seeds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), jax.process_index()), local_dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc // dev_cnt,
        )
        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

        @pmap
        def run_burnin_chain(seed):
            start = q_z.sample(n_hmc // dev_cnt, seed=seed)
            num_adaptation_steps = int(num_burnin_steps * 0.8)

            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                inner_kernel=mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps
            )

            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel,
                num_adaptation_steps=num_adaptation_steps
            )
            
            results = tfp.mcmc.sample_chain(
                num_results=num_burnin_steps,
                current_state=start,
                kernel=mc_kernel,
                trace_fn=None,
                return_final_kernel_results=True,
                seed=seed
            )    
            kernel_results = results.final_kernel_results
            step_size = unnest.get_innermost(kernel_results, 'step_size')
            num_leapfrog_steps = unnest.get_innermost(kernel_results, 'num_leapfrog_steps')
            
            return results.all_states, step_size, num_leapfrog_steps
            
        
        @pmap
        def run_chain(seed, dev_idx):            
            final_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=step_size[dev_idx],
                num_leapfrog_steps=num_leapfrog_steps[dev_idx]
            )

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                current_state=all_states[dev_idx, -1],
                kernel=final_kernel,
                trace_fn=lambda _, pkr: None,
                seed=seed
            ).all_states
            

        start = time.time()
        if num_burnin_steps > 0:
            # tuple: (all_states, step_size, num_leapfrog_steps), dim 0 of each tensor is device idx 
            all_states, step_size, num_leapfrog_steps = run_burnin_chain(seeds)
            # all_samples is (num_processes, num_devices_per_process, num_steps, n_hmc_per_device, 22)
            all_samples = jax.experimental.multihost_utils.process_allgather(all_states)
            
            all_samples = all_samples.reshape(-1, 22)
            mle_cov = jnp.cov(all_samples, rowvar=False)
            proposed_normal_distribution = tfd.MultivariateNormalFullCovariance(
                loc=all_samples.mean(axis=0),
                covariance_matrix=mle_cov,
            )

            elbo_lens_sim = sim.LensSimulator(
                self.phys_model,
                self.sim_config,
                bs=1,
            )

            # pick lower elbo distribution
            @functools.partial(jit, static_argnums=(0,))
            def elbo(qz):
                z = qz.sample(n_vi, seed=jax.random.PRNGKey(0))
                lps = qz.log_prob(z)
                return jnp.mean(lps - self.prob_model.log_prob(elbo_lens_sim, z)[0])

            burnin_elbo = elbo(proposed_normal_distribution)
            q_z_elbo = elbo(q_z)
            print(f'Burn-in elbo: {burnin_elbo}')
            print(f'q_z elbo: {q_z_elbo}')
            if burnin_elbo < q_z_elbo:
                momentum_distribution = tfd.MultivariateNormalFullCovariance(
                loc=jnp.zeros_like(q_z.mean()),
                covariance_matrix=jnp.linalg.inv(mle_cov),
            )
                print('Switched to burnin model')
        

        dev_idxs = jnp.arange(local_dev_cnt)
        all_states = run_chain(seeds, dev_idxs)
        all_samples = jax.experimental.multihost_utils.process_allgather(all_states)
        all_samples = jnp.swapaxes(all_samples.reshape(all_samples.shape[0] * all_samples.shape[1], *all_samples.shape[2:]), 1, 2)
        
        end = time.time()
        print(f"Sampling took {(end - start):.1f}s")
        return all_samples, mle_cov    # mle_cov for debugging only

    def HMC_alt(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            num_burnin_steps=250,
            num_results=750,
            max_leapfrog_steps=30,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc // dev_cnt,
        )
        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

        @pmap
        def run_burnin_chain(seed):
            start = q_z.sample(n_hmc // dev_cnt, seed=seed)
            num_adaptation_steps = int(num_burnin_steps * 0.8)

            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                inner_kernel=mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps
            )

            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel,
                num_adaptation_steps=num_adaptation_steps
            )
            
            results = tfp.mcmc.sample_chain(
                num_results=num_burnin_steps,
                current_state=start,
                kernel=mc_kernel,
                trace_fn=None,
                return_final_kernel_results=True,
                seed=seed
            )    
            kernel_results = results.final_kernel_results
            step_size = unnest.get_innermost(kernel_results, 'step_size')
            num_leapfrog_steps = unnest.get_innermost(kernel_results, 'num_leapfrog_steps')
            
            return results.all_states, step_size, num_leapfrog_steps
            
        
        @pmap
        def run_chain(seed, dev_idx):            
            final_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=step_size[dev_idx],
                num_leapfrog_steps=num_leapfrog_steps[dev_idx]
            )

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                current_state=all_states[dev_idx, -1],
                kernel=final_kernel,
                trace_fn=lambda _, pkr: None,
                seed=seed
            )
            

        start = time.time()
        # tuple: (all_states, step_size, num_leapfrog_steps), dim 0 of each tensor is device idx 
        all_states, step_size, num_leapfrog_steps = run_burnin_chain(seeds)
        
        burnin_samples = jnp.transpose(all_states, (1, 2, 0, 3)).reshape((-1, 22))
        mle_cov = jnp.cov(burnin_samples, rowvar=False)

        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(mle_cov),
        )

        dev_idxs = jnp.arange(dev_cnt)
        ret = run_chain(seeds, dev_idxs)
        
        end = time.time()
        print(f"Sampling took {(end - start):.1f}s")
        return ret, mle_cov    # mle_cov for debugging only
