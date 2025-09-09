"""
Helper functions for GigaLens inference and visualization.

This module provides utility functions for:
1. Running inference pipelines (MAP, SVI, HMC) on gravitational lensing systems
2. Visualizing results through various plots:
   - Image comparisons (true vs predicted)
   - Residual analysis and Gaussianity tests
   - Loss histories for optimization
   - Corner plots for parameter distributions
3. Computing diagnostics like chi-squared statistics and noise maps
4. Parameter manipulation and indexing

The functions here streamline the workflow of fitting lens models and analyzing their results.
"""

from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
# from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.jax.simulator import LensSimulator
import jax
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from scipy.stats import norm, kstest
import corner as corner
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import os

def params_jax_to_lists(params_jax):
    """Convert nested parameter structure of JAX arrays to nested Python lists"""
    params_list = []
    for i in range(len(params_jax)):
        params_list.append([])
        for j in range(len(params_jax[i])):
            params_list[i].append({})
            for key in params_jax[i][j]:
                params_list[i][j][key] = params_jax[i][j][key].tolist()
    return params_list

def params_lists_to_jax(params_list):
    """Convert nested parameter structure of Python lists back to JAX arrays"""
    params = []
    for i in range(len(params_list)):
        params.append([])
        for j in range(len(params_list[i])):
            params[i].append({})
            for key in params_list[i][j]:
                params[i][j][key] = jnp.array(params_list[i][j][key])
    return params

def index_params(pars, i):
    #* gets the params just for the ith system
    o1 = []
    for i1 in pars:
        o2 = []
        for i2 in i1:
            out_d = {}
            for k in i2:
                out_d[k] = i2[k][i:i+1]
            o2.append(out_d)
        o1.append(o2)
            
    return o1

class PipelineConfig:
    """
    Configuration class for the GigaLens inference pipeline.

    This class encapsulates the configuration settings for the inference pipeline, including:
    - Which steps to run (MAP, SVI, HMC)
    - Number of optimization steps for each step
    - Optimizers for MAP and SVI stages
    - Other parameters for the three stages of the pipeline
    """
    def __init__(self, steps=["MAP", "SVI", "HMC"], 
            map_steps=350, map_n_samples=500, map_optimizer=None,
            n_vi=1000, svi_steps=1500, svi_start=None, tree_struct=None, precision_parameterization=False, svi_optimizer=None,
            hmc_burnin_steps=250, hmc_num_results=750, n_hmc=50, qz=None, init_eps=0.3, init_l=3):
        
        self.total_devices = jax.device_count()
        self.tree_struct = tree_struct
        
        self.steps = steps
        if "MAP" in steps:
            self.map_steps = map_steps
            self.map_n_samples = map_n_samples
            # Default MAP optimizer
            if map_optimizer is None:
                # schedule_fn = optax.polynomial_schedule(init_value=-1e-2, end_value=-1e-2/3,
                #                                 power=0.5, transition_steps=500)
                # self.map_optimizer = optax.chain(
                #     optax.scale_by_adam(),
                #     optax.scale_by_schedule(schedule_fn),
                # )

                self.map_optimizer = optax.adabelief(1e-2, b1=0.95, b2=0.99, nesterov=True)
            else:
                self.map_optimizer = map_optimizer
                
        if "SVI" in steps:
            self.n_vi = n_vi
            self.svi_steps = svi_steps
            self.precision_parameterization = precision_parameterization
            # Default SVI optimizer
            if svi_optimizer is None:
                # schedule_fn = optax.polynomial_schedule(init_value=-1e-6, end_value=-3e-3,
                #                                     power=2, transition_steps=300)
                # self.svi_optimizer = optax.chain(
                #     optax.scale_by_adam(),
                #     optax.scale_by_schedule(schedule_fn),
                # )
                self.svi_optimizer = optax.adabelief(1e-4, b1=0.95, b2=0.99)
            else:
                self.svi_optimizer = svi_optimizer

            if "MAP" not in steps:
                if svi_start is None:
                    raise ValueError("svi_start must be provided if MAP is not run")
                self.svi_start = svi_start

        if "HMC" in steps:
            self.hmc_burnin_steps = hmc_burnin_steps
            self.hmc_num_results = hmc_num_results
            self.n_hmc = n_hmc
            self.init_eps = init_eps
            self.init_l = init_l
            if "SVI" not in steps:
                if qz is None:
                    raise ValueError("qz must be provided if SVI is not run")
                self.qz = qz

#* All result objects should be simple and pickleable automatically
class MAPResults:
    """
    Results class for the MAP stage of the inference pipeline.

    This class encapsulates the results of the MAP stage, including:
    - The best-fit parameters
    - The chi-squared loss history (the minumum loss for each step)
    - The time taken to run the MAP stage

    It detects the implementation, and extracts these results from the returned values of the MAP function, which differ between implementations.
    """


    def __init__(self, MAP_estimate, MAP_chisq_hist, time_taken, model_seq, pipeline_config, from_save=False):
        if from_save:
            if type(model_seq).__name__ == "ModellingSequenceMultinode":
                _, tree_struct = jax.tree.flatten(model_seq.prob_model.prior.sample(1,seed=jax.random.PRNGKey(0)))
                best_z = jax.tree.unflatten(tree_struct, list(MAP_estimate.reshape((-1, 22)).T))
                select_index = lambda x: x[0]
                best_z = jax.tree.map(select_index, best_z)
                best_x = model_seq.prob_model.bijector.forward(best_z)
            else:
                best_z = MAP_estimate.reshape((-1, 22))
                best_x = model_seq.prob_model.bij.forward(list(best_z.T))
        elif type(model_seq).__name__ == "ModellingSequence":
            best_z, best_x = self.init_GL1(MAP_estimate, MAP_chisq_hist, model_seq, pipeline_config)
        elif type(model_seq).__name__ == "ModellingSequenceMultinode":
            best_z, best_x = self.init_GL2(MAP_estimate, MAP_chisq_hist, model_seq, pipeline_config)
        elif type(model_seq).__name__ == "HarryModellingSequence":
            best_z, best_x, MAP_chisq_hist = self.init_GLH(MAP_estimate, MAP_chisq_hist, model_seq, pipeline_config)
        else:
            raise ValueError(f"Invalid model sequence type {type(model_seq).__name__}")

        self.MAP_chisq_hist = MAP_chisq_hist

        self.best_z = best_z
        self.MAP_best = best_x
        self.time_taken = time_taken

    def init_GL1(self, MAP_estimate, MAP_chisq_hist, model_seq, pipeline_config):
        """
        Process the MAP results from GIGALens 1.0. (Original implementation)

        This function takes in the parameter values for each of the MAP particles, and pick the one with the highest log probability.
        It then converts the best-fit parameters to the physical space
        Args:
            MAP_estimate: 2-D array of shape (n_particles, n_params). The parameters for each particle in the unconstrained space
            MAP_chisq_hist: 1-D array of shape (n_steps) Not currently used
            model_seq: The model sequence object, used to get the prob_model and phys_model
            pipeline_config: The pipeline configuration object, used to get the number of samples

        Returns:
            best: 1-D array of shape (n_params) The parameters for the best-fit particle in the unconstrained space
            map_best_x: pytree of the normal parameter structure with leaves of shape (1,). The best-fit parameters in the physical space
        """
        prob_model = model_seq.prob_model
        phys_model = model_seq.phys_model
        sim_config = model_seq.sim_config

        lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=pipeline_config.map_n_samples), MAP_estimate)[0]
        best = MAP_estimate[jnp.nanargmax(lps)][jnp.newaxis,:] #! nanargmax is very important
        map_best_x = prob_model.bij.forward(list(best.T))

        return best, map_best_x
    
    def init_GL2(self, MAP_estimate, MAP_chisq_hist, model_seq, pipeline_config):
        """
        Process the MAP results from GIGALens 2.0 2024. (Nico's multi-node implementation)

        This function takes in the parameter values for each of the MAP particles, and pick the one with the highest log probability.
        It then converts the best-fit parameters to the physical space

        Args:
            MAP_estimate: 2-D array of shape (n_particles, n_params). The parameters for each particle in the unconstrained space
            MAP_chisq_hist: 1-D array of shape (n_steps) Not currently used
            model_seq: The model sequence object, used to get the prob_model and phys_model
            pipeline_config: The pipeline configuration object, used to get the number of samples and number of devices

        Returns:
            best: 1-D array of shape (n_params) The parameters for the best-fit particle in the unconstrained space
            map_best_x: pytree of the normal parameter structure with leaves of shape (1,). The best-fit parameters in the physical space
        """
        prob_model = model_seq.prob_model
        phys_model = model_seq.phys_model
        sim_config = model_seq.sim_config

        n_samples_s = (pipeline_config.map_n_samples // pipeline_config.total_devices) * pipeline_config.total_devices
        lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=n_samples_s), MAP_estimate)[0] 
        select_index = lambda x: x[jnp.nanargmax(lps)]
        best = jax.tree.map(select_index, MAP_estimate)
        map_best_x = prob_model.bijector.forward(best)

        return best, map_best_x
    
    def init_GLH(self, MAP_estimate, MAP_chisq_hist, model_seq, pipeline_config):
        """
        Process the MAP results from GIGALens 2.0 2025. (Harry's implementation)

        This function takes in the parameter values for each of the MAP particles AT EACH STEP, and picks the one with the highest log probability.
        It then converts the best-fit parameters to the physical space

        Args:
            MAP_estimate: 3-D array of shape (n_steps, n_particles, n_params). The parameters for each particle in the unconstrained space
            MAP_chisq_hist: 2-D array of shape (n_steps, n_particles). The chi-squared loss for each particle at each step
            model_seq: The model sequence object, used to get the prob_model and phys_model
            pipeline_config: The pipeline configuration object, used to get the number of samples and number of devices
        
        Returns:
            best: 1-D array of shape (n_params) The parameters for the best-fit particle in the unconstrained space
            map_best_x: pytree of the normal parameter structure with leaves of shape (1,). The best-fit parameters in the physical space
            map_loss_history: 1-D array of shape (n_steps). The minimum chi-squared loss for each step
        """
        prob_model = model_seq.prob_model
        phys_model = model_seq.phys_model
        sim_config = model_seq.sim_config

        map_loss_history = jnp.min(MAP_chisq_hist, axis=1)
        best_step_idx = jnp.argmin(map_loss_history)
        best_sample_idx = jnp.argmin(MAP_chisq_hist[best_step_idx])

        best = MAP_estimate[best_step_idx][best_sample_idx][jnp.newaxis, :].reshape((-1, 22))
        map_best_x = prob_model.bij.forward(list(best.T))

        # MAP_estimate = MAP_estimate[-1]

        # lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=pipeline_config.map_n_samples), MAP_estimate)[0]
        # best = MAP_estimate[jnp.nanargmax(lps)][jnp.newaxis,:] #! nanargmax is very important
        # map_best_x = prob_model.bij.forward(list(best.T))

        return best, map_best_x, map_loss_history
    
    def save(self, results_dir):
        best_z = jax.experimental.multihost_utils.process_allgather(self.best_z)
        chisq_hist = jax.experimental.multihost_utils.process_allgather(np.squeeze(self.MAP_chisq_hist))
        if jax.process_index() == 0:
            np.save(os.path.join(results_dir, 'map_best_z.npy'), best_z)
            np.save(os.path.join(results_dir, 'map_losses.npy'), chisq_hist)
    
    @classmethod
    def load(cls, results_dir, model_seq):
        map_best_z = np.load(os.path.join(results_dir, 'map_best_z.npy'))
        map_losses = np.squeeze(np.load(os.path.join(results_dir, 'map_losses.npy')))
        return cls(map_best_z, map_losses, 0, model_seq, None, from_save=True)

class SVIResults:
    """
    Results class for the SVI stage of the inference pipeline.

    This class encapsulates the results of the SVI stage, including:
    - The surrogate posterior distribution
    - The mean of the surrogate posterior distribution
    - A set of samples from the surrogate posterior distribution
    - The ELBO loss history
    - The time taken to run the SVI stage

    It detects the implementation, and extracts these results from the returned values of the SVI function, which differ between implementations.
    """
    def __init__(self, qz, SVI_loss_hist, time_taken, model_seq, pipeline_config, n_samples=1000, from_save=False):

        if from_save and type(model_seq).__name__ == "ModellingSequenceMultinode":
            #* Too much of a pain to pass the tree struct through right now.
            SVI_mean = None
            svi_samples_x = None
        elif type(model_seq).__name__ == "ModellingSequenceMultinode":
            svi_samples_x, SVI_mean = self.init_GL2(qz, model_seq, pipeline_config, n_samples)
        elif type(model_seq).__name__ == "ModellingSequence" or type(model_seq).__name__ == "HarryModellingSequence":
            svi_samples_x, SVI_mean = self.init_GL1(qz, model_seq, pipeline_config, n_samples)
        else:
            raise ValueError(f"Invalid model sequence type {type(model_seq).__name__}")
        
        self.qz = qz
        self.SVI_mean = SVI_mean
        self.SVI_samples = svi_samples_x
        self.SVI_loss_hist = SVI_loss_hist
        self.time_taken = time_taken
    
    def init_GL1(self, qz, model_seq, pipeline_config, n_samples=1000):
        """
        Process the SVI results from GIGALens 1.0. (Original implementation).
        Also used for Harry's implementation, as the SVI returns the same format as GIGALens 1.0.

        Args:
            qz: The surrogate posterior distribution returned by the SVI function
            model_seq: The model sequence object, used to get the prob_model
            pipeline_config: The pipeline configuration object (not used)
            n_samples: The number of samples to draw from the surrogate posterior distribution

        Returns:
            svi_samples_x: A pytree of the normal parameter structure with leaves of shape (n_samples,). The samples from the surrogate posterior distribution
            SVI_mean: A pytree of the normal parameter structure with leaves of shape (1,). The mean of the surrogate posterior distribution
        """
        prob_model = model_seq.prob_model

        svi_samples_z = qz.sample(n_samples, seed=jax.random.PRNGKey(0))
        svi_samples_x = prob_model.bij.forward(list(svi_samples_z.T))

        SVI_mean = prob_model.bij.forward(list(qz.mean().T))

        return svi_samples_x, SVI_mean
    
    def init_GL2(self, qz, model_seq, pipeline_config, n_samples=1000):
        """
        Process the SVI results from GIGALens 2.0 2024. (Nico's multi-node implementation)
        Differs from the original implementation in that the unconstrained parameters are also expected to be in the tree structure.

        Args:
            qz: The surrogate posterior distribution returned by the SVI function
            model_seq: The model sequence object, used to get the prob_model
            pipeline_config: The pipeline configuration object, used to get the parameter tree structure
            n_samples: The number of samples to draw from the surrogate posterior distribution

        Returns:
            svi_samples_x: A pytree of the normal parameter structure with leaves of shape (n_samples,). The samples from the surrogate posterior distribution
            SVI_mean: A pytree of the normal parameter structure with leaves of shape (1,). The mean of the surrogate posterior distribution
        """
        prob_model = model_seq.prob_model

        mean = prob_model.bijector.forward(jax.tree.unflatten(pipeline_config.tree_struct, qz.mean()))

        svi_samples_z = qz.sample(n_samples, seed=jax.random.PRNGKey(0))
        svi_samples_x = prob_model.bijector.forward(jax.tree.unflatten(pipeline_config.tree_struct, svi_samples_z.T))

        return svi_samples_x, mean
    
    def save(self, results_dir):
        if jax.process_index() == 0:
            jnp.save(os.path.join(results_dir, 'loss_history.npy'), jnp.array(self.SVI_loss_hist))
            jnp.save(os.path.join(results_dir, 'qz_scale_tril.npy'), self.qz.scale_tril)
            jnp.save(os.path.join(results_dir, 'qz_loc.npy'), self.qz.loc)

    @classmethod
    def load(cls, results_dir, model_seq):
        loss_hist = np.load(os.path.join(results_dir, 'loss_history.npy'))
        qz_scale_tril = np.load(os.path.join(results_dir, 'qz_scale_tril.npy'))
        qz_loc = np.load(os.path.join(results_dir, 'qz_loc.npy'))
        qz = tfd.MultivariateNormalTriL(loc=qz_loc, scale_tril=qz_scale_tril)
        return cls(qz, loss_hist, 0, model_seq, None, from_save=True)

class HMCResults:
    """
    Results class for the HMC stage of the inference pipeline.

    This class encapsulates the results of the HMC stage, including:
    - The samples from the posterior distribution in the physical space
    - The samples from the posterior distribution in the unconstrained space
    - The median of the posterior distribution in the physical space
    - The R-hat statistic
    - The time taken to run the HMC stage

    It detects the implementation, and extracts these results from the returned values of the HMC function, which differ between implementations.
    """
    def __init__(self, samples, time_taken, model_seq):

        prob_model = model_seq.prob_model
        if type(model_seq).__name__ == "ModellingSequence":
            HMC_samples, HMC_median, rhat, HMC_samples_z = self.init_GL1(samples, model_seq)
        elif type(model_seq).__name__ == "ModellingSequenceMultinode":
            HMC_samples, HMC_median, rhat, HMC_samples_z = self.init_GL2(samples, model_seq)
            # print("Multinode HMC Result Processing isn't implemented yet")
        elif type(model_seq).__name__ == "HarryModellingSequence":
            HMC_samples, HMC_median, rhat, HMC_samples_z = self.init_GLH(samples, model_seq)
        else:
            raise ValueError(f"Invalid model sequence type {type(model_seq).__name__}")

        self.HMC_samples = HMC_samples
        self.HMC_median = HMC_median
        self.HMC_rhat = rhat
        self.time_taken = time_taken
        self.HMC_samples_z = HMC_samples_z

    def init_GLH(self, samples, model_seq):
        """
        Process the HMC results from GIGALens 2.0 2025. (Harry's implementation)

        Args:
            samples: The samples from the HMC chain. 4-D array of shape (num_devices, num_chains_per_device, num_steps, n_params)
            model_seq: The model sequence object, used to get the prob_model

        Returns:
            HMC_samples: A pytree of the normal parameter structure with leaves of shape (total_hmc_samples,). The samples from the posterior distribution in the physical space
            HMC_median: A pytree of the normal parameter structure with leaves of shape (1,). The median of the posterior distribution in the physical space
            rhat: And array of shape (n_params). The R-hat statistic for each parameter
            HMC_samples_z: The samples from the posterior distribution in the unconstrained space. 4-D array of shape (num_devices, num_chains_per_device, num_steps, n_params)
        """
        prob_model = model_seq.prob_model
        phys_model = model_seq.phys_model
        sim_config = model_seq.sim_config

        rhat= tfp.mcmc.potential_scale_reduction(jnp.transpose(samples, (1,2,0,3)), independent_chain_ndims=2)
    
        #* Return the results of HMC
        smp = jnp.transpose(samples, (1, 2, 0, 3)).reshape((-1, 22))
        HMC_samples = prob_model.bij.forward(list(smp.T))

        HMC_median = prob_model.bij.forward(list(np.median(smp,axis=0)))
        # lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=smp.shape[0]), smp)[0]
        # best = smp[jnp.nanargmax(lps)][jnp.newaxis,:] #! nanargmax is very important
        # HMC_best = prob_model.bij.forward(list(best.T))

        return HMC_samples, HMC_median, rhat, samples

    def init_GL1(self, samples, model_seq):
        """
        Process the HMC results from GIGALens 1.0. (Original implementation)
        Works the same as for Harry's implementation, except that samples is an object which has an attribute (all_states) containing the samples.
        """
        return self.init_GLH(samples.all_states, model_seq)
    
    def init_GL2(self, samples, model_seq):
        print("Multinode HMC Result Processing isn't implemented yet")
        return None, None, None, None
    
    def save(self, results_dir):
        if jax.process_index() == 0:
            np.save(os.path.join(results_dir, 'hmc_samples_z.npy'), self.HMC_samples_z)
    
    @classmethod
    def load(cls, results_dir, model_seq):
        samples = np.load(os.path.join(results_dir, 'hmc_samples_z.npy'))
        return cls(samples, 0, model_seq)


def gather_Nico_HMC_samples(samples, tree_struct, num_results, total_devices, n_hmc_gpu):
    mesh = jax.sharding.Mesh(jax.devices(), 'devices') 
    partition_spec_hmc = jax.sharding.PartitionSpec(None, 'devices')
    sharding_hmc = jax.sharding.NamedSharding(mesh, partition_spec_hmc) 
    shard_hmc_fn = lambda samples_gpu: jax.make_array_from_single_device_arrays((num_results, total_devices * n_hmc_gpu), sharding_hmc, [samples_gpu])

    samples = samples.all_states.transpose((2, 0, 1)) # (22, 750, 8)
    samples_gpu = jax.tree.unflatten(tree_struct, samples)

    sharded_samples_hmc = jax.tree.map(shard_hmc_fn, samples_gpu)

    return jax.experimental.multihost_utils.process_allgather(sharded_samples_hmc)

def run_pipeline(model_seq, pipeline_config):
    """
    Execute the GigaLens inference pipeline with configurable stages.
    
    Runs the three-stage gravitational lens modeling pipeline:
    1. MAP: Gradient-based optimization to find best-fit parameters
    2. SVI: Variational inference to approximate posterior with Gaussian surrogate
    3. HMC: Hamiltonian Monte Carlo sampling for full posterior characterization
    
    Parameters
    ----------
    model_seq : ModellingSequence
        The modeling sequence object containing the physical model, probabilistic model,
        and simulation configuration and the functions for each stage of the pipeline
    pipeline_config : PipelineConfig
        Configuration object specifying which stages to run and their parameters
        
    Returns
    -------
    dict
        Dictionary containing results from executed stages:
        - "MAP": MAPResults object (if MAP was run)
        - "SVI": SVIResults object (if SVI was run) 
        - "HMC": HMCResults object (if HMC was run)
        
    Notes
    -----
    - MAP results are used as starting point for SVI
    - SVI results (surrogate posterior) are used as starting point for HMC
    - Each stage can be run independently if proper starting values are provided in the pipeline config
    """
    

    cfg = pipeline_config

    run_map = "MAP" in cfg.steps
    run_svi = "SVI" in cfg.steps
    run_hmc = "HMC" in cfg.steps

    results = {}

    #* RUNNING MAP---------------------------------
    if run_map:
        print("Starting MAP")
        
        start = time.perf_counter()
        map_estimate, map_chisq_hist = model_seq.MAP(cfg.map_optimizer, seed=0, num_steps=cfg.map_steps, n_samples=cfg.map_n_samples) #num_steps=350
        end = time.perf_counter()
        
        results["MAP"] = MAPResults(map_estimate, map_chisq_hist, end - start, model_seq, pipeline_config)
    
    #* RUNNING SVI---------------------------------
    if run_svi:
        print("Starting SVI")
        
        best_z = results["MAP"].best_z if run_map else cfg.svi_start

        if type(model_seq).__name__ == "ModellingSequenceMultinode":
            _, tree_struct = jax.tree.flatten(best_z)
            pipeline_config.tree_struct = tree_struct

        start = time.perf_counter()
        qz, svi_loss_hist = model_seq.SVI(best_z, cfg.svi_optimizer, n_vi=cfg.n_vi, num_steps=cfg.svi_steps, 
                                          precision_parameterization=cfg.precision_parameterization)
        end = time.perf_counter()
        
        results["SVI"] = SVIResults(qz, svi_loss_hist, end - start, model_seq, pipeline_config)
    
    #* RUNNING HMC---------------------------------
    if run_hmc:
        print("Starting HMC")

        qz = qz if run_svi else cfg.qz

        start = time.perf_counter()
        if type(model_seq).__name__ == "ModellingSequenceMultinode":

            mean = jax.device_get(qz.loc)
            scale_tril = jax.device_get(qz.scale_tril)

            qz_unsharded = tfd.MultivariateNormalTriL(loc=mean, scale_tril=scale_tril)

            samples = model_seq.HMC(
                qz_unsharded, tree_struct=cfg.tree_struct, num_burnin_steps=cfg.hmc_burnin_steps, num_results=cfg.hmc_num_results, n_hmc=cfg.n_hmc,
                init_eps=cfg.init_eps, init_l=cfg.init_l
            )
            samples = gather_Nico_HMC_samples(samples, cfg.tree_struct, cfg.hmc_num_results, jax.device_count(), cfg.n_hmc//jax.device_count())
        else:
            samples = model_seq.HMC(qz, num_burnin_steps=cfg.hmc_burnin_steps, num_results=cfg.hmc_num_results, n_hmc=cfg.n_hmc)
        end = time.perf_counter()

        results["HMC"] = HMCResults(samples, end - start, model_seq)
    
    return results
    


def simulate_system(observed_img, prior, ModellingSequenceType, sim_config, phys_model, 
    map_steps=350, map_n_samples=500, map_optimizer=None, 
    n_vi=1000, svi_steps=1500, precision_parameterization=False, svi_optimizer=None, 
    n_hmc=50, hmc_burnin_steps=250, hmc_num_results=750, init_eps=0.3, init_l=3):
    """
    Run the complete typical GigaLens inference pipeline on a gravitational lensing system.
    
    This is a convenience wrapper around run_pipeline that:
    1. Creates a ForwardProbModel from the observed image and prior
    2. Instantiates the specified ModellingSequence implementation  
    3. Configures and executes the full MAP → SVI → HMC pipeline
    
    Parameters
    ----------
    observed_img : array-like
        A 2-D array. The image of the lensing system to fit
    prior : tfd.Distribution
        A tensorflow_probability distribution object. The prior distribution for the model parameters
    ModellingSequenceType : class
        The ModellingSequence class to instantiate (e.g., ModellingSequence, ModellingSequenceMultinode, HarryModellingSequence)
    sim_config : SimulatorConfig object
        Configuration settings for the lens simulator.
    phys_model : PhysicalModel object
        Physical model describing the lens system.
    map_steps : int, optional
        Number of optimization steps for MAP (default: 350)
    map_n_samples : int, optional
        Number of parallel particles for MAP optimization (default: 500)
    map_optimizer : optax.GradientTransformation, optional
        Optimizer for MAP stage. If None, uses default polynomial schedule with Adam (default: None)
    n_vi : int, optional
        In SVI, number of samples to draw from the surrogate posterior for the ELBO calculation (default: 1000)
    svi_steps : int, optional
        Number of optimization steps for SVI (default: 1500)
    precision_parameterization : bool, optional
        In SVI, whether to parameterize the surrogate posterior using the precision matrix instead of the covariance matrix (default: False)
    svi_optimizer : optax.GradientTransformation, optional
        Optimizer for SVI stage. If None, uses default polynomial schedule with Adam (default: None)
    n_hmc : int, optional
        Number of HMC chains to run in parallel (default: 50)
    hmc_burnin_steps : int, optional
        Number of burn-in steps for HMC (default: 250)
    hmc_num_results : int, optional
        Number of posterior samples to collect from HMC per chain (default: 750)
     
    Returns
    -------
    dict
        Contains results from all inference stages:
        - MAP: MAPResults object
        - SVI: SVIResults object
        - HMC: HMCResults object
    """
    prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
    model_seq = ModellingSequenceType(phys_model, prob_model, sim_config)
    
    pipeline_config = PipelineConfig(map_steps=map_steps, map_n_samples=map_n_samples, map_optimizer=map_optimizer,
                                     n_vi=n_vi, svi_steps=svi_steps, precision_parameterization=precision_parameterization, svi_optimizer=svi_optimizer,
                                     hmc_burnin_steps=hmc_burnin_steps, hmc_num_results=hmc_num_results, n_hmc=n_hmc, init_eps=init_eps, init_l=init_l)
    
    results = run_pipeline(model_seq, pipeline_config)
    
    return results


def get_noise_image(image, background_rms, exp_time):
    return np.sqrt(image / exp_time + background_rms**2)

def get_chisq(true_img, predicted_img, background_rms=0.2, exp_time=100):
    emap = get_noise_image(predicted_img, background_rms, exp_time)

    return np.sum(np.square((true_img-predicted_img)/emap))

def plot_image(fig, ax, img, extent=None, title=None, residual=False, colorbar=True):
    """
    Plot an image using my chosen standards for coloring, 
    which changes depending on whether the image is a residual or not.
    """
    if not residual:
        #* Meaning actual lensing image
        cnorm = matplotlib.colors.Normalize(vmin=0)
        cmap = 'inferno'
    else:
        #* Meaning residual image
        cnorm = matplotlib.colors.CenteredNorm()
        cmap = 'bwr'
    
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)


    im = ax.imshow(img, cmap=cmap, norm=cnorm, extent=extent, origin='lower')
    if colorbar:
        fig.colorbar(im, cax=cax)
    if title is not None:
        ax.set_title(title)
    
    ax.set_xlim((extent[0], extent[1]))
    ax.set_ylim((extent[2], extent[3]))
    ax.axis('off')
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot
from lenstronomy.Data.imaging_data import ImageData
from astropy.visualization.mpl_normalize import simple_norm

def add_caustics(ax, params, model_seq, lens_objects=['EPL', 'SHEAR']):
    kwargs_data = sim_util.data_configure_simple(model_seq.sim_config.num_pix*40, model_seq.sim_config.delta_pix/20)
    data = ImageData(**kwargs_data)
    _coords = data
    lensModel = LensModel(lens_model_list=lens_objects) #just need a list of the mass parameters, something like ['EPL', 'SHEAR']
    params = jax.tree.map(lambda a : np.array(a), params)
    kwargs_lens = params[0] #the values for the above parameters
    

    lens_plot.caustics_plot(ax, _coords, lensModel, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green')


def histogram_residuals(fig, ax, flat_residual, title, bins=50):
    
    mu, std = norm.fit(flat_residual)
    p = kstest(flat_residual, norm.cdf).pvalue

    dummy_x = np.linspace(np.min(flat_residual), np.max(flat_residual), 100)
    ax.hist(flat_residual, bins=bins, density=True, label=f"mu={mu:.4f} \nstd={std:.4f} \np={p:.4f}")
    ax.plot(dummy_x, norm.pdf(dummy_x, mu, std))
    ax.set_title(title)
    ax.legend()

def plot_image_results(fig, axs, true_img, lens_sim=None, predicted_params=None, 
                       predicted_img=None, resimulate=True, display_true_chisq=False, true_params=None, prefix="",
                       plot_caustics=False, model_seq=None):
    """
    Plot the results of a lensing fit. Given a set of predicted parameters, compare the predicted image to the true image.
    Displays normalized residuals, and a histogram of the residuals to check that they are gaussian noise
    """
    if resimulate:
        if lens_sim is None:
            raise ValueError("lens_sim must be provided if resimulate is True")
        predicted_img = lens_sim.simulate(predicted_params)
    elif predicted_img is None:
        raise ValueError("predicted_img must be provided if resimulate is False")

    if display_true_chisq:
        true_chisq = get_chisq(true_img, lens_sim.simulate(true_params))
    
    noise_map = get_noise_image(true_img, 0.2, 100)

    residual = (true_img - predicted_img)/noise_map

    chisq = np.sum(np.square(residual))
    dof = true_img.shape[0]*true_img.shape[1] - 22 #! Change if number of params changes
    #! Do I want to do sqrt curve cmap for the images?\
    numPix = model_seq.sim_config.num_pix
    deltaPix = model_seq.sim_config.delta_pix
    extent = (-numPix/2*deltaPix, numPix/2*deltaPix, -numPix/2*deltaPix, numPix/2*deltaPix)
    plot_image(fig, axs[0], true_img, extent=extent,
               title=f"True Image" + (f"(Red Chisq:{true_chisq/dof:.3f})" if display_true_chisq else ""))
    if plot_caustics and (true_params is not None):
        add_caustics(axs[0], true_params, model_seq)
    plot_image(fig, axs[1], predicted_img, extent=extent, title=f"{prefix} Model Fit (Red Chisq:{chisq/dof:.3f})")
    if plot_caustics and (predicted_params is not None):
        add_caustics(axs[1], predicted_params, model_seq)
    plot_image(fig, axs[2], residual, extent=extent, title=f"{prefix} Normalized Residual", residual=True)

    if display_true_chisq:
        print("True Chisq", true_chisq)
        print("Model Fit Chisq", chisq)
    flat_residual = residual.flatten()
    histogram_residuals(fig, axs[3], flat_residual, f"{prefix} Global Gaussianity Test")
    

def plot_loss_histories(fig, axs, map_chisq_hist, svi_loss_hist):
    """
    Plot the loss histories of the MAP and SVI stages of the inference pipeline.
    """
    axs[0].plot(map_chisq_hist)
    axs[0].set_title("MAP Loss History")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Chi-squared Loss")
    axs[0].set_ylim(bottom=0, top=3)

    axs[1].plot(svi_loss_hist)
    axs[1].set_title("SVI Loss History")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("ELBO")

def cornerplot_labels(example_params):
    """
    Generate the labels for the cornerplot based on the tree structure of the parameters.
    """
    tups = [(0, 0), (0, 1), (1, 0), (2, 0)]
    # get labels and pts for the MAP
    label_prefixes = ['', '', 'lens_', 'src_']
    labels = []
    
    for (i, j), label_prefix in zip(tups, label_prefixes):
        labels.extend((label_prefix + key for key in example_params[i][j].keys()))
    return labels

def flatten_label_order(tree):
    tups = [(0, 0), (0, 1), (1, 0), (2, 0)]
    flat = []
    for (i, j) in tups:
       flat.extend((arr.item() for arr in tree[i][j].values()))
    flat = np.array(flat)
    return flat

def cornerplot_posterior(labels, raw_samples, fig=None, truth=None, overplots=None, color='black', truth_color='black', overplot_color='red'):
    """
    Create a cornerplot of the a set of samples in the physical space.
    Option to overplot a single point, such as the MAP best fit
    Can also overplot a second point as crossed vertical and horizontal lines (most often the truth or median of the samples)
    """
    tups = [(0, 0), (0, 1), (1, 0), (2, 0)]

    if overplots is not None:
        overplot_pts = []
        for (i, j) in tups:
            overplot_pts.extend((arr.item() for arr in overplots[i][j].values()))
        overplot_pts = np.array(overplot_pts)

    if truth is not None:
        truth_overplot_pts = []
        for (i,j) in tups:
            truth_overplot_pts.extend((arr.item() for arr in truth[i][j].values()))
        truth_overplot_pts = np.array(truth_overplot_pts)
    else:
        truth_overplot_pts = None

    samples = np.vstack([np.array(list(raw_samples[i][j].values())) for i, j in tups]).T
    histargs = {'density': True, 'color': color}
    
    fig = corner.corner(samples, fig=fig, truths=truth_overplot_pts, truth_color=truth_color, 
        show_titles=True, title_fmt='.3f',
        labels=labels, hist_kwargs=histargs, color=color)

    if overplots is not None:
        corner.overplot_points(fig, overplot_pts[np.newaxis], marker='*', markersize=12, mfc=overplot_color, mec=overplot_color)
    
    return fig

def cornerplot_results(map_best, svi_samples=None, HMC_samples=None, true_params=None, hmc_median=None):
    """
    Cornerplot of the results of the inference pipeline, including MAP, SVI, and HMC.
    """
    labels = cornerplot_labels(map_best)

    fig = cornerplot_posterior(labels, svi_samples, truth=true_params, overplots=map_best, color='blue', truth_color='green', overplot_color='red')
    cornerplot_posterior(labels, HMC_samples, fig=fig, truth=hmc_median)

def get_errors_diff(HMC_samples, true_params):
    
    lower_err = jax.tree.map(lambda x: -(jnp.percentile(x, 16)-jnp.median(x)), HMC_samples)
    upper_err = jax.tree.map(lambda x: jnp.percentile(x, 84)-jnp.median(x), HMC_samples)
    median = jax.tree.map(lambda x: jnp.median(x), HMC_samples)

    median_diff = jax.tree.map(lambda x, y: x-y, median, true_params)

    flat_lower_err = flatten_label_order(lower_err)
    flat_upper_err = flatten_label_order(upper_err)
    flat_median_diff = flatten_label_order(median_diff)

    return flat_median_diff, flat_lower_err, flat_upper_err

def normalize_residuals(median_diff, upper_err, lower_err):
    pos_res = median_diff > 0
    scale = np.zeros_like(median_diff)
    scale[pos_res] = lower_err[pos_res]
    scale[~pos_res] = upper_err[~pos_res]

    residual_norm = median_diff/scale
    chisq = 1/(residual_norm.shape[0]-1) * np.sum(np.square(residual_norm))
    return residual_norm, chisq

def residualplot_params(save_dirs, true_params_all, prob_models):
    median_diffs = []
    upper_errs = []
    lower_errs = []
    for i, save_dir in enumerate(save_dirs):
        select_index = lambda a : a[i]
        true_params = jax.tree.map(select_index, true_params_all)
        samples = np.load(os.path.join(save_dir, 'hmc_samples_z.npy'))
        smp = jnp.transpose(samples, (1, 2, 0, 3)).reshape((-1, 22))
        HMC_samples = prob_models[i].bij.forward(list(smp.T))
        diff, low, high = get_errors_diff(HMC_samples, true_params)
        median_diffs.append(diff)
        lower_errs.append(low)
        upper_errs.append(high)
    
    median_diffs = jnp.array(median_diffs)
    upper_errs = jnp.array(upper_errs)
    lower_errs = jnp.array(lower_errs)



    labels = cornerplot_labels(HMC_samples)
    n_params = len(labels)
    fig, axs = plt.subplots(n_params//3 + 1, 3)
    fig.set_size_inches(40,50)
    axs = axs.flatten()

    num_systems = len(save_dirs)
    for i, label in enumerate(labels):
        axs[i].errorbar(jnp.arange(num_systems), median_diffs[:, i], yerr=[lower_errs[:, i], upper_errs[:, i]], fmt='o', linestyle='')
        axs[i].axhline(y=0, color='black', linestyle=':', alpha=0.5)
        axs[i].set_title(label)

    plt.show()

    fig, axs = plt.subplots(n_params//3 + 1, 3)
    fig.set_size_inches(20,25)
    axs = axs.flatten()

    for i, label in enumerate(labels):
        z_scores, chisq = normalize_residuals(median_diffs[:, i], upper_errs[:, i], lower_errs[:, i])
        outliers = jnp.where(jnp.abs(z_scores) > 5)[0]
        if len(outliers) > 0:
            print(f"{label} has outliers at indices: {outliers}")
        histogram_residuals(fig, axs[i], z_scores, f'{label}, chisq: {chisq:.3f}', bins=10)


def display_results(r, true_img, lens_sim, true_params=None, save_dir=None, 
    show=True, make_cornerplot=True, plot_caustics=False, model_seq=None):
    """
    Display all results of the inference pipeline, including:
    - Comparing predicted images to true images (MAP best sample and HMC median)
    - Plotting the loss histories of the MAP and SVI stages
    - Plotting the cornerplot of the results, including the MAP best fit, SVI samples, and HMC samples
    """
    
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(12,3)
    plot_image_results(fig, axs, true_img, prefix="MAP",
                       lens_sim=lens_sim, predicted_params=r['MAP'].MAP_best, 
                       resimulate=True, true_params=true_params, plot_caustics=plot_caustics, model_seq=model_seq)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'map_results.png'))
    if show:
        plt.show()
    plt.close(fig)

    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(12,3)
    plot_image_results(fig, axs, true_img, prefix="HMC",
                       lens_sim=lens_sim, predicted_params=r['HMC'].HMC_median, 
                       resimulate=True, true_params=true_params, plot_caustics=plot_caustics, model_seq=model_seq)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'hmc_results.png'))
    if show:
        plt.show()
    plt.close(fig)
    
    fig, axs = plt.subplots(1, 2)
    plot_loss_histories(fig, axs, r['MAP'].MAP_chisq_hist, r['SVI'].SVI_loss_hist)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'loss_histories.png'))
    if show:
        plt.show()
    plt.close(fig)
    
    if make_cornerplot:
        # HMC_samp_reduced = jax.random.choice(jax.random.PRNGKey(0), r['HMC'].HMC_samples, (1000,), replace=False)
        cornerplot_results(r['MAP'].MAP_best, r['SVI'].SVI_samples, r['HMC'].HMC_samples, true_params=true_params, hmc_median=r['HMC'].HMC_median)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'cornerplot.png'))
        if show:
            plt.show()
        plt.close()

def make_default_prior():
    """
    Make the default prior from the original GIGALens paper.
    """
    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
                    gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
                    e1=tfd.Normal(0, 0.1),
                    e2=tfd.Normal(0, 0.1),
                    center_x=tfd.Normal(0, 0.05),
                    center_y=tfd.Normal(0, 0.05),
                )
            ),
            tfd.JointDistributionNamed(
                dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
            ),
        ]
    )
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
                    n_sersic=tfd.Uniform(2, 6),
                    e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                    e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                    center_x=tfd.Normal(0, 0.05),
                    center_y=tfd.Normal(0, 0.05),
                    Ie=tfd.LogNormal(jnp.log(300.0), 0.3),
                )
            )
        ]
    )

    source_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
                    n_sersic=tfd.Uniform(0.5, 4),
                    e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                    center_x=tfd.Normal(0, 0.25),
                    center_y=tfd.Normal(0, 0.25),
                    Ie=tfd.LogNormal(jnp.log(150.0), 0.5),
                )
            )
        ]
    )

    prior = tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )
    return prior