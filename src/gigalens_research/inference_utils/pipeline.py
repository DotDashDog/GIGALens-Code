from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.jax.simulator import LensSimulator
import jax
import tensorflow_probability.substrates.jax as tfp
import os
import time
tfd = tfp.distributions

class PipelineConfig:
    """
    Configuration class for the GigaLens inference pipeline.

    This class encapsulates the configuration settings for the inference pipeline, including:
    - Which steps to run (MAP, SVI, HMC)
    - Number of optimization steps for each step
    - Optimizers for MAP and SVI stages
    - Other parameters for the three stages of the pipeline
    """
    # def __init__(self, steps=["MAP", "SVI", "HMC"], 
    #         map_steps=350, map_n_samples=500, map_optimizer=None,
    #         n_vi=1000, svi_steps=1500, svi_start=None, precision_parameterization=False, svi_optimizer=None,
    #         hmc_burnin_steps=250, hmc_num_results=750, n_hmc=50, qz=None, init_eps=0.3, init_l=3):
    
    def __init__(self, steps=["MAP", "SVI", "HMC"],
        map_kwargs={}, map_func=None, svi_kwargs={}, svi_func=None, hmc_kwargs={}, hmc_func=None):
        
        self.total_devices = jax.device_count()
        self.map_kwargs = map_kwargs
        self.svi_kwargs = svi_kwargs
        self.hmc_kwargs = hmc_kwargs
        self.steps = steps
        self.map_func = map_func
        self.svi_func = svi_func
        self.hmc_func = hmc_func

        if "MAP" in steps:
            # Default MAP optimizer
            if 'optimizer' not in map_kwargs or map_kwargs['optimizer'] is None:
                map_kwargs['optimizer'] = optax.adabelief(1e-2, b1=0.95, b2=0.99, nesterov=True)
                
        if "SVI" in steps:
            # Default SVI optimizer
            if 'optimizer' not in svi_kwargs or svi_kwargs['optimizer'] is None:
                svi_kwargs['optimizer'] = optax.adabelief(1e-4, b1=0.95, b2=0.99)

            if "MAP" not in steps:
                if 'start' not in svi_kwargs:
                    raise ValueError("SVI must be given a starting point if MAP is not run")

        if "HMC" in steps:
            if "SVI" not in steps:
                if 'qz' not in hmc_kwargs:
                    raise ValueError("qz must be provided if SVI is not run")

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

    def __init__(self, MAP_estimate, MAP_chisq_hist, time_taken, model_seq, from_save=False):
        best_z = jnp.squeeze(MAP_estimate)[jnp.newaxis]
        best_x = model_seq.prob_model.bij.forward(list(best_z.T))

        self.MAP_chisq_hist = MAP_chisq_hist

        self.best_z = best_z
        self.MAP_best = best_x
        self.time_taken = time_taken
    
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
        return cls(map_best_z, map_losses, -1, model_seq, from_save=True)

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
    def __init__(self, qz, SVI_loss_hist, time_taken, model_seq, n_samples=1000):
        # svi_samples_x, SVI_mean = self.init_GL1(qz, model_seq, n_samples)

        prob_model = model_seq.prob_model

        svi_samples_z = qz.sample(n_samples, seed=jax.random.PRNGKey(0))
        svi_samples_x = prob_model.bij.forward(list(svi_samples_z.T))

        SVI_mean = prob_model.bij.forward(list(qz.mean().T))

        self.qz = qz
        self.SVI_mean = SVI_mean
        self.SVI_samples = svi_samples_x
        self.SVI_loss_hist = SVI_loss_hist
        self.time_taken = time_taken
        
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
        return cls(qz, loss_hist, -1, model_seq)

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
    def __init__(self, samples_z, time_taken, model_seq):

        # HMC_samples, HMC_median, rhat, HMC_samples_z = self.init_GL2(samples, model_seq)
        prob_model = model_seq.prob_model

        rhat= tfp.mcmc.potential_scale_reduction(jnp.transpose(samples_z, (2, 0, 1 ,3)), independent_chain_ndims=2)
    
        #* Return the results of HMC
        smp = samples_z.reshape((-1, 22))
        HMC_samples = prob_model.bij.forward(list(smp.T))

        HMC_median = prob_model.bij.forward(list(np.median(smp,axis=0)))

        self.HMC_samples = HMC_samples
        self.HMC_median = HMC_median
        self.HMC_rhat = rhat
        self.time_taken = time_taken
        self.HMC_samples_z = samples_z
    
    def save(self, results_dir):
        if jax.process_index() == 0:
            np.save(os.path.join(results_dir, 'hmc_samples_z.npy'), self.HMC_samples_z)
    
    @classmethod
    def load(cls, results_dir, model_seq):
        samples = np.load(os.path.join(results_dir, 'hmc_samples_z.npy'))
        if len(samples.shape) == 3:
            #* If it was saved with the shape (num_hmc, num_steps, n_params)
            samples = samples[np.newaxis] #* Add devices dimension
        return cls(samples, -1, model_seq)

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

        if cfg.map_func is None:
            map_func = model_seq.MAP
        else:
            map_func = cfg.map_func
        
        map_kwargs = cfg.map_kwargs.copy()
        optimizer = map_kwargs['optimizer']
        map_kwargs.pop('optimizer')
        
        start = time.perf_counter()
        map_estimate, map_chisq, lp = map_func(**cfg.map_kwargs)
        end = time.perf_counter()
        
        results["MAP"] = MAPResults(map_estimate, map_chisq, end - start, model_seq)
    
    #* RUNNING SVI---------------------------------
    if run_svi:
        print("Starting SVI")
        
        if cfg.svi_func is None:
            svi_func = model_seq.SVI
        else:
            svi_func = cfg.svi_func
        
        svi_kwargs = cfg.svi_kwargs.copy()
        if not run_map:
            best_z = svi_kwargs['start']
            svi_kwargs.pop('start')
        else:
            best_z = results["MAP"].best_z

        optimizer = svi_kwargs['optimizer']
        svi_kwargs.pop('optimizer')
            
        start = time.perf_counter()
        qz, svi_loss_hist = svi_func(best_z, optimizer, **svi_kwargs)
        end = time.perf_counter()
        
        results["SVI"] = SVIResults(qz, svi_loss_hist, end - start, model_seq)
    
    #* RUNNING HMC---------------------------------
    if run_hmc:
        print("Starting HMC")

        if cfg.hmc_func is None:
            hmc_func = model_seq.HMC
        else:
            hmc_func = cfg.hmc_func

        hmc_kwargs = cfg.hmc_kwargs.copy()
        if not run_svi:
            qz = hmc_kwargs['qz']
            hmc_kwargs.pop('qz')
        
        start = time.perf_counter()

        samples = hmc_func(qz, **hmc_kwargs)
        end = time.perf_counter()

        results["HMC"] = HMCResults(samples, end - start, model_seq)
    
    return results
    


def simulate_system(observed_img, prior, ModellingSequenceType, sim_config, phys_model,
    map_kwargs={}, svi_kwargs={}, hmc_kwargs={}, background_rms=0.2, exp_time=100, hmc_alt_multi=False):
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
    map_kwargs : dict
        Keyword arguments for the MAP stage
    svi_kwargs : dict
        Keyword arguments for the SVI stage
    hmc_kwargs : dict
        Keyword arguments for the HMC stage
     
    Returns
    -------
    dict
        Contains results from all inference stages:
        - MAP: MAPResults object
        - SVI: SVIResults object
        - HMC: HMCResults object
    """
    prob_model = ForwardProbModel(prior, observed_img, background_rms=background_rms, exp_time=exp_time)
    model_seq = ModellingSequenceType(phys_model, prob_model, sim_config)
    
    pipeline_config = PipelineConfig(map_kwargs = map_kwargs, svi_kwargs = svi_kwargs, hmc_kwargs = hmc_kwargs, hmc_func = model_seq.HMC_alt_multi if hmc_alt_multi else None)
    
    results = run_pipeline(model_seq, pipeline_config)
    
    return results

