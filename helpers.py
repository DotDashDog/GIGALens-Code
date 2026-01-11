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


NOTES FOR ELDEN:
- The precision_parameterization parameter I use for SVI means that gradient descent is done on the precision matrix rather than the covariance matrix.
    It's implemented in my inference.py file
    I'll ask about whether we should use it at the next meeting. If we decide not to, just don't pass it to SVI
- I have my ModellingSequence objects named ModellingSequence (the default GIGALens one) and HarryModellingSequence (Harry's multinode one). 
    If you do things differently, you'll have to change the references to the ModellingSequence class names in the results objects
- The way residualplot_params (used for hundred systems) works is very dependent on the way I store the modelling results. So probably don't use it for now.
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
import corner
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Plots import lens_plot
from lenstronomy.Data.imaging_data import ImageData

def params_jax_to_lists(params_jax):
    """Convert nested parameter structure of JAX arrays to nested Python lists"""
    # params_list = []
    # for i in range(len(params_jax)):
    #     params_list.append([])
    #     for j in range(len(params_jax[i])):
    #         params_list[i].append({})
    #         for key in params_jax[i][j]:
    #             params_list[i][j][key] = params_jax[i][j][key].tolist()
    # return params_list
    return jax.tree.map(lambda a : list(a), params_jax)

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
    # return jax.tree.map(lambda a : jnp.array(a), params_list, is_leaf=lambda x : isinstance(x, list) and isinstance(x[0], int))

def index_params(params, i):
    #* gets the params just for the ith system
    # o1 = []
    # for i1 in params:
    #     o2 = []
    #     for i2 in i1:
    #         out_d = {}
    #         for k in i2:
    #             out_d[k] = i2[k][i:i+1]
    #         o2.append(out_d)
    #     o1.append(o2)
            
    return jax.tree.map(lambda a : a[i], params)

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
        best_z = MAP_estimate.reshape((-1, 22))
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
            map_func = model_seq.MAP_multi
        else:
            map_func = cfg.map_func
        
        map_kwargs = cfg.map_kwargs.copy()
        optimizer = map_kwargs['optimizer']
        map_kwargs.pop('optimizer')
        
        start = time.perf_counter()
        map_estimate, map_chisq_hist = map_func(**cfg.map_kwargs)
        end = time.perf_counter()
        
        results["MAP"] = MAPResults(map_estimate, map_chisq_hist, end - start, model_seq)
    
    #* RUNNING SVI---------------------------------
    if run_svi:
        print("Starting SVI")
        
        if cfg.svi_func is None:
            svi_func = model_seq.SVI_multi
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
            hmc_func = model_seq.HMC_multi
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
        # cnorm = matplotlib.colors.Normalize(vmin=0)
        # Use LogNorm for logarithmic scaling with inferno colormap
        cnorm = matplotlib.colors.LogNorm(vmin=max(img.min(), 1e0), vmax=img.max())
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
    if extent is not None:
        ax.set_xlim((extent[0], extent[1]))
        ax.set_ylim((extent[2], extent[3]))
    ax.axis('off')


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

def cornerplot_labels(example_params, latex=False):
    """
    Generate the labels for the cornerplot based on the tree structure of the parameters.
    """
    tups = [(0, 0), (0, 1), (1, 0), (2, 0)]
    # get labels and pts for the MAP
    label_prefixes = ['', '', 'lens_', 'src_']
    labels = []
    
    for (i, j), label_prefix in zip(tups, label_prefixes):
        labels.extend((label_prefix + key for key in example_params[i][j].keys()))

    if latex:
        labels = [latex_label(label) for label in labels]

    return labels

def flatten_params_to_labeled_dict(params):
    tups = [(0, 0), (0, 1), (1, 0), (2, 0)]
    label_prefixes = ['', '', 'lens_', 'src_']

    flat_dict = {}
    for (i, j), label_prefix in zip(tups, label_prefixes):
        flat_dict.update({label_prefix + key: params[i][j][key] for key in params[i][j].keys()})
    return flat_dict

# def flatten_label_order(tree):
#     tups = [(0, 0), (0, 1), (1, 0), (2, 0)]
#     flat = []
#     for (i, j) in tups:
#        flat.extend((arr.item() for arr in tree[i][j].values()))
#     flat = np.array(flat)
#     return flat

def cornerplot_posterior(raw_samples, fig=None, truth=None, overplots=None, color='black', truth_color='black', overplot_color='red', plot_params=None):
    """
    Create a cornerplot of a set of samples in the physical space.
    Option to overplot a single point, such as the MAP best fit
    Can also overplot a second point as crossed vertical and horizontal lines (most often the truth or median of the samples)
    """
    flat_samples = flatten_params_to_labeled_dict(raw_samples)
    if plot_params is None:
        plot_params = flat_samples.keys()
        # flat_samples = {k:flat_samples[k] for k in plot_params}
        

    if overplots is not None:
        flat_overplots = flatten_params_to_labeled_dict(overplots)
            #flat_overplots = {k:flat_overplots[k] for k in plot_params}
        overplot_pts = np.squeeze(np.stack([flat_overplots[key] for key in plot_params]))

    if truth is not None:
        flat_truth = flatten_params_to_labeled_dict(truth)
        # if plot_params is not None:
        #     flat_truth = {k:flat_truth[k] for k in plot_params}
        truth_overplot_pts = np.squeeze(np.stack([flat_truth[key] for key in plot_params]))
    else:
        truth_overplot_pts = None

    samples = np.vstack([flat_samples[key] for key in plot_params]).T
    histargs = {'density': True, 'color': color}
    labels = [latex_label(label) for label in flat_samples.keys()]
    fig = corner.corner(samples, fig=fig, truths=truth_overplot_pts, truth_color=truth_color, 
        show_titles=True, title_fmt='.3f',
        labels=labels, hist_kwargs=histargs, color=color)

    if overplots is not None:
        corner.overplot_points(fig, overplot_pts[np.newaxis], marker='*', markersize=20, mfc=overplot_color, mec=overplot_color)
    
    return fig

def cornerplot_results(map_best, svi_samples=None, HMC_samples=None, true_params=None, hmc_median=None, plot_params=None, svi_label='SVI', hmc_label='HMC', legend_loc='upper right', legend_kwargs=None, truth_label='Truth', map_label='MAP'):
    """
    Cornerplot of the results of the inference pipeline, including MAP, SVI, and HMC.
    """

    svi_color = 'blue'
    hmc_color = 'black'

    fig = cornerplot_posterior(svi_samples, truth=true_params, overplots=map_best, color=svi_color, truth_color='black', overplot_color='red', plot_params=plot_params)
    cornerplot_posterior(HMC_samples, fig=fig, color=hmc_color, plot_params=plot_params)

    # Build a single consolidated legend
    handles = []
    if (svi_label is not None):
        handles.append(Patch(facecolor=svi_color, edgecolor='none', alpha=0.6, label=svi_label))
    if (hmc_label is not None):
        handles.append(Patch(facecolor=hmc_color, edgecolor='none', alpha=0.6, label=hmc_label))
    if (true_params is not None) and (truth_label is not None):
        handles.append(Line2D([0], [0], color='black', lw=1.5, label=truth_label))
    if (map_best is not None) and (map_label is not None):
        handles.append(Line2D([0], [0], marker='*', markersize=12, linestyle='none', markerfacecolor='red', markeredgecolor='red', label=map_label))

    if legend_kwargs is None:
        legend_kwargs = {}
    if 'fontsize' not in legend_kwargs:
        legend_kwargs['fontsize'] = 12
    prev_leg = getattr(fig, "_corner_legend_obj", None)
    if prev_leg is not None:
        try:
            prev_leg.remove()
        except Exception:
            pass
    new_leg = fig.legend(handles=handles, loc=legend_loc, frameon=False, **legend_kwargs)
    setattr(fig, "_corner_legend_obj", new_leg)

def get_errors_diff(HMC_samples, true_params):
    
    lower_err = jax.tree.map(lambda x: -(jnp.percentile(x, 16)-jnp.median(x)), HMC_samples)
    upper_err = jax.tree.map(lambda x: jnp.percentile(x, 84)-jnp.median(x), HMC_samples)
    median = jax.tree.map(lambda x: jnp.median(x), HMC_samples)

    median_diff = jax.tree.map(lambda x, y: x-y, median, true_params)

    # flat_lower_err, _ = jax.tree.flatten(lower_err)
    # flat_upper_err, _ = jax.tree.flatten(upper_err)
    # flat_median_diff, _ = jax.tree.flatten(median_diff)
    # flat_true_params, _ = jax.tree.flatten(true_params)

    return median_diff, lower_err, upper_err

def normalize_residuals(median_diff, upper_err, lower_err):
    pos_res = median_diff > 0
    scale = np.zeros_like(median_diff)
    scale[pos_res] = lower_err[pos_res]
    scale[~pos_res] = upper_err[~pos_res]

    residual_norm = median_diff/scale
    chisq = 1/(residual_norm.shape[0]-1) * np.sum(np.square(residual_norm))
    return residual_norm, chisq

def residualplot_params(save_dirs, true_params_all, prob_models, make_hist=False, figsize=(20,15), plot_kwargs={}):

    median_diffs = []
    upper_errs = []
    lower_errs = []
    for i, save_dir in enumerate(save_dirs):
        true_params = jax.tree.map(lambda a : a[i], true_params_all)

        samples = np.load(os.path.join(save_dir, 'hmc_samples_z.npy')).reshape((-1, 22))
        HMC_samples = prob_models[i].bij.forward(list(samples.T))

        diff, low, high = get_errors_diff(HMC_samples, true_params)

        median_diffs.append(diff)
        lower_errs.append(low)
        upper_errs.append(high)

    
    #* Turn list of trees into a single tree of jnp arrays
    def list_to_tree(list_of_trees):
        return jax.tree.map(lambda *xs: jnp.array(xs), *list_of_trees)

    median_diffs = flatten_params_to_labeled_dict(list_to_tree(median_diffs))
    upper_errs = flatten_params_to_labeled_dict(list_to_tree(upper_errs))
    lower_errs = flatten_params_to_labeled_dict(list_to_tree(lower_errs))
    true_params_flat = flatten_params_to_labeled_dict(true_params_all)

    # labels = cornerplot_labels(true_params_all, latex=True)
    # n_params = len(labels)
    n_params = len(median_diffs)
    ncols = 5
    fig, axs = plt.subplots(n_params//ncols + 1, ncols)
    fig.set_size_inches(*figsize)
    fig.tight_layout()
    axs = axs.flatten()

    num_systems = len(save_dirs)

    for i, label in enumerate(median_diffs.keys()):
        axs[i].axhline(y=0, color='red', linestyle=':', alpha=0.5)
        axs[i].errorbar(true_params_flat[label], median_diffs[label], yerr=[lower_errs[label], upper_errs[label]], **(plot_kwargs |  dict(fmt='o', linestyle='')))
        axs[i].set_title(latex_label(label))
        yabs_max = abs(max(axs[i].get_ylim(), key=abs))
        axs[i].set_ylim(ymin=-yabs_max, ymax=yabs_max)

    # Turn off unused axes in the subplot grid
    for j in range(len(median_diffs.keys()), len(axs)):
        axs[j].axis('off')
    if not make_hist:
        return fig, axs
    else:
        plt.show()

    if make_hist:
        fig, axs = plt.subplots(n_params//3 + 1, 3)
        fig.set_size_inches(20,25)
        axs = axs.flatten()

        
        for i, label in enumerate(median_diffs.keys()):
            z_scores, chisq = normalize_residuals(median_diffs[label], upper_errs[label], lower_errs[label])
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
                    theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                    gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
                    e1=tfd.Normal(0, 0.2),
                    e2=tfd.Normal(0, 0.2),
                    center_x=tfd.Normal(0, 0.06),
                    center_y=tfd.Normal(0, 0.06),
                )
            ),
            tfd.JointDistributionNamed(
                dict(gamma1=tfd.Normal(0, 0.1), gamma2=tfd.Normal(0, 0.1))
            ),
        ]
    )
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
                    n_sersic=tfd.Uniform(0.5, 8),
                    e1=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
                    e2=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
                    center_x=tfd.Normal(0, 0.02),
                    center_y=tfd.Normal(0, 0.02),
                    Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
                )
            )
        ]
    )

    source_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
                    n_sersic=tfd.Uniform(0.5, 8),
                    e1=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                    center_x=tfd.Normal(0, 0.5),
                    center_y=tfd.Normal(0, 0.5),
                    Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
                )
            )
        ]
    )

    prior = tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )
    return prior

def latex_label(label):

    latex_label_map = {
        # Mass
        "theta_E": r"$\theta_E$",
        "gamma": r"$\gamma_{epl}$",
        "e1": r"$\epsilon_{epl,1}$",
        "e2": r"$\epsilon_{epl,2}$",
        "center_x": r"$x_{epl}$",
        "center_y": r"$y_{epl}$",
        "gamma1": r"$\gamma_{ext,1}$",
        "gamma2": r"$\gamma_{ext,2}$",

        # Lens Light
        "lens_R_sersic": r"$R_{l}$",
        "lens_n_sersic": r"$n_{l}$",
        "lens_e1": r"$\epsilon_{l,1}$",
        "lens_e2": r"$\epsilon_{l,2}$",
        "lens_center_x": r"$x_{l}$",
        "lens_center_y": r"$y_{l}$",
        "lens_Ie": r"$I_l$",

        # Source Light
        "src_R_sersic": r"$R_{s}$",
        "src_n_sersic": r"$n_{s}$",
        "src_e1": r"$\epsilon_{s,1}$",
        "src_e2": r"$\epsilon_{s,2}$",
        "src_center_x": r"$x_{s}$",
        "src_center_y": r"$y_{s}$",
        "src_Ie": r"$I_s$",
    }

    return latex_label_map[label]