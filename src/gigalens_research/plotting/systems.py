import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

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

from gigalens_research.inference_utils import get_noise_image, get_chisq
from gigalens_research.inference_utils.params import to_dict_params

def plot_image(fig, ax, img, extent=None, title=None, residual=False, colorbar=True, remove_axis=True, log_vmin=1e-2, log_norm=True):
    """
    Plot an image using my chosen standards for coloring, 
    which changes depending on whether the image is a residual or not.
    """
    if not residual:
        #* Meaning actual lensing image
        # cnorm = matplotlib.colors.Normalize(vmin=0)
        # Use LogNorm for logarithmic scaling with inferno colormap
        if log_norm == True:
            cnorm = matplotlib.colors.LogNorm(vmin=jnp.fmax(img.min(), log_vmin), vmax=jnp.abs(jnp.nanmax(img)), clip=True)
        else:
            cnorm = cnorm = matplotlib.colors.Normalize()
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
    if remove_axis:
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)


def add_caustics(ax, params, sim_config, lens_objects=['EPL', 'SHEAR']):
    kwargs_data = sim_util.data_configure_simple(sim_config.num_pix*40, sim_config.delta_pix/20)
    data = ImageData(**kwargs_data)
    _coords = data
    lensModel = LensModel(lens_model_list=lens_objects) #just need a list of the mass parameters, something like ['EPL', 'SHEAR']
    params = jax.tree.map(lambda a : np.array(a), params)
    # New gigalens params are dict-keyed: {'lens_mass': {'0': {..}, '1': {..}}, ..}.
    # lenstronomy wants a list of lens param dicts ordered to match lens_model_list.
    params = to_dict_params(params)
    lens_mass = params['lens_mass']
    kwargs_lens = [lens_mass[k] for k in sorted(lens_mass, key=int)]

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
                       plot_caustics=False, model_seq=None, background_rms=0.2, exp_time=100.0, use_backward=False, log_vmin=1e-3):
    """
    Plot the results of a lensing fit. Given a set of predicted parameters, compare the predicted image to the true image.
    Displays normalized residuals, and a histogram of the residuals to check that they are gaussian noise
    """
    # Normalise to the new gigalens dict-keyed param structure; legacy list-form
    # truth (vela/GL2 pickles) would otherwise break the dict-expecting simulator.
    if predicted_params is not None:
        predicted_params = to_dict_params(predicted_params)
    if true_params is not None:
        true_params = to_dict_params(true_params)
    if resimulate:
        if lens_sim is None:
            raise ValueError("lens_sim must be provided if resimulate is True")
        if not use_backward:
            predicted_img = lens_sim.simulate(predicted_params)
        else:
            orig_err_map = get_noise_image(true_img, background_rms, exp_time)
            predicted_img = lens_sim.lstsq_simulate(predicted_params, true_img, orig_err_map)
    elif predicted_img is None:
        raise ValueError("predicted_img must be provided if resimulate is False")

    if display_true_chisq:
        true_chisq = get_chisq(true_img, lens_sim.simulate(true_params))
    
    noise_map = get_noise_image(predicted_img, background_rms, exp_time)

    residual = (true_img - predicted_img)/noise_map

    chisq = np.sum(np.square(residual))
    dof = true_img.shape[0]*true_img.shape[1] - 22 #! Change if number of params changes
    #! Do I want to do sqrt curve cmap for the images?
    if plot_caustics:
        numPix = model_seq.sim_config.num_pix
        deltaPix = model_seq.sim_config.delta_pix
        extent = (-numPix/2*deltaPix, numPix/2*deltaPix, -numPix/2*deltaPix, numPix/2*deltaPix)
    else:
        extent =None
    plot_image(fig, axs[0], true_img, extent=extent,
               title=f"True Image" + (f"(Red Chisq:{true_chisq/dof:.3f})" if display_true_chisq else ""), log_vmin=log_vmin)
    if plot_caustics and (true_params is not None):
        add_caustics(axs[0], true_params, model_seq)
    plot_image(fig, axs[1], predicted_img, extent=extent, title=f"{prefix} Model Fit (Red Chisq:{chisq/dof:.5f})", log_vmin=log_vmin)
    if plot_caustics and (predicted_params is not None):
        add_caustics(axs[1], predicted_params, model_seq)
    plot_image(fig, axs[2], residual, extent=extent, title=f"{prefix} Normalized Residual", residual=True)

    if display_true_chisq:
        print("True Chisq", true_chisq)
        print("Model Fit Chisq", chisq)
    flat_residual = residual.flatten()
    histogram_residuals(fig, axs[3], flat_residual, f"{prefix} Global Gaussianity Test")