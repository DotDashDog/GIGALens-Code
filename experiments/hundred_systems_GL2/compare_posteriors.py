#%%
import corner
import time
import os
import sys
import json
from datetime import datetime
from os.path import expanduser

home = expanduser("~/")
srcdir = os.path.join(home, 'gigalens/src/')
# srcdir = os.path.join(home, "gigalens-multinode/gigalens_hackathon/src/")
sys.path.insert(0, srcdir)
sys.path.insert(0, home+'/GIGALens-Code')
print('MASTER BRANCH GIGALENS')

import jax
# jax.config.update("jax_enable_x64", True)

from gigalens.jax.inference import HarryModellingSequence
from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

import tensorflow_probability.substrates.jax as tfp
from jax import random
from jax import numpy as jnp
import numpy as np
import optax
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from corner import corner
tfd = tfp.distributions
import pickle
import helpers
from helpers import *
import pandas as pd
import yaml

jax.distributed.initialize(
    coordinator_address="localhost:12346",
    num_processes=1,
    process_id=0
)

# jax.distributed.initialize()
save_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/100standard80px")


refined_save_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/100standard80px_refined_more")

# finished_systems = [4, 18, 53, 56, 81, 98]
sysnum=60
base_result_dir = os.path.join(save_dir, f"{sysnum}")
refined_result_dir = os.path.join(refined_save_dir, f"{sysnum}")


kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
# observed_img = np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy'))
systems_dir = os.path.join(home, "GIGALens-Code/SystemSaves")
f = np.load(os.path.join(systems_dir, "100SystemsStandard80px.npz"))
keys = f.files
observed_imgs = jnp.array([f[key] for key in keys])

filename = os.path.join(systems_dir, '100SystemsStandardParams.yaml')
with open(filename, 'r') as file:
    true_params = params_lists_to_jax(yaml.safe_load(file))

prior = helpers.make_default_prior()

phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])

sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=2, kernel=kernel) 
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
#%%
base_results = {}

observed_img = observed_imgs[sysnum]
prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)

base_results["MAP"] = MAPResults.load(base_result_dir, model_seq)
base_results["SVI"] = SVIResults.load(base_result_dir, model_seq)
base_results["HMC"] = HMCResults.load(base_result_dir, model_seq)

refined_results = {}

refined_results["MAP"] = MAPResults.load(refined_result_dir, model_seq)
refined_results["SVI"] = SVIResults.load(refined_result_dir, model_seq)
refined_results["HMC"] = HMCResults.load(refined_result_dir, model_seq)

# print(f"Loaded from:", results_dir)
# rhat_max = np.max(results["HMC"].HMC_rhat)
# print("Final MAP chisq:", results["MAP"].MAP_chisq_hist[-1], 
#     "Final ELBO:",results["SVI"].SVI_loss_hist[-1], 
#     "Worst HMC rhat:", np.max(results["HMC"].HMC_rhat), "SVI steps:", results["SVI"].SVI_loss_hist.shape[0])

# print("Min ESS:", np.min(tfp.mcmc.effective_sample_size(results["HMC"].HMC_samples_z.reshape((-1, *results["HMC"].HMC_samples_z.shape[2:])), cross_chain_dims=1)))
# print(jax.tree.map(lambda x: tfp.mcmc.effective_sample_size(x,cross_chain_dims=1), prob_model.bij.forward(results["HMC"].HMC_samples_z)))
# break
# display_results(results, observed_img, lens_sim, save_dir=results_dir, show=True, make_cornerplot=True, 
#     true_params=index_params(true_params, sysnum), plot_caustics=True, model_seq=model_seq)
        
labels = cornerplot_labels(base_results["MAP"].MAP_best)

fig = cornerplot_posterior(labels, refined_results["HMC"].HMC_samples, overplots=refined_results["MAP"].MAP_best, color='purple', overplot_color='orange')
cornerplot_posterior(labels, base_results["HMC"].HMC_samples, truth=index_params(true_params, sysnum), overplots=base_results["MAP"].MAP_best, color='blue', truth_color='green', overplot_color='red', fig=fig)
plt.show()
# %%
