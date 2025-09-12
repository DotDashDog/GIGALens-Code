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
save_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/100standard80px_refined")
finished_systems = []
for fname in os.listdir(save_dir):
    try:
        int(fname)
    except ValueError:
        continue
    else:
        finished_systems.append(int(fname.split('/')[-1].split('.')[0]))

finished_systems.sort()

refined_save_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/100standard80px_refined")
refined_systems = []
for fname in os.listdir(refined_save_dir):
    try:
        int(fname)
    except ValueError:
        continue
    else:
        refined_systems.append(int(fname.split('/')[-1].split('.')[0]))

refined_systems.sort()

# finished_systems = [4, 18, 53, 56, 81, 98]

results_dirs = {n : os.path.join(refined_save_dir if n in refined_systems else save_dir, f"{n}") for n in finished_systems}

print(results_dirs)
# skip = []
# for num in finished_systems:
#     if os.path.exists(os.path.join(save_dir, f"{num}", 'cornerplot.png')):
#         # print(f"System {num} already has cornerplot")
#         skip.append(num)




kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
# observed_img = np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy'))
systems_dir = os.path.join(home, "GIGALens-Code/SystemSaves")
f = np.load(os.path.join(systems_dir, "100SystemsStandard80px.npz"))
keys = f.files
observed_imgs = jnp.array([f[key] for key in keys])

filename = os.path.join(systems_dir, '100SystemsStandardParams.yaml')
with open(filename, 'r') as file:
    true_params = params_lists_to_jax(yaml.safe_load(file))

select_index = lambda a : a[np.array(finished_systems)]
true_params = jax.tree.map(select_index, true_params)

prior = helpers.make_default_prior()

phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])

sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=2, kernel=kernel) 
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
#%%

# def show_with_caustic(params):
#     simulated = lens_sim.simulate(params) #or however you obtain your simulated image
#     numPix = model_seq.sim_config.num_pix
#     deltaPix = model_seq.sim_config.delta_pix
#     kwargs_data = sim_util.data_configure_simple(numPix*40, deltaPix/20)
#     data = ImageData(**kwargs_data)
#     _coords = data
#     lensModel = LensModel(lens_model_list=['EPL', 'SHEAR']) #just need a list of the mass parameters, something like ['EPL', 'SHEAR']
#     params = jax.tree.map(lambda a : np.array(a), params)
#     kwargs_lens = params[0] #the values for the above parameters

#     plt.figure(figsize=(16,4))
#     # norm = simple_norm(psf, 'log', percent=99.)
#     extent = (-numPix/2*deltaPix, numPix/2*deltaPix, -numPix/2*deltaPix, numPix/2*deltaPix)
#     print(extent)
#     ax = plt.subplot(111)
#     ax.set_xlim((extent[0], extent[1]))
#     ax.set_ylim((extent[0], extent[1]))
#     plt.imshow(simulated, extent=extent,origin='lower', cmap='inferno')
#     lens_plot.caustics_plot(ax, _coords, lensModel, kwargs_lens, fast_caustic=True, color_crit='red', color_caustic='green')

#idxes = list(range(100))#[4, 18, 52, 54, 56, 94]
# finished_systems = list(range(100))
rhat_maxes = []
for sysnum in finished_systems:
    results = {}
    # for sysnum in np.sort(finished_systems):
    # if sysnum in skip:
    #     print(f"System {sysnum} already has cornerplot, skipping")
    #     continue
    observed_img = observed_imgs[sysnum]
    prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
    model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)

    
    results_dir = results_dirs[sysnum]
    results["MAP"] = MAPResults.load(results_dir, model_seq)
    results["SVI"] = SVIResults.load(results_dir, model_seq)
    results["HMC"] = HMCResults.load(results_dir, model_seq)
    print(f"System {sysnum}:")
    # print(f"Loaded from:", results_dir)
    rhat_max = np.max(results["HMC"].HMC_rhat)
    print("Final MAP chisq:", results["MAP"].MAP_chisq_hist[-1], 
        "Final ELBO:",results["SVI"].SVI_loss_hist[-1], 
        "Worst HMC rhat:", np.max(results["HMC"].HMC_rhat), "SVI steps:", results["SVI"].SVI_loss_hist.shape[0])

    rhat_maxes.append(rhat_max)
    
    display_results(results, observed_img, lens_sim, save_dir=results_dir, show=False, make_cornerplot=False, 
        true_params=index_params(true_params, sysnum), plot_caustics=True, model_seq=model_seq)
        

df = pd.DataFrame({'system' : finished_systems, 'rhat': rhat_maxes})
# df.to_csv(os.path.join(save_dir, 'rhat_results.csv'))
# %%
# result_dirs = [os.path.join(save_dir, f"{sysnum}") for sysnum in finished_systems]
prob_models = [ForwardProbModel(prior, observed_imgs[sysnum], background_rms=0.2, exp_time=100) for sysnum in finished_systems]

# true_file = os.path.join(home, "GIGALens-Code", 'SystemSaves', '100SystemsStandardParams.yaml')
# with open(true_file, 'r') as file:
#     loaded_params_list = yaml.safe_load(file)
    
# true_params = params_lists_to_jax(loaded_params_list)


residualplot_params(list(results_dirs.values()), true_params, prob_models)

# %%

lens_Ie_prior = tfd.LogNormal(jnp.log(300.0), 0.3)
source_Ie_prior = tfd.LogNormal(jnp.log(150.0), 0.5)

lens_Ie = true_params[1][0]['Ie']
source_Ie = true_params[2][0]['Ie']

lens_Ie_60 = true_params[1][0]['Ie'][60]
source_Ie_60 = true_params[2][0]['Ie'][60]
plt.axvline(lens_Ie_60, color='C0', linestyle='--', alpha=0.7, label='System 60 Lens Ie')
plt.axvline(source_Ie_60, color='C1', linestyle='--', alpha=0.7, label='System 60 Source Ie')

dummy_x = np.linspace(0, np.maximum(np.max(lens_Ie), np.max(source_Ie)))

plt.plot(dummy_x, lens_Ie_prior.prob(dummy_x), color='C0', label='Lens Ie Prior')
plt.hist(lens_Ie, label="Systems' Lens Ie", histtype='step', density=True, color='C0')
plt.plot(dummy_x, source_Ie_prior.prob(dummy_x), color='C1', label='Source Ie Prior')
plt.hist(source_Ie, label="Systems' Source Ie", histtype='step', density=True, color='C1')
plt.title("Source vs. Lens Ie (Hundred Systems)")
plt.legend()
plt.show()
# %%
