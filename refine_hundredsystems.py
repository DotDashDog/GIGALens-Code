#%%
import corner
import time
import os
import sys
import json
import argparse
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
import os
import yaml


jax.distributed.initialize()
# jax.distributed.initialize(
#     coordinator_address="localhost:12346",
#     num_processes=1,
#     process_id=0
# )

jax.experimental.multihost_utils.sync_global_devices("run_start")
kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
# observed_img = np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy'))

prior = helpers.make_default_prior()

phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
# prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=2, kernel=kernel) 

# model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)

# pipeline_config = PipelineConfig(
#     steps=["MAP", "SVI", "HMC"], map_steps=350, map_n_samples=500, n_vi=1000, 
#     svi_steps=1500, hmc_burnin_steps=250, hmc_num_results=750, n_hmc=50)

# results = run_pipeline(model_seq, pipeline_config)

systems_dir = os.path.join(home, "GIGALens-Code/SystemSaves")

f = np.load(os.path.join(systems_dir, "100SystemsStandard80px.npz"))
keys = f.files
observed_imgs = jnp.array([f[key] for key in keys])

# filename = os.path.join(systems_dir, '100SystemsStandardParams.yaml')
# with open(filename, 'r') as file:
#     true_params = params_lists_to_jax(yaml.safe_load(file))

# print(f"Simulating systems {start_idx} to {end_idx-1} (inclusive)")
prev_save_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/100standard80px")
final_save_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/100standard80px_refined")
# finished_systems = [int(f.split('/')[-1].split('.')[0]) for f in os.listdir(final_save_dir)]

# idxes = list(range(start_idx, end_idx))
# idxes = [4, 18, 52, 54, 56, 94]
# idxes = list(range(100))

rhat_df = pd.read_csv(os.path.join(prev_save_dir, 'rhat_results.csv')).set_index('system')
for i in rhat_df.index:
    max_rhat = rhat_df['rhat'][i]
    observed_img = observed_imgs[i]
    if max_rhat < 1.01:
        print(f"System {i} already has passable rhat, skipping")
        continue
    elif max_rhat < 1.1:
        results = simulate_system(
            observed_img, prior, HarryModellingSequence, sim_config, phys_model, 
            map_steps=1000, map_n_samples=2000,
            precision_parameterization=False, n_vi=1000, svi_steps=5000,
            n_hmc=64, hmc_num_results=7000, hmc_burnin_steps=2000,
            init_eps=0.3, init_l=3
        )
    else:
        results = simulate_system(
            observed_img, prior, HarryModellingSequence, sim_config, phys_model, 
            map_steps=1000, map_n_samples=2000,
            precision_parameterization=False, n_vi=2000, svi_steps=4999,
            n_hmc=64, hmc_num_results=10000,hmc_burnin_steps=3000,
            init_eps=0.3, init_l=3
        )
    
    # if i in finished_systems:
    #     print(f"System {i} already finished, skipping")
    #     continue
    
    

    

    #* Intensive settings are n_vi = 10000, svi_steps = 5000, hmc_num_results = 5000

    if jax.process_index() == 0:
        print(f"System {i}:")
        print("MAP time taken: ", results["MAP"].time_taken)
        print("SVI time taken: ", results["SVI"].time_taken)
        print("HMC time taken: ", results["HMC"].time_taken)

        # Create directory for saving results if it doesn't exist
        results_dir = os.path.join(home, final_save_dir, f"{i}")
        # results_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/example")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        results["MAP"].save(results_dir)
        results["SVI"].save(results_dir)
        results["HMC"].save(results_dir)


# final_save_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/100standard80px_refined")
# for sysnum in finished_systems:
#     observed_img = observed_imgs[sysnum]
#     prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
#     model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)

    
#     results_dir = os.path.join(prelim_save_dir, f"{sysnum}")

#     map_results_prev = MAPResults.load(results_dir, model_seq)
#     svi_results_prev = SVIResults.load(results_dir, model_seq)
#     hmc_results_prev = HMCResults.load(results_dir, model_seq)

#     rhat = np.max(hmc_results_prev.HMC_rhat)

#     if rhat < 1.01:
#         print(f"System {sysnum} has rhat < 1.01, already acceptable")
#         continue
#     elif rhat > 1.01 and rhat < 1.1:
#         print(f"System {sysnum} has rhat > 1.01 and < 1.1, refining by longer HMC")
#         pipeline_config_hmc = PipelineConfig(
#         steps=['HMC'], hmc_burnin_steps=250, hmc_num_results=5000, n_hmc=64, qz=svi_results_prev.qz,
#         )
#         results = run_pipeline(model_seq, pipeline_config_hmc)
#         results["MAP"] = map_results_prev
#         results["SVI"] = svi_results_prev
#     elif rhat > 1.1:
#         print("System {sysnum} has rhat > 1.1, running more intensive pipeline")
#         pipeline_config = PipelineConfig(steps=["MAP", "SVI", "HMC"], 
#             map_steps=1000, map_n_samples=1000, 
#             n_vi=5000, svi_steps=5000, 
#             hmc_burnin_steps=250, hmc_num_results=5000, n_hmc=64)
#         results = run_pipeline(model_seq, pipeline_config)
        


#     if jax.process_index() == 0:
#         print(f"System {sysnum}:")
#         print("MAP time taken: ", results["MAP"].time_taken)
#         print("SVI time taken: ", results["SVI"].time_taken)
#         print("HMC time taken: ", results["HMC"].time_taken)

#         # Create directory for saving results if it doesn't exist
#         results_dir = os.path.join(home, final_save_dir, f"{sysnum}")
#         if not os.path.exists(results_dir):
#             os.makedirs(results_dir)

#         results["MAP"].save(results_dir)
#         results["SVI"].save(results_dir)
#         results["HMC"].save(results_dir)

# %%
