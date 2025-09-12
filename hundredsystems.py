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


# def parse_arguments():
#     """Parse command line arguments for system range."""
#     parser = argparse.ArgumentParser(description='Simulate a range of lensing systems')
#     parser.add_argument('start', type=int, 
#                       help='Starting system index (inclusive)')
#     parser.add_argument('end', type=int,
#                       help='Ending system index (exclusive)')
    
#     args = parser.parse_args()
    
#     # Validate arguments
#     if args.start < 0:
#         raise ValueError("Start index must be non-negative")
#     if args.end <= args.start:
#         raise ValueError("End index must be greater than start index")
#     if args.start >= 100:
#         raise ValueError("Start index must be less than 100")
#     if args.end > 100:
#         raise ValueError("End index must be less than or equal to 100")
    
#     return args.start, args.end


# # Parse command line arguments
# start_idx, end_idx = parse_arguments()

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
save_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/100standard80px")
# finished_systems = [int(f.split('/')[-1].split('.')[0]) for f in os.listdir(save_dir)]

# idxes = list(range(start_idx, end_idx))
# idxes = [4, 18, 52, 54, 56, 94]
idxes = list(range(100))
for i in idxes:
    # if i in finished_systems:
    #     print(f"System {i} already finished, skipping")
    #     continue
    observed_img = observed_imgs[i]

    results = simulate_system(
        observed_img, prior, HarryModellingSequence, sim_config, phys_model, 
        map_steps=1000, map_n_samples=2000,
        precision_parameterization=False, n_vi=1000, svi_steps=5000,
        n_hmc=64, hmc_num_results=1500,
        init_eps=0.3, init_l=3, hmc_burnin_steps=500
    )

    #* Intensive settings are n_vi = 10000, svi_steps = 5000, hmc_num_results = 5000

    if jax.process_index() == 0:
        print(f"System {i}:")
        print("MAP time taken: ", results["MAP"].time_taken)
        print("SVI time taken: ", results["SVI"].time_taken)
        print("HMC time taken: ", results["HMC"].time_taken)

        # Create directory for saving results if it doesn't exist
        results_dir = os.path.join(home, save_dir, f"{i}")
        # results_dir = os.path.join(home, f"GIGALens-Code/pipeline_results/example")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        results["MAP"].save(results_dir)
        results["SVI"].save(results_dir)
        results["HMC"].save(results_dir)
        
# %%
