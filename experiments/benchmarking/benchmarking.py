#%%
import corner
import time
import os
import sys
import json
from datetime import datetime
from os.path import expanduser

code_version = "Nico"

home = expanduser("~/")
if code_version == "Harry":
    srcdir = os.path.join(home, 'gigalens/src/')
elif code_version == "Nico":
    srcdir = os.path.join(home, "gigalens-multinode/gigalens_hackathon/src/")
else:
    raise ValueError(f"Invalid code version: {code_version}")

sys.path.insert(0, srcdir)
sys.path.insert(0, home+'/GIGALens-Code')
print(f'{code_version} GIGALENS IMPLEMENTATION')

import jax
# jax.config.update("jax_enable_x64", True)

if code_version == "Harry":
    from gigalens.jax.inference import HarryModellingSequence
elif code_version == "Nico":
    from gigalens.jax.inference import ModellingSequenceMultinode
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

#* For slurm jobs
jax.distributed.initialize()

#* For local, single-gpu testing
# jax.distributed.initialize(
#     coordinator_address="localhost:12346",
#     num_processes=1,
#     process_id=0
# )

save_dir = os.path.join(home, f"GIGALens-Code/benchmarking_results")

jax.experimental.multihost_utils.sync_global_devices("run_start")
kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
observed_img = np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy'))

prior = helpers.make_default_prior()

phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=kernel) 

if code_version == "Harry":
    model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)
elif code_version == "Nico":
    model_seq = ModellingSequenceMultinode(phys_model, prob_model, sim_config)

multipliers = [1, 4, 8, 16]
for mult in multipliers:
    #* Load starting points
    map_standard = MAPResults.load(os.path.join(home, "GIGALens-Code", "benchmarking_results", "benchmark_starts"), model_seq)
    svi_start = map_standard.best_z
    svi_standard = SVIResults.load(os.path.join(home, "GIGALens-Code", "benchmarking_results", "benchmark_starts"), model_seq)
    hmc_start_qz = svi_standard.qz

    if code_version == "Nico":
        _, tree_struct  = jax.tree.flatten(svi_start)
    else:
        tree_struct = None

    #* Run full pipeline
    pipeline_config_map = PipelineConfig(
        steps = ["MAP"], map_steps=1000, map_n_samples=1000*mult,
    )

    map_results = run_pipeline(model_seq, pipeline_config_map)['MAP']
    # print(map_results.best_z)
    # print(svi_start)

    pipeline_config_svi = PipelineConfig(
        steps=['SVI'], n_vi=1000*mult, svi_steps=1500, svi_start=svi_start
    )
    svi_results = run_pipeline(model_seq, pipeline_config_svi)['SVI']

    pipeline_config_hmc = PipelineConfig(
        steps=['HMC'], hmc_burnin_steps=250, hmc_num_results=750, n_hmc=75*mult, qz=hmc_start_qz, tree_struct=tree_struct
    )
    hmc_results = run_pipeline(model_seq, pipeline_config_hmc)['HMC']
    # pipeline_config = PipelineConfig(
    #     steps=["MAP", "SVI", "HMC"], map_steps=1000, map_n_samples=1000*mult, n_vi=1000*mult, 
    #     svi_steps=1500, hmc_burnin_steps=250, hmc_num_results=750, n_hmc=75*mult)

    # results_full = run_pipeline(model_seq, pipeline_config)

    #* Run 1 epoch of MAP and SVI
    pipeline_config_map_1epoch = PipelineConfig(
        steps=["MAP"], map_steps=1, map_n_samples=1000*mult, 
    )
    map_results_1epoch = run_pipeline(model_seq, pipeline_config_map_1epoch)["MAP"]

    pipeline_config_svi_1epoch = PipelineConfig(
        steps = ['SVI'], n_vi=1000*mult, svi_steps=1, svi_start=svi_start
    )
    svi_results_1epoch = run_pipeline(model_seq, pipeline_config_svi_1epoch)["SVI"]

    #* Run just burnin steps for HMC
    pipeline_config_hmc_burnin = PipelineConfig(
        steps=["HMC"], hmc_burnin_steps=250, hmc_num_results=1, n_hmc=75*mult, qz=hmc_start_qz, tree_struct=tree_struct)
    results_hmc_burnin = run_pipeline(model_seq, pipeline_config_hmc_burnin)["HMC"]

    print("MAP time taken: ", map_results.time_taken)
    print("SVI time taken: ", svi_results.time_taken)
    print("HMC time taken: ", hmc_results.time_taken)

    # save results using append-only approach to avoid race conditions
    csv_file = os.path.join(save_dir, f'benchmark_results_final_gather_{code_version}.csv')
    
    # Create new row with current results
    new_row = {
        'map_time_1epoch': map_results_1epoch.time_taken,
        'svi_time_1epoch': svi_results_1epoch.time_taken, 
        'hmc_time_burnin': results_hmc_burnin.time_taken,
        'map_time': map_results.time_taken,
        'svi_time': svi_results.time_taken, 
        'hmc_time': hmc_results.time_taken,
        'map_n_samples': pipeline_config_map.map_n_samples,
        'n_vi': pipeline_config_svi.n_vi,
        'n_hmc': pipeline_config_hmc.n_hmc,
        'devices': jax.device_count(),
        'process_id': jax.process_index(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'N/A'),
    }
    
    # Append to CSV file (write header if file doesn't exist)
    df_new = pd.DataFrame([new_row])
    df_new.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
# %%
