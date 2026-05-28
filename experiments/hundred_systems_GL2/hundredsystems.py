#%%
import time
import os
import sys
import json
import argparse
from datetime import datetime
from os.path import expanduser
import socket

conda_env = sys.path[1]
del sys.path[1]

home = expanduser("~/")
srcdir = os.path.join(home, 'gigalens/src/')
# srcdir = os.path.join(home, "gigalens-multinode/gigalens_hackathon/src/")
sys.path.append(home+'/gigalens'+'/src')
sys.path.append(home+'/GIGALens-Code')
sys.path.append(conda_env)
print('MASTER BRANCH GIGALENS')

import jax

sys.path.append(f'{os.environ['HOME']}/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages')
jax.distributed.initialize(local_device_ids=None)

if jax.process_index() == 0:
    print(sys.path)
    print(f"Hostname: {socket.gethostname()}")
    # print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    print(f"Visible JAX devices: {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

from gigalens_research.inference_utils import (
    InferenceContext, Pipeline, MAPStage, SVIStage, HMCStage,
)

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

jax.experimental.multihost_utils.sync_global_devices("run_start")
kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)

prior = helpers.make_default_prior()

phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=2, kernel=kernel) 

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
idxes = list(range(4, 100))
for i in idxes:
    observed_img = observed_imgs[i]

    prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
    model_seq = ModellingSequence(phys_model, prob_model, sim_config)
    ctx = InferenceContext.from_modelling_sequence(model_seq)

    pipeline = Pipeline(ctx, seed=0)
    pipeline.add(MAPStage(num_steps=1000, n_samples=2000))
    pipeline.add(SVIStage(num_steps=5000, n_vi=1000))
    pipeline.add(HMCStage(n_hmc=64, num_results=1500, num_burnin_steps=500))

    #* Intensive settings are n_vi = 10000, svi_steps = 5000, hmc_num_results = 5000

    results_dir = os.path.join(home, save_dir, f"{i}")
    pipeline.run(out_dir=results_dir, resume=True)

# %%
