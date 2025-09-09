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

systems_dir = os.path.join(home, "GIGALens-Code/SystemSaves")
f = np.load(os.path.join(systems_dir, "100SystemsStandard80px.npz"))
keys = f.files
observed_imgs = jnp.array([f[key] for key in keys])

observed_img = observed_imgs[4]#np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy'))

prior = helpers.make_default_prior()

phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=2, kernel=kernel) 

model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)

schedule_fn = optax.polynomial_schedule(init_value=-1e-2, end_value=-1e-2/3, 
                                          power=0.5, transition_steps=500)
opt = optax.chain(
    # optax.scale_by_adam(),
    # optax.scale_by_schedule(schedule_fn),
    optax.adabelief(1e-2, b1=0.95, b2=0.99),  
)

print('Starting MAP')
map_samples, map_losses = model_seq.MAP(opt, seed=0, num_steps=350, n_samples=500)

map_loss_history = jnp.min(map_losses, axis=1)
best_step_idx = jnp.argmin(map_loss_history)
best_sample_idx = jnp.argmin(map_losses[best_step_idx])


best = map_samples[best_step_idx][best_sample_idx][jnp.newaxis, :]
map_x = prob_model.bij.forward(list(best.T))

# SVI
print('Starting SVI')
schedule_fn = optax.polynomial_schedule(init_value=-1e-6, end_value=-3e-3, 
                                        power=2, transition_steps=300)
opt = optax.chain(
    # optax.scale_by_adam(),
    # optax.scale_by_schedule(schedule_fn),
    optax.adabelief(2e-3, b1=0.95, b2=0.99),
)

qz, loss_history = model_seq.SVI(best, opt, n_vi=1000, num_steps=1000)

print('Starting HMC')
samples = model_seq.HMC_multi_alt(qz, n_hmc=50,num_burnin_steps=250, num_results=750)


# %%
