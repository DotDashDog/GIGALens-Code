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

sys.path.insert(0, srcdir)
sys.path.insert(0, home+'/GIGALens-Code')
print('Harry GIGALENS IMPLEMENTATION')

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

#* For slurm jobs
# jax.distributed.initialize()

#* For local, single-gpu testing
jax.distributed.initialize(
    coordinator_address="localhost:12346",
    num_processes=1,
    process_id=0
)

save_dir = os.path.join(home, f"GIGALens-Code/benchmarking_results")

jax.experimental.multihost_utils.sync_global_devices("run_start")
kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
observed_img = np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy'))

prior = helpers.make_default_prior()

phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=kernel) 


model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)

pipeline_config = PipelineConfig(
    steps=["MAP", "SVI"], map_steps=350, map_n_samples=1000, n_vi=1000, 
    svi_steps=1500)

#%%
results = run_pipeline(model_seq, pipeline_config)
map_results = results["MAP"]
svi_results = results["SVI"]
#%%

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
plot_loss_histories(fig, axs, map_results.MAP_chisq_hist, svi_results.SVI_loss_hist)

plt.show()

#%%
