# %%
import os
from os.path import expanduser
home = expanduser("~/")

import sys
sys.path.insert(0, home+'/gigalens'+'/src')
sys.path.insert(0, home+'/GIGALens-Code')
print('MASTER BRANCH GIGALENS')

srcdir = os.path.join(home, "gigalens/src/")
code_dir = os.path.join(home, "GIGALens-Code")
# %%
from gigalens.jax.inference import HarryModellingSequence
from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.model import PhysicalModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

import tensorflow_probability.substrates.jax as tfp
import jax
from jax import random
import numpy as np
import optax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import optax
from helpers import *
tfd = tfp.distributions

# jax.distributed.initialize(
#     coordinator_address="localhost:12346",
#     num_processes=1,
#     process_id=0
# )
jax.distributed.initialize()

# %%
import json
from myfunctions import readJson

prior, phys_model_back, phys_model_for, singlesources = readJson("prior_shapelets.json")

# %%
from astropy.io import fits

f=fits.open('psf238.fits') 
psf=jnp.array(f[0].data).astype(jnp.float32)

observed_img = np.load("cutout238b.npy").astype(np.float32)

f=fits.open('final_94_drz.fits')
background_rms=0.00766512
exp_time=f[0].header["EXPTIME"]
deltaPix = f[0].header["D002SCAL"]
numPix = np.shape(observed_img)[0]
print(f"Numpix:",numPix, "DeltaPix:",deltaPix)

# %%
kernel = psf #np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
# observed_img = np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy')).astype(np.float32)
sim_config = SimulatorConfig(delta_pix=deltaPix, num_pix=numPix, supersample=2, kernel=kernel)
# phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=True)], [sersic.SersicEllipse(use_lstsq=True)])
lens_sim = LensSimulator(phys_model_back, sim_config, bs=1)

# systems_dir = os.path.join(home, "GIGALens-Code/SystemSaves")
# f = np.load(os.path.join(systems_dir, "100SystemsStandard80px.npz"))
# observed_imgs = jnp.array([f[key] for key in f.files])

# observed_img = observed_imgs[4]

prob_model = BackwardProbModel(prior, observed_img, background_rms=background_rms, exp_time=exp_time)
model_seq = HarryModellingSequence(phys_model_back, prob_model, sim_config)

# %%
map_optimizer = optax.adabelief(1e-2, b1=0.95, b2=0.99) #nesterov=True may not be implemented in current optax
map_estimate, map_loss_hist = model_seq.MAP(map_optimizer, seed=0, n_samples=500, num_steps=350)

# %%
# plt.plot(map_loss_hist)
# plt.ylim(bottom=0, top=3)
# plt.show()

# %%
# lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=500), map_estimate)[0]
# best = map_estimate[jnp.nanargmax(lps)][jnp.newaxis,:]
map_loss_history = jnp.min(map_loss_hist, axis=1)
best_step_idx = jnp.argmin(map_loss_history)
best_sample_idx = jnp.argmin(map_loss_hist[best_step_idx])

best = map_estimate[best_step_idx][best_sample_idx][jnp.newaxis, :].reshape((-1, map_estimate.shape[-1]))
best_x = prob_model.bij.forward(list(best.T))

# %%
svi_optimizer = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = model_seq.SVI(best, svi_optimizer, n_vi=600, num_steps=500)

# %%
# plt.plot(loss_hist)

# %%
samples = model_seq.HMC(qz, num_burnin_steps=250, num_results=750)

# %%
smp = jnp.transpose(samples, (1, 2, 0, 3)).reshape((-1, samples.shape[-1]))
hmc_median = jnp.median(smp, axis=0)[jnp.newaxis,:]

# %%
rhat= tfp.mcmc.potential_scale_reduction(jnp.transpose(samples, (1,2,0,3)), independent_chain_ndims=2)

# %%
rhat


