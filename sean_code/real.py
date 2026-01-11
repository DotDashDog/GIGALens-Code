import os
# Suppress warnings and errors for this demo (else, we get a lot of XLA timer warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Reproducibility flag for JAX (slight performance hit)
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import sys


import socket
import jax

# Initialize distributed JAX with full GPU visibility
jax.distributed.initialize(
    # coordinator_address=os.environ.get("JAX_COORDINATOR_ADDR"),
    # num_processes=int(os.environ.get("SLURM_NTASKS")),
    # process_id=int(os.environ.get("SLURM_PROCID")),
    local_device_ids=None  # Allow access to all local GPUs
)

sys.path.append(f'{os.environ['HOME']}/multinode2.0/src')
sys.path.append(f'{os.environ['HOME']}/.conda/envs/gigajax3.0/lib/python3.12/site-packages')

if jax.process_index() == 0:
    print(sys.path)

    print(f"Hostname: {socket.gethostname()}")
    # print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    print(f"Visible JAX devices: {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from jax import numpy as jnp
import numpy as np
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.profiles.light import sersic
from gigalens.jax.simulator import LensSimulator
from gigalens.jax.model import ForwardProbModel, BackwardProbModel
from gigalens.jax.inference import ModellingSequence
import optax

import matplotlib.pyplot as plt

import scipy
import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors

import myfunctions 

def imshow_with_colorbar(ax, data, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(data, cmap='viridis', **kwargs)
    ax.get_figure().colorbar(im, cax=cax, orientation='vertical')
    
def imshow_with_colorbar_sqrt_scale(ax, data, **kwargs):
    imshow_with_colorbar(ax, data, norm=matplotlib.colors.PowerNorm(gamma=0.5), **kwargs)

import lenstronomy.Util.image_util as image_util
# exp_time: exposure time to quantify the Poisson noise level
# background_rms: background rms value
def get_noisy_image(image, background_rms, exp_time):
    poisson_noise = image_util.add_poisson(image, exp_time=exp_time)
    bkg_noise = image_util.add_background(image, sigma_bkd=background_rms)
    image_noisy = image + bkg_noise + poisson_noise
    return image_noisy

# new_pixel_value = orig_pixel_value + poisson_noise + bkg_noise
# new_pixel_value ~ Poisson(orig_pix_val / exp_time) + N(0, background_rms^2)
# Var(new_pix_val) = orig_pix_val / exp_time + background_rms^2
# residual = new_pix_val - old_pix_val
# E[residual] = 0; Var(residual) = Var(new_pix_val)

#load basic information
from astropy.io import fits
f=fits.open('/global/homes/s/seanjx/gigalens/238/psf238.fits') 
psf=jnp.array(f[0].data)
observed_img = np.load("/global/homes/s/seanjx/gigalens/238/cutout238b.npy")
f=fits.open('/global/homes/s/seanjx/gigalens/238/final_94_drz.fits')
background_rms=0.00766512
exp_time=f[0].header["EXPTIME"]
deltaPix = f[0].header["D002SCAL"]
numPix = np.shape(observed_img)[0]

# observed_img = np.load("/global/homes/s/seanjx/multinode/src/gigalens/assets/demo.npy")
# psf = np.load("/global/homes/s/seanjx/multinode/src/gigalens/assets/psf.npy")
# numPix = 60

path = "/global/homes/s/seanjx/gigalens/238/output/238_2025-04-26 15:27:11.797415"
prior, phys_model, phys_model_Forward, SingleSources = myfunctions.readJson(path+"/prior.json")[0:4]
err_map = np.load(path+"/err_map.npy")
sim_config = SimulatorConfig(delta_pix=deltaPix, num_pix=numPix, supersample=2, kernel=psf)
lens_sim = LensSimulator(phys_model, sim_config, bs=1)


def run(observed_image):
    observed_image = jnp.array(observed_image)
    
    output_dir = os.path.join(os.environ['PSCRATCH'], 'gigalens_multinode', '238')
    if jax.process_index() == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    if jax.process_index() == 0:
        print(f'Starting run on 238')
    jax.experimental.multihost_utils.sync_global_devices("run_start")

    fig, ax = plt.subplots()
    imshow_with_colorbar_sqrt_scale(ax, observed_image)
    if jax.process_index() == 0:
        plt.savefig(os.path.join(output_dir, 'observed.png'))
    
    # set up optimizer
    schedule_fn = optax.polynomial_schedule(init_value=-1e-2, end_value=-1e-2/3, 
                                          power=0.5, transition_steps=500)
    opt = optax.chain(
      # optax.scale_by_adam(),
      # optax.scale_by_schedule(schedule_fn),
      optax.adabelief(1e-2, b1=0.95, b2=0.99),  
    )
    
    # get MAP of Noisy
    prob_model = BackwardProbModel(prior, observed_image, background_rms=background_rms, exp_time=exp_time)
    model_seq = ModellingSequence(phys_model, prob_model, sim_config)

    prev_time = time.time()
    map_samples, map_losses = model_seq.MAP_multi(opt, seed=0, num_steps=1000)
    map_time = time.time() - prev_time
    if jax.process_index() == 0:
        print(f'MAP time: {map_time}')
    
    map_loss_history = jnp.min(map_losses, axis=1)
    best_step_idx = jnp.argmin(map_loss_history)
    best_sample_idx = jnp.argmin(map_losses[best_step_idx])

    # plot map loss
    fig, ax = plt.subplots()
    ax.set_title("MAP Loss")
    ax.plot(map_loss_history)
    ax.axvline(best_step_idx, linestyle='--')
    ax.axhline(map_loss_history[best_step_idx], linestyle='--', label=map_loss_history[best_step_idx])
    ax.legend()

    a = jax.experimental.multihost_utils.process_allgather(map_samples)
    b = jax.experimental.multihost_utils.process_allgather(map_losses)
    if jax.process_index() == 0:
        plt.savefig(os.path.join(output_dir, 'map_loss.png'))
        np.save(os.path.join(output_dir, 'map_samples.npy'), a)
        np.save(os.path.join(output_dir, 'map_losses.npy'), b)
    
    
    best = map_samples[best_step_idx][best_sample_idx][jnp.newaxis, :]
    map_x = prob_model.bij.forward(list(best.T))
    
    # SVI
    schedule_fn = optax.polynomial_schedule(init_value=-1e-6, end_value=-3e-3, 
                                          power=2, transition_steps=300)
    opt = optax.chain(
      # optax.scale_by_adam(),
      # optax.scale_by_schedule(schedule_fn),
      optax.adabelief(2e-3, b1=0.95, b2=0.99),
    )
    
    prev_time = time.time()
    qz, loss_history = model_seq.SVI_multi(best, opt, n_vi=1000, num_steps=1000)
    svi_time = time.time() - prev_time
    if jax.process_index() == 0:
        print(f'SVI time: {svi_time}')
    
    fig, ax = plt.subplots()
    ax.set_title("SVI Loss")
    ax.plot(loss_history)
    min_loss_idx = min(range(len(loss_history)), key=lambda idx: loss_history[idx])
    ax.axvline(min_loss_idx, linestyle='--')
    ax.axhline(loss_history[min_loss_idx], linestyle='--', label=loss_history[min_loss_idx])
    ax.legend()
    if jax.process_index() == 0:
        plt.savefig(os.path.join(output_dir, 'svi_loss.png'))
        jnp.save(os.path.join(output_dir, 'loss_history.npy'), jnp.array(loss_history))
        jnp.save(os.path.join(output_dir, 'qz_scale_tril.npy'), qz.scale_tril)
        jnp.save(os.path.join(output_dir, 'qz_loc.npy'), qz.loc)

    prev_time = time.time()
    samples, replacement_cov = model_seq.HMC_alt_multi(qz, n_hmc=16, n_vi=1000, num_burnin_steps=250, num_results=750)
    hmc_time = time.time() - prev_time
    if jax.process_index() == 0:
        print(f'HMC time: {hmc_time}')
    if jax.process_index() == 0:
        np.save(os.path.join(output_dir, 'hmc_samples.npy'), samples)
        if replacement_cov is not None:
            np.save(os.path.join(output_dir, 'replacement_cov.npy'), replacement_cov)

run(observed_img)
