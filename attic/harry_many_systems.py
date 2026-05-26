import os
# Suppress warnings and errors for this demo (else, we get a lot of XLA timer warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Reproducibility flag for JAX (slight performance hit)
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import sys


import socket
import jax

# Initialize distributed JAX with full GPU visibility
# jax.distributed.initialize(
#     # coordinator_address=os.environ.get("JAX_COORDINATOR_ADDR"),
#     # num_processes=int(os.environ.get("SLURM_NTASKS")),
#     # process_id=int(os.environ.get("SLURM_PROCID")),
#     local_device_ids=None  # Allow access to all local GPUs
# )

jax.distributed.initialize(
    coordinator_address="localhost:12346",
    num_processes=1,
    process_id=0
)
# sys.path.append(f'{os.environ['HOME']}/gigalens_personal/gigalens/src')
# sys.path.append(f'{os.environ['HOME']}/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages')

home = os.path.expanduser("~/")
# srcdir = os.path.join(home, "gigalens-multinode/gigalens_hackathon/src/")
srcdir = os.path.join(home, "gigalens/src/")
sys.path.insert(0, srcdir)
sys.path.insert(0, home+'/GIGALens-Code')

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
from gigalens.jax.model import ForwardProbModel
from gigalens.jax.inference import HarryModellingSequence
import optax

import matplotlib.pyplot as plt

import scipy
import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors

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

def get_noise_image(image, background_rms, exp_time):
    return np.sqrt(image / exp_time + background_rms**2)

lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
                gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
                e1=tfd.Normal(0, 0.1),
                e2=tfd.Normal(0, 0.1),
                center_x=tfd.Normal(0, 0.05),
                center_y=tfd.Normal(0, 0.05),
            )
        ),
        tfd.JointDistributionNamed(
            dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
        ),
    ]
)
lens_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
                n_sersic=tfd.Uniform(2, 6),
                e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                center_x=tfd.Normal(0, 0.05),
                center_y=tfd.Normal(0, 0.05),
                Ie=tfd.LogNormal(jnp.log(500.0), 0.3),
            )
        )
    ]
)

source_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
                n_sersic=tfd.Uniform(0.5, 4),
                e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                center_x=tfd.Normal(0, 0.25),
                center_y=tfd.Normal(0, 0.25),
                Ie=tfd.LogNormal(jnp.log(150.0), 0.5),
            )
        )
    ]
)

prior = tfd.JointDistributionSequential(
    [lens_prior, lens_light_prior, source_light_prior]
)

kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
# delta_pix misleading when supersample != 1? Actual angular resolution is delta_pix / supersample?
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=100, supersample=2, kernel=kernel)
phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
lens_sim = LensSimulator(phys_model, sim_config, bs=1)

background_rms = 0.2
exp_time = 100

from enum import Enum

class LensingConfig(Enum):
    CUSP = "cusp"
    CROSS = "cross"
    FOLD = "fold"
    DOUBLE = "double"

def get_sample_from_config(lensing_config: LensingConfig):
    sample = prior.sample(1, seed=jax.random.PRNGKey(21))
    for li in sample:
        for d in li:
            for k in d.keys():
                d[k] = d[k].item()
    kwargs_source = sample[2][0]
    if lensing_config == LensingConfig.CUSP or lensing_config == 'cusp':
        kwargs_source['center_x'] += 0.05
        kwargs_source['center_y'] -= 0.1
    elif lensing_config == LensingConfig.CROSS or lensing_config == 'cross':
        kwargs_lens = sample[0][0]
        kwargs_source['center_x'] = kwargs_lens['center_x']
        kwargs_source['center_y'] = kwargs_lens['center_y']
    elif lensing_config == LensingConfig.FOLD or lensing_config == 'fold':
        kwargs_source['center_x'] += 0.1
        kwargs_source['center_y'] += 0.05
    elif lensing_config == LensingConfig.DOUBLE or lensing_config == 'double':
        y = 0.07
        gamma1 = y * np.cos(1)
        gamma2 = y * np.sin(1)
        lens_params = [{'theta_E': 1.51, 'gamma': 2.34, 'e1': .174, 'e2': .03, 'center_x': 0, 'center_y': -0},
        {'gamma1': gamma1, 'gamma2': gamma2}]
        lens_light_params = [{'R_sersic': 1.5, 'n_sersic': 3, 'e1': 0.05, 'e2': 0.05, 'center_x': 0, 'center_y': 0, 'Ie': 50}]
        source_light_params = [{'R_sersic': 0.2, 'n_sersic': 1, 'e1': 0, 'e2': 0.1, 'center_x': -0.2, 'center_y': 0.3, 'Ie': 500}
        ]
        sample = [lens_params, lens_light_params, source_light_params]
    else:
        raise Exception('Unknown lensing_config')
    return sample


def run(seed):
    output_dir = os.path.join(os.environ['PSCRATCH'], 'gigalens_multinode', str(seed))
    if jax.process_index() == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    if jax.process_index() == 0:
        print(f'Starting run with seed {seed}')
    jax.experimental.multihost_utils.sync_global_devices("run_start")
    
    np.random.seed(seed)
    sample = prior.sample(1, seed=jax.random.PRNGKey(seed))
    
    sample_image = lens_sim.simulate(sample)
    observed_image = get_noisy_image(sample_image, background_rms=background_rms, exp_time=exp_time)
    noise_image = get_noise_image(sample_image, background_rms=background_rms, exp_time=exp_time)

    
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
    prob_model = ForwardProbModel(prior, observed_image, background_rms=background_rms, exp_time=exp_time)
    model_seq = HarryModellingSequence(phys_model, prob_model, sim_config)

    prev_time = time.time()
    map_samples, map_losses = model_seq.MAP(opt, seed=0, num_steps=350)
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
    qz, loss_history = model_seq.SVI(best, opt, n_vi=1000, num_steps=1000)
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
    samples = model_seq.HMC(qz, n_hmc=64,num_burnin_steps=250, num_results=10000)
    hmc_time = time.time() - prev_time
    if jax.process_index() == 0:
        print(f'HMC time: {hmc_time}')
    if jax.process_index() == 0:
        np.save(os.path.join(output_dir, 'hmc_samples.npy'), samples)

for i in range(10):
    run(i)
