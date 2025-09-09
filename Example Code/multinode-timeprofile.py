#%%
import corner
import time
import os
import sys
import json
from datetime import datetime
from os.path import expanduser

home = expanduser("~")
# sys.path.append("/global/homes/n/nratier/tests_hackathon/gigalens/src")
sys.path.insert(0, home+'/gigalens-multinode/gigalens_hackathon/src')
sys.path.insert(0, home+'/GIGALens-Code')
srcdir = os.path.join(home, "gigalens-multinode/gigalens_hackathon/src/")
print('MASTER BRANCH GIGALENS')

import jax
# jax.config.update("jax_enable_x64", True)

from gigalens.jax.inference import ModellingSequence
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
from helpers import cornerplot_labels, cornerplot_posterior, plot_image_results

print("\nStarting...")

# jax.distributed.initialize(local_device_ids=range(4)) # 4 nodes
# For single node execution without SLURM
jax.distributed.initialize(
    coordinator_address="localhost:12346",
    num_processes=1,
    process_id=0
)

# jax.distributed.initialize()

total_devices = jax.device_count() 

# devices accessible
verbose = (jax.process_index() == 0)
print(f"{jax.process_index()}: local devices: {jax.local_devices()}")
if verbose: print(f"Global devices: {jax.devices()}")

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

# kernel = np.load('/global/homes/n/nratier/tests_hackathon/gigalens/src/gigalens/assets/psf.npy').astype(np.float64)
kernel = np.load(os.path.join(srcdir, 'gigalens/assets/psf.npy')).astype(np.float32)
observed_img = np.load(os.path.join(srcdir, 'gigalens/assets/demo.npy'))

phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [sersic.SersicEllipse(use_lstsq=False)])
prob_model = ForwardProbModel(prior, observed_img, background_rms=0.2, exp_time=100)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=kernel) 

model_seq = ModellingSequence(phys_model, prob_model, sim_config)
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
print("\nmap ----------------")
schedule_fn = optax.polynomial_schedule(init_value=-1e-2, end_value=-1e-2/3, power=0.5, transition_steps=500)
opt = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(schedule_fn),
)
n_samples = 500 

# with jax.profiler.trace("GIGALens-Code/jax-trace", create_perfetto_link=True):
start = time.perf_counter()
map_estimate = model_seq.MAP(opt, n_samples=n_samples, num_steps=150, seed=0)
end = time.perf_counter()
map_time = end - start
print("\nTime for map:", map_time)


# n_samples_s = (n_samples // total_devices) * total_devices
# lps = prob_model.log_prob(LensSimulator(phys_model, sim_config, bs=n_samples_s), map_estimate)[0] 
# #print("\nlogprobs map shape:", lps.shape)
# #print(lps)
# select_index = lambda x: x[jnp.nanargmax(lps)]
# best = jax.tree.map(select_index, map_estimate)

# # map_best_x = prob_model.bij.forward(list(best.T))

# print("\nmap:", best)

# map_best_x = prob_model.bijector.forward(best)

# fig, axs = plt.subplots(1, 4)
# fig.set_size_inches(16,3)
# plot_image_results(fig, axs, observed_img, prefix="MAP",
#                     lens_sim=lens_sim, predicted_params=map_best_x, 
#                     resimulate=True)
# plt.savefig(os.path.join(home, "GIGALens-Code/plots/map_multinode.png"))
# plt.close()

# # pickle.dump(best, open(os.path.join(home, "GIGALens-Code/map_best_test.pkl"), "wb"))
# with open(os.path.join(home, "GIGALens-Code/map_best_test.pkl"), "rb") as f:
#     best = pickle.load(f)

# print("\nsvi ----------------")
# schedule_fn = optax.polynomial_schedule(init_value=-1e-6, end_value=-3e-3, 
#                                       power=2, transition_steps=300)
# opt = optax.chain(
#   optax.scale_by_adam(),
#   optax.scale_by_schedule(schedule_fn),
# )
# n_vi = 500
# start = time.perf_counter()
# qz, loss_hist, tree_struct = model_seq.SVI(best, opt, n_vi=n_vi, num_steps=1500, precision_parameterization=True)
# end = time.perf_counter()
# svi_time = end - start
# print("\nTime for svi:", svi_time)
# plt.plot(loss_hist)
# plt.savefig(os.path.join(home, "GIGALens-Code/plots/sviloss_multinode.png"))
# plt.close()

# # with open(os.path.join(home, "GIGALens-Code/svi_loss_hist.pkl"), "wb") as f:
# #     pickle.dump(loss_hist, f)
# #%%
# fig, axs = plt.subplots(1, 4)
# fig.set_size_inches(16,3)
# mean = prob_model.bijector.forward(jax.tree.unflatten(tree_struct, qz.mean()))
# plot_image_results(fig, axs, observed_img, prefix="SVI",
#                     lens_sim=lens_sim, predicted_params=mean, 
#                     resimulate=True)
# plt.savefig(os.path.join(home, "GIGALens-Code/plots/svi_multinode.png"))
# plt.close()

# labels = cornerplot_labels(map_best_x)
# svi_samples_z = qz.sample(1000, seed=jax.random.PRNGKey(0))
# print("\nsvi_samples_z", svi_samples_z.shape)
# svi_samples_x = prob_model.bijector.forward(jax.tree.unflatten(tree_struct, svi_samples_z.T))

# # print(jax.tree.flatten(svi_samples_x)[0].shape)
# # print(jax.tree.flatten(svi_samples_x)[1])
# # print("SVI NAN:", jnp.isnan(jnp.stack(jax.tree.flatten(svi_samples_x)[0])).any())
# fig = cornerplot_posterior(labels, svi_samples_x, overplots=map_best_x, color='blue')

# plt.savefig(os.path.join(home, "GIGALens-Code/plots/svi_corner_multinode.png"))
# plt.close()

# print("\nhmc ----------------")
# n_hmc = 64
# num_results = 750
# samples = model_seq.HMC(qz, tree_struct, n_hmc=n_hmc, num_burnin_steps=250, num_results=num_results)

# total_devices = jax.device_count() # * jax.local_device_count()
# n_hmc = (n_hmc // total_devices) * total_devices
# n_hmc_gpu = n_hmc // total_devices

# print("\nProcess index:", jax.process_index())
# print("\nSamples from HMC shape:", samples.all_states.shape)
# n_params = len(qz.mean())
# print("\nn_params:", n_params)

# mesh = jax.sharding.Mesh(jax.devices(), 'devices') 
# partition_spec_hmc = jax.sharding.PartitionSpec(None, 'devices')
# sharding_hmc = jax.sharding.NamedSharding(mesh, partition_spec_hmc) 
# shard_hmc_fn = lambda samples_gpu: jax.make_array_from_single_device_arrays((num_results, total_devices * n_hmc_gpu), sharding_hmc, [samples_gpu])

# samples = samples.all_states.transpose((2, 0, 1)) # (22, 750, 8)
# print('\nSamples shape after transpose:', samples.shape)
# # print("Expected sharding:\n", jax.debug.visualize_sharding(samples.shape, sharding_hmc))
# # print("Actual sharding:\n", jax.debug.visualize_array_sharding(samples))
# samples_gpu = jax.tree.unflatten(tree_struct, samples) # so first dim is n_params. tree on each gpu

# sharded_samples_hmc = jax.tree.map(shard_hmc_fn, samples_gpu)
# print("\nsharded_samples_hmc", sharded_samples_hmc)


# rhat_fn = lambda sharded_samples: tfp.mcmc.potential_scale_reduction(sharded_samples, independent_chain_ndims=1) # is 1 since we had (750, 8*total_devices)
# rhat = jax.tree.map(rhat_fn, sharded_samples_hmc)
# print("\nrhat", rhat)


# best_hmc = jax.tree.map(lambda x: jnp.median(x), sharded_samples_hmc)
# best_hmc_phys = prob_model.bijector.forward(best_hmc)
# print("\nbest_hmc_phys", best_hmc_phys)

# physical_samples = prob_model.bijector.forward(sharded_samples_hmc)

# fig, axs = plt.subplots(1, 4)
# fig.set_size_inches(16,3)
# plot_image_results(fig, axs, observed_img, prefix="HMC",
#                     lens_sim=lens_sim, predicted_params=best_hmc_phys, 
#                     resimulate=True)
# plt.savefig(os.path.join(home, "GIGALens-Code/plots/hmc_multinode.png"))
# plt.close()

# fig = cornerplot_posterior(labels, physical_samples, truth=best_hmc_phys, color='black')

# plt.savefig(os.path.join(home, "GIGALens-Code/plots/hmc_corner_multinode.png"))
# plt.close()

