import jax

jax.distributed.initialize()

import sys

sys.path.append("/global/homes/e/evanod/gigalens-multinode-2026/src")

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import BackwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

import corner as corner
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
import time
import numpy as np
import optax
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, kstest
from astropy.visualization import simple_norm

tfd = tfp.distributions

# Showing all available devices
total_devices = jax.device_count()
verbose = jax.process_index() == 0
print(f"{jax.process_index()}: local devices: {jax.local_devices()}")
if verbose:
    print(f"Global devices: {jax.devices()}")

# priors!
lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                theta_E=tfd.LogNormal(jnp.log(2.0), 0.5),
                gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
                e1=tfd.TruncatedNormal(0, 0.25,-0.3, 0.3),
                e2=tfd.TruncatedNormal(0, 0.25, -0.3, 0.3),
                center_x=tfd.Normal(0, 0.1),
                center_y=tfd.Normal(0, 0.1),
            )
        ),
        tfd.JointDistributionNamed(
            dict(gamma1=tfd.Normal(0, 0.1), gamma2=tfd.Normal(0, 0.1))
        ),
    ]
)
lens_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.05), 0.1), #sean's has 0.05 width
                n_sersic=tfd.Uniform(1, 15), #seans has 1,10
                e1=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5), #sean's has -0.3, 0.3
                e2=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
                center_x=tfd.Normal(-3.25, 0.05),
                center_y=tfd.Normal(-3.25, 0.05)
            )
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.1),
                n_sersic=tfd.Uniform(1, 10),
                e1=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                center_x=tfd.Normal(0, 0.05),
                center_y=tfd.Normal(0, 0.05)
            )
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.1),
                n_sersic=tfd.Uniform(1, 10),
                e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                center_x=tfd.Normal(0, 0.05),
                center_y=tfd.Normal(0, 0.05)
            )
        )
    ]
)

source_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.2), 0.5),
                n_sersic=tfd.Uniform(0.01, 15), #seans has lower bound of 0.5
                e1=tfd.TruncatedNormal(0, 0.25, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0, 0.25, -0.5, 0.5),
                center_x=tfd.Normal(0, 0.5),
                center_y=tfd.Normal(0, 0.5)
            )
        ),
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.2), 0.5),
                n_sersic=tfd.Uniform(0.01, 15),
                e1=tfd.TruncatedNormal(0, 0.25, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0, 0.25, -0.5, 0.5),
                center_x=tfd.Normal(0, 0.5),
                center_y=tfd.Normal(0, 0.5)
            )
        )
    ]
)
prior = tfd.JointDistributionSequential(
    [lens_prior, lens_light_prior, source_light_prior]
)

print("Setting up models")
# set up models and whatnot
kernel = np.load("psf94.npy").astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=120, supersample=2, kernel=kernel)
phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=True),sersic.SersicEllipse(use_lstsq=True),sersic.SersicEllipse(use_lstsq=True)], [sersic.SersicEllipse(use_lstsq=True), sersic.SersicEllipse(use_lstsq=True)])
lens_sim = LensSimulator(phys_model, sim_config, bs=1)
observed_img = np.load('cutout238b.npy')
background_rms = 0.007616264 #background_rms from photutils.background
exp_time = 1197.699462 #exp_time from header["EXPTIME"]
prob_model = BackwardProbModel(prior, jnp.array(observed_img), background_rms=background_rms, exp_time=exp_time)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

# MAP
print("Starting MAP")
start = time.perf_counter()
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99, nesterov=True)
best, lps, chisq = model_seq.MAP(opt, seed=0, n_samples=5000, num_steps=500, pbar_interval=0)
end = time.perf_counter()
print("MAP time taken: ", (end - start))
np.save("best.npy", best)

# SVI
print("Starting SVI")
start = time.perf_counter()
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = model_seq.SVI(best, opt, n_vi=5000, num_steps=1050, pbar_interval=0)
jnp.save("svi_samples.npy", np.array(qz.sample(sample_shape=(50000), seed=jax.random.PRNGKey(0))))
plt.plot(loss_hist)
jnp.save("loss_hist", loss_hist)
plt.savefig("svi.png")
jnp.savez('qz.npz', loc=qz.loc, scale_tril=qz.scale_tril)
end = time.perf_counter()
print("SVI time taken: ", (end - start))

# HMC
print("Starting HMC")
start = time.perf_counter()
samples = model_seq.HMC(qz, n_hmc=64, num_burnin_steps=10000, num_results=10000, pbar_interval=0) #works with 10,20, trying with fewer
jnp.save("samples.npy", samples)
end = time.perf_counter()
print("HMC time taken: ", (end - start))

# get rhat and ess
start = time.perf_counter()
rhat= tfp.mcmc.potential_scale_reduction(samples, independent_chain_ndims=2)
ess = tfp.mcmc.effective_sample_size(samples, cross_chain_dims=[1,2])
numParams = len(rhat)
rhatess = []
for i in range(0,numParams):
    rhatess.append(str((i, rhat[i].item(), ess[i].item(), rhat[i].item()<1.01)))
with open("rhatess.txt", 'w') as file:
    for item in rhatess:
        file.write(f"{item}\n")

# convert HMC output into plottable samples
print("Starting Sampling + Plotting")
smp = jnp.transpose(samples, (1, 2, 0, 3)).reshape((-1, numParams))
smp_physical = prob_model.bij.forward(list(smp.T))
tups = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
label_prefixes = ["", "", "otherobj_", "lens1_", "lens2_", "src1_", "src2_"]
labels = []
for (i, j), label_prefix in zip(tups, label_prefixes):
    labels.extend((label_prefix + key for key in smp_physical[i][j].keys()))
median_params = [
    [
        {key: np.median(value) for key, value in d.items()}
        for d in list_of_dicts
    ]
    for list_of_dicts in smp_physical
]
with open("medianparams.txt", 'w') as file:
    for item in median_params:
        file.write(f"{item}\n")


# # plot HMC samples
# plt.style.use('default')
# plt_samples = np.vstack(
#     [np.array(list(smp_physical[i][j].values())) for i, j in tups]
# ).T
# fig = corner.corner(
#     plt_samples,
#     show_titles=True,
#     title_fmt=".3f",
#     labels=labels
# )
# _ = fig.suptitle("All HMC Parameters")
# plt.savefig("cornerplot.png")
# plt.close()
# end = time.perf_counter()
# print("Sampling + Plotting time taken: ", (end - start))

# # show final model
# print("Starting Final Model")
# err_map = np.sqrt(background_rms**2 + observed_img/exp_time)
# simulated = lens_sim.lstsq_simulate(median_params, jnp.array(observed_img), err_map)
# residual = simulated - jnp.array(observed_img)
# norm_residual = residual/err_map
# chisq = np.sum(np.square(norm_residual))
# dof = len(observed_img) * len(observed_img[0]) - numParams
# redchisq = chisq / dof
# norm = simple_norm(observed_img, "sqrt", percent=99.)
# fig, ax = plt.subplots(1,4, figsize=(25,4))
# image_sys = ax[0].imshow(observed_img, norm=norm, cmap="viridis", origin='lower')
# image_sim = ax[1].imshow(simulated, norm=norm, cmap="viridis", origin='lower')
# image_res = ax[2].imshow(residual, cmap="coolwarm", origin='lower')
# image_nres = ax[3].imshow(norm_residual, cmap="coolwarm", origin='lower')
# ax[0].set_title("Observed System")
# ax[1].set_title("Simulated System")
# ax[2].set_title("Residuals")
# ax[3].set_title(f"Normalized Residuals (redchisq: {redchisq:.3f})")
# plt.colorbar(image_res)
# plt.colorbar(image_nres)
# plt.savefig("finalmodel.png")
# plt.close()
# end = time.perf_counter()
# print("Final Model time taken: ", (end - start))