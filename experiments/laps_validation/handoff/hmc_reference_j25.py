#!/usr/bin/env python
# coding: utf-8
r"""Converged BASIC-HMC reference posterior for the gigalens lens demo.

This is the jax-2025 (image ``jax-2025-06-07`` / ``gigalens_multinode_env``)
HMC-only variant of ``hmc_reference.py``. The LAPS warm/prior + overlay parts
have been REMOVED -- LAPS runs on the jax-2026 image and the orchestrator does
the LAPS + overlay comparison separately. This script produces ONLY the HMC
reference posterior and its convergence diagnostics.

RUN ON GPU (the orchestrator does the full-length run; do NOT run during review):

    PYTHONPATH=<gigalens_multinode_env site-packages>:\
              /global/u1/l/linusu/gigalens/src:\
              /global/u1/l/linusu/GIGALens-Code/src \
    srun --jobid=<JOBID> --ntasks=1 shifter --module=gpu,nccl-plugin \
         --image=docker:ghcr.io/nvidia/jax:jax-2025-06-07 \
         bash -c "python hmc_reference_j25.py"

What it does
------------
1. Builds the NEW scene-API model + MAP + full SVI ONCE -> model_seq, qz, markers
   (reused verbatim from experiments/laps_validation/jax-demo.ipynb).
2. BASIC HMC reference -- gigalens' own basic (preconditioned) HMC, NOT NUTS:
       samples = model_seq.HMC(qz, num_burnin_steps=500, num_results=1500,
                               pbar_interval=25)
   Return shape: (num_results, num_devices, n_hmc_per_device, num_params)
   = (1500, D, 50//D, 22) -- axis 0 = draws, axes 1,2 = independent chains,
   axis 3 = the 22 unconstrained params.
3. Convergence check on the LONG chains (here R-hat IS reliable):
       rhat = tfp.mcmc.potential_scale_reduction(samples, independent_chain_ndims=2)
       ess  = tfp.mcmc.effective_sample_size(samples, cross_chain_dims=[1,2])
   PASS iff max(R-hat) < 1.01.
4. Convert HMC samples to physical space (reshape (-1,22) -> bij.forward), extract
   the 8 mass params in MASS_ORDER -> hmc_mass (N,8); saved to hmc_ref/hmc_mass.npy.
5. Standalone HMC reference corner (with truths) -> hmc_ref/hmc_corner.png;
   R-hat + per-config mean+/-std -> hmc_ref/hmc_summary.json.
"""
import os
import json

import jax
jax.config.update("jax_enable_x64", True)

# NEW gigalens "scene" API
from gigalens.jax.inference import ModellingSequence
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

import tensorflow_probability.substrates.jax as tfp
import numpy as np
import optax
from jax import numpy as jnp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from corner import corner

tfd = tfp.distributions

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hmc_ref")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Build model + MAP + SVI ONCE (verbatim from jax-demo.ipynb)              #
# --------------------------------------------------------------------------- #
epl_p = dict(
    theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
    gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
    e1=tfd.Normal(0, 0.1),
    e2=tfd.Normal(0, 0.1),
    center_x=tfd.Normal(0, 0.05),
    center_y=tfd.Normal(0, 0.05),
)
shear_p = dict(
    gamma1=tfd.Normal(0, 0.05),
    gamma2=tfd.Normal(0, 0.05),
)
lens_light_p = dict(
    R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15),
    n_sersic=tfd.Uniform(2, 6),
    e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
    e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
    center_x=tfd.Normal(0, 0.05),
    center_y=tfd.Normal(0, 0.05),
    Ie=tfd.LogNormal(jnp.log(500.0), 0.3),
)
source_light_p = dict(
    R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15),
    n_sersic=tfd.Uniform(0.5, 4),
    e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
    e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
    center_x=tfd.Normal(0, 0.25),
    center_y=tfd.Normal(0, 0.25),
    Ie=tfd.LogNormal(jnp.log(150.0), 0.5),
)

truth = [[
    {'theta_E': 1.1, 'gamma': 2.0, 'e1': 0.1, 'e2': 0.1, 'center_y': 0.0, 'center_x': 0.1},
    {'gamma2': 0.03, 'gamma1': -0.01}
], [
    {'R_sersic': 0.8, 'n_sersic': 2.5, 'e1': 0.09534746574143645, 'e2': 0.14849487967198177, 'center_x': 0.1, 'center_y': 0.0, 'Ie': 499.3695906504067}
], [
    {'R_sersic': 0.25, 'n_sersic': 1.5, 'e1': 0., 'e2': 0., 'center_x': 0.09566681002252231, 'center_y': -0.0639623054267272, 'Ie': 149.58828877085668}
]]

ASSETS = "/global/u1/l/linusu/gigalens/src/gigalens/assets"
kernel = np.load(f"{ASSETS}/psf.npy").astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=2, kernel=kernel)

lens_light = Component(sersic.SersicEllipse(use_lstsq=False), lens_light_p)
source_light = Component(sersic.SersicEllipse(use_lstsq=False), source_light_p)
model = LensModel([
    Plane(mass=[Component(epl.EPL(50), epl_p), Component(shear.Shear(), shear_p)],
          light=[lens_light]),
    Plane(deflection_ratio=1.0, light=[source_light]),
])

observed_img = np.load(f"{ASSETS}/demo.npy")
ds = Dataset(jnp.asarray(observed_img), sim_config,
             background_rms=0.2, exp_time=100, sees="all")
prob_model = ProbModel(model, ds, mode="forward")
model_seq = ModellingSequence(prob_model)
DIM = int(model.num_free_params)
print("dim (num free params):", DIM)

# MAP then SVI -> qz (verbatim defaults from the demo)
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = model_seq.MAP(opt, seed=0)

opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = model_seq.SVI(best, opt, n_vi=1000, num_steps=1500)

# truth markers for the 8 mass params (EPL 6 + Shear 2), demo dict order
markers = []
for i in truth[0][0].keys():
    markers.append(truth[0][0][i])
for j in truth[0][1].keys():
    markers.append(truth[0][1][j])

# 8 mass params, ordered to match `markers` and `labels`
MASS_ORDER = [
    "planes/0/mass/0/theta_E",
    "planes/0/mass/0/gamma",
    "planes/0/mass/0/e1",
    "planes/0/mass/0/e2",
    "planes/0/mass/0/center_y",
    "planes/0/mass/0/center_x",
    "planes/0/mass/1/gamma2",
    "planes/0/mass/1/gamma1",
]
labels = [r'$\theta_E$', r'$\gamma$', r'$\epsilon_2$', r'$\epsilon_1$',
          r'$y$', r'$x$', r'$\gamma_{2,ext}$', r'$\gamma_{1,ext}$']


def to_mass(samples_z):
    """(N, 22) UNCONSTRAINED z  ->  (N, 8) physical mass params in MASS_ORDER."""
    smp = np.asarray(samples_z).reshape((-1, DIM))           # (N, 22)
    phys = prob_model.bij.forward(list(jnp.asarray(smp).T))  # dict: key -> (N,)
    return np.stack([np.asarray(phys[k]) for k in MASS_ORDER], axis=1)  # (N, 8)


# --------------------------------------------------------------------------- #
# 2. BASIC HMC reference (gigalens basic HMC, NOT NUTS) -- doubled burnin/results
# --------------------------------------------------------------------------- #
print("\n=== BASIC HMC reference (num_burnin_steps=500, num_results=1500) ===")
samples = model_seq.HMC(qz, num_burnin_steps=500, num_results=1500, pbar_interval=25)
samples = np.asarray(samples)
print("HMC samples shape (draws, chain1, chain2, params):", samples.shape)

# --------------------------------------------------------------------------- #
# 3. Convergence check -- exactly as the notebook (axes per shape above)       #
# --------------------------------------------------------------------------- #
rhat = np.asarray(tfp.mcmc.potential_scale_reduction(samples, independent_chain_ndims=2))
ess = np.asarray(tfp.mcmc.effective_sample_size(samples, cross_chain_dims=[1, 2]))
print("\nper-param R-hat / ESS (22 unconstrained params):")
for j in range(rhat.shape[0]):
    print(f"  z{j:02d}  Rhat={rhat[j]:.4f}   ESS={ess[j]:.0f}")
rhat_max = float(np.nanmax(rhat))
print(f"\nmax(R-hat) = {rhat_max:.4f}")
hmc_good = rhat_max < 1.01
print(f"HMC REFERENCE CONVERGENCE: {'PASS' if hmc_good else 'FAIL'} "
      f"(criterion max(R-hat) < 1.01; got {rhat_max:.4f})")

# --------------------------------------------------------------------------- #
# 4. HMC -> physical mass params, save                                        #
# --------------------------------------------------------------------------- #
hmc_mass = to_mass(samples)                          # (N, 8)
np.save(os.path.join(OUT, "hmc_mass.npy"), hmc_mass)
print(f"\nsaved hmc_ref/hmc_mass.npy  shape={hmc_mass.shape}")

# --------------------------------------------------------------------------- #
# 5. Standalone HMC reference corner + summary.json                           #
# --------------------------------------------------------------------------- #
lo = hmc_mass.min(axis=0)
hi = hmc_mass.max(axis=0)
pad = 0.05 * (hi - lo)
pad = np.where(pad > 0, pad, 1e-6)
rng = list(zip(lo - pad, hi + pad))

fig = corner(hmc_mass, color="black", truths=markers, truth_color="green",
             show_titles=True, title_fmt=".3f", labels=labels, range=rng, bins=30)
fig.suptitle("HMC reference posterior (lens mass params)")
fig.savefig(os.path.join(OUT, "hmc_corner.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved hmc_ref/hmc_corner.png")


def mstd(a):
    return dict(mean=a.mean(axis=0).tolist(), std=a.std(axis=0).tolist())


summary = {
    "params": MASS_ORDER,
    "labels": labels,
    "truth": [float(m) for m in markers],
    "hmc": mstd(hmc_mass),
    "hmc_rhat_max": rhat_max,
    "hmc_rhat_per_param": rhat.tolist(),
    "hmc_ess_per_param": ess.tolist(),
    "hmc_converged": bool(hmc_good),
    "n_hmc": int(hmc_mass.shape[0]),
}
with open(os.path.join(OUT, "hmc_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("saved hmc_ref/hmc_summary.json")

# --------------------------------------------------------------------------- #
# 6. Compact per-param table -- HMC vs truth                                  #
# --------------------------------------------------------------------------- #
hm, hs = hmc_mass.mean(0), hmc_mass.std(0)
print("\n" + "=" * 60)
print(f"{'param':<22}{'truth':>9}{'HMC mean+/-std':>24}")
print("-" * 60)
for k in range(len(MASS_ORDER)):
    print(f"{MASS_ORDER[k]:<22}{markers[k]:>9.3f}"
          f"{hm[k]:>13.3f}+/-{hs[k]:<6.3f}")
print("=" * 60)
print(f"HMC convergence: {'PASS' if hmc_good else 'FAIL'} (max R-hat={rhat_max:.4f})")
