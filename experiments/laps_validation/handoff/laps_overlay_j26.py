#!/usr/bin/env python
# coding: utf-8
r"""Overlay the warm- and prior-init LAPS posteriors against the ALREADY-COMPUTED
BASIC-HMC reference posterior for the gigalens lens demo, to adjudicate which LAPS
init is correct.

RUN ON GPU on the jax-2026 image (the orchestrator does this; do NOT run during
static review):

    JAX_ENABLE_X64=1 python laps_overlay_j26.py

Differs from hmc_reference.py in exactly one way: the HMC chain is NOT run here.
Instead the saved reference `hmc_ref/hmc_mass.npy` (shape (N,8), already
physical-space mass params in MASS_ORDER) is LOADED. Everything else -- the
scene-API model + MAP + SVI, the physical-space mass extraction, the LAPS
warm/prior runs, and the 3-way overlay corner -- is reused verbatim.

What it does
------------
1. Builds the NEW scene-API model + MAP + SVI ONCE  -> prob_model, qz,
   markers (reused verbatim from hmc_reference.py / jax-demo.ipynb).
2. LOADS the HMC reference: hmc_mass = np.load("hmc_ref/hmc_mass.npy")  (N,8).
3. LAPS comparison: run_laps warm + prior, num_chains=512, p2_keep_per_chain=1
   (CLEAN ensemble -- no multi-sample amplification). Same physical extraction
   -> warm_mass, prior_mass; saved to hmc_ref/warm_mass.npy, hmc_ref/prior_mass.npy.
4. Overlay corner (HMC black w/ truths green, warm blue, prior orange; shared
   range=; identical bins; legend) -> hmc_ref/overlay_corner.png.
5. Per-param table + hmc_ref/overlay_summary.json: HMC/warm/prior mean+/-std,
   truth, and std_warm/std_hmc, std_prior/std_hmc ratios (~1 = matches reference;
   >>1 = over-dispersed).
"""
import os
import json

import jax
jax.config.update("jax_enable_x64", True)

# NEW gigalens "scene" API
from gigalens.jax.inference import MAP, SVI
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
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from corner import corner

tfd = tfp.distributions

import sys
sys.path.insert(0, "/global/u1/l/linusu/GIGALens-Code/experiments/laps_validation")
sys.path.insert(0, "/global/u1/l/linusu/GIGALens-Code/src")  # for gigalens_research
from handoff.laps_handoff import run_laps  # validated LAPS defaults baked in

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hmc_ref")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Build model + MAP + SVI ONCE (verbatim from hmc_reference.py)            #
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
DIM = int(model.num_free_params)
print("dim (num free params):", DIM)

# MAP then SVI -> qz (verbatim defaults from the demo)
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = MAP(prob_model, opt, seed=0)

opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = SVI(prob_model, best, opt, n_vi=1000, num_steps=1500)

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
# 2. LOAD the already-computed HMC reference (NOT re-run here)                 #
# --------------------------------------------------------------------------- #
hmc_mass_path = os.path.join(OUT, "hmc_mass.npy")
hmc_mass = np.load(hmc_mass_path)                    # (N, 8) physical, MASS_ORDER
assert hmc_mass.ndim == 2 and hmc_mass.shape[1] == len(MASS_ORDER), (
    f"hmc_mass has shape {hmc_mass.shape}; expected (N, {len(MASS_ORDER)})")
print(f"loaded HMC reference {hmc_mass_path}  shape={hmc_mass.shape}")

# --------------------------------------------------------------------------- #
# 3. LAPS comparison -- warm + prior init, CLEAN ensemble (1 sample/chain)     #
# --------------------------------------------------------------------------- #
print("\n=== LAPS warm-init (num_chains=512, p2_keep_per_chain=1) ===")
res_warm = run_laps(prob_model, qz, init_mode="warm", num_chains=512, p2_keep_per_chain=1)
print("\n=== LAPS prior-init (num_chains=512, p2_keep_per_chain=1) ===")
res_prior = run_laps(prob_model, qz, init_mode="prior", num_chains=512, p2_keep_per_chain=1)

warm_mass = to_mass(res_warm.samples)                # (512, 8)
prior_mass = to_mass(res_prior.samples)              # (512, 8)
np.save(os.path.join(OUT, "warm_mass.npy"), warm_mass)
np.save(os.path.join(OUT, "prior_mass.npy"), prior_mass)
print(f"warm_mass shape={warm_mass.shape}  prior_mass shape={prior_mass.shape}")

# --------------------------------------------------------------------------- #
# 4. Overlay corner with SHARED ranges + identical bins + legend              #
# --------------------------------------------------------------------------- #
# Shared per-param range spanning all three posteriors (so axes are identical).
allcat = np.concatenate([hmc_mass, warm_mass, prior_mass], axis=0)
lo = allcat.min(axis=0)
hi = allcat.max(axis=0)
pad = 0.05 * (hi - lo)
pad = np.where(pad > 0, pad, 1e-6)
rng = list(zip(lo - pad, hi + pad))
ckw = dict(labels=labels, range=rng, bins=30, plot_datapoints=False,
           smooth=1.0, hist_kwargs=dict(density=True))

# HMC drawn first (with truths); warm + prior overlaid on the same fig/axes.
fig = corner(hmc_mass, color="black", truths=markers,
             truth_color="green", **ckw)
corner(warm_mass, color="C0", fig=fig, **ckw)   # blue
corner(prior_mass, color="C1", fig=fig, **ckw)  # orange
handles = [
    Line2D([0], [0], color="black", lw=2, label="HMC reference"),
    Line2D([0], [0], color="C0", lw=2, label="LAPS warm"),
    Line2D([0], [0], color="C1", lw=2, label="LAPS prior"),
    Line2D([0], [0], color="green", lw=2, label="truth"),
]
fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=False)
fig.suptitle("Lens mass params: HMC reference vs LAPS warm vs LAPS prior", fontsize=14)
fig.savefig(os.path.join(OUT, "overlay_corner.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved hmc_ref/overlay_corner.png")

# --------------------------------------------------------------------------- #
# 5. Per-param table + overlay_summary.json -- width/center comparison        #
# --------------------------------------------------------------------------- #
hm, hs = hmc_mass.mean(0), hmc_mass.std(0)
wm, ws = warm_mass.mean(0), warm_mass.std(0)
pm, ps = prior_mass.mean(0), prior_mass.std(0)

# std ratios: ~1 => matches HMC width; >>1 => over-dispersed.
eps = 1e-12
ratio_warm = ws / np.where(hs > eps, hs, np.nan)
ratio_prior = ps / np.where(hs > eps, hs, np.nan)

print("\n" + "=" * 116)
print(f"{'param':<22}{'truth':>9}{'HMC mean+/-std':>20}"
      f"{'warm mean+/-std':>22}{'prior mean+/-std':>22}"
      f"{'sW/sH':>10}{'sP/sH':>10}")
print("-" * 116)
for k in range(len(MASS_ORDER)):
    print(f"{MASS_ORDER[k]:<22}{markers[k]:>9.3f}"
          f"{hm[k]:>11.3f}+/-{hs[k]:<6.3f}"
          f"{wm[k]:>11.3f}+/-{ws[k]:<6.3f}"
          f"{pm[k]:>11.3f}+/-{ps[k]:<6.3f}"
          f"{ratio_warm[k]:>10.2f}{ratio_prior[k]:>10.2f}")
print("=" * 116)
print("\nWidth read: std_warm/std_hmc ~ 1 => warm matches HMC width; "
      "std_prior/std_hmc >> 1 => prior OVER-dispersed. HMC is the adjudicator.")

summary = {
    "params": MASS_ORDER,
    "labels": labels,
    "truth": [float(m) for m in markers],
    "hmc": dict(mean=hm.tolist(), std=hs.tolist()),
    "warm": dict(mean=wm.tolist(), std=ws.tolist()),
    "prior": dict(mean=pm.tolist(), std=ps.tolist()),
    "std_ratio_warm_over_hmc": ratio_warm.tolist(),
    "std_ratio_prior_over_hmc": ratio_prior.tolist(),
    "n_hmc": int(hmc_mass.shape[0]),
    "n_warm": int(warm_mass.shape[0]),
    "n_prior": int(prior_mass.shape[0]),
}
with open(os.path.join(OUT, "overlay_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("saved hmc_ref/overlay_summary.json")
