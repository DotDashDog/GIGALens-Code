#!/usr/bin/env python
# coding: utf-8
r"""Phase-2 acceptance test: does full-LAPS prior-init fail because Phase-2 (MAMS)
acceptance collapses from a cold Phase-1 ensemble? Reproduces the failing default
config (num_unadjusted_steps=300, num_adjusted_steps=200) for init in {prior,warm},
dumps p2_accept trajectory + final physical width vs HMC. PRODUCES NUMBERS, no verdict.
Model build verbatim from laps_overlay_j26.py / diag_stepsize_run.py.
"""
import os, json
import jax
jax.config.update("jax_enable_x64", True)
from gigalens.jax.inference import ModellingSequence
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
import tensorflow_probability.substrates.jax as tfp
import numpy as np, optax
from jax import numpy as jnp
tfd = tfp.distributions
from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted_JIT
print("jax devices:", jax.devices())

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_p2accept"); os.makedirs(OUT, exist_ok=True)

epl_p = dict(theta_E=tfd.LogNormal(jnp.log(1.25), 0.25), gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
    e1=tfd.Normal(0, 0.1), e2=tfd.Normal(0, 0.1), center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05))
shear_p = dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
lens_light_p = dict(R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15), n_sersic=tfd.Uniform(2, 6),
    e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3), e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
    center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05), Ie=tfd.LogNormal(jnp.log(500.0), 0.3))
source_light_p = dict(R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15), n_sersic=tfd.Uniform(0.5, 4),
    e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5), e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
    center_x=tfd.Normal(0, 0.25), center_y=tfd.Normal(0, 0.25), Ie=tfd.LogNormal(jnp.log(150.0), 0.5))

ASSETS = "/global/u1/l/linusu/gigalens/src/gigalens/assets"
kernel = np.load(f"{ASSETS}/psf.npy").astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=2, kernel=kernel)
lens_light = Component(sersic.SersicEllipse(use_lstsq=False), lens_light_p)
source_light = Component(sersic.SersicEllipse(use_lstsq=False), source_light_p)
model = LensModel([
    Plane(mass=[Component(epl.EPL(50), epl_p), Component(shear.Shear(), shear_p)], light=[lens_light]),
    Plane(deflection_ratio=1.0, light=[source_light])])
observed_img = np.load(f"{ASSETS}/demo.npy")
ds = Dataset(jnp.asarray(observed_img), sim_config, background_rms=0.2, exp_time=100, sees="all")
prob_model = ProbModel(model, ds, mode="forward")
model_seq = ModellingSequence(prob_model)
DIM = int(model.num_free_params); print("dim:", DIM)
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = model_seq.MAP(opt, seed=0)
print("MAP best_chisq (min):", float(np.min(np.asarray(best_chisq))))
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, _ = model_seq.SVI(best, opt, n_vi=1000, num_steps=1500); print("SVI done.", flush=True)


import numpy as np
OUT3=os.path.join(HERE,"diag_precond"); os.makedirs(OUT3,exist_ok=True)
def perdim_std(res): return np.asarray(res.samples).reshape((-1,DIM)).std(0)
qz_var=np.asarray(qz.sample((4096,),seed=jax.random.PRNGKey(123))).var(0)  # correct per-dim metric
print("qz_var min/max:",float(qz_var.min()),float(qz_var.max()),flush=True)
runs={}
print("\n=== warm ref (default) ===",flush=True)
runs["warm"]=perdim_std(LAPS_late_adjusted_JIT(model_seq,qz,init_mode="warm",num_chains=512,seed=0))
print("=== prior + CORRECT precond (default budget) ===",flush=True)
runs["prior_precond"]=perdim_std(LAPS_late_adjusted_JIT(model_seq,qz,init_mode="prior",num_chains=512,seed=0,p2_precond_var=qz_var))
print("=== prior + CORRECT precond + big budget (1000 unadj + 500 adj) ===",flush=True)
runs["prior_precond_big"]=perdim_std(LAPS_late_adjusted_JIT(model_seq,qz,init_mode="prior",num_chains=512,seed=0,p2_precond_var=qz_var,num_unadjusted_steps=1000,num_adjusted_steps=500))
warm=runs["warm"]; order=np.argsort(warm)
for name in ("prior_precond","prior_precond_big"):
    r=runs[name]/warm
    print(f"\n--- {name}: max ratio={r.max():.1f} median={np.median(r):.1f} (1.0=converged; prior-default baseline was ~236x max / 22x median) ---")
    print(f"{'dim':>3} {'warm_std':>11} {'this_std':>12} {'ratio':>8}")
    for i in order[:8]:
        print(f"{i:>3} {warm[i]:11.3e} {runs[name][i]:12.3e} {r[i]:8.1f}")
np.savez(os.path.join(OUT3,"precond.npz"),qz_var=qz_var,**runs)
print("\nDIAG DONE",flush=True)
