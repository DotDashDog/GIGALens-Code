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
OUT2=os.path.join(HERE,"diag_budget"); os.makedirs(OUT2,exist_ok=True)
def perdim_std(res):
    s=np.asarray(res.samples).reshape((-1,DIM)); return s.std(0)
# warm control (default budgets) = converged reference in sampling space
print("\n=== warm control (default) ===",flush=True)
rw=LAPS_late_adjusted_JIT(model_seq,qz,init_mode="warm",num_chains=512,seed=0)
warm_std=perdim_std(rw)
# prior at big budget
print("\n=== prior BIG budget (3000 unadj + 1500 adj) ===",flush=True)
rp=LAPS_late_adjusted_JIT(model_seq,qz,init_mode="prior",num_chains=512,seed=0,
        num_unadjusted_steps=3000,num_adjusted_steps=1500)
prior_std=perdim_std(rp)
ratio=prior_std/warm_std
order=np.argsort(warm_std)  # tightest sampling dims first
print(f"\nprior-BIG p2_accept last={float(np.asarray(rp.p2_accept)[-1]):.3f} p2_eps last={float(rp.p2_step_size[-1]):.3e} switch@{int(rp.switch_index)} p1len={int(rp.phase1_len)}")
print(f"{'dim':>3} {'warm_std':>10} {'priorBIG_std':>12} {'ratio':>8}")
for i in order[:10]:
    print(f"{i:>3} {warm_std[i]:10.3e} {prior_std[i]:12.3e} {ratio[i]:8.1f}")
print(f"\nSUMMARY prior-BIG/warm ratio: max={ratio.max():.1f} median={np.median(ratio):.1f} (1.0=converged)")
np.savez(os.path.join(OUT2,"budget.npz"),warm_std=warm_std,prior_std=prior_std,ratio=ratio)
print("DIAG DONE",flush=True)
