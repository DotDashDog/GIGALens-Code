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
from gigalens.jax.inference import MAP, SVI
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
DIM = int(model.num_free_params); print("dim:", DIM)
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = MAP(prob_model, opt, seed=0)
print("MAP best_chisq (min):", float(np.min(np.asarray(best_chisq))))
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, _ = SVI(prob_model, best, opt, n_vi=1000, num_steps=1500); print("SVI done.", flush=True)

results = {}
for init in ("prior", "warm"):
    print(f"\n=== full LAPS init={init} (default budgets 300/200) ===", flush=True)
    res = LAPS_late_adjusted_JIT(prob_model, qz, init_mode=init, num_chains=512, seed=0)
    p2a = np.asarray(res.p2_accept)
    settled = np.asarray(res.p2_settled_accept)
    p2ss = np.asarray(res.p2_step_size)
    smp = np.asarray(res.samples).reshape((-1, DIM))
    results[init] = dict(
        p2_accept_first=float(p2a[0]), p2_accept_last=float(p2a[-1]),
        p2_accept_mean=float(np.mean(p2a)), p2_accept_min=float(np.min(p2a)), p2_accept_max=float(np.max(p2a)),
        settled=[float(x) for x in settled], p2_step_first=float(p2ss[0]), p2_step_last=float(p2ss[-1]),
        p2_final_step=float(res.p2_final_step_size), target_accept=float(res.target_accept),
        switched=bool(res.switched), switch_index=int(res.switch_index), phase1_len=int(res.phase1_len),
        n_p2_steps=int(p2a.shape[0]))
    print(f"[{init}] p2_accept first/mean/last = {p2a[0]:.4f}/{np.mean(p2a):.4f}/{p2a[-1]:.4f}  "
          f"target={float(res.target_accept)}  switched={bool(res.switched)} switch_idx={int(res.switch_index)} "
          f"p1_len={int(res.phase1_len)}  p2_final_eps={float(res.p2_final_step_size):.3e}", flush=True)
    np.save(os.path.join(OUT, f"{init}_samples_z.npy"), smp)

json.dump(results, open(os.path.join(OUT, "p2accept_summary.json"), "w"), indent=2)
print("\nSUMMARY:", json.dumps(results, indent=2))
print("DIAG DONE", flush=True)
