#!/usr/bin/env python
# coding: utf-8
r"""Phase-1 step-size controller diagnostic: prior-init vs warm-init LAPS.

PRODUCES NUMBERS + a plot; NO verdict. Builds the gigalens lens demo model ONCE
(scene API + MAP + SVI -> model_seq, qz, verbatim from laps_overlay_j26.py), then
runs LAPS_late_adjusted_JIT with phase2_enabled=False, num_unadjusted_steps=6000,
512 chains, seed=0 for init_mode in {prior, warm}, saving the Phase-1 controller
trajectories (eps, D-tilde, EEVPD obs/wanted, per-dim ensemble std).
"""
import os
import json

import jax
jax.config.update("jax_enable_x64", True)

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

tfd = tfp.distributions

from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted_JIT

print("jax devices:", jax.devices())

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_stepsize")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Build model + MAP + SVI ONCE (verbatim from laps_overlay_j26.py)         #
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

opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = model_seq.MAP(opt, seed=0)
MAP_CHISQ = float(np.min(np.asarray(best_chisq)))
print("MAP best_chisq (min):", MAP_CHISQ)

opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = model_seq.SVI(best, opt, n_vi=1000, num_steps=1500)
print("SVI done.")

# --------------------------------------------------------------------------- #
# 2. Run BOTH LAPS Phase-1-only runs (reproduce failing default config)       #
# --------------------------------------------------------------------------- #
NSTEP = 6000
results = {}
trajectories = {}

for init in ("prior", "warm"):
    print(f"\n=== LAPS Phase-1 diagnostic: init_mode={init} ===", flush=True)
    res = LAPS_late_adjusted_JIT(
        model_seq, qz, init_mode=init, num_chains=512,
        num_unadjusted_steps=NSTEP, num_adjusted_steps=1,
        early_stop=False, phase2_enabled=False, seed=0)

    ss = np.asarray(res.p1_step_size)          # (T,)
    Dt = np.asarray(res.p1_D_tilde)            # (T,)
    eo = np.asarray(res.p1_eevpd_obs)          # (T,)
    ew = np.asarray(res.p1_eevpd_wanted)       # (T,)
    obs_sq = np.asarray(res.p1_obs_sq)         # (T, dim)
    obs_mn = np.asarray(res.p1_obs_mean)       # (T, dim)
    std = np.sqrt(np.maximum(obs_sq - obs_mn**2, 0.0))   # (T, dim) per-dim ensemble std
    print(f"  phase1_len={res.phase1_len}  ss.shape={ss.shape}  std.shape={std.shape}", flush=True)

    trajectories[init] = dict(step_size=ss, D_tilde=Dt, eevpd_obs=eo,
                              eevpd_wanted=ew, ens_std=std)
    results[init] = dict(res_ss=ss, res_Dt=Dt, res_eo=eo, res_ew=ew, res_std=std)

# --------------------------------------------------------------------------- #
# 3. Identify tightest dims from WARM-final std; compare to prior at same dims #
# --------------------------------------------------------------------------- #
warm_std = results["warm"]["res_std"]
prior_std = results["prior"]["res_std"]
warm_final = warm_std[-1]                       # (dim,)
prior_final = prior_std[-1]                      # (dim,)
# tightest few dims = smallest warm-final std
n_tight = 3
tight_dims = np.argsort(warm_final)[:n_tight].tolist()
print("\ntightest warm dims (idx):", tight_dims)
print("warm-final std at tight dims:", warm_final[tight_dims].tolist())
print("prior-final std at tight dims:", prior_final[tight_dims].tolist())

# --------------------------------------------------------------------------- #
# 4. Summary JSON                                                             #
# --------------------------------------------------------------------------- #
def _idx(a, i):
    return float(a[i]) if i < len(a) else float("nan")

summary = {"n_steps": NSTEP, "dim": DIM, "n_tight": n_tight,
           "tight_dims": tight_dims,
           "MAP_chisq": MAP_CHISQ}
for init in ("prior", "warm"):
    ss = results[init]["res_ss"]; Dt = results[init]["res_Dt"]
    eo = results[init]["res_eo"]; ew = results[init]["res_ew"]
    std = results[init]["res_std"]
    summary[init] = dict(
        step_size_0=float(ss[0]),
        step_size_final=float(ss[-1]),
        step_size_min=float(np.min(ss)),
        step_size_max=float(np.max(ss)),
        D_tilde_0=float(Dt[0]),
        D_tilde_3000=_idx(Dt, 3000),
        D_tilde_final=float(Dt[-1]),
        eevpd_obs_final=float(eo[-1]),
        eevpd_wanted_final=float(ew[-1]),
        tight_dim_std_final=[float(std[-1][d]) for d in tight_dims],
    )
# tightest-dim std comparison prior vs warm target
summary["tight_dim_std_prior_final"] = [float(prior_final[d]) for d in tight_dims]
summary["tight_dim_std_warm_target"] = [float(warm_final[d]) for d in tight_dims]
summary["tight_dim_std_ratio_prior_over_warm"] = [
    float(prior_final[d] / warm_final[d]) if warm_final[d] > 0 else float("nan")
    for d in tight_dims]

with open(os.path.join(OUT, "stepsize_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nsaved stepsize_summary.json:")
print(json.dumps(summary, indent=2))

# --------------------------------------------------------------------------- #
# 5. Save raw arrays                                                          #
# --------------------------------------------------------------------------- #
np.savez(os.path.join(OUT, "stepsize_traj.npz"),
         prior_step_size=trajectories["prior"]["step_size"],
         prior_D_tilde=trajectories["prior"]["D_tilde"],
         prior_eevpd_obs=trajectories["prior"]["eevpd_obs"],
         prior_eevpd_wanted=trajectories["prior"]["eevpd_wanted"],
         prior_ens_std=trajectories["prior"]["ens_std"],
         warm_step_size=trajectories["warm"]["step_size"],
         warm_D_tilde=trajectories["warm"]["D_tilde"],
         warm_eevpd_obs=trajectories["warm"]["eevpd_obs"],
         warm_eevpd_wanted=trajectories["warm"]["eevpd_wanted"],
         warm_ens_std=trajectories["warm"]["ens_std"],
         tight_dims=np.asarray(tight_dims))
print("saved stepsize_traj.npz")

# --------------------------------------------------------------------------- #
# 6. Plot 2x2                                                                 #
# --------------------------------------------------------------------------- #
steps = np.arange(NSTEP)
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# (a) step_size
ax[0, 0].plot(steps, trajectories["prior"]["step_size"], "C0-", label="prior")
ax[0, 0].plot(steps, trajectories["warm"]["step_size"], "C1--", label="warm")
ax[0, 0].set_yscale("log")
ax[0, 0].set_title("(a) Phase-1 step size eps")
ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("eps"); ax[0, 0].legend()

# (b) D_tilde
ax[0, 1].plot(steps, trajectories["prior"]["D_tilde"], "C0-", label="prior")
ax[0, 1].plot(steps, trajectories["warm"]["D_tilde"], "C1--", label="warm")
ax[0, 1].set_yscale("log")
ax[0, 1].set_title("(b) equipartition divergence D-tilde")
ax[0, 1].set_xlabel("step"); ax[0, 1].set_ylabel("D-tilde"); ax[0, 1].legend()

# (c) EEVPD obs and wanted
ax[1, 0].plot(steps, trajectories["prior"]["eevpd_obs"], "C0-", label="prior obs")
ax[1, 0].plot(steps, trajectories["prior"]["eevpd_wanted"], "C0:", label="prior wanted")
ax[1, 0].plot(steps, trajectories["warm"]["eevpd_obs"], "C1--", label="warm obs")
ax[1, 0].plot(steps, trajectories["warm"]["eevpd_wanted"], "C1-.", label="warm wanted")
ax[1, 0].set_yscale("log")
ax[1, 0].set_title("(c) EEVPD observed vs wanted")
ax[1, 0].set_xlabel("step"); ax[1, 0].set_ylabel("EEVPD"); ax[1, 0].legend()

# (d) tightest-dim ensemble std, with warm-target horizontal lines
for k, d in enumerate(tight_dims):
    ax[1, 1].plot(steps, prior_std[:, d], "C0-", alpha=0.7,
                  label=f"prior dim{d}" if k == 0 else None)
    ax[1, 1].plot(steps, warm_std[:, d], "C1--", alpha=0.7,
                  label=f"warm dim{d}" if k == 0 else None)
    ax[1, 1].axhline(warm_final[d], color="k", ls=":", lw=0.8,
                     label="warm target" if k == 0 else None)
ax[1, 1].set_yscale("log")
ax[1, 1].set_title(f"(d) tightest-dim ensemble std (dims {tight_dims})")
ax[1, 1].set_xlabel("step"); ax[1, 1].set_ylabel("ensemble std"); ax[1, 1].legend()

fig.suptitle("LAPS Phase-1 step-size controller: prior (solid) vs warm (dashed)",
             fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stepsize.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved stepsize.png")

# --------------------------------------------------------------------------- #
# 7. Pre-registered primary numbers                                          #
# --------------------------------------------------------------------------- #
print("\n===== PRE-REGISTERED PRIMARY NUMBERS =====")
print(f"prior step_size[-1] = {summary['prior']['step_size_final']:.6g}")
print(f"warm  step_size[-1] = {summary['warm']['step_size_final']:.6g}")
print(f"prior D_tilde 0/3000/6000 = {summary['prior']['D_tilde_0']:.6g} / "
      f"{summary['prior']['D_tilde_3000']:.6g} / {summary['prior']['D_tilde_final']:.6g}")
print(f"warm  D_tilde 0/3000/6000 = {summary['warm']['D_tilde_0']:.6g} / "
      f"{summary['warm']['D_tilde_3000']:.6g} / {summary['warm']['D_tilde_final']:.6g}")
print(f"prior EEVPD obs/wanted final = {summary['prior']['eevpd_obs_final']:.6g} / "
      f"{summary['prior']['eevpd_wanted_final']:.6g}")
print(f"warm  EEVPD obs/wanted final = {summary['warm']['eevpd_obs_final']:.6g} / "
      f"{summary['warm']['eevpd_wanted_final']:.6g}")
print(f"tight dims {tight_dims}: prior std {summary['tight_dim_std_prior_final']} "
      f"vs warm target {summary['tight_dim_std_warm_target']} "
      f"(ratio {summary['tight_dim_std_ratio_prior_over_warm']})")

print("DIAG DONE", flush=True)
