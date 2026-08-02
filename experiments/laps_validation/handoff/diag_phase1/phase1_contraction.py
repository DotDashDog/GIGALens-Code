#!/usr/bin/env python
# coding: utf-8
r"""Phase-1 contraction diagnostic for LAPS on the gigalens lens demo.

Discriminates H1 (budget/switch-too-early: tight dirs DO contract toward the
true posterior width given enough Phase-1 steps) vs H3 (structural: tight dirs
PLATEAU above the true width regardless of budget) by running two LONG
Phase-1-ONLY LAPS runs (warm vs prior init) and tracking the per-dim ensemble
std in sampling (unconstrained) space per Phase-1 step.

Model + MAP + SVI + physical mass extraction are copied VERBATIM from
laps_overlay_j26.py. Do NOT change the physics. Do NOT edit the sampler.

Modes (env var):
    SMOKE=1     -> tiny 16-chain/50-step prior run to confirm it runs + time.
    (default)   -> probe timing at 512 chains, pick STEPS for ~15-20 min/run,
                   run warm + prior Phase-1-only, save all deliverables.
    LAPS_STEPS=N -> override the auto-picked STEPS.
    PROBE_STEPS=N -> steps used for the 512-chain timing probe (default 120).
"""
import os
import json
import time

import jax
jax.config.update("jax_enable_x64", True)

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
from matplotlib import pyplot as plt

tfd = tfp.distributions

from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted_JIT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
os.makedirs(OUT, exist_ok=True)
HMC_REF = os.path.join(os.path.dirname(HERE), "hmc_ref")

print("jax devices:", jax.devices())

# --------------------------------------------------------------------------- #
# 1. Build model + MAP + SVI ONCE (VERBATIM from laps_overlay_j26.py)          #
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

opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = MAP(prob_model, opt, seed=0)

opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, loss_hist = SVI(prob_model, best, opt, n_vi=1000, num_steps=1500)

markers = []
for i in truth[0][0].keys():
    markers.append(truth[0][0][i])
for j in truth[0][1].keys():
    markers.append(truth[0][1][j])

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
    """(N, DIM) UNCONSTRAINED z -> (N, 8) physical mass params in MASS_ORDER."""
    smp = np.asarray(samples_z).reshape((-1, DIM))
    phys = prob_model.bij.forward(list(jnp.asarray(smp).T))
    return np.stack([np.asarray(phys[k]) for k in MASS_ORDER], axis=1)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def std_traj(res):
    """Per-Phase-1-step per-dim ensemble std in sampling space -> (T1, dim)."""
    var_t = np.asarray(res.p1_obs_sq) - np.asarray(res.p1_obs_mean) ** 2
    return np.sqrt(np.clip(var_t, 0, None))


SMOKE = os.environ.get("SMOKE", "0") == "1"

# --------------------------------------------------------------------------- #
# SMOKE: confirm it runs on the real model + rough timing                      #
# --------------------------------------------------------------------------- #
if SMOKE:
    t0 = time.time()
    res = LAPS_late_adjusted_JIT(
        prob_model, qz, init_mode="prior",
        num_chains=16, num_unadjusted_steps=50, num_adjusted_steps=1,
        early_stop=False, phase2_enabled=False, seed=0)
    dt = time.time() - t0
    st = std_traj(res)
    print(f"[SMOKE] ran OK. wall={dt:.1f}s  T1={res.phase1_len}  "
          f"std_t shape={st.shape}  per-step(incl compile)={dt/50:.3f}s")
    print(f"[SMOKE] switch_index={int(res.switch_index)} "
          f"switch_index_paper={int(res.switch_index_paper)}")
    raise SystemExit(0)

# --------------------------------------------------------------------------- #
# 2. Timing probe at 512 chains -> pick STEPS for ~15-20 min per run           #
# --------------------------------------------------------------------------- #
if "LAPS_STEPS" in os.environ:
    STEPS = int(os.environ["LAPS_STEPS"])
    print(f"STEPS overridden by env -> {STEPS}")
else:
    # Two probes at 512 chains subtract the (fixed) compile cost: run_p1_chunk is
    # compiled ONCE per call (all chunks share shape), so compile is independent of
    # step count. per_step = (t_big - t_small) / (P_big - P_small).
    P_small, P_big = 25, 200
    t0 = time.time()
    LAPS_late_adjusted_JIT(prob_model, qz, init_mode="prior", num_chains=512,
                           num_unadjusted_steps=P_small, num_adjusted_steps=1,
                           early_stop=False, phase2_enabled=False, seed=0)
    t_small = time.time() - t0
    t0 = time.time()
    LAPS_late_adjusted_JIT(prob_model, qz, init_mode="prior", num_chains=512,
                           num_unadjusted_steps=P_big, num_adjusted_steps=1,
                           early_stop=False, phase2_enabled=False, seed=0)
    t_big = time.time() - t0
    per_step = max((t_big - t_small) / (P_big - P_small), 1e-3)
    compile_est = max(t_small - P_small * per_step, 0.0)
    target_s = 1050.0  # ~17.5 min of stepping
    STEPS = int((target_s) / per_step)
    STEPS = max(2000, min(STEPS, 6000))
    print(f"[PROBE] 512ch t({P_small})={t_small:.1f}s t({P_big})={t_big:.1f}s  "
          f"per_step~{per_step:.3f}s compile~{compile_est:.1f}s  -> STEPS={STEPS} "
          f"(est {(compile_est+STEPS*per_step)/60:.1f} min/run)")

common = dict(num_chains=512, num_unadjusted_steps=STEPS, num_adjusted_steps=1,
              early_stop=False, phase2_enabled=False, seed=0)

# --------------------------------------------------------------------------- #
# 3. The two long Phase-1-only runs                                            #
# --------------------------------------------------------------------------- #
t0 = time.time()
res_warm = LAPS_late_adjusted_JIT(prob_model, qz, init_mode="warm", **common)
t_warm = time.time() - t0
print(f"[warm] done wall={t_warm:.1f}s ({t_warm/60:.1f} min)  T1={res_warm.phase1_len}")

t0 = time.time()
res_prior = LAPS_late_adjusted_JIT(prob_model, qz, init_mode="prior", **common)
t_prior = time.time() - t0
print(f"[prior] done wall={t_prior:.1f}s ({t_prior/60:.1f} min)  T1={res_prior.phase1_len}")

std_warm = std_traj(res_warm)     # (T1, dim)
std_prior = std_traj(res_prior)   # (T1, dim)
T1, dim = std_prior.shape

# target per-dim std = warm run's std at its FINAL step (warm matches HMC)
target = std_warm[-1]             # (dim,)
# rank dims by target tightness (smallest target std = tightest)
order_tight = np.argsort(target)
tight8 = order_tight[:8]

sw_prior_paper = int(res_prior.switch_index_paper)
sw_prior = int(res_prior.switch_index)
sw_warm_paper = int(res_warm.switch_index_paper)
sw_warm = int(res_warm.switch_index)
print(f"switch idx: prior(paper)={sw_prior_paper} prior(active)={sw_prior} "
      f"warm(paper)={sw_warm_paper} warm(active)={sw_warm}")

# --------------------------------------------------------------------------- #
# Deliverable 1: traj.npz                                                       #
# --------------------------------------------------------------------------- #
np.savez(
    os.path.join(OUT, "traj.npz"),
    std_warm=std_warm, std_prior=std_prior, target=target,
    tight8=tight8, STEPS=STEPS, T1=T1, dim=dim,
    switch_index_prior_paper=sw_prior_paper, switch_index_prior=sw_prior,
    switch_index_warm_paper=sw_warm_paper, switch_index_warm=sw_warm,
    t_warm=t_warm, t_prior=t_prior,
)
print("saved traj.npz")

# --------------------------------------------------------------------------- #
# Deliverable 2: contraction.png (8 tightest dims, log-y)                       #
# --------------------------------------------------------------------------- #
steps_ax = np.arange(T1)
final_third_start = (2 * T1) // 3
fig, ax = plt.subplots(figsize=(11, 7))
cmap = plt.get_cmap("tab10")
for i, d in enumerate(tight8):
    c = cmap(i)
    ax.plot(steps_ax, std_prior[:, d], color=c, lw=1.6,
            label=f"dim {d} (target={target[d]:.2e})")
    ax.axhline(target[d], color=c, ls="--", lw=1.1, alpha=0.9)
# vertical dotted line at prior switch_index_paper (post-hoc literal switch)
if sw_prior_paper < T1:
    ax.axvline(sw_prior_paper, color="k", ls=":", lw=1.5,
               label=f"prior switch_index_paper={sw_prior_paper}")
else:
    ax.axvline(T1 - 1, color="k", ls=":", lw=1.5,
               label=f"switch_index_paper={sw_prior_paper} (never fired)")
ax.axvspan(final_third_start, T1 - 1, color="grey", alpha=0.10,
           label="final third (slope test)")
ax.set_yscale("log")
ax.set_xlabel("Phase-1 step")
ax.set_ylabel("ensemble std (sampling space)")
ax.set_title("LAPS Phase-1 contraction: 8 tightest dims (prior=solid, "
             "warm-target=dashed)")
ax.legend(fontsize=7, ncol=2, loc="upper right")
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "contraction.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved contraction.png")

# --------------------------------------------------------------------------- #
# Deliverable 3: stragglers.png (final-step straggler fraction, bar chart)      #
# --------------------------------------------------------------------------- #
prior_final = np.asarray(res_prior.samples).reshape((-1, dim))   # (M, dim)
mean_final = prior_final.mean(axis=0)
straggler_frac = np.array([
    np.mean(np.abs(prior_final[:, d] - mean_final[d]) > 3.0 * target[d])
    for d in tight8
])
fig, ax = plt.subplots(figsize=(9, 5))
xpos = np.arange(len(tight8))
ax.bar(xpos, straggler_frac, color="C1")
ax.set_xticks(xpos)
ax.set_xticklabels([f"dim {d}\n(t={target[d]:.1e})" for d in tight8],
                   fontsize=8)
ax.set_ylabel("fraction chains |x - mean| > 3*warm-target")
ax.set_title("Prior-init final-step straggler fraction (8 tightest dims)")
for x, v in zip(xpos, straggler_frac):
    ax.text(x, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
ax.grid(True, axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stragglers.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved stragglers.png")

# --------------------------------------------------------------------------- #
# Deliverable 4: summary.json + printed table                                  #
# --------------------------------------------------------------------------- #
def logslope_final_third(y):
    """slope of log(y) over the final third of steps (negative = descending)."""
    seg = y[final_third_start:]
    xs = np.arange(len(seg))
    ly = np.log(np.clip(seg, 1e-300, None))
    if len(seg) < 2:
        return float("nan")
    return float(np.polyfit(xs, ly, 1)[0])


sw_step_for_table = sw_prior_paper if sw_prior_paper < T1 else T1 - 1
rows = []
print("\n" + "=" * 110)
print(f"{'dim':>4}{'target_std':>13}{'prior@switch':>14}{'prior@final':>13}"
      f"{'final/target':>14}{'log-slope(final3)':>19}{'descending?':>13}")
print("-" * 110)
for d in tight8:
    tgt = float(target[d])
    p_sw = float(std_prior[sw_step_for_table, d])
    p_fin = float(std_prior[-1, d])
    ratio = p_fin / tgt if tgt > 0 else float("nan")
    slope = logslope_final_third(std_prior[:, d])
    desc = "yes" if slope < 0 else "no(flat/up)"
    rows.append(dict(dim=int(d), target_std=tgt, prior_at_switch=p_sw,
                     prior_at_final=p_fin, ratio_final_over_target=ratio,
                     logslope_final_third=slope))
    print(f"{int(d):>4}{tgt:>13.3e}{p_sw:>14.3e}{p_fin:>13.3e}"
          f"{ratio:>14.2f}{slope:>19.2e}{desc:>13}")
print("=" * 110)

summary = dict(
    STEPS=STEPS, T1=int(T1), dim=int(dim),
    final_third_start=int(final_third_start),
    switch_index_prior_paper=sw_prior_paper,
    switch_index_prior_active=sw_prior,
    switch_index_warm_paper=sw_warm_paper,
    switch_index_warm_active=sw_warm,
    t_warm_s=t_warm, t_prior_s=t_prior,
    tight8_dims=[int(x) for x in tight8],
    per_dim=rows,
    straggler_frac_tight8=straggler_frac.tolist(),
)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("saved summary.json")

# --------------------------------------------------------------------------- #
# Deliverable 5: PHYSICAL ANCHOR — 8 mass params, prior/warm/HMC std           #
# --------------------------------------------------------------------------- #
with open(os.path.join(HMC_REF, "hmc_summary.json")) as f:
    hmc_summary = json.load(f)
hmc_std = np.array(hmc_summary["hmc"]["std"])         # (8,) MASS_ORDER
hmc_mean = np.array(hmc_summary["hmc"]["mean"])

prior_mass = to_mass(res_prior.samples)               # (M, 8)
warm_mass = to_mass(res_warm.samples)                 # (M, 8)
ps = prior_mass.std(axis=0)
ws = warm_mass.std(axis=0)

mass_order_hmc = np.argsort(hmc_std)                  # rank by HMC tightness
print("\n" + "=" * 104)
print("PHYSICAL ANCHOR (mass params, ranked by HMC tightness)")
print(f"{'param':<24}{'HMC_std':>12}{'warm_std':>12}{'prior_std':>12}"
      f"{'warm/HMC':>11}{'prior/HMC':>11}")
print("-" * 104)
phys_rows = []
for k in mass_order_hmc:
    wr = ws[k] / hmc_std[k] if hmc_std[k] > 0 else float("nan")
    pr = ps[k] / hmc_std[k] if hmc_std[k] > 0 else float("nan")
    print(f"{MASS_ORDER[k]:<24}{hmc_std[k]:>12.3e}{ws[k]:>12.3e}"
          f"{ps[k]:>12.3e}{wr:>11.2f}{pr:>11.2f}")
    phys_rows.append(dict(param=MASS_ORDER[k], label=labels[k],
                          hmc_std=float(hmc_std[k]), warm_std=float(ws[k]),
                          prior_std=float(ps[k]),
                          warm_over_hmc=float(wr), prior_over_hmc=float(pr)))
print("=" * 104)

summary["physical_anchor"] = phys_rows
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nDONE. artifacts in:", OUT)
