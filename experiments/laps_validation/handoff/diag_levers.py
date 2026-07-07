#!/usr/bin/env python
# coding: utf-8
r"""DC-7.1 lever test: do the blackjax-faithful cold-start mechanisms (stocktake
F1/F2/F3) rescue prior-seeded LAPS on the demo lens?

Arms (512 chains, 300 unadj + 200 adj, early_stop=False so every arm gets the
FULL Phase-1 budget -- removes the false-firing-switch confound; switch indices
still recorded post-hoc):
  A0   prior, levers off, seed 0      (same-code control / edit-regression check)
  A1   prior + velocity_init="gradient", seed 0          (F1: THE hypothesis test)
  A1b  prior + velocity_init="gradient", seed 1          (single-seed guard)
  A1c  prior + F1 + L0_inf (bj first-step no-refresh companion), seed 0
  A2   prior + F1 + L0_inf + nan_eps_halving + precond_source="final", seed 0

Outputs (diag_levers/): per-arm mass samples, summary.json, corner overlays
(shared-range + HMC-zoomed), per-dim ensemble-std trajectories (scale view).
PRODUCES NUMBERS AND PLOTS, no verdict. Model build verbatim from
laps_overlay_j26.py / diag_p2accept.py.
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
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from corner import corner
tfd = tfp.distributions
from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted_JIT
print("jax devices:", jax.devices())

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_levers"); os.makedirs(OUT, exist_ok=True)

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

# ---- HMC reference + truth markers (from the already-computed artifacts) ----
MASS_ORDER = [
    "planes/0/mass/0/theta_E", "planes/0/mass/0/gamma",
    "planes/0/mass/0/e1", "planes/0/mass/0/e2",
    "planes/0/mass/0/center_y", "planes/0/mass/0/center_x",
    "planes/0/mass/1/gamma2", "planes/0/mass/1/gamma1",
]
labels = [r'$\theta_E$', r'$\gamma$', r'$\epsilon_2$', r'$\epsilon_1$',
          r'$y$', r'$x$', r'$\gamma_{2,ext}$', r'$\gamma_{1,ext}$']
hmc_mass = np.load(os.path.join(HERE, "hmc_ref", "hmc_mass.npy"))
markers = json.load(open(os.path.join(HERE, "hmc_ref", "overlay_summary.json")))["truth"]
hs = hmc_mass.std(0)
print(f"HMC ref: {hmc_mass.shape}, std={hs}")


def to_mass(samples_z):
    smp = np.asarray(samples_z).reshape((-1, DIM))
    phys = prob_model.bij.forward(list(jnp.asarray(smp).T))
    return np.stack([np.asarray(phys[k]) for k in MASS_ORDER], axis=1)


logp_one = jax.jit(jax.vmap(lambda z: prob_model.log_prob(z)[0]))

ARMS = {
    "A0_baseline":  dict(seed=0),
    "A1_f1":        dict(seed=0, velocity_init="gradient"),
    "A1b_f1_seed1": dict(seed=1, velocity_init="gradient"),
    "A1c_f1_Linf":  dict(seed=0, velocity_init="gradient", L0_inf=True),
    "A2_all":       dict(seed=0, velocity_init="gradient", L0_inf=True,
                         nan_eps_halving=True, precond_source="final"),
}

results, summary = {}, {}
for name, kw in ARMS.items():
    print(f"\n=== {name}: prior-init 512 chains, 300/200, early_stop=False, {kw} ===",
          flush=True)
    res = LAPS_late_adjusted_JIT(model_seq, qz, init_mode="prior", num_chains=512,
                                 early_stop=False, **kw)
    smp = np.asarray(res.samples).reshape((-1, DIM))
    mass = to_mass(smp)
    np.save(os.path.join(OUT, f"{name}_mass.npy"), mass)
    np.save(os.path.join(OUT, f"{name}_samples_z.npy"), smp)
    ratio = mass.std(0) / hs
    lp = np.asarray(logp_one(jnp.asarray(smp)))
    # per-dim ensemble std trajectory in z (scale-contraction view)
    var_t = np.asarray(res.p1_obs_sq) - np.asarray(res.p1_obs_mean) ** 2
    std_t = np.sqrt(np.clip(var_t, 0, None))
    np.save(os.path.join(OUT, f"{name}_p1_std_traj.npy"), std_t)
    nanf = np.asarray(res.p1_nan_frac)
    np.save(os.path.join(OUT, f"{name}_p1_delta_max.npy"),
            np.asarray(res.p1_delta_max))
    results[name] = dict(mass=mass, std_t=std_t)
    summary[name] = dict(
        kwargs={k: str(v) for k, v in kw.items()},
        ratio=ratio.tolist(), ratio_median=float(np.median(ratio)),
        ratio_max=float(np.max(ratio)),
        logp_median=float(np.median(lp)), logp_max=float(np.max(lp)),
        nan_frac_max=float(nanf.max()), nan_frac_mean=float(nanf.mean()),
        switch_index_paper=int(res.switch_index_paper),
        switched=bool(res.switched), switch_index=int(res.switch_index),
        phase1_len=int(res.phase1_len),
        p2_final_eps=float(res.p2_final_step_size),
        p2_accept_last=float(np.asarray(res.p2_accept)[-1]),
        precond_var=np.asarray(res.precond_var).tolist(),
    )
    s = summary[name]
    print(f"[{name}] ratio median={s['ratio_median']:.2f} max={s['ratio_max']:.2f}  "
          f"logp median={s['logp_median']:.4g} max={s['logp_max']:.4g}  "
          f"nan_frac max={s['nan_frac_max']:.4f}  "
          f"switch_paper@{s['switch_index_paper']}  p2_eps={s['p2_final_eps']:.3e}  "
          f"p2_acc_last={s['p2_accept_last']:.3f}", flush=True)

json.dump(summary, open(os.path.join(OUT, "levers_summary.json"), "w"), indent=2)

# ---- corner overlays: shared-range AND HMC-zoomed ----
plot_arms = ["A0_baseline", "A1_f1", "A1c_f1_Linf", "A2_all"]
colors = ["C1", "C3", "C2", "C4"]
arm_labels = ["A0 baseline prior", "A1 F1 grad-init", "A1c F1+L0_inf",
              "A2 all levers"]
for tag, rng in (
    ("full", None),
    ("zoom", [(m - 6 * s, m + 6 * s) for m, s in zip(hmc_mass.mean(0), hs)]),
):
    if rng is None:
        allcat = np.concatenate([hmc_mass] + [results[a]["mass"] for a in plot_arms])
        lo, hi = allcat.min(0), allcat.max(0)
        pad = np.where((hi - lo) > 0, 0.05 * (hi - lo), 1e-6)
        rng = list(zip(lo - pad, hi + pad))
    ckw = dict(labels=labels, range=rng, bins=30, plot_datapoints=False,
               smooth=1.0, hist_kwargs=dict(density=True))
    fig = corner(hmc_mass, color="black", truths=markers, truth_color="green", **ckw)
    for a, col in zip(plot_arms, colors):
        m = results[a]["mass"]
        clip = np.all([(m[:, i] >= rng[i][0]) & (m[:, i] <= rng[i][1])
                       for i in range(len(rng))], axis=0)
        if clip.sum() >= 10:
            corner(m[clip], color=col, fig=fig, **ckw)
        else:
            print(f"(zoom) {a}: only {int(clip.sum())} samples inside HMC±6σ — "
                  f"omitted from zoom corner", flush=True)
    handles = ([Line2D([0], [0], color="black", lw=2, label="HMC reference")] +
               [Line2D([0], [0], color=c, lw=2, label=l)
                for c, l in zip(colors, arm_labels)] +
               [Line2D([0], [0], color="green", lw=2, label="truth")])
    fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=False)
    fig.suptitle(f"DC-7.1 lever test ({tag}): HMC vs prior-LAPS arms", fontsize=14)
    fig.savefig(os.path.join(OUT, f"levers_corner_{tag}.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved levers_corner_{tag}.png", flush=True)

# ---- per-dim Phase-1 std trajectories: A0 vs A1 (the scale-lock view) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, a in zip(axes, ["A0_baseline", "A1_f1"]):
    st = results[a]["std_t"]
    for i in range(st.shape[1]):
        ax.semilogy(st[:, i], lw=0.8)
    ax.set_title(f"{a}: per-dim ensemble std (z) vs Phase-1 step")
    ax.set_xlabel("Phase-1 step")
axes[0].set_ylabel("ensemble std (unconstrained)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "levers_p1_std.png"), dpi=110)
plt.close(fig)
print("saved levers_p1_std.png", flush=True)

print("\nSUMMARY:", json.dumps({k: {kk: vv for kk, vv in v.items()
                                    if kk != "precond_var"}
                                for k, v in summary.items()}, indent=2))
print("DIAG DONE", flush=True)
