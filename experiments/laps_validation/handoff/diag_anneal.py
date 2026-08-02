#!/usr/bin/env python
# coding: utf-8
r"""Likelihood-annealing warm-up bridge for PRIOR-seeded LAPS on the gigalens lens demo.

Diagnosis being addressed: prior-init full-LAPS over-disperses 20-370x vs HMC because
prior draws start catastrophically far (ensemble median logp ~-1.57e5 vs warm ~+150) and
Phase-1 locks into a broad basin. This script bridges broad-prior -> tight-posterior via a
likelihood-tempered beta ladder of short UNADJUSTED Phase-1 runs, then hands the warmed
ensemble to FULL LAPS (adjusted phase on the true posterior).

Tempered target:  logp_beta(z) = log_prior(z) + beta * log_like(z)[0]
  beta=0 -> prior (prior-init draws already in equilibrium); beta=1 -> true posterior.

PRODUCES NUMBERS, no verdict. Model build verbatim from diag_p2accept.py.
"""
import os, json, time
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
from gigalens_research.inference.laps_late_adjusted import (
    LAPS_late_adjusted, LAPS_late_adjusted_JIT)
print("jax devices:", jax.devices(), flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_anneal"); os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# MODEL BUILD -- verbatim from diag_p2accept.py (scene API + MAP + SVI)
# ---------------------------------------------------------------------------
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
DIM = int(model.num_free_params); print("dim:", DIM, flush=True)
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = MAP(prob_model, opt, seed=0)
print("MAP best_chisq (min):", float(np.min(np.asarray(best_chisq))), flush=True)
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, _ = SVI(prob_model, best, opt, n_vi=1000, num_steps=1500); print("SVI done.", flush=True)

# ---------------------------------------------------------------------------
# Batched true-posterior log_prob helper (returns (M,))
# ---------------------------------------------------------------------------
def true_logp(z):
    return np.asarray(prob_model.log_prob(jnp.asarray(z))[0])

def tempered_logp(z, beta):
    return np.asarray(prob_model.log_prior(jnp.asarray(z))
                      + beta * prob_model.log_like(jnp.asarray(z))[0])

# ---------------------------------------------------------------------------
# Prior-init positions (unconstrained) -- exactly as init_mode='prior'
# ---------------------------------------------------------------------------
M = 512
STEPS_PER_BETA = 200
betas = np.geomspace(0.003, 1.0, 10)
print(f"M={M}  STEPS_PER_BETA={STEPS_PER_BETA}  betas={np.array2string(betas, precision=4)}", flush=True)

k_prior = jax.random.fold_in(jax.random.key(0), 0x9E3779B9)
start = prob_model.prior.sample(M, seed=k_prior)          # constrained draws
pos = jnp.stack(prob_model.bij.inverse(start)).T          # (M, DIM) unconstrained
pos = pos.astype(jnp.float64)

# Pre-registered baseline: prior-init median TRUE logp (~-1.57e5 expected)
prior_logp = true_logp(pos)
print(f"[prior-init] median true logp = {np.median(prior_logp):.4e}  "
      f"(min {np.min(prior_logp):.3e}, max {np.max(prior_logp):.3e})", flush=True)
prior_std = np.std(np.asarray(pos), axis=0)

# ---------------------------------------------------------------------------
# ANNEALING LADDER: chained short unadjusted Phase-1 runs over beta
# ---------------------------------------------------------------------------
track = []   # per-beta: (beta, median_true_logp, median_tempered_logp, per-dim std vector)
t0 = time.time()
for i, beta in enumerate(betas):
    b = float(beta)
    logp_beta = (lambda bb: (lambda z: prob_model.log_prior(z) + bb * prob_model.log_like(z)[0]))(b)
    tb = time.time()
    res = LAPS_late_adjusted(logp_beta, qz, init_positions=pos, num_chains=M,
                             num_unadjusted_steps=STEPS_PER_BETA, num_adjusted_steps=1,
                             early_stop=False, phase2_enabled=False, seed=0)
    pos = jnp.asarray(res.samples).reshape((M, -1))       # ensemble at this beta
    mt = float(np.median(true_logp(pos)))
    mtemp = float(np.median(tempered_logp(pos, b)))
    sd = np.std(np.asarray(pos), axis=0)
    track.append((b, mt, mtemp, sd))
    print(f"[beta {i+1:2d}/{len(betas)}  beta={b:.5f}] median true_logp={mt:.4e}  "
          f"median tempered_logp={mtemp:.4e}  elapsed={time.time()-tb:.1f}s", flush=True)

annealed = np.asarray(pos)
anneal_time = time.time() - t0
annealed_true_logp = true_logp(annealed)
annealed_std = np.std(annealed, axis=0)
print(f"\n[ANNEALED (beta=1)] median true logp = {np.median(annealed_true_logp):.4e}  "
      f"(min {np.min(annealed_true_logp):.3e}, max {np.max(annealed_true_logp):.3e})", flush=True)
print(f"[ANNEALED] per-dim std: median={np.median(annealed_std):.4e}  "
      f"max={np.max(annealed_std):.4e}  ladder_time={anneal_time:.1f}s", flush=True)

# ---------------------------------------------------------------------------
# REFERENCE: warm-init full LAPS (default budgets) -> converged-truth per-dim std
# ---------------------------------------------------------------------------
print("\n=== warm-init full LAPS (default budgets) [reference] ===", flush=True)
tw = time.time()
warm = LAPS_late_adjusted_JIT(prob_model, qz, init_mode="warm", num_chains=M, seed=0)
warm_smp = np.asarray(warm.samples).reshape((-1, DIM))
warm_std = np.std(warm_smp, axis=0)
warm_logp = true_logp(warm_smp)
print(f"[warm full LAPS] median true logp = {np.median(warm_logp):.4e}  "
      f"per-dim std median={np.median(warm_std):.4e} max={np.max(warm_std):.4e}  "
      f"time={time.time()-tw:.1f}s", flush=True)

# ---------------------------------------------------------------------------
# FINAL: full LAPS (adjusted) seeded from the ANNEALED ensemble
# ---------------------------------------------------------------------------
print("\n=== final full LAPS from ANNEALED ensemble (default budgets) ===", flush=True)
tf = time.time()
log_prob = lambda z: prob_model.log_prob(z)[0]
final = LAPS_late_adjusted(log_prob, qz, init_positions=jnp.asarray(annealed),
                           num_chains=M, seed=0)
final_smp = np.asarray(final.samples).reshape((-1, DIM))
final_std = np.std(final_smp, axis=0)
final_logp = true_logp(final_smp)
print(f"[final full LAPS] median true logp = {np.median(final_logp):.4e}  time={time.time()-tf:.1f}s", flush=True)

ratio = final_std / warm_std          # width ratio to converged truth, per dim
print(f"\n[FINAL/WARM per-dim std ratio] MAX={np.max(ratio):.3f}  MEDIAN={np.median(ratio):.3f}  "
      f"(no-anneal prior baseline was ~236x max / 22x median; warm=1.0)", flush=True)

# ---------------------------------------------------------------------------
# TABLE + SAVE
# ---------------------------------------------------------------------------
print("\n===== BETA LADDER TRACKING TABLE =====")
print(f"{'beta':>10} {'median_true_logp':>18} {'median_temp_logp':>18} {'std_median':>12} {'std_max':>12}")
for (b, mt, mtemp, sd) in track:
    print(f"{b:>10.5f} {mt:>18.4e} {mtemp:>18.4e} {np.median(sd):>12.4e} {np.max(sd):>12.4e}")

print("\n===== FINAL PER-DIM STD RATIO (final / warm) =====")
print(f"{'dim':>4} {'final_std':>12} {'warm_std':>12} {'ratio':>10}")
for d in range(DIM):
    print(f"{d:>4} {final_std[d]:>12.4e} {warm_std[d]:>12.4e} {ratio[d]:>10.3f}")

np.savez(os.path.join(OUT, "anneal.npz"),
         betas=betas, M=M, steps_per_beta=STEPS_PER_BETA,
         prior_median_logp=np.median(prior_logp), prior_std=prior_std,
         track_beta=np.array([t[0] for t in track]),
         track_median_true_logp=np.array([t[1] for t in track]),
         track_median_temp_logp=np.array([t[2] for t in track]),
         track_std=np.array([t[3] for t in track]),
         annealed=annealed, annealed_median_true_logp=np.median(annealed_true_logp),
         annealed_std=annealed_std,
         warm_samples_z=warm_smp, warm_std=warm_std, warm_median_logp=np.median(warm_logp),
         final_samples_z=final_smp, final_std=final_std, final_median_logp=np.median(final_logp),
         ratio=ratio, ratio_max=np.max(ratio), ratio_median=np.median(ratio))

summary = dict(
    dim=DIM, M=M, steps_per_beta=STEPS_PER_BETA, betas=[float(b) for b in betas],
    map_best_chisq=float(np.min(np.asarray(best_chisq))),
    prior_median_true_logp=float(np.median(prior_logp)),
    annealed_median_true_logp=float(np.median(annealed_true_logp)),
    annealed_std_median=float(np.median(annealed_std)), annealed_std_max=float(np.max(annealed_std)),
    warm_median_logp=float(np.median(warm_logp)),
    warm_std_median=float(np.median(warm_std)), warm_std_max=float(np.max(warm_std)),
    final_median_logp=float(np.median(final_logp)),
    ratio_max=float(np.max(ratio)), ratio_median=float(np.median(ratio)),
    track=[dict(beta=float(t[0]), median_true_logp=float(t[1]),
                median_temp_logp=float(t[2]), std_median=float(np.median(t[3])),
                std_max=float(np.max(t[3]))) for t in track])
json.dump(summary, open(os.path.join(OUT, "anneal_summary.json"), "w"), indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=2))
print("DIAG DONE", flush=True)
