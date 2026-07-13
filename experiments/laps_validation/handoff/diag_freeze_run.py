#!/usr/bin/env python
# coding: utf-8
r"""ONE diagnostic: is prior-init LAPS over-dispersion caused by chains entering
BAD regions (non-finite log-posterior/gradient -> frozen by the NaN-guard) vs
healthy-but-slow dynamics?

PRODUCES NUMBERS + a plot. No verdict. Does NOT edit the sampler / model / env.

Model-build block (scene-API model + MAP + SVI -> model_seq, prob_model, qz) is
copied VERBATIM from laps_overlay_j26.py. The LAPS calls from that file are NOT
run; instead we run the pre-registered freeze diagnostic.
"""
import os
import json
import traceback

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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_freeze")
os.makedirs(OUT, exist_ok=True)


def main():
    print("jax.devices():", jax.devices())

    # ----------------------------------------------------------------- #
    # Model + MAP + SVI  (VERBATIM from laps_overlay_j26.py)             #
    # ----------------------------------------------------------------- #
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
    map_lp = float(np.asarray(best_lp).reshape(-1)[0])
    map_chisq = float(np.asarray(best_chisq).reshape(-1)[0])
    print("MAP best_lp:", map_lp, "  MAP reduced chi^2:", map_chisq)

    opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
    qz, loss_hist = model_seq.SVI(best, opt, n_vi=1000, num_steps=1500)

    # ----------------------------------------------------------------- #
    # Diagnostic machinery                                              #
    # ----------------------------------------------------------------- #
    NUM_CHAINS = 256
    SEED = 0

    def log_prob(z):
        return prob_model.log_prob(z)[0]

    # per-chain grad: vmap grad of scalar log_prob over the ensemble
    g = jax.jit(jax.vmap(jax.grad(lambda z: log_prob(z[None])[0])))
    lp_batch = jax.jit(log_prob)

    def stats(x, tag):
        """x: (N, dim) unconstrained ensemble. Returns dict of the pre-registered
        quantities: frac non-finite logp / grad, frac |x|>5, and 5/50/95
        percentiles of logp and of |x| (L2 norm in unconstrained space)."""
        x = jnp.asarray(x)
        logp = np.asarray(lp_batch(x))               # (N,)
        grad = np.asarray(g(x))                       # (N, dim)
        xn = np.asarray(jnp.linalg.norm(x, axis=1))   # (N,) L2 in unconstrained space

        finite_logp = np.isfinite(logp)
        finite_grad = np.all(np.isfinite(grad), axis=1)
        frac_nf_logp = float(np.mean(~finite_logp))
        frac_nf_grad = float(np.mean(~finite_grad))
        frac_xgt5 = float(np.mean(xn > 5.0))

        # percentiles computed over FINITE entries only (non-finite counted above)
        lp_fin = logp[finite_logp]
        lp_pct = (np.percentile(lp_fin, [5, 50, 95]).tolist()
                  if lp_fin.size else [float("nan")] * 3)
        xn_pct = np.percentile(xn[np.isfinite(xn)], [5, 50, 95]).tolist() \
            if np.any(np.isfinite(xn)) else [float("nan")] * 3
        d = dict(tag=tag, n=int(x.shape[0]),
                 frac_nonfinite_logp=frac_nf_logp,
                 frac_nonfinite_grad=frac_nf_grad,
                 frac_absx_gt5=frac_xgt5,
                 logp_p5=lp_pct[0], logp_p50=lp_pct[1], logp_p95=lp_pct[2],
                 absx_p5=xn_pct[0], absx_p50=xn_pct[1], absx_p95=xn_pct[2])
        print(f"[{tag:>18}] nf_logp={frac_nf_logp:.4f} nf_grad={frac_nf_grad:.4f} "
              f"|x|>5={frac_xgt5:.4f}  logp[p5/p50/p95]="
              f"{lp_pct[0]:.3g}/{lp_pct[1]:.3g}/{lp_pct[2]:.3g}  "
              f"|x|[p5/p50/p95]={xn_pct[0]:.3g}/{xn_pct[1]:.3g}/{xn_pct[2]:.3g}")
        return d

    results = {}

    # ---- (1) INIT ensembles -------------------------------------------------
    # prior: EXACT idiom of the LAPS wrapper's init_mode='prior' (same folded key)
    k_prior = jax.random.fold_in(jax.random.key(SEED), 0x9E3779B9)
    start = prob_model.prior.sample(NUM_CHAINS, seed=k_prior)      # constrained
    x_prior = jnp.stack(prob_model.bij.inverse(start)).T           # (256, dim) unconstr
    x_prior = x_prior.astype(jnp.float64)
    results["init_prior"] = stats(x_prior, "init_prior")

    # warm: qz surrogate
    k_warm = jax.random.fold_in(jax.random.key(SEED), 12345)
    x_warm = qz.sample((NUM_CHAINS,), seed=k_warm)                 # (256, dim)
    x_warm = jnp.asarray(x_warm).astype(jnp.float64)
    results["init_warm"] = stats(x_warm, "init_warm")

    # ---- (2) LADDER over Phase-1 (prior init) -------------------------------
    from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted_JIT
    for steps in (300, 1500):
        print(f"\n--- Phase-1 ladder: num_unadjusted_steps={steps} (prior init) ---")
        res = LAPS_late_adjusted_JIT(
            model_seq, qz, init_mode="prior", num_chains=NUM_CHAINS,
            num_unadjusted_steps=steps, num_adjusted_steps=1,
            early_stop=False, phase2_enabled=False, seed=SEED)
        ens = np.asarray(res.samples).reshape((NUM_CHAINS, DIM))
        results[f"prior_step{steps}"] = stats(ens, f"prior_step{steps}")

    # ---- (3) Save summary + plot -------------------------------------------
    with open(os.path.join(OUT, "freeze_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    xgrid = [0, 300, 1500]
    prior_keys = ["init_prior", "prior_step300", "prior_step1500"]
    nf_logp = [results[k]["frac_nonfinite_logp"] for k in prior_keys]
    nf_grad = [results[k]["frac_nonfinite_grad"] for k in prior_keys]
    xgt5 = [results[k]["frac_absx_gt5"] for k in prior_keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xgrid, nf_logp, "o-", color="C3", label="frac non-finite logp (prior)")
    ax.plot(xgrid, nf_grad, "s-", color="C0", label="frac non-finite grad (prior)")
    ax.plot(xgrid, xgt5, "^-", color="C2", label="frac |x|>5 (prior)")
    # warm-init reference points at x=0
    w = results["init_warm"]
    ax.plot(0, w["frac_nonfinite_logp"], "x", color="C3", ms=11, mew=2.5,
            label="warm ref: nf logp")
    ax.plot(0, w["frac_nonfinite_grad"], "x", color="C0", ms=11, mew=2.5,
            label="warm ref: nf grad")
    ax.plot(0, w["frac_absx_gt5"], "x", color="C2", ms=11, mew=2.5,
            label="warm ref: |x|>5")
    ax.set_xlabel("Phase-1 unadjusted steps")
    ax.set_ylabel("fraction of chains")
    ax.set_title("Freeze diagnostic: bad-region fraction vs Phase-1 step (prior init)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "freeze.png"), dpi=120)
    plt.close(fig)

    print("\nMAP reduced chi^2 =", map_chisq)
    print("saved:", os.path.join(OUT, "freeze_summary.json"))
    print("saved:", os.path.join(OUT, "freeze.png"))
    print("PRIMARY nonfinite-grad frac  init_prior =",
          results["init_prior"]["frac_nonfinite_grad"],
          " prior_step1500 =", results["prior_step1500"]["frac_nonfinite_grad"])
    print("DIAG DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
