#!/usr/bin/env python
"""Compare LAPS warm-vs-cold init and chain counts on the REAL gigalens lens demo.

Reuses ONE MAP+SVI fit (built once, up front) across a small grid of LAPS configs:

    [(init, M) for init in ("warm", "cold") for M in (512, 128)]

For each config we:
  * time the run_laps wall clock,
  * collect the ground-truth-FREE internal health from diagnose(),
  * convert the LAPS ensemble to physical space and compute a RECOVERY metric
    against the known demo truth (this demo HAS a truth; a real lens would not),
  * save a per-config diagnose PNG and corner PNG (no clobbering),
  * record one summary row; failures are caught per-config so one OOM does not
    kill the rest of the grid.

DO NOT run this on a login node -- it needs the GPU. The orchestrator runs it on
the GPU node. Keeps run_laps budgets at their validated defaults: we vary only
init mode and chain count M at fixed budget.
"""
import os
import sys
import json
import time
import traceback

# --------------------------------------------------------------------------- #
# EDIT HERE: grid + seed (budgets stay at run_laps validated defaults).         #
# --------------------------------------------------------------------------- #
CHAIN_COUNTS = (512, 128)
INIT_MODES = ("warm", "prior")
CONFIGS = [(init, M) for init in INIT_MODES for M in CHAIN_COUNTS]
SEED = 0

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../handoff
LAPS_VAL_DIR = os.path.dirname(HERE)                              # .../laps_validation
OUT_DIR = os.path.join(HERE, "compare_out_v5")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Matplotlib headless BEFORE any pyplot import (corner pulls in pyplot).         #
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np

# --------------------------------------------------------------------------- #
# sys.path inserts (same as the notebook) so the handoff + gigalens_research     #
# packages import. handoff.laps_handoff needs LAPS_VAL_DIR (parent of handoff)   #
# on the path; gigalens_research needs GIGALens-Code/src.                        #
# --------------------------------------------------------------------------- #
sys.path.insert(0, LAPS_VAL_DIR)
sys.path.insert(0, "/global/u1/l/linusu/GIGALens-Code/src")

import jax
jax.config.update("jax_enable_x64", True)

# NEW gigalens "scene" API
from gigalens.jax.inference import ModellingSequence
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

import tensorflow_probability.substrates.jax as tfp
from jax import random
import optax
from jax import numpy as jnp
tfd = tfp.distributions

from corner import corner

# The handoff helpers under test.
from handoff.laps_handoff import run_laps, diagnose


# =========================================================================== #
# 1. MODEL SETUP + MAP + SVI  (done ONCE; identical to jax-demo.ipynb)          #
# =========================================================================== #
def build_model_and_fit():
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

    # Ground truth for the bundled demo image (used by the RECOVERY metric only).
    truth = [[
        {'theta_E': 1.1, 'gamma': 2.0, 'e1': 0.1, 'e2': 0.1, 'center_y': 0.0, 'center_x': 0.1},
        {'gamma2': 0.03, 'gamma1': -0.01}
    ], [
        {'R_sersic': 0.8, 'n_sersic': 2.5, 'e1': 0.09534746574143645, 'e2': 0.14849487967198177,
         'center_x': 0.1, 'center_y': 0.0, 'Ie': 499.3695906504067}
    ], [
        {'R_sersic': 0.25, 'n_sersic': 1.5, 'e1': 0., 'e2': 0., 'center_x': 0.09566681002252231,
         'center_y': -0.0639623054267272, 'Ie': 149.58828877085668}
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
    print("dim (num free params):", model.num_free_params)

    # MAP then SVI -> qz (single time, reused across all configs).
    opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
    best, best_lp, best_chisq = model_seq.MAP(opt, seed=0)

    opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
    qz, loss_hist = model_seq.SVI(best, opt, n_vi=1000, num_steps=1500)

    # truth markers for the 8 lens-mass params (EPL 6 + Shear 2), notebook order.
    markers = []
    for k in truth[0][0].keys():
        markers.append(truth[0][0][k])
    for k in truth[0][1].keys():
        markers.append(truth[0][1][k])

    return model_seq, prob_model, qz, markers


# 8 lens-mass params, column order matched to `markers` + `labels` below.
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
LABELS = [r'$\theta_E$', r'$\gamma$', r'$\epsilon_2$', r'$\epsilon_1$',
          r'$y$', r'$x$', r'$\gamma_{2,ext}$', r'$\gamma_{1,ext}$']


def samples_to_mass(prob_model, res):
    """LAPS ensemble (num_chains, dim) UNCONSTRAINED -> physical 8 mass cols (N, 8)."""
    smp = np.asarray(res.samples)
    dim = smp.shape[-1]                 # (M, K, dim) or legacy (M, dim) -> dim is last
    smp = smp.reshape((-1, dim))        # flatten (chain, sample) -> (M*K, dim)
    phys = prob_model.bij.forward(list(jnp.asarray(smp).T))   # dict: key -> (N,)
    mass_samps = np.stack([np.asarray(phys[k]) for k in MASS_ORDER], axis=1)
    return mass_samps


def recovery_metric(mass_samps, markers):
    """Per-param bias in sigma units (mean_hat - truth)/std_hat over the 8 mass params.

    Returns (max_abs_bias_sigma, worst_param_label, full per-param bias array).
    The metric is a per-parameter standardized bias: it measures whether the
    posterior MEAN sits within a few posterior-widths of truth. Blind spot: it
    says nothing about the WIDTH being correct (a too-narrow posterior inflates
    sigma-bias; a too-wide one hides a biased mean) and nothing about non-mass
    nuisance params -- so read it alongside the corner + diagnose plots.
    """
    truth = np.asarray(markers, np.float64)
    mean_hat = mass_samps.mean(axis=0)
    std_hat = mass_samps.std(axis=0)
    bias_sigma = (mean_hat - truth) / np.where(std_hat > 0, std_hat, np.nan)
    abs_bias = np.abs(bias_sigma)
    worst_idx = int(np.nanargmax(abs_bias))
    return float(np.nanmax(abs_bias)), LABELS[worst_idx], bias_sigma


# =========================================================================== #
# 2/3. RUN THE GRID                                                            #
# =========================================================================== #
def main():
    model_seq, prob_model, qz, markers = build_model_and_fit()

    rows = []
    for init, M in CONFIGS:
        tag = f"{init}_{M}"
        row = {
            "init": init, "M": M, "wall_s": None, "switched": None,
            "n_samples": M, "p2_accept_final": None,
            "p2_accept_on_target_flag": None, "rhat_max": None, "ess_min": None,
            "max_bias_sigma": None, "worst_param": None, "error": None,
        }
        print(f"\n========== CONFIG init={init} M={M} ==========")
        try:
            t0 = time.perf_counter()
            res = run_laps(model_seq, qz, init_mode=init, num_chains=M, seed=SEED, p2_keep_per_chain=16)
            wall = time.perf_counter() - t0
            row["wall_s"] = round(wall, 2)

            # ground-truth-free health (per-config PNG so they don't clobber).
            diag_png = os.path.join(OUT_DIR, f"diagnose_{tag}.png")
            health = diagnose(res, param_names=MASS_ORDER + [
                f"z{j}" for j in range(8, int(np.asarray(res.samples).shape[-1]))],
                out_png=diag_png, verbose=True)
            row["switched"] = bool(health["switched"])
            row["p2_accept_final"] = (None if not np.isfinite(health["p2_accept_final"])
                                      else round(float(health["p2_accept_final"]), 4))
            row["p2_accept_on_target_flag"] = bool(health["flags"]["p2_accept_on_target"])
            row["rhat_max"] = round(float(health["rhat_max"]), 4)
            row["ess_min"] = round(float(health["ess_min"]), 1)

            # recovery vs truth in physical (constrained) space.
            mass_samps = samples_to_mass(prob_model, res)
            max_bias, worst, bias_sigma = recovery_metric(mass_samps, markers)
            row["max_bias_sigma"] = round(max_bias, 3)
            row["worst_param"] = worst

            # per-config corner with notebook labels/markers.
            fig = corner(mass_samps, show_titles=True, title_fmt='.3f',
                         labels=LABELS, truths=list(markers))
            fig.suptitle(f"Lens mass params — {init} M={M}")
            corner_png = os.path.join(OUT_DIR, f"corner_{tag}.png")
            fig.savefig(corner_png, dpi=110, bbox_inches="tight")
            plt.close(fig)

            print(f"OK  init={init} M={M}  wall={wall:.1f}s  switched={row['switched']}  "
                  f"rhat_max={row['rhat_max']}  ess_min={row['ess_min']}  "
                  f"max|bias|={row['max_bias_sigma']}sig @ {worst}")
        except Exception as e:  # OOM or anything else: record + continue
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"FAILED: init={init} M={M}: {row['error']}")
            traceback.print_exc()
        rows.append(row)

    # =================== 4. TABLE + summary.json =========================== #
    cols = ["init", "M", "wall_s", "switched", "n_samples", "p2_accept_final",
            "p2_accept_on_target_flag", "rhat_max", "ess_min",
            "max_bias_sigma", "worst_param", "error"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n================= LAPS warm/cold x M COMPARISON =================")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"seed": SEED, "configs": CONFIGS, "rows": rows}, f, indent=2)
    print(f"\nwrote {summary_path}")
    print(f"artifacts in {OUT_DIR}/ : diagnose_<cfg>.png, corner_<cfg>.png, summary.json")


if __name__ == "__main__":
    main()
