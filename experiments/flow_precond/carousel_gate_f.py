#!/usr/bin/env python
"""Carousel GATE F (plan §5.2): train one-shot flows on carousel-dPIE, gate them.

Trains Phase-A-only and Phase-A+B one-shot flows (fixed range 6 + trainable
DiagScale — the demo-validated v4 architecture, nothing data-derived) on the
33-param carousel posterior, then evaluates the pre-registered gates:

  GATE F(a)  ELBO: neg-ELBO tail (mean last 100) <= SVI final loss 291453.1
             (family nesting; same sign convention as SVIStage, verified).
  GATE F(b)  pocket coverage: log q_flow(zP) - log q_flow(zM) >= -8 nats
             (zP/zM = basin medians from basin_slice.npz; true lp ratio ~ -3).
             Pre-registered: A-only FAILS (mode-seeking drops the ~5% pocket
             14 sigma from the SVI solution), A+B PASSES.
  GATE F(c)  pullback-scale on MAIN-BASIN MAMS64 draws (pocket excluded via
             z[:,6] > -22.35): per-dim sd in [0.5, 2], |mean| <= 1, for A+B
             (gate) and A-only (recorded). Pocket-draw pullback stats reported
             separately as a diagnostic (an A-only flow that drops the pocket
             SHOULD place those draws far out -- that is F(b) seen from the
             sample side, not a scale failure).
  Diagnostics: max|T^-1(z)| over all draws (box check), per-phase loss
             histories, learned exp(s) range.

Phase-B data: the fresh 64-chain MAMS run (messy_tests/dpie/mams/arrays.npz,
(64,1000,33)) -- DEVIATION from the plan's named mclmc file (8x10k). Measured
occupancies (checkpoint entry has full provenance): MAMS64 9.57% (per-chain
range [0.001, 0.951] -- pocket-segregated chains, estimate uncertain ~2x),
MCLMC 14.57%; Laplace proxy 5.4%. MAMS64 is MH-exact in law and the least
over-weighted MH-correct set available; forward-KL trains toward its weights.

GPU only. Outputs to carousel_gate_f_out/ (flows cached with config keys).
"""
import os
import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "carousel_gate_f_out")
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

sys.path.insert(0, HERE)
import carousel_model
from gigalens_research.inference import flows
import demo_validation as dv  # reuse the demo-validated train_flow loop
dv.OUT_DIR = OUT_DIR          # redirect its flow caches here

SEED = 0
SPLINE_CFG = dict(num_layers=6, num_bins=8, spline_range=6.0, hidden_dims=(64, 64),
                  trainable_scale=True)
PHASE_A_STEPS = 3000
PHASE_A_DRAWS = 128
PHASE_A_LR = 3e-3
PHASE_B_STEPS = 1000
PHASE_B_LR = 1e-4

DPIE = os.path.join(HERE, "..", "sim_carousel", "messy_tests", "dpie")
BASIN = "/pscratch/sd/l/linusu/carousel_diag/basin_slice/basin_slice.npz"
SVI_FINAL_LOSS = 291453.1085118712  # svi manifest final_loss (verified 2026-07-08)
POCKET_COL, POCKET_THR = 6, -22.35  # z[:,6] > -22.35 => pocket (verified col map)


def main():
    model_seq, prob_model = carousel_model.build()
    dim = 33
    names = prob_model.z_param_names
    assert names[POCKET_COL] == "planes/0/mass/1/center_x", names[POCKET_COL]
    lp_fn = lambda z: prob_model.log_prob(z)[0]

    svi = np.load(os.path.join(DPIE, "svi", "arrays.npz"))
    qz_loc, qz_scale_tril = svi["qz_loc"], svi["qz_scale_tril"]

    mams = np.load(os.path.join(DPIE, "mams", "arrays.npz"))
    z_mams = mams["samples_z"].reshape(-1, dim)
    pocket_mask = z_mams[:, POCKET_COL] > POCKET_THR
    print(f"MAMS64 draws: {z_mams.shape[0]}, pocket occupancy "
          f"{pocket_mask.mean():.3%}")

    basin = np.load(BASIN)
    zP, zM = jnp.asarray(basin["zP"]), jnp.asarray(basin["zM"])

    model_card = {
        "script": os.path.abspath(__file__),
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64), "dim": dim,
        "spline": {**SPLINE_CFG, "phase_a": [PHASE_A_STEPS, PHASE_A_DRAWS, PHASE_A_LR],
                   "phase_b": [PHASE_B_STEPS, PHASE_B_LR, "mams64"]},
        "phase_b_data": dict(source="messy_tests/dpie/mams/arrays.npz",
                             n=int(z_mams.shape[0]),
                             pocket_occupancy=float(pocket_mask.mean())),
        "seed": SEED,
    }
    print("MODEL CARD:", json.dumps(model_card, indent=2))

    key = jax.random.key(SEED + 10)
    sp_init, sp_make = flows.make_whitened_spline_flow(
        key, dim, jnp.asarray(qz_loc), jnp.asarray(qz_scale_tril), **SPLINE_CFG)
    sp_key = (f"car_r{int(SPLINE_CFG['spline_range'])}b{SPLINE_CFG['num_bins']}"
              f"ts1lr{PHASE_A_LR:g}")

    t0 = time.perf_counter()
    params_a, hist_a = dv.train_flow(
        f"A_{sp_key}", sp_init, sp_make, lp_fn, dim, n_draws=PHASE_A_DRAWS,
        num_steps=PHASE_A_STEPS, lr=PHASE_A_LR, seed=SEED + 11)
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    params_ab, hist_ab = dv.train_flow(
        f"AB_{sp_key}", params_a, sp_make, lp_fn, dim, n_draws=PHASE_A_DRAWS,
        num_steps=0, lr=PHASE_A_LR, seed=SEED + 12,
        phase_b_samples=z_mams, phase_b_steps=PHASE_B_STEPS, phase_b_lr=PHASE_B_LR)
    t_b = time.perf_counter() - t0

    summary = {"model_card": model_card, "timings_s": dict(phase_a=t_a, phase_b=t_b),
               "training": {
                   "A": dict(steps=int(hist_a["a"].size),
                             diverged=bool(hist_a.get("diverged"))),
                   "B": dict(steps=int(hist_ab["b"].size)
                             if hist_ab["b"] is not None else 0,
                             diverged=bool(hist_ab.get("diverged"))),
               }}

    def log_q(make, params, z):
        bij = make(params)
        u = bij.inverse(z)
        ildj = bij.inverse_log_det_jacobian(z, event_ndims=1)
        return np.asarray(flows._std_normal_logprob(u) + ildj), np.asarray(u)

    batched_lp = jax.vmap(lp_fn)

    def direct_neg_elbo(params, n_batches=5):
        """Fresh Monte-Carlo neg-ELBO of the FINAL flow (Phase B shifts it off the
        Phase-A history, so each flow gets its own estimate; 5x128 draws)."""
        vals = [float(flows.neg_elbo_loss(
            params, sp_make, batched_lp, jax.random.key(1000 + i),
            PHASE_A_DRAWS, dim)) for i in range(n_batches)]
        return float(np.mean(vals)), float(np.std(vals))

    results = {}
    for tag, params, hist in (("A_only", params_a, hist_a),
                              ("A_plus_B", params_ab, hist_ab)):
        if hist.get("diverged"):
            results[tag] = dict(status="TRAINING DIVERGED")
            continue
        # GATE F(a): direct neg-ELBO of this flow (mean +- sd over 5 batches)
        tail, tail_sd = direct_neg_elbo(params)
        # GATE F(b): pocket coverage
        lq_pm, _ = log_q(sp_make, params, jnp.stack([zP, zM]))
        pocket_ratio = float(lq_pm[0] - lq_pm[1])
        # GATE F(c): pullback of main-basin draws (subsample for memory)
        rng = np.random.default_rng(SEED)
        main_idx = rng.choice(np.where(~pocket_mask)[0], size=4000, replace=False)
        pocket_idx = np.where(pocket_mask)[0]
        pocket_idx = rng.choice(pocket_idx, size=min(2000, pocket_idx.size),
                                replace=False)
        _, u_main = log_q(sp_make, params, jnp.asarray(z_mams[main_idx]))
        _, u_pocket = log_q(sp_make, params, jnp.asarray(z_mams[pocket_idx]))
        sd, mu = u_main.std(0), np.abs(u_main.mean(0))
        bij = sp_make(params)
        results[tag] = dict(
            status="OK",
            elbo_direct=tail, elbo_direct_sd=tail_sd,
            elbo_gate_pass=bool(tail <= SVI_FINAL_LOSS),
            pocket_log_ratio=pocket_ratio,
            pocket_gate_pass=bool(pocket_ratio >= -8.0),
            pullback_main_sd=[float(sd.min()), float(sd.max())],
            pullback_main_maxabsmean=float(mu.max()),
            pullback_scale_pass=bool((sd >= 0.5).all() and (sd <= 2.0).all()
                                     and (mu <= 1.0).all()),
            pullback_pocket_sd=[float(u_pocket.std(0).min()),
                                float(u_pocket.std(0).max())],
            pullback_pocket_maxabs=float(np.abs(u_pocket).max()),
            max_abs_u_main=float(np.abs(u_main).max()),
            exp_s_range=[float(np.exp(np.asarray(
                params["log_scale"]).min())), float(np.exp(np.asarray(
                    params["log_scale"]).max()))] if "log_scale" in params else None,
        )
        print(tag, json.dumps(results[tag], indent=2))

    summary["gates"] = results

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist_a["a"], label="Phase A (neg-ELBO)")
    if hist_ab["b"] is not None and len(hist_ab["b"]):
        ax.plot(np.arange(len(hist_ab["b"])) + PHASE_A_STEPS, hist_ab["b"],
                label="Phase B (fwd-KL)")
    ax.axhline(SVI_FINAL_LOSS, color="r", ls="--", label="SVI final loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend()
    ax.set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gate_f_losses.png"), dpi=140)

    with open(os.path.join(OUT_DIR, "gate_f_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY:", json.dumps(summary["gates"], indent=2, default=str))


if __name__ == "__main__":
    main()
