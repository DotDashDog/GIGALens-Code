#!/usr/bin/env python
"""Demo-lens validation of flow-preconditioned MAMS and plain NeuTra (plan §5.2/§5.3 dry run).

Four arms, identical 8-chain budgets, on the 22-param demo lens (easiest system;
vanilla MAMS is the in-family reference; the demo posterior itself was validated
against HMC in experiments/laps_validation):

  A) vanilla MAMS on z                       (reference)
  B) flow-MAMS, whitened-spline flow, Phase A only
  C) flow-MAMS, whitened-spline flow, Phase A + Phase B (forward-KL on arm-A-style
     MAMS samples from the GATE I run -- exercises the Phase-B code path end to end)
  D) plain NeuTra (NumPyro-faithful): IAF flow (1-sample ELBO, Adam 3e-3) + NUTS

All sampling diagnostics are computed on DECODED z = T(u) (never raw u). Agreement
gates (pre-registered in docs/logs/carousel-mclmc-sampling.md) compare each arm to
arm A per parameter:

  mean gate:  |mean_i - mean_A| <= 4 * sqrt(sd_i^2/ESS_i + sd_A^2/ESS_A)   (all 22 params)
  width gate: sd_i/sd_A in [1 - 4*se_r, 1 + 4*se_r], se_r = sqrt(1/(2(ESS_i-1)) + 1/(2(ESS_A-1)))
  flow gate (B/C): flow neg-ELBO <= SVI final loss (flow family strictly contains
     the affine SVI family, so it must do at least as well; same sign convention:
     lower = better, verified against model_seq.SVI loss history)

GPU only. Artifacts to OUT_DIR; flows/MAP/SVI cached as .npz so reruns skip stages.
"""
import os
import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)  # survive SIGTERM with output intact

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "demo_validation_out")
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

import gigalens
import gigalens_research
from gigalens_research.inference.mams import MAMS_JIT
from gigalens_research.inference.flow_precond import (
    TransformedProbModel, FlowModelSeq, make_latent_qz,
)
from gigalens_research.inference import flows
from gigalens_research.inference.neutra_nuts import neutra_nuts

SEED = 0
N_CHAINS = 8
NUM_BURNIN = 300
NUM_RESULTS = 300
NUTS_WARMUP = 1000        # NumPyro default warmup
NUTS_RESULTS = 300        # same kept-draw budget as MAMS arms

# Phase A (plan §4.2/§4.4): reverse-KL, 128 draws/step. Phase B: forward-KL on MAMS
# samples from the GATE I run (demo posterior; exercises the code path -- on the demo
# there is no secondary mode for it to add, so B should change little).
# v4 (one-shot config, user-approved): FIXED range 6 + trainable per-dim scale layer
# replaces v3's data-derived range 35. exp(s) absorbs scale corrections by gradient
# descent (flows.py test: sd-30 direction through a +-6 box), so no posterior samples
# are needed to size the box. v3 (range 35, no scale) remains in the record as the
# data-derived alternative.
SPLINE_CFG = dict(num_layers=6, num_bins=8, spline_range=6.0, hidden_dims=(64, 64),
                  trainable_scale=True)
# Flow-MAMS arms get the same adaptation budget NUTS gets (1000 warmup): v2 showed
# 300 burnin from identity mass cannot learn residual flow-imperfection scales.
FLOW_MAMS_BURNIN = 1000
PHASE_A_STEPS = 3000
PHASE_A_DRAWS = 128
# lr 3e-3 (was 1e-3): s must travel log(scale-mismatch) ~ 2 nats on the demo; adam
# travel ~ lr*steps, so 1e-3 x 3000 was marginal. 5e-3 measured stable, 2e-2 not
# (flows.py one-shot test) -- 3e-3 leaves x4 headroom on both sides.
PHASE_A_LR = 3e-3
PHASE_B_STEPS = 1000
PHASE_B_LR = 1e-4
# NumPyro NeuTra preset (numpyro AutoIAFNormal + examples/neutra.py defaults):
IAF_CFG = dict(num_flows=3)  # hidden_dims default [dim, dim] inside the builder
IAF_STEPS = 10000
IAF_LR = 3e-3
IAF_DRAWS = 1                # Trace_ELBO default = 1 particle

GATE_I_ARRAYS = os.path.join(HERE, "gate_i_out", "gate_i_arrays.npz")

MODEL_CARD = {
    "script": os.path.abspath(__file__),
    "gigalens": gigalens.__file__,
    "gigalens_research": gigalens_research.__file__,
    "jax": jax.__version__,
    "x64": bool(jax.config.jax_enable_x64),
    "devices": [str(d) for d in jax.devices()],
    "seed": SEED,
    "arms": ("A=vanilla MAMS, B=spline-A flow-MAMS, C=spline-A+B flow-MAMS, "
             "D=faithful unwhitened-IAF NeuTra-NUTS (expected to record divergence), "
             "E=whitened-IAF NeuTra-NUTS (documented deviation: SVI whitening composed "
             "under the NumPyro IAF recipe)"),
    "budgets": dict(n_chains=N_CHAINS, mams=[NUM_BURNIN, NUM_RESULTS],
                    nuts=[NUTS_WARMUP, NUTS_RESULTS]),
    "spline": {**SPLINE_CFG, "phase_a": [PHASE_A_STEPS, PHASE_A_DRAWS, PHASE_A_LR],
               "phase_b": [PHASE_B_STEPS, PHASE_B_LR, GATE_I_ARRAYS]},
    "iaf": {**IAF_CFG, "steps": IAF_STEPS, "lr": IAF_LR, "n_draws": IAF_DRAWS},
}
print("MODEL CARD:", json.dumps(MODEL_CARD, indent=2))


def build_model_and_fit():
    """Demo lens + MAP + SVI; identical to gate_i_identity.py (kept in sync by hand)."""
    from gigalens.jax.inference import ModellingSequence
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import Dataset, ProbModel
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear

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
    dim = model.num_free_params
    print("dim (num free params):", dim)

    cache = os.path.join(OUT_DIR, "map_svi_cache.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        qz_loc, qz_scale_tril = c["qz_loc"], c["qz_scale_tril"]
        svi_final_loss = float(c["svi_final_loss"])
        print("MAP/SVI loaded from cache. svi final loss:", svi_final_loss)
    else:
        opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
        best, best_lp, best_chisq = model_seq.MAP(opt, seed=0)
        print("MAP done. best_lp:", np.asarray(best_lp).max(), "best red-chi2:",
              np.asarray(best_chisq).min())
        opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
        qz, loss_hist = model_seq.SVI(best, opt, n_vi=128, num_steps=1500)
        svi_final_loss = float(np.asarray(loss_hist)[-1])
        print("SVI done. final loss:", svi_final_loss)
        qz_loc = np.asarray(qz.loc)
        qz_scale_tril = np.asarray(qz.scale.to_dense())
        np.savez(cache, qz_loc=qz_loc, qz_scale_tril=qz_scale_tril,
                 svi_final_loss=svi_final_loss)

    # Fresh host arrays: MAMS closes over qz.covariance() inside shard_map and new
    # JAX rejects SVI's Explicit-sharded arrays there (see gate_i_identity.py).
    qz = tfd.MultivariateNormalTriL(jnp.asarray(qz_loc), jnp.asarray(qz_scale_tril))
    return model_seq, prob_model, qz, qz_loc, qz_scale_tril, svi_final_loss, dim


# --------------------------------------------------------------------------- #
# Flow training                                                                #
# --------------------------------------------------------------------------- #
def train_flow(tag, init_params, make_bij, lp_fn, dim, *, n_draws, num_steps, lr,
               seed, phase_b_samples=None, phase_b_steps=0, phase_b_lr=1e-4):
    """Phase A reverse-KL (+ optional Phase B forward-KL). Returns (params, hists).

    `tag` must encode the architecture/training config (cache key) -- caches carry
    no hash, so a config change with an unchanged tag would silently reuse stale
    params.
    """
    cache = os.path.join(OUT_DIR, f"flow_{tag}.npz")
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        leaves, treedef = jax.tree_util.tree_flatten(init_params)
        cached_leaves = [jnp.asarray(c[f"leaf_{i}"]) for i in range(len(leaves))]
        if not all(bool(jnp.isfinite(l).all()) for l in cached_leaves):
            print(f"flow[{tag}] cache contains non-finite params -- retraining")
            os.remove(cache)
        else:
            params = jax.tree_util.tree_unflatten(treedef, cached_leaves)
            ha = c["hist_a"]
            print(f"flow[{tag}] loaded from cache; final A loss "
                  f"{float(ha[-1]):.2f}" if ha.size else
                  f"flow[{tag}] loaded from cache; no Phase A steps")
            return params, {"a": ha,
                            "b": c["hist_b"] if "hist_b" in c.files else None,
                            "diverged": bool(c["diverged"]) if "diverged" in c.files
                            else False}

    # Losses come from the 11-green-tested library (flows.py), not re-implemented
    # here: neg_elbo_loss expects a batched target_log_prob_fn (n, dim) -> (n,).
    batched_lp = jax.vmap(lp_fn)

    opt = optax.adam(lr)

    @jax.jit
    def step_a(params, opt_state, key):
        loss, g = jax.value_and_grad(flows.neg_elbo_loss)(
            params, make_bij, batched_lp, key, n_draws, dim)
        updates, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    def _params_finite(p):
        return all(bool(jnp.isfinite(l).all()) for l in jax.tree_util.tree_leaves(p))

    params, opt_state = init_params, opt.init(init_params)
    keys = jax.random.split(jax.random.key(seed), num_steps) if num_steps else []
    hist_a = []
    diverged = False
    t0 = time.perf_counter()
    for i in range(num_steps):
        prev = params
        params, opt_state, loss = step_a(params, opt_state, keys[i])
        hist_a.append(float(loss))  # loss is evaluated at `prev`
        # The observed failure mode is finite loss -> NaN grad -> NaN params, so
        # gate on the UPDATED params (a NaN loss alone lags one step behind).
        if not np.isfinite(hist_a[-1]) or not _params_finite(params):
            print(f"flow[{tag}] Phase A DIVERGED at step {i}; reverting to last "
                  "finite params and stopping. Divergence is a recorded finding.")
            params, diverged = prev, True
            break
        if i % 500 == 0:
            print(f"flow[{tag}] A step {i}: {hist_a[-1]:.2f}")
    print(f"flow[{tag}] Phase A done in {time.perf_counter()-t0:.1f}s, final "
          + (f"{hist_a[-1]:.2f}" if hist_a else "n/a (0 steps)"))

    hist_b = None
    if phase_b_samples is not None and phase_b_steps > 0 and not diverged:
        zs = jnp.asarray(phase_b_samples)

        def fkl(params):
            return flows.forward_kl_loss(params, make_bij, zs)

        opt_b = optax.adam(phase_b_lr)

        @jax.jit
        def step_b(params, opt_state):
            loss, g = jax.value_and_grad(fkl)(params)
            updates, opt_state = opt_b.update(g, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        opt_state = opt_b.init(params)
        hist_b = []
        for i in range(phase_b_steps):
            prev = params
            params, opt_state, loss = step_b(params, opt_state)
            hist_b.append(float(loss))
            if not np.isfinite(hist_b[-1]) or not _params_finite(params):
                print(f"flow[{tag}] Phase B DIVERGED at step {i}; reverting to "
                      "last finite params and stopping.")
                params, diverged = prev, True
                break
            if i % 200 == 0:
                print(f"flow[{tag}] B step {i}: {hist_b[-1]:.4f}")
        if hist_b and np.isfinite(hist_b[-1]):
            print(f"flow[{tag}] Phase B done, final {hist_b[-1]:.4f}")

    leaves = jax.tree_util.tree_leaves(params)
    save = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
    save["hist_a"] = np.asarray(hist_a)
    save["diverged"] = np.asarray(diverged)
    if hist_b is not None:
        save["hist_b"] = np.asarray(hist_b)
    np.savez(cache, **save)
    return params, {"a": np.asarray(hist_a),
                    "b": np.asarray(hist_b) if hist_b is not None else None,
                    "diverged": diverged}


# --------------------------------------------------------------------------- #
# Diagnostics on decoded z                                                     #
# --------------------------------------------------------------------------- #
def split_rhat_ess(x):
    """x: (chains, draws, dim). Rank-normalized split-Rhat and bulk-ESS per dim
    (Vehtari et al. 2021, simplified: rank-normalize, split, Gelman Rhat + Geyer ESS)."""
    from scipy.stats import norm as _norm
    c, n, d = x.shape
    half = n // 2
    xs = x[:, : 2 * half, :].reshape(c, 2, half, d).reshape(2 * c, half, d)
    # rank-normalize
    flat = xs.reshape(-1, d)
    ranks = np.argsort(np.argsort(flat, axis=0), axis=0) + 1
    zval = _norm.ppf((ranks - 0.375) / (flat.shape[0] + 0.25))
    zs = zval.reshape(2 * c, half, d)
    m = zs.mean(axis=1)
    B = half * m.var(axis=0, ddof=1)
    W = zs.var(axis=1, ddof=1).mean(axis=0)
    rhat = np.sqrt((W * (half - 1) / half + B / half) / W)
    # Geyer initial-positive-sequence ESS on rank-normalized chains
    ess = np.empty(d)
    za = zs - zs.mean(axis=1, keepdims=True)
    for j in range(d):
        acov = np.array([
            np.mean([np.dot(za[i, : half - t, j], za[i, t:, j]) / half
                     for i in range(2 * c)])
            for t in range(half - 1)
        ])
        var = acov[0] * half / (half - 1.0)
        rho = 1.0 - (var - acov) / var if var > 0 else np.zeros_like(acov)
        s, t = 0.0, 1
        while t + 1 < len(rho):
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            s += pair
            t += 2
        ess[j] = (2 * c * half) / (1.0 + 2.0 * s)
    return rhat, ess


def decode_batch(prob_model, flow_bij, u_samples):
    """(chains, draws, dim) u -> z via T (single vmap; flow forward is cheap)."""
    c, n, d = u_samples.shape
    flat = jnp.asarray(u_samples.reshape(-1, d))
    z = jax.vmap(flow_bij.forward)(flat)
    return np.asarray(z).reshape(c, n, d)


def main():
    (model_seq, prob_model, qz, qz_loc, qz_scale_tril,
     svi_final_loss, dim) = build_model_and_fit()
    lp_fn = lambda z: prob_model.log_prob(z)[0]

    results = {}
    timings = {}

    # ---- Arm A: vanilla MAMS -------------------------------------------------
    t0 = time.perf_counter()
    samples_a = np.asarray(MAMS_JIT(
        model_seq, qz, n_hmc=N_CHAINS, num_burnin_steps=NUM_BURNIN,
        num_results=NUM_RESULTS, seed=SEED, progress_bar=False))
    timings["A_vanilla_mams"] = time.perf_counter() - t0
    results["A"] = dict(z=samples_a)

    # ---- Flows ---------------------------------------------------------------
    key = jax.random.key(SEED + 1)
    k_spline, k_iaf = jax.random.split(key)

    sp_init, sp_make = flows.make_whitened_spline_flow(
        k_spline, dim, jnp.asarray(qz_loc), jnp.asarray(qz_scale_tril), **SPLINE_CFG)
    sp_key = (f"r{int(SPLINE_CFG['spline_range'])}b{SPLINE_CFG['num_bins']}"
              f"ts{int(SPLINE_CFG.get('trainable_scale', False))}"
              f"lr{PHASE_A_LR:g}")

    t0 = time.perf_counter()
    sp_params_a, sp_hist = train_flow(
        f"spline_A_{sp_key}", sp_init, sp_make, lp_fn, dim, n_draws=PHASE_A_DRAWS,
        num_steps=PHASE_A_STEPS, lr=PHASE_A_LR, seed=SEED + 2)
    timings["flow_spline_A"] = time.perf_counter() - t0

    gate_arrays = np.load(GATE_I_ARRAYS)
    phase_b_z = gate_arrays["samples_vanilla"].reshape(-1, dim)

    t0 = time.perf_counter()
    sp_params_ab, sp_hist_ab = train_flow(
        f"spline_AB_{sp_key}", sp_params_a, sp_make, lp_fn, dim, n_draws=PHASE_A_DRAWS,
        num_steps=0, lr=PHASE_A_LR, seed=SEED + 3,
        phase_b_samples=phase_b_z, phase_b_steps=PHASE_B_STEPS,
        phase_b_lr=PHASE_B_LR)
    timings["flow_spline_B"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    iaf_init, iaf_make = flows.make_numpyro_iaf_flow(k_iaf, dim, **IAF_CFG)
    iaf_params, iaf_hist = train_flow(
        "iaf", iaf_init, iaf_make, lp_fn, dim, n_draws=IAF_DRAWS,
        num_steps=IAF_STEPS, lr=IAF_LR, seed=SEED + 4)
    timings["flow_iaf"] = time.perf_counter() - t0

    # Whitened-IAF (arm E): the identical NumPyro recipe, but the IAF acts in the
    # SVI-whitened space (T = Affine_qz o IAF). Documented deviation from vanilla
    # NeuTra; kept alongside faithful arm D so D's outcome is a finding, not a gap.
    k_iaf_w = jax.random.split(key, 3)[2]
    whiten = flows.Affine(jnp.asarray(qz_loc), jnp.asarray(qz_scale_tril))
    iafw_init, iafw_inner_make = flows.make_numpyro_iaf_flow(k_iaf_w, dim, **IAF_CFG)
    iafw_make = lambda p: flows.Sequential([iafw_inner_make(p), whiten])
    t0 = time.perf_counter()
    iafw_params, iafw_hist = train_flow(
        "iaf_whitened", iafw_init, iafw_make, lp_fn, dim, n_draws=IAF_DRAWS,
        num_steps=IAF_STEPS, lr=IAF_LR, seed=SEED + 5)
    timings["flow_iaf_whitened"] = time.perf_counter() - t0

    # Flow gate (B/C): neg-ELBO (mean over last 100 steps) <= SVI final loss.
    flow_elbo_tail = float(np.mean(sp_hist["a"][-100:]))
    print(f"FLOW GATE: spline neg-ELBO tail {flow_elbo_tail:.2f} vs SVI final "
          f"{svi_final_loss:.2f} (must be <=)")

    # ---- Sampling arms (fail fast on bad flows; isolate failures per arm) -----
    arm_status = {"A": "OK"}

    def flow_ok(tag, hist, params):
        if hist.get("diverged"):
            arm_status[tag] = "SKIPPED: flow training diverged (recorded finding)"
            return False
        if not all(bool(jnp.isfinite(l).all())
                   for l in jax.tree_util.tree_leaves(params)):
            arm_status[tag] = "SKIPPED: non-finite flow params"
            return False
        return True

    qz_u = make_latent_qz(dim)
    for tag, params, hist in (("B", sp_params_a, sp_hist),
                              ("C", sp_params_ab, sp_hist_ab)):
        if not flow_ok(tag, hist, params):
            print(f"arm {tag}: {arm_status[tag]}")
            continue
        try:
            bij = sp_make(params)
            wrapped = FlowModelSeq(model_seq, TransformedProbModel(prob_model, bij))
            t0 = time.perf_counter()
            u = np.asarray(MAMS_JIT(
                wrapped, qz_u, n_hmc=N_CHAINS, num_burnin_steps=FLOW_MAMS_BURNIN,
                num_results=NUM_RESULTS, seed=SEED, progress_bar=False))
            timings[f"{tag}_flow_mams"] = time.perf_counter() - t0
            results[tag] = dict(z=decode_batch(prob_model, bij, u), u=u)
            arm_status[tag] = "OK"
        except Exception as e:  # isolate: one arm's failure must not kill the run
            arm_status[tag] = f"FAILED: {type(e).__name__}: {e}"
            print(f"arm {tag}: {arm_status[tag]}")

    # ---- Arms D/E: NeuTra (IAF + NUTS); D faithful-unwhitened, E whitened ------
    for tag, params, make, hist in (("D", iaf_params, iaf_make, iaf_hist),
                                    ("E", iafw_params, iafw_make, iafw_hist)):
        if not flow_ok(tag, hist, params):
            print(f"arm {tag}: {arm_status[tag]}")
            continue
        try:
            bij = make(params)
            neutra_pm = TransformedProbModel(prob_model, bij)
            nuts_lp = lambda u: neutra_pm.log_prob(u)[0]
            t0 = time.perf_counter()
            nuts_out = neutra_nuts(nuts_lp, dim, n_chains=N_CHAINS,
                                   num_warmup=NUTS_WARMUP,
                                   num_results=NUTS_RESULTS, seed=SEED,
                                   target_accept=0.8)
            timings[f"{tag}_neutra_nuts"] = time.perf_counter() - t0
            u_s = np.asarray(nuts_out["samples"])
            results[tag] = dict(z=decode_batch(prob_model, bij, u_s), u=u_s,
                                divergences=int(np.sum(
                                    nuts_out.get("num_divergences", 0))))
            arm_status[tag] = "OK"
        except Exception as e:
            arm_status[tag] = f"FAILED: {type(e).__name__}: {e}"
            print(f"arm {tag}: {arm_status[tag]}")

    # ---- Pullback-scale gate (v3, pre-registered predictions 7/8) --------------
    # Per-dim sd of T^{-1}(arm-A posterior draws) must be in [0.5, 2] and |mean| <= 1
    # for the Phase-A+B flow (7); Phase-A-only is PREDICTED to fail it (mode-seeking
    # reverse KL, prediction 8). This is the demo analog of the carousel GATE F
    # pocket claim and a candidate addition to GATE F for the carousel stage.
    zA_flat = jnp.asarray(results["A"]["z"].reshape(-1, dim))
    pullback = {}
    for name, params in (("spline_A", sp_params_a), ("spline_AB", sp_params_ab)):
        uu = np.asarray(sp_make(params).inverse(zA_flat))
        sd, mu = uu.std(0), np.abs(uu.mean(0))
        pullback[name] = dict(
            sd_range=[float(sd.min()), float(sd.max())], max_abs_mean=float(mu.max()),
            pass_=bool((sd >= 0.5).all() and (sd <= 2.0).all() and (mu <= 1.0).all()))
        print(f"pullback-scale[{name}]: sd [{sd.min():.2f},{sd.max():.2f}] "
              f"|mean|max {mu.max():.2f} pass={pullback[name]['pass_']}")

    # ---- Diagnostics + gates ---------------------------------------------------
    summary = {"model_card": MODEL_CARD, "timings_s": timings,
               "svi_final_loss": svi_final_loss,
               "arm_status": arm_status,
               "flow_training": {
                   "spline_A": dict(steps=int(sp_hist["a"].size),
                                    diverged=bool(sp_hist.get("diverged"))),
                   "spline_B": dict(
                       steps=int(sp_hist_ab["b"].size)
                       if sp_hist_ab["b"] is not None else 0,
                       diverged=bool(sp_hist_ab.get("diverged"))),
                   "iaf": dict(steps=int(iaf_hist["a"].size),
                               diverged=bool(iaf_hist.get("diverged"))),
                   "iaf_whitened": dict(steps=int(iafw_hist["a"].size),
                                        diverged=bool(iafw_hist.get("diverged"))),
               },
               "flow_gate": {"spline_neg_elbo_tail": flow_elbo_tail,
                             "pass": flow_elbo_tail <= svi_final_loss},
               # Pre-registered prediction (5): Phase B trains -- loss finite and
               # decreasing (final < initial).
               "phase_b_gate": {
                   "pass": bool(
                       sp_hist_ab["b"] is not None and len(sp_hist_ab["b"]) > 0
                       and np.isfinite(sp_hist_ab["b"][-1])
                       and sp_hist_ab["b"][-1] < sp_hist_ab["b"][0]),
                   "first": float(sp_hist_ab["b"][0])
                   if sp_hist_ab["b"] is not None and len(sp_hist_ab["b"]) else None,
                   "final": float(sp_hist_ab["b"][-1])
                   if sp_hist_ab["b"] is not None and len(sp_hist_ab["b"]) else None,
               },
               "pullback_scale_gate": pullback,
               "arms": {}}
    diag = {}
    for tag, r in results.items():
        rhat, ess = split_rhat_ess(r["z"])
        diag[tag] = (rhat, ess)
        summary["arms"][tag] = dict(
            max_rhat=float(np.max(rhat)), min_ess=float(np.min(ess)),
            worst_rhat_param=int(np.argmax(rhat)), worst_ess_param=int(np.argmin(ess)),
            # Pre-registered health gate: max split-Rhat < 1.05 (prediction 4).
            rhat_health_pass=bool(np.max(rhat) < 1.05),
        )
        if "u" in r:
            u_rhat, u_ess = split_rhat_ess(r["u"])
            summary["arms"][tag]["u_max_rhat"] = float(np.max(u_rhat))
            summary["arms"][tag]["u_min_ess"] = float(np.min(u_ess))
        if "divergences" in r:
            # Pre-registered rule: 0 divergent transitions expected on this easy
            # whitened target; ANY divergence fails the gate and is a recorded finding.
            summary["arms"][tag]["divergences"] = r["divergences"]
            summary["arms"][tag]["divergence_gate_pass"] = bool(r["divergences"] == 0)

    completed = [t for t in ("B", "C", "D", "E") if t in results]
    mean_a = results["A"]["z"].reshape(-1, dim).mean(0)
    sd_a = results["A"]["z"].reshape(-1, dim).std(0, ddof=1)
    ess_a = diag["A"][1]
    for tag in completed:
        zi = results[tag]["z"].reshape(-1, dim)
        mean_i, sd_i, ess_i = zi.mean(0), zi.std(0, ddof=1), diag[tag][1]
        se_mean = np.sqrt(sd_i**2 / ess_i + sd_a**2 / ess_a)
        mean_ok = np.abs(mean_i - mean_a) <= 4 * se_mean
        se_r = np.sqrt(1 / (2 * (ess_i - 1)) + 1 / (2 * (ess_a - 1)))
        ratio = sd_i / sd_a
        width_ok = (ratio >= 1 - 4 * se_r) & (ratio <= 1 + 4 * se_r)
        summary["arms"][tag].update(
            mean_gate_pass=bool(mean_ok.all()), width_gate_pass=bool(width_ok.all()),
            mean_gate_worst_sigma=float(np.max(np.abs(mean_i - mean_a) / se_mean)),
            width_ratio_range=[float(ratio.min()), float(ratio.max())],
            n_mean_fail=int((~mean_ok).sum()), n_width_fail=int((~width_ok).sum()),
        )
        # v3: agreement gates are only interpretable as bias evidence when the arm
        # is converged (health gate). An unconverged arm is a budget/adaptation
        # finding, not a correctness verdict (the v2 B/C failure taught this).
        summary["arms"][tag]["agreement_interpretable"] = \
            summary["arms"][tag]["rhat_health_pass"]
        if tag == "D":
            # Arm D completing at all falsifies prediction (6); its agreement stats
            # are then diagnostic under (6)'s "investigate", NOT part of the
            # 154-test accounting (checkpoint AMENDED v2 (iii)).
            summary["arms"][tag]["diagnostic_only"] = True

    # Pre-registered prediction (3): direct B-vs-C width comparison (Phase B on a
    # unimodal target is a small perturbation). Direct comparison beats inferring
    # from B~A and C~A, where the shared arm-A noise partially cancels.
    if "B" in results and "C" in results:
        zb = results["B"]["z"].reshape(-1, dim)
        zc = results["C"]["z"].reshape(-1, dim)
        sd_b, sd_c = zb.std(0, ddof=1), zc.std(0, ddof=1)
        ess_b, ess_c = diag["B"][1], diag["C"][1]
        se_r_bc = np.sqrt(1 / (2 * (ess_b - 1)) + 1 / (2 * (ess_c - 1)))
        ratio_bc = sd_c / sd_b
        bc_ok = (ratio_bc >= 1 - 4 * se_r_bc) & (ratio_bc <= 1 + 4 * se_r_bc)
        summary["bc_width_gate"] = dict(
            pass_=bool(bc_ok.all()), n_fail=int((~bc_ok).sum()),
            ratio_range=[float(ratio_bc.min()), float(ratio_bc.max())],
        )
    else:
        summary["bc_width_gate"] = dict(
            pass_=None, note="not computed: arm B and/or C did not complete")

    # ---- Plots (before metrics are trusted) ------------------------------------
    # Traces: worst-ESS param of arm A, all arms.
    wp = summary["arms"]["A"]["worst_ess_param"]
    present = ["A"] + completed
    fig, axes = plt.subplots(len(present), 1, figsize=(10, 2.5 * len(present)),
                             sharex=False, squeeze=False)
    for ax, tag in zip(axes[:, 0], present):
        for ci in range(N_CHAINS):
            ax.plot(results[tag]["z"][ci, :, wp], lw=0.5, alpha=0.7)
        ax.set_ylabel(f"{tag}: z[{wp}]")
    axes = axes[:, 0]
    axes[-1].set_xlabel("draw")
    fig.suptitle(f"traces, worst-ESS param z[{wp}] (all chains)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "traces_worst_param.png"), dpi=140)

    # Mean/width agreement per param.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    markers = {"B": "o", "C": "s", "D": "^", "E": "v"}
    for tag in completed:
        mk = markers[tag]
        zi = results[tag]["z"].reshape(-1, dim)
        se = np.sqrt(zi.std(0, ddof=1)**2 / diag[tag][1] + sd_a**2 / ess_a)
        ax1.plot(np.abs(zi.mean(0) - mean_a) / se, mk, label=tag, alpha=0.8)
        ax2.plot(zi.std(0, ddof=1) / sd_a, mk, label=tag, alpha=0.8)
    ax1.axhline(4, color="r", ls="--"); ax1.set_ylabel("|Δmean| / SE (gate: <4)")
    ax2.axhline(1, color="k", lw=0.5); ax2.set_ylabel("sd ratio vs A")
    ax2.set_xlabel("z column"); ax1.legend()
    fig.suptitle("agreement vs arm A (vanilla MAMS)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "agreement.png"), dpi=140)

    # Flow loss histories.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sp_hist["a"], label="spline Phase A (neg-ELBO)")
    if sp_hist_ab["b"] is not None:
        ax.plot(np.arange(len(sp_hist_ab["b"])) + PHASE_A_STEPS, sp_hist_ab["b"],
                label="spline Phase B (fwd-KL)")
    if iaf_hist["a"].size:
        ax.plot(iaf_hist["a"], label="IAF 1-draw ELBO (faithful)", alpha=0.5)
    if iafw_hist["a"].size:
        ax.plot(iafw_hist["a"], label="IAF 1-draw ELBO (whitened)", alpha=0.5)
    ax.axhline(svi_final_loss, color="r", ls="--", label="SVI final loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "flow_losses.png"), dpi=140)

    np.savez(os.path.join(OUT_DIR, "demo_validation_arrays.npz"),
             arms=np.array(present),
             **{f"z_{t}": results[t]["z"] for t in results},
             **{f"u_{t}": results[t]["u"] for t in results if "u" in results[t]},
             **{f"rhat_{t}": diag[t][0] for t in present},
             **{f"ess_{t}": diag[t][1] for t in present})
    with open(os.path.join(OUT_DIR, "demo_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
