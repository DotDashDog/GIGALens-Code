#!/usr/bin/env python3
"""float32-basis ablation for the scene-API shapelet/sersiclet likelihood.

Profiling (gigalens/wip/profile_scene_likelihood.py) found the gradient of the
lstsq log-likelihood is dominated by the light-basis generation + its backward
pass, and the kernel is ~co-limited by HBM bandwidth and half-rate Ampere FP64.
This script tests whether rendering the light basis in **float32** (gram/solve
and chi^2 reduction stay float64) is *scientifically* acceptable, by the
pre-registered falsifier:

  (A) grad(log_prob) agrees float32-basis vs float64-basis to rel <= 1e-4,
      evaluated NEAR THE TYPICAL SET (post-warmup MCLMC positions), and
  (B) MCLMC's adapted step_size and per-chain min-ESS match the float64 run
      within 5%, same seed / same bootstrap qz.

It reuses the known-converging Vela scheme from
experiments/TestNewAPI/TestSersiclets.ipynb (PartialTruthBootstrapQzStage ->
MCLMC, n_chains=8) so sampler convergence is not an added confound. The basis
precision is the ONLY difference between the two runs (same qz, same seed).

This is a PROPOSING harness (Mode B): it prints an UNCERTIFIED proposed verdict;
the human grades it. Run inside the canonical Shifter container on a GPU node.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import statistics
import time

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from gigalens_research.simtests.system import from_vela_dir
from gigalens_research.simtests.registry import get_inference_builder
# Import for side effect: registers the "epl_shear_sersic_elliptical_sersiclets"
# inference builder via @register_inference_builder (registry is import-populated).
import gigalens_research.simtests.experiments.vela_elliptical_sersiclets  # noqa: F401
from gigalens_research.inference_utils.pipeline import InferenceContext
from gigalens_research.simtests.pipelines import PartialTruthBootstrapQzStage
from gigalens_research.inference import MCLMC

DATA = "/global/homes/l/linusu/GIGALens-Code/data"
BUILDER = "epl_shear_sersic_elliptical_sersiclets"


def build_system(basis_precision, num_pix, supersample):
    cam, sim_num, rep = "12", "01", 0
    name = f"vela{sim_num}_cam{cam}_rep{str(rep).zfill(2)}_a0.500_f814w"
    source = f"vela{sim_num}_cam{cam}_a0.500_f814w"
    system = from_vela_dir(
        system_dir=f"{DATA}/vela_sim_systems/{name}",
        source_dir=f"{DATA}/vela_sources/{source}",
        system_id=f"vela{sim_num}_cam{cam}_rep{str(rep).zfill(2)}",
        delta_pix=0.03, num_pix=num_pix, supersample=supersample,
        background_rms=0.002, exp_time=2000.0,
    )
    system.likelihood_precision = "float64"
    system.conv_precision = "float32"
    system.basis_precision = basis_precision
    return system


def grad_fn(model_seq):
    lens_sim = model_seq.make_lens_sim(1)

    def lp(z):
        return jnp.squeeze(model_seq.prob_model.log_prob(lens_sim, z)[0])

    return jax.jit(jax.grad(lp))


def time_fn(fn, arg, n_warmup=3, n_iter=30):
    jax.block_until_ready(fn(arg))
    for _ in range(n_warmup - 1):
        jax.block_until_ready(fn(arg))
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def per_chain_min_ess(positions):
    """positions: (n_chains, n_samples, dim) -> per-chain min-over-dims ESS, and global min."""
    ess_list = []
    for c in range(positions.shape[0]):
        ess = np.asarray(tfp.mcmc.effective_sample_size(jnp.asarray(positions[c])))
        ess_list.append(float(np.nanmin(ess)))
    return np.array(ess_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=15)
    ap.add_argument("--num-pix", type=int, default=200)
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--n-chains", type=int, default=8)
    ap.add_argument("--num-burnin", type=int, default=2000)
    ap.add_argument("--num-results", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-rel-thresh", type=float, default=1e-4)
    ap.add_argument("--adapt-rel-thresh", type=float, default=0.05)
    args = ap.parse_args()

    print("JAX", jax.__version__, "| x64:", jax.config.jax_enable_x64,
          "| devices:", jax.devices())
    print(f"config: n_max={args.n_max}, num_pix={args.num_pix}, ss={args.supersample}, "
          f"n_chains={args.n_chains}, burnin={args.num_burnin}, results={args.num_results}, "
          f"seed={args.seed}")

    # --- build ONE model (float64), derive the float32-basis twin by swapping ONLY
    # sim_config.basis_precision. Reusing the SAME LensModel + ProbModel (via from_scene)
    # makes the model object literally identical between the two runs — basis precision
    # is then the sole difference — and sidesteps re-instantiating the source profile
    # (whose params list carries shared mutable state across constructions). ---
    import dataclasses
    from gigalens.jax.inference import ModellingSequence
    sys64 = build_system(None, args.num_pix, args.supersample)
    ms64 = get_inference_builder(BUILDER)(sys64, n_max=args.n_max)
    cfg32 = dataclasses.replace(ms64.datasets[0].sim_config, basis_precision="float32")
    ms32 = ModellingSequence.from_scene(ms64.scene_model, ms64.prob_model, cfg32)
    _s64, _s32 = ms64.make_lens_sim(1), ms32.make_lens_sim(1)
    assert _s64.basis_precision is None and _s32.basis_precision == "float32", (
        _s64.basis_precision, _s32.basis_precision)
    print("trace_mode:", _s64.trace_mode,
          "| basis_precision (sim):", _s64.basis_precision, "/", _s32.basis_precision,
          "| depth:", _s64.depth,
          "| num_free:", ms64.model.num_free_params)

    print("\n[1/4] bootstrapping qz (PartialTruthBootstrapQzStage, free=source) ...")
    ctx = InferenceContext.from_prob_model(ms64)
    stage = PartialTruthBootstrapQzStage(
        system=sys64, free=("source",), map_num_steps=200, map_n_samples=50)
    qz = stage.derive_artifacts(stage.run(ctx, {}, seed=args.seed).arrays)["qz"]

    # --- float64 MCLMC: adapted hyperparams + typical-set positions ---
    print("\n[2/4] float64-basis MCLMC ...")
    hist64 = MCLMC(ms64, qz, n_hmc=args.n_chains, num_burnin_steps=args.num_burnin,
                   num_results=args.num_results, debug_output=True, seed=args.seed)
    ss64 = np.asarray(hist64.step_size[:, -1])
    L64 = np.asarray(hist64.L[:, -1])
    pos64 = np.asarray(hist64.position[:, -args.num_results:, :])
    ess64 = per_chain_min_ess(pos64)

    # --- grad agreement near the typical set (last post-warmup position per chain) ---
    print("\n[3/4] grad(log_prob) agreement near the typical set ...")
    typical = jnp.asarray(pos64[:, -1, :])  # (n_chains, dim)
    g64f = grad_fn(ms64)
    g32f = grad_fn(ms32)
    rels = []
    for c in range(typical.shape[0]):
        z = typical[c]
        g64 = np.asarray(g64f(z))
        g32 = np.asarray(g32f(z))
        denom = max(np.max(np.abs(g64)), 1e-30)
        rels.append(float(np.max(np.abs(g64 - g32)) / denom))
    grad_rel_max = max(rels)

    # timing at one typical-set point
    t64 = time_fn(g64f, typical[0])
    t32 = time_fn(g32f, typical[0])

    # --- float32 MCLMC: same qz, same seed ---
    print("\n[4/4] float32-basis MCLMC (same qz, same seed) ...")
    hist32 = MCLMC(ms32, qz, n_hmc=args.n_chains, num_burnin_steps=args.num_burnin,
                   num_results=args.num_results, debug_output=True, seed=args.seed)
    ss32 = np.asarray(hist32.step_size[:, -1])
    L32 = np.asarray(hist32.L[:, -1])
    pos32 = np.asarray(hist32.position[:, -args.num_results:, :])
    ess32 = per_chain_min_ess(pos32)

    # --- compare ---
    def rel(a, b):
        return abs(float(np.mean(a)) - float(np.mean(b))) / max(abs(float(np.mean(a))), 1e-30)

    ss_rel = rel(ss64, ss32)
    L_rel = rel(L64, L32)
    ess_rel = abs(float(np.min(ess64)) - float(np.min(ess32))) / max(float(np.min(ess64)), 1e-30)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"grad rel-err (max over {typical.shape[0]} typical-set pts): {grad_rel_max:.2e}"
          f"   [thresh {args.grad_rel_thresh:.0e}]")
    print(f"per-eval grad time:  f64 {t64:.3f} ms   f32 {t32:.3f} ms   "
          f"speedup {t64 / t32:.2f}x")
    print(f"adapted step_size:   f64 {np.mean(ss64):.5g}   f32 {np.mean(ss32):.5g}   "
          f"rel {ss_rel:.2%}   [thresh {args.adapt_rel_thresh:.0%}]")
    print(f"adapted L:           f64 {np.mean(L64):.5g}   f32 {np.mean(L32):.5g}   "
          f"rel {L_rel:.2%}")
    print(f"min-ESS (global):    f64 {np.min(ess64):.1f}   f32 {np.min(ess32):.1f}   "
          f"rel {ess_rel:.2%}   [thresh {args.adapt_rel_thresh:.0%}]")
    print(f"per-chain ESS f64: {np.array2string(ess64, precision=0)}")
    print(f"per-chain ESS f32: {np.array2string(ess32, precision=0)}")

    grad_ok = grad_rel_max <= args.grad_rel_thresh
    ss_ok = ss_rel <= args.adapt_rel_thresh
    ess_ok = ess_rel <= args.adapt_rel_thresh
    print("\nPROPOSED VERDICT (UNCERTIFIED — human grades):")
    print(f"  (A) grad agreement   : {'meets' if grad_ok else 'FAILS'} 1e-4")
    print(f"  (B) step_size within : {'meets' if ss_ok else 'FAILS'} 5%")
    print(f"  (B) min-ESS within   : {'meets' if ess_ok else 'FAILS'} 5%")
    verdict = "float32-basis PROPOSED-ACCEPTABLE" if (grad_ok and ss_ok and ess_ok) \
        else "float32-basis PROPOSED-REJECTED / needs scrutiny"
    print(f"  => {verdict}  (speedup {t64 / t32:.2f}x on the grad eval)")
    print("\nNOTE: read the per-chain ESS and the adapted step_size spread, not just the "
          "means — a mean within 5% can hide a divergent chain.")


if __name__ == "__main__":
    main()
