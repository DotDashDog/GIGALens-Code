#!/usr/bin/env python3
"""Seed control for the float32-basis ablation (resolves claim C-4).

The single-seed ablation (basis_precision_ablation.py) found the adapted MCLMC L
differing 42% between float64 and float32 basis, with noisy ESS — uninterpretable
without knowing the natural run-to-run spread. This script measures that spread:
the SAME bootstrap qz (built once, float64) is sampled by MCLMC at several seeds,
in BOTH precisions. The real falsifier:

    float32 perturbs adaptation IFF |f32 - f64| EXCEEDS the f64-vs-f64(seed) spread.

If the float64 runs already span tens of % in L / min-ESS, float32 is exonerated.

Prints incrementally (flushed) so progress is visible mid-run. Run inside the
canonical Shifter container on a GPU node.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from gigalens_research.simtests.system import from_vela_dir
from gigalens_research.simtests.registry import get_inference_builder
from gigalens_research.inference_utils.pipeline import InferenceContext
from gigalens_research.simtests.pipelines import PartialTruthBootstrapQzStage
from gigalens_research.inference import MCLMC
from gigalens.jax.inference import ModellingSequence
import gigalens_research.simtests.experiments.vela_elliptical_sersiclets  # noqa: F401 (registers builder)

DATA = "/global/homes/l/linusu/GIGALens-Code/data"
BUILDER = "epl_shear_sersic_elliptical_sersiclets"


def build_system(basis_precision, num_pix, supersample):
    name = "vela01_cam12_rep00_a0.500_f814w"
    system = from_vela_dir(
        system_dir=f"{DATA}/vela_sim_systems/{name}",
        source_dir=f"{DATA}/vela_sources/vela01_cam12_a0.500_f814w",
        system_id="vela01_cam12_rep00",
        delta_pix=0.03, num_pix=num_pix, supersample=supersample,
        background_rms=0.002, exp_time=2000.0,
    )
    system.likelihood_precision = "float64"
    system.conv_precision = "float32"
    system.basis_precision = basis_precision
    return system


def per_chain_min_ess(positions):
    out = []
    for c in range(positions.shape[0]):
        ess = np.asarray(tfp.mcmc.effective_sample_size(jnp.asarray(positions[c])))
        out.append(float(np.nanmin(ess)))
    return np.array(out)


def run_one(model_seq, qz, n_chains, num_burnin, num_results, seed):
    hist = MCLMC(model_seq, qz, n_hmc=n_chains, num_burnin_steps=num_burnin,
                 num_results=num_results, debug_output=True, seed=seed)
    ss = float(np.mean(np.asarray(hist.step_size[:, -1])))
    L = float(np.mean(np.asarray(hist.L[:, -1])))
    pos = np.asarray(hist.position[:, -num_results:, :])
    ess = per_chain_min_ess(pos)
    return ss, L, float(np.min(ess)), ess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=15)
    ap.add_argument("--num-pix", type=int, default=200)
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--n-chains", type=int, default=8)
    ap.add_argument("--num-burnin", type=int, default=2000)
    ap.add_argument("--num-results", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--qz-seed", type=int, default=0)
    args = ap.parse_args()

    print("JAX", jax.__version__, "| devices:", jax.devices(), flush=True)
    print(f"config: n_max={args.n_max}, ss={args.supersample}, n_chains={args.n_chains}, "
          f"burnin={args.num_burnin}, results={args.num_results}, seeds={args.seeds}, "
          f"qz_seed={args.qz_seed}", flush=True)

    sys64 = build_system(None, args.num_pix, args.supersample)
    ms64 = get_inference_builder(BUILDER)(sys64, n_max=args.n_max)
    cfg32 = dataclasses.replace(ms64.datasets[0].sim_config, basis_precision="float32")
    ms32 = ModellingSequence.from_scene(ms64.scene_model, ms64.prob_model, cfg32)
    print(f"depth={ms64.make_lens_sim(1).depth}, num_free={ms64.model.num_free_params}",
          flush=True)

    # One shared bootstrap qz (float64), as in the ablation.
    ctx = InferenceContext.from_prob_model(ms64)
    stage = PartialTruthBootstrapQzStage(
        system=sys64, free=("source",), map_num_steps=200, map_n_samples=50)
    qz = stage.derive_artifacts(stage.run(ctx, {}, seed=args.qz_seed).arrays)["qz"]
    print("bootstrap qz ready\n", flush=True)

    results = {"float64": {}, "float32": {}}
    for label, ms in (("float64", ms64), ("float32", ms32)):
        for seed in args.seeds:
            ss, L, min_ess, ess = run_one(
                ms, qz, args.n_chains, args.num_burnin, args.num_results, seed)
            results[label][seed] = (ss, L, min_ess, ess)
            print(f"[{label} seed={seed}] step_size={ss:.4f}  L={L:.3f}  "
                  f"min_ESS={min_ess:.1f}  per_chain_ESS={np.array2string(ess, precision=0)}",
                  flush=True)

    def spread(vals):
        a = np.array(vals)
        return a.min(), a.max(), a.mean(), (a.max() - a.min()) / max(abs(a.mean()), 1e-30)

    print("\n" + "=" * 72, flush=True)
    print("SEED-CONTROL SUMMARY (resolves C-4)", flush=True)
    print("=" * 72, flush=True)
    for label in ("float64", "float32"):
        Ls = [results[label][s][1] for s in args.seeds]
        esss = [results[label][s][2] for s in args.seeds]
        ss_ = [results[label][s][0] for s in args.seeds]
        lo, hi, mu, rng = spread(Ls)
        elo, ehi, emu, erng = spread(esss)
        slo, shi, smu, srng = spread(ss_)
        print(f"{label}:  L [{lo:.2f}, {hi:.2f}] mean {mu:.2f} (spread {rng:.0%})   "
              f"min_ESS [{elo:.1f}, {ehi:.1f}] mean {emu:.1f} (spread {erng:.0%})   "
              f"step_size [{slo:.4f}, {shi:.4f}] (spread {srng:.0%})", flush=True)

    print("\nINTERPRETATION (UNCERTIFIED — human grades):", flush=True)
    print("  Compare the float64 L/min_ESS SPREAD above to the float64-vs-float32 gap.", flush=True)
    print("  If the float32 means fall inside the float64 across-seed range, the single-seed", flush=True)
    print("  42% L gap is seed noise (float32 exonerated for adaptation). If float32 sits", flush=True)
    print("  OUTSIDE the float64 range, float32 genuinely perturbs adaptation.", flush=True)


if __name__ == "__main__":
    main()
