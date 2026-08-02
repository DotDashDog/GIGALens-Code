"""Deterministic repro script for MCLMC adaptation collapse on vela shapelets.

Mirrors TestNewAPI/TestNewAPI.ipynb cell sequence:
  - system vela01_cam12_rep03_a0.500_f814w loaded via from_vela_dir
  - VelaBootstrapQzStage (run inline, no Pipeline)
  - MCLMC_JIT with debug_output=True

CLI flags:
  --n-max INT         shapelet order (default: 25)
  --num-burnin-steps  (default: 2000)
  --num-results       (default: 500)
  --n-chains          (default: 8)
  --diag-scale FLOAT  qz diagonal scale^2 (default: 1e-8)
  --seed INT          (default: 0)
  --x64              enable float64 via jax_enable_x64
  --out-dir PATH      output directory (default: diagnosis_2026-06/<tag>/)
  --tag STR           run tag for subdirectory naming (default: auto)

Outputs (under <out-dir>/<tag>/):
  config.json         all run parameters
  hist_position.npy   shape (n_chains, total_steps, dim)
  hist_step_size.npy  shape (n_chains, total_steps)
  hist_L.npy          shape (n_chains, total_steps)
  hist_nonan.npy      shape (n_chains, total_steps)  -- step_size_adapt success
  hist_xi.npy         shape (n_chains, total_steps)
  hist_energy_change_raw.npy  shape (n_chains, total_steps)  -- pre-zeroing
  hist_kernel_nonan.npy       shape (n_chains, total_steps)  -- kernel rejection flag
  hist_step_norm.npy          shape (n_chains, total_steps)  -- ||x_t - x_{t-1}||
  hist_inverse_mass_matrix.npy shape (n_chains, total_steps, dim, dim)
  summary.json        per-phase summary statistics

Environment:
  Set XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
  before running on shared GPU to avoid evicting the user's Jupyter kernel.

Usage (from the GPU node via srun --overlap):
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
  export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/homes/l/linusu/GIGALens-Code/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \\
      python /global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/repro_vela_shapelets.py \\
      --n-max 25 --num-burnin-steps 2000 --num-results 500 --n-chains 8 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (mirrors TestNewAPI cell 0 / repro_mclmc_238.py approach)
# ---------------------------------------------------------------------------
home = os.path.expanduser("~/")
_gigalens_src = os.path.join(home, "gigalens/src")
_gigalens_code_src = os.path.join(home, "GIGALens-Code/src")
for _p in [_gigalens_src, _gigalens_code_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Repro MCLMC adaptation on vela01_cam12_rep03 shapelets")
    p.add_argument("--n-max", type=int, default=25)
    p.add_argument("--num-burnin-steps", type=int, default=2000)
    p.add_argument("--num-results", type=int, default=500)
    p.add_argument("--n-chains", type=int, default=8)
    p.add_argument("--diag-scale", type=float, default=1e-8,
                   help="Diagonal scale for qz (as variance, i.e. scale_tril = diag(sqrt(diag_scale)))")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x64", action="store_true", help="Enable float64 (jax_enable_x64)")
    p.add_argument("--likelihood-precision", choices=["float32", "mixed", "float64"], default=None,
                   help="Sets System.likelihood_precision. 'mixed'/'float64' imply --x64. "
                        "'float64' = full float64 forward+reduction (the high-n_max fix).")
    p.add_argument("--high-precision", action="store_true",
                   help="Deprecated alias for --likelihood-precision mixed (implies --x64).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: <script_dir>/diagnosis_2026-06/)")
    p.add_argument("--tag", type=str, default=None,
                   help="Run tag for subdirectory naming (default: auto-generated)")
    p.add_argument("--bootstrap-map-steps", type=int, default=200)
    p.add_argument("--bootstrap-map-samples", type=int, default=100)
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip bootstrap MAP; use raw truth params as qz center")
    p.add_argument("--regularize-mm", action="store_true",
                   help="F4: Stan-style shrinkage + PSD floor on every mass-matrix window.")
    return p.parse_args()


def main():
    args = _parse_args()

    # Resolve precision: --high-precision is a deprecated alias for 'mixed'.
    prec = args.likelihood_precision
    if prec is None and args.high_precision:
        prec = "mixed"
    args.likelihood_precision = prec
    # mixed/float64 require x64 to be real, not silently truncated.
    if prec in ("mixed", "float64"):
        args.x64 = True

    # JAX config must be set before importing jax.numpy (via other imports)
    if args.x64:
        import jax
        jax.config.update("jax_enable_x64", True)
        print("[repro] float64 enabled", flush=True)

    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    print(f"[repro] JAX {jax.__version__} devices: {jax.devices()}", flush=True)

    # ---------------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_out = args.out_dir or os.path.join(script_dir, "diagnosis_2026-06")
    if args.tag is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = (f"nmax{args.n_max}_nb{args.num_burnin_steps}_nr{args.num_results}"
               f"_nc{args.n_chains}_ds{args.diag_scale:.0e}_seed{args.seed}"
               f"{'_x64' if args.x64 else ''}")
        tag = tag + f"_{ts}"
    else:
        tag = args.tag
    out_dir = os.path.join(base_out, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[repro] output dir: {out_dir}", flush=True)

    # ---------------------------------------------------------------------------
    # Load system (mirrors notebook cell 4)
    # ---------------------------------------------------------------------------
    data_dir = os.path.join(home, "GIGALens-Code/data")
    system_name = "vela01_cam12_rep03_a0.500_f814w"
    system_dir = os.path.join(data_dir, "vela_sim_systems", system_name)
    source_dir = os.path.join(data_dir, "vela_sources", system_name)

    from gigalens_research.simtests.system import from_vela_dir
    system = from_vela_dir(
        system_dir=system_dir,
        source_dir=source_dir,
        system_id="vela01_cam12_rep03",
        delta_pix=0.03,
        num_pix=200,
        supersample=1,
        background_rms=0.002,
        exp_time=2000.0,
    )
    print(f"[repro] system loaded: {system.system_id}", flush=True)

    # ---------------------------------------------------------------------------
    # Build inference model (mirrors get_inference_builder + vela_shapelets)
    # ---------------------------------------------------------------------------
    # Import vela_shapelets to trigger registration of "epl_shear_sersic_shapelets"
    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401
    from gigalens_research.simtests.registry import get_inference_builder
    # Set the precision flag on the SYSTEM before building prob_model, so it propagates through
    # system.sim_config to BOTH the bootstrap (VelaBootstrapQzStage reads system.sim_config) and
    # the sampler (MCLMC_JIT reads the prob_model's sim_config). Mirrors the notebook
    # (`system.likelihood_precision = "float64"`).
    if args.likelihood_precision is not None:
        system.likelihood_precision = args.likelihood_precision
    prob_model = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=args.n_max)
    print(f"[repro] model built: n_max={args.n_max}  "
          f"likelihood_precision={prob_model.datasets[0].sim_config.likelihood_precision}", flush=True)

    # ---------------------------------------------------------------------------
    # VelaBootstrapQzStage — inline (no Pipeline)
    # ---------------------------------------------------------------------------
    from gigalens_research.simtests.pipelines import VelaBootstrapQzStage

    t0 = time.perf_counter()
    if args.no_bootstrap:
        # Use raw truth in unconstrained space + diag_scale directly
        print("[repro] --no-bootstrap: constructing qz from truth directly", flush=True)
        import jax.tree_util as jtu
        truth_x = system.truth_x
        # Drop Ie from lens_light (as VelaBootstrapQzStage does)
        lens_light_no_ie = [{k: v for k, v in truth_x[1][0].items() if k != "Ie"}]
        # Use zeros for shapelet geometry (beta=log(0.7), cx=0, cy=0)
        shp_params = [{"beta": jnp.array(0.7), "center_x": jnp.array(0.0), "center_y": jnp.array(0.0)}]
        true_params_shp = [truth_x[0], lens_light_no_ie, shp_params]
        true_params_normed = jtu.tree_map(
            lambda x: jnp.squeeze(jnp.asarray(x)), true_params_shp
        )
        true_z = jnp.stack(prob_model.bij.inverse(true_params_normed))
        d = true_z.shape[-1]
        # Keep loc/scale_tril dtype-consistent (under x64 jnp.ones is float64 while the
        # bij.inverse of float32 truth is float32 -> tfd dtype mismatch). Mirror the
        # pipelines.py qz fix: cast scale_tril to loc.dtype.
        loc = jnp.asarray(true_z)
        scale_tril = jnp.diag(jnp.ones(d, loc.dtype) * jnp.sqrt(jnp.asarray(args.diag_scale, loc.dtype)))
        qz = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
    else:
        print(f"[repro] running VelaBootstrapQzStage (map_steps={args.bootstrap_map_steps}, "
              f"map_samples={args.bootstrap_map_samples}, diag_scale={args.diag_scale})", flush=True)
        bootstrap_stage = VelaBootstrapQzStage(
            system=system,
            n_max=args.n_max,
            map_num_steps=args.bootstrap_map_steps,
            map_n_samples=args.bootstrap_map_samples,
            diag_scale=args.diag_scale,
        )

        # Build inference context (what pipeline.run would do)
        from gigalens_research.inference_utils.pipeline import InferenceContext
        ctx = InferenceContext.from_prob_model(prob_model)

        # Run the stage directly
        stage_result = bootstrap_stage.run(ctx, {}, seed=args.seed)
        artifacts = bootstrap_stage.derive_artifacts(stage_result.arrays)
        qz = artifacts["qz"]

    bootstrap_time = time.perf_counter() - t0
    print(f"[repro] qz obtained in {bootstrap_time:.1f}s; dim={qz.mean().shape[-1]}", flush=True)
    print(f"[repro] qz mean[:4]: {jnp.asarray(qz.mean())[:4]}", flush=True)

    # ---------------------------------------------------------------------------
    # MCLMC run
    # ---------------------------------------------------------------------------
    from gigalens_research.inference import MCLMC_JIT

    print(f"[repro] starting MCLMC: n_chains={args.n_chains}, "
          f"num_burnin={args.num_burnin_steps}, num_results={args.num_results}, "
          f"seed={args.seed}", flush=True)

    t1 = time.perf_counter()
    hist = MCLMC_JIT(
        prob_model=prob_model,
        qz=qz,
        n_hmc=args.n_chains,
        num_burnin_steps=args.num_burnin_steps,
        num_results=args.num_results,
        desired_energy_variance=5e-4,
        frac_tune1=0.2,
        frac_tune2=0.6,
        frac_tune3=0.2,
        progress_bar=False,
        seed=args.seed,
        debug_output=True,
        regularize_mass_matrix=args.regularize_mm,
    )
    mclmc_time = time.perf_counter() - t1
    print(f"[repro] MCLMC done in {mclmc_time:.1f}s", flush=True)

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    print(f"[repro] saving to {out_dir}", flush=True)

    # Save all Hist fields
    fields_to_save = {
        "position": hist.position,
        "step_size": hist.step_size,
        "L": hist.L,
        "nonan": hist.nonan,
        "xi": hist.xi,
        "energy_change_raw": hist.energy_change_raw,
        "kernel_nonan": hist.kernel_nonan,
        "step_norm": hist.step_norm,
        "inverse_mass_matrix": hist.inverse_mass_matrix,
    }
    for name, arr in fields_to_save.items():
        path = os.path.join(out_dir, f"hist_{name}.npy")
        np.save(path, np.asarray(arr))
        print(f"[repro]   saved {name}: shape={arr.shape}", flush=True)

    # Config
    total_steps = args.num_burnin_steps + args.num_results
    n_steps1 = round(args.num_burnin_steps * 0.2)
    n_steps2 = round(args.num_burnin_steps * 0.6)
    n_steps3 = round(args.num_burnin_steps * 0.2)
    config = {
        "n_max": args.n_max,
        "num_burnin_steps": args.num_burnin_steps,
        "num_results": args.num_results,
        "n_chains": args.n_chains,
        "diag_scale": args.diag_scale,
        "seed": args.seed,
        "x64": args.x64,
        "likelihood_precision": args.likelihood_precision,
        "bootstrap_map_steps": args.bootstrap_map_steps,
        "bootstrap_map_samples": args.bootstrap_map_samples,
        "no_bootstrap": args.no_bootstrap,
        "total_steps": total_steps,
        "tune1_steps": n_steps1,
        "tune2_steps": n_steps2,
        "tune3_steps": n_steps3,
        "dim": int(hist.position.shape[-1]),
        "jax_version": jax.__version__,
        "mclmc_wall_time_s": mclmc_time,
        "bootstrap_wall_time_s": bootstrap_time,
        "tag": tag,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Per-phase summary
    n_chains_actual = hist.position.shape[0]
    dim = hist.position.shape[-1]
    # Phase boundaries: [0, n_steps1) = tune1, [n_steps1, n_steps1+n_steps2) = tune2,
    # ... tune3 ..., then sampling
    tune1_end = n_steps1
    tune2_end = n_steps1 + n_steps2
    tune3_end = n_steps1 + n_steps2 + n_steps3

    def phase_stats(name, start, end):
        sl = slice(start, end)
        ss = np.asarray(hist.step_size[:, sl])
        xi = np.asarray(hist.xi[:, sl])
        knonan = np.asarray(hist.kernel_nonan[:, sl])
        snorm = np.asarray(hist.step_norm[:, sl])
        ecraw = np.asarray(hist.energy_change_raw[:, sl])
        n_steps_phase = end - start
        return {
            "name": name,
            "steps": n_steps_phase,
            "step_size_init": float(ss[:, 0].mean()) if n_steps_phase > 0 else None,
            "step_size_final": float(ss[:, -1].mean()) if n_steps_phase > 0 else None,
            "step_size_min": float(ss.min()) if n_steps_phase > 0 else None,
            "step_size_max": float(ss.max()) if n_steps_phase > 0 else None,
            "xi_q10": float(np.nanpercentile(xi, 10)) if n_steps_phase > 0 else None,
            "xi_q50": float(np.nanpercentile(xi, 50)) if n_steps_phase > 0 else None,
            "xi_q90": float(np.nanpercentile(xi, 90)) if n_steps_phase > 0 else None,
            "kernel_nonan_frac": float(knonan.mean()) if n_steps_phase > 0 else None,
            "kernel_rejected_frac": float((~knonan).mean()) if n_steps_phase > 0 else None,
            "step_norm_median": float(np.median(snorm)) if n_steps_phase > 0 else None,
            "step_norm_p10": float(np.percentile(snorm, 10)) if n_steps_phase > 0 else None,
            "energy_change_raw_rms": float(np.sqrt(np.nanmean(ecraw**2))) if n_steps_phase > 0 else None,
        }

    phases = [
        phase_stats("tune1", 0, tune1_end),
        phase_stats("tune2", tune1_end, tune2_end),
        phase_stats("tune3", tune2_end, tune3_end),
        phase_stats("sampling", tune3_end, total_steps),
    ]

    # Mass matrix eigenvalue summary at each window boundary
    # Windows are at: step1 = mm_start + w1-1, step2 = mm_start + w1 + w2 -1, step3 = mm_start + w1+w2+w3-1
    mm_start = n_steps1
    n_mm_steps = round(0.67 * n_steps2)
    ratios = [1, 2, 4]
    total_ratio = sum(ratios)
    w_sizes = [max(1, round(n_mm_steps * r / total_ratio)) for r in ratios]
    w_sizes[-1] = n_mm_steps - sum(w_sizes[:-1])
    pos, window_ends = mm_start, []
    for ws in w_sizes:
        pos += ws
        window_ends.append(pos - 1)

    mm_eig_stats = []
    imm = np.asarray(hist.inverse_mass_matrix)  # (n_chains, total_steps, dim, dim)
    for we in window_ends:
        if 0 <= we < total_steps:
            # Chain 0 (all chains share same mass matrix)
            m = imm[0, we]
            try:
                eigs = np.linalg.eigvalsh(m)
                mm_eig_stats.append({
                    "window_end_step": we,
                    "eig_min": float(eigs.min()),
                    "eig_max": float(eigs.max()),
                    "eig_min_pos": float(eigs[eigs > 0].min()) if np.any(eigs > 0) else None,
                    "cond": float(eigs.max() / max(eigs[eigs > 0].min(), 1e-300)) if np.any(eigs > 0) else None,
                    "n_negative_eigs": int(np.sum(eigs < 0)),
                })
            except Exception as ex:
                mm_eig_stats.append({"window_end_step": we, "error": str(ex)})

    summary = {
        "config": config,
        "phases": phases,
        "mass_matrix_window_eigenvalues": mm_eig_stats,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print key summary to stdout
    print("\n=== SUMMARY ===", flush=True)
    for ph in phases:
        if ph["steps"] == 0:
            continue
        print(f"  {ph['name']:10s}: eps {ph['step_size_init']:.4g} -> {ph['step_size_final']:.4g}"
              f"  xi[10,50,90]=[{ph['xi_q10']:.3g},{ph['xi_q50']:.3g},{ph['xi_q90']:.3g}]"
              f"  kern_rej={ph['kernel_rejected_frac']:.3f}"
              f"  step_norm_med={ph['step_norm_median']:.3g}", flush=True)
    for e in mm_eig_stats:
        if "error" in e:
            print(f"  MM window step={e['window_end_step']}: ERROR {e['error']}", flush=True)
        else:
            print(f"  MM window step={e['window_end_step']:4d}: eig [{e['eig_min']:.3g}, {e['eig_max']:.3g}]"
                  f"  cond={e['cond']:.3g}  n_neg={e['n_negative_eigs']}", flush=True)

    print(f"\n[repro] all outputs written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
