"""
Batch driver that reproduces the modeling loop from
``experiments/shapelets_systematics/notebooks/sim_system_complex.ipynb`` for one or more
(sim_num, rep) pairs.

Per system the script:

  1. Loads the lensed image and the truth parameters.
  2. Runs a short MAP fit with the lens parameters pinned to truth so that
     the source shapelet (beta + centre) gets a well-defined "truth"
     reference.
  3. Plots ``mass_truth_free_source.png`` for the resulting truth-style
     image reconstruction.
  4. Runs MCLMC starting from the truth (and the recovered shapelet
     parameters), saving:

       - ``mclmc_diagnostics.png``
       - ``cornerplot.png``
       - ``mclmc_samples.npy``
       - ``zscore_summary.json``

All outputs land in
``results/shapelets_systematics/vela{sim}_cam{cam}_rep{rep:02}_{filter_tag}/n_max{nmax:02}/``.

CLI arguments mirror the configuration knobs that vary between runs
(``--n_max``, ``--supersample``, ``--num_pix``, sampler tuning, etc.).
The defaults reproduce the n_max=10 run currently in the notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
import blackjax
from matplotlib import pyplot as plt

from gigalens.jax.inference.mclmc import MCLMC_JIT
from gigalens_research.inference_utils import PipelineConfig, run_pipeline
from gigalens_research.plotting import plot_image_results
from vela_utilities import (
    DEFAULT_BACKGROUND_RMS,
    DEFAULT_CAM,
    DEFAULT_EXP_TIME,
    DEFAULT_NUM_PIX,
    DEFAULT_SUPERSAMPLE,
    build_true_params_shp,
    cornerplot_all,
    free_source_fixed_lens_model,
    load_vela_sim_system,
    plot_diagnostics,
    run_save_dir,
    system_save_dir,
    vela_system_model,
    zscore_summary,
)

tfd = tfp.distributions


DEFAULT_SIM_NUMS = [
    "01", "03", "04", "07", "08", "10", "15", "21", "22", "23", "25", "26",
]


def _parse_int_list(values):
    """Accept '0,1,2' or '0 1 2' style strings as a list of ints."""
    out = []
    for v in values:
        for chunk in str(v).replace(",", " ").split():
            out.append(int(chunk))
    return out


def _parse_str_list(values):
    """Same as `_parse_int_list` but keep the elements as strings (for
    sim_nums, which are zero-padded labels like '01')."""
    out = []
    for v in values:
        for chunk in str(v).replace(",", " ").split():
            out.append(chunk)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run MCLMC modeling on Vela-based simulated systems.",
    )

    # Modeling configuration
    p.add_argument("--n-max", type=int, default=10,
                   help="Shapelet n_max (default: 10).")
    p.add_argument("--supersample", type=int, default=DEFAULT_SUPERSAMPLE,
                   help=f"SimulatorConfig.supersample (default: "
                        f"{DEFAULT_SUPERSAMPLE}).")
    p.add_argument("--num-pix", type=int, default=DEFAULT_NUM_PIX,
                   help=f"Pixel grid size (default: {DEFAULT_NUM_PIX}).")
    p.add_argument("--background-rms", type=float, default=DEFAULT_BACKGROUND_RMS,
                   help=f"Background RMS noise (default: "
                        f"{DEFAULT_BACKGROUND_RMS}).")
    p.add_argument("--exp-time", type=float, default=DEFAULT_EXP_TIME,
                   help=f"Exposure time used for the noise model (default: "
                        f"{DEFAULT_EXP_TIME}).")
    p.add_argument("--no-shapelets", action="store_true",
                   help="Use the Sersic source branch instead of shapelets.")

    # System selection
    p.add_argument("--cam", type=str, default=DEFAULT_CAM,
                   help=f"Camera ID (default: {DEFAULT_CAM}).")
    p.add_argument("--sim-nums", nargs="+", default=DEFAULT_SIM_NUMS,
                   help="Vela source IDs to run, e.g. '01 03 07'. "
                        "Comma-separated lists also accepted.")
    p.add_argument("--reps", nargs="+", default=["0", "1", "2", "3", "4"],
                   help="Rep indices to run, e.g. '0 1 2'.")

    # MAP (fixed-lens bootstrap) configuration
    p.add_argument("--map-num-steps", type=int, default=200,
                   help="Number of MAP optimisation steps (default: 200).")
    p.add_argument("--map-n-samples", type=int, default=100,
                   help="Number of starts for MAP (default: 100).")

    # MCLMC configuration
    p.add_argument("--num-burnin-steps", type=int, default=4000,
                   help="MCLMC burn-in steps (default: 4000).")
    p.add_argument("--num-results", type=int, default=4000,
                   help="MCLMC kept samples per chain (default: 4000).")
    p.add_argument("--n-hmc", type=int, default=8,
                   help="Number of MCLMC chains (default: 8).")
    p.add_argument("--desired-energy-variance", type=float, default=5e-4,
                   help="Target EEVPD for the MCLMC tuner (default: 5e-4).")
    p.add_argument("--frac-tune1", type=float, default=0.2,
                   help="Fraction of burn-in for stage 1 tuning.")
    p.add_argument("--frac-tune2", type=float, default=0.6,
                   help="Fraction of burn-in for stage 2 (mass-matrix) tuning.")
    p.add_argument("--frac-tune3", type=float, default=0.2,
                   help="Fraction of burn-in for stage 3 (L) tuning.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-progress-bar", action="store_true")

    # Plotting
    p.add_argument("--no-cornerplot", action="store_true",
                   help="Skip the cornerplot (slow on large samples).")

    # Bookkeeping
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (sim_num, rep) combinations whose save_dir "
                        "already contains mclmc_samples.npy. Useful for "
                        "resuming a partial sweep.")

    args = p.parse_args(argv)
    args.sim_nums = _parse_str_list(args.sim_nums)
    args.reps = _parse_int_list(args.reps)
    return args


def _try(label, fn, *args, **kwargs):
    """Run `fn(*args, **kwargs)` and swallow any exception so a single
    failing post-processing step doesn't kill the whole run. Returns the
    function's return value on success and ``None`` on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[warn] {label} failed: {exc}")
        traceback.print_exc()
        return None


def run_one_system(args, sim_num, rep):
    """Run the full modeling chain for one (sim_num, rep) and return a dict
    summarising its outputs.

    The MCLMC samples are persisted as early as possible so that a downstream
    plotting / diagnostic failure (e.g. NaN-poisoned samples crashing
    matplotlib) cannot lose the chain on disk.
    """
    use_shapelets = not args.no_shapelets
    sys_dir = system_save_dir(sim_num, rep, cam=args.cam)
    save_dir = run_save_dir(sys_dir, args.n_max)
    samples_path = os.path.join(save_dir, "mclmc_samples.npy")
    summary_path = os.path.join(save_dir, "zscore_summary.json")

    print(f"\n--------------------- {sim_num}-{rep:02d} "
          f"(n_max={args.n_max}) ---------------------")
    print(f"save_dir: {save_dir}")

    if args.skip_existing and os.path.exists(samples_path):
        print(f"[skip] {samples_path} already exists.")
        return {"sim_num": sim_num, "rep": rep,
                "status": "skipped (already done)"}

    os.makedirs(save_dir, exist_ok=True)

    observed_img, true_params, sim_config, _ = load_vela_sim_system(
        sim_num, rep, cam=args.cam,
        num_pix=args.num_pix, supersample=args.supersample,
    )

    # 1. Free-source fixed-lens MAP -> shapelet "truth"
    prob_model_fixed_lens, _ = free_source_fixed_lens_model(
        sim_config, observed_img, true_params,
        background_rms=args.background_rms, exp_time=args.exp_time,
        use_shapelets=use_shapelets, n_max=args.n_max,
    )
    pipelinecfg = PipelineConfig(
        steps=["MAP"],
        map_kwargs={"num_steps": args.map_num_steps,
                    "n_samples": args.map_n_samples},
    )
    results_fixed_lens = run_pipeline(prob_model_fixed_lens, pipelinecfg)
    shp_true = results_fixed_lens["MAP"].MAP_best[2][0]

    # 2. Free Vela model + truth-as-reference
    prob_model, lens_sim = vela_system_model(
        sim_config, observed_img,
        background_rms=args.background_rms, exp_time=args.exp_time,
        use_shapelets=use_shapelets, n_max=args.n_max,
    )
    true_params_shp = build_true_params_shp(true_params, shp_true)
    true_z = jnp.stack(prob_model.bij.inverse(true_params_shp)).T

    def _truth_image_plot():
        fig, axs = plt.subplots(1, 4)
        fig.set_size_inches(20, 5)
        plot_image_results(
            fig, axs, jnp.array(observed_img), prefix="True Param",
            lens_sim=lens_sim, predicted_params=true_params_shp,
            background_rms=args.background_rms, exp_time=args.exp_time,
            use_backward=True,
        )
        plt.savefig(os.path.join(save_dir, "mass_truth_free_source.png"))
        plt.close(fig)

    _try("mass_truth_free_source plot", _truth_image_plot)

    # 3. MCLMC starting from truth
    best = jax.device_get(true_z)
    default_start = jnp.diag(jnp.ones((best.shape[-1],))) * 1e-3
    qz = tfd.MultivariateNormalTriL(
        loc=jnp.squeeze(best), scale_tril=default_start,
    )

    mclmc_start = time.perf_counter()
    debug_hist = MCLMC_JIT(
        prob_model, qz,
        n_hmc=args.n_hmc,
        num_burnin_steps=args.num_burnin_steps,
        num_results=args.num_results,
        desired_energy_variance=args.desired_energy_variance,
        frac_tune1=args.frac_tune1,
        frac_tune2=args.frac_tune2,
        frac_tune3=args.frac_tune3,
        seed=args.seed,
        debug_output=True,
        progress_bar=not args.no_progress_bar,
    )
    mclmc_samples = debug_hist.position[:, -args.num_results:, :]
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        mclmc_samples,
    )
    mclmc_elapsed = time.perf_counter() - mclmc_start
    print(f"MCLMC wall time: {mclmc_elapsed:.1f} s")

    # 4. Save samples BEFORE any plotting/diagnostics that might crash on
    # NaNs. The whole point of this ordering is to make sure a failing
    # cornerplot can never lose the chain.
    jnp.save(samples_path, mclmc_samples)
    print(f"Saved samples to {samples_path}")

    # 5. NaN heads-up (cheap, robust, and useful for the slurm log).
    n_nan = int(jnp.sum(~jnp.isfinite(mclmc_samples)))
    if n_nan:
        print(
            f"[warn] {n_nan} non-finite entries in mclmc_samples "
            f"(shape {tuple(mclmc_samples.shape)}); downstream plots may "
            "fail."
        )

    # 6. R-hat / ESS (NaN-safe arithmetic; we just report).
    rhat = blackjax.diagnostics.potential_scale_reduction(
        mclmc_samples, chain_axis=0, sample_axis=1,
    )
    ess = blackjax.diagnostics.effective_sample_size(
        mclmc_samples, chain_axis=0, sample_axis=1,
    )
    max_rhat = float(jnp.max(rhat))
    min_ess = float(jnp.min(ess))
    print(f"max R-hat: {max_rhat:.4f}")
    print(f"min ESS:   {min_ess:.1f}")

    # 7. Plots that can fail on NaN samples — each isolated.
    _try("plot_diagnostics", plot_diagnostics,
         debug_hist, args.num_burnin_steps,
         args.frac_tune1, args.frac_tune2, args.frac_tune3, save_dir)

    if not args.no_cornerplot:
        _try("cornerplot_all", cornerplot_all,
             mclmc_samples, qz, prob_model, save_dir, true_params_shp)

    # 8. Z-score summary (best-effort).
    summary = _try("zscore_summary", zscore_summary,
                   mclmc_samples, prob_model, true_params_shp, true_params)
    if summary is not None:
        print("label : predicted | true | z-score")
        print(summary)

    summary_payload = {
        "sim_num": sim_num, "rep": rep, "n_max": args.n_max,
        "supersample": args.supersample, "num_pix": args.num_pix,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_nonfinite_samples": n_nan,
        "mclmc_seconds": mclmc_elapsed,
        "mass_zscore": summary,
    }
    try:
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to write {summary_path}: {exc}")

    return {
        "sim_num": sim_num, "rep": rep, "max_rhat": max_rhat,
        "min_ess": min_ess, "n_nonfinite_samples": n_nan,
        "status": "ok" if n_nan == 0 else "ok (with NaNs)",
    }


def main(argv=None):
    args = parse_args(argv)

    print(f"JAX devices: {jax.devices()}")
    print(f"local device count: {jax.local_device_count()}")
    print(f"sim_nums: {args.sim_nums}")
    print(f"reps:     {args.reps}")
    print(f"n_max:    {args.n_max}")
    print(f"supersample: {args.supersample}, num_pix: {args.num_pix}")

    overall_start = time.perf_counter()
    results = []
    for sim_num in args.sim_nums:
        for rep in args.reps:
            try:
                res = run_one_system(args, sim_num, rep)
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"[ERROR] vela{sim_num}_rep{rep:02d}: {exc}")
                traceback.print_exc()
                res = {
                    "sim_num": sim_num, "rep": rep,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            results.append(res)
    elapsed = time.perf_counter() - overall_start

    print("\n=== Summary ===")
    print(f"Total wall time: {elapsed:.1f} s")
    for r in results:
        status = r.get("status", "ok")
        rhat = r.get("max_rhat")
        rhat_str = f"max R-hat = {rhat:.4f}" if rhat is not None else "no R-hat"
        print(f"  vela{r['sim_num']}_rep{r['rep']:02d}: {status} | {rhat_str}")


if __name__ == "__main__":
    main()
