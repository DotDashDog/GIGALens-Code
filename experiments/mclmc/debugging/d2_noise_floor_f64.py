"""D2 — logp noise-floor measurement (float64 version).

IMPORTANT: This script enables jax_enable_x64 at startup and must be run
in a SEPARATE process from the float32 version. Never toggle mid-process.

Same anchor points and methodology as d2_noise_floor_f32.py but:
  - jax_enable_x64=True from the start
  - JAX_PLATFORMS=cpu (float64 on GPU can be slow; CPU is fine for this)
  - Smaller batch: n_max in {10, 25}, same anchors but only 2 rays per point

We compare the float32 vs float64 noise amplitude to assess the f32/f64 ratio.
Prediction: if H1 is active, f32/f64 noise ratio should be ~1e6-1e9 at n_max=25.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# Enable x64 FIRST, before any jax imports
import jax
jax.config.update("jax_enable_x64", True)
print("[d2_f64] float64 enabled", flush=True)

home = os.path.expanduser("~/")
for _p in [
    os.path.join(home, "gigalens/src"),
    os.path.join(home, "GIGALens-Code/src"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = os.path.join(
    home,
    "GIGALens-Code/experiments/mclmc/debugging/diagnosis_2026-06/d1_d2",
)
PHASE0_DIR = os.path.join(
    home,
    "GIGALens-Code/experiments/mclmc/debugging/diagnosis_2026-06",
)
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "d2_f64_log.txt")
_log_fh = open(LOG_PATH, "w", buffering=1)


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()


DE_TARGET = 0.092
N_RAYS = 2
N_INCREMENTS = 200
T_MIN, T_MAX = 1e-8, 1e-1


def measure_noise_on_ray(logp_fn, z0, direction, dtype, n_increments=N_INCREMENTS,
                          t_min=T_MIN, t_max=T_MAX):
    import jax.numpy as jnp

    t_vals = np.logspace(np.log10(t_min), np.log10(t_max), n_increments)
    z0_jnp = jnp.array(z0, dtype=dtype)
    dir_jnp = jnp.array(direction, dtype=dtype)
    dir_jnp = dir_jnp / jnp.linalg.norm(dir_jnp)

    logp_vals = np.zeros(n_increments)
    for i, t in enumerate(t_vals):
        z_t = z0_jnp + dtype(t) * dir_jnp
        try:
            lp = float(logp_fn(z_t))
            logp_vals[i] = lp if np.isfinite(lp) else np.nan
        except Exception:
            logp_vals[i] = np.nan

    # Polynomial noise estimate
    window_size = 20
    step_size = 5
    log_t = np.log10(t_vals)
    residuals = []

    for start in range(0, n_increments - window_size, step_size):
        end = start + window_size
        t_window = log_t[start:end]
        lp_window = logp_vals[start:end]
        mask = np.isfinite(lp_window)
        if mask.sum() < window_size // 2:
            continue
        t_w = t_window[mask]
        lp_w = lp_window[mask]
        try:
            coeffs = np.polyfit(t_w - t_w.mean(), lp_w, deg=3)
            lp_fit = np.polyval(coeffs, t_w - t_w.mean())
            residuals.extend((lp_w - lp_fit).tolist())
        except Exception:
            pass

    noise_amplitude = float(np.sqrt(np.mean(np.array(residuals)**2))) if residuals else np.nan

    return {
        "t_vals": t_vals.tolist(),
        "logp_vals": logp_vals.tolist(),
        "noise_amplitude": noise_amplitude,
        "noise_vs_target": float(noise_amplitude / DE_TARGET) if np.isfinite(noise_amplitude) else None,
        "n_finite": int(np.sum(np.isfinite(logp_vals))),
    }


def main():
    log("D2 noise floor (float64) starting")
    log(f"Output: {OUT_DIR}")
    log(f"JAX {jax.__version__} devices: {jax.devices()}")
    log(f"x64 enabled: {jax.config.jax_enable_x64}")

    import jax.numpy as jnp

    # Check memory usage (limit for CPU runs)
    import resource
    def get_rss_gb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    log(f"Initial RSS: {get_rss_gb():.2f} GB")

    # Load system
    log("Loading system...")
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

    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401
    from gigalens_research.simtests.registry import get_inference_builder
    import gigalens.jax.simulator as gj_sim

    # Load Phase 0 positions
    pos_a = np.load(os.path.join(PHASE0_DIR, "run_a_nmax25_ds1e-8_nb2000", "hist_position.npy"))
    pos_b = np.load(os.path.join(PHASE0_DIR, "run_b_nmax10_ds1e-8_nb2000", "hist_position.npy"))

    late_a = pos_a[:, -20:, :].reshape(-1, 17)
    late_b = pos_b[:, -20:, :].reshape(-1, 17)

    rng = np.random.default_rng(42)
    frozen_a_pts = late_a[rng.choice(len(late_a), size=3, replace=False)]
    frozen_b_pts = late_b[rng.choice(len(late_b), size=3, replace=False)]

    ray_rng = np.random.default_rng(99)

    all_results = {}
    DTYPE = jnp.float64

    for n_max in [10, 25]:
        log(f"\n=== n_max={n_max} (float64) ===")
        log(f"  RSS before model build: {get_rss_gb():.2f} GB")

        # Guard: abort if memory is too high
        if get_rss_gb() > 25:
            log(f"  ABORT: RSS {get_rss_gb():.2f} GB > 25 GB limit")
            break

        t0 = time.perf_counter()
        model_seq = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=n_max)
        prob_model = model_seq.prob_model
        simulator = gj_sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)

        # Get bootstrap point
        log("  Getting bootstrap qz...")
        import tensorflow_probability.substrates.jax as tfp
        tfd = tfp.distributions
        from gigalens_research.simtests.pipelines import VelaBootstrapQzStage
        from gigalens_research.inference_utils.pipeline import InferenceContext

        bootstrap_stage = VelaBootstrapQzStage(
            system=system,
            n_max=n_max,
            map_num_steps=200,
            map_n_samples=100,
            diag_scale=1e-6,
        )
        ctx = InferenceContext.from_modelling_sequence(model_seq)
        stage_result = bootstrap_stage.run(ctx, {}, seed=0)
        artifacts = bootstrap_stage.derive_artifacts(stage_result.arrays)
        qz = artifacts["qz"]
        z_bootstrap = np.array(qz.mean())
        log(f"  Bootstrap z[:4]: {z_bootstrap[:4]}")

        qz_samples = np.array(qz.sample(3, seed=jax.random.PRNGKey(789)))  # float32

        prior_pts = []
        key2 = jax.random.PRNGKey(321)
        for _ in range(10):
            key2, subkey = jax.random.split(key2)
            try:
                x_prior = prob_model.prior.sample(seed=subkey)
                z_prior = np.array(jnp.stack(prob_model.bij.inverse(x_prior)))
                if z_prior.shape == (17,):
                    prior_pts.append(z_prior.astype(np.float64))
                    if len(prior_pts) >= 3:
                        break
            except Exception:
                pass

        # Build logp function in float64
        def logp_fn(z):
            z_f64 = jnp.array(z, dtype=jnp.float64)
            val, _ = prob_model.log_prob(simulator, z_f64[jnp.newaxis, :])
            return val[0]

        logp_jit = jax.jit(logp_fn)
        # Warm-up
        log("  Compiling float64 logp...")
        z_bs_f64 = jnp.array(z_bootstrap, dtype=DTYPE)
        _ = logp_jit(z_bs_f64)
        log("  Compiled OK")
        log(f"  RSS after compile: {get_rss_gb():.2f} GB")

        results_for_nmax = {}

        # Helper
        def probe_anchor(label, points):
            all_res = []
            for pt_idx, z_pt in enumerate(points):
                pt_noise = []
                for ray_idx in range(N_RAYS):
                    direction = ray_rng.standard_normal(17)
                    res = measure_noise_on_ray(logp_jit, z_pt, direction, DTYPE)
                    res["ray_idx"] = ray_idx
                    res["pt_idx"] = pt_idx
                    res["anchor"] = label
                    all_res.append(res)
                    if np.isfinite(res["noise_amplitude"]):
                        pt_noise.append(res["noise_amplitude"])
                log(f"    {label} pt {pt_idx}: mean noise = "
                    f"{np.mean(pt_noise):.4e}" if pt_noise else "    no finite rays")
            return all_res

        log("  Probing anchor (a): bootstrap")
        results_for_nmax["anchor_a_bootstrap"] = probe_anchor("bootstrap", [z_bootstrap.astype(np.float64)])

        log("  Probing anchor (b1): qz samples")
        results_for_nmax["anchor_b1_qz_samples"] = probe_anchor("qz_sample", qz_samples.astype(np.float64))

        log("  Probing anchor (b2): frozen run_a")
        results_for_nmax["anchor_b2_frozen_run_a"] = probe_anchor("frozen_run_a", frozen_a_pts.astype(np.float64))

        log("  Probing anchor (b3): healthy run_b")
        results_for_nmax["anchor_b3_healthy_run_b"] = probe_anchor("healthy_run_b", frozen_b_pts.astype(np.float64))

        if prior_pts:
            log("  Probing anchor (c): prior draws")
            results_for_nmax["anchor_c_prior_draws"] = probe_anchor("prior_draw", [p.astype(np.float64) for p in prior_pts[:3]])

        all_results[f"n_max_{n_max}"] = results_for_nmax
        log(f"  n_max={n_max} done in {time.perf_counter()-t0:.1f}s, RSS={get_rss_gb():.2f} GB")

    # Save
    out_path = os.path.join(OUT_DIR, "d2_f64_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nD2 float64 results saved to {out_path}")

    # Print summary
    log(f"\n=== D2 SUMMARY TABLE (float64, dE_target={DE_TARGET}) ===")
    log(f"{'n_max':>6}  {'anchor':>20}  {'noise_mean':>12}  {'noise/tgt':>10}")
    log("-" * 60)
    for nmax_key in sorted(all_results.keys()):
        n_max = nmax_key.replace("n_max_", "")
        for anchor_key, entries in all_results[nmax_key].items():
            noises = [e["noise_amplitude"] for e in entries if np.isfinite(e["noise_amplitude"])]
            if noises:
                noise_mean = np.mean(noises)
                noise_vs_tgt = noise_mean / DE_TARGET
                label = entries[0]["anchor"] if entries else anchor_key
                log(f"{n_max:>6}  {label:>20}  {noise_mean:>12.4e}  {noise_vs_tgt:>10.2e}")
            else:
                log(f"{n_max:>6}  {anchor_key:>20}  {'NaN':>12}  {'NaN':>10}")

    log("\nD2 float64 complete.")
    _log_fh.close()


if __name__ == "__main__":
    main()
