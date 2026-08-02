"""D2 — logp noise-floor measurement (float32 version).

For n_max in {10, 25}, anchors (a) + 3 from each (b) subset + 3 from (c):
  - logp along 4 random unit rays per point
  - 200 increments t in logspace(-8, -1) (in z units)
  - float32 on GPU

For each ray:
  - fit smooth low-order polynomial over sliding windows
  - noise amplitude = RMS residual
  - probe gradient consistency: compare directional derivative (autodiff) vs
    central finite differences at matched scales

Pre-registered predictions (H1):
  - float32 noise at n_max=25 near anchors (a)/(b) within ~2 orders of magnitude
    of dE_target=0.092 and >> float64 noise (ratio ~1e6-1e9)
  - noise at n_max=25 larger than at n_max=10 by >= 10x
  - The Phase 0 anomaly: tune1 energy RMS at n_max=25/ds1e-8 was 0.029, smaller
    than n_max=10 (0.339) — does noise floor vary strongly with position scale
    around bootstrap point?

FALSIFIER: float32 noise <= 1e-3 everywhere => H1 demoted.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

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

LOG_PATH = os.path.join(OUT_DIR, "d2_f32_log.txt")
_log_fh = open(LOG_PATH, "w", buffering=1)


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()


DE_TARGET = 0.092  # sqrt(17 * 5e-4)
N_RAYS = 4
N_INCREMENTS = 200
T_MIN, T_MAX = 1e-8, 1e-1


def measure_noise_on_ray(logp_fn, z0, direction, n_increments=N_INCREMENTS,
                          t_min=T_MIN, t_max=T_MAX):
    """Measure noise amplitude of logp along a 1D ray.

    Returns:
        t_vals: array of step sizes
        logp_vals: array of logp values
        noise_amplitude: RMS residual from local polynomial fit
        grad_consistency: dict with autodiff vs finite-diff stats
    """
    import jax
    import jax.numpy as jnp

    t_vals = np.logspace(np.log10(t_min), np.log10(t_max), n_increments)
    z0_jnp = jnp.array(z0, dtype=jnp.float32)
    dir_jnp = jnp.array(direction, dtype=jnp.float32)
    dir_jnp = dir_jnp / jnp.linalg.norm(dir_jnp)  # normalize

    # Evaluate logp at z0 + t * dir for each t
    logp_vals = np.zeros(n_increments)
    for i, t in enumerate(t_vals):
        z_t = z0_jnp + float(t) * dir_jnp
        try:
            lp = float(logp_fn(z_t))
            logp_vals[i] = lp if np.isfinite(lp) else np.nan
        except Exception:
            logp_vals[i] = np.nan

    # Also evaluate at z0
    try:
        lp0 = float(logp_fn(z0_jnp))
    except Exception:
        lp0 = np.nan

    # Fit smooth polynomial in sliding windows (log-t space)
    # Window size = 20 increments, step = 5
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
        # Fit degree-3 polynomial
        try:
            coeffs = np.polyfit(t_w - t_w.mean(), lp_w, deg=3)
            lp_fit = np.polyval(coeffs, t_w - t_w.mean())
            residuals.extend((lp_w - lp_fit).tolist())
        except Exception:
            pass

    if residuals:
        noise_amplitude = float(np.sqrt(np.mean(np.array(residuals)**2)))
    else:
        noise_amplitude = np.nan

    # Gradient consistency at a few scales
    grad_stats = {}
    vg_fn = jax.jit(jax.value_and_grad(logp_fn))

    # Evaluate at 5 selected scales for gradient consistency
    for scale_exp in [-6, -5, -4, -3, -2]:
        t_scale = 10.0 ** scale_exp
        z_plus = z0_jnp + t_scale * dir_jnp
        z_minus = z0_jnp - t_scale * dir_jnp
        try:
            lp_plus = float(logp_fn(z_plus))
            lp_minus = float(logp_fn(z_minus))
            fd_deriv = (lp_plus - lp_minus) / (2 * t_scale)

            val0, grad0 = vg_fn(z0_jnp)
            ad_deriv = float(jnp.dot(grad0, dir_jnp))

            if np.isfinite(fd_deriv) and np.isfinite(ad_deriv):
                rel_err = abs(fd_deriv - ad_deriv) / (abs(ad_deriv) + 1e-10)
                grad_stats[f"t_{-scale_exp}"] = {
                    "fd_deriv": fd_deriv,
                    "ad_deriv": ad_deriv,
                    "abs_err": abs(fd_deriv - ad_deriv),
                    "rel_err": float(rel_err),
                }
        except Exception as e:
            grad_stats[f"t_{-scale_exp}"] = {"error": str(e)}

    return {
        "t_vals": t_vals.tolist(),
        "logp_vals": logp_vals.tolist(),
        "logp_at_origin": float(lp0),
        "noise_amplitude": noise_amplitude,
        "noise_vs_target": float(noise_amplitude / DE_TARGET) if np.isfinite(noise_amplitude) else None,
        "grad_consistency": grad_stats,
        "n_finite": int(np.sum(np.isfinite(logp_vals))),
    }


def main():
    log("D2 noise floor (float32) starting")
    log(f"Output: {OUT_DIR}")

    import jax
    import jax.numpy as jnp

    log(f"JAX {jax.__version__} devices: {jax.devices()}")

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
    # Pick 3 from each subset for D2
    frozen_a_pts = late_a[rng.choice(len(late_a), size=3, replace=False)]
    frozen_b_pts = late_b[rng.choice(len(late_b), size=3, replace=False)]

    # Random directions for rays
    ray_rng = np.random.default_rng(99)

    all_results = {}

    for n_max in [10, 25]:
        log(f"\n=== n_max={n_max} (float32) ===")
        t0 = time.perf_counter()

        prob_model = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=n_max)
        simulator = gj_sim.LensSimulator(prob_model.model, prob_model.datasets[0].sim_config, bs=1)

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
        ctx = InferenceContext.from_prob_model(prob_model)
        stage_result = bootstrap_stage.run(ctx, {}, seed=0)
        artifacts = bootstrap_stage.derive_artifacts(stage_result.arrays)
        qz = artifacts["qz"]
        z_bootstrap = np.array(qz.mean())
        log(f"  Bootstrap z: {z_bootstrap[:4]}")

        # qz samples
        key = jax.random.PRNGKey(789)
        qz_samples = np.array(qz.sample(3, seed=key))  # (3, 17)

        # Prior draws
        prior_pts = []
        key2 = jax.random.PRNGKey(321)
        for _ in range(10):
            key2, subkey = jax.random.split(key2)
            try:
                x_prior = prob_model.prior.sample(seed=subkey)
                z_prior = np.array(jnp.stack(prob_model.bij.inverse(x_prior)))
                if z_prior.shape == (17,):
                    prior_pts.append(z_prior)
                    if len(prior_pts) >= 3:
                        break
            except Exception:
                pass
        log(f"  Got {len(prior_pts)} prior samples")

        # Build logp function
        def logp_fn(z):
            val, _ = prob_model.log_prob(simulator, z[jnp.newaxis, :])
            return val[0]

        logp_jit = jax.jit(logp_fn)
        # Warm-up
        _ = logp_jit(jnp.array(z_bootstrap, dtype=jnp.float32))
        log("  logp compiled")

        results_for_nmax = {}

        # Anchor (a): bootstrap
        log("  Probing anchor (a): bootstrap point")
        anchor_results = []
        for ray_idx in range(N_RAYS):
            direction = ray_rng.standard_normal(17)
            direction /= np.linalg.norm(direction)
            log(f"    Ray {ray_idx+1}/{N_RAYS}...")
            res = measure_noise_on_ray(logp_jit, z_bootstrap, direction)
            res["ray_idx"] = ray_idx
            res["anchor"] = "bootstrap"
            anchor_results.append(res)
            log(f"      noise={res['noise_amplitude']:.4e}, "
                f"noise/target={res['noise_vs_target']:.2e}, "
                f"n_finite={res['n_finite']}")
        results_for_nmax["anchor_a_bootstrap"] = anchor_results

        # Anchor (b1): qz samples
        log("  Probing anchor (b1): qz samples")
        b1_results = []
        for pt_idx, z_pt in enumerate(qz_samples):
            for ray_idx in range(N_RAYS):
                direction = ray_rng.standard_normal(17)
                res = measure_noise_on_ray(logp_jit, z_pt, direction)
                res["ray_idx"] = ray_idx
                res["pt_idx"] = pt_idx
                res["anchor"] = "qz_sample"
                b1_results.append(res)
            noise_here = np.mean([r["noise_amplitude"] for r in b1_results[-N_RAYS:]
                                  if np.isfinite(r["noise_amplitude"])])
            log(f"    qz pt {pt_idx}: mean noise over {N_RAYS} rays = {noise_here:.4e}")
        results_for_nmax["anchor_b1_qz_samples"] = b1_results

        # Anchor (b2): frozen run_a positions
        log("  Probing anchor (b2): frozen run_a positions")
        b2_results = []
        for pt_idx, z_pt in enumerate(frozen_a_pts):
            for ray_idx in range(N_RAYS):
                direction = ray_rng.standard_normal(17)
                res = measure_noise_on_ray(logp_jit, z_pt, direction)
                res["ray_idx"] = ray_idx
                res["pt_idx"] = pt_idx
                res["anchor"] = "frozen_run_a"
                b2_results.append(res)
            noise_here = np.mean([r["noise_amplitude"] for r in b2_results[-N_RAYS:]
                                  if np.isfinite(r["noise_amplitude"])])
            log(f"    frozen_a pt {pt_idx}: mean noise over {N_RAYS} rays = {noise_here:.4e}")
        results_for_nmax["anchor_b2_frozen_run_a"] = b2_results

        # Anchor (b3): healthy run_b positions
        log("  Probing anchor (b3): healthy run_b positions")
        b3_results = []
        for pt_idx, z_pt in enumerate(frozen_b_pts):
            for ray_idx in range(N_RAYS):
                direction = ray_rng.standard_normal(17)
                res = measure_noise_on_ray(logp_jit, z_pt, direction)
                res["ray_idx"] = ray_idx
                res["pt_idx"] = pt_idx
                res["anchor"] = "healthy_run_b"
                b3_results.append(res)
            noise_here = np.mean([r["noise_amplitude"] for r in b3_results[-N_RAYS:]
                                  if np.isfinite(r["noise_amplitude"])])
            log(f"    healthy_b pt {pt_idx}: mean noise over {N_RAYS} rays = {noise_here:.4e}")
        results_for_nmax["anchor_b3_healthy_run_b"] = b3_results

        # Anchor (c): prior draws
        if prior_pts:
            log("  Probing anchor (c): prior draws")
            c_results = []
            for pt_idx, z_pt in enumerate(prior_pts[:3]):
                for ray_idx in range(N_RAYS):
                    direction = ray_rng.standard_normal(17)
                    res = measure_noise_on_ray(logp_jit, z_pt, direction)
                    res["ray_idx"] = ray_idx
                    res["pt_idx"] = pt_idx
                    res["anchor"] = "prior_draw"
                    c_results.append(res)
                noise_here = np.mean([r["noise_amplitude"] for r in c_results[-N_RAYS:]
                                      if np.isfinite(r["noise_amplitude"])])
                log(f"    prior pt {pt_idx}: mean noise over {N_RAYS} rays = {noise_here:.4e}")
            results_for_nmax["anchor_c_prior_draws"] = c_results

        all_results[f"n_max_{n_max}"] = results_for_nmax
        log(f"  n_max={n_max} done in {time.perf_counter()-t0:.1f}s")

    # Save results
    out_path = os.path.join(OUT_DIR, "d2_f32_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nD2 float32 results saved to {out_path}")

    # Print summary table
    log(f"\n=== D2 SUMMARY TABLE (float32, dE_target={DE_TARGET}) ===")
    log(f"{'n_max':>6}  {'anchor':>20}  {'noise_mean':>12}  {'noise/tgt':>10}  "
        f"{'n_rays_finite':>14}")
    log("-" * 80)
    for nmax_key in sorted(all_results.keys()):
        n_max = nmax_key.replace("n_max_", "")
        for anchor_key, entries in all_results[nmax_key].items():
            noises = [e["noise_amplitude"] for e in entries if np.isfinite(e["noise_amplitude"])]
            if noises:
                noise_mean = np.mean(noises)
                noise_vs_tgt = noise_mean / DE_TARGET
                n_fin = len(noises)
            else:
                noise_mean = np.nan
                noise_vs_tgt = np.nan
                n_fin = 0
            label = entries[0]["anchor"] if entries else anchor_key
            log(f"{n_max:>6}  {label:>20}  {noise_mean:>12.4e}  {noise_vs_tgt:>10.2e}  "
                f"{n_fin:>14}")

    log("\nD2 float32 complete.")
    _log_fh.close()


if __name__ == "__main__":
    main()
