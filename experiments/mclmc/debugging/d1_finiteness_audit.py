"""D1 — Finiteness audit: tests H2 (NaN grads from ill-conditioned lstsq solve).

For each n_max in {10, 15, 20, 25, 30} and each anchor class:
  (a) bootstrap/truth point
  (b) bulk proxies: ~20 qz draws at diag_scale 1e-6, ~20 late frozen positions from run_a,
      ~20 late positions from healthy run_b
  (c) far-field control: ~20 prior draws mapped to z

Reports: fraction non-finite logp, fraction non-finite grad (any component),
and for failures localizes the first non-finite stage:
  - prior log_prob
  - bijector fldj
  - simulator output (lstsq image)
  - lstsq coeffs

Pre-registered prediction (H2): NaN grads appear by n_max=25 near anchors (a)/(b);
Phase 0 run (a) showed ZERO kernel rejections during collapse, so we EXPECT D1 to come
back mostly clean near (a)/(b) at n_max=25 — confirming H2 is NOT the primary driver.
FALSIFIER for that expectation: substantial NaN-grad fractions at (a)/(b).

Usage:
  python d1_finiteness_audit.py [--gpu]
  (CPU only mode is the default since D1 just calls jax.value_and_grad)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

# Path setup
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

# Redirect all output to a log file
LOG_PATH = os.path.join(OUT_DIR, "d1_log.txt")
_log_fh = open(LOG_PATH, "w", buffering=1)


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()


def main():
    log("D1 finiteness audit starting")
    log(f"Output: {OUT_DIR}")

    import jax
    import jax.numpy as jnp

    try:
        devices = jax.devices()
        log(f"JAX {jax.__version__} devices: {devices}")
    except Exception as e:
        log(f"JAX {jax.__version__} device query failed: {e} — continuing anyway")

    # -------------------------------------------------------------------------
    # Build the vela shapelets model infrastructure (shared across all n_max)
    # -------------------------------------------------------------------------
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
    log(f"System loaded: {system.system_id}")

    # Import vela_shapelets to trigger registration
    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401
    from gigalens_research.simtests.registry import get_inference_builder
    import gigalens.jax.simulator as gj_sim

    # -------------------------------------------------------------------------
    # Load Phase 0 positions (anchors b)
    # -------------------------------------------------------------------------
    log("Loading Phase 0 positions...")
    pos_a = np.load(os.path.join(PHASE0_DIR, "run_a_nmax25_ds1e-8_nb2000", "hist_position.npy"))
    pos_b = np.load(os.path.join(PHASE0_DIR, "run_b_nmax10_ds1e-8_nb2000", "hist_position.npy"))

    # Late frozen positions from run_a (8 chains, last 20 steps) -> (160, 17)
    late_a = pos_a[:, -20:, :].reshape(-1, 17)   # collapsed n_max=25 positions
    # Late positions from healthy run_b (8 chains, last 20 steps) -> (160, 17)
    late_b = pos_b[:, -20:, :].reshape(-1, 17)   # healthy n_max=10 positions

    # Pick 20 representative points from each
    rng = np.random.default_rng(42)
    idx_a = rng.choice(len(late_a), size=min(20, len(late_a)), replace=False)
    idx_b = rng.choice(len(late_b), size=min(20, len(late_b)), replace=False)
    frozen_a_pts = late_a[idx_a]   # shape (20, 17)
    frozen_b_pts = late_b[idx_b]   # shape (20, 17)

    log(f"Phase 0 positions loaded: frozen_a={frozen_a_pts.shape}, frozen_b={frozen_b_pts.shape}")

    # -------------------------------------------------------------------------
    # Helper: localize first non-finite stage
    # -------------------------------------------------------------------------
    def localize_failure(prob_model, simulator, z_pt):
        """Return the first stage that produces a non-finite value."""
        try:
            import jax.numpy as jnp
            import jax
            z = jnp.array(z_pt, dtype=jnp.float32)
            z_list = list(z)

            # Stage 1: bijector forward
            try:
                x = prob_model.bij.forward(z_list)
                x_finite = jax.tree_util.tree_all(
                    jax.tree_util.tree_map(lambda v: jnp.all(jnp.isfinite(v)), x)
                )
            except Exception as e:
                return f"bijector_forward_exception:{e}"
            if not x_finite:
                return "bijector_forward"

            # Stage 2: prior log_prob
            try:
                lp = prob_model.prior.log_prob(x)
                if not jnp.all(jnp.isfinite(lp)):
                    return "prior_logp"
            except Exception as e:
                return f"prior_logp_exception:{e}"

            # Stage 3: bijector fldj
            try:
                fldj = prob_model.bij.forward_log_det_jacobian(z_list)
                if not jnp.isfinite(fldj):
                    return "bijector_fldj"
            except Exception as e:
                return f"bijector_fldj_exception:{e}"

            # Stage 4: simulator lstsq_simulate
            try:
                im_sim, coeffs = simulator.lstsq_simulate(x, prob_model.observed_image, prob_model.err_map)
                if not jnp.all(jnp.isfinite(im_sim)):
                    return "lstsq_image"
                if not jnp.all(jnp.isfinite(coeffs)):
                    return "lstsq_coeffs"
            except Exception as e:
                return f"lstsq_exception:{e}"

            # Stage 5: likelihood
            try:
                log_like = prob_model.observed_dist.log_prob(im_sim)
                if not jnp.isfinite(log_like):
                    return "likelihood"
            except Exception as e:
                return f"likelihood_exception:{e}"

            return "all_finite"
        except Exception as e:
            return f"unknown_exception:{e}"

    # -------------------------------------------------------------------------
    # Main audit loop
    # -------------------------------------------------------------------------
    n_max_list = [10, 15, 20, 25, 30]
    all_results = {}

    for n_max in n_max_list:
        log(f"\n=== n_max={n_max} ===")
        t0 = time.perf_counter()

        # Build model for this n_max
        try:
            prob_model = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=n_max)
        except Exception as e:
            log(f"  ERROR building model: {e}")
            continue

        simulator = gj_sim.LensSimulator(prob_model.model, prob_model.datasets[0].sim_config, bs=1)
        log(f"  Model built in {time.perf_counter()-t0:.1f}s")

        # Get bootstrap point (anchor a) via VelaBootstrapQzStage
        log("  Getting bootstrap qz...")
        try:
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
            z_bootstrap = np.array(qz.mean())  # shape (17,)
            log(f"  Bootstrap z[:4]: {z_bootstrap[:4]}")
        except Exception as e:
            log(f"  ERROR getting bootstrap: {e}")
            traceback.print_exc(file=_log_fh)
            continue

        # Define log_prob function for grad computation
        def make_logp_fn(prob_model, simulator):
            import jax
            import jax.numpy as jnp

            def logp(z):
                val, _ = prob_model.log_prob(simulator, z[jnp.newaxis, :])
                return val[0]

            return jax.jit(jax.value_and_grad(logp))

        vg_fn = make_logp_fn(prob_model, simulator)

        # Warm-up compile
        log("  Compiling value_and_grad...")
        try:
            _ = vg_fn(jnp.array(z_bootstrap, dtype=jnp.float32))
            log("  Compiled OK")
        except Exception as e:
            log(f"  Compile ERROR: {e}")
            traceback.print_exc(file=_log_fh)
            continue

        def audit_points(label, points, prob_model, simulator, vg_fn):
            """Audit a batch of points, return dict of stats."""
            import jax
            import jax.numpy as jnp

            n = len(points)
            logp_vals = np.zeros(n)
            grad_finite = np.zeros(n, dtype=bool)
            logp_finite = np.zeros(n, dtype=bool)
            localizations = []

            for i, z_pt in enumerate(points):
                z = jnp.array(z_pt, dtype=jnp.float32)
                try:
                    val, grad = vg_fn(z)
                    lp = float(val)
                    logp_vals[i] = lp
                    logp_finite[i] = np.isfinite(lp)
                    gf = bool(np.all(np.isfinite(np.array(grad))))
                    grad_finite[i] = gf

                    if not (logp_finite[i] and gf):
                        loc = localize_failure(prob_model, simulator, z_pt)
                        localizations.append(loc)
                    else:
                        localizations.append("ok")
                except Exception as e:
                    logp_vals[i] = np.nan
                    logp_finite[i] = False
                    grad_finite[i] = False
                    localizations.append(f"exception:{e}")

            frac_nonfinite_logp = float(np.mean(~logp_finite))
            frac_nonfinite_grad = float(np.mean(~grad_finite))

            # Count localizations
            loc_counts = {}
            for loc in localizations:
                loc_counts[loc] = loc_counts.get(loc, 0) + 1

            return {
                "label": label,
                "n_points": n,
                "frac_nonfinite_logp": frac_nonfinite_logp,
                "frac_nonfinite_grad": frac_nonfinite_grad,
                "logp_mean": float(np.nanmean(logp_vals)),
                "logp_std": float(np.nanstd(logp_vals)),
                "localizations": loc_counts,
            }

        results_for_nmax = {}

        # --- Anchor (a): bootstrap point ---
        log("  Anchor (a): bootstrap point")
        res_a = audit_points("bootstrap", [z_bootstrap], prob_model, simulator, vg_fn)
        results_for_nmax["anchor_a_bootstrap"] = res_a
        log(f"    logp={res_a['logp_mean']:.4f}, nonfinite_logp={res_a['frac_nonfinite_logp']:.3f}, "
            f"nonfinite_grad={res_a['frac_nonfinite_grad']:.3f}, loc={res_a['localizations']}")

        # --- Anchor (b): qz draws at diag_scale 1e-6 ---
        log("  Anchor (b1): qz draws diag_scale=1e-6")
        import jax
        key = jax.random.PRNGKey(123)
        # Sample 20 points from qz
        qz_samples = np.array(qz.sample(20, seed=key))  # (20, 17)
        res_b1 = audit_points("qz_ds1e-6", qz_samples, prob_model, simulator, vg_fn)
        results_for_nmax["anchor_b1_qz_ds1e-6"] = res_b1
        log(f"    logp={res_b1['logp_mean']:.4f}±{res_b1['logp_std']:.4f}, "
            f"nonfinite_logp={res_b1['frac_nonfinite_logp']:.3f}, "
            f"nonfinite_grad={res_b1['frac_nonfinite_grad']:.3f}, loc={res_b1['localizations']}")

        # --- Anchor (b): late frozen positions from run_a (n_max=25 collapsed) ---
        log("  Anchor (b2): late frozen positions from run_a (n_max=25 collapsed)")
        res_b2 = audit_points("frozen_run_a", frozen_a_pts, prob_model, simulator, vg_fn)
        results_for_nmax["anchor_b2_frozen_run_a"] = res_b2
        log(f"    logp={res_b2['logp_mean']:.4f}±{res_b2['logp_std']:.4f}, "
            f"nonfinite_logp={res_b2['frac_nonfinite_logp']:.3f}, "
            f"nonfinite_grad={res_b2['frac_nonfinite_grad']:.3f}, loc={res_b2['localizations']}")

        # --- Anchor (b): late positions from healthy run_b (n_max=10) ---
        log("  Anchor (b3): late positions from healthy run_b (n_max=10)")
        res_b3 = audit_points("healthy_run_b", frozen_b_pts, prob_model, simulator, vg_fn)
        results_for_nmax["anchor_b3_healthy_run_b"] = res_b3
        log(f"    logp={res_b3['logp_mean']:.4f}±{res_b3['logp_std']:.4f}, "
            f"nonfinite_logp={res_b3['frac_nonfinite_logp']:.3f}, "
            f"nonfinite_grad={res_b3['frac_nonfinite_grad']:.3f}, loc={res_b3['localizations']}")

        # --- Anchor (c): prior draws ---
        log("  Anchor (c): prior draws")
        prior_samples = []
        key2 = jax.random.PRNGKey(456)
        for i in range(20):
            key2, subkey = jax.random.split(key2)
            try:
                x_prior = prob_model.prior.sample(seed=subkey)
                z_prior = np.array(jax.numpy.stack(prob_model.bij.inverse(x_prior)))
                if z_prior.shape == (17,):
                    prior_samples.append(z_prior)
            except Exception as e:
                log(f"    prior sample {i} failed: {e}")
        log(f"    Got {len(prior_samples)} prior samples")
        if prior_samples:
            res_c = audit_points("prior_draws", prior_samples, prob_model, simulator, vg_fn)
            results_for_nmax["anchor_c_prior_draws"] = res_c
            log(f"    logp={res_c['logp_mean']:.4f}±{res_c['logp_std']:.4f}, "
                f"nonfinite_logp={res_c['frac_nonfinite_logp']:.3f}, "
                f"nonfinite_grad={res_c['frac_nonfinite_grad']:.3f}, loc={res_c['localizations']}")

        all_results[f"n_max_{n_max}"] = results_for_nmax
        log(f"  n_max={n_max} done in {time.perf_counter()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    out_path = os.path.join(OUT_DIR, "d1_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nD1 results saved to {out_path}")

    # Print summary table
    log("\n=== D1 SUMMARY TABLE ===")
    log(f"{'n_max':>6}  {'anchor':>25}  {'nonfinite_logp':>14}  {'nonfinite_grad':>14}  {'localization'}")
    log("-" * 95)
    for nmax_key in sorted(all_results.keys()):
        n_max = nmax_key.replace("n_max_", "")
        for anchor_key, res in all_results[nmax_key].items():
            locs = [k for k, v in res["localizations"].items() if k != "ok"]
            loc_str = ", ".join(locs) if locs else "all_ok"
            log(f"{n_max:>6}  {res['label']:>25}  {res['frac_nonfinite_logp']:>14.3f}  "
                f"{res['frac_nonfinite_grad']:>14.3f}  {loc_str}")

    log("\nD1 audit complete.")
    _log_fh.close()


if __name__ == "__main__":
    main()
