"""D3 — Direct EEVPD(eps) curve measurement.

Measures Var(dE)/dim vs eps for the real MCLMC kernel on the real n_max∈{10,25}
vela shapelets model, bypassing ALL adaptation.  Tests whether the step-size
controller's assumed xi∝eps^6 scaling holds in float32 at n_max=25.

Pre-registered predictions (from plan valiant-hatching-kettle.md):
- Healthy (n_max=10): log-log slope ≈6 over several decades, 5e-4 crossing at eps~O(0.1–1).
- H1 (noise floor): at n_max=25/float32 the curve flattens to an eps-independent floor;
  floor ≥ 5e-4 ⟹ H4 (target unreachable). float64 floor should be ~1e6–1e12 lower.
- FALSIFIER: clean slope-6 down to Var(dE)/dim ≪ 5e-4 in float32 ⟹ pure controller (H3).

Usage (inside the shifter container with GPU allocation):
  python d3_eevpd_vs_eps.py --precision float32 --n-max 25
  python d3_eevpd_vs_eps.py --precision float64 --n-max 25
  python d3_eevpd_vs_eps.py --precision float32 --n-max 10
  python d3_eevpd_vs_eps.py --precision float64 --n-max 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — same as repro_vela_shapelets.py
# ---------------------------------------------------------------------------
home = os.path.expanduser("~/")
for _p in [
    os.path.join(home, "sidecar_jax_upgrade"),
    os.path.join(home, "gigalens/src"),
    os.path.join(home, "GIGALens-Code/src"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n-max", type=int, required=True, help="Shapelet order (10 or 25)")
    p.add_argument("--precision", choices=["float32", "float64"], default="float32")
    p.add_argument("--n-eps", type=int, default=25, help="Number of eps grid points")
    p.add_argument("--eps-lo", type=float, default=1e-7, help="Minimum eps")
    p.add_argument("--eps-hi", type=float, default=10.0, help="Maximum eps")
    p.add_argument("--K", type=int, default=64, help="Momenta samples per eps point")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip bootstrap MAP; use run_a step-0 position as qz.mean proxy "
                        "(avoids OOM at n_max=25; step-0 pos differs from true qz.mean by ~1e-4)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric constructors (dim=17)
# ---------------------------------------------------------------------------
DIM = 17
DESIRED_ENERGY_VAR = 5e-4  # xi = 1 target
TARGET_DE_RMS = (DIM * DESIRED_ENERGY_VAR) ** 0.5  # ≈ 0.092


def _diag_metric(scale, dim=DIM):
    """Return scalar * identity as 2D array."""
    return np.eye(dim, dtype=np.float64) * scale


def _build_metrics(imm_a_final, imm_b_final):
    """Four metrics as float64 numpy arrays (shape (17,17)).

    (i)   1e-8 * I   — run_a init metric
    (ii)  1e-6 * I   — baseline init
    (iii) final adapted metric from run_a (n_max=25, collapsed)
    (iv)  final adapted metric from run_b (n_max=10, healthy)
    """
    return {
        "diag_1e-8_I": _diag_metric(1e-8),
        "diag_1e-6_I": _diag_metric(1e-6),
        "run_a_final": imm_a_final.astype(np.float64),
        "run_b_final": imm_b_final.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse()

    # --- JAX config MUST come before any jax import ---
    if args.precision == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
        print(f"[D3] float64 enabled", flush=True)

    import jax
    import jax.numpy as jnp
    print(f"[D3] JAX {jax.__version__} devices: {jax.devices()}", flush=True)
    print(f"[D3] n_max={args.n_max} precision={args.precision} K={args.K} n_eps={args.n_eps}", flush=True)

    # ---------------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_out = args.out_dir or os.path.join(script_dir, "diagnosis_2026-06", "d3")
    out_dir = os.path.join(
        base_out,
        f"nmax{args.n_max}_{args.precision}_K{args.K}_eps{args.n_eps}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[D3] output dir: {out_dir}", flush=True)

    # ---------------------------------------------------------------------------
    # Load run_a / run_b artifacts (positions + final metrics)
    # ---------------------------------------------------------------------------
    diag_dir = os.path.join(script_dir, "diagnosis_2026-06")
    run_a_dir = os.path.join(diag_dir, "run_a_nmax25_ds1e-8_nb2000")
    run_b_dir = os.path.join(diag_dir, "run_b_nmax10_ds1e-8_nb2000")

    pos_a = np.load(os.path.join(run_a_dir, "hist_position.npy"))  # (8, 2500, 17)
    pos_b = np.load(os.path.join(run_b_dir, "hist_position.npy"))
    imm_a_all = np.load(os.path.join(run_a_dir, "hist_inverse_mass_matrix.npy"))  # (8,2500,17,17)
    imm_b_all = np.load(os.path.join(run_b_dir, "hist_inverse_mass_matrix.npy"))

    # Final adapted metric = last step, chain 0 (all chains share the same metric)
    imm_a_final = imm_a_all[0, -1]  # (17,17)
    imm_b_final = imm_b_all[0, -1]

    print(f"[D3] run_a final metric[0,0]={imm_a_final[0,0]:.3e}  run_b final metric[0,0]={imm_b_final[0,0]:.3e}", flush=True)

    # ---------------------------------------------------------------------------
    # Build model / bootstrap (same logic as repro_vela_shapelets.py)
    # ---------------------------------------------------------------------------
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    from gigalens_research.simtests.system import from_vela_dir
    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401 — triggers registration
    from gigalens_research.simtests.registry import get_inference_builder
    from gigalens_research.simtests.pipelines import VelaBootstrapQzStage
    from gigalens_research.inference_utils.pipeline import InferenceContext
    import gigalens.jax.simulator as sim

    data_dir = os.path.join(home, "GIGALens-Code/data")
    system_name = "vela01_cam12_rep03_a0.500_f814w"
    system = from_vela_dir(
        system_dir=os.path.join(data_dir, "vela_sim_systems", system_name),
        source_dir=os.path.join(data_dir, "vela_sources", system_name),
        system_id="vela01_cam12_rep03",
        delta_pix=0.03, num_pix=200, supersample=1,
        background_rms=0.002, exp_time=2000.0,
    )
    print(f"[D3] system loaded", flush=True)

    prob_model = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=args.n_max)

    # Mirror MCLMC_JIT exactly: build lens_sim and wrap log_prob to return scalar
    lens_sim = sim.LensSimulator(
        prob_model.model,
        prob_model.datasets[0].sim_config,
        bs=1,
    )

    def logdensity_fn(z):
        """Exactly mirrors MCLMC_JIT's log_prob: prob_model.log_prob(lens_sim, z)[0]"""
        return prob_model.log_prob(lens_sim, z)[0]

    print(f"[D3] model built n_max={args.n_max}", flush=True)

    if not args.skip_bootstrap:
        print(f"[D3] running bootstrap ...", flush=True)
        t0 = time.perf_counter()
        bootstrap_stage = VelaBootstrapQzStage(
            system=system, n_max=args.n_max,
            map_num_steps=200, map_n_samples=100,
            diag_scale=1e-8,
        )
        ctx = InferenceContext.from_prob_model(prob_model)
        stage_result = bootstrap_stage.run(ctx, {}, seed=0)
        artifacts = bootstrap_stage.derive_artifacts(stage_result.arrays)
        qz = artifacts["qz"]
        bootstrap_point = np.asarray(qz.mean())
        print(f"[D3] bootstrap done in {time.perf_counter()-t0:.1f}s, "
              f"qz.mean()[:4]={bootstrap_point[:4]}", flush=True)
    else:
        # Use run_a step-0 position as proxy for qz.mean() at n_max=25.
        # Chains are initialised from qz.sample() with diag_scale=1e-8 (std~1e-4),
        # so step-0 pos differs from qz.mean by ~1e-4 — negligible for this test.
        # run_b step-0 is used for n_max=10.
        if args.n_max == 25:
            bootstrap_point = pos_a[0, 0, :]  # (17,) numpy
        else:
            bootstrap_point = pos_b[0, 0, :]
        print(f"[D3] --skip-bootstrap: using saved step-0 position as qz.mean proxy "
              f"(n_max={args.n_max}, diff from true mean ~1e-4)", flush=True)
        print(f"[D3]   bootstrap_point[:4]={bootstrap_point[:4]}", flush=True)

    bootstrap_point = np.asarray(bootstrap_point)
    dim = bootstrap_point.shape[-1]
    assert dim == DIM, f"Expected dim=17, got {dim}"

    # ---------------------------------------------------------------------------
    # Build anchor set
    # ---------------------------------------------------------------------------
    # (i) bootstrap point
    # (ii) 4 late frozen positions from run_a  (chains 0,2,4,6, last step)
    # (iii) 4 late positions from run_b healthy (chains 0,2,4,6, last step)
    anchors_a_late = pos_a[[0, 2, 4, 6], -1, :]  # (4, 17)
    anchors_b_late = pos_b[[0, 2, 4, 6], -1, :]  # (4, 17)

    anchor_dict = {
        "bootstrap": np.asarray(bootstrap_point)[None],          # (1,17)
        "run_a_late": anchors_a_late,                             # (4,17)
        "run_b_late": anchors_b_late,                             # (4,17)
    }

    # ---------------------------------------------------------------------------
    # Build kernel (from blackjax_updated_utils.py baseline commit 71357bd)
    # ---------------------------------------------------------------------------
    from gigalens_research.inference.blackjax_updated_utils import (
        _build_kernel_shardmap,
        isokinetic_mclachlan_smart,
    )
    from blackjax.mcmc.integrators import IntegratorState

    # ---------------------------------------------------------------------------
    # Measurement function
    # ---------------------------------------------------------------------------
    L_val = float(np.sqrt(DIM))  # ≈ 4.123, same as run init

    def measure_eevpd_at_anchor(anchor_z, metric_np, eps_grid, K, seed):
        """
        For each eps in eps_grid, draw K random unit momenta, take ONE kernel step,
        record dE.  Returns arrays (n_eps, K) of energy changes.

        anchor_z:   numpy (17,)  — position in z-space
        metric_np:  numpy (17,17) — inverse mass matrix
        eps_grid:   numpy (n_eps,) — step sizes to test
        K:          int — momenta per eps
        seed:       int
        """
        import jax
        import jax.numpy as jnp
        from blackjax.util import generate_unit_vector

        # Route through numpy float64 so the active JAX default dtype wins:
        # float32 .npy anchors otherwise stay float32 under x64 and poison a
        # lax.cond branch in the kernel with mixed dtypes.
        metric_jnp = jnp.asarray(np.asarray(metric_np, dtype=np.float64))
        kernel = _build_kernel_shardmap(logdensity_fn, metric_jnp, isokinetic_mclachlan_smart)

        anchor = jnp.asarray(np.asarray(anchor_z, dtype=np.float64))
        # Compute logdensity and grad at anchor
        ld, ldg = jax.value_and_grad(logdensity_fn)(anchor)
        print(f"      logdensity={float(ld):.4g}  grad_norm={float(jnp.linalg.norm(ldg)):.4g}  "
              f"finite_ld={bool(jnp.isfinite(ld))}  finite_grad={bool(jnp.all(jnp.isfinite(ldg)))}", flush=True)

        # Use SAME base state for all eps (anchor, unit momentum varies)
        # We resample momenta from the same base key so results are comparable across eps
        rng = jax.random.PRNGKey(seed)
        # Draw K momenta (all steps from same anchor)
        momentum_keys = jax.random.split(rng, K)
        momenta = jax.vmap(lambda k: generate_unit_vector(k, anchor))(momentum_keys)  # (K, 17)

        def dE_for_eps(eps):
            """Run K steps (one per momentum) at this eps; return (K,) dE array."""
            def one_step(momentum, step_key):
                state = IntegratorState(
                    position=anchor,
                    momentum=momentum,
                    logdensity=ld,
                    logdensity_grad=ldg,
                )
                step_keys = jax.random.split(step_key, 2)
                _, info = kernel(step_keys[0], state, jnp.array(L_val), jnp.array(eps))
                # MCLMCInfoWithExtras: raw dE is in info.extras.energy_change_raw
                return info.extras.energy_change_raw  # raw, pre-zeroing

            step_keys = jax.random.split(rng, K)
            # Chunk the K-batch: a full vmap(K=64) materializes 64 copies of the
            # 351-component basis (~16 GiB at n_max=25) in one allocation and
            # OOMs on the shared GPU. lax.map over chunks bounds peak memory at
            # chunk_size simultaneous model evaluations.
            chunk_size = 8
            assert K % chunk_size == 0
            momenta_c = momenta.reshape(K // chunk_size, chunk_size, -1)
            keys_c = step_keys.reshape(K // chunk_size, chunk_size, -1)
            dEs = jax.lax.map(
                lambda mk: jax.vmap(one_step)(mk[0], mk[1]), (momenta_c, keys_c)
            )
            return dEs.reshape(K)

        # JIT compile once (outside loop)
        dE_for_eps_jit = jax.jit(dE_for_eps)

        # Warmup
        _ = dE_for_eps_jit(eps_grid[0]).block_until_ready()

        all_dE = np.zeros((len(eps_grid), K), dtype=np.float64)
        for i, eps in enumerate(eps_grid):
            dEs = np.asarray(dE_for_eps_jit(float(eps)).block_until_ready())
            all_dE[i] = dEs
            n_finite = np.sum(np.isfinite(dEs))
            n_nan = K - n_finite
            vdE_dim = float(np.nanvar(dEs) / DIM) if n_finite > 1 else float("nan")
            print(f"      eps={eps:.3e}  Var(dE)/dim={vdE_dim:.3e}  "
                  f"median|dE|={float(np.nanmedian(np.abs(dEs))):.3e}  "
                  f"NaN={n_nan}/{K}", flush=True)

        return all_dE

    # ---------------------------------------------------------------------------
    # EPS grid
    # ---------------------------------------------------------------------------
    eps_grid = np.logspace(np.log10(args.eps_lo), np.log10(args.eps_hi), args.n_eps)
    print(f"[D3] eps grid: {args.n_eps} points logspace({args.eps_lo:.0e},{args.eps_hi:.0e})", flush=True)
    print(f"[D3] eps range: {eps_grid[0]:.3e} .. {eps_grid[-1]:.3e}", flush=True)

    # ---------------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------------
    metrics = _build_metrics(imm_a_final, imm_b_final)

    # ---------------------------------------------------------------------------
    # Determine which (metric × anchor) combinations to run
    # The mandatory ones are:
    #   n_max=25: metric (i)=diag_1e-8_I  × bootstrap anchor  (both precisions)
    #   n_max=10: metric (iv)=run_b_final  × run_b_late anchors (both precisions)
    # We also run all metrics × bootstrap anchor to be thorough (within time budget)
    # ---------------------------------------------------------------------------
    if args.n_max == 25:
        # Mandatory: metric(i) × bootstrap
        # Also run metric(ii)(iii)(iv) × bootstrap for comparison
        combos = [
            ("diag_1e-8_I",  "bootstrap",  True),   # MANDATORY at n_max=25
            ("diag_1e-6_I",  "bootstrap",  True),
            ("run_a_final",  "bootstrap",  True),
            ("run_b_final",  "bootstrap",  True),
            ("diag_1e-8_I",  "run_a_late", True),   # frozen anchor points
            ("run_b_final",  "run_a_late", True),
        ]
    else:  # n_max=10
        # Mandatory: metric(iv) × run_b_late
        # Also run metric(iv) × bootstrap for comparison
        combos = [
            ("run_b_final",  "run_b_late", True),   # MANDATORY at n_max=10
            ("diag_1e-8_I",  "bootstrap",  True),
            ("run_b_final",  "bootstrap",  True),
            ("diag_1e-6_I",  "bootstrap",  True),
        ]

    # ---------------------------------------------------------------------------
    # Run all combos
    # ---------------------------------------------------------------------------
    results = {}
    t_start = time.perf_counter()

    for metric_name, anchor_name, do_run in combos:
        if not do_run:
            continue
        metric_np = metrics[metric_name]
        anchors = anchor_dict[anchor_name]  # (N_anchors, 17)

        print(f"\n[D3] === metric={metric_name}  anchor={anchor_name}  "
              f"n_anchors={anchors.shape[0]} ===", flush=True)

        combo_key = f"{metric_name}__{anchor_name}"
        all_dE_anchors = []

        for ai, anchor_z in enumerate(anchors):
            print(f"[D3]   anchor {ai+1}/{anchors.shape[0]}", flush=True)
            dE_arr = measure_eevpd_at_anchor(
                anchor_z=anchor_z,
                metric_np=metric_np,
                eps_grid=eps_grid,
                K=args.K,
                seed=args.seed + ai,
            )
            all_dE_anchors.append(dE_arr)  # (n_eps, K)

        # Stack: (n_anchors, n_eps, K)
        all_dE_anchors = np.stack(all_dE_anchors, axis=0)
        results[combo_key] = all_dE_anchors

        # Save per-combo NPZ
        fname = os.path.join(out_dir, f"dE_{combo_key}.npz")
        np.savez(fname, dE=all_dE_anchors, eps=eps_grid,
                 metric_name=metric_name, anchor_name=anchor_name,
                 dim=DIM, K=args.K)
        print(f"[D3]   saved {fname}", flush=True)

    print(f"\n[D3] all combos done in {time.perf_counter()-t_start:.1f}s", flush=True)

    # ---------------------------------------------------------------------------
    # Analysis: compute Var(dE)/dim, fit slope, find floor, crossing
    # ---------------------------------------------------------------------------
    summary = {
        "n_max": args.n_max,
        "precision": args.precision,
        "K": args.K,
        "n_eps": args.n_eps,
        "eps_lo": args.eps_lo,
        "eps_hi": args.eps_hi,
        "dim": DIM,
        "desired_energy_var": DESIRED_ENERGY_VAR,
        "target_dE_rms": TARGET_DE_RMS,
        "collapsed_eps_run_a": 1.6746e-5,
        "combos": {},
    }

    for combo_key, dE_4d in results.items():
        # dE_4d: (n_anchors, n_eps, K)
        # Collapse anchors: pool all samples
        n_anchors = dE_4d.shape[0]
        # per-eps: pool over anchors and K
        dE_pooled = dE_4d.reshape(n_anchors, len(eps_grid), -1)  # (n_anchors, n_eps, K)

        var_dim = np.zeros(len(eps_grid))
        median_abs_dE = np.zeros(len(eps_grid))
        n_nan_frac = np.zeros(len(eps_grid))

        for i in range(len(eps_grid)):
            vals = dE_pooled[:, i, :].ravel()
            finite = vals[np.isfinite(vals)]
            n_nan_frac[i] = 1.0 - len(finite) / len(vals)
            if len(finite) > 1:
                var_dim[i] = np.var(finite, ddof=1) / DIM
                median_abs_dE[i] = np.median(np.abs(finite))
            else:
                var_dim[i] = np.nan
                median_abs_dE[i] = np.nan

        # Fit log-log slope in mid-eps region (where Var(dE)/dim is finite and not at floor)
        # Use eps range 1e-4 to 1e-1 for slope fit (avoid floor and saturation)
        mask_fit = (eps_grid >= 1e-4) & (eps_grid <= 1e-1) & np.isfinite(var_dim) & (var_dim > 0)
        if mask_fit.sum() >= 3:
            slope, intercept = np.polyfit(np.log10(eps_grid[mask_fit]), np.log10(var_dim[mask_fit]), 1)
        else:
            slope, intercept = float("nan"), float("nan")

        # Floor estimate: minimum finite Var(dE)/dim over the small-eps region (eps < 1e-3)
        mask_small = (eps_grid < 1e-3) & np.isfinite(var_dim)
        floor_level = float(np.nanmin(var_dim[mask_small])) if mask_small.sum() > 0 else float("nan")
        floor_onset_eps = float(eps_grid[np.where(var_dim <= floor_level * 3)[0][0]]) if mask_small.sum() > 0 else float("nan")

        # Find eps where Var(dE)/dim crosses 5e-4 (from below, i.e., searching from large eps)
        above_target = var_dim >= DESIRED_ENERGY_VAR
        # crossing: smallest eps where var_dim >= 5e-4
        crossing_eps = float(eps_grid[np.where(above_target)[0][0]]) if above_target.any() else float("nan")

        # xi at collapsed eps (1.67e-5): interpolate
        collapsed_eps = 1.6746e-5
        xi_at_collapsed = float(np.interp(np.log10(collapsed_eps), np.log10(eps_grid), var_dim / DESIRED_ENERGY_VAR))

        # Save summary for this combo
        combo_summary = {
            "slope_mid_eps": float(slope),
            "slope_intercept": float(intercept),
            "floor_level_var_dE_per_dim": floor_level,
            "floor_onset_eps": floor_onset_eps,
            "crossing_eps_5e-4": crossing_eps,
            "xi_at_collapsed_eps": xi_at_collapsed,
            "eps_grid": eps_grid.tolist(),
            "var_dim": var_dim.tolist(),
            "median_abs_dE": median_abs_dE.tolist(),
            "nan_frac": n_nan_frac.tolist(),
        }
        summary["combos"][combo_key] = combo_summary

        print(f"\n[D3] ANALYSIS: {combo_key}", flush=True)
        print(f"       slope (log-log, eps 1e-4..1e-1): {slope:.2f}  (expected ~6.0)", flush=True)
        print(f"       floor level (Var(dE)/dim): {floor_level:.3e}", flush=True)
        print(f"       floor onset eps: {floor_onset_eps:.3e}", flush=True)
        print(f"       5e-4 crossing eps: {crossing_eps:.3e}  (run_b settled at eps≈0.79)", flush=True)
        print(f"       xi at collapsed eps={collapsed_eps:.2e}: {xi_at_collapsed:.3g}  (run_a observed xi_q50≈0.029, q90≈2.87)", flush=True)
        print(f"       floor >= 5e-4 (H4 unreachable): {floor_level >= DESIRED_ENERGY_VAR}", flush=True)

    # ---------------------------------------------------------------------------
    # Save summary JSON
    # ---------------------------------------------------------------------------
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[D3] summary written to {summary_path}", flush=True)

    # ---------------------------------------------------------------------------
    # Make plots (one per combo)
    # ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for combo_key, cs in summary["combos"].items():
            eps_arr = np.array(cs["eps_grid"])
            vd = np.array(cs["var_dim"])
            med = np.array(cs["median_abs_dE"])

            fig, ax = plt.subplots(figsize=(8, 5))
            valid = np.isfinite(vd) & (vd > 0)
            ax.loglog(eps_arr[valid], vd[valid], "b.-", label="Var(dE)/dim (measured)")
            ax.axhline(DESIRED_ENERGY_VAR, color="r", ls="--", label=f"target {DESIRED_ENERGY_VAR}")

            # Slope-6 reference line anchored at the run_b settled eps
            ref_eps = 0.79
            ref_y = np.interp(np.log10(ref_eps), np.log10(eps_arr[valid]), np.log10(vd[valid]))
            ref_y = 10 ** ref_y
            slope6_y = ref_y * (eps_arr / ref_eps) ** 6
            ax.loglog(eps_arr, slope6_y, "g--", alpha=0.6, label="slope-6 reference")

            slope_val = cs["slope_mid_eps"]
            floor_val = cs["floor_level_var_dE_per_dim"]
            ax.set_title(f"D3 EEVPD: {combo_key}\nn_max={args.n_max} {args.precision}  slope={slope_val:.2f}  floor={floor_val:.2e}")
            ax.set_xlabel("step size eps")
            ax.set_ylabel("Var(dE)/dim")
            ax.axvline(1.6746e-5, color="orange", ls=":", alpha=0.7, label="run_a collapsed eps")
            ax.axvline(0.79, color="purple", ls=":", alpha=0.7, label="run_b settled eps")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)

            figpath = os.path.join(out_dir, f"eevpd_{combo_key}.png")
            fig.savefig(figpath, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[D3] plot saved: {figpath}", flush=True)
    except Exception as e:
        print(f"[D3] WARNING: plotting failed: {e}", flush=True)

    # ---------------------------------------------------------------------------
    # Verdict (per pre-registered predictions)
    # ---------------------------------------------------------------------------
    print("\n" + "="*60, flush=True)
    print("D3 VERDICT CHECKLIST", flush=True)
    print("="*60, flush=True)
    for combo_key, cs in summary["combos"].items():
        slope = cs["slope_mid_eps"]
        floor = cs["floor_level_var_dE_per_dim"]
        crossing = cs["crossing_eps_5e-4"]
        xi_at_coll = cs["xi_at_collapsed_eps"]
        print(f"\n  {combo_key}:", flush=True)
        print(f"    slope={slope:.2f}  floor={floor:.2e}  crossing_eps={crossing:.2e}  xi@collapsed={xi_at_coll:.2g}", flush=True)
        if not np.isnan(floor) and floor >= DESIRED_ENERGY_VAR:
            print(f"    *** H4 CONFIRMED: floor {floor:.2e} >= target {DESIRED_ENERGY_VAR:.0e} — target UNREACHABLE in this precision ***", flush=True)
        elif not np.isnan(slope) and abs(slope - 6.0) > 2.0:
            print(f"    *** ANOMALOUS SLOPE {slope:.2f} — not slope-6 regime ***", flush=True)
        else:
            print(f"    slope near 6.0 and floor below target — model-side numerics OK at this combo", flush=True)
    print("="*60, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
