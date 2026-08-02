"""E2 — condition-number map of the lstsq marginalization.

Pre-registered hypothesis: the design/gram conditioning is position-dependent
and tracks the D3 floor: it is much worse at the frozen run-(a) positions than
at the bootstrap, explaining the 10^9× floor variation.  Additionally, the
normal equations needlessly SQUARE the conditioning: cond(gram) ≈ cond(A)^2.

Pre-registered predictions:
  (i)   cond(A) at frozen anchors ≥ ~10^3–10^5× cond(A) at bootstrap (gram
        cond gap ~10^6–10^10×, matching the 10^9× floor gap).
  (ii)  cond(gram) ≈ cond(A)^2 within a factor ~10 everywhere.
  (iii) n_max=10 conds are several orders below n_max=25 at the same anchor.

Falsifiers:
  - cond ~constant across anchors → position dependence lives elsewhere.
  - cond(gram) ≪ cond(A)^2 → the squaring story is wrong.

Usage (CPU login node, float64 only):
  python e2_cond_map.py --n-max 25 --out-dir diagnosis_2026-06/e2
  python e2_cond_map.py --n-max 10 --out-dir diagnosis_2026-06/e2

  # Or compute both at once (they run sequentially):
  python e2_cond_map.py --n-max 25 --n-max-also 10 --out-dir diagnosis_2026-06/e2

NOTE: ALL computation is in float64 (jax_enable_x64=True enabled before any
jax import).  The float32 gram path is explicitly also measured (form gram in
float32, evaluate eigenspectrum in float64) to separate formation error from
solve error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — mirrors d3_eevpd_vs_eps.py
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
    p.add_argument("--n-max", type=int, required=True,
                   help="Primary shapelet order (10 or 25)")
    p.add_argument("--n-max-also", type=int, default=None,
                   help="Also run a second n_max in the same invocation")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: diagnosis_2026-06/e2 relative to script)")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip bootstrap (use run_a step-0 proxy)")
    p.add_argument("--n-prior-draws", type=int, default=4,
                   help="Number of prior-draw far-field control anchors")
    return p.parse_args()


# ---------------------------------------------------------------------------
# D3 floor values (pre-measured; used for the money-plot correlation)
# ---------------------------------------------------------------------------
# Key: (n_max, combo_key) — combo_key = "{metric}__{anchor_class}"
# Value: floor_level_var_dE_per_dim from d3 summary.json (float32 run).
# "bootstrap" combos have floor=0 in f32 (clamped to float32 ulp level);
# we represent those as the ulp noise estimate from REPORT.md: ~7e-6.
# The "run_a_late" combos have the catastrophic floor of ~1.4e5.
D3_FLOORS_F32 = {
    # n_max=25
    (25, "diag_1e-8_I__bootstrap"):  6e-6,    # floor≈0 in json = f32 ulp level
    (25, "diag_1e-6_I__bootstrap"):  6e-6,
    (25, "run_a_final__bootstrap"):  6e-6,
    (25, "run_b_final__bootstrap"):  6e-6,
    (25, "diag_1e-8_I__run_a_late"): 1.42e5,
    (25, "run_b_final__run_a_late"): 1.41e5,
    # n_max=10
    (10, "run_b_final__run_b_late"): 2.70e-6,
    (10, "diag_1e-8_I__bootstrap"):  6e-6,
    (10, "run_b_final__bootstrap"):  6e-6,
    (10, "diag_1e-6_I__bootstrap"):  6e-6,
}


# ---------------------------------------------------------------------------
# Build the prob model (mirrors d3_eevpd_vs_eps.py)
# ---------------------------------------------------------------------------

def build_model(n_max, home=home):
    from gigalens_research.simtests.system import from_vela_dir
    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401
    from gigalens_research.simtests.registry import get_inference_builder
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
    prob_model = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=n_max)
    lens_sim = sim.LensSimulator(prob_model.model, prob_model.datasets[0].sim_config, bs=1)
    return prob_model, lens_sim, system


# ---------------------------------------------------------------------------
# Extract design matrix A and gram = A^T A (weighted)
# ---------------------------------------------------------------------------

def get_design_matrix(prob_model, lens_sim, anchor_z, jnp, jax):
    """
    Form the weighted design matrix A  (shape [n_pixels, n_components]).

    The `lstsq_simulate` code computes:
        X = ret * W   (n_pixels × n_components, weighted basis images)
        gram = X^T @ X

    We replicate this in float64, returning A (= X) and the observed image /
    err_map as arrays.

    Returns:
        A_f64  : np.ndarray, shape (n_pix, n_comp), float64
        A_f32  : np.ndarray, same shape, float32  (formed under float32 JAX ops)
        gram_f64 : np.ndarray, (n_comp, n_comp), float64
        gram_f32_as_f64 : np.ndarray, the f32 gram re-read as f64
        obs_image : np.ndarray
        err_map   : np.ndarray
        coeffs    : np.ndarray, shapelet amplitudes solved at this anchor
        sv_A      : np.ndarray, singular values (from lstsq)
        nan_pre_clamp : bool, True if any non-finite values found before nan_to_num
    """
    from gigalens.jax.simulator import _shared_kernel_component_conv

    # Decode z → physical params.
    # log_prob does: z = list(z.T); x = bij.forward(z)
    # For a single anchor (shape (17,)), z.T is still (17,), list() gives 17 scalars.
    # The bijector expects a list of per-parameter arrays (one per latent dim).
    # For a single-chain (bs=1) we pass z as shape (1,17) and transpose to
    # get a list of 17 arrays of shape (1,).
    z_jax = jnp.asarray(np.asarray(anchor_z, dtype=np.float64))   # (17,)
    z_list = list(z_jax[None, :].T)   # list of 17 arrays of shape (1,)
    x = prob_model.bij.forward(z_list)   # x = [lens_params, lens_light_params, src_params]

    lens_params = x[0]
    lens_light_params = x[1] if len(lens_sim.phys_model.lens_light) > 0 else []
    source_light_params = x[2] if len(lens_sim.phys_model.lens_light) > 0 else x[1]

    beta_x, beta_y = lens_sim._beta(lens_params)

    # Build basis (mirrors lstsq_simulate build_basis block)
    img_stack = jnp.zeros((0, *lens_sim.img_X.shape))
    for lightModel, p in zip(lens_sim.phys_model.lens_light, lens_light_params):
        img_stack = jnp.concatenate(
            (img_stack, lightModel.light(lens_sim.img_X, lens_sim.img_Y, **p)), axis=0)
    for lightModel, p in zip(lens_sim.phys_model.source_light, source_light_params):
        img_stack = jnp.concatenate(
            (img_stack, lightModel.light(beta_x, beta_y, **p)), axis=0)

    # Check for non-finite pre-clamp
    nan_pre_clamp = bool(jnp.any(~jnp.isfinite(img_stack)))

    img_stack = jnp.nan_to_num(img_stack)
    img_stack = jnp.transpose(img_stack, (3, 0, 1, 2))  # bs, n_comp, h, w

    # conv/pool (supersample=1 for this system so pool is identity)
    ret = (
        _shared_kernel_component_conv(img_stack, lens_sim.flat_kernel)
        if lens_sim.flat_kernel is not None
        else img_stack
    )
    if lens_sim.supersample != 1:
        from objax.functional import average_pool_2d
        ret = average_pool_2d(ret, size=(lens_sim.supersample, lens_sim.supersample), padding="SAME")
    ret = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, n_comp

    # Build error map (stored on prob_model as .err_map)
    err_map = jnp.asarray(np.asarray(prob_model.err_map, dtype=np.float64))
    obs = prob_model.observed_image

    # Weighted: W = 1/err_map
    W = (1.0 / err_map)[..., jnp.newaxis]  # (h, w, 1)
    ret_sq = ret[0]  # (h, w, n_comp)  — batch size 1

    # Weighted basis: X = ret * W   (this is the design matrix A)
    X = ret_sq * W   # (h, w, n_comp)

    n_h, n_w, n_comp = X.shape
    n_pix = n_h * n_w

    # A (float64)
    A_f64 = np.asarray(X.reshape(n_pix, n_comp), dtype=np.float64)

    # gram_f64 = A^T A
    gram_f64 = A_f64.T @ A_f64  # (n_comp, n_comp)

    # Also form gram in float32 to see what the library actually computes
    A_f32 = A_f64.astype(np.float32)
    gram_f32 = (A_f32.T @ A_f32).astype(np.float64)  # store as f64 to compare

    # Solve for coefficients using lstsq (float64; for amplitude diagnostics only)
    Y_weighted = np.asarray(
        (jnp.asarray(np.asarray(obs, dtype=np.float64)) * jnp.squeeze(W)).reshape(-1),
        dtype=np.float64
    )
    coeffs, _res, _rank, sv_A = np.linalg.lstsq(A_f64, Y_weighted, rcond=None)

    return (A_f64, A_f32, gram_f64, gram_f32,
            np.asarray(obs), np.asarray(err_map),
            coeffs.flatten(), sv_A, nan_pre_clamp)


# ---------------------------------------------------------------------------
# Condition number analysis
# ---------------------------------------------------------------------------

def analyze_matrix(A_f64, A_f32, gram_f64, gram_f32, label, coeffs, sv_A):
    """
    Compute and return a dict of condition-related metrics.
    All linear-algebra done with numpy (float64).
    """
    result = {}

    # --- Design matrix A (float64 SVD) ---
    # sv_A already computed by lstsq, but let's do full SVD for spectrum
    t0 = time.perf_counter()
    # Use scipy for full SVD (numpy may be slower)
    # Actually use numpy — no scipy available guaranteed
    U, sv_full, Vt = np.linalg.svd(A_f64, full_matrices=False)
    result["svd_time_s"] = time.perf_counter() - t0

    result["cond_A_f64"] = float(sv_full[0] / sv_full[-1]) if sv_full[-1] > 0 else float("inf")
    result["cond_A_sq"] = result["cond_A_f64"] ** 2
    result["sv_max"] = float(sv_full[0])
    result["sv_min"] = float(sv_full[-1])
    result["sv_spectrum"] = sv_full.tolist()  # full spectrum

    # --- Gram (float64) eigenspectrum ---
    # gram_f64 = A^T A; eigenvalues should be sv^2
    ev_gram_f64 = np.linalg.eigvalsh(gram_f64)  # ascending
    ev_gram_f64_sorted = np.sort(ev_gram_f64)[::-1]  # descending
    result["gram_f64_ev_spectrum"] = ev_gram_f64_sorted.tolist()
    pos_ev = ev_gram_f64_sorted[ev_gram_f64_sorted > 0]
    if len(pos_ev) > 0:
        result["cond_gram_f64"] = float(pos_ev[0] / pos_ev[-1])
    else:
        result["cond_gram_f64"] = float("inf")
    result["gram_f64_n_negative_ev"] = int(np.sum(ev_gram_f64 < 0))
    result["gram_f64_n_zero_ev"] = int(np.sum(np.abs(ev_gram_f64) < 1e-30 * ev_gram_f64_sorted[0]))

    # --- Gram (float32, evaluated as float64) ---
    ev_gram_f32 = np.linalg.eigvalsh(gram_f32)  # ascending, but computed in f64
    ev_gram_f32_sorted = np.sort(ev_gram_f32)[::-1]
    result["gram_f32_ev_spectrum"] = ev_gram_f32_sorted.tolist()
    pos_ev_f32 = ev_gram_f32_sorted[ev_gram_f32_sorted > 0]
    if len(pos_ev_f32) > 0:
        result["cond_gram_f32"] = float(pos_ev_f32[0] / pos_ev_f32[-1])
    else:
        result["cond_gram_f32"] = float("inf")
    result["gram_f32_n_negative_ev"] = int(np.sum(ev_gram_f32 < 0))
    result["gram_f32_n_zero_ev"] = int(np.sum(np.abs(ev_gram_f32) < 1e-30 * ev_gram_f32_sorted[0] if len(ev_gram_f32_sorted) > 0 else False))

    # --- Amplitude diagnostics ---
    result["amp_norm"] = float(np.linalg.norm(coeffs))
    result["amp_max_abs"] = float(np.max(np.abs(coeffs)))
    result["amp_min_abs"] = float(np.min(np.abs(coeffs)))

    # --- Squaring ratio ---
    result["ratio_gram_f64_over_cond_A_sq"] = (result["cond_gram_f64"] / result["cond_A_sq"]
                                                 if result["cond_A_sq"] > 0 else float("nan"))
    result["ratio_gram_f32_over_gram_f64"] = (result["cond_gram_f32"] / result["cond_gram_f64"]
                                                if result["cond_gram_f64"] > 0 else float("nan"))

    return result


# ---------------------------------------------------------------------------
# Run one n_max
# ---------------------------------------------------------------------------

def run_one_n_max(n_max, args, base_out, jnp, jax):
    print(f"\n{'='*70}", flush=True)
    print(f"E2: n_max={n_max}", flush=True)
    print(f"{'='*70}", flush=True)

    out_dir = os.path.join(base_out, f"nmax{n_max}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "e2_run.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"E2 n_max={n_max}  start: {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    # --- Load run_a / run_b positions ---
    diag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnosis_2026-06")
    run_a_dir = os.path.join(diag_dir, "run_a_nmax25_ds1e-8_nb2000")
    run_b_dir = os.path.join(diag_dir, "run_b_nmax10_ds1e-8_nb2000")

    pos_a = np.load(os.path.join(run_a_dir, "hist_position.npy"))  # (8,2500,17)
    pos_b = np.load(os.path.join(run_b_dir, "hist_position.npy"))

    # Bootstrap proxy (step-0, chain 0 of the appropriate run)
    if n_max == 25:
        bootstrap_point = pos_a[0, 0, :]
    else:
        bootstrap_point = pos_b[0, 0, :]

    # Frozen run_a positions (all 8 chains, last step)
    anchors_a_late = pos_a[:, -1, :]  # (8, 17)

    # Healthy run_b positions (all 8 chains, last step) — as controls
    anchors_b_late = pos_b[:, -1, :]  # (8, 17)

    # Prior-draw far-field controls (sample from standard normal — z-space)
    rng = np.random.default_rng(12345)
    prior_draws = rng.standard_normal((args.n_prior_draws, 17))

    anchor_sets = {
        "bootstrap": bootstrap_point[None],   # (1, 17)
        "run_a_late": anchors_a_late,          # (8, 17)
        "run_b_late": anchors_b_late,          # (8, 17)
        "prior_draws": prior_draws,            # (n_prior_draws, 17)
    }

    log(f"[E2] n_max={n_max}: loading model...")
    t0 = time.perf_counter()
    prob_model, lens_sim, system = build_model(n_max)
    log(f"[E2] model loaded in {time.perf_counter()-t0:.1f}s")

    all_results = {}   # anchor_set_name -> list of per-anchor dicts
    all_sv_spectra = {}  # anchor_set_name -> list of sv arrays
    all_gram_ev_f64 = {}
    all_gram_ev_f32 = {}

    for set_name, anchors in anchor_sets.items():
        log(f"\n[E2] ---- anchor set: {set_name} (n_anchors={anchors.shape[0]}) ----")
        set_results = []
        sv_list = []
        gram_ev_f64_list = []
        gram_ev_f32_list = []

        for ai, anchor_z in enumerate(anchors):
            log(f"[E2]   anchor {ai+1}/{anchors.shape[0]}")
            t_anc = time.perf_counter()

            try:
                (A_f64, A_f32, gram_f64, gram_f32,
                 obs, err_map, coeffs, sv_A, nan_pre_clamp) = get_design_matrix(
                    prob_model, lens_sim, anchor_z, jnp, jax)
            except Exception as e:
                log(f"[E2]   ERROR building design matrix: {e}")
                set_results.append({"error": str(e)})
                sv_list.append(None)
                gram_ev_f64_list.append(None)
                gram_ev_f32_list.append(None)
                continue

            log(f"[E2]   A shape: {A_f64.shape}  nan_pre_clamp={nan_pre_clamp}")

            try:
                metrics = analyze_matrix(A_f64, A_f32, gram_f64, gram_f32,
                                         f"{set_name}[{ai}]", coeffs, sv_A)
            except Exception as e:
                log(f"[E2]   ERROR in analyze_matrix: {e}")
                set_results.append({"error": str(e)})
                continue

            metrics["anchor_z"] = anchor_z.tolist()
            metrics["nan_pre_clamp"] = nan_pre_clamp
            metrics["n_max"] = n_max
            metrics["anchor_set"] = set_name
            metrics["anchor_idx"] = ai

            dt = time.perf_counter() - t_anc
            metrics["wall_time_s"] = dt

            log(f"[E2]   cond(A)={metrics['cond_A_f64']:.3e}  "
                f"cond(A)^2={metrics['cond_A_sq']:.3e}  "
                f"cond(gram_f64)={metrics['cond_gram_f64']:.3e}  "
                f"cond(gram_f32)={metrics['cond_gram_f32']:.3e}  "
                f"gram_f32_neg_ev={metrics['gram_f32_n_negative_ev']}")
            log(f"[E2]   sv_max={metrics['sv_max']:.3e}  sv_min={metrics['sv_min']:.3e}")
            log(f"[E2]   amp_norm={metrics['amp_norm']:.3e}  amp_max_abs={metrics['amp_max_abs']:.3e}")
            log(f"[E2]   ratio cond_gram_f64/cond(A)^2 = {metrics['ratio_gram_f64_over_cond_A_sq']:.3e}")
            log(f"[E2]   ratio cond_gram_f32/cond_gram_f64 = {metrics['ratio_gram_f32_over_gram_f64']:.3e}")
            log(f"[E2]   done in {dt:.1f}s")

            set_results.append(metrics)
            sv_list.append(np.array(metrics["sv_spectrum"]))
            gram_ev_f64_list.append(np.array(metrics["gram_f64_ev_spectrum"]))
            gram_ev_f32_list.append(np.array(metrics["gram_f32_ev_spectrum"]))

        all_results[set_name] = set_results
        all_sv_spectra[set_name] = sv_list
        all_gram_ev_f64[set_name] = gram_ev_f64_list
        all_gram_ev_f32[set_name] = gram_ev_f32_list

    # --- Save raw spectra as NPZ ---
    npz_path = os.path.join(out_dir, "spectra.npz")
    save_dict = {}
    for set_name, sv_list in all_sv_spectra.items():
        for ai, sv in enumerate(sv_list):
            if sv is not None:
                save_dict[f"sv_{set_name}_{ai}"] = sv
    for set_name, ev_list in all_gram_ev_f64.items():
        for ai, ev in enumerate(ev_list):
            if ev is not None:
                save_dict[f"gram_ev_f64_{set_name}_{ai}"] = ev
    for set_name, ev_list in all_gram_ev_f32.items():
        for ai, ev in enumerate(ev_list):
            if ev is not None:
                save_dict[f"gram_ev_f32_{set_name}_{ai}"] = ev
    np.savez(npz_path, **save_dict)
    log(f"\n[E2] spectra saved to {npz_path}")

    # --- Build summary dict ---
    summary = {
        "n_max": n_max,
        "float": "float64",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "anchor_sets": {},
    }
    for set_name, results in all_results.items():
        anchor_summaries = []
        for r in results:
            if "error" in r:
                anchor_summaries.append(r)
                continue
            # Drop full spectra from summary JSON (stored in npz)
            s = {k: v for k, v in r.items()
                 if k not in ("sv_spectrum", "gram_f64_ev_spectrum",
                              "gram_f32_ev_spectrum", "anchor_z")}
            anchor_summaries.append(s)
        summary["anchor_sets"][set_name] = anchor_summaries

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        # Convert any non-JSON-serializable types
        def _fix(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")
        json.dump(summary, f, indent=2, default=_fix)
    log(f"[E2] summary saved to {summary_path}")

    log_f.close()
    return all_results, all_sv_spectra, all_gram_ev_f64, all_gram_ev_f32, summary, out_dir


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(results_by_nmax, out_dir_by_nmax, base_out):
    """
    Produce three required plots:
      1. Singular-value spectra (one line per anchor, anchor class colour-coded, n_max line style).
      2. Money plot: D3 floor (x) vs cond(gram) (y), log-log scatter.
      3. cond(A)^2 vs cond(gram) parity plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[E2] matplotlib not available — skipping plots", flush=True)
        return

    # Colour scheme per anchor set
    SET_COLORS = {
        "bootstrap":   "#1f77b4",   # blue
        "run_a_late":  "#d62728",   # red (frozen/bad)
        "run_b_late":  "#2ca02c",   # green (healthy)
        "prior_draws": "#9467bd",   # purple
    }
    SET_LABELS = {
        "bootstrap":   "bootstrap",
        "run_a_late":  "frozen run-a (collapse)",
        "run_b_late":  "run-b late (healthy)",
        "prior_draws": "prior draws (far-field)",
    }
    NMAX_LS = {10: "--", 25: "-"}

    # =========================================================
    # Plot 1: Singular-value spectra
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for n_max, (all_results, all_sv_spectra, _, _, _, _) in results_by_nmax.items():
        ls = NMAX_LS.get(n_max, "-")
        for set_name, sv_list in all_sv_spectra.items():
            color = SET_COLORS.get(set_name, "gray")
            first = True
            for ai, sv in enumerate(sv_list):
                if sv is None or len(sv) == 0:
                    continue
                label = (f"n_max={n_max} {SET_LABELS.get(set_name, set_name)}"
                         if first else None)
                ax1.semilogy(np.arange(1, len(sv)+1), sv,
                             ls=ls, color=color, alpha=0.6, lw=1.2,
                             label=label)
                first = False

    ax1.set_xlabel("Singular value index (descending)")
    ax1.set_ylabel("Singular value (float64)")
    ax1.set_title("E2: Design matrix A singular-value spectra\n"
                  "(solid=n_max=25, dashed=n_max=10; colour=anchor class)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, which="both", alpha=0.3)
    plot1_path = os.path.join(base_out, "e2_sv_spectra.png")
    fig1.savefig(plot1_path, dpi=130, bbox_inches="tight")
    plt.close(fig1)
    print(f"[E2] plot saved: {plot1_path}", flush=True)

    # =========================================================
    # Plot 2: Money plot — D3 floor vs cond(gram)
    # We pair each (n_max, anchor_class) to the D3 floor entry.
    # anchor classes: bootstrap → "bootstrap"; run_a_late → "run_a_late"
    # For run_b_late and prior_draws we don't have D3 floors; skip.
    # We use cond(gram_f32) as the relevant quantity (library runs in f32).
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Collect points
    money_points = []  # (d3_floor, cond_gram_f32, cond_gram_f64, label, color)

    for n_max, (all_results, _, _, _, _, _) in results_by_nmax.items():
        for set_name, results in all_results.items():
            # Map set_name → D3 combo anchor name
            d3_anchor_name = {
                "bootstrap": "bootstrap",
                "run_a_late": "run_a_late",
                "run_b_late": "run_b_late",
            }.get(set_name, None)
            if d3_anchor_name is None:
                continue

            # Find D3 floor: use the first matching combo key
            matching_floors = []
            for k, v in D3_FLOORS_F32.items():
                if k[0] == n_max and k[1].endswith("__" + d3_anchor_name):
                    matching_floors.append(v)
            if not matching_floors:
                continue
            d3_floor = matching_floors[0]  # same floor for all metrics at that anchor

            color = SET_COLORS.get(set_name, "gray")
            for ai, r in enumerate(results):
                if "error" in r:
                    continue
                cg_f32 = r.get("cond_gram_f32", None)
                cg_f64 = r.get("cond_gram_f64", None)
                if cg_f32 is None or not np.isfinite(cg_f32):
                    cg_f32 = cg_f64  # fallback
                if cg_f32 is None or not np.isfinite(cg_f32):
                    continue
                label = f"n{n_max} {set_name}[{ai}]"
                money_points.append((d3_floor, cg_f32, cg_f64, label, color, n_max, set_name))

    if money_points:
        x = np.array([p[0] for p in money_points])
        y = np.array([p[1] for p in money_points])
        colors = [p[4] for p in money_points]

        ax2.scatter(x, y, c=colors, s=80, zorder=5, edgecolors="k", lw=0.5)

        # Add point labels
        for d3_floor, cg_f32, cg_f64, label, color, n_max, set_name in money_points:
            ax2.annotate(label, (d3_floor, cg_f32), fontsize=6, alpha=0.7,
                         xytext=(5, 2), textcoords="offset points")

        # Reference: if cond(gram) ∝ D3_floor, draw slope-1 line
        x_range = np.logspace(np.log10(min(x)*0.5), np.log10(max(x)*2), 100)
        # Anchor reference line at median
        if len(x) > 0:
            ref_x = np.median(x)
            ref_y = np.median(y)
            slope1_line = ref_y * (x_range / ref_x)
            ax2.loglog(x_range, slope1_line, "k--", alpha=0.4, label="slope-1 ref")

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("D3 measured floor: Var(dE)/dim (float32, target=5e-4)")
        ax2.set_ylabel("cond(gram, f32) — library gram condition")
        ax2.set_title("E2 MONEY PLOT: D3 noise floor vs gram condition number\n"
                      "Red=frozen run-a (collapse); Blue=bootstrap; Green=run-b healthy")
        ax2.axvline(5e-4, color="r", ls=":", alpha=0.5, label="target 5e-4")
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No paired D3/E2 points found", transform=ax2.transAxes,
                 ha="center", va="center")

    plot2_path = os.path.join(base_out, "e2_money_plot.png")
    fig2.savefig(plot2_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"[E2] plot saved: {plot2_path}", flush=True)

    # =========================================================
    # Plot 3: cond(A)^2 vs cond(gram) parity
    # =========================================================
    fig3, ax3 = plt.subplots(figsize=(7, 6))

    all_cond_A_sq = []
    all_cond_gram_f64 = []
    all_cond_gram_f32 = []
    all_labels = []
    all_colors = []

    for n_max, (all_results, _, _, _, _, _) in results_by_nmax.items():
        for set_name, results in all_results.items():
            color = SET_COLORS.get(set_name, "gray")
            for ai, r in enumerate(results):
                if "error" in r:
                    continue
                ca2 = r.get("cond_A_sq", None)
                cg64 = r.get("cond_gram_f64", None)
                cg32 = r.get("cond_gram_f32", None)
                if ca2 is None or not np.isfinite(ca2):
                    continue
                if cg64 is None or not np.isfinite(cg64):
                    continue
                all_cond_A_sq.append(ca2)
                all_cond_gram_f64.append(cg64)
                all_cond_gram_f32.append(cg32 if (cg32 and np.isfinite(cg32)) else cg64)
                all_labels.append(f"n{n_max} {set_name}[{ai}]")
                all_colors.append(color)

    if all_cond_A_sq:
        xs = np.array(all_cond_A_sq)
        ys64 = np.array(all_cond_gram_f64)
        ys32 = np.array(all_cond_gram_f32)

        ax3.scatter(xs, ys64, c=all_colors, marker="o", s=80, zorder=5,
                    edgecolors="k", lw=0.5, label="cond(gram,f64)")
        ax3.scatter(xs, ys32, c=all_colors, marker="^", s=60, zorder=4,
                    edgecolors="k", lw=0.5, alpha=0.6, label="cond(gram,f32)")

        # y=x parity line
        all_x = np.concatenate([xs, xs])
        all_y = np.concatenate([ys64, ys32])
        mn = min(min(all_x), min(all_y))
        mx = max(max(all_x), max(all_y))
        parity = np.logspace(np.log10(mn*0.5), np.log10(mx*2), 100)
        ax3.loglog(parity, parity, "k--", alpha=0.5, label="y=x (squaring exact)")
        ax3.loglog(parity, parity * 10, "k:", alpha=0.3, label="y=10x")
        ax3.loglog(parity, parity / 10, "k:", alpha=0.3)

        for label, xs_i, ys_i, col in zip(all_labels, xs, ys64, all_colors):
            ax3.annotate(label, (xs_i, ys_i), fontsize=6, alpha=0.7,
                         xytext=(5, 2), textcoords="offset points")

        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel("cond(A)^2")
        ax3.set_ylabel("cond(gram)")
        ax3.set_title("E2: cond(A)^2 vs cond(gram) — parity plot\n"
                      "circles=f64 gram; triangles=f32 gram; dashed=y=x")
        ax3.legend(fontsize=8)
        ax3.grid(True, which="both", alpha=0.3)

        # Add legend for anchor classes manually
        from matplotlib.lines import Line2D
        legend_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                              markersize=8, label=SET_LABELS.get(s, s))
                      for s, c in SET_COLORS.items()]
        ax3.legend(handles=legend_els + [
            Line2D([0], [0], ls='--', color='k', alpha=0.5, label='y=x'),
            Line2D([0], [0], ls=':', color='k', alpha=0.3, label='y=10x or y=x/10'),
        ], fontsize=7, loc="upper left")

    plot3_path = os.path.join(base_out, "e2_parity_cond_sq.png")
    fig3.savefig(plot3_path, dpi=130, bbox_inches="tight")
    plt.close(fig3)
    print(f"[E2] plot saved: {plot3_path}", flush=True)

    return plot1_path, plot2_path, plot3_path


# ---------------------------------------------------------------------------
# Verdict: compare against pre-registered predictions
# ---------------------------------------------------------------------------

def print_verdict(results_by_nmax):
    print("\n" + "="*70, flush=True)
    print("E2 VERDICT vs PRE-REGISTERED PREDICTIONS", flush=True)
    print("="*70, flush=True)

    # Gather cond(A) per anchor set and n_max
    cond_A_by = {}  # (n_max, set_name) -> list of cond(A)
    cond_gram64_by = {}
    cond_gram32_by = {}

    for n_max, (all_results, _, _, _, _, _) in results_by_nmax.items():
        for set_name, results in all_results.items():
            key = (n_max, set_name)
            cond_A_by[key] = []
            cond_gram64_by[key] = []
            cond_gram32_by[key] = []
            for r in results:
                if "error" in r:
                    continue
                cond_A_by[key].append(r.get("cond_A_f64", float("nan")))
                cond_gram64_by[key].append(r.get("cond_gram_f64", float("nan")))
                cond_gram32_by[key].append(r.get("cond_gram_f32", float("nan")))

    def median_safe(lst):
        vals = [v for v in lst if v and np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    # Prediction (i): cond(A) at frozen anchors >> bootstrap
    for n_max in sorted(results_by_nmax.keys()):
        cA_boot = median_safe(cond_A_by.get((n_max, "bootstrap"), []))
        cA_late = median_safe(cond_A_by.get((n_max, "run_a_late"), []))
        cG_boot = median_safe(cond_gram64_by.get((n_max, "bootstrap"), []))
        cG_late = median_safe(cond_gram64_by.get((n_max, "run_a_late"), []))
        cG32_boot = median_safe(cond_gram32_by.get((n_max, "bootstrap"), []))
        cG32_late = median_safe(cond_gram32_by.get((n_max, "run_a_late"), []))

        print(f"\nn_max={n_max}:", flush=True)
        print(f"  cond(A): bootstrap={cA_boot:.3e}  run_a_late(median)={cA_late:.3e}", flush=True)
        if np.isfinite(cA_boot) and np.isfinite(cA_late) and cA_boot > 0:
            ratio_A = cA_late / cA_boot
            print(f"  ratio cond(A)[frozen/bootstrap] = {ratio_A:.2e}", flush=True)
        print(f"  cond(gram_f64): bootstrap={cG_boot:.3e}  run_a_late(median)={cG_late:.3e}", flush=True)
        print(f"  cond(gram_f32): bootstrap={cG32_boot:.3e}  run_a_late(median)={cG32_late:.3e}", flush=True)

        print(f"\n  PREDICTION (i): cond(A) at frozen >= 10^3–10^5 × bootstrap", flush=True)
        if np.isfinite(cA_boot) and np.isfinite(cA_late) and cA_boot > 0:
            if ratio_A >= 1e3:
                print(f"  → CONFIRMED (ratio={ratio_A:.2e} >= 1e3)", flush=True)
            else:
                print(f"  → FAILED (ratio={ratio_A:.2e} < 1e3; "
                      "position dependence does NOT explain D3 floor gap)", flush=True)
        else:
            print(f"  → INCONCLUSIVE (nan/inf)", flush=True)

        print(f"\n  PREDICTION (ii): cond(gram_f64) ≈ cond(A)^2 within ~10×", flush=True)
        for set_name in ["bootstrap", "run_a_late"]:
            all_results_nmax = results_by_nmax[n_max][0]  # tuple index 0 = all_results dict
            rs = all_results_nmax.get(set_name, [])
            for r in rs:
                if "error" in r:
                    continue
                ratio = r.get("ratio_gram_f64_over_cond_A_sq", float("nan"))
                cA = r.get("cond_A_f64", float("nan"))
                cA2 = r.get("cond_A_sq", float("nan"))
                cG = r.get("cond_gram_f64", float("nan"))
                print(f"    {set_name}[{r.get('anchor_idx',0)}]: "
                      f"cond(A)={cA:.2e}  cond(A)^2={cA2:.2e}  "
                      f"cond(gram_f64)={cG:.2e}  ratio={ratio:.2e}", flush=True)
                if np.isfinite(ratio):
                    if 0.1 <= ratio <= 10:
                        print(f"      → CONFIRMED (ratio within 10×)", flush=True)
                    else:
                        print(f"      → FAILED (ratio={ratio:.2e} outside 10×)", flush=True)

    print(f"\n  PREDICTION (iii): n_max=10 conds several orders below n_max=25", flush=True)
    for set_name in ["bootstrap"]:
        ca25 = median_safe(cond_A_by.get((25, set_name), []))
        ca10 = median_safe(cond_A_by.get((10, set_name), []))
        if np.isfinite(ca25) and np.isfinite(ca10) and ca10 > 0:
            ratio = ca25 / ca10
            print(f"  cond(A): n_max=25={ca25:.3e}  n_max=10={ca10:.3e}  ratio={ratio:.2e}", flush=True)
            if ratio >= 10:
                print(f"  → CONFIRMED (ratio={ratio:.2e} >= 10)", flush=True)
            else:
                print(f"  → FAILED (ratio={ratio:.2e} < 10)", flush=True)
        else:
            print(f"  → INCONCLUSIVE", flush=True)

    print("\n" + "="*70, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse()

    # Float64 MUST be enabled before any jax import
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    print(f"[E2] JAX {jax.__version__}  float64 enabled  devices: {jax.devices()}", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_out = args.out_dir or os.path.join(script_dir, "diagnosis_2026-06", "e2")
    os.makedirs(base_out, exist_ok=True)
    print(f"[E2] output base: {base_out}", flush=True)

    n_maxes = [args.n_max]
    if args.n_max_also is not None and args.n_max_also != args.n_max:
        n_maxes.append(args.n_max_also)

    results_by_nmax = {}  # n_max -> (all_results, all_sv, all_gram64, all_gram32, summary, out_dir)

    for n_max in n_maxes:
        ret = run_one_n_max(n_max, args, base_out, jnp, jax)
        results_by_nmax[n_max] = ret

    print("\n[E2] making plots...", flush=True)
    make_plots(results_by_nmax, {nm: ret[5] for nm, ret in results_by_nmax.items()}, base_out)

    print_verdict(results_by_nmax)

    print(f"\n[E2] DONE. All artifacts under {base_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
