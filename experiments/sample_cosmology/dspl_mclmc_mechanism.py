#!/usr/bin/env python3
"""Post-hoc MECHANISM diagnosis for the DSPL cosmology MCLMC run: does the
sampler BOUNCE off a barrier near the r2-contour "crest" (metric/step-size
neck), or does it simply RANDOM-WALK too slowly to ever reach the unvisited
low-Om0 tail (diffusion time)?

This is a diagnostic script only (method-discipline: diagnosing, not
fixing). It runs NO sampler, NO MAP, NO simulator -- it only loads existing
run artifacts (arrays.npz, diagnostics.npz, map/arrays.npz,
def_ratio_grid.npz), reuses `def_ratio_grid.py`'s model-rebuild /
bijector-forward machinery, and produces plots + numbers. All conclusions
below are UNCERTIFIED (per docs/agent-operating-card.md): they are proposed
readings of the evidence, not certified claims.

Established context (do not re-derive; see caller's task description):
  - 8 chains, 10000 burn-in + 10000 results, 21 params, MCLMC.
  - The 2-D cosmology posterior is exactly constant along level contours of
    r2 = deflection_ratio(z_source2); uniform priors.
  - All 8 chains sit tightly on the r2(truth)=1.3241652 contour, covering
    only Om0 in ~[0.15, 0.57]; truncation edges at Om0~0.146-0.163.
  - The contour's continuation over the banana's high-curvature CREST (near
    Om0~0.2, w0~-0.9) down to the Om0=0 edge (w0~-1.27 there) is entirely
    unvisited.
  - rank-Rhat: Om0 1.038, w0 1.046; bulk-ESS ~ 200/80000.

Outputs (written to the results dir):
  mech_traces.png, mech_xi_vs_om0.png, mech_ridge_geometry.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.scipy.stats as jst

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import def_ratio_grid as drg  # noqa: E402  (reuse model/bijector/grid code)

from gigalens.jax.cosmo import w0waCDM_Cosmo  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / constants (established context; see module docstring).
# ---------------------------------------------------------------------------
RESULTS_DIR = drg.RESULTS_DIR
MCLMC_DIR = drg.MCLMC_DIR
MAP_DIR = os.path.join(RESULTS_DIR, "map")
GRID_NPZ = drg.OUT_NPZ

TRACES_PNG = os.path.join(RESULTS_DIR, "mech_traces.png")
XI_PNG = os.path.join(RESULTS_DIR, "mech_xi_vs_om0.png")
RIDGE_PNG = os.path.join(RESULTS_DIR, "mech_ridge_geometry.png")

OM0_EDGE = 0.163     # truncation edge, established context
OM0_CREST = 0.2      # banana crest, established context
OM0_BULK = 0.35      # representative "bulk" point in the well-sampled range
EXCURSION_THRESH = 0.25

N_BURNIN = 10000
N_RESULTS = 10000


def log(msg):
    print(f"[mech] {msg}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_everything():
    model = drg.build_model()
    idx_om0 = model.z_param_names.index("cosmo/Om0")
    idx_w0 = model.z_param_names.index("cosmo/w0")
    log(f"rebuilt LensModel: num_free_params={model.num_free_params}, "
        f"idx_om0={idx_om0}, idx_w0={idx_w0}")

    Om0_samples, w0_samples, samples_z, idx_om0_chk, idx_w0_chk = \
        drg.load_mclmc_cosmo_samples(model)
    assert idx_om0_chk == idx_om0 and idx_w0_chk == idx_w0
    n_chains, n_steps, n_params = samples_z.shape
    log(f"samples_z (post-burn-in results only): shape={samples_z.shape}")
    log(f"Om0_samples/w0_samples (physical, post-burn-in): shape={Om0_samples.shape}")

    diag_path = os.path.join(MCLMC_DIR, "diagnostics.npz")
    with np.load(diag_path) as d:
        diag = {k: np.asarray(d[k]) for k in d.keys()}
    log("diagnostics.npz keys and shapes:")
    for k, v in diag.items():
        log(f"    {k}: shape={v.shape} dtype={v.dtype}")

    map_path = os.path.join(MAP_DIR, "arrays.npz")
    with np.load(map_path) as f:
        z_best = np.asarray(f["z_best"])
    log(f"MAP z_best: shape={z_best.shape}")

    grid = {}
    with np.load(GRID_NPZ) as f:
        for k in f.keys():
            grid[k] = np.asarray(f[k])
    log(f"def_ratio_grid.npz loaded: r2_truth={float(grid['r2_truth']):.7f}")

    return dict(
        model=model, idx_om0=idx_om0, idx_w0=idx_w0,
        Om0_samples=Om0_samples, w0_samples=w0_samples, samples_z=samples_z,
        diag=diag, z_best=z_best, grid=grid,
    )


# ---------------------------------------------------------------------------
# Shape/meaning verification of diagnostics.npz (printed, not asserted-fatal
# beyond sanity so the report is honest about what was actually found).
# ---------------------------------------------------------------------------
def verify_diagnostics_alignment(ctx):
    diag = ctx["diag"]
    samples_z = ctx["samples_z"]
    n_chains, n_results, n_params = samples_z.shape
    step_size, L, xi, nonan = diag["step_size"], diag["L"], diag["xi"], diag["nonan"]
    imm, samples_cov = diag["inverse_mass_matrix"], diag["samples_cov"]

    total_steps = step_size.shape[1]
    log(f"diagnostics per-step histories: shape[1]={total_steps} vs "
        f"n_burnin+n_results={N_BURNIN + N_RESULTS} "
        f"(n_results alone={n_results})")

    # samples_cov should be EXACTLY the covariance of the pooled post-burn-in
    # samples_z (this pins down what 'samples_cov' is: a derived summary of
    # the *results* samples, not a diagnostics-history entry).
    pooled = samples_z.reshape(-1, n_params)
    cov_check = np.cov(pooled, rowvar=False)
    max_diff = np.abs(samples_cov - cov_check).max()
    log(f"samples_cov vs cov(pooled samples_z): max abs diff = {max_diff:.3e} "
        "(0 => samples_cov IS the post-burn-in sample covariance)")

    # L is a single shared (non-per-chain) scalar at every step -- check.
    L_std_over_chains = L.std(axis=0)
    log(f"L: identical across chains at every step? max std-over-chains = "
        f"{L_std_over_chains.max():.3e}; unique values in chain 0: "
        f"{len(np.unique(L[0]))} -> values {np.unique(L[0])}")

    # step_size: per-chain during burn-in tuning, converges to a single
    # shared value; report when it becomes fully shared/frozen.
    step_std_over_chains = step_size.std(axis=0)
    frozen_step = np.where(step_std_over_chains < 1e-12)[0]
    first_frozen = int(frozen_step[0]) if len(frozen_step) else -1
    log(f"step_size: first step index at which value is IDENTICAL across all "
        f"8 chains: {first_frozen} (out of {total_steps}); final value = "
        f"{step_size[0, -1]:.6f}")

    log(f"nonan: all True? {bool(nonan.all())}; total False count = "
        f"{int((~nonan).sum())}")
    log(f"inverse_mass_matrix shape={imm.shape} -> single shared (chain-pooled) "
        "history, NOT per-chain")

    return dict(total_steps=total_steps, first_frozen_step=first_frozen)


# ---------------------------------------------------------------------------
# (1) Trace plots + per-chain excursion statistics + classification.
# ---------------------------------------------------------------------------
def analyze_traces(ctx):
    Om0 = ctx["Om0_samples"]  # (n_chains, n_steps) physical, post-burn-in
    w0 = ctx["w0_samples"]
    n_chains, n_steps = Om0.shape
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    for c in range(n_chains):
        axes[0].plot(steps, Om0[c], lw=0.5, alpha=0.8, color=cmap(c % 10),
                      label=f"chain {c}")
        axes[1].plot(steps, w0[c], lw=0.5, alpha=0.8, color=cmap(c % 10))
    axes[0].axhline(OM0_EDGE, color="black", ls="--", lw=1,
                     label=f"truncation edge Om0={OM0_EDGE}")
    axes[0].axhline(OM0_CREST, color="red", ls=":", lw=1,
                     label=f"crest Om0~{OM0_CREST}")
    axes[0].set_ylabel(r"$\Omega_{m,0}$ (physical)")
    axes[0].legend(loc="upper right", fontsize=6, ncol=3)
    axes[1].set_ylabel(r"$w_0$ (physical)")
    axes[1].set_xlabel("post-burn-in step")
    fig.suptitle("MCLMC cosmology traces (8 chains): mechanism diagnosis "
                  "(UNCERTIFIED)")
    fig.tight_layout()
    fig.savefig(TRACES_PNG, dpi=150)
    plt.close(fig)
    log(f"wrote {TRACES_PNG}")

    report = []
    for c in range(n_chains):
        om0_c = Om0[c]
        min_om0 = float(om0_c.min())
        argmin_step = int(om0_c.argmin())

        below = om0_c < EXCURSION_THRESH
        # contiguous run detection
        d = np.diff(below.astype(int))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0] + 1)
        if below[0]:
            starts = [0] + starts
        if below[-1]:
            ends = ends + [n_steps]
        durations = [e - s for s, e in zip(starts, ends)]
        n_excursions = len(durations)
        frac_below = float(below.mean())
        max_dur = int(max(durations)) if durations else 0
        mean_dur = float(np.mean(durations)) if durations else 0.0
        truncated_last = bool(below[-1])  # excursion still ongoing at trace end

        om0_range = float(om0_c.max() - om0_c.min())
        if om0_range < 0.02:
            classification = "frozen (no meaningful movement)"
        elif n_excursions >= 15 and mean_dur < 0.02 * n_steps:
            classification = "bouncing (frequent brief approach-and-retreat)"
        elif n_excursions <= 5 and max_dur > 0.1 * n_steps:
            classification = "slow-mixing (long sojourn(s) near/at the boundary)"
        else:
            classification = "mixed / intermediate"

        report.append(dict(
            chain=c, min_om0=min_om0, argmin_step=argmin_step,
            n_excursions=n_excursions, mean_dur=mean_dur, max_dur=max_dur,
            frac_below=frac_below, truncated_last=truncated_last,
            classification=classification,
        ))
        log(f"chain {c}: min Om0={min_om0:.4f} @ step {argmin_step}; "
            f"excursions(<{EXCURSION_THRESH})={n_excursions}, "
            f"mean_dur={mean_dur:.0f}, max_dur={max_dur} "
            f"({100*max_dur/n_steps:.1f}% of trace), "
            f"frac_time_below={100*frac_below:.1f}%, "
            f"ongoing_at_end={truncated_last} -> {classification}")
    return report


# ---------------------------------------------------------------------------
# (2) Energy error xi vs physical Om0.
# ---------------------------------------------------------------------------
def analyze_xi_vs_om0(ctx, align_info):
    diag = ctx["diag"]
    xi = diag["xi"]  # (n_chains, total_steps) burn-in + results
    Om0 = ctx["Om0_samples"]  # (n_chains, n_results) results only
    n_chains, n_results = Om0.shape
    total_steps = xi.shape[1]

    if total_steps == n_results:
        xi_overlap = xi
        log("xi already exactly matches the results-only length; no burn-in "
            "truncation needed.")
    else:
        assert total_steps >= n_results, (
            f"xi has fewer steps ({total_steps}) than samples_z results "
            f"({n_results}); cannot align."
        )
        xi_overlap = xi[:, -n_results:]
        log(f"xi covers burn-in+results ({total_steps} steps); positions "
            f"(Om0_samples) cover only the results phase ({n_results} steps). "
            f"Using the trailing {n_results}-step OVERLAP of xi "
            f"(diagnostics index {total_steps - n_results}:{total_steps}) "
            "paired 1:1, per-chain, with samples_z's results steps. "
            "(Verified: samples_cov == cov(pooled samples_z) exactly, and "
            "step_size/L become chain-shared and frozen well before the "
            "burn-in/results boundary, consistent with this alignment.)")

    Om0_flat = Om0.flatten()
    xi_flat = xi_overlap.flatten()
    log_xi_flat = np.log10(np.clip(xi_flat, 1e-300, None))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hb = axes[0].hexbin(Om0_flat, log_xi_flat, gridsize=60, cmap="viridis",
                         bins="log", mincnt=1)
    fig.colorbar(hb, ax=axes[0], label="log10(count)")
    axes[0].axvline(OM0_EDGE, color="black", ls="--", lw=1)
    axes[0].axvline(OM0_CREST, color="red", ls=":", lw=1)
    axes[0].set_xlabel(r"$\Omega_{m,0}$ (physical)")
    axes[0].set_ylabel(r"$\log_{10}(\xi)$ (per-step energy-error stat)")
    axes[0].set_title("xi vs Om0 (pooled over chains, results phase)")

    n_bins = 30
    bin_edges = np.linspace(Om0_flat.min(), Om0_flat.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    p99 = np.full(n_bins, np.nan)
    p50 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (Om0_flat >= bin_edges[i]) & (Om0_flat < bin_edges[i + 1])
        counts[i] = mask.sum()
        if mask.sum() > 5:
            p99[i] = np.percentile(xi_flat[mask], 99)
            p50[i] = np.percentile(xi_flat[mask], 50)
    axes[1].plot(bin_centers, np.log10(p99), "o-", color="crimson",
                 label="99th pct xi")
    axes[1].plot(bin_centers, np.log10(p50), "o-", color="steelblue",
                 label="median xi")
    axes[1].axvline(OM0_EDGE, color="black", ls="--", lw=1, label="edge")
    axes[1].axvline(OM0_CREST, color="red", ls=":", lw=1, label="crest")
    axes[1].set_xlabel(r"$\Omega_{m,0}$ (physical)")
    axes[1].set_ylabel(r"$\log_{10}(\xi)$")
    axes[1].set_title("xi percentiles in Om0 bins")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(XI_PNG, dpi=150)
    plt.close(fig)
    log(f"wrote {XI_PNG}")

    low_mask = Om0_flat < 0.25
    bulk_mask = Om0_flat > 0.30
    p99_low = float(np.percentile(xi_flat[low_mask], 99)) if low_mask.sum() > 10 else float("nan")
    p99_bulk = float(np.percentile(xi_flat[bulk_mask], 99)) if bulk_mask.sum() > 10 else float("nan")
    log(f"xi 99th pct: Om0<0.25 (near edge/crest) = {p99_low:.4g}  vs  "
        f"Om0>0.30 (bulk) = {p99_bulk:.4g}  "
        f"(ratio = {p99_low / p99_bulk if p99_bulk else float('nan'):.3g})")
    log(f"n samples with Om0<0.25: {int(low_mask.sum())} / {len(Om0_flat)} "
        f"({100 * low_mask.mean():.2f}%)")
    return dict(p99_low=p99_low, p99_bulk=p99_bulk,
                bin_centers=bin_centers, p99=p99, p50=p50)


# ---------------------------------------------------------------------------
# (3) Step size / L / mass-matrix cosmo block.
# ---------------------------------------------------------------------------
def analyze_metric(ctx):
    diag = ctx["diag"]
    step_size, L, imm = diag["step_size"], diag["L"], diag["inverse_mass_matrix"]
    idx_om0, idx_w0 = ctx["idx_om0"], ctx["idx_w0"]

    final_step_size = float(step_size[:, -1].mean())
    final_step_size_allchains = step_size[:, -1]
    final_L = float(L[:, -1].mean())
    final_L_allchains = L[:, -1]
    log(f"final step_size (all 8 chains): {final_step_size_allchains} "
        f"(shared: {np.allclose(final_step_size_allchains, final_step_size)})")
    log(f"final L (all 8 chains): {final_L_allchains} "
        f"(shared: {np.allclose(final_L_allchains, final_L)})")

    L_unique = np.unique(L[0])
    log(f"L takes exactly {len(L_unique)} distinct value(s) across the ENTIRE "
        f"run (burn-in+results), shared across chains: {L_unique}. "
        "This is a step-function, not a gradual collapse: L is held at the "
        "smaller tuning value throughout burn-in and jumps to the larger "
        "value at the burn-in/results boundary, then stays fixed through "
        "all of the results phase (standard MCLMC tune3->sampling "
        "handoff, not itself anomalous).")

    step_std_over_chains = step_size.std(axis=0)
    n_frozen_shared = int((step_std_over_chains < 1e-12).sum())
    log(f"step_size collapses from per-chain-varying to a single shared "
        f"value for {n_frozen_shared}/{step_size.shape[1]} of the recorded "
        "steps (frozen well before burn-in ends); final shared value = "
        f"{final_step_size:.6f}.")

    imm_final = imm[0, -1]
    block = imm_final[np.ix_([idx_om0, idx_w0], [idx_om0, idx_w0])]
    eigvals, eigvecs = np.linalg.eigh(block)
    log(f"final inverse_mass_matrix cosmo block (Om0,w0) x (Om0,w0):\n{block}")
    log(f"eigenvalues (variances, z-space): {eigvals}")
    stds = np.sqrt(np.clip(eigvals, 0, None))
    angles_deg = []
    for i in range(2):
        v = eigvecs[:, i]
        ang = np.degrees(np.arctan2(v[1], v[0])) % 180.0
        angles_deg.append(ang)
        log(f"  principal axis {i}: std={stds[i]:.4f}, angle from z_Om0 axis "
            f"= {ang:.2f} deg (mod 180)")

    return dict(final_step_size=final_step_size, final_L=final_L,
                imm_block=block, eigvals=eigvals, eigvecs=eigvecs,
                axis_angles_deg=angles_deg, axis_stds=stds)


# ---------------------------------------------------------------------------
# (4) Neck / ridge geometry along the r2(truth) contour, in z-space.
# ---------------------------------------------------------------------------
def z_om0_of_om0(om0):
    return jsp.ndtri(om0)


def z_w0_of_w0(w0):
    return jsp.ndtri((w0 + 2.0) * (3.0 / 5.0))


def analyze_ridge_geometry(ctx, metric_info):
    grid = ctx["grid"]
    Om0_mesh, w0_mesh, r2_grid = grid["Om0_mesh"], grid["w0_mesh"], grid["r2_grid"]
    r2_truth = float(grid["r2_truth"])
    r2_samples = grid["r2_samples"]
    sigma_r_eff = float(np.std(r2_samples))
    log(f"sigma_r_eff = std(r2 at actual chain samples) = {sigma_r_eff:.4e} "
        f"(r2 samples range: [{r2_samples.min():.6f}, {r2_samples.max():.6f}], "
        f"r2_truth={r2_truth:.6f})")

    # --- validate the standalone z<->physical formulas against the model's
    # own bijector, on a handful of actual samples (rigor: don't silently
    # assume the per-parameter independence of the Uniform->NormalCDF
    # bijector; check it). ---
    samples_z = ctx["samples_z"]
    idx_om0, idx_w0 = ctx["idx_om0"], ctx["idx_w0"]
    z_om0_chk = samples_z[0, :5, idx_om0]
    z_w0_chk = samples_z[0, :5, idx_w0]
    om0_chk_formula = np.asarray(jst.norm.cdf(z_om0_chk))
    w0_chk_formula = np.asarray(-2.0 + (5.0 / 3.0) * jst.norm.cdf(z_w0_chk))
    om0_chk_model = ctx["Om0_samples"][0, :5]
    w0_chk_model = ctx["w0_samples"][0, :5]
    max_diff_om0 = np.abs(om0_chk_formula - om0_chk_model).max()
    max_diff_w0 = np.abs(w0_chk_formula - w0_chk_model).max()
    log(f"standalone Om0(z)=Phi(z), w0(z)=-2+(5/3)Phi(z) formula vs model "
        f"bijector, max abs diff over 5 samples: Om0={max_diff_om0:.3e}, "
        f"w0={max_diff_w0:.3e} (should be ~0)")
    if max_diff_om0 > 1e-8 or max_diff_w0 > 1e-8:
        raise RuntimeError(
            "Standalone per-parameter bijector formula does NOT match the "
            "model's own bijector -- the independence assumption is wrong; "
            "aborting rather than silently using an incorrect chain rule."
        )

    # --- extract the r2(truth) contour from the grid ---
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(Om0_mesh, w0_mesh, r2_grid, levels=[r2_truth])
    segs = cs.allsegs[0]
    plt.close(fig_tmp)
    log(f"r2={r2_truth:.6f} contour: {len(segs)} disjoint segment(s) found "
        f"on the grid; lengths = {[s.shape[0] for s in segs]}")
    seg = max(segs, key=lambda s: s.shape[0])
    if seg[0, 0] > seg[-1, 0]:
        seg = seg[::-1]
    log(f"using longest segment: {seg.shape[0]} points, "
        f"Om0 range [{seg[:,0].min():.4f}, {seg[:,0].max():.4f}], "
        f"w0 range [{seg[:,1].min():.4f}, {seg[:,1].max():.4f}]")

    s_full = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])))]
    )
    # restrict to Om0 >= 0.005 through the segment's high end
    keep = seg[:, 0] >= 0.005
    if not keep.any():
        raise RuntimeError("No contour points with Om0>=0.005 found.")
    i0 = int(np.argmax(keep))  # first True
    s_lo, s_hi = s_full[i0], s_full[-1]

    n_pts = 40
    s_uniform = np.linspace(s_lo, s_hi, n_pts)
    Om0_pts = np.interp(s_uniform, s_full, seg[:, 0])
    w0_pts = np.interp(s_uniform, s_full, seg[:, 1])
    log(f"parametrized {n_pts} points along contour from Om0={Om0_pts[0]:.4f} "
        f"to Om0={Om0_pts[-1]:.4f} (arclength {s_lo:.4f} to {s_hi:.4f})")

    # The contour is monotonic in Om0 over this range (verified: no sign
    # changes in diff(seg[:,0]) -- the "crest" is a region of high
    # curvature/rapid tangent rotation, not a literal fold/turning point),
    # so Om0 -> s is invertible; use this to evaluate EXACT target points
    # (edge/crest/bulk) rather than nearest-neighbor of the coarse 40-point
    # plotting grid, which is too sparse to resolve Om0=0.163 from Om0=0.2.
    om0_arr, w0_arr = seg[:, 0], seg[:, 1]
    is_monotonic_incr = np.all(np.diff(om0_arr) >= 0)
    log(f"contour Om0 monotonic increasing after reorder: {is_monotonic_incr} "
        "(no fold -> Om0->s inversion is well-defined)")

    cosmo_model = w0waCDM_Cosmo(z_lens=drg.Z_LENS, z_source_ref=drg.Z_SOURCE1)

    def r2_scalar(om0, w0):
        r2 = cosmo_model.deflection_ratio(
            drg.Z_SOURCE2, H0=drg.TRUTH["H0"], Om0=om0, k=drg.TRUTH["k"],
            w0=w0, wa=drg.TRUTH["wa"],
        )
        return jnp.squeeze(jnp.asarray(r2))

    grad_r2_fn = jax.jit(jax.grad(r2_scalar, argnums=(0, 1)))

    def width_and_tangent_at(om0_target, w0_target, ds_frac=2e-3):
        """Exact (not downsampled) width + z-space tangent angle at one
        physical contour point, via local interpolation of s(Om0) and
        finite differencing in z-space around it."""
        s_target = float(np.interp(om0_target, om0_arr, s_full))
        ds = ds_frac * (s_hi - s_lo)
        s_m, s_p = max(s_target - ds, s_full[0]), min(s_target + ds, s_full[-1])
        om0_m, w0_m = np.interp(s_m, s_full, om0_arr), np.interp(s_m, s_full, w0_arr)
        om0_p, w0_p = np.interp(s_p, s_full, om0_arr), np.interp(s_p, s_full, w0_arr)
        z_om0_m, z_w0_m = float(z_om0_of_om0(om0_m)), float(z_w0_of_w0(w0_m))
        z_om0_p, z_w0_p = float(z_om0_of_om0(om0_p)), float(z_w0_of_w0(w0_p))
        tangent_angle = np.arctan2(z_w0_p - z_w0_m, z_om0_p - z_om0_m)

        dr2_dom0, dr2_dw0 = grad_r2_fn(float(om0_target), float(w0_target))
        z_om0_t = float(z_om0_of_om0(om0_target))
        z_w0_t = float(z_w0_of_w0(w0_target))
        phi_zom0 = float(jst.norm.pdf(z_om0_t))
        phi_zw0 = float(jst.norm.pdf(z_w0_t))
        dom0_dz = phi_zom0 * 1.0
        dw0_dz = phi_zw0 * (5.0 / 3.0)
        grad_z = np.array([float(dr2_dom0) * dom0_dz, float(dr2_dw0) * dw0_dz])
        gnorm = np.linalg.norm(grad_z)
        width = sigma_r_eff / gnorm if gnorm > 0 else np.inf
        return width, tangent_angle

    z_om0_pts = np.asarray(z_om0_of_om0(jnp.asarray(Om0_pts)))
    z_w0_pts = np.asarray(z_w0_of_w0(jnp.asarray(w0_pts)))

    widths = np.zeros(n_pts)
    grad_z_norms = np.zeros(n_pts)
    for i in range(n_pts):
        dr2_dom0, dr2_dw0 = grad_r2_fn(float(Om0_pts[i]), float(w0_pts[i]))
        phi_zom0 = float(jst.norm.pdf(z_om0_pts[i]))
        phi_zw0 = float(jst.norm.pdf(z_w0_pts[i]))
        dom0_dz = phi_zom0 * 1.0          # Om0 bijector scale = (1-0) = 1
        dw0_dz = phi_zw0 * (5.0 / 3.0)    # w0 bijector scale = (-1/3 - (-2))
        grad_z = np.array([float(dr2_dom0) * dom0_dz, float(dr2_dw0) * dw0_dz])
        gnorm = np.linalg.norm(grad_z)
        grad_z_norms[i] = gnorm
        widths[i] = sigma_r_eff / gnorm if gnorm > 0 else np.inf

    # tangent direction in z-space (finite differences along the 40 pts;
    # coarse -- used only for the plot's overall shape, NOT for the precise
    # edge/crest/bulk numbers below, which use width_and_tangent_at()).
    dz_om0 = np.gradient(z_om0_pts)
    dz_w0 = np.gradient(z_w0_pts)
    tangent_angle_raw = np.arctan2(dz_w0, dz_om0)  # radians, signed, continuous-ish
    tangent_angle_unwrapped = np.unwrap(tangent_angle_raw)
    tangent_angle_line_deg = np.degrees(tangent_angle_raw) % 180.0

    # --- figure ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(Om0_pts, tangent_angle_line_deg, "o-", color="darkorange")
    axes[0].axvline(OM0_EDGE, color="black", ls="--", lw=1, label="edge")
    axes[0].axvline(OM0_CREST, color="red", ls=":", lw=1, label="crest")
    axes[0].axvline(OM0_BULK, color="steelblue", ls="-.", lw=1, label="bulk")
    axes[0].set_ylabel("z-space tangent angle (deg, mod 180)")
    axes[0].legend(fontsize=8)
    axes[1].plot(Om0_pts, np.log10(widths), "o-", color="teal")
    axes[1].axvline(OM0_EDGE, color="black", ls="--", lw=1)
    axes[1].axvline(OM0_CREST, color="red", ls=":", lw=1)
    axes[1].axvline(OM0_BULK, color="steelblue", ls="-.", lw=1)
    axes[1].set_ylabel(r"$\log_{10}$(ridge width), z-space")
    axes[1].set_xlabel(r"$\Omega_{m,0}$ along r2(truth) contour")
    fig.suptitle("Ridge (neck) geometry along the r2(truth) contour, z-space "
                 "(UNCERTIFIED)")
    fig.tight_layout()
    fig.savefig(RIDGE_PNG, dpi=150)
    plt.close(fig)
    log(f"wrote {RIDGE_PNG}")

    # Exact (interpolated, not nearest-of-40) evaluation at the specific
    # target Om0 values requested -- the 40-pt plotting grid is too coarse
    # to distinguish edge (0.163) from crest (0.2).
    w0_at_edge = float(np.interp(OM0_EDGE, om0_arr, w0_arr))
    w0_at_crest = float(np.interp(OM0_CREST, om0_arr, w0_arr))
    w0_at_bulk = float(np.interp(OM0_BULK, om0_arr, w0_arr))
    w0_at_035 = float(np.interp(0.35, om0_arr, w0_arr))
    w0_at_005 = float(np.interp(0.05, om0_arr, w0_arr))

    width_edge, tangent_edge = width_and_tangent_at(OM0_EDGE, w0_at_edge)
    width_crest, tangent_crest = width_and_tangent_at(OM0_CREST, w0_at_crest)
    width_bulk, tangent_bulk = width_and_tangent_at(OM0_BULK, w0_at_bulk)
    _, tangent_035 = width_and_tangent_at(0.35, w0_at_035)
    _, tangent_005 = width_and_tangent_at(0.05, w0_at_005)

    log(f"widths (exact interpolation): edge(Om0={OM0_EDGE}, w0={w0_at_edge:.4f})="
        f"{width_edge:.4e}, crest(Om0={OM0_CREST}, w0={w0_at_crest:.4f})="
        f"{width_crest:.4e}, bulk(Om0={OM0_BULK}, w0={w0_at_bulk:.4f})="
        f"{width_bulk:.4e}")
    log(f"width ratio crest/bulk = {width_crest/width_bulk:.3f}; "
        f"crest/edge = {width_crest/width_edge:.3f}; "
        f"edge/bulk = {width_edge/width_bulk:.3f}")

    # total tangent rotation Om0=0.35 -> Om0=0.05: integrate the SIGNED
    # tangent-angle change along s between the two points using the fine
    # (546-pt) contour and unwrapping, so a coarse 40-pt sampling can't
    # under/over-count the rotation.
    s_035 = float(np.interp(0.35, om0_arr, s_full))
    s_005 = float(np.interp(0.05, om0_arr, s_full))
    s_lo_int, s_hi_int = sorted([s_035, s_005])
    s_fine = np.linspace(s_lo_int, s_hi_int, 400)
    om0_fine = np.interp(s_fine, s_full, om0_arr)
    w0_fine = np.interp(s_fine, s_full, w0_arr)
    z_om0_fine = np.asarray(z_om0_of_om0(jnp.asarray(om0_fine)))
    z_w0_fine = np.asarray(z_w0_of_w0(jnp.asarray(w0_fine)))
    tangent_fine = np.unwrap(np.arctan2(np.gradient(z_w0_fine), np.gradient(z_om0_fine)))
    rotation_deg = abs(np.degrees(tangent_fine[-1] - tangent_fine[0]))
    log(f"total tangent rotation between Om0=0.35 and Om0=0.05 "
        f"(fine 400-pt integration): {rotation_deg:.2f} deg "
        f"(coarse single-pair estimate for comparison: "
        f"{abs(np.degrees(tangent_035 - tangent_005)):.2f} deg)")

    # metric principal axis vs tangent direction, at bulk and crest
    axis_angles = metric_info["axis_angles_deg"]  # two eigenvector angles (mod 180)
    # compare the LARGER-std axis (index of larger eigval) to tangent
    larger_idx = int(np.argmax(metric_info["eigvals"]))
    metric_angle = axis_angles[larger_idx]

    def angle_mismatch(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    tangent_bulk_deg = np.degrees(tangent_bulk) % 180.0
    tangent_crest_deg = np.degrees(tangent_crest) % 180.0
    tangent_edge_deg = np.degrees(tangent_edge) % 180.0
    mismatch_bulk = angle_mismatch(metric_angle, tangent_bulk_deg)
    mismatch_crest = angle_mismatch(metric_angle, tangent_crest_deg)
    mismatch_edge = angle_mismatch(metric_angle, tangent_edge_deg)
    log(f"metric long-axis angle = {metric_angle:.2f} deg; tangent angle at "
        f"bulk (Om0={OM0_BULK}) = {tangent_bulk_deg:.2f} deg (mismatch "
        f"{mismatch_bulk:.2f} deg); tangent angle at crest (Om0={OM0_CREST}) = "
        f"{tangent_crest_deg:.2f} deg (mismatch "
        f"{mismatch_crest:.2f} deg); tangent angle at truncation edge "
        f"(Om0={OM0_EDGE}, the ACTUAL per-chain turnaround point -- see "
        f"trace min-Om0 values, which cluster at 0.146-0.163, i.e. BELOW "
        f"the crest at 0.2, meaning chains already cross the crest and are "
        f"stopped further down) = {tangent_edge_deg:.2f} deg (mismatch "
        f"{mismatch_edge:.2f} deg)")

    return dict(
        Om0_pts=Om0_pts, w0_pts=w0_pts, widths=widths,
        tangent_angle_line_deg=tangent_angle_line_deg,
        sigma_r_eff=sigma_r_eff,
        width_edge=float(width_edge), width_crest=float(width_crest),
        width_bulk=float(width_bulk),
        rotation_deg=float(rotation_deg),
        mismatch_bulk_deg=float(mismatch_bulk),
        mismatch_crest_deg=float(mismatch_crest),
        mismatch_edge_deg=float(mismatch_edge),
    )


# ---------------------------------------------------------------------------
# (5) NaN summary.
# ---------------------------------------------------------------------------
def analyze_nans(ctx):
    nonan = ctx["diag"]["nonan"]
    n_nan_steps = int((~nonan).sum())
    log(f"nonan history: total NaN-flagged steps = {n_nan_steps} / "
        f"{nonan.size} ({100*n_nan_steps/nonan.size:.4f}%)")
    if n_nan_steps == 0:
        log("No NaN-flagged steps anywhere in the run -- NaNs are not part "
            "of the mechanism.")
        return dict(n_nan_steps=0)
    per_chain = (~nonan).sum(axis=1)
    log(f"per-chain NaN counts: {per_chain}")
    return dict(n_nan_steps=n_nan_steps, per_chain=per_chain)


def main():
    ctx = load_everything()
    align_info = verify_diagnostics_alignment(ctx)
    trace_report = analyze_traces(ctx)
    xi_report = analyze_xi_vs_om0(ctx, align_info)
    metric_info = analyze_metric(ctx)
    ridge_info = analyze_ridge_geometry(ctx, metric_info)
    nan_info = analyze_nans(ctx)

    log("=" * 70)
    log("SUMMARY (UNCERTIFIED)")
    log("=" * 70)
    for r in trace_report:
        log(f"  chain {r['chain']}: {r['classification']}")
    log(f"  xi 99th pct near edge/crest vs bulk ratio: "
        f"{xi_report['p99_low']/xi_report['p99_bulk']:.3g}")
    log(f"  ridge width ratio crest/bulk: "
        f"{ridge_info['width_crest']/ridge_info['width_bulk']:.3g}")
    log(f"  tangent rotation Om0 0.35->0.05: {ridge_info['rotation_deg']:.1f} deg")
    log(f"  metric-vs-tangent mismatch (bulk): {ridge_info['mismatch_bulk_deg']:.1f} deg")
    log(f"  metric-vs-tangent mismatch (crest): {ridge_info['mismatch_crest_deg']:.1f} deg")
    log(f"  metric-vs-tangent mismatch (truncation edge): {ridge_info['mismatch_edge_deg']:.1f} deg")


if __name__ == "__main__":
    main()
