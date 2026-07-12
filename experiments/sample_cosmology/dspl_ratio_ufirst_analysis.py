#!/usr/bin/env python3
"""Post-run analysis for Run D (banded u-first ratio coordinates) — pre-registered.

Reads the completed pipeline's artifacts from
`results/sample_cosmology/dspl_ratio_ufirst/` and evaluates the design
checkpoint's predictions (docs/logs/sample-cosmology-dspl.md, "Run D"):

  P1  rank-R̂(Om0) and rank-R̂(w0) (PHYSICAL space) < 1.01
  P2  bulk-ESS of BOTH cosmology z-columns within 2x of the median nuisance ESS
  P3  mass(Om0 < 0.146) from the pooled Om0 marginal = 0.103 +/- 0.02
  P4  0 nonfinite-flagged MCLMC steps
  P5  MAP chi2/nu ~= 1 (final value of the MAP stage's chisq_hist)
  monitors: max/min sampled Om0 (degenerate-sliver watch: flips live at
  Om0 >= 0.785), per-chain Om0 range (truncation watch at the former
  0.146-0.163 edge).

Plots (the PRIMARY phenomenon check per the checkpoint's blind-spot (a)):
  - ratio_ufirst_traces.png : per-chain Om0 and w0 traces with the former
    truncation edge marked; free crossings = hypothesis, bounce = falsifier.
  - ratio_ufirst_overlay.png: grid posterior bands (def_ratio_grid.npz) with
    the run's pooled (Om0, w0) samples — must show the full arm to Om0=0.

Writes `ratio_ufirst_run_summary.json` (observed vs predicted, verdicts).
Transforming 80k z-samples through the bisection map is GPU-cheap; run this
inside the same allocation as the sampler (see run_dspl_ratio_ufirst.sh).
"""
from __future__ import annotations

import json
import os

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dspl_ratio_ufirst as ru
import dspl_ratio_coords as rc
from dspl_arm_init import GRID_NPZ

from gigalens_research.inference_utils.posterior import SamplerPosterior

MCLMC_ARRAYS = os.path.join(ru.RESULTS_DIR, "mclmc", "arrays.npz")
MCLMC_DIAG = os.path.join(ru.RESULTS_DIR, "mclmc", "diagnostics.npz")
MAP_ARRAYS = os.path.join(ru.RESULTS_DIR, "map", "arrays.npz")

OM0_EDGE = 0.146          # T1 truncation edge (mass threshold of P3)
OM0_TURNAROUND = 0.163    # T2 measured turnaround
OM0_SLIVER = 0.785        # degenerate-sliver lower edge (validator, 2026-07-11)

GROUP_KEY = "cosmo/Om0|cosmo/w0"


def main():
    model, *_ = ru.build_grouped_model_ufirst()
    names = model.z_param_names
    n_dim = model.num_free_params

    with np.load(MCLMC_ARRAYS) as d:
        samples_z = np.asarray(d["samples_z"])          # (n_chains, n_steps, D)
    n_chains, n_steps, n_params = samples_z.shape
    if n_params != n_dim:
        raise RuntimeError(f"samples have {n_params} params, model has {n_dim}")

    icosmo = [names.index("cosmo/Om0"), names.index("cosmo/w0")]
    inuis = [i for i in range(n_dim) if i not in icosmo]

    # ---- z -> physical (GPU; batched through the bisection solve) ------------
    z_flat = jnp.asarray(samples_z.reshape(-1, n_params))
    theta = jax.jit(model.bijector.forward)(z_flat)
    omw = np.asarray(theta[GROUP_KEY]).reshape(n_chains, n_steps, 2)
    om, w0 = omw[..., 0], omw[..., 1]

    # ---- P1/P2: convergence metrics ------------------------------------------
    sp_z = SamplerPosterior(None, samples_z)
    rhat_z, ess_z = np.asarray(sp_z.rhat), np.asarray(sp_z.ess)
    sp_phys = SamplerPosterior(None, omw)
    rhat_phys, ess_phys = np.asarray(sp_phys.rhat), np.asarray(sp_phys.ess)

    ess_cosmo_z = {names[i]: float(ess_z[i]) for i in icosmo}
    med_nuis_ess = float(np.median(ess_z[inuis]))
    worst_nuis = dict(
        rhat=float(np.max(rhat_z[inuis])),
        ess=float(np.min(ess_z[inuis])),
    )
    p1 = bool(rhat_phys[0] < 1.01 and rhat_phys[1] < 1.01)
    p2 = bool(all(e >= med_nuis_ess / 2.0 for e in ess_cosmo_z.values()))
    p2_within_2x = bool(all(e >= med_nuis_ess / 2.0 and e <= med_nuis_ess * 2.0
                            for e in ess_cosmo_z.values()))

    # ---- P3: arm mass ---------------------------------------------------------
    mass_low = float(np.mean(om < OM0_EDGE))
    p3 = bool(abs(mass_low - 0.103) <= 0.02)

    # ---- P4: nonfinite steps --------------------------------------------------
    nonan_report = {}
    p4 = None
    if os.path.exists(MCLMC_DIAG):
        with np.load(MCLMC_DIAG) as d:
            if "nonan" in d:
                nonan = np.asarray(d["nonan"])
                # semantics per baseline mech analysis: nonan True/1 = step OK
                n_total = int(nonan.size)
                n_ok = int(np.sum(nonan != 0))
                nonan_report = dict(n_total=n_total, n_flagged=n_total - n_ok,
                                    unique_values=sorted(
                                        float(v) for v in np.unique(nonan)[:4]))
                p4 = bool(n_total - n_ok == 0)

    # ---- P5: MAP chi2/nu ------------------------------------------------------
    with np.load(MAP_ARRAYS) as d:
        chisq_hist = np.asarray(d["chisq_hist"])
    map_red_chi2 = float(chisq_hist[-1])
    p5 = bool(abs(map_red_chi2 - 1.0) < 0.2)

    # ---- monitors -------------------------------------------------------------
    per_chain_om_min = np.min(om, axis=1)
    per_chain_om_max = np.max(om, axis=1)
    monitors = dict(
        om_min=float(om.min()), om_max=float(om.max()),
        per_chain_om_min=[float(v) for v in per_chain_om_min],
        per_chain_om_max=[float(v) for v in per_chain_om_max],
        frac_steps_above_sliver=float(np.mean(om >= OM0_SLIVER)),
        frac_steps_below_edge=float(np.mean(om < OM0_EDGE)),
        chains_reaching_below_0p05=int(np.sum(per_chain_om_min < 0.05)),
    )
    # P6 (Run D checkpoint): full-arc coverage INCLUDING both edges — every
    # chain below Om0=0.05 AND above Om0=0.50 (Run C stopped at 0.385).
    p6 = bool(np.all(per_chain_om_min < 0.05) and np.all(per_chain_om_max > 0.50))

    # ---- plots ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for c in range(n_chains):
        axes[0].plot(om[c], lw=0.4, alpha=0.8)
        axes[1].plot(w0[c], lw=0.4, alpha=0.8)
    axes[0].axhline(OM0_EDGE, color="k", ls="--", lw=1,
                    label=f"baseline truncation edge {OM0_EDGE}")
    axes[0].axhline(OM0_TURNAROUND, color="k", ls=":", lw=1)
    axes[0].set_ylabel("Om0"); axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("w0"); axes[1].set_xlabel("result step")
    fig.suptitle("Run D: per-chain physical traces (8 chains)")
    fig.tight_layout()
    traces_png = os.path.join(ru.RESULTS_DIR, "ratio_ufirst_traces.png")
    fig.savefig(traces_png, dpi=140); plt.close(fig)

    with np.load(GRID_NPZ) as g:
        Om0_mesh, w0_mesh = np.asarray(g["Om0_mesh"]), np.asarray(g["w0_mesh"])
        prob, mass_levels = np.asarray(g["prob"]), np.asarray(g["mass_levels"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(Om0_mesh, w0_mesh, prob, levels=sorted(mass_levels),
               colors=["#bbbbbb", "#888888", "#333333"], linewidths=1.0)
    idx = np.random.default_rng(0).choice(om.size, size=20000, replace=False)
    ax.scatter(om.ravel()[idx], w0.ravel()[idx], s=1.5, alpha=0.15,
               color="tab:red", rasterized=True, label="Run D samples (20k of 80k)")
    ax.axvline(OM0_EDGE, color="k", ls="--", lw=1)
    ax.set_xlabel("Om0"); ax.set_ylabel("w0")
    ax.set_title("Run D samples vs grid posterior bands (99.7/95.5/68%)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    overlay_png = os.path.join(ru.RESULTS_DIR, "ratio_ufirst_overlay.png")
    fig.savefig(overlay_png, dpi=140); plt.close(fig)

    # ---- summary ---------------------------------------------------------------
    summary = dict(
        n_chains=n_chains, n_steps=n_steps, n_params=n_params,
        P1_rank_rhat_Om0=float(rhat_phys[0]), P1_rank_rhat_w0=float(rhat_phys[1]),
        P1_threshold=1.01, P1_passed=p1,
        P2_ess_cosmo_z=ess_cosmo_z, P2_median_nuisance_ess=med_nuis_ess,
        P2_passed_not_below_half=p2, P2_within_2x_band=p2_within_2x,
        P3_mass_below_0146=mass_low, P3_predicted="0.103 +/- 0.02", P3_passed=p3,
        P4_nonan=nonan_report, P4_passed=p4,
        P5_map_red_chi2=map_red_chi2, P5_passed=p5,
        P6_full_arc_all_chains=p6,
        P6_passed=p6,
        worst_nuisance=worst_nuis,
        rhat_z_cosmo={names[i]: float(rhat_z[i]) for i in icosmo},
        monitors=monitors,
        plots=[traces_png, overlay_png],
    )
    out = os.path.join(ru.RESULTS_DIR, "ratio_ufirst_run_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[analysis] wrote {out}")


if __name__ == "__main__":
    main()
