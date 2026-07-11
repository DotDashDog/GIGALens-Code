#!/usr/bin/env python3
"""Post-run analysis for Run A (free-r2 reparameterization).

Design checkpoint: `docs/logs/sample-cosmology-dspl.md`, "Run A". This script
is meant to be run AFTER `dspl_free_r2.py --run` has produced
`results/sample_cosmology/dspl_free_r2/mclmc/arrays.npz` -- it does NOT sample
anything itself. It:

  1. Loads the MCLMC samples and transforms them to physical space via the
     r2-model's own bijector; extracts the plane-2 `deflection_ratio` (= r2)
     column (index identified via `model.z_param_names`, per the project's
     "C-8" convention -- see `gigalens/src/gigalens/jax/scene.py::_z_param_names`).
  2. Reports rank-R-hat / bulk-ESS (arviz-backed, via the project's own
     `SamplerPosterior`) for r2 AND the worst nuisance parameter -- per the
     operating card, always the WORST (max R-hat / min ESS), never a mean.
  3. Estimates hhat(r2) by Gaussian KDE of the pooled (chain-flattened) r2
     samples, evaluated on `def_ratio_grid.npz`'s (Om0, w0) grid via
     r2_grid = deflection_ratio(Om0, w0). Because the r2-model's prior on r2
     is UNIFORM over its support, the posterior KDE hhat(r2) is already
     proportional to the r2 likelihood (no 1/prior_r reweighting needed) --
     see the design checkpoint's "analytic (Om0,w0) reconstruction" clause.
     The reconstructed p(Om0, w0) is then just hhat(r2(Om0,w0)), normalized
     over the grid.
  4. Computes mass(Om0 < 0.146) and compares to the pre-registered prediction
     0.103 +/- 0.02.
  5. Writes `r2_reconstruction.png`: left = per-chain r2 traces; right =
     reconstructed 68/95.5/99.7% contours (color) over the grid-search bands
     from `def_ratio_grid.npz` (black), with the truth star.

Every number/plot below is a PROPOSED (UNCERTIFIED) reading, per this repo's
operating card (docs/agent-operating-card.md) -- it is not a certified claim
until a grader inspects the artifacts.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspl_free_r2 as free_r2  # noqa: E402
import def_ratio_grid as drg    # noqa: E402  (reuse logsumexp_norm / get_mass_levels)

from gigalens_research.inference_utils.posterior import SamplerPosterior  # noqa: E402

RESULTS_DIR = free_r2.RESULTS_DIR
MCLMC_DIR = os.path.join(RESULTS_DIR, "mclmc")
GRID_NPZ = drg.OUT_NPZ  # results/.../dspl_cosmology_newapi/def_ratio_grid.npz

OUT_PNG = os.path.join(RESULTS_DIR, "r2_reconstruction.png")
OUT_JSON = os.path.join(RESULTS_DIR, "r2_reconstruction_summary.json")

MASS_THRESHOLD_OM0 = 0.146  # def_ratio_grid.py / T1 finding: truncation edge
PREDICTED_MASS = 0.103
PREDICTED_MASS_TOL = 0.02


def load_mclmc_samples():
    npz_path = os.path.join(MCLMC_DIR, "arrays.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found -- run `dspl_free_r2.py --run` first (this "
            "script only analyzes an existing MCLMC run; it never launches one).")
    with np.load(npz_path) as f:
        samples_z = np.asarray(f["samples_z"])  # (n_chains, n_steps, n_params)
    return samples_z


def main():
    model = free_r2.build_r2_model()
    print(f"[reconstruct] r2_model.num_free_params = {model.num_free_params}")
    print(f"[reconstruct] r2_model.z_param_names   = {model.z_param_names}")
    idx_r2 = model.z_param_names.index("planes/2/geometry/deflection_ratio")
    print(f"[reconstruct] r2 is z_param_names[{idx_r2}]")

    samples_z = load_mclmc_samples()
    n_chains, n_steps, n_params = samples_z.shape
    if n_params != model.num_free_params:
        raise ValueError(
            f"mclmc/arrays.npz has {n_params} params but the rebuilt r2-model "
            f"has {model.num_free_params}; build_r2_model() no longer matches "
            "the model that produced this run.")

    # -- z -> physical r2, via the model's own bijector (same op as
    #    def_ratio_grid.py's load_mclmc_cosmo_samples) --------------------------
    z_flat = jnp.asarray(samples_z.reshape(-1, n_params))
    x = model.bijector.forward(list(z_flat.T))
    r2_flat = np.asarray(x["planes/2/geometry/deflection_ratio"])
    r2_samples = r2_flat.reshape(n_chains, n_steps)

    # -- convergence: r2 AND worst nuisance (never a mean; operating card) -----
    sp = SamplerPosterior(None, samples_z)
    rhat_all, ess_all = sp.rhat, sp.ess
    rhat_r2, ess_r2 = float(rhat_all[idx_r2]), float(ess_all[idx_r2])

    nuisance_idx = [i for i in range(n_params) if i != idx_r2]
    nuisance_rhat = rhat_all[nuisance_idx]
    nuisance_ess = ess_all[nuisance_idx]
    worst_rhat_i = nuisance_idx[int(np.argmax(nuisance_rhat))]
    worst_ess_i = nuisance_idx[int(np.argmin(nuisance_ess))]
    median_nuisance_ess = float(np.median(nuisance_ess))

    print("\n[reconstruct] --- convergence (UNCERTIFIED; see operating card) ---")
    print(f"  r2:  Rhat={rhat_r2:.4f}  ESS={ess_r2:.1f}")
    print(f"  worst-Rhat nuisance: {model.z_param_names[worst_rhat_i]} "
          f"Rhat={float(rhat_all[worst_rhat_i]):.4f}")
    print(f"  worst-ESS nuisance:  {model.z_param_names[worst_ess_i]} "
          f"ESS={float(ess_all[worst_ess_i]):.1f}")
    print(f"  median nuisance ESS = {median_nuisance_ess:.1f}")

    falsifier_rhat = rhat_r2 >= 1.01
    falsifier_ess = ess_r2 < 0.5 * median_nuisance_ess
    print(f"  Run A falsifier (Rhat(r2)>=1.01):                {falsifier_rhat}")
    print(f"  Run A falsifier (ESS(r2) < half median nuisance): {falsifier_ess}")
    if falsifier_rhat or falsifier_ess:
        print("  *** FALSIFIER TRIGGERED: residual hard geometry in r2 -- the "
              "coordinate-disease hypothesis (Run A's premise) would be WRONG. ***")

    # -- KDE reconstruction of p(Om0, w0) ---------------------------------------
    # Uniform prior on the free r2 parameter -> the posterior KDE of the pooled
    # r2 samples is already proportional to the r2 LIKELIHOOD (no 1/prior_r
    # reweighting needed): posterior_r(r2) ∝ likelihood(r2) * prior_r(r2), and
    # prior_r is constant over its support.
    kde = gaussian_kde(r2_samples.reshape(-1))

    with np.load(GRID_NPZ) as g:
        Om0_mesh = g["Om0_mesh"]
        w0_mesh = g["w0_mesh"]
        r2_grid = g["r2_grid"]
        grid_mass_levels = g["mass_levels"]
        grid_prob = g["prob"]
        r2_truth = float(g["r2_truth"])
        truth_Om0 = float(g["truth_Om0"])
        truth_w0 = float(g["truth_w0"])

    hhat_grid = kde(r2_grid.flatten()).reshape(r2_grid.shape)
    with np.errstate(divide="ignore"):
        log_hhat = np.log(np.clip(hhat_grid, 1e-300, None))
    log_prob_recon = drg.logsumexp_norm(log_hhat)
    prob_recon = np.exp(log_prob_recon)
    mass_levels_recon = drg.get_mass_levels(prob_recon, levels=(0.68, 0.955, 0.997))

    mass_below = float(prob_recon[Om0_mesh < MASS_THRESHOLD_OM0].sum())
    print("\n[reconstruct] --- reconstructed (Om0, w0) mass (UNCERTIFIED) ---")
    print(f"  mass(Om0 < {MASS_THRESHOLD_OM0}) = {mass_below:.4f}  "
          f"(pre-registered prediction {PREDICTED_MASS} +/- {PREDICTED_MASS_TOL})")
    gap = mass_below - PREDICTED_MASS
    within_tol = abs(gap) <= PREDICTED_MASS_TOL
    print(f"  gap = {gap:+.4f}  ({'within' if within_tol else 'OUTSIDE'} tolerance)")

    # -- figure: per-chain r2 traces (left) + reconstructed vs grid contours (right) --
    fig, (ax_trace, ax_contour) = plt.subplots(1, 2, figsize=(13, 5.5))

    cmap = plt.get_cmap("tab10")
    for c in range(n_chains):
        ax_trace.plot(r2_samples[c], lw=0.5, alpha=0.8, color=cmap(c % 10),
                      label=f"chain {c}")
    ax_trace.axhline(r2_truth, color="red", ls="--", lw=1, label="r2(truth)")
    ax_trace.set_xlabel("MCLMC step (results segment)")
    ax_trace.set_ylabel("r2 = deflection_ratio(z_source2)")
    ax_trace.set_title("Per-chain r2 traces")
    ax_trace.legend(fontsize=6, ncol=2, loc="best")

    cs_recon = ax_contour.contour(
        Om0_mesh, w0_mesh, prob_recon, levels=sorted(mass_levels_recon),
        colors=["tab:blue"], linestyles=[":", "--", "-"], linewidths=1.5, zorder=3,
    )
    ax_contour.clabel(
        cs_recon,
        fmt={sorted(mass_levels_recon)[0]: "99.7%", sorted(mass_levels_recon)[1]: "95.5%",
             sorted(mass_levels_recon)[2]: "68%"},
        fontsize=7,
    )
    cs_grid = ax_contour.contour(
        Om0_mesh, w0_mesh, grid_prob, levels=sorted(grid_mass_levels.tolist()),
        colors="k", linestyles=[":", "--", "-"], linewidths=1.0, zorder=2,
    )
    ax_contour.axvline(MASS_THRESHOLD_OM0, color="gray", lw=0.8, ls=":",
                        label=f"Om0={MASS_THRESHOLD_OM0}")
    ax_contour.scatter([truth_Om0], [truth_w0], marker="*", s=200, color="red",
                       edgecolor="k", zorder=5, label="truth")
    ax_contour.set_xlim(0, 1)
    ax_contour.set_ylim(-2, -1 / 3)
    ax_contour.set_xlabel(r"$\Omega_{m,0}$")
    ax_contour.set_ylabel(r"$w_0$")
    ax_contour.set_title(
        "Reconstructed p(Om0,w0) from r2 posterior (blue)\nvs. grid-search (black)",
        fontsize=10,
    )
    ax_contour.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"\n[reconstruct] wrote {OUT_PNG}")

    summary = {
        "n_chains": n_chains, "n_steps": n_steps, "n_params": n_params,
        "idx_r2": idx_r2,
        "rhat_r2": rhat_r2, "ess_r2": ess_r2,
        "worst_rhat_nuisance_name": model.z_param_names[worst_rhat_i],
        "worst_rhat_nuisance_value": float(rhat_all[worst_rhat_i]),
        "worst_ess_nuisance_name": model.z_param_names[worst_ess_i],
        "worst_ess_nuisance_value": float(ess_all[worst_ess_i]),
        "median_nuisance_ess": median_nuisance_ess,
        "falsifier_rhat_triggered": bool(falsifier_rhat),
        "falsifier_ess_triggered": bool(falsifier_ess),
        "mass_om0_below_0p146": mass_below,
        "predicted_mass": PREDICTED_MASS,
        "predicted_mass_tol": PREDICTED_MASS_TOL,
        "within_tolerance": bool(within_tol),
        "r2_truth": r2_truth,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[reconstruct] wrote {OUT_JSON}")
    print("\n[reconstruct] All numbers above are PROPOSED (UNCERTIFIED) readings "
          "-- see docs/agent-operating-card.md rule 5. Update "
          "docs/logs/sample-cosmology-dspl.md with the outcome (rule 9).")


if __name__ == "__main__":
    main()
