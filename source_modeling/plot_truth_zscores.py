"""
Quick script to summarize MCLMC results across the Vela-based simulated
systems, restricted to mass parameters.

Convergence is judged on the joint MCLMC samples via R-hat < 1.1.

Outputs two figures, each with one subplot per mass parameter and every
converged system shown as a point in each subplot:

  1. ``mass_truth_zscores.png``  - the asymmetric z-score
     ``(truth - median) / sigma``, where ``sigma`` is the upper or lower
     1-sigma half-width of the marginal posterior depending on which side
     of the median the truth falls.
  2. ``mass_truth_residuals.png`` - the physical-space residual
     ``truth - median`` with asymmetric 1-sigma error bars taken from the
     16th/84th posterior percentiles.
"""

import os
import pickle
import sys
from os.path import expanduser

home = expanduser("~/")

# Make local gigalens importable, matching the notebook setup.
sys.path.insert(0, os.path.join(home, "gigalens", "src"))

import jax
import numpy as np
from jax import numpy as jnp
import matplotlib.pyplot as plt
import blackjax

import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import bijectors as tfb

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYSTEMS_DIR = os.path.join(home, "GIGALens-Code", "source_modeling", "vela_sim_systems")
CAM = "12"
SIM_NUMS = ["01", "03", "04", "07", "08", "10", "15", "21", "22", "23", "25", "26"]
N_MAX = 10            # shapelet n_max used in the notebook
RHAT_THRESHOLD = 1.1  # convergence cutoff
ZSCORE_OUTPUT_PATH = os.path.join(SYSTEMS_DIR, "mass_truth_zscores.png")
RESIDUAL_OUTPUT_PATH = os.path.join(SYSTEMS_DIR, "mass_truth_residuals.png")


# ---------------------------------------------------------------------------
# Build the prior + bijector exactly as in vela_system_model() from
# sim_system_complex.ipynb. Only the bijector is needed here; observed image
# and sim_config are not.
# ---------------------------------------------------------------------------
def build_prior(n_max=N_MAX):
    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                    gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
                    e1=tfd.TruncatedNormal(0, 0.2, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(0, 0.2, -0.5, 0.5),
                    center_x=tfd.Normal(0, 0.06),
                    center_y=tfd.Normal(0, 0.06),
                )
            ),
            tfd.JointDistributionNamed(
                dict(
                    gamma1=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
                    gamma2=tfd.Normal(0, 0.1, -0.5, 0.5),
                )
            ),
        ]
    )
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
                    n_sersic=tfd.Uniform(0.5, 8),
                    e1=tfd.TruncatedNormal(0, 0.1, -0.2, 0.2),
                    e2=tfd.TruncatedNormal(0, 0.1, -0.2, 0.2),
                    center_x=tfd.Normal(0, 0.02),
                    center_y=tfd.Normal(0, 0.02),
                )
            )
        ]
    )
    source_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    beta=tfd.LogNormal(jnp.log(0.7), 0.4),
                    center_x=tfd.Normal(0, 0.5),
                    center_y=tfd.Normal(0, 0.5),
                )
            ),
        ]
    )
    return tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )


def make_bijector(prior):
    """Replicate ProbModel.bij exactly: pack -> default event space bijector."""
    example = prior.sample(seed=jax.random.PRNGKey(0))
    pack_bij = tfb.pack_sequence_as(example)
    return tfb.Chain([prior.experimental_default_event_space_bijector(), pack_bij])


# ---------------------------------------------------------------------------
# Same asymmetric z-score the notebook uses (cell that prints
# "label : predicted | true | z-score")
# ---------------------------------------------------------------------------
def stdev_calc(truth, med, sig_low, sig_up):
    sigma = jnp.where(truth > med, sig_up - med, med - sig_low)
    return (truth - med) / sigma


# Mass-parameter spec: profile name, key in true_params[0][i], display label.
MASS_PARAMS = [
    (0, "theta_E",  r"$\theta_E$"),
    (0, "gamma",    r"$\gamma_{\mathrm{epl}}$"),
    (0, "e1",       r"$\epsilon_{\mathrm{epl},1}$"),
    (0, "e2",       r"$\epsilon_{\mathrm{epl},2}$"),
    (0, "center_x", r"$x_{\mathrm{epl}}$"),
    (0, "center_y", r"$y_{\mathrm{epl}}$"),
    (1, "gamma1",   r"$\gamma_{\mathrm{ext},1}$"),
    (1, "gamma2",   r"$\gamma_{\mathrm{ext},2}$"),
]


def main():
    bij = make_bijector(build_prior())

    # Per-parameter values across all converged systems.
    zscores = {label: [] for _, _, label in MASS_PARAMS}
    truths = {label: [] for _, _, label in MASS_PARAMS}
    medians = {label: [] for _, _, label in MASS_PARAMS}
    lo_errs = {label: [] for _, _, label in MASS_PARAMS}
    hi_errs = {label: [] for _, _, label in MASS_PARAMS}
    converged_systems = []
    skipped = []

    for sim_num in SIM_NUMS:
        save_dir = os.path.join(
            SYSTEMS_DIR, f"vela{sim_num}_cam{CAM}_a0.500_f814w"
        )
        samples_path = os.path.join(save_dir, "mclmc_samples.npy")
        truth_path = os.path.join(save_dir, "true_params")
        if not (os.path.exists(samples_path) and os.path.exists(truth_path)):
            skipped.append((sim_num, "missing files"))
            continue

        mclmc_samples = jnp.asarray(np.load(samples_path))  # (chains, samples, dim)
        with open(truth_path, "rb") as f:
            true_params = pickle.load(f)

        rhat = blackjax.diagnostics.potential_scale_reduction(
            mclmc_samples, chain_axis=0, sample_axis=1
        )
        max_rhat = float(jnp.max(rhat))

        if not np.isfinite(max_rhat) or max_rhat >= RHAT_THRESHOLD:
            skipped.append((sim_num, f"R-hat={max_rhat:.3f}"))
            print(f"[skip] vela{sim_num}: max R-hat = {max_rhat:.3f}")
            continue

        print(f"[ok]   vela{sim_num}: max R-hat = {max_rhat:.3f}")
        converged_systems.append(sim_num)

        med_z = jnp.median(mclmc_samples, axis=(0, 1))
        lo_z = jnp.quantile(mclmc_samples, 0.159, axis=(0, 1))
        hi_z = jnp.quantile(mclmc_samples, 1 - 0.159, axis=(0, 1))

        med_x = bij.forward(list(med_z.T))
        lo_x = bij.forward(list(lo_z.T))
        hi_x = bij.forward(list(hi_z.T))

        for prof_idx, key, label in MASS_PARAMS:
            truth = float(jnp.squeeze(true_params[0][prof_idx][key]))
            med = float(jnp.squeeze(med_x[0][prof_idx][key]))
            lo = float(jnp.squeeze(lo_x[0][prof_idx][key]))
            hi = float(jnp.squeeze(hi_x[0][prof_idx][key]))
            zscores[label].append(float(stdev_calc(truth, med, lo, hi)))
            truths[label].append(truth)
            medians[label].append(med)
            lo_errs[label].append(med - lo)
            hi_errs[label].append(hi - med)

    if not converged_systems:
        print("No systems converged below the R-hat threshold; nothing to plot.")
        return

    x_positions = np.arange(len(converged_systems))
    x_labels = [f"vela{s}" for s in converged_systems]

    def _grid():
        n_params = len(MASS_PARAMS)
        ncols = 4
        nrows = int(np.ceil(n_params / ncols))
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True
        )
        return fig, axs.flatten()

    # ------------------------------------------------------------------
    # Plot 1: asymmetric z-scores
    # ------------------------------------------------------------------
    fig_z, axs_z = _grid()
    for i, (_, _, label) in enumerate(MASS_PARAMS):
        ax = axs_z[i]
        zs = np.asarray(zscores[label])
        ax.axhspan(-1, 1, color="gray", alpha=0.15, zorder=0)
        ax.axhspan(-2, 2, color="gray", alpha=0.08, zorder=0)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.scatter(x_positions, zs, color="C0", zorder=3)
        ax.set_title(label)
        ax.set_ylabel(r"$(\mathrm{truth} - \mathrm{median}) / \sigma$")
        ymax = max(3.0, float(np.nanmax(np.abs(zs))) * 1.1) if zs.size else 3.0
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    for j in range(len(MASS_PARAMS), len(axs_z)):
        axs_z[j].axis("off")
    fig_z.suptitle(
        f"Truth z-scores for mass parameters across converged Vela systems "
        f"(R-hat < {RHAT_THRESHOLD}, N = {len(converged_systems)})"
    )
    fig_z.tight_layout()
    fig_z.savefig(ZSCORE_OUTPUT_PATH, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Plot 2: physical-space residuals (truth - median) with asymmetric
    # 1-sigma error bars (from the 16th/84th percentiles).
    # ------------------------------------------------------------------
    fig_r, axs_r = _grid()
    for i, (_, _, label) in enumerate(MASS_PARAMS):
        ax = axs_r[i]
        truth = np.asarray(truths[label])
        med = np.asarray(medians[label])
        lo = np.asarray(lo_errs[label])
        hi = np.asarray(hi_errs[label])
        diff = truth - med
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.errorbar(
            x_positions, diff, yerr=[lo, hi],
            fmt="o", color="C0", ecolor="C0", capsize=3, zorder=3,markersize=2
        )
        ax.set_title(label)
        ax.set_ylabel(r"$\mathrm{truth} - \mathrm{median}$")
        # Symmetrize the y-range around zero so the y=0 line stays centered.
        y_extent = np.concatenate([np.abs(diff - lo), np.abs(diff + hi), [0.0]])
        ymax = float(np.nanmax(y_extent)) * 1.1 if y_extent.size else 1.0
        ymax = max(ymax, 1e-6)
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    for j in range(len(MASS_PARAMS), len(axs_r)):
        axs_r[j].axis("off")
    fig_r.suptitle(
        f"Truth - posterior median for mass parameters with 1-sigma error bars "
        f"(R-hat < {RHAT_THRESHOLD}, N = {len(converged_systems)})"
    )
    fig_r.tight_layout()
    fig_r.savefig(RESIDUAL_OUTPUT_PATH, dpi=150, bbox_inches="tight")

    print(f"\nSaved z-score figure to {ZSCORE_OUTPUT_PATH}")
    print(f"Saved residual figure to {RESIDUAL_OUTPUT_PATH}")
    print(f"Converged ({len(converged_systems)}): {converged_systems}")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
