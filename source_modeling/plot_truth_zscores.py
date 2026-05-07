"""
Quick script to summarize MCLMC results across the Vela-based simulated
systems, restricted to mass parameters.

Each Vela source has multiple simulated systems with different lens
parameters (folders named ``vela{XX}_cam{CC}_rep{RR}_a0.500_f814w``).
Convergence is judged on the joint MCLMC samples via R-hat < 1.1, and
all reps with the same Vela source share an x-axis column on the plots.

Outputs two figures, each with one subplot per mass parameter:

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
import re
import sys
from collections import defaultdict
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
N_MAX = 10            # shapelet n_max used in the notebook
RHAT_THRESHOLD = 1.1  # convergence cutoff
COLUMN_JITTER = 0.5   # full horizontal width allotted to a single source column
ZSCORE_OUTPUT_PATH = os.path.join(SYSTEMS_DIR, "mass_truth_zscores.png")
RESIDUAL_OUTPUT_PATH = os.path.join(SYSTEMS_DIR, "mass_truth_residuals.png")

# Folder name pattern: vela{vela}_cam{cam}_rep{rep}_a{a}_{filter}
FOLDER_PATTERN = re.compile(
    r"^vela(?P<vela>\d+)_cam(?P<cam>\d+)_rep(?P<rep>\d+)_a[\d.]+_[a-z0-9]+$"
)


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


def discover_systems():
    """Return [(vela_num, rep_num, folder_path), ...] sorted by (vela, rep)."""
    systems = []
    for entry in sorted(os.listdir(SYSTEMS_DIR)):
        full_path = os.path.join(SYSTEMS_DIR, entry)
        if not os.path.isdir(full_path):
            continue
        match = FOLDER_PATTERN.match(entry)
        if match is None or match.group("cam") != CAM:
            continue
        systems.append(
            (match.group("vela"), match.group("rep"), full_path)
        )
    systems.sort(key=lambda t: (t[0], t[1]))
    return systems


def main():
    bij = make_bijector(build_prior())

    # Per-parameter values keyed by (vela_num, rep_num).
    zscores = {label: {} for _, _, label in MASS_PARAMS}
    truths = {label: {} for _, _, label in MASS_PARAMS}
    medians = {label: {} for _, _, label in MASS_PARAMS}
    lo_errs = {label: {} for _, _, label in MASS_PARAMS}
    hi_errs = {label: {} for _, _, label in MASS_PARAMS}

    converged = []  # list of (vela_num, rep_num)
    skipped = []

    for vela, rep, save_dir in discover_systems():
        tag = f"vela{vela}_rep{rep}"
        samples_path = os.path.join(save_dir, "mclmc_samples.npy")
        truth_path = os.path.join(save_dir, "true_params")
        if not (os.path.exists(samples_path) and os.path.exists(truth_path)):
            skipped.append((tag, "missing files"))
            continue

        mclmc_samples = jnp.asarray(np.load(samples_path))  # (chains, samples, dim)
        with open(truth_path, "rb") as f:
            true_params = pickle.load(f)

        rhat = blackjax.diagnostics.potential_scale_reduction(
            mclmc_samples, chain_axis=0, sample_axis=1
        )
        max_rhat = float(jnp.max(rhat))

        if not np.isfinite(max_rhat) or max_rhat >= RHAT_THRESHOLD:
            skipped.append((tag, f"R-hat={max_rhat:.3f}"))
            print(f"[skip] {tag}: max R-hat = {max_rhat:.3f}")
            continue

        print(f"[ok]   {tag}: max R-hat = {max_rhat:.3f}")
        converged.append((vela, rep))

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
            zscores[label][(vela, rep)] = float(stdev_calc(truth, med, lo, hi))
            truths[label][(vela, rep)] = truth
            medians[label][(vela, rep)] = med
            lo_errs[label][(vela, rep)] = med - lo
            hi_errs[label][(vela, rep)] = hi - med

    if not converged:
        print("No systems converged below the R-hat threshold; nothing to plot.")
        return

    # --- Group converged systems by Vela source for x-axis layout ---
    reps_per_vela = defaultdict(list)
    for vela, rep in converged:
        reps_per_vela[vela].append(rep)
    velas = sorted(reps_per_vela)
    vela_x = {vela: i for i, vela in enumerate(velas)}

    # Deterministic horizontal offsets for reps within a column.
    def rep_offsets(reps):
        reps = sorted(set(reps))
        if len(reps) == 1:
            return {reps[0]: 0.0}
        # Spread reps evenly across +/- COLUMN_JITTER/2.
        offs = np.linspace(-COLUMN_JITTER / 2, COLUMN_JITTER / 2, len(reps))
        return dict(zip(reps, offs))

    offsets_per_vela = {v: rep_offsets(reps_per_vela[v]) for v in velas}
    all_reps = sorted({rep for _, rep in converged})
    rep_color = {
        rep: plt.get_cmap("tab10")(i % 10) for i, rep in enumerate(all_reps)
    }

    def system_xy(label, source):
        """Return parallel arrays (x, y, color) for converged points of `label`.
        `source` is one of the per-parameter dicts above."""
        xs, ys, cs = [], [], []
        for (vela, rep), val in source[label].items():
            xs.append(vela_x[vela] + offsets_per_vela[vela][rep])
            ys.append(val)
            cs.append(rep_color[rep])
        return np.asarray(xs), np.asarray(ys), cs

    def _grid():
        n_params = len(MASS_PARAMS)
        ncols = 4
        nrows = int(np.ceil(n_params / ncols))
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True
        )
        return fig, axs.flatten()

    def add_legend(fig):
        from matplotlib.lines import Line2D
        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=rep_color[r], markeredgecolor=rep_color[r],
                label=f"rep{r}",
            )
            for r in all_reps
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=len(all_reps),
            frameon=False, bbox_to_anchor=(0.5, -0.02),
        )

    x_ticks = list(vela_x.values())
    x_labels = [f"vela{v}" for v in velas]
    # Vertical dividers go between columns (half-integer positions).
    column_boundaries = [t + 0.5 for t in x_ticks[:-1]]

    def add_column_dividers(ax):
        for x in column_boundaries:
            ax.axvline(x, color="lightgray", lw=0.8, zorder=0)
        if x_ticks:
            ax.set_xlim(x_ticks[0] - 0.5, x_ticks[-1] + 0.5)

    # ------------------------------------------------------------------
    # Plot 1: asymmetric z-scores
    # ------------------------------------------------------------------
    fig_z, axs_z = _grid()
    for i, (_, _, label) in enumerate(MASS_PARAMS):
        ax = axs_z[i]
        xs, ys, cs = system_xy(label, zscores)
        ax.axhspan(-1, 1, color="gray", alpha=0.15, zorder=0)
        ax.axhspan(-2, 2, color="gray", alpha=0.08, zorder=0)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        add_column_dividers(ax)
        ax.scatter(xs, ys, color=cs, zorder=3, s=20)
        ax.set_title(label)
        ax.set_ylabel(r"$(\mathrm{truth} - \mathrm{median}) / \sigma$")
        ymax = max(3.0, float(np.nanmax(np.abs(ys))) * 1.1) if ys.size else 3.0
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    for j in range(len(MASS_PARAMS), len(axs_z)):
        axs_z[j].axis("off")
    fig_z.suptitle(
        f"Truth z-scores for mass parameters across converged Vela systems "
        f"(R-hat < {RHAT_THRESHOLD}, "
        f"N_sources = {len(velas)}, N_systems = {len(converged)})"
    )
    fig_z.tight_layout()
    add_legend(fig_z)
    fig_z.savefig(ZSCORE_OUTPUT_PATH, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Plot 2: physical-space residuals (truth - median) with asymmetric
    # 1-sigma error bars (from the 16th/84th percentiles).
    # ------------------------------------------------------------------
    fig_r, axs_r = _grid()
    for i, (_, _, label) in enumerate(MASS_PARAMS):
        ax = axs_r[i]
        truth_vals = np.asarray([truths[label][k] for k in zscores[label]])
        med_vals = np.asarray([medians[label][k] for k in zscores[label]])
        lo = np.asarray([lo_errs[label][k] for k in zscores[label]])
        hi = np.asarray([hi_errs[label][k] for k in zscores[label]])
        diff = truth_vals - med_vals
        xs = []
        cs = []
        for (vela, rep) in zscores[label]:
            xs.append(vela_x[vela] + offsets_per_vela[vela][rep])
            cs.append(rep_color[rep])
        xs = np.asarray(xs)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        add_column_dividers(ax)
        # Plot one errorbar per point so each rep can carry its own color.
        for x, d, l, h, c in zip(xs, diff, lo, hi, cs):
            ax.errorbar(
                x, d, yerr=[[l], [h]],
                fmt="o", color=c, ecolor=c, capsize=3, zorder=3, markersize=3,
            )
        ax.set_title(label)
        ax.set_ylabel(r"$\mathrm{truth} - \mathrm{median}$")
        if diff.size:
            y_extent = np.concatenate([np.abs(diff - lo), np.abs(diff + hi), [0.0]])
            ymax = float(np.nanmax(y_extent)) * 1.1
        else:
            ymax = 1.0
        ax.set_ylim(-max(ymax, 1e-6), max(ymax, 1e-6))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    for j in range(len(MASS_PARAMS), len(axs_r)):
        axs_r[j].axis("off")
    fig_r.suptitle(
        f"Truth - posterior median for mass parameters with 1-sigma error bars "
        f"(R-hat < {RHAT_THRESHOLD}, "
        f"N_sources = {len(velas)}, N_systems = {len(converged)})"
    )
    fig_r.tight_layout()
    add_legend(fig_r)
    fig_r.savefig(RESIDUAL_OUTPUT_PATH, dpi=150, bbox_inches="tight")

    print(f"\nSaved z-score figure to {ZSCORE_OUTPUT_PATH}")
    print(f"Saved residual figure to {RESIDUAL_OUTPUT_PATH}")
    print(
        f"Converged: {len(converged)} systems across {len(velas)} sources "
        f"({velas})"
    )
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for tag, reason in skipped:
            print(f"  {tag}: {reason}")


if __name__ == "__main__":
    main()
