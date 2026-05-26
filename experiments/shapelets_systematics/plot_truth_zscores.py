"""
Quick script to summarize MCLMC results across the Vela-based simulated
systems, restricted to mass parameters.

Each Vela source has multiple simulated systems with different lens
parameters (folders named ``vela{XX}_cam{CC}_rep{RR}_a0.500_f814w``).
Input system folders live under ``data/vela_sim_systems``. MCLMC samples for
different shapelet ``n_max`` values are stored under matching system folders in
``results/shapelets_systematics``. Convergence is judged on each run individually via
R-hat < 1.1, and all reps for the same Vela source share an x-axis column
on the plots. Marker shape encodes the shapelet ``n_max``, and color
encodes the rep.

Outputs four figures, each with one subplot per mass parameter:

  1. ``mass_truth_zscores.png``  - the asymmetric z-score
     ``(truth - median) / sigma``, where ``sigma`` is the upper or lower
     1-sigma half-width of the marginal posterior depending on which side
     of the median the truth falls.
  2. ``mass_truth_residuals.png`` - the physical-space residual
     ``truth - median`` with asymmetric 1-sigma error bars taken from the
     16th/84th posterior percentiles.
  3. ``mass_truth_percent_errors.png`` - the signed percent error
     ``100 * (median - truth) / abs(truth)`` with asymmetric 1-sigma error
     bars scaled by ``abs(truth)``.
  4. ``mass_abs_zscore_vs_nmax.png`` - violin plots of the absolute
     z-score distribution at each shapelet order.
"""

import os
import pickle
import re
from collections import defaultdict
from os.path import expanduser

home = expanduser("~/")

import jax
import numpy as np
from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
import blackjax

import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import bijectors as tfb

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYSTEMS_DIR = os.path.join(home, "GIGALens-Code", "data", "vela_sim_systems")
RESULTS_DIR = os.path.join(home, "GIGALens-Code", "results", "shapelets_systematics")
CAM = "12"
# Marker shape per shapelet n_max. Keys = n_max value, values = matplotlib marker.
N_MAX_MARKERS = {30: "o", 20:"p", 15: "s", 10: "^"} # 
RHAT_THRESHOLD = 1.1  # convergence cutoff
COLUMN_JITTER = 0.5   # full horizontal width allotted to a single source column
ZSCORE_OUTPUT_PATH = os.path.join(RESULTS_DIR, "mass_truth_zscores.png")
RESIDUAL_OUTPUT_PATH = os.path.join(RESULTS_DIR, "mass_truth_residuals.png")
PERCENT_ERROR_OUTPUT_PATH = os.path.join(RESULTS_DIR, "mass_truth_percent_errors.png")
ABS_ZSCORE_NMAX_OUTPUT_PATH = os.path.join(RESULTS_DIR, "mass_abs_zscore_vs_nmax.png")

# Folder name pattern: vela{vela}_cam{cam}_rep{rep}_a{a}_{filter}
FOLDER_PATTERN = re.compile(
    r"^vela(?P<vela>\d+)_cam(?P<cam>\d+)_rep(?P<rep>\d+)_a[\d.]+_[a-z0-9]+$"
)
NMAX_SUBDIR_PATTERN = re.compile(r"^n_max(?P<n>\d+)$")


# ---------------------------------------------------------------------------
# Build the prior + bijector exactly as in vela_system_model() from
# sim_system_complex.ipynb. The shapelet n_max only affects the source-light
# amp_names that are solved for via lstsq; the prior parameters that the
# MCLMC samples actually live in (mass + lens light + source beta/center)
# are independent of n_max, so a single bijector works for both runs.
# ---------------------------------------------------------------------------
def build_prior():
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


def discover_runs():
    """Yield (vela, rep, n_max, run_dir, system_dir) for every result run."""
    runs = []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        result_system_dir = os.path.join(RESULTS_DIR, entry)
        if not os.path.isdir(result_system_dir):
            continue
        match = FOLDER_PATTERN.match(entry)
        if match is None or match.group("cam") != CAM:
            continue
        vela = match.group("vela")
        rep = match.group("rep")
        system_dir = os.path.join(SYSTEMS_DIR, entry)
        for sub in sorted(os.listdir(result_system_dir)):
            sub_match = NMAX_SUBDIR_PATTERN.match(sub)
            if sub_match is None:
                continue
            run_dir = os.path.join(result_system_dir, sub)
            if not os.path.isdir(run_dir):
                continue
            n_max = int(sub_match.group("n"))
            runs.append((vela, rep, n_max, run_dir, system_dir))
    runs.sort(key=lambda t: (t[0], t[1], t[2]))
    return runs


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bij = make_bijector(build_prior())

    # Per-parameter values keyed by (vela_num, rep_num, n_max).
    zscores = {label: {} for _, _, label in MASS_PARAMS}
    truths = {label: {} for _, _, label in MASS_PARAMS}
    medians = {label: {} for _, _, label in MASS_PARAMS}
    lo_errs = {label: {} for _, _, label in MASS_PARAMS}
    hi_errs = {label: {} for _, _, label in MASS_PARAMS}

    converged = []  # list of (vela, rep, n_max)
    skipped = []

    for vela, rep, n_max, run_dir, system_dir in discover_runs():
        tag = f"vela{vela}_rep{rep}_n_max{n_max}"
        if n_max not in N_MAX_MARKERS:
            skipped.append((tag, f"no marker assigned for n_max={n_max}"))
            continue

        samples_path = os.path.join(run_dir, "mclmc_samples.npy")
        truth_path = os.path.join(system_dir, "true_params")
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
        run_key = (vela, rep, n_max)
        converged.append(run_key)

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
            zscores[label][run_key] = float(stdev_calc(truth, med, lo, hi))
            truths[label][run_key] = truth
            medians[label][run_key] = med
            lo_errs[label][run_key] = med - lo
            hi_errs[label][run_key] = hi - med

    if not converged:
        print("No runs converged below the R-hat threshold; nothing to plot.")
        return

    # --- Group converged systems by Vela source for x-axis layout ---
    reps_per_vela = defaultdict(set)
    for vela, rep, _ in converged:
        reps_per_vela[vela].add(rep)
    velas = sorted(reps_per_vela)
    vela_x = {vela: i for i, vela in enumerate(velas)}

    def rep_offsets(reps):
        """Deterministic horizontal offsets for reps within a column."""
        reps = sorted(reps)
        if len(reps) == 1:
            return {reps[0]: 0.0}
        offs = np.linspace(-COLUMN_JITTER / 2, COLUMN_JITTER / 2, len(reps))
        return dict(zip(reps, offs))

    offsets_per_vela = {v: rep_offsets(reps_per_vela[v]) for v in velas}
    all_reps = sorted({rep for _, rep, _ in converged})
    rep_color = {
        rep: plt.get_cmap("tab10")(i % 10) for i, rep in enumerate(all_reps)
    }
    n_max_list = sorted({n for _, _, n in converged})
    source_color = {
        vela: plt.get_cmap("tab20")(i % 20) for i, vela in enumerate(velas)
    }

    def x_for(run_key):
        vela, rep, _ = run_key
        return vela_x[vela] + offsets_per_vela[vela][rep]

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
        rep_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=rep_color[r], markeredgecolor=rep_color[r],
                label=f"rep{r}",
            )
            for r in all_reps
        ]
        nmax_handles = [
            Line2D(
                [0], [0], marker=N_MAX_MARKERS[n], linestyle="none",
                markerfacecolor="black", markeredgecolor="black",
                label=f"n_max={n}",
            )
            for n in n_max_list
        ]
        fig.legend(
            handles=rep_handles + nmax_handles,
            loc="lower center", ncol=len(rep_handles) + len(nmax_handles),
            frameon=False, bbox_to_anchor=(0.5, -0.02),
        )

    def add_source_legend(fig):
        from matplotlib.lines import Line2D
        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=source_color[v], markeredgecolor=source_color[v],
                label=f"vela{v}",
            )
            for v in velas
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=min(len(handles), 6),
            frameon=False, bbox_to_anchor=(0.5, -0.04),
        )

    x_ticks = list(vela_x.values())
    x_labels = [f"vela{v}" for v in velas]
    column_boundaries = [t + 0.5 for t in x_ticks[:-1]]

    def add_column_dividers(ax):
        for x in column_boundaries:
            ax.axvline(x, color="lightgray", lw=0.8, zorder=0)
        if x_ticks:
            ax.set_xlim(x_ticks[0] - 0.5, x_ticks[-1] + 0.5)

    # Group converged keys by n_max so we can use one marker per scatter call.
    keys_by_nmax = defaultdict(list)
    for run_key in converged:
        keys_by_nmax[run_key[2]].append(run_key)

    # ------------------------------------------------------------------
    # Plot 1: asymmetric z-scores
    # ------------------------------------------------------------------
    fig_z, axs_z = _grid()
    for i, (_, _, label) in enumerate(MASS_PARAMS):
        ax = axs_z[i]
        ax.axhspan(-1, 1, color="gray", alpha=0.15, zorder=0)
        ax.axhspan(-2, 2, color="gray", alpha=0.08, zorder=0)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        add_column_dividers(ax)
        all_ys = []
        for n_max in n_max_list:
            keys = keys_by_nmax[n_max]
            xs = np.asarray([x_for(k) for k in keys])
            ys = np.asarray([zscores[label][k] for k in keys])
            cs = [rep_color[k[1]] for k in keys]
            ax.scatter(
                xs, ys, color=cs, marker=N_MAX_MARKERS[n_max], s=20, zorder=3,
            )
            all_ys.append(ys)
        all_ys = np.concatenate(all_ys) if all_ys else np.empty(0)
        ax.set_title(label)
        ax.set_ylabel(r"$(\mathrm{truth} - \mathrm{median}) / \sigma$")
        ymax = max(3.0, float(np.nanmax(np.abs(all_ys))) * 1.1) if all_ys.size else 3.0
        ax.set_ylim(-ymax, ymax)
        # ysc = matplotlib.scale.SymmetricalLogScale(axis, *, base=10, linthresh=2, subs=None, linscale=1)
        ax.set_yscale('asinh')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    for j in range(len(MASS_PARAMS), len(axs_z)):
        axs_z[j].axis("off")
    fig_z.suptitle(
        f"Truth z-scores for mass parameters across converged Vela systems "
        f"(R-hat < {RHAT_THRESHOLD}, "
        f"N_sources = {len(velas)}, N_runs = {len(converged)})"
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
        ax.axhline(0, color="black", lw=0.8, ls="--")
        add_column_dividers(ax)
        all_diff_lo = []
        all_diff_hi = []
        for n_max in n_max_list:
            marker = N_MAX_MARKERS[n_max]
            for run_key in keys_by_nmax[n_max]:
                vela, rep, _ = run_key
                diff = truths[label][run_key] - medians[label][run_key]
                lo = lo_errs[label][run_key]
                hi = hi_errs[label][run_key]
                color = rep_color[rep]
                ax.errorbar(
                    x_for(run_key), diff, yerr=[[lo], [hi]],
                    fmt=marker, color=color, ecolor=color,
                    capsize=3, zorder=3, markersize=4,
                )
                all_diff_lo.append(diff - lo)
                all_diff_hi.append(diff + hi)
        ax.set_title(label)
        ax.set_ylabel(r"$\mathrm{truth} - \mathrm{median}$")
        if all_diff_lo:
            extents = np.abs(np.concatenate([all_diff_lo, all_diff_hi]))
            ymax = float(np.nanmax(extents)) * 1.1
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
        f"N_sources = {len(velas)}, N_runs = {len(converged)})"
    )
    fig_r.tight_layout()
    add_legend(fig_r)
    fig_r.savefig(RESIDUAL_OUTPUT_PATH, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Plot 3: signed percent errors, with the same posterior 1-sigma
    # intervals expressed as a percent of |truth|.
    # ------------------------------------------------------------------
    fig_p, axs_p = _grid()
    for i, (_, _, label) in enumerate(MASS_PARAMS):
        ax = axs_p[i]
        ax.axhline(0, color="black", lw=0.8, ls="--")
        add_column_dividers(ax)
        all_pct_lo = []
        all_pct_hi = []
        for n_max in n_max_list:
            marker = N_MAX_MARKERS[n_max]
            for run_key in keys_by_nmax[n_max]:
                vela, rep, _ = run_key
                truth = truths[label][run_key]
                denom = abs(truth)
                if denom == 0:
                    print(
                        f"[skip] percent error undefined for {label} "
                        f"{run_key}: truth is zero"
                    )
                    continue
                pct = 100.0 * (medians[label][run_key] - truth) / denom
                lo = 100.0 * lo_errs[label][run_key] / denom
                hi = 100.0 * hi_errs[label][run_key] / denom
                color = rep_color[rep]
                ax.errorbar(
                    x_for(run_key), pct, yerr=[[lo], [hi]],
                    fmt=marker, color=color, ecolor=color,
                    capsize=3, zorder=3, markersize=4,
                )
                all_pct_lo.append(pct - lo)
                all_pct_hi.append(pct + hi)
        ax.set_title(label)
        ax.set_ylabel(r"$100(\mathrm{median} - \mathrm{truth}) / |\mathrm{truth}|$")
        if all_pct_lo:
            extents = np.abs(np.concatenate([all_pct_lo, all_pct_hi]))
            ymax = float(np.nanmax(extents)) * 1.1
        else:
            ymax = 1.0
        ax.set_ylim(-max(ymax, 1e-6), max(ymax, 1e-6))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    for j in range(len(MASS_PARAMS), len(axs_p)):
        axs_p[j].axis("off")
    fig_p.suptitle(
        f"Signed percent error for mass parameters with 1-sigma error bars "
        f"(R-hat < {RHAT_THRESHOLD}, "
        f"N_sources = {len(velas)}, N_runs = {len(converged)})"
    )
    fig_p.tight_layout()
    add_legend(fig_p)
    fig_p.savefig(PERCENT_ERROR_OUTPUT_PATH, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Plot 4: shapelet order vs absolute z-score. Use violin plots rather
    # than per-run scatter so the distribution at each n_max is readable.
    # ------------------------------------------------------------------
    rng_null = np.random.default_rng(12345)
    null_abs_z = np.abs(rng_null.normal(size=20000))
    if len(n_max_list) > 1:
        nmax_spacing = float(np.median(np.diff(n_max_list)))
    else:
        nmax_spacing = 5.0
    null_x = max(n_max_list) + nmax_spacing
    violin_width = min(2.4, 0.45 * nmax_spacing)

    fig_a, axs_a = _grid()
    for i, (_, _, label) in enumerate(MASS_PARAMS):
        ax = axs_a[i]
        all_abs_z = [
            np.asarray([abs(zscores[label][k]) for k in keys_by_nmax[n_max]])
            for n_max in n_max_list
        ]
        nonempty = [ys for ys in all_abs_z if ys.size]
        violin = ax.violinplot(
            all_abs_z,
            positions=n_max_list,
            widths=violin_width,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body in violin["bodies"]:
            body.set_facecolor("C0")
            body.set_edgecolor("C0")
            body.set_alpha(0.35)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            violin[part].set_color("C0")
            violin[part].set_linewidth(1.0)

        # Add a lightly jittered dot layer so single-point n_max values and
        # sample counts remain visible without dominating the violins.
        rng = np.random.default_rng(0)
        for n_max, ys in zip(n_max_list, all_abs_z):
            xs = n_max + rng.uniform(-0.45, 0.45, size=ys.size)
            ax.scatter(
                xs, ys, color="black", s=6, alpha=0.22, linewidths=0,
                zorder=3,
            )

        # Calibrated null: if inference were unbiased and uncertainties were
        # calibrated, |z| would be distributed as |N(0, 1)|. Draw it as a
        # grey hatched right-half violin so it is visibly not another dataset.
        null_violin = ax.violinplot(
            [null_abs_z],
            positions=[null_x],
            widths=violin_width,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in null_violin["bodies"]:
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.maximum(verts[:, 0], null_x)
            body.set_facecolor("0.7")
            body.set_edgecolor("0.25")
            body.set_alpha(0.35)
            body.set_hatch("///")
            body.set_linewidth(1.0)
        null_violin["cmedians"].set_color("0.25")
        null_violin["cmedians"].set_linewidth(1.2)

        all_abs_z = np.concatenate(all_abs_z) if all_abs_z else np.empty(0)
        ax.axhline(1, color="gray", lw=0.8, ls=":", zorder=0)
        ax.axhline(2, color="gray", lw=0.8, ls="--", zorder=0)
        ax.set_title(label)
        ax.set_ylabel(r"$|(\mathrm{truth} - \mathrm{median}) / \sigma|$")
        ax.set_xlabel(r"shapelet $n_\mathrm{max}$")
        ax.set_xticks(n_max_list + [null_x])
        ax.set_xticklabels([str(n) for n in n_max_list] + ["null\n|N(0,1)|"])
        if nonempty:
            ymax = max(3.0, float(np.nanmax(all_abs_z)) * 1.1)
        else:
            ymax = 3.0
        ax.set_ylim(0, ymax)
        ax.set_xlim(min(n_max_list) - nmax_spacing * 0.6, null_x + nmax_spacing * 0.6)
        ax.set_yscale("asinh")
    for j in range(len(MASS_PARAMS), len(axs_a)):
        axs_a[j].axis("off")
    fig_a.suptitle(
        f"Absolute truth z-score vs. shapelet order for mass parameters "
        f"(R-hat < {RHAT_THRESHOLD}, "
        f"N_sources = {len(velas)}, N_runs = {len(converged)})"
    )
    fig_a.tight_layout()
    fig_a.savefig(ABS_ZSCORE_NMAX_OUTPUT_PATH, dpi=150, bbox_inches="tight")

    n_runs_per_nmax = {n: len(keys_by_nmax[n]) for n in n_max_list}
    print(f"\nSaved z-score figure to {ZSCORE_OUTPUT_PATH}")
    print(f"Saved residual figure to {RESIDUAL_OUTPUT_PATH}")
    print(f"Saved percent-error figure to {PERCENT_ERROR_OUTPUT_PATH}")
    print(f"Saved abs-zscore-vs-nmax figure to {ABS_ZSCORE_NMAX_OUTPUT_PATH}")
    print(
        f"Converged: {len(converged)} runs across {len(velas)} sources "
        f"(per n_max: {n_runs_per_nmax})"
    )
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for tag, reason in skipped:
            print(f"  {tag}: {reason}")


if __name__ == "__main__":
    main()
