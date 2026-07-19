"""Stage-specific *debug* diagnostics (e.g. MCLMC tuning histories).

Unlike :mod:`.convergence` (which works off a finished posterior), these plots
visualize a stage's *internal* run history — the kind of thing you capture with
``MCLMCStage(..., debug=True)`` to see *where* sampling went wrong. Because each
inference algorithm fails differently, every stage gets its own plotter,
registered by stage-class name and dispatched through
:func:`plot_stage_diagnostics`.

Adding diagnostics for a new stage:

1. Capture the debug arrays in the stage's ``run`` (under a ``debug`` flag)
   into ``StageResult.diagnostics``, and expose any plot-relevant config via
   ``InferenceStage.diagnostics_config``.
2. Write a plotter here and decorate it with
   ``@register_diagnostic_plotter("YourStage")``.

The plotter receives a
:class:`~gigalens_research.inference_utils.pipeline.StageDiagnostics` (arrays +
config + ctx) and returns a Matplotlib ``Figure``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .labels import z_column_labels

# Registry: stage-class name -> plotter(StageDiagnostics, **kwargs) -> Figure.
_DIAGNOSTIC_PLOTTERS: Dict[str, Callable[..., Figure]] = {}


def register_diagnostic_plotter(stage_class: str) -> Callable[[Callable], Callable]:
    """Register ``fn`` as the diagnostic plotter for ``stage_class`` (the stage
    class *name*, e.g. ``"MCLMCStage"``). See module docstring."""

    def deco(fn: Callable[..., Figure]) -> Callable[..., Figure]:
        _DIAGNOSTIC_PLOTTERS[stage_class] = fn
        return fn

    return deco


def has_diagnostic_plotter(stage_class: str) -> bool:
    return stage_class in _DIAGNOSTIC_PLOTTERS


def plot_stage_diagnostics(diagnostics, **kwargs) -> Figure:
    """Render a stage's debug diagnostics, dispatching on its stage class.

    ``diagnostics`` is the
    :class:`~gigalens_research.inference_utils.pipeline.StageDiagnostics`
    returned by ``Pipeline.diagnostics(stage)`` or ``diagnostics_from_disk``.

    Raises a clear error if the stage was not run with ``debug=True`` (no
    captured arrays) or if no plotter is registered for its class.
    """
    if not getattr(diagnostics, "arrays", None):
        raise ValueError(
            f"Stage {getattr(diagnostics, 'stage_name', '?')!r} has no captured "
            "diagnostics. Re-run the stage with debug=True (e.g. "
            "MCLMCStage(..., debug=True)) to record them."
        )
    plotter = _DIAGNOSTIC_PLOTTERS.get(diagnostics.stage_class)
    if plotter is None:
        raise KeyError(
            f"No diagnostic plotter registered for stage class "
            f"{diagnostics.stage_class!r}. Available: "
            f"{sorted(_DIAGNOSTIC_PLOTTERS)}. Register one with "
            "@register_diagnostic_plotter(...)."
        )
    return plotter(diagnostics, **kwargs)


# ---------------------------------------------------------------------------
# MCLMC
# ---------------------------------------------------------------------------

def _tuning_boundaries(config: Dict) -> Tuple[int, int, int]:
    """Step indices where MCLMC's three tuning stages end."""
    nb = int(config.get("num_burnin_steps", 0))
    f1 = float(config.get("frac_tune1", 0.0))
    f2 = float(config.get("frac_tune2", 0.0))
    f3 = float(config.get("frac_tune3", 0.0))
    return int(f1 * nb), int((f1 + f2) * nb), int((f1 + f2 + f3) * nb)


@register_diagnostic_plotter("MCLMCStage")
def plot_mclmc_diagnostics(
    diagnostics,
    *,
    chain: int = 0,
    smooth: int = 30,
    figsize: Tuple[float, float] = (10, 9),
) -> Figure:
    """Five stacked panels of an MCLMC tuning run, vs. step:

    1. per-chain step size,
    2. per-chain trajectory length ``L``,
    3. inverse-mass-matrix eigenvalue spread (min/mean/max, log y), with the
       final output samples' covariance eigenvalue spread overlaid as horizontal
       dashed lines (min/mean/max),
    4. the per-step energy-error ratio ``xi`` for one chain (raw + smoothed),
    5. a success heatmap (green = finite step, red = NaN/blow-up).

    Vertical dashed lines mark the boundaries of MCLMC's three tuning stages
    (step size, mass matrix, ``L``); anything after the last line is sampling.

    ``chain`` selects which chain to show for the ``xi`` panel. Captured by
    ``MCLMCStage(..., debug=True)``.
    """
    arr = diagnostics.arrays
    stage1, stage2, stage3 = _tuning_boundaries(diagnostics.config)

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=figsize)
    ax_ss, ax_L, ax_eig, ax_xi, ax_nan = axs

    # 1. step size (transpose -> step on x, one line per chain)
    if "step_size" in arr:
        ax_ss.plot(np.asarray(arr["step_size"]).T)
    ax_ss.set_title("Chain-wise step size")
    ax_ss.set_yscale('log')
    ax_ss.set_ylabel("step size")

    # 2. trajectory length L
    if "L" in arr:
        ax_L.plot(np.asarray(arr["L"]).T)
    ax_L.set_title("Chain-wise L")
    ax_L.set_ylabel("L")

    # 3. inverse-mass-matrix eigenvalues (stored chain-0 only; replicated)
    if "inverse_mass_matrix" in arr:
        imm = np.asarray(arr["inverse_mass_matrix"])[0]  # (n_steps, dim, dim)
        # Symmetric PD covariance -> use eigvalsh (real, ascending).
        eig = np.linalg.eigvalsh(imm)  # (n_steps, dim)
        ax_eig.plot(eig.min(axis=1), label="min", color="tab:blue")
        ax_eig.plot(eig.mean(axis=1), label="mean", color="black")
        ax_eig.plot(eig.max(axis=1), label="max", color="tab:red")
        ax_eig.set_yscale("log")
        # Overlay the eigenvalue spread of the final output samples' covariance
        # as horizontal dashed lines (min/mean/max), matching the trace colors.
        # The inverse mass matrix targets this posterior covariance, so the
        # traces above should settle toward these levels.
        if "samples_cov" in arr:
            cov_eig = np.linalg.eigvalsh(np.asarray(arr["samples_cov"]))
            for val, color, lbl in (
                (cov_eig.min(), "tab:blue", "sample cov min"),
                (cov_eig.mean(), "black", "sample cov mean"),
                (cov_eig.max(), "tab:red", "sample cov max"),
            ):
                ax_eig.axhline(val, color=color, linestyle="--", linewidth=1, label=lbl)
        ax_eig.legend(fontsize=8)
    ax_eig.set_title("Inverse mass-matrix eigenvalues")
    ax_eig.set_ylabel("eigenvalue")

    # 4. xi (energy-error ratio) for one chain, raw + smoothed
    if "xi" in arr:
        xi = np.asarray(arr["xi"])
        c = chain % xi.shape[0]
        xi_c = xi[c]
        ax_xi.plot(xi_c, alpha=0.4, color="tab:blue")
        if smooth and smooth > 1 and xi_c.size >= smooth:
            kern = np.ones(smooth) / smooth
            ax_xi.plot(np.convolve(xi_c, kern, mode="same"), color="tab:blue")
        ax_xi.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_xi.set_yscale("log")
        ax_xi.set_ylabel(f"xi (chain {c})")
    ax_xi.set_title("Energy-error ratio")

    # 5. success / NaN heatmap (chains x steps), restricted to the tuning span
    if "nonan" in arr:
        nonan = np.asarray(arr["nonan"])
        upto = stage2 if stage2 > 0 else nonan.shape[1]
        ax_nan.imshow(
            nonan[:, :upto], aspect="auto", interpolation="none", cmap="RdYlGn",
            vmin=0, vmax=1,
        )
        ax_nan.set_ylabel("chain")
    ax_nan.set_title("Finite-step mask (green = ok, red = NaN)")
    ax_nan.set_xlabel("step")

    # Tuning-stage boundaries on every panel.
    for ax in axs:
        for x, color in ((stage1, "tab:red"), (stage2, "tab:blue"), (stage3, "tab:green")):
            if x > 0:
                ax.axvline(x, color=color, linestyle="--", linewidth=1)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MAMS (Metropolis-adjusted microcanonical sampler)
# ---------------------------------------------------------------------------

@register_diagnostic_plotter("MAMSStage")
def plot_mams_diagnostics(
    diagnostics,
    *,
    chain: int = 0,
    smooth: int = 30,
    figsize: Tuple[float, float] = (10, 11),
) -> Figure:
    """Six stacked panels of a MAMS tuning run, vs. step:

    1. per-chain step size,
    2. per-chain trajectory length ``L``,
    3. inverse-mass-matrix eigenvalue spread (min/mean/max, log y), with the
       final output samples' covariance eigenvalue spread overlaid as horizontal
       dashed lines,
    4. per-transition Metropolis acceptance rate (per-chain raw + cross-chain
       smoothed mean), with the dual-averaging target drawn as a dashed line,
    5. trajectory length in integrator steps (the shared deterministic Halton
       schedule),
    6. a success heatmap (green = finite step, red = NaN/blow-up).

    The MAMS analogue of :func:`plot_mclmc_diagnostics`: panels 1-3 and 6 are
    identical, but the unadjusted energy-error ratio ``xi`` is replaced by the
    two quantities MAMS actually tunes against -- the MH acceptance rate
    (panel 4, driven to ``target_acceptance`` by dual averaging) and the
    deterministic number of integrator steps per trajectory (panel 5).

    Vertical dashed lines mark the boundaries of the three tuning stages (step
    size, mass matrix, ``L``); anything after the last line is sampling.
    ``chain`` selects which chain to highlight for the integration-steps panel.
    Captured by ``MAMSStage(..., debug=True)``.
    """
    arr = diagnostics.arrays
    stage1, stage2, stage3 = _tuning_boundaries(diagnostics.config)
    target_acc = diagnostics.config.get("target_acceptance")

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=figsize)
    ax_ss, ax_L, ax_eig, ax_acc, ax_n, ax_nan = axs

    # 1. step size (transpose -> step on x, one line per chain)
    if "step_size" in arr:
        ax_ss.plot(np.asarray(arr["step_size"]).T)
    ax_ss.set_title("Chain-wise step size")
    ax_ss.set_yscale("log")
    ax_ss.set_ylabel("step size")

    # 2. trajectory length L
    if "L" in arr:
        ax_L.plot(np.asarray(arr["L"]).T)
    ax_L.set_title("Chain-wise L")
    ax_L.set_ylabel("L")

    # 3. inverse-mass-matrix eigenvalues (stored chain-0 only; replicated)
    if "inverse_mass_matrix" in arr:
        imm = np.asarray(arr["inverse_mass_matrix"])[0]  # (n_steps, dim, dim)
        eig = np.linalg.eigvalsh(imm)  # (n_steps, dim)
        ax_eig.plot(eig.min(axis=1), label="min", color="tab:blue")
        ax_eig.plot(eig.mean(axis=1), label="mean", color="black")
        ax_eig.plot(eig.max(axis=1), label="max", color="tab:red")
        ax_eig.set_yscale("log")
        if "samples_cov" in arr:
            cov_eig = np.linalg.eigvalsh(np.asarray(arr["samples_cov"]))
            for val, color, lbl in (
                (cov_eig.min(), "tab:blue", "sample cov min"),
                (cov_eig.mean(), "black", "sample cov mean"),
                (cov_eig.max(), "tab:red", "sample cov max"),
            ):
                ax_eig.axhline(val, color=color, linestyle="--", linewidth=1, label=lbl)
        ax_eig.legend(fontsize=8)
    ax_eig.set_title("Inverse mass-matrix eigenvalues")
    ax_eig.set_ylabel("eigenvalue")

    # 4. acceptance rate: per-chain raw (faint) + cross-chain smoothed mean,
    #    with the dual-averaging target overlaid. This is the MAMS analogue of
    #    MCLMC's xi panel -- it's what the step-size adaptation targets.
    if "acceptance_rate" in arr:
        acc = np.asarray(arr["acceptance_rate"])  # (n_chains, n_steps)
        ax_acc.plot(acc.T, color="tab:blue", alpha=0.15)
        mean_acc = acc.mean(axis=0)
        if smooth and smooth > 1 and mean_acc.size >= smooth:
            kern = np.ones(smooth) / smooth
            ax_acc.plot(np.convolve(mean_acc, kern, mode="same"), color="black",
                        label="chain-mean (smoothed)")
        else:
            ax_acc.plot(mean_acc, color="black", label="chain-mean")
        if target_acc is not None:
            ax_acc.axhline(float(target_acc), color="tab:green", linestyle="--",
                           linewidth=1, label=f"target {float(target_acc):.2f}")
        ax_acc.set_ylim(0, 1)
        ax_acc.legend(fontsize=8)
    ax_acc.set_title("Metropolis acceptance rate")
    ax_acc.set_ylabel("p(accept)")

    # 5. trajectory length in integrator steps (shared Halton schedule;
    #    deterministic and identical across chains, so show one chain's series).
    if "num_integration_steps" in arr:
        nis = np.asarray(arr["num_integration_steps"])
        series = nis[chain % nis.shape[0]] if nis.ndim == 2 else nis
        ax_n.plot(series, color="tab:purple")
    ax_n.set_title("Integrator steps per trajectory")
    ax_n.set_ylabel("n steps")

    # 6. success / NaN heatmap (chains x steps), restricted to the tuning span
    if "nonan" in arr:
        nonan = np.asarray(arr["nonan"])
        upto = stage2 if stage2 > 0 else nonan.shape[1]
        ax_nan.imshow(
            nonan[:, :upto], aspect="auto", interpolation="none", cmap="RdYlGn",
            vmin=0, vmax=1,
        )
        ax_nan.set_ylabel("chain")
    ax_nan.set_title("Finite-step mask (green = ok, red = NaN)")
    ax_nan.set_xlabel("step")

    # Tuning-stage boundaries on every panel.
    for ax in axs:
        for x, color in ((stage1, "tab:red"), (stage2, "tab:blue"), (stage3, "tab:green")):
            if x > 0:
                ax.axvline(x, color=color, linestyle="--", linewidth=1)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PT-MCLMC (parallel-tempered MCLMC)
# ---------------------------------------------------------------------------

@register_diagnostic_plotter("PTMCLMCStage")
def plot_pt_mclmc_diagnostics(
    diagnostics,
    *,
    figsize: Tuple[float, float] = (10, 11),
) -> Figure:
    """Four stacked panels of a PT-MCLMC (parallel-tempered MCLMC) run, vs.
    round, for a PT novice:

    (a) cold-rung transport trace: mean occupancy of ``indicator`` per round
        (over walkers) if captured, else mean ``cold_logdensity`` per round;
        vertical dashed lines at the ``metric_windows`` boundaries and a
        shaded burn-in region ``[0, num_burnin_rounds)``;
    (b) swap acceptance (transport bottleneck check): a bar per rung
        boundary, with a red 0.3 "bottleneck" reference line and a green
        0.4-0.7 healthy band shaded;
    (c) EEVPD per rung vs. round (log-y), one line per rung labeled by its
        beta, with the 1e-4..2e-3 health band shaded;
    (d) mean MCLMC step size per rung vs. round (log-y).

    Requires the stage to have been run with ``debug=True``. Degrades
    gracefully (skips/annotates a panel) when an expected array is absent.
    """
    arr = diagnostics.arrays
    config = diagnostics.config
    num_burnin = int(config.get("num_burnin_rounds", 0))
    metric_windows = config.get("metric_windows", ())

    fig, axs = plt.subplots(4, 1, figsize=figsize)
    ax_transport, ax_swap, ax_eevpd, ax_step = axs

    # (a) cold-rung transport trace.
    if "cold_indicator" in arr:
        occ = np.asarray(arr["cold_indicator"]).mean(axis=1)
        ax_transport.plot(occ, color="tab:blue")
        ax_transport.set_ylabel("cold occupancy\n(indicator, mean over walkers)")
    elif "cold_logdensity" in arr:
        occ = np.asarray(arr["cold_logdensity"]).mean(axis=1)
        ax_transport.plot(occ, color="tab:blue")
        ax_transport.set_ylabel("mean cold log-density")
    if num_burnin > 0:
        ax_transport.axvspan(0, num_burnin, color="grey", alpha=0.15,
                              label=f"burn-in (0-{num_burnin})")
    ax_transport.set_title("Cold-rung transport trace (has the ladder equilibrated?)")
    ax_transport.legend(fontsize=8, loc="upper right")

    # (b) swap acceptance per boundary: bottleneck check.
    if "swap_attempts" in arr and "swap_accepts" in arr:
        attempts = np.asarray(arr["swap_attempts"], dtype=np.float64)
        accepts = np.asarray(arr["swap_accepts"], dtype=np.float64)
        rate = np.divide(accepts, attempts,
                          out=np.full_like(accepts, np.nan),
                          where=attempts > 0)
        x = np.arange(rate.shape[0])
        colors = ["tab:red" if (np.isfinite(r) and r < 0.3) else "tab:blue" for r in rate]
        ax_swap.bar(x, rate, color=colors)
        ax_swap.axhspan(0.4, 0.7, color="tab:green", alpha=0.15, label="healthy 0.4-0.7")
        ax_swap.axhline(0.3, color="tab:red", linestyle="--", linewidth=1,
                         label="0.3 bottleneck threshold")
        ax_swap.set_xticks(x)
        ax_swap.set_xticklabels([f"{i}<->{i+1}" for i in x], fontsize=8)
        ax_swap.set_ylim(0, 1)
        ax_swap.legend(fontsize=8, loc="upper right")
    ax_swap.set_title("Swap acceptance (transport bottleneck check)")
    ax_swap.set_ylabel("p(accept)")

    # (c) EEVPD per rung vs. round.
    if "eevpd" in arr:
        eevpd = np.asarray(arr["eevpd"])  # (n_rounds, n_rungs)
        betas = np.asarray(arr.get("betas", np.arange(eevpd.shape[1])))
        for r in range(eevpd.shape[1]):
            label = f"rung {r} (beta={betas[r]:.3g})" if r < len(betas) else f"rung {r}"
            ax_eevpd.plot(eevpd[:, r], label=label, alpha=0.8)
        ax_eevpd.axhspan(1e-4, 2e-3, color="tab:green", alpha=0.12, label="health band")
        ax_eevpd.set_yscale("log")
        ax_eevpd.legend(fontsize=7, ncol=2, loc="upper right")
    ax_eevpd.set_title("EEVPD per rung (step-size adapter target)")
    ax_eevpd.set_ylabel("EEVPD")

    # (d) mean step size per rung vs. round.
    if "step_size_mean" in arr:
        ss = np.asarray(arr["step_size_mean"])  # (n_rounds, n_rungs)
        betas = np.asarray(arr.get("betas", np.arange(ss.shape[1])))
        for r in range(ss.shape[1]):
            label = f"rung {r} (beta={betas[r]:.3g})" if r < len(betas) else f"rung {r}"
            ax_step.plot(ss[:, r], label=label, alpha=0.8)
        ax_step.set_yscale("log")
        ax_step.legend(fontsize=7, ncol=2, loc="upper right")
    ax_step.set_title("Mean MCLMC step size per rung")
    ax_step.set_ylabel("step size")
    ax_step.set_xlabel("round")

    if num_burnin > 0:
        for ax in (ax_eevpd, ax_step):
            ax.axvspan(0, num_burnin, color="grey", alpha=0.1)
    for ax in (ax_transport, ax_eevpd, ax_step):
        for w in metric_windows:
            ax.axvline(int(w), color="tab:orange", linestyle="--", linewidth=1)

    fig.tight_layout()
    return fig


def _z_space_labels(ctx, dim: int) -> Optional[list]:
    """Display labels for the sampler's z columns, in z-column order, or ``None``.

    Z-SPACE. ``ctx.prob_model.z_param_names`` is the scene's own column→name map and
    the ONE correct source for it (see ``gigalens.jax.scene._z_param_names``). This
    used to rebuild the names itself, by probing ``bij.forward`` and sorting the
    output dict's keys — precisely the reconstruction that map exists to retire, and
    the one whose off-by-reversal ("C-8") mislabel it was written to work around.

    Returns ``None``, so the caller falls back to generic ``z[i]``, when the map is
    absent or its length disagrees with ``dim``: a z-space debug figure is worth
    drawing unlabeled, but never worth mislabeling, and a captured ``samples_z`` can
    outlive the model it came from.
    """
    names = getattr(getattr(ctx, "prob_model", None), "z_param_names", None)
    if not names or len(names) != dim:
        return None
    return z_column_labels(names)


def plot_mclmc_surrogate_corner(
    diagnostics,
    *,
    max_samples: int = 5000,
    seed: int = 0,
    sample_color: str = "black",
    surrogate_color: str = "tab:red",
    **corner_kwargs,
):
    """Corner plot of the final MCLMC draws vs. a Gaussian surrogate.

    The surrogate is a multivariate normal with mean equal to the sample mean
    and covariance equal to the **final inverse mass matrix** used during
    sampling (the last tuning step's matrix). This shows how Gaussian the
    posterior is and whether the preconditioner the sampler actually used
    matches the realized draw covariance.

    Everything is in the **unconstrained (z) space**: that is where both the
    sampler positions and the inverse mass matrix live, and the bijector to
    physical space is nonlinear, so the IMM covariance is only meaningful here.
    Axes are labeled with z-space parameter names when they can be recovered
    unambiguously, else generic ``z[i]``.

    Requires the stage to have been run with ``debug=True`` (so ``samples_z``
    and ``inverse_mass_matrix`` were captured). Returns a Matplotlib ``Figure``.

    Parameters
    ----------
    max_samples : int, default 5000
        Cap on the number of (chain-flattened) draws plotted; the surrogate is
        drawn with the same count. Subsampling is deterministic given ``seed``.
    """
    import corner as _corner_pkg
    from matplotlib.lines import Line2D

    arr = diagnostics.arrays
    missing = [k for k in ("samples_z", "inverse_mass_matrix") if k not in arr]
    if missing:
        raise ValueError(
            f"Surrogate corner needs {missing} in the captured diagnostics; "
            "re-run the stage with debug=True (MCLMCStage(..., debug=True))."
        )

    samples = np.asarray(arr["samples_z"])           # (n_chains, n_steps, dim)
    flat = samples.reshape(-1, samples.shape[-1])     # (N, dim)
    dim = flat.shape[-1]
    # Final inverse mass matrix actually used for sampling: chain-0 is stored,
    # and the last tuning step is the one carried into the sampling phase.
    imm = np.asarray(arr["inverse_mass_matrix"])[0]   # (n_steps, dim, dim)
    cov = imm[-1]

    mean = flat.mean(axis=0)
    rng = np.random.default_rng(seed)

    # Subsample the real draws (deterministically) and draw an equal-sized
    # surrogate, so neither side dominates the histograms by sheer count.
    n_plot = min(max_samples, flat.shape[0])
    if n_plot < flat.shape[0]:
        idx = rng.choice(flat.shape[0], size=n_plot, replace=False)
        real = flat[idx]
    else:
        real = flat
    surrogate = rng.multivariate_normal(mean, cov, size=n_plot)

    labels = _z_space_labels(diagnostics.ctx, dim) or [f"z[{i}]" for i in range(dim)]

    # Shared axis range over both clouds so corner's fig-reuse overlay is stable.
    combined = np.vstack([real, surrogate])
    lo = np.quantile(combined, 0.0005, axis=0)
    hi = np.quantile(combined, 0.9995, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    lo -= 0.05 * span; hi += 0.05 * span
    shared_range = list(zip(lo.tolist(), hi.tolist()))

    defaults = dict(labels=labels, show_titles=False, plot_datapoints=False,
                    plot_density=False, range=shared_range)
    defaults.update(corner_kwargs)

    pre_existing = set(plt.get_fignums())
    fig = _corner_pkg.corner(real, color=sample_color,
                             hist_kwargs={"density": True, "color": sample_color},
                             **defaults)
    _corner_pkg.corner(surrogate, fig=fig, color=surrogate_color,
                       hist_kwargs={"density": True, "color": surrogate_color},
                       **defaults)
    # Drop any orphan figure strict corner versions may spawn on overlay.
    for fid in list(plt.get_fignums()):
        if fid not in pre_existing and plt.figure(fid) is not fig:
            plt.close(fid)

    fig.legend(
        handles=[
            Line2D([0], [0], color=sample_color, lw=2, label="final samples"),
            Line2D([0], [0], color=surrogate_color, lw=2,
                   label="MVN(sample mean, final inverse mass matrix)"),
        ],
        loc="upper right", frameon=False, fontsize=10,
    )
    fig.suptitle("MCLMC final draws vs. Gaussian surrogate "
                 "(unconstrained z-space)", fontsize=13)
    return fig
