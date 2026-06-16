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
    3. inverse-mass-matrix eigenvalue spread (min/mean/max, log y),
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
