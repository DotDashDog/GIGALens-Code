"""Convergence and training-curve diagnostic plots.

Each plotter takes a :class:`Posterior` (or a raw 1-D loss array, for MAP/SVI
histories) and renders into a provided axes. They share no state and don't
modify the posterior.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.axes import Axes

from .labels import flatten_param_names, flatten_params, latex_label


def _param_labels(posterior) -> list:
    """Best-effort parameter labels from the bijector's nested structure."""
    try:
        x = posterior.z_to_x(posterior.median_z)
        return list(flatten_param_names(x))
    except Exception:
        return [f"param[{i}]" for i in range(posterior.n_params)]


def _samples_x_chains(posterior):
    """Push ``samples_z`` through the bijector and flatten to a matrix.

    Returns ``(keys, samples_x)`` where ``keys`` is a list of flat parameter
    names and ``samples_x`` has shape ``(n_chains, n_steps, n_flat_params)``.
    Returns ``(None, None)`` if the bijector is unavailable.
    """
    try:
        sz = posterior.samples_z          # (n_chains, n_steps, n_params)
        n_chains, n_steps, _ = sz.shape
        flat_z = sz.reshape(-1, sz.shape[-1])          # (n_chains*n_steps, n_params)
        x = posterior.z_to_x(flat_z)                   # nested list-of-dicts, batch dim 0
        flat_dict = flatten_params(x)                  # {name: (n_chains*n_steps,)}
        keys = list(flat_dict.keys())
        mat = np.stack([np.asarray(flat_dict[k]) for k in keys], axis=-1)
        return keys, mat.reshape(n_chains, n_steps, -1)
    except Exception:
        return None, None


def plot_running_rhat(
    ax: Axes,
    posterior,
    *,
    schedule: Optional[Sequence[int]] = None,
    params: Optional[Sequence[int]] = None,
    aggregate: Optional[str] = None,
    threshold: float = 1.01,
    log_x: bool = True,
) -> None:
    """``R̂ - 1`` as a function of how many burn-in steps were used, on a log y axis.

    Why ``R̂ - 1``: the usual convergence target ``R̂ ≲ 1.01`` is only
    visible at log scale once you're close to convergence; plotting the raw
    R-hat tends to flatten everything against 1.0 long before the chains are
    actually well-mixed. The ``threshold`` argument (default ``1.01``)
    expresses the convergence cut-off in *R-hat* units; it is drawn at
    ``threshold - 1`` on the plot.

    ``aggregate`` selects how to collapse the per-parameter R-hats:

    - ``None`` (default): one line per parameter (or per index in ``params``).
      LaTeX labels are used in the legend if known.
    - ``'max'``: only the worst R-hat across parameters at each step.
    - ``'mean'``: the mean across parameters.
    """
    schedule_arr, rhat = posterior.running_rhat(schedule=schedule)
    labels = _param_labels(posterior)
    if params is not None:
        rhat = rhat[:, list(params)]
        labels = [labels[i] for i in params]

    # Clip to avoid log(0) when a chain is so well-converged that the
    # estimator returns exactly 1.0 (or, more rarely, slightly less).
    y = np.maximum(rhat - 1.0, 1e-6)

    if aggregate is None:
        for i, lbl in enumerate(labels):
            ax.plot(schedule_arr, y[:, i], label=latex_label(lbl), alpha=0.7)
        if y.shape[1] <= 12:
            ax.legend(fontsize=8, ncol=2, loc="upper right")
    elif aggregate == "max":
        ax.plot(schedule_arr, np.nanmax(y, axis=1), label="max", color="black")
        ax.legend()
    elif aggregate == "mean":
        ax.plot(schedule_arr, np.nanmean(y, axis=1), label="mean", color="black")
        ax.legend()
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")

    ax.axhline(threshold - 1.0, color="grey", linestyle="--", linewidth=1,
               label=f"R̂={threshold}")
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("# samples used")
    ax.set_ylabel(r"$\hat{R} - 1$")
    ax.set_title("Running R-hat")


def plot_running_ess(
    ax: Axes,
    posterior,
    *,
    schedule: Optional[Sequence[int]] = None,
    params: Optional[Sequence[int]] = None,
    aggregate: Optional[str] = "min",
    log_x: bool = True,
    log_y: bool = True,
) -> None:
    """ESS as a function of how many steps were used.

    ``aggregate`` is ``'min'`` by default (worst-parameter ESS, which is the
    most useful single number for "are we done sampling"). ``None`` plots one
    line per parameter.
    """
    schedule_arr, ess = posterior.running_ess(schedule=schedule)
    labels = _param_labels(posterior)
    if params is not None:
        ess = ess[:, list(params)]
        labels = [labels[i] for i in params]

    if aggregate is None:
        for i, lbl in enumerate(labels):
            ax.plot(schedule_arr, ess[:, i], label=latex_label(lbl), alpha=0.7)
        if ess.shape[1] <= 12:
            ax.legend(fontsize=8, ncol=2)
    elif aggregate == "min":
        ax.plot(schedule_arr, np.nanmin(ess, axis=1), color="black", label="min")
    elif aggregate == "mean":
        ax.plot(schedule_arr, np.nanmean(ess, axis=1), color="black", label="mean")
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("# samples used")
    ax.set_ylabel("ESS")
    ax.set_title("Running ESS")


def plot_chain_traces(
    ax: Axes,
    posterior,
    *,
    param: int = 0,
    max_chains: int = 16,
    alpha: float = 0.5,
) -> None:
    """Plot the per-chain trace for a single parameter (by index).

    Traces are shown in **physical space** (bijector applied), so the y-axis
    matches the corner plot and convergence diagnostics. ``param`` indexes into
    the flat physical-space parameter vector in the same order as
    ``flatten_param_names`` / the corner plot columns.

    Falls back to raw ``z``-space (with a note in the title) if the bijector
    is unavailable.

    Caps at ``max_chains`` lines for readability.
    """
    keys, samples_x = _samples_x_chains(posterior)
    if samples_x is not None:
        samples = samples_x
        labels = keys
        space_note = ""
    else:
        samples = posterior.samples_z
        labels = [f"param[{i}]" for i in range(samples.shape[-1])]
        space_note = " [z-space]"

    n_chains = samples.shape[0]
    chains_to_show = list(range(min(n_chains, max_chains)))
    for c in chains_to_show:
        ax.plot(samples[c, :, param], alpha=alpha, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(latex_label(labels[param]) if param < len(labels) else f"param[{param}]")
    ax.set_title(f"Chain traces ({len(chains_to_show)}/{n_chains} chains){space_note}")


def plot_loss_history(
    ax: Axes,
    history: np.ndarray,
    *,
    title: str = "Loss",
    ylabel: str = "loss",
    xlabel: str = "step",
    log_y: bool = False,
) -> None:
    """Simple 1-D loss/objective curve. Works for MAP chi-squared, SVI ELBO,
    or anything else 1-D."""
    history = np.asarray(history).squeeze()
    ax.plot(history)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
