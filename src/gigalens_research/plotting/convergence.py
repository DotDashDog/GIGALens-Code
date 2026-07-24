"""Convergence and training-curve diagnostic plots.

Each plotter takes a :class:`Posterior` (or a raw 1-D loss array, for MAP/SVI
histories) and renders into a provided axes. They share no state and don't
modify the posterior.

Two parameter spaces, never mixed
---------------------------------
R-hat and ESS are computed per column of the sampler's unconstrained ``z``
vector, so ``params=`` selects **z columns** and they are named from
``prob_model.z_param_names``. Chain traces are pushed through the bijector and so
live in **x space**, indexed and named from
:func:`~gigalens_research.param_index.param_sites` — the same records, in the same
order, that the corner plot resolves. The bijector reorders, so an index means
different things in the two spaces; each helper below states which one it is in.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes

from ..param_index import ParamSite, param_sites, site_labels, sites_to_matrix
from .labels import z_column_labels


def _z_labels(posterior) -> List[str]:
    """Display labels for the sampler's z columns, in z-column order.

    Z-SPACE. ``rhat`` / ``ess`` and their running variants return one entry per
    column of the unconstrained z vector, so ``rhat[:, i]`` is z column ``i``.
    ``prob_model.z_param_names`` is the scene's own column→name map and the ONE
    correct source for it (see ``gigalens.jax.scene._z_param_names``): names read off
    a bijector output dict's key order are in a *different* order, which is the
    recurring "C-8" mislabel. There is therefore no safe fallback — every other name
    order disagrees with the sampler's — so a missing or mismatched map raises.
    """
    prob_model = getattr(getattr(posterior, "ctx", None), "prob_model", None)
    names = getattr(prob_model, "z_param_names", None)
    if not names:
        raise AttributeError(
            "this posterior's prob_model publishes no z_param_names, so the "
            "sampler's column→name map is unknown and its z columns cannot be "
            "labelled. Only a scene-backed prob_model publishes it. There is no "
            "fallback on purpose: every name order derivable without it disagrees "
            "with the sampler's, so a guess would silently mislabel every line."
        )
    if len(names) != posterior.n_params:
        raise ValueError(
            f"prob_model.z_param_names carries {len(names)} names but this "
            f"posterior has {posterior.n_params} sampler columns, so the "
            "column→name map does not belong to these samples."
        )
    return z_column_labels(names)


def _worst_indices(values: np.ndarray, idx: np.ndarray, n: int, *,
                   largest: bool) -> np.ndarray:
    """The ``n`` entries of ``idx`` with the largest / smallest ``values[idx]``.

    NaN always sorts to the worst end (a non-finite diagnostic is a failure, not
    a pass), so ``largest=True`` maps NaN→+inf and ``largest=False`` maps
    NaN→−inf before ranking.
    """
    v = np.asarray(values, dtype=float)[idx]
    key = np.where(np.isnan(v), (np.inf if largest else -np.inf), v)
    order = np.argsort(-key if largest else key)
    return idx[order][:max(1, n)]


def _annotate_worst(ax, posterior, n: int, params, *, kind: str) -> None:
    """Title + a small corner table naming the ``n`` worst z-columns.

    ``kind='rhat'`` ranks by largest R-hat; ``kind='ess'`` by smallest bulk-ESS
    and shows each column's ``bulk/tail`` ESS. Values come from the labeled
    full-chain report :attr:`SamplerPosterior.convergence`, so every number is
    tied to the parameter it belongs to (that is the whole point of this panel).
    ``params`` (if given) restricts the ranking to the shown z-columns.
    """
    report = posterior.convergence
    labels = _z_labels(posterior)
    idx = (np.arange(len(labels)) if params is None
           else np.asarray(list(params), dtype=int))
    if kind == "rhat":
        rhat = np.asarray(report.rhat, dtype=float)
        order = _worst_indices(rhat, idx, n, largest=True)
        ax.set_title(f"Running R-hat — worst: {labels[order[0]]} "
                     f"({rhat[order[0]]:.3f})")
        body = "\n".join(f"{labels[i]}: {rhat[i]:.3f}" for i in order)
        ax.text(0.97, 0.97, body, transform=ax.transAxes, fontsize=7,
                va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    elif kind == "ess":
        bulk = np.asarray(report.ess_bulk, dtype=float)
        tail = np.asarray(report.ess_tail, dtype=float)
        order = _worst_indices(bulk, idx, n, largest=False)
        ax.set_title(f"Running ESS — worst: {labels[order[0]]} "
                     f"(bulk {bulk[order[0]]:.0f})")
        body = "worst ESS (bulk/tail):\n" + "\n".join(
            f"{labels[i]}: {bulk[i]:.0f}/{tail[i]:.0f}" for i in order)
        ax.text(0.03, 0.97, body, transform=ax.transAxes, fontsize=7,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown annotate kind {kind!r}")


def _samples_x_chains(
    posterior,
) -> Tuple[Optional[List[ParamSite]], Optional[np.ndarray]]:
    """The chains in physical space: ``(sites, samples_x)``, with ``samples_x``
    shaped ``(n_chains, n_steps, len(sites))``.

    X-SPACE. One column per free parameter, in :func:`param_sites` order — the same
    records and the same order :func:`~gigalens_research.plotting.plot_corner`
    resolves — so trace column ``i`` and corner column ``i`` are the same parameter.

    Returns ``(None, None)`` when x space is unreachable (no bijector, or a
    posterior with no scene behind it and hence no path space); the caller then
    plots raw z and says so, rather than passing z off as physical.
    """
    try:
        sz = posterior.samples_z                       # (n_chains, n_steps, n_params)
        n_chains, n_steps, _ = sz.shape
        sites = param_sites(posterior)
        # z_to_x over the chain-flattened draws: one bijector call, and the flat
        # {unique_key: (n_chains*n_steps,)} dict sites_to_matrix expects.
        x = posterior.z_to_x(sz.reshape(-1, sz.shape[-1]))
        mat = sites_to_matrix(sites, x)                # (n_chains*n_steps, len(sites))
        return sites, mat.reshape(n_chains, n_steps, len(sites))
    except Exception:
        return None, None


def plot_running_rhat(
    ax: Axes,
    posterior,
    *,
    schedule: Optional[Sequence[int]] = None,
    params: Optional[Sequence[int]] = None,
    aggregate: Optional[str] = None,
    annotate_worst: Optional[int] = None,
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

    ``params`` selects **sampler (z) columns** by index, since that is what R-hat
    is computed per; ``prob_model.z_param_names`` is the column→name map. This is a
    different indexing from :func:`plot_chain_traces`, which is x space.

    ``aggregate`` selects how to collapse the per-parameter R-hats:

    - ``None`` (default): one line per parameter (or per index in ``params``).
      LaTeX labels are used in the legend if known.
    - ``'max'``: only the worst R-hat across parameters at each step.
    - ``'mean'``: the mean across parameters.

    ``annotate_worst`` (an int ``n``): overwrite the title with the single
    worst-R-hat parameter and draw a small corner table of the ``n`` worst
    z-columns (name → final R-hat), taken from the labeled full-chain report.
    This is what lets a ``max`` aggregate curve still say *which* parameter is
    the offender. ``None`` (default) leaves the plain title.
    """
    schedule_arr, rhat = posterior.running_rhat(schedule=schedule)
    if params is not None:
        rhat = rhat[:, list(params)]

    # Clip to avoid log(0) when a chain is so well-converged that the
    # estimator returns exactly 1.0 (or, more rarely, slightly less).
    y = np.maximum(rhat - 1.0, 1e-6)

    if aggregate is None:
        # Only this branch names a line, and _z_labels refuses to guess a
        # column→name map; an aggregate plot that labels nothing shouldn't inherit
        # that requirement.
        labels = _z_labels(posterior)
        if params is not None:
            labels = [labels[i] for i in params]
        for i, lbl in enumerate(labels):
            ax.plot(schedule_arr, y[:, i], label=lbl, alpha=0.7)
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
    if annotate_worst is not None:
        _annotate_worst(ax, posterior, annotate_worst, params, kind="rhat")


def plot_running_ess(
    ax: Axes,
    posterior,
    *,
    schedule: Optional[Sequence[int]] = None,
    params: Optional[Sequence[int]] = None,
    aggregate: Optional[str] = "min",
    annotate_worst: Optional[int] = None,
    log_x: bool = True,
    log_y: bool = True,
) -> None:
    """ESS as a function of how many steps were used.

    ``aggregate`` is ``'min'`` by default (worst-parameter ESS, which is the
    most useful single number for "are we done sampling"). ``None`` plots one
    line per parameter.

    As in :func:`plot_running_rhat`, ``params`` selects **sampler (z) columns** by
    index — not x-space / corner-plot columns.

    ``annotate_worst`` (an int ``n``): overwrite the title with the single
    lowest-ESS parameter and draw a corner table of the ``n`` worst z-columns
    (name → ``bulk/tail`` ESS), from the labeled full-chain report.
    """
    schedule_arr, ess = posterior.running_ess(schedule=schedule)
    if params is not None:
        ess = ess[:, list(params)]

    if aggregate is None:
        labels = _z_labels(posterior)
        if params is not None:
            labels = [labels[i] for i in params]
        for i, lbl in enumerate(labels):
            ax.plot(schedule_arr, ess[:, i], label=lbl, alpha=0.7)
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
    if annotate_worst is not None:
        _annotate_worst(ax, posterior, annotate_worst, params, kind="ess")


def plot_chain_traces(
    ax: Axes,
    posterior,
    *,
    param: int = 0,
    z_param: Optional[int] = None,
    max_chains: int = 16,
    alpha: float = 0.5,
) -> None:
    """Plot the per-chain trace for a single parameter (by index).

    Default (``z_param=None``): traces are shown in **physical (x) space**
    (bijector applied), so the y-axis matches the corner plot. ``param`` indexes
    the free parameters in :func:`~gigalens_research.param_index.param_sites`
    order, which *is* the corner plot's column order — index ``i`` is the same
    parameter in both figures. It is not the sampler's z-column order that
    :func:`plot_running_rhat` and :func:`plot_running_ess` index with
    ``params=``; the bijector reorders. Falls back to raw ``z``-space (with a
    note in the title and generic ``z[i]`` labels) if the bijector is unavailable.

    ``z_param`` (a z-column index): trace that **sampler (z) column** directly,
    labelled from ``prob_model.z_param_names``. This is the space R-hat/ESS are
    computed in, so it is the right view for the parameter those diagnostics
    flag as worst — and it indexes the same way as ``plot_running_*``'s
    ``params=``. Overrides ``param``.

    Caps at ``max_chains`` lines for readability.
    """
    if z_param is not None:
        # z-space: index and labels line up with the R-hat/ESS panels.
        samples = posterior.samples_z
        labels = _z_labels(posterior)
        space, space_note, idx = "z", " [z-space]", z_param
    else:
        sites, samples_x = _samples_x_chains(posterior)
        if samples_x is not None:
            samples = samples_x
            labels = site_labels(sites)
            space, space_note = "x", ""
        else:
            # Raw sampler columns are a different space in a different order; label
            # them generically and say so rather than pass them off as physical.
            samples = posterior.samples_z
            labels = [f"z[{i}]" for i in range(samples.shape[-1])]
            space, space_note = "z", " [z-space]"
        idx = param

    if not 0 <= idx < len(labels):
        raise IndexError(
            f"{'z_param' if z_param is not None else 'param'}={idx} is out of "
            f"range: this trace's {space} space has {len(labels)} parameters "
            f"(0..{len(labels) - 1})."
        )

    n_chains = samples.shape[0]
    chains_to_show = list(range(min(n_chains, max_chains)))
    for c in chains_to_show:
        ax.plot(samples[c, :, idx], alpha=alpha, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(labels[idx])
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
