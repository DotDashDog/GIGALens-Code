"""Post-hoc per-run diagnostic plotting for completed campaigns.

The ``run`` driver only persists arrays + scalar metrics — it does **no**
plotting, so corner-plotting never sits on the GPU critical path.  This module
renders per-run diagnostic figures *after the fact* from the saved stage arrays,
on CPU, and is shardable so it can run as its own Slurm array.

For each completed ``(system, sweep_point)`` run it rebuilds the
:class:`InferenceContext` from the campaign's inference builder, loads the
stage posteriors from disk via :class:`PipelineReport.from_disk`, and writes the
requested panels to ``<run_dir>/plots/<panel>.png``.

Panels (``--panels``):

``image``, ``convergence``, ``source``, ``corner``, ``z_scores``,
``source_comparison`` (from :class:`PosteriorReport`) and ``diagnostics`` (the
MCLMC tuning history, requires the run used ``mclmc_debug: true``).  ``corner``
is the expensive one; drop it from ``--panels`` for fast lightweight plotting.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless; never needs a display
import matplotlib.pyplot as plt


# All panels, in render order. ``corner`` is the costly one.
ALL_PANELS: Tuple[str, ...] = (
    "image", "convergence", "source", "corner", "z_scores",
    "source_comparison", "diagnostics",
)

# Stage-name preference for "the posterior to report on" (richest first).
# Matched as a case-insensitive substring of the stage instance name. Bootstrap
# stages are excluded so e.g. ``bootstrap_map`` is never mistaken for a ``map``.
_TERMINAL_PRIORITY = ("mclmc", "hmc", "svi", "map")


def _ordered_stages(stage_names: Sequence[str]) -> List[str]:
    """Stage names ordered richest-posterior-first, bootstrap stages dropped."""
    cands = [n for n in stage_names if "bootstrap" not in n.lower()]
    ranked = sorted(
        cands,
        key=lambda n: next((i for i, k in enumerate(_TERMINAL_PRIORITY)
                            if k in n.lower()), len(_TERMINAL_PRIORITY)),
    )
    return ranked


def _stage_dirs(out_dir: str) -> List[str]:
    return [n for n in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, n))
            and n != "plots" and not n.endswith(".stale") and ".stale-" not in n]


def _load_terminal_posterior(out_dir: str, ctx: Any, stage: Optional[str]):
    """Load the posterior for the chosen (or auto-picked) sampler stage.

    Tries candidates in priority order, skipping any stage whose class has no
    posterior view (``TypeError``) or isn't registered (``KeyError``) — e.g. a
    bootstrap stage, or a renamed/removed stage from an older run.
    Returns ``(stage_name, posterior)`` or ``(None, None)``.
    """
    from gigalens_research.inference_utils.pipeline import posterior_from_disk

    candidates = [stage] if stage else _ordered_stages(_stage_dirs(out_dir))
    for name in candidates:
        try:
            return name, posterior_from_disk(out_dir, name, ctx)
        except (TypeError, KeyError):
            continue
    return None, None


def _try_truth_source(system: Any):
    """Best-effort truth-source callable for the source-comparison panel.

    Vela systems store the source dir in ``truth_assets['vela_source_dir']``;
    other campaigns simply get no source-comparison panel.
    """
    src_dir = (getattr(system, "truth_assets", None) or {}).get("vela_source_dir")
    if not src_dir or not os.path.isdir(src_dir):
        return None
    try:
        from gigalens_research.simulations import load_vela_source
        from gigalens_research.inference_utils.truth_diagnostics import (
            truth_source_from_light_model,
        )
        vela = load_vela_source(src_dir)
        return truth_source_from_light_model(vela.light, system.truth_x[2][0])
    except Exception as exc:  # pragma: no cover - best effort
        warnings.warn(f"[plot] could not build truth source for "
                      f"{system.system_id}: {exc}", stacklevel=2)
        return None


def _render_panel(panel: str, report: Any, *, out_dir: str, ctx: Any,
                  stage: str, z_score_group: str):
    """Return the Figure for one panel name (raises if not applicable)."""
    if panel == "image":
        return report.image_panel()
    if panel == "convergence":
        return report.convergence_panel()
    if panel == "source":
        return report.source_panel()
    if panel == "corner":
        return report.corner()
    if panel == "z_scores":
        return report.z_score_panel(group=z_score_group)
    if panel == "source_comparison":
        return report.source_comparison_panel()
    if panel == "diagnostics":
        # MCLMC tuning history; requires the run used mclmc_debug: true.
        from gigalens_research.inference_utils.pipeline import diagnostics_from_disk
        from gigalens_research.plotting.diagnostics import plot_stage_diagnostics
        return plot_stage_diagnostics(diagnostics_from_disk(out_dir, stage, ctx))
    raise ValueError(f"unknown panel {panel!r}; choose from {ALL_PANELS}")


def _plot_one_run(
    *,
    ctx: Any,
    system: Any,
    out_dir: str,
    panels: Sequence[str],
    stage: Optional[str],
    z_score_group: str,
    overwrite: bool,
    verbose: bool,
) -> Dict[str, bool]:
    """Render the requested panels for one run. Returns ``{panel: ok}``."""
    from gigalens_research.plotting import PosteriorReport

    plots_dir = os.path.join(out_dir, "plots")
    targets = {p: os.path.join(plots_dir, f"{p}.png") for p in panels}
    if not overwrite and all(os.path.exists(t) for t in targets.values()):
        if verbose:
            print(f"[plot] {os.path.basename(out_dir)}: up to date, skipping")
        return {p: True for p in panels}

    term, posterior = _load_terminal_posterior(out_dir, ctx, stage)
    if posterior is None:
        warnings.warn(f"[plot] no loadable sampler posterior in {out_dir} "
                      f"(stages: {_stage_dirs(out_dir)})", stacklevel=2)
        return {p: False for p in panels}

    report = PosteriorReport(
        posterior,
        prefix=f"{system.system_id} [{os.path.basename(out_dir)}] ",
        truth_x=system.truth_x,
        truth_source_fn=_try_truth_source(system),
    )

    os.makedirs(plots_dir, exist_ok=True)
    results: Dict[str, bool] = {}
    for panel in panels:
        if not overwrite and os.path.exists(targets[panel]):
            results[panel] = True
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = _render_panel(panel, report, out_dir=out_dir, ctx=ctx,
                                    stage=term, z_score_group=z_score_group)
            fig.savefig(targets[panel], bbox_inches="tight", dpi=150)
            plt.close(fig)
            results[panel] = True
        except Exception as exc:
            # One un-renderable panel (e.g. diagnostics without mclmc_debug, or
            # z_scores with no shared truth params) must not abort the run.
            warnings.warn(f"[plot] {system.system_id} panel {panel!r} failed: "
                          f"{exc}", stacklevel=2)
            results[panel] = False
    if verbose:
        ok = [p for p, v in results.items() if v]
        print(f"[plot] {os.path.basename(out_dir)} ({term}): saved {ok} -> {plots_dir}")
    return results


def plot_campaign(
    campaign_spec: Any,
    base_dir: str,
    *,
    panels: Optional[Sequence[str]] = None,
    stage: Optional[str] = None,
    z_score_group: str = "mass",
    shard_i: int = 0,
    shard_n: int = 1,
    include_failed: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Render per-run diagnostic plots for one shard of a campaign.

    Mirrors ``run_campaign``'s sharding: ``--shard i/N`` renders the strided
    slice ``runs[i::N]``, so a Slurm array covers all runs in parallel.
    """
    from .run import enumerate_runs
    from .system import System
    from .index import load_run_json
    from .registry import get_inference_builder
    from gigalens_research.inference_utils.pipeline import InferenceContext

    panels = list(panels) if panels else list(ALL_PANELS)
    unknown = [p for p in panels if p not in ALL_PANELS]
    if unknown:
        raise ValueError(f"unknown panel(s) {unknown}; choose from {ALL_PANELS}")

    dataset_dir = os.path.join(base_dir, "dataset")
    runs_dir = os.path.join(base_dir, "runs")
    builder = get_inference_builder(campaign_spec.inference.builder)

    all_runs = enumerate_runs(campaign_spec, dataset_dir)
    my_runs = all_runs[shard_i::shard_n]
    if verbose:
        print(f"[plot] campaign {campaign_spec.name!r}: shard {shard_i}/{shard_n} "
              f"-> {len(my_runs)}/{len(all_runs)} runs; panels={panels}")

    n_done = n_skip = n_fail = 0
    for system_id, sweep_point in my_runs:
        sweep_name = campaign_spec.sweep_dir_name(sweep_point)
        out_dir = os.path.join(runs_dir, system_id, sweep_name)

        rec = load_run_json(out_dir)
        if rec is None:
            n_skip += 1
            continue
        if not include_failed and rec.get("status") != "ok":
            if verbose:
                print(f"[plot] {system_id}/{sweep_name}: status="
                      f"{rec.get('status')!r}, skipping (use --include-failed)")
            n_skip += 1
            continue

        try:
            system = System.load(dataset_dir, system_id)
            model_seq = builder(system, **sweep_point)
            ctx = InferenceContext.from_modelling_sequence(model_seq)
            _plot_one_run(
                ctx=ctx, system=system, out_dir=out_dir, panels=panels,
                stage=stage, z_score_group=z_score_group,
                overwrite=overwrite, verbose=verbose,
            )
            n_done += 1
        except Exception as exc:
            warnings.warn(f"[plot] run {system_id}/{sweep_name} failed: {exc}",
                          stacklevel=2)
            n_fail += 1

    if verbose:
        print(f"[plot] done: {n_done} plotted, {n_skip} skipped, {n_fail} errored.")
