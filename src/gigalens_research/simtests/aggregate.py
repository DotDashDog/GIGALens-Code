"""Aggregation, figures, and campaign-level diagnostics.

:func:`aggregate_campaign` is the main entry point.  It:
1. Builds (or refreshes) the run index via :func:`~index.build_index`.
2. Produces convergence figures (R-hat / ESS distributions, NaN rates).
3. Produces truth-recovery figures (z-scores, residuals, percent errors).
4. Calls any registered *campaign metrics* for population-level analyses.

Figures are saved to ``<base_dir>/aggregate/``.

Campaign metrics
----------------
Register population-level metrics (functions over the full index and all
posteriors) with :func:`register_campaign_metric`::

    @register_campaign_metric("population_level_demo")
    def demo(index_df, posteriors_by_run, campaign_spec, agg_dir):
        ...  # e.g. hierarchical inference or a combined cosmological test

This is the primary extension seam for cosmological-level analyses.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Campaign-metric registry
# ---------------------------------------------------------------------------

_CAMPAIGN_METRICS: Dict[str, Callable] = {}


def register_campaign_metric(name: str) -> Callable:
    """Decorator to register a population-level campaign metric."""
    def decorator(fn: Callable) -> Callable:
        _CAMPAIGN_METRICS[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def aggregate_campaign(
    campaign_spec: Any,
    base_dir: str,
    *,
    rebuild_index: bool = True,
    run_campaign_metrics: bool = True,
    verbose: bool = True,
) -> "Any":
    """Aggregate results and produce figures for a finished (or partial) campaign.

    Returns the index :class:`pandas.DataFrame`.
    """
    try:
        import pandas as pd
    except ImportError:
        warnings.warn("[aggregate] pandas not available; skipping aggregation.")
        return {}

    from .index import build_index

    agg_dir = os.path.join(base_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    runs_dir = os.path.join(base_dir, "runs")

    if verbose:
        print(f"[aggregate] campaign={campaign_spec.name!r}  agg_dir={agg_dir}")

    df = build_index(runs_dir, campaign_spec.name)
    if df is None or (hasattr(df, "empty") and df.empty):
        if verbose:
            print("[aggregate] no completed runs found; nothing to aggregate.")
        return df

    n_ok = int((df.get("status", pd.Series()) == "ok").sum()) if hasattr(df, "get") else 0
    if verbose:
        print(f"[aggregate] {len(df)} rows, {n_ok} with status=ok")

    # Write refreshed index.csv
    df.to_csv(os.path.join(base_dir, "index.csv"), index=False)

    try:
        _plot_convergence(df, agg_dir, campaign_spec)
    except Exception as exc:
        warnings.warn(f"[aggregate] convergence figures failed: {exc}", stacklevel=2)

    try:
        _plot_truth_recovery(df, agg_dir, campaign_spec)
    except Exception as exc:
        warnings.warn(f"[aggregate] truth recovery figures failed: {exc}", stacklevel=2)

    if run_campaign_metrics:
        for name, fn in _CAMPAIGN_METRICS.items():
            try:
                fn(df, campaign_spec, agg_dir)
            except Exception as exc:
                warnings.warn(
                    f"[aggregate] campaign metric {name!r} failed: {exc}", stacklevel=2
                )

    if verbose:
        print(f"[aggregate] done. Figures saved to {agg_dir}/")

    return df


# ---------------------------------------------------------------------------
# Convergence figures
# ---------------------------------------------------------------------------


def _plot_convergence(df: Any, agg_dir: str, campaign_spec: Any) -> None:
    import matplotlib.pyplot as plt

    ok = df[df["status"] == "ok"] if "status" in df.columns else df

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{campaign_spec.name} — Convergence", fontsize=12)

    # R-hat histogram
    ax = axes[0]
    if "max_rhat" in ok.columns:
        rhat = ok["max_rhat"].dropna().astype(float)
        ax.hist(rhat, bins=30, edgecolor="white")
        ax.axvline(1.01, color="red", linestyle="--", label="1.01 threshold")
        ax.set_xlabel("max R-hat")
        ax.set_ylabel("count")
        ax.set_title("Max R-hat per run")
        ax.legend()

    # ESS histogram
    ax = axes[1]
    if "min_ess" in ok.columns:
        ess = ok["min_ess"].dropna().astype(float)
        ax.hist(ess, bins=30, edgecolor="white")
        ax.set_xlabel("min ESS")
        ax.set_title("Min ESS per run")

    # NaN rate
    ax = axes[2]
    if "nan_rate" in ok.columns:
        nanr = ok["nan_rate"].dropna().astype(float)
        ax.hist(nanr, bins=30, edgecolor="white")
        ax.set_xlabel("NaN rate")
        ax.set_title("Non-finite sample fraction")

    plt.tight_layout()
    path = os.path.join(agg_dir, "convergence.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Truth recovery figures
# ---------------------------------------------------------------------------


def _plot_truth_recovery(df: Any, agg_dir: str, campaign_spec: Any) -> None:
    """Produce truth-recovery summary figures.

    Figures produced (if enough data exists):
    1. ``zscores_scatter.png`` — per-parameter z-score scatter across all runs.
    2. ``abs_zscore_vs_sweep.png`` — |z| violin per sweep point (for sweeps).
    3. ``percent_error.png`` — percent error per parameter (for named sweeps).
    """
    import matplotlib.pyplot as plt

    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if ok.empty:
        return

    # Collect z-score columns (ending in "_z")
    z_cols = [c for c in ok.columns if c.endswith("_z") and not c.startswith("sys_")]
    if not z_cols:
        return

    # 1. Z-score scatter
    _plot_zscore_scatter(ok, z_cols, agg_dir, campaign_spec)

    # 2. |z| vs sweep point (only if >1 sweep point)
    sweep_cols = [c for c in ok.columns if c.startswith("sweep_")]
    if sweep_cols:
        _plot_abs_zscore_vs_sweep(ok, z_cols, sweep_cols, agg_dir, campaign_spec)

    # 3. Percent errors
    _plot_percent_errors(ok, z_cols, agg_dir, campaign_spec)


def _plot_zscore_scatter(ok: Any, z_cols: List[str], agg_dir: str, campaign_spec: Any) -> None:
    import matplotlib.pyplot as plt

    mass_z_cols = [c for c in z_cols
                   if not c.startswith("lens_") and not c.startswith("src_")]
    if not mass_z_cols:
        mass_z_cols = z_cols

    n_params = len(mass_z_cols)
    fig, axes = plt.subplots(1, max(n_params, 1), figsize=(max(4 * n_params, 8), 5),
                              squeeze=False)
    fig.suptitle(f"{campaign_spec.name} — Mass parameter z-scores", fontsize=11)

    sweep_cols = [c for c in ok.columns if c.startswith("sweep_")]
    sweep_vals = ok[sweep_cols[0]].astype(str) if sweep_cols else None
    unique_sweeps = list(sweep_vals.unique()) if sweep_vals is not None else ["default"]
    colors = plt.cm.tab10.colors

    for ax_i, col in enumerate(mass_z_cols):
        ax = axes[0][ax_i]
        z = ok[col].dropna().astype(float)
        for si, sv in enumerate(unique_sweeps):
            mask = (sweep_vals == sv) if sweep_vals is not None else ok.index == ok.index
            z_sv = ok.loc[mask if sweep_vals is not None else ok.index, col].dropna().astype(float)
            ax.scatter(np.arange(len(z_sv)) + si * 0.2, z_sv,
                       alpha=0.7, s=20, label=sv if len(unique_sweeps) > 1 else None,
                       color=colors[si % len(colors)])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(3, color="red", linewidth=0.6, linestyle="--")
        ax.axhline(-3, color="red", linewidth=0.6, linestyle="--")
        ax.set_title(col.replace("_z", ""))
        ax.set_ylabel("z-score" if ax_i == 0 else "")
        if len(unique_sweeps) > 1 and ax_i == 0:
            ax.legend(fontsize=7, title=sweep_cols[0].replace("sweep_", "") if sweep_cols else "sweep")

    plt.tight_layout()
    plt.savefig(os.path.join(agg_dir, "zscores_scatter.png"), dpi=150)
    plt.close(fig)


def _plot_abs_zscore_vs_sweep(
    ok: Any, z_cols: List[str], sweep_cols: List[str], agg_dir: str, campaign_spec: Any
) -> None:
    import matplotlib.pyplot as plt

    mass_z_cols = [c for c in z_cols
                   if not c.startswith("lens_") and not c.startswith("src_")]
    if not mass_z_cols or len(ok[sweep_cols[0]].unique()) < 2:
        return

    sweep_col = sweep_cols[0]
    sweep_label = sweep_col.replace("sweep_", "")
    sweep_vals_sorted = sorted(ok[sweep_col].dropna().unique(),
                                key=lambda x: (float(x) if _is_numeric(x) else x))

    fig, axes = plt.subplots(1, len(mass_z_cols), figsize=(4 * len(mass_z_cols), 5), squeeze=False)
    fig.suptitle(f"{campaign_spec.name} — |z| vs {sweep_label}", fontsize=11)

    null_z = np.abs(np.random.default_rng(0).standard_normal(10000))

    for ax_i, col in enumerate(mass_z_cols):
        ax = axes[0][ax_i]
        data_by_sweep = [
            ok.loc[ok[sweep_col] == sv, col].dropna().abs().values.astype(float)
            for sv in sweep_vals_sorted
        ]
        labels = [str(sv) for sv in sweep_vals_sorted]

        parts = ax.violinplot(
            [d for d in data_by_sweep if len(d) > 0],
            positions=range(len(data_by_sweep)),
            showmedians=True,
        )
        ax.violinplot([null_z], positions=[len(data_by_sweep)], showmedians=True)
        ax.set_xticks(range(len(data_by_sweep) + 1))
        ax.set_xticklabels(labels + ["|N(0,1)|"], fontsize=8)
        ax.set_title(col.replace("_z", ""))
        ax.set_ylabel("|z|" if ax_i == 0 else "")
        ax.set_xlabel(sweep_label)

    plt.tight_layout()
    plt.savefig(os.path.join(agg_dir, "abs_zscore_vs_sweep.png"), dpi=150)
    plt.close(fig)


def _plot_percent_errors(ok: Any, z_cols: List[str], agg_dir: str, campaign_spec: Any) -> None:
    import matplotlib.pyplot as plt

    pct_cols_json = [c for c in ok.columns if c == "percent_error_json"]
    if not pct_cols_json:
        return

    import json
    all_pct: list = []
    for _, row in ok.iterrows():
        v = row.get("percent_error_json")
        if v and isinstance(v, str):
            try:
                all_pct.append(json.loads(v))
            except Exception:
                pass

    if not all_pct:
        return

    param_names = sorted(all_pct[0].keys())
    mass_params = [p for p in param_names
                   if not p.startswith("lens_") and not p.startswith("src_")]
    if not mass_params:
        mass_params = param_names

    fig, axes = plt.subplots(1, len(mass_params), figsize=(4 * len(mass_params), 5), squeeze=False)
    fig.suptitle(f"{campaign_spec.name} — Mass parameter percent error", fontsize=11)

    for ax_i, param in enumerate(mass_params):
        ax = axes[0][ax_i]
        vals = [d.get(param) for d in all_pct if d.get(param) is not None]
        if vals:
            ax.hist(vals, bins=20, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(param)
        ax.set_xlabel("% error" if ax_i == 0 else "")

    plt.tight_layout()
    plt.savefig(os.path.join(agg_dir, "percent_error.png"), dpi=150)
    plt.close(fig)


def _is_numeric(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
