"""Metric functions over ``(Posterior, System)`` pairs.

All registered metrics have the signature::

    fn(posterior: Posterior, system: System) -> Any

where ``Any`` is typically a float, a list of floats, or a flat dict of
``label -> float``.  The return value must be JSON-serialisable (no numpy
arrays; use ``float()`` to convert).

Registered metrics
------------------
- ``max_rhat`` — maximum Gelman–Rubin R-hat across all sampled parameters.
- ``min_ess`` — minimum ESS across all sampled parameters.
- ``worst_rhat_param`` — name of the parameter carrying that maximum R-hat.
- ``nan_rate`` — fraction of non-finite sample entries.
- ``all_zscores`` — asymmetric z-scores for *all* shared parameters.
- ``mass_zscores`` — z-scores for mass parameters only.
- ``percent_error`` — ``100 × (median − truth) / |truth|`` for shared params.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

from .registry import register_metric


# ---------------------------------------------------------------------------
# Convergence / chain diagnostics
# ---------------------------------------------------------------------------


@register_metric("max_rhat")
def max_rhat(posterior: Any, system: Any) -> float:
    """Maximum Gelman–Rubin R-hat across all parameters (NaN-safe)."""
    if not hasattr(posterior, "rhat"):
        return float("nan")
    rh = np.asarray(posterior.rhat)
    return float(np.nanmax(rh)) if np.isfinite(rh).any() else float("nan")


@register_metric("min_ess")
def min_ess(posterior: Any, system: Any) -> float:
    """Minimum ESS across all parameters (NaN-safe)."""
    if not hasattr(posterior, "ess"):
        return float("nan")
    es = np.asarray(posterior.ess)
    return float(np.nanmin(es)) if np.isfinite(es).any() else float("nan")


@register_metric("worst_rhat_param")
def worst_rhat_param(posterior: Any, system: Any) -> Optional[str]:
    """Name of the parameter with the largest R-hat — the :func:`max_rhat`
    offender, so a campaign CSV can point at *which* z-column drove the worst
    R-hat in each run.

    Uses the labeled convergence report (``posterior.convergence``); returns the
    z-column name, or ``None`` if diagnostics are unavailable.
    """
    report = getattr(posterior, "convergence", None)
    if report is None:
        return None
    try:
        worst = report.worst(1)
    except Exception as exc:
        warnings.warn(f"[metrics.worst_rhat_param] failed: {exc}", stacklevel=2)
        return None
    return worst[0].name if worst else None


@register_metric("nan_rate")
def nan_rate(posterior: Any, system: Any) -> float:
    """Fraction of non-finite entries in the raw sample array."""
    if not hasattr(posterior, "samples_z"):
        return 0.0
    arr = np.asarray(posterior.samples_z)
    return float(np.mean(~np.isfinite(arr)))


# ---------------------------------------------------------------------------
# Truth recovery
# ---------------------------------------------------------------------------


def _scene_truth(posterior: Any, system: Any):
    """``system.truth_x`` as a scene-nested truth.

    Persisted truths are in the OLD 3-group form (mass / lens light / source
    light), which cannot say which plane a parameter is on; the truth-comparison
    layer now works in the scene's path space. ``truth_x_to_scene_params`` maps
    the old form onto the model's actual structure by role+index, which is the
    only place that correspondence is recoverable.
    """
    truth = system.truth_x
    if isinstance(truth, dict) and "planes" in truth:
        return truth  # already scene-nested
    from gigalens_research.inference_utils.params import truth_x_to_scene_params
    return truth_x_to_scene_params(truth, posterior.scene)


@register_metric("all_zscores")
def all_zscores(posterior: Any, system: Any) -> Dict[str, float]:
    """Asymmetric ±1σ z-scores for all shared parameters (truth vs posterior).

    Returns a flat dict ``{param_label: z_score}`` using the same sign
    convention as :func:`gigalens_research.inference_utils.z_scores`:
    positive z means truth is above the posterior median.
    Only parameters present in *both* the truth and the posterior are scored.
    """
    from gigalens_research.inference_utils import z_scores
    if not hasattr(posterior, "quantiles_z"):
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = z_scores(posterior, _scene_truth(posterior, system))
        return {k: float(v) for k, v in scores.items()}
    except Exception as exc:
        warnings.warn(f"[metrics.all_zscores] failed: {exc}", stacklevel=2)
        return {}


@register_metric("mass_zscores")
def mass_zscores(posterior: Any, system: Any) -> Dict[str, float]:
    """Z-scores for mass parameters only."""
    from gigalens_research.inference_utils.truth_diagnostics import filter_keys_by_kind
    scores = all_zscores(posterior, system)
    return {k: scores[k] for k in filter_keys_by_kind(list(scores), "mass")}


@register_metric("percent_error")
def percent_error(posterior: Any, system: Any) -> Dict[str, float]:
    """``100 × (posterior_median − truth) / |truth|`` for shared parameters.

    Parameters with ``|truth| < 1e-12`` are skipped to avoid division by zero.
    """
    if not hasattr(posterior, "median_z"):
        return {}
    try:
        from gigalens_research.param_index import (
            param_sites, sites_to_matrix, truth_row,
        )
        sites = param_sites(posterior)
        median = sites_to_matrix(sites, posterior.z_to_x(posterior.median_z))[0]
        truth = truth_row(sites, _scene_truth(posterior, system), warn=False)
        out: Dict[str, float] = {}
        for i, site in enumerate(sites):
            t = truth[i]
            if not np.isfinite(t) or abs(t) < 1e-12:
                continue
            out[site.key] = 100.0 * (median[i] - t) / abs(t)
        return out
    except Exception as exc:
        warnings.warn(f"[metrics.percent_error] failed: {exc}", stacklevel=2)
        return {}


@register_metric("median_reduced_chisq")
def median_reduced_chisq(posterior: Any, system: Any) -> float:
    """Reduced chi-squared ``χ²/ν`` of the median posterior sample.

    Identical to the value shown in ``PosteriorReport.image_panel``: simulate the
    forward model at the median parameters, form the normalized residual against
    the prob_model's per-pixel error map, sum the squares, and divide by
    ``ν = N_pix − N_params``. A well-specified, converged fit sits near 1.
    """
    if not hasattr(posterior, "simulate"):
        return float("nan")
    try:
        observed = np.asarray(posterior.ctx.prob_model.observed_image)
        predicted = np.asarray(posterior.simulate(point="median"))
        err_map = np.asarray(posterior.err_map_at(predicted))
        residual = (observed - predicted) / err_map
        chisq = float(np.sum(residual ** 2))
        ndof = max(observed.size - posterior.n_params, 1)
        return chisq / ndof
    except Exception as exc:
        warnings.warn(f"[metrics.median_reduced_chisq] failed: {exc}", stacklevel=2)
        return float("nan")


# ---------------------------------------------------------------------------
# Helpers for run.py (not registered as metrics but imported from this module)
# ---------------------------------------------------------------------------


def peak_gpu_bytes() -> int:
    """Return the current peak GPU memory high-water mark across all local devices.

    Uses ``jax.local_devices()[d].memory_stats()['peak_bytes_in_use']``.
    Returns ``-1`` if JAX memory stats are unavailable (e.g. CPU-only builds).
    """
    try:
        import jax
        peaks = []
        for d in jax.local_devices():
            stats = d.memory_stats()
            if stats is not None:
                peaks.append(stats.get("peak_bytes_in_use", 0))
        return max(peaks) if peaks else -1
    except Exception:
        return -1
