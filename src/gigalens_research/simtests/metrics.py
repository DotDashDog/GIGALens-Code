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
            scores = z_scores(posterior, system.truth_x)
        return {k: float(v) for k, v in scores.items()}
    except Exception as exc:
        warnings.warn(f"[metrics.all_zscores] failed: {exc}", stacklevel=2)
        return {}


@register_metric("mass_zscores")
def mass_zscores(posterior: Any, system: Any) -> Dict[str, float]:
    """Z-scores for mass parameters only (no ``lens_`` or ``src_`` prefix)."""
    scores = all_zscores(posterior, system)
    return {
        k: v for k, v in scores.items()
        if not k.startswith("lens_") and not k.startswith("src_")
    }


@register_metric("percent_error")
def percent_error(posterior: Any, system: Any) -> Dict[str, float]:
    """``100 × (posterior_median − truth) / |truth|`` for shared parameters.

    Parameters with ``|truth| < 1e-12`` are skipped to avoid division by zero.
    """
    if not hasattr(posterior, "median_z"):
        return {}
    try:
        from gigalens_research.inference_utils.truth_diagnostics import _flat_floats
        flat_truth = _flat_floats(system.truth_x)
        flat_med = _flat_floats(posterior.z_to_x(posterior.median_z))
        out: Dict[str, float] = {}
        for k, t in flat_truth.items():
            if k not in flat_med or abs(t) < 1e-12:
                continue
            out[k] = 100.0 * (flat_med[k] - t) / abs(t)
        return out
    except Exception as exc:
        warnings.warn(f"[metrics.percent_error] failed: {exc}", stacklevel=2)
        return {}


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
