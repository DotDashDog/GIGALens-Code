"""Diagnostics for simulated systems where the truth is known.

These are data-layer functions; the matching plotters live in
:mod:`gigalens_research.plotting.truth`, and the report-level wiring is on
:class:`PosteriorReport`. Keep this module light on dependencies — it is
imported by both modeling and plotting code.

Conventions
-----------
- ``truth_x`` follows the same nested structure as the posterior's physical
  parameters: ``[[mass_dicts], [lens_light_dicts], [source_light_dicts]]``.
- Asymmetric ±1σ z-scores match :func:`vela_utilities.stdev_calc` — the
  ``sigma`` denominator is the *upper* quantile gap when truth is above the
  median, otherwise the *lower* gap. With ``low_q=0.159`` / ``high_q=0.841``
  this corresponds to the empirical ±1σ.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np


def _flatten_params(nested_x):
    """Local import of :func:`plotting.labels.flatten_params`. Lazy to avoid
    a circular import (the plotting package eagerly pulls in ``reports``,
    which in turn imports this module)."""
    from ..plotting.labels import flatten_params
    return flatten_params(nested_x)


# Group ordering matches gigalens' params convention:
#   params[0] = mass, params[1] = lens light, params[2] = source light.
_GROUP_PREFIX_BY_NAME = {
    "mass": "",
    "lens_light": "lens_",
    "src_light": "src_",
    "source_light": "src_",  # convenience alias
}


def filter_labels_by_group(labels: Sequence[str], group: str) -> list:
    """Subset a flat parameter-label list to a single group ('mass',
    'lens_light', 'src_light'), or return everything for 'all' / ``None``.

    Mass params have no prefix; lens light has ``lens_``; source light has
    ``src_``. See :mod:`plotting.labels` for the flattening convention.
    """
    if group is None or group == "all":
        return list(labels)
    if group not in _GROUP_PREFIX_BY_NAME:
        raise ValueError(
            f"group must be one of 'mass', 'lens_light', 'src_light', "
            f"'all', or None; got {group!r}."
        )
    prefix = _GROUP_PREFIX_BY_NAME[group]
    if group == "mass":
        return [l for l in labels if not l.startswith("lens_") and not l.startswith("src_")]
    return [l for l in labels if l.startswith(prefix)]


def _flat_floats(nested_x) -> Dict[str, float]:
    """Flatten a posterior point ``x`` and reduce each leaf to a plain float."""
    flat = _flatten_params(nested_x)
    return {k: float(np.squeeze(np.asarray(v))) for k, v in flat.items()}


def z_scores(
    posterior,
    truth_x,
    *,
    low_q: float = 0.159,
    high_q: float = 0.841,
) -> Dict[str, float]:
    """Per-parameter z-score of the truth relative to the posterior's
    asymmetric ±1σ quantiles.

    ``z = (truth - median) / sigma`` where ``sigma`` is the *upper* (high_q)
    quantile gap when ``truth > median`` and the *lower* (low_q) gap otherwise.
    Returns a flat dict keyed by the same labels :func:`flatten_params`
    produces. Parameters with degenerate (zero-width) intervals get ``NaN``.

    Only parameters present in *both* the truth and the posterior are scored.
    This is the common case when the truth and fitted models differ in their
    light parameterization (e.g. an ``ImageBasedLight`` truth fit with a
    shapelet source): the shared mass/center parameters are scored and the
    unmatched light parameters are skipped with a warning.

    Works on any :class:`Posterior` that supports :meth:`quantiles_z`
    (samplers and surrogates); not meaningful for :class:`PointEstimate`.
    """
    if not hasattr(posterior, "quantiles_z"):
        raise TypeError(
            f"z_scores requires a posterior with quantiles_z(); "
            f"{type(posterior).__name__} has no posterior uncertainty."
        )
    # grouped_free_x regroups a scene-backed flat bijector output into the legacy
    # 3-group label space (pass-through for legacy posteriors), so the truth (3-group)
    # and the posterior points share labels and the shared params are actually scored.
    # In multi-profile groups every parameter carries a ``__<i>`` profile-index
    # suffix (see plotting.labels.flatten_params), so each profile's params are
    # scored separately rather than colliding onto one column. This requires the
    # truth's profiles to be in the same order as the model's within each group.
    flat_truth = _flat_floats(truth_x)
    flat_med = _flat_floats(posterior.grouped_free_x(posterior.z_to_x(posterior.median_z)))
    flat_lo = _flat_floats(posterior.grouped_free_x(posterior.z_to_x(posterior.quantiles_z(low_q))))
    flat_hi = _flat_floats(posterior.grouped_free_x(posterior.z_to_x(posterior.quantiles_z(high_q))))
    skipped = [k for k in flat_truth if k not in flat_med]
    if skipped:
        warnings.warn(
            f"z_scores: {len(skipped)} truth parameter(s) are not in the "
            f"posterior and will be skipped: {skipped}.",
            stacklevel=2,
        )
    out: Dict[str, float] = {}
    for k, t in flat_truth.items():
        if k not in flat_med:
            continue
        m = flat_med[k]
        sigma = (flat_hi[k] - m) if t > m else (m - flat_lo[k])
        out[k] = (t - m) / sigma if sigma > 0 else float("nan")
    return out


def source_comparison(
    posterior,
    truth_source: Union[np.ndarray, Callable[..., Any]],
    *,
    extent: Optional[Tuple[float, float, float, float]] = None,
    point: str = "median",
    grid_pix: Optional[int] = None,
    fov_arcsec: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """Render the recovered source plane and the truth on a shared grid.

    Returns ``(truth, recovered, residual, extent)`` with ``residual =
    truth - recovered``.

    The truth can be supplied in either of two ways:

    1. **Pre-rendered**: ``truth_source`` is a square 2-D array, and
       ``extent`` gives its field of view. The recovered source is rendered
       on a matching grid (``grid_pix`` and ``fov_arcsec`` are read from the
       truth array's shape and the ``extent``).

    2. **Callable**: ``truth_source`` is a callable ``f(X, Y) -> image`` that
       evaluates the truth at arbitrary source-plane coordinates. ``X``/``Y``
       are shaped ``(grid_pix, grid_pix, 1)`` to match gigalens'
       light-profile convention. In this mode, the grid is set by
       ``grid_pix`` (default 400), ``fov_arcsec`` (default from
       ``posterior.ctx.sim_config``), and ``center`` (default: the recovered
       source's center, i.e. the first source-light component's
       ``center_x``/``center_y`` at the chosen point). This is the natural
       mode for interpolated truths like a Vela ``ImageBasedLight``.

    Example for case 2:

    .. code-block:: python

        truth_fn = lambda X, Y: vela_light.light(X, Y, **truth_x[2][0])
        truth, rec, res, ext = source_comparison(
            posterior, truth_fn, grid_pix=400, fov_arcsec=2.0,
        )
    """
    if callable(truth_source):
        recovered, extent = posterior.source_plane(
            point=point,
            grid_pix=400 if grid_pix is None else int(grid_pix),
            fov_arcsec=fov_arcsec,
            center=center,
        )
        h, w = recovered.shape
        gx = jnp.linspace(extent[0], extent[1], w)
        gy = jnp.linspace(extent[2], extent[3], h)
        # Match the (grid, grid, 1) "depth"-axis convention of gigalens light
        # profiles so e.g. ImageBasedLight's stack interpolator works.
        X = jnp.broadcast_to(gx[None, :, None], (h, w, 1))
        Y = jnp.broadcast_to(gy[:, None, None], (h, w, 1))
        truth_arr = jnp.squeeze(truth_source(X, Y))
        truth = np.asarray(truth_arr)
        if truth.shape != recovered.shape:
            raise ValueError(
                f"truth_source callable returned shape {truth.shape}, "
                f"expected {recovered.shape}. Check that it squeezes to 2-D "
                f"after broadcasting against (grid, grid, 1) coordinates."
            )
    else:
        truth = np.asarray(truth_source)
        if truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
            raise ValueError(
                f"truth_source must be a square 2-D array or a callable; "
                f"got array of shape {truth.shape}."
            )
        if extent is None:
            raise ValueError(
                "`extent` is required when `truth_source` is a pre-rendered "
                "array (to know its field of view). Use a callable to defer "
                "grid choice to the recovered source's defaults."
            )
        g = int(truth.shape[0])
        fov = float(extent[1] - extent[0])
        cx = 0.5 * (extent[0] + extent[1])
        cy = 0.5 * (extent[2] + extent[3])
        recovered, _ = posterior.source_plane(
            point=point, grid_pix=g, fov_arcsec=fov, center=(cx, cy),
        )

    residual = truth - recovered
    return truth, recovered, residual, extent


def truth_source_from_light_model(light_model, source_params: Dict[str, Any]):
    """Wrap a gigalens :class:`LightProfile` + truth source params as a
    callable suitable for :func:`source_comparison`.

    Example:

    .. code-block:: python

        # truth_x[2][0] = {"center_x": ..., "center_y": ...} for ImageBasedLight
        fn = truth_source_from_light_model(vela_light, truth_x[2][0])
        source_comparison(posterior, fn, grid_pix=400, fov_arcsec=2.0)
    """
    # Coerce any leaves to plain floats once so JAX doesn't re-trace per call.
    params_static = {
        k: float(np.squeeze(np.asarray(v))) if np.isscalar(v) or np.ndim(np.asarray(v)) == 0 or np.size(v) == 1
        else jnp.asarray(v)
        for k, v in source_params.items()
    }
    def fn(X, Y):
        return light_model.light(X, Y, **params_static)
    fn.__doc__ = f"Truth source via {type(light_model).__name__}({source_params})"
    return fn
