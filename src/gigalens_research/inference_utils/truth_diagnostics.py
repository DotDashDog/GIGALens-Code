"""Diagnostics for simulated systems where the truth is known.

These are data-layer functions; the matching plotters live in
:mod:`gigalens_research.plotting.truth`, and the report-level wiring is on
:class:`PosteriorReport`. Keep this module light on dependencies — it is
imported by both modeling and plotting code.

Conventions
-----------
- ``truth_x`` names parameters the way the model does: either scene-nested
  (``{"planes": {"lens": {"mass": {"host": {...}}}}, "cosmo": {...}}``) or
  path-keyed (``{"planes/lens/mass/host/theta_E": ...}``). A parameter is matched
  by *where it lives*, so results are keyed by scene path — and a plane/component
  key is its name where the scene names it, ``str(index)`` where it does not.
- A truth persisted by ``simtests`` is still in the old 3-group form; it is
  adapted at the boundary by
  :func:`gigalens_research.inference_utils.params.truth_x_to_scene_params`
  rather than re-persisted (D2).
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


from ..param_index import (
    KINDS,
    kind_of_key,
    param_sites,
    sites_to_matrix,
    truth_row,
)


def filter_keys_by_kind(keys: Sequence[str], kind: Optional[str]) -> list:
    """Subset path keys to one parameter class: ``'cosmology'``, ``'geometry'``,
    ``'mass'``, ``'light'``, or everything for ``'all'`` / ``None``.

    Classifies by scene path (``planes/0/mass/0/theta_E`` -> mass), so it reads
    the model's own structure rather than matching name prefixes. The old
    prefix version defined "mass" by *negation* — anything without a
    ``lens_``/``src_``/``cosmo_`` prefix — so every new group silently leaked
    into the mass panel until someone remembered to add it here.
    """
    if kind is None or kind == "all":
        return list(keys)
    if kind not in KINDS:
        raise ValueError(
            f"kind must be one of {', '.join(repr(k) for k in KINDS)}, 'all', or "
            f"None; got {kind!r}."
        )
    return [k for k in keys if kind_of_key(k) == kind]


def _flat_floats(x_flat) -> Dict[str, float]:
    """Reduce each leaf of a flat bijector output to a plain float."""
    return {k: float(np.squeeze(np.asarray(v))) for k, v in x_flat.items()}


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
    Returns a flat dict keyed by scene path (``planes/0/mass/0/theta_E``).
    Parameters with degenerate (zero-width) intervals get ``NaN``.

    Only parameters the truth actually defines are scored. Truth and fitted
    models legitimately differ in their light parameterization (an
    ``ImageBasedLight`` truth fit with a shapelet source, say): the shared
    mass/center parameters are scored and the rest are skipped with a warning.

    ``truth_x`` is a scene-nested point (``{"planes": {...}, "cosmo": {...}}``)
    or a path-keyed dict. Both name parameters the way the model does, so a
    parameter is matched by *where it lives*, not by a reconstructed label.

    Works on any :class:`Posterior` that supports :meth:`quantiles_z`
    (samplers and surrogates); not meaningful for :class:`PointEstimate`.
    """
    if not hasattr(posterior, "quantiles_z"):
        raise TypeError(
            f"z_scores requires a posterior with quantiles_z(); "
            f"{type(posterior).__name__} has no posterior uncertainty."
        )
    sites = param_sites(posterior)
    median = sites_to_matrix(sites, posterior.z_to_x(posterior.median_z))[0]
    lower = sites_to_matrix(sites, posterior.z_to_x(posterior.quantiles_z(low_q)))[0]
    upper = sites_to_matrix(sites, posterior.z_to_x(posterior.quantiles_z(high_q)))[0]
    # NaN wherever the truth is silent on a parameter; truth_row warns about it.
    truth = truth_row(sites, truth_x, what="z_scores truth")

    out: Dict[str, float] = {}
    for i, site in enumerate(sites):
        if not np.isfinite(truth[i]):
            continue
        m = median[i]
        sigma = (upper[i] - m) if truth[i] > m else (m - lower[i])
        out[site.key] = (truth[i] - m) / sigma if sigma > 0 else float("nan")
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
