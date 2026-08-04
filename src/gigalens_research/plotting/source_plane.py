"""Source-plane plotting, keyed on a :class:`Posterior`.

- :func:`plot_source_plane` renders the intrinsic (unlensed, un-convolved) source
  brightness at a representative point of the posterior.
- :func:`plot_caustics` / :func:`plot_critical_curves` / :func:`plot_image_border`
  overlay the source-plane and image-plane curves for the same point.

**The curve math lives in gigalens, not here.** Everything below is a thin
posterior-keyed adapter over :mod:`gigalens.jax.analysis` -- ``LensSolver`` for the
critical curves and caustics, ``analysis.source_plane`` for the ray-traced cutout
border. This module used to carry its own tracer (``_native_alpha`` +
``_critical_and_caustic_curves``, differentiating each ``profile.deriv`` with
``jax.jacfwd``); that implementation now lives in the gigalens package, where it is
pinned against lenstronomy and against the simulator's own ``trace``. Keeping a second
copy here meant two lens maps that could silently drift apart.

What stays here is the part that genuinely depends on research-side machinery: the
``Posterior`` -> ``(solver, params)`` adaptation, and the source-plane *image*, which is
rendered from the posterior's lstsq-solved coefficients (:meth:`Posterior.source_plane`).

Source-plane awareness: a model can carry several source planes at different redshifts.
The lens map to a plane with deflection ratio ``r`` is ``beta = theta - r * alpha(theta)``,
so the critical curve, its caustic, AND the cutout border all depend on ``r``. Pass
``deflection_ratio=`` to select a plane; the per-plane ratios come from
:meth:`Posterior.source_plane_views`. Passing ``None`` (the default) lets the solver
derive ``r`` from the plane's own geometry.
"""

from __future__ import annotations

from typing import Optional, Tuple

from matplotlib.axes import Axes

from gigalens.jax.analysis.lens_solver import LensSolver
from gigalens.jax.analysis.lens_solver import plot_caustics as _plot_caustics
from gigalens.jax.analysis.lens_solver import (
    plot_critical_curves as _plot_critical_curves,
)
from gigalens.jax.analysis.source_plane import plot_image_border as _plot_image_border

from .image import plot_image

__all__ = [
    "plot_source_plane",
    "plot_caustics",
    "plot_critical_curves",
    "plot_image_border",
]


def _solver_and_params(posterior, point: str = "median") -> Tuple[LensSolver, dict]:
    """Adapt a :class:`Posterior` to the gigalens lens solver.

    Returns the solver built from the posterior's own ``ProbModel`` -- so the curves come
    from exactly the model being fit -- together with the structured params at ``point``.

    Construction is cheap (no tracing or compilation happens here), so this is called per
    overlay rather than cached; a cached solver would have to be invalidated on ``point``.

    Raises ``NotImplementedError`` for multiplane (>=2 mass planes) models, where the
    image->source map is not ``theta - r*alpha`` with a single deflection field and
    single-plane critical curves are not valid. The message comes from ``LensSolver``.
    """
    # LensSolver reads exactly two attributes off what it is given -- ``.model`` and
    # ``.high_precision`` (``.simulators`` is an optional getattr) -- so a forward-mode
    # context with no ProbModel supplies a stand-in carrying those. See
    # :meth:`SceneContext.solver_source`.
    prob = getattr(posterior.ctx, "prob_model", None)
    if prob is None:
        prob = posterior.ctx.solver_source()
    return LensSolver(prob), posterior.params_at(point)


# ---------------------------------------------------------------------------
# Source-plane image
# ---------------------------------------------------------------------------


def plot_source_plane(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    grid_pix: int = 400,
    fov_arcsec: Optional[float] = None,
    center: Optional[tuple] = None,
    title: Optional[str] = None,
    scale: str = "asinh",
    linear_width: Optional[float] = None,
    log_vmin: float = 1e-2,
    with_caustics: bool = True,
    caustic_color: str = "green",
    with_image_border: bool = True,
    border_color: str = "white",
    dataset: int = 0,
    plane_index: Optional[int] = None,
    deflection_ratio: Optional[float] = None,
) -> None:
    """Render the intrinsic source brightness on a fine source-plane grid.

    Delegates to :meth:`Posterior.source_plane` for the image data and
    :func:`plot_image` for the styling. When ``with_caustics=True`` (default), overlays
    the **caustic** curves (source-plane) for this plane; when ``with_image_border=True``
    (default), also overlays the cutout's image-plane field-of-view boundary ray-traced
    into the source plane (:func:`plot_image_border`) -- source flux outside that curve
    falls off the edge of the observed cutout.

    ``plane_index`` restricts the rendered source light to one (lensed) plane and selects
    which plane's curves to draw; ``deflection_ratio`` overrides that plane's ``r``.
    ``None`` (both defaults) means "all sources, and let the solver derive ``r`` from the
    geometry".

    Framing note: unless ``center`` is given, :meth:`Posterior.source_plane` centers this
    panel's field of view on the first rendered source Component's ``(center_x,
    center_y)``, whereas the ``Observed`` panel is centered on the data frame (0, 0). Pass
    ``center=(0.0, 0.0)`` to align the windows.
    """
    img, extent = posterior.source_plane(
        point=point, grid_pix=grid_pix, fov_arcsec=fov_arcsec, center=center,
        dataset=dataset, plane_index=plane_index,
    )
    plot_image(
        ax, img,
        extent=extent,
        title=title or f"Source plane ({point})",
        scale=scale, linear_width=linear_width, log_vmin=log_vmin,
        colorbar=True, remove_axis=False,
    )
    ax.set_xlabel("x [arcsec]"); ax.set_ylabel("y [arcsec]")
    if with_caustics:
        plot_caustics(ax, posterior, point=point, color=caustic_color,
                      plane=plane_index, deflection_ratio=deflection_ratio)
    if with_image_border:
        plot_image_border(ax, posterior, point=point, color=border_color,
                          plane=plane_index, deflection_ratio=deflection_ratio)


# ---------------------------------------------------------------------------
# Curve overlays -- posterior-keyed adapters over gigalens.jax.analysis
# ---------------------------------------------------------------------------


def plot_caustics(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    color: str = "green",
    linewidth: float = 0.7,
    linestyle: str = "-",
    supersample: int = 20,
    plane: Optional[int] = None,
    deflection_ratio: Optional[float] = None,
    label: Optional[str] = None,
    **kw,
) -> None:
    """Overlay only the caustic curves on ``ax`` (a source-plane axes).

    Caustics live in the source plane; this is the natural overlay for
    :func:`plot_source_plane`. ``plane`` / ``deflection_ratio`` select which source
    plane's caustic to draw. Extra keywords reach
    :meth:`gigalens.jax.analysis.lens_solver.LensSolver.critical_and_caustics` (e.g.
    ``window``, ``scale``, ``max_grid``).
    """
    solver, params = _solver_and_params(posterior, point)
    _plot_caustics(ax, solver, params, color=color, linewidth=linewidth,
                   linestyle=linestyle, label=label, supersample=supersample,
                   plane=plane, deflection_ratio=deflection_ratio, **kw)


def plot_critical_curves(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    color: str = "red",
    linewidth: float = 1.5,
    linestyle: str = "-",
    supersample: int = 20,
    plane: Optional[int] = None,
    deflection_ratio: Optional[float] = None,
    label: Optional[str] = None,
    **kw,
) -> None:
    """Overlay only the critical curves on ``ax`` (an image-plane axes).

    Critical curves live in the image (lens) plane and, like caustics, move with the
    source plane's ``deflection_ratio`` (the Jacobian ``I - r * d alpha/d theta`` depends
    on ``r``).
    """
    solver, params = _solver_and_params(posterior, point)
    _plot_critical_curves(ax, solver, params, color=color, linewidth=linewidth,
                          linestyle=linestyle, label=label, supersample=supersample,
                          plane=plane, deflection_ratio=deflection_ratio, **kw)


def plot_image_border(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    color: str = "white",
    linewidth: float = 1.2,
    linestyle: str = "--",
    n_per_side: int = 250,
    plane: Optional[int] = None,
    deflection_ratio: Optional[float] = None,
    label: Optional[str] = None,
) -> None:
    """Overlay the ray-traced cutout border on a source-plane axes.

    The cutout's image-plane field of view is mapped into the source plane, marking which
    part of it is visible in the observed image. Natural companion to
    :func:`plot_caustics`: the caustic is a property of the mass model, this border a
    property of the observation.
    """
    solver, params = _solver_and_params(posterior, point)
    _plot_image_border(ax, solver, params, color=color, linewidth=linewidth,
                       linestyle=linestyle, label=label, n_per_side=n_per_side,
                       plane=plane, deflection_ratio=deflection_ratio)
