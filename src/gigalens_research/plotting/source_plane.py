"""Source-plane and caustic/critical curve plotting.

Two flavors of plot, both keyed on a :class:`Posterior` object so they pick
up the model context automatically:

- :func:`plot_source_plane` renders the intrinsic (unlensed, un-convolved)
  source brightness at a representative point of the posterior.
- :func:`plot_caustics_critical` overlays caustic and critical curves on a
  given axes, computed via lenstronomy from the same point.

The lens-class dispatch from gigalens to lenstronomy names is in
:data:`LENS_MODEL_NAME_MAP`; extend it to support new mass profiles.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .image import plot_image


#: Mapping from a gigalens mass-profile class name to the lenstronomy
#: ``lens_model_list`` name. Extend when adding new profiles.
LENS_MODEL_NAME_MAP: Dict[str, str] = {
    "EPL": "EPL",
    "Shear": "SHEAR",
    "SIE": "SIE",
    "NFW": "NFW",
    "PointMass": "POINT_MASS",
}


def _default_lens_model_list(phys_model) -> List[str]:
    out = []
    for lens in phys_model.lenses:
        name = type(lens).__name__
        if name not in LENS_MODEL_NAME_MAP:
            raise KeyError(
                f"No lenstronomy mapping for gigalens lens class {name!r}. "
                f"Add it to gigalens_research.plotting.source_plane.LENS_MODEL_NAME_MAP "
                f"or pass lens_model_list= explicitly."
            )
        out.append(LENS_MODEL_NAME_MAP[name])
    return out


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
) -> None:
    """Render the intrinsic source brightness on a fine source-plane grid.

    Delegates to :meth:`Posterior.source_plane` for the image data and
    :func:`plot_image` for the styling. When ``with_caustics=True`` (default),
    overlays the caustic curves — which live in the source plane — using
    :func:`plot_caustics`. When ``with_image_border=True`` (default), also
    overlays the cutout's image-plane field-of-view boundary ray-traced into the
    source plane (via :func:`plot_image_border`); source flux outside that curve
    falls off the edge of the observed cutout and is not constrained by it.

    Framing note: unless ``center`` is given, :meth:`Posterior.source_plane`
    centers this panel's field of view on the first source component's
    ``(center_x, center_y)``, whereas the ``Observed`` panel is centered on the
    data frame (0, 0). The two panels therefore share one absolute arcsec
    coordinate system but use *different windows*, so the source appears shifted
    relative to the cutout even though every overlay here is internally
    registered. Pass ``center=(0.0, 0.0)`` to align the windows.
    """
    img, extent = posterior.source_plane(
        point=point, grid_pix=grid_pix, fov_arcsec=fov_arcsec, center=center,
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
        plot_caustics(ax, posterior, point=point, color=caustic_color)
    if with_image_border:
        plot_image_border(ax, posterior, point=point, color=border_color)


def _lens_model_and_kwargs(
    posterior,
    *,
    point: str = "median",
    lens_model_list: Optional[List[str]] = None,
):
    """Build a lenstronomy ``LensModel`` and its ``kwargs_lens`` from the
    posterior at ``point``.

    The lens kwargs are pulled straight from the gigalens mass parameters
    (``x['lens_mass']``) and squeezed to plain floats, so the resulting model
    lives in the same arcsec coordinate system as the gigalens simulator — which
    is why its caustics/curves overlay directly on the source-plane and
    image-plane axes.
    """
    from lenstronomy.LensModel.lens_model import LensModel

    if lens_model_list is None:
        lens_model_list = _default_lens_model_list(posterior.ctx.phys_model)
    lens_model = LensModel(lens_model_list=lens_model_list)
    x = posterior.z_to_x(posterior._point_z(point))
    # New gigalens params are dict-keyed: x['lens_mass'] = {'0': {..}, '1': {..}}.
    # lenstronomy wants a list of param dicts ordered to match lens_model_list
    # (which is built from phys_model.lenses in the same profile order).
    lens_mass = x["lens_mass"]
    kwargs_lens = [
        jax.tree_util.tree_map(
            lambda a: float(np.squeeze(np.asarray(a))), lens_mass[k]
        )
        for k in sorted(lens_mass, key=int)
    ]
    return lens_model, kwargs_lens


def _critical_and_caustic_curves(
    posterior,
    *,
    point: str = "median",
    lens_model_list: Optional[List[str]] = None,
    supersample: int = 20,
):
    """Compute (critical, caustic) curves via lenstronomy and return them as
    matched lists of ``(x_array, y_array)`` polylines.

    Critical-curve segments live in the image plane; caustic segments live in
    the source plane (by definition: caustics are the lens map of critical
    curves). Use them with :func:`plot_caustics` / :func:`plot_critical_curves`
    or overlay yourself.
    """
    # Local imports: lenstronomy is heavy and only needed for this view.
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

    sc = posterior.ctx.sim_config
    lens_model, kwargs_lens = _lens_model_and_kwargs(
        posterior, point=point, lens_model_list=lens_model_list,
    )

    ext = LensModelExtensions(lens_model)
    # Returns four lists of arrays (one entry per disjoint segment).
    crit_x, crit_y, caust_x, caust_y = ext.critical_curve_caustics(
        kwargs_lens,
        compute_window=sc.num_pix * sc.delta_pix,
        grid_scale=sc.delta_pix / supersample,
        center_x=0., center_y=0.,
    )
    critical = list(zip(crit_x, crit_y))
    caustics = list(zip(caust_x, caust_y))
    return critical, caustics


def _image_border_in_source_plane(
    posterior,
    *,
    point: str = "median",
    lens_model_list: Optional[List[str]] = None,
    n_per_side: int = 250,
):
    """Ray-trace the cutout's image-plane border into the source plane.

    Walks the rectangular field-of-view boundary of the data (the same window
    the ``Observed`` panel spans, ``[-N delta/2, +N delta/2]`` on each axis) and
    maps each point through the lens equation ``beta = theta - alpha(theta)``
    with :meth:`LensModel.ray_shooting`. The returned closed polyline bounds the
    region of the source plane that has at least one image inside the cutout —
    i.e. the part of the source actually probed by the observed image. Source
    flux outside this curve is lensed entirely off the edge of the cutout.

    Returns ``(beta_x, beta_y)`` arrays tracing the mapped border (closed).
    """
    lens_model, kwargs_lens = _lens_model_and_kwargs(
        posterior, point=point, lens_model_list=lens_model_list,
    )
    sc = posterior.ctx.sim_config
    half = sc.num_pix * sc.delta_pix / 2.0

    # Perimeter of the image-plane FOV, traversed counter-clockwise and closed.
    t = np.linspace(-half, half, n_per_side)
    ones = np.ones_like(t)
    edge_x = np.concatenate([t, half * ones, t[::-1], -half * ones, [-half]])
    edge_y = np.concatenate([-half * ones, t, half * ones, t[::-1], [-half]])

    beta_x, beta_y = lens_model.ray_shooting(edge_x, edge_y, kwargs_lens)
    return np.asarray(beta_x), np.asarray(beta_y)


def plot_image_border(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    lens_model_list: Optional[List[str]] = None,
    color: str = "white",
    linewidth: float = 1.2,
    linestyle: str = "--",
    n_per_side: int = 250,
    label: Optional[str] = None,
) -> None:
    """Overlay the ray-traced cutout border on a source-plane axes.

    The cutout's image-plane field of view is mapped to the source plane via
    :func:`_image_border_in_source_plane`; the resulting curve marks which part
    of the source plane is visible in the observed image. Natural companion to
    :func:`plot_caustics` on a :func:`plot_source_plane` axes.
    """
    bx, by = _image_border_in_source_plane(
        posterior, point=point, lens_model_list=lens_model_list,
        n_per_side=n_per_side,
    )
    ax.plot(bx, by, color=color, linewidth=linewidth, linestyle=linestyle,
            label=label)


def plot_caustics(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    lens_model_list: Optional[List[str]] = None,
    color: str = "green",
    linewidth: float = 1.5,
    linestyle: str = "-",
    supersample: int = 20,
    label: Optional[str] = None,
) -> None:
    """Overlay only the caustic curves on ``ax`` (typically a source-plane axes).

    Caustics live in the source plane; this is the natural overlay for
    :func:`plot_source_plane`. To overlay both caustics and critical curves
    (the usual lenstronomy preset) on an image-plane axes, use
    :func:`plot_caustics_critical` instead.
    """
    _, caustics = _critical_and_caustic_curves(
        posterior, point=point, lens_model_list=lens_model_list, supersample=supersample,
    )
    for i, (xs, ys) in enumerate(caustics):
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                label=label if i == 0 else None)


def plot_critical_curves(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    lens_model_list: Optional[List[str]] = None,
    color: str = "red",
    linewidth: float = 1.5,
    linestyle: str = "-",
    supersample: int = 20,
    label: Optional[str] = None,
) -> None:
    """Overlay only the critical curves on ``ax`` (typically an image-plane axes)."""
    critical, _ = _critical_and_caustic_curves(
        posterior, point=point, lens_model_list=lens_model_list, supersample=supersample,
    )
    for i, (xs, ys) in enumerate(critical):
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                label=label if i == 0 else None)


def plot_caustics_critical(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    lens_model_list: Optional[List[str]] = None,
    color_critical: str = "red",
    color_caustic: str = "green",
    supersample: int = 20,
    **_unused,
) -> None:
    """Overlay both caustic *and* critical curves on a single ``ax``.

    Image-plane convention: caustics (source plane) and critical curves
    (image plane) are drawn together. For source-plane-only or
    image-plane-only views, use :func:`plot_caustics` or
    :func:`plot_critical_curves` respectively.
    """
    critical, caustics = _critical_and_caustic_curves(
        posterior, point=point, lens_model_list=lens_model_list, supersample=supersample,
    )
    for xs, ys in critical:
        ax.plot(xs, ys, color=color_critical, linewidth=1.5)
    for xs, ys in caustics:
        ax.plot(xs, ys, color=color_caustic, linewidth=1.5)
