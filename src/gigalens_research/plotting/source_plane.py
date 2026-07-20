"""Source-plane and caustic/critical curve plotting.

Two flavors of plot, both keyed on a :class:`Posterior` object so they pick
up the model context automatically:

- :func:`plot_source_plane` renders the intrinsic (unlensed, un-convolved)
  source brightness at a representative point of the posterior.
- :func:`plot_caustics` / :func:`plot_critical_curves` overlay the caustic
  (source-plane) or critical (image-plane) curves computed from the same point.

Curves are computed **natively** from the gigalens lens model — the same mass
profiles being fit — via :func:`_critical_and_caustic_curves`. There is no
lenstronomy translation layer (no name map): the deflection comes straight from
each ``profile.deriv``, its Jacobian from :func:`jax.jacfwd`, and the
critical-curve contours from ``skimage.measure.find_contours``.

Source-plane awareness: a model can carry several source planes at different
redshifts. The lens map to a source plane with deflection ratio ``r`` is
``beta = theta - r * alpha(theta)``, so BOTH the critical curve
(``det(I - r * d alpha/d theta) = 0``) and its caustic (the lens map of that
curve) depend on ``r``. Pass ``deflection_ratio=`` to render the pair for a
given source plane; the per-plane ratios come from
:meth:`Posterior.source_plane_views`.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.axes import Axes

from .image import plot_image

# Upper bound on the (square) grid used to trace critical/caustic curves; see
# the comment at its use site in _critical_and_caustic_curves.
_MAX_CAUSTIC_GRID_PIX = 800


# ---------------------------------------------------------------------------
# Native deflection from the gigalens lens model
# ---------------------------------------------------------------------------


def _native_alpha(posterior, *, point: str = "median"):
    """Build a jax-traceable deflection ``alpha(x, y) -> (alpha_x, alpha_y)`` from
    the posterior's gigalens lens model at ``point``.

    The returned closure sums every mass Component's ``profile.deriv`` with its
    fitted parameters (squeezed to scalars), so it is the *reference* reduced
    deflection — the deflection to the cosmology's ``z_source_ref`` (ratio 1).
    Scale it by a source plane's ``deflection_ratio`` to get that plane's lens map.

    Raises ``NotImplementedError`` for the multiplane (>=2 mass planes) trace
    mode: there the image->source map is not ``theta - r * alpha`` with a single
    deflection field, so single-plane critical/caustic curves are not valid. (The
    natural generalization is the Jacobian of the full multiplane ``trace``; not
    yet implemented.)
    """
    scene = posterior._scene_model
    sim = posterior._sim_for(0)
    trace_mode = getattr(sim, "trace_mode", "deflection_ratio")
    if trace_mode != "deflection_ratio":
        raise NotImplementedError(
            "native caustic/critical curves currently support only the "
            "single-deflector (deflection_ratio) trace mode; this model is "
            f"{trace_mode!r} (>=2 mass planes), for which single-plane caustics "
            "are not valid. Generalizing to the multiplane trace Jacobian is "
            "future work."
        )
    params = scene.to_params(dict(posterior.z_to_x(posterior._point_z(point))))
    mass = []
    for i, plane in enumerate(scene.planes):
        for j, comp in enumerate(plane.mass):
            kw = {k: jnp.squeeze(jnp.asarray(v))
                  for k, v in params["planes"][i]["mass"][j].items()}
            mass.append((comp.profile, kw))
    if not mass:
        raise ValueError(
            "no mass Components in this model; cannot compute caustic/critical curves.")

    def alpha(x, y):
        ax = jnp.zeros_like(x)
        ay = jnp.zeros_like(y)
        for profile, kw in mass:
            dx, dy = profile.deriv(x, y, **kw)
            ax = ax + dx
            ay = ay + dy
        return ax, ay

    return alpha


def _critical_and_caustic_curves(
    posterior,
    *,
    point: str = "median",
    supersample: int = 20,
    deflection_ratio: float = 1.0,
):
    """Compute (critical, caustic) curves natively from the gigalens lens model.

    For a source plane with deflection ratio ``r`` the lens map is
    ``beta = theta - r * alpha(theta)`` with Jacobian ``A = I - r * d alpha/d theta``.
    Critical curves are the ``det(A) = 0`` contours (image plane); each is mapped
    through the lens equation to its caustic (source plane). Returns matched lists
    of ``(x_array, y_array)`` polylines, one entry per disjoint segment.

    At ``deflection_ratio=1`` this reduces exactly to lenstronomy's
    ``critical_curve_caustics`` (``det(A) = 1/magnification``,
    ``beta = theta - alpha``) — the parity check used to certify the native path.
    """
    from skimage.measure import find_contours

    sc = posterior.ctx.sim_config
    dr = float(deflection_ratio)
    alpha = _native_alpha(posterior, point=point)

    compute_window = sc.num_pix * sc.delta_pix
    grid_scale = sc.delta_pix / supersample
    num_pix = int(round(compute_window / grid_scale))
    if num_pix % 2 == 1:
        num_pix += 1
    # num_pix scales as sc.num_pix * supersample, so for large images the
    # default supersample=20 (inherited from the old lenstronomy tracer, whose
    # base grid was much smaller than sc.num_pix) can demand a many-thousand^2
    # grid; jax.vmap(jax.jacfwd(...)) over that OOMs during XLA autotuning.
    # Cap the grid size directly, coarsening grid_scale but keeping the FOV.
    if num_pix > _MAX_CAUSTIC_GRID_PIX:
        num_pix = _MAX_CAUSTIC_GRID_PIX
        grid_scale = compute_window / num_pix
    coord = (jnp.arange(num_pix) - (num_pix - 1) / 2.0) * grid_scale
    # X varies along axis 1 (columns), Y along axis 0 (rows): matches the skimage
    # find_contours convention (v[:, 0] = row = y, v[:, 1] = col = x).
    X, Y = jnp.meshgrid(coord, coord)
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    def alpha_vec(p):
        ax, ay = alpha(p[0], p[1])
        return jnp.stack([ax, ay])

    # Per-point 2x2 deflection Jacobian: jac[a, b] = d alpha_a / d theta_b.
    jac = jax.vmap(jax.jacfwd(alpha_vec))(pts)  # (Npts, 2, 2)
    Jxx, Jxy = jac[:, 0, 0], jac[:, 0, 1]
    Jyx, Jyy = jac[:, 1, 0], jac[:, 1, 1]
    det_A = (1.0 - dr * Jxx) * (1.0 - dr * Jyy) - (dr ** 2) * (Jxy * Jyx)
    det_img = np.asarray(det_A).reshape(num_pix, num_pix)

    half = grid_scale * (num_pix - 1) / 2.0
    critical, caustics = [], []
    for v in find_contours(det_img, 0.0):
        ra = v[:, 1] * grid_scale - half
        dec = v[:, 0] * grid_scale - half
        critical.append((ra, dec))
        ax, ay = alpha(jnp.asarray(ra), jnp.asarray(dec))
        ra_c = ra - dr * np.asarray(ax)
        dec_c = dec - dr * np.asarray(ay)
        caustics.append((ra_c, dec_c))
    return critical, caustics


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
    deflection_ratio: float = 1.0,
) -> None:
    """Render the intrinsic source brightness on a fine source-plane grid.

    Delegates to :meth:`Posterior.source_plane` for the image data and
    :func:`plot_image` for the styling. When ``with_caustics=True`` (default),
    overlays the **caustic** curves (source-plane) for this plane's
    ``deflection_ratio`` via :func:`plot_caustics`. When ``with_image_border=True``
    (default), also overlays the cutout's image-plane field-of-view boundary
    ray-traced into the source plane (via :func:`plot_image_border`); source flux
    outside that curve falls off the edge of the observed cutout.

    ``plane_index`` restricts the rendered source light to one (lensed) plane;
    ``deflection_ratio`` is that plane's ratio (so the caustic/border match the
    rendered source). Both default to "all sources, reference ratio 1".

    Framing note: unless ``center`` is given, :meth:`Posterior.source_plane`
    centers this panel's field of view on the first rendered source Component's
    ``(center_x, center_y)``, whereas the ``Observed`` panel is centered on the
    data frame (0, 0). Pass ``center=(0.0, 0.0)`` to align the windows.
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
                      deflection_ratio=deflection_ratio)
    if with_image_border:
        plot_image_border(ax, posterior, point=point, color=border_color,
                          deflection_ratio=deflection_ratio)


# ---------------------------------------------------------------------------
# Image-plane border mapped to the source plane
# ---------------------------------------------------------------------------


def _image_border_in_source_plane(
    posterior,
    *,
    point: str = "median",
    deflection_ratio: float = 1.0,
    n_per_side: int = 250,
):
    """Ray-trace the cutout's image-plane border into the source plane.

    Walks the rectangular field-of-view boundary of the data and maps each point
    through the lens equation ``beta = theta - r * alpha(theta)`` (native
    deflection, scaled by this plane's ratio ``r``). The returned closed polyline
    bounds the region of the source plane that has at least one image inside the
    cutout. Returns ``(beta_x, beta_y)`` arrays (closed).
    """
    alpha = _native_alpha(posterior, point=point)
    sc = posterior.ctx.sim_config
    dr = float(deflection_ratio)
    half = sc.num_pix * sc.delta_pix / 2.0

    # Perimeter of the image-plane FOV, traversed counter-clockwise and closed.
    t = np.linspace(-half, half, n_per_side)
    ones = np.ones_like(t)
    edge_x = np.concatenate([t, half * ones, t[::-1], -half * ones, [-half]])
    edge_y = np.concatenate([-half * ones, t, half * ones, t[::-1], [-half]])

    ax, ay = alpha(jnp.asarray(edge_x), jnp.asarray(edge_y))
    beta_x = edge_x - dr * np.asarray(ax)
    beta_y = edge_y - dr * np.asarray(ay)
    return np.asarray(beta_x), np.asarray(beta_y)


def plot_image_border(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    deflection_ratio: float = 1.0,
    color: str = "white",
    linewidth: float = 1.2,
    linestyle: str = "--",
    n_per_side: int = 250,
    label: Optional[str] = None,
) -> None:
    """Overlay the ray-traced cutout border on a source-plane axes.

    The cutout's image-plane field of view is mapped to the source plane via
    :func:`_image_border_in_source_plane` at this plane's ``deflection_ratio``;
    the resulting curve marks which part of the source plane is visible in the
    observed image. Natural companion to :func:`plot_caustics`.
    """
    bx, by = _image_border_in_source_plane(
        posterior, point=point, deflection_ratio=deflection_ratio,
        n_per_side=n_per_side,
    )
    ax.plot(bx, by, color=color, linewidth=linewidth, linestyle=linestyle,
            label=label)


# ---------------------------------------------------------------------------
# Curve overlays
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
    deflection_ratio: float = 1.0,
    label: Optional[str] = None,
) -> None:
    """Overlay only the caustic curves on ``ax`` (a source-plane axes).

    Caustics live in the source plane; this is the natural overlay for
    :func:`plot_source_plane`. ``deflection_ratio`` selects which source plane's
    caustic to draw (default 1 = reference redshift).
    """
    _, caustics = _critical_and_caustic_curves(
        posterior, point=point, supersample=supersample,
        deflection_ratio=deflection_ratio,
    )
    for i, (xs, ys) in enumerate(caustics):
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                label=label if i == 0 else None)


def plot_critical_curves(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    color: str = "red",
    linewidth: float = 1.5,
    linestyle: str = "-",
    supersample: int = 20,
    deflection_ratio: float = 1.0,
    label: Optional[str] = None,
) -> None:
    """Overlay only the critical curves on ``ax`` (an image-plane axes).

    Critical curves live in the image (lens) plane and, like caustics, move with
    the source plane's ``deflection_ratio`` (the Jacobian ``I - r * d alpha/d theta``
    depends on ``r``)."""
    critical, _ = _critical_and_caustic_curves(
        posterior, point=point, supersample=supersample,
        deflection_ratio=deflection_ratio,
    )
    for i, (xs, ys) in enumerate(critical):
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                label=label if i == 0 else None)


def plot_caustics_critical(
    ax: Axes,
    posterior,
    *,
    point: str = "median",
    color_critical: str = "red",
    color_caustic: str = "green",
    supersample: int = 20,
    deflection_ratio: float = 1.0,
    **_unused,
) -> None:
    """Overlay both caustic *and* critical curves on a single ``ax``.

    Kept for convenience, but note the two curves live in *different* planes
    (caustics: source; critical: image). The reports draw them separately —
    critical on the image/lens panel, caustics on the source panel. Prefer
    :func:`plot_caustics` / :func:`plot_critical_curves` for a single plane.
    """
    critical, caustics = _critical_and_caustic_curves(
        posterior, point=point, supersample=supersample,
        deflection_ratio=deflection_ratio,
    )
    for xs, ys in critical:
        ax.plot(xs, ys, color=color_critical, linewidth=1.5)
    for xs, ys in caustics:
        ax.plot(xs, ys, color=color_caustic, linewidth=1.5)
