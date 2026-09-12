from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from voronoi_src.pixelized_simulator import PixelizedSourceSimulator


def compute_degeneracy_stats(
    simulator: PixelizedSourceSimulator,
    lens_params,
) -> dict[str, Any]:
    """Return sub-pixel and per-vertex degenerate-triangle statistics at fixed mass."""
    seed_x = simulator.seed_xy[:, 0]
    seed_y = simulator.seed_xy[:, 1]
    Mx, My = simulator._beta(lens_params, seed_x, seed_y)
    M = jnp.stack([Mx, My], axis=-1)

    sub_x = simulator.subpix_xy[:, 0]
    sub_y = simulator.subpix_xy[:, 1]
    betax, betay = simulator._beta(lens_params, sub_x, sub_y)
    P = jnp.stack([betax, betay], axis=-1)

    tri = simulator.subpix_tri
    inside = tri >= 0
    tri_safe = jnp.where(inside, tri, jnp.zeros_like(tri))
    vids = simulator.simplices[tri_safe]
    a = M[vids[:, 0]]
    b = M[vids[:, 1]]
    c = M[vids[:, 2]]
    _, _, _, is_deg = jax.vmap(simulator._barycentric_weights)(P, a, b, c)
    is_deg = np.asarray(jax.device_get(is_deg))
    inside_np = np.asarray(jax.device_get(inside))

    n_inside = int(np.sum(inside_np))
    n_deg = int(np.sum(inside_np & is_deg))
    frac = float(n_deg / max(n_inside, 1))

    # Per-vertex: fraction of incident sub-pixels that are degenerate.
    I = simulator.I
    tri_np = np.asarray(jax.device_get(tri_safe))
    vids_np = np.asarray(jax.device_get(simulator.simplices))[tri_np]  # (J_sub, 3)
    inside_mask = inside_np
    vids_inside = vids_np[inside_mask].reshape(-1)
    deg_inside = np.repeat(is_deg[inside_mask].astype(np.int64), 3)
    vertex_counts = np.zeros(I, dtype=np.int64)
    vertex_deg = np.zeros(I, dtype=np.int64)
    np.add.at(vertex_counts, vids_inside, 1)
    np.add.at(vertex_deg, vids_inside, deg_inside)
    vertex_frac = np.where(vertex_counts > 0, vertex_deg / np.maximum(vertex_counts, 1), 0.0)

    return {
        "degenerate_subpix_count": n_deg,
        "inside_subpix_count": n_inside,
        "degenerate_subpix_fraction": frac,
        "degenerate_vertex_fraction_max": float(np.max(vertex_frac)),
        "degenerate_vertex_fraction_mean": float(np.mean(vertex_frac[vertex_counts > 0]))
        if np.any(vertex_counts > 0)
        else 0.0,
    }


def plot_caustic_vertex_overlay(
    *,
    lenses,
    lens_params,
    seed_xy_image: np.ndarray,
    extent: float,
    out_path: str,
    num_pix_grid: int = 200,
) -> None:
    """Overlay seed vertices on a coarse critical-curve proxy (det J = 0 contour)."""
    import matplotlib.pyplot as plt

    from voronoi_src.delaunay_mesh import _squeeze_lens_params

    lp = _squeeze_lens_params(lens_params)
    x = np.linspace(-extent, extent, num_pix_grid)
    y = np.linspace(-extent, extent, num_pix_grid)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    def det_jacobian(xv, yv):
        pts = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)

        def beta_vec(th):
            bx, by = th[0], th[1]
            for lens, p in zip(lenses, lp):
                ax, ay = lens.deriv(bx, by, **p)
                bx = bx - ax
                by = by - ay
            return jnp.stack([jnp.squeeze(bx), jnp.squeeze(by)])

        def det_one(pt):
            return jnp.linalg.det(jax.jacfwd(beta_vec)(pt))

        return jax.vmap(det_one)(pts).reshape(xv.shape)

    det = np.asarray(jax.device_get(det_jacobian(jnp.asarray(xx), jnp.asarray(yy))))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.contour(xx, yy, det, levels=[0.0], colors="cyan", linewidths=1.0)
    ax.scatter(
        seed_xy_image[:, 0],
        seed_xy_image[:, 1],
        s=8,
        c="white",
        edgecolors="k",
        linewidths=0.2,
        alpha=0.85,
    )
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    ax.set_title("Caustic proxy (det J = 0) + mesh vertices")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
