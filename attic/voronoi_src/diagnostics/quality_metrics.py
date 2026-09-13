from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def alternating_pattern_score(
    source_coeffs: np.ndarray,
    edges: np.ndarray,
    *,
    bright_fraction: float = 0.1,
) -> float:
    """
    Mean per-edge sign product (s_i - <s>) * (s_k - <s>), bounded in [-1, 1],
    restricted to edges incident to the brightest ``bright_fraction`` vertices.

    Strongly negative values indicate checkerboard / alternating tiles among
    the brightest vertices; positive values indicate locally smooth bright
    structure. Using the sign keeps the score scale-free so the same
    threshold applies across systems and reg-strength regimes.
    """
    s = np.asarray(source_coeffs, dtype=np.float64).reshape(-1)
    edges_np = np.asarray(edges, dtype=np.int32)
    if edges_np.size == 0 or s.size == 0:
        return 0.0

    threshold = np.quantile(s, 1.0 - float(bright_fraction))
    bright = s >= threshold
    if not np.any(bright):
        return 0.0

    i, k = edges_np[:, 0], edges_np[:, 1]
    mask = bright[i] | bright[k]
    if not np.any(mask):
        return 0.0

    smean = float(np.mean(s))
    di = s[i[mask]] - smean
    dk = s[k[mask]] - smean
    signs = np.sign(di) * np.sign(dk)
    return float(np.mean(signs))


def vertex_density_kde_grid(
    points_xy: np.ndarray,
    extent: float,
    *,
    out_npix: int = 120,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """2D Gaussian KDE of point density on a square grid."""
    from scipy.stats import gaussian_kde

    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 3:
        grid = np.zeros((out_npix, out_npix), dtype=np.float32)
        ext = (-extent, extent, -extent, extent)
        return grid, ext

    kde = gaussian_kde(pts.T, bw_method=bandwidth)
    gx = np.linspace(-extent, extent, out_npix)
    gy = np.linspace(-extent, extent, out_npix)
    xx, yy = np.meshgrid(gx, gy, indexing="xy")
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(out_npix, out_npix)
    density = density / max(float(np.max(density)), np.finfo(float).tiny)
    return density.astype(np.float32), (-extent, extent, -extent, extent)


def plot_vertex_density_maps(
    *,
    seed_xy_image: np.ndarray,
    seed_xy_source: np.ndarray,
    extent_image: float,
    extent_source: float,
    out_path: str,
    title: str = "Vertex density",
) -> dict[str, Any]:
    """Save image-plane and source-plane vertex-density heatmaps."""
    img_density, img_ext = vertex_density_kde_grid(seed_xy_image, extent_image)
    src_density, src_ext = vertex_density_kde_grid(seed_xy_source, extent_source)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = axs[0].imshow(img_density, origin="lower", cmap="viridis", extent=img_ext)
    axs[0].set_title("Image-plane vertex density")
    axs[0].set_xlabel("x [arcsec]")
    axs[0].set_ylabel("y [arcsec]")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(src_density, origin="lower", cmap="viridis", extent=src_ext)
    axs[1].set_title("Source-plane vertex density")
    axs[1].set_xlabel("source x [arcsec]")
    axs[1].set_ylabel("source y [arcsec]")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        "image_plane_density_max": float(np.max(img_density)),
        "source_plane_density_max": float(np.max(src_density)),
        "image_plane_density_peak_xy": _peak_on_grid(img_density, img_ext),
        "source_plane_density_peak_xy": _peak_on_grid(src_density, src_ext),
    }


def _peak_on_grid(grid: np.ndarray, extent: tuple[float, float, float, float]) -> tuple[float, float]:
    arr = np.asarray(grid)
    iy, ix = np.unravel_index(np.argmax(arr), arr.shape)
    x0, x1, y0, y1 = extent
    x = np.linspace(x0, x1, arr.shape[1])[ix]
    y = np.linspace(y0, y1, arr.shape[0])[iy]
    return (float(x), float(y))


def lambda_evidence_scan(
    *,
    prob_model,
    simulator,
    x_truth_mass: tuple,
    lambda_values: list[float] | np.ndarray,
    z_truth: jnp.ndarray | None = None,
) -> list[dict[str, float]]:
    """
    Evaluate chi^2 and Bayesian evidence terms at truth for a grid of lambda.

    ``x_truth_mass`` is (lens_params, lens_light_params) in physical space; only
    lambda is varied across the scan.
    """
    del z_truth
    lam_list = [float(v) for v in lambda_values]
    rows: list[dict[str, float]] = []

    for lam_float in lam_list:
        x = (
            x_truth_mass[0],
            x_truth_mass[1],
            [{"lambda": jnp.array([lam_float], dtype=jnp.float32)}],
        )
        z = jnp.squeeze(jnp.stack(prob_model.bij.inverse(x)), axis=-1)
        terms = prob_model.debug_terms(simulator, z)
        rows.append(
            {
                "lambda": lam_float,
                "chi2_mean": float(terms["chi2_mean"]),
                "chi2": float(terms["chi2"]),
                "sHs": float(terms["sHs"]),
                "logdetA": float(terms["logdetA"]),
                "logdetH": float(terms["logdetH"]),
                "logZ": float(terms["logZ"]),
                "log_prior": float(terms["log_prior"]),
                "log_target": float(terms["log_target"]),
                "minus2logZ": float(-2.0 * terms["logZ"]),
            }
        )
    return rows


def pick_evidence_optimal_lambda(evidence_rows: list[dict[str, float]]) -> dict[str, float]:
    """Return the row with minimum -2 ln Z."""
    return min(evidence_rows, key=lambda r: r["minus2logZ"])
