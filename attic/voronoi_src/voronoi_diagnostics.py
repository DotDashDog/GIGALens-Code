from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class Stage0Diagnostics:
    data: np.ndarray
    model: np.ndarray
    norm_resid: np.ndarray
    source_plane: np.ndarray
    map_loss: np.ndarray
    title: str
    image_extent: Tuple[float, float, float, float]
    source_extent: Tuple[float, float, float, float]
    lens_light: Optional[np.ndarray] = None
    lensed_source: Optional[np.ndarray] = None
    basis_coverage: Optional[np.ndarray] = None


def plot_stage0_panel(
    diag: Stage0Diagnostics,
    *,
    save_path: Optional[str] = None,
    cmap: str = "magma",
):
    ncols = 8 if diag.lens_light is not None else 5
    fig, axs = plt.subplots(1, ncols, figsize=(4.4 * ncols, 4))
    if ncols == 5:
        ax_data, ax_model, ax_resid, ax_src, ax_loss = axs
    else:
        ax_data, ax_model, ax_lens, ax_lensed_src, ax_coverage, ax_resid, ax_src, ax_loss = axs

    def _log_norm(arr):
        positive = np.asarray(arr)[np.asarray(arr) > 0]
        if positive.size == 0:
            return None
        vmin = max(float(np.nanpercentile(positive, 0.5)), np.finfo(float).tiny)
        vmax = float(np.nanpercentile(positive, 99.8))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return None
        return LogNorm(vmin=vmin, vmax=vmax)

    im0 = ax_data.imshow(
        diag.data,
        origin="lower",
        cmap=cmap,
        norm=_log_norm(diag.data),
        extent=diag.image_extent,
    )
    ax_data.set_title("Data")
    ax_data.set_xlabel("x [arcsec]")
    ax_data.set_ylabel("y [arcsec]")
    plt.colorbar(im0, ax=ax_data, fraction=0.046, pad=0.04)

    im1 = ax_model.imshow(
        diag.model,
        origin="lower",
        cmap=cmap,
        norm=_log_norm(diag.model),
        extent=diag.image_extent,
    )
    ax_model.set_title("Model")
    ax_model.set_xlabel("x [arcsec]")
    ax_model.set_ylabel("y [arcsec]")
    plt.colorbar(im1, ax=ax_model, fraction=0.046, pad=0.04)

    if ncols == 8:
        im_lens = ax_lens.imshow(
            diag.lens_light,
            origin="lower",
            cmap=cmap,
            norm=_log_norm(diag.lens_light),
            extent=diag.image_extent,
        )
        ax_lens.set_title("Lens light")
        ax_lens.set_xlabel("x [arcsec]")
        ax_lens.set_ylabel("y [arcsec]")
        plt.colorbar(im_lens, ax=ax_lens, fraction=0.046, pad=0.04)

        im_src = ax_lensed_src.imshow(
            diag.lensed_source,
            origin="lower",
            cmap=cmap,
            norm=_log_norm(np.maximum(diag.lensed_source, 0)),
            extent=diag.image_extent,
        )
        ax_lensed_src.set_title("Lensed source")
        ax_lensed_src.set_xlabel("x [arcsec]")
        ax_lensed_src.set_ylabel("y [arcsec]")
        plt.colorbar(im_src, ax=ax_lensed_src, fraction=0.046, pad=0.04)

        im_cov = ax_coverage.imshow(
            diag.basis_coverage,
            origin="lower",
            cmap="viridis",
            extent=diag.image_extent,
        )
        ax_coverage.set_title("Basis coverage")
        ax_coverage.set_xlabel("x [arcsec]")
        ax_coverage.set_ylabel("y [arcsec]")
        plt.colorbar(im_cov, ax=ax_coverage, fraction=0.046, pad=0.04)

    im2 = ax_resid.imshow(
        diag.norm_resid,
        origin="lower",
        cmap="coolwarm",
        vmin=-5,
        vmax=5,
        extent=diag.image_extent,
    )
    ax_resid.set_title("Norm residual (data-model)/err")
    ax_resid.set_xlabel("x [arcsec]")
    ax_resid.set_ylabel("y [arcsec]")
    plt.colorbar(im2, ax=ax_resid, fraction=0.046, pad=0.04)

    im3 = ax_src.imshow(
        diag.source_plane,
        origin="lower",
        cmap=cmap,
        extent=diag.source_extent,
    )
    ax_src.set_title("Source plane (rendered)")
    ax_src.set_xlabel("x [arcsec]")
    ax_src.set_ylabel("y [arcsec]")
    plt.colorbar(im3, ax=ax_src, fraction=0.046, pad=0.04)

    ax_loss.plot(diag.map_loss)
    ax_loss.set_title("MAP best chi^2 history")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("chi^2 (mean)")

    fig.suptitle(diag.title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=180)
        plt.close(fig)
    return fig


def render_source_on_grid(
    *,
    source_values: jnp.ndarray,  # (I,)
    seed_xy_src: jnp.ndarray,  # (I,2) in source plane (for this mass model)
    out_npix: int = 120,
    out_extent: float = 1.5,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Lightweight visualization helper: grid the irregular source pixel values
    by nearest-neighbour assignment in source plane.

    Stage 0 only: this is for plotting, not for likelihood evaluation.
    """
    s = np.asarray(source_values)
    xy = np.asarray(seed_xy_src)
    gx = np.linspace(-out_extent, out_extent, out_npix)
    gy = np.linspace(-out_extent, out_extent, out_npix)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # brute-force nearest neighbour (small I)
    d2 = np.sum((pts[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    nn = np.argmin(d2, axis=1)
    img = s[nn].reshape(out_npix, out_npix)
    extent = (-out_extent, out_extent, -out_extent, out_extent)
    return img, extent

