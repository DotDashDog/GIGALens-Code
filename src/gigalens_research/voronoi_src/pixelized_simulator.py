from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gigalens.jax.simulator import _convolve_components
from gigalens.simulator import LensWCS
from objax.functional import average_pool_2d

from .delaunay_mesh import DelaunayMesh


@dataclass(frozen=True)
class PixelizedSourceOutputs:
    basis_images: jnp.ndarray  # (I, H, W) PSF-blurred basis images
    lens_light: jnp.ndarray  # (H, W) PSF-blurred lens light model image
    degenerate_subpix: jnp.ndarray  # scalar int count


class PixelizedSourceSimulator:
    """
    Experimental simulator that produces a pixelised source basis under a
    lens model using a frozen image-plane Delaunay triangulation.

    This intentionally does not implement the full LensSimulatorInterface;
    it provides just what the PixelizedSourceProbModel needs.
    """

    def __init__(
        self,
        *,
        lenses,
        lens_light_profiles,
        sim_config,
        mesh: DelaunayMesh,
    ):
        self.lenses = lenses
        self.lens_light_profiles = lens_light_profiles
        self.sim_config = sim_config
        self.mesh = mesh

        self.supersample = int(sim_config.supersample)
        self.num_pix = int(sim_config.num_pix)
        self.num_pix_sup = self.num_pix * self.supersample

        # Pixel <-> angle transform consistent with gigalens.jax.simulator.LensSimulator
        transform_pix2angle = (
            jnp.eye(2, dtype=jnp.float32) * jnp.float32(sim_config.delta_pix)
            if sim_config.transform_pix2angle is None
            else jnp.array(sim_config.transform_pix2angle, dtype=jnp.float32)
        )
        self.conversion_factor = jnp.linalg.det(transform_pix2angle)
        self.transform_pix2angle = transform_pix2angle / float(self.supersample)

        # Coordinate grids at supersampled resolution. get_coords was removed from
        # gigalens; build the grid via LensWCS exactly as LensSimulator does. Pass the
        # *undivided* transform / delta_pix — LensWCS divides by supersample internally.
        wcs = LensWCS(
            n=self.num_pix,
            supersample=self.supersample,
            transform_pix2angle=sim_config.transform_pix2angle,
            pix_scale=sim_config.delta_pix,
            shift=getattr(sim_config, "angular_shift", (0.0, 0.0)),
        )
        img_X, img_Y = wcs.pixel_grid()
        # Flattened subpixel coordinates used by the mesh lookup (static)
        self.subpix_xy = jnp.array(mesh.subpix_xy, dtype=jnp.float32)  # (J_sub, 2)
        # Also keep grid-shaped coordinates for lens-light rendering / scatter
        self.img_X = jnp.array(img_X, dtype=jnp.float32)  # (H_sup, W_sup)
        self.img_Y = jnp.array(img_Y, dtype=jnp.float32)

        # PSF kernel (shared across basis images and lens light)
        self.flat_kernel = None
        if sim_config.kernel is not None:
            from lenstronomy.Util.kernel_util import subgrid_kernel

            kernel = subgrid_kernel(
                np.array(sim_config.kernel, dtype=np.float32),
                sim_config.supersample,
                odd=True,
            )[::-1, ::-1, np.newaxis, np.newaxis]
            kernel = jnp.array(kernel, dtype=jnp.float32)
            self.flat_kernel = jnp.transpose(kernel, (2, 3, 0, 1))  # (1,1,kh,kw)

        # Static mesh lookup arrays (int32)
        self.simplices = jnp.array(mesh.simplices, dtype=jnp.int32)  # (T,3)
        self.subpix_tri = jnp.array(mesh.subpix_tri, dtype=jnp.int32)  # (J_sub,)
        self.I = int(mesh.seed_xy.shape[0])

        # Seed vertices in image plane (static float32)
        self.seed_xy = jnp.array(mesh.seed_xy, dtype=jnp.float32)  # (I,2)

    def _beta(self, lens_params: List[Dict], x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        beta_x, beta_y = x, y
        for lens, p in zip(self.lenses, lens_params):
            f_xi, f_yi = lens.deriv(x, y, **p)
            beta_x = beta_x - f_xi
            beta_y = beta_y - f_yi
        return beta_x, beta_y

    @staticmethod
    def _barycentric_weights(
        p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, area_eps: float = 1e-6
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute barycentric weights of p in triangle (a,b,c) in 2D.

        Returns (w_a, w_b, w_c, is_degenerate) where is_degenerate is bool.
        """
        # Signed double-area of triangle
        v0 = b - a
        v1 = c - a
        v2 = p - a
        denom = v0[0] * v1[1] - v0[1] * v1[0]
        is_deg = jnp.abs(denom) < area_eps
        denom_safe = jnp.where(is_deg, jnp.sign(denom) * area_eps + area_eps, denom)

        # Solve for weights in basis of v0, v1
        # v2 = u*v0 + v*v1
        u = (v2[0] * v1[1] - v2[1] * v1[0]) / denom_safe
        v = (v0[0] * v2[1] - v0[1] * v2[0]) / denom_safe
        # Guard against extreme ratios when denom is small but not flagged.
        u = jnp.clip(u, -10.0, 10.0)
        v = jnp.clip(v, -10.0, 10.0)
        w_a = 1.0 - u - v
        w_b = u
        w_c = v
        return w_a, w_b, w_c, is_deg

    def _build_basis_sup(
        self,
        lens_params: List[Dict],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return basis at supersampled resolution (I, H_sup, W_sup).
        Also returns count of degenerate sub-pixels.
        """
        # Ray-trace mesh vertices (seeds) to source plane.
        seed_x = self.seed_xy[:, 0]
        seed_y = self.seed_xy[:, 1]
        Mx, My = self._beta(lens_params, seed_x, seed_y)
        M = jnp.stack([Mx, My], axis=-1)  # (I,2)

        # Ray-trace subpixels to source plane.
        sub_x = self.subpix_xy[:, 0]
        sub_y = self.subpix_xy[:, 1]
        betax, betay = self._beta(lens_params, sub_x, sub_y)
        P = jnp.stack([betax, betay], axis=-1)  # (J_sub,2)

        # Gather triangle vertex ids per subpixel
        tri = self.subpix_tri
        inside = tri >= 0

        # For outside points, pick dummy triangle (0) but mask later.
        tri_safe = jnp.where(inside, tri, jnp.zeros_like(tri))
        vids = self.simplices[tri_safe]  # (J_sub,3)

        a_id = vids[:, 0]
        b_id = vids[:, 1]
        c_id = vids[:, 2]

        a = M[a_id]
        b = M[b_id]
        c = M[c_id]

        # vmapped barycentric weights
        w_a, w_b, w_c, is_deg = jax.vmap(self._barycentric_weights)(P, a, b, c)

        # Mask outside and degenerate triangles
        valid = inside & (~is_deg)
        w_a = jnp.where(valid, w_a, 0.0)
        w_b = jnp.where(valid, w_b, 0.0)
        w_c = jnp.where(valid, w_c, 0.0)

        # If triangles fold in the source plane, barycentric weights can be
        # negative. Clipping without renormalizing makes the basis blocky.
        # For Stage 0, clip and renormalize to preserve partition of unity.
        w_a = jnp.clip(w_a, 0.0, 1.0)
        w_b = jnp.clip(w_b, 0.0, 1.0)
        w_c = jnp.clip(w_c, 0.0, 1.0)
        w_sum = w_a + w_b + w_c
        w_sum = jnp.where(w_sum > 0, w_sum, 1.0)
        w_a = w_a / w_sum
        w_b = w_b / w_sum
        w_c = w_c / w_sum

        # Scatter (J_sub, I) dense matrix with 3 non-zeros per row.
        J_sub = P.shape[0]
        K = jnp.zeros((J_sub, self.I), dtype=jnp.float32)
        row = jnp.arange(J_sub, dtype=jnp.int32)
        K = K.at[row, a_id].add(w_a)
        K = K.at[row, b_id].add(w_b)
        K = K.at[row, c_id].add(w_c)

        # Reshape to (I, H_sup, W_sup): currently K is (J_sub, I) with J_sub = H_sup*W_sup
        K = jnp.transpose(K, (1, 0))
        basis_sup = jnp.reshape(K, (self.I, self.num_pix_sup, self.num_pix_sup))
        degenerate_count = jnp.sum(inside & is_deg)
        return basis_sup, degenerate_count

    def _psf_and_downsample(self, img_sup: jnp.ndarray) -> jnp.ndarray:
        """
        Apply shared PSF convolution and average pooling to (I, H_sup, W_sup).
        """
        # _convolve_components expects (bs, depth, h, w). Use bs=1; a single shared
        # PSF (flat_kernel.shape[0] == 1) takes the single-kernel FFT conv path.
        img = img_sup[jnp.newaxis, ...]  # (1, I, H_sup, W_sup)
        if self.flat_kernel is not None:
            img = _convolve_components(img, self.flat_kernel)
        if self.supersample != 1:
            img = average_pool_2d(img, size=(self.supersample, self.supersample), padding="SAME")
        img = jnp.squeeze(img, axis=0)  # (I, H, W)
        return img * self.conversion_factor

    def render_lens_light_sup(self, lens_light_params: List[Dict]) -> jnp.ndarray:
        img = jnp.zeros_like(self.img_X)
        for prof, p in zip(self.lens_light_profiles, lens_light_params):
            img = img + prof.light(self.img_X, self.img_Y, **p)
        return jnp.nan_to_num(img)

    def basis_and_lens_light(
        self,
        params,
    ) -> PixelizedSourceOutputs:
        lens_params = params[0]
        lens_light_params = params[1]

        basis_sup, degenerate_count = self._build_basis_sup(lens_params)
        basis = self._psf_and_downsample(basis_sup)

        lens_light_sup = self.render_lens_light_sup(lens_light_params)
        lens_light = self._psf_and_downsample(lens_light_sup[jnp.newaxis, ...])[0]

        return PixelizedSourceOutputs(
            basis_images=basis,
            lens_light=lens_light,
            degenerate_subpix=degenerate_count,
        )

