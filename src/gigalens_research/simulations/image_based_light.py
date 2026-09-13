"""Image-backed gigalens light profile.

:class:`ImageBasedLight` lifts a discrete 2-D surface-brightness image to a
*continuous* light profile via bilinear interpolation. It plugs into a
:class:`gigalens.model.PhysicalModel` like any other light profile and is the
natural way to lens a real (e.g. Vela cosmological-simulation) image.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import gigalens.profile


class ImageBasedLight(gigalens.profile.LightProfile):
    """Light profile backed by a bilinearly-interpolated 2-D image.

    Parameters
    ----------
    image : ndarray, shape (N, N)
        Square surface-brightness image. Units are whatever the caller wants
        the simulator to return per arcsec² (e.g. cps/arcsec²); the profile
        does not rescale.
    image_pixel_scale : float
        Pixel scale of ``image``, in arcseconds per pixel. The image is
        centered on (0, 0): pixel ``(i, j)`` corresponds to source-plane
        coordinate ``((j - (N-1)/2) * scale, (i - (N-1)/2) * scale)``.

    Notes
    -----
    Pixel orientation follows the convention in
    ``experiments/vela_sim_systems/lens_vela_system.ipynb``: the first axis
    of ``image`` is treated as ``x`` and the second as ``y`` when the
    interpolator is queried via ``points = stack([x, y], axis=-1)``. If your
    upstream pipeline transposes the image before display (as the notebook
    does for plotting), be consistent here so the truth and the recovered
    source share a frame.

    The profile exposes ``center_x`` / ``center_y`` as free parameters; the
    image content is treated as a fixed (non-trainable) template. Like every
    gigalens light profile it also carries an amplitude parameter, here named
    ``amp`` (a dimensionless multiplier on the image, default 1.0). The scene API
    requires the amplitude name to be declared (the base class appends
    ``_amp`` to ``params`` when ``use_lstsq=False``); before this was declared the
    profile could not be placed in a :class:`gigalens.jax.scene.LensModel`.
    """

    _name = "IMAGEBASEDLIGHT"
    _params = ["center_x", "center_y"]
    _amp = "amp"

    def __init__(self, image, image_pixel_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        image = jnp.asarray(image, dtype=jnp.float32)
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError(
                f"ImageBasedLight requires a square 2-D image; got shape "
                f"{tuple(image.shape)}."
            )
        self.source_img = image
        self.image_pixel_scale = float(image_pixel_scale)

        num_pix = int(self.source_img.shape[0])
        half_extent = (num_pix - 1) / 2.0 * self.image_pixel_scale
        coords_1d = np.linspace(
            -half_extent, half_extent, num_pix, dtype=np.float32,
        )
        self.interpolator = jax.scipy.interpolate.RegularGridInterpolator(
            (coords_1d, coords_1d),
            self.source_img,
            method="linear",
            fill_value=0.0,
        )

    def light(self, x, y, center_x, center_y, amp=1.0):
        points = jnp.stack([x - center_x, y - center_y], axis=-1)
        return amp * self.interpolator(points)
