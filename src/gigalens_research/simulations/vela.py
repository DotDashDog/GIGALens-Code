"""Load a Vela-catalog source as an :class:`ImageBasedLight`.

The conventional layout (see
``experiments/vela_sim_systems/lens_vela_system.ipynb``) for a single Vela
source directory is::

    <source_dir>/
        source_image.npy   # 2-D image, surface brightness in nJy / pixel
        psf.npy            # 2-D PSF kernel (for the instrument)
        metadata.json      # {"source_pixel_scale_arcsec": float,
                           #  "instrument_pixel_scale_arcsec": float,
                           #  "photfnu_Jy": float, ...}

:func:`load_vela_source` reads these three files, converts the image to the
units the gigalens simulator expects (cps / arcsec²), wraps it in an
:class:`ImageBasedLight`, and returns a :class:`VelaSource` bundle.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .image_based_light import ImageBasedLight


@dataclass
class VelaSource:
    """Bundle of everything needed to lens a Vela source.

    Attributes
    ----------
    light : ImageBasedLight
        The light profile to drop into a :class:`gigalens.model.PhysicalModel`
        (e.g. ``PhysicalModel(mass=[...], lens_light=[...], src_light=[vela.light])``).
    psf : ndarray
        The instrument PSF kernel, suitable for
        :class:`gigalens.simulator.SimulatorConfig`'s ``kernel`` argument.
    surface_brightness_cps_per_arcsec2 : ndarray
        The image actually backing ``light``, in cps / arcsec². Useful for
        plotting the truth source or passing to
        :func:`gigalens_research.inference_utils.source_comparison` as a
        pre-rendered truth.
    source_pixel_scale_arcsec : float
        Pixel scale of the source image (arcsec / pixel).
    instrument_pixel_scale_arcsec : float
        Detector pixel scale (arcsec / pixel) for the corresponding
        observation.
    metadata : dict
        The raw metadata.json contents, in case callers need extra fields.
    """

    light: ImageBasedLight
    psf: np.ndarray
    surface_brightness_cps_per_arcsec2: np.ndarray
    source_pixel_scale_arcsec: float
    instrument_pixel_scale_arcsec: float
    metadata: Dict[str, Any]

    def source_extent(self) -> Tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` of the native source-image grid,
        suitable for ``ax.imshow(extent=...)`` and for the pre-rendered
        truth path of :func:`source_comparison`."""
        n = self.surface_brightness_cps_per_arcsec2.shape[0]
        half = n * self.source_pixel_scale_arcsec / 2.0
        return (-half, half, -half, half)


def load_vela_source(
    source_dir: str,
    *,
    image_filename: str = "source_image.npy",
    psf_filename: str = "psf.npy",
    metadata_filename: str = "metadata.json",
    transpose_image: bool = False,
) -> VelaSource:
    """Build a :class:`VelaSource` from a Vela source directory.

    The conversion mirrors ``lens_vela_system.ipynb``:

    1. ``source_image.npy`` is interpreted as nJy / pixel.
    2. Divide by ``source_pixel_scale_arcsec ** 2`` to get nJy / arcsec².
    3. Multiply by ``1e-9 / photfnu_Jy`` to get cps / arcsec² (gigalens'
       expected surface-brightness units).
    4. Wrap in :class:`ImageBasedLight` with the source-image pixel scale.

    Parameters
    ----------
    source_dir : str
        Directory containing ``source_image.npy``, ``psf.npy``,
        ``metadata.json``.
    image_filename, psf_filename, metadata_filename : str
        Override the conventional filenames if your layout differs.
    transpose_image : bool, default False
        If True, transpose the source image before wrapping it. Some
        downstream display code in the notebooks renders the source
        transposed; setting this flag once at load time avoids having to
        remember to transpose everywhere else.

    Returns
    -------
    VelaSource
        See :class:`VelaSource`.

    Examples
    --------
    >>> vela = load_vela_source("vela_sources/vela07_cam0_a0.500_f814w")
    >>> phys = PhysicalModel(mass, lens_light, [vela.light])
    >>> sim_cfg = SimulatorConfig(
    ...     delta_pix=vela.instrument_pixel_scale_arcsec,
    ...     num_pix=200, supersample=4, kernel=vela.psf,
    ... )

    And, for truth-aware diagnostics on a fit of this system:

    >>> from gigalens_research.inference_utils import truth_source_from_light_model
    >>> truth_fn = truth_source_from_light_model(vela.light, truth_x[2][0])
    >>> PosteriorReport(post, truth_x=truth_x, truth_source_fn=truth_fn)
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Vela source dir not found: {source_dir}")

    image_path = os.path.join(source_dir, image_filename)
    psf_path = os.path.join(source_dir, psf_filename)
    meta_path = os.path.join(source_dir, metadata_filename)

    for p in (image_path, psf_path, meta_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing Vela source file: {p}")

    source_img_nJy = np.load(image_path)
    psf = np.load(psf_path)
    with open(meta_path) as f:
        meta = json.load(f)

    if transpose_image:
        source_img_nJy = source_img_nJy.T

    try:
        src_scale = float(meta["source_pixel_scale_arcsec"])
        inst_scale = float(meta["instrument_pixel_scale_arcsec"])
        photfnu_Jy = float(meta["photfnu_Jy"])
    except KeyError as e:
        raise KeyError(
            f"metadata.json is missing required key {e!s}. Expected: "
            f"source_pixel_scale_arcsec, instrument_pixel_scale_arcsec, "
            f"photfnu_Jy."
        ) from None

    sb_nJy_per_arcsec2 = source_img_nJy / (src_scale ** 2)
    # cps/arcsec² = (nJy / arcsec²) × (1 Jy / 1e9 nJy) / (Jy per count·s)
    sb_cps_per_arcsec2 = sb_nJy_per_arcsec2 * 1e-9 / photfnu_Jy

    light = ImageBasedLight(sb_cps_per_arcsec2, src_scale)

    return VelaSource(
        light=light,
        psf=np.asarray(psf),
        surface_brightness_cps_per_arcsec2=np.asarray(sb_cps_per_arcsec2),
        source_pixel_scale_arcsec=src_scale,
        instrument_pixel_scale_arcsec=inst_scale,
        metadata=meta,
    )
