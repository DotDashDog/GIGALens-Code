"""Route-B Vela generator: *simulate* fresh lensed Vela systems inside simtests.

Unlike ``vela_existing`` (which only adapts pre-simulated on-disk products), this
module performs the full forward simulation itself, so every scientific choice is
an explicit, recorded campaign parameter rather than something baked into a
notebook:

    draw lens truth  ->  lens the pristine Vela source (ImageBasedLight)
                     ->  PSF-convolve  ->  add noise  ->  save System

Registered here:

- ``"vela_simulated"`` generator.

It reuses the canonical unit conversion (``simulations/vela.py``) and the
framework noise function (``generate._add_noise``) so there is a single source of
truth for each.

------------------------------------------------------------------------------
DELIBERATELY-UNSPECIFIED CHOICES (the generator REFUSES to run until set)
------------------------------------------------------------------------------
These are physics decisions the user is still finalising. Each is required with
**no default**, and generation raises a clear error if it is missing, so a
dataset can never be produced with a silently-assumed value:

- ``scale_factor``       Source redshift, via the VELA scale factor ``a``
                         (z = 1/a - 1). Lensing here is redshift-free, so this
                         acts only through source angular size + brightness.
                         (Decision: "choose VELA scale factors".)
- ``source_flux_scale``  Multiplicative scaling of the source surface brightness
                         (sets lensed-arc S/N). NOT calibrated here. Target
                         guidance from the user: "lensed arcs ~0.5x the
                         brightness of the source" -- explicitly NOT codified
                         yet; this is only a manual knob.
- ``psf_fwhm_arcsec``    FWHM of the Gaussian PSF (arcsec). PSF is applied
                         POST-lensing, so the source image MUST be the pristine
                         (pre-PSF) VELA image -- see the EXTNAME handling below.
- ``background_rms`` and ``exp_time``
                         Noise scheme (lenstronomy-form Poisson + Gaussian).
                         Values pending the user's research-group discussion.

------------------------------------------------------------------------------
PRISTINE-IMAGE PROVENANCE (science-critical)
------------------------------------------------------------------------------
VERIFIED against a real file (vela01-cam12-a0.400_f814w_v3, June 2026): the HDU
layout is

  [0] EXTNAME=IMAGE_PSF               193x193  PSF-convolved "mock observed"
  [1] EXTNAME=IMAGE_PRISTINE          800x800  pristine, WITH dust scattering
  [2] EXTNAME=IMAGE_PRISTINE_NONSCATTER 800x800 pristine, dust attenuation removed

So the *primary* HDU is the PSF-convolved image; the pristine images are
extensions. (The MAST prose claiming IMAGE_PRISTINE is the primary HDU is wrong
for these files -- always trust the EXTNAME, never the index.) The old
``Velalens.py`` read ``hdul[1]`` which here genuinely IS ``IMAGE_PRISTINE``, so
it was correct -- but indexing is fragile, so this generator selects by EXTNAME.

Because we PSF-convolve ourselves after lensing, the source must be a pristine
image. Two pristine variants exist; ``source_variant`` selects which:

  * ``IMAGE_PRISTINE`` (default) -- includes Sunrise dust radiative transfer;
    this is the observationally-realistic galaxy appearance.
  * ``IMAGE_PRISTINE_NONSCATTER`` -- dust attenuation removed (special tests).

The pixel scales / units / photometry are split across HDUs (PIXSIZE, PIXKPC,
IMUNIT, FLUX_NJY, ABMAG live in the pristine HDU; TPIX, PHOTFNU, ABZP, the
distances and cosmology live in the primary/IMAGE_PSF HDU), so extraction reads
the chosen pristine HDU first and falls back to the primary header. We assert
the pristine ``IMUNIT`` is nanoJanskies and that the image sum matches
``FLUX_NJY`` so a units mismatch fails loud rather than silently mis-scaling.

The source metadata is stamped with ``source_builder`` + ``source_extname``; the
generator refuses to reuse a source directory whose marker does not match the
requested variant unless ``allow_unverified_sources=True``.
"""
from __future__ import annotations

import json
import os
import tarfile
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gigalens_research.simtests.registry import register_generator


# ---------------------------------------------------------------------------
# Defaults for the *non*-science-critical / structural choices. The science
# choices that must be set consciously have NO default (see module docstring).
# ---------------------------------------------------------------------------

_HOME = os.path.expanduser("~")
_DEFAULT_SOURCE_ROOT = os.path.join(_HOME, "GIGALens-Code", "data", "vela_sources_pristine")
_DEFAULT_DATADIR = os.path.join(_HOME, "GIGALens-Code", "data", "vela_downloads")

_DEFAULT_VELA_IDS = ["01", "03", "04", "07", "08", "10", "15", "21", "22", "23", "25", "26"]
_DEFAULT_CAM = "12"
_DEFAULT_FILTER = "f814w"
_DEFAULT_VERSION = "v3"

_VELA_BASE_URL = "https://archive.stsci.edu/hlsps/vela"
_HST_FILTERS = {
    "acs": ["f435w", "f606w", "f775w", "f814w", "f850lp"],
    "wfc3": ["f275w", "f336w", "f105w", "f125w", "f140w", "f160w"],
}

# Provenance markers written into source metadata.json so we can prove a source
# image is the pristine (pre-PSF) one and was produced by THIS pipeline.
_PRISTINE_EXTNAME = "IMAGE_PRISTINE"            # default: with dust scattering
_PRISTINE_VARIANTS = ("IMAGE_PRISTINE", "IMAGE_PRISTINE_NONSCATTER")
_SOURCE_BUILDER = "vela_simulated"


# ===========================================================================
# Truth (generation) prior -- EDITABLE BASELINE
# ===========================================================================
#
# This is the verbatim baseline from experiments/vela_sim_systems/
# lens_vela_system.ipynb (cell 4) -- the prior that produced the original Vela
# systems. It is intentionally SEPARATE from the inference prior
# (vela_shapelets.vela_inference_prior); the science lives in the mismatch.
#
# "Baseline then iterate": revise the hyperparameters below directly. Every
# distribution and number is exposed -- nothing is hidden.

def vela_truth_prior_baseline():
    """Return the (lens, lens_light, source) truth prior, as a TFP joint.

    Source is an ImageBasedLight, whose only sampled params are the source-plane
    offset (center_x, center_y); its morphology and absolute brightness come
    from the pristine Vela image (and the separate ``source_flux_scale``).
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    lens_prior = tfd.JointDistributionNamed({
        '0': tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
            gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0),
            e1=tfd.Normal(0.0, 0.1),
            e2=tfd.Normal(0.0, 0.1),
            center_x=tfd.Normal(0.0, 0.03),
            center_y=tfd.Normal(0.0, 0.03),
        )),
        '1': tfd.JointDistributionNamed(dict(
            gamma1=tfd.Normal(0.0, 0.05),
            gamma2=tfd.Normal(0.0, 0.05),
        )),
    })
    # Lens light is rendered with use_lstsq=False at generation, so Ie (absolute
    # amplitude) IS a sampled truth parameter here -- it sets the lens/source
    # blending. Inference solves Ie by lstsq, so it is truth-only (not z-scored).
    lens_light_prior = tfd.JointDistributionNamed({
        '0': tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.15),
            n_sersic=tfd.Uniform(1.0, 6.0),
            e1=tfd.TruncatedNormal(0.0, 0.05, -0.15, 0.15),
            e2=tfd.TruncatedNormal(0.0, 0.05, -0.15, 0.15),
            center_x=tfd.Normal(0.0, 0.01),
            center_y=tfd.Normal(0.0, 0.01),
            Ie=tfd.LogNormal(jnp.log(20.0), 0.3),
        )),
    })
    source_prior = tfd.JointDistributionNamed({
        '0': tfd.JointDistributionNamed(dict(
            center_x=tfd.Normal(0.0, 0.25),
            center_y=tfd.Normal(0.0, 0.25),
        )),
    })
    return tfd.JointDistributionNamed({
        'lens_mass': lens_prior,
        'lens_light': lens_light_prior,
        'source_light': source_prior,
    })


# ===========================================================================
# Pristine-source acquisition (download + EXTNAME-verified extraction)
# ===========================================================================

def _instrument_for_filter(filt: str) -> str:
    for inst, filters in _HST_FILTERS.items():
        if filt.lower() in filters:
            return inst
    raise ValueError(f"Unknown HST filter {filt!r}. Known: {_HST_FILTERS}")


def _normalize_sim(sim: str) -> str:
    """'01' -> 'vela01'; 'vela01' -> 'vela01'."""
    s = str(sim)
    return s if s.startswith("vela") else f"vela{s}"


def _normalize_cam(cam: str) -> str:
    """'12' -> 'cam12'; 'cam12' -> 'cam12'."""
    c = str(cam)
    return c if c.startswith("cam") else f"cam{c}"


def _source_dir_name(sim: str, cam: str, scale_factor: str, filt: str) -> str:
    return f"{_normalize_sim(sim)}_{_normalize_cam(cam)}_{scale_factor}_{filt.lower()}"


def _gaussian_psf(fwhm_arcsec: float, pixel_scale_arcsec: float,
                  size: Optional[int] = None) -> np.ndarray:
    """Normalised 2-D Gaussian PSF kernel at ``pixel_scale_arcsec``.

    (Ported from Velalens.py; the FWHM is a *required generation parameter*,
    NOT read from the FITS header, because the PSF is our modelling choice.)
    """
    sigma_pix = (fwhm_arcsec / pixel_scale_arcsec) / 2.3548200
    if size is None:
        size = int(np.ceil(sigma_pix * 8)) // 2 * 2 + 1  # next odd
        size = max(size, 25)
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_pix ** 2))
    kernel /= kernel.sum()
    return kernel


def _download_fits(sim: str, cam: str, scale_factor: str, filt: str,
                   datadir: str, version: str) -> str:
    """Locate or download the VELA FITS file; return its path."""
    inst = _instrument_for_filter(filt)
    sim_n, cam_n = _normalize_sim(sim), _normalize_cam(cam)
    fname = f"hlsp_vela_hst_{inst}_{sim_n}-{cam_n}-{scale_factor}_{filt.lower()}_{version}_sim.fits"

    for root, _dirs, files in os.walk(datadir):
        if fname in files:
            return os.path.join(root, fname)

    tar_fname = f"hlsp_vela_hst_{inst}_{sim_n}_{filt.lower()}_{version}_sim.tar.gz"
    tar_url = f"{_VELA_BASE_URL}/{sim_n}/{tar_fname}"
    tar_path = os.path.join(datadir, tar_fname)
    os.makedirs(datadir, exist_ok=True)
    print(f"[vela_simulated] downloading {tar_url}")
    urllib.request.urlretrieve(tar_url, tar_path)

    fits_path = None
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if scale_factor in base and cam_n in base:
                tf.extract(member, path=datadir)
                if base == fname:
                    fits_path = os.path.join(datadir, member.name)
    os.remove(tar_path)
    if fits_path is None:
        raise FileNotFoundError(
            f"{fname} not in downloaded archive; check sim/cam/scale_factor/filter "
            f"({sim_n}/{cam_n}/{scale_factor}/{filt})."
        )
    return fits_path


def _extract_pristine(fits_path: str, source_extname: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read the requested pristine HDU (by EXTNAME) + assemble source metadata.

    Selects the source HDU strictly by EXTNAME (never by index), reads pixel
    scales/units/photometry from the pristine HDU first and falls back to the
    primary (IMAGE_PSF) header for the instrument/cosmology keys, asserts the
    pristine unit is nanoJanskies, and sanity-checks the image flux against the
    header's FLUX_NJY.
    """
    from astropy.io import fits

    if source_extname.upper() not in _PRISTINE_VARIANTS:
        raise ValueError(
            f"source_variant {source_extname!r} is not a pristine variant "
            f"{_PRISTINE_VARIANTS}; refusing to use a PSF-convolved image as source."
        )

    with fits.open(fits_path) as hdul:
        extnames = [str(h.header.get("EXTNAME", "")).upper() for h in hdul]
        want = source_extname.upper()
        if want not in extnames:
            raise ValueError(
                f"No HDU with EXTNAME={want!r} in {fits_path}. Found {extnames}. "
                f"Refusing to guess by index."
            )
        idx = extnames.index(want)
        if hdul[idx].data is None:
            raise ValueError(f"{want} HDU in {fits_path} has no data.")
        source_image = np.asarray(hdul[idx].data, dtype=np.float64)
        src_hdr = hdul[idx].header
        primary_hdr = hdul[0].header

        def hget(key, default=None):
            if key in src_hdr:
                return src_hdr[key]
            return primary_hdr.get(key, default)

        # Units guard: the conversion in _load_pristine_source assumes nJy/pixel.
        imunit = str(hget("IMUNIT", "")).strip().lower()
        if imunit not in ("nanojanskies", "nanojansky", "njy"):
            raise ValueError(
                f"Pristine {want} IMUNIT={imunit!r} (expected nanoJanskies) in "
                f"{fits_path}. The nJy->cps conversion would be wrong; aborting."
            )

        flux_njy = hget("FLUX_NJY")          # present on IMAGE_PRISTINE, not NONSCATTER
        if flux_njy is not None:
            img_sum = float(source_image.sum())
            if not np.isclose(img_sum, float(flux_njy), rtol=1e-3):
                raise ValueError(
                    f"Image sum {img_sum:.6g} nJy != header FLUX_NJY "
                    f"{float(flux_njy):.6g} for {want} in {fits_path} -- units mismatch."
                )

        meta = {
            "source_builder": _SOURCE_BUILDER,
            "source_extname": want,
            "fits_path": os.path.basename(fits_path),
            "source_image_shape": list(source_image.shape),
            "source_image_unit": str(hget("IMUNIT")),
            # geometry / redshift
            "source_pixel_scale_arcsec": float(hget("PIXSIZE")),
            "source_pixel_scale_kpc": float(hget("PIXKPC")),
            "instrument_pixel_scale_arcsec": float(hget("TPIX")),
            "redshift": float(hget("REDSHIFT")),
            # photometry actually used in the nJy->cps conversion
            "photfnu_Jy": float(hget("PHOTFNU")),
            "ab_zeropoint": float(hget("ABZP")) if hget("ABZP") is not None else None,
            # pristine source brightness -- directly relevant to source-flux/S-N calibration
            "pristine_flux_nJy": float(flux_njy) if flux_njy is not None else None,
            "pristine_ABMAG": float(hget("ABMAG")) if hget("ABMAG") is not None else None,
            "sb_factor": float(hget("SBFACTOR")) if hget("SBFACTOR") is not None else None,
            # mock-observed (IMAGE_PSF) photometry, for reference
            "mock_AB_mag_apparent": float(primary_hdr.get("MAG")) if "MAG" in primary_hdr else None,
            "mock_AB_mag_absolute": float(primary_hdr.get("ABSMAG")) if "ABSMAG" in primary_hdr else None,
            "approx_psf_fwhm_arcsec": float(primary_hdr.get("APROXPSF")) if "APROXPSF" in primary_hdr else None,
            "psf_file": str(primary_hdr.get("PSFFILE")) if "PSFFILE" in primary_hdr else None,
            # distances / cosmology
            "luminosity_distance_mpc": float(hget("LUMDIST")) if hget("LUMDIST") is not None else None,
            "angular_diameter_distance_mpc": float(hget("ANGDIST")) if hget("ANGDIST") is not None else None,
            "distance_modulus_mag": float(hget("DISTMOD")) if hget("DISTMOD") is not None else None,
            "effective_wavelength_um": float(hget("EFLAMBDA")) if hget("EFLAMBDA") is not None else None,
            "cosmology": {
                "H0": float(hget("H0")) if hget("H0") is not None else None,
                "Omega_m": float(hget("WM")) if hget("WM") is not None else None,
                "Omega_Lambda": float(hget("WV")) if hget("WV") is not None else None,
            },
        }
    return source_image, meta


def ensure_pristine_source(
    sim: str, cam: str, scale_factor: str, filt: str,
    *, source_root: str, datadir: str, version: str,
    source_variant: str = _PRISTINE_EXTNAME,
    allow_unverified_sources: bool = False,
) -> str:
    """Return a source directory holding the PRISTINE image + metadata.

    Layout written: ``<source_root>/<name>/{source_image.npy, metadata.json}``.
    No ``psf.npy`` is written: the PSF is a generation-time choice, not a source
    asset. Downloads + extracts (by EXTNAME) if the verified source is missing.
    The directory name encodes the variant so the two pristine variants never
    collide.
    """
    variant = source_variant.upper()
    tag = "" if variant == _PRISTINE_EXTNAME else "_nonscatter"
    name = _source_dir_name(sim, cam, scale_factor, filt) + tag
    sdir = os.path.join(source_root, name)
    img_path = os.path.join(sdir, "source_image.npy")
    meta_path = os.path.join(sdir, "metadata.json")

    if os.path.isfile(img_path) and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        verified = (meta.get("source_extname") == variant
                    and meta.get("source_builder") == _SOURCE_BUILDER)
        if verified or allow_unverified_sources:
            return sdir
        raise ValueError(
            f"Source dir {sdir!r} exists but is NOT marked as a verified "
            f"{variant} image from {_SOURCE_BUILDER} (found "
            f"source_extname={meta.get('source_extname')!r}). Legacy Vela sources "
            f"carry no such marker and must not be assumed pristine. Delete it to "
            f"re-extract, or pass allow_unverified_sources=True only if you have "
            f"independently confirmed it is the pristine pre-PSF {variant} image."
        )

    fits_path = _download_fits(sim, cam, scale_factor, filt, datadir, version)
    source_image, meta = _extract_pristine(fits_path, variant)
    os.makedirs(sdir, exist_ok=True)
    np.save(img_path, source_image)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[vela_simulated] extracted {variant} source -> {sdir} "
          f"(z={meta['redshift']}, {meta['source_pixel_scale_arcsec']:.6f}\"/px, "
          f"ABMAG={meta.get('pristine_ABMAG')})")
    return sdir


def _load_pristine_source(source_dir: str, transpose_image: bool):
    """Load pristine source, convert nJy/px -> cps/arcsec^2, return components.

    Conversion mirrors gigalens_research.simulations.vela.load_vela_source
    (vela.py L153-155): SB[cps/arcsec^2] = (img_nJy / src_scale^2) * 1e-9 / photfnu.
    """
    with open(os.path.join(source_dir, "metadata.json")) as f:
        meta = json.load(f)
    img_nJy = np.load(os.path.join(source_dir, "source_image.npy"))
    if transpose_image:
        img_nJy = img_nJy.T
    src_scale = float(meta["source_pixel_scale_arcsec"])
    inst_scale = float(meta["instrument_pixel_scale_arcsec"])
    photfnu = float(meta["photfnu_Jy"])
    sb_cps_per_arcsec2 = img_nJy / (src_scale ** 2) * 1e-9 / photfnu
    return np.asarray(sb_cps_per_arcsec2), src_scale, inst_scale, meta


# ===========================================================================
# Generator
# ===========================================================================

def _require(extra: Dict[str, Any], key: str, why: str) -> Any:
    if extra.get(key) is None:
        raise ValueError(
            f"[vela_simulated] required parameter {key!r} is unspecified. {why} "
            f"Set it explicitly in the campaign YAML's dataset block; the generator "
            f"refuses to assume a value."
        )
    return extra[key]


@register_generator("vela_simulated")
def generate_vela_simulated(spec: Any, dataset_dir: str, seed: int) -> None:
    """Simulate fresh lensed Vela systems and save them in System format.

    All ``DatasetSpec.extra`` keys (campaign YAML ``dataset:`` block):

    REQUIRED (no default -- fail loud if missing):
      scale_factor       VELA scale factor a, e.g. "a0.500" (z = 1/a - 1).
      source_flux_scale  Multiplicative source-SB scale (UNCALIBRATED knob;
                         the "arcs ~0.5x source" target is NOT yet codified).
      psf_fwhm_arcsec    Gaussian PSF FWHM in arcsec (applied post-lensing).
      background_rms     Gaussian background sigma (image units).
      exp_time           Effective exposure time (s) for the Poisson term.

    EXPOSED (have a documented structural default; override freely):
      vela_ids           list of sim IDs (default: the 12 standard).
      cam                camera, default "12".
      filter             HST filter, default "f814w".
      version            HLSP version tag, default "v3".
      n_reps             noise/truth realisations per source, default 4.
      num_pix            output image size, default 200.
      supersample        sim supersampling, default 4. NOTE: inference MUST use
                         the same value or the model cannot reproduce the data;
                         it is persisted to each System's meta.json.
      transpose_image    transpose source before lensing, default False.
      source_variant     IMAGE_PRISTINE (default, with dust) or
                         IMAGE_PRISTINE_NONSCATTER (dust removed).
      source_root        where pristine sources live/are written.
      datadir            FITS download cache.
      allow_unverified_sources  reuse legacy source dirs (default False).
      likelihood_precision      default "float64"; conv_precision (None).
    """
    import jax
    import jax.numpy as jnp
    from jax import random

    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.jax.physical_model import PhysicalModel
    from gigalens.simulator import SimulatorConfig

    from gigalens_research.simulations.image_based_light import ImageBasedLight
    from gigalens_research.simtests.generate import _add_noise, _hash_dataset
    from gigalens_research.simtests.system import System, write_manifest

    extra = dict(spec.extra)

    # --- required, fail-loud science choices --------------------------------
    scale_factor = _require(
        extra, "scale_factor",
        "This is the source redshift (z = 1/a - 1) and the whole point of the "
        "revision; there is no sensible default.")
    source_flux_scale = _require(
        extra, "source_flux_scale",
        "Sets lensed-arc S/N. Calibration to 'arcs ~0.5x source brightness' is "
        "intentionally NOT implemented yet -- set a manual float once decided.")
    psf_fwhm_arcsec = float(_require(
        extra, "psf_fwhm_arcsec",
        "Gaussian PSF FWHM (arcsec). PSF is applied post-lensing; pick a width."))
    background_rms = float(_require(
        extra, "background_rms",
        "Gaussian background sigma for the noise model (pending group input)."))
    exp_time = float(_require(
        extra, "exp_time",
        "Exposure time (s) for the Poisson noise term (pending group input)."))
    source_flux_scale = float(source_flux_scale)

    # --- exposed structural choices -----------------------------------------
    vela_ids = list(extra.get("vela_ids", _DEFAULT_VELA_IDS))
    cam = str(extra.get("cam", _DEFAULT_CAM))
    filt = str(extra.get("filter", _DEFAULT_FILTER))
    version = str(extra.get("version", _DEFAULT_VERSION))
    n_reps = int(extra.get("n_reps", 4))
    num_pix = int(extra.get("num_pix", 200))
    supersample = int(extra.get("supersample", 4))
    transpose_image = bool(extra.get("transpose_image", False))
    source_root = os.path.expanduser(str(extra.get("source_root", _DEFAULT_SOURCE_ROOT)))
    datadir = os.path.expanduser(str(extra.get("datadir", _DEFAULT_DATADIR)))
    source_variant = str(extra.get("source_variant", _PRISTINE_EXTNAME)).upper()
    if source_variant not in _PRISTINE_VARIANTS:
        raise ValueError(
            f"[vela_simulated] source_variant={source_variant!r} must be one of "
            f"{_PRISTINE_VARIANTS} (a PSF-convolved image must never be the source)."
        )
    allow_unverified = bool(extra.get("allow_unverified_sources", False))
    likelihood_precision = extra.get("likelihood_precision", "float64")
    conv_precision = extra.get("conv_precision", None)

    prior = vela_truth_prior_baseline()
    base_key = random.PRNGKey(seed)

    system_ids: List[str] = []
    seed_log: Dict[str, int] = {}
    sys_index = 0

    for sim in vela_ids:
        source_dir = ensure_pristine_source(
            sim, cam, scale_factor, filt,
            source_root=source_root, datadir=datadir, version=version,
            source_variant=source_variant,
            allow_unverified_sources=allow_unverified,
        )
        sb_cps, src_scale, delta_pix, src_meta = _load_pristine_source(
            source_dir, transpose_image)

        # Source brightness scaling (manual, uncalibrated knob).
        light = ImageBasedLight(np.asarray(sb_cps) * source_flux_scale, src_scale)

        psf = _gaussian_psf(psf_fwhm_arcsec, delta_pix)

        sim_config = SimulatorConfig(
            delta_pix=delta_pix, num_pix=num_pix, supersample=supersample,
            kernel=psf, likelihood_precision=likelihood_precision,
            conv_precision=conv_precision,
        )
        phys_model = PhysicalModel(
            [epl.EPL(50), shear.Shear()],
            [sersic.SersicEllipse(use_lstsq=False)],
            [light],
        )
        lens_sim = LensSimulator(phys_model, sim_config, bs=1)

        for rep in range(n_reps):
            sys_key = random.fold_in(base_key, sys_index)
            truth_key, noise_key = random.split(sys_key)

            truth = prior.sample(seed=truth_key)
            truth_batched = jax.tree.map(lambda a: jnp.asarray(a)[jnp.newaxis], truth)

            # simulate() may return (1, H, W) or a squeezed (H, W) depending on
            # the gigalens/JAX version; collapse to 2-D and verify before noising.
            noiseless = jnp.squeeze(lens_sim.simulate(truth_batched))
            if noiseless.shape != (num_pix, num_pix):
                raise ValueError(
                    f"simulated image shape {noiseless.shape} != "
                    f"({num_pix}, {num_pix}) for {sim}; refusing to save."
                )
            noisy = _add_noise(noiseless, background_rms, exp_time, noise_key)

            system_id = (f"{_normalize_sim(sim)}_{_normalize_cam(cam)}"
                         f"_{scale_factor}_rep{rep:02d}")
            sys = System(
                system_id=system_id,
                observed_image=np.asarray(noisy),
                truth_x=jax.tree.map(lambda a: np.asarray(a), truth),
                delta_pix=delta_pix,
                num_pix=num_pix,
                supersample=supersample,
                psf=np.asarray(psf),
                noise_kind="forward",
                background_rms=background_rms,
                exp_time=exp_time,
                likelihood_precision=likelihood_precision,
                conv_precision=conv_precision,
                truth_assets={
                    "vela_source_dir": source_dir,
                    "source_redshift": src_meta["redshift"],
                    "source_flux_scale": source_flux_scale,
                    "psf_kind": "gaussian",
                    "psf_fwhm_arcsec": psf_fwhm_arcsec,
                },
            )
            sys.save(dataset_dir)
            system_ids.append(system_id)
            seed_log[system_id] = sys_index
            sys_index += 1

        print(f"[vela_simulated] {_normalize_sim(sim)}: {n_reps} systems "
              f"(z={src_meta['redshift']}, scale_factor={scale_factor}).")

    dataset_hash = _hash_dataset(dataset_dir, system_ids)
    write_manifest(
        dataset_dir,
        generator="vela_simulated",
        seed=seed,
        system_ids=system_ids,
        dataset_hash=dataset_hash,
        extra={
            "scale_factor": scale_factor,
            "source_variant": source_variant,
            "source_flux_scale": source_flux_scale,
            "psf_kind": "gaussian",
            "psf_fwhm_arcsec": psf_fwhm_arcsec,
            "background_rms": background_rms,
            "exp_time": exp_time,
            "num_pix": num_pix,
            "supersample": supersample,
            "transpose_image": transpose_image,
            "cam": cam,
            "filter": filt,
            "version": version,
            "n_reps": n_reps,
            "likelihood_precision": likelihood_precision,
            "truth_prior": "vela_truth_prior_baseline (notebook cell-4 baseline)",
            "source_builder": _SOURCE_BUILDER,
            "source_extname": _PRISTINE_EXTNAME,
            "seed_fold_index": seed_log,
        },
    )
    print(f"[vela_simulated] wrote {len(system_ids)} systems to {dataset_dir}.")
