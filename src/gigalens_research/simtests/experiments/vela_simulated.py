"""Route-B Vela generator (v2): *simulate* fresh lensed Vela systems inside simtests.

Unlike ``vela_existing`` (which only adapts pre-simulated on-disk products), this
module performs the full forward simulation itself on the SCENE API, so every
scientific choice is an explicit, recorded campaign parameter rather than
something baked into a notebook::

    build truth prior (YAML-tunable)
      -> draw lens truth
      -> render lens light only / lensed source only (cutout + 2x canvas)
      -> CALIBRATE source amplitude to a lens-to-source flux ratio
      -> CUTOFF CHECK (flux outside cutout, border surface brightness); redraw on fail
      -> render full image, PSF-convolve (instrument PSF), add instrument noise
      -> save System (+ per-system calibration numbers in truth_assets / manifest)

Registered here:

- ``"vela_simulated"`` generator.

It reuses the canonical unit conversion (``simulations/vela.py``) and the
framework noise function (``generate._add_noise``) so there is a single source of
truth for each.

------------------------------------------------------------------------------
CAMPAIGN YAML ``dataset:`` BLOCK -- FULL REFERENCE
------------------------------------------------------------------------------
Every knob below is read from ``DatasetSpec.extra``. Keys marked REQUIRED have
no default and generation raises if they are missing (a dataset can never be
produced with a silently-assumed physics value). See
``experiments/vela_revised_v2/campaign.yaml`` for a complete, annotated example.

Source selection / geometry
  scale_factor        REQUIRED. VELA scale factor ``a`` (z = 1/a - 1); "a0.400" = z 1.5.
  vela_ids            list of sim IDs (default: the 12 standard).
  cam / filter / version   camera "12", HST filter "f814w", HLSP version "v3".
  n_reps              truth+noise realisations per source (default 1).
  num_pix, supersample     output grid (default 200 px, supersample 4). Inference
                      reads these back from meta.json so it always matches.
  transpose_image     transpose the source before lensing (default False).
  source_variant      IMAGE_PRISTINE (default, with dust) | IMAGE_PRISTINE_NONSCATTER.
  source_crop_radius_arcsec   null (default: no crop) or a radius: source surface
                      brightness beyond this distance from the flux centroid is
                      zeroed (removes companions / faint outskirts). Recorded.
  source_recenter     False (default) | True: shift the source image (integer
                      source pixels) so its flux centroid sits at (0, 0). Recorded.
  source_root / datadir / allow_unverified_sources / likelihood_precision /
  conv_precision      as before (structural).

psf: (REQUIRED block)
  kind: gaussian      fwhm_arcsec (REQUIRED), size_pix (optional, odd).
  kind: stdpsf        STScI empirical effective PSF (Anderson & King "STDPSF"
                      library, 4x oversampled, native detector pixels). Keys:
                      era ("SM4" | "SM3"; REQUIRED), chip_xy ([x, y] detector
                      position used to pick the nearest grid PSF; REQUIRED),
                      size_pix (odd kernel size at delta_pix; REQUIRED),
                      subsamples (fine samples per output pixel when resampling;
                      default 6), url_base (default: STScI HST1PASS library),
                      file (optional local FITS path; skips the download).
                      APPROXIMATION (recorded): the ePSF already includes the
                      native (0.05") pixel response and is NOT deconvolved before
                      resampling to delta_pix, so the kernel is marginally broader
                      than a true delta_pix-pixel PSF. Drizzle broadening of real
                      0.03" mosaics is not modelled either.
  kind: file          path to a .npy kernel already sampled at delta_pix (must be
                      odd, square; renormalised to unit sum, original sum recorded).

noise: (REQUIRED block)
  kind: explicit      background_rms (cps / pixel), exp_time (s).
  kind: instrument    derive background_rms from the instrument:
                      sky_mag_arcsec2, exp_time (total, s), n_reads, read_noise_e,
                      dark_e_per_s (per native pixel), native_pixel_arcsec
                      (all REQUIRED); zeropoint_ab (null -> derived from the source
                      FITS PHOTFNU, which is the instrument zeropoint of the mock).
                      sigma_native = sqrt(sky*t + dark*t + n_reads*RN^2) / t  [cps]
                      background_rms = sigma_native * (delta_pix / native_pixel)
                      (white-noise equivalent per output pixel; the correlated
                      noise of a drizzled mosaic is ignored). The object Poisson
                      term is added separately by ``_add_noise`` (variance I/exp_time,
                      i.e. gain 1 e-/count, as for cps images).
  Legacy: top-level ``background_rms`` + ``exp_time`` == ``noise: {kind: explicit}``.

calibration: (REQUIRED block; exactly one of)
  source_to_lens_flux_ratio: r    set the source amplitude per system so that
                      (lensed source flux) / (lens light flux) == r, both measured
                      inside the cutout (``measure_in: cutout``, default) or on the
                      cutoff canvas (``measure_in: canvas``).
  source_flux_scale: s            fixed multiplicative source amplitude (old knob).

cutoff: (REQUIRED block, or ``cutoff: null`` to disable explicitly)
  canvas_factor       render the lensed source on canvas_factor x num_pix to
                      measure flux outside the cutout (default 2).
  canvas_supersample  supersample for the canvas render (default: same as main).
  max_flux_outside    reject the draw if (source flux outside cutout)/(total) exceeds this.
  max_border_sb_sigma reject if the max noiseless lensed-source surface brightness
                      in the outer ``border_width_pix`` rows/cols of the cutout
                      exceeds this many background sigmas (after calibration).
  border_width_pix    width of the border band (default 3).
  max_redraws         give up (raise) after this many rejected draws per system.
  All accepted draws record their metrics and the number of rejections.

truth_prior: (optional) overrides merged onto ``TRUTH_PRIOR_BASELINE`` (the cell-4
  prior of the original notebook). Structure mirrors the 3-group truth::

      truth_prior:
        lens_mass:
          "0": {theta_E: {dist: LogNormal, median: 1.1, sigma: 0.2}}
        lens_light:
          "0": {Ie: {dist: LogNormal, median: 20.0, sigma: 0.3}}

  Distributions: LogNormal(median, sigma), Normal(loc, scale),
  TruncatedNormal(loc, scale, low, high), Uniform(low, high), Fixed(value).
  Unknown groups / components / params / dist names raise (typo guard). The fully
  resolved spec is written to the manifest.

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
for these files -- always trust the EXTNAME, never the index.) Because we
PSF-convolve ourselves after lensing, the source must be a pristine image.

The pixel scales / units / photometry are split across HDUs (PIXSIZE, PIXKPC,
IMUNIT, FLUX_NJY, ABMAG live in the pristine HDU; TPIX, PHOTFNU, ABZP, the
distances and cosmology live in the primary/IMAGE_PSF HDU), so extraction reads
the chosen pristine HDU first and falls back to the primary header. We assert
the pristine ``IMUNIT`` is nanoJanskies and that the image sum matches
``FLUX_NJY`` (when present) so a units mismatch fails loud.

The source metadata is stamped with ``source_builder`` + ``source_extname``; the
generator refuses to reuse a source directory whose marker does not match the
requested variant unless ``allow_unverified_sources=True``.
"""
from __future__ import annotations

import copy
import json
import os
import tarfile
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gigalens_research.simtests.registry import register_generator


# ---------------------------------------------------------------------------
# Structural defaults (non-science). Science choices have NO default.
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

# STScI empirical "standard PSF" library (Anderson & King effective PSFs), used by
# hst1pass. 4x oversampled, native detector pixels, distorted (FLT) frame.
_STDPSF_URL_BASE = "https://www.stsci.edu/~jayander/HST1PASS/LIB/PSFs/STDPSFs"
_STDPSF_DETECTOR = {"acs": "ACSWFC", "wfc3_uvis": "WFC3UV", "wfc3_ir": "WFC3IR"}
_STDPSF_NATIVE_PIXEL_ARCSEC = {"ACSWFC": 0.05, "WFC3UV": 0.04, "WFC3IR": 0.13}
_STDPSF_OVERSAMPLING = 4

_PRISTINE_EXTNAME = "IMAGE_PRISTINE"
_PRISTINE_VARIANTS = ("IMAGE_PRISTINE", "IMAGE_PRISTINE_NONSCATTER")
_SOURCE_BUILDER = "vela_simulated"

_GROUPS = ("lens_mass", "lens_light", "source_light")


# ===========================================================================
# Truth (generation) prior -- YAML-tunable spec + factory
# ===========================================================================
#
# Baseline = verbatim cell-4 prior of experiments/vela_sim_systems/
# lens_vela_system.ipynb (the prior that produced the original Vela systems),
# plus the source amplitude ``amp`` (Fixed 1.0 here; overwritten by the
# calibration step). It is intentionally SEPARATE from the inference prior
# (vela_shapelets._vela_scene_lens_priors); the science lives in the mismatch.

TRUTH_PRIOR_BASELINE: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
    "lens_mass": {
        "0": {  # EPL
            "theta_E": {"dist": "LogNormal", "median": 1.25, "sigma": 0.25},
            "gamma": {"dist": "TruncatedNormal", "loc": 2.0, "scale": 0.25, "low": 1.0, "high": 3.0},
            "e1": {"dist": "Normal", "loc": 0.0, "scale": 0.1},
            "e2": {"dist": "Normal", "loc": 0.0, "scale": 0.1},
            "center_x": {"dist": "Normal", "loc": 0.0, "scale": 0.03},
            "center_y": {"dist": "Normal", "loc": 0.0, "scale": 0.03},
        },
        "1": {  # external shear
            "gamma1": {"dist": "Normal", "loc": 0.0, "scale": 0.05},
            "gamma2": {"dist": "Normal", "loc": 0.0, "scale": 0.05},
        },
    },
    "lens_light": {
        "0": {  # SersicEllipse, rendered with use_lstsq=False so Ie is a truth param
            "R_sersic": {"dist": "LogNormal", "median": 1.6, "sigma": 0.15},
            "n_sersic": {"dist": "Uniform", "low": 1.0, "high": 6.0},
            "e1": {"dist": "TruncatedNormal", "loc": 0.0, "scale": 0.05, "low": -0.15, "high": 0.15},
            "e2": {"dist": "TruncatedNormal", "loc": 0.0, "scale": 0.05, "low": -0.15, "high": 0.15},
            "center_x": {"dist": "Normal", "loc": 0.0, "scale": 0.01},
            "center_y": {"dist": "Normal", "loc": 0.0, "scale": 0.01},
            "Ie": {"dist": "LogNormal", "median": 20.0, "sigma": 0.3},
        },
    },
    "source_light": {
        "0": {  # ImageBasedLight: source-plane offset + amplitude
            "center_x": {"dist": "Normal", "loc": 0.0, "scale": 0.25},
            "center_y": {"dist": "Normal", "loc": 0.0, "scale": 0.25},
            "amp": {"dist": "Fixed", "value": 1.0},
        },
    },
}

_DIST_KEYS = {
    "LogNormal": {"median", "sigma"},
    "Normal": {"loc", "scale"},
    "TruncatedNormal": {"loc", "scale", "low", "high"},
    "Uniform": {"low", "high"},
    "Fixed": {"value"},
}


def resolve_truth_prior_spec(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep-merge ``overrides`` onto :data:`TRUTH_PRIOR_BASELINE` with a typo guard.

    Only existing group / component / parameter keys may be overridden (a new
    parameter cannot be introduced through YAML: the profiles are fixed). Each
    override leaf replaces the whole distribution spec for that parameter.
    """
    spec = copy.deepcopy(TRUTH_PRIOR_BASELINE)
    if not overrides:
        return spec
    if not isinstance(overrides, dict):
        raise ValueError("[vela_simulated] truth_prior must be a mapping of groups.")
    for group, comps in overrides.items():
        if group not in spec:
            raise ValueError(f"[vela_simulated] truth_prior: unknown group {group!r}; "
                             f"expected one of {list(spec)}.")
        if not isinstance(comps, dict):
            raise ValueError(f"[vela_simulated] truth_prior.{group} must be a mapping.")
        for comp, params in comps.items():
            comp = str(comp)
            if comp not in spec[group]:
                raise ValueError(f"[vela_simulated] truth_prior.{group}: unknown component "
                                 f"{comp!r}; expected one of {list(spec[group])}.")
            if not isinstance(params, dict):
                raise ValueError(f"[vela_simulated] truth_prior.{group}.{comp} must be a mapping.")
            for pname, dspec in params.items():
                if pname not in spec[group][comp]:
                    raise ValueError(f"[vela_simulated] truth_prior.{group}.{comp}: unknown "
                                     f"parameter {pname!r}; expected one of "
                                     f"{list(spec[group][comp])}.")
                _validate_dist_spec(dspec, f"truth_prior.{group}.{comp}.{pname}")
                spec[group][comp][pname] = dict(dspec)
    return spec


def _validate_dist_spec(dspec: Any, where: str) -> None:
    if not isinstance(dspec, dict) or "dist" not in dspec:
        raise ValueError(f"[vela_simulated] {where}: expected {{dist: ..., ...}}, got {dspec!r}.")
    name = dspec["dist"]
    if name not in _DIST_KEYS:
        raise ValueError(f"[vela_simulated] {where}: unknown dist {name!r}; "
                         f"supported: {sorted(_DIST_KEYS)}.")
    keys = set(dspec) - {"dist"}
    if keys != _DIST_KEYS[name]:
        raise ValueError(f"[vela_simulated] {where}: {name} needs exactly keys "
                         f"{sorted(_DIST_KEYS[name])}, got {sorted(keys)}.")


def _make_dist(dspec: Dict[str, Any], where: str):
    """Build one TFP distribution from a validated spec (float64 leaves)."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    _validate_dist_spec(dspec, where)
    f = lambda k: jnp.asarray(float(dspec[k]), dtype=jnp.float64)  # noqa: E731
    name = dspec["dist"]
    if name == "LogNormal":
        if float(dspec["median"]) <= 0:
            raise ValueError(f"[vela_simulated] {where}: LogNormal median must be > 0.")
        return tfd.LogNormal(jnp.log(f("median")), f("sigma"))
    if name == "Normal":
        return tfd.Normal(f("loc"), f("scale"))
    if name == "TruncatedNormal":
        return tfd.TruncatedNormal(f("loc"), f("scale"), f("low"), f("high"))
    if name == "Uniform":
        return tfd.Uniform(f("low"), f("high"))
    if name == "Fixed":
        return tfd.Deterministic(f("value"))
    raise AssertionError(name)


def build_truth_prior(spec: Dict[str, Any]):
    """Return the 3-group TFP joint ``{lens_mass:{'0':..}, lens_light:.., source_light:..}``."""
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    groups = {}
    for group in _GROUPS:
        comps = {}
        for comp, params in spec[group].items():
            comps[comp] = tfd.JointDistributionNamed({
                p: _make_dist(d, f"truth_prior.{group}.{comp}.{p}") for p, d in params.items()
            })
        groups[group] = tfd.JointDistributionNamed(comps)
    return tfd.JointDistributionNamed(groups)


def vela_truth_prior_baseline():
    """Baseline truth prior as a TFP joint (kept for callers of the v1 name)."""
    return build_truth_prior(resolve_truth_prior_spec(None))


def _sample_to_legacy(sample: Dict[str, Any]) -> List[List[Dict[str, float]]]:
    """Dict sample -> legacy 3-list ``[[mass...], [lens_light...], [source...]]`` of floats."""
    out = []
    for group in _GROUPS:
        comps = sample[group]
        out.append([{p: float(np.asarray(v)) for p, v in comps[k].items()}
                    for k in sorted(comps, key=int)])
    return out


# ===========================================================================
# Pristine-source acquisition (download + EXTNAME-verified extraction)
# ===========================================================================

def _instrument_for_filter(filt: str) -> str:
    for inst, filters in _HST_FILTERS.items():
        if filt.lower() in filters:
            return inst
    raise ValueError(f"Unknown HST filter {filt!r}. Known: {_HST_FILTERS}")


def _normalize_sim(sim: str) -> str:
    s = str(sim)
    return s if s.startswith("vela") else f"vela{s}"


def _normalize_cam(cam: str) -> str:
    c = str(cam)
    return c if c.startswith("cam") else f"cam{c}"


def _source_dir_name(sim: str, cam: str, scale_factor: str, filt: str) -> str:
    return f"{_normalize_sim(sim)}_{_normalize_cam(cam)}_{scale_factor}_{filt.lower()}"


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
    """Read the requested pristine HDU (by EXTNAME) + assemble source metadata."""
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

        imunit = str(hget("IMUNIT", "")).strip().lower()
        if imunit not in ("nanojanskies", "nanojansky", "njy"):
            raise ValueError(
                f"Pristine {want} IMUNIT={imunit!r} (expected nanoJanskies) in "
                f"{fits_path}. The nJy->cps conversion would be wrong; aborting."
            )

        flux_njy = hget("FLUX_NJY")  # present on IMAGE_PRISTINE for most sims, not NONSCATTER
        if flux_njy is not None:
            img_sum = float(source_image.sum())
            if not np.isclose(img_sum, float(flux_njy), rtol=1e-3):
                raise ValueError(
                    f"Image sum {img_sum:.6g} nJy != header FLUX_NJY "
                    f"{float(flux_njy):.6g} for {want} in {fits_path} -- units mismatch."
                )

        def fopt(key):
            v = hget(key)
            return float(v) if v is not None else None

        meta = {
            "source_builder": _SOURCE_BUILDER,
            "source_extname": want,
            "fits_path": os.path.basename(fits_path),
            "source_image_shape": list(source_image.shape),
            "source_image_unit": str(hget("IMUNIT")),
            "source_pixel_scale_arcsec": float(hget("PIXSIZE")),
            "source_pixel_scale_kpc": float(hget("PIXKPC")),
            "instrument_pixel_scale_arcsec": float(hget("TPIX")),
            "redshift": float(hget("REDSHIFT")),
            "photfnu_Jy": float(hget("PHOTFNU")),
            "ab_zeropoint": fopt("ABZP"),
            "pristine_flux_nJy": fopt("FLUX_NJY"),
            "pristine_ABMAG": fopt("ABMAG"),
            "image_sum_nJy": float(source_image.sum()),
            "sb_factor": fopt("SBFACTOR"),
            "mock_AB_mag_apparent": float(primary_hdr.get("MAG")) if "MAG" in primary_hdr else None,
            "mock_AB_mag_absolute": float(primary_hdr.get("ABSMAG")) if "ABSMAG" in primary_hdr else None,
            "approx_psf_fwhm_arcsec": float(primary_hdr.get("APROXPSF")) if "APROXPSF" in primary_hdr else None,
            "psf_file": str(primary_hdr.get("PSFFILE")) if "PSFFILE" in primary_hdr else None,
            "luminosity_distance_mpc": fopt("LUMDIST"),
            "angular_diameter_distance_mpc": fopt("ANGDIST"),
            "distance_modulus_mag": fopt("DISTMOD"),
            "effective_wavelength_um": fopt("EFLAMBDA"),
            "cosmology": {"H0": fopt("H0"), "Omega_m": fopt("WM"), "Omega_Lambda": fopt("WV")},
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

    Conversion mirrors gigalens_research.simulations.vela.load_vela_source:
    SB[cps/arcsec^2] = (img_nJy / src_scale^2) * 1e-9 / photfnu.
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
    return np.asarray(sb_cps_per_arcsec2, dtype=np.float64), src_scale, inst_scale, meta


def _flux_centroid(img: np.ndarray) -> Tuple[float, float]:
    """(row, col) flux-weighted centroid in pixel units."""
    tot = float(img.sum())
    if tot <= 0:
        raise ValueError("[vela_simulated] source image has non-positive total flux.")
    yy, xx = np.indices(img.shape)
    return float((yy * img).sum() / tot), float((xx * img).sum() / tot)


def preprocess_source(sb: np.ndarray, src_scale: float, *,
                      crop_radius_arcsec: Optional[float],
                      recenter: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Optional crop (zero beyond a radius from the flux centroid) and recentering.

    Both are recorded so the truth source is reproducible from the manifest.
    """
    img = np.array(sb, dtype=np.float64, copy=True)
    info: Dict[str, Any] = {"crop_radius_arcsec": crop_radius_arcsec, "recenter": bool(recenter)}
    total0 = float(img.sum())
    cy, cx = _flux_centroid(img)
    if crop_radius_arcsec is not None:
        r_pix = float(crop_radius_arcsec) / src_scale
        yy, xx = np.indices(img.shape)
        mask = np.hypot(yy - cy, xx - cx) > r_pix
        img[mask] = 0.0
        info["crop_flux_removed_frac"] = 1.0 - float(img.sum()) / total0
        cy, cx = _flux_centroid(img)
    if recenter:
        n = img.shape[0]
        c = (n - 1) / 2.0
        dy, dx = int(round(c - cy)), int(round(c - cx))
        shifted = np.zeros_like(img)
        ys, yd = (slice(0, n - dy), slice(dy, n)) if dy >= 0 else (slice(-dy, n), slice(0, n + dy))
        xs, xd = (slice(0, n - dx), slice(dx, n)) if dx >= 0 else (slice(-dx, n), slice(0, n + dx))
        shifted[yd, xd] = img[ys, xs]
        info["recenter_shift_arcsec"] = [dx * src_scale, dy * src_scale]
        info["recenter_flux_lost_frac"] = 1.0 - float(shifted.sum()) / max(float(img.sum()), 1e-300)
        img = shifted
    cy2, cx2 = _flux_centroid(img)
    info["centroid_offset_arcsec_after"] = [(cx2 - (img.shape[1] - 1) / 2.0) * src_scale,
                                            (cy2 - (img.shape[0] - 1) / 2.0) * src_scale]
    return img, info


# ===========================================================================
# PSF
# ===========================================================================

def _gaussian_psf(fwhm_arcsec: float, pixel_scale_arcsec: float,
                  size: Optional[int] = None) -> np.ndarray:
    """Normalised 2-D Gaussian PSF kernel sampled at ``pixel_scale_arcsec``."""
    sigma_pix = (fwhm_arcsec / pixel_scale_arcsec) / 2.3548200
    if size is None:
        size = int(np.ceil(sigma_pix * 8)) // 2 * 2 + 1  # next odd
        size = max(size, 25)
    if size % 2 == 0:
        raise ValueError(f"[vela_simulated] PSF size_pix must be odd, got {size}.")
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_pix ** 2))
    kernel /= kernel.sum()
    return kernel


def measure_fwhm(img: np.ndarray, spacing: float) -> float:
    """FWHM (in arcsec, given ``spacing`` per sample) from the azimuthally-averaged
    radial profile about the brightest sample; the half-max radius is linearly
    interpolated on the first crossing between successive unique radii.
    """
    img = np.asarray(img, dtype=np.float64)
    py, px = np.unravel_index(int(np.argmax(img)), img.shape)
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - py, xx - px).ravel()
    ru, inv = np.unique(np.round(r, 6), return_inverse=True)
    prof = np.bincount(inv, weights=img.ravel()) / np.bincount(inv)
    half = prof[0] / 2.0
    below = np.nonzero(prof < half)[0]
    if len(below) == 0:
        raise ValueError("[vela_simulated] PSF never falls below half max; kernel too small.")
    i = int(below[0])
    r_half = np.interp(half, [prof[i], prof[i - 1]], [ru[i], ru[i - 1]])
    return float(2.0 * r_half * spacing)


def resample_oversampled_psf(epsf: np.ndarray, sample_spacing_arcsec: float,
                             delta_pix: float, size_pix: int, subsamples: int) -> np.ndarray:
    """Resample an oversampled PSF (regular grid, spacing ``sample_spacing_arcsec``,
    centred on its central sample) onto a ``size_pix`` x ``size_pix`` kernel at
    ``delta_pix`` by averaging ``subsamples^2`` bilinear samples per output pixel.
    Flux outside the oversampled footprint is treated as zero. Unit-normalised.
    """
    from scipy.interpolate import RegularGridInterpolator

    epsf = np.asarray(epsf, dtype=np.float64)
    if epsf.ndim != 2 or epsf.shape[0] != epsf.shape[1] or epsf.shape[0] % 2 == 0:
        raise ValueError(f"[vela_simulated] oversampled PSF must be odd square, got {epsf.shape}.")
    if size_pix % 2 == 0:
        raise ValueError(f"[vela_simulated] PSF size_pix must be odd, got {size_pix}.")
    n = epsf.shape[0]
    coords = (np.arange(n) - (n - 1) / 2.0) * sample_spacing_arcsec
    interp = RegularGridInterpolator((coords, coords), epsf, method="linear",
                                     bounds_error=False, fill_value=0.0)
    half = size_pix // 2
    centers = np.arange(-half, half + 1) * delta_pix
    offs = (np.arange(subsamples) + 0.5) / subsamples - 0.5
    fine = (centers[:, None] + offs[None, :] * delta_pix).ravel()
    yy, xx = np.meshgrid(fine, fine, indexing="ij")
    vals = interp(np.stack([yy.ravel(), xx.ravel()], axis=-1)).reshape(yy.shape)
    kernel = vals.reshape(size_pix, subsamples, size_pix, subsamples).mean(axis=(1, 3))
    s = kernel.sum()
    if s <= 0:
        raise ValueError("[vela_simulated] resampled PSF has non-positive sum.")
    return kernel / s


def _stdpsf_detector(filt: str) -> str:
    inst = _instrument_for_filter(filt)
    if inst == "acs":
        return _STDPSF_DETECTOR["acs"]
    # WFC3: UVIS vs IR by filter wavelength
    return _STDPSF_DETECTOR["wfc3_ir"] if filt.lower() in ("f105w", "f125w", "f140w", "f160w") \
        else _STDPSF_DETECTOR["wfc3_uvis"]


def _ensure_stdpsf_file(filt: str, era: str, datadir: str, url_base: str,
                        file_override: Optional[str]) -> Tuple[str, str]:
    det = _stdpsf_detector(filt)
    if file_override:
        path = os.path.expanduser(str(file_override))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[vela_simulated] psf.file {path!r} not found.")
        return path, det
    fname = f"STDPSF_{det}_{filt.upper()}_{era.upper()}.fits"
    pdir = os.path.join(datadir, "psf")
    path = os.path.join(pdir, fname)
    if not os.path.isfile(path):
        os.makedirs(pdir, exist_ok=True)
        url = f"{url_base}/{det}/{fname}"
        print(f"[vela_simulated] downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path, det


def build_psf(psf_spec: Any, delta_pix: float, filt: str, datadir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the detector-sampled kernel (at ``delta_pix``) from the ``psf:`` block."""
    if not isinstance(psf_spec, dict) or "kind" not in psf_spec:
        raise ValueError("[vela_simulated] dataset.psf must be a mapping with a 'kind' "
                         "(gaussian | stdpsf | file); no default PSF.")
    kind = str(psf_spec["kind"]).lower()
    meta: Dict[str, Any] = {"kind": kind, "delta_pix": float(delta_pix)}

    if kind == "gaussian":
        fwhm = float(_require(psf_spec, "fwhm_arcsec", "Gaussian PSF FWHM (arcsec).", "psf"))
        size = psf_spec.get("size_pix")  # physics-default-ok: kernel extent only; None -> >= 8 sigma, min 25 px
        kernel = _gaussian_psf(fwhm, delta_pix, None if size is None else int(size))
        meta.update({"fwhm_arcsec_requested": fwhm})

    elif kind == "stdpsf":
        era = str(_require(psf_spec, "era", "STDPSF era: 'SM4' (post-2009) or 'SM3'.", "psf"))
        chip_xy = _require(psf_spec, "chip_xy", "detector [x, y] used to pick the nearest grid PSF.", "psf")
        size_pix = int(_require(psf_spec, "size_pix", "odd kernel size at delta_pix.", "psf"))
        subsamples = int(psf_spec.get("subsamples", 6))  # physics-default-ok: resampling quadrature only
        url_base = str(psf_spec.get("url_base", _STDPSF_URL_BASE))  # physics-default-ok: download location only
        path, det = _ensure_stdpsf_file(filt, era, datadir, url_base, psf_spec.get("file"))  # physics-default-ok: optional local override of the same product
        try:
            from photutils.psf import STDPSFGrid
        except ImportError as exc:  # pragma: no cover
            raise ImportError("[vela_simulated] psf.kind=stdpsf needs photutils (STDPSFGrid).") from exc
        grid = STDPSFGrid(path)
        xy = np.asarray(grid.grid_xypos, dtype=np.float64)
        i = int(np.argmin(((xy - np.asarray(chip_xy, dtype=np.float64)) ** 2).sum(axis=1)))
        epsf = np.asarray(grid.data[i], dtype=np.float64)
        os_ = np.asarray(grid.oversampling).ravel()
        if not np.all(os_ == _STDPSF_OVERSAMPLING):
            raise ValueError(f"[vela_simulated] unexpected STDPSF oversampling {os_}.")
        native = _STDPSF_NATIVE_PIXEL_ARCSEC[det]
        spacing = native / _STDPSF_OVERSAMPLING
        kernel = resample_oversampled_psf(epsf, spacing, delta_pix, size_pix, subsamples)
        fine_fwhm = measure_fwhm(epsf, spacing)
        meta.update({
            "file": os.path.basename(path), "detector": det, "era": era.upper(),
            "grid_xy_used": [float(xy[i, 0]), float(xy[i, 1])], "chip_xy_requested": list(chip_xy),
            "native_pixel_arcsec": native, "oversampling": _STDPSF_OVERSAMPLING,
            "subsamples": subsamples, "size_pix": size_pix,
            "fwhm_arcsec_native_epsf": fine_fwhm,
            "approximation": ("ePSF includes the native pixel response and is not "
                              "deconvolved before resampling; drizzle broadening not modelled"),
        })

    elif kind == "file":
        path = os.path.expanduser(str(_require(psf_spec, "path", "path to a .npy kernel at delta_pix.", "psf")))
        kernel = np.load(path).astype(np.float64)
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 == 0:
            raise ValueError(f"[vela_simulated] psf.file kernel must be odd square, got {kernel.shape}.")
        meta.update({"file": path, "original_sum": float(kernel.sum())})
        kernel = kernel / kernel.sum()
    else:
        raise ValueError(f"[vela_simulated] psf.kind={kind!r} must be gaussian | stdpsf | file.")

    meta["size_pix"] = int(kernel.shape[0])
    meta["fwhm_arcsec_measured"] = measure_fwhm(kernel, delta_pix)
    meta["ee_within_0p25_arcsec"] = float(_encircled(kernel, delta_pix, 0.25))
    return kernel.astype(np.float64), meta


def _encircled(kernel: np.ndarray, delta_pix: float, radius_arcsec: float) -> float:
    n = kernel.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.indices(kernel.shape)
    return float(kernel[np.hypot(yy - c, xx - c) * delta_pix <= radius_arcsec].sum() / kernel.sum())


# ===========================================================================
# Noise
# ===========================================================================

def derive_background_rms(*, sky_mag_arcsec2: float, zeropoint_ab: float, exp_time: float,
                          n_reads: int, read_noise_e: float, dark_e_per_s: float,
                          native_pixel_arcsec: float, delta_pix: float) -> Dict[str, float]:
    """Instrument noise -> Gaussian background sigma per output pixel (cps).

    sigma_native = sqrt(sky*t + dark*t + n_reads*RN^2) / t   [cps per native pixel]
    background_rms = sigma_native * (delta_pix / native_pixel)  (white-noise equivalent)
    """
    if exp_time <= 0 or n_reads < 1:
        raise ValueError("[vela_simulated] noise: exp_time must be > 0 and n_reads >= 1.")
    sky_cps_native = 10.0 ** (-0.4 * (sky_mag_arcsec2 - zeropoint_ab)) * native_pixel_arcsec ** 2
    var_e2 = sky_cps_native * exp_time + dark_e_per_s * exp_time + n_reads * read_noise_e ** 2
    sigma_native = float(np.sqrt(var_e2) / exp_time)
    return {
        "sky_cps_per_native_pixel": float(sky_cps_native),
        "sigma_cps_per_native_pixel": sigma_native,
        "background_rms": sigma_native * (delta_pix / native_pixel_arcsec),
    }


def resolve_noise(extra: Dict[str, Any], delta_pix: float, photfnu_Jy: float) -> Tuple[float, float, Dict[str, Any]]:
    """Return (background_rms, exp_time, noise_meta) from ``noise:`` (or legacy keys)."""
    spec = extra.get("noise")  # physics-default-ok: legacy top-level keys handled below, else raise
    if spec is None:
        if extra.get("background_rms") is None or extra.get("exp_time") is None:  # physics-default-ok: raise path
            raise ValueError("[vela_simulated] dataset.noise block is required "
                             "({kind: explicit, background_rms, exp_time} or {kind: instrument, ...}).")
        spec = {"kind": "explicit", "background_rms": extra["background_rms"], "exp_time": extra["exp_time"]}
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ValueError("[vela_simulated] dataset.noise must be a mapping with 'kind'.")
    kind = str(spec["kind"]).lower()
    if kind == "explicit":
        bkg = float(_require(spec, "background_rms", "Gaussian background sigma (cps/pixel).", "noise"))
        t = float(_require(spec, "exp_time", "exposure time (s) for the Poisson term.", "noise"))
        return bkg, t, {"kind": kind, "background_rms": bkg, "exp_time": t}
    if kind == "instrument":
        keys = {k: _require(spec, k, f"instrument noise needs {k}.", "noise") for k in
                ("sky_mag_arcsec2", "exp_time", "n_reads", "read_noise_e", "dark_e_per_s", "native_pixel_arcsec")}
        zp = spec.get("zeropoint_ab")  # physics-default-ok: None -> derived from the mock's own PHOTFNU (recorded)
        zp_source = "config"
        if zp is None:
            zp = -2.5 * np.log10(photfnu_Jy / 3631.0)
            zp_source = "derived from source FITS PHOTFNU"
        d = derive_background_rms(
            sky_mag_arcsec2=float(keys["sky_mag_arcsec2"]), zeropoint_ab=float(zp),
            exp_time=float(keys["exp_time"]), n_reads=int(keys["n_reads"]),
            read_noise_e=float(keys["read_noise_e"]), dark_e_per_s=float(keys["dark_e_per_s"]),
            native_pixel_arcsec=float(keys["native_pixel_arcsec"]), delta_pix=delta_pix)
        meta = {"kind": kind, **{k: float(v) for k, v in keys.items()}, "n_reads": int(keys["n_reads"]),
                "zeropoint_ab": float(zp), "zeropoint_source": zp_source, **d,
                "note": "white-noise equivalent per output pixel; drizzle correlations ignored; "
                        "object Poisson term added separately with gain 1 e-/count"}
        return float(d["background_rms"]), float(keys["exp_time"]), meta
    raise ValueError(f"[vela_simulated] noise.kind={kind!r} must be explicit | instrument.")


# ===========================================================================
# Calibration + cutoff helpers (pure numpy; unit-tested)
# ===========================================================================

def flux_outside_cutout(canvas_img: np.ndarray, num_pix: int) -> float:
    """Fraction of the canvas image's flux outside the centred num_pix x num_pix cutout."""
    n = canvas_img.shape[0]
    if n < num_pix or (n - num_pix) % 2:
        raise ValueError(f"[vela_simulated] canvas {n} must be >= num_pix {num_pix} with even margin.")
    c = (n - num_pix) // 2
    total = float(canvas_img.sum())
    if total <= 0:
        raise ValueError("[vela_simulated] lensed source has non-positive flux on the canvas.")
    return 1.0 - float(canvas_img[c:c + num_pix, c:c + num_pix].sum()) / total


def border_max(img: np.ndarray, width: int) -> float:
    """Max pixel value in the outer ``width`` rows/columns."""
    w = int(width)
    return float(max(img[:w].max(), img[-w:].max(), img[:, :w].max(), img[:, -w:].max()))


def calibrate_amp(*, ratio: float, lens_flux: float, source_flux_unit_amp: float) -> float:
    """Source amplitude giving (source flux)/(lens flux) == ratio (fluxes at amp=1)."""
    if lens_flux <= 0 or source_flux_unit_amp <= 0:
        raise ValueError(f"[vela_simulated] cannot calibrate: lens_flux={lens_flux}, "
                         f"source_flux(amp=1)={source_flux_unit_amp}.")
    return float(ratio) * lens_flux / source_flux_unit_amp


def _resolve_calibration(extra: Dict[str, Any]) -> Dict[str, Any]:
    spec = extra.get("calibration")  # physics-default-ok: legacy source_flux_scale handled below, else raise
    if spec is None and extra.get("source_flux_scale") is not None:  # physics-default-ok: raise path
        spec = {"source_flux_scale": extra["source_flux_scale"]}
    if not isinstance(spec, dict):
        raise ValueError("[vela_simulated] dataset.calibration block is required: "
                         "{source_to_lens_flux_ratio: r} or {source_flux_scale: s}.")
    has_ratio = spec.get("source_to_lens_flux_ratio") is not None  # physics-default-ok: presence test
    has_scale = spec.get("source_flux_scale") is not None  # physics-default-ok: presence test
    if has_ratio == has_scale:
        raise ValueError("[vela_simulated] calibration: set exactly one of "
                         "source_to_lens_flux_ratio / source_flux_scale.")
    measure_in = str(spec.get("measure_in", "cutout")).lower()  # physics-default-ok: documented default (cutout = what the fit sees)
    if measure_in not in ("cutout", "canvas"):
        raise ValueError("[vela_simulated] calibration.measure_in must be cutout | canvas.")
    out = {"measure_in": measure_in}
    if has_ratio:
        out["mode"] = "ratio"
        out["source_to_lens_flux_ratio"] = float(spec["source_to_lens_flux_ratio"])
    else:
        out["mode"] = "scale"
        out["source_flux_scale"] = float(spec["source_flux_scale"])
    return out


def _resolve_cutoff(extra: Dict[str, Any], supersample: int) -> Optional[Dict[str, Any]]:
    if "cutoff" not in extra:
        raise ValueError("[vela_simulated] dataset.cutoff block is required "
                         "(or `cutoff: null` to disable the cut-off check explicitly).")
    spec = extra["cutoff"]
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("[vela_simulated] dataset.cutoff must be a mapping or null.")
    out = {
        "canvas_factor": int(spec.get("canvas_factor", 2)),  # physics-default-ok: measurement canvas only
        "canvas_supersample": int(spec.get("canvas_supersample", supersample)),  # physics-default-ok: numerics of the check only
        "max_flux_outside": float(_require(spec, "max_flux_outside", "flux-outside-cutout threshold.", "cutoff")),
        "max_border_sb_sigma": float(_require(spec, "max_border_sb_sigma", "border SB threshold (sigma).", "cutoff")),
        "border_width_pix": int(spec.get("border_width_pix", 3)),  # physics-default-ok: band width of the check
        "max_redraws": int(_require(spec, "max_redraws", "maximum rejected draws per system.", "cutoff")),
    }
    if out["canvas_factor"] < 1:
        raise ValueError("[vela_simulated] cutoff.canvas_factor must be >= 1.")
    return out


def _require(d: Dict[str, Any], key: str, why: str, block: str = "dataset") -> Any:
    if d.get(key) is None:  # physics-default-ok: raise path
        raise ValueError(
            f"[vela_simulated] required parameter {block}.{key!r} is unspecified. {why} "
            f"Set it explicitly in the campaign YAML; the generator refuses to assume a value."
        )
    return d[key]


# ===========================================================================
# Scene model + rendering
# ===========================================================================

def _build_truth_scene_model(prior_spec: Dict[str, Any], light):
    """Scene LensModel whose Components carry the truth prior (Fixed -> constant)."""
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.scene import Component, Plane, LensModel

    def comp_params(group, comp):
        out = {}
        for p, d in prior_spec[group][comp].items():
            out[p] = float(d["value"]) if d["dist"] == "Fixed" else _make_dist(d, f"{group}.{comp}.{p}")
        return out

    planes = [
        Plane(mass=[Component(epl.EPL(50), comp_params("lens_mass", "0")),
                    Component(shear.Shear(), comp_params("lens_mass", "1"))],
              light=[Component(sersic.SersicEllipse(use_lstsq=False), comp_params("lens_light", "0"))]),
        Plane(deflection_ratio=1.0, light=[Component(light, comp_params("source_light", "0"))]),
    ]
    return LensModel(planes)


def _render(sim, model, truth_legacy) -> np.ndarray:
    import jax.numpy as jnp
    from gigalens_research.inference_utils.params import truth_x_to_scene_params
    params = truth_x_to_scene_params(truth_legacy, model)
    img = np.asarray(jnp.squeeze(sim.simulate(params)), dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"[vela_simulated] render returned shape {img.shape}.")
    return img


def _with(truth_legacy, group_idx: int, comp_idx: int, **kv):
    t = copy.deepcopy(truth_legacy)
    t[group_idx][comp_idx].update(kv)
    return t


# ===========================================================================
# Generator
# ===========================================================================

@register_generator("vela_simulated")
def generate_vela_simulated(spec: Any, dataset_dir: str, seed: int) -> None:
    """Simulate fresh lensed Vela systems and save them in System format.

    See the module docstring for the full ``dataset:`` reference.
    """
    import jax
    from jax import random

    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.simulator import SimulatorConfig

    from gigalens_research.simulations.image_based_light import ImageBasedLight
    from gigalens_research.simtests.generate import _add_noise, _hash_dataset
    from gigalens_research.simtests.system import System, write_manifest

    jax.config.update("jax_enable_x64", True)
    extra = dict(spec.extra)

    # --- required science choices -------------------------------------------
    scale_factor = str(_require(extra, "scale_factor",
                                "Source redshift via the VELA scale factor (z = 1/a - 1)."))
    psf_spec = _require(extra, "psf", "PSF block {kind: gaussian|stdpsf|file, ...}.")
    calib = _resolve_calibration(extra)

    # --- structural choices ---------------------------------------------------
    vela_ids = [str(v) for v in extra.get("vela_ids", _DEFAULT_VELA_IDS)]
    cam = str(extra.get("cam", _DEFAULT_CAM))
    filt = str(extra.get("filter", _DEFAULT_FILTER))
    version = str(extra.get("version", _DEFAULT_VERSION))
    n_reps = int(extra.get("n_reps", 1))
    num_pix = int(extra.get("num_pix", 200))  # physics-default-ok: documented structural default, persisted to meta.json
    supersample = int(extra.get("supersample", 4))  # physics-default-ok: documented structural default, persisted to meta.json
    transpose_image = bool(extra.get("transpose_image", False))
    source_root = os.path.expanduser(str(extra.get("source_root", _DEFAULT_SOURCE_ROOT)))
    datadir = os.path.expanduser(str(extra.get("datadir", _DEFAULT_DATADIR)))
    source_variant = str(extra.get("source_variant", _PRISTINE_EXTNAME)).upper()
    if source_variant not in _PRISTINE_VARIANTS:
        raise ValueError(f"[vela_simulated] source_variant={source_variant!r} must be one of "
                         f"{_PRISTINE_VARIANTS}.")
    crop_radius = extra.get("source_crop_radius_arcsec")  # physics-default-ok: None = no crop, recorded in manifest
    crop_radius = None if crop_radius is None else float(crop_radius)
    recenter = bool(extra.get("source_recenter", False))
    allow_unverified = bool(extra.get("allow_unverified_sources", False))
    likelihood_precision = extra.get("likelihood_precision", "float64")  # physics-default-ok: gigalens default, persisted
    conv_precision = extra.get("conv_precision", None)  # physics-default-ok: None = basis dtype, persisted
    cutoff = _resolve_cutoff(extra, supersample)
    border_width = cutoff["border_width_pix"] if cutoff is not None else 3
    prior_spec = resolve_truth_prior_spec(extra.get("truth_prior"))  # physics-default-ok: None = baseline, resolved spec recorded
    prior = build_truth_prior(prior_spec)
    base_key = random.PRNGKey(seed)

    system_ids: List[str] = []
    per_system: Dict[str, Dict[str, Any]] = {}
    psf_meta: Optional[Dict[str, Any]] = None
    noise_meta: Optional[Dict[str, Any]] = None
    source_info: Dict[str, Any] = {}
    sys_index = 0

    for sim in vela_ids:
        source_dir = ensure_pristine_source(
            sim, cam, scale_factor, filt, source_root=source_root, datadir=datadir,
            version=version, source_variant=source_variant,
            allow_unverified_sources=allow_unverified)
        sb_raw, src_scale, delta_pix, src_meta = _load_pristine_source(source_dir, transpose_image)
        sb, pre_info = preprocess_source(sb_raw, src_scale, crop_radius_arcsec=crop_radius,
                                         recenter=recenter)
        source_info[_normalize_sim(sim)] = pre_info
        unlensed_flux_cps = float(sb.sum()) * src_scale ** 2  # at amp = 1
        photfnu = float(src_meta["photfnu_Jy"])

        # PSF / noise are per delta_pix (same for every source of a campaign; built once).
        if psf_meta is None:
            psf, psf_meta = build_psf(psf_spec, delta_pix, filt, datadir)
            background_rms, exp_time, noise_meta = resolve_noise(extra, delta_pix, photfnu)
            print(f"[vela_simulated] PSF {psf_meta['kind']}: {psf_meta['size_pix']} px, "
                  f"FWHM {psf_meta['fwhm_arcsec_measured']:.3f}\"; noise {noise_meta['kind']}: "
                  f"background_rms={background_rms:.4g} cps/px, exp_time={exp_time:g} s")

        light = ImageBasedLight(sb, src_scale)
        model = _build_truth_scene_model(prior_spec, light)
        cfg = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=supersample,
                              kernel=psf, likelihood_precision=likelihood_precision,
                              conv_precision=conv_precision)
        sim_cut = SceneSimulator(model, cfg)
        sim_canvas = None
        if cutoff is not None:
            n_canvas = num_pix * cutoff["canvas_factor"]
            if (n_canvas - num_pix) % 2:
                n_canvas += 1
            cfg_canvas = SimulatorConfig(delta_pix=delta_pix, num_pix=n_canvas,
                                         supersample=cutoff["canvas_supersample"], kernel=psf,
                                         likelihood_precision=likelihood_precision,
                                         conv_precision=conv_precision)
            sim_canvas = SceneSimulator(model, cfg_canvas)

        for rep in range(n_reps):
            sys_key = random.fold_in(base_key, sys_index)
            attempt = 0
            rejections: List[Dict[str, float]] = []
            while True:
                truth_key, noise_key = random.split(random.fold_in(sys_key, attempt))
                truth = _sample_to_legacy(prior.sample(seed=truth_key))

                lens_only = _render(sim_cut, model, _with(truth, 2, 0, amp=0.0))
                src_unit = _render(sim_cut, model, _with(_with(truth, 1, 0, Ie=0.0), 2, 0, amp=1.0))
                lens_flux_cut, src_flux_cut_unit = float(lens_only.sum()), float(src_unit.sum())
                metrics: Dict[str, Any] = {}
                if sim_canvas is not None:
                    src_canvas_unit = _render(sim_canvas, model,
                                              _with(_with(truth, 1, 0, Ie=0.0), 2, 0, amp=1.0))
                    lens_canvas = _render(sim_canvas, model, _with(truth, 2, 0, amp=0.0))
                    metrics["flux_outside_frac"] = flux_outside_cutout(src_canvas_unit, num_pix)
                    metrics["lens_flux_canvas"] = float(lens_canvas.sum())
                    metrics["source_flux_canvas_unit_amp"] = float(src_canvas_unit.sum())

                if calib["mode"] == "ratio":
                    if calib["measure_in"] == "canvas":
                        if sim_canvas is None:
                            raise ValueError("[vela_simulated] calibration.measure_in=canvas needs a cutoff block.")
                        amp = calibrate_amp(ratio=calib["source_to_lens_flux_ratio"],
                                            lens_flux=metrics["lens_flux_canvas"],
                                            source_flux_unit_amp=metrics["source_flux_canvas_unit_amp"])
                    else:
                        amp = calibrate_amp(ratio=calib["source_to_lens_flux_ratio"],
                                            lens_flux=lens_flux_cut, source_flux_unit_amp=src_flux_cut_unit)
                else:
                    amp = calib["source_flux_scale"]

                metrics.update({
                    "theta_E": truth[0][0]["theta_E"],
                    "amp": amp,
                    "lens_flux_cutout": lens_flux_cut,
                    "source_flux_cutout": amp * src_flux_cut_unit,
                    "source_to_lens_ratio_cutout": amp * src_flux_cut_unit / lens_flux_cut,
                    "source_flux_unlensed_cps": amp * unlensed_flux_cps,
                    "source_ab_mag_unlensed": -2.5 * np.log10(amp * unlensed_flux_cps * photfnu / 3631.0),
                    "magnification_cutout": src_flux_cut_unit / unlensed_flux_cps,
                    "border_sb_sigma": amp * border_max(src_unit, border_width) / background_rms,
                })
                if "source_flux_canvas_unit_amp" in metrics:
                    metrics["magnification_canvas"] = metrics["source_flux_canvas_unit_amp"] / unlensed_flux_cps

                if cutoff is None:
                    break
                ok = (metrics["flux_outside_frac"] <= cutoff["max_flux_outside"]
                      and metrics["border_sb_sigma"] <= cutoff["max_border_sb_sigma"])
                if ok:
                    break
                rejections.append({"attempt": attempt, "theta_E": truth[0][0]["theta_E"],
                                   "flux_outside_frac": metrics["flux_outside_frac"],
                                   "border_sb_sigma": metrics["border_sb_sigma"]})
                attempt += 1
                if attempt > cutoff["max_redraws"]:
                    raise RuntimeError(
                        f"[vela_simulated] {_normalize_sim(sim)} rep {rep}: {attempt} draws rejected by the "
                        f"cut-off check (max_flux_outside={cutoff['max_flux_outside']}, "
                        f"max_border_sb_sigma={cutoff['max_border_sb_sigma']}). Last: {rejections[-1]}. "
                        f"Loosen the thresholds, shrink theta_E / the source (crop), or enlarge num_pix.")

            truth = _with(truth, 2, 0, amp=amp)
            full = _render(sim_cut, model, truth)
            recon = lens_only + amp * src_unit
            lin_err = float(np.abs(full - recon).max() / max(full.max(), 1e-300))
            if lin_err > 1e-6:
                raise AssertionError(f"[vela_simulated] render is not linear in components "
                                     f"(max rel err {lin_err:.3g}); refusing to save.")
            if full.shape != (num_pix, num_pix):
                raise ValueError(f"[vela_simulated] simulated image shape {full.shape} != ({num_pix}, {num_pix}).")
            noisy = _add_noise(full, background_rms, exp_time, noise_key)

            system_id = f"{_normalize_sim(sim)}_{_normalize_cam(cam)}_{scale_factor}_rep{rep:02d}"
            metrics.update({"n_redraws": len(rejections), "seed_fold_index": sys_index,
                            "accepted_attempt": attempt, "peak_cps": float(full.max())})
            sys_obj = System(
                system_id=system_id,
                observed_image=noisy,
                truth_x=truth,
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
                    "source_flux_scale": amp,
                    "psf_kind": psf_meta["kind"],
                    "psf_fwhm_arcsec": psf_meta["fwhm_arcsec_measured"],
                    **{k: v for k, v in metrics.items() if np.isscalar(v)},
                },
            )
            sys_obj.save(dataset_dir)
            np.save(os.path.join(dataset_dir, "systems", system_id, "noiseless_image.npy"),
                    full.astype(np.float32))
            np.save(os.path.join(dataset_dir, "systems", system_id, "lens_light_only.npy"),
                    lens_only.astype(np.float32))
            with open(os.path.join(dataset_dir, "systems", system_id, "generation.json"), "w") as f:
                json.dump({"metrics": metrics, "rejections": rejections,
                           "source_preprocessing": pre_info}, f, indent=2)
            system_ids.append(system_id)
            per_system[system_id] = metrics
            sys_index += 1
            print(f"[vela_simulated] {system_id}: theta_E={truth[0][0]['theta_E']:.2f} "
                  f"amp={amp:.3g} ratio={metrics['source_to_lens_ratio_cutout']:.3f} "
                  f"mu={metrics['magnification_cutout']:.1f} outside="
                  f"{metrics.get('flux_outside_frac', float('nan')):.3%} "
                  f"border={metrics['border_sb_sigma']:.2f}sigma redraws={len(rejections)}")

    dataset_hash = _hash_dataset(dataset_dir, system_ids)
    write_manifest(
        dataset_dir,
        generator="vela_simulated",
        seed=seed,
        system_ids=system_ids,
        dataset_hash=dataset_hash,
        extra={
            "generator_version": 2,
            "scale_factor": scale_factor,
            "source_variant": source_variant,
            "cam": cam, "filter": filt, "version": version, "n_reps": n_reps,
            "num_pix": num_pix, "supersample": supersample, "transpose_image": transpose_image,
            "likelihood_precision": likelihood_precision, "conv_precision": conv_precision,
            "source_preprocessing": {"crop_radius_arcsec": crop_radius, "recenter": recenter,
                                     "per_source": source_info},
            "psf": psf_meta,
            "noise": noise_meta,
            "calibration": calib,
            "cutoff": cutoff,
            "truth_prior": prior_spec,
            "source_builder": _SOURCE_BUILDER,
            "source_extname": source_variant,
            "per_system": per_system,
            # legacy flat keys read by older plotting scripts
            "psf_kind": psf_meta["kind"],
            "psf_fwhm_arcsec": psf_meta["fwhm_arcsec_measured"],
            "background_rms": noise_meta["background_rms"],
            "exp_time": noise_meta["exp_time"],
            "source_flux_scale": calib.get("source_flux_scale"),
        },
    )
    print(f"[vela_simulated] wrote {len(system_ids)} systems to {dataset_dir}.")
