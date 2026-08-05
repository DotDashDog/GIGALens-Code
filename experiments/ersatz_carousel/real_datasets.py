#!/usr/bin/env python
"""Load the carousel's real per-source FITS cutouts into gigalens ``ImageData``.

Each ``real_cutouts/source{ext}.fits`` carries its own PSF, background RMS and
exposure time -- these are properties of *that cutout* (its band, its footprint on
the mosaic), not of the model. Verified: ``BKG_RMS`` ranges 1.88-4.93 across the nine
files here; only ``EXPTIME`` happens to be constant (620s for all nine, because they
share one exposure). Treat neither as safe to hardcode or share across sources.

Getting the pairing wrong is invisible downstream: every plane still renders, with
someone else's noise model and seeing, and nothing crashes. This is exactly the
mispairing :func:`translate_old_params.cutout_extensions` exists to prevent -- planes
are ordered by redshift, cutouts are named by source number, and the two orders agree
for only three of nine. Accordingly, the one entry point here
(:func:`real_datasets_for_model`) always drives the file lookup through
``cutout_extensions(spec)``, never a hand-written or number-sorted list, and cross-
checks ``spec``'s redshifts against the model's plane redshifts before touching a
single file -- so a ``model``/``spec`` pair built from two different JSONs (or from
JSONs whose source lists disagree in order) fails loudly instead of quietly pairing
the wrong noise model to the wrong plane.

Usage
-----
    from translate_old_params import build
    from real_datasets import real_datasets_for_model

    model, spec = build("MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json",
                        zero_negative_amplitudes=True)
    datasets = real_datasets_for_model(model, spec, "real_cutouts", delta_pix=0.2)
    prob_model = ProbModel(model, datasets, mode="forward")

Not handled here: the ``CENTROIDS`` HDU (present in every file, unused everywhere in
this repo so far -- there is nothing to generalize from yet) and per-source masks
beyond what ``MASK`` already carries (``boiler(2).py`` additionally blanks
``[150:250, 150:250]`` for two sources in the original fit; that is a fit-time choice,
not a cutout property, and does not belong in a loader that only reads files).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from translate_old_params import cutout_extensions

#: HDUs every cutout must carry. ``CENTROIDS`` and ``PRIMARY`` are checked separately
#: since they're consulted for header keys, not pixel data.
_REQUIRED_HDUS = ("DATA", "STAT", "MASK", "PSF")


@dataclass(frozen=True)
class RealCutout:
    """One source's real observation: everything ``ImageData`` needs, plus provenance.

    ``image``/``error_map``/``mask``/``psf`` are plain ``numpy`` arrays (``ImageData``
    converts on construction); ``ext``/``path`` are kept so an error raised later,
    deep inside model construction, can still be traced back to a specific file.
    """

    ext: str
    path: str
    image: np.ndarray
    error_map: np.ndarray
    mask: np.ndarray
    psf: np.ndarray
    background_rms: float
    exp_time: float


def load_real_cutout(dir_path: str, ext: str, *, psf_sum_tol: float = 1e-4) -> RealCutout:
    """Read and validate ``{dir_path}/source{ext}.fits``.

    ``ext`` should come from :func:`translate_old_params.cutout_extensions`, in plane
    order -- never a hand-sorted source-number list (see that function's docstring
    for why six of nine planes disagree between the two orderings).

    Validates, rather than trusting the file: every required HDU and header key is
    present, ``STAT`` has no negative pixels (``sqrt`` would otherwise silently
    produce NaN error bars), ``DATA``/``STAT``/``MASK`` agree on shape, and the PSF
    is normalised (sums to 1 within ``psf_sum_tol``) -- an unnormalised kernel
    silently rescales the entire forward model, with no error anywhere downstream.
    """
    path = os.path.join(dir_path, f"source{ext}.fits")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no cutout for source ext {ext!r}: {path} does not exist. Check "
            "cutout_extensions(spec) against the files actually in dir_path.")

    with fits.open(path) as hdul:
        present = {h.name for h in hdul}
        missing = [name for name in _REQUIRED_HDUS if name not in present]
        if missing:
            raise ValueError(
                f"{path}: missing HDU(s) {missing}; has {sorted(present)}")
        if "BKG_RMS" not in hdul["DATA"].header:
            raise KeyError(f"{path}: DATA header has no BKG_RMS")
        if "EXPTIME" not in hdul["PRIMARY"].header:
            raise KeyError(f"{path}: PRIMARY header has no EXPTIME")

        image = np.asarray(hdul["DATA"].data, dtype=np.float64)
        stat = np.asarray(hdul["STAT"].data, dtype=np.float64)
        mask = np.asarray(hdul["MASK"].data).astype(bool)
        psf = np.asarray(hdul["PSF"].data, dtype=np.float64)
        background_rms = float(hdul["DATA"].header["BKG_RMS"])
        exp_time = float(hdul["PRIMARY"].header["EXPTIME"])

    if np.any(stat < 0):
        raise ValueError(
            f"{path}: STAT has {int(np.sum(stat < 0))} negative pixel(s); "
            "sqrt(STAT) would be NaN there.")
    error_map = np.sqrt(stat)

    if not (image.shape == mask.shape == error_map.shape):
        raise ValueError(
            f"{path}: DATA {image.shape}, STAT {error_map.shape}, MASK {mask.shape} "
            "disagree on shape.")

    psf_sum = float(psf.sum())
    if abs(psf_sum - 1.0) > psf_sum_tol:
        raise ValueError(
            f"{path}: PSF sums to {psf_sum:.6f}, not 1 (tol {psf_sum_tol:g}); an "
            "unnormalised kernel silently rescales the whole forward model. "
            "Renormalise or investigate before using it.")

    return RealCutout(ext=ext, path=path, image=image, error_map=error_map, mask=mask,
                      psf=psf, background_rms=background_rms, exp_time=exp_time)


def real_datasets_for_model(
    model: Any, spec: Dict[str, Any], dir_path: str, *,
    delta_pix: float, supersample: int = 1,
    likelihood_precision: str = "float64", conv_precision: str = "float32",
    mode: Optional[str] = None, psf_sum_tol: float = 1e-4,
) -> List[Any]:
    """One ``ImageData`` per source plane, its PSF/background_rms/exp_time read from
    its own real cutout -- never a hand-written or positional pairing.

    ``model`` and ``spec`` should come from the same JSON (typically the same call to
    :func:`translate_old_params.build`). Before opening any file, this checks that
    ``len(cutout_extensions(spec))`` matches the model's source-plane count and that
    ``spec["sources"]``' redshifts agree with the model's plane redshifts, in order --
    the one failure mode nothing else here could otherwise detect: a ``model`` built
    from a different (or differently-ordered) JSON than ``spec``.

    ``sees`` for each ``ImageData`` is the plane's own ``light`` component list (by
    object identity, as ``ImageData.resolve_sees`` requires) -- so the returned
    datasets are only ever safe to use against the exact ``model`` passed in, not a
    separately-built one with the same-looking planes.
    """
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.scene_prob_model import ImageData

    exts = cutout_extensions(spec)
    planes = [p for p in model.planes if p.has_light]
    if len(exts) != len(planes):
        raise ValueError(
            f"{len(exts)} cutouts vs {len(planes)} source planes -- zip would "
            "silently drop the tail. spec and model disagree on source count.")
    for i, (source, plane) in enumerate(zip(spec["sources"], planes)):
        if abs(source["redshift"] - plane.redshift) > 1e-6:
            raise ValueError(
                f"plane {i}: spec redshift {source['redshift']:g} != model plane "
                f"redshift {plane.redshift:g}. model and spec were built from "
                "different JSONs, or from JSONs whose source lists disagree in "
                "order.")

    datasets = []
    for ext, plane in zip(exts, planes):
        cutout = load_real_cutout(dir_path, ext, psf_sum_tol=psf_sum_tol)
        cfg = SimulatorConfig(
            delta_pix=delta_pix, num_pix=cutout.image.shape, supersample=supersample,
            kernel=cutout.psf, likelihood_precision=likelihood_precision,
            conv_precision=conv_precision)
        datasets.append(ImageData(
            cutout.image, cfg, background_rms=cutout.background_rms,
            exp_time=cutout.exp_time, mask=cutout.mask, sees=plane.light, mode=mode))
    return datasets


# --------------------------------------------------------------------------------
def main() -> int:
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json", default=os.path.join(here, "MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json"))
    parser.add_argument("--cutouts", default=os.path.join(here, "real_cutouts"))
    parser.add_argument("--delta-pix", type=float, default=0.2)
    parser.add_argument("--zero-negative-amplitudes", action="store_true")
    args = parser.parse_args()

    from translate_old_params import build

    model, spec = build(args.json, zero_negative_amplitudes=args.zero_negative_amplitudes)
    datasets = real_datasets_for_model(model, spec, args.cutouts, delta_pix=args.delta_pix)
    exts = cutout_extensions(spec)
    print(f"{len(datasets)} real datasets, matched by cutout_extensions(spec):")
    for ext, source, ds in zip(exts, spec["sources"], datasets):
        print(f"  src ext {ext:<7} z = {source['redshift']:<6.3f}  "
              f"image {ds.image.shape}  psf {ds.sim_config.kernel.shape}  "
              f"n_light_seen {len(ds.sees)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
