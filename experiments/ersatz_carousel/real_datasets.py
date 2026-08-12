#!/usr/bin/env python
"""Match the carousel's real per-source FITS cutouts to a model, two ways.

Each ``real_cutouts/source{ext}.fits`` carries its own PSF, background RMS and
exposure time -- these are properties of *that cutout* (its band, its footprint on
the mosaic), not of the model. Verified: ``BKG_RMS`` ranges 1.88-4.93 across the nine
files here; only ``EXPTIME`` happens to be constant (620s for all nine, because they
share one exposure). Treat neither as safe to hardcode or share across sources.

Getting the pairing wrong is invisible downstream: every plane still renders, with
someone else's noise model and seeing, and nothing crashes. This is exactly the
mispairing :func:`translate_old_params.cutout_extensions` exists to prevent -- planes
are ordered by redshift, cutouts are named by source number, and the two orders agree
for only three of nine. Accordingly, both entry points below
(:func:`real_datasets_for_model`, :func:`real_simulators_for_model`) drive the file
lookup through ``cutout_extensions(spec)`` via the shared :func:`_matched_exts_and_planes`,
never a hand-written or number-sorted list, and cross-check ``spec``'s redshifts
against the model's plane redshifts before touching a single file -- so a
``model``/``spec`` pair built from two different JSONs (or from JSONs whose source
lists disagree in order) fails loudly instead of quietly pairing the wrong noise
model to the wrong plane.

Two entry points, for two different jobs
-----------------------------------------
**Fitting the real image** -- :func:`real_datasets_for_model` returns ``ImageData``,
whose ``error_map`` is derived from the *observed* pixels (``sqrt(bg**2 +
clip(image, 0, inf)/exp_time)``, per ``ImageData``'s own construction). That is only
valid for the image that produced it.

**Simulating a mock with the real PSF** -- :func:`real_simulators_for_model` returns
a bare ``SceneSimulator`` per plane (real PSF, no observed pixels involved) plus that
cutout's ``background_rms``/``exp_time``, and stops there. It deliberately does NOT
hand back a noised image or an ``ImageData``: the Poisson-approximation term of the
noise depends on the actual flux in the image being noised
(``gigalens.jax.utils.noise.noise_sigma``: ``sqrt(|image|/exp_time + sigma_bkd**2)``),
so a real cutout's stored ``STAT``/error map is the noise level for the TRUE
(observed) image and is wrong for a simulated one -- reusing it would silently hand a
recovery test a noise level no real observation could ever supply in advance. Noise
the simulated image yourself, from ITS OWN flux, via ``add_noise``::

    from translate_old_params import build
    from real_datasets import real_simulators_for_model
    from gigalens.jax.utils.noise import add_noise
    import jax

    model, spec = build("MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json",
                        zero_negative_amplitudes=True)
    sims = real_simulators_for_model(model, spec, "real_cutouts", delta_pix=0.2,
                                     supersample=3)
    params = model.to_params({})
    key = jax.random.PRNGKey(0)
    for entry, subkey in zip(sims, jax.random.split(key, len(sims))):
        image = entry.simulator.simulate(params)
        noisy = add_noise(subkey, image, entry.exp_time, entry.background_rms)

(One ``key`` per plane, via ``jax.random.split`` -- reusing a single key across
planes would correlate their noise realizations; see ``add_noise``'s own docstring.)

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


def _matched_exts_and_planes(model: Any, spec: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """``cutout_extensions(spec)`` zipped against ``model``'s light-bearing planes,
    after checking the two agree on count and per-plane redshift, in order.

    Shared by every public function below so the one guard against a mismatched
    ``model``/``spec`` pair -- built from different JSONs, or from JSONs whose source
    lists disagree in order -- lives in exactly one place rather than drifting out of
    sync between a dataset path and a simulator path.
    """
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
    return list(zip(exts, planes))


def real_datasets_for_model(
    model: Any, spec: Dict[str, Any], dir_path: str, *,
    delta_pix: float, supersample: int = 1,
    likelihood_precision: str = "float64", conv_precision: str = "float32",
    mode: Optional[str] = None, psf_sum_tol: float = 1e-4,
) -> List[Any]:
    """One ``ImageData`` per source plane, its PSF/background_rms/exp_time read from
    its own real cutout -- never a hand-written or positional pairing. For fitting
    the real image; see :func:`real_simulators_for_model` for simulating a mock.

    ``model`` and ``spec`` should come from the same JSON (typically the same call to
    :func:`translate_old_params.build`) -- see :func:`_matched_exts_and_planes`.

    ``sees`` for each ``ImageData`` is the plane's own ``light`` component list (by
    object identity, as ``ImageData.resolve_sees`` requires) -- so the returned
    datasets are only ever safe to use against the exact ``model`` passed in, not a
    separately-built one with the same-looking planes.
    """
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.scene_prob_model import ImageData

    datasets = []
    for ext, plane in _matched_exts_and_planes(model, spec):
        cutout = load_real_cutout(dir_path, ext, psf_sum_tol=psf_sum_tol)
        cfg = SimulatorConfig(
            delta_pix=delta_pix, num_pix=cutout.image.shape, supersample=supersample,
            kernel=cutout.psf, likelihood_precision=likelihood_precision,
            conv_precision=conv_precision)
        datasets.append(ImageData(
            cutout.image, cfg, background_rms=cutout.background_rms,
            exp_time=cutout.exp_time, mask=cutout.mask, sees=plane.light, mode=mode))
    return datasets


@dataclass(frozen=True)
class RealSimulator:
    """One source plane's ``SceneSimulator``, built with its own real PSF, plus the
    ``background_rms``/``exp_time`` needed to noise whatever it renders.

    Deliberately NOT an ``error_map``: the noise level appropriate for a simulated
    image depends on that image's own flux (see the module docstring), so this hands
    back the two raw numbers and lets the caller noise its own simulated image with
    them (e.g. via ``gigalens.jax.utils.noise.add_noise``), instead of a precomputed
    sigma that would be silently wrong for anything but the real observed pixels.
    """

    ext: str
    redshift: float
    simulator: Any               # gigalens.jax.scene_simulator.SceneSimulator
    background_rms: float
    exp_time: float


def real_simulators_for_model(
    model: Any, spec: Dict[str, Any], dir_path: str, *,
    delta_pix: float, supersample: int = 1,
    likelihood_precision: str = "float64", conv_precision: str = "float64",
    psf_sum_tol: float = 1e-4,
) -> List[RealSimulator]:
    """One ``SceneSimulator`` per source plane, rendering with that plane's own real
    PSF -- for SIMULATING a mock with the carousel's real seeing, not for fitting the
    real image (see :func:`real_datasets_for_model` for that, and the module
    docstring for why the two need different noise handling).

    ``model`` and ``spec`` are matched the same way as :func:`real_datasets_for_model`
    (see :func:`_matched_exts_and_planes`). ``num_pix`` is still read from each
    cutout's own ``DATA`` shape -- the simulated grid should match the real footprint
    it is mimicking -- but the cutout's pixel VALUES (image, error map, mask) are
    otherwise unused; only its PSF and header-derived ``background_rms``/``exp_time``
    feed the returned :class:`RealSimulator`.

    ``supersample`` is forwarded to every ``SimulatorConfig`` unchanged (default 1,
    i.e. no supersampling) -- raise it if the real PSF is undersampled at the
    carousel's native ``delta_pix`` and the render needs a finer sub-grid.
    """
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.scene_simulator import SceneSimulator

    sims = []
    for ext, plane in _matched_exts_and_planes(model, spec):
        cutout = load_real_cutout(dir_path, ext, psf_sum_tol=psf_sum_tol)
        cfg = SimulatorConfig(
            delta_pix=delta_pix, num_pix=cutout.image.shape, supersample=supersample,
            kernel=cutout.psf, likelihood_precision=likelihood_precision,
            conv_precision=conv_precision)
        simulator = SceneSimulator(model, cfg, sees=plane.light)
        sims.append(RealSimulator(
            ext=ext, redshift=plane.redshift, simulator=simulator,
            background_rms=cutout.background_rms, exp_time=cutout.exp_time))
    return sims


# --------------------------------------------------------------------------------
def main() -> int:
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json", default=os.path.join(here, "MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json"))
    parser.add_argument("--cutouts", default=os.path.join(here, "real_cutouts"))
    parser.add_argument("--delta-pix", type=float, default=0.2)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--zero-negative-amplitudes", action="store_true")
    parser.add_argument("--simulate", action="store_true",
                        help="build SceneSimulators + noise a mock instead of "
                             "ImageData datasets")
    args = parser.parse_args()

    from translate_old_params import build

    model, spec = build(args.json, zero_negative_amplitudes=args.zero_negative_amplitudes)

    if args.simulate:
        import jax
        from gigalens.jax.utils.noise import add_noise, noise_sigma

        sims = real_simulators_for_model(model, spec, args.cutouts,
                                         delta_pix=args.delta_pix,
                                         supersample=args.supersample)
        params = model.to_params({})
        key = jax.random.PRNGKey(0)
        print(f"{len(sims)} real simulators, matched by cutout_extensions(spec):")
        for entry, subkey in zip(sims, jax.random.split(key, len(sims))):
            image = entry.simulator.simulate(params)
            noisy = add_noise(subkey, image, entry.exp_time, entry.background_rms)
            sigma = noise_sigma(image, entry.exp_time, entry.background_rms)
            print(f"  src ext {entry.ext:<7} z = {entry.redshift:<6.3f}  "
                  f"image {image.shape}  bkg_rms {entry.background_rms:.4f}  "
                  f"exp_time {entry.exp_time:g}  "
                  f"noise sigma [{float(sigma.min()):.4g}, {float(sigma.max()):.4g}]")
        return 0

    datasets = real_datasets_for_model(model, spec, args.cutouts, delta_pix=args.delta_pix,
                                       supersample=args.supersample)
    exts = cutout_extensions(spec)
    print(f"{len(datasets)} real datasets, matched by cutout_extensions(spec):")
    for ext, source, ds in zip(exts, spec["sources"], datasets):
        print(f"  src ext {ext:<7} z = {source['redshift']:<6.3f}  "
              f"image {ds.image.shape}  psf {ds.sim_config.kernel.shape}  "
              f"n_light_seen {len(ds.sees)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
