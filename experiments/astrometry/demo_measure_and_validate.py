"""End-to-end demo: measure point-source astrometry, then certify the covariance.

Simulates a quad with a Gaussian PSF, measures positions and their full
covariance, and runs the validation suite that decides whether that covariance
may be believed. Everything here is synthetic, so the "truth" is known and the
pull test is meaningful; on real data the pull test is run on a *simulation
matched to that data* and the systematics scan is run on the data itself.

Run (NERSC login node is fine for the smoke settings; use a compute node for
``--realizations 200``)::

    python experiments/astrometry/demo_measure_and_validate.py --realizations 60

Roughly 7 s per fit, so the default 60 realizations take about 8 minutes.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.gaussian import Gaussian

from gigalens_research.astrometry import (
    Frame, NoiseSpec, PSFSpec, SystematicsBudget, measure_astrometry)
from gigalens_research.astrometry.validate import (
    TruthScene, frame_roundtrip_check, plot_pull_diagnostics,
    psf_systematics_scan, pull_test, simulate_scene)

NUM_PIX = 60
DELTA_PIX = 0.05
FWHM = 0.15
BACKGROUND_RMS = 0.01
EXPOSURE_TIME = 1000.0

# A quad. The closest pair sits at ~0.52", which bounds the usable search box.
TRUTH_RA = np.array([0.60, -0.55, 0.10, -0.20])
TRUTH_DEC = np.array([0.15, 0.20, -0.62, 0.58])
TRUTH_AMP = np.array([100.0, 80.0, 60.0, 40.0])


def gaussian_kernel(fwhm: float, num_pix: int = 25, delta_pix: float = DELTA_PIX,
                    stretch_y: float = 0.0) -> np.ndarray:
    """A normalised Gaussian kernel; ``stretch_y`` perturbs it for the PSF scan."""
    x, y = util.make_grid(numPix=num_pix, deltapix=delta_pix)
    k = util.array2image(Gaussian().function(
        x, y * (1.0 + stretch_y), amp=1.0, sigma=fwhm / 2.3548,
        center_x=0.0, center_y=0.0))
    return k / k.sum()


def build_scene() -> TruthScene:
    return TruthScene(
        num_pix=NUM_PIX,
        frame=Frame.from_pixel_scale(DELTA_PIX, NUM_PIX),
        psf=PSFSpec(kernel=gaussian_kernel(FWHM)),
        noise=NoiseSpec(background_rms=BACKGROUND_RMS,
                        exposure_time=EXPOSURE_TIME),
        ra=TRUTH_RA, dec=TRUTH_DEC, amp=TRUTH_AMP)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=60,
                    help="noise draws for the pull test (200+ for a real check)")
    ap.add_argument("--out-dir", default="astrometry_demo",
                    help="where plots are written")
    ap.add_argument("--skip-pull-test", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scene = build_scene()
    # Half the closest separation is the ceiling; stay comfortably under it so
    # the optimiser cannot swap two images between realizations.
    measure_kwargs = dict(search_radius=0.18, pso_particles=80,
                          pso_iterations=200)

    print("=" * 72)
    print("1. Frame round-trip — does the coordinate convention hold?")
    print("=" * 72)
    fr = frame_roundtrip_check(scene, measure_kwargs=measure_kwargs)
    print(f"  pix2angle round trip error : {fr['pix2angle_roundtrip_max_err']:.2e}")
    print(f"  worst recovered offset     : {fr['worst_offset_mas']:.2f} mas")
    print(f"  passed                     : {fr['passed']}")
    print("  cutout corners (arcsec):")
    for name, (ra, dec) in fr["corners"].items():
        print(f"    {name}: ({ra:+.3f}, {dec:+.3f})")
    print("  ^ compare these against the corners of the gigalens grid you will")
    print("    use. Agreement here is NOT evidence that the two frames match.")

    print()
    print("=" * 72)
    print("2. A single measurement")
    print("=" * 72)
    image = simulate_scene(scene, seed=0)
    t0 = time.time()
    result = measure_astrometry(
        image, frame=scene.frame, psf=scene.psf, noise=scene.noise,
        init_ra=TRUTH_RA + 0.02, init_dec=TRUTH_DEC - 0.02, **measure_kwargs)
    print(f"  ({time.time() - t0:.1f}s)")
    print(result.summary())
    err = np.hypot(result.x_img - TRUTH_RA, result.y_img - TRUTH_DEC) * 1e3
    print(f"  true offsets from truth (mas): {np.round(err, 2)}")

    print()
    print("=" * 72)
    print("3. PSF systematics — the error the pull test cannot see")
    print("=" * 72)
    variants = [("nominal", PSFSpec(kernel=gaussian_kernel(FWHM))),
                ("fwhm +2%", PSFSpec(kernel=gaussian_kernel(FWHM * 1.02))),
                ("fwhm -2%", PSFSpec(kernel=gaussian_kernel(FWHM * 0.98))),
                ("stretched", PSFSpec(kernel=gaussian_kernel(FWHM, stretch_y=0.03)))]
    scan = psf_systematics_scan(scene, variants, seed=1,
                                measure_kwargs=measure_kwargs)
    print(scan.report(statistical_sigma=np.sqrt(np.diag(result.cov_stat))))
    budget = scan.budget()

    if args.skip_pull_test:
        print("\n(pull test skipped)")
        return

    print()
    print("=" * 72)
    print(f"4. Pull test — {args.realizations} realizations")
    print("=" * 72)
    t0 = time.time()
    pulls = pull_test(
        scene, n_realizations=args.realizations, measure_kwargs=measure_kwargs,
        seed0=1000,
        progress=lambda d, t: print(f"  {d}/{t}", end="\r", flush=True))
    print(f"\n  ({time.time() - t0:.0f}s)")
    plot_path = os.path.join(args.out_dir, "pull_diagnostics.png")
    plot_pull_diagnostics(pulls, path=plot_path)
    print(f"  plot -> {plot_path}   (read this before the numbers below)")
    print()
    print(pulls.report())

    print()
    print("=" * 72)
    print("5. The measurement gigalens should actually be given")
    print("=" * 72)
    final = measure_astrometry(
        image, frame=scene.frame, psf=scene.psf, noise=scene.noise,
        init_ra=TRUTH_RA + 0.02, init_dec=TRUTH_DEC - 0.02,
        systematics=budget, **measure_kwargs)
    print(final.summary())
    print()
    stat = np.sqrt(np.diag(final.cov_stat))
    tot = np.sqrt(np.diag(final.cov_img))
    print(f"  statistical only : median {np.median(stat) * 1e3:.2f} mas, "
          f"max |corr| {np.abs(final.cov_stat / np.outer(stat, stat) - np.eye(len(stat))).max():.3f}")
    print(f"  with systematics : median {np.median(tot) * 1e3:.2f} mas, "
          f"max |corr| {np.abs(final.correlation - np.eye(len(tot))).max():.3f}")
    print("  The variance grows modestly; the correlation is the real change,")
    print("  and it is the part a diagonal sigma_img would throw away.")
    print()
    print("  Handover:")
    print("    from gigalens.jax.point_source_position import PointSourcePositionData")
    print("    data = PointSourcePositionData(src, **result.to_gigalens_kwargs())")


if __name__ == "__main__":
    main()
