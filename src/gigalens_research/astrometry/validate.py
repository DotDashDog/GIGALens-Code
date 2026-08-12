"""Deciding whether a measured astrometric covariance may be believed.

A covariance is a statement about an ensemble: *if this measurement were
repeated, the scatter would look like this*. No single fit can check such a
statement, which is why :func:`~gigalens_research.astrometry.measure.measure_astrometry`
reports its covariance as a claim and this module exists to test it. The failure
mode being guarded against is specific and unpleasant: an astrometric error bar
that is too small produces a lens posterior that is too tight and centred in the
wrong place, and every downstream diagnostic — convergence, effective sample
size, corner plots — looks perfectly healthy.

Four things need running, and they catch different faults.

1. :func:`pull_test` — inject and recover under the *same* PSF and model.
   Certifies the statistical part: the Laplace approximation, the noise model,
   and the correlation structure. Its whitened form is the only test here that
   can see a wrong off-diagonal, because per-coordinate pulls are identical for
   a diagonal and a correlated matrix with the same variances.

2. :func:`psf_systematics_scan` — refit the *same* data with a family of
   plausible PSFs. This is the only probe of the dominant real-world error, and
   the pull test is structurally blind to it: simulating and fitting with one
   PSF assumes away exactly the thing being tested. Its output is a calibrated
   :class:`~gigalens_research.astrometry.measure.SystematicsBudget`.

3. :func:`frame_roundtrip_check` — recover a source injected at a known angular
   position. Catches a flipped axis or an origin offset between the measurement
   frame and the lens-model grid, which no amount of internal consistency will
   reveal.

4. :func:`model_perturbation_check` — refit with the light model changed
   (a component dropped, added, or re-centred). The resulting shift is a
   systematic in the same currency as the PSF scan.

Read the plots before the numbers. :func:`plot_pull_diagnostics` draws the pull
histograms, the chi-squared distribution against its theoretical curve, and the
per-image residual scatter against the reported error ellipse; a covariance that
is wrong in an interesting way (a tail, a bias in one image, a correlation with
the wrong sign) is obvious there and invisible in a summary statistic. A pull
width of 1.0 achieved by averaging an over-estimate and an under-estimate is
still a broken covariance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
import lenstronomy.Util.image_util as image_util

from gigalens_research.astrometry.measure import (
    AstrometryResult,
    Frame,
    NoiseSpec,
    PSFSpec,
    SystematicsBudget,
    common_mode_jacobian,
    measure_astrometry,
)

__all__ = [
    "PullTestResult",
    "SystematicsScanResult",
    "TruthScene",
    "decompose_common_mode",
    "frame_roundtrip_check",
    "model_perturbation_check",
    "plot_pull_diagnostics",
    "psf_systematics_scan",
    "pull_test",
    "simulate_scene",
]


# ---------------------------------------------------------------------------
# Truth scene and simulation
# ---------------------------------------------------------------------------


@dataclass
class TruthScene:
    """A known configuration to inject and recover.

    For the certification to mean anything about the real measurement, this must
    resemble the real cutout in the ways that matter: the same pixel scale, the
    same PSF, the same noise level, and — most easily forgotten — the same
    image separations and flux ratios. Astrometric precision is a strong
    function of how blended the images are, so a validation run on four widely
    separated bright images certifies nothing about a close pair near the
    deflector core.
    """

    num_pix: int
    frame: Frame
    psf: PSFSpec
    noise: NoiseSpec
    ra: np.ndarray
    dec: np.ndarray
    amp: np.ndarray
    lens_light_model_list: Sequence[str] = ()
    kwargs_lens_light: Sequence[Dict[str, Any]] = ()
    supersampling_factor: int = 1

    def __post_init__(self) -> None:
        self.ra = np.asarray(self.ra, dtype=float)
        self.dec = np.asarray(self.dec, dtype=float)
        self.amp = np.asarray(self.amp, dtype=float)
        if not (self.ra.shape == self.dec.shape == self.amp.shape):
            raise ValueError(
                f"TruthScene ra/dec/amp must share a shape; got {self.ra.shape}, "
                f"{self.dec.shape}, {self.amp.shape}."
            )

    @property
    def n_images(self) -> int:
        return int(self.ra.size)

    @property
    def truth_interleaved(self) -> np.ndarray:
        """Truth positions in gigalens' ``[x0, y0, x1, y1, ...]`` order."""
        out = np.empty(2 * self.n_images)
        out[0::2] = self.ra
        out[1::2] = self.dec
        return out


def simulate_scene(scene: TruthScene, seed: int,
                   psf: Optional[PSFSpec] = None) -> np.ndarray:
    """Render ``scene`` and add noise for the given ``seed``.

    ``psf`` overrides the scene's own PSF, which is how the PSF-mismatch studies
    generate data under one PSF and fit it under another.
    """
    use_psf = scene.psf if psf is None else psf
    kwargs_data: Dict[str, Any] = {
        "image_data": np.zeros((scene.num_pix, scene.num_pix))}
    kwargs_data.update(scene.frame.to_kwargs())
    kwargs_data.update(scene.noise.to_kwargs())
    data = ImageData(**kwargs_data)

    lens_light = (LightModel(list(scene.lens_light_model_list))
                  if scene.lens_light_model_list else None)
    model = ImageModel(
        data, use_psf.to_psf(),
        lens_light_model_class=lens_light,
        point_source_class=PointSource(point_source_type_list=["UNLENSED"]),
        kwargs_numerics={"supersampling_factor": int(scene.supersampling_factor)})
    clean = model.image(
        kwargs_lens=None, kwargs_source=None,
        kwargs_lens_light=(list(scene.kwargs_lens_light)
                           if scene.lens_light_model_list else None),
        kwargs_ps=[{"ra_image": scene.ra, "dec_image": scene.dec,
                    "point_amp": scene.amp}])

    # lenstronomy's noise helpers draw from the legacy global numpy stream, so
    # seed it rather than a local Generator; otherwise realizations repeat.
    np.random.seed(int(seed))
    noisy = clean.copy()
    if scene.noise.noise_map is not None:
        noisy = noisy + np.random.normal(0.0, scene.noise.noise_map)
    else:
        if scene.noise.exposure_time is not None:
            noisy = noisy + image_util.add_poisson(clean, scene.noise.exposure_time)
        noisy = noisy + image_util.add_background(clean, scene.noise.background_rms)
    return noisy


# ---------------------------------------------------------------------------
# Pull test
# ---------------------------------------------------------------------------


@dataclass
class PullTestResult:
    """Outcome of an inject-and-recover ensemble."""

    residuals: np.ndarray          # (n_ok, 2n) measured - truth, arcsec, interleaved
    pulls: np.ndarray              # (n_ok, 2n) residual / marginal sigma
    whitened: np.ndarray           # (n_ok, 2n) L^-1 residual
    chi2: np.ndarray               # (n_ok,) |whitened|^2
    chi2_diag: np.ndarray          # (n_ok,) same, ignoring off-diagonals
    reported_cov: np.ndarray       # (2n, 2n) mean reported covariance
    seeds: np.ndarray
    failures: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def n_realizations(self) -> int:
        return int(self.residuals.shape[0])

    @property
    def dof(self) -> int:
        return int(self.residuals.shape[1])

    def summary(self) -> Dict[str, Any]:
        """Calibration statistics, each with the sampling error it must be judged against."""
        n, k = self.n_realizations, self.dof
        if n < 2:
            raise RuntimeError(
                f"A pull test needs at least 2 successful realizations; got {n}."
            )
        emp = np.cov(self.residuals, rowvar=False)
        # Whitened empirical covariance: identity iff the reported matrix is right,
        # including its off-diagonals.
        chol = np.linalg.cholesky(self.reported_cov)
        w = np.linalg.inv(chol)
        whitened_emp = w @ emp @ w.T
        eig = np.linalg.eigvalsh(whitened_emp)

        out: Dict[str, Any] = {
            "n_realizations": n,
            "dof": k,
            "pull_mean": self.pulls.mean(axis=0),
            "pull_std": self.pulls.std(axis=0, ddof=1),
            "pull_mean_all": float(self.pulls.mean()),
            "pull_std_all": float(self.pulls.std(ddof=1)),
            "pull_mean_se": 1.0 / np.sqrt(n),
            "pull_std_se": 1.0 / np.sqrt(2.0 * (n - 1)),
            "whitened_mean_all": float(self.whitened.mean()),
            "whitened_std_all": float(self.whitened.std(ddof=1)),
            "chi2_mean": float(self.chi2.mean()),
            "chi2_expected": float(k),
            "chi2_mean_se": float(np.sqrt(2.0 * k / n)),
            "chi2_diag_mean": float(self.chi2_diag.mean()),
            "whitened_cov_eig_min": float(eig.min()),
            "whitened_cov_eig_max": float(eig.max()),
            # A sample covariance from n draws in k dimensions has badly spread
            # eigenvalues even when it is drawn from the identity: the extremes
            # follow Marchenko-Pastur with ratio q = k/(n-1). Judging them
            # against a fixed window like [0.7, 1.4] declares a correct
            # covariance broken whenever n is not much larger than k, which for
            # a quad means anything under a few hundred realizations.
            "mp_ratio": float(k / (n - 1)),
            "mp_eig_lo": float((1.0 - np.sqrt(k / (n - 1))) ** 2)
            if k < (n - 1) else 0.0,
            "mp_eig_hi": float((1.0 + np.sqrt(k / (n - 1))) ** 2),
            "variance_ratio": float(np.trace(emp) / np.trace(self.reported_cov)),
            "n_failures": len(self.failures),
        }
        try:
            from scipy import stats
            out["chi2_ks_pvalue"] = float(
                stats.kstest(self.chi2, "chi2", args=(k,)).pvalue)
            # A high mean chi2 has two very different causes: every error bar is
            # slightly too small, or most are fine and a few fits went badly. The
            # median is insensitive to the second, so comparing a robust scale
            # against the mean-based one tells them apart — and they call for
            # opposite responses (rescale the covariance vs. fix the optimiser).
            out["chi2_robust_scale"] = float(
                np.median(self.chi2) / stats.chi2.median(k))
            out["chi2_outlier_fraction"] = float(
                np.mean(self.chi2 > stats.chi2.ppf(0.99, k)))
        except Exception:                                # scipy optional
            out["chi2_ks_pvalue"] = float("nan")
            out["chi2_robust_scale"] = float("nan")
            out["chi2_outlier_fraction"] = float("nan")
        return out

    def report(self, tol_sigma: float = 3.0) -> str:
        """Human-readable verdict.

        ``tol_sigma`` is how many sampling-error units a statistic may stray
        before it is called a failure. The defaults are deliberately loose: with
        a few dozen realizations the sampling error on a pull width is several
        percent, and a test that flags noise is a test that gets ignored.
        """
        s = self.summary()
        lines = [
            f"Pull test: {s['n_realizations']} realizations, {s['dof']} coordinates"
            + (f" ({s['n_failures']} fits failed and were dropped)"
               if s["n_failures"] else ""),
            "",
            f"  pull mean  = {s['pull_mean_all']:+.3f}  "
            f"(expect 0 +/- {s['pull_mean_se'] / np.sqrt(s['dof']):.3f})",
            f"  pull width = {s['pull_std_all']:.3f}  "
            f"(expect 1 +/- {s['pull_std_se'] / np.sqrt(s['dof']):.3f})",
            f"  chi2 mean  = {s['chi2_mean']:.2f}  "
            f"(expect {s['chi2_expected']:.0f} +/- {s['chi2_mean_se']:.2f})",
            f"  chi2 KS p  = {s['chi2_ks_pvalue']:.3f}",
            f"  chi2 robust scale = {s['chi2_robust_scale']:.3f} (expect 1); "
            f"{s['chi2_outlier_fraction']:.1%} of fits beyond the 99th "
            f"percentile (expect 1%)",
            f"  whitened empirical covariance eigenvalues in "
            f"[{s['whitened_cov_eig_min']:.2f}, {s['whitened_cov_eig_max']:.2f}] "
            f"(Marchenko-Pastur band for a correct matrix at this N: "
            f"[{s['mp_eig_lo']:.2f}, {s['mp_eig_hi']:.2f}])",
            f"  chi2 using the diagonal only = {s['chi2_diag_mean']:.2f}"
            f"  <- differs from {s['chi2_mean']:.2f} exactly insofar as the "
            f"correlations matter",
            "",
        ]
        fails = []
        n_eff = np.sqrt(s["dof"] * s["n_realizations"])
        if abs(s["pull_mean_all"]) > tol_sigma / n_eff:
            fails.append(
                f"pull mean {s['pull_mean_all']:+.3f} is biased — the estimator "
                f"is not centred on the truth")
        if abs(s["pull_std_all"] - 1.0) > tol_sigma * s["pull_std_se"] / np.sqrt(s["dof"]):
            direction = "under" if s["pull_std_all"] > 1 else "over"
            fails.append(
                f"pull width {s['pull_std_all']:.3f} != 1 — the covariance "
                f"{direction}-estimates the scatter")
        if abs(s["chi2_mean"] - s["dof"]) > tol_sigma * s["chi2_mean_se"]:
            fails.append(
                f"chi2 mean {s['chi2_mean']:.2f} != {s['dof']} — the joint "
                f"covariance is wrong even if the marginals are right")
        margin = 1.0 + 0.5 * s["mp_ratio"]      # MP is asymptotic; leave slack at small N
        if (s["whitened_cov_eig_min"] < s["mp_eig_lo"] / margin
                or s["whitened_cov_eig_max"] > s["mp_eig_hi"] * margin):
            fails.append(
                f"whitened covariance eigenvalues "
                f"[{s['whitened_cov_eig_min']:.2f}, {s['whitened_cov_eig_max']:.2f}] "
                f"fall outside the Marchenko-Pastur band "
                f"[{s['mp_eig_lo']:.2f}, {s['mp_eig_hi']:.2f}] — some direction "
                f"in the {s['dof']}-dim space is mis-weighted, which points at "
                f"an off-diagonal term rather than an overall scale")
        if s["n_failures"] > 0.05 * (s["n_realizations"] + s["n_failures"]):
            fails.append(
                f"{s['n_failures']} of "
                f"{s['n_realizations'] + s['n_failures']} fits failed — the "
                f"surviving sample is a selected subset, so the statistics "
                f"above describe the fits that converged, not the estimator")

        lines.append("VERDICT: statistical covariance is calibrated"
                     if not fails else "VERDICT: FAILED")
        for f in fails:
            lines.append(f"  - {f}")
        if (np.isfinite(s["chi2_robust_scale"])
                and s["chi2_mean"] > s["dof"] + s["chi2_mean_se"]
                and s["chi2_robust_scale"] < 1.15
                and s["chi2_outlier_fraction"] > 0.03):
            lines.append(
                f"  note: the excess is a tail, not a scale — the median chi2 "
                f"implies a scale of {s['chi2_robust_scale']:.2f} while "
                f"{s['chi2_outlier_fraction']:.0%} of fits sit beyond the 99th "
                f"percentile. Inflating the covariance would be the wrong fix; "
                f"look at the badly-fit realizations instead.")
        if s["mp_ratio"] > 0.25:
            lines.append(
                f"  (weak test: {s['n_realizations']} realizations for "
                f"{s['dof']} coordinates. The eigenvalue check has little power "
                f"below ~10x as many realizations as coordinates.)")
        lines.append("")
        lines.append(
            "  Note: this certifies the statistical covariance only. Both the "
            "simulation and the fit used the same PSF and the same light model, "
            "so PSF and model systematics are assumed away here by construction "
            "— run psf_systematics_scan for those.")
        return "\n".join(lines)


def pull_test(
    scene: TruthScene,
    *,
    n_realizations: int = 200,
    measure_kwargs: Optional[Dict[str, Any]] = None,
    fit_psf: Optional[PSFSpec] = None,
    seed0: int = 0,
    init_jitter: float = 0.02,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PullTestResult:
    """Inject ``scene`` at ``n_realizations`` noise draws and recover it.

    Parameters
    ----------
    scene:
        Truth configuration; see :class:`TruthScene`.
    n_realizations:
        Number of independent noise draws. The sampling error on the pull width
        is :math:`1/\\sqrt{2 N k}` for ``k`` coordinates, so 200 realizations of
        a quad measure the width to about 1%, and 20 only to about 4% — enough
        to catch a factor-of-two error, not enough to certify a 10% one. Each
        realization costs one full optimisation, so this is the expensive step;
        run it on a compute node.
    measure_kwargs:
        Extra keyword arguments forwarded to
        :func:`~gigalens_research.astrometry.measure.measure_astrometry`. Use
        the *same* settings as the real measurement — a validation run with a
        different search radius or optimiser budget certifies a different
        estimator.
    fit_psf:
        PSF used for fitting, if different from the one used for simulating.
        Leave ``None`` for the statistical test.
    init_jitter:
        1-sigma scatter, arcsec, added to the truth to seed each fit, so the
        test measures the estimator rather than the quality of the starting
        guess.
    progress:
        Optional ``callback(done, total)``.

    Notes
    -----
    Any systematics budget in ``measure_kwargs`` is *removed* before fitting:
    a common-mode term inflates the reported covariance without inflating the
    simulated scatter, so leaving it in would drive the pull width below 1 by
    construction and the test would be measuring its own configuration rather
    than the estimator.
    """
    kwargs = dict(measure_kwargs or {})
    kwargs["systematics"] = SystematicsBudget()   # dropped; see the docstring note
    kwargs.setdefault("lens_light_model_list", scene.lens_light_model_list)
    kwargs.setdefault("supersampling_factor", scene.supersampling_factor)

    truth = scene.truth_interleaved
    rng_init = np.random.default_rng(seed0 + 987654321)

    residuals, covs, ok_seeds = [], [], []
    failures: List[Tuple[int, str]] = []
    for i in range(int(n_realizations)):
        seed = seed0 + i
        try:
            image = simulate_scene(scene, seed)
            jit_ra = rng_init.normal(0.0, init_jitter, scene.n_images)
            jit_dec = rng_init.normal(0.0, init_jitter, scene.n_images)
            res = measure_astrometry(
                image, frame=scene.frame,
                psf=scene.psf if fit_psf is None else fit_psf,
                noise=scene.noise,
                init_ra=scene.ra + jit_ra, init_dec=scene.dec + jit_dec,
                **kwargs)
            meas = np.empty(2 * scene.n_images)
            meas[0::2] = res.x_img
            meas[1::2] = res.y_img
            residuals.append(meas - truth)
            covs.append(res.cov_img)
            ok_seeds.append(seed)
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            # A fit that did not converge is data about the estimator, so it is
            # recorded and the ensemble continues. A ValueError is not: that
            # means the *configuration* is wrong, and every realization would
            # hit it, so it propagates rather than quietly shrinking the sample
            # to a biased subset that still reports a confident verdict.
            failures.append((seed, f"{type(exc).__name__}: {exc}"))
        if progress is not None:
            progress(i + 1, int(n_realizations))

    if not residuals:
        raise RuntimeError(
            f"Every one of {n_realizations} fits failed; first error: "
            f"{failures[0][1] if failures else 'unknown'}")

    residuals = np.asarray(residuals)
    reported = np.mean(np.asarray(covs), axis=0)
    reported = 0.5 * (reported + reported.T)

    sig = np.sqrt(np.diag(reported))
    pulls = residuals / sig
    w = np.linalg.inv(np.linalg.cholesky(reported))
    whitened = residuals @ w.T
    chi2 = np.sum(whitened ** 2, axis=1)
    chi2_diag = np.sum(pulls ** 2, axis=1)

    return PullTestResult(residuals=residuals, pulls=pulls, whitened=whitened,
                          chi2=chi2, chi2_diag=chi2_diag, reported_cov=reported,
                          seeds=np.asarray(ok_seeds), failures=failures)


# ---------------------------------------------------------------------------
# Systematics
# ---------------------------------------------------------------------------


def decompose_common_mode(x: np.ndarray, y: np.ndarray,
                          shift: np.ndarray) -> Dict[str, float]:
    """Project an interleaved displacement field onto translation/rotation/scale.

    Returns the four best-fit coefficients and the RMS of what is left over. The
    leftover matters: it is the part of the systematic that is *not* common-mode
    and therefore belongs in ``sigma_independent`` rather than in the correlated
    block.
    """
    jac = common_mode_jacobian(x, y)
    coef, *_ = np.linalg.lstsq(jac, np.asarray(shift, dtype=float), rcond=None)
    resid = shift - jac @ coef
    return {"t_x": float(coef[0]), "t_y": float(coef[1]),
            "rotation": float(coef[2]), "scale": float(coef[3]),
            "residual_rms": float(np.sqrt(np.mean(resid ** 2)))}


@dataclass
class SystematicsScanResult:
    """Position shifts induced by refitting under alternative PSFs."""

    shifts: np.ndarray                     # (n_variants, 2n), arcsec, interleaved
    decompositions: List[Dict[str, float]]
    labels: List[str]
    reference: np.ndarray                  # (2n,) positions under the nominal PSF

    def budget(self) -> SystematicsBudget:
        """The :class:`SystematicsBudget` implied by the scatter across variants.

        Each term is the RMS of the corresponding coefficient over the PSF
        family. This is only as good as that family is representative: a scan
        over three PSFs that happen to be similar will report a small budget and
        prove nothing. The family should span what is actually uncertain about
        the PSF — different stars, different epochs, different reconstruction
        settings.
        """
        if not self.decompositions:
            raise RuntimeError("No PSF variants were scanned.")
        arr = {k: np.array([d[k] for d in self.decompositions])
               for k in ("t_x", "t_y", "rotation", "scale", "residual_rms")}
        trans = float(np.sqrt(np.mean(np.concatenate([arr["t_x"], arr["t_y"]]) ** 2)))
        return SystematicsBudget(
            sigma_translation=trans,
            sigma_rotation=float(np.sqrt(np.mean(arr["rotation"] ** 2))),
            sigma_scale=float(np.sqrt(np.mean(arr["scale"] ** 2))),
            sigma_independent=float(np.sqrt(np.mean(arr["residual_rms"] ** 2))))

    def report(self, statistical_sigma: Optional[np.ndarray] = None) -> str:
        b = self.budget()
        lines = [f"PSF systematics scan over {len(self.labels)} variants:", ""]
        for lab, d in zip(self.labels, self.decompositions):
            lines.append(
                f"  {lab:<24s} shift = ({d['t_x'] * 1e3:+7.2f}, "
                f"{d['t_y'] * 1e3:+7.2f}) mas, rot = {d['rotation'] * 1e3:+.3g} mrad, "
                f"scale = {d['scale'] * 1e3:+.3g}e-3, leftover "
                f"{d['residual_rms'] * 1e3:.2f} mas")
        lines += ["", "  implied budget:",
                  f"    sigma_translation = {b.sigma_translation * 1e3:.2f} mas",
                  f"    sigma_rotation    = {b.sigma_rotation * 1e3:.3g} mrad",
                  f"    sigma_scale       = {b.sigma_scale:.3g}",
                  f"    sigma_independent = {b.sigma_independent * 1e3:.2f} mas"]
        if statistical_sigma is not None:
            stat = float(np.median(np.asarray(statistical_sigma)))
            ratio = b.sigma_translation / max(stat, 1e-30)
            lines += ["", f"  median statistical sigma = {stat * 1e3:.2f} mas; "
                          f"common-mode systematic is {ratio:.1f}x that."]
            if ratio > 1.0:
                lines.append(
                    "  The systematic dominates. A diagonal cov_img would be "
                    "wrong in the one direction the lens model cares about most.")
        return "\n".join(lines)


def psf_systematics_scan(
    scene: TruthScene,
    psf_variants: Sequence[Tuple[str, PSFSpec]],
    *,
    seed: int = 0,
    measure_kwargs: Optional[Dict[str, Any]] = None,
    image: Optional[np.ndarray] = None,
) -> SystematicsScanResult:
    """Refit one dataset under several PSFs and decompose the induced shifts.

    The pull test cannot see PSF error, because it simulates and fits with the
    same kernel. This does: hold the data fixed, vary the PSF over a family that
    represents what is genuinely uncertain, and watch where the positions go.
    Because a PSF error displaces every image in a correlated way, the shift is
    reported through :func:`decompose_common_mode` rather than as per-image
    scatter, and :meth:`SystematicsScanResult.budget` turns it into the
    :class:`~gigalens_research.astrometry.measure.SystematicsBudget` that
    ``measure_astrometry`` will add to the statistical covariance.

    Parameters
    ----------
    scene:
        Truth configuration. Only used to simulate the data (when ``image`` is
        not given) and to seed the fits.
    psf_variants:
        ``(label, PSFSpec)`` pairs. The first is treated as nominal, and all
        shifts are measured relative to it.
    image:
        Real data to use instead of a simulation. Preferred when available: the
        point is to learn how *this* dataset responds to PSF error.
    """
    if len(psf_variants) < 2:
        raise ValueError(
            "psf_systematics_scan needs at least two PSF variants — a nominal "
            "one and something to compare it against.")
    kwargs = dict(measure_kwargs or {})
    kwargs["systematics"] = SystematicsBudget()
    kwargs.setdefault("lens_light_model_list", scene.lens_light_model_list)
    kwargs.setdefault("supersampling_factor", scene.supersampling_factor)
    data = simulate_scene(scene, seed) if image is None else np.asarray(image, float)

    fits = []
    for _, spec in psf_variants:
        res = measure_astrometry(
            data, frame=scene.frame, psf=spec, noise=scene.noise,
            init_ra=scene.ra, init_dec=scene.dec, **kwargs)
        v = np.empty(2 * scene.n_images)
        v[0::2] = res.x_img
        v[1::2] = res.y_img
        fits.append(v)

    reference = fits[0]
    shifts = np.asarray(fits[1:]) - reference
    labels = [lab for lab, _ in psf_variants[1:]]
    x, y = reference[0::2], reference[1::2]
    decomps = [decompose_common_mode(x, y, s) for s in shifts]
    return SystematicsScanResult(shifts=shifts, decompositions=decomps,
                                 labels=labels, reference=reference)


def model_perturbation_check(
    scene: TruthScene,
    variants: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    seed: int = 0,
    measure_kwargs: Optional[Dict[str, Any]] = None,
    image: Optional[np.ndarray] = None,
) -> SystematicsScanResult:
    """Same idea as :func:`psf_systematics_scan`, for light-model choices.

    ``variants`` are ``(label, extra_measure_kwargs)`` pairs — for example one
    fit with the deflector light modelled and one without, or with the Sersic
    index free versus fixed. Unmodelled light under the images biases positions
    coherently toward the galaxy, so this belongs in the same budget as the PSF.
    """
    if len(variants) < 2:
        raise ValueError("model_perturbation_check needs at least two variants.")
    base = dict(measure_kwargs or {})
    base["systematics"] = SystematicsBudget()
    data = simulate_scene(scene, seed) if image is None else np.asarray(image, float)

    fits = []
    for _, extra in variants:
        kw = dict(base)
        kw.update(extra)
        kw.setdefault("lens_light_model_list", scene.lens_light_model_list)
        kw.setdefault("supersampling_factor", scene.supersampling_factor)
        res = measure_astrometry(
            data, frame=scene.frame, psf=scene.psf, noise=scene.noise,
            init_ra=scene.ra, init_dec=scene.dec, **kw)
        v = np.empty(2 * scene.n_images)
        v[0::2] = res.x_img
        v[1::2] = res.y_img
        fits.append(v)

    reference = fits[0]
    shifts = np.asarray(fits[1:]) - reference
    x, y = reference[0::2], reference[1::2]
    return SystematicsScanResult(
        shifts=shifts, decompositions=[decompose_common_mode(x, y, s) for s in shifts],
        labels=[lab for lab, _ in variants[1:]], reference=reference)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


def frame_roundtrip_check(scene: TruthScene, *, seed: int = 0,
                          measure_kwargs: Optional[Dict[str, Any]] = None,
                          tol_mas: float = 5.0) -> Dict[str, Any]:
    """Recover sources injected at known angular coordinates.

    This is the cheapest test here and the one that catches the most damaging
    class of error, because a frame mismatch between the measurement and the
    lens-model grid is invisible to every other diagnostic: the fit converges,
    the covariance is finite, the pulls are unit width, and the lens model is
    wrong. Passing means only that :class:`Frame` is self-consistent with
    lenstronomy's convention — it does *not* prove the frame matches the
    gigalens grid. For that, compare
    :meth:`~gigalens_research.astrometry.measure.Frame.corner_coordinates`
    against the corners of the grid the lens model will use.
    """
    kwargs = dict(measure_kwargs or {})
    kwargs["systematics"] = SystematicsBudget()
    kwargs.setdefault("lens_light_model_list", scene.lens_light_model_list)
    kwargs.setdefault("supersampling_factor", scene.supersampling_factor)
    image = simulate_scene(scene, seed)
    res = measure_astrometry(
        image, frame=scene.frame, psf=scene.psf, noise=scene.noise,
        init_ra=scene.ra, init_dec=scene.dec, **kwargs)

    dx = (res.x_img - scene.ra) * 1e3
    dy = (res.y_img - scene.dec) * 1e3
    worst = float(np.max(np.hypot(dx, dy)))
    x_pix, y_pix = scene.frame.angle2pix(scene.ra, scene.dec)
    ra_rt, dec_rt = scene.frame.pix2angle(x_pix, y_pix)
    return {
        "passed": bool(worst < tol_mas),
        "worst_offset_mas": worst,
        "offset_x_mas": dx,
        "offset_y_mas": dy,
        "pix2angle_roundtrip_max_err": float(
            max(np.max(np.abs(ra_rt - scene.ra)), np.max(np.abs(dec_rt - scene.dec)))),
        "corners": scene.frame.corner_coordinates(scene.num_pix),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_pull_diagnostics(result: PullTestResult, *, path: Optional[str] = None,
                          title: str = "astrometric covariance calibration"):
    """Draw the pull test. Look at this before reading :meth:`PullTestResult.report`.

    Four panels: per-coordinate pull histograms against the unit normal, the
    chi-squared distribution against its theoretical density, the whitened
    empirical covariance (identity if the reported matrix is right), and the
    per-image residual scatter against the reported 1- and 2-sigma ellipses.
    """
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = result.summary()
    k = result.dof
    n_img = k // 2
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    grid = np.linspace(-4, 4, 200)
    for j in range(k):
        ax.hist(result.pulls[:, j], bins=20, range=(-4, 4), histtype="step",
                density=True, alpha=0.7)
    ax.plot(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi), "k-", lw=2,
            label="N(0,1)")
    ax.set_xlabel("pull, per coordinate"); ax.set_ylabel("density")
    ax.set_title(f"pulls: mean {s['pull_mean_all']:+.3f}, "
                 f"width {s['pull_std_all']:.3f}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(result.chi2, bins=25, density=True, alpha=0.5, label="whitened")
    ax.hist(result.chi2_diag, bins=25, density=True, histtype="step",
            label="diagonal only")
    try:
        from scipy import stats
        xs = np.linspace(0, max(result.chi2.max(), k * 3), 300)
        ax.plot(xs, stats.chi2.pdf(xs, k), "k-", lw=2, label=f"chi2({k})")
    except Exception:
        pass
    ax.axvline(k, color="k", ls=":")
    ax.set_xlabel("chi2"); ax.set_title(
        f"chi2 mean {s['chi2_mean']:.2f} vs {k} expected")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    emp = np.cov(result.residuals, rowvar=False)
    w = np.linalg.inv(np.linalg.cholesky(result.reported_cov))
    m = w @ emp @ w.T
    im = ax.imshow(m, cmap="RdBu_r", vmin=0, vmax=2)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("whitened empirical covariance\n(identity if correct)")

    ax = axes[1, 1]
    colors = plt.cm.tab10(np.arange(n_img) % 10)
    for i in range(n_img):
        rx = result.residuals[:, 2 * i] * 1e3
        ry = result.residuals[:, 2 * i + 1] * 1e3
        ax.scatter(rx, ry, s=6, alpha=0.5, color=colors[i], label=f"image {i}")
        c = result.reported_cov[np.ix_([2 * i, 2 * i + 1], [2 * i, 2 * i + 1])] * 1e6
        vals, vecs = np.linalg.eigh(c)
        th = np.linspace(0, 2 * np.pi, 128)
        for nsig in (1, 2):
            e = vecs @ (np.sqrt(vals)[:, None] * np.vstack([np.cos(th), np.sin(th)]))
            ax.plot(nsig * e[0], nsig * e[1], color=colors[i], lw=1.2)
    ax.set_xlabel("dx (mas)"); ax.set_ylabel("dy (mas)")
    ax.set_title("residuals vs reported 1,2-sigma ellipses")
    ax.legend(fontsize=7); ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return fig
