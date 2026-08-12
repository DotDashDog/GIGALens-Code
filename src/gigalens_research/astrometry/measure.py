"""Fit point sources to a cutout and report positions with a full covariance.

The measurement is a lenstronomy forward model of the pixels: ``n`` free point
sources (lenstronomy's ``UNLENSED`` point-source type), optionally on top of a
parametric lens-light model, convolved with a supplied PSF and compared to the
data under a Gaussian pixel likelihood. Positions come from optimising that
model; the covariance comes from the curvature of the same likelihood at the
optimum (a Laplace approximation).

Three choices in here are load-bearing, and each one exists to close a failure
mode that is otherwise silent.

**No lens model is used.** The point sources are ``UNLENSED``: ``2n`` free
coordinates with no lens equation relating them. That is deliberate. The whole
purpose of the measurement is to feed a gigalens lens-model inference, so
folding a lens model into the measurement would make the data a function of the
thing being inferred, and the resulting posterior would be tightened by its own
prior. The cost is real — a lens model is genuinely informative, and ignoring it
gives larger error bars than a joint fit would — but a joint fit is a different
analysis, not a better version of this one.

**The covariance is marginal, not conditional.** Point-source amplitudes and any
lens-light parameters are free and non-linear (``linear_solver=False``), the
Hessian is taken over *all* of them, and the position block is cut out of the
inverse: ``Sigma = (H^-1)[pos, pos]``, never ``(H[pos, pos])^-1``. The second
form is the covariance at *fixed* nuisance parameters and is too small whenever
position trades off against flux or against the deflector light — which is
exactly the regime of a real lens, where the images sit on the galaxy.

**The output is interleaved.** gigalens orders ``cov_img`` as
``[x0, y0, x1, y1, ...]``; lenstronomy's parameter vector is blocked,
``[ra_0..ra_n-1, dec_0..dec_n-1]``. The two have identical shape and different
meaning, so no validation can catch a mix-up downstream. Rather than extract a
blocked matrix and permute it afterwards, :func:`measure_astrometry` looks up
each coordinate's index by name from ``param_class.num_param()`` and builds the
interleaved order directly, so the result does not depend on lenstronomy's
internal blocking at all. (``tests/test_astrometry_measure.py`` pins this
against ``gigalens``' own ``interleave_xy_cov`` helper.)

What the Laplace covariance does *not* contain is any error in the PSF, the
astrometric solution, or the light model — all of which move every image at
once, and all of which are far more damaging to a lens fit than the
per-image statistical scatter, because a coherent shift is nearly degenerate
with the lens model's own position and shear freedom. :class:`SystematicsBudget`
adds that common mode explicitly; it is not estimated from the data, because a
single image cannot estimate it. See
:mod:`~gigalens_research.astrometry.validate` for how to calibrate it.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Workflow.fitting_sequence import FittingSequence

__all__ = [
    "AstrometryResult",
    "Frame",
    "NoiseSpec",
    "PSFSpec",
    "SystematicsBudget",
    "common_mode_jacobian",
    "measure_astrometry",
]


def common_mode_jacobian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Derivative of interleaved positions w.r.t. ``(t_x, t_y, phi, s)``.

    The four columns are a rigid shift in each axis, a rotation about the image
    centroid, and an isotropic scale change — the leading ways a PSF, WCS or
    light-model error moves *all* the images together. :class:`SystematicsBudget`
    propagates a variance through this matrix;
    :func:`gigalens_research.astrometry.validate.decompose_common_mode` projects
    a measured displacement field onto it. Both must use the same convention,
    which is why it lives here rather than being written out twice.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    dx = x - x.mean()
    dy = y - y.mean()
    jac = np.zeros((2 * n, 4))
    jac[0::2, 0] = 1.0
    jac[1::2, 1] = 1.0
    jac[0::2, 2] = -dy
    jac[1::2, 2] = dx
    jac[0::2, 3] = dx
    jac[1::2, 3] = dy
    return jac


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """The angular coordinate frame of the cutout.

    This is the input most likely to be wrong, and the one with no internal
    check: every number this module reports is a coordinate *in this frame*, and
    gigalens will interpret them in whatever frame its own simulation grid
    defines. If the two disagree — a flipped RA axis, a half-pixel offset, a
    different origin — the positions are still self-consistent, the fit still
    looks good, the covariance is still finite, and the lens inference is
    quietly wrong.

    So build this from the same WCS that defines the gigalens grid, and check it
    with :meth:`corner_coordinates` (or the round-trip test in
    :mod:`~gigalens_research.astrometry.validate`) rather than trusting it.

    Attributes
    ----------
    transform_pix2angle:
        ``(2, 2)`` matrix mapping pixel offsets to angular offsets, arcsec.
    ra_at_xy_0, dec_at_xy_0:
        Angular coordinate of pixel ``(0, 0)``, arcsec.
    """

    transform_pix2angle: np.ndarray
    ra_at_xy_0: float = 0.0
    dec_at_xy_0: float = 0.0

    def __post_init__(self) -> None:
        m = np.asarray(self.transform_pix2angle, dtype=float)
        if m.shape != (2, 2):
            raise ValueError(
                f"Frame.transform_pix2angle must be (2, 2); got {m.shape}."
            )
        if not np.all(np.isfinite(m)):
            raise ValueError("Frame.transform_pix2angle contains non-finite entries.")
        if abs(np.linalg.det(m)) < 1e-12:
            raise ValueError(
                "Frame.transform_pix2angle is singular — it does not define an "
                "invertible pixel-to-angle map."
            )
        object.__setattr__(self, "transform_pix2angle", m)

    @classmethod
    def from_pixel_scale(cls, delta_pix: float, num_pix: int,
                         center: Tuple[float, float] = (0.0, 0.0)) -> "Frame":
        """A square, axis-aligned frame with ``center`` at the image centre.

        Convenience for simulations and tests. Real data should come from a WCS.
        """
        m = np.array([[delta_pix, 0.0], [0.0, delta_pix]], dtype=float)
        half = (num_pix - 1) / 2.0
        return cls(transform_pix2angle=m,
                   ra_at_xy_0=center[0] - half * delta_pix,
                   dec_at_xy_0=center[1] - half * delta_pix)

    @property
    def pixel_scale(self) -> float:
        """Mean linear pixel scale in arcsec (``sqrt|det|``)."""
        return float(np.sqrt(abs(np.linalg.det(self.transform_pix2angle))))

    def to_kwargs(self) -> Dict[str, Any]:
        return {"transform_pix2angle": self.transform_pix2angle.copy(),
                "ra_at_xy_0": float(self.ra_at_xy_0),
                "dec_at_xy_0": float(self.dec_at_xy_0)}

    def pix2angle(self, x_pix, y_pix):
        """Map pixel indices to angular coordinates (arcsec)."""
        x_pix = np.asarray(x_pix, dtype=float)
        y_pix = np.asarray(y_pix, dtype=float)
        m = self.transform_pix2angle
        ra = self.ra_at_xy_0 + m[0, 0] * x_pix + m[0, 1] * y_pix
        dec = self.dec_at_xy_0 + m[1, 0] * x_pix + m[1, 1] * y_pix
        return ra, dec

    def angle2pix(self, ra, dec):
        """Map angular coordinates (arcsec) to pixel indices."""
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        inv = np.linalg.inv(self.transform_pix2angle)
        dx = ra - self.ra_at_xy_0
        dy = dec - self.dec_at_xy_0
        return inv[0, 0] * dx + inv[0, 1] * dy, inv[1, 0] * dx + inv[1, 1] * dy

    def corner_coordinates(self, num_pix: int) -> Dict[str, Tuple[float, float]]:
        """Angular coordinates of the four pixel corners, for eyeballing.

        Compare these against the corners of the gigalens grid you intend to
        use. It is a cheap way to catch an axis flip or an origin offset before
        it becomes a wrong lens model.
        """
        n = num_pix - 1
        out = {}
        for name, (i, j) in {"xy_0_0": (0, 0), "xy_n_0": (n, 0),
                             "xy_0_n": (0, n), "xy_n_n": (n, n)}.items():
            ra, dec = self.pix2angle(i, j)
            out[name] = (float(ra), float(dec))
        return out


@dataclass(frozen=True)
class PSFSpec:
    """The point-spread function used to model the images.

    Attributes
    ----------
    kernel:
        2-D pixel kernel. Normalised to unit sum on construction.
    supersampling_factor:
        If the kernel is sampled finer than the data, the factor by which. A
        kernel at the data's own sampling (factor 1) undersamples the profile
        for a typical space-based PSF and will bias positions toward pixel
        centres; prefer an oversampled kernel with the matching factor here.
    """

    kernel: np.ndarray
    supersampling_factor: int = 1

    def __post_init__(self) -> None:
        k = np.asarray(self.kernel, dtype=float)
        if k.ndim != 2:
            raise ValueError(f"PSFSpec.kernel must be 2-D; got shape {k.shape}.")
        if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
            raise ValueError(
                f"PSFSpec.kernel must have odd side lengths so its centre is a "
                f"pixel; got {k.shape}. An even kernel puts the PSF centre on a "
                f"pixel boundary and biases every position by half a pixel."
            )
        if not np.all(np.isfinite(k)):
            raise ValueError("PSFSpec.kernel contains non-finite entries.")
        total = k.sum()
        if total <= 0:
            raise ValueError(f"PSFSpec.kernel must have positive sum; got {total}.")
        if int(self.supersampling_factor) < 1:
            raise ValueError("PSFSpec.supersampling_factor must be >= 1.")
        object.__setattr__(self, "kernel", k / total)
        object.__setattr__(self, "supersampling_factor", int(self.supersampling_factor))

    def to_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"psf_type": "PIXEL",
                              "kernel_point_source": self.kernel.copy()}
        if self.supersampling_factor > 1:
            kw["point_source_supersampling_factor"] = self.supersampling_factor
        return kw

    def to_psf(self) -> PSF:
        return PSF(**self.to_kwargs())


@dataclass(frozen=True)
class NoiseSpec:
    """The pixel noise model.

    Give either an explicit ``noise_map`` (per-pixel 1-sigma, the honest choice
    for real data, where drizzling has already correlated and rescaled the
    noise) or ``background_rms`` with an optional ``exposure_time`` so
    lenstronomy builds background + Poisson variance itself.

    Getting this wrong rescales the whole covariance by a constant, which the
    pull test in :mod:`~gigalens_research.astrometry.validate` will catch as a
    pull width that is not 1.
    """

    background_rms: Optional[float] = None
    exposure_time: Optional[Any] = None
    noise_map: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if (self.noise_map is None) == (self.background_rms is None):
            if self.noise_map is None:
                raise ValueError(
                    "NoiseSpec needs a noise model: pass noise_map for a "
                    "per-pixel 1-sigma map, or background_rms (optionally with "
                    "exposure_time) to have lenstronomy build one."
                )
            raise ValueError(
                "NoiseSpec got both noise_map and background_rms; pass exactly "
                "one so it is unambiguous which noise model is in force."
            )
        if self.noise_map is not None:
            nm = np.asarray(self.noise_map, dtype=float)
            if nm.ndim != 2:
                raise ValueError(f"NoiseSpec.noise_map must be 2-D; got {nm.shape}.")
            if not np.all(np.isfinite(nm)) or np.any(nm <= 0):
                raise ValueError(
                    "NoiseSpec.noise_map must be finite and strictly positive."
                )
            object.__setattr__(self, "noise_map", nm)

    def to_kwargs(self) -> Dict[str, Any]:
        if self.noise_map is not None:
            return {"noise_map": self.noise_map.copy()}
        kw: Dict[str, Any] = {"background_rms": float(self.background_rms)}
        if self.exposure_time is not None:
            kw["exposure_time"] = self.exposure_time
        return kw


@dataclass(frozen=True)
class SystematicsBudget:
    """A common-mode astrometric error, added to the statistical covariance.

    The Laplace covariance describes how much the *pixel noise* moves the
    fitted positions. It says nothing about the PSF being slightly wrong, the
    astrometric solution being slightly rotated, or the deflector light being
    imperfectly modelled. Those errors are shared across images, so they show up
    as correlated displacements — and a coherent displacement of all four images
    is almost exactly what a small change in the lens centroid, or in the
    external shear, would do. That is why a diagonal error bar is not a
    conservative approximation here: inflating the diagonal spends error budget
    in every direction *except* the one the systematic actually moves.

    The model is a linear error propagation over four latent nuisances — a rigid
    translation, a rotation about the image centroid, and an isotropic scale
    error — plus an uncorrelated per-image floor:

    .. math::
        \\Sigma_\\mathrm{sys} = J \\, \\mathrm{diag}(\\sigma^2) \\, J^\\top
                               + \\sigma_\\mathrm{ind}^2 I

    Every term defaults to zero, so the budget is opt-in and never silently
    inflates anything. Calibrate the terms with the PSF-mismatch study in
    :mod:`~gigalens_research.astrometry.validate`; do not guess them, and in
    particular do not read them off the same fit whose errors they describe.

    Attributes
    ----------
    sigma_translation:
        1-sigma rigid shift per axis, arcsec. Typically the dominant term.
    sigma_rotation:
        1-sigma rotation about the image centroid, radians.
    sigma_scale:
        1-sigma fractional plate-scale error (dimensionless).
    sigma_independent:
        1-sigma uncorrelated per-image, per-axis floor, arcsec. Use for
        image-specific effects such as a nearby contaminant.
    """

    sigma_translation: float = 0.0
    sigma_rotation: float = 0.0
    sigma_scale: float = 0.0
    sigma_independent: float = 0.0

    def __post_init__(self) -> None:
        for name in ("sigma_translation", "sigma_rotation", "sigma_scale",
                     "sigma_independent"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v < 0:
                raise ValueError(
                    f"SystematicsBudget.{name} must be finite and >= 0; got {v}."
                )
            object.__setattr__(self, name, v)

    @property
    def is_zero(self) -> bool:
        return (self.sigma_translation == 0.0 and self.sigma_rotation == 0.0
                and self.sigma_scale == 0.0 and self.sigma_independent == 0.0)

    def covariance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Build the ``(2n, 2n)`` interleaved systematic covariance at ``(x, y)``."""
        x = np.asarray(x, dtype=float)
        n = x.size
        jac = common_mode_jacobian(x, y)
        var = np.array([self.sigma_translation ** 2, self.sigma_translation ** 2,
                        self.sigma_rotation ** 2, self.sigma_scale ** 2])
        cov = jac @ np.diag(var) @ jac.T
        if self.sigma_independent > 0:
            cov = cov + np.eye(2 * n) * self.sigma_independent ** 2
        return cov


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class AstrometryResult:
    """Measured positions and their covariance, ready for gigalens.

    ``cov_img`` is in gigalens' interleaved ``[x0, y0, x1, y1, ...]`` order and
    goes straight into
    :class:`~gigalens.jax.point_source_position.PointSourcePositionData` via
    :meth:`to_gigalens_kwargs`. It must *not* be passed through
    ``interleave_xy_cov`` again — that helper is for a raw lenstronomy chain
    covariance, and applying it here would silently scramble the matrix back
    into blocked order.
    """

    x_img: np.ndarray
    y_img: np.ndarray
    cov_img: np.ndarray
    cov_stat: np.ndarray
    cov_sys: np.ndarray
    amp: np.ndarray
    amp_err: np.ndarray
    kwargs_result: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_images(self) -> int:
        return int(self.x_img.size)

    @property
    def sigma_img(self) -> np.ndarray:
        """Marginal per-coordinate 1-sigma, shape ``(n, 2)``.

        Provided for reporting and for a deliberately-degraded comparison run.
        Passing this to gigalens instead of :attr:`cov_img` discards precisely
        the correlations that motivate the whole exercise.
        """
        return np.sqrt(np.diag(self.cov_img)).reshape(self.n_images, 2)

    @property
    def correlation(self) -> np.ndarray:
        """Correlation matrix of :attr:`cov_img`."""
        s = np.sqrt(np.diag(self.cov_img))
        return self.cov_img / np.outer(s, s)

    def to_gigalens_kwargs(self) -> Dict[str, np.ndarray]:
        """Keyword arguments for ``PointSourcePositionData``.

        Usage::

            data = PointSourcePositionData(src_component, **result.to_gigalens_kwargs())
        """
        return {"x_img": self.x_img.copy(),
                "y_img": self.y_img.copy(),
                "cov_img": self.cov_img.copy()}

    def summary(self) -> str:
        """A short human-readable report, in milliarcseconds."""
        lines = [f"{self.n_images} images, positions (arcsec) and 1-sigma (mas):"]
        sig = self.sigma_img
        for i in range(self.n_images):
            lines.append(
                f"  [{i}] x = {self.x_img[i]:+.6f} +/- {sig[i, 0] * 1e3:6.2f}   "
                f"y = {self.y_img[i]:+.6f} +/- {sig[i, 1] * 1e3:6.2f}   "
                f"amp = {self.amp[i]:.4g} +/- {self.amp_err[i]:.3g}"
            )
        off = np.abs(self.correlation - np.eye(2 * self.n_images))
        lines.append(f"max |off-diagonal correlation| = {off.max():.3f}")
        for key in ("reduced_chi2", "hessian_step_stability", "cov_condition"):
            if key in self.diagnostics:
                lines.append(f"{key} = {self.diagnostics[key]:.4g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _position_indices(param_class) -> Tuple[np.ndarray, np.ndarray]:
    """Interleaved index arrays into the lenstronomy argument vector.

    Returns ``(order, block_order)``: ``order`` is the ``2n`` indices in
    gigalens' ``[x0, y0, x1, y1, ...]`` order, ``block_order`` the same indices
    in lenstronomy's native ``[ra..., dec...]`` blocking. Both are looked up by
    *name*, so neither depends on lenstronomy's internal parameter layout.
    """
    _, names = param_class.num_param()
    ra_idx = [i for i, nm in enumerate(names) if nm == "ra_image"]
    dec_idx = [i for i, nm in enumerate(names) if nm == "dec_image"]
    if not ra_idx or len(ra_idx) != len(dec_idx):
        raise RuntimeError(
            f"Could not locate matched ra_image/dec_image parameters in the "
            f"lenstronomy argument vector (found {len(ra_idx)} ra, "
            f"{len(dec_idx)} dec). Are the positions fixed, or is "
            f"num_point_source_list wrong?"
        )
    order = np.empty(2 * len(ra_idx), dtype=int)
    order[0::2] = ra_idx
    order[1::2] = dec_idx
    return order, np.asarray(ra_idx + dec_idx, dtype=int)


def _neg_logl(fitting_seq) -> Any:
    like = fitting_seq.likelihoodModule

    def f(args: np.ndarray) -> float:
        val = like.logL(np.asarray(args, dtype=float))
        if isinstance(val, tuple):          # some lenstronomy versions return (logL, extra)
            val = val[0]
        return -float(val)

    return f


def _hessian(f, args: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``args``."""
    a = np.asarray(args, dtype=float)
    n = a.size
    h = np.asarray(steps, dtype=float)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = h[i]
            ej = np.zeros(n); ej[j] = h[j]
            vals = (f(a + ei + ej), f(a + ei - ej), f(a - ei + ej), f(a - ei - ej))
            if not all(np.isfinite(v) for v in vals):
                raise RuntimeError(
                    f"The likelihood is not finite in the neighbourhood of the "
                    f"best fit while differentiating parameters ({i}, {j}). This "
                    f"usually means the optimum sits on a parameter bound, so "
                    f"the curvature there is not a covariance at all. Widen the "
                    f"bounds, or check the initial guess."
                )
            hess[i, j] = hess[j, i] = (vals[0] - vals[1] - vals[2] + vals[3]) / (
                4.0 * h[i] * h[j])
    return hess


def _marginal_position_covariance(hess: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Invert the full Hessian, then cut out the position block.

    The order matters and is the whole point: inverting first marginalises over
    amplitudes and lens light, cutting first would condition on them.
    """
    eig = np.linalg.eigvalsh(hess)
    if np.min(eig) <= 0:
        raise RuntimeError(
            f"The Hessian of -logL is not positive definite (smallest eigenvalue "
            f"{np.min(eig):.3e}), so the fit is not at a minimum and its curvature "
            f"is not a covariance. Re-run the optimiser, or check for a "
            f"degenerate/unconstrained parameter."
        )
    cov_full = np.linalg.inv(hess)
    return cov_full[np.ix_(order, order)]


def _default_amp_guess(image: np.ndarray, frame: Frame, psf: PSFSpec,
                       ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Rough amplitudes from the peak pixel under each image."""
    x_pix, y_pix = frame.angle2pix(ra, dec)
    ny, nx = image.shape
    peak = float(psf.kernel.max()) * psf.supersampling_factor ** 2
    out = np.empty(ra.size)
    for k in range(ra.size):
        i = int(np.clip(round(float(y_pix[k])), 0, ny - 1))
        j = int(np.clip(round(float(x_pix[k])), 0, nx - 1))
        lo_i, hi_i = max(0, i - 1), min(ny, i + 2)
        lo_j, hi_j = max(0, j - 1), min(nx, j + 2)
        out[k] = max(float(image[lo_i:hi_i, lo_j:hi_j].max()), 0.0) / max(peak, 1e-12)
    return np.where(out > 0, out, 1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def measure_astrometry(
    image: np.ndarray,
    *,
    frame: Frame,
    psf: PSFSpec,
    noise: NoiseSpec,
    init_ra: Sequence[float],
    init_dec: Sequence[float],
    init_amp: Optional[Sequence[float]] = None,
    search_radius: float = 0.3,
    amp_bounds: Tuple[float, float] = (1e-4, 1e8),
    lens_light_model_list: Sequence[str] = (),
    kwargs_lens_light_params: Optional[List[List[Dict[str, Any]]]] = None,
    mask: Optional[np.ndarray] = None,
    supersampling_factor: int = 1,
    systematics: SystematicsBudget = SystematicsBudget(),
    pso_particles: int = 80,
    pso_iterations: int = 200,
    polish_iterations: int = 400,
    extra_fitting_steps: Sequence[Any] = (),
    hessian_step_scale: float = 1.0,
    verbose: bool = False,
) -> AstrometryResult:
    """Measure ``n`` point-source positions and their joint covariance.

    Parameters
    ----------
    image:
        2-D cutout, in the same units the noise model describes.
    frame:
        Angular frame of the cutout — **must** match the gigalens grid; see
        :class:`Frame`.
    psf, noise:
        PSF kernel and pixel noise model.
    init_ra, init_dec:
        Starting positions, arcsec in ``frame``. Their length fixes ``n``. These
        need only be good to within ``search_radius``; a by-eye click on each
        image is normally enough.
    init_amp:
        Starting amplitudes. Estimated from the peak pixel under each image when
        omitted.
    search_radius:
        Half-width of the uniform position bound around each initial guess,
        arcsec. Keep it comfortably smaller than the image separation so the
        optimiser cannot swap two images between fits — a swap silently
        permutes the rows of ``cov_img``.
    lens_light_model_list, kwargs_lens_light_params:
        Optional parametric light for the deflector, as a lenstronomy model list
        and the matching ``[init, sigma, fixed, lower, upper]`` list of kwargs
        lists. For a real lens this is not optional in practice: unmodelled
        galaxy light under the images pulls the fitted positions toward the
        galaxy centre, and that bias is common-mode, so it lands squarely in the
        direction the lens model is most sensitive to.
    mask:
        Optional 0/1 pixel mask (1 = use).
    supersampling_factor:
        Numerical supersampling of the image model.
    systematics:
        Common-mode budget added to the statistical covariance; see
        :class:`SystematicsBudget`. Defaults to zero, i.e. statistical only.
    pso_particles, pso_iterations:
        Particle-swarm settings for the global optimisation.
    polish_iterations:
        Nelder-Mead iterations run after the swarm. Do not set this to zero
        without checking what it costs you. A swarm that stops slightly short of
        the optimum scatters the reported position from fit to fit, and that
        scatter is invisible to the Hessian — which describes the curvature at
        wherever the optimiser happened to stop, not the distance from there to
        the true minimum. The result is a covariance that is too small by an
        amount no internal diagnostic reveals; it shows up only as a pull width
        above 1 in :func:`~gigalens_research.astrometry.validate.pull_test`.
    extra_fitting_steps:
        Appended to the fitting sequence after the PSO, in lenstronomy's
        ``[[name, kwargs], ...]`` form. The Hessian is always evaluated at
        whatever the final step returns.
    hessian_step_scale:
        Multiplier on the automatic finite-difference step. The result carries a
        ``hessian_step_stability`` diagnostic comparing the covariance at this
        step and at twice it; if that is not small, the step is wrong.
    verbose:
        Passed to lenstronomy.

    Returns
    -------
    AstrometryResult
        Positions, the interleaved ``(2n, 2n)`` covariance, and diagnostics.

    Notes
    -----
    The returned covariance is only a *claim*. Nothing in this function can
    check it — the Laplace approximation, the noise model, and the PSF are all
    assumptions, and a fit that is confidently wrong looks exactly like a fit
    that is right. Run :func:`gigalens_research.astrometry.validate.pull_test`
    before the numbers are used for science.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D; got shape {image.shape}.")
    ra0 = np.asarray(init_ra, dtype=float)
    dec0 = np.asarray(init_dec, dtype=float)
    if ra0.shape != dec0.shape or ra0.ndim != 1 or ra0.size == 0:
        raise ValueError(
            f"init_ra and init_dec must be matching 1-D sequences; got shapes "
            f"{ra0.shape} and {dec0.shape}."
        )
    n = ra0.size
    if float(search_radius) <= 0:
        raise ValueError("search_radius must be positive.")

    if n > 1:
        sep = np.hypot(ra0[:, None] - ra0[None, :], dec0[:, None] - dec0[None, :])
        min_sep = float(np.min(sep[~np.eye(n, dtype=bool)]))
        if min_sep <= 2.0 * search_radius:
            raise ValueError(
                f"search_radius={search_radius:g} is too large for the closest "
                f"image pair (separation {min_sep:.4g} arcsec): their search "
                f"boxes overlap, so the optimiser may swap the two images and "
                f"silently permute the rows of cov_img. Use "
                f"search_radius < {min_sep / 2:.4g}."
            )

    amp0 = (_default_amp_guess(image, frame, psf, ra0, dec0)
            if init_amp is None else np.asarray(init_amp, dtype=float))
    if amp0.shape != ra0.shape:
        raise ValueError(f"init_amp must have shape {ra0.shape}; got {amp0.shape}.")

    kwargs_data: Dict[str, Any] = {"image_data": image}
    kwargs_data.update(frame.to_kwargs())
    kwargs_data.update(noise.to_kwargs())

    kwargs_model: Dict[str, Any] = {"point_source_model_list": ["UNLENSED"]}
    kwargs_params: Dict[str, Any] = {
        "point_source_model": [
            [{"ra_image": ra0.copy(), "dec_image": dec0.copy(),
              "point_amp": amp0.copy()}],
            [{"ra_image": np.full(n, 0.1 * search_radius),
              "dec_image": np.full(n, 0.1 * search_radius),
              "point_amp": 0.3 * np.abs(amp0) + 1e-8}],
            [{}],
            [{"ra_image": ra0 - search_radius, "dec_image": dec0 - search_radius,
              "point_amp": np.full(n, float(amp_bounds[0]))}],
            [{"ra_image": ra0 + search_radius, "dec_image": dec0 + search_radius,
              "point_amp": np.full(n, float(amp_bounds[1]))}],
        ]
    }
    if lens_light_model_list:
        if kwargs_lens_light_params is None:
            raise ValueError(
                "lens_light_model_list was given without kwargs_lens_light_params; "
                "pass the matching [init, sigma, fixed, lower, upper] lists."
            )
        kwargs_model["lens_light_model_list"] = list(lens_light_model_list)
        kwargs_params["lens_light_model"] = kwargs_lens_light_params

    kwargs_numerics = {"supersampling_factor": int(supersampling_factor)}
    kwargs_data_joint = {
        "multi_band_list": [[kwargs_data, psf.to_kwargs(), kwargs_numerics]],
        "multi_band_type": "multi-linear",
    }
    # linear_solver=False keeps amplitudes as sampled parameters so the Hessian
    # marginalises over them; with the linear solver on they are profiled out and
    # the position errors come back too small wherever flux and position trade off.
    kwargs_constraints = {"linear_solver": False, "num_point_source_list": [n]}
    kwargs_likelihood: Dict[str, Any] = {"check_bounds": True}
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != image.shape:
            raise ValueError(
                f"mask shape {m.shape} does not match image shape {image.shape}."
            )
        kwargs_likelihood["image_likelihood_mask_list"] = [m.astype(float)]

    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints,
                                  kwargs_likelihood, kwargs_params, verbose=verbose)

    steps: List[Any] = [["PSO", {"sigma_scale": 1.0,
                                 "n_particles": int(pso_particles),
                                 "n_iterations": int(pso_iterations)}]]
    if int(polish_iterations) > 0:
        steps.append(["SIMPLEX", {"n_iterations": int(polish_iterations),
                                  "method": "Nelder-Mead"}])
    steps.extend(list(extra_fitting_steps))
    # lenstronomy's SIMPLEX step prints its progress regardless of `verbose`,
    # which buries the diagnostics under thousands of lines in an ensemble run.
    quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(
        io.StringIO())
    with quiet:
        fitting_seq.fit_sequence(steps)

    best = fitting_seq.best_fit()
    args = np.asarray(fitting_seq.param_class.kwargs2args(**best), dtype=float)
    order, block_order = _position_indices(fitting_seq.param_class)
    if order.size != 2 * n:
        raise RuntimeError(
            f"Expected {2 * n} free position parameters, found {order.size}."
        )

    _, names = fitting_seq.param_class.num_param()
    base = frame.pixel_scale * 0.02
    step = np.array([base if nm in ("ra_image", "dec_image")
                     else max(abs(args[i]), 1.0) * 1e-3
                     for i, nm in enumerate(names)], dtype=float)
    step *= float(hessian_step_scale)

    f = _neg_logl(fitting_seq)
    hess = _hessian(f, args, step)
    cov_stat = _marginal_position_covariance(hess, order)

    # Step-size stability: the covariance must not care which step we chose.
    hess2 = _hessian(f, args, step * 2.0)
    cov_stat2 = _marginal_position_covariance(hess2, order)
    d1, d2 = np.sqrt(np.diag(cov_stat)), np.sqrt(np.diag(cov_stat2))
    stability = float(np.max(np.abs(d2 - d1) / np.maximum(d1, 1e-30)))

    ps = best["kwargs_ps"][0]
    x_img = np.asarray(ps["ra_image"], dtype=float).copy()
    y_img = np.asarray(ps["dec_image"], dtype=float).copy()
    amp = np.asarray(ps["point_amp"], dtype=float).copy()

    cov_full = np.linalg.inv(hess)
    amp_idx = [i for i, nm in enumerate(names) if nm == "point_amp"]
    amp_err = (np.sqrt(np.maximum(np.diag(cov_full)[amp_idx], 0.0))
               if len(amp_idx) == n else np.full(n, np.nan))

    cov_sys = systematics.covariance(x_img, y_img)
    cov_img = cov_stat + cov_sys
    cov_img = 0.5 * (cov_img + cov_img.T)

    n_data = int(fitting_seq.likelihoodModule.effective_num_data_points(**best))
    chi2 = 2.0 * f(args)
    dof = max(n_data - args.size, 1)

    diagnostics: Dict[str, Any] = {
        "n_images": n,
        "n_free_params": int(args.size),
        "n_data_points": n_data,
        "logL": float(-f(args)),
        "reduced_chi2": float(chi2 / dof),
        "hessian_step_stability": stability,
        "hessian_condition": float(np.linalg.cond(hess)),
        "cov_condition": float(np.linalg.cond(cov_img)),
        "hessian_step": step.copy(),
        "param_names": list(names),
        "position_arg_indices_interleaved": order.copy(),
        "position_arg_indices_blocked": block_order.copy(),
        "systematics": dataclasses.asdict(systematics),
        "sys_fraction_of_total_variance": float(
            np.trace(cov_sys) / np.trace(cov_img)) if np.trace(cov_img) > 0 else 0.0,
    }
    if stability > 0.05:
        diagnostics["warning_hessian_step"] = (
            f"Covariance changed by {stability:.1%} when the finite-difference "
            f"step was doubled; the reported errors are not step-independent."
        )

    return AstrometryResult(
        x_img=x_img, y_img=y_img, cov_img=cov_img, cov_stat=cov_stat,
        cov_sys=cov_sys, amp=amp, amp_err=amp_err,
        kwargs_result=best, diagnostics=diagnostics,
    )
