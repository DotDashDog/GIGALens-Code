"""Point-source inference plotting primitives.

Pure plotters in the style of :mod:`.image`: a
:class:`~gigalens_research.inference_utils.point_source.PointSourcePrediction`
(and optionally a :class:`~gigalens_research.inference_utils.point_source.\
PointSourceDraws`) in, matplotlib axes drawn. The compound panel in :mod:`.reports`
composes them; :mod:`gigalens_research.inference_utils.point_source` provides the
arrays.

What each plot is for
---------------------
The imaging row is ``observed | model | normalized residual | residual histogram``.
Positions get the same shape, with two structural differences forced by the data:

- **Scale.** Image separations are ~arcsec while astrometric sigmas are ~0.005-0.05",
  so a single axes showing all N images cannot show a residual. :func:`plot_positions`
  draws the configuration; :func:`plot_position_zoom` draws one image at the noise
  scale, and the report gives every image its own.
- **Count.** A quad is 8 numbers. :func:`plot_position_pulls` shows all of them
  individually; there is deliberately no histogram-with-Gaussian-fit companion, because
  a normality test on 8 points measures nothing and would invite reading a KS p-value
  that carries no information.

Beyond the analogue, three plots exist for failure modes that have no imaging
counterpart: :func:`plot_chi2_decomposition` (is this chi2 an astrometric residual, or
a saturated honesty charge from a model that cannot reproduce the images?),
:func:`plot_solver_health` (do the solves in the posterior actually reach the source
plane?) and :func:`plot_trust_occupancy` (are the iterates pinned on the trust-region
boundary, i.e. is the likelihood reporting a wall rather than a fit?).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse

from ..inference_utils.datasets import covariance_ellipse
from ..inference_utils.point_source import SOLVER_CONVERGED_ARCSEC

__all__ = [
    "plot_positions",
    "plot_position_zoom",
    "plot_position_pulls",
    "plot_source_position",
    "plot_chi2_decomposition",
    "plot_solver_health",
    "plot_trust_occupancy",
    "plot_magnifications",
    "plot_flux_channel",
    "plot_time_delay_channel",
]

#: Contour levels for the posterior-predictive clouds, as enclosed-probability of a
#: 2-D Gaussian at 1 and 2 sigma. Matches how ``corner`` states its own levels, so a
#: predictive cloud and a corner panel in the same report mean the same thing.
_CLOUD_LEVELS = (1.0 - np.exp(-0.5), 1.0 - np.exp(-2.0))

_OBS_COLOR = "black"
_PRED_COLOR = "crimson"
_CLOUD_COLOR = "steelblue"


def _draw_cloud(ax: Axes, xs: np.ndarray, ys: np.ndarray, *, color: str = _CLOUD_COLOR,
                label: Optional[str] = None) -> bool:
    """Posterior-predictive cloud as filled 1/2-sigma contours, corner-style.

    Delegates to ``corner.hist2d`` — the same renderer behind :func:`plot_corner` — so
    a predictive cloud and a corner panel are the same estimator with the same level
    convention, not two different smoothings that happen to sit in one report.

    Returns ``False`` (drawing nothing) when the sample is too small or too degenerate
    for a density estimate, so callers can fall back to plain markers instead of
    showing a contour built from a handful of points.
    """
    import corner as _corner_pkg

    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    good = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[good], ys[good]
    if xs.size < 50:
        return False
    if not (np.ptp(xs) > 0 and np.ptp(ys) > 0):
        return False

    _corner_pkg.hist2d(
        xs, ys, ax=ax, new_fig=False, levels=_CLOUD_LEVELS, smooth=1.0,
        color=color, plot_density=False, plot_datapoints=True, fill_contours=True,
        contour_kwargs={"linewidths": 0.8},
        data_kwargs={"alpha": 0.12, "ms": 1.5, "color": color},
    )
    if label:
        ax.plot([], [], color=color, lw=3, alpha=0.6, label=label)
    return True


def _square_frame(ax: Axes, xs: np.ndarray, ys: np.ndarray, *, pad: float = 1.25,
                  include_existing: bool = True) -> None:
    """Give ``ax`` a padded SQUARE frame and true equal aspect.

    Equal aspect with ``adjustable="datalim"`` makes matplotlib discard limits set
    before it ("Ignoring fixed x limits...") and reframe on the data, silently
    dropping the padding; a square range plus ``adjustable="box"`` gets equal aspect
    without that fight.

    ``include_existing`` unions in whatever is already on the axes — the caller draws
    curves (critical curves, caustics) first, so the frame ends up covering them
    instead of cropping a caustic that ran outside the marker bounding box.
    """
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    lo_x, hi_x = (xs.min(), xs.max()) if xs.size else (0.0, 0.0)
    lo_y, hi_y = (ys.min(), ys.max()) if ys.size else (0.0, 0.0)

    if include_existing:
        d = ax.dataLim
        if np.isfinite([d.x0, d.x1, d.y0, d.y1]).all() and d.width >= 0:
            lo_x, hi_x = min(lo_x, d.x0), max(hi_x, d.x1)
            lo_y, hi_y = min(lo_y, d.y0), max(hi_y, d.y1)

    span = max(hi_x - lo_x, hi_y - lo_y, 1e-9) * pad
    cx, cy = 0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y)
    ax.set_xlim(cx - 0.5 * span, cx + 0.5 * span)
    ax.set_ylim(cy - 0.5 * span, cy + 0.5 * span)
    ax.set_aspect("equal", adjustable="box")


def _is_diagonal(cov: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether every per-image covariance is diagonal (uncorrelated x/y)."""
    cov = np.asarray(cov, dtype=float)
    off = np.abs(cov[:, 0, 1])
    scale = np.sqrt(np.abs(cov[:, 0, 0] * cov[:, 1, 1]))
    return bool(np.all(off <= tol * np.maximum(scale, 1e-300)))


# ---------------------------------------------------------------------------
# Image plane
# ---------------------------------------------------------------------------


def plot_positions(
    ax: Axes,
    pred,
    *,
    draws=None,
    title: Optional[str] = None,
    with_labels: bool = True,
    sigma_scale: float = 1.0,
) -> None:
    """Image-plane configuration: observed vs predicted image positions.

    The system-scale view. Error ellipses are drawn at their true size, which for
    real astrometry is far below a printed point — that is the honest rendering, and
    the reason :func:`plot_position_zoom` exists. ``sigma_scale`` inflates them purely
    for visibility on this overview; it is annotated on the axes whenever it is not 1,
    so an inflated ellipse can never be mistaken for the measurement.

    Critical curves are the caller's job (``plot_critical_curves`` from
    :mod:`.source_plane`), since they need the posterior rather than the prediction.
    """
    x_obs, y_obs = np.asarray(pred.x_obs), np.asarray(pred.y_obs)

    if draws is not None:
        for i in range(pred.n_images):
            _draw_cloud(ax, draws.x_pred[i], draws.y_pred[i],
                        label="posterior predictive" if i == 0 else None)

    for i in range(pred.n_images):
        w, h, angle = covariance_ellipse(pred.cov[i], n_sigma=sigma_scale)
        ax.add_patch(Ellipse((x_obs[i], y_obs[i]), w, h, angle=angle,
                             facecolor="none", edgecolor=_OBS_COLOR, lw=0.8))

    ax.plot(x_obs, y_obs, "o", mfc="none", mec=_OBS_COLOR, ms=7, mew=1.2,
            linestyle="none", label="observed")
    ax.plot(pred.x_pred, pred.y_pred, "x", color=_PRED_COLOR, ms=7, mew=1.5,
            linestyle="none", label=f"model ({pred.point_label})")

    if with_labels:
        for i in range(pred.n_images):
            ax.annotate(str(i), (x_obs[i], y_obs[i]),
                        textcoords="offset points", xytext=(7, 7), fontsize=8)

    _square_frame(ax, x_obs, y_obs)
    ax.set_xlabel(r'$x$ ["]')
    ax.set_ylabel(r'$y$ ["]')
    if sigma_scale != 1.0:
        ax.text(0.02, 0.02, f"error ellipses x{sigma_scale:g}", transform=ax.transAxes,
                fontsize=7, color=_OBS_COLOR, alpha=0.8)
    ax.set_title(title if title is not None else "Image positions", fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)


def plot_position_zoom(
    ax: Axes,
    pred,
    image: int,
    *,
    draws=None,
    min_sigma_frame: float = 3.0,
    draw_quantile: float = 0.95,
) -> None:
    """One image at the noise scale: error ellipses, prediction, predictive cloud.

    Coordinates are offsets from the OBSERVED position in milliarcsec, so every zoom
    panel in a report shares one interpretable scale regardless of where its image
    sits on the sky.

    The frame is the largest of ``min_sigma_frame`` sigma and the requested quantile
    of the predictive cloud, capped at the solver's trust radius — the hard bound the
    iterates cannot cross, so a frame beyond it would show empty space by construction.
    """
    mas = 1e3
    x0, y0 = float(pred.x_obs[image]), float(pred.y_obs[image])
    sig = float(pred.sigma_iso[image])

    half = min_sigma_frame * sig
    if draws is not None:
        dx = np.asarray(draws.x_pred[image]) - x0
        dy = np.asarray(draws.y_pred[image]) - y0
        r = np.hypot(dx, dy)
        r = r[np.isfinite(r)]
        if r.size:
            half = max(half, float(np.quantile(r, draw_quantile)) * 1.2)
    half = min(half, float(pred.trust_radius) * 1.05)
    half = max(half, 1e-12)

    if draws is not None:
        _draw_cloud(ax, (np.asarray(draws.x_pred[image]) - x0) * mas,
                    (np.asarray(draws.y_pred[image]) - y0) * mas)

    for n_sig, alpha in ((1.0, 1.0), (2.0, 0.5)):
        w, h, angle = covariance_ellipse(pred.cov[image], n_sigma=n_sig)
        ax.add_patch(Ellipse((0.0, 0.0), w * mas, h * mas, angle=angle,
                             facecolor="none", edgecolor=_OBS_COLOR,
                             lw=1.0, alpha=alpha))

    # The trust boundary, when the frame actually reaches it.
    if float(pred.trust_radius) <= half:
        ax.add_patch(Circle((0.0, 0.0), float(pred.trust_radius) * mas,
                            facecolor="none", edgecolor="darkorange",
                            lw=0.8, linestyle=":"))

    ax.plot([0.0], [0.0], "o", mfc="none", mec=_OBS_COLOR, ms=6, mew=1.2)
    ax.plot([(float(pred.x_pred[image]) - x0) * mas],
            [(float(pred.y_pred[image]) - y0) * mas],
            "x", color=_PRED_COLOR, ms=7, mew=1.5)

    lim = half * mas
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"image {image}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlabel(r"$\Delta x$ [mas]", fontsize=7)
    ax.set_ylabel(r"$\Delta y$ [mas]", fontsize=7)


def plot_position_pulls(ax: Axes, pred, *, title: Optional[str] = None) -> None:
    """Whitened position residuals, one bar per observable.

    ``2 n_images`` numbers — all of them, individually. For a diagonal covariance the
    two components per image are the familiar x and y pulls; for correlated astrometry
    they are the Cholesky-whitened components (uncorrelated, unit variance, squares
    summing to the image's Mahalanobis chi2), and the axis says so rather than
    labelling a rotated quantity 'x'.
    """
    pulls = np.asarray(pred.pulls)
    n = pred.n_images
    diagonal = _is_diagonal(pred.cov)
    comp_names = ("x", "y") if diagonal else (r"$w_1$", r"$w_2$")

    positions = np.arange(2 * n, dtype=float)
    values = pulls.reshape(-1)
    colors = [_PRED_COLOR if k % 2 else _CLOUD_COLOR for k in range(2 * n)]

    for band, alpha in ((2.0, 0.08), (1.0, 0.12)):
        ax.axhspan(-band, band, color="grey", alpha=alpha, lw=0)
    ax.axhline(0.0, color="black", lw=0.8)

    ax.bar(positions, values, width=0.7, color=colors, edgecolor="none")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{i}{c}" for i in range(n) for c in comp_names], fontsize=7)
    ax.set_xlabel("image / component", fontsize=8)
    ax.set_ylabel("whitened residual [$\\sigma$]", fontsize=8)

    lim = max(3.0, float(np.max(np.abs(values))) * 1.15) if values.size else 3.0
    ax.set_ylim(-lim, lim)
    if title is None:
        title = "Position pulls"
        if not diagonal:
            title += " (Cholesky-whitened)"
    ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# Source plane
# ---------------------------------------------------------------------------


def plot_source_position(
    ax: Axes,
    pred,
    *,
    draws=None,
    title: Optional[str] = None,
) -> None:
    """Source plane: sampled source position, and where each image delenses to.

    Two things decide a point-source fit and both are visible here: whether the source
    sits inside the caustic (its image multiplicity), and how tightly the solved image
    positions delens back onto one point. The spread of the ``beta(theta_hat_i)``
    markers around ``beta_src`` IS the source-plane residual that
    :func:`plot_solver_health` reduces to a distribution — at convergence they are one
    marker.

    Caustics are the caller's job (:func:`.source_plane.plot_caustics`), which needs
    the posterior and an explicit window.
    """
    if draws is not None:
        _draw_cloud(ax, draws.beta_x, draws.beta_y, label="posterior predictive")

    ax.plot(pred.beta_img_x, pred.beta_img_y, "s", mfc="none", mec="darkorange",
            ms=6, mew=1.0, linestyle="none",
            label=r"$\beta(\hat{\theta}_i)$ (delensed images)")
    ax.plot([pred.beta_x], [pred.beta_y], "*", color=_PRED_COLOR, ms=13,
            linestyle="none", label=r"$\beta_{\rm src}$")

    _square_frame(ax, np.append(np.asarray(pred.beta_img_x), pred.beta_x),
                  np.append(np.asarray(pred.beta_img_y), pred.beta_y))
    ax.set_xlabel(r'$\beta_x$ ["]')
    ax.set_ylabel(r'$\beta_y$ ["]')
    ax.set_title(title if title is not None else "Source plane", fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)


# ---------------------------------------------------------------------------
# chi2 decomposition
# ---------------------------------------------------------------------------


def plot_chi2_decomposition(ax: Axes, pred, *, draws=None,
                            title: Optional[str] = None, log: bool = True) -> None:
    """Per-image chi2, split into the terms the likelihood actually sums.

    The distinction this panel exists to make: a **displacement** chi2 is an
    astrometric residual — the model reproduces the image and misses it by so many
    sigma. A **honesty** chi2 is the saturated first-order charge for an image the
    model cannot reproduce at all; it is bounded (~``(cap/sigma)^2`` per image, the
    dashed ceiling), so a fit resting on it has a finite, flat, and therefore
    sampler-trappable chi2 that a total-only reduced chi2 cannot distinguish from a
    poor-but-genuine fit. Chains have frozen there for an entire run.

    A bar dominated by the anchor term means the same thing with the source-plane
    anchor switched on: the solve is not reaching the source plane.

    The bars are the decomposition at ONE representative point, which cannot say
    whether unconverged solves dominate the *posterior* — a median draw can look
    healthy while most of the posterior mass does not. Pass ``draws`` and the panel
    also reports the share of the position chi2 that the honesty charge carries across
    the posterior, which is the question that matters when deciding whether a fit is
    real.

    The recomputed parts are checked against the term's own scored chi2; a mismatch is
    stated on the axes rather than silently drawn over.
    """
    n = pred.n_images
    idx = np.arange(n)
    segments = [("displacement", np.asarray(pred.chi2_displacement), _CLOUD_COLOR),
                ("honesty (saturated)", np.asarray(pred.chi2_honesty), "darkorange")]
    if pred.chi2_anchor is not None:
        segments.append(("source anchor", np.asarray(pred.chi2_anchor), "seagreen"))
    if pred.chi2_flux is not None:
        segments.append(("inverse flux", np.asarray(pred.chi2_flux), "purple"))

    bottom = np.zeros(n)
    for name, vals, color in segments:
        vals = np.nan_to_num(np.asarray(vals, dtype=float).reshape(n), nan=0.0)
        ax.bar(idx, vals, bottom=bottom, width=0.65, label=name, color=color,
               edgecolor="none")
        bottom = bottom + vals

    # Asymptotic ceiling of the saturated honesty charge, per image: as |s| grows,
    # sat*|s|^2 -> (cap/sigma)^2. Derived from the dataset's own cap and sigma rather
    # than hardcoded, so it stays correct if the cap convention changes.
    with np.errstate(divide="ignore", invalid="ignore"):
        ceiling = (np.asarray(pred.honesty_cap) / np.asarray(pred.sigma_iso)) ** 2
    ceiling_val = None
    if np.all(np.isfinite(ceiling)) and np.ptp(ceiling) <= 1e-6 * max(ceiling.max(), 1.0):
        ceiling_val = float(ceiling[0])
        ax.axhline(ceiling_val, color="darkorange", lw=0.9, ls="--",
                   label="saturated-charge ceiling")

    if log:
        ax.set_yscale("log")
        positive = bottom[bottom > 0]
        if positive.size:
            # The ceiling is included in the range on purpose: "how far below the
            # saturated charge is this fit?" is the comparison the panel is for, and a
            # legend entry for a line scrolled off the top of the axes is worse than
            # no line at all.
            hi = positive.max() if ceiling_val is None else max(positive.max(),
                                                                ceiling_val)
            ax.set_ylim(max(positive.min() * 0.2, 1e-8), hi * 3.0)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(i) for i in idx])
    ax.set_xlabel("image", fontsize=8)
    ax.set_ylabel(r"$\chi^2$ contribution", fontsize=8)

    if title is None:
        title = rf"$\chi^2$ decomposition ($\chi^2/\nu$ = {pred.red_chi2:.3g})"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    # Notes go along the BOTTOM: on a log axis the strip below the shortest bar is the
    # one region reliably free of both bars and the legend.
    notes = []
    if draws is not None:
        hf = np.asarray(draws.honesty_fraction, dtype=float)
        hf = hf[np.isfinite(hf)]
        if hf.size:
            notes.append((f"honesty $\\geq$50% of position $\\chi^2$ in "
                          f"{np.mean(hf > 0.5):.0%} of draws", "black"))
    if not pred.chi2_closes:
        notes.append((f"parts sum {pred.chi2_parts_sum:.6g} != term $\\chi^2$ "
                      f"{float(pred.chi2_total):.6g}", "red"))
    for row, (text, color) in enumerate(notes):
        ax.text(0.5, 0.02 + 0.07 * row, text, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.5, color=color,
                bbox={"facecolor": "white", "alpha": 0.85, "pad": 1.5, "lw": 0})


# ---------------------------------------------------------------------------
# Solver health (small panels)
# ---------------------------------------------------------------------------


def plot_solver_health(ax: Axes, draws, *, bins: int = 40,
                       title: Optional[str] = None) -> None:
    """Distribution of the per-draw source-plane residual, with the solve tolerance.

    Chains can mix beautifully while a fraction of draws never solve the lens
    equation, so this is an independent gate on the posterior rather than a summary of
    the fit: mass to the right of the line is posterior probability the likelihood
    scored at unconverged iterates.
    """
    r = np.asarray(draws.max_src_residual, dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        ax.text(0.5, 0.5, "no finite residuals", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        return
    lo = min(r.min(), SOLVER_CONVERGED_ARCSEC) * 0.5
    hi = max(r.max(), SOLVER_CONVERGED_ARCSEC) * 2.0
    ax.hist(r, bins=np.geomspace(lo, hi, bins), color=_CLOUD_COLOR)
    ax.axvline(SOLVER_CONVERGED_ARCSEC, color="red", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r'$\max_i |\beta(\hat{\theta}_i) - \beta_{\rm src}|$ ["]', fontsize=7)
    ax.set_ylabel("draws", fontsize=7)
    ax.tick_params(labelsize=6)
    frac = draws.frac_unconverged
    ax.set_title(title if title is not None
                 else f"solver: {frac:.1%} unconverged", fontsize=9)


def plot_trust_occupancy(ax: Axes, draws, *, bins: int = 40,
                         title: Optional[str] = None) -> None:
    """How close the solved iterates sit to their trust-region boundary.

    Every iterate is confined to ``trust_region_frac`` times the minimum image
    separation around its own observed seed. Mass piled against 1.0 means iterates are
    resting on that boundary: the likelihood there is reporting the edge of the
    solver's confinement, not a located root, and the residual it scores is an
    artifact of where the projection stopped.
    """
    f = np.asarray(draws.trust_frac, dtype=float).reshape(-1)
    f = f[np.isfinite(f)]
    if f.size == 0:
        ax.text(0.5, 0.5, "no finite iterates", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        return
    ax.hist(f, bins=np.linspace(0.0, max(1.05, float(f.max())), bins),
            color=_CLOUD_COLOR)
    ax.axvline(1.0, color="red", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\hat{\theta} - \theta_{\rm obs}|\,/\,R_{\rm trust}$", fontsize=7)
    ax.set_ylabel("image-draws", fontsize=7)
    ax.tick_params(labelsize=6)
    pinned = float(np.mean(f > 0.99))
    ax.set_title(title if title is not None
                 else f"trust region: {pinned:.1%} pinned", fontsize=9)


# ---------------------------------------------------------------------------
# Extra observable channels
# ---------------------------------------------------------------------------


def plot_magnifications(ax: Axes, pred, *, draws=None,
                        title: Optional[str] = None) -> None:
    """Signed magnification at the solved positions, per image.

    The quantity the flux channel is built on: ``1 / det A``. Sign is parity, so it is
    shown as ``|mu|`` on a log axis with the parity printed — a saddle image and a
    minimum of the same brightness are physically different and should not share a
    bar. Where draws are supplied, the spread is the posterior's, not a fit error.
    """
    n = pred.n_images
    idx = np.arange(n)
    mu = np.asarray(pred.mu, dtype=float)

    if draws is not None:
        data = [np.abs(np.asarray(draws.mu[i], dtype=float)) for i in range(n)]
        data = [d[np.isfinite(d) & (d > 0)] for d in data]
        if all(d.size > 1 for d in data):
            ax.violinplot(data, positions=idx, showextrema=False, widths=0.7)

    ax.plot(idx, np.abs(mu), "x", color=_PRED_COLOR, ms=8, mew=1.5, linestyle="none",
            label=f"model ({pred.point_label})")
    for i in range(n):
        if np.isfinite(mu[i]):
            ax.annotate("+" if mu[i] > 0 else "−", (idx[i], abs(mu[i])),
                        textcoords="offset points", xytext=(8, -3), fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xlabel("image", fontsize=8)
    ax.set_ylabel(r"$|\mu|$", fontsize=8)
    ax.set_title(title if title is not None else "Magnification (parity marked)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)


def plot_flux_channel(ax: Axes, pred, *, title: Optional[str] = None) -> None:
    """Flux channel in the observable the likelihood actually scores: ``1/F``.

    Plotting ``F`` would misrepresent both the residual and the error bar — the
    likelihood is Gaussian in inverse flux with the delta-method sigma
    ``sigma_F / F^2``, precisely because ``F`` has a pole at the critical curve while
    ``1/F = det A / amp`` passes through it linearly. So the panel is drawn where the
    Gaussian lives.
    """
    if pred.inv_flux_obs is None:
        raise ValueError(
            "this dataset has no flux channel (PointSourcePositionData, or "
            "PointSourceObsData built without flux_obs/sigma_flux); there is nothing "
            "to plot.")
    idx = np.arange(pred.n_images)
    ax.errorbar(idx, pred.inv_flux_obs, yerr=pred.sigma_inv_flux, fmt="o",
                mfc="none", mec=_OBS_COLOR, ecolor=_OBS_COLOR, ms=6,
                capsize=3, linestyle="none", label="observed")
    ax.plot(idx, pred.inv_flux_pred, "x", color=_PRED_COLOR, ms=8, mew=1.5,
            linestyle="none", label=f"model ({pred.point_label})")
    ax.set_xticks(idx)
    ax.set_xlabel("image", fontsize=8)
    ax.set_ylabel(r"$1/F$", fontsize=8)
    chi2 = None if pred.chi2_flux is None else float(np.sum(pred.chi2_flux))
    if title is None:
        title = "Flux channel" + ("" if chi2 is None else rf" ($\chi^2$ = {chi2:.3g})")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)


def plot_time_delay_channel(ax: Axes, pred, *, title: Optional[str] = None) -> None:
    """Relative time delays vs image 0, observed with error bars against the model.

    Image 0 is the reference and carries no observable, so it is not drawn. Delays are
    evaluated at the solved positions where ``grad Phi = 0``, which makes this channel
    first-order insensitive to astrometric noise.
    """
    if pred.td_obs is None:
        raise ValueError(
            "this dataset has no time-delay channel (built without td_obs/sigma_td); "
            "there is nothing to plot.")
    idx = np.arange(1, pred.n_images)
    ax.errorbar(idx, np.asarray(pred.td_obs)[1:], yerr=np.asarray(pred.sigma_td),
                fmt="o", mfc="none", mec=_OBS_COLOR, ecolor=_OBS_COLOR, ms=6,
                capsize=3, linestyle="none", label="observed")
    if pred.td_pred is not None:
        ax.plot(idx, np.asarray(pred.td_pred)[1:], "x", color=_PRED_COLOR, ms=8,
                mew=1.5, linestyle="none", label=f"model ({pred.point_label})")
    ax.set_xticks(idx)
    ax.set_xlabel("image (relative to image 0)", fontsize=8)
    ax.set_ylabel(r"$\Delta t$ [days]", fontsize=8)
    chi2 = None if pred.chi2_td is None else float(np.sum(pred.chi2_td))
    if title is None:
        title = "Time delays" + ("" if chi2 is None else rf" ($\chi^2$ = {chi2:.3g})")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)
