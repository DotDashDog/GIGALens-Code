"""Point-source predictions from a :class:`Posterior`.

The imaging side of the view layer answers "what image does this model predict?"
(:meth:`Posterior.simulate`) and "how far is it from the data, in sigma?"
(:meth:`Posterior.err_map_at` + ``normalized_residual``). This module answers the same
two questions for a :mod:`gigalens.jax.point_source_position` observation, where the
prediction is ``2 n_images`` numbers produced by the term's own lens-equation solve
rather than a rendered array.

Two views, mirroring the imaging pair:

- :class:`PointSourcePrediction` — everything at ONE representative point (the analogue
  of a rendered model image + its residual map).
- :class:`PointSourceDraws` — the same quantities over thinned posterior draws (the
  posterior-predictive cloud, and the per-draw solver-health distributions).

Everything is computed through the term's OWN methods (``solve``,
``_newton_from_jacobian``, ``_delens_and_jacobian``). Nothing here re-implements the
lens equation: a second solver in the plotting layer would be a second thing to keep in
sync with the likelihood, and the failure it would hide (plots that disagree with the
scored chi2) is exactly what these panels exist to expose.

The chi2 decomposition
----------------------
``PointSourcePositionLikelihoodTerm`` scores

``chi2_i = |theta_obs - theta_hat|^2_C + sat(s) |s|^2_C   (+ |beta(theta_hat) - beta_src|^2 / sigma_anchor^2)``

and returns only the total. The three parts are what distinguish "this model fits" from
"this model cannot reproduce the images and is paying the bounded honesty charge to sit
on the saturation shelf" — a measured failure mode in which chains froze for a whole run
(``docs/logs/point-source-sbc.md``, C-2/P-4). So the parts are recomputed here from the
same solve, and — because a decomposition nobody checks is a decomposition that quietly
stops matching — :attr:`PointSourcePrediction.chi2_closes` reports whether they sum back
to the term's own scored chi2. The plotters surface a mismatch on the figure instead of
drawing a breakdown that no longer describes the likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .datasets import (
    KIND_POINT_SOURCE,
    KIND_POINT_SOURCE_LOSS,
    dataset_kind,
    image_position_covariances,
    isotropic_sigma,
    mahalanobis_sq,
    whiten,
)
from .params import mass_params_list

#: Default number of posterior draws behind a predictive cloud. Thinned from the full
#: chains, not subsampled at random: an even in-chain stride approximates independent
#: draws far better than a random pick from the pooled array, which preserves
#: autocorrelation clumps and makes a contour look tighter in places than it is.
DEFAULT_PREDICTIVE_DRAWS = 2000

#: Source-plane residual (arcsec) below which the lens-equation solve counts as
#: converged. Matches the campaign gate in
#: ``simtests/experiments/lenstronomy_point_source.py``.
SOLVER_CONVERGED_ARCSEC = 1e-4

#: Batch size for the draw-wise solve. The per-draw work is tiny (2 n_images
#: observables); this only bounds peak memory of the vmapped deflection evaluations.
_DRAW_CHUNK = 1000

#: Relative tolerance for "the recomputed chi2 parts sum to the term's chi2".
_CHI2_CLOSE_RTOL = 1e-6


def _squeeze_batch(arr: np.ndarray, batched: bool) -> np.ndarray:
    """Drop the trailing size-1 parameter-batch axis added by a single-point solve."""
    a = np.asarray(arr)
    return a[..., 0] if batched and a.ndim and a.shape[-1] == 1 else a


def _magnification(det: np.ndarray) -> np.ndarray:
    """Signed magnification ``1 / det A``, with the critical curve left as ``inf``.

    ``det A`` passes through zero at a critical curve, where the magnification is
    genuinely infinite; ``inf`` is the honest value and matplotlib simply drops it.
    Silencing the divide keeps a legitimate configuration from emitting a warning
    that reads like a bug.
    """
    det = np.asarray(det, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / det


@dataclass
class PointSourcePrediction:
    """Model prediction for one point-source observation at one representative point.

    Positions are arcsec, delays days, ``mu`` is the signed magnification
    ``1 / det A``. Per-image arrays are length ``n_images`` and aligned with the
    dataset's ``x_img`` / ``y_img`` ordering.
    """

    # -- the observation ----------------------------------------------------
    x_obs: np.ndarray                 #: (n,) observed image positions
    y_obs: np.ndarray                 #: (n,)
    cov: np.ndarray                   #: (n, 2, 2) astrometric covariances
    sigma_iso: np.ndarray             #: (n,) rms of the covariance principal axes

    # -- the prediction -----------------------------------------------------
    x_pred: np.ndarray                #: (n,) solved image positions theta_hat
    y_pred: np.ndarray                #: (n,)
    beta_x: float                     #: sampled source position
    beta_y: float
    beta_img_x: np.ndarray            #: (n,) delensed solved positions beta(theta_hat)
    beta_img_y: np.ndarray            #: (n,)
    mu: np.ndarray                    #: (n,) signed magnification at theta_hat

    # -- residuals ----------------------------------------------------------
    pulls: np.ndarray                 #: (n, 2) whitened position residuals

    # -- chi2 decomposition -------------------------------------------------
    chi2_displacement: np.ndarray     #: (n,) |theta_obs - theta_hat|^2_C
    chi2_honesty: np.ndarray          #: (n,) saturated first-order honesty charge
    chi2_anchor: Optional[np.ndarray]  #: (n,) source-plane anchor, or None if off
    chi2_flux: Optional[np.ndarray]   #: (n,) inverse-flux channel, or None
    chi2_td: Optional[np.ndarray]     #: (n-1,) time-delay channel, or None
    chi2_total: float                 #: the term's OWN scored chi2 (authoritative)
    event_size: int                   #: observables behind ``chi2_total``

    # -- solver health ------------------------------------------------------
    src_residual: np.ndarray          #: (n,) |beta(theta_hat) - beta_src|, arcsec
    trust_frac: np.ndarray            #: (n,) |theta_hat - theta_obs| / trust radius
    trust_radius: float               #: arcsec
    honesty_cap: np.ndarray           #: (n,) saturation scale of the honesty charge

    # -- optional channels --------------------------------------------------
    inv_flux_obs: Optional[np.ndarray] = None      #: (n,) 1/F observed
    inv_flux_pred: Optional[np.ndarray] = None     #: (n,) |det A|_eps / amp
    sigma_inv_flux: Optional[np.ndarray] = None    #: (n,) delta-method sigma on 1/F
    flux_obs: Optional[np.ndarray] = None          #: (n,) F observed (labels only)
    td_obs: Optional[np.ndarray] = None            #: (n,) observed delays, td[0] = 0
    td_pred: Optional[np.ndarray] = None           #: (n,) predicted delays
    sigma_td: Optional[np.ndarray] = None          #: (n-1,) delay sigmas

    # -- provenance ---------------------------------------------------------
    point_label: str = "median"
    n_images: int = 0

    @property
    def red_chi2(self) -> float:
        """The term's scored chi2 per observable."""
        return float(self.chi2_total) / max(int(self.event_size), 1)

    @property
    def chi2_parts_sum(self) -> float:
        """Sum of the recomputed parts, for comparison against :attr:`chi2_total`."""
        total = float(np.sum(self.chi2_displacement) + np.sum(self.chi2_honesty))
        for part in (self.chi2_anchor, self.chi2_flux, self.chi2_td):
            if part is not None:
                total += float(np.sum(part))
        return total

    @property
    def chi2_closes(self) -> bool:
        """Whether the recomputed decomposition reproduces the term's own chi2.

        ``False`` means the likelihood's scoring has moved on from what this module
        recomputes (a new channel, a changed whitening); the breakdown is then not a
        description of the scored chi2 and the plotters say so on the figure.
        """
        ref = abs(float(self.chi2_total))
        return abs(self.chi2_parts_sum - float(self.chi2_total)) <= _CHI2_CLOSE_RTOL * max(ref, 1.0)

    @property
    def converged(self) -> np.ndarray:
        """(n,) boolean: images whose solve reached the source plane."""
        return np.asarray(self.src_residual) <= SOLVER_CONVERGED_ARCSEC


@dataclass
class PointSourceDraws:
    """The same quantities over ``k`` thinned posterior draws.

    Per-image arrays are ``(n_images, k)``; per-draw arrays are ``(k,)``.
    """

    x_pred: np.ndarray                #: (n, k)
    y_pred: np.ndarray                #: (n, k)
    beta_x: np.ndarray                #: (k,)
    beta_y: np.ndarray                #: (k,)
    chi2: np.ndarray                  #: (k,) the term's scored chi2 per draw
    src_residual: np.ndarray          #: (n, k)
    trust_frac: np.ndarray            #: (n, k)
    mu: np.ndarray                    #: (n, k)
    chi2_displacement: np.ndarray     #: (n, k)
    chi2_honesty: np.ndarray          #: (n, k)
    chi2_anchor: Optional[np.ndarray]  #: (n, k) or None
    event_size: int
    n_draws: int
    n_images: int

    @property
    def red_chi2(self) -> np.ndarray:
        """(k,) scored chi2 per observable, per draw."""
        return np.asarray(self.chi2) / max(int(self.event_size), 1)

    @property
    def max_src_residual(self) -> np.ndarray:
        """(k,) worst per-draw source-plane residual — the campaign's health metric."""
        return np.max(np.asarray(self.src_residual), axis=0)

    @property
    def frac_unconverged(self) -> float:
        """Fraction of draws with any image failing the solve tolerance."""
        return float(np.mean(self.max_src_residual > SOLVER_CONVERGED_ARCSEC))

    @property
    def honesty_fraction(self) -> np.ndarray:
        """(k,) share of the position chi2 carried by the honesty charge.

        Near 0 the fit is a genuine astrometric residual; near 1 the draw cannot
        reproduce the images and is sitting on the saturated charge. This is the
        scalar the phantom-shelf failure shows up in.
        """
        disp = np.sum(np.asarray(self.chi2_displacement), axis=0)
        hon = np.sum(np.asarray(self.chi2_honesty), axis=0)
        total = disp + hon
        return np.where(total > 0, hon / np.where(total > 0, total, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Locating the term
# ---------------------------------------------------------------------------


def point_source_term(posterior, dataset: int):
    """The :class:`LikelihoodTerm` for point-source dataset index ``dataset``.

    ``ProbModel.terms`` is built one-per-dataset by ``zip(datasets, sees)``, so term
    and dataset indices agree — unlike ``ProbModel.simulators``, which is compacted to
    the imaging terms only.
    """
    prob = getattr(posterior.ctx, "prob_model", None)
    if prob is None:
        raise TypeError(
            "point-source predictions need a ProbModel: this is a forward-mode scene "
            "with no observed data, so there are no observed image positions to "
            "predict against.")
    datasets = list(getattr(prob, "datasets", []) or [])
    if not 0 <= dataset < len(datasets):
        raise IndexError(
            f"dataset index {dataset} out of range: the ProbModel has "
            f"{len(datasets)} dataset(s).")
    ds = datasets[dataset]
    kind = dataset_kind(ds)
    if kind == KIND_POINT_SOURCE_LOSS:
        raise TypeError(
            f"dataset {dataset} is a {type(ds).__name__} from gigalens.jax."
            "point_source — the three-term hand-weighted loss. Its 'chi2' is a "
            "weighted loss over a stand-in event count, not a calibrated Gaussian, so "
            "pulls and a reduced-chi2 drawn from it would claim a calibration that "
            "module does not provide. These plotters cover "
            "gigalens.jax.point_source_position (PointSourcePositionData / "
            "PointSourceObsData), whose chi2 is a genuine goodness-of-fit.")
    if kind != KIND_POINT_SOURCE:
        raise TypeError(
            f"dataset {dataset} is {type(ds).__name__} (kind {kind!r}), not a "
            "point-source observation.")
    term = prob.terms[dataset]
    if not hasattr(term, "solve"):
        raise TypeError(
            f"the likelihood term for dataset {dataset} ({type(term).__name__}) has no "
            "solve(); point-source predictions come from the term's own "
            "lens-equation solve.")
    return term


# ---------------------------------------------------------------------------
# The shared per-draw computation
# ---------------------------------------------------------------------------


def _solve_bundle(term, params) -> Dict[str, Any]:
    """Everything both views need, from ONE solve, as jax arrays.

    Batch dims ride inside ``params``, so this is shape-agnostic: leading axis
    ``n_images``, trailing axes the parameter batch.
    """
    from gigalens.jax.point_source_position import _delens_and_jacobian

    ds = term.dataset
    n = ds.n_images
    tx, ty, bsx, bsy = term.solve(params)
    mass_params = mass_params_list(term.model, params, term.lens_i,
                                   len(term.mass_profiles))
    (bx, by), jac = _delens_and_jacobian(term.mass_profiles, mass_params, tx, ty)
    sx, sy = term._newton_from_jacobian(bx, by, jac, bsx, bsy)

    bd = tx.ndim - 1
    xr = term.x.reshape((n,) + (1,) * bd)
    yr = term.y.reshape((n,) + (1,) * bd)

    a_xx, a_xy, a_yx, a_yy = jac
    det = a_xx * a_yy - a_xy * a_yx

    out = {
        "tx": tx, "ty": ty, "bsx": bsx, "bsy": bsy,
        "bx": bx, "by": by, "sx": sx, "sy": sy,
        "det": det,
        "dx": tx - xr, "dy": ty - yr,
        "src_residual": jnp.hypot(bx - bsx, by - bsy),
        "disp": jnp.hypot(tx - xr, ty - yr),
    }
    return out


def _chi2_parts_np(term, bundle: Dict[str, np.ndarray], cov: np.ndarray,
                   batched_shape: tuple):
    """Recompute the position chi2 parts in numpy from a solved bundle.

    Mirrors ``PointSourcePositionLikelihoodTerm._position_chi2`` with the
    per-coordinate whitening generalized to the full covariance: for a diagonal
    covariance ``mahalanobis_sq`` is exactly ``(dx/sig_x)^2 + (dy/sig_y)^2``, so this
    reproduces the current scoring identically and extends to correlated astrometry
    without a second formula.

    ``sat`` is computed from the UNWHITENED squared step length, as the term does —
    it is a rotation-invariant arcsec scale compared against the noise-scaled cap,
    not a whitened quantity.
    """
    res = np.stack([bundle["dx"], bundle["dy"]], axis=1)      # (n, 2, *batch)
    step = np.stack([bundle["sx"], bundle["sy"]], axis=1)
    disp = mahalanobis_sq(res, cov)                           # (n, *batch)
    step_m = mahalanobis_sq(step, cov)
    step_sq = bundle["sx"] ** 2 + bundle["sy"] ** 2
    cap = np.asarray(term.cap).reshape((-1,) + (1,) * (len(batched_shape)))
    sat = cap ** 2 / (cap ** 2 + step_sq)
    honesty = sat * step_m

    anchor = None
    if term.src_anchor_sigma is not None:
        anchor = ((bundle["bsx"] - bundle["bx"]) ** 2
                  + (bundle["bsy"] - bundle["by"]) ** 2) / term.src_anchor_sigma ** 2
    return disp, honesty, anchor


def _flux_and_td(term, params, bundle):
    """Predicted inverse fluxes and time delays, or ``(None, None, ...)`` if the
    dataset has no such channel. Mirrors ``PointSourceObsLikelihoodTerm.log_like``."""
    ds = term.dataset
    inv_flux_pred = chi2_flux = td_pred = chi2_td = None

    if getattr(ds, "has_flux", False):
        amp = term.model.component_params(
            params, term.src_i, "light", term.src_j)["amp"]
        det = bundle["det"]
        n = ds.n_images
        bd = det.ndim - 1
        sig_inv = np.asarray(term.sig_inv_flux).reshape((n,) + (1,) * bd)
        inv_f = np.asarray(term.inv_flux_obs).reshape((n,) + (1,) * bd)
        eps = term._FLUX_DET_SMOOTH * np.asarray(amp) * sig_inv
        det_sm = np.sqrt(np.asarray(det) ** 2 + eps ** 2)
        inv_flux_pred = det_sm / np.asarray(amp)
        chi2_flux = ((inv_f - inv_flux_pred) / sig_inv) ** 2

    if getattr(ds, "has_td", False):
        from gigalens.jax.point_source import _fermat_potential, time_delay_days
        mass_params = mass_params_list(term.model, params, term.lens_i,
                                       len(term.mass_profiles))
        fermat = _fermat_potential(term.mass_profiles, mass_params,
                                   bundle["tx"], bundle["ty"],
                                   bundle["bsx"], bundle["bsy"])
        td = time_delay_days(fermat, term.cosmo_profile, params["cosmo"],
                             term._comoving, term.z_lens, term.z_source)
        td = np.asarray(td - td[0:1])
        td_pred = td
        n = ds.n_images
        bd = td.ndim - 1
        td_obs = np.asarray(term.td_obs).reshape((n,) + (1,) * bd)
        sig_td = np.asarray(term.sig_td).reshape((n - 1,) + (1,) * bd)
        chi2_td = ((td[1:] - td_obs[1:]) / sig_td) ** 2

    return inv_flux_pred, chi2_flux, td_pred, chi2_td


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def point_source_prediction(posterior, *, dataset: int = 0,
                            point: str = "median") -> PointSourcePrediction:
    """Predictions for point-source ``dataset`` at one representative point."""
    term = point_source_term(posterior, dataset)
    ds = term.dataset
    params = posterior.params_at(point)
    bundle_j = _solve_bundle(term, params)
    chi2_total = float(np.asarray(term.log_like(params)[1]).reshape(-1)[0])

    # A single point still round-trips through the bijector, which works in batches,
    # so most quantities arrive with a trailing length-1 batch axis. Normalize once.
    probe = np.asarray(bundle_j["tx"])
    batched = probe.ndim == 2 and probe.shape[-1] == 1
    bundle = {k: _squeeze_batch(np.asarray(v), batched) for k, v in bundle_j.items()}

    cov = image_position_covariances(ds)
    disp, honesty, anchor = _chi2_parts_np(term, bundle, cov, ())

    inv_flux_pred, chi2_flux, td_pred, chi2_td = _flux_and_td(term, params, bundle_j)
    inv_flux_pred = _squeeze_batch(inv_flux_pred, batched) if inv_flux_pred is not None else None
    chi2_flux = _squeeze_batch(chi2_flux, batched) if chi2_flux is not None else None
    td_pred = _squeeze_batch(td_pred, batched) if td_pred is not None else None
    chi2_td = _squeeze_batch(chi2_td, batched) if chi2_td is not None else None

    # Pulls follow the package's imaging convention — OBSERVED minus predicted, as in
    # :func:`gigalens_research.plotting.image.normalized_residual` — so a pull chart and
    # a residual map are read the same way. ``dx`` is the term's own (predicted minus
    # observed) displacement, which the chi2 squares and the sign of never reaches.
    res_obs = np.stack([-bundle["dx"], -bundle["dy"]], axis=1)      # (n, 2)
    pulls = whiten(res_obs, cov)

    return PointSourcePrediction(
        x_obs=np.asarray(ds.x_img), y_obs=np.asarray(ds.y_img),
        cov=cov, sigma_iso=isotropic_sigma(cov),
        x_pred=bundle["tx"], y_pred=bundle["ty"],
        beta_x=float(np.asarray(bundle["bsx"]).reshape(-1)[0]),
        beta_y=float(np.asarray(bundle["bsy"]).reshape(-1)[0]),
        beta_img_x=bundle["bx"], beta_img_y=bundle["by"],
        mu=_magnification(bundle["det"]),
        pulls=pulls,
        chi2_displacement=disp, chi2_honesty=honesty, chi2_anchor=anchor,
        chi2_flux=chi2_flux, chi2_td=chi2_td,
        chi2_total=chi2_total, event_size=int(term.event_size),
        src_residual=bundle["src_residual"],
        trust_frac=bundle["disp"] / float(ds.trust_radius_arcsec),
        trust_radius=float(ds.trust_radius_arcsec),
        honesty_cap=np.asarray(ds.honesty_cap_arcsec),
        inv_flux_obs=(np.asarray(ds.inv_flux_obs)
                      if getattr(ds, "has_flux", False) else None),
        inv_flux_pred=inv_flux_pred,
        sigma_inv_flux=(np.asarray(ds.sigma_inv_flux)
                        if getattr(ds, "has_flux", False) else None),
        flux_obs=(np.asarray(ds.flux_obs) if getattr(ds, "has_flux", False) else None),
        td_obs=(np.asarray(ds.td_obs) if getattr(ds, "has_td", False) else None),
        td_pred=td_pred,
        sigma_td=(np.asarray(ds.sigma_td) if getattr(ds, "has_td", False) else None),
        point_label=posterior.point_label(point),
        n_images=int(ds.n_images),
    )


def thin_draws_z(posterior, n_draws: int) -> Optional[np.ndarray]:
    """``(k, n_params)`` thinned draws in z-space, or ``None`` for a point estimate.

    Chains are thinned by an even in-chain stride (not a random pick from the pooled
    array), which approximates independent draws far better: a random subsample keeps
    autocorrelation clumps, and a contour drawn from clumped draws reads tighter in
    places than the posterior actually is. Non-chain posteriors (e.g. an SVI
    surrogate) have no autocorrelation to thin, so their own draw mechanism is used.
    """
    if n_draws <= 0:
        raise ValueError(f"n_draws must be positive; got {n_draws}.")
    if hasattr(posterior, "samples_z"):
        sz = np.asarray(posterior.samples_z)
        n_chains, n_steps, n_params = sz.shape
        per_chain = min(n_steps, max(1, int(math.ceil(n_draws / n_chains))))
        idx = np.unique(np.linspace(0, n_steps - 1, num=per_chain).round().astype(int))
        thin = sz[:, idx, :].reshape(-1, n_params)
        return thin[np.isfinite(thin).all(axis=1)]
    if hasattr(posterior, "flat_z"):
        flat = np.asarray(posterior.flat_z)
        if flat.shape[0] > n_draws:
            rng = np.random.default_rng(getattr(posterior, "seed", 0))
            flat = flat[rng.choice(flat.shape[0], size=n_draws, replace=False)]
        return flat[np.isfinite(flat).all(axis=1)]
    return None


def point_source_draws(posterior, *, dataset: int = 0,
                       n_draws: int = DEFAULT_PREDICTIVE_DRAWS,
                       ) -> Optional[PointSourceDraws]:
    """Posterior-predictive point-source quantities over thinned draws.

    Returns ``None`` for a posterior with no draw distribution (a MAP point
    estimate), which is what the plotters branch on to fall back from contours to a
    single marked prediction.
    """
    term = point_source_term(posterior, dataset)
    ds = term.dataset
    z = thin_draws_z(posterior, n_draws)
    if z is None or z.shape[0] == 0:
        return None

    model = term.model
    prob = posterior.ctx.prob_model

    @jax.jit
    def _batch(zb):
        params = model.to_params(prob.bij.forward(zb))
        bundle = _solve_bundle(term, params)
        _, chi2 = term.log_like(params)
        return bundle, chi2

    parts: List[Dict[str, np.ndarray]] = []
    chi2s: List[np.ndarray] = []
    for start in range(0, z.shape[0], _DRAW_CHUNK):
        bundle, chi2 = _batch(jnp.asarray(z[start:start + _DRAW_CHUNK]))
        parts.append({k: np.asarray(v) for k, v in bundle.items()})
        chi2s.append(np.asarray(chi2).reshape(-1))

    bundle = {k: np.concatenate([p[k] for p in parts], axis=-1) for k in parts[0]}
    chi2 = np.concatenate(chi2s)

    cov = image_position_covariances(ds)
    disp, honesty, anchor = _chi2_parts_np(term, bundle, cov, (chi2.shape[0],))

    return PointSourceDraws(
        x_pred=bundle["tx"], y_pred=bundle["ty"],
        beta_x=np.asarray(bundle["bsx"]).reshape(-1),
        beta_y=np.asarray(bundle["bsy"]).reshape(-1),
        chi2=chi2,
        src_residual=bundle["src_residual"],
        trust_frac=bundle["disp"] / float(ds.trust_radius_arcsec),
        mu=_magnification(bundle["det"]),
        chi2_displacement=disp, chi2_honesty=honesty, chi2_anchor=anchor,
        event_size=int(term.event_size),
        n_draws=int(chi2.shape[0]), n_images=int(ds.n_images),
    )
