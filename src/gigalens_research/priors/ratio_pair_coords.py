"""Ratio-PAIR grouped prior over (Om0, w0) — two deflection ratios as coordinates.

Context (lab log ``docs/logs/sample-cosmology-dspl.md`` + the multiplane band
overlay, ``results/sample_cosmology/multiplane_band_overlay``): with a single
extra source plane the likelihood constrains ONE scalar r(Om0, w0) and its level
set is a rotating semi-infinite ridge — the triangular charts in
``ratio_coords.py`` (om-first Run C, u-first Run D) chart that ridge with one
data-stiff coordinate plus one leftover coordinate, and both inherit residual
disease from the leftover direction (bracket drift / support band). With >= 2
source planes at well-separated redshifts the likelihood constrains >= 2
deflection ratios whose contours cross (measured 10-16 deg for the 7-plane
z_lens=0.49 system), so BOTH sampling coordinates can be data-stiff ratios:

    z = (z1, z2)  ->  r_a = box_a(NormalCDF(z1)),  r_b = box_b(NormalCDF(z2))
                  ->  (Om0, w0) = the root of  (r_a_fn, r_b_fn)(Om0, w0) = (r_a, r_b)

where ``box_i`` squashes into the IMAGE bounding box of ratio i over the
(Om0, w0) prior box (computed at construction on a grid), and the root is found
by damped Newton in logit-theta space (gradients via the implicit function
theorem — never through the iteration). The inverse direction is closed-form
(two ratio evaluations). The Jacobian factors through the r-intermediate, so
both log-det directions are analytic:

    log|det dtheta/dz| = sum_i [ log(width_i) + log phi(z_i) ]
                       - log|det d(r_a, r_b)/d(Om0, w0)|

Exactness: the prior DENSITY over (Om0, w0) is the plain uniform box — a
diffeomorphism of sampling coordinates never changes the posterior. Two
quantified departures from a full-box diffeomorphism, both measured by
:func:`validate_ratio_pair` (raise-over-default, per repo discipline):

1. NO-PREIMAGE REGION (sampler wall). The image of the box under (r_a, r_b) is
   a curved region; its bounding box strictly contains it. z-points whose
   r-target falls in the bounding box but outside the image have no cosmology
   preimage: Newton cannot converge, and ``_forward_log_det_jacobian`` returns
   ``-inf`` there (the sampler's non-finite rejection treats it as zero
   density). The validator reports the no-preimage fraction of the sampling box
   (``r_box_empty_frac``) — sampling-efficiency cost, not a correctness cost.

2. DEGENERATE EDGES (chart collapse, intrinsic to ANY ratio pair — measured
   on the z_lens=0.49 7-plane system,
   ``experiments/sample_cosmology/ratio_pair_coverage.py``, 2026-07-17):

   * Om0 -> 1: dark energy vanishes and EVERY deflection ratio loses
     w0-sensitivity, so the w0-column of the Jacobian -> 0 and det -> 0 at
     the edge, crossing zero just inside it.
   * Om0 -> 0: matter vanishes and the GRADIENTS of all deflection ratios
     align (the same physics that makes every single-ratio constraint band
     near-vertical at low Om0), so det crosses zero along a fold curve at
     Om0 ~ 0.02-0.05 (all w0) and the map reverses orientation on the far
     side with LARGE opposite-sign det (measured |det_w| up to ~60 in the
     strip): a genuine fold, NOT coverable by a det tolerance.

   Consequently the FULL box Om0 in [0, 1] is not a diffeomorphism domain for
   any pair, and :func:`validate_ratio_pair` rejects it (det sign flip beyond
   any honest ``det_atol``). Use the chart on an Om0-TRIMMED box past both
   zero crossings (measured trims for the 7-plane system in the coverage
   script's summary JSON; (0.044, 0.994) certifies with det_atol=0 and
   float64 round-trip for all tested pairs). The trim is an explicit support
   amendment: quote the excluded prior strips with any run, and verify
   against YOUR posterior that they carry no mass (they do not, for data
   constraining ratios at sub-percent precision with contours far from both
   edges).

Status: UNCERTIFIED research machinery. Promote to
``gigalens.jax.grouped_priors`` only after certification.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens_research.priors.ratio_coords import deflection_ratio_u_fn

__all__ = [
    "RatioPairBijector",
    "RatioPairUniform",
    "deflection_ratio_pair_fn",
    "validate_ratio_pair",
]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_S_CLIP = 8.5          # |logit-theta| bound: ndtr saturates in float64 near 8.3
_BACKTRACK = (1.0, 0.3, 0.1)


def _log_phi(z):
    return -0.5 * jnp.square(z) - _LOG_SQRT_2PI


def deflection_ratio_pair_fn(
    cosmo,
    z_pair: Sequence[float],
    *,
    fixed: Dict[str, float],
) -> Callable:
    """Build ``r_pair_fn(Om0, w0) -> (r_a, r_b)`` from a cosmology profile.

    Each coordinate is the deflection ratio of one source plane (relative to
    ``cosmo.z_source_ref``), via :func:`deflection_ratio_u_fn` with weight 1.0
    so the map matches the scene's own ratios exactly.

    Args:
      cosmo: a ``CosmoBase`` profile (e.g. ``w0waCDM_Cosmo(z_lens, z_source_ref)``).
      z_pair: exactly two DISTINCT source-plane redshifts, neither equal to
        ``cosmo.z_source_ref`` (the reference ratio is identically 1 — not a
        coordinate). Prefer a well-separated pair: the chart's conditioning is
        the crossing angle of the two ratio contours (band-overlay measurement:
        low-z/high-z pairs cross at 10-16 deg for the 7-plane system, same-group
        pairs at < 4 deg).
      fixed: EVERY non-sampled parameter of ``cosmo``, explicitly (as in
        :func:`deflection_ratio_u_fn`).
    """
    if len(z_pair) != 2:
        raise ValueError(f"z_pair must have exactly 2 redshifts; got {z_pair!r}.")
    z_a, z_b = float(z_pair[0]), float(z_pair[1])
    if z_a == z_b:
        raise ValueError(f"z_pair redshifts must be distinct; got {z_pair!r}.")
    z_ref = float(np.asarray(cosmo.z_source_ref).reshape(()))
    for z_i in (z_a, z_b):
        if z_i == z_ref:
            raise ValueError(
                f"z_pair contains the reference redshift {z_ref}: its ratio is "
                "identically 1 and cannot serve as a coordinate.")
    r_a_fn = deflection_ratio_u_fn(cosmo, (z_a,), (1.0,), fixed=fixed)
    r_b_fn = deflection_ratio_u_fn(cosmo, (z_b,), (1.0,), fixed=fixed)

    def r_pair_fn(om0, w0):
        return r_a_fn(om0, w0), r_b_fn(om0, w0)

    return r_pair_fn


def _image_grids(r_pair_fn, om0_bounds, w0_bounds, n_grid):
    """Concrete (numpy) image of the box on an n x n grid: (om, w, ra, rb)."""
    om = jnp.linspace(om0_bounds[0], om0_bounds[1], int(n_grid))
    w = jnp.linspace(w0_bounds[0], w0_bounds[1], int(n_grid))
    om_mesh, w_mesh = jnp.meshgrid(om, w, indexing="ij")
    ra, rb = jnp.vectorize(r_pair_fn)(om_mesh, w_mesh)
    return (np.asarray(om_mesh), np.asarray(w_mesh),
            np.asarray(ra), np.asarray(rb))


def _make_theta_solver(
    r_pair_fn: Callable,
    om0_bounds: Tuple[float, float],
    w0_bounds: Tuple[float, float],
    *,
    n_newton: int,
    residual_rtol: float,
    init_table: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    r_scales: Tuple[float, float],
):
    """Scalar solver for ``r_pair_fn(theta) = (ra, rb)`` on the box.

    Damped Newton in logit-theta space s (theta = lo + (hi-lo) * ndtr(s), so
    iterates never leave the open box), seeded from the nearest entry of a
    precomputed (s, r) table, with a 3-point backtracking line search on the
    whitened residual norm. Wrapped in ``custom_vjp`` with implicit-function-
    theorem gradients: d theta = J^{-1} d r with J = d(ra, rb)/d(Om0, w0) at
    the root — never differentiate through the iteration. Gradients are ZEROED
    where the whitened residual exceeds ``residual_rtol`` (no root: the
    no-preimage region, or the degenerate Om0 -> 1 sliver), matching the
    ``-inf`` density the bijector assigns there.
    """
    om_lo, om_hi = float(om0_bounds[0]), float(om0_bounds[1])
    w_lo, w_hi = float(w0_bounds[0]), float(w0_bounds[1])
    widths = (om_hi - om_lo, w_hi - w_lo)
    s1_tab, s2_tab, ra_tab, rb_tab = (jnp.asarray(t) for t in init_table)
    sa, sb = float(r_scales[0]), float(r_scales[1])

    def _theta(s1, s2):
        return (om_lo + widths[0] * ndtr(s1), w_lo + widths[1] * ndtr(s2))

    def _res(s1, s2, ra, rb):
        r1, r2 = r_pair_fn(*_theta(s1, s2))
        return (r1 - ra) / sa, (r2 - rb) / sb

    def _newton(ra, rb):
        # Seed FROM the inputs (ra*0 + ...): under shard_map a plain-constant
        # fori_loop carry is unvarying while the body mixes in device-varying
        # inputs, and the loop rejects the mismatched carry types (see the
        # bisection-solver comment in ratio_coords.py).
        base = ra * 0 + rb * 0
        d2 = jnp.square((ra_tab - ra) / sa) + jnp.square((rb_tab - rb) / sb)
        k = jnp.argmin(d2)
        s1 = base + s1_tab[k]
        s2 = base + s2_tab[k]

        res_fn = lambda s: jnp.stack(_res(s[0], s[1], ra, rb))
        jac_fn = jax.jacfwd(res_fn)

        def body(_, s12):
            s = jnp.stack(s12)
            g = res_fn(s)
            J = jac_fn(s)
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            # Adjugate solve of the 2x2 system; guard the degenerate sliver
            # (det -> 0 at Om0 -> 1) so the step stays finite — the step is
            # then capped, and non-convergence there is handled by the
            # residual mask, not by the iteration.
            safe = jnp.where(jnp.abs(det) < 1e-300, 1e-300, det)
            d1 = -(J[1, 1] * g[0] - J[0, 1] * g[1]) / safe
            d2_ = -(-J[1, 0] * g[0] + J[0, 0] * g[1]) / safe
            step = jnp.stack([d1, d2_])
            norm = jnp.sqrt(jnp.sum(jnp.square(step)))
            step = step * jnp.minimum(1.0, 4.0 / jnp.maximum(norm, 1e-300))

            best_s, best_f = s, jnp.sum(jnp.square(g))
            for t in _BACKTRACK:
                cand = jnp.clip(s + t * step, -_S_CLIP, _S_CLIP)
                f = jnp.sum(jnp.square(res_fn(cand)))
                take = f < best_f
                best_s = jnp.where(take, cand, best_s)
                best_f = jnp.where(take, f, best_f)
            return best_s[0], best_s[1]

        s1, s2 = jax.lax.fori_loop(0, n_newton, body, (s1, s2))
        return _theta(s1, s2)

    @jax.custom_vjp
    def solve(ra, rb):
        return _newton(ra, rb)

    def fwd(ra, rb):
        om, w0 = _newton(ra, rb)
        return (om, w0), (om, w0, ra, rb)

    def bwd(saved, cot):
        om, w0, ra, rb = saved
        g_om, g_w0 = cot
        r_stacked = lambda th: jnp.stack(r_pair_fn(th[0], th[1]))
        J = jax.jacfwd(r_stacked)(jnp.stack([om, w0]))
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        safe = jnp.where(jnp.abs(det) < 1e-300, 1e-300, det)
        # g_r = J^{-T} g_theta (adjugate transpose over det).
        g_ra = (J[1, 1] * g_om - J[1, 0] * g_w0) / safe
        g_rb = (-J[0, 1] * g_om + J[0, 0] * g_w0) / safe
        r1, r2 = r_pair_fn(om, w0)
        res = jnp.maximum(jnp.abs((r1 - ra) / sa), jnp.abs((r2 - rb) / sb))
        ok = jnp.where(res <= residual_rtol, 1.0, 0.0)
        return g_ra * ok, g_rb * ok

    solve.defvjp(fwd, bwd)
    return solve


class RatioPairBijector(tfb.Bijector):
    """Diffeomorphism ``R^2 -> {theta in box reachable through the r-chart}``
    (module doc). Event ndims 1 (last axis is the (Om0, w0) 2-vector).
    ``_forward`` runs the Newton solve; ``_inverse`` is analytic; both log-det
    directions are analytic through the r-intermediate (no autodiff through the
    solve). ``_forward_log_det_jacobian`` is ``-inf`` where the r-target has no
    preimage (whitened residual > ``residual_rtol``)."""

    def __init__(
        self,
        r_pair_fn: Callable,
        om0_bounds: Tuple[float, float],
        w0_bounds: Tuple[float, float],
        *,
        n_newton: int = 40,
        n_image_grid: int = 201,
        residual_rtol: float = 1e-8,
        init_table_size: int = 25,
        validate_args: bool = False,
        name: str = "ratio_pair_coords",
    ):
        parameters = dict(locals())
        self._om_lo, self._om_hi = float(om0_bounds[0]), float(om0_bounds[1])
        self._w_lo, self._w_hi = float(w0_bounds[0]), float(w0_bounds[1])
        if not (self._om_hi > self._om_lo) or not (self._w_hi > self._w_lo):
            raise ValueError(
                f"bounds must be increasing; got om0={om0_bounds}, w0={w0_bounds}.")
        if int(n_newton) < 15:
            raise ValueError(
                f"n_newton={n_newton} < 15 cannot reach float64 residuals from "
                "a table init on this geometry; use >= 15.")
        self._r_pair_scalar = r_pair_fn
        self._r_pair = jnp.vectorize(r_pair_fn)
        self._residual_rtol = float(residual_rtol)

        # Image bounding box of each ratio over the theta-box (concrete, eager).
        _, _, ra_g, rb_g = _image_grids(
            r_pair_fn, om0_bounds, w0_bounds, n_image_grid)
        if not (np.isfinite(ra_g).all() and np.isfinite(rb_g).all()):
            raise ValueError(
                "r_pair_fn is non-finite on the construction grid; the "
                "ratio-pair map is unusable on this box.")
        self._ra_box = (float(ra_g.min()), float(ra_g.max()))
        self._rb_box = (float(rb_g.min()), float(rb_g.max()))
        wa = self._ra_box[1] - self._ra_box[0]
        wb = self._rb_box[1] - self._rb_box[0]
        if not (wa > 0.0 and wb > 0.0):
            raise ValueError(
                f"degenerate ratio image (widths {wa}, {wb}): a ratio is "
                "constant on the box and cannot serve as a coordinate.")
        self._r_widths = (wa, wb)

        # Newton init table: interior theta-grid (cell centers), stored in
        # logit-theta coordinates with the matching ratio values.
        k = int(init_table_size)
        cent = (np.arange(k) + 0.5) / k
        s1_1d = np.asarray(ndtri(cent))
        om_1d = self._om_lo + (self._om_hi - self._om_lo) * cent
        w_1d = self._w_lo + (self._w_hi - self._w_lo) * cent
        om_t, w_t = np.meshgrid(om_1d, w_1d, indexing="ij")
        ra_t, rb_t = (np.asarray(a) for a in
                      jnp.vectorize(r_pair_fn)(jnp.asarray(om_t), jnp.asarray(w_t)))
        s1_t, s2_t = np.meshgrid(s1_1d, s1_1d, indexing="ij")
        init_table = (s1_t.ravel(), s2_t.ravel(), ra_t.ravel(), rb_t.ravel())

        self._solve = jnp.vectorize(_make_theta_solver(
            r_pair_fn, om0_bounds, w0_bounds,
            n_newton=int(n_newton), residual_rtol=self._residual_rtol,
            init_table=init_table, r_scales=self._r_widths))

        def _logabsdet(om, w0):
            r_stacked = lambda th: jnp.stack(r_pair_fn(th[0], th[1]))
            J = jax.jacfwd(r_stacked)(jnp.stack([om, w0]))
            return jnp.log(jnp.abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))

        self._logabsdet = jnp.vectorize(_logabsdet)
        super().__init__(
            forward_min_event_ndims=1,
            is_constant_jacobian=False,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    # -- pieces ----------------------------------------------------------------
    def _r_from_z(self, z):
        z1, z2 = z[..., 0], z[..., 1]
        ra = self._ra_box[0] + self._r_widths[0] * ndtr(z1)
        rb = self._rb_box[0] + self._r_widths[1] * ndtr(z2)
        return ra, rb

    @staticmethod
    def _clip_unit(t):
        dt = jnp.result_type(t)
        return jnp.clip(t, jnp.finfo(dt).tiny, 1.0 - jnp.finfo(dt).epsneg)

    def _whitened_residual(self, om, w0, ra, rb):
        r1, r2 = self._r_pair(om, w0)
        return jnp.maximum(jnp.abs((r1 - ra) / self._r_widths[0]),
                           jnp.abs((r2 - rb) / self._r_widths[1]))

    def _fldj_from_parts(self, z1, z2, om, w0):
        return (
            math.log(self._r_widths[0]) + _log_phi(z1)
            + math.log(self._r_widths[1]) + _log_phi(z2)
            - self._logabsdet(om, w0)
        )

    # -- bijector interface ------------------------------------------------------
    def _forward(self, z):
        ra, rb = self._r_from_z(z)
        om, w0 = self._solve(ra, rb)
        return jnp.stack([om, w0], axis=-1)

    def _inverse(self, x):
        om, w0 = x[..., 0], x[..., 1]
        ra, rb = self._r_pair(om, w0)
        z1 = ndtri(self._clip_unit((ra - self._ra_box[0]) / self._r_widths[0]))
        z2 = ndtri(self._clip_unit((rb - self._rb_box[0]) / self._r_widths[1]))
        return jnp.stack([z1, z2], axis=-1)

    def _forward_log_det_jacobian(self, z):
        ra, rb = self._r_from_z(z)
        om, w0 = self._solve(ra, rb)
        fldj = self._fldj_from_parts(z[..., 0], z[..., 1], om, w0)
        res = self._whitened_residual(om, w0, ra, rb)
        return jnp.where(res <= self._residual_rtol, fldj, -jnp.inf)

    def _inverse_log_det_jacobian(self, x):
        om, w0 = x[..., 0], x[..., 1]
        z = self._inverse(x)
        return -self._fldj_from_parts(z[..., 0], z[..., 1], om, w0)


def validate_ratio_pair(
    r_pair_fn: Callable,
    om0_bounds: Tuple[float, float],
    w0_bounds: Tuple[float, float],
    *,
    det_atol: float = 0.0,
    roundtrip_atol: float = 0.0,
    n_grid: int = 201,
    n_roundtrip: int = 41,
    n_newton: int = 40,
    n_image_grid: int = 201,
) -> Dict[str, float]:
    """Grid-check that the ratio pair supports the chart; raise if it doesn't.

    On an ``n_grid x n_grid`` grid over the box (autodiff of the ACTUAL
    ``r_pair_fn``):

    1. ``r_pair_fn`` and its Jacobian finite everywhere (always fatal).
    2. The WHITENED Jacobian determinant (theta normalized by box widths, r by
       image widths — dimensionless, comparable across pairs) keeps one sign.
       Zero-crossings are tolerated only where ``|det_w| <= det_atol`` (the
       documented Om0 -> 1 degeneracy, and nothing else); any flip beyond the
       tolerance raises — the chart is folded and the pair choice is wrong
       (e.g. a near-parallel same-redshift-group pair).
    3. End-to-end round-trip ``forward(inverse(theta))`` through an actual
       :class:`RatioPairBijector` on an ``n_roundtrip`` grid: the max
       box-normalized theta error OUTSIDE the degenerate sliver
       (``|det_w| <= det_atol``) must be <= ``roundtrip_atol``.

    Strict 0.0 defaults: the caller must pass measured, explicitly derived
    tolerances (repo discipline — raise, never default). Reports the sliver
    extent, the no-preimage fraction of the sampling box
    (``r_box_empty_frac``), and the image boxes; attach to the model card.
    """
    om_lo, om_hi = float(om0_bounds[0]), float(om0_bounds[1])
    w_lo, w_hi = float(w0_bounds[0]), float(w0_bounds[1])
    if not (om_hi > om_lo) or not (w_hi > w_lo):
        raise ValueError(
            f"bounds must be increasing; got om0={om0_bounds}, w0={w0_bounds}.")

    om = jnp.linspace(om_lo, om_hi, int(n_grid))
    w = jnp.linspace(w_lo, w_hi, int(n_grid))
    om_mesh, w_mesh = jnp.meshgrid(om, w, indexing="ij")
    ra_g, rb_g = jnp.vectorize(r_pair_fn)(om_mesh, w_mesh)

    def _det(om_, w0_):
        r_stacked = lambda th: jnp.stack(r_pair_fn(th[0], th[1]))
        J = jax.jacfwd(r_stacked)(jnp.stack([om_, w0_]))
        return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    det = jnp.vectorize(_det)(om_mesh, w_mesh)
    finite = (bool(jnp.isfinite(ra_g).all()) and bool(jnp.isfinite(rb_g).all())
              and bool(jnp.isfinite(det).all()))
    if not finite:
        raise ValueError(
            "r_pair_fn or its Jacobian is non-finite on the validation grid; "
            "the ratio-pair map is unusable on this box.")

    wa = float(jnp.max(ra_g) - jnp.min(ra_g))
    wb = float(jnp.max(rb_g) - jnp.min(rb_g))
    det_w = det * (om_hi - om_lo) * (w_hi - w_lo) / (wa * wb)

    sign = 1.0 if float(jnp.mean(det_w)) >= 0.0 else -1.0
    signed = sign * det_w
    degen = jnp.abs(det_w) <= float(det_atol)
    n_flip = int(jnp.sum(signed < 0.0))
    n_flip_beyond = int(jnp.sum(signed < -float(det_atol)))
    degen_om = jnp.where(degen, om_mesh, jnp.nan)
    degen_om_min = float(jnp.nanmin(degen_om)) if bool(degen.any()) else float("nan")

    # No-preimage fraction of the r sampling box: bin the image into the
    # bounding box; empty cells have no theta preimage at grid resolution.
    hist, _, _ = np.histogram2d(
        np.asarray(ra_g).ravel(), np.asarray(rb_g).ravel(),
        bins=max(16, int(n_grid) // 4),
        range=[[float(jnp.min(ra_g)), float(jnp.max(ra_g))],
               [float(jnp.min(rb_g)), float(jnp.max(rb_g))]])
    r_box_empty_frac = float((hist == 0).mean())

    report = dict(
        det_sign=sign,
        min_signed_det_w=float(jnp.min(signed)),
        max_abs_det_w=float(jnp.max(jnp.abs(det_w))),
        n_grid_flips=n_flip,
        n_grid_flips_beyond_atol=n_flip_beyond,
        degen_frac=float(jnp.mean(degen)),
        degen_om0_min=degen_om_min,
        ra_box=(float(jnp.min(ra_g)), float(jnp.max(ra_g))),
        rb_box=(float(jnp.min(rb_g)), float(jnp.max(rb_g))),
        r_box_empty_frac=r_box_empty_frac,
        det_atol=float(det_atol),
        roundtrip_atol=float(roundtrip_atol),
        n_grid=int(n_grid),
    )

    if n_flip_beyond > 0:
        raise ValueError(
            f"ratio-pair Jacobian determinant changes sign beyond det_atol="
            f"{det_atol}: {n_flip_beyond} grid point(s), most-negative signed "
            f"whitened det {report['min_signed_det_w']:.3e}. The chart is "
            f"folded — this redshift pair cannot coordinate the box (pick a "
            f"pair with a larger contour crossing angle, or shrink the box). "
            f"Report: {report}")

    # End-to-end round-trip through an actual bijector (same construction path
    # the sampler will use).
    bij = RatioPairBijector(
        r_pair_fn, om0_bounds, w0_bounds,
        n_newton=n_newton, n_image_grid=n_image_grid)
    om_r = jnp.linspace(om_lo, om_hi, int(n_roundtrip))
    w_r = jnp.linspace(w_lo, w_hi, int(n_roundtrip))
    om_rm, w_rm = jnp.meshgrid(om_r, w_r, indexing="ij")
    theta = jnp.stack([om_rm, w_rm], axis=-1)
    theta_rt = bij.forward(bij.inverse(theta))
    err = jnp.maximum(
        jnp.abs(theta_rt[..., 0] - om_rm) / (om_hi - om_lo),
        jnp.abs(theta_rt[..., 1] - w_rm) / (w_hi - w_lo))
    det_w_rt = jnp.vectorize(_det)(om_rm, w_rm) * (
        (om_hi - om_lo) * (w_hi - w_lo) / (wa * wb))
    outside = jnp.abs(det_w_rt) > float(det_atol)
    err_outside = jnp.where(outside, err, 0.0)
    report["max_roundtrip_err"] = float(jnp.max(err))
    report["max_roundtrip_err_outside_degen"] = float(jnp.max(err_outside))
    report["n_roundtrip"] = int(n_roundtrip)

    if report["max_roundtrip_err_outside_degen"] > float(roundtrip_atol):
        raise ValueError(
            f"ratio-pair round-trip error "
            f"{report['max_roundtrip_err_outside_degen']:.3e} (box-normalized, "
            f"outside the |det_w|<={det_atol} sliver) exceeds roundtrip_atol="
            f"{roundtrip_atol}; the Newton chart does not reliably cover the "
            f"box for this pair. Report: {report}")
    return report


class RatioPairUniform(tfd.Independent):
    """Uniform prior on the (Om0, w0) box whose event-space bijector is the
    ratio-PAIR map (event_shape [2]); both sampling coordinates are data-stiff
    deflection ratios. Drop-in tuple-key usage, as RatioCoordsUniform::

        Component(w0waCDM_Cosmo(z_lens, z_source_ref), {
            "H0": 70.0, "k": 0.0, "wa": 0.0,
            ("Om0", "w0"): RatioPairUniform(
                deflection_ratio_pair_fn(cosmo, (z_a, z_b),
                                         fixed=dict(H0=70.0, k=0.0, wa=0.0)),
                (0.0, 1.0), (-2.0, -1.0 / 3.0),
                det_atol=..., roundtrip_atol=...),
        })

    The prior DENSITY is exactly the independent-uniform box; only the sampling
    coordinates differ. Two quantified caveats (module doc): the no-preimage
    region of the r sampling box is rejected via ``-inf`` forward log-det
    (report: ``r_box_empty_frac``), and the chart FOLDS at both Om0 extremes
    of the full physical range — pass an Om0-trimmed box past the two
    measured det-zero crossings ((0.044, 0.994) for the 7-plane system, 0.056 for the tightest same-group pair; the
    validator hard-rejects a box containing either fold).
    :func:`validate_ratio_pair` runs at construction with caller-supplied
    measured tolerances unless ``skip_validation=True`` (tests only); the
    report is attached as ``self.ratio_pair_report``."""

    def __init__(
        self,
        r_pair_fn: Callable,
        om0_bounds: Tuple[float, float],
        w0_bounds: Tuple[float, float],
        *,
        det_atol: float = 0.0,
        roundtrip_atol: float = 0.0,
        n_newton: int = 40,
        n_image_grid: int = 201,
        n_validation_grid: int = 201,
        n_roundtrip: int = 41,
        residual_rtol: float = 1e-8,
        skip_validation: bool = False,
        dtype=None,
        validate_args: bool = False,
        name: str = "RatioPairUniform",
    ):
        if skip_validation:
            self.ratio_pair_report = None
        else:
            self.ratio_pair_report = validate_ratio_pair(
                r_pair_fn, om0_bounds, w0_bounds,
                det_atol=det_atol, roundtrip_atol=roundtrip_atol,
                n_grid=n_validation_grid, n_roundtrip=n_roundtrip,
                n_newton=n_newton, n_image_grid=n_image_grid)
        self._esb = RatioPairBijector(
            r_pair_fn, om0_bounds, w0_bounds,
            n_newton=n_newton, n_image_grid=n_image_grid,
            residual_rtol=residual_rtol, validate_args=validate_args)
        if dtype is None:
            dtype = jnp.zeros(()).dtype  # ambient default float (float64 iff x64)
        low = jnp.asarray([om0_bounds[0], w0_bounds[0]], dtype)
        high = jnp.asarray([om0_bounds[1], w0_bounds[1]], dtype)
        super().__init__(
            tfd.Uniform(low=low, high=high),
            reinterpreted_batch_ndims=1,
            validate_args=validate_args,
            name=name,
        )

    def _default_event_space_bijector(self):
        return self._esb
