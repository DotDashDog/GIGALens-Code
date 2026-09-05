"""Ratio-pair + wa grouped prior over (Om0, w0, wa): two deflection ratios plus wa
as sampling coordinates.

Why (lab log ``docs/logs/ersatz-carousel-cosmology.md``): on a single-mass-plane
system with many source planes the likelihood sees cosmology only through the
deflection ratios r_i(Om0, w0, wa). Their Jacobian has singular values ~1 : 0.06 :
0.002 (ersatz carousel, 8 ratios), i.e. two data-stiff directions and one
likelihood-null direction along which the posterior is a long, CURVED filament
bounded by the prior box. In the gaussianized-box chart that filament rotates by
~50 deg, so one frozen MCLMC metric is misaligned with it almost everywhere: the
tuned metric's soft direction is locally stiff (integrator instability, energy
spikes) and chains stall where the filament meets the wa wall. Charting the
filament as a straight coordinate axis removes the rotation.

The chart::

    z = (z1, z2, z3)
      -> r_a = box_a(ndtr(z1)),  r_b = box_b(ndtr(z2)),  wa = lo_wa + w_wa * ndtr(z3)
      -> (Om0, w0) = root of (r_a_fn, r_b_fn)(Om0, w0; wa) = (r_a, r_b)

``box_i`` squashes into the IMAGE bounding box of ratio i over the whole 3-D prior
box; the root is found by damped Newton in logit-theta space at fixed wa (implicit-
function-theorem gradients, including d theta / d wa = -J^{-1} dr/dwa -- never
through the iteration). The (r_a, r_b, wa) -> (Om0, w0, wa) Jacobian is block
triangular, so the forward log-det is analytic::

    log|d theta / dz| = sum_i [log width_i + log phi(z_i)] - log|det d(r_a,r_b)/d(Om0,w0)|_wa

Exactness: the prior DENSITY stays the independent uniform box -- a change of
sampling coordinates never changes the posterior. Departures from a full-box
diffeomorphism, measured by :func:`validate_ratio_pair_wa` (raise-over-default):

1. NO-PREIMAGE REGION: r-targets inside the bounding box but outside the image
   have no root; ``_forward_log_det_jacobian`` returns ``-inf`` there (the sampler
   rejects). Burn-in cost only; MAP/truth init lands inside.
2. FOLDS: det d(r_a,r_b)/d(Om0,w0) crosses zero (a) near Om0 -> 0 (matter vanishes,
   ratio gradients align) and Om0 -> 1 (w0 unidentifiable) -- the same intrinsic
   edges as :mod:`ratio_pair_coords` -- so pass an Om0-TRIMMED box; and (b) for the
   ersatz-carousel pair (0.962, 4.090) a thin sliver at w0 > -0.67, Om0 < 0.23,
   wa ~ 0.2-0.4 (whitened |det| <= 2e-5). The validator reports the sliver's
   extent and the caller must pass a measured ``det_atol`` covering it; quote the
   sliver with any run and check the posterior carries no mass there.

Status: UNCERTIFIED research machinery (pilot-validated 2026-09-05, see the log).
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

__all__ = [
    "RatioPairWaBijector",
    "RatioPairWaUniform",
    "deflection_ratio_pair_wa_fn",
    "validate_ratio_pair_wa",
]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_S_CLIP = 8.5          # |logit-theta| bound: ndtr saturates in float64 near 8.3
_BACKTRACK = (1.0, 0.3, 0.1)
Bounds = Tuple[float, float]


def _log_phi(z):
    return -0.5 * jnp.square(z) - _LOG_SQRT_2PI


def deflection_ratio_pair_wa_fn(cosmo, z_pair: Sequence[float], *,
                                fixed: Dict[str, float]) -> Callable:
    """Build ``r_pair_fn(Om0, w0, wa) -> (r_a, r_b)`` from a ``w0waCDM``-type profile.

    ``fixed`` must list EVERY non-sampled cosmology parameter (e.g. ``dict(H0=70.0,
    k=0.0)``) -- no silent defaults. Prefer the lowest and highest source redshifts
    (largest ratio-contour crossing angle); neither may equal ``cosmo.z_source_ref``.
    """
    if len(z_pair) != 2 or float(z_pair[0]) == float(z_pair[1]):
        raise ValueError(f"z_pair must be two distinct redshifts; got {z_pair!r}.")
    z_ref = float(np.asarray(cosmo.z_source_ref).reshape(()))
    za, zb = float(z_pair[0]), float(z_pair[1])
    if za == z_ref or zb == z_ref:
        raise ValueError(f"z_pair contains the reference redshift {z_ref} (ratio == 1).")
    sampled = {"Om0", "w0", "wa"}
    if sampled & set(fixed):
        raise ValueError("fixed must not contain Om0/w0/wa -- they are the sampled triple.")
    missing = set(cosmo._params) - sampled - set(fixed)
    if missing:
        raise ValueError(
            f"fixed is missing cosmology parameter(s) {sorted(missing)} of "
            f"{type(cosmo).__name__} (params {cosmo._params}).")
    extra = set(fixed) - set(cosmo._params)
    if extra:
        raise ValueError(
            f"fixed contains {sorted(extra)}, not parameters of {type(cosmo).__name__}.")
    fixed_f = {k: float(v) for k, v in fixed.items()}

    def r_pair_fn(om0, w0, wa):
        ra = jnp.reshape(cosmo.deflection_ratio(za, Om0=om0, w0=w0, wa=wa, **fixed_f), ())
        rb = jnp.reshape(cosmo.deflection_ratio(zb, Om0=om0, w0=w0, wa=wa, **fixed_f), ())
        return ra, rb

    return r_pair_fn


def _det2(r_pair_fn, om, w0, wa):
    J = jax.jacfwd(lambda th: jnp.stack(r_pair_fn(th[0], th[1], wa)))(jnp.stack([om, w0]))
    return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]


class RatioPairWaBijector(tfb.Bijector):
    """Diffeomorphism ``R^3 -> {(Om0, w0, wa) in box reachable through the chart}``
    (module doc). Event ndims 1; last axis is the (Om0, w0, wa) 3-vector, in that
    order (the order the grouped tuple key must use)."""

    def __init__(self, r_pair_fn: Callable, om0_bounds: Bounds, w0_bounds: Bounds,
                 wa_bounds: Bounds, *, n_newton: int = 40,
                 n_image_grid: Tuple[int, int, int] = (61, 61, 41),
                 init_table_size: Tuple[int, int, int] = (17, 17, 13),
                 residual_rtol: float = 1e-8, validate_args: bool = False,
                 name: str = "ratio_pair_wa"):
        parameters = dict(locals())
        lo = np.array([om0_bounds[0], w0_bounds[0], wa_bounds[0]], float)
        hi = np.array([om0_bounds[1], w0_bounds[1], wa_bounds[1]], float)
        if not np.all(hi > lo):
            raise ValueError(
                f"bounds must be increasing; got {om0_bounds}, {w0_bounds}, {wa_bounds}.")
        if int(n_newton) < 15:
            raise ValueError(
                f"n_newton={n_newton} < 15 cannot reach float64 residuals; use >= 15.")
        self._lo, self._hi, self._wid = lo, hi, hi - lo
        self._rp = r_pair_fn
        self._rpv = jnp.vectorize(r_pair_fn)
        self._rtol = float(residual_rtol)

        # Image bounding boxes of r_a, r_b over the 3-D box (concrete, eager).
        axes = [jnp.linspace(lo[i], hi[i], int(n_image_grid[i])) for i in range(3)]
        om, w, wa = jnp.meshgrid(*axes, indexing="ij")
        ra, rb = self._rpv(om, w, wa)
        if not (bool(jnp.isfinite(ra).all()) and bool(jnp.isfinite(rb).all())):
            raise ValueError("r_pair_fn is non-finite on the construction grid.")
        self._ra_box = (float(ra.min()), float(ra.max()))
        self._rb_box = (float(rb.min()), float(rb.max()))
        self._rw = (self._ra_box[1] - self._ra_box[0], self._rb_box[1] - self._rb_box[0])
        if not (self._rw[0] > 0.0 and self._rw[1] > 0.0):
            raise ValueError(f"degenerate ratio image (widths {self._rw}).")

        # Newton init table over interior cell centres of the 3-D box; the nearest
        # entry in whitened (ra, rb, wa) seeds the solve.
        k = tuple(int(v) for v in init_table_size)
        cent = [(np.arange(k[i]) + 0.5) / k[i] for i in range(3)]
        omt, wt, wat = np.meshgrid(
            *[lo[i] + self._wid[i] * cent[i] for i in range(3)], indexing="ij")
        rat, rbt = (np.asarray(a) for a in
                    self._rpv(jnp.asarray(omt), jnp.asarray(wt), jnp.asarray(wat)))
        s1t, s2t, _ = np.meshgrid(ndtri(cent[0]), ndtri(cent[1]), cent[2], indexing="ij")
        self._tab = tuple(jnp.asarray(a.ravel()) for a in (s1t, s2t, rat, rbt, wat))
        self._solve = jnp.vectorize(self._make_solver(int(n_newton)))
        self._logabsdet = jnp.vectorize(
            lambda om_, w0_, wa_: jnp.log(jnp.abs(_det2(r_pair_fn, om_, w0_, wa_))))
        super().__init__(forward_min_event_ndims=1, is_constant_jacobian=False,
                         validate_args=validate_args, parameters=parameters, name=name)

    # -- solver -----------------------------------------------------------------
    def _make_solver(self, n_newton: int):
        lo, wid = self._lo, self._wid
        rp = self._rp
        sa, sb = self._rw
        rtol = self._rtol
        s1t, s2t, rat, rbt, wat = self._tab
        wa_w = wid[2]

        def _theta(s1, s2):
            return lo[0] + wid[0] * ndtr(s1), lo[1] + wid[1] * ndtr(s2)

        def _res(s, ra, rb, wa):
            r1, r2 = rp(*_theta(s[0], s[1]), wa)
            return jnp.stack([(r1 - ra) / sa, (r2 - rb) / sb])

        def _newton(ra, rb, wa):
            # Seed FROM the inputs so the fori_loop carry inherits their sharding
            # (shard_map varying-type rule, see ratio_coords.py).
            base = ra * 0 + rb * 0 + wa * 0
            d2 = (jnp.square((rat - ra) / sa) + jnp.square((rbt - rb) / sb)
                  + jnp.square((wat - wa) / wa_w))
            k = jnp.argmin(d2)
            s = jnp.stack([base + s1t[k], base + s2t[k]])
            res_fn = lambda s_: _res(s_, ra, rb, wa)
            jac_fn = jax.jacfwd(res_fn)
            # Early exit: stop once the whitened residual reaches 1e-13 (the float64
            # floor is ~1e-15; residual_rtol only gates acceptance). Newton converges
            # quadratically from the table seed (4-6 iterations on the ersatz-carousel
            # pair); the iteration is the dominant per-step cost of the chart on GPU.
            f_stop = 1e-26

            def cond(carry):
                i, _, f = carry
                return jnp.logical_and(i < n_newton, f > f_stop)

            def body(carry):
                i, s_, f_cur = carry
                g = res_fn(s_)
                J = jac_fn(s_)
                det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                safe = jnp.where(jnp.abs(det) < 1e-300, 1e-300, det)
                step = jnp.stack([-(J[1, 1] * g[0] - J[0, 1] * g[1]) / safe,
                                  -(-J[1, 0] * g[0] + J[0, 0] * g[1]) / safe])
                nrm = jnp.sqrt(jnp.sum(jnp.square(step)))
                step = step * jnp.minimum(1.0, 4.0 / jnp.maximum(nrm, 1e-300))
                best_s, best_f = s_, jnp.sum(jnp.square(g))
                for t in _BACKTRACK:
                    cand = jnp.clip(s_ + t * step, -_S_CLIP, _S_CLIP)
                    f = jnp.sum(jnp.square(res_fn(cand)))
                    take = f < best_f
                    best_s = jnp.where(take, cand, best_s)
                    best_f = jnp.where(take, f, best_f)
                return i + 1, best_s, best_f

            f0 = jnp.sum(jnp.square(res_fn(s)))
            _, s, _ = jax.lax.while_loop(cond, body, (jnp.zeros((), jnp.int32) + (base * 0).astype(jnp.int32), s, f0))
            return _theta(s[0], s[1])

        @jax.custom_vjp
        def solve(ra, rb, wa):
            return _newton(ra, rb, wa)

        def fwd(ra, rb, wa):
            om, w0 = _newton(ra, rb, wa)
            return (om, w0), (om, w0, ra, rb, wa)

        def bwd(saved, cot):
            om, w0, ra, rb, wa = saved
            g_th = jnp.stack([cot[0], cot[1]])
            full = jax.jacfwd(lambda q: jnp.stack(rp(q[0], q[1], q[2])))(
                jnp.stack([om, w0, wa]))
            J, Jw = full[:, :2], full[:, 2]
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            safe = jnp.where(jnp.abs(det) < 1e-300, 1e-300, det)
            # theta = theta(r, wa): dtheta/dr = J^-1, dtheta/dwa = -J^-1 Jw.
            jinvT_g = jnp.stack([(J[1, 1] * g_th[0] - J[1, 0] * g_th[1]) / safe,
                                 (-J[0, 1] * g_th[0] + J[0, 0] * g_th[1]) / safe])
            g_wa = -jnp.dot(jinvT_g, Jw)
            r1, r2 = rp(om, w0, wa)
            res = jnp.maximum(jnp.abs((r1 - ra) / sa), jnp.abs((r2 - rb) / sb))
            ok = jnp.where(res <= rtol, 1.0, 0.0)   # no root -> zero grad (density -inf)
            return jinvT_g[0] * ok, jinvT_g[1] * ok, g_wa * ok

        solve.defvjp(fwd, bwd)
        return solve

    # -- pieces -----------------------------------------------------------------
    def _parts(self, z):
        ra = self._ra_box[0] + self._rw[0] * ndtr(z[..., 0])
        rb = self._rb_box[0] + self._rw[1] * ndtr(z[..., 1])
        wa = self._lo[2] + self._wid[2] * ndtr(z[..., 2])
        return ra, rb, wa

    @staticmethod
    def _clip_unit(t):
        dt = jnp.result_type(t)
        return jnp.clip(t, jnp.finfo(dt).tiny, 1.0 - jnp.finfo(dt).epsneg)

    def _fldj_parts(self, z, om, w0, wa):
        return (math.log(self._rw[0]) + _log_phi(z[..., 0])
                + math.log(self._rw[1]) + _log_phi(z[..., 1])
                + math.log(self._wid[2]) + _log_phi(z[..., 2])
                - self._logabsdet(om, w0, wa))

    # -- bijector interface -------------------------------------------------------
    def _forward(self, z):
        ra, rb, wa = self._parts(z)
        om, w0 = self._solve(ra, rb, wa)
        return jnp.stack([om, w0, wa], axis=-1)

    def _inverse(self, x):
        om, w0, wa = x[..., 0], x[..., 1], x[..., 2]
        ra, rb = self._rpv(om, w0, wa)
        z1 = ndtri(self._clip_unit((ra - self._ra_box[0]) / self._rw[0]))
        z2 = ndtri(self._clip_unit((rb - self._rb_box[0]) / self._rw[1]))
        z3 = ndtri(self._clip_unit((wa - self._lo[2]) / self._wid[2]))
        return jnp.stack([z1, z2, z3], axis=-1)

    def _forward_log_det_jacobian(self, z):
        ra, rb, wa = self._parts(z)
        om, w0 = self._solve(ra, rb, wa)
        r1, r2 = self._rpv(om, w0, wa)
        res = jnp.maximum(jnp.abs((r1 - ra) / self._rw[0]),
                          jnp.abs((r2 - rb) / self._rw[1]))
        return jnp.where(res <= self._rtol, self._fldj_parts(z, om, w0, wa), -jnp.inf)

    def _inverse_log_det_jacobian(self, x):
        z = self._inverse(x)
        return -self._fldj_parts(z, x[..., 0], x[..., 1], x[..., 2])


def validate_ratio_pair_wa(r_pair_fn: Callable, om0_bounds: Bounds, w0_bounds: Bounds,
                           wa_bounds: Bounds, *, det_atol: float = 0.0,
                           roundtrip_atol: float = 0.0,
                           n_grid: Tuple[int, int, int] = (41, 31, 21),
                           n_roundtrip: Tuple[int, int, int] = (9, 9, 7),
                           n_newton: int = 40,
                           n_image_grid: Tuple[int, int, int] = (61, 61, 41)
                           ) -> Dict[str, object]:
    """Grid-check that the pair supports the chart on the 3-D box; raise if it doesn't.

    1. ``r_pair_fn`` and det d(r_a,r_b)/d(Om0,w0) finite everywhere (fatal).
    2. The WHITENED 2x2 determinant keeps one sign; zero crossings are tolerated
       only where ``|det_w| <= det_atol``. The report gives the flipped region's
       (Om0, w0, wa) extent so the caller can check it carries no posterior mass.
    3. Round-trip ``forward(inverse(theta))`` through an actual bijector on an
       ``n_roundtrip`` interior grid, outside the ``|det_w| <= det_atol`` sliver,
       must be ``<= roundtrip_atol`` (box-normalized).

    Strict 0.0 defaults: pass measured tolerances (raise-over-default).
    """
    lo = np.array([om0_bounds[0], w0_bounds[0], wa_bounds[0]], float)
    hi = np.array([om0_bounds[1], w0_bounds[1], wa_bounds[1]], float)
    if not np.all(hi > lo):
        raise ValueError(
            f"bounds must be increasing; got {om0_bounds}, {w0_bounds}, {wa_bounds}.")
    axes = [jnp.linspace(lo[i], hi[i], int(n_grid[i])) for i in range(3)]
    om, w, wa = jnp.meshgrid(*axes, indexing="ij")
    ra, rb = jnp.vectorize(r_pair_fn)(om, w, wa)
    det_fn = jnp.vectorize(lambda o, w0_, a: _det2(r_pair_fn, o, w0_, a))
    det = det_fn(om, w, wa)
    if not (bool(jnp.isfinite(ra).all()) and bool(jnp.isfinite(rb).all())
            and bool(jnp.isfinite(det).all())):
        raise ValueError("r_pair_fn or its Jacobian is non-finite on the validation grid.")
    wa_, wb_ = float(ra.max() - ra.min()), float(rb.max() - rb.min())
    whiten = (hi[0] - lo[0]) * (hi[1] - lo[1]) / (wa_ * wb_)
    det_w = np.asarray(det) * whiten
    sign = 1.0 if float(np.median(det_w)) >= 0.0 else -1.0
    signed = sign * det_w
    degen = np.abs(det_w) <= float(det_atol)
    flipped = signed < 0.0
    beyond = signed < -float(det_atol)
    om_n, w_n, wa_n = (np.asarray(a) for a in (om, w, wa))

    def _extent(mask):
        if not mask.any():
            return None
        return dict(om0=(float(om_n[mask].min()), float(om_n[mask].max())),
                    w0=(float(w_n[mask].min()), float(w_n[mask].max())),
                    wa=(float(wa_n[mask].min()), float(wa_n[mask].max())))

    report: Dict[str, object] = dict(
        det_sign=sign, min_signed_det_w=float(signed.min()),
        max_abs_det_w=float(np.abs(det_w).max()),
        n_grid_flips=int(flipped.sum()), n_grid_flips_beyond_atol=int(beyond.sum()),
        flip_frac=float(flipped.mean()), flip_extent=_extent(flipped),
        degen_frac=float(degen.mean()), degen_extent=_extent(degen),
        ra_box=(float(ra.min()), float(ra.max())), rb_box=(float(rb.min()), float(rb.max())),
        det_atol=float(det_atol), roundtrip_atol=float(roundtrip_atol),
        n_grid=tuple(int(v) for v in n_grid),
    )
    if report["n_grid_flips_beyond_atol"] > 0:
        raise ValueError(
            f"ratio-pair+wa Jacobian determinant changes sign beyond det_atol={det_atol}: "
            f"{report['n_grid_flips_beyond_atol']} grid point(s), most-negative signed "
            f"whitened det {report['min_signed_det_w']:.3e}, in {report['flip_extent']}. "
            f"The chart is folded there -- shrink the box, pick a pair with a larger "
            f"crossing angle, or pass a MEASURED det_atol covering a sliver you have "
            f"checked carries no posterior mass. Report: {report}")

    bij = RatioPairWaBijector(r_pair_fn, om0_bounds, w0_bounds, wa_bounds,
                              n_newton=n_newton, n_image_grid=n_image_grid)
    raxes = [jnp.linspace(lo[i], hi[i], int(n_roundtrip[i]) + 2)[1:-1] for i in range(3)]
    om_r, w_r, wa_r = jnp.meshgrid(*raxes, indexing="ij")
    theta = jnp.stack([om_r, w_r, wa_r], axis=-1)
    theta_rt = bij.forward(bij.inverse(theta))
    err = np.asarray(jnp.max(jnp.abs(theta_rt - theta) / jnp.asarray(hi - lo), axis=-1))
    det_w_rt = np.asarray(det_fn(om_r, w_r, wa_r)) * whiten
    outside = np.abs(det_w_rt) > float(det_atol)
    report["max_roundtrip_err"] = float(err.max())
    report["max_roundtrip_err_outside_degen"] = float(np.where(outside, err, 0.0).max())
    report["n_roundtrip"] = tuple(int(v) for v in n_roundtrip)
    if report["max_roundtrip_err_outside_degen"] > float(roundtrip_atol):
        raise ValueError(
            f"ratio-pair+wa round-trip error {report['max_roundtrip_err_outside_degen']:.3e} "
            f"(box-normalized, outside the |det_w|<={det_atol} sliver) exceeds "
            f"roundtrip_atol={roundtrip_atol}. Report: {report}")
    return report


class RatioPairWaUniform(tfd.Independent):
    """Uniform prior on the (Om0, w0, wa) box whose event-space bijector is the
    ratio-pair + wa chart (event_shape [3]). Tuple-key usage -- the key order MUST
    be ``("Om0", "w0", "wa")``::

        cosmo = w0waCDM_Cosmo(z_lens, z_source_ref)
        Component(cosmo, {
            "H0": 70.0, "k": 0.0,
            ("Om0", "w0", "wa"): RatioPairWaUniform(
                deflection_ratio_pair_wa_fn(cosmo, (z_low, z_high),
                                            fixed=dict(H0=70.0, k=0.0)),
                (0.05, 0.99), (-2.0, -1.0 / 3.0), (-3.0, 2.0),
                det_atol=..., roundtrip_atol=...),
        })

    The prior density is exactly the independent uniform box; only the sampling
    coordinates differ. ``LensModel(unconstrain="gaussian")`` leaves grouped priors'
    own bijectors alone, so this chart survives that policy. Validation runs at
    construction with caller-supplied measured tolerances unless
    ``skip_validation=True``; the report is attached as ``self.ratio_pair_wa_report``.
    """

    def __init__(self, r_pair_fn: Callable, om0_bounds: Bounds, w0_bounds: Bounds,
                 wa_bounds: Bounds, *, det_atol: float = 0.0, roundtrip_atol: float = 0.0,
                 n_newton: int = 40, n_image_grid: Tuple[int, int, int] = (61, 61, 41),
                 init_table_size: Tuple[int, int, int] = (17, 17, 13),
                 n_validation_grid: Tuple[int, int, int] = (41, 31, 21),
                 n_roundtrip: Tuple[int, int, int] = (9, 9, 7), residual_rtol: float = 1e-8,
                 skip_validation: bool = False, dtype=None, validate_args: bool = False,
                 name: str = "RatioPairWaUniform"):
        if skip_validation:
            self.ratio_pair_wa_report = None
        else:
            self.ratio_pair_wa_report = validate_ratio_pair_wa(
                r_pair_fn, om0_bounds, w0_bounds, wa_bounds, det_atol=det_atol,
                roundtrip_atol=roundtrip_atol, n_grid=n_validation_grid,
                n_roundtrip=n_roundtrip, n_newton=n_newton, n_image_grid=n_image_grid)
        self._esb = RatioPairWaBijector(
            r_pair_fn, om0_bounds, w0_bounds, wa_bounds, n_newton=n_newton,
            n_image_grid=n_image_grid, init_table_size=init_table_size,
            residual_rtol=residual_rtol, validate_args=validate_args)
        if dtype is None:
            dtype = jnp.zeros(()).dtype
        low = jnp.asarray([om0_bounds[0], w0_bounds[0], wa_bounds[0]], dtype)
        high = jnp.asarray([om0_bounds[1], w0_bounds[1], wa_bounds[1]], dtype)
        super().__init__(tfd.Uniform(low=low, high=high), reinterpreted_batch_ndims=1,
                         validate_args=validate_args, name=name)

    def _default_event_space_bijector(self):
        return self._esb
