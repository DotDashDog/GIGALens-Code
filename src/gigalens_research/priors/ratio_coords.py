"""Ratio-coordinates grouped prior over (Om0, w0) — exact reparameterization.

Context (lab log ``docs/logs/sample-cosmology-dspl.md``, claims C-1..C-3): the
DSPL pixel likelihood depends on cosmology only through the plane-2 deflection
ratio ``r2(Om0, w0)``, whose level contours form a rotating, semi-infinite thin
ridge in the baseline (Om0, w0)+NormalCDF sampling coordinates — a geometry a
single frozen global MCLMC metric cannot cover (Run B), truncating ~10% of the
posterior. Run A showed that sampling r2 as a *free* parameter is trivially
easy; this module keeps full single-run forward modeling instead, by making the
data-stiff scalar a *sampling coordinate* while (Om0, w0) remain the model
parameters.

Construction — a triangular diffeomorphism ``z = (z1, z2) -> (Om0, w0)``:

    Om0 = om_lo + (om_hi - om_lo) * NormalCDF(z1)                (as baseline)
    u   = u_a + (u_b - u_a) * NormalCDF(z2),
          with the CONDITIONAL bracket u_a = u_fn(Om0, w0_lo),
                                       u_b = u_fn(Om0, w0_hi)
    w0  = the root of u_fn(Om0, w0) = u   (bracketed bisection; gradients via
                                           the implicit function theorem)

where ``u_fn(Om0, w0)`` is a data-stiff scalar — for the DSPL system the
deflection ratio itself, and in general any weighted combination of the
system's deflection ratios (see :func:`deflection_ratio_u_fn`). The Jacobian is
triangular (z1 -> Om0 only), so both log-det directions are analytic:

    log|det dtheta/dz| = log(om_hi - om_lo) + log phi(z1)
                       + log|u_b - u_a|     + log phi(z2) - log|du_fn/dw0|

Why conditioning on Om0 and solving w0 (not the reverse): r2 is NOT monotone in
Om0 at fixed w0 — the contour crest near (Om0~0.2, w0~-0.94) means a horizontal
cut crosses a level contour twice — while r2 IS monotone in w0 at fixed Om0
over the prior box, except for a structurally degenerate sliver near Om0 -> 1
where dark energy vanishes and w0 becomes unidentifiable (dr2/dw0 -> 0, with
quadrature-noise-level sign flips; measured on
``results/sample_cosmology/dspl_cosmology_newapi/def_ratio_grid.npz``: flips
confined to Om0 >~ 0.79, |dr2/dw0| ~ 1e-9 vs median ~1e-2, posterior mass
~2e-17, >= 7.9 sigma_r from the data contour). :func:`validate_ratio_coords`
measures this sliver on the actual ``u_fn`` at construction and raises unless
the caller passes explicit tolerances covering it — raise, never default.

Exactness: the posterior in (Om0, w0) is invariant under ANY diffeomorphism of
the sampling coordinates, so the quality of ``u_fn`` (quadrature resolution,
choice of ratio weights) affects sampling GEOMETRY only, never correctness —
the scene computes its own deflection ratios from (Om0, w0) downstream of the
map. The one caveat is the degenerate sliver above: there the map degrades
gracefully (the bracket width |u_b - u_a| and du_fn/dw0 vanish at the same
rate, so the log-det stays finite), but within the sliver the w0-root is not
unique and gradients of the solve can be large or wrong-signed. The validator
bounds the sliver; it carries ~zero posterior mass for the DSPL system.

Known edge behavior: at Om0 bounds hit EXACTLY (requires |z1| >~ 38 so that
NormalCDF saturates), a degenerate bracket u_a == u_b yields log-det -> -inf;
measure-zero and handled by the sampler's non-finite rejection.

Status: UNCERTIFIED research machinery (Run C, awaiting design-checkpoint
approval + grader inspection). Promote to ``gigalens.jax.grouped_priors`` only
after certification.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

__all__ = [
    "RatioCoordsBijector",
    "RatioCoordsUniform",
    "deflection_ratio_u_fn",
    "validate_ratio_coords",
]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _log_phi(z):
    """log of the standard normal pdf."""
    return -0.5 * jnp.square(z) - _LOG_SQRT_2PI


def deflection_ratio_u_fn(
    cosmo,
    z_sources: Sequence[float],
    weights: Sequence[float],
    *,
    fixed: Dict[str, float],
) -> Callable:
    """Build ``u_fn(Om0, w0) -> scalar`` from a cosmology profile's deflection ratios.

    ``u = sum_i weights[i] * cosmo.deflection_ratio(z_sources[i], Om0=..., w0=...,
    **fixed)``. For a single extra source plane (the DSPL system) pass one
    redshift with weight 1.0, making ``u`` the deflection ratio itself; for more
    planes pass the data-stiff combination (e.g. the top Fisher-whitened one).
    An imperfect combination degrades sampling geometry, not correctness (see
    module docstring).

    Args:
      cosmo: a ``CosmoBase`` profile (e.g. ``w0waCDM_Cosmo(z_lens, z_source_ref)``).
      z_sources: source-plane redshifts whose deflection ratios enter ``u``.
      weights: one weight per redshift (no silent normalization).
      fixed: EVERY non-sampled parameter of ``cosmo`` (e.g. ``dict(H0=70.0,
        k=0.0, wa=0.0)``), passed explicitly — no silent defaults. ``Om0``/``w0``
        are the sampled pair and must not appear here.
    """
    zs = tuple(float(z) for z in z_sources)
    ws = tuple(float(w) for w in weights)
    if len(zs) == 0 or len(zs) != len(ws):
        raise ValueError(
            f"z_sources and weights must be non-empty and the same length; got "
            f"{len(zs)} redshift(s) and {len(ws)} weight(s).")
    if "Om0" in fixed or "w0" in fixed:
        raise ValueError(
            "fixed must not contain 'Om0' or 'w0' — they are the sampled pair.")
    sampled = {"Om0", "w0"}
    missing = set(cosmo._params) - sampled - set(fixed)
    if missing:
        raise ValueError(
            f"fixed is missing cosmology parameter(s) {sorted(missing)} of "
            f"{type(cosmo).__name__} (params {cosmo._params}); every non-sampled "
            "parameter must be given explicitly — no silent default.")
    extra = set(fixed) - set(cosmo._params)
    if extra:
        raise ValueError(
            f"fixed contains {sorted(extra)}, not parameter(s) of "
            f"{type(cosmo).__name__} (params {cosmo._params}).")
    fixed_f = {k: float(v) for k, v in fixed.items()}

    def u_fn(om0, w0):
        tot = jnp.zeros((), dtype=jnp.result_type(om0, w0))
        for z_i, a_i in zip(zs, ws):
            r_i = cosmo.deflection_ratio(z_i, Om0=om0, w0=w0, **fixed_f)
            tot = tot + a_i * jnp.reshape(r_i, ())
        return tot

    return u_fn


def validate_ratio_coords(
    u_fn: Callable,
    om0_bounds: Tuple[float, float],
    w0_bounds: Tuple[float, float],
    *,
    du_dw_atol: float = 0.0,
    excursion_atol: float = 0.0,
    n_om: int = 201,
    n_w: int = 201,
) -> Dict[str, float]:
    """Grid-check that ``u_fn`` supports the triangular map; raise if it doesn't.

    Requirements checked on an ``n_om x n_w`` grid over the box (autodiff of the
    ACTUAL ``u_fn``, not a saved table):

    1. ``u_fn`` and ``du_fn/dw0`` finite everywhere (always fatal if violated).
    2. ``du_fn/dw0`` keeps one sign. Flips are tolerated only where
       ``|du_fn/dw0| <= du_dw_atol`` (the documented w0-unidentifiability sliver,
       e.g. Om0 -> 1 in [w0]wCDM); any flip beyond the tolerance raises.
    3. Interior u-values stay inside the endpoint bracket
       ``[min(u_a, u_b), max(u_a, u_b)]`` to within ``excursion_atol`` (an
       excursion beyond the bracket means the bisection cannot reach that
       sliver); a larger excursion raises.

    The default tolerances are 0.0 (strict): for a u_fn with a degenerate edge
    the caller must pass measured, explicitly-derived tolerances. Returns the
    report dict (attach it to the model card).
    """
    om_lo, om_hi = float(om0_bounds[0]), float(om0_bounds[1])
    w_lo, w_hi = float(w0_bounds[0]), float(w0_bounds[1])
    if not (om_hi > om_lo) or not (w_hi > w_lo):
        raise ValueError(
            f"bounds must be increasing; got om0={om0_bounds}, w0={w0_bounds}.")

    om = jnp.linspace(om_lo, om_hi, int(n_om))
    w = jnp.linspace(w_lo, w_hi, int(n_w))
    om_mesh, w_mesh = jnp.meshgrid(om, w, indexing="ij")

    u_vec = jnp.vectorize(u_fn)
    du_dw_vec = jnp.vectorize(jax.grad(u_fn, argnums=1))
    u_grid = u_vec(om_mesh, w_mesh)
    du_dw = du_dw_vec(om_mesh, w_mesh)

    if not (bool(jnp.isfinite(u_grid).all()) and bool(jnp.isfinite(du_dw).all())):
        raise ValueError(
            "u_fn or du_fn/dw0 is non-finite somewhere on the validation grid; "
            "the ratio-coordinates map is unusable on this box.")

    # Dominant sign = the sign of the mean derivative (majority direction).
    sign = 1.0 if float(jnp.mean(du_dw)) >= 0.0 else -1.0
    signed = sign * du_dw
    worst = float(jnp.min(signed))          # most-negative signed derivative
    n_flip = int(jnp.sum(signed < 0.0))
    n_flip_beyond = int(jnp.sum(signed < -float(du_dw_atol)))

    # Interior excursion beyond the endpoint bracket, per om row.
    u_a = u_grid[:, 0][:, None]
    u_b = u_grid[:, -1][:, None]
    lo_b = jnp.minimum(u_a, u_b)
    hi_b = jnp.maximum(u_a, u_b)
    excursion = float(jnp.max(jnp.maximum(u_grid - hi_b, lo_b - u_grid)))
    min_bracket = float(jnp.min(jnp.abs(u_b - u_a)))

    report = dict(
        sign=sign,
        min_signed_du_dw=worst,
        max_abs_du_dw=float(jnp.max(jnp.abs(du_dw))),
        n_grid_flips=n_flip,
        n_grid_flips_beyond_atol=n_flip_beyond,
        max_interior_excursion=max(excursion, 0.0),
        min_endpoint_bracket=min_bracket,
        du_dw_atol=float(du_dw_atol),
        excursion_atol=float(excursion_atol),
        n_om=int(n_om),
        n_w=int(n_w),
    )

    if n_flip_beyond > 0:
        raise ValueError(
            f"u_fn is not monotone in w0 at fixed Om0: {n_flip_beyond} grid "
            f"point(s) have sign-flipped du/dw0 with magnitude beyond "
            f"du_dw_atol={du_dw_atol} (most-negative signed derivative "
            f"{worst:.3e}; dominant sign {sign:+.0f}). Either the conditioning "
            f"choice is wrong for this u_fn, or pass a measured, explicitly "
            f"derived du_dw_atol covering a documented degenerate sliver. "
            f"Report: {report}")
    if excursion > float(excursion_atol):
        raise ValueError(
            f"u_fn exceeds its w0-endpoint bracket in the interior by "
            f"{excursion:.3e} > excursion_atol={excursion_atol}; the bisection "
            f"bracket cannot reach that sliver. Report: {report}")
    return report


def _make_w_solver(u_fn: Callable, w_lo: float, w_hi: float, n_bisect: int):
    """Scalar solver for ``u_fn(om, w) = u`` on [w_lo, w_hi].

    Fixed-iteration bracketed bisection (robust to a near-flat du/dw; never
    leaves the bracket), wrapped in ``custom_vjp`` with implicit-function-
    theorem gradients:  dw = (du - u_Om0 dOm0) / u_w0.
    """

    def _bisect(om, u):
        dt = jnp.result_type(om, u)
        # Seed the bracket FROM the inputs (om*0 + u*0): under shard_map (the
        # sharded MCLMC kernel) a plain-constant fori_loop carry is unvarying
        # while the body output mixes in the device-varying om/u, and the loop
        # rejects the mismatched carry types.
        base = (om * 0 + u * 0).astype(dt)
        lo = base + jnp.asarray(w_lo, dt)
        hi = base + jnp.asarray(w_hi, dt)
        # Local orientation: works wherever u_fn is monotone at this om and
        # degrades gracefully (stays in the bracket) in a degenerate sliver.
        s = jnp.sign(u_fn(om, hi) - u_fn(om, lo))

        def body(_, lohi):
            lo, hi = lohi
            mid = 0.5 * (lo + hi)
            go_right = s * (u_fn(om, mid) - u) < 0.0
            return jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid)

        lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
        return 0.5 * (lo + hi)

    @jax.custom_vjp
    def solve(om, u):
        return _bisect(om, u)

    def fwd(om, u):
        w = _bisect(om, u)
        return w, (om, u, w)

    def bwd(res, g):
        om, _, w = res
        du_dom = jax.grad(u_fn, argnums=0)(om, w)
        du_dw = jax.grad(u_fn, argnums=1)(om, w)
        # A root pinned at a bracket edge means the target was outside the
        # bracket (an inactive constraint in the u-first interval logic): the
        # true derivative of the clamped output is 0, and the implicit rule
        # (which assumes an interior root) would emit garbage — measured
        # ~1e15 via 1/slope at the edge. Margin ≫ the 2^-n_bisect resolution.
        margin = (w_hi - w_lo) * 1e-9
        interior = jnp.where((w > w_lo + margin) & (w < w_hi - margin), 1.0, 0.0)
        return (-(du_dom / du_dw) * g * interior, g / du_dw * interior)

    solve.defvjp(fwd, bwd)
    return solve


class RatioCoordsBijector(tfb.Bijector):
    """Triangular diffeomorphism ``R^2 -> (om0_bounds) x (w0_bounds)`` (module doc).

    Event ndims 1 (last axis is the (Om0, w0) 2-vector). ``_forward`` runs the
    bracketed w0-solve; ``_inverse`` is analytic; both log-det directions are
    analytic (triangular Jacobian — no autodiff through the solve).
    """

    def __init__(
        self,
        u_fn: Callable,
        om0_bounds: Tuple[float, float],
        w0_bounds: Tuple[float, float],
        *,
        n_bisect: int = 80,
        validate_args: bool = False,
        name: str = "ratio_coords",
    ):
        parameters = dict(locals())
        self._om_lo, self._om_hi = float(om0_bounds[0]), float(om0_bounds[1])
        self._w_lo, self._w_hi = float(w0_bounds[0]), float(w0_bounds[1])
        if not (self._om_hi > self._om_lo) or not (self._w_hi > self._w_lo):
            raise ValueError(
                f"bounds must be increasing; got om0={om0_bounds}, w0={w0_bounds}.")
        if int(n_bisect) < 50:
            raise ValueError(
                f"n_bisect={n_bisect} < 50 cannot reach float64 resolution on the "
                f"w0 bracket; use >= 50.")
        self._u_scalar = u_fn
        self._u = jnp.vectorize(u_fn)
        self._du_dw = jnp.vectorize(jax.grad(u_fn, argnums=1))
        self._solve = jnp.vectorize(
            _make_w_solver(u_fn, self._w_lo, self._w_hi, int(n_bisect)))
        super().__init__(
            forward_min_event_ndims=1,
            is_constant_jacobian=False,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    # -- pieces ----------------------------------------------------------------
    def _om_from_z1(self, z1):
        return self._om_lo + (self._om_hi - self._om_lo) * ndtr(z1)

    def _bracket(self, om):
        dt = jnp.result_type(om)
        u_a = self._u(om, jnp.asarray(self._w_lo, dt))
        u_b = self._u(om, jnp.asarray(self._w_hi, dt))
        return u_a, u_b

    @staticmethod
    def _clip_unit(t):
        # Keep ndtri finite: open-interval clamp (same spirit as DiskBijector's
        # boundary floor). Activates only at box edges / the degenerate sliver.
        dt = jnp.result_type(t)
        return jnp.clip(t, jnp.finfo(dt).tiny, 1.0 - jnp.finfo(dt).epsneg)

    def _fldj_from_parts(self, z1, z2, om, w0):
        # log|det dtheta/dz| of the triangular map (module docstring).
        u_a, u_b = self._bracket(om)
        return (
            math.log(self._om_hi - self._om_lo) + _log_phi(z1)
            + jnp.log(jnp.abs(u_b - u_a)) + _log_phi(z2)
            - jnp.log(jnp.abs(self._du_dw(om, w0)))
        )

    # -- bijector interface ------------------------------------------------------
    def _forward(self, z):
        z1, z2 = z[..., 0], z[..., 1]
        om = self._om_from_z1(z1)
        u_a, u_b = self._bracket(om)
        u = u_a + (u_b - u_a) * ndtr(z2)
        w0 = self._solve(om, u)
        return jnp.stack([om, w0], axis=-1)

    def _inverse(self, x):
        om, w0 = x[..., 0], x[..., 1]
        z1 = ndtri(self._clip_unit((om - self._om_lo) / (self._om_hi - self._om_lo)))
        u_a, u_b = self._bracket(om)
        z2 = ndtri(self._clip_unit((self._u(om, w0) - u_a) / (u_b - u_a)))
        return jnp.stack([z1, z2], axis=-1)

    def _forward_log_det_jacobian(self, z):
        z1, z2 = z[..., 0], z[..., 1]
        om = self._om_from_z1(z1)
        u_a, u_b = self._bracket(om)
        u = u_a + (u_b - u_a) * ndtr(z2)
        w0 = self._solve(om, u)
        return self._fldj_from_parts(z1, z2, om, w0)

    def _inverse_log_det_jacobian(self, x):
        om, w0 = x[..., 0], x[..., 1]
        z = self._inverse(x)
        return -self._fldj_from_parts(z[..., 0], z[..., 1], om, w0)


class RatioCoordsUniform(tfd.Independent):
    """Uniform prior on the (Om0, w0) box whose event-space bijector is the
    ratio-coordinates map (event_shape [2]).

    Drops into a scene cosmology Component as a grouped tuple key::

        Component(w0waCDM_Cosmo(z_lens, z_source_ref), {
            "H0": 70.0, "k": 0.0, "wa": 0.0,
            ("Om0", "w0"): RatioCoordsUniform(u_fn, (0.0, 1.0), (-2.0, -1/3),
                                              du_dw_atol=..., excursion_atol=...),
        })

    The prior DENSITY is exactly the baseline's independent-uniform box (the
    posterior over (Om0, w0) is unchanged); only the sampling coordinates
    differ. :func:`validate_ratio_coords` runs at construction (see its
    docstring for the tolerances) unless ``skip_validation=True`` (tests only);
    the report is attached as ``self.ratio_coords_report`` for the model card.

    Subclass-of-``tfd.Independent`` with an overridden
    ``_default_event_space_bijector``, per the ``UniformBij`` precedent
    (``experiments/sample_cosmology``) and ``gigalens.jax.grouped_priors``.
    """

    def __init__(
        self,
        u_fn: Callable,
        om0_bounds: Tuple[float, float],
        w0_bounds: Tuple[float, float],
        *,
        du_dw_atol: float = 0.0,
        excursion_atol: float = 0.0,
        n_bisect: int = 80,
        n_validation_grid: int = 201,
        skip_validation: bool = False,
        dtype=None,
        validate_args: bool = False,
        name: str = "RatioCoordsUniform",
    ):
        if skip_validation:
            self.ratio_coords_report = None
        else:
            self.ratio_coords_report = validate_ratio_coords(
                u_fn, om0_bounds, w0_bounds,
                du_dw_atol=du_dw_atol, excursion_atol=excursion_atol,
                n_om=n_validation_grid, n_w=n_validation_grid)
        self._esb = RatioCoordsBijector(
            u_fn, om0_bounds, w0_bounds, n_bisect=n_bisect,
            validate_args=validate_args)
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




# ------------------------------------------------------------------------------
# u-first ordering (Run D) — banded global squash of the stiff scalar.
#
# Run C's om-first ordering left a residual disease: the conditional bracket
# [u(Om0,w_lo), u(Om0,w_hi)] drifts with Om0, so the likelihood band ROTATES in
# (z1, z2) (measured -8deg -> -84deg along the DSPL contour) and a frozen global
# metric truncates where the rotation outruns it (Run C outcome, lab log). The
# u-first ordering makes the likelihood a function of z1 ALONE:
#
#     u   = u_lo_band + (u_hi_band - u_lo_band) * NormalCDF(z1)   (fixed range)
#     Om0 = om_lo + (om_hi_eff(u) - om_lo) * NormalCDF(z2)
#     w0  = root of u_fn(Om0, w0) = u                              (as om-first)
#
# so the likelihood band in z-space is an axis-aligned slab — zero rotation at
# every point of the posterior, by construction.
#
# THE BAND (support amendment, measured on the real DSPL u_fn 2026-07-11): u has
# interior critical points on the full box — u_a(Om0) = u_fn(Om0, w_lo) dips
# before rising (minimum at Om0 ~ 0.02, corner slope -4.7) and
# u_b(Om0) = u_fn(Om0, w_hi) wiggles — so level sets of u change topology and a
# FULL-BOX "u as a global coordinate" diffeomorphism cannot exist. The map
# therefore squashes z1 into the band
#
#     u in ( u_a(start end),  min over Om0 of u_b )
#
# ("start end" = om_lo for a rising u_a, om_hi for a falling one), inside which
# u_a is single-crossing (validated) and the u_b constraint is never binding,
# giving Om0-interval [om_lo, root of u_a = u] (mirrored for falling u_a). The
# prior becomes an UNNORMALIZED TRUNCATED uniform: theta with u(theta) outside
# the band is unreachable. For the DSPL system the band is (1.28671, 1.34072)
# while the data sit at u* = 1.32392 +/- 6.7e-4 — the truncation edges are
# ~55 sigma and ~25 sigma away (excluded posterior mass < erfc(25/sqrt(2))
# ~ 1e-138); the excluded PRIOR volume is reported by the validator and must be
# quoted with any run. This is an explicit, quantified support amendment — not
# a silent approximation.
#
# Jacobian through the (u, Om0) intermediate is triangular, so the log-det is
# analytic:  log|det dtheta/dz| = log(band width) + log phi(z1)
#                               + log(om-interval width) + log phi(z2)
#                               - log|du_fn/dw0|.
# (The om-interval endpoint's u-derivative does not enter the determinant; it
# DOES carry forward-map gradients, via the solver's implicit-vjp rule, which
# zeroes itself at bracket-edge clamps.)
# ------------------------------------------------------------------------------


def validate_u_first_ratio_coords(
    u_fn: Callable,
    om0_bounds: Tuple[float, float],
    w0_bounds: Tuple[float, float],
    *,
    du_dw_atol: float = 0.0,
    excursion_atol: float = 0.0,
    curve_atol: float = 0.0,
    n_om: int = 201,
    n_w: int = 201,
) -> Dict[str, float]:
    """Validate ``u_fn`` for the banded u-first map; raise if unusable.

    Runs :func:`validate_ratio_coords` (w0-monotonicity + excursion — the
    w0-solve is shared), then derives the u-band and requires, on the grid:

    1. ``u_a`` finite with a well-defined dominant direction; the band floor is
       ``u_a`` at the branch's START end (om_lo if rising, om_hi if falling) —
       a dip BELOW the floor before the branch is tolerated (it is outside the
       band), but above the floor ``u_a`` must be single-crossing: signed slope
       >= -``curve_atol`` wherever ``u_a >= floor``.
    2. Band ceiling = min over the grid of ``u_b``; the band must be non-empty
       (floor < ceiling) — otherwise no u-value admits the construction.

    Reports the band, the excluded prior-volume fraction (grid measure of
    ``u < floor`` or ``u > ceiling`` over the box), and the curve diagnostics.
    """
    report = validate_ratio_coords(
        u_fn, om0_bounds, w0_bounds, du_dw_atol=du_dw_atol,
        excursion_atol=excursion_atol, n_om=n_om, n_w=n_w)

    om_lo, om_hi = float(om0_bounds[0]), float(om0_bounds[1])
    w_lo, w_hi = float(w0_bounds[0]), float(w0_bounds[1])
    om = jnp.linspace(om_lo, om_hi, int(n_om))
    du_dom_vec = jnp.vectorize(jax.grad(u_fn, argnums=0))
    u_vec = jnp.vectorize(u_fn)

    u_a = u_vec(om, jnp.full_like(om, w_lo))
    u_b = u_vec(om, jnp.full_like(om, w_hi))
    slope_a = du_dom_vec(om, jnp.full_like(om, w_lo))
    if not (bool(jnp.isfinite(u_a).all()) and bool(jnp.isfinite(u_b).all())
            and bool(jnp.isfinite(slope_a).all())):
        raise ValueError(
            "endpoint curve u_a/u_b non-finite on the validation grid; the "
            "u-first map is unusable on this box.")

    sign_a = 1.0 if float(u_a[-1]) >= float(u_a[0]) else -1.0
    floor = float(u_a[0]) if sign_a > 0 else float(u_a[-1])
    ceiling = float(jnp.min(u_b))
    report["u_a_sign"] = sign_a
    report["u_band_floor"] = floor
    report["u_band_ceiling"] = ceiling
    if not (ceiling > floor):
        raise ValueError(
            f"empty u-band: floor u_a(start)={floor} >= ceiling min(u_b)="
            f"{ceiling}; the banded u-first construction does not apply to "
            f"this u_fn/box. Report: {report}")

    # STRICTLY above the floor: the branch-start endpoint has u_a == floor and
    # may legitimately descend into a below-band dip (the DSPL curve does);
    # only points above the floor can spoil single-crossing within the band.
    above = u_a > floor
    viol = jnp.where(above, sign_a * slope_a, jnp.inf)
    worst = float(jnp.min(viol))
    n_flip_beyond = int(jnp.sum(viol < -float(curve_atol)))
    report["u_a_min_signed_slope_above_floor"] = worst
    report["u_a_n_flips_beyond_atol"] = n_flip_beyond
    if n_flip_beyond > 0:
        raise ValueError(
            f"endpoint curve u_a is not single-crossing above the band floor: "
            f"{n_flip_beyond} grid point(s) with signed slope < -curve_atol="
            f"{curve_atol} (worst {worst:.3e}) where u_a >= floor={floor}. "
            f"Report: {report}")

    # Excluded prior volume: grid measure of u outside the band over the box.
    w = jnp.linspace(w_lo, w_hi, int(n_w))
    om_mesh, w_mesh = jnp.meshgrid(om, w, indexing="ij")
    u_grid = u_vec(om_mesh, w_mesh)
    report["excluded_prior_volume_frac"] = float(
        jnp.mean((u_grid < floor) | (u_grid > ceiling)))
    report["curve_atol"] = float(curve_atol)
    return report


class UFirstRatioCoordsBijector(tfb.Bijector):
    """Banded u-first diffeomorphism ``R^2 -> {theta in box : u(theta) in band}``
    (section doc). Event ndims 1. ``_forward`` runs one Om0-root solve + the w0
    solve; ``_inverse`` is analytic modulo the Om0-root solve; both log-det
    directions are analytic."""

    def __init__(
        self,
        u_fn: Callable,
        om0_bounds: Tuple[float, float],
        w0_bounds: Tuple[float, float],
        *,
        n_bisect: int = 80,
        n_band_grid: int = 201,
        validate_args: bool = False,
        name: str = "u_first_ratio_coords",
    ):
        parameters = dict(locals())
        self._om_lo, self._om_hi = float(om0_bounds[0]), float(om0_bounds[1])
        self._w_lo, self._w_hi = float(w0_bounds[0]), float(w0_bounds[1])
        if not (self._om_hi > self._om_lo) or not (self._w_hi > self._w_lo):
            raise ValueError(
                f"bounds must be increasing; got om0={om0_bounds}, w0={w0_bounds}.")
        if int(n_bisect) < 50:
            raise ValueError(
                f"n_bisect={n_bisect} < 50 cannot reach float64 resolution; "
                f"use >= 50.")
        self._u_scalar = u_fn
        self._u = jnp.vectorize(u_fn)
        self._du_dw = jnp.vectorize(jax.grad(u_fn, argnums=1))
        self._solve_w = jnp.vectorize(
            _make_w_solver(u_fn, self._w_lo, self._w_hi, int(n_bisect)))
        # Om0-root solver on the u_a curve: the w-solver machinery with swapped
        # arguments — solve om in [om_lo, om_hi] with u_fn(om, w_lo) = u.
        self._solve_om = jnp.vectorize(_make_w_solver(
            lambda w_end, om: u_fn(om, w_end),
            self._om_lo, self._om_hi, int(n_bisect)))

        # The band (concrete, eager; see section doc). Curves evaluated on a
        # construction grid — validate_u_first_ratio_coords is the authority
        # on whether this shape family applies; here we only re-derive the
        # numbers and hard-fail on an empty band.
        om_grid = jnp.linspace(self._om_lo, self._om_hi, int(n_band_grid))
        u_a = self._u(om_grid, jnp.full_like(om_grid, self._w_lo))
        u_b = self._u(om_grid, jnp.full_like(om_grid, self._w_hi))
        self._sign_a = 1.0 if float(u_a[-1]) >= float(u_a[0]) else -1.0
        self._u_lo_band = float(u_a[0]) if self._sign_a > 0 else float(u_a[-1])
        self._u_hi_band = float(jnp.min(u_b))
        # u_a value at the branch's FAR end: beyond it the u_a constraint is
        # inactive and the Om0-interval endpoint is the box edge.
        self._u_a_far = float(u_a[-1]) if self._sign_a > 0 else float(u_a[0])
        if not (self._u_hi_band > self._u_lo_band):
            raise ValueError(
                f"empty u-band [{self._u_lo_band}, {self._u_hi_band}]; run "
                "validate_u_first_ratio_coords for the full report.")
        super().__init__(
            forward_min_event_ndims=1,
            is_constant_jacobian=False,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    # -- pieces ----------------------------------------------------------------
    def _om_interval(self, u):
        """Om0-interval of the u-contour within the band.

        Only the u_a constraint can bind inside the band (the ceiling is
        min u_b). A rising u_a bounds Om0 above by its root; a falling one
        bounds it below. The root is used only where the constraint is active
        (u on the branch side of u_a's far end) — `jnp.where` on that concrete
        activity test keeps clamped-root garbage gradients out of the graph
        (the solver's edge mask is the second line of defense)."""
        dt = jnp.result_type(u)
        # Seed w_end FROM u (u*0 + const) so it inherits u's sharding/varying
        # type: under shard_map a plain-constant custom_vjp argument makes the
        # bwd rule's cotangent type (varying) mismatch the primal input type
        # (unvarying) — the same class of failure as the bisection carry.
        w_end = u * 0 + jnp.asarray(self._w_lo, dt)
        r_a = self._solve_om(w_end, u)
        lo = jnp.full_like(u, self._om_lo)
        hi = jnp.full_like(u, self._om_hi)
        if self._sign_a > 0:
            hi = jnp.where(u < self._u_a_far, jnp.minimum(hi, r_a), hi)
        else:
            lo = jnp.where(u < self._u_a_far, jnp.maximum(lo, r_a), lo)
        return lo, hi

    def _u_from_z1(self, z1):
        return self._u_lo_band + (self._u_hi_band - self._u_lo_band) * ndtr(z1)

    def _fldj_from_parts(self, z1, z2, om_int_lo, om_int_hi, om, w0):
        return (
            math.log(self._u_hi_band - self._u_lo_band) + _log_phi(z1)
            + jnp.log(jnp.abs(om_int_hi - om_int_lo)) + _log_phi(z2)
            - jnp.log(jnp.abs(self._du_dw(om, w0)))
        )

    # -- bijector interface ------------------------------------------------------
    def _forward(self, z):
        z1, z2 = z[..., 0], z[..., 1]
        u = self._u_from_z1(z1)
        lo, hi = self._om_interval(u)
        om = lo + (hi - lo) * ndtr(z2)
        w0 = self._solve_w(om, u)
        return jnp.stack([om, w0], axis=-1)

    def _inverse(self, x):
        om, w0 = x[..., 0], x[..., 1]
        u = self._u(om, w0)
        z1 = ndtri(RatioCoordsBijector._clip_unit(
            (u - self._u_lo_band) / (self._u_hi_band - self._u_lo_band)))
        lo, hi = self._om_interval(u)
        z2 = ndtri(RatioCoordsBijector._clip_unit((om - lo) / (hi - lo)))
        return jnp.stack([z1, z2], axis=-1)

    def _forward_log_det_jacobian(self, z):
        z1, z2 = z[..., 0], z[..., 1]
        u = self._u_from_z1(z1)
        lo, hi = self._om_interval(u)
        om = lo + (hi - lo) * ndtr(z2)
        w0 = self._solve_w(om, u)
        return self._fldj_from_parts(z1, z2, lo, hi, om, w0)

    def _inverse_log_det_jacobian(self, x):
        om, w0 = x[..., 0], x[..., 1]
        z = self._inverse(x)
        u = self._u(om, w0)
        lo, hi = self._om_interval(u)
        return -self._fldj_from_parts(z[..., 0], z[..., 1], lo, hi, om, w0)


class UFirstRatioCoordsUniform(tfd.Independent):
    """Uniform prior on the (Om0, w0) box whose event-space bijector is the
    banded u-FIRST ratio-coordinates map (event_shape [2]); Run D counterpart
    of :class:`RatioCoordsUniform` — same drop-in tuple-key usage.

    SUPPORT AMENDMENT (section doc): the sampler can only reach theta with
    ``u(theta)`` inside the band, so this is effectively an UNNORMALIZED
    truncated-uniform prior (``log_prob`` stays the plain box value — the
    truncation constant is irrelevant to MCMC). Quote the validator report's
    ``excluded_prior_volume_frac`` and the band's sigma-distance from the data
    with any run that uses it."""

    def __init__(
        self,
        u_fn: Callable,
        om0_bounds: Tuple[float, float],
        w0_bounds: Tuple[float, float],
        *,
        du_dw_atol: float = 0.0,
        excursion_atol: float = 0.0,
        curve_atol: float = 0.0,
        n_bisect: int = 80,
        n_validation_grid: int = 201,
        skip_validation: bool = False,
        dtype=None,
        validate_args: bool = False,
        name: str = "UFirstRatioCoordsUniform",
    ):
        if skip_validation:
            self.ratio_coords_report = None
        else:
            self.ratio_coords_report = validate_u_first_ratio_coords(
                u_fn, om0_bounds, w0_bounds,
                du_dw_atol=du_dw_atol, excursion_atol=excursion_atol,
                curve_atol=curve_atol,
                n_om=n_validation_grid, n_w=n_validation_grid)
        self._esb = UFirstRatioCoordsBijector(
            u_fn, om0_bounds, w0_bounds, n_bisect=n_bisect,
            n_band_grid=n_validation_grid, validate_args=validate_args)
        if dtype is None:
            dtype = jnp.zeros(()).dtype
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
