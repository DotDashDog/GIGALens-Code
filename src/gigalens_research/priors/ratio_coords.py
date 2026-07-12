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
        return (-(du_dom / du_dw) * g, g / du_dw)

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
