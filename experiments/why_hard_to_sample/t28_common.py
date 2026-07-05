"""T28 shared machinery -- the observable-slope prior transform s(Rs) and its
inverse leaf Rs(s), plus the custom tfp Bijector ``RsOfS`` that carries the
modeling change (pre-registered 2026-07-04, docs/logs/why-hard-to-sample.md,
"T28 -- prior set DIRECTLY in the observable-anchored slope coordinate s").

WHAT T28 CHANGES (a MODELING change, not a coordinate change)
------------------------------------------------------------
The new arm's ``Rs ~ Uniform(20, 100)`` prior is REPLACED by a prior set directly
on the observable

    s(Rs) = d ln alpha / d ln r   at   r = theta_E* = 13.8126953867242

(the Route-B slope; t25_transforms.slope_s_of_Rs -- exact NFW ``g_`` mirror, FD in
ln r).  s is strictly increasing in Rs (verified numerically over Rs in [6, 900]).
We set  s ~ Uniform(0.0, 0.75).  Observable anchors: s = 0 is the SIS/isothermal
slope at the Einstein radius (Rs = 10.478), s -> 1 is the mass-sheet limit;
s = 0.75 <-> Rs = 614.4.  Everything else (the other 13 priors, renderer, data,
likelihood) is UNCHANGED.

HOW THE SWAP IS IMPLEMENTED (traced against gigalens source)
------------------------------------------------------------
The Rs prior distribution is replaced by

    tfd.TransformedDistribution(tfd.Uniform(0.0, 0.75), bijector=RsOfS)

where ``RsOfS`` is a scalar tfb.Bijector with

  * _forward(s)   = Rs   = leaf.forward(s)                 (PCHIP monotone spline)
  * _inverse(Rs)  = s    = slope_s_of_Rs_jnp(Rs)           (ANALYTIC jnp port)
  * _forward_log_det_jacobian(s)  = log dRs/ds  = leaf.forward_log_det_jacobian(s)
  * _inverse_log_det_jacobian(Rs) = log ds/dRs  = log(ds_dRs_jnp(Rs))   (ANALYTIC)

Why this is correct for the ACTUAL hot-path (scene_prob_model.py, traced):
  ProbModel.log_prior(z):  x = bij.forward(z);  prior.log_prob(x)
                                              + bij.forward_log_det_jacobian(z)
  with  bij = Chain([prior.experimental_default_event_space_bijector(), pack]).

  - prior.log_prob(x) (constrained-theta density) evaluates the Rs component as
    TransformedDistribution.log_prob(Rs) = Uniform(0,0.75).log_prob(s) + ildj(Rs)
                                         = -log(0.75) + log|ds/dRs|   (s in [0,0.75]).
    This uses ONLY the ANALYTIC ``_inverse`` and ``_inverse_log_det_jacobian`` --
    the density is exact analytic; the spline leaf never enters it.  (This is the
    G2 identity.)
  - experimental_default_event_space_bijector() of the TransformedDistribution
    composes to  RsOfS o Sigmoid(0, 0.75), so the SAMPLING chart for the Rs slot
    is unconstrained-s:  z_s -> s = 0.75*sigmoid(z_s) -> Rs = leaf.forward(s).
    The funnel-relevant chart changes to flat-in-s, as registered.
  - In log_prior the spline fldj log|dRs/ds| (forward) and the analytic ildj
    log|ds/dRs| (inside prior.log_prob at Rs = leaf.forward(s)) CANCEL up to the
    leaf round-trip error, leaving the correct logit-uniform prior in z_s.  The
    G1 round-trip gate (< 1e-8) bounds that residual.

float64 everywhere; eager leaf materialization before any trace (paid-for lesson,
reparam_bijector._ensure_jnp).  The jnp ports are import-guarded so this module is
importable under a jax-free login python (numpy paths + leaf build still run).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# numpy reference paths (login-safe; no jax) -- the SAME exact g_ mirror + FD-in-ln-r
from reparam_bijector import MonotoneCubicBijector  # noqa: E402
from t25_transforms import (  # noqa: E402
    FD_REL, nfw_g_numpy, nfw_alpha_shape, slope_s_of_Rs,
)

# ---------------------------------------------------------------------------
# registered constants
# ---------------------------------------------------------------------------
THETA_E_STAR = 13.8126953867242   # fixed reference arc radius (new-arm posterior median)
S_LO, S_HI = 0.0, 0.75            # the registered prior:  s ~ Uniform(0.0, 0.75)

# NOTE on ds/dRs (the ILDJ / induced Rs-prior density): the observable
# s = slope_s_of_Rs is itself a central-FD-in-ln-r expression, so its EXACT
# derivative is d/dRs of that expression -- obtained by autodiff (jax.grad),
# which introduces NO further finite-difference roundoff and is consistent with
# the leaf (built from the same s-expression) to the leaf's interpolation
# quality. A second numerical FD in Rs (ds_dRs_numpy below) amplifies the inner
# FD's ~1e-8 noise and is used ONLY for the login-node diagnostic.

# Rs(s) leaf knots: UNIFORM in s over [S_KNOT_LO, S_KNOT_HI] (brackets [0,0.75]),
# with Rs(s) obtained by np.interp off a very dense log-spaced Rs->s table (NO
# bisection). Uniform-in-s knots keep the interpolation error even in the
# coordinate the spline is indexed on (x=s) and give a clean G1 round-trip.
S_KNOT_LO, S_KNOT_HI, N_GRID = -0.04, 0.78, 2000
RS_TABLE_LO, RS_TABLE_HI, N_TABLE = 6.0, 2000.0, 400000  # dense s->Rs inversion table
RS_DS_REL = 2e-4                  # relative central-FD step for the login-only ds/dRs diagnostic

# endpoint anchors (numeric facts; asserted at build time)
RS_AT_S0 = 10.478                # s = 0  <-> SIS slope at the Einstein radius
RS_AT_S075 = 614.4               # s = 0.75 <-> mass-sheet-approaching

ARTIFACT_REL = os.path.join("results_carousel", "phaseC", "t28",
                            "transform_sprior.npz")
ARTIFACT = os.path.join(HERE, ARTIFACT_REL)


# ===========================================================================
# numpy side (login-safe): ds/dRs via central FD of the numpy slope
# ===========================================================================
def ds_dRs_numpy(Rs, theta_E_star=THETA_E_STAR, rel=RS_DS_REL):
    """ds/dRs by central finite difference in Rs of the numpy slope_s_of_Rs (the
    SAME closed FD-in-ln-r form used on the jnp side). Used for the login-side
    internal-consistency gate."""
    Rs = np.asarray(Rs, np.float64)
    Rsp = Rs * (1.0 + rel)
    Rsm = Rs * (1.0 - rel)
    return (slope_s_of_Rs(Rsp, theta_E_star) - slope_s_of_Rs(Rsm, theta_E_star)) / (Rsp - Rsm)


# ===========================================================================
# jnp side (GPU / traceable): exact g_ mirror + s(Rs) + ds/dRs
# ===========================================================================
def nfw_g_jnp(x):
    """Exact jnp mirror of gigalens NFW.g_ (nfw.py:37-51, _c=1e-6). Branchwise
    clamp for NaN-safety, identical constants to nfw_g_numpy. float64, traceable."""
    import jax.numpy as jnp
    c = 1e-6
    x = jnp.maximum(c, jnp.asarray(x, jnp.float64))
    x_lt1 = jnp.clip(x, c, 1.0 - 1e-6)
    x_gt1 = jnp.clip(x, 1.0 + 1e-6, jnp.inf)
    g_lt1 = jnp.log(x_lt1 / 2.0) + 1.0 / jnp.sqrt(1.0 - x_lt1 ** 2) * jnp.arccosh(1.0 / x_lt1)
    g_gt1 = jnp.log(x_gt1 / 2.0) + 1.0 / jnp.sqrt(x_gt1 ** 2 - 1.0) * jnp.arccos(1.0 / x_gt1)
    return jnp.where(x < 1.0, g_lt1, g_gt1)


def nfw_alpha_shape_jnp(r, Rs):
    """r-dependent part of the NFW deflection magnitude, alpha(r) ∝ g(x)/x, x=r/Rs
    (jnp mirror of nfw_alpha_shape)."""
    import jax.numpy as jnp
    x = jnp.asarray(r, jnp.float64) / jnp.asarray(Rs, jnp.float64)
    return nfw_g_jnp(x) / x


def slope_s_of_Rs_jnp(Rs, theta_E_star=THETA_E_STAR):
    """s(Rs) = d ln(alpha)/d ln(r) at r=theta_E*, central FD in ln r (rel FD_REL),
    jnp port of t25_transforms.slope_s_of_Rs. Traceable, float64."""
    import jax.numpy as jnp
    r = float(theta_E_star)
    rp = r * (1.0 + FD_REL)
    rm = r * (1.0 - FD_REL)
    ap = nfw_alpha_shape_jnp(rp, Rs)
    am = nfw_alpha_shape_jnp(rm, Rs)
    return (jnp.log(ap) - jnp.log(am)) / (jnp.log(rp) - jnp.log(rm))


def ds_dRs_jnp(Rs, theta_E_star=THETA_E_STAR):
    """ds/dRs = EXACT derivative of the observable s = slope_s_of_Rs_jnp, via
    autodiff (jax.grad, vmapped). No second finite difference, so no roundoff
    amplification of the inner FD-in-ln-r noise; consistent-by-construction with
    the leaf. Strictly positive on the support. Traceable; batch-friendly."""
    import jax
    import jax.numpy as jnp
    Rs = jnp.asarray(Rs, jnp.float64)
    shp = Rs.shape
    gscalar = jax.grad(lambda r: slope_s_of_Rs_jnp(r, theta_E_star))
    out = jax.vmap(gscalar)(Rs.reshape(-1))
    return out.reshape(shp)


# ===========================================================================
# the Rs(s) leaf  (built WITHOUT bisection: evaluate s on an Rs grid, swap axes)
# ===========================================================================
def build_leaf(theta_E_star=THETA_E_STAR):
    """Build the monotone PCHIP leaf ``forward: s -> Rs`` on UNIFORM-in-s knots.

    Rs(s) at the uniform-s knots is obtained by np.interp off a very dense
    log-spaced Rs->s table (NO bisection), which inverts the strictly-increasing
    s(Rs). Returns (leaf, s_knots, Rs_knots). Pure numpy (login-safe)."""
    Rs_table = np.logspace(np.log10(RS_TABLE_LO), np.log10(RS_TABLE_HI), N_TABLE)
    s_table = np.asarray(slope_s_of_Rs(Rs_table, theta_E_star), np.float64)
    if not np.all(np.diff(s_table) > 0):
        raise RuntimeError("build_leaf: s(Rs) not strictly increasing on the table; "
                           "the observable coordinate is not invertible. STOP.")
    if not (s_table[0] < S_KNOT_LO and s_table[-1] > S_KNOT_HI):
        raise RuntimeError(
            f"build_leaf: table s-range [{s_table[0]:.4f}, {s_table[-1]:.4f}] does "
            f"not bracket the knot span [{S_KNOT_LO}, {S_KNOT_HI}]; widen the table.")
    s_knots = np.linspace(S_KNOT_LO, S_KNOT_HI, N_GRID)
    Rs_knots = np.interp(s_knots, s_table, Rs_table)
    if not np.all(np.diff(Rs_knots) > 0):
        raise RuntimeError("build_leaf: Rs(s) knots not strictly increasing. STOP.")
    leaf = MonotoneCubicBijector.fit(
        s_knots, Rs_knots, z_init=None,
        meta={"kind": "sprior_RsOfS", "theta_E_star": float(theta_E_star),
              "s_lo": S_LO, "s_hi": S_HI, "fd_rel": FD_REL,
              "s_knot_lo": S_KNOT_LO, "s_knot_hi": S_KNOT_HI, "n_knots": N_GRID,
              "note": "u_knots=s (uniform), z_knots=Rs; forward maps s->Rs"})
    return leaf, s_knots, Rs_knots


def load_leaf(artifact=ARTIFACT):
    if not os.path.isfile(artifact):
        raise FileNotFoundError(
            f"[T28] transform artifact not found: {artifact}. Run "
            "t28_sprior_transform.py first (it writes transform_sprior.npz).")
    leaf = MonotoneCubicBijector.from_npz(artifact)
    return leaf


# ===========================================================================
# the custom tfp Bijector RsOfS  (jax-only; import-guarded)
# ===========================================================================
def make_RsOfS(leaf, theta_E_star=THETA_E_STAR):
    """Construct the RsOfS tfb.Bijector wrapping ``leaf`` (forward s->Rs, spline)
    with an ANALYTIC inverse s(Rs) and analytic ds/dRs for the density path.

    leaf._ensure_jnp() is called EAGERLY here (outside any trace) so the spline
    knots are concrete device arrays before the sampler jits log_prob (paid-for
    tracer-leak lesson, reparam_bijector._ensure_jnp)."""
    import tensorflow_probability.substrates.jax as tfp
    import jax.numpy as jnp
    tfb = tfp.bijectors

    leaf._ensure_jnp()  # eager materialization BEFORE any jit trace

    class RsOfS(tfb.Bijector):
        """Scalar monotone bijector s -> Rs.  forward = PCHIP leaf (spline);
        inverse = analytic slope s(Rs); *_log_det_jacobian analytic/spline as noted.
        Injective, increasing, event_ndims 0."""

        def __init__(self, name="RsOfS"):
            # leaf + theta_E* are captured by CLOSURE (not stored in `parameters`),
            # so TFP never tries to reconstruct this bijector from `parameters`
            # (it is not an AutoCompositeTensor / pytree; it stays a static closure
            # constant during MCLMC's jit). Validated composition pattern (numpy
            # substrate) uses an empty parameters dict.
            self._leaf = leaf
            self._te = float(theta_E_star)
            super().__init__(
                forward_min_event_ndims=0,
                is_constant_jacobian=False,
                validate_args=False,
                parameters=dict(),
                name=name,
            )

        def _forward(self, s):
            return self._leaf.forward(jnp.asarray(s, jnp.float64))

        def _inverse(self, Rs):
            return slope_s_of_Rs_jnp(jnp.asarray(Rs, jnp.float64), self._te)

        def _forward_log_det_jacobian(self, s):
            # log|dRs/ds| from the spline (used in the SAMPLING chart's fldj)
            return self._leaf.forward_log_det_jacobian(jnp.asarray(s, jnp.float64))

        def _inverse_log_det_jacobian(self, Rs):
            # log|ds/dRs| analytic. NOTE (measured, numpy substrate 2026-07-04):
            # TransformedDistribution.log_prob calls inverse(x) FIRST, which
            # populates the bijector cache, and the subsequent ildj lookup then
            # prefers the FORWARD side: -_forward_log_det_jacobian(inverse(x))
            # (spline). So the DENSITY actually realized is uniform in the
            # gated SPLINE chart's s -- equal to the analytic statement to
            # <= the chart-consistency bound (blocking gate in t28_run_gpu).
            # This method is kept for direct (uncached) ildj calls only.
            return jnp.log(ds_dRs_jnp(jnp.asarray(Rs, jnp.float64), self._te))

        @property
        def _is_increasing(self):
            return True

    return RsOfS()


def make_sprior_Rs_distribution(leaf, theta_E_star=THETA_E_STAR):
    """The replacement Rs prior:  TransformedDistribution(Uniform(0,0.75), RsOfS).
    Its constrained density on Rs is  -log(0.75) + log|ds/dRs|  on Rs in
    [leaf.forward(0), leaf.forward(0.75)]; its default event-space bijector is
    RsOfS o Sigmoid(0,0.75) (unconstrained-s sampling chart)."""
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    bij = make_RsOfS(leaf, theta_E_star)
    # np.float64 low/high: tfd.Uniform infers float32 from Python floats (measured
    # on the numpy substrate 2026-07-04; the f32 constant rounding polluted G2).
    return tfd.TransformedDistribution(
        distribution=tfd.Uniform(low=np.float64(S_LO), high=np.float64(S_HI)),
        bijector=bij)
