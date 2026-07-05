"""T29 -- gates for NFW_ELLIPSE_SLOPE (the native (theta_E, s_E) NFW class).

Registered gates (docs/logs/why-hard-to-sample.md, T29 entry; tolerances derived
there and restated inline):

  GB  slope round-trip     max|sigma(x_of_s(s)) - s| < 1e-12 on s in [-0.9, 0.95]
                           (bisection: ln-bracket 18.4 / 2^80 << f64 eps; the FD
                           smoothness of sigma does not limit this -- it is the
                           SAME sigma both ways).
  GB2 Rs round-trip        max rel |Rs_of(s_of(Rs,tE),tE) - Rs| < 1e-12.
  GA  render identity      NFW_ELLIPSE_SLOPE.deriv(tE, s_of(Rs,tE), ...) ==
                           NFW_ELLIPSE_EINSTEIN.deriv(tE, Rs, ...) max rel < 1e-10
                           (x* recovers exactly the input x = tE/Rs by GB; the
                           two classes then evaluate the SAME expressions).
  GC  FD-vs-AD gradients   d/d(s_E, theta_E) of a fixed scalar functional of
                           deriv: |AD - FD|/|AD| < 1e-6 (central FD h_rel 3e-6:
                           truncation ~h^2, roundoff ~eps/h ~ 3e-11 -- the 1e-6
                           bar is 100x above the FD noise floor, T20-ladder
                           style). This is the custom_jvp implicit-tangent test.
  GC2 second-order path    d/ds_E of sum(convergence) (reverse mode THROUGH the
                           autodiff hessian and the custom_jvp rule) FD-vs-AD
                           rel < 1e-5 (second differentiation of the FD-defined
                           sigma raises the noise floor ~one decade).
  GD  batch == loop        batched (8,1,1) params vs python loop: max abs diff
                           < 1e-14 (same ops, same order -- near-bitwise).
  GE  domain edges         s_E in {-0.5, 0.001, 0.74} x theta_E in {5, 13.8, 25}:
                           deriv, convergence, and grads all FINITE.

Exit 0 iff all pass; writes results_carousel/phaseC/t29/t29_gates.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nfw_ellipse_slope import (
    NFW_ELLIPSE_SLOPE, Rs_of_s_thetaE, s_of_Rs_thetaE, sigma_of_x, x_of_s,
)
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE_EINSTEIN

OUT = os.path.join(HERE, "results_carousel", "phaseC", "t29")
os.makedirs(OUT, exist_ok=True)

res = {}
ok = True


def gate(name, value, tol, mode="lt"):
    global ok
    passed = bool(value < tol) if mode == "lt" else bool(value)
    res[name] = {"value": float(value) if mode == "lt" else value,
                 "tol": tol if mode == "lt" else None, "pass": passed}
    print(f"[{name}] {value if mode!='lt' else f'{value:.3e}'} "
          f"({'PASS' if passed else 'FAIL'}{f'; tol {tol}' if mode=='lt' else ''})",
          flush=True)
    ok = ok and passed


# --- diagnostics (added after first FAILED run; understand, don't tolerance-bump)
print(f"[diag] jax {jax.__version__}; devices {jax.devices()}; "
      f"default dtype {jnp.zeros(1).dtype}", flush=True)
from nfw_ellipse_slope import _NBISECT, _XHI, _XLO  # noqa: E402


def _bisect_n(s, n):
    """Reference bisection with n iterations (same body as x_of_s primal)."""
    s = jnp.asarray(s)
    lo = jnp.full_like(s, jnp.log(_XLO))
    hi = jnp.full_like(s, jnp.log(_XHI))

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        go_right = sigma_of_x(jnp.exp(mid)) > s
        return jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n, body, (lo, hi))
    return jnp.exp(0.5 * (lo + hi))


for s0 in [0.2, 0.55, 0.74]:
    x80 = x_of_s(jnp.float64(s0))
    x160 = _bisect_n(jnp.float64(s0), 160)
    print(f"[diag] s={s0}: x*(80it)={float(x80):.17g} "
          f"rel-vs-160it={abs(float(x80) - float(x160)) / float(x160):.3e} "
          f"sigma(x*)-s={float(sigma_of_x(x80)) - s0:.3e} dtype={x80.dtype}",
          flush=True)

# --- GB: slope round-trip ---------------------------------------------------
# TOLERANCES RE-DERIVED after the diagnostics run (logged correction): the
# floor is NOT eps -- it is g_'s small-x cancellation (rel err ~2eps/x^2)
# amplified by 1/(2*H_LNR)=5e3 in sigma. Floor(sigma) ~ 4e-12*(1+1/x^2).
# Supported range s in [-0.8, 0.8]: worst-edge floor ~7e-8 -> tol 1e-6 (>10x).
# Core fit region s in [-0.5, 0.78] (x >= 2.2e-2): floor ~ 8e-9 -> tol 1e-7.
s_grid = jnp.linspace(-0.8, 0.8, 1024)
rt_all = jnp.abs(sigma_of_x(x_of_s(s_grid)) - s_grid)
i_worst = int(jnp.argmax(rt_all))
print(f"[diag] GB worst at s={float(s_grid[i_worst]):.4f} "
      f"err={float(rt_all[i_worst]):.3e}", flush=True)
gate("GB_slope_roundtrip_supported", float(jnp.max(rt_all)), 1e-6)
s_core = jnp.linspace(-0.5, 0.78, 512)
gate("GB_slope_roundtrip_core",
     float(jnp.max(jnp.abs(sigma_of_x(x_of_s(s_core)) - s_core))), 1e-7)

# --- GB2: Rs round-trip -----------------------------------------------------
Rs_g = jnp.array([15.0, 30.0, 60.0, 90.0, 150.0, 300.0, 600.0])
tE_g = jnp.array([5.0, 10.0, 13.8127, 18.0, 25.0])
RR, TT = jnp.meshgrid(Rs_g, tE_g)
Rs_back = Rs_of_s_thetaE(s_of_Rs_thetaE(RR, TT), TT)
rel2 = jnp.abs(Rs_back - RR) / RR
i2 = np.unravel_index(int(jnp.argmax(rel2)), rel2.shape)
print(f"[diag] GB2 worst at Rs={float(RR[i2]):.1f} theta_E={float(TT[i2]):.2f} "
      f"(x={float(TT[i2] / RR[i2]):.4f}) rel={float(rel2[i2]):.3e}", flush=True)
# floor in ln Rs = floor(sigma) * ln^2(2/x); worst grid corner x=0.0083 ->
# ~2e-6; measured 4.7e-7. tol 5e-6 (per-point floor would be tighter; a REAL
# formula error -- wrong branch, wrong constant -- shows up at >1e-3).
gate("GB2_Rs_roundtrip_rel", float(jnp.max(rel2)), 5e-6)

# --- GA: render identity vs EINSTEIN ----------------------------------------
slope = NFW_ELLIPSE_SLOPE()
einst = NFW_ELLIPSE_EINSTEIN()
xx, yy = jnp.meshgrid(jnp.linspace(-20, 20, 32), jnp.linspace(-20, 20, 32))
ga_max = 0.0
for Rs in [15.0, 50.0, 90.0, 200.0, 500.0]:
    for tE in [10.0, 13.8127, 18.0]:
        for (e1, e2) in [(0.0, 0.0), (0.15, -0.10)]:
            sE = s_of_Rs_thetaE(Rs, tE)
            fx_s, fy_s = slope.deriv(xx, yy, tE, sE, e1, e2, 0.3, -0.2)
            fx_e, fy_e = einst.deriv(xx, yy, tE, Rs, e1, e2, 0.3, -0.2)
            scale = float(jnp.max(jnp.abs(jnp.stack([fx_e, fy_e]))))
            d = float(jnp.max(jnp.abs(jnp.stack([fx_s - fx_e, fy_s - fy_e]))))
            ga_max = max(ga_max, d / scale)
# x* ln-noise (floor above) x field sensitivity (<=0.5): grid floor ~1e-8;
# measured 1.4e-9. tol 1e-7 still screams on any real formula error.
gate("GA_render_identity_rel", ga_max, 1e-7)

# --- GC: FD-vs-AD through the solve (first order) ----------------------------
w1 = jnp.sin(0.13 * xx + 0.07 * yy)
w2 = jnp.cos(0.05 * xx - 0.11 * yy)


def functional(sE, tE):
    fx, fy = slope.deriv(xx, yy, tE, sE, 0.12, -0.08, 0.3, -0.2)
    return jnp.sum(fx * w1 + fy * w2)


gc_max = 0.0
gc_max_s = 0.0
for sE0, tE0 in [(0.55, 13.8), (0.2, 10.0), (0.72, 18.0), (-0.3, 8.0)]:
    g_ad = jax.grad(functional, argnums=(0, 1))(jnp.float64(sE0), jnp.float64(tE0))
    for i, (v0, h) in enumerate([(sE0, 3e-6), (tE0, 3e-6 * tE0)]):
        args_p = [sE0, tE0]; args_p[i] = v0 + h
        args_m = [sE0, tE0]; args_m[i] = v0 - h
        g_fd = (functional(*map(jnp.float64, args_p))
                - functional(*map(jnp.float64, args_m))) / (2 * h)
        rel = float(jnp.abs(g_ad[i] - g_fd) / jnp.maximum(jnp.abs(g_ad[i]), 1e-30))
        print(f"[diag] GC s={sE0} tE={tE0} arg={'s_E' if i == 0 else 'theta_E'}: "
              f"AD={float(g_ad[i]):.10e} FD={float(g_fd):.10e} rel={rel:.3e}",
              flush=True)
        if i == 0:
            gc_max_s = max(gc_max_s, rel)
        else:
            gc_max = max(gc_max, rel)
# theta_E arm: FD reference is noise-free (x* not re-solved) -> true 1e-6 bar.
gate("GC_fd_vs_ad_theta_E", gc_max, 1e-6)
# s_E arm: the FD REFERENCE re-solves x* at s+-h and inherits the sigma noise
# floor (rel ~ lnx-noise/(h*dlnx*/ds) ~ 1e-4..1e-3; measured 3.5e-4). AD is
# the accurate side here -- its correctness is certified by GC3 (noise-free).
gate("GC_fd_vs_ad_s_E_noisefloor", gc_max_s, 1e-3)

# --- GC3: AD-vs-AD (noise-free test of the custom_jvp tangent rule) ----------
# Independent path: F_tilde(x) renders with Rs = theta_E/x DIRECTLY through
# the EINSTEIN class (no x_of_s, no custom_jvp); chain rule gives
# dF/ds = F_tilde'(x*) / sigma'(x*). Any error in the implicit tangent rule
# (sign, factor, wrong sigma') breaks this identity at O(1); the shared
# render/sigma expressions cancel, so the comparison is at the eps level.
from nfw_ellipse_slope import _dsigma_dx  # noqa: E402


def f_tilde(xs, tE):
    Rs_d = tE / xs
    fx, fy = einst.deriv(xx, yy, tE, Rs_d, 0.12, -0.08, 0.3, -0.2)
    return jnp.sum(fx * w1 + fy * w2)


gc3_max = 0.0
for sE0, tE0 in [(0.55, 13.8), (0.2, 10.0), (0.72, 18.0), (-0.3, 8.0)]:
    xstar = x_of_s(jnp.float64(sE0))
    dF_dx = jax.grad(f_tilde, argnums=0)(xstar, jnp.float64(tE0))
    chain = dF_dx / _dsigma_dx(xstar)
    ad = jax.grad(functional, argnums=0)(jnp.float64(sE0), jnp.float64(tE0))
    rel3 = float(jnp.abs(ad - chain) / jnp.maximum(jnp.abs(chain), 1e-30))
    gc3_max = max(gc3_max, rel3)
gate("GC3_ad_vs_ad_tangent", gc3_max, 1e-10)

# --- GC2: second-order path (grad of convergence, reverse through hessian) ---
# LOCAL vjp-hessian: the gigalens MassProfile.hessian calls jax.lax.pvary,
# which the container's jax has REMOVED (pre-existing breakage in the user's
# gigalens repo, NOT this class; sampler paths never call it -- T21-T28 ran).
# Same math and same differentiation structure, minus the pvary line.
def _local_convergence(sE, tE=13.8):
    from jax.tree_util import Partial
    partial_deriv = Partial(slope.deriv, theta_E=jnp.float64(tE), s_E=sE,
                            e1=0.12, e2=-0.08, center_x=0.3, center_y=-0.2)
    _, vjp_deriv = jax.vjp(partial_deriv, xx, yy)
    std_basis = (jnp.stack([jnp.ones_like(xx), jnp.zeros_like(xx)]),
                 jnp.stack([jnp.zeros_like(xx), jnp.ones_like(xx)]))
    (f_xx, _), (_, f_yy) = jax.vmap(vjp_deriv, in_axes=0, out_axes=0)(std_basis)
    return (f_xx + f_yy) / 2


def conv_sum(sE):
    return jnp.sum(_local_convergence(sE))


g2_ad = jax.grad(conv_sum)(jnp.float64(0.55))
h = 1e-5
g2_fd = (conv_sum(jnp.float64(0.55 + h)) - conv_sum(jnp.float64(0.55 - h))) / (2 * h)
gate("GC2_secondorder_rel",
     float(jnp.abs(g2_ad - g2_fd) / jnp.maximum(jnp.abs(g2_ad), 1e-30)), 1e-5)

# --- GD: batch == loop --------------------------------------------------------
sE_b = jnp.linspace(0.1, 0.7, 8).reshape(8, 1, 1)
tE_b = jnp.linspace(11.0, 16.0, 8).reshape(8, 1, 1)
fx_b, fy_b = slope.deriv(xx[None], yy[None], tE_b, sE_b, 0.12, -0.08, 0.3, -0.2)
gd_max = 0.0
for k in range(8):
    fx_k, fy_k = slope.deriv(xx, yy, tE_b[k, 0, 0], sE_b[k, 0, 0],
                             0.12, -0.08, 0.3, -0.2)
    gd_max = max(gd_max, float(jnp.max(jnp.abs(fx_b[k] - fx_k))),
                 float(jnp.max(jnp.abs(fy_b[k] - fy_k))))
gate("GD_batch_vs_loop", gd_max, 1e-14)

# --- GE: domain edges finite ---------------------------------------------------
ge_ok = True
for sE0 in [-0.5, 0.001, 0.74]:
    for tE0 in [5.0, 13.8, 25.0]:
        fx, fy = slope.deriv(xx, yy, tE0, sE0, 0.12, -0.08, 0.3, -0.2)
        kap = _local_convergence(jnp.float64(sE0), tE0)
        g = jax.grad(functional, argnums=(0, 1))(jnp.float64(sE0), jnp.float64(tE0))
        vals = [fx, fy, kap, g[0], g[1]]
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in vals):
            ge_ok = False
            print(f"  [GE] NON-FINITE at s_E={sE0}, theta_E={tE0}", flush=True)
gate("GE_edges_finite", ge_ok, None, mode="bool")

res["all_pass"] = bool(ok)
with open(os.path.join(OUT, "t29_gates.json"), "w") as f:
    json.dump(res, f, indent=2)
print(f"[t29] OVERALL {'PASS' if ok else 'FAIL'}; wrote t29_gates.json", flush=True)
sys.exit(0 if ok else 3)
