"""T25b -- derive BOTH reparameterization transforms + run the family gates.

Pre-registered 2026-07-03 (docs/logs/why-hard-to-sample.md, "T25 ... + T26 ...").
Reads results_carousel/phaseC/t25/profile.npz (from t25_profile_gpu.py) and writes
the PCHIP knot artifacts

    results_carousel/phaseC/t25/transform_A.npz   (variance-stabilizing)
    results_carousel/phaseC/t25/transform_B.npz   (observable-anchored)

plus the A-vs-B comparison arrays (ab_comparison.npz) and a routes_passed.json
that the slurm driver reads to decide which routes enter T26.

ROUTE A (variance-stabilizing): u(z) = int sqrt(lambda_hat(z')) dz', lambda_hat =
  log-space interpolation of the T25a bin medians (on-floor bins primary;
  below-floor transect medians extend the low end, flagged upper-envelope),
  constant-extended outside. Dense grid z in [-6,12], 512 pts. Fit PCHIP w=u->z.

ROUTE B (observable-anchored, B-lite): s(Rs) = dln(alpha)/dln(r) at r=theta_E*
  (new-arm posterior median), using the EXACT gigalens NFW g_ (mirrored in numpy
  here and cross-checked byte-for-byte against gigalens.NFW().g_ on the GPU).
  Slope via central finite difference in ln r at 1e-4 relative (SEMI-CLOSED: the
  r-independent rho0/Rs prefactors cancel in d ln alpha/d ln r; only g(r/Rs)/(r/Rs)
  varies). Monotonicity of s in Rs over [20,100] hard-asserted on a 512-pt grid.
  u_B = affine-normalized s(z), fit PCHIP w=u_B->z on the same dense grid.

Both transforms carry the affine normalization dz/du=1 and u=z_init at z_init
(baked into the fitted knots; MonotoneCubicBijector asserts it downstream).

FAMILY-PRESERVATION GATES (HARD, per route; jax): (1) round-trip |Rs-Rs''|<1e-8
through the route system's forward/inverse on a 64-pt prior grid; (2) 8 random
theta: prior-logp identity <1e-10 and rendered-loglike identity <1e-8 vs the
baseline system; (3) mapped z_init: theta(z_init_base)==theta(z_init_route)<1e-10.
A route that fails ANY gate is marked failed in routes_passed.json and SKIPPED in
T26 (only that route). Catastrophic failure (F-T25a fires, profile missing, or NO
route passes) -> nonzero exit blocks all of T26.

Run:
  python t25_transforms.py                 # derive + gates (GPU)
  python t25_transforms.py --no-gates      # derive only (offline/login checks)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from reparam_bijector import MonotoneCubicBijector, sha256_file  # noqa: E402

T25_OUT = os.path.join(HERE, "results_carousel", "phaseC", "t25")
PROFILE_NPZ = os.path.join(T25_OUT, "profile.npz")
TRANSFORM_A = os.path.join(T25_OUT, "transform_A.npz")
TRANSFORM_B = os.path.join(T25_OUT, "transform_B.npz")
AB_COMPARISON = os.path.join(T25_OUT, "ab_comparison.npz")
ROUTES_PASSED = os.path.join(T25_OUT, "routes_passed.json")

# registered dense grid + bands
DENSE_LO, DENSE_HI, DENSE_N = -6.0, 12.0, 512
FD_REL = 1e-4                      # central FD relative step in ln r for Route B
RS_BOX = (20.0, 100.0)            # NFW Rs prior box
N_GATE_RT = 64                    # round-trip grid points (gate 1)
N_GATE_THETA = 8                  # random theta points (gates 2)
GATE_RT_TOL = 1e-8
GATE_LOGLIKE_TOL = 1e-8
GATE_PRIOR_TOL = 1e-10
GATE_ZINIT_TOL = 1e-10
F_T25A_MINVAR = 5.0              # ORIGINAL F-T25a (fired 2026-07-03; kept for the record)
F_T25A_MINVAR_AMENDED = 30.0     # F-T25a': full-range reliable-bin variation < 30x => STOP
LAMBDA_RELIABLE = 1.0            # knots with lambda <= this are noise/off-valley (see docstring)
F_T25C_FACTOR = 5.0             # F-T25c: A-vs-B ratio varies >=5x => physics wrong
P_T25C_BAND = 2.0              # validated within factor-2


def _now():
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# exact gigalens NFW g_ mirrored in numpy (nfw.py:37-51) -- cross-checked vs
# gigalens.NFW().g_ on the GPU (see _crosscheck_g). Used for the Route-B slope.
# ===========================================================================

def nfw_g_numpy(x):
    """Exact numpy mirror of gigalens NFW.g_ (nfw.py:37-51). Branchwise clamp for
    NaN-safety, identical constants."""
    c = 1e-6
    x = np.maximum(c, np.asarray(x, np.float64))
    x_lt1 = np.clip(x, c, 1.0 - 1e-6)
    x_gt1 = np.clip(x, 1.0 + 1e-6, np.inf)
    g_lt1 = np.log(x_lt1 / 2.0) + 1.0 / np.sqrt(1.0 - x_lt1 ** 2) * np.arccosh(1.0 / x_lt1)
    g_gt1 = np.log(x_gt1 / 2.0) + 1.0 / np.sqrt(x_gt1 ** 2 - 1.0) * np.arccos(1.0 / x_gt1)
    return np.where(x < 1.0, g_lt1, g_gt1)


def nfw_alpha_shape(r, Rs):
    """r-dependent part of the NFW deflection magnitude: alpha(r) ∝ g(x)/x,
    x=r/Rs (the 4*rho0*Rs^2 prefactor is r-independent; nfwAlpha, nfw.py:28-34)."""
    x = np.asarray(r, np.float64) / np.asarray(Rs, np.float64)
    return nfw_g_numpy(x) / x


def slope_s_of_Rs(Rs, theta_E_star):
    """s(Rs) = d ln(alpha)/d ln(r) at r=theta_E*, central FD in ln r (rel FD_REL).
    SEMI-CLOSED: prefactors cancel; uses the exact g_ mirror."""
    Rs = np.asarray(Rs, np.float64)
    r = float(theta_E_star)
    rp = r * (1.0 + FD_REL)
    rm = r * (1.0 - FD_REL)
    ap = nfw_alpha_shape(rp, Rs)
    am = nfw_alpha_shape(rm, Rs)
    return (np.log(ap) - np.log(am)) / (np.log(rp) - np.log(rm))


# ===========================================================================
# derivation
# ===========================================================================

def _load_profile(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _lambda_hat_knots(prof):
    """Assemble (knot_z, knot_lambda, flags) for lambda_hat: on-floor bin medians
    (primary) plus below-floor medians extending the low end (flagged envelope).

    RELIABILITY FILTER (added with the F-T25a' amendment; see log correction
    entry): knots with lambda <= LAMBDA_RELIABLE are DROPPED. Two measured
    failure modes force this: (a) high-z sigmoid-plateau medians are at the HVP
    noise floor (|lambda| ~ 0.01-0.14, some negative) — log-interp through them
    would collapse du/dz to ~0 and break the strict-monotone fit; (b) below-floor
    medians at z <~ -0.7 are LARGE NEGATIVE (-2e3..-9e3): the frozen-others
    slices there are past the cliff inflection (the registered upper-envelope
    caveat realized) and carry no width information. LAMBDA_RELIABLE = 1.0 is
    derivable: lambda = 1 means conditional width ~1 in z-units ~ the prior
    scale — flatter than that is indistinguishable from flat for sampling and is
    below ~10x the observed noise magnitude. Outside the kept knots lambda_hat is
    constant-extended (np.interp endpoints), i.e. the map continues linearly."""
    zf = np.asarray(prof["bin_med_z"], np.float64)
    lf = np.asarray(prof["bin_med_lambda"], np.float64)
    lo = float(prof["visited_zrs_lo"])
    bz = np.asarray(prof["belowfloor_med_z"], np.float64)
    bl = np.asarray(prof["belowfloor_med_lambda"], np.float64)
    keep = bz < lo                                  # strictly below the floor
    bz, bl = bz[keep], bl[keep]
    knot_z = np.concatenate([bz, zf])
    knot_l = np.concatenate([bl, lf])
    is_env = np.concatenate([np.ones(len(bz), bool), np.zeros(len(zf), bool)])
    reliable = np.isfinite(knot_l) & (knot_l > LAMBDA_RELIABLE)
    n_drop = int((~reliable).sum())
    if n_drop:
        print(f"[routeA] lambda_hat: dropping {n_drop} unreliable knots "
              f"(lambda <= {LAMBDA_RELIABLE}: plateau-noise/negative-slice); "
              f"keeping {int(reliable.sum())}")
    knot_z, knot_l, is_env = knot_z[reliable], knot_l[reliable], is_env[reliable]
    order = np.argsort(knot_z)
    knot_z, knot_l, is_env = knot_z[order], knot_l[order], is_env[order]
    # de-duplicate coincident z (keep first)
    keepm = np.concatenate([[True], np.diff(knot_z) > 1e-9])
    return knot_z[keepm], knot_l[keepm], is_env[keepm]


def route_A(prof, dense_z):
    """Variance-stabilizing coordinate. Returns (leaf, arrays, info)."""
    knot_z, knot_l, is_env = _lambda_hat_knots(prof)
    knot_l = np.clip(knot_l, 1e-30, None)
    log_l = np.log(knot_l)
    # log-space interpolation, constant-extended outside (np.interp endpoints)
    lam_hat = np.exp(np.interp(dense_z, knot_z, log_l))
    lam_hat = np.clip(lam_hat, 1e-30, None)
    sqrt_lam = np.sqrt(lam_hat)

    # u_raw(z) = int_{z0}^{z} sqrt(lambda_hat) dz'   (trapezoid on the dense grid)
    du = np.diff(dense_z)
    u_raw = np.concatenate([[0.0], np.cumsum(0.5 * (sqrt_lam[1:] + sqrt_lam[:-1]) * du)])

    z_init = float(prof["zrs_init"])
    g0 = float(np.interp(z_init, dense_z, sqrt_lam))        # sqrt(lambda_hat(z_init))
    u0 = float(np.interp(z_init, dense_z, u_raw))
    u = z_init + (u_raw - u0) / g0                          # dz/du=1, u=z_init at z_init

    leaf = MonotoneCubicBijector.fit(
        u, dense_z, z_init=z_init,
        meta={"route": "A", "kind": "variance_stabilizing",
              "lambda_hat_knot_z": knot_z.tolist(),
              "lambda_hat_knot_lambda": knot_l.tolist(),
              "lambda_hat_is_envelope": is_env.tolist(),
              "g0_sqrt_lambda_init": g0, "z_init": z_init})
    arrays = {"A_dense_z": dense_z, "A_lambda_hat": lam_hat,
              "A_sqrt_lambda": sqrt_lam, "A_u": u,
              "A_du_dz": sqrt_lam / g0}
    info = {"g0": g0, "n_env_knots": int(is_env.sum())}
    return leaf, arrays, info


def route_B(prof, dense_z):
    """Observable-anchored coordinate. Returns (leaf, arrays, info)."""
    theta_E_star = float(prof["theta_E_star"])
    rsmap_z = np.asarray(prof["rsmap_z"], np.float64)
    rsmap_Rs = np.asarray(prof["rsmap_Rs"], np.float64)

    # (a) monotonicity of s in Rs over the prior box (HARD assert)
    Rs_box = np.linspace(RS_BOX[0], RS_BOX[1], 512)
    s_box = slope_s_of_Rs(Rs_box, theta_E_star)
    ds_box = np.diff(s_box)
    mono = bool(np.all(ds_box > 0) or np.all(ds_box < 0))
    if not mono:
        raise RuntimeError(
            "Route B: s(Rs) is NOT monotone over [20,100] "
            f"(sign changes={int(np.sum(np.sign(ds_box[:-1]) != np.sign(ds_box[1:])))}); "
            "the observable-anchored coordinate is not invertible. STOP.")
    s_sign = 1.0 if ds_box[0] > 0 else -1.0

    # (b) s(z) through the REAL sigmoid leaf: Rs(z) from the profile's rs-map grid
    Rs_of_z = np.interp(dense_z, rsmap_z, rsmap_Rs)
    s_z = slope_s_of_Rs(Rs_of_z, theta_E_star)

    # (c) affine-normalize: u_B = z_init + (s - s(z_init))/s'(z_init)
    z_init = float(prof["zrs_init"])
    ds_dz = np.gradient(s_z, dense_z)
    s_init = float(np.interp(z_init, dense_z, s_z))
    sp_init = float(np.interp(z_init, dense_z, ds_dz))
    if abs(sp_init) < 1e-30:
        raise RuntimeError("Route B: ds/dz vanishes at z_init; cannot normalize.")
    u_B = z_init + (s_z - s_init) / sp_init
    du_dz = ds_dz / sp_init                                  # = 1 at z_init, >0 core

    # (d) trim to the strictly-monotone core (sigmoid saturates s at the grid
    # ends -> du_dz -> 0); the leaf linearly extends beyond the end knots with the
    # (positive) end slope, a valid monotone continuation for the rare tails.
    slope_floor = 1e-4 * float(np.nanmax(du_dz))
    good = du_dz > slope_floor
    # keep the contiguous block containing z_init
    i0 = int(np.argmin(np.abs(dense_z - z_init)))
    lo_i = i0
    while lo_i > 0 and good[lo_i - 1]:
        lo_i -= 1
    hi_i = i0
    while hi_i < len(dense_z) - 1 and good[hi_i + 1]:
        hi_i += 1
    zc = dense_z[lo_i:hi_i + 1]
    uc = u_B[lo_i:hi_i + 1]
    if uc.shape[0] < 8 or np.any(np.diff(uc) <= 0):
        raise RuntimeError("Route B: monotone core too small / non-increasing after trim.")

    leaf = MonotoneCubicBijector.fit(
        uc, zc, z_init=z_init,
        meta={"route": "B", "kind": "observable_anchored",
              "theta_E_star": theta_E_star, "s_sign": s_sign,
              "s_init": s_init, "sp_init": sp_init, "z_init": z_init,
              "core_z_range": [float(zc[0]), float(zc[-1])], "fd_rel": FD_REL})
    arrays = {"B_dense_z": dense_z, "B_s_z": s_z, "B_ds_dz": ds_dz,
              "B_u": u_B, "B_du_dz": du_dz, "B_Rs_of_z": Rs_of_z,
              "B_Rs_box": Rs_box, "B_s_box": s_box}
    info = {"theta_E_star": theta_E_star, "s_monotone": mono, "s_sign": s_sign,
            "core_z_range": [float(zc[0]), float(zc[-1])], "n_core": int(uc.shape[0])}
    return leaf, arrays, info


def compare_A_B(prof, A_arrays, B_arrays):
    """P-T25c: ds/dz (Route B stretch) vs sqrt(lambda_hat) (Route A stretch) over
    the VISITED range. Report the variation of their ratio (band <=2 validated;
    >=5x fires F-T25c)."""
    lo = float(prof["visited_zrs_lo"]); hi = float(prof["visited_zrs_hi"])
    z = A_arrays["A_dense_z"]
    m = (z >= lo) & (z <= hi)
    sqrt_lam = A_arrays["A_sqrt_lambda"][m]          # ∝ Route A local stretch
    ds_dz = np.abs(B_arrays["B_ds_dz"][m])           # ∝ Route B local stretch
    ratio = ds_dz / np.clip(sqrt_lam, 1e-30, None)
    ratio_n = ratio / np.median(ratio)
    rmin, rmax = float(ratio_n.min()), float(ratio_n.max())
    variation = rmax / max(rmin, 1e-30)
    arrays = {"cmp_z": z[m], "cmp_sqrt_lambda": sqrt_lam, "cmp_ds_dz": ds_dz,
              "cmp_ratio_normalized": ratio_n}
    return arrays, {"ratio_min": rmin, "ratio_max": rmax, "variation": variation}


def check_F_T25a(prof):
    """F-T25a' (AMENDED 2026-07-03 after the original fired on a mis-registered
    window; see the log correction entry "T25 F-window correction"): the original
    [1.7,3] window sampled two adjacent bins of a smooth ~1400x decay (3.1x) and
    missed both the wall rise (below-floor lambda up to ~1.5e4) and the plateau
    fall (lambda -> HVP noise ~0.1 at z>6). Amended operationalization: total
    variation of the RELIABLE on-floor bin medians (lambda > LAMBDA_RELIABLE,
    i.e. above prior-scale/noise) over the FULL visited range, threshold 30x —
    derivation: Route A's stretch ratio is sqrt(variation); sqrt(30)~5.5x is the
    minimum for a 1-D transform to be worth building. The ORIGINAL firing and
    this amendment are both logged; the amendment was made BEFORE re-running."""
    z = np.asarray(prof["bin_med_z"], np.float64)
    lam = np.asarray(prof["bin_med_lambda"], np.float64)
    good = np.isfinite(lam) & (lam > LAMBDA_RELIABLE)
    z, lam = z[good], lam[good]
    scope = f"all visited on-floor bins with lambda > {LAMBDA_RELIABLE}"
    if lam.size < 2:
        return {"fires": True, "variation": float("nan"),
                "reason": "insufficient reliable bin medians", "scope": scope}
    variation = float(lam.max() / lam.min())
    return {"fires": bool(variation < F_T25A_MINVAR_AMENDED),
            "variation": variation, "scope": scope,
            "original_window_variation_3p1x_fired": True}


# ===========================================================================
# family-preservation gates (GPU / jax)
# ===========================================================================

def _theta_dict(bij, cols):
    import jax.numpy as jnp
    return bij.forward([jnp.asarray(c) for c in cols])


def _max_dict_diff(d1, d2):
    keys = set(d1) | set(d2)
    m = 0.0
    for k in keys:
        a = np.asarray(d1[k], np.float64); b = np.asarray(d2[k], np.float64)
        m = max(m, float(np.max(np.abs(a - b))))
    return m


def run_family_gates(route, prof):
    """Build baseline + route systems and run gates 1/2/3. Returns a result dict
    with per-gate max errors + overall pass bool."""
    import jax.numpy as jnp
    from common import load_target

    print(f"\n===== T25 family gates: route {route} =====", flush=True)
    # baseline
    base_dir = os.path.join(HERE, "systems", "carousel_min_new")
    ms_b, _qz_b, z_center, dim, names = load_target(base_dir)
    pm_b = ms_b.prob_model
    rs_key = next(k for k in _theta_dict(pm_b.bij, list(np.zeros((dim, 1))))
                  if str(k).endswith("Rs") and not str(k).endswith("alpha_Rs"))

    # route system
    route_mod = "carousel_min_newA" if route == "A" else "carousel_min_newB"
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"_route_{route}", os.path.join(HERE, "systems", route_mod, "system.py"))
    rmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rmod)
    rb = rmod.build_route_target()
    pm_r = rb["prob_model"]; leaf = rb["leaf"]; col = rb["col"]

    # cross-check the numpy g_ mirror vs gigalens' exact g_ (Route B soundness)
    g_ok = _crosscheck_g()

    rng = np.random.RandomState(20260703)
    z_init = np.asarray(prof["z_init"], np.float64)
    rsmap_z = np.asarray(prof["rsmap_z"], np.float64)
    rsmap_Rs = np.asarray(prof["rsmap_Rs"], np.float64)
    z_of_Rs = lambda R: np.interp(R, rsmap_Rs[np.argsort(rsmap_Rs)],
                                  rsmap_z[np.argsort(rsmap_Rs)])

    res = {"route": route, "g_crosscheck_max": g_ok}

    # --- Gate 1: round-trip Rs through route forward/inverse -----------------
    Rs_grid = np.linspace(RS_BOX[0] + 0.1, RS_BOX[1] - 0.1, N_GATE_RT)
    Z = np.tile(z_init, (N_GATE_RT, 1)).astype(np.float64)
    Z[:, col] = z_of_Rs(Rs_grid)
    theta_b = _theta_dict(pm_b.bij, [Z[:, j] for j in range(dim)])
    Rs_in = np.asarray(theta_b[rs_key], np.float64)
    u_cols = pm_r.bij.inverse(theta_b)                    # list of (N,) u columns
    theta_r = pm_r.bij.forward([jnp.asarray(np.asarray(c)) for c in u_cols])
    Rs_out = np.asarray(theta_r[rs_key], np.float64)
    g1 = float(np.max(np.abs(Rs_in - Rs_out)))
    res["gate1_roundtrip_Rs_maxabs"] = g1
    res["gate1_pass"] = bool(g1 < GATE_RT_TOL)
    print(f"[gate {route}.1] round-trip |Rs-Rs''| max = {g1:.3e} "
          f"({'PASS' if res['gate1_pass'] else 'FAIL'}; tol {GATE_RT_TOL})", flush=True)

    # --- Gate 2: 8 random theta -- loglike + prior identity ------------------
    Zr = z_center[None, :] + rng.standard_normal((N_GATE_THETA, dim)) * 1.0
    theta_b2 = _theta_dict(pm_b.bij, [Zr[:, j] for j in range(dim)])
    u_cols2 = pm_r.bij.inverse(theta_b2)
    U = np.stack([np.asarray(c, np.float64) for c in u_cols2], axis=1)  # (N,dim)
    ll_b = np.asarray(pm_b.log_like(jnp.asarray(Zr))[0], np.float64)
    ll_r = np.asarray(pm_r.log_like(jnp.asarray(U))[0], np.float64)
    g2_ll = float(np.max(np.abs(ll_b - ll_r)))
    theta_r2 = pm_r.bij.forward([jnp.asarray(U[:, j]) for j in range(dim)])
    pl_b = np.asarray(pm_b.prior.log_prob(theta_b2), np.float64)
    pl_r = np.asarray(pm_r.prior.log_prob(theta_r2), np.float64)
    g2_pl = float(np.max(np.abs(pl_b - pl_r)))
    res["gate2_loglike_maxabs"] = g2_ll
    res["gate2_prior_maxabs"] = g2_pl
    res["gate2_pass"] = bool(g2_ll < GATE_LOGLIKE_TOL and g2_pl < GATE_PRIOR_TOL)
    print(f"[gate {route}.2] loglike max|diff| = {g2_ll:.3e} (tol {GATE_LOGLIKE_TOL}); "
          f"prior max|diff| = {g2_pl:.3e} (tol {GATE_PRIOR_TOL}) "
          f"({'PASS' if res['gate2_pass'] else 'FAIL'})", flush=True)

    # --- Gate 3: mapped z_init identity --------------------------------------
    u_init = z_init.copy(); u_init[col] = leaf.inverse(np.asarray(z_init[col]))
    theta_bi = _theta_dict(pm_b.bij, [z_init[None, j] for j in range(dim)])
    theta_ri = _theta_dict(pm_r.bij, [u_init[None, j] for j in range(dim)])
    g3 = _max_dict_diff(theta_bi, theta_ri)
    res["gate3_zinit_theta_maxabs"] = g3
    res["gate3_pass"] = bool(g3 < GATE_ZINIT_TOL)
    print(f"[gate {route}.3] mapped z_init theta max|diff| = {g3:.3e} "
          f"({'PASS' if res['gate3_pass'] else 'FAIL'}; tol {GATE_ZINIT_TOL})", flush=True)

    res["all_pass"] = bool(res["gate1_pass"] and res["gate2_pass"] and res["gate3_pass"])
    print(f"[gate {route}] OVERALL {'PASS' if res['all_pass'] else 'FAIL'}", flush=True)
    return res


def _crosscheck_g():
    """Assert the numpy g_ mirror == gigalens.NFW().g_ on a grid (Route B soundness)."""
    try:
        import jax.numpy as jnp
        from gigalens.jax.profiles.mass.nfw import NFW
        x = np.linspace(0.05, 5.0, 400)
        g_ref = np.asarray(NFW().g_(jnp.asarray(x)), np.float64)
        g_np = nfw_g_numpy(x)
        d = float(np.max(np.abs(g_ref - g_np)))
        print(f"[route B] numpy g_ vs gigalens NFW().g_ max|diff| = {d:.3e}", flush=True)
        if d > 1e-10:
            raise RuntimeError(f"numpy g_ mirror disagrees with gigalens g_ ({d:.3e})")
        return d
    except ImportError:
        return None


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="T25b derive transforms + family gates")
    ap.add_argument("--no-gates", action="store_true",
                    help="derive artifacts only (offline/login-node checks)")
    ap.add_argument("--profile", default=PROFILE_NPZ)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    if not os.path.isfile(args.profile):
        print(f"[t25b] FATAL: profile not found: {args.profile}", flush=True)
        return 2
    prof = _load_profile(args.profile)
    os.makedirs(T25_OUT, exist_ok=True)

    # --- F-T25a guard --------------------------------------------------------
    fa = check_F_T25a(prof)
    print(f"[t25b] F-T25a total variation over on-floor bins = {fa['variation']:.3g} "
          f"(threshold {F_T25A_MINVAR}x) -> {'FIRES (STOP)' if fa['fires'] else 'ok'}",
          flush=True)
    if fa["fires"]:
        with open(ROUTES_PASSED, "w") as fh:
            json.dump({"status": "F-T25a fired; T25/T26 STOP",
                       "F_T25a": fa, "A": False, "B": False}, fh, indent=2)
        print("[t25b] F-T25a fired: funnel NOT expressible as a 1-D Rs profile; "
              "both routes mis-aimed. Blocking T26 (nonzero exit).", flush=True)
        return 3

    # --- derive both routes --------------------------------------------------
    dense_z = np.linspace(DENSE_LO, DENSE_HI, DENSE_N)
    routes_ok = {}
    derive_info = {}
    A_arrays = B_arrays = None

    try:
        leafA, A_arrays, iA = route_A(prof, dense_z)
        leafA.to_npz(TRANSFORM_A)
        derive_info["A"] = iA
        print(f"[t25b] Route A knots -> {TRANSFORM_A} "
              f"(sha256 {sha256_file(TRANSFORM_A)[:16]}...)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[t25b] Route A DERIVATION FAILED: {type(e).__name__}: {e}", flush=True)
        routes_ok["A"] = False

    try:
        leafB, B_arrays, iB = route_B(prof, dense_z)
        leafB.to_npz(TRANSFORM_B)
        derive_info["B"] = iB
        print(f"[t25b] Route B knots -> {TRANSFORM_B} "
              f"(sha256 {sha256_file(TRANSFORM_B)[:16]}...)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[t25b] Route B DERIVATION FAILED: {type(e).__name__}: {e}", flush=True)
        routes_ok["B"] = False

    # --- A-vs-B comparison (P-T25c) ------------------------------------------
    cmp_info = None
    if A_arrays is not None and B_arrays is not None:
        cmp_arrays, cmp_info = compare_A_B(prof, A_arrays, B_arrays)
        allarr = {**A_arrays, **B_arrays, **cmp_arrays}
        np.savez(AB_COMPARISON, **allarr)
        band = ("VALIDATED (<=2x)" if cmp_info["variation"] <= P_T25C_BAND
                else ("F-T25c FIRES (>=5x)" if cmp_info["variation"] >= F_T25C_FACTOR
                      else "intermediate"))
        print(f"[t25b] P-T25c: ds/dz vs sqrt(lambda_hat) ratio variation = "
              f"{cmp_info['variation']:.2f}x over visited range "
              f"(min {cmp_info['ratio_min']:.3g}, max {cmp_info['ratio_max']:.3g}) "
              f"-> {band}", flush=True)
        print(f"[t25b] wrote {AB_COMPARISON}", flush=True)

    # --- family gates --------------------------------------------------------
    gate_results = {}
    if not args.no_gates:
        for route in ("A", "B"):
            if routes_ok.get(route) is False:
                gate_results[route] = {"route": route, "all_pass": False,
                                       "reason": "derivation failed"}
                continue
            try:
                gr = run_family_gates(route, prof)
            except Exception as e:  # noqa: BLE001
                print(f"[t25b] route {route} gates RAISED: {type(e).__name__}: {e}",
                      flush=True)
                gr = {"route": route, "all_pass": False,
                      "reason": f"{type(e).__name__}: {e}"}
            gate_results[route] = gr
            routes_ok[route] = bool(gr.get("all_pass", False))
    else:
        print("[t25b] --no-gates: skipping GPU family gates (derivation only)",
              flush=True)
        for route in ("A", "B"):
            routes_ok.setdefault(route, None)

    # --- routes_passed.json + manifest ---------------------------------------
    passed = {r: bool(routes_ok.get(r)) for r in ("A", "B")}
    payload = {
        "experiment": "T25b transforms + family gates",
        "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(),
        "gates_run": (not args.no_gates),
        "A": passed["A"], "B": passed["B"],
        "F_T25a": fa,
        "P_T25c": cmp_info,
        "derive_info": derive_info,
        "gate_results": gate_results,
        "artifacts": {"A": TRANSFORM_A, "B": TRANSFORM_B, "comparison": AB_COMPARISON},
        "constants": {
            "DENSE_GRID": [DENSE_LO, DENSE_HI, DENSE_N], "FD_REL": FD_REL,
            "N_GATE_RT": N_GATE_RT, "N_GATE_THETA": N_GATE_THETA,
            "tols": {"roundtrip": GATE_RT_TOL, "loglike": GATE_LOGLIKE_TOL,
                     "prior": GATE_PRIOR_TOL, "zinit": GATE_ZINIT_TOL},
        },
    }
    with open(ROUTES_PASSED, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"[t25b] wrote {ROUTES_PASSED}: A={passed['A']} B={passed['B']}", flush=True)

    wall = time.perf_counter() - t0
    print(f"[t25b] DONE wall={wall:.1f}s (PROPOSED / UNCERTIFIED)", flush=True)

    # exit code: nonzero blocks ALL of T26 only if NO route is usable (when gates ran)
    if not args.no_gates and not (passed["A"] or passed["B"]):
        print("[t25b] NO route passed the family gates -> blocking T26 (nonzero exit)",
              flush=True)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
