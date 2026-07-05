"""T27 -- (M200, c) pushforward of existing carousel new-arm chains (DESCRIPTIVE).

Pre-registered 2026-07-04 (docs/logs/why-hard-to-sample.md, "T27 -- (M200, c)
pushforward of existing chains"). ZERO-GPU, login-node numpy/scipy/matplotlib only
(NO jax). Pushes the converged Route-A posterior samples (and the T21 baseline)
forward through the exact gigalens NFW deflection convention + a fixed-wCDM
cosmology to (M200, c) and makes a 2-parameter cornerplot. This is a pure
pushforward of samples -- NO reparameterization, NO hypothesis test.

STATUS: proposed (UNCERTIFIED). Producer may not self-certify.

=====================================================================
CONVENTION TRACE (traced in gigalens SOURCE, file:line -- NOT lenstronomy memory)
=====================================================================
gigalens NFW deflection normalization
  /global/homes/l/linusu/gigalens/src/gigalens/jax/profiles/mass/nfw.py
  - NFW.deriv         (nfw.py:20-25):  rho0 = alpha_Rs / (4 Rs^2 (1 - ln2))
  - NFW.nfwAlpha      (nfw.py:27-34):  a = 4 rho0 Rs g(x)/x^2 ; x=R/Rs ;
                                       deflection vec = a * (dx, dy)
      => |alpha|(R) = a * R = 4 rho0 Rs^2 g(x)/x  (reduced deflection)
      => at R=Rs (x=1, g(1)=1-ln2): |alpha|(Rs) = 4 rho0 Rs^2 (1-ln2) = alpha_Rs
         so alpha_Rs IS the reduced deflection angle at r=Rs (standard convention).
  - NFW.g_            (nfw.py:36-51):  exact g(x); mirrored in numpy by
                                       t25_transforms.nfw_g_numpy (verified 1e-10).
  - NFW_ELLIPSE_EINSTEIN.deriv (nfw.py:144-146):
                       rho0 = theta_E^2 / (4 Rs^3 g(theta_E/Rs))
      equating the two rho0 forms (this is the arm the chains were sampled under):
         alpha_Rs = (1-ln2) theta_E^2 / (Rs g(theta_E/Rs))     [closed form]
      This exact relation is REUSED (not re-derived) from
      t15_carousel_decompose.py:83-86 (theta_E_to_alpha_Rs) and nfw.py:106/146.

Multi-plane deflection_ratio (does alpha_Rs reference which source plane?)
  /global/homes/l/linusu/gigalens/src/gigalens/cosmo.py:9-17,149-157
  - CosmoBase docstring (cosmo.py:10-14): z_source_ref is "where the mass
    deflections are normalized, i.e. where deflection_ratio == 1. Mass parameters
    (theta_E, alpha_Rs, shear, ...) are only meaningful relative to this reference."
  - deflection_ratio(z_source) = lensing_distance(z_source)/lensing_distance(z_ref)
    (cosmo.py:154-157); lensing_distance = D_ls/D_s (cosmo.py:149-152).
  - The carousel system builds wCDM_Cosmo(z_lens=0.49, z_source_ref=1.432) with a
    SINGLE lensed source plane at z=1.432 (systems/carousel_min_common.py:63-64,
    132-140). Since z_source == z_source_ref, deflection_ratio == 1 for this plane.
  CONCLUSION: alpha_Rs is the reduced deflection referenced to z_s=1.432, so
  Sigma_cr uses (z_l=0.49, z_s=1.432). No deflection_ratio rescaling is needed.

Cosmology (fixed wCDM, read from source)
  systems/carousel_min_common.py:63-64  z_lens=0.49, z_source=1.432
  systems/carousel_min_common.py:132-135 H0=70.0, Om0=0.3, k=0.0, w0=-1.0
  E(z) replicates gigalens efunc EXACTLY incl. radiation term:
    jax/cosmo.py:13-26  matter=Om0(1+z)^3, rad=Or0(1+z)^4,
                        Ode0=1-Om0-Or0(-Ok0), DE=Ode0(1+z)^(3(1+w0))
    cosmo.py:23-25      Or0 = 2.469e-5 h^-2 (1 + 0.2271 Neff), Neff=3.04 (cosmo.py:6)
    cosmo.py:7          c = 299792.458 km/s
  Distances implemented here in scipy quadrature (flat, k=0):
    D_C(0,z) = (c/H0) int_0^z dz'/E(z');  D_A(z)=D_C/(1+z);
    D_ls (flat) = (D_C(z_s)-D_C(z_l))/(1+z_s).

Physical chain to (M200, c)  [all per the registration]
  rho0_ang  = alpha_Rs / (4 Rs^2 (1-ln2))          [1/arcsec] (Sigma_cr per arcsec)
  Sigma_cr  = c^2/(4 pi G) * D_s/(D_l D_ls)          [Msun/kpc^2]
  kpc_per_arcsec(z) = D_A(z)[kpc] * (pi/180/3600)
  rho0_phys = rho0_ang * Sigma_cr / kpc_per_arcsec(z_l)   [Msun/kpc^3]
  Rs_kpc    = Rs_arcsec * kpc_per_arcsec(z_l)
  rho_cr(z) = 3 H(z)^2 / (8 pi G)                    [Msun/kpc^3]
  solve  rho0_phys = (200/3) rho_cr(z_l) c^3 / m(c),  m(c)=ln(1+c)-c/(1+c)
         for c by bisection ; R200 = c Rs_kpc ; M200 = (4/3) pi R200^3 * 200 rho_cr(z_l)

HARD GATE (registered): round trip (Rs,theta_E) -> (M200,c) -> (Rs,theta_E) on a
64-pt grid over Rs in [25,99], theta_E in [13.5,14.1], using ONLY the forward
physics relations run in reverse. Must close < 1e-6 relative or STOP.

Run:  python t27_pushforward.py            (login node; numpy/scipy/matplotlib)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from reparam_bijector import MonotoneCubicBijector          # noqa: E402
from t25_transforms import nfw_g_numpy                       # noqa: E402  (verified 1e-10)

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PHASEC = os.path.join(HERE, "results_carousel", "phaseC")
ROUTEA_DIR = os.path.join(PHASEC, "t26", "routeA")
BASE_DIR = os.path.join(PHASEC, "t21", "new")
LEAF_NPZ = os.path.join(PHASEC, "t25", "transform_A.npz")
OUT_DIR = os.path.join(PHASEC, "t27")
SEEDS = (1, 2, 3)
RESULTS_START = 2000          # results phase = steps 2000..3999
COL_RS = 0                    # routeA: u_Rs ; baseline: z_Rs
COL_THETA_E = 5              # theta_E (z-space == theta-space; identity event bij)
SUBSAMPLE_N = 20000
SUBSAMPLE_SEED = 20260703

# ---------------------------------------------------------------------------
# fixed wCDM cosmology params (READ from systems/carousel_min_common.py source)
# ---------------------------------------------------------------------------
Z_LENS = 0.49               # carousel_min_common.py:63
Z_SOURCE = 1.432            # carousel_min_common.py:64
H0 = 70.0                   # carousel_min_common.py:134
OM0 = 0.3                   # carousel_min_common.py:134
K_CURV = 0.0                # carousel_min_common.py:134
W0 = -1.0                   # carousel_min_common.py:134
NEFF = 3.04                 # gigalens/cosmo.py:6
C_LIGHT_KM_S = 299792.458   # gigalens/cosmo.py:7

# ---------------------------------------------------------------------------
# physical constants (SI; stated values)
# ---------------------------------------------------------------------------
G_SI = 6.67430e-11               # m^3 kg^-1 s^-2  (CODATA 2018)
C_SI = 299792458.0               # m/s (exact)
MSUN_KG = 1.98892e30             # kg
MPC_M = 3.085677581491367e22     # m
KPC_M = 3.085677581491367e19     # m
ARCSEC_RAD = np.pi / (180.0 * 3600.0)   # rad/arcsec
LN2 = float(np.log(2.0))

DELTA = 200.0                    # overdensity (M200/c200 convention)


def _now():
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# cosmology (scipy quadrature; replicates gigalens efunc incl. radiation)
# ===========================================================================
def _efunc(z):
    """Dimensionless Friedmann E(z), matching gigalens jax/cosmo.py:13-26 exactly
    (flat k=0): matter + radiation + dark energy. Radiation is negligible here
    (~1e-5) but kept for faithfulness to the simulator's efunc."""
    h = H0 / 100.0
    Or0 = 2.469e-5 * h ** -2.0 * (1.0 + 0.2271 * NEFF)   # cosmo.py:23-25
    Ok0 = -K_CURV / H0 ** 2
    Ode0 = 1.0 - OM0 - Or0 - Ok0
    matter = OM0 * (1.0 + z) ** 3
    rad = Or0 * (1.0 + z) ** 4
    curv = Ok0 * (1.0 + z) ** 2
    de = Ode0 * (1.0 + z) ** (3.0 * (1.0 + W0))
    return np.sqrt(matter + rad + de + curv)


def comoving_distance_Mpc(z):
    """D_C(0,z) [Mpc] = (c/H0) int_0^z dz'/E(z')  (scipy quadrature)."""
    val, _ = quad(lambda zp: 1.0 / _efunc(zp), 0.0, z, epsabs=1e-12, epsrel=1e-12)
    return (C_LIGHT_KM_S / H0) * val


def angular_distance_Mpc(z):
    return comoving_distance_Mpc(z) / (1.0 + z)


def angular_distance_ls_Mpc(z1, z2):
    """flat: D_A(z1,z2) = (D_C(z2)-D_C(z1))/(1+z2)."""
    return (comoving_distance_Mpc(z2) - comoving_distance_Mpc(z1)) / (1.0 + z2)


def sigma_cr_Msun_kpc2(z_l, z_s):
    """Sigma_cr = c^2/(4 pi G) * D_s/(D_l D_ls)  [Msun/kpc^2]."""
    D_s = angular_distance_Mpc(z_s) * MPC_M
    D_l = angular_distance_Mpc(z_l) * MPC_M
    D_ls = angular_distance_ls_Mpc(z_l, z_s) * MPC_M
    sig_SI = C_SI ** 2 / (4.0 * np.pi * G_SI) * D_s / (D_l * D_ls)   # kg/m^2
    return sig_SI * KPC_M ** 2 / MSUN_KG                             # Msun/kpc^2


def rho_cr_Msun_kpc3(z):
    """rho_cr(z) = 3 H(z)^2/(8 pi G)  [Msun/kpc^3]."""
    H_z_SI = (H0 * _efunc(z)) * 1000.0 / MPC_M       # 1/s
    rho_SI = 3.0 * H_z_SI ** 2 / (8.0 * np.pi * G_SI)   # kg/m^3
    return rho_SI * KPC_M ** 3 / MSUN_KG                # Msun/kpc^3


def kpc_per_arcsec(z):
    return angular_distance_Mpc(z) * 1000.0 * ARCSEC_RAD


def m_of_c(c):
    return np.log(1.0 + c) - c / (1.0 + c)


# ===========================================================================
# forward + inverse physics chains
# ===========================================================================
def theta_E_to_alpha_Rs(theta_E, Rs):
    """alpha_Rs = (1-ln2) theta_E^2/(Rs g(theta_E/Rs)) (nfw.py:106/146; t15:83-86)."""
    theta_E = np.asarray(theta_E, np.float64)
    Rs = np.asarray(Rs, np.float64)
    return (1.0 - LN2) * theta_E ** 2 / (Rs * nfw_g_numpy(theta_E / Rs))


def alpha_Rs_to_theta_E(alpha_Rs, Rs, lo=1e-4, hi=500.0, iters=100):
    """Invert theta_E^2/g(theta_E/Rs) = alpha_Rs Rs/(1-ln2) by bisection.
    LHS monotone-increasing in theta_E (t15:88-92)."""
    alpha_Rs = np.atleast_1d(np.asarray(alpha_Rs, np.float64))
    Rs = np.broadcast_to(np.asarray(Rs, np.float64), alpha_Rs.shape).copy()
    target = alpha_Rs * Rs / (1.0 - LN2)
    lo_a = np.full_like(alpha_Rs, lo)
    hi_a = np.full_like(alpha_Rs, hi)

    def f(te):
        return te ** 2 / nfw_g_numpy(te / Rs) - target

    if np.any(f(lo_a) > 0) or np.any(f(hi_a) < 0):
        raise ValueError("alpha_Rs_to_theta_E: root not bracketed; widen bracket.")
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        pos = f(mid) > 0
        hi_a = np.where(pos, mid, hi_a)
        lo_a = np.where(pos, lo_a, mid)
    return 0.5 * (lo_a + hi_a)


def solve_c(rho0_phys, rho_cr_zl, lo=0.1, hi=100.0, iters=200):
    """Solve rho0_phys = (200/3) rho_cr c^3/m(c) for c by bisection.
    RHS = (200/3) rho_cr c^3/m(c) is monotone-increasing in c (dln(c^3/m)/dc>0).
    Returns (c, converged_mask, bracket_used). Auto-widens hi if the target
    exceeds the [lo,hi] RHS range (registered bracket [0.1,100] may be too tight
    for railed-Rs samples; any widening is reported)."""
    rho0_phys = np.atleast_1d(np.asarray(rho0_phys, np.float64))
    K = (DELTA / 3.0) * rho_cr_zl

    def rhs(c):
        return K * c ** 3 / m_of_c(c)

    hi_used = float(hi)
    # widen hi until it brackets every sample (RHS strictly increasing in c)
    widened = False
    while np.any(rhs(np.full_like(rho0_phys, hi_used)) < rho0_phys) and hi_used < 1e4:
        hi_used *= 2.0
        widened = True
    lo_a = np.full_like(rho0_phys, lo)
    hi_a = np.full_like(rho0_phys, hi_used)
    # samples below the RHS(lo) floor cannot be bracketed downward -> flag
    too_low = rhs(lo_a) > rho0_phys
    for _ in range(iters):
        mid = 0.5 * (lo_a + hi_a)
        pos = rhs(mid) > rho0_phys
        hi_a = np.where(pos, mid, hi_a)
        lo_a = np.where(pos, lo_a, mid)
    c = 0.5 * (lo_a + hi_a)
    converged = ~too_low
    return c, converged, {"lo": lo, "hi_registered": hi, "hi_used": hi_used,
                          "widened": widened, "n_below_floor": int(too_low.sum())}


def forward_physics(Rs_arcsec, theta_E, cosmo):
    """(Rs_arcsec, theta_E) -> dict(M200, c, alpha_Rs, rho0_ang, rho0_phys, Rs_kpc)."""
    Rs_arcsec = np.asarray(Rs_arcsec, np.float64)
    theta_E = np.asarray(theta_E, np.float64)
    alpha_Rs = theta_E_to_alpha_Rs(theta_E, Rs_arcsec)                 # arcsec
    rho0_ang = alpha_Rs / (4.0 * Rs_arcsec ** 2 * (1.0 - LN2))          # 1/arcsec
    rho0_phys = rho0_ang * cosmo["Sigma_cr"] / cosmo["kpc_arcsec_l"]    # Msun/kpc^3
    Rs_kpc = Rs_arcsec * cosmo["kpc_arcsec_l"]                          # kpc
    c, conv, brk = solve_c(rho0_phys, cosmo["rho_cr_l"])
    R200 = c * Rs_kpc                                                   # kpc
    M200 = (4.0 / 3.0) * np.pi * R200 ** 3 * DELTA * cosmo["rho_cr_l"]  # Msun
    return {"M200": M200, "c": c, "alpha_Rs": alpha_Rs, "rho0_ang": rho0_ang,
            "rho0_phys": rho0_phys, "Rs_kpc": Rs_kpc, "R200": R200,
            "converged": conv, "bracket": brk}


def inverse_physics(M200, c, cosmo):
    """(M200, c) -> (Rs_arcsec, theta_E) using ONLY the forward relations reversed.
    No reuse of forward intermediates (this is the gate's inverse leg)."""
    M200 = np.asarray(M200, np.float64)
    c = np.asarray(c, np.float64)
    rho0_phys = (DELTA / 3.0) * cosmo["rho_cr_l"] * c ** 3 / m_of_c(c)     # Msun/kpc^3
    R200 = (M200 / ((4.0 / 3.0) * np.pi * DELTA * cosmo["rho_cr_l"])) ** (1.0 / 3.0)
    Rs_kpc = R200 / c
    Rs_arcsec = Rs_kpc / cosmo["kpc_arcsec_l"]
    rho0_ang = rho0_phys * cosmo["kpc_arcsec_l"] / cosmo["Sigma_cr"]       # 1/arcsec
    alpha_Rs = rho0_ang * 4.0 * Rs_arcsec ** 2 * (1.0 - LN2)               # arcsec
    theta_E = alpha_Rs_to_theta_E(alpha_Rs, Rs_arcsec)
    return Rs_arcsec, theta_E


# ===========================================================================
# HARD GATE
# ===========================================================================
def hard_gate(cosmo):
    """Round trip on a 64-pt grid (8x8) over Rs in [25,99], theta_E in [13.5,14.1].
    Must close < 1e-6 relative. Returns (passed, info)."""
    print("\n===== T27 HARD GATE: (Rs,theta_E)->(M200,c)->(Rs,theta_E) =====", flush=True)
    Rs_g = np.linspace(25.0, 99.0, 8)
    te_g = np.linspace(13.5, 14.1, 8)
    RR, TT = np.meshgrid(Rs_g, te_g)
    Rs_in = RR.ravel()
    te_in = TT.ravel()
    fwd = forward_physics(Rs_in, te_in, cosmo)
    Rs_out, te_out = inverse_physics(fwd["M200"], fwd["c"], cosmo)
    rel_Rs = np.abs(Rs_out - Rs_in) / np.abs(Rs_in)
    rel_te = np.abs(te_out - te_in) / np.abs(te_in)
    max_rel = float(max(rel_Rs.max(), rel_te.max()))
    passed = bool(max_rel < 1e-6) and bool(fwd["converged"].all())
    print(f"[gate] grid = 8x8 = {Rs_in.size} pts "
          f"(Rs in [25,99], theta_E in [13.5,14.1])", flush=True)
    print(f"[gate] c-solve bracket: registered hi={fwd['bracket']['hi_registered']}, "
          f"used hi={fwd['bracket']['hi_used']} "
          f"(widened={fwd['bracket']['widened']}, "
          f"below_floor={fwd['bracket']['n_below_floor']})", flush=True)
    print(f"[gate] c range over grid = [{fwd['c'].min():.3f}, {fwd['c'].max():.3f}]",
          flush=True)
    print(f"[gate] max rel |Rs-Rs''| = {rel_Rs.max():.3e}", flush=True)
    print(f"[gate] max rel |te-te''| = {rel_te.max():.3e}", flush=True)
    print(f"[gate] max rel round-trip = {max_rel:.3e} "
          f"({'PASS' if passed else 'FAIL'}; tol 1e-6)", flush=True)
    if not passed:
        print("[gate] *** GATE FAILED -- STOP. Do not trust the pushforward. ***",
              flush=True)
    return passed, {"max_rel_roundtrip": max_rel,
                    "max_rel_Rs": float(rel_Rs.max()),
                    "max_rel_theta_E": float(rel_te.max()),
                    "grid_n": int(Rs_in.size),
                    "c_grid_min": float(fwd["c"].min()),
                    "c_grid_max": float(fwd["c"].max()),
                    "bracket": fwd["bracket"],
                    "all_converged": bool(fwd["converged"].all())}


# ===========================================================================
# data loading
# ===========================================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_routeA(leaf):
    """Route-A: col0 = u_Rs -> z_Rs via leaf._forward_np -> Rs=20+80 sigmoid(z_Rs);
    col5 = theta_E. Pool 3 seeds, results phase only."""
    Rs_list, te_list = [], []
    for s in SEEDS:
        p = os.path.join(ROUTEA_DIR, f"t0_seed{s}.npz")
        pos = np.load(p)["position"][:, RESULTS_START:, :]     # (8, 2000, 14)
        u_Rs = pos[..., COL_RS].ravel()
        z_Rs = leaf._forward_np(u_Rs)
        Rs = 20.0 + 80.0 * _sigmoid(z_Rs)
        te = pos[..., COL_THETA_E].ravel()
        Rs_list.append(Rs)
        te_list.append(te)
    return np.concatenate(Rs_list), np.concatenate(te_list)


def load_baseline():
    """T21 baseline: col0 = z_Rs directly -> Rs=20+80 sigmoid(z_Rs); col5 = theta_E."""
    Rs_list, te_list = [], []
    for s in SEEDS:
        p = os.path.join(BASE_DIR, f"t0_seed{s}.npz")
        pos = np.load(p)["position"][:, RESULTS_START:, :]
        z_Rs = pos[..., COL_RS].ravel()
        Rs = 20.0 + 80.0 * _sigmoid(z_Rs)
        te = pos[..., COL_THETA_E].ravel()
        Rs_list.append(Rs)
        te_list.append(te)
    return np.concatenate(Rs_list), np.concatenate(te_list)


def subsample(n, rng):
    """Deterministic index subsample to <= SUBSAMPLE_N."""
    if n <= SUBSAMPLE_N:
        return np.arange(n)
    return np.sort(rng.choice(n, size=SUBSAMPLE_N, replace=False))


# ===========================================================================
# plotting
# ===========================================================================
def make_corner(A, B, cosmo, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    lA = np.log10(A["M200"]); cA = A["c"]
    lB = np.log10(B["M200"]); cB = B["c"]

    qs = (5, 50, 95)
    def q(a):
        return np.percentile(a, qs)
    qlA, qcA, qlB, qcB = q(lA), q(cA), q(lB), q(cB)

    x_lo = min(lA.min(), lB.min()); x_hi = max(lA.max(), lB.max())
    y_lo = min(cA.min(), cB.min()); y_hi = max(cA.max(), cB.max())
    xpad = 0.03 * (x_hi - x_lo + 1e-9); ypad = 0.03 * (y_hi - y_lo + 1e-9)
    x_lo -= xpad; x_hi += xpad; y_lo -= ypad; y_hi += ypad

    fig = plt.figure(figsize=(8.2, 8.2))
    gs = GridSpec(2, 2, width_ratios=[4, 1.3], height_ratios=[1.3, 4],
                  hspace=0.05, wspace=0.05)
    ax2d = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0], sharex=ax2d)
    axright = fig.add_subplot(gs[1, 1], sharey=ax2d)

    cA_col = "#1f77b4"; cB_col = "#d62728"

    # 2D: theta_E is pinned (~13.81), so (M200,c) collapse to a ~1-D degeneracy
    # in Rs. A hist2d on a razor-thin line reads as blank, so scatter BOTH sets
    # (Route A filled/behind, baseline outline-colour in front) to show extents.
    ax2d.scatter(lA, cA, s=3.0, c=cA_col, alpha=0.06, edgecolors="none",
                 rasterized=True, label="Route A")
    ax2d.scatter(lB, cB, s=3.0, c=cB_col, alpha=0.06, edgecolors="none",
                 rasterized=True, label="T21 baseline")
    ax2d.set_xlim(x_lo, x_hi); ax2d.set_ylim(y_lo, y_hi)
    ax2d.set_xlabel(r"$\log_{10}\, M_{200}\ [\mathrm{M_\odot}]$")
    ax2d.set_ylabel(r"concentration $c_{200}$")
    ax2d.text(0.03, 0.05,
              r"$\theta_E$ pinned $\Rightarrow$ near-1D $(M_{200},c)$ degeneracy in $R_s$",
              transform=ax2d.transAxes, fontsize=8, color="0.3")

    # marginals
    xbins = np.linspace(x_lo, x_hi, 90)
    ybins = np.linspace(y_lo, y_hi, 90)
    axtop.hist(lA, bins=xbins, density=True, color=cA_col, alpha=0.55,
               label="Route A")
    axtop.hist(lB, bins=xbins, density=True, histtype="step", color=cB_col,
               lw=1.6, label="T21 baseline")
    axright.hist(cA, bins=ybins, density=True, color=cA_col, alpha=0.55,
                 orientation="horizontal")
    axright.hist(cB, bins=ybins, density=True, histtype="step", color=cB_col,
                 lw=1.6, orientation="horizontal")
    for qq, col, ls in [(qlA, cA_col, "-"), (qlB, cB_col, "--")]:
        for v in qq:
            axtop.axvline(v, color=col, ls=ls, lw=0.8, alpha=0.7)
    for qq, col, ls in [(qcA, cA_col, "-"), (qcB, cB_col, "--")]:
        for v in qq:
            axright.axhline(v, color=col, ls=ls, lw=0.8, alpha=0.7)
    axtop.tick_params(labelbottom=False)
    axright.tick_params(labelleft=False)
    axtop.set_yticks([]); axright.set_xticks([])

    # quantile annotation box
    txt = (
        "5/50/95%  (Route A | T21 baseline)\n"
        r"$\log_{10}M_{200}$: "
        f"{qlA[0]:.2f}/{qlA[1]:.2f}/{qlA[2]:.2f} | "
        f"{qlB[0]:.2f}/{qlB[1]:.2f}/{qlB[2]:.2f}\n"
        r"$c_{200}$: "
        f"{qcA[0]:.1f}/{qcA[1]:.1f}/{qcA[2]:.1f} | "
        f"{qcB[0]:.1f}/{qcB[1]:.1f}/{qcB[2]:.1f}"
    )
    fig.text(0.62, 0.80, txt, fontsize=8.5, va="top", ha="left",
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))
    axtop.legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle("T27  (M200, c) pushforward -- carousel new-arm posterior\n"
                 "Route A (filled) vs T21 baseline (outline)  [proposed / UNCERTIFIED]",
                 fontsize=11)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}", flush=True)
    return {"logM200": qs, "qlA": qlA.tolist(), "qlB": qlB.tolist(),
            "qcA": qcA.tolist(), "qcB": qcB.tolist()}


# ===========================================================================
# main
# ===========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72, flush=True)
    print("T27 -- (M200, c) pushforward (login-node, numpy/scipy; UNCERTIFIED)",
          flush=True)
    print("=" * 72, flush=True)

    # --- cosmology --------------------------------------------------------
    Sigma_cr = sigma_cr_Msun_kpc2(Z_LENS, Z_SOURCE)
    rho_cr_l = rho_cr_Msun_kpc3(Z_LENS)
    kpc_arcsec_l = kpc_per_arcsec(Z_LENS)
    cosmo = {"Sigma_cr": Sigma_cr, "rho_cr_l": rho_cr_l,
             "kpc_arcsec_l": kpc_arcsec_l}
    print("\n[cosmo] fixed wCDM (H0=%.1f Om0=%.2f k=%.1f w0=%.1f, Neff=%.2f)"
          % (H0, OM0, K_CURV, W0, NEFF), flush=True)
    print(f"[cosmo] D_A(z_l=0.49)   = {angular_distance_Mpc(Z_LENS):.3f} Mpc", flush=True)
    print(f"[cosmo] D_A(z_s=1.432)  = {angular_distance_Mpc(Z_SOURCE):.3f} Mpc", flush=True)
    print(f"[cosmo] D_A(z_l,z_s)    = {angular_distance_ls_Mpc(Z_LENS, Z_SOURCE):.3f} Mpc",
          flush=True)
    print(f"[cosmo] Sigma_cr        = {Sigma_cr:.4e} Msun/kpc^2", flush=True)
    print(f"[cosmo] rho_cr(z_l)     = {rho_cr_l:.4e} Msun/kpc^3", flush=True)
    print(f"[cosmo] kpc_per_arcsec(z_l) = {kpc_arcsec_l:.5f} kpc/arcsec", flush=True)

    # --- HARD GATE (before touching samples) ------------------------------
    gate_pass, gate_info = hard_gate(cosmo)
    if not gate_pass:
        payload = {"experiment": "T27 (M200,c) pushforward",
                   "status": "GATE FAILED -- STOP",
                   "gate": gate_info, "timestamp_utc": _now()}
        with open(os.path.join(OUT_DIR, "t27_summary.json"), "w") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print("\n[t27] HARD GATE FAILED -- refusing to produce the pushforward.",
              flush=True)
        return 1

    # --- load samples -----------------------------------------------------
    print("\n[data] loading chains (results phase, 3 seeds pooled)", flush=True)
    leaf = MonotoneCubicBijector.from_npz(LEAF_NPZ)
    assert leaf.meta.get("route") == "A", "transform_A is not Route A"
    Rs_A, te_A = load_routeA(leaf)
    Rs_B, te_B = load_baseline()
    print(f"[data] Route A pooled n = {Rs_A.size}  "
          f"Rs[arcsec] med {np.median(Rs_A):.3f} [{Rs_A.min():.2f},{Rs_A.max():.2f}]  "
          f"theta_E med {np.median(te_A):.4f}", flush=True)
    print(f"[data] baseline pooled n = {Rs_B.size}  "
          f"Rs[arcsec] med {np.median(Rs_B):.3f} [{Rs_B.min():.2f},{Rs_B.max():.2f}]  "
          f"theta_E med {np.median(te_B):.4f}", flush=True)

    # sanity: theta_E must be ~13.8 (identity event-space bijector check)
    for nm, te in (("routeA", te_A), ("baseline", te_B)):
        if not (13.5 < np.median(te) < 14.2):
            raise RuntimeError(f"{nm} theta_E median {np.median(te):.4f} "
                               "outside expected ~13.8 -- column mis-ID?")

    # deterministic subsample
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    iA = subsample(Rs_A.size, rng)
    iB = subsample(Rs_B.size, rng)
    Rs_A, te_A = Rs_A[iA], te_A[iA]
    Rs_B, te_B = Rs_B[iB], te_B[iB]
    print(f"[data] subsampled to n_A={Rs_A.size} n_B={te_B.size} "
          f"(RandomState({SUBSAMPLE_SEED}))", flush=True)

    # --- pushforward ------------------------------------------------------
    print("\n[push] pushing samples forward to (M200, c)", flush=True)
    A = forward_physics(Rs_A, te_A, cosmo)
    B = forward_physics(Rs_B, te_B, cosmo)
    for nm, D, conv in (("routeA", A, A["converged"]), ("baseline", B, B["converged"])):
        nbad = int((~conv).sum())
        if nbad:
            print(f"[push] WARNING {nm}: {nbad} samples below c-solve floor "
                  f"(bracket {D['bracket']})", flush=True)

    # --- quantiles + summary table ---------------------------------------
    def qtab(name, D):
        M = D["M200"]; c = D["c"]; lM = np.log10(M)
        qM = np.percentile(M, [5, 50, 95])
        qlM = np.percentile(lM, [5, 50, 95])
        qc = np.percentile(c, [5, 50, 95])
        return {"name": name, "n": int(M.size),
                "M200_5_50_95": qM.tolist(),
                "log10M200_5_50_95": qlM.tolist(),
                "c_5_50_95": qc.tolist(),
                "M200_median": float(qM[1]), "c_median": float(qc[1])}
    tA = qtab("routeA", A)
    tB = qtab("baseline", B)

    print("\n" + "=" * 72, flush=True)
    print("T27 SUMMARY  (M200 [Msun], c200)   5% / 50% / 95%", flush=True)
    print("-" * 72, flush=True)
    for t in (tA, tB):
        qM = t["M200_5_50_95"]; qlM = t["log10M200_5_50_95"]; qc = t["c_5_50_95"]
        print(f"{t['name']:>9} (n={t['n']:>5}) : "
              f"M200 = {qM[0]:.3e} / {qM[1]:.3e} / {qM[2]:.3e}", flush=True)
        print(f"{'':>9}            log10M200 = {qlM[0]:.3f} / {qlM[1]:.3f} / {qlM[2]:.3f}"
              f"   c = {qc[0]:.2f} / {qc[1]:.2f} / {qc[2]:.2f}", flush=True)
    # truncation-bias view: median shifts A vs B
    dlogM = tA["log10M200_5_50_95"][1] - tB["log10M200_5_50_95"][1]
    dc = tA["c_5_50_95"][1] - tB["c_5_50_95"][1]
    print("-" * 72, flush=True)
    print(f"truncation-bias (Route A - baseline medians): "
          f"dlog10M200 = {dlogM:+.3f} dex, dc = {dc:+.2f}", flush=True)
    print("=" * 72, flush=True)

    # --- write npz --------------------------------------------------------
    npz_path = os.path.join(OUT_DIR, "t27_m200c.npz")
    np.savez(npz_path,
             routeA_M200=A["M200"], routeA_c=A["c"],
             routeA_Rs_arcsec=Rs_A, routeA_theta_E=te_A,
             routeA_alpha_Rs=A["alpha_Rs"], routeA_Rs_kpc=A["Rs_kpc"],
             baseline_M200=B["M200"], baseline_c=B["c"],
             baseline_Rs_arcsec=Rs_B, baseline_theta_E=te_B,
             baseline_alpha_Rs=B["alpha_Rs"], baseline_Rs_kpc=B["Rs_kpc"])
    print(f"\n[out] wrote {npz_path}", flush=True)

    # --- corner plot ------------------------------------------------------
    corner_q = make_corner(A, B, cosmo, os.path.join(OUT_DIR, "t27_corner.png"))

    # --- summary json -----------------------------------------------------
    summary = {
        "experiment": "T27 (M200,c) pushforward of carousel new-arm chains",
        "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(),
        "note": "Pure pushforward of existing samples; NO reparameterization, "
                "NO hypothesis test. Descriptive readout only.",
        "hard_gate": gate_info,
        "hard_gate_passed": gate_pass,
        "quantiles": {"routeA": tA, "baseline": tB},
        "truncation_bias": {"dlog10M200_median_dex": dlogM, "dc_median": dc},
        "corner_quantiles": corner_q,
        "cosmology": {
            "z_lens": Z_LENS, "z_source": Z_SOURCE, "H0": H0, "Om0": OM0,
            "k": K_CURV, "w0": W0, "Neff": NEFF, "c_km_s": C_LIGHT_KM_S,
            "source": "systems/carousel_min_common.py:63-64,132-135; "
                      "gigalens/cosmo.py:6-7; jax/cosmo.py:13-26",
            "Sigma_cr_Msun_kpc2": Sigma_cr, "rho_cr_zl_Msun_kpc3": rho_cr_l,
            "kpc_per_arcsec_zl": kpc_arcsec_l,
            "D_A_zl_Mpc": angular_distance_Mpc(Z_LENS),
            "D_A_zs_Mpc": angular_distance_Mpc(Z_SOURCE),
            "D_A_ls_Mpc": angular_distance_ls_Mpc(Z_LENS, Z_SOURCE),
        },
        "convention_citations": {
            "alpha_Rs_is_reduced_deflection_at_Rs": "nfw.py:20-34 (NFW.deriv/nfwAlpha)",
            "rho0_from_alpha_Rs": "nfw.py:21,106 (alpha_Rs/(4 Rs^2 (1-ln2)))",
            "theta_E_to_alpha_Rs_closed_form": "nfw.py:106/146 + t15_carousel_decompose.py:83-86",
            "nfw_g_numpy_verified": "t25_transforms.py:91-100 (vs gigalens g_ to 1e-10)",
            "z_source_ref_normalization": "gigalens/cosmo.py:9-17,149-157 "
                "(alpha_Rs referenced to z_s=1.432; deflection_ratio==1 single plane)",
        },
        "constants_SI": {"G": G_SI, "c": C_SI, "Msun_kg": MSUN_KG,
                         "Mpc_m": MPC_M, "kpc_m": KPC_M},
        "inputs": {"routeA": ROUTEA_DIR, "baseline": BASE_DIR, "leaf": LEAF_NPZ,
                   "seeds": list(SEEDS), "results_phase_start": RESULTS_START,
                   "col_Rs": COL_RS, "col_theta_E": COL_THETA_E,
                   "subsample_n": SUBSAMPLE_N, "subsample_seed": SUBSAMPLE_SEED},
        "stated_expectation": "M200 ~ 1e14-1e15 Msun, c ~ 2-10 (halo-study range)",
    }
    with open(os.path.join(OUT_DIR, "t27_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"[out] wrote {os.path.join(OUT_DIR, 't27_summary.json')}", flush=True)
    print("\n[t27] DONE (proposed / UNCERTIFIED)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
