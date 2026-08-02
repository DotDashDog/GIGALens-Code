"""T8 -- Does GN-eigenvalue BREATHING (not bending) set T3's h*?

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T8, approved by the
human 2026-07-02). Driven by the T6/E1 finding: on sys60 the local Gauss-Newton
metric M(z)=J^T W J has a top eigenvalue that BREATHES x22 across the typical set
while its stiff DIRECTION rotates only slowly at h*-scales. T8 asks whether that
breathing -- not the rotation -- is what makes T3's second-difference curvature
D2(h) leave its plateau at h_dev.

Method (verbatim design): along the T3 directions with clear D2 transitions
(random_1, random_2, axis[planes/0/mass/0/gamma]) plus the stiffest clone-cov
eigendirection (its -12-nat dip; D2 'stable', so we use h* = 3.5e-3 for its range
and SAY SO), walk z = x0 + t*e over ~25 t-values (dense-linear near +/-h_dev plus
log-spaced out to +/-4*h_dev, including t=0). At each z we form the SAME
standardized GN metric E1 used (M_zhat = gn_metric(scale_cols(J_z, std_z), W),
i.e. curvature in zhat = z/std_z units) and record:
  g(t)       = e_hat^T M_zhat e_hat   (the GN quadratic form along the transect)
  lambda1(t), lambda2(t)             (top two eigenvalues of M_zhat)
  overlap(t) = |v1(t) . e_hat|        (alignment of stiffest eigvec with e_hat)
where e_hat is the transect direction expressed in zhat coordinates and
unit-normalized (see standardize_direction): because zhat = z/std_z, a raw-z step
t*e maps to a zhat step t*(e/std_z), so e_hat = (e/std_z)/||e/std_z||.

Attribution (e^T M e = sum_k lambda_k (v_k . e)^2, an exact identity for the
orthonormal eigenbasis, with Sum_k (v_k.e_hat)^2 = 1): over |t| <= h_dev we report
the variation (max-min) of g and decompose it two ways --
  breathing share: hold the overlaps o_k = (v_k . e_hat) fixed at their t=0 values
                   and vary only the lambda_k(t)   -> g_breath(t)
  bending   share: hold the lambda_k fixed at their t=0 values and vary only the
                   overlaps o_k(t)                 -> g_bend(t)
each share = (max-min of that reconstruction) / (max-min of g). They need NOT sum
to 1 (cross terms + rank-continuity approximation), which we state in the JSON.

x0 and the direction vectors are reconstructed EXACTLY as replot_macro_with_clone.py
does (same rng stream seeded by the T3 JSON's args.seed for the randoms; eigh(cov)
for the eigendirections; unit axis from param_names) and each is VERIFIED
sign-sensitively against the T3 JSON's stored g_dot_e and sigma_dir before use
(replot's verify_direction, imported). Any mismatch RAISES.

The chi^2 render-path gate (E1 Task A, verified to machine zero there) is re-run at
startup here on this script's own probe points -- nothing downstream is valid
without it. E1 does not factor the gate into an importable function, so its loop is
replicated here VERBATIM in convention, reusing E1's build_jax_ops render/W/aux.

jax and scipy (via replot) are imported ONLY inside functions so the module imports
cleanly under a plain conda python for offline smoke tests (--smoke) of the
pure-numpy attribution algebra.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reused verbatim from E1 (module-level numpy only; no jax/scipy at import time).
from e1_fisher_survey import (
    gn_metric, scale_cols, eig_desc, build_jax_ops, TINY, H_STAR,
)

# --- pre-registered constants (T8 checkpoint) ------------------------------
T8_DIRECTIONS = ("random_1", "random_2", "axis[planes/0/mass/0/gamma]", "stiffest")
N_T_LIN = 13          # dense-linear samples across [-h_dev, +h_dev] (includes 0, +/-h_dev)
N_T_LOG = 6           # log-spaced samples per side over (h_dev, 4*h_dev]
T_OUTER_FACTOR = 4.0  # outer reach = +/- T_OUTER_FACTOR * h_dev
K_ATTR = 22           # use the full eigenbasis in the attribution (exact identity)
# Pre-registered prediction / falsifier (restated next to measured numbers):
T8_PREDICT_GVAR = 3.0     # g varies by >= 3x within |t| <= h_dev on transition dirs
T8_PREDICT_LAMBDA_SHARE = 0.70   # ... with the lambda (breathing) share >= 70%
T8_FALSIFY_GVAR = 1.5     # g constant within ~1.5x over |t| <= 2*h_dev => breathing
                          #   does NOT explain h*


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pure-numpy machinery (offline-testable) -- direction transform, decomposition
# ===========================================================================

def standardize_direction(e_raw, std_z):
    """Express a raw-z direction e_raw in standardized zhat = z/std_z coordinates
    and unit-normalize. A raw-z step t*e_raw is a zhat step t*(e_raw/std_z), so the
    transect direction in zhat is e_hat = (e_raw/std_z)/||e_raw/std_z||. Returns
    (e_hat, scale) where scale = ||e_raw/std_z|| relates the zhat arclength to t
    (a zhat-step of length |t|*scale corresponds to raw-z displacement t*e_raw)."""
    v = np.asarray(e_raw, dtype=np.float64) / np.asarray(std_z, dtype=np.float64)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        raise ValueError("standardize_direction: zero direction after scaling.")
    return v / nrm, float(nrm)


def quad_form_profile(M_zhat, e_hat, k=K_ATTR):
    """Decompose g = e_hat^T M e_hat over the (descending) eigenbasis of M.

    Returns dict with g, lambdas (descending, length k), overlaps o_k = v_k.e_hat
    (SIGNED, length k), lambda1, lambda2, overlap1 = |o_1|. The identity
    g = sum_k lambda_k o_k^2 holds over the FULL basis; with k < n it is a partial
    sum (still exact for k=n=22 here)."""
    w, V = eig_desc(M_zhat)            # descending
    e_hat = np.asarray(e_hat, dtype=np.float64)
    o = V.T @ e_hat                    # (n,) signed overlaps by rank
    g = float(np.sum(w * o ** 2))      # = e_hat^T M e_hat over full basis
    return {
        "g": g,
        "lambdas": w[:k].copy(),
        "overlaps": o[:k].copy(),
        "lambda1": float(w[0]),
        "lambda2": float(w[1]),
        "overlap1": float(abs(o[0])),
    }


def _shares_from_window(g, g_breath, g_bend, mask):
    """Breathing vs bending decomposition of g's variation over a t-window.

    g(t)        = sum_k lambda_k(t) o_k(t)^2  (o_k = v_k.e_hat, signed)
    g_breath(t) = sum_k lambda_k(t) o_k(0)^2  (overlaps frozen at t=0)  -> breathing
    g_bend(t)   = sum_k lambda_k(0) o_k(t)^2  (lambdas  frozen at t=0)  -> bending
    share = (max-min of that reconstruction over `mask`) / (max-min of g over `mask`).
    Shares need NOT sum to 1 (cross terms + rank-continuity approximation)."""
    def span(a):
        aa = np.asarray(a)[mask]
        return float(np.nanmax(aa) - np.nanmin(aa)) if aa.size else float("nan")
    g_var = span(g)
    b_share = span(g_breath) / g_var if g_var > 0 else float("nan")
    n_share = span(g_bend) / g_var if g_var > 0 else float("nan")
    # g's multiplicative variation (max/min) -- the >=3x / <=1.5x prediction lives here
    gg = np.asarray(g)[mask]
    gmin = float(np.nanmin(gg)) if gg.size else float("nan")
    gmax = float(np.nanmax(gg)) if gg.size else float("nan")
    g_ratio = gmax / gmin if gmin > 0 else float("inf")
    return {"g_var_absolute": g_var, "g_ratio_max_over_min": g_ratio,
            "breathing_share": b_share, "bending_share": n_share,
            "g_min_in_window": gmin, "g_max_in_window": gmax}


def build_t_grid(h_dev, n_lin=N_T_LIN, n_log=N_T_LOG, outer=T_OUTER_FACTOR):
    """Symmetric t-grid: dense-linear across [-h_dev, +h_dev] (n_lin points incl.
    0 and +/-h_dev) plus log-spaced points (n_log per side) over (h_dev, outer*h_dev].
    Returns a sorted unique array of ~ n_lin + 2*n_log t-values (includes 0)."""
    h_dev = float(h_dev)
    lin = np.linspace(-h_dev, h_dev, n_lin)
    log_pos = np.geomspace(h_dev * 1.15, outer * h_dev, n_log)
    ts = np.concatenate([lin, log_pos, -log_pos, [0.0]])
    ts = np.unique(np.round(ts, 15))
    return np.sort(ts)


# ===========================================================================
# chi^2 render-path gate (replicated from E1 Task A -- E1 does not factor it out)
# ===========================================================================

def chi2_gate(ops, z_points, tag="T8"):
    """Recompute chi^2 from the render path and reconcile with log_prob's reduced
    chi^2 aux at each z in z_points. Convention IDENTICAL to E1: aux = reduced
    chi^2 = chi2_total/event_size, so chi2_from_aux = aux * event_size. RAISE if
    relative match > 1e-6 (a wrong render invalidates all downstream metrics)."""
    W = ops["W"]; obs = ops["obs"]; event_size = ops["event_size"]
    checks = []
    for i, z in enumerate(z_points):
        z = np.asarray(z, dtype=np.float64)
        render = np.asarray(ops["render"](z), dtype=np.float64)
        chi2_mine = float(np.sum(W * (obs - render) ** 2))
        red_aux = float(ops["red_chi2"](z))
        chi2_from_aux = red_aux * event_size
        rel = abs(chi2_mine - chi2_from_aux) / max(abs(chi2_from_aux), TINY)
        checks.append({"probe_point": i, "chi2_recomputed": chi2_mine,
                       "reduced_chi2_aux": red_aux, "event_size": event_size,
                       "chi2_from_aux": chi2_from_aux, "relative_error": rel})
        print(f"[{tag}] chi2 gate pt{i}: recomputed={chi2_mine:.10e} "
              f"aux*event_size={chi2_from_aux:.10e} rel={rel:.3e}")
        if rel > 1e-6:
            raise RuntimeError(
                f"[{tag}] chi^2 reconciliation FAILED at probe {i}: rel={rel:.3e} "
                "> 1e-6. Render path != likelihood model image -- STOP.")
    worst = max(c["relative_error"] for c in checks)
    print(f"[{tag}] chi^2 gate PASSED (worst rel={worst:.3e})")
    return checks, worst


# ===========================================================================
# Offline smoke test (numpy only) -- attribution algebra on synthetic M(t)
# ===========================================================================

def _rot(n, i, j, phi):
    G = np.eye(n)
    c, s = np.cos(phi), np.sin(phi)
    G[i, i] = c; G[i, j] = -s; G[j, i] = s; G[j, j] = c
    return G


def _synthetic_transect(mode, ts, seed=0):
    """Build M(t) with a KNOWN breathing/bending mix and return the per-t lambdas
    (descending) and signed overlaps o_k=v_k.e_hat for the fixed probe direction.

    mode='breathing': eigenvectors FIXED, lambda1 scales linearly with t  -> share
                       should be ~1 breathing / ~0 bending.
    mode='bending'  : lambdas FIXED, eigenbasis rotates linearly with t    -> ~0/~1.
    mode='mixed'    : both                                                 -> both>0.
    """
    rng = np.random.default_rng(seed)
    n = 5
    Q0, _ = np.linalg.qr(rng.standard_normal((n, n)))
    base = np.array([100.0, 10.0, 3.0, 1.0, 0.3])   # descending eigenvalues
    e_hat = rng.standard_normal(n); e_hat /= np.linalg.norm(e_hat)
    lam_list, ov_list = [], []
    for t in ts:
        lam = base.copy()
        Q = Q0
        if mode in ("breathing", "mixed"):
            lam = base.copy()
            lam[0] = base[0] * (1.0 + 1.5 * t)         # top eigenvalue breathes
        if mode in ("bending", "mixed"):
            Q = Q0 @ _rot(n, 0, 1, 0.8 * t)            # basis rotates
        M = Q @ np.diag(lam) @ Q.T
        prof = quad_form_profile(0.5 * (M + M.T), e_hat, k=n)
        lam_list.append(prof["lambdas"]); ov_list.append(prof["overlaps"])
    return np.array(lam_list), np.array(ov_list), e_hat


def smoke_attribution(verbose=True):
    ts = build_t_grid(0.02)
    mask = np.abs(ts) <= 0.02
    out = {}
    ok = True

    def spans(mode):
        lam, ov, _ = _synthetic_transect(mode, ts)
        i0 = int(np.argmin(np.abs(ts)))               # t=0 reference
        o2 = ov ** 2
        g = np.sum(lam * o2, axis=1)
        g_breath = np.sum(lam * (o2[i0][None, :]), axis=1)   # overlaps frozen
        g_bend = np.sum(lam[i0][None, :] * o2, axis=1)       # lambdas frozen
        sh = _shares_from_window(g, g_breath, g_bend, mask)
        sp = lambda a: float(np.nanmax(a[mask]) - np.nanmin(a[mask]))
        return sh, sp(g_breath), sp(g_bend)

    # Pure cases: exactly ONE mechanism moves g, so the share recovers to 1 / 0.
    for mode, exp_b, exp_n in (("breathing", 1.0, 0.0), ("bending", 0.0, 1.0)):
        sh, _, _ = spans(mode)
        out[mode] = sh
        ok = ok and abs(sh["breathing_share"] - exp_b) < 0.08 \
                and abs(sh["bending_share"] - exp_n) < 0.08
        if verbose:
            print(f"[smoke] {mode:9s}: breathing_share={sh['breathing_share']:.3f} "
                  f"bending_share={sh['bending_share']:.3f} "
                  f"g_ratio={sh['g_ratio_max_over_min']:.3f} "
                  f"(expect b~{exp_b}, n~{exp_n})")

    # Mixed case: breathing raises lambda1 while the top eigvec rotates AWAY from
    # e_hat -- the two partly CANCEL in g (so dividing by g's near-flat range would
    # inflate the shares; that is the documented "need not sum to 1" regime). We
    # instead validate that BOTH reconstructions move substantially, i.e. both
    # mechanisms are individually active, via their absolute spans.
    sh_m, sp_b, sp_n = spans("mixed")
    out["mixed"] = sh_m
    frac_b = sp_b / (sp_b + sp_n) if (sp_b + sp_n) > 0 else float("nan")
    ok = ok and 0.15 < frac_b < 0.85 and sp_b > 0 and sp_n > 0
    if verbose:
        print(f"[smoke] mixed    : g_breath span={sp_b:.3g} g_bend span={sp_n:.3g} "
              f"breath-fraction={frac_b:.3f} (expect both active, 0.15<frac<0.85; "
              f"g nearly cancels: g_ratio={sh_m['g_ratio_max_over_min']:.3f})")
    if verbose:
        print(f"[smoke] attribution recovery -> {'PASS' if ok else 'FAIL'}")
    out["pass"] = bool(ok)
    return out


def run_smoke():
    print("=== T8 offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_attribution()
    print(f"[smoke] overall: {'PASS' if a['pass'] else 'FAIL'}")
    return {"attribution": a, "pass": bool(a["pass"])}


# ===========================================================================
# Direction reconstruction (reuse replot's verified builders/verifier)
# ===========================================================================

def reconstruct_directions(t3_json, clone_cov):
    """Rebuild + verify the T3 direction vectors EXACTLY as replot_macro_with_clone.
    Imported lazily (replot pulls scipy at import time; keep offline import clean)."""
    from replot_macro_with_clone import rebuild_directions, verify_direction
    grad_ad = np.asarray(t3_json["grad_ad"], dtype=np.float64)
    by_name = rebuild_directions(t3_json, clone_cov)
    dmap = {d["name"]: d for d in t3_json["directions"]}
    verified = {}
    for name in T8_DIRECTIONS:
        if name not in dmap:
            raise KeyError(f"[T8] direction {name!r} not in T3 JSON directions "
                           f"({list(dmap)})")
        d = dmap[name]
        e = verify_direction(d, by_name[name], grad_ad, clone_cov)  # RAISES on mismatch
        verified[name] = {"e_raw": np.asarray(e, dtype=np.float64), "t3": d}
    return verified


def h_dev_for(d):
    """That direction's D2 deviation scale from the T3 summary; 'stable' -> use h*.
    Returns (h_dev, note)."""
    hd = d["summary"]["h_D2_deviate"]
    if isinstance(hd, str):    # 'stable' (e.g. stiffest) -- no D2 transition
        return H_STAR, f"D2 'stable' (no transition); using h* = {H_STAR:g} for range"
    return float(hd), "h_D2_deviate from T3 summary"


# ===========================================================================
# Plot
# ===========================================================================

def plot_transects(per_dir, out_path):
    """<=4x4 panels: per direction (row) g(t), lambda1/2(t), overlap(t) vs t with
    +/-h_dev marked, plus the T3 D2 plateau-deviation factor |D2/plateau| vs h."""
    names = list(per_dir.keys())
    nrows = len(names)
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 3.4 * nrows), squeeze=False)
    for r, name in enumerate(names):
        R = per_dir[name]
        ts = np.asarray(R["t"]); hd = R["h_dev"]
        g = np.asarray(R["g"]); l1 = np.asarray(R["lambda1"])
        l2 = np.asarray(R["lambda2"]); ov = np.asarray(R["overlap"])

        ax = axes[r][0]
        ax.plot(ts, g, "-o", ms=3, color="#1f5f7a")
        ax.set_yscale("log"); ax.set_ylabel(f"{name}\ng(t)=e^T M e", fontsize=8)
        ax.set_title("GN quadratic form g(t)", fontsize=9)

        ax = axes[r][1]
        ax.plot(ts, l1, "-o", ms=3, color="#8a3b3b", label="lambda1")
        ax.plot(ts, l2, "-s", ms=2.5, color="#c08a3b", label="lambda2")
        ax.set_yscale("log"); ax.set_title("top eigenvalues", fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[r][2]
        ax.plot(ts, ov, "-o", ms=3, color="#3b7a4a")
        ax.set_ylim(-0.02, 1.02); ax.set_title("overlap |v1.e_hat|", fontsize=9)

        for c in range(3):
            for s in (-1, 1):
                axes[r][c].axvline(s * hd, color="purple", ls=":", lw=1.0)
            axes[r][c].axvline(0, color="0.7", lw=0.5)
            axes[r][c].set_xlabel("t (raw-z displacement)", fontsize=8)
            axes[r][c].tick_params(labelsize=7)

        ax = axes[r][3]
        h = np.asarray(R["t3_D2_h"]); fac = np.abs(np.asarray(R["t3_D2_factor"]))
        ax.plot(h, fac, "-o", ms=3, color="#555555")
        ax.axhspan(1 / 3, 3, color="0.85", alpha=0.6, label="D2 plateau band [1/3,3]")
        ax.axvline(hd, color="purple", ls=":", lw=1.2, label="h_dev")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("h (spacing)", fontsize=8)
        ax.set_title("T3 D2 plateau-deviation |D2/plateau|", fontsize=9)
        ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    fig.suptitle("T8 breathing transects: does GN-eigenvalue breathing set T3's "
                 "h*? -- PROPOSED (UNCERTIFIED)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T8 breathing transects (does GN "
                                            "eigenvalue breathing set T3's h*?)")
    p.add_argument("--data-dir")
    p.add_argument("--t3-json")
    p.add_argument("--clone")
    p.add_argument("--samples", help="clone_source.npz (pooled post-burn-in z-samples)")
    p.add_argument("--out-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true",
                   help="run the offline numpy-only smoke test and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        res = run_smoke()
        if not res["pass"]:
            raise SystemExit("[T8] smoke tests FAILED")
        return

    for req in ("data_dir", "t3_json", "clone", "samples", "out_dir", "seed"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    os.makedirs(args.out_dir, exist_ok=True)

    from common import assert_x64, load_target
    assert_x64()

    # --- inputs -------------------------------------------------------------
    t3 = json.load(open(args.t3_json))
    dim = int(t3["dim"])
    clone = np.load(args.clone)
    clone_cov = np.asarray(clone["cov"], dtype=np.float64)
    if clone_cov.shape != (dim, dim):
        raise ValueError(f"clone cov {clone_cov.shape} != ({dim},{dim})")

    # x0 EXACTLY as replot: from the T3 provenance's flat index into the samples.
    src = np.load(args.samples)
    pos = np.asarray(src["position"], dtype=np.float64)
    if pos.shape[-1] != dim:
        raise ValueError(f"samples dim {pos.shape} != {dim}")
    flat = pos.reshape(-1, dim)
    prov = t3["x0_provenance"]
    x0 = flat[prov["flat_index"]].copy()
    # cross-check provenance points at THIS samples file
    if os.path.abspath(args.samples) != os.path.abspath(prov["samples_path"]):
        print(f"[T8] NOTE: --samples ({args.samples}) differs from T3 provenance "
              f"samples_path ({prov['samples_path']}); using --samples (verified "
              "byte-identical clone_source elsewhere).")

    # standardization: SAME as E1 -- per-coordinate std from pooled samples.
    std_z = np.std(flat, axis=0, ddof=1)

    # directions (reconstructed + verified sign-sensitively; RAISES on mismatch)
    verified = reconstruct_directions(t3, clone_cov)
    print(f"[T8] {len(verified)} directions verified against stored g_dot_e/sigma_dir")

    prob_model, qz, z_center, dim2, param_names = load_target(args.data_dir)
    if dim2 != dim:
        raise ValueError(f"data-dir dim {dim2} != T3 dim {dim}")
    ops = build_jax_ops(prob_model, param_names)

    # --- HARD chi^2 gate (E1 Task A) on x0 + a couple of transect endpoints ---
    rng = np.random.default_rng(args.seed)
    gate_pts = [x0]
    # two extra probe points: x0 nudged along the first two directions at ~h*
    for name in T8_DIRECTIONS[:2]:
        e = verified[name]["e_raw"]
        gate_pts.append(x0 + H_STAR * e)
    chi2_checks, worst_rel = chi2_gate(ops, gate_pts, tag="T8")

    # --- transects ----------------------------------------------------------
    per_dir = {}
    W = ops["W"]
    for name in T8_DIRECTIONS:
        d = verified[name]["t3"]
        e_raw = verified[name]["e_raw"]
        h_dev, hd_note = h_dev_for(d)
        e_hat, e_scale = standardize_direction(e_raw, std_z)
        ts = build_t_grid(h_dev)
        g_arr, l1_arr, l2_arr, ov_arr = [], [], [], []
        lam_rows, ov_rows = [], []
        for t in ts:
            z = x0 + t * e_raw
            Jz = np.asarray(ops["jac_render"](z), dtype=np.float64)   # (6400, 22)
            M_zhat = gn_metric(scale_cols(Jz, std_z), W)
            prof = quad_form_profile(M_zhat, e_hat, k=K_ATTR)
            g_arr.append(prof["g"]); l1_arr.append(prof["lambda1"])
            l2_arr.append(prof["lambda2"]); ov_arr.append(prof["overlap1"])
            lam_rows.append(prof["lambdas"]); ov_rows.append(prof["overlaps"])
        g_arr = np.asarray(g_arr)
        lam_rows = np.asarray(lam_rows); ov_rows = np.asarray(ov_rows)

        # attribution over |t| <= h_dev
        i0 = int(np.argmin(np.abs(ts)))
        o2 = ov_rows ** 2
        g_check = np.sum(lam_rows * o2, axis=1)
        g_breath = np.sum(lam_rows * o2[i0][None, :], axis=1)
        g_bend = np.sum(lam_rows[i0][None, :] * o2, axis=1)
        mask = np.abs(ts) <= h_dev
        shares = _shares_from_window(g_arr, g_breath, g_bend, mask)
        # also report a 2x-window ratio (the falsifier is phrased on |t| <= 2*h_dev)
        mask2 = np.abs(ts) <= 2.0 * h_dev
        sh2 = _shares_from_window(g_arr, g_breath, g_bend, mask2)

        # T3 D2 plateau-deviation factor for the reference panel
        h_arr = np.asarray(d["h"], dtype=np.float64)
        D2_arr = np.asarray(d["D2"], dtype=np.float64)
        plateau = float(d["summary"]["plateau_D2"])
        D2_factor = D2_arr / plateau if plateau != 0 else D2_arr * np.nan

        per_dir[name] = {
            "kind": d["kind"], "h_dev": h_dev, "h_dev_note": hd_note,
            "sigma_dir": float(d["sigma_dir"]),
            "e_scale_zhat_per_t": e_scale,
            "t": ts.tolist(),
            "g": g_arr.tolist(),
            "lambda1": l1_arr, "lambda2": l2_arr, "overlap": ov_arr,
            "g_identity_max_abs_err": float(np.max(np.abs(g_arr - g_check))),
            "attribution_window_hdev": shares,
            "attribution_window_2hdev": sh2,
            "t3_D2_h": h_arr.tolist(),
            "t3_D2_factor": D2_factor.tolist(),
            "t3_plateau_D2": plateau,
        }
        print(f"[T8] {name:32s} h_dev={h_dev:.3e} g_ratio(|t|<=h_dev)="
              f"{shares['g_ratio_max_over_min']:.2f} "
              f"breathing_share={shares['breathing_share']:.2f} "
              f"bending_share={shares['bending_share']:.2f} "
              f"(g-identity err {per_dir[name]['g_identity_max_abs_err']:.2e})")

    # --- plot + JSON --------------------------------------------------------
    plot_path = os.path.join(args.out_dir, "t8_breathing_transects.png")
    plot_transects(per_dir, plot_path)

    # transition directions (exclude stiffest, whose D2 is 'stable') for the
    # verdict read; stiffest reported for context.
    trans = [n for n in T8_DIRECTIONS if not isinstance(
        verified[n]["t3"]["summary"]["h_D2_deviate"], str)]

    doc = {
        "experiment": "T8 -- does GN-eigenvalue breathing set T3's h*?",
        "status": "proposed (UNCERTIFIED) -- grader inspects artifacts, not this summary",
        "timestamp_utc": _now(),
        "args": {k: v for k, v in vars(args).items()},
        "dim": dim,
        "param_names": param_names,
        "x0_provenance": prov,
        "standardization": {
            "note": "SAME as E1: zhat = z/std_z (per-coordinate, ddof=1, pooled "
                    "post-burn-in samples). M_zhat = gn_metric(scale_cols(J_z, std_z), W). "
                    "e_hat = (e_raw/std_z)/||e_raw/std_z|| (direction in zhat coords).",
            "std_z": std_z.tolist(),
        },
        "render_path_chi2_gate": {
            "convention": "aux = reduced chi^2 = chi2_total/event_size; "
                          "chi2_from_aux = aux * event_size (identical to E1 Task A)",
            "checks": chi2_checks, "worst_relative_error": worst_rel,
        },
        "directions": per_dir,
        "prediction": {
            "text": f"g(t) varies by >= {T8_PREDICT_GVAR:g}x within |t| <= h_dev on "
                    f"transition directions, with the lambda (breathing) share "
                    f">= {int(T8_PREDICT_LAMBDA_SHARE*100)}% of g's variation",
            "g_ratio_threshold": T8_PREDICT_GVAR,
            "breathing_share_threshold": T8_PREDICT_LAMBDA_SHARE,
        },
        "falsifier": {
            "text": f"g(t) constant within ~{T8_FALSIFY_GVAR:g}x over |t| <= 2*h_dev "
                    "while D2 transitions => breathing does NOT explain h*",
            "g_ratio_threshold_2hdev": T8_FALSIFY_GVAR,
        },
        "measured_on_transition_dirs": {
            n: {
                "g_ratio_hdev": per_dir[n]["attribution_window_hdev"]["g_ratio_max_over_min"],
                "breathing_share_hdev": per_dir[n]["attribution_window_hdev"]["breathing_share"],
                "bending_share_hdev": per_dir[n]["attribution_window_hdev"]["bending_share"],
                "g_ratio_2hdev": per_dir[n]["attribution_window_2hdev"]["g_ratio_max_over_min"],
            } for n in trans
        },
        "attribution_note": "breathing_share + bending_share need NOT sum to 1 "
                            "(cross terms between lambda and overlap changes + a "
                            "rank-continuity approximation in tracking eigenvectors "
                            "across t). Reported separately.",
        "verdict_fields": {
            "T8_breathing_sets_hstar": "proposed (UNCERTIFIED): compare per-direction "
                "g_ratio_hdev to the >=3x prediction and <=1.5x (2*h_dev) falsifier, "
                "and breathing_share to the >=70% prediction; read the transect plot "
                "against the T3 D2-deviation panel.",
            "note": "this script does NOT adjudicate; a grader inspects plots + numbers.",
        },
        "plots": {"transects": os.path.basename(plot_path)},
    }
    json_path = os.path.join(args.out_dir, "t8_results.json")
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[T8] wrote {json_path} and {plot_path}")

    print("\n=== T8 summary (PROPOSED / UNCERTIFIED) ===")
    print(f"  chi^2 render gate: PASSED, worst rel {worst_rel:.3e}")
    for n in T8_DIRECTIONS:
        s = per_dir[n]["attribution_window_hdev"]
        tag = "(transition)" if n in trans else "(D2 stable; h* range)"
        print(f"  {n:32s} {tag}: g_ratio(|t|<=h_dev)={s['g_ratio_max_over_min']:.2f} "
              f"breathing={s['breathing_share']:.2f} bending={s['bending_share']:.2f}")
    print(f"  prediction: g_ratio >= {T8_PREDICT_GVAR:g} & breathing_share >= "
          f"{T8_PREDICT_LAMBDA_SHARE:.0%} ; falsifier: g_ratio(2*h_dev) <= "
          f"{T8_FALSIFY_GVAR:g}")
    print("[T8] done.")


if __name__ == "__main__":
    main()
