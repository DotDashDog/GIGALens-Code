"""T14 -- Close the last link: does the residual Hessian term wash out the Fisher comb?

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T14, approved by the
human 2026-07-03). Builds directly on C-4 / T12 / T13':

  * T12: on sys60 the GN metric M = J^T W J has a SUBGRID COMB in lambda1 (top
    eigenvalue) vs the compact counter-image sub-pixel displacement x_c -- narrow
    teeth at the supersample=2 subgrid pitch (~0.52 px, ~1.9/px), spanning
    peak-to-trough ~30000.
  * T13' Step 6: that comb lives in the ss2 MODEL's Fisher metric UNCONDITIONALLY --
    it is IDENTICAL against the original ss2 data and against the accurate ss128 data
    d' (J is data-independent; W shifts only via the err map). Yet sampling is slow
    ONLY on the self-consistent (ss2 data + ss2 model) pairing; on d' the ss2 model
    samples FAST (min-ESS ~1100) despite the same comb.

T14 closes the last causal link: the EXACT posterior Hessian is
    H = -grad^2 logp = M - sum_i W_i r_i grad^2 m_i - grad^2 log_prior
(r_i = obs_i - m_i, m_i the model image). On self-consistent data r ~ noise => the
residual term is small => H ~ M => the posterior COMBS (E1 measured GN-dominance
exactly there). On accurate data the ~14 sigma un-fittable counter-image residual
multiplies the SAME rendering-aliasing second derivatives grad^2 m_i, so the residual
term carries an ANTI-COMB that cancels M's comb => H is SMOOTH at the tooth scale =>
the posterior does NOT inherit the comb => fast sampling.

DESIGN (verbatim from the checkpoint): the T12 top-spike dial (97 points, +/-2 px
MEASURED x_c displacement), ss2 model, run TWICE -- against the ORIGINAL ss2 data and
against d' (ss128) -- computing per point, in the SAME standardized zhat units as
E1/T12 (std_z from the ORIGINAL run's pooled samples, ONE common standardization for
both targets so curvatures are comparable):
  (i)   g_M(t) = e^T M e,  M = gn_metric(scale_cols(J_z, std_z), W)      [GN]
  (ii)  g_H(t) = e^T H_zhat e,  H_zhat = diag(std_z)(-hessian(logp))diag(std_z) [exact]
  (iii) f(t)   = logp_scalar(z) = prob_model.log_prob(z)[0]  (includes prior+logdet)
  (iv)  g_R(t) = g_M(t) - g_H(t)   (everything-not-GN quadratic form = residual+prior)
with e = the per-point TOP eigenvector of M_zhat (the counter-image-dominated stiff
direction), so g_M(t) = lambda1(t) reproduces T12's comb EXACTLY ("as T12"), and
g_H / g_R are the log-density curvature / residual+prior curvature ALONG that same
stiff direction. (Ambiguity resolved: e is the per-point stiff eigenvector, not a
fixed vector -- this makes g_M literally the T12/T13'-Step-6 comb and makes g_R the
non-GN curvature on the stiff direction, which is where the cancellation is claimed.)

CANCELLATION IS ADDITIVE: g_M = g_H + g_R exactly (per point). All detrending is done
in LINEAR space (not log lambda) precisely because the cancellation is an ADDITIVE
relationship; log space would break it, and g_H / g_R can be non-positive (H need not
be positive-definite along e when evaluated off the target's own typical set).

PREDICTIONS (restated next to every measured number in the JSON):
  ORIGINAL data: g_H tracks g_M tooth-for-tooth (detrended corr >= 0.8; tooth
    peak-to-trough within x3), f corrugated at tooth scale by >= 3 nats.
  d' (NEW): g_M combs identically (known); g_H teeth suppressed >= 10x vs original,
    f tooth corrugation reduced >= 10x, and g_R combs IN PHASE with g_M (detrended
    corr(g_R, g_M) >= 0.8) -- the direct signature of cancellation.
  FALSIFIER: d' g_H teeth >= 1/3 of g_M's while the arm demonstrably samples fast =>
    wash-out story wrong (stop and rethink before writing any mechanism claim).

This script PRODUCES artifacts + PROPOSED (UNCERTIFIED) verdict fields; it does NOT
adjudicate. A grader inspects the plots/JSON, not this printed summary.

REUSE (import, not duplicate): from t12_flank_crossing -- blob_stats,
window_flat_indices, moving_median, detrend_log, peak_to_trough, fourier_metrics,
chi2_gate, build_dial, calibrate_dial, load_sys60, _pixel_gridlines and the dial
constants; from e1_fisher_survey -- build_jax_ops, gn_metric, scale_cols, eig_desc,
TINY; from common -- assert_x64, load_target. Nothing is copied.

jax is imported ONLY inside functions so the module imports cleanly under a plain
conda python (no jax) for offline smoke tests (--smoke) of the pure-numpy analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- reused machinery (all numpy-only at import; no jax) --------------------
from t12_flank_crossing import (
    blob_stats, window_flat_indices, moving_median, detrend_log, peak_to_trough,
    fourier_metrics, chi2_gate, build_dial, calibrate_dial, load_sys60,
    _pixel_gridlines,
    N_DIAL, DIAL_RANGE_PX, PERIOD_PX, HALF_PERIODS, CHUNK,
)
from e1_fisher_survey import build_jax_ops, gn_metric, scale_cols, eig_desc, TINY

# --- pre-registered predictions / falsifiers (restated by JSON) ------------
OLD_CORR_MIN = 0.8        # ORIGINAL: detrended corr(g_H, g_M) >= 0.8
OLD_P2T_FACTOR = 3.0      # ORIGINAL: g_H tooth p2t within x3 of g_M's
OLD_F_NATS_MIN = 3.0      # ORIGINAL: f corrugated >= 3 nats at tooth scale
NEW_SUPPRESS_MIN = 10.0   # NEW: g_H teeth suppressed >= 10x vs original
NEW_F_REDUCE_MIN = 10.0   # NEW: f tooth corrugation reduced >= 10x
NEW_GR_CORR_MIN = 0.8     # NEW: detrended corr(g_R, g_M) >= 0.8 (cancellation)
FALSIFY_GH_FRAC = 1.0 / 3.0   # NEW g_H teeth >= gM/3 while fast => wash-out wrong

DEFAULT_RUN_DIR = "/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# LINEAR-space detrend analysis (reuses T12's detrend_log + peak_to_trough,
# which are generic: detrend_log subtracts a moving median of WHATEVER series it
# is handed -- here the LINEAR g / f, not log lambda).
# ===========================================================================

def detr_series(disp, y, period=PERIOD_PX):
    """Linear moving-median detrend of y vs measured displacement disp (1-px window).
    Returns (us, detr) sorted by disp -- reuses T12's detrend_log verbatim."""
    us, detr, _win, _dx = detrend_log(np.asarray(disp, float), np.asarray(y, float),
                                      period)
    return us, detr


def tooth_amplitude(disp, y, period=PERIOD_PX, half_periods=HALF_PERIODS):
    """Detrended peak-to-trough AMPLITUDE (linear, native units) over the central
    +/- half_periods px: max(detr) - min(detr). Handles non-positive y (unlike a
    log ratio). Reuses T12's peak_to_trough for the central-window masking; the
    amplitude is d_max - d_min (NOT the exp() ratio, which is meaningless in linear
    space)."""
    us, detr = detr_series(disp, y, period)
    pt = peak_to_trough(us, detr, half_periods, period)
    return {"amplitude": float(pt["d_max"] - pt["d_min"]),
            "d_max": pt["d_max"], "d_min": pt["d_min"],
            "n_in_window": pt["n_in_window"]}


def detr_corr(disp, ya, yb, period=PERIOD_PX):
    """Pearson corr of the linearly-detrended ya vs yb over the FULL scan. Both are
    sorted by the SAME disp inside detrend_log, so the samples are index-aligned."""
    _ua, da = detr_series(disp, ya, period)
    _ub, db = detr_series(disp, yb, period)
    if np.std(da) < TINY or np.std(db) < TINY:
        return float("nan")
    return float(np.corrcoef(da, db)[0, 1])


# ===========================================================================
# Offline smoke tests (numpy only; no jax/GPU)
# ===========================================================================

def _synth_dial():
    """A synthetic dial: smooth positive breathing envelope + a sharp subgrid comb.
    Returns (disp, env, comb) with comb at ~1.9/px (0.52 px pitch, as T12)."""
    disp = np.linspace(-DIAL_RANGE_PX, DIAL_RANGE_PX, N_DIAL)
    env = 1.0e5 * np.exp(0.30 * disp)                        # smooth, positive, ~4px scale
    teeth = np.clip(np.cos(2 * np.pi * disp / 0.52), 0.0, 1.0) ** 6   # sharp, subgrid
    comb = 3.0e7 * teeth
    return disp, env, comb


def smoke_suppression(verbose=True):
    """Synthetic g_M = env + comb; OLD g_H = g_M (H ~ M) minus a tiny smooth prior;
    NEW g_H = env only (comb cancelled). Recover: OLD corr(g_H,g_M) ~ 1, OLD p2t
    ratio ~ 1, NEW suppression p2t(g_H,NEW)/p2t(g_H,OLD) << 0.1, and the cancellation
    corr(g_R_NEW, g_M) ~ 1 with g_R_NEW = g_M - g_H_NEW = comb."""
    disp, env, comb = _synth_dial()
    prior = 5.0e2 * disp                                    # tiny smooth prior term
    g_M = env + comb
    g_H_old = g_M - prior                                   # OLD: H ~ M (residual~0)
    g_H_new = env - prior                                   # NEW: comb cancelled
    g_R_new = g_M - g_H_new                                 # = comb + prior ~ comb

    corr_old = detr_corr(disp, g_H_old, g_M)
    amp_gM = tooth_amplitude(disp, g_M)["amplitude"]
    amp_gH_old = tooth_amplitude(disp, g_H_old)["amplitude"]
    amp_gH_new = tooth_amplitude(disp, g_H_new)["amplitude"]
    p2t_ratio_old = amp_gH_old / amp_gM
    suppression = amp_gH_new / amp_gH_old
    corr_canc = detr_corr(disp, g_R_new, g_M)

    ok = (corr_old > 0.99 and abs(p2t_ratio_old - 1.0) < 0.05
          and suppression < 0.01 and corr_canc > 0.99)
    if verbose:
        print(f"[smoke] suppression/tracking: OLD corr(g_H,g_M)={corr_old:.4f} (~1); "
              f"OLD p2t(g_H)/p2t(g_M)={p2t_ratio_old:.4f} (~1); "
              f"NEW suppression p2t(g_H_NEW)/p2t(g_H_OLD)={suppression:.2e} (<<0.1); "
              f"cancellation corr(g_R_NEW,g_M)={corr_canc:.4f} (~1) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return {"old_corr_gH_gM": corr_old, "old_p2t_ratio": p2t_ratio_old,
            "new_suppression": suppression, "cancellation_corr_gR_gM": corr_canc,
            "amp_gM": amp_gM, "amp_gH_old": amp_gH_old, "amp_gH_new": amp_gH_new,
            "pass": bool(ok)}


def smoke_cancellation(verbose=True):
    """Explicit cancellation: g_R = comb exactly ==> detrended corr(g_R, g_M) -> 1."""
    disp, env, comb = _synth_dial()
    g_M = env + comb
    g_R = comb.copy()
    corr = detr_corr(disp, g_R, g_M)
    ok = corr > 0.99
    if verbose:
        print(f"[smoke] pure cancellation: corr(g_R=comb, g_M=env+comb)={corr:.4f} "
              f"(-> 1) -> {'PASS' if ok else 'FAIL'}")
    return {"corr_gR_gM": corr, "pass": bool(ok)}


def smoke_f_teeth(verbose=True):
    """f-teeth extraction on a planted 5-nat corrugation. The planted teeth have the
    REAL comb SHAPE -- sharp, narrow subgrid teeth on a between-teeth baseline (T12's
    comb is narrow teeth, not a smooth sinusoid; the 1-px moving MEDIAN is designed to
    ride that baseline). f = smooth trend + 5.0 * teeth, teeth in [0, 1], so the
    detrended peak-to-trough is 5.0 nats. Recover 5 +/- 0.5 nats."""
    disp = np.linspace(-DIAL_RANGE_PX, DIAL_RANGE_PX, N_DIAL)
    trend = -3000.0 + 1.0 * disp - 0.2 * disp ** 2          # gentle smooth log-density trend
    # Grid-aligned narrow teeth at the ~0.52 px subgrid pitch: height EXACTLY
    # planted_p2t on a zero baseline, so discreteness cannot clip the peak (a smooth
    # sinusoid's peak falls between grid points and undercounts). This isolates the
    # extraction machinery -- the real g_M/g_H teeth are likewise narrow (T12).
    planted_p2t = 5.0
    dx = float(np.median(np.diff(np.sort(disp))))
    period_samples = max(3, int(round(0.52 / dx)))          # ~12 samples ~ 0.52 px
    teeth = np.zeros_like(disp)
    teeth[::period_samples] = planted_p2t                   # narrow teeth on 0 baseline
    f = trend + teeth
    amp = tooth_amplitude(disp, f)["amplitude"]
    ok = abs(amp - planted_p2t) < 0.5
    if verbose:
        print(f"[smoke] f-teeth: recovered p2t={amp:.3f} nats (planted "
              f"{planted_p2t}) -> {'PASS' if ok else 'FAIL'}")
    return {"recovered_nats": amp, "planted_nats": planted_p2t, "pass": bool(ok)}


def run_smoke():
    print("=== T14 offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_suppression(); b = smoke_cancellation(); c = smoke_f_teeth()
    allpass = a["pass"] and b["pass"] and c["pass"]
    print(f"[smoke] overall: {'PASS' if allpass else 'FAIL'}")
    return {"suppression": a, "cancellation": b, "f_teeth": c, "pass": bool(allpass)}


# ===========================================================================
# NEW target loader (d' via build_prob_model -- NO qz needed; T14 never samples)
# ===========================================================================

def load_new_target(sys_dir, supersample=2):
    """Build the ss2 model on the re-simulated d' data via the qz-free
    build_prob_model(supersample) helper in systems/sys60_ss16data/system.py.
    Returns (prob_model, dim, param_names). No qz: T14 computes curvature, never samples."""
    import importlib.util
    import sys as _sys
    sys_dir = os.path.abspath(sys_dir)
    sys_py = os.path.join(sys_dir, "system.py")
    if not os.path.isfile(sys_py):
        raise FileNotFoundError(f"[T14] no system.py in {sys_dir}")
    if sys_dir not in _sys.path:
        _sys.path.insert(0, sys_dir)
    spec = importlib.util.spec_from_file_location("_t14_sys60_ss16data", sys_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_prob_model(supersample=supersample)


# ===========================================================================
# One target: g_M, g_H, f, x_c along the SHARED dial z-path Z (97 points)
# ===========================================================================

def compute_target(prob_model, param_names, Z, dial, std_z, gate_idx, tag):
    """Evaluate (g_M, g_H, f, x_c) at every dial point for one target.

    Z    : (N, dim) SHARED dial z-path (identical for both targets -- render is
           data-independent, so the path is built ONCE on the OLD model).
    std_z: (dim,) ONE common standardization (original run's pooled samples).
    e = per-point TOP eigenvector of M_zhat (the stiff/counter-image direction);
    g_M = e^T M e = lambda1 (as T12); g_H = e^T H_zhat e with H_zhat =
    diag(std_z)(-hessian(logp0))diag(std_z); f = logp0(z). Batched Jacobians +
    batched logp; Hessians one-by-one (jit once per target via ops['hess_logp'])."""
    import jax

    ops = build_jax_ops(prob_model, param_names)
    ops["render_batch"] = jax.jit(jax.vmap(ops["render"]))
    W = np.asarray(ops["W"], dtype=np.float64)
    pm = prob_model
    logp_batch = jax.jit(jax.vmap(lambda z: pm.log_prob(z)[0]))
    hess_logp = ops["hess_logp"]                   # jax.jit(jax.hessian(logp0)) -- jit once
    batched_jac = jax.jit(jax.vmap(ops["jac_render"]))

    # chi^2 render gate against THIS target's own log_prob aux (obs/err differ).
    gate_checks, gate_worst = chi2_gate(ops, [np.asarray(Z[i], float) for i in gate_idx],
                                        tag=f"T14/{tag}")

    N, dim = Z.shape
    g_M = np.empty(N); g_H = np.empty(N); f = np.empty(N)
    x_c = np.empty(N); y_c = np.empty(N); A = np.empty(N)
    e_all = np.empty((N, dim))

    t0 = time.perf_counter()
    for lo in range(0, N, CHUNK):
        hi = min(N, lo + CHUNK)
        Js = np.asarray(batched_jac(Z[lo:hi]), dtype=np.float64)      # (nb, 6400, dim)
        Ims = np.asarray(ops["render_batch"](Z[lo:hi]), dtype=np.float64)
        fs = np.asarray(logp_batch(Z[lo:hi]), dtype=np.float64)       # (nb,)
        for k in range(hi - lo):
            j = lo + k
            M = gn_metric(scale_cols(Js[k], std_z), W)                # standardized GN
            w, V = eig_desc(M)
            e = V[:, 0]                                               # stiff direction
            g_M[j] = float(w[0])                                      # e^T M e = lambda1
            Hraw = np.asarray(hess_logp(Z[j]), dtype=np.float64)      # hessian(logp0)
            Hz = std_z[:, None] * (-Hraw) * std_z[None, :]            # H_zhat = -grad^2 logp
            g_H[j] = float(e @ Hz @ e)
            f[j] = float(fs[k])
            bs = blob_stats(Ims[k])
            x_c[j] = bs["x_c"]; y_c[j] = bs["y_c"]; A[j] = bs["A"]
            e_all[j] = e
        if (lo // CHUNK) % 1 == 0:      # print progress per chunk (~32 pts)
            print(f"[T14] {tag}: {hi:3d}/{N} points ({time.perf_counter()-t0:5.1f}s)")

    g_R = g_M - g_H                                                   # everything-not-GN

    # center displacement on the dial=0 (base) measured centroid, exactly as T12.
    i0 = int(np.argmin(np.abs(dial)))
    disp = x_c - float(x_c[i0])

    # per-point stiff-direction stability (diagnostic): |e_j . e_{j-1}|.
    e_overlap = np.ones(N)
    for j in range(1, N):
        e_overlap[j] = abs(float(e_all[j] @ e_all[j - 1]))

    return {
        "tag": tag,
        "gate": {"checks": gate_checks, "worst_relative_error": gate_worst},
        "g_M": g_M, "g_H": g_H, "g_R": g_R, "f": f,
        "x_c": x_c, "y_c": y_c, "A": A, "disp": disp, "i0": i0,
        "e_top": e_all, "e_overlap": e_overlap,
        "W": W,
    }


# ===========================================================================
# Analysis (per target + cross-target), reusing T12's linear detrend
# ===========================================================================

def analyze(old, new, std_z):
    """All T14 metrics + pre-registered comparisons. LINEAR-space detrend throughout
    (cancellation is additive: g_M = g_H + g_R)."""
    disp_o = old["disp"]; disp_n = new["disp"]

    def amp(res, key, disp):
        return tooth_amplitude(disp, res[key])["amplitude"]

    amp_gM_old = amp(old, "g_M", disp_o); amp_gH_old = amp(old, "g_H", disp_o)
    amp_gR_old = amp(old, "g_R", disp_o); amp_f_old = amp(old, "f", disp_o)
    amp_gM_new = amp(new, "g_M", disp_n); amp_gH_new = amp(new, "g_H", disp_n)
    amp_gR_new = amp(new, "g_R", disp_n); amp_f_new = amp(new, "f", disp_n)

    # tracking (per target): detrended corr(g_H, g_M)
    corr_gH_gM_old = detr_corr(disp_o, old["g_H"], old["g_M"])
    corr_gH_gM_new = detr_corr(disp_n, new["g_H"], new["g_M"])
    # cancellation: detrended corr(g_R, g_M)
    corr_gR_gM_old = detr_corr(disp_o, old["g_R"], old["g_M"])
    corr_gR_gM_new = detr_corr(disp_n, new["g_R"], new["g_M"])

    # cross-target: g_M should be ~identical (M is data-independent; W ~identical)
    gM_o = np.asarray(old["g_M"]); gM_n = np.asarray(new["g_M"])
    gM_cross_corr = float(np.corrcoef(gM_o, gM_n)[0, 1])
    gM_max_rel_diff = float(np.max(np.abs(gM_n - gM_o) / np.maximum(np.abs(gM_o), TINY)))

    # suppression ratios (NEW vs OLD, same definition)
    suppression_gH = amp_gH_old / amp_gH_new if amp_gH_new > 0 else float("inf")
    reduction_f = amp_f_old / amp_f_new if amp_f_new > 0 else float("inf")
    old_p2t_ratio = amp_gH_old / amp_gM_old if amp_gM_old > 0 else float("nan")
    new_gH_over_gM = amp_gH_new / amp_gM_new if amp_gM_new > 0 else float("nan")

    # comb frequency of detrended g_M (diagnostic; expect ~1.9/px subgrid, T12)
    def peak_freq(disp, y):
        us, detr = detr_series(disp, y)
        return fourier_metrics(us, detr)["recovered_peak_freq"]

    checks = {
        "OLD": {
            "corr_gH_gM": {"measured": corr_gH_gM_old, "predicted": f">= {OLD_CORR_MIN}",
                           "meets": bool(corr_gH_gM_old >= OLD_CORR_MIN)},
            "gH_p2t_over_gM_p2t": {"measured": old_p2t_ratio,
                                   "predicted": f"within x{OLD_P2T_FACTOR:g} "
                                                f"([{1/OLD_P2T_FACTOR:.3f}, {OLD_P2T_FACTOR:g}])",
                                   "meets": bool(1.0/OLD_P2T_FACTOR <= old_p2t_ratio <= OLD_P2T_FACTOR)},
            "f_teeth_nats": {"measured": amp_f_old, "predicted": f">= {OLD_F_NATS_MIN} nats",
                             "meets": bool(amp_f_old >= OLD_F_NATS_MIN)},
        },
        "NEW": {
            "gH_suppression_old_over_new": {"measured": suppression_gH,
                                            "predicted": f">= {NEW_SUPPRESS_MIN}x",
                                            "meets": bool(suppression_gH >= NEW_SUPPRESS_MIN)},
            "f_reduction_old_over_new": {"measured": reduction_f,
                                         "predicted": f">= {NEW_F_REDUCE_MIN}x",
                                         "meets": bool(reduction_f >= NEW_F_REDUCE_MIN)},
            "corr_gR_gM": {"measured": corr_gR_gM_new, "predicted": f">= {NEW_GR_CORR_MIN}",
                           "meets": bool(corr_gR_gM_new >= NEW_GR_CORR_MIN)},
            "falsifier_gH_over_gM": {"measured": new_gH_over_gM,
                                     "falsifier": f">= {FALSIFY_GH_FRAC:.3f} (g_H >= g_M/3) "
                                                  "while arm samples fast => wash-out wrong",
                                     "fired": bool(new_gH_over_gM >= FALSIFY_GH_FRAC)},
        },
        "cross_target": {
            "gM_corr_old_vs_new": gM_cross_corr,
            "gM_max_rel_diff_old_vs_new": gM_max_rel_diff,
            "note": "M = J^T W J is data-independent up to the small W (err-map) shift; "
                    "g_M should be ~identical between targets.",
        },
    }

    amplitudes = {
        "amp_gM_old": amp_gM_old, "amp_gH_old": amp_gH_old, "amp_gR_old": amp_gR_old,
        "amp_f_old_nats": amp_f_old,
        "amp_gM_new": amp_gM_new, "amp_gH_new": amp_gH_new, "amp_gR_new": amp_gR_new,
        "amp_f_new_nats": amp_f_new,
    }
    corrs = {
        "corr_gH_gM_old": corr_gH_gM_old, "corr_gH_gM_new": corr_gH_gM_new,
        "corr_gR_gM_old": corr_gR_gM_old, "corr_gR_gM_new": corr_gR_gM_new,
    }
    diag = {
        "combfreq_gM_old_per_px": peak_freq(disp_o, old["g_M"]),
        "combfreq_gM_new_per_px": peak_freq(disp_n, new["g_M"]),
        "e_top_min_consecutive_overlap_old": float(np.min(old["e_overlap"][1:])),
        "e_top_min_consecutive_overlap_new": float(np.min(new["e_overlap"][1:])),
    }
    return {"amplitudes": amplitudes, "correlations": corrs, "checks": checks,
            "diagnostics": diag}


# ===========================================================================
# Plots
# ===========================================================================

def _yscale_g(ax, *arrays):
    allpos = all(np.all(np.asarray(a) > 0) for a in arrays)
    ax.set_yscale("log" if allpos else "symlog")


def plot_main(old, new, ana, out_path):
    """Fig 1 (4x2): rows = g_M, g_H, f, detrended overlays; cols = OLD, NEW.
    Row 3: col0 = detrended g_H vs g_M (OLD tracking); col1 = detrended g_R vs g_M
    (NEW cancellation). log-y for g's (symlog where non-positive), pixel gridlines."""
    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    cols = [("OLD (ss2 data)", old, "#8a3b3b"), ("NEW / d' (ss128 data)", new, "#2a6a8a")]

    for c, (title, res, color) in enumerate(cols):
        disp = np.asarray(res["disp"]); x_c = np.asarray(res["x_c"])
        x_c0 = float(x_c[res["i0"]])
        # row 0: g_M
        ax = axes[0, c]; _pixel_gridlines(ax, x_c0, x_c)
        ax.plot(disp, res["g_M"], "-o", ms=2.5, color=color, lw=1.0)
        _yscale_g(ax, res["g_M"])
        ax.set_title(f"{title}: g_M = e^T M e (GN, =lambda1)", fontsize=10)
        ax.set_ylabel("g_M (std curvature)")
        # row 1: g_H
        ax = axes[1, c]; _pixel_gridlines(ax, x_c0, x_c)
        ax.plot(disp, res["g_H"], "-o", ms=2.5, color=color, lw=1.0)
        _yscale_g(ax, res["g_H"])
        ax.set_title(f"{title}: g_H = e^T (-grad^2 logp) e (exact)", fontsize=10)
        ax.set_ylabel("g_H (std curvature)")
        # row 2: f
        ax = axes[2, c]; _pixel_gridlines(ax, x_c0, x_c)
        ax.plot(disp, res["f"], "-o", ms=2.5, color=color, lw=1.0)
        ax.set_title(f"{title}: f = logp (nats)", fontsize=10)
        ax.set_ylabel("logp (nats)")
        for r in range(3):
            axes[r, c].set_xlabel("measured x_c displacement (px)")
            axes[r, c].tick_params(labelsize=7)

    # row 3 col 0: OLD tracking (detrended g_H vs g_M)
    ax = axes[3, 0]
    uo, dM = detr_series(old["disp"], old["g_M"])
    _uo, dH = detr_series(old["disp"], old["g_H"])
    ax.plot(uo, dM, "-o", ms=2.5, color="#333333", lw=1.0, label="detr g_M")
    ax.plot(uo, dH, "-s", ms=2.0, color="#8a3b3b", lw=0.9, alpha=0.8, label="detr g_H")
    ax.axvspan(-HALF_PERIODS, HALF_PERIODS, color="0.92", zorder=0)
    ax.set_title(f"OLD: detrended g_H tracks g_M  "
                 f"(corr={ana['correlations']['corr_gH_gM_old']:.3f})", fontsize=10)
    ax.set_xlabel("measured x_c displacement (px)")
    ax.set_ylabel("detrended (linear, std curvature)")
    ax.legend(fontsize=7)

    # row 3 col 1: NEW cancellation (detrended g_R vs g_M)
    ax = axes[3, 1]
    un, dMn = detr_series(new["disp"], new["g_M"])
    _un, dRn = detr_series(new["disp"], new["g_R"])
    _un2, dHn = detr_series(new["disp"], new["g_H"])
    ax.plot(un, dMn, "-o", ms=2.5, color="#333333", lw=1.0, label="detr g_M")
    ax.plot(un, dRn, "-^", ms=2.5, color="#2a6a8a", lw=1.0, label="detr g_R (=g_M-g_H)")
    ax.plot(un, dHn, "-s", ms=2.0, color="#c08a2a", lw=0.8, alpha=0.7,
            label="detr g_H (suppressed)")
    ax.axvspan(-HALF_PERIODS, HALF_PERIODS, color="0.92", zorder=0)
    ax.set_title(f"NEW: cancellation -- detr g_R in phase with g_M  "
                 f"(corr={ana['correlations']['corr_gR_gM_new']:.3f})", fontsize=10)
    ax.set_xlabel("measured x_c displacement (px)")
    ax.set_ylabel("detrended (linear, std curvature)")
    ax.legend(fontsize=7)
    for c in (0, 1):
        axes[3, c].tick_params(labelsize=7)

    fig.suptitle("T14: exact Hessian vs GN comb along the T12 top-spike dial, ss2 model "
                 "on OLD (ss2) vs NEW/d' (ss128) data -- PROPOSED (UNCERTIFIED)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_f_nats(old, new, ana, out_path):
    """Fig 2 (small): detrended-f comparison in NATS (OLD vs NEW), the sampler-relevant
    corrugation."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    uo, dfo = detr_series(old["disp"], old["f"])
    un, dfn = detr_series(new["disp"], new["f"])
    ax.plot(uo, dfo, "-o", ms=3, color="#8a3b3b",
            label=f"OLD (p2t={ana['amplitudes']['amp_f_old_nats']:.2f} nats)")
    ax.plot(un, dfn, "-^", ms=3, color="#2a6a8a",
            label=f"NEW/d' (p2t={ana['amplitudes']['amp_f_new_nats']:.2f} nats)")
    ax.axvspan(-HALF_PERIODS, HALF_PERIODS, color="0.92", zorder=0,
               label="+/-1.5 px window")
    ax.set_xlabel("measured x_c displacement (px)")
    ax.set_ylabel("detrended log-density f (nats)")
    red = ana["checks"]["NEW"]["f_reduction_old_over_new"]["measured"]
    ax.set_title(f"T14: log-density tooth corrugation (nats) -- reduction OLD/NEW = "
                 f"{red:.1f}x -- PROPOSED (UNCERTIFIED)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T14 exact-Hessian vs GN-comb dial")
    p.add_argument("--old-data-dir", help="systems/sys60 (original ss2 data)")
    p.add_argument("--new-sys-dir", help="systems/sys60_ss16data (d' via build_prob_model)")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                   help="original MCLMC run dir (arrays.npz -> z0 + std_z)")
    p.add_argument("--t10-dir", help="T10 output dir (spike_list.json: top spike z0)")
    p.add_argument("--out-dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="run offline smoke tests and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        res = run_smoke()
        out = None
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out = os.path.join(args.out_dir, "t14_smoke.json")
            with open(out, "w") as f:
                json.dump(res, f, indent=2)
        if not res["pass"]:
            raise SystemExit("[T14] smoke tests FAILED")
        return

    for req in ("old_data_dir", "new_sys_dir", "t10_dir", "out_dir"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    os.makedirs(args.out_dir, exist_ok=True)

    from common import assert_x64, load_target

    # --- z0 (T12 top census spike) + std_z (original run pooled samples) ----
    spike_list = json.load(open(os.path.join(args.t10_dir, "spike_list.json")))
    top = next(s for s in spike_list["spikes"] if s["rank"] == 0)
    print(f"[T14] top spike: seg{top['segment']} ch{top['chain']} step{top['step']} "
          f"lam1={top['lambda1']:.3e}")

    arrays = np.load(os.path.join(args.run_dir, "arrays.npz"))
    samples_z = np.asarray(arrays["samples_z"], dtype=np.float64)   # (C, R, dim)
    C, Rr, dim = samples_z.shape
    std_z = np.std(samples_z.reshape(-1, dim), axis=0, ddof=1)      # ONE common std
    z0 = samples_z[int(top["chain"]), int(top["step"])].copy()
    print(f"[T14] samples_z {samples_z.shape}; std_z[:3]={std_z[:3]}; z0[:3]={z0[:3]}")

    assert_x64()

    # --- OLD target (original ss2 data) -------------------------------------
    print("[T14] === building OLD target (systems/sys60, ss2 data) ===")
    old_seq, _qz, _zc, dim_o, names_o = load_target(args.old_data_dir)
    if dim_o != dim:
        raise ValueError(f"[T14] OLD dim {dim_o} != run dim {dim}")

    # --- SHARED dial z-path: built ONCE on the OLD model. render(z) is the MODEL
    # image (data-independent), so calibration + Z are IDENTICAL for both targets;
    # evaluating both targets at the SAME Z guarantees an identical z-path (T12 dial).
    old_ops = build_jax_ops(old_seq, names_o)
    theta0, theta_to_z, rt_err = build_dial(old_seq, z0)
    R, d_dir, c0 = calibrate_dial(theta_to_z, old_ops)
    print(f"[T14] dial calibrated: R=\n{R}\n d_dir={d_dir} x_c0={c0[0]:.4f} "
          f"y_c0={c0[1]:.4f} rt_err={rt_err:.2e}")
    dial = np.linspace(-DIAL_RANGE_PX, DIAL_RANGE_PX, N_DIAL)
    Z = np.stack([np.asarray(theta_to_z(float(dial[i] * d_dir[0]),
                                        float(dial[i] * d_dir[1])), dtype=np.float64)
                  for i in range(N_DIAL)])
    gate_idx = [0, N_DIAL // 2, N_DIAL - 1]     # 3 points along the dial for the gate

    # --- NEW target (d' via build_prob_model; no qz) ----------------
    print("[T14] === building NEW target (systems/sys60_ss16data, d' ss128 data) ===")
    new_seq, dim_n, names_n = load_new_target(args.new_sys_dir, supersample=2)
    if dim_n != dim:
        raise ValueError(f"[T14] NEW dim {dim_n} != run dim {dim}")
    if list(names_n) != list(names_o):
        raise ValueError(f"[T14] param-name order differs OLD vs NEW:\n{names_o}\n{names_n}")

    # --- compute both targets at the SAME Z ---------------------------------
    old = compute_target(old_seq, names_o, Z, dial, std_z, gate_idx, "OLD")
    new = compute_target(new_seq, names_n, Z, dial, std_z, gate_idx, "NEW")

    # --- W difference over the counter-image window -------------------------
    win_idx = window_flat_indices()
    Wo = old["W"][win_idx]; Wn = new["W"][win_idx]
    W_max_rel_diff = float(np.max(np.abs(Wn - Wo) / np.maximum(np.abs(Wo), TINY)))
    print(f"[T14] max relative W diff over counter-image window = {W_max_rel_diff:.3e}")

    # --- analysis -----------------------------------------------------------
    ana = analyze(old, new, std_z)

    # --- figures ------------------------------------------------------------
    fig1 = os.path.join(args.out_dir, "t14_main.png")
    fig2 = os.path.join(args.out_dir, "t14_f_nats.png")
    plot_main(old, new, ana, fig1)
    plot_f_nats(old, new, ana, fig2)

    # --- JSON (all raw arrays + metrics + gates + W-diff) -------------------
    def raw(res):
        return {k: np.asarray(res[k]).tolist() for k in
                ("g_M", "g_H", "g_R", "f", "x_c", "y_c", "A", "disp", "e_overlap")}

    out = {
        "experiment": "T14",
        "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(),
        "seed": args.seed,
        "inputs": {
            "old_data_dir": os.path.abspath(args.old_data_dir),
            "new_sys_dir": os.path.abspath(args.new_sys_dir),
            "run_dir": os.path.abspath(args.run_dir),
            "t10_dir": os.path.abspath(args.t10_dir),
            "top_spike": top,
        },
        "design_notes": {
            "e_direction": "per-point TOP eigenvector of M_zhat (stiff/counter-image "
                           "direction); g_M = e^T M e = lambda1 (as T12).",
            "H_zhat": "diag(std_z) (-hessian(prob_model.log_prob(z)[0])) diag(std_z); "
                      "INCLUDES prior + logdet (prior smooth at tooth scale).",
            "g_R": "g_M - g_H = everything-not-GN quadratic form = residual term "
                   "(sum W_i r_i grad^2 m_i) + prior Hessian, along e.",
            "detrend": "LINEAR moving median (1-px window of MEASURED displacement), "
                       "reusing T12 detrend_log; amplitudes are d_max-d_min over the "
                       "central +/-1.5 px (linear, native units / nats). Linear because "
                       "the cancellation g_M = g_H + g_R is ADDITIVE.",
            "std_z": "ONE common standardization from the ORIGINAL run's pooled samples "
                     "(so OLD/NEW curvatures are directly comparable).",
            "z_path": "SHARED: calibrated ONCE on the OLD model (render is "
                      "data-independent) and both targets evaluated at the SAME 97 Z.",
        },
        "std_z": std_z.tolist(),
        "z0": z0.tolist(),
        "param_names": list(names_o),
        "dial": dial.tolist(),
        "calibration": {"R": R.tolist(), "d_dir_px_to_theta": d_dir.tolist(),
                        "x_c0": float(c0[0]), "y_c0": float(c0[1]),
                        "roundtrip_err": rt_err},
        "gates": {"OLD": old["gate"], "NEW": new["gate"],
                  "note": "the two gates reconcile against DIFFERENT observed/err maps "
                          "(OLD ss2 data vs NEW d'); both must pass -- that is expected."},
        "W_difference": {
            "max_relative_diff_counter_image_window": W_max_rel_diff,
            "window_pixels": win_idx.tolist(),
            "note": "W = 1/err^2; err map derives from each target's OWN observed image, "
                    "so W differs slightly. M and hence g_M differ by ~this over the "
                    "counter-image window.",
        },
        "raw": {"OLD": raw(old), "NEW": raw(new)},
        "amplitudes": ana["amplitudes"],
        "correlations": ana["correlations"],
        "diagnostics": ana["diagnostics"],
        "prereg_checks": ana["checks"],
        "figures": {"main": os.path.abspath(fig1), "f_nats": os.path.abspath(fig2)},
        "verdict": "proposed (UNCERTIFIED) -- no adjudication (a grader inspects the "
                   "plots/JSON, not this script).",
    }
    out_json = os.path.join(args.out_dir, "t14_results.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # --- printed summary (restates prereg next to measured) -----------------
    print("\n=== T14 summary (PROPOSED, UNCERTIFIED) ===")
    print(f"  W max rel diff (counter-image window): {W_max_rel_diff:.3e}")
    print(f"  cross-target g_M corr OLD vs NEW: "
          f"{ana['checks']['cross_target']['gM_corr_old_vs_new']:.5f} "
          f"(max rel diff {ana['checks']['cross_target']['gM_max_rel_diff_old_vs_new']:.3e})")
    for arm in ("OLD", "NEW"):
        print(f"  --- {arm} ---")
        for k, v in ana["checks"][arm].items():
            print(f"    {k}: {v}")
    print(f"  amplitudes: {ana['amplitudes']}")
    print(f"  saved: {out_json}\n           {fig1}\n           {fig2}")


if __name__ == "__main__":
    main()
