"""T12 -- Flank-crossing / pixelation check: the MICRO-mechanism of the lambda1 spikes.

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T12, approved by the
human 2026-07-02). Builds directly on C-3 / T10 / T11: on sys60 essentially ALL the
local Gauss-Newton stiffness lambda1 = top eigenvalue of M_zhat is carried by ONE
compact lensed counter-image at pixel (row 57, col 12) of the 80x80 grid (observed
peak 15.3 vs bg rms 0.2 ~ 75 sigma). lambda1 spikes x10-500 along the on-ridge chain
(T10). T12 asks the micro-question: does lambda1 OSCILLATE with the SUB-PIXEL PHASE
of the counter-image centroid x_c (a steep flank sweeping across pixel sample points)
-- and is that oscillation PHYSICAL pixel-integration information (stable under finer
rendering) or a RENDERING-FIDELITY ARTIFACT of supersample=2 (shrinks at
supersample=4)?

THREE PARTS (verbatim from the checkpoint):
  Part 1 -- dial scan (supersample=2). At the TOP census spike, one MID spike (rank 5
    of the 12), and one matched BASELINE (spike_list.json), impose a controlled
    SOURCE-center shift in theta-space (planes/1/light/0/center_x, center_y), mapped
    BACK through the bijector to z, that translates the counter-image centroid x_c by
    a MEASURED +/-2 pixels in 97 uniform dial steps. First CALIBRATE: finite-difference
    renders w.r.t. the two source-center thetas -> the 2x2 response d(x_c,y_c)/d(cx,cy);
    pick the theta-direction that moves x_c purely along image-x, unit-normalized so the
    dial variable is EXPECTED image-plane displacement in pixels. Per scan point: lambda1
    (census convention), lambda1 restricted to the 15x15 blob window (M rebuilt from the
    window's rows of J only), the MEASURED centroid x_c, and the blob flux A. Detrend
    log lambda1 by a moving median (window = one pixel period of measured displacement);
    oscillation peak-to-trough ratio (central +/-1.5 periods) + FFT amplitude at the
    pixel frequency (1/px) vs the local continuum. PRE-REGISTERED: predict ratio >= 10
    and Fourier peak >= 5x continuum; FALSIFY if ratio <= 2.
  Part 2 -- supersample control. Rebuild via load_target(supersample=4) (own chi^2
    gate) and rerun the TOP-spike scan identically. Report amplitude ratio ss4/ss2 of
    the pixel-frequency Fourier amplitude AND of the detrended peak-to-trough. Branch:
    >= 0.7 physical / <= 0.5 artifact / between = mixed. (The ss4 likelihood is a
    slightly different posterior; the comparison is of oscillation AMPLITUDE at matched
    dial points, NOT of absolute lambda1.)
  Part 3 -- census correlational cross-check. Reconstruct the 2048 census z-points from
    t10_results.json (chain, start, segment_len) + arrays.npz; ONE render each (batched
    vmap, chunked; renders only, no Jacobians). Per step: x_c, A, phases phi_x=frac(x_c),
    phi_y=frac(y_c). REUSE the census lambda1 arrays from t10_results.json (do NOT
    recompute). OLS of log lambda1 on [1, log A] (R2_flux) vs [1, log A, first two
    Fourier harmonics of phi_x and phi_y] (R2_full); DeltaR2 = R2_full - R2_flux.
    PREDICT DeltaR2 >= 0.4; FALSIFY <= 0.1; if flux alone gives R2 >= 0.6 with
    DeltaR2 <= 0.1, note the smooth magnification-branch signature explicitly.

REUSE (import, not duplicate): gn_metric / scale_cols / eig_desc / build_jax_ops /
TINY from e1_fisher_survey. The standardized-units metric is EXACTLY the census one:
M_zhat = gn_metric(scale_cols(J_z, std_z), W), lambda1 = top eigenvalue, with std_z
from the run's pooled results samples (a property of the RUN, identical for ss2/ss4).
The HARD chi^2 render-path gate (E1 Task A) is replicated VERBATIM and re-run at
startup for EACH model build (ss2 AND ss4) -- it reconciles against that model's OWN
log_prob, so it must pass for both.

This script PRODUCES artifacts + PROPOSED (UNCERTIFIED) verdict fields; it does NOT
adjudicate (a grader inspects plots/JSON, not the printed summary).

jax is imported ONLY inside functions so the module imports cleanly under a plain
conda python (no jax) for offline smoke tests (--smoke) of the pure-numpy machinery:
blob_stats centroid on a synthetic Gaussian blob; detrend+FFT on a synthetic
oscillation-plus-trend; the OLS DeltaR2 on synthetic phase-modulated data.
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

# Reused verbatim (module-level numpy only; no jax/scipy at import time).
from e1_fisher_survey import gn_metric, scale_cols, eig_desc, build_jax_ops, TINY

# --- pre-registered constants (T12 checkpoint) -----------------------------
IMG_SIDE = 80
HOTSPOT_ROW = 57                 # nominal counter-image pixel (T11)
HOTSPOT_COL = 12
BLOB_HALF = 7                    # 15x15 window (2*7+1)
GUARD_FACTOR = 10.0              # blob lost if window max < 10x edge-median
SRC_CX_KEY = "planes/1/light/0/center_x"   # source-center theta keys (flat dict)
SRC_CY_KEY = "planes/1/light/0/center_y"
DELTA_THETA = 1e-3               # finite-difference step in theta (source center)
DIAL_RANGE_PX = 2.0              # scan +/- this many EXPECTED px of x_c displacement
N_DIAL = 97                      # uniform dial steps
PERIOD_PX = 1.0                  # one pixel period (measured-displacement units)
HALF_PERIODS = 1.5               # peak-to-trough window = central +/-1.5 periods
PIX_FREQ = 1.0                   # pixel frequency (cycles / px)
CONT_BAND = (0.3, 3.0)           # continuum band (cycles / px)
EXCLUDE_FRAC = 0.15              # exclude +/-15% around the pixel frequency
N_FFT_RESAMPLE = 512             # uniform-resample length for the FFT
CHUNK = 32                       # batched-jacfwd / render chunk

# Pre-registered predictions / falsifiers (restated next to measured numbers).
PREDICT_RATIO = 10.0             # detrended peak-to-trough >= 10
PREDICT_FOURIER = 5.0            # pixel-freq Fourier peak >= 5x continuum
FALSIFY_RATIO = 2.0             # peak-to-trough <= 2 => sub-pixel phase does not explain
SS4_PHYSICAL = 0.7               # amplitude ratio ss4/ss2 >= 0.7 => physical
SS4_ARTIFACT = 0.5               # <= 0.5 => rendering artifact (0.5-0.7 = mixed)
PREDICT_DR2 = 0.4                # census DeltaR2(phase | flux) >= 0.4
FALSIFY_DR2 = 0.1                # DeltaR2 <= 0.1 => phase does not explain the variance
FLUX_R2_NOTE = 0.6               # flux-alone R2 >= 0.6 with small DeltaR2 => magnif branch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pure-numpy blob / oscillation / regression machinery (offline-testable)
# ===========================================================================

def blob_stats(image, row0=HOTSPOT_ROW, col0=HOTSPOT_COL, half=BLOB_HALF,
               guard_factor=GUARD_FACTOR, side=IMG_SIDE):
    """Windowed flux-weighted centroid + flux of the counter-image blob.

    image : (side, side) or (side*side,) model image.
    Returns dict: x_c (column centroid, image-x, sub-pixel px), y_c (row centroid,
    image-y), A (window flux sum), peak_above (window max / window edge-median),
    plus the window bounds. GUARD: raises if peak_above < guard_factor (blob lost).
    """
    img = np.asarray(image, dtype=np.float64).reshape(side, side)
    r_lo, r_hi = row0 - half, row0 + half + 1
    c_lo, c_hi = col0 - half, col0 + half + 1
    if r_lo < 0 or c_lo < 0 or r_hi > side or c_hi > side:
        raise ValueError(f"[T12] blob window [{r_lo}:{r_hi},{c_lo}:{c_hi}] out of "
                         f"bounds for {side}x{side} image.")
    win = img[r_lo:r_hi, c_lo:c_hi]
    # edge (border) pixels of the window
    edge = np.concatenate([win[0, :], win[-1, :], win[1:-1, 0], win[1:-1, -1]])
    edge_med = float(np.median(edge))
    edge_mad = float(np.median(np.abs(edge - edge_med)))
    wmax = float(np.max(win))
    # PEDESTAL SUBTRACTION: the window sits on the extended lens-light wing
    # (~2 counts at 30 px from center), so the raw max/edge-median ratio is
    # meaningless (first run: 17.2/2.08 = 8.3 despite a healthy blob) and a raw
    # flux-weighted centroid is biased toward the window center. Subtract the
    # edge-median pedestal, then guard on CONTRAST above pedestal: >= guard_factor
    # x max(edge MAD, background rms 0.2).
    peak_above = wmax - edge_med
    # Guard vs the BACKGROUND RMS (0.2), not the edge MAD: MAD reflects pedestal
    # curvature/blob-wing structure, not noise (2nd run: a healthy 35-sigma blob
    # at peak_above=7.0 tripped a MAD-scaled guard when the dial dimmed the
    # counter-image 15->7 -- real magnification change, not blob loss).
    # 10 x 0.2 = 2.0 still catches actual disappearance.
    guard_floor = guard_factor * 0.2
    if peak_above < guard_floor:
        raise RuntimeError(
            f"[T12] blob LOST at ({row0},{col0}): peak-above-pedestal "
            f"{peak_above:.3e} < {guard_factor:g}x max(edge MAD {edge_mad:.3e}, "
            f"bg rms 0.2) = {guard_floor:.3e} (pedestal {edge_med:.3e}). The "
            "counter-image left the fixed window -- STOP (dial range too large or "
            "wrong hotspot).")
    # Fit a PLANE to the edge pixels (lens-light wing has a gradient across the
    # window; subtracting only the median level leaves a tilt that biases the
    # centroid by ~0.1 px). Least-squares [1, r, c] on the border ring.
    nwin = win.shape[0]
    rr_e = np.concatenate([np.zeros(nwin), np.full(nwin, nwin - 1),
                           np.arange(1, nwin - 1), np.arange(1, nwin - 1)])
    cc_e = np.concatenate([np.arange(nwin), np.arange(nwin),
                           np.zeros(nwin - 2), np.full(nwin - 2, nwin - 1)])
    vals_e = np.concatenate([win[0, :], win[-1, :], win[1:-1, 0], win[1:-1, -1]])
    Ae = np.column_stack([np.ones_like(rr_e), rr_e, cc_e])
    coef, *_ = np.linalg.lstsq(Ae, vals_e, rcond=None)
    rr_w, cc_w = np.mgrid[0:nwin, 0:nwin]
    plane = coef[0] + coef[1] * rr_w + coef[2] * cc_w
    sub = np.clip(win - plane, 0.0, None)
    rows = np.arange(r_lo, r_hi)[:, None]
    cols = np.arange(c_lo, c_hi)[None, :]
    tot = float(np.sum(sub))
    x_c = float(np.sum(sub * cols) / tot)     # column centroid = image-x
    y_c = float(np.sum(sub * rows) / tot)     # row centroid    = image-y
    return {"x_c": x_c, "y_c": y_c, "A": tot, "pedestal": edge_med,
            "peak_above_pedestal": peak_above,
            "window": [r_lo, r_hi, c_lo, c_hi]}


def window_flat_indices(row0=HOTSPOT_ROW, col0=HOTSPOT_COL, half=BLOB_HALF,
                        side=IMG_SIDE):
    """Row-major flat pixel indices of the 15x15 blob window (for windowed J rows)."""
    rr, cc = np.meshgrid(np.arange(row0 - half, row0 + half + 1),
                         np.arange(col0 - half, col0 + half + 1), indexing="ij")
    return (rr * side + cc).ravel().astype(np.int64)


def moving_median(y, win):
    """Centered moving median (window `win`, odd); edges shrink the window."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    win = int(win)
    if win < 1:
        win = 1
    if win % 2 == 0:
        win += 1
    h = win // 2
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - h)
        hi = min(n, i + h + 1)
        out[i] = np.median(y[lo:hi])
    return out


def detrend_log(u, log_lambda, period=PERIOD_PX):
    """Detrend log lambda1 by a moving median with window = one pixel period of the
    MEASURED displacement u (px). Returns (u_sorted, detrended, win_samples, dx)."""
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(log_lambda, dtype=np.float64)
    order = np.argsort(u)
    us = u[order]
    ys = y[order]
    dx = float(np.median(np.diff(us))) if us.size > 1 else 1.0
    win = max(3, int(round(period / dx)) if dx > 0 else 3)
    med = moving_median(ys, win)
    return us, ys - med, int(win if win % 2 else win + 1), dx


def peak_to_trough(u, detr, half_periods=HALF_PERIODS, period=PERIOD_PX):
    """Peak-to-trough RATIO of the detrended lambda1 within the central +/-1.5
    periods: exp(max(d) - min(d)) (d is detrended LOG lambda1). Also returns the
    log half-amplitude 0.5*(d_max - d_min) and the number of points in the window."""
    u = np.asarray(u, dtype=np.float64)
    d = np.asarray(detr, dtype=np.float64)
    mask = np.abs(u) <= half_periods * period
    if mask.sum() < 3:
        mask = np.ones_like(u, dtype=bool)
    dm = d[mask]
    dmax, dmin = float(np.max(dm)), float(np.min(dm))
    return {"ratio": float(np.exp(dmax - dmin)),
            "log_half_amplitude": 0.5 * (dmax - dmin),
            "n_in_window": int(mask.sum()), "d_max": dmax, "d_min": dmin}


def fourier_metrics(u, detr, target_freq=PIX_FREQ, band=CONT_BAND,
                    exclude_frac=EXCLUDE_FRAC, n_res=N_FFT_RESAMPLE):
    """FFT of the detrended series uniformly resampled vs measured displacement u.

    Returns dict: amp_at_pixel_freq (spectral amplitude at target_freq), continuum
    median amplitude in `band` excluding +/-exclude_frac around target_freq, their
    ratio (the pre-registered Fourier peak metric), the recovered in-band peak
    frequency, and the resampling spacing du."""
    u = np.asarray(u, dtype=np.float64)
    d = np.asarray(detr, dtype=np.float64)
    order = np.argsort(u)
    us, ds = u[order], d[order]
    ug = np.linspace(us.min(), us.max(), n_res)
    dg = np.interp(ug, us, ds)
    dg = dg - np.mean(dg)
    du = float(ug[1] - ug[0])
    F = np.fft.rfft(dg)
    freqs = np.fft.rfftfreq(n_res, d=du)          # cycles / px
    amp = (2.0 / n_res) * np.abs(F)
    i_target = int(np.argmin(np.abs(freqs - target_freq)))
    amp_target = float(amp[i_target])
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    cont_mask = in_band & (np.abs(freqs - target_freq) > exclude_frac * target_freq)
    cont_med = float(np.median(amp[cont_mask])) if cont_mask.any() else float("nan")
    ratio = amp_target / cont_med if cont_med > 0 else float("inf")
    peak_freq = float(freqs[in_band][np.argmax(amp[in_band])]) if in_band.any() else float("nan")
    return {"amp_at_pixel_freq": amp_target, "continuum_median_amp": cont_med,
            "fourier_peak_ratio": float(ratio), "recovered_peak_freq": peak_freq,
            "freq_resolution": du, "target_freq_used": float(freqs[i_target])}


def phase_design(log_A, phi_x, phi_y):
    """Full OLS design [1, log A, sin/cos(2 pi phi_x), sin/cos(4 pi phi_x),
    sin/cos(2 pi phi_y), sin/cos(4 pi phi_y)] and the flux-only design [1, log A]."""
    log_A = np.asarray(log_A, dtype=np.float64)
    phi_x = np.asarray(phi_x, dtype=np.float64)
    phi_y = np.asarray(phi_y, dtype=np.float64)
    one = np.ones_like(log_A)
    X_flux = np.column_stack([one, log_A])
    X_full = np.column_stack([
        one, log_A,
        np.sin(2 * np.pi * phi_x), np.cos(2 * np.pi * phi_x),
        np.sin(4 * np.pi * phi_x), np.cos(4 * np.pi * phi_x),
        np.sin(2 * np.pi * phi_y), np.cos(2 * np.pi * phi_y),
        np.sin(4 * np.pi * phi_y), np.cos(4 * np.pi * phi_y),
    ])
    return X_flux, X_full


def ols_r2(X, y):
    """OLS fit; returns (R2, beta, residuals). R2 = 1 - SSres/SStot."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, beta, resid


# ===========================================================================
# chi^2 render-path gate (replicated VERBATIM from E1 Task A / T10 / T11)
# ===========================================================================

def chi2_gate(ops, z_points, tag="T12"):
    """Recompute chi^2 from the render path and reconcile with log_prob's reduced
    chi^2 aux. Convention IDENTICAL to E1/T8/T9/T10/T11: chi2_from_aux = aux *
    event_size. RAISE if relative match > 1e-6 (a wrong render invalidates all
    downstream). Re-run per model build (ss2 AND ss4) against that model's own aux."""
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
# Offline smoke tests (numpy only; no jax/GPU)
# ===========================================================================

def smoke_blob(verbose=True):
    """A synthetic Gaussian blob at a KNOWN sub-pixel center is recovered to <0.02 px
    by blob_stats' flux-weighted windowed centroid."""
    side = IMG_SIDE
    yy, xx = np.mgrid[0:side, 0:side]
    cx_true, cy_true = HOTSPOT_COL + 0.37, HOTSPOT_ROW - 0.24
    sigma = 1.3
    # A pure Gaussian blob (a raw flux-weighted centroid is unbiased only without a
    # background pedestal; a pedestal would pull the centroid toward the window center).
    img = 15.0 * np.exp(-((xx - cx_true) ** 2 + (yy - cy_true) ** 2) / (2 * sigma ** 2))
    bs = blob_stats(img)
    ex = abs(bs["x_c"] - cx_true)
    ey = abs(bs["y_c"] - cy_true)
    ok = ex < 0.02 and ey < 0.02
    if verbose:
        print(f"[smoke] blob centroid: x_c={bs['x_c']:.4f} (true {cx_true:.4f}, "
              f"err {ex:.4f}) y_c={bs['y_c']:.4f} (true {cy_true:.4f}, err {ey:.4f}) "
              f"A={bs['A']:.2f} peak_above={bs['peak_above_pedestal']:.1f} -> "
              f"{'PASS' if ok else 'FAIL'}")
    return {"x_c": bs["x_c"], "cx_true": cx_true, "err_x": ex,
            "y_c": bs["y_c"], "cy_true": cy_true, "err_y": ey, "pass": bool(ok)}


def smoke_oscillation(verbose=True):
    """Detrend+FFT recover a planted 1-cycle/px oscillation on top of a smooth trend.

    Plant log lambda = trend(u) + amp*sin(2 pi f0 u), f0 = 1/px, on a non-uniform u
    grid. The moving-median detrend removes the trend; the FFT peak must land at f0,
    the Fourier peak ratio must be large, and the peak-to-trough ratio must recover
    ~exp(2*amp)."""
    rng = np.random.default_rng(3)
    u = np.linspace(-DIAL_RANGE_PX, DIAL_RANGE_PX, N_DIAL)
    u = u + 0.01 * rng.standard_normal(u.size)     # mild non-uniformity
    f0 = 1.0
    amp = 0.4
    trend = 0.7 * u + 0.3 * u ** 2                  # smooth low-frequency trend
    log_lam = trend + amp * np.sin(2 * np.pi * f0 * u)
    us, detr, win, dx = detrend_log(u, log_lam)
    pt = peak_to_trough(us, detr)
    fo = fourier_metrics(us, detr)
    freq_ok = abs(fo["recovered_peak_freq"] - f0) < 0.15
    peakratio_ok = fo["fourier_peak_ratio"] > 5.0
    # detrended peak-to-trough ~ exp(2*amp) once the trend is gone
    pt_expected = float(np.exp(2 * amp))
    pt_ok = abs(pt["ratio"] - pt_expected) / pt_expected < 0.35
    ok = freq_ok and peakratio_ok and pt_ok
    if verbose:
        print(f"[smoke] oscillation: detrend win={win} samples (dx={dx:.3f}px); "
              f"recovered peak freq={fo['recovered_peak_freq']:.3f}/px (true {f0}); "
              f"fourier peak ratio={fo['fourier_peak_ratio']:.1f} (>5); "
              f"peak-to-trough={pt['ratio']:.2f} (expect ~{pt_expected:.2f}) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return {"recovered_peak_freq": fo["recovered_peak_freq"], "true_freq": f0,
            "fourier_peak_ratio": fo["fourier_peak_ratio"],
            "peak_to_trough": pt["ratio"], "pt_expected": pt_expected,
            "detrend_win_samples": win, "pass": bool(ok)}


def smoke_delta_r2(verbose=True):
    """OLS DeltaR2 recovers a planted phase contribution over a flux baseline.

    y = b0 + b1*logA + phase(phi_x, phi_y) + small noise, with phi drawn iid and
    (by construction) uncorrelated with logA. The flux-only R2 captures the logA
    part; the full model additionally captures the planted phase part. Recovered
    DeltaR2 must match the planted variance share to within 0.05."""
    rng = np.random.default_rng(5)
    n = 2048
    log_A = rng.normal(4.0, 0.6, n)
    phi_x = rng.uniform(0, 1, n)
    phi_y = rng.uniform(0, 1, n)
    flux_part = 0.5 + 1.2 * log_A
    phase_part = (0.8 * np.sin(2 * np.pi * phi_x) + 0.5 * np.cos(4 * np.pi * phi_x)
                  + 0.6 * np.sin(2 * np.pi * phi_y))
    noise = 0.15 * rng.standard_normal(n)
    y = flux_part + phase_part + noise
    X_flux, X_full = phase_design(log_A, phi_x, phi_y)
    r2_flux, _, _ = ols_r2(X_flux, y)
    r2_full, _, _ = ols_r2(X_full, y)
    dr2 = r2_full - r2_flux
    # planted variance-share of the phase part (orthogonal design => ~var ratio)
    planted = np.var(phase_part) / np.var(y)
    ok = abs(dr2 - planted) < 0.05 and dr2 > 0.1
    if verbose:
        print(f"[smoke] DeltaR2: R2_flux={r2_flux:.3f} R2_full={r2_full:.3f} "
              f"DeltaR2={dr2:.3f} (planted phase share ~{planted:.3f}) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return {"r2_flux": r2_flux, "r2_full": r2_full, "delta_r2": dr2,
            "planted_share": float(planted), "pass": bool(ok)}


def run_smoke():
    print("=== T12 offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_blob(); b = smoke_oscillation(); c = smoke_delta_r2()
    allpass = a["pass"] and b["pass"] and c["pass"]
    print(f"[smoke] overall: {'PASS' if allpass else 'FAIL'}")
    return {"blob": a, "oscillation": b, "delta_r2": c, "pass": bool(allpass)}


# ===========================================================================
# System loader with supersample override (system.py path; NO common.py edit)
# ===========================================================================

def load_sys60(data_dir, supersample=None):
    """Load the sys60 target via its system.py, passing an optional supersample
    override. supersample=None reproduces the reference (notebook) build exactly;
    an int overrides ONLY SimulatorConfig.supersample (T12 Part 2 control). This
    replicates common.load_target's system.py dispatch so common.py stays untouched."""
    import importlib.util
    import sys as _sys
    data_dir = os.path.abspath(data_dir)
    sys_py = os.path.join(data_dir, "system.py")
    if not os.path.isfile(sys_py):
        raise FileNotFoundError(f"[T12] no system.py in data-dir: {data_dir}")
    tag = "base" if supersample is None else f"ss{supersample}"
    mod_name = "_t12_system_" + os.path.basename(data_dir).replace(".", "_") + "_" + tag
    if data_dir not in _sys.path:
        _sys.path.insert(0, data_dir)
    spec = importlib.util.spec_from_file_location(mod_name, sys_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_target(supersample=supersample)


# ===========================================================================
# theta-space dial (source-center shift, mapped back to z through the bijector)
# ===========================================================================

def build_dial(model_seq, z0):
    """Return (theta0, theta_to_z) for a base point z0. theta0 is the constrained
    source-center theta dict at z0 (jax scalars); theta_to_z(dcx, dcy) perturbs the
    two source-center thetas by (dcx, dcy), maps the FULL constrained dict back to z
    via bij.inverse, and returns the z vector (jitted). Round-trip verified: at
    (0,0) theta_to_z reproduces z0."""
    import jax
    import jax.numpy as jnp
    pm = model_seq.prob_model
    theta0 = pm.bij.forward(list(jnp.asarray(z0, dtype=jnp.float64)))
    if SRC_CX_KEY not in theta0 or SRC_CY_KEY not in theta0:
        raise KeyError(f"[T12] source-center keys {SRC_CX_KEY!r}/{SRC_CY_KEY!r} not "
                       f"in bijector output keys: {sorted(theta0.keys())}")

    def theta_to_z(dcx, dcy):
        d = dict(theta0)
        d[SRC_CX_KEY] = theta0[SRC_CX_KEY] + dcx
        d[SRC_CY_KEY] = theta0[SRC_CY_KEY] + dcy
        cols = pm.bij.inverse(d)
        return jnp.stack(cols)

    jf = jax.jit(theta_to_z)
    z_rt = np.asarray(jf(0.0, 0.0), dtype=np.float64)
    rt_err = float(np.max(np.abs(z_rt - np.asarray(z0, dtype=np.float64))))
    if rt_err > 1e-8:
        raise RuntimeError(f"[T12] bijector round-trip error {rt_err:.3e} > 1e-8 at "
                           "the base point -- theta<->z mapping is inconsistent.")
    theta0_np = {k: float(np.asarray(v)) for k, v in theta0.items()}
    return theta0_np, jf, rt_err


def calibrate_dial(theta_to_z, ops, delta=DELTA_THETA):
    """Finite-difference the MEASURED centroid response to the two source-center
    thetas -> R = d(x_c, y_c)/d(cx, cy) (2x2). Choose the theta-direction d that moves
    x_c PURELY along image-x with unit gain (R @ d = (1,0)), so a dial value s (px)
    is the EXPECTED x_c displacement. Returns (R, d_dir, centroid0)."""
    def centroid(dcx, dcy):
        z = np.asarray(theta_to_z(dcx, dcy), dtype=np.float64)
        img = np.asarray(ops["render"](z), dtype=np.float64)
        bs = blob_stats(img)
        return np.array([bs["x_c"], bs["y_c"]])
    c0 = centroid(0.0, 0.0)
    dcx_p, dcx_m = centroid(+delta, 0.0), centroid(-delta, 0.0)
    dcy_p, dcy_m = centroid(0.0, +delta), centroid(0.0, -delta)
    dcolx = (dcx_p - dcx_m) / (2 * delta)      # d(x_c,y_c)/d(cx)
    dcoly = (dcy_p - dcy_m) / (2 * delta)      # d(x_c,y_c)/d(cy)
    R = np.column_stack([dcolx, dcoly])         # [[dxc/dcx, dxc/dcy],[dyc/dcx, dyc/dcy]]
    d_dir = np.linalg.solve(R, np.array([1.0, 0.0]))   # R @ d = e_x
    return R, d_dir, c0


# ===========================================================================
# One dial scan (measured x_c, flux A, full + windowed lambda1 per point)
# ===========================================================================

def run_scan(model_seq, ops, z0, std_z, W, tag):
    """Calibrate the dial at z0 then scan +/-DIAL_RANGE_PX expected-px in N_DIAL
    steps. Returns a rich dict with the calibration, the per-point table (measured
    x_c/y_c, displacement, A, lambda1 full, lambda1 windowed), and the oscillation
    metrics (detrended peak-to-trough + FFT) computed vs MEASURED displacement."""
    import jax
    theta0, theta_to_z, rt_err = build_dial(model_seq, z0)
    R, d_dir, c0 = calibrate_dial(theta_to_z, ops)
    print(f"[T12] {tag}: calibration R=\n{R}\n  d_dir(px->theta)={d_dir}  "
          f"x_c0={c0[0]:.4f} y_c0={c0[1]:.4f}  round-trip err={rt_err:.2e}")

    dial = np.linspace(-DIAL_RANGE_PX, DIAL_RANGE_PX, N_DIAL)
    dcx = dial * d_dir[0]
    dcy = dial * d_dir[1]
    # z for every dial point (jitted scalar map; cheap)
    Z = np.stack([np.asarray(theta_to_z(float(dcx[i]), float(dcy[i])),
                             dtype=np.float64) for i in range(N_DIAL)])

    win_idx = window_flat_indices()
    batched_jac = jax.jit(jax.vmap(ops["jac_render"]))

    x_c = np.empty(N_DIAL); y_c = np.empty(N_DIAL); A = np.empty(N_DIAL)
    peak_above = np.empty(N_DIAL)
    lam1 = np.empty(N_DIAL); lam1_win = np.empty(N_DIAL)
    t0 = time.perf_counter()
    for lo in range(0, N_DIAL, CHUNK):
        hi = min(N_DIAL, lo + CHUNK)
        Js = np.asarray(batched_jac(Z[lo:hi]), dtype=np.float64)   # (nb, 6400, 22)
        Ims = np.asarray(ops["render_batch"](Z[lo:hi]), dtype=np.float64)  # (nb, 6400)
        for k in range(Js.shape[0]):
            j = lo + k
            bs = blob_stats(Ims[k])
            x_c[j] = bs["x_c"]; y_c[j] = bs["y_c"]; A[j] = bs["A"]
            peak_above[j] = bs["peak_above_pedestal"]
            Jk = Js[k]
            lam1[j] = eig_desc(gn_metric(scale_cols(Jk, std_z), W))[0][0]
            lam1_win[j] = eig_desc(
                gn_metric(scale_cols(Jk[win_idx], std_z), W[win_idx]))[0][0]
        print(f"[T12] {tag} scan chunk {lo:3d}-{hi:3d}/{N_DIAL} "
              f"({time.perf_counter()-t0:5.1f}s)")

    # center on the dial=0 (base) measured centroid
    i0 = int(np.argmin(np.abs(dial)))
    x_c0 = float(x_c[i0])
    disp = x_c - x_c0                    # MEASURED image-x displacement (px)

    log_lam = np.log(lam1)
    us, detr, win_samples, dx = detrend_log(disp, log_lam)
    pt = peak_to_trough(us, detr)
    fo = fourier_metrics(us, detr)
    # windowed lambda1 oscillation (same machinery, reported alongside)
    us_w, detr_w, _, _ = detrend_log(disp, np.log(lam1_win))
    pt_w = peak_to_trough(us_w, detr_w)
    fo_w = fourier_metrics(us_w, detr_w)

    print(f"[T12] {tag}: measured x_c range [{x_c.min():.3f},{x_c.max():.3f}] px "
          f"(disp [{disp.min():.3f},{disp.max():.3f}]); lambda1 range "
          f"[{lam1.min():.3e},{lam1.max():.3e}]; peak-to-trough={pt['ratio']:.2f} "
          f"fourier_peak_ratio={fo['fourier_peak_ratio']:.2f}")

    return {
        "tag": tag,
        "calibration": {"R": R.tolist(), "d_dir_px_to_theta": d_dir.tolist(),
                        "x_c0": x_c0, "y_c0": float(y_c[i0]),
                        "delta_theta": DELTA_THETA, "roundtrip_err": rt_err,
                        "note": "R = d(x_c,y_c)/d(source cx,cy); d_dir = R^-1 e_x so a "
                                "unit dial = 1 px EXPECTED x_c displacement; analysis "
                                "uses MEASURED x_c, never the dial."},
        "dial": dial.tolist(),
        "x_c": x_c.tolist(), "y_c": y_c.tolist(), "displacement": disp.tolist(),
        "A": A.tolist(), "peak_above_pedestal": peak_above.tolist(),
        "lambda1": lam1.tolist(), "lambda1_window": lam1_win.tolist(),
        "detrend": {"window_samples": win_samples, "dx_px": dx},
        "oscillation_full": {"peak_to_trough": pt, "fourier": fo},
        "oscillation_window": {"peak_to_trough": pt_w, "fourier": fo_w},
        # arrays kept for plotting (not serialized directly)
        "_disp": disp, "_lam1": lam1, "_lam1_win": lam1_win, "_A": A,
        "_us": us, "_detr": detr, "_fo": fo, "_x_c": x_c,
    }


# ===========================================================================
# Plots
# ===========================================================================

def _pixel_gridlines(ax, x_c0, x_c_arr):
    """Vertical lines at image-x pixel boundaries (integer+0.5) mapped to the
    displacement axis (disp = x_c - x_c0)."""
    lo, hi = np.floor(x_c_arr.min()), np.ceil(x_c_arr.max())
    for n in np.arange(lo - 1, hi + 1):
        b = n + 0.5 - x_c0
        ax.axvline(b, color="0.85", lw=0.6, zorder=0)


def plot_scans(scans, ss4_scan, out_path):
    """Fig 1: lambda1 (and blob-window lambda1) vs measured x_c displacement for the
    3 scans + the ss4 overlay on the top-spike panel; pixel-boundary gridlines."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    order = ["top spike", "mid spike", "baseline"]
    axmap = {order[0]: axes[0, 0], order[1]: axes[0, 1], order[2]: axes[1, 0]}
    for name, sc in scans.items():
        ax = axmap[name]
        disp = sc["_disp"]; x_c0 = sc["calibration"]["x_c0"]
        _pixel_gridlines(ax, x_c0, sc["_x_c"])
        ax.plot(disp, sc["_lam1"], "-o", ms=2.5, color="#8a3b3b", lw=1.0,
                label="lambda1 (full, ss2)")
        ax.plot(disp, sc["_lam1_win"], "-s", ms=2.0, color="#3b6a8a", lw=0.8,
                alpha=0.8, label="lambda1 (blob window, ss2)")
        if name == "top spike" and ss4_scan is not None:
            ax.plot(ss4_scan["_disp"], ss4_scan["_lam1"], "-^", ms=2.5,
                    color="#2a8a2a", lw=1.0, alpha=0.85, label="lambda1 (full, ss4)")
        ax.set_yscale("log")
        ax.set_xlabel("measured x_c displacement (px)")
        ax.set_ylabel("lambda1 (top GN eigenvalue, standardized)")
        ax.set_title(f"{name}: pt={sc['oscillation_full']['peak_to_trough']['ratio']:.1f} "
                     f"Fpk={sc['oscillation_full']['fourier']['fourier_peak_ratio']:.1f}",
                     fontsize=9)
        ax.legend(fontsize=6, loc="best")
        ax.tick_params(labelsize=7)
    axes[1, 1].axis("off")
    fig.suptitle("T12 Part 1/2: lambda1 vs measured counter-image x_c displacement "
                 "(pixel gridlines) -- PROPOSED (UNCERTIFIED)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_detrend_fft(scans, ss4_scan, out_path):
    """Fig 2: detrended series (col 0) + FFT spectrum (col 1) per scan (rows); ss4
    overlaid on the top-spike row."""
    names = ["top spike", "mid spike", "baseline"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for r, name in enumerate(names):
        sc = scans[name]
        axd, axf = axes[r, 0], axes[r, 1]
        axd.plot(sc["_us"], sc["_detr"], "-o", ms=2.5, color="#8a3b3b",
                 label="ss2")
        axd.axvspan(-HALF_PERIODS, HALF_PERIODS, color="0.9", zorder=0,
                    label="+/-1.5 period window")
        # FFT spectrum (recompute on the resampled series for display)
        us = np.asarray(sc["_us"]); d = np.asarray(sc["_detr"])
        ug = np.linspace(us.min(), us.max(), N_FFT_RESAMPLE)
        dg = np.interp(ug, us, d); dg -= dg.mean()
        du = ug[1] - ug[0]
        freqs = np.fft.rfftfreq(N_FFT_RESAMPLE, d=du)
        amp = (2.0 / N_FFT_RESAMPLE) * np.abs(np.fft.rfft(dg))
        axf.plot(freqs, amp, "-", color="#8a3b3b", label="ss2")
        if name == "top spike" and ss4_scan is not None:
            axd.plot(ss4_scan["_us"], ss4_scan["_detr"], "-^", ms=2.5,
                     color="#2a8a2a", alpha=0.8, label="ss4")
            us4 = np.asarray(ss4_scan["_us"]); d4 = np.asarray(ss4_scan["_detr"])
            ug4 = np.linspace(us4.min(), us4.max(), N_FFT_RESAMPLE)
            dg4 = np.interp(ug4, us4, d4); dg4 -= dg4.mean()
            amp4 = (2.0 / N_FFT_RESAMPLE) * np.abs(np.fft.rfft(dg4))
            axf.plot(np.fft.rfftfreq(N_FFT_RESAMPLE, d=ug4[1]-ug4[0]), amp4, "-",
                     color="#2a8a2a", alpha=0.8, label="ss4")
        axf.axvline(PIX_FREQ, color="purple", ls="--", lw=1.0, label="1/px")
        axf.set_xlim(0, CONT_BAND[1])
        axd.set_xlabel("measured x_c displacement (px)")
        axd.set_ylabel("detrended log lambda1")
        axd.set_title(f"{name}: detrended", fontsize=9)
        axf.set_xlabel("frequency (cycles / px)")
        axf.set_ylabel("FFT amplitude")
        axf.set_title(f"{name}: spectrum", fontsize=9)
        axd.legend(fontsize=6); axf.legend(fontsize=6)
        axd.tick_params(labelsize=7); axf.tick_params(labelsize=7)
    fig.suptitle("T12 Part 1/2: detrended log lambda1 + FFT (pixel frequency marked) "
                 "-- PROPOSED (UNCERTIFIED)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_census_phase(phi_x, phi_y, detr_logl, spike_mask, out_path):
    """Fig 3: (a) phi_x vs detrended log lambda1 (spikes marked); (b) 2-D phi_x-phi_y
    hexbin colored by median detrended log lambda1 (spikes overplotted)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    ax.scatter(phi_x[~spike_mask], detr_logl[~spike_mask], s=5, c="0.6",
               alpha=0.4, label="census steps")
    ax.scatter(phi_x[spike_mask], detr_logl[spike_mask], s=28, c="#8a3b3b",
               edgecolor="k", lw=0.4, zorder=3, label="T10 spikes")
    ax.set_xlabel("phi_x = frac(x_c)")
    ax.set_ylabel("detrended log lambda1 (flux-residual)")
    ax.set_title("phase vs stiffness residual", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    hb = ax.hexbin(phi_x, phi_y, C=detr_logl, gridsize=20,
                   reduce_C_function=np.median, cmap="viridis")
    ax.scatter(phi_x[spike_mask], phi_y[spike_mask], s=30, facecolor="none",
               edgecolor="red", lw=1.0, zorder=3, label="T10 spikes")
    ax.set_xlabel("phi_x = frac(x_c)"); ax.set_ylabel("phi_y = frac(y_c)")
    ax.set_title("median detrended log lambda1 over (phi_x, phi_y)", fontsize=10)
    ax.legend(fontsize=8)
    fig.colorbar(hb, ax=ax, label="median detrended log lambda1")
    fig.suptitle("T12 Part 3: census sub-pixel phase vs stiffness -- PROPOSED "
                 "(UNCERTIFIED)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sanity(scans, out_path):
    """Fig 4: A and x_c vs dial (sanity: both should be smooth) for the 3 scans."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"top spike": "#8a3b3b", "mid spike": "#3b6a8a", "baseline": "#2a8a2a"}
    for name, sc in scans.items():
        dial = np.asarray(sc["dial"])
        axes[0].plot(dial, sc["_A"], "-o", ms=2.5, color=colors[name], label=name)
        axes[1].plot(dial, sc["_x_c"], "-o", ms=2.5, color=colors[name], label=name)
    axes[0].set_xlabel("dial (expected x_c displacement, px)")
    axes[0].set_ylabel("blob flux A (window sum)")
    axes[0].set_title("A vs dial (sanity: smooth)", fontsize=10)
    axes[1].set_xlabel("dial (expected x_c displacement, px)")
    axes[1].set_ylabel("measured x_c (px)")
    axes[1].set_title("x_c vs dial (sanity: smooth, ~linear)", fontsize=10)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle("T12 Part 1: dial sanity (flux + centroid smooth) -- PROPOSED "
                 "(UNCERTIFIED)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T12 flank-crossing / pixelation check")
    p.add_argument("--data-dir")
    p.add_argument("--run-dir",
                   default="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc",
                   help="MCLMC run dir with arrays.npz (z lookup + std_z)")
    p.add_argument("--t10-dir", help="T10 output dir (spike_list.json + t10_results.json)")
    p.add_argument("--out-dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-ss4", action="store_true",
                   help="skip Part 2 (supersample=4 control) -- debugging only")
    p.add_argument("--smoke", action="store_true",
                   help="run the offline numpy-only smoke tests and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        res = run_smoke()
        if not res["pass"]:
            raise SystemExit("[T12] smoke tests FAILED")
        return

    for req in ("data_dir", "t10_dir", "out_dir"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)   # reserved (deterministic; no random draws)

    from common import assert_x64

    # --- inputs -------------------------------------------------------------
    spike_list = json.load(open(os.path.join(args.t10_dir, "spike_list.json")))
    t10 = json.load(open(os.path.join(args.t10_dir, "t10_results.json")))
    spikes = spike_list["spikes"]
    baselines = spike_list["baselines"]
    top = next(s for s in spikes if s["rank"] == 0)
    mid = next(s for s in spikes if s["rank"] == 5)          # 6th of 12
    base = next(b for b in baselines if b["matched_spike_rank"] == 0)
    print(f"[T12] scan points: TOP seg{top['segment']} ch{top['chain']} "
          f"step{top['step']} lam1={top['lambda1']:.3e}; MID seg{mid['segment']} "
          f"ch{mid['chain']} step{mid['step']} lam1={mid['lambda1']:.3e}; "
          f"BASELINE seg{base['segment']} ch{base['chain']} step{base['step']} "
          f"lam1={base['lambda1']:.3e}")

    arrays = np.load(os.path.join(args.run_dir, "arrays.npz"))
    samples_z = np.asarray(arrays["samples_z"], dtype=np.float64)   # (C, R, dim)
    C, Rr, dim = samples_z.shape
    flat = samples_z.reshape(-1, dim)
    std_z = np.std(flat, axis=0, ddof=1)          # property of the RUN (ss-independent)
    std_z_t10 = np.asarray(t10["standardization"]["std_z"], dtype=np.float64)
    std_z_match = float(np.max(np.abs(std_z - std_z_t10)))
    print(f"[T12] samples_z {samples_z.shape}; std_z vs T10 max|diff|={std_z_match:.3e} "
          f"(must be ~0 -- same run/std)")

    def z_of(pt):
        return samples_z[int(pt["chain"]), int(pt["step"])].copy()
    z_top, z_mid, z_base = z_of(top), z_of(mid), z_of(base)

    assert_x64()

    # =====================================================================
    # Part 1 -- dial scans at ss2 (reference build)
    # =====================================================================
    model_seq, qz, z_center, dim2, param_names = load_sys60(args.data_dir, supersample=None)
    if dim2 != dim:
        raise ValueError(f"data-dir dim {dim2} != run dim {dim}")
    ops = build_jax_ops(model_seq, param_names)
    import jax
    ops["render_batch"] = jax.jit(jax.vmap(ops["render"]))
    W = ops["W"]
    print(f"[T12] ss2 model built; event_size={ops['event_size']}")

    # chi^2 gate (ss2) at the three base points
    gate_ss2, worst_ss2 = chi2_gate(ops, [z_top, z_mid, z_base], tag="T12/ss2")

    scans = {}
    scans["top spike"] = run_scan(model_seq, ops, z_top, std_z, W, "top spike (ss2)")
    scans["mid spike"] = run_scan(model_seq, ops, z_mid, std_z, W, "mid spike (ss2)")
    scans["baseline"] = run_scan(model_seq, ops, z_base, std_z, W, "baseline (ss2)")

    # =====================================================================
    # Part 2 -- supersample=4 control (TOP-spike scan, own chi^2 gate)
    # =====================================================================
    ss4_scan = None
    ss4_block = None
    if not args.skip_ss4:
        print("[T12] === Part 2: rebuilding model at supersample=4 ===")
        model_seq4, _, _, dim4, names4 = load_sys60(args.data_dir, supersample=4)
        ops4 = build_jax_ops(model_seq4, names4)
        ops4["render_batch"] = jax.jit(jax.vmap(ops4["render"]))
        gate_ss4, worst_ss4 = chi2_gate(ops4, [z_top], tag="T12/ss4")
        ss4_scan = run_scan(model_seq4, ops4, z_top, std_z, ops4["W"], "top spike (ss4)")

        f2 = scans["top spike"]["oscillation_full"]["fourier"]["amp_at_pixel_freq"]
        f4 = ss4_scan["oscillation_full"]["fourier"]["amp_at_pixel_freq"]
        p2 = scans["top spike"]["oscillation_full"]["peak_to_trough"]["log_half_amplitude"]
        p4 = ss4_scan["oscillation_full"]["peak_to_trough"]["log_half_amplitude"]
        fourier_ratio = f4 / f2 if f2 > 0 else float("inf")
        pt_ratio = p4 / p2 if p2 > 0 else float("inf")
        both = [r for r in (fourier_ratio, pt_ratio) if np.isfinite(r)]
        branch = "mixed"
        if both and min(both) >= SS4_PHYSICAL:
            branch = "physical"
        elif both and max(both) <= SS4_ARTIFACT:
            branch = "artifact"
        ss4_block = {
            "gate": {"checks": gate_ss4, "worst_relative_error": worst_ss4},
            "amp_pixel_freq_ss2": f2, "amp_pixel_freq_ss4": f4,
            "fourier_amplitude_ratio_ss4_over_ss2": fourier_ratio,
            "log_half_amplitude_ss2": p2, "log_half_amplitude_ss4": p4,
            "peak_to_trough_amplitude_ratio_ss4_over_ss2": pt_ratio,
            "branch_thresholds": {"physical>=": SS4_PHYSICAL, "artifact<=": SS4_ARTIFACT},
            "branch (proposed, UNCERTIFIED)": branch,
            "note": "ss4 likelihood is a slightly different posterior; this compares "
                    "oscillation AMPLITUDE at matched dial points, NOT absolute lambda1.",
        }
        print(f"[T12] ss4/ss2 fourier-amp ratio={fourier_ratio:.3f}  "
              f"peak-to-trough-amp ratio={pt_ratio:.3f}  branch={branch}")

    # =====================================================================
    # Part 3 -- census correlational cross-check (renders only; reuse census lambda1)
    # =====================================================================
    print("[T12] === Part 3: census phase cross-check (2048 renders) ===")
    seg_defs = [(int(s["segment"]), int(s["chain"]), int(s["start"]))
                for s in t10["segments"]]
    seg_len = int(t10["args"]["segment_len"])
    census_z = []
    census_lam1 = []
    seg_index_of = []     # (segment, seg_step) per pooled point (for spike matching)
    for (segid, chain, start) in seg_defs:
        seg = next(s for s in t10["segments"] if s["segment"] == segid)
        lam = np.asarray(seg["lambda1"], dtype=np.float64)
        if lam.size != seg_len:
            raise ValueError(f"[T12] segment {segid} lambda1 len {lam.size} != "
                             f"segment_len {seg_len}")
        zseg = samples_z[chain, start:start + seg_len]
        census_z.append(zseg)
        census_lam1.append(lam)
        for k in range(seg_len):
            seg_index_of.append((segid, k))
    census_z = np.concatenate(census_z, axis=0)         # (2048, dim)
    census_lam1 = np.concatenate(census_lam1, axis=0)    # (2048,)
    n_census = census_z.shape[0]
    print(f"[T12] reconstructed {n_census} census z-points ({len(seg_defs)} segments "
          f"x {seg_len}); reusing t10 lambda1 (NOT recomputed)")

    x_c = np.empty(n_census); y_c = np.empty(n_census); A = np.empty(n_census)
    t0 = time.perf_counter()
    for lo in range(0, n_census, 128):
        hi = min(n_census, lo + 128)
        Ims = np.asarray(ops["render_batch"](census_z[lo:hi]), dtype=np.float64)
        for k in range(Ims.shape[0]):
            bs = blob_stats(Ims[k])
            x_c[lo + k] = bs["x_c"]; y_c[lo + k] = bs["y_c"]; A[lo + k] = bs["A"]
        if lo % 512 == 0:
            print(f"[T12] census render {hi}/{n_census} ({time.perf_counter()-t0:5.1f}s)")
    phi_x = x_c - np.floor(x_c)
    phi_y = y_c - np.floor(y_c)

    log_lam = np.log(census_lam1)
    log_A = np.log(A)
    X_flux, X_full = phase_design(log_A, phi_x, phi_y)
    r2_flux, beta_flux, resid_flux = ols_r2(X_flux, log_lam)
    r2_full, beta_full, _ = ols_r2(X_full, log_lam)
    delta_r2 = r2_full - r2_flux
    magnif_signature = (r2_flux >= FLUX_R2_NOTE) and (delta_r2 <= FALSIFY_DR2)
    print(f"[T12] census: R2_flux={r2_flux:.3f} R2_full={r2_full:.3f} "
          f"DeltaR2={delta_r2:.3f} (predict >= {PREDICT_DR2}; falsify <= {FALSIFY_DR2}); "
          f"magnification-branch signature={magnif_signature}")

    # spike mask over the pooled census (T10 top-12 spikes)
    spike_keys = {(int(s["segment"]), int(s["seg_step"])) for s in spikes}
    spike_mask = np.array([(seg_index_of[i] in spike_keys) for i in range(n_census)])
    detr_logl = resid_flux         # detrended log lambda1 = flux-only residual

    # =====================================================================
    # Plots
    # =====================================================================
    fig1 = os.path.join(args.out_dir, "t12_scans_lambda1.png")
    fig2 = os.path.join(args.out_dir, "t12_detrend_fft.png")
    fig3 = os.path.join(args.out_dir, "t12_census_phase.png")
    fig4 = os.path.join(args.out_dir, "t12_dial_sanity.png")
    plot_scans(scans, ss4_scan, fig1)
    plot_detrend_fft(scans, ss4_scan, fig2)
    plot_census_phase(phi_x, phi_y, detr_logl, spike_mask, fig3)
    plot_sanity(scans, fig4)

    # =====================================================================
    # JSON (everything; verdict fields restate thresholds next to measured)
    # =====================================================================
    def _scan_json(sc):
        return {k: v for k, v in sc.items() if not k.startswith("_")}

    scan_verdicts = {}
    for name, sc in scans.items():
        pt = sc["oscillation_full"]["peak_to_trough"]["ratio"]
        fpk = sc["oscillation_full"]["fourier"]["fourier_peak_ratio"]
        scan_verdicts[name] = {
            "measured_peak_to_trough": pt,
            "predict_peak_to_trough>=": PREDICT_RATIO,
            "falsify_peak_to_trough<=": FALSIFY_RATIO,
            "measured_fourier_peak_ratio": fpk,
            "predict_fourier_peak>=": PREDICT_FOURIER,
            "verdict": "proposed (UNCERTIFIED)",
        }

    doc = {
        "experiment": "T12 -- flank-crossing / pixelation check (micro-mechanism of "
                      "the lambda1 spikes)",
        "status": "proposed (UNCERTIFIED) -- grader inspects artifacts, not this summary",
        "timestamp_utc": _now(),
        "args": {k: v for k, v in vars(args).items()},
        "dim": int(dim), "param_names": param_names,
        "hotspot": {"row": HOTSPOT_ROW, "col": HOTSPOT_COL, "blob_half": BLOB_HALF,
                    "window": "15x15", "guard_factor": GUARD_FACTOR},
        "standardization": {
            "std_z": std_z.tolist(),
            "std_z_vs_t10_max_abs_diff": std_z_match,
            "note": "M_zhat = gn_metric(scale_cols(J_z, std_z), W); lambda1 = top "
                    "eigenvalue. std_z from pooled run results samples (ddof=1) -- a "
                    "property of the RUN, identical for ss2 and ss4.",
        },
        "dial_definition": {
            "space": "theta (constrained); perturb source-center "
                     f"{SRC_CX_KEY} & {SRC_CY_KEY}, map back to z via bij.inverse",
            "delta_theta_fd": DELTA_THETA,
            "range_px": DIAL_RANGE_PX, "n_steps": N_DIAL,
            "note": "dial variable = EXPECTED image-x displacement (px) via R^-1 e_x; "
                    "analysis uses MEASURED x_c, never the dial.",
        },
        "part1_scans": {name: _scan_json(sc) for name, sc in scans.items()},
        "part1_scan_verdicts (proposed, UNCERTIFIED)": scan_verdicts,
        "part1_prediction": {
            "text": f"detrended peak-to-trough >= {PREDICT_RATIO:g} AND pixel-frequency "
                    f"Fourier peak >= {PREDICT_FOURIER:g}x continuum",
            "falsifier": f"peak-to-trough <= {FALSIFY_RATIO:g} => sub-pixel phase does "
                         "NOT explain the spikes (smooth magnification/deflection "
                         "alternative leads)",
        },
        "part2_supersample_control": (
            {"skipped": True} if ss4_scan is None else {
                "ss4_scan": _scan_json(ss4_scan),
                "comparison": ss4_block,
            }),
        "part2_gate_ss2": {"checks": gate_ss2, "worst_relative_error": worst_ss2},
        "part3_census": {
            "n_points": int(n_census),
            "segment_defs": [{"segment": s, "chain": c, "start": st}
                             for (s, c, st) in seg_defs],
            "segment_len": seg_len,
            "reused_census_lambda1": "t10_results.json segments[*].lambda1 (NOT recomputed)",
            "regression": {
                "R2_flux": r2_flux, "R2_full": r2_full, "delta_R2": delta_r2,
                "flux_design": "[1, log A]",
                "full_design": "[1, log A, sin/cos(2pi phi_x), sin/cos(4pi phi_x), "
                               "sin/cos(2pi phi_y), sin/cos(4pi phi_y)]",
                "beta_flux": beta_flux.tolist(),
                "beta_full": beta_full.tolist(),
            },
            "prediction": {
                "text": f"DeltaR2(phase | flux) >= {PREDICT_DR2:g}",
                "falsifier": f"DeltaR2 <= {FALSIFY_DR2:g} => phase does not explain the "
                             "census stiffness variance",
                "measured_delta_R2": delta_r2,
            },
            "magnification_branch_signature": {
                "condition": f"flux-alone R2 >= {FLUX_R2_NOTE:g} AND DeltaR2 <= "
                             f"{FALSIFY_DR2:g}",
                "fired": bool(magnif_signature),
                "note": "if fired, log A (blob magnification/brightness) carries the "
                        "census variance -- the smooth magnification-branch signature.",
            },
            "n_spikes_flagged": int(spike_mask.sum()),
        },
        "verdict_fields": {
            "T12_part1_flank_oscillation": "proposed (UNCERTIFIED): compare each scan's "
                "peak_to_trough to >=10 (falsify <=2) and fourier_peak_ratio to >=5; read "
                "fig1 (lambda1 vs measured x_c, pixel gridlines) + fig2 (detrend+FFT).",
            "T12_part2_pixelation": "proposed (UNCERTIFIED): ss4/ss2 amplitude ratio >=0.7 "
                "physical / <=0.5 artifact / between mixed (fourier-amp AND peak-to-trough).",
            "T12_part3_census": "proposed (UNCERTIFIED): compare DeltaR2 to >=0.4 "
                "(falsify <=0.1); if flux-alone R2>=0.6 with small DeltaR2, magnification "
                "branch. Read fig3 (phase scatter + hexbin).",
            "note": "this script does NOT adjudicate; a grader inspects plots + numbers.",
        },
        "plots": {"scans_lambda1": os.path.basename(fig1),
                  "detrend_fft": os.path.basename(fig2),
                  "census_phase": os.path.basename(fig3),
                  "dial_sanity": os.path.basename(fig4)},
    }
    json_path = os.path.join(args.out_dir, "t12_results.json")
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[T12] wrote {json_path}, {fig1}, {fig2}, {fig3}, {fig4}")

    # --- compact printed summary (PROPOSED / UNCERTIFIED) -------------------
    print("\n=== T12 summary (PROPOSED / UNCERTIFIED) ===")
    print(f"  chi^2 gate ss2: worst rel {worst_ss2:.3e}")
    for name, sc in scans.items():
        o = sc["oscillation_full"]
        print(f"  [P1 {name:10s}] peak-to-trough={o['peak_to_trough']['ratio']:8.2f} "
              f"(predict >= {PREDICT_RATIO:g}, falsify <= {FALSIFY_RATIO:g}); "
              f"fourier_peak={o['fourier']['fourier_peak_ratio']:6.2f} "
              f"(predict >= {PREDICT_FOURIER:g})")
    if ss4_block is not None:
        print(f"  [P2 ss4/ss2] fourier-amp ratio="
              f"{ss4_block['fourier_amplitude_ratio_ss4_over_ss2']:.3f}, "
              f"peak-to-trough-amp ratio="
              f"{ss4_block['peak_to_trough_amplitude_ratio_ss4_over_ss2']:.3f} => "
              f"{ss4_block['branch (proposed, UNCERTIFIED)']} "
              f"(physical >= {SS4_PHYSICAL}, artifact <= {SS4_ARTIFACT})")
    print(f"  [P3 census] R2_flux={r2_flux:.3f} R2_full={r2_full:.3f} "
          f"DeltaR2={delta_r2:.3f} (predict >= {PREDICT_DR2}, falsify <= {FALSIFY_DR2}; "
          f"magnif-branch={magnif_signature})")
    print("[T12] done.")


if __name__ == "__main__":
    main()
