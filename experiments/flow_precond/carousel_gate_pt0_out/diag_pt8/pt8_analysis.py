"""PT-8 asymptote characterization: do warm-beta_min ladders asymptote to the
SAME cold-pocket weight as beta_min=0.40, or to beta_min-DEPENDENT lower
values?

Reuses the VALIDATED PART-A/PART-B primitives from diag_pt7/pt7_analysis.py
(transport_ground_truth, zca_whiten, kmeans2, bc_pipeline,
null_calibrate_BC_star, detector_for_hot_rung, thin_window_mask) by import,
not reimplementation -- diag_pt7's PART-A reproduces pt6_report_table.json
exactly and its detector is the pinned procedure. Do not reinvent them here.

Read-only analysis of arrays_D2_pt8_{L60_s60,L60_s61,L70_s60,L70_s61}.npz
(a GPU run may still be producing/checkpointing these -- glob for whichever
exist, process what's there, never crash on missing/short data) plus
arrays_D2_pt7_L40.npz reused as the beta_min=0.40 anchor.

No fabricated results: an arm whose rounds_done does not yet cover the
required window is reported insufficient_data=True with a verdict of
INSUFFICIENT_DATA, never a SHORTFALL/REACHES-BAND/UNRESOLVED science verdict.
"""
import os
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OUTDIR = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
DIAG = os.path.join(OUTDIR, "diag_pt8")
PT7_DIR = os.path.join(OUTDIR, "diag_pt7")
sys.path.insert(0, PT7_DIR)
import pt7_analysis as pt7  # noqa: E402  (validated primitives, reused not reinvented)

THIN_B = 5
WIN_LO = 500  # fixed per spec: exponential fit + flat window always start at round 500
N_BOOT = 300
BOOT_SEED = 0
MIN_FIT_POINTS = 100      # need >=100 rounds in [WIN_LO, rounds_done) to attempt the exp fit
MIN_FLAT_POINTS = 50      # need >=50 rounds in the flat window to trust A_flat/IAT
C25_LO, C25_HI = 0.32, 0.49
SHORTFALL_THRESH = 0.32
C25_POINT = 0.42  # the C-25 reference point noted in the task (faint sub-0.42 pull check)
ROLL_W = 25
FRONTIER_HOLD = 100  # rounds the rolling mean must stay >=0.32 to count as "in band"

# PT-8 arm -> (beta_min, seed); PT-7 L40 reused as the beta_min=0.40 anchor (single seed).
ARM_INFO = {
    "L60_s60": dict(beta_min=0.60, seed=60, path=os.path.join(OUTDIR, "arrays_D2_pt8_L60_s60.npz")),
    "L60_s61": dict(beta_min=0.60, seed=61, path=os.path.join(OUTDIR, "arrays_D2_pt8_L60_s61.npz")),
    "L70_s60": dict(beta_min=0.70, seed=60, path=os.path.join(OUTDIR, "arrays_D2_pt8_L70_s60.npz")),
    "L70_s61": dict(beta_min=0.70, seed=61, path=os.path.join(OUTDIR, "arrays_D2_pt8_L70_s61.npz")),
}
ANCHOR_TAG = "L40_anchor"
ANCHOR_INFO = dict(beta_min=0.40, seed=60, path=os.path.join(OUTDIR, "arrays_D2_pt7_L40.npz"))

BETA_MIN_TO_TAGS = {0.60: ["L60_s60", "L60_s61"], 0.70: ["L70_s60", "L70_s61"]}

COLORS = {"L60_s60": "#d95f02", "L60_s61": "#fdae61", "L70_s60": "#e7298a",
          "L70_s61": "#f4a6c6", ANCHOR_TAG: "#1b9e77"}


# ============================================================
# PART 1 -- exponential-approach fit + chain-bootstrap CI
# ============================================================

def exp_model(t, tau, A):
    """w(t) = A*(1 - exp(-(t-500)/tau)). Parameter order (tau, A) matches the
    prototyped p0/bounds ordering: p0=[500, 0.3], bounds=([50,0.1],[8000,0.65])."""
    return A * (1.0 - np.exp(-(t - WIN_LO) / tau))


def fit_exp_point(t, w, p0=(500.0, 0.3), bounds=((50.0, 0.1), (8000.0, 0.65))):
    """Single deterministic fit on the (already chain-averaged) series w(t).
    Returns (tau_hat, A_hat). Raises on fit failure (caller handles)."""
    popt, _ = curve_fit(exp_model, t, w, p0=list(p0), bounds=bounds, maxfev=20000)
    tau_hat, A_hat = float(popt[0]), float(popt[1])
    return tau_hat, A_hat


def bootstrap_exp_over_chains(cold_ind, win_lo, win_hi, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample the 16 chains (nsys axis) WITH REPLACEMENT, recompute the
    per-round chain-mean series, refit the exponential, 300 reps. Returns
    (A_boot, tau_boot) arrays of successful-fit values only (failed fits, e.g.
    degenerate resamples that don't converge, are silently dropped -- the
    caller checks how many succeeded)."""
    rng = np.random.default_rng(seed)
    nsys = cold_ind.shape[1]
    t = np.arange(win_lo, win_hi)
    A_list, tau_list = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, nsys, size=nsys)
        w = cold_ind[win_lo:win_hi][:, idx].mean(axis=1).astype(np.float64)
        try:
            tau_hat, A_hat = fit_exp_point(t, w)
            tau_list.append(tau_hat)
            A_list.append(A_hat)
        except Exception:
            continue
    return np.asarray(A_list), np.asarray(tau_list)


# ============================================================
# PART 2 -- integrated autocorrelation time (IAT) + autocorr-corrected SE
# ============================================================

def iat_series(x):
    """IAT = 1 + 2*sum(acf[1:k]) up to (not including) the first negative lag,
    the standard cheap "initial positive sequence" heuristic. Floored at 1.0
    (no series can have fewer than 1 effective sample per sample)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 1.0
    xc = x - x.mean()
    var = np.dot(xc, xc) / n
    if var <= 0:
        return 1.0
    acf_full = np.correlate(xc, xc, mode="full")[n - 1:] / (var * n)
    s = 0.0
    for k in range(1, n):
        if acf_full[k] < 0:
            break
        s += acf_full[k]
    iat = 1.0 + 2.0 * s
    return float(max(iat, 1.0))


def autocorr_se(window):
    """std(window)/sqrt(N_eff), N_eff = len(window)/IAT."""
    window = np.asarray(window, dtype=np.float64)
    n = len(window)
    if n < 2:
        return np.nan, 1.0, float(n)
    iat = iat_series(window)
    n_eff = n / iat
    se = float(window.std(ddof=1) / np.sqrt(max(n_eff, 1e-9)))
    return se, iat, n_eff


# ============================================================
# PART 3 -- per-arm asymptote characterization (fit + flat + guard + verdict)
# ============================================================

def cold_occ_series(cold_ind):
    return cold_ind.mean(axis=1).astype(np.float64)


def rolling_mean(x, w=ROLL_W):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < w:
        return np.array([])
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


def compute_asymptote_for_arm(d, tag, beta_min, seed, win_lo=WIN_LO,
                               n_boot=N_BOOT, boot_seed=BOOT_SEED):
    cold_ind = np.asarray(d["cold_ind"])
    rounds_done = int(np.asarray(d["rounds_done"]).reshape(-1)[0]) if "rounds_done" in d.files \
        else cold_ind.shape[0]
    n_avail = min(rounds_done, cold_ind.shape[0])

    result = dict(tag=tag, beta_min=beta_min, seed=seed, rounds_done=rounds_done)

    # RT_pocket (cumulative, window-independent) is always safe to extract via
    # pt7's transport_ground_truth "always safe" fields, regardless of whether
    # the windowed fields below are well-defined yet.
    gt = pt7.transport_ground_truth(d, win_lo=500, win_hi=max(501, n_avail), thin_b=THIN_B)
    result["RT_pocket"] = gt["RT_pocket"]

    if n_avail - win_lo < MIN_FIT_POINTS:
        result.update(insufficient_data=True,
                      reason=f"rounds_done={rounds_done} (avail={n_avail}) gives only "
                             f"{max(n_avail - win_lo, 0)} rounds in [{win_lo},.), need >={MIN_FIT_POINTS}",
                      verdict="INSUFFICIENT_DATA", detector=None)
        return result

    win_hi = n_avail
    t = np.arange(win_lo, win_hi)
    w = cold_occ_series(cold_ind[win_lo:win_hi])

    # --- point fit (single fit on the full 16-chain-averaged series) ---
    try:
        tau_point, A_point = fit_exp_point(t, w)
    except Exception as e:
        result.update(insufficient_data=True, reason=f"point fit failed: {e!r}",
                      verdict="INSUFFICIENT_DATA", detector=None)
        return result

    # --- bootstrap over chains for CIs ---
    A_boot, tau_boot = bootstrap_exp_over_chains(cold_ind, win_lo, win_hi, n_boot, boot_seed)
    n_boot_ok = len(A_boot)
    if n_boot_ok < 0.5 * n_boot:
        result.update(insufficient_data=True,
                      reason=f"bootstrap fit success rate too low ({n_boot_ok}/{n_boot})",
                      verdict="INSUFFICIENT_DATA", detector=None)
        return result
    A_pctl = np.percentile(A_boot, [2.5, 50, 97.5])
    tau_pctl = np.percentile(tau_boot, [2.5, 50, 97.5])
    tau_median = float(tau_pctl[1])

    # --- direct flat-window estimate ---
    flat_start = int(min(win_lo + 5 * tau_median, win_hi - 300))
    flat_start = max(flat_start, win_lo)
    if win_hi - flat_start < MIN_FLAT_POINTS:
        result.update(insufficient_data=True,
                      reason=f"flat window [{flat_start},{win_hi}) has only "
                             f"{win_hi - flat_start} rounds, need >={MIN_FLAT_POINTS}",
                      tau=dict(median=tau_median, ci_lo=float(tau_pctl[0]), ci_hi=float(tau_pctl[2])),
                      verdict="INSUFFICIENT_DATA", detector=None)
        return result

    flat_window = cold_occ_series(cold_ind[flat_start:win_hi])
    A_flat = float(flat_window.mean())
    se_ac, iat, n_eff = autocorr_se(flat_window)

    # --- dual-estimator guard ---
    A_fit_halfwidth = (float(A_pctl[2]) - float(A_pctl[0])) / 2.0
    A_flat_halfwidth = 1.96 * se_ac
    combined_halfwidth = A_fit_halfwidth + A_flat_halfwidth
    dual_agree = bool(abs(A_point - A_flat) <= combined_halfwidth)

    still_rising = bool(tau_median > (win_hi - win_lo) / 2.0)

    # --- shortfall trichotomy (primary: A_flat CI, assumption-lighter) ---
    ci_lo = A_flat - 1.96 * se_ac
    ci_hi = A_flat + 1.96 * se_ac
    if ci_hi < SHORTFALL_THRESH:
        verdict = "SHORTFALL"
    elif ci_lo >= SHORTFALL_THRESH:
        verdict = "REACHES-BAND"
    else:
        verdict = "UNRESOLVED"
    if (not dual_agree) or still_rising:
        verdict = "UNRESOLVED"

    # --- late-window detector, BC* RE-DERIVED at this window's actual N ---
    detector = detector_late_window(d, flat_start, win_hi)

    result.update(
        insufficient_data=False,
        win_hi=win_hi,
        tau=dict(median=tau_median, ci_lo=float(tau_pctl[0]), ci_hi=float(tau_pctl[2])),
        A_fit=dict(point=A_point, boot_median=float(A_pctl[1]),
                   ci_lo=float(A_pctl[0]), ci_hi=float(A_pctl[2]), n_boot_ok=n_boot_ok),
        A_flat=dict(value=A_flat, ci_lo=float(ci_lo), ci_hi=float(ci_hi), se_ac=se_ac),
        flat_start=flat_start,
        IAT=iat,
        N_eff=n_eff,
        N_window=len(flat_window),
        dual_estimator_agree=dual_agree,
        dual_estimator_diff=float(abs(A_point - A_flat)),
        dual_estimator_combined_halfwidth=float(combined_halfwidth),
        still_rising=still_rising,
        verdict=verdict,
        detector=detector,
    )
    return result


def detector_late_window(d, flat_start, win_hi, thin_b=THIN_B):
    """PART-B detector on the LATE window [flat_start, win_hi), reusing pt7's
    pinned whiten->kmeans->BC pipeline but with BC* RE-DERIVED at the ACTUAL
    N of this window (never reuse PT-7's cached 0.4275 -- different N).

    Also computes POCKET PURITY/RECALL of the k-means minority cluster against
    the ind_thin pocket ground truth for the SAME (snapshot, chain) points in
    the SAME flattening order as pos_thin[sel, 0, :, :].reshape(N, dim) (both
    are C-order flattens of the (n_sel, nsys) axes, so ind_thin[sel, 0, :]
    .reshape(-1) lines up element-for-element with the k-means labels). This
    reuses pt7's PINNED zca_whiten/bc_pipeline primitives directly (rather than
    pt7.detector_for_hot_rung, which does not expose the k-means labels) so no
    part of the pinned procedure itself is touched -- same whitening, same
    kmeans call, same seed, same BC/f/FIRES formulas; cross-checked to give
    identical BC/f/FIRES/y against pt7.detector_for_hot_rung.
    """
    if "pos_thin" not in d.files:
        return None
    ind_thin = d["ind_thin"]
    n_thin = ind_thin.shape[0]
    sel = pt7.thin_window_mask(n_thin, thin_b, flat_start, win_hi)
    if sel.sum() == 0:
        return None
    pos_thin = d["pos_thin"]
    X0 = pos_thin[sel, 0, :, :]
    n_sel, nsys, dim = X0.shape
    N = n_sel * nsys
    BC_star, _ = pt7.null_calibrate_BC_star(N, dim)

    X = X0.reshape(N, dim)
    Xw = pt7.zca_whiten(X)
    res = pt7.bc_pipeline(Xw, seed=pt7.KMEANS_SEED)
    fires = bool(res["BC"] >= BC_star and res["f"] >= 0.05)

    labels = res["labels"]
    pocket_truth = ind_thin[sel, 0, :].astype(np.uint8).reshape(-1)  # same order as labels
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))
    minority_label = 0 if n0 <= n1 else 1
    minority_mask = labels == minority_label
    n_minority = int(minority_mask.sum())
    n_minority_pocket = int(np.sum(pocket_truth[minority_mask] == 1))
    n_pocket_total = int(np.sum(pocket_truth == 1))
    pocket_purity = float(n_minority_pocket / n_minority) if n_minority > 0 else float("nan")
    pocket_recall = float(n_minority_pocket / n_pocket_total) if n_pocket_total > 0 else float("nan")

    return dict(BC=float(res["BC"]), BC_star=float(BC_star), f=float(res["f"]), FIRES=fires,
                N=int(res["N"]), dim=int(dim), y=res["y"],
                pocket_purity=pocket_purity, pocket_recall=pocket_recall,
                minority_n=n_minority, minority_n_pocket=n_minority_pocket,
                pocket_total=n_pocket_total)


# ============================================================
# PART 4 -- combine 2 seeds per beta_min
# ============================================================

def combine_seeds(results, tags):
    """A beta_min gets a verdict only if BOTH seeds agree; else seed-split."""
    present = [t for t in tags if t in results]
    if len(present) < len(tags):
        return "INSUFFICIENT_DATA (missing arm)"
    verdicts = [results[t]["verdict"] for t in present]
    if any(v == "INSUFFICIENT_DATA" for v in verdicts):
        return "INSUFFICIENT_DATA"
    if len(set(verdicts)) == 1:
        return verdicts[0]
    return f"seed-split -> unresolved ({verdicts})"


def combine_A_flat(results, tags):
    """Combine 2 seeds' A_flat point+CI into one beta_min-level estimate:
    mean of the two point estimates, half-width via independent-SE propagation
    (SE_mean = sqrt(se1^2+se2^2)/2). Returns None if either seed lacks A_flat."""
    vals, ses = [], []
    for t in tags:
        r = results.get(t)
        if r is None or r.get("insufficient_data") or "A_flat" not in r:
            return None
        vals.append(r["A_flat"]["value"])
        ses.append(r["A_flat"]["se_ac"])
    vals = np.asarray(vals)
    ses = np.asarray(ses)
    mean_val = float(vals.mean())
    se_mean = float(np.sqrt(np.sum(ses ** 2)) / len(ses))
    return dict(value=mean_val, se=se_mean, ci_lo=mean_val - 1.96 * se_mean,
                ci_hi=mean_val + 1.96 * se_mean, per_seed=vals.tolist())


# ============================================================
# PART 5 -- cross-arm slope test (A_flat vs beta_min)
# ============================================================

def slope_test(anchor_result, combined_060, combined_070):
    points = []  # (beta_min, value, ci_lo, ci_hi, label)
    if anchor_result is not None and not anchor_result.get("insufficient_data"):
        af = anchor_result["A_flat"]
        points.append((0.40, af["value"], af["ci_lo"], af["ci_hi"], "L40_anchor (single seed)"))
    if combined_060 is not None:
        points.append((0.60, combined_060["value"], combined_060["ci_lo"], combined_060["ci_hi"], "L60 (2-seed combined)"))
    if combined_070 is not None:
        points.append((0.70, combined_070["value"], combined_070["ci_lo"], combined_070["ci_hi"], "L70 (2-seed combined)"))

    if len(points) < 3:
        return dict(interpretation="UNTESTABLE (fewer than 3 of {0.40,0.60,0.70} have a resolved A_flat)",
                    points=[dict(beta_min=p[0], A_flat=p[1], ci_lo=p[2], ci_hi=p[3], label=p[4]) for p in points])

    points.sort(key=lambda p: p[0])
    (b0, v0, lo0, hi0, _l0), (b1, v1, lo1, hi1, _l1), (b2, v2, lo2, hi2, _l2) = points

    monotone_decreasing = (v0 > v1 > v2)
    # "beyond CIs" -> each pair's CIs must not overlap for a confirmed decreasing slope
    beyond_ci_01 = hi1 < lo0
    beyond_ci_12 = hi2 < lo1
    beyond_ci_02 = hi2 < lo0

    all_overlap_common_band = not (hi0 < lo1 or hi1 < lo0) and not (hi1 < lo2 or hi2 < lo1) and not (hi0 < lo2 or hi2 < lo0)

    slope = float((v2 - v0) / (b2 - b0))

    if monotone_decreasing and beyond_ci_01 and beyond_ci_12:
        interp = "beta_min-DEPENDENT: A_flat decreases monotonically with beta_min beyond combined CIs"
    elif all_overlap_common_band:
        interp = "UNIFORM sub-band bias: all three A_flat values consistent within combined CIs (flat vs beta_min)"
    else:
        interp = "AMBIGUOUS: neither a clean monotone-beyond-CI decrease nor full CI overlap"

    c25_sub_pull_flag = bool(v0 < C25_POINT)  # PT-7 L40 A~0.387 < C-25 point 0.42

    return dict(
        interpretation=interp,
        slope_per_unit_beta_min=slope,
        monotone_decreasing=bool(monotone_decreasing),
        pairwise_beyond_ci=dict(p040_vs_060=bool(beyond_ci_01), p060_vs_070=bool(beyond_ci_12), p040_vs_070=bool(beyond_ci_02)),
        points=[dict(beta_min=b0, A_flat=v0, ci_lo=lo0, ci_hi=hi0),
                dict(beta_min=b1, A_flat=v1, ci_lo=lo1, ci_hi=hi1),
                dict(beta_min=b2, A_flat=v2, ci_lo=lo2, ci_hi=hi2)],
        C25_point=C25_POINT,
        L40_below_C25_point_flag=c25_sub_pull_flag,
    )


# ============================================================
# PART 6 -- frontier (time-to-band / wall-to-band), ALL arms regardless of verdict
# ============================================================

# PT-8's OWN MEASURED per-round wall-clock rates (s/round), NOT PT-7's --
# L60 = 12722 s / 2500 rounds; L70 = 12586 s / 3200 rounds; L40 anchor is the
# PT-7 rate since that arm's wall-clock was measured under PT-7, reused here
# only as the beta_min=0.40 reference point.
WALL_RATE_S_PER_ROUND = {0.60: 12722.0 / 2500.0, 0.70: 12586.0 / 3200.0, 0.40: 7.87}


def time_to_band(cold_ind, w=ROLL_W, thresh=C25_LO, hold=FRONTIER_HOLD):
    occ = cold_occ_series(cold_ind)
    rm = rolling_mean(occ, w)
    if len(rm) == 0:
        return None
    ok = rm >= thresh
    for i in range(len(ok)):
        if ok[i] and np.all(ok[i:i + hold]) and (i + hold <= len(ok)):
            # round index of this rolling-mean sample (rm[i] centers on window [i, i+w))
            return int(i + w - 1)
    return None


def frontier_analysis(results, arm_data, anchor_result=None, anchor_data=None):
    """Frontier for ALL arms (PT-8's 4 + the PT-7 L40 anchor), regardless of
    verdict. x=beta_min, y=wall-to-band in hours, using PT-8's OWN MEASURED
    per-round wall rates (WALL_RATE_S_PER_ROUND). An arm whose rolling-mean
    cold occupancy never crosses+holds the C-25 band by rounds_done is
    reported CENSORED at a lower-bound wall time (rounds_done * rate)."""
    items = list(results.items())
    data_by_tag = dict(arm_data)
    if anchor_result is not None and anchor_data is not None:
        items.append((ANCHOR_TAG, anchor_result))
        data_by_tag[ANCHOR_TAG] = anchor_data

    rows = []
    for tag, r in items:
        d = data_by_tag.get(tag)
        if d is None:
            continue
        beta_min = r["beta_min"]
        rate = WALL_RATE_S_PER_ROUND.get(round(beta_min, 2))
        if rate is None:
            continue
        cold_ind = np.asarray(d["cold_ind"])
        rounds_done = int(r.get("rounds_done", cold_ind.shape[0]))
        ttb = time_to_band(cold_ind)
        if ttb is not None:
            rows.append(dict(tag=tag, beta_min=beta_min, seed=r.get("seed"),
                              verdict=r.get("verdict", "INSUFFICIENT_DATA"),
                              time_to_band=ttb, rate_s_per_round=rate,
                              wall_to_band_hours=ttb * rate / 3600.0,
                              rounds_done=rounds_done, censored=False))
        else:
            rows.append(dict(tag=tag, beta_min=beta_min, seed=r.get("seed"),
                              verdict=r.get("verdict", "INSUFFICIENT_DATA"),
                              time_to_band=None, rate_s_per_round=rate,
                              wall_to_band_hours=rounds_done * rate / 3600.0,
                              rounds_done=rounds_done, censored=True))
    if not rows:
        return "no arm has both cold_ind data and a known PT-8 wall rate"
    return rows


# ============================================================
# PART 7 -- plots
# ============================================================

def plot_asymptote(results, arm_data, anchor_result, anchor_data, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhspan(C25_LO, C25_HI, color="gray", alpha=0.15, label=f"C-25 band [{C25_LO},{C25_HI}]")
    any_plotted = False

    def _plot_one(tag, d, r, color):
        nonlocal any_plotted
        occ = cold_occ_series(np.asarray(d["cold_ind"]))
        rm = rolling_mean(occ, ROLL_W)
        if len(rm) == 0:
            return
        rounds = np.arange(ROLL_W - 1, ROLL_W - 1 + len(rm))
        ax.plot(rounds, rm, color=color, lw=1.4, alpha=0.9, label=f"{tag} (rolling mean)")
        any_plotted = True
        if r is not None and not r.get("insufficient_data"):
            t_fit = np.arange(WIN_LO, r["win_hi"])
            fit_curve = exp_model(t_fit, r["tau"]["median"], r["A_fit"]["point"])
            ax.plot(t_fit, fit_curve, color=color, lw=1.8, ls="--", alpha=0.8)
            af = r["A_flat"]
            ax.axhline(af["value"], color=color, lw=1.0, ls=":", alpha=0.7)
            ax.axhspan(af["ci_lo"], af["ci_hi"], color=color, alpha=0.08)

    if anchor_data is not None:
        _plot_one(ANCHOR_TAG, anchor_data, anchor_result, COLORS[ANCHOR_TAG])
    for tag, d in arm_data.items():
        _plot_one(tag, d, results.get(tag), COLORS.get(tag, "steelblue"))

    ax.set_xlabel("round")
    ax.set_ylabel(f"cold-rung pocket occupancy (rolling mean, w={ROLL_W})")
    ax.set_title("PT-8: cold-occ asymptote vs round -- warm arms climb into C-25 band or flatten below?")
    if any_plotted:
        ax.legend(fontsize=7, loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "no arm has sufficient data yet (PT-8 run in progress)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_A_vs_betamin(slope_result, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhspan(C25_LO, C25_HI, color="gray", alpha=0.15, label=f"C-25 band [{C25_LO},{C25_HI}]")
    ax.axhline(C25_POINT, color="k", lw=1.0, ls="--", alpha=0.6, label=f"C-25 point ({C25_POINT})")
    pts = slope_result.get("points", [])
    if pts:
        bmins = [p["beta_min"] for p in pts]
        vals = [p["A_flat"] for p in pts]
        lo = [p["A_flat"] - p["ci_lo"] for p in pts]
        hi = [p["ci_hi"] - p["A_flat"] for p in pts]
        ax.errorbar(bmins, vals, yerr=[lo, hi], fmt="o-", color="#d95f02", capsize=4, lw=1.6)
    else:
        ax.text(0.5, 0.5, "no beta_min has a resolved A_flat yet",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("beta_min")
    ax.set_ylabel("A_flat (asymptotic cold-rung occupancy)")
    ax.set_title(f"PT-8: A_flat vs beta_min\n{slope_result.get('interpretation', '')}", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_detector(results, anchor_result, path):
    present = [(t, r["detector"]) for t, r in results.items() if r.get("detector") is not None]
    if anchor_result is not None and anchor_result.get("detector") is not None:
        present.append((ANCHOR_TAG, anchor_result["detector"]))
    n = max(len(present), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]
    if not present:
        axes[0].text(0.5, 0.5, "no arm has a resolved late window yet",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()
    for i, (t, det) in enumerate(present):
        ax = axes[i]
        color = COLORS.get(t, "steelblue")
        ax.hist(det["y"], bins=40, color=color, alpha=0.85)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        fire_str = "FIRE" if det["FIRES"] else "no-fire"
        purity = det.get("pocket_purity", float("nan"))
        recall = det.get("pocket_recall", float("nan"))
        ax.set_title(f"{t}: {fire_str}\nBC={det['BC']:.3f} BC*={det['BC_star']:.3f} f={det['f']:.3f} "
                     f"N={det['N']}\nPOCKET PURITY={purity:.3f} (recall={recall:.3f})",
                     fontsize=8)
        ax.set_xlabel("y = X_w . v (split axis)")
    fig.suptitle("PT-8: late-window pinned hot-rung multimodality detector (BC* re-derived per window N)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(frontier_rows, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    if isinstance(frontier_rows, str) or not frontier_rows:
        msg = frontier_rows if isinstance(frontier_rows, str) else "no frontier data"
        ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True, transform=ax.transAxes)
    else:
        for r in frontier_rows:
            b, wtb, t = r["beta_min"], r["wall_to_band_hours"], r["tag"]
            color = COLORS.get(t, "#7570b3")
            if not r["censored"]:
                ax.scatter([b], [wtb], color=color, s=70, marker="o", zorder=3)
                ax.annotate(f"{t}\n{wtb:.2f}h", (b, wtb), fontsize=7,
                            xytext=(5, 5), textcoords="offset points", color=color)
            else:
                ax.scatter([b], [wtb], facecolors="none", edgecolors=color, s=90,
                           marker="o", linewidths=1.6, zorder=3)
                ax.annotate("", xy=(b, wtb * 1.14), xytext=(b, wtb),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
                ax.annotate(f"{t}\ncensored (>{wtb:.2f}h)", (b, wtb), fontsize=7,
                            xytext=(5, -16), textcoords="offset points", color=color)
    ax.set_xlabel("beta_min")
    ax.set_ylabel("wall-to-band (hours, PT-8-measured s/round)")
    ax.set_title("PT-8: frontier -- wall-to-band vs beta_min (ALL arms)\n"
                 "open marker + up-arrow = censored lower bound (never reached band by rounds_done)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================
# PART 8 -- glue / load / report
# ============================================================

def load_pt8_arms():
    data = {}
    for tag, info in ARM_INFO.items():
        matches = glob.glob(info["path"])
        if matches:
            data[tag] = np.load(matches[0])
    return data


def load_anchor():
    matches = glob.glob(ANCHOR_INFO["path"])
    if matches:
        return np.load(matches[0])
    return None


def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items() if k != "y" and k != "labels"}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def build_report(results, anchor_result, arm_data, slope_result, frontier_result):
    arms = {t: to_native(r) for t, r in results.items()}
    anchor_entry = to_native(anchor_result) if anchor_result is not None else None

    per_beta_min = {}
    for beta_min, tags in BETA_MIN_TO_TAGS.items():
        per_beta_min[str(beta_min)] = combine_seeds(results, tags)

    frontier_out = frontier_result if isinstance(frontier_result, str) else to_native(frontier_result)

    report = dict(
        summary=dict(
            n_pt8_arms_found=len(arm_data),
            n_pt8_arms_total=len(ARM_INFO),
            anchor_found=anchor_result is not None,
            C25_band=[C25_LO, C25_HI],
            slope_interpretation=slope_result.get("interpretation"),
            A_vs_betamin=slope_result.get("points"),
            frontier=frontier_out,
        ),
        per_beta_min_combined_verdict=per_beta_min,
        arms=arms,
        anchor=anchor_entry,
        slope_test=to_native(slope_result),
    )
    return report


def main(write=True):
    os.makedirs(DIAG, exist_ok=True)
    arm_data = load_pt8_arms()
    anchor_data = load_anchor()
    print(f"Found {len(arm_data)}/{len(ARM_INFO)} PT-8 arm files: {sorted(arm_data.keys())}")
    print(f"Anchor (PT-7 L40) found: {anchor_data is not None}")

    results = {}
    for tag, d in arm_data.items():
        info = ARM_INFO[tag]
        r = compute_asymptote_for_arm(d, tag, info["beta_min"], info["seed"])
        results[tag] = r
        if r.get("insufficient_data"):
            print(f"  {tag} (beta_min={info['beta_min']}, seed={info['seed']}): "
                  f"INSUFFICIENT_DATA -- {r.get('reason')}")
        else:
            print(f"  {tag} (beta_min={info['beta_min']}, seed={info['seed']}): "
                  f"verdict={r['verdict']} A_flat={r['A_flat']['value']:.4f}"
                  f"[{r['A_flat']['ci_lo']:.4f},{r['A_flat']['ci_hi']:.4f}] "
                  f"A_fit={r['A_fit']['point']:.4f} tau={r['tau']['median']:.1f} "
                  f"dual_agree={r['dual_estimator_agree']} RT_pocket={r['RT_pocket']}")

    anchor_result = None
    if anchor_data is not None:
        anchor_result = compute_asymptote_for_arm(anchor_data, ANCHOR_TAG,
                                                    ANCHOR_INFO["beta_min"], ANCHOR_INFO["seed"])
        if anchor_result.get("insufficient_data"):
            print(f"  {ANCHOR_TAG}: INSUFFICIENT_DATA -- {anchor_result.get('reason')}")
        else:
            print(f"  {ANCHOR_TAG}: verdict={anchor_result['verdict']} "
                  f"A_flat={anchor_result['A_flat']['value']:.4f} A_fit={anchor_result['A_fit']['point']:.4f} "
                  f"tau={anchor_result['tau']['median']:.1f}")

    combined_060 = combine_A_flat(results, BETA_MIN_TO_TAGS[0.60])
    combined_070 = combine_A_flat(results, BETA_MIN_TO_TAGS[0.70])
    slope_result = slope_test(anchor_result, combined_060, combined_070)
    print(f"\nSlope test: {slope_result['interpretation']}")

    frontier_result = frontier_analysis(results, arm_data, anchor_result, anchor_data)
    if isinstance(frontier_result, str):
        print(f"Frontier: {frontier_result}")
    else:
        n_censored = sum(1 for r in frontier_result if r["censored"])
        print(f"Frontier: {len(frontier_result)} arm(s) total, "
              f"{len(frontier_result) - n_censored} reached band, {n_censored} censored")
        for r in frontier_result:
            status = f"ttb={r['time_to_band']}" if not r["censored"] else "CENSORED (lower bound)"
            print(f"  {r['tag']} (beta_min={r['beta_min']}): {status} "
                  f"wall_to_band={r['wall_to_band_hours']:.3f}h (rate={r['rate_s_per_round']:.3f} s/rd)")

    if write:
        plot_asymptote(results, arm_data, anchor_result, anchor_data, os.path.join(DIAG, "pt8_asymptote.png"))
        plot_A_vs_betamin(slope_result, os.path.join(DIAG, "pt8_A_vs_betamin.png"))
        plot_detector(results, anchor_result, os.path.join(DIAG, "pt8_detector.png"))
        plot_frontier(frontier_result, os.path.join(DIAG, "pt8_frontier.png"))
        print(f"Plots written to {DIAG}/pt8_{{asymptote,A_vs_betamin,detector,frontier}}.png")

        report = build_report(results, anchor_result, arm_data, slope_result, frontier_result)
        report_path = os.path.join(DIAG, "pt8_report_table.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1)
        print(f"Report written to {report_path}")
        return report
    return None


# ============================================================
# SELF-TEST
# ============================================================

def selftest_pt7_reproduction():
    """(a) Reproduce the prototyped PT-7 exponential-fit point estimates on
    [500,1000) for L40/L60/L70: A~{0.387,0.319,0.221}, tau~{204,149,496}."""
    print("\n=== SELF-TEST (a): PT-7 exp-fit + bootstrap reproduction ===")
    targets = {"L40": (0.387, 204), "L60": (0.319, 149), "L70": (0.221, 496)}
    all_ok = True
    for tag, (A_target, tau_target) in targets.items():
        path = os.path.join(OUTDIR, f"arrays_D2_pt7_{tag}.npz")
        d = np.load(path)
        cold_ind = np.asarray(d["cold_ind"])
        win_hi = 1000
        t = np.arange(WIN_LO, win_hi)
        w = cold_occ_series(cold_ind[WIN_LO:win_hi])
        tau_point, A_point = fit_exp_point(t, w)
        A_boot, tau_boot = bootstrap_exp_over_chains(cold_ind, WIN_LO, win_hi, N_BOOT, BOOT_SEED)
        A_pctl = np.percentile(A_boot, [2.5, 50, 97.5])
        tau_pctl = np.percentile(tau_boot, [2.5, 50, 97.5])

        ok_A = abs(A_point - A_target) < 0.01
        ok_tau = abs(tau_point - tau_target) < 5
        all_ok &= (ok_A and ok_tau)
        print(f"  {tag}: A_point={A_point:.4f} (target~{A_target}) [{'OK' if ok_A else 'MISMATCH'}]  "
              f"tau_point={tau_point:.2f} (target~{tau_target}) [{'OK' if ok_tau else 'MISMATCH'}]")
        print(f"       bootstrap CI: A 2.5/50/97.5 = {A_pctl[0]:.4f}/{A_pctl[1]:.4f}/{A_pctl[2]:.4f}  "
              f"tau 2.5/50/97.5 = {tau_pctl[0]:.2f}/{tau_pctl[1]:.2f}/{tau_pctl[2]:.2f}  "
              f"(n_boot_ok={len(A_boot)}/{N_BOOT})")
    print(f"SELF-TEST (a) RESULT: {'ALL MATCH' if all_ok else 'MISMATCH DETECTED'}")
    return all_ok


def selftest_iat_sane():
    """(b) Confirm the IAT/autocorr-SE routine returns a sane (>=1) IAT on a
    PT-7 series."""
    print("\n=== SELF-TEST (b): IAT sanity on PT-7 L40 [500:1000) ===")
    d = np.load(os.path.join(OUTDIR, "arrays_D2_pt7_L40.npz"))
    cold_ind = np.asarray(d["cold_ind"])
    w = cold_occ_series(cold_ind[500:1000])
    se, iat, n_eff = autocorr_se(w)
    ok = iat >= 1.0 and np.isfinite(iat) and np.isfinite(se) and se > 0
    print(f"  IAT={iat:.3f}  N_eff={n_eff:.2f}  SE_ac={se:.4f}  (N={len(w)})  [{'OK' if ok else 'FAIL'}]")
    print(f"SELF-TEST (b) RESULT: {'OK' if ok else 'FAIL'}")
    return ok


def selftest_missing_pt8_no_crash():
    """(c) Confirm the script does not crash when PT-8 arms are absent (point
    at a scratch dir with none of the files)."""
    print("\n=== SELF-TEST (c): missing PT-8 arms robustness ===")
    fake_dir = os.path.join(DIAG, "_selftest_missing_scratch")
    os.makedirs(fake_dir, exist_ok=True)
    global ARM_INFO
    real_info = ARM_INFO
    fake_info = {t: dict(v, path=os.path.join(fake_dir, f"arrays_D2_pt8_{t}.npz")) for t, v in real_info.items()}
    ARM_INFO = fake_info
    try:
        data = load_pt8_arms()
        anchor = load_anchor()  # real anchor still exists -- also exercise that path
        results = {}
        for tag, d in data.items():
            info = fake_info[tag]
            results[tag] = compute_asymptote_for_arm(d, tag, info["beta_min"], info["seed"])
        anchor_result = None
        if anchor is not None:
            anchor_result = compute_asymptote_for_arm(anchor, ANCHOR_TAG, ANCHOR_INFO["beta_min"], ANCHOR_INFO["seed"])
        combined_060 = combine_A_flat(results, BETA_MIN_TO_TAGS[0.60])
        combined_070 = combine_A_flat(results, BETA_MIN_TO_TAGS[0.70])
        slope_result = slope_test(anchor_result, combined_060, combined_070)
        frontier_result = frontier_analysis(results, data, anchor_result, anchor)
        report = build_report(results, anchor_result, data, slope_result, frontier_result)
        plot_asymptote(results, data, anchor_result, anchor, os.path.join(fake_dir, "_tmp_asym.png"))
        plot_A_vs_betamin(slope_result, os.path.join(fake_dir, "_tmp_slope.png"))
        plot_detector(results, anchor_result, os.path.join(fake_dir, "_tmp_det.png"))
        plot_frontier(frontier_result, os.path.join(fake_dir, "_tmp_front.png"))
        print(f"  0/{len(real_info)} PT-8 arms found -> summary={report['summary']}; no crash.")
        ok = True
    except Exception as e:
        print(f"  SELF-TEST (c) FAILED with exception: {e!r}")
        ok = False
    finally:
        ARM_INFO = real_info
        import shutil
        shutil.rmtree(fake_dir, ignore_errors=True)
    return ok


def selftest_current_pt8_no_crash():
    """Bonus robustness check: run the REAL main() against whatever PT-8
    checkpoint state currently exists on disk (mid-run / short arrays) and
    confirm no crash. Does not check any science content."""
    print("\n=== SELF-TEST (bonus): current on-disk PT-8 state, no-crash check ===")
    try:
        main(write=True)
        print("  main() ran end-to-end against current on-disk PT-8 state -- no crash.")
        return True
    except Exception as e:
        print(f"  SELF-TEST (bonus) FAILED with exception: {e!r}")
        return False


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok_a = selftest_pt7_reproduction()
        ok_b = selftest_iat_sane()
        ok_c = selftest_missing_pt8_no_crash()
        ok_bonus = selftest_current_pt8_no_crash()
        print(f"\n=== SELF-TEST SUMMARY: pt7_repro={ok_a} iat_sane={ok_b} "
              f"missing_pt8_ok={ok_c} current_pt8_no_crash={ok_bonus} ===")
        sys.exit(0 if (ok_a and ok_b and ok_c and ok_bonus) else 1)
    main()
