"""PT-7 sweep diagnostic: transport ground truth + a PINNED reference-free
hot-rung multimodality detector.

Read-only analysis of arrays_D2_pt7_{L40,L50,L60,L70}.npz (may not all exist
yet -- a GPU sweep may still be running; this script globs for whichever
exist and skips the rest without crashing).

PART A ground-truth formulas (per_rung_occ / cold_occ source / RT_pocket /
swap_acc) are copied to match diag_pt6/make_figs.py EXACTLY, verified by the
self-test below to reproduce diag_pt6/pt6_report_table.json bit-for-rounding.

PART B is a PINNED reference-free multimodality detector on the hottest rung:
ZCA-whiten -> k-means (k=2) -> project onto the centroid-split axis -> Sarle's
bimodality coefficient (non-excess kurtosis) -> null-calibrated threshold from
200 matched-N,dim standard-normal replicas (same whiten/kmeans/BC pipeline).

No new runs. No fabricated results: an arm whose window is not yet covered by
its data (rounds_done too small) is reported INSUFFICIENT_DATA, never a verdict.
"""
import os
import sys
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

try:
    from sklearn.cluster import KMeans as _SKKMeans
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

OUTDIR = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
DIAG = os.path.join(OUTDIR, "diag_pt7")

TAGS = ["L40", "L50", "L60", "L70"]
COLORS = {"L40": "#1b9e77", "L50": "#7570b3", "L60": "#d95f02", "L70": "#e7298a"}

THIN_B = 5
WIN_LO_DEFAULT, WIN_HI_DEFAULT = 500, 1000
N_NULL = 200
NULL_PCTL = 99.0
KMEANS_N_INIT = 20
KMEANS_SEED = 0


# ============================================================
# PART A -- transport ground truth (formulas matched to make_figs.py)
# ============================================================

def thin_window_mask(n_thin, thin_b, win_lo, win_hi):
    """Boolean mask over thin-snapshot index selecting rounds in [win_lo, win_hi).
    Matches diag_pt6/make_figs.py's construction exactly:
        thin_rounds = arange(0, thin_b*n_thin, thin_b); sel = (thin_rounds>=lo)&(<hi)
    """
    thin_rounds = np.arange(0, thin_b * n_thin, thin_b)
    return (thin_rounds >= win_lo) & (thin_rounds < win_hi)


def transport_ground_truth(d, win_lo=WIN_LO_DEFAULT, win_hi=WIN_HI_DEFAULT, thin_b=THIN_B):
    """Reproduce make_figs.py's PT-6 report-table quantities, generalized to a
    parametrized window (needed for the smoke self-test, where [500:1000) is
    empty).

    Sources (matched to make_figs.py, NOT guessed):
      - per_rung_occ[r]   <- ind_thin[sel, r, :].mean()                (Fig-3 / report_perrung)
      - hot_rung_occ      <- per_rung_occ[0]                            (report_perrung[t][0])
      - cold_occ          <- cold_ind[win_lo:win_hi].mean()             (report table's cold_occ_500_1000;
                                                                          round-resolution source, NOT ind_thin[-1] --
                                                                          make_figs.py uses a DIFFERENT array for this
                                                                          field than for per-rung occ; verified they
                                                                          differ slightly on PT-6 Lcert: 0.2455 vs 0.2425)
      - RT_pocket         <- int(round_trips_pocket.sum())              (cumulative, whole run, no windowing)
      - swap_acc[b]       <- swap_accepts.sum(1) / swap_attempts.sum(1) (cumulative, whole run, no windowing)
    """
    betas = np.asarray(d["betas"])
    nrung = len(betas)
    ind_thin = d["ind_thin"]  # (n_thin, nrung, nsys)
    n_thin = ind_thin.shape[0]
    cold_ind = d["cold_ind"]  # (rounds, nsys)

    sel = thin_window_mask(n_thin, thin_b, win_lo, win_hi)
    n_sel = int(sel.sum())
    cold_ok = cold_ind.shape[0] >= win_hi

    # swap acceptance and RT_pocket do not depend on the window -> always safe
    att = d["swap_attempts"].astype(np.float64)
    acc = d["swap_accepts"].astype(np.float64)
    denom = att.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        swap_acc = np.where(denom > 0, acc.sum(axis=1) / np.where(denom > 0, denom, 1.0), np.nan)
    rt_pocket = int(np.asarray(d["round_trips_pocket"]).sum())

    out = dict(
        betas=betas.tolist(),
        n_rungs=int(nrung),
        n_thin_total=int(n_thin),
        n_thin_in_window=n_sel,
        RT_pocket=rt_pocket,
        swap_acc=swap_acc.tolist(),
        swap_acc_mean=float(np.nanmean(swap_acc)),
        swap_acc_min=float(np.nanmin(swap_acc)),
    )

    if n_sel == 0 or not cold_ok:
        out.update(
            insufficient_data=True,
            per_rung_occ=None,
            cold_occ=None,
            hot_rung_occ=None,
            cold_hot_ratio=None,
            transport_verdict="INSUFFICIENT_DATA",
            hot_rung_discovery=None,
        )
        return out

    per_rung_occ = ind_thin[sel].astype(np.float64).mean(axis=(0, 2))  # (nrung,)
    cold_occ = float(cold_ind[win_lo:win_hi].mean())
    hot_rung_occ = float(per_rung_occ[0])
    cold_hot_ratio = cold_occ / max(hot_rung_occ, 1e-9)

    if cold_hot_ratio > 0.5 and rt_pocket >= 10:
        verdict = "SUCCESS"
    elif cold_hot_ratio < 0.2 and rt_pocket <= 2:
        verdict = "FAIL"
    else:
        verdict = "MIDDLE"

    out.update(
        insufficient_data=False,
        per_rung_occ=per_rung_occ.tolist(),
        cold_occ=cold_occ,
        hot_rung_occ=hot_rung_occ,
        cold_hot_ratio=float(cold_hot_ratio),
        transport_verdict=verdict,
        hot_rung_discovery=bool(hot_rung_occ > 0.02),
    )
    return out


# ============================================================
# PART B -- pinned reference-free hot-rung multimodality detector
# ============================================================

def zca_whiten(X):
    """Full ZCA whitening. X: (N, dim) -> X_w with ~identity covariance."""
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = np.cov(Xc, rowvar=False)
    dim = Sigma.shape[0]
    eps = 1e-6 * np.trace(Sigma) / dim
    Sigma = Sigma + eps * np.eye(dim)
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 1e-300, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    Xw = Xc @ W
    return Xw


def _kmeans2_numpy(X, n_init=KMEANS_N_INIT, seed=KMEANS_SEED, max_iter=300, tol=1e-8):
    """Fixed-seed 2-means fallback (kmeans++-style init, n_init restarts,
    keep lowest-inertia result). Used only if sklearn is unavailable."""
    n, dim = X.shape
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_centroids = None
    best_labels = None
    for _ in range(n_init):
        idx0 = rng.integers(0, n)
        c0 = X[idx0]
        d2 = np.sum((X - c0) ** 2, axis=1)
        s = d2.sum()
        probs = d2 / s if s > 0 else np.full(n, 1.0 / n)
        idx1 = rng.choice(n, p=probs)
        centroids = np.stack([c0, X[idx1]]).astype(np.float64)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d0 = np.sum((X - centroids[0]) ** 2, axis=1)
            d1 = np.sum((X - centroids[1]) ** 2, axis=1)
            new_labels = (d1 < d0).astype(int)
            new_c0 = X[new_labels == 0].mean(axis=0) if np.any(new_labels == 0) else centroids[0]
            new_c1 = X[new_labels == 1].mean(axis=0) if np.any(new_labels == 1) else centroids[1]
            new_centroids = np.stack([new_c0, new_c1])
            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            labels = new_labels
            if shift < tol:
                break
        d0 = np.sum((X - centroids[0]) ** 2, axis=1)
        d1 = np.sum((X - centroids[1]) ** 2, axis=1)
        labels = (d1 < d0).astype(int)
        inertia = float(np.sum(np.where(labels == 0, d0, d1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.copy()
            best_labels = labels.copy()
    return best_centroids, best_labels


def kmeans2(X, n_init=KMEANS_N_INIT, seed=KMEANS_SEED):
    if HAVE_SKLEARN:
        km = _SKKMeans(n_clusters=2, n_init=n_init, random_state=seed).fit(X)
        return km.cluster_centers_, km.labels_
    return _kmeans2_numpy(X, n_init=n_init, seed=seed)


def bc_pipeline(Xw, seed=KMEANS_SEED):
    """whiten(already applied) -> kmeans(k=2) -> split-axis projection -> BC.
    Returns dict with BC, f (minority fraction), y (1-D projection), labels.
    """
    centroids, labels = kmeans2(Xw, seed=seed)
    c1, c2 = centroids[0], centroids[1]
    v = c2 - c1
    norm = np.linalg.norm(v)
    if norm < 1e-300:
        v = np.zeros_like(v)
        v[0] = 1.0
    else:
        v = v / norm
    y = Xw @ v
    n1 = int(np.sum(labels == 0))
    n2 = int(np.sum(labels == 1))
    N = len(labels)
    f = min(n1, n2) / N
    g = skew(y)
    k = kurtosis(y, fisher=False)
    BC = (g ** 2 + 1) / k
    return dict(BC=float(BC), f=float(f), y=y, labels=labels, n1=n1, n2=n2, N=N)


_NULL_CACHE = {}


def null_calibrate_BC_star(N, dim, n_null=N_NULL, pctl=NULL_PCTL, kmeans_seed=KMEANS_SEED):
    """BC* = pctl-th percentile of BC values from n_null standard-normal
    (N, dim) replicas run through the IDENTICAL kmeans->project->BC pipeline.
    Cached on (N, dim) since it is deterministic given those and is meant to
    be computed once and reused across arms with matching (N, dim).
    """
    key = (int(N), int(dim), int(n_null), float(pctl), int(kmeans_seed))
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    bcs = np.empty(n_null, dtype=np.float64)
    for s in range(n_null):
        rng = np.random.default_rng(s)
        Z = rng.standard_normal((N, dim))
        # IDENTICAL pipeline as the real data: the null must pass through the
        # SAME zca_whiten step (else the real data is exactly-identity-cov while
        # the null retains finite-sample residual cov the k-means axis can
        # exploit -> mismatched BC reference). Whitening N(0,I) draws is cheap
        # and makes whiten->kmeans->project->BC truly identical across data/null.
        Zw = zca_whiten(Z)
        res = bc_pipeline(Zw, seed=kmeans_seed)
        bcs[s] = res["BC"]
    BC_star = float(np.percentile(bcs, pctl))
    _NULL_CACHE[key] = (BC_star, bcs)
    return BC_star, bcs


def detector_for_hot_rung(pos_thin, sel, BC_star):
    """pos_thin: (n_thin, nrung, nsys, dim). sel: bool mask over n_thin axis.
    Runs the pinned detector on the hottest rung (index 0) within the window.
    """
    X = pos_thin[sel, 0, :, :]  # (n_sel, nsys, dim)
    n_sel, nsys, dim = X.shape
    X = X.reshape(n_sel * nsys, dim)
    Xw = zca_whiten(X)
    res = bc_pipeline(Xw, seed=KMEANS_SEED)
    fires = bool(res["BC"] >= BC_star and res["f"] >= 0.05)
    return dict(N=res["N"], dim=int(dim), BC=res["BC"], BC_star=float(BC_star),
                f=res["f"], FIRES=fires, y=res["y"])


# ============================================================
# glue: run Part A + Part B over whichever PT-7 arms exist
# ============================================================

def load_pt7_arms(outdir=OUTDIR, tags=TAGS):
    data = {}
    for t in tags:
        path = os.path.join(outdir, f"arrays_D2_pt7_{t}.npz")
        matches = glob.glob(path)
        if matches:
            data[t] = np.load(matches[0])
    return data


def analyze_arms(data, win_lo=WIN_LO_DEFAULT, win_hi=WIN_HI_DEFAULT, thin_b=THIN_B):
    """Returns (ground_truth: {tag: dict}, detector: {tag: dict or None}, BC_star or None)."""
    ground_truth = {t: transport_ground_truth(d, win_lo, win_hi, thin_b) for t, d in data.items()}

    ref_dims = None
    for t, d in data.items():
        gt = ground_truth[t]
        if gt.get("insufficient_data") or "pos_thin" not in d.files:
            continue
        ind_thin = d["ind_thin"]
        sel = thin_window_mask(ind_thin.shape[0], thin_b, win_lo, win_hi)
        X0 = d["pos_thin"][sel, 0, :, :]
        ref_dims = (X0.shape[0] * X0.shape[1], X0.shape[2])
        break

    detector = {t: None for t in data}
    BC_star = None
    if ref_dims is not None:
        N_ref, dim_ref = ref_dims
        BC_star, _null_bcs = null_calibrate_BC_star(N_ref, dim_ref)
        for t, d in data.items():
            gt = ground_truth[t]
            if gt.get("insufficient_data") or "pos_thin" not in d.files:
                continue
            ind_thin = d["ind_thin"]
            sel = thin_window_mask(ind_thin.shape[0], thin_b, win_lo, win_hi)
            det = detector_for_hot_rung(d["pos_thin"], sel, BC_star)
            det["detector_vs_truth_agree"] = bool(det["FIRES"] == bool(gt["hot_rung_occ"] > 0.02))
            detector[t] = det

    return ground_truth, detector, BC_star


# ============================================================
# PART C -- plots
# ============================================================

def _label_for(t, gt):
    if gt.get("betas"):
        return f"{t} (beta_min={gt['betas'][0]:.3f})"
    return t


def plot_occ_profile(ground_truth, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False
    for t in TAGS:
        gt = ground_truth.get(t)
        if gt is None or gt.get("insufficient_data") or gt.get("per_rung_occ") is None:
            continue
        occ = np.asarray(gt["per_rung_occ"])
        ax.plot(np.arange(len(occ)), occ, marker="o", lw=1.6,
                color=COLORS.get(t), label=_label_for(t, gt))
        any_plotted = True
    ax.set_xlabel("rung index (0=hottest .. n-1=coldest)")
    ax.set_ylabel(f"pocket occupancy, time-avg over rounds [{WIN_LO_DEFAULT}:{WIN_HI_DEFAULT})")
    ax.set_title("PT-7: per-rung pocket occupancy vs rung\n(FLAT=transport, MONOTONE-DECAYING=fail)")
    if any_plotted:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "no arm has sufficient data in window yet", ha="center",
                va="center", transform=ax.transAxes)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_detector(detector, path):
    present = [t for t in TAGS if detector.get(t) is not None]
    n = max(len(present), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]
    if not present:
        axes[0].text(0.5, 0.5, "no arm has sufficient data\nfor the detector yet",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()
    for i, t in enumerate(present):
        det = detector[t]
        ax = axes[i]
        ax.hist(det["y"], bins=40, color=COLORS.get(t, "steelblue"), alpha=0.85)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        fire_str = "FIRE" if det["FIRES"] else "no-fire"
        ax.set_title(f"{t}: {fire_str}\nBC={det['BC']:.3f} BC*={det['BC_star']:.3f} f={det['f']:.3f}",
                     fontsize=9)
        ax.set_xlabel("y = X_w . v (split axis)")
    fig.suptitle("PT-7: pinned reference-free hot-rung multimodality detector", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_cold_occ_t(data, path, win_lo=WIN_LO_DEFAULT, win_hi=WIN_HI_DEFAULT, w=25):
    fig, ax = plt.subplots(figsize=(8, 5))

    def roll_mean(x, w):
        if len(x) < w:
            return np.array([])
        c = np.cumsum(np.insert(x.astype(np.float64), 0, 0))
        return (c[w:] - c[:-w]) / w

    any_plotted = False
    for t in TAGS:
        d = data.get(t)
        if d is None:
            continue
        cold_ind = d["cold_ind"]
        if cold_ind.shape[0] < win_hi:
            continue
        occ = cold_ind[win_lo:win_hi].mean(axis=1).astype(np.float64)
        rm = roll_mean(occ, w)
        if len(rm) == 0:
            continue
        rounds = win_lo + np.arange(w - 1, w - 1 + len(rm))
        ax.plot(rounds, rm, label=f"{t}", color=COLORS.get(t), lw=1.6)
        any_plotted = True
    ax.set_xlabel("round")
    ax.set_ylabel(f"cold-rung pocket occupancy (rolling mean, w={w})")
    ax.set_title(f"PT-7: cold-rung pocket occupancy vs round, [{win_lo}:{win_hi})")
    if any_plotted:
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no arm covers this window yet", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_swap_acc(ground_truth, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(0.45, 0.60, color="gray", alpha=0.15, label="matched-alpha band [0.45,0.60]")
    any_plotted = False
    for t in TAGS:
        gt = ground_truth.get(t)
        if gt is None:
            continue
        rate = np.asarray(gt["swap_acc"])
        if rate.size == 0:
            continue
        bidx = np.arange(len(rate))
        ax.plot(bidx, rate, marker="o", label=_label_for(t, gt), color=COLORS.get(t), lw=1.6)
        any_plotted = True
    ax.set_xlabel("boundary index (0 = hottest pair)")
    ax.set_ylabel("cumulative swap acceptance rate")
    ax.set_title("PT-7: per-boundary cumulative swap acceptance")
    if any_plotted:
        ax.legend(fontsize=8, loc="best")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================
# PART D -- report
# ============================================================

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


def build_report(ground_truth, detector, BC_star):
    arms = {}
    for t in TAGS:
        gt = ground_truth.get(t)
        if gt is None:
            continue
        det = detector.get(t)
        entry = dict(gt)
        entry["detector"] = None
        entry["detector_vs_truth_agree"] = None
        if det is not None:
            entry["detector"] = dict(BC=det["BC"], BC_star=det["BC_star"], f=det["f"], FIRES=det["FIRES"])
            entry["detector_vs_truth_agree"] = det["detector_vs_truth_agree"]
        arms[t] = entry

    n_agree = sum(1 for t in arms if arms[t].get("detector_vs_truth_agree") is True)
    n_with_detector = sum(1 for t in arms if arms[t].get("detector") is not None)

    success_arms = [t for t in arms if arms[t].get("transport_verdict") == "SUCCESS"]
    fail_arms = [t for t in arms if arms[t].get("transport_verdict") == "FAIL"]
    warmest_success = None
    coldest_fail = None
    if success_arms:
        warmest_success = max(success_arms, key=lambda t: arms[t]["betas"][0])
    if fail_arms:
        coldest_fail = min(fail_arms, key=lambda t: arms[t]["betas"][0])

    summary = dict(
        n_arms_found=len(arms),
        n_arms_total=len(TAGS),
        BC_star=BC_star,
        n_with_detector=n_with_detector,
        n_agree=n_agree,
        agree_str=f"{n_agree}/{n_with_detector}",
        warmest_success_arm=warmest_success,
        coldest_fail_arm=coldest_fail,
        beta_min_bracket=dict(
            warmest_success_beta_min=arms[warmest_success]["betas"][0] if warmest_success else None,
            coldest_fail_beta_min=arms[coldest_fail]["betas"][0] if coldest_fail else None,
        ),
    )
    report = dict(summary=summary, arms=arms)
    return to_native(report)


# ============================================================
# main
# ============================================================

def main(outdir=OUTDIR, diag=DIAG, win_lo=WIN_LO_DEFAULT, win_hi=WIN_HI_DEFAULT):
    os.makedirs(diag, exist_ok=True)
    data = load_pt7_arms(outdir)
    print(f"Found {len(data)}/{len(TAGS)} PT-7 arm files: {sorted(data.keys())}")
    if not data:
        print("No PT-7 arrays found yet -- writing empty report, no plots with data.")

    ground_truth, detector, BC_star = analyze_arms(data, win_lo, win_hi)

    for t in TAGS:
        gt = ground_truth.get(t)
        if gt is None:
            print(f"  {t}: (missing)")
            continue
        if gt.get("insufficient_data"):
            print(f"  {t}: INSUFFICIENT_DATA (n_thin_in_window={gt['n_thin_in_window']}, "
                  f"n_thin_total={gt['n_thin_total']}) -- RT_pocket={gt['RT_pocket']} "
                  f"swap_acc_mean={gt['swap_acc_mean']:.4f}")
        else:
            det = detector.get(t)
            det_str = "n/a"
            if det is not None:
                det_str = f"BC={det['BC']:.3f} BC*={det['BC_star']:.3f} f={det['f']:.3f} FIRES={det['FIRES']}"
            print(f"  {t}: verdict={gt['transport_verdict']} cold_hot_ratio={gt['cold_hot_ratio']:.4f} "
                  f"RT_pocket={gt['RT_pocket']} hot_rung_occ={gt['hot_rung_occ']:.4f} | detector: {det_str}")

    plot_occ_profile(ground_truth, os.path.join(diag, "pt7_occ_profile.png"))
    plot_detector(detector, os.path.join(diag, "pt7_detector.png"))
    plot_cold_occ_t(data, os.path.join(diag, "pt7_cold_occ_t.png"), win_lo, win_hi)
    plot_swap_acc(ground_truth, os.path.join(diag, "pt7_swap_acc.png"))
    print(f"Plots written to {diag}/pt7_{{occ_profile,detector,cold_occ_t,swap_acc}}.png")

    report = build_report(ground_truth, detector, BC_star)
    report_path = os.path.join(diag, "pt7_report_table.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"Report written to {report_path}")
    return report


# ============================================================
# SELF-TEST
# ============================================================

def selftest_pt6_reproduction(outdir=OUTDIR):
    """PART A must reproduce diag_pt6/pt6_report_table.json on the PT-6 arrays."""
    tags6 = ["Lcert", "La", "Lb", "Lc"]
    ref_path = os.path.join(outdir, "diag_pt6", "pt6_report_table.json")
    with open(ref_path) as f:
        ref = json.load(f)

    print("\n=== SELF-TEST 1: PT-6 reproduction (PART A) ===")
    all_ok = True
    for t in tags6:
        d = np.load(os.path.join(outdir, f"arrays_D2_pt6_{t}.npz"))
        gt = transport_ground_truth(d, win_lo=WIN_LO_DEFAULT, win_hi=WIN_HI_DEFAULT, thin_b=THIN_B)
        r = ref[t]

        checks = [
            ("cold_occ", gt["cold_occ"], r["cold_occ_500_1000"]),
            ("hot_rung_occ", gt["hot_rung_occ"], r["hot_rung_occ_500_1000"]),
            ("RT_pocket", gt["RT_pocket"], r["RT_pocket"]),
            ("swap_acc_min", gt["swap_acc_min"], r["swap_acc_min"]),
            ("swap_acc_mean", gt["swap_acc_mean"], r["swap_acc_mean"]),
        ]
        print(f"\n  arm {t}:")
        for name, mine, theirs in checks:
            ok = np.isclose(mine, theirs, atol=1e-9, rtol=1e-6)
            all_ok &= ok
            flag = "OK" if ok else "MISMATCH"
            print(f"    {name:15s} mine={mine!r:>22} ref={theirs!r:>22}  [{flag}]")
        per_rung_mine = np.asarray(gt["per_rung_occ"])
        per_rung_ref = np.asarray(r["per_rung_occ_500_1000"])
        ok = per_rung_mine.shape == per_rung_ref.shape and np.allclose(per_rung_mine, per_rung_ref, atol=1e-9)
        all_ok &= ok
        flag = "OK" if ok else "MISMATCH"
        print(f"    {'per_rung_occ':15s} mine={np.round(per_rung_mine, 4).tolist()}")
        print(f"    {'':15s} ref ={np.round(per_rung_ref, 4).tolist()}  [{flag}]")

    print(f"\nSELF-TEST 1 RESULT: {'ALL MATCH' if all_ok else 'MISMATCH DETECTED'}")
    return all_ok


def selftest_smoke_detector_mechanics(outdir=OUTDIR):
    """PART B mechanics check on arrays_D2_pt7smoke.npz. [500:1000) is empty
    for this file (n_thin=6, ROUNDS=30) -- use ALL available thin indices
    instead, purely to confirm whiten->kmeans->BC->null->fire runs end to end
    without shape errors. This is a mechanics test, not science."""
    print("\n=== SELF-TEST 2: smoke detector mechanics (PART B) ===")
    path = os.path.join(outdir, "arrays_D2_pt7smoke.npz")
    d = np.load(path)
    ind_thin = d["ind_thin"]
    n_thin = ind_thin.shape[0]
    print(f"  smoke file: n_thin={n_thin}, ROUNDS={d['cold_ind'].shape[0]}, "
          f"nrung={ind_thin.shape[1]}, nsys={ind_thin.shape[2]}, "
          f"pos_thin={d['pos_thin'].shape}")

    sel_all = np.ones(n_thin, dtype=bool)  # use ALL available thin snapshots
    X = d["pos_thin"][sel_all, 0, :, :]
    n_sel, nsys, dim = X.shape
    N = n_sel * nsys
    print(f"  window=ALL -> N={N} (={n_sel} thin x {nsys} sys), dim={dim}")

    try:
        BC_star, _ = null_calibrate_BC_star(N, dim)
        det = detector_for_hot_rung(d["pos_thin"], sel_all, BC_star)
        print(f"  BC={det['BC']:.4f} BC*={det['BC_star']:.4f} f={det['f']:.4f} "
              f"FIRES={det['FIRES']} -- pipeline ran end-to-end, no shape errors")
        return True, det
    except Exception as e:
        print(f"  SELF-TEST 2 FAILED with exception: {e!r}")
        return False, None


def selftest_missing_arms_no_crash(outdir=OUTDIR, diag=DIAG):
    """Confirm the glue code does not crash if some/all of the 4 PT-7 arm
    files are absent (simulated by pointing at a tag set that (mostly) does
    not exist)."""
    print("\n=== SELF-TEST 3: missing-arms robustness ===")
    fake_dir = os.path.join(diag, "_selftest_missing_scratch")
    os.makedirs(fake_dir, exist_ok=True)
    data = load_pt7_arms(fake_dir)  # empty dir -> zero arms found
    try:
        ground_truth, detector, BC_star = analyze_arms(data)
        report = build_report(ground_truth, detector, BC_star)
        plot_occ_profile(ground_truth, os.path.join(fake_dir, "_tmp_occ.png"))
        plot_detector(detector, os.path.join(fake_dir, "_tmp_det.png"))
        plot_cold_occ_t(data, os.path.join(fake_dir, "_tmp_cold.png"))
        plot_swap_acc(ground_truth, os.path.join(fake_dir, "_tmp_swap.png"))
        print(f"  0/4 arms found -> report={report['summary']}; no crash.")
        ok = True
    except Exception as e:
        print(f"  SELF-TEST 3 FAILED with exception: {e!r}")
        ok = False
    finally:
        import shutil
        shutil.rmtree(fake_dir, ignore_errors=True)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok1 = selftest_pt6_reproduction()
        ok2, _ = selftest_smoke_detector_mechanics()
        ok3 = selftest_missing_arms_no_crash()
        print(f"\n=== SELF-TEST SUMMARY: pt6_repro={ok1} smoke_mechanics={ok2} missing_arms_ok={ok3} ===")
        sys.exit(0 if (ok1 and ok2 and ok3) else 1)
    main()
