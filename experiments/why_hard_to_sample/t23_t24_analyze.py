"""T23 + T24 analysis (LOGIN-NODE stage -- numpy/scipy/matplotlib ONLY, NO jax).

Reads the UNCERTIFIED census arrays produced by t23_momentum_gpu.py /
t24_census_gpu.py and produces, under the same t23/t24 dirs:

  T23 (PLOTS FIRST, then metrics):
    * xi (log) vs z_Rs unconstrained coordinate over ALL results-phase steps
      (raw positions read directly from the login-readable t21 npz files);
    * spike-vs-calm ECDF of |z_k| for every hard-bounded param;
    * spike-vs-calm ECDF of c_t and of stability number eps*sqrt(max(c,0));
    * turn-angle histograms spike vs calm;
    * C3 loading heatmap (mean |loading| per fixed eigenvector, spike vs calm);
    * metrics json: per-arm wall statistic (spike-median-|z_k| quantile within
      the calm distribution), stiff statistic (median c spike/calm), clone
      controls, and the registered P/F verdict fields -- all "proposed
      (UNCERTIFIED)".

  T24:
    * per-segment overlay plots (c_t & xi twin-axis traces, census spikes marked);
    * per-segment + pooled stats json: spike rate (c>3x segment median),
      max/median, widths, spacings, Spearman rho(c,xi) & rho(c, xi smoothed w=5),
      comparison to frac(xi>10) (registered 2x band), clone versions, F-T24 fields.

Registered thresholds (evaluated but marked UNCERTIFIED; adjudication is the
orchestrator's):
  P-T23-wall  MET if spike |z_Rs| median quantile within calm >= 0.80
  F-T23-wall  FIRES if that quantile in [0.35, 0.65]
  P-T23-stiff MET if median c_t(spike)/c_t(calm) >= 10
  F-T23-stiff FIRES if that ratio < 2
  P-T23-clone (sanity) clone shows neither signature (quantile in [0.35,0.65]
              AND ratio < 2)
  T24 predictions: pooled rho >= 0.4; census spike rate within 2x of frac(xi>10)
              (old ~0.15, new ~0.08); widths 1-3 steps; clone near-zero.
  F-T24  FIRES if rho < 0.15 on >= 6/8 segments in BOTH arms AND spike-rate
              mismatch > 5x.

Run: /global/homes/l/linusu/.conda/envs/gigalens_multinode_env/bin/python \
        t23_t24_analyze.py [--arm both] [--smoke]
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
from scipy.stats import spearmanr

import t23_t24_common as C

# registered thresholds
WALL_MET = 0.80
WALL_FALSE_LO, WALL_FALSE_HI = 0.35, 0.65
STIFF_MET = 10.0
STIFF_FALSE = 2.0
T24_RHO_MET = 0.4
T24_RHO_FALSE = 0.15
T24_SPIKE_BAND = 2.0
T24_SPIKE_MISMATCH = 5.0
REF_XI_FRAC = {"old": 0.15, "new": 0.08}   # registered reference frac(xi>10)


def _now():
    return datetime.now(timezone.utc).isoformat()


def ecdf(x):
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x)]
    y = np.arange(1, len(x) + 1) / max(1, len(x))
    return x, y


def quantile_within(value, dist):
    """Fraction of `dist` <= value (the quantile position of `value`)."""
    dist = np.asarray(dist, float)
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0 or not np.isfinite(value):
        return float("nan")
    return float(np.mean(dist <= value))


def load_t23(arm, smoke):
    suf = "_smoke" if smoke else ""
    d = dict(np.load(os.path.join(C.T23_OUT, f"t23_{arm}{suf}.npz"), allow_pickle=True))
    man = json.load(open(os.path.join(C.T23_OUT, f"t23_{arm}{suf}.manifest.json")))
    return d, man


def load_t24(arm, smoke):
    suf = "_smoke" if smoke else ""
    d = dict(np.load(os.path.join(C.T24_OUT, f"t24_{arm}{suf}.npz"), allow_pickle=True))
    man = json.load(open(os.path.join(C.T24_OUT, f"t24_{arm}{suf}.manifest.json")))
    return d, man


# ===========================================================================
# T23 plots
# ===========================================================================

def plot_xi_vs_zRs(arm, rs_index, out_path):
    """xi (log) vs z_Rs over ALL results-phase steps (raw t21 positions)."""
    zs, xis = [], []
    for s in C.SEEDS:
        run = C.load_run(C.seed_npz(arm, s))
        pos = run["position"][:, C.RESULTS_LO:C.RESULTS_HI, rs_index].reshape(-1)
        xi = run["xi"][:, C.RESULTS_LO:C.RESULTS_HI].reshape(-1)
        zs.append(pos); xis.append(xi)
    z = np.concatenate(zs); xi = np.concatenate(xis)
    xi = np.clip(xi, 1e-6, None)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    hb = ax.hexbin(z, np.log10(xi), gridsize=60, cmap="viridis", bins="log", mincnt=1)
    ax.axhline(np.log10(C.XI_SPIKE_THRESH), color="crimson", lw=1, ls="--",
               label=f"xi={C.XI_SPIKE_THRESH:g}")
    ax.set_xlabel(f"z_Rs (unconstrained coord, column {rs_index})")
    ax.set_ylabel("log10(xi)")
    ax.set_title(f"T23 {arm}: xi vs z_Rs over ALL results-phase steps")
    ax.legend(fontsize=8)
    fig.colorbar(hb, ax=ax, label="log10 count")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_bounded_absz_ecdf(arm, d, man, out_path):
    """ECDF of |z_k| spike vs calm for every hard-bounded param."""
    idx = np.asarray(d["bounded_idx"]); names = man["bounded"]["names"]
    sp = d["real_spike_z"]; ca = d["real_calm_z"]
    n = len(idx)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for j, (k, nm) in enumerate(zip(idx, names)):
        ax = axes[j // ncol][j % ncol]
        xs, ys = ecdf(np.abs(sp[:, k])); ax.plot(xs, ys, color="C3", label="spike")
        xc, yc = ecdf(np.abs(ca[:, k])); ax.plot(xc, yc, color="C2", label="calm")
        ax.set_title(f"|{nm}| (col {k})", fontsize=8)
        ax.set_xlabel("|z_k|"); ax.set_ylabel("ECDF"); ax.legend(fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"T23 {arm}: |z_k| spike vs calm (hard-bounded params) -- wall check")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_ct_stability_ecdf(arm, d, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for grp, col in [("real_spike", "C3"), ("real_calm", "C2"),
                     ("clone_spike", "C1"), ("clone_calm", "C0")]:
        ct = d[f"{grp}_c_t"]; eps = d[f"{grp}_eps"]
        xs, ys = ecdf(ct[ct > -np.inf])
        ls = "--" if grp.startswith("clone") else "-"
        axes[0].plot(xs, ys, color=col, ls=ls, label=grp)
        stab = eps * np.sqrt(np.clip(ct, 0, None))
        xs2, ys2 = ecdf(stab)
        axes[1].plot(xs2, ys2, color=col, ls=ls, label=grp)
    axes[0].set_xlabel("c_t (curvature along motion)"); axes[0].set_ylabel("ECDF")
    axes[0].set_title("c_t ECDF"); axes[0].legend(fontsize=7)
    axes[0].set_xscale("symlog")
    axes[1].set_xlabel("eps*sqrt(max(c_t,0)) (stability number)"); axes[1].set_ylabel("ECDF")
    axes[1].set_title("stability number ECDF"); axes[1].legend(fontsize=7)
    fig.suptitle(f"T23 {arm}: c_t & stability number, spike vs calm (+ clone)")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_turn_hist(arm, d, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    bins = np.linspace(-1, 1, 41)
    for ax, key, ttl in [(axes[0], "turn_prev", "cos(Dz_{t-1},Dz_t)"),
                         (axes[1], "turn_next", "cos(Dz_t,Dz_{t+1})")]:
        for grp, col in [("real_spike", "C3"), ("real_calm", "C2")]:
            v = d[f"{grp}_{key}"]; v = v[np.isfinite(v)]
            ax.hist(v, bins=bins, density=True, histtype="step", color=col, label=grp)
        ax.axvline(-1, color="0.7", lw=0.5)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("cos(turn)"); ax.legend(fontsize=7)
    fig.suptitle(f"T23 {arm}: turn-angle cosines (reversal ~ -1) spike vs calm")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_loading_heatmap(arm, d, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, grp in [(axes[0], "real_spike"), (axes[1], "real_calm")]:
        L = np.abs(d[f"{grp}_loadings"])           # (N,14)
        mean_load = np.nanmean(L, axis=0)
        ax.bar(np.arange(len(mean_load)), mean_load, color="C3" if "spike" in grp else "C2")
        ax.set_title(f"{grp}: mean |loading| on eig(-H(z_ref))", fontsize=9)
        ax.set_xlabel("eigenvector index (0=stiffest)")
    axes[0].set_ylabel("mean |loading|")
    fig.suptitle(f"T23 {arm}: C3 direction identity -- unit-Dz loadings on fixed eigenbasis")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_path, dpi=130); plt.close(fig)


# ===========================================================================
# T23 metrics
# ===========================================================================

def t23_metrics(arm, d, man):
    idx = list(np.asarray(d["bounded_idx"])); names = man["bounded"]["names"]
    rs_index = man["bounded"]["rs_index"]; a_rs = man["bounded"]["alpha_rs_index"]

    def wall_stat(sp_z, ca_z):
        out = {}
        for k, nm in zip(idx, names):
            med = float(np.median(np.abs(sp_z[:, k])))
            out[nm] = {"col": int(k), "spike_median_absz": med,
                       "quantile_within_calm": quantile_within(med, np.abs(ca_z[:, k]))}
        return out

    real_wall = wall_stat(d["real_spike_z"], d["real_calm_z"])
    clone_wall = wall_stat(d["clone_spike_z"], d["clone_calm_z"])

    def med_ratio(sp, ca):
        a = np.nanmedian(sp); b = np.nanmedian(ca)
        return float(a / b) if b not in (0.0,) and np.isfinite(b) and b != 0 else float("nan")

    real_stiff = med_ratio(d["real_spike_c_t"], d["real_calm_c_t"])
    clone_stiff = med_ratio(d["clone_spike_c_t"], d["clone_calm_c_t"])

    # registered wall coordinate: new -> Rs; old -> max quantile of {Rs, alpha_Rs}
    def _q(name_col):
        for nm, info in real_wall.items():
            if info["col"] == name_col:
                return info["quantile_within_calm"], nm
        return float("nan"), None
    if arm == "new":
        wall_q, wall_name = _q(rs_index)
    else:
        cands = [c for c in (rs_index, a_rs) if c is not None]
        qs = [(_q(c)[0], _q(c)[1]) for c in cands]
        qs = [(q, nm) for q, nm in qs if np.isfinite(q)]
        wall_q, wall_name = max(qs) if qs else (float("nan"), None)

    clone_wall_q = (clone_wall.get(wall_name, {}) or {}).get("quantile_within_calm", float("nan"))

    verdicts = {
        "P_T23_wall_MET": bool(np.isfinite(wall_q) and wall_q >= WALL_MET),
        "F_T23_wall_FIRES": bool(np.isfinite(wall_q) and WALL_FALSE_LO <= wall_q <= WALL_FALSE_HI),
        "P_T23_stiff_MET": bool(np.isfinite(real_stiff) and real_stiff >= STIFF_MET),
        "F_T23_stiff_FIRES": bool(np.isfinite(real_stiff) and real_stiff < STIFF_FALSE),
        "P_T23_clone_clean": bool(
            np.isfinite(clone_wall_q) and WALL_FALSE_LO <= clone_wall_q <= WALL_FALSE_HI
            and np.isfinite(clone_stiff) and clone_stiff < STIFF_FALSE),
        "note": "all verdicts PROPOSED (UNCERTIFIED); orchestrator adjudicates",
    }
    return {
        "registered_wall_coord": {"name": wall_name, "quantile": wall_q,
                                  "clone_quantile": clone_wall_q},
        "wall_stat_real": real_wall, "wall_stat_clone": clone_wall,
        "stiff_ratio_real": real_stiff, "stiff_ratio_clone": clone_stiff,
        "median_c_real": {"spike": float(np.nanmedian(d["real_spike_c_t"])),
                          "calm": float(np.nanmedian(d["real_calm_c_t"]))},
        "median_c_clone": {"spike": float(np.nanmedian(d["clone_spike_c_t"])),
                           "calm": float(np.nanmedian(d["clone_calm_c_t"]))},
        "verdicts": verdicts,
    }


# ===========================================================================
# T24 plots + stats
# ===========================================================================

def moving_avg(x, w=5):
    x = np.asarray(x, float)
    if len(x) < w:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def run_lengths_above(mask):
    """Return (widths, onset_indices) of consecutive True runs in a 1-D mask."""
    widths, onsets = [], []
    i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            widths.append(j - i); onsets.append(i); i = j
        else:
            i += 1
    return widths, onsets


def segment_stats(c_row, xi_row, factor=C.CENSUS_SPIKE_FACTOR):
    med = float(np.nanmedian(c_row))
    thr = factor * med
    mask = np.asarray(c_row) > thr
    widths, onsets = run_lengths_above(mask)
    spacings = list(np.diff(onsets)) if len(onsets) > 1 else []
    rho = float(spearmanr(c_row, xi_row, nan_policy="omit").correlation)
    rho_sm = float(spearmanr(c_row, moving_avg(xi_row, 5), nan_policy="omit").correlation)
    return {
        "median_c": med, "max_c": float(np.nanmax(c_row)),
        "spike_rate": float(np.mean(mask)), "n_spikes": int(len(widths)),
        "widths": [int(w) for w in widths], "spacings": [int(s) for s in spacings],
        "frac_xi_gt10": float(np.mean(np.asarray(xi_row) > C.XI_SPIKE_THRESH)),
        "rho_c_xi": rho, "rho_c_xi_smooth5": rho_sm,
    }


def plot_t24_segments(arm, d, out_path):
    steps = d["steps"]; c = d["real_c_t"]; xi = d["real_xi"]
    Cn = c.shape[0]
    fig, axes = plt.subplots(Cn, 1, figsize=(9, 1.7 * Cn), sharex=True)
    if Cn == 1:
        axes = [axes]
    for ci in range(Cn):
        ax = axes[ci]; ax2 = ax.twinx()
        ax.plot(steps, c[ci], color="C0", lw=0.8, label="c_t")
        med = np.nanmedian(c[ci]); thr = C.CENSUS_SPIKE_FACTOR * med
        ax.axhline(thr, color="C0", lw=0.5, ls=":")
        sp = np.where(c[ci] > thr)[0]
        ax.plot(steps[sp], c[ci][sp], "o", color="crimson", ms=3)
        ax2.plot(steps, np.clip(xi[ci], 1e-6, None), color="C1", lw=0.6, alpha=0.7)
        ax2.set_yscale("log")
        ax.set_ylabel(f"c ch{ci}", fontsize=7)
        if ci == 0:
            ax.set_title(f"T24 {arm}: per-segment c_t (blue, spikes red) & xi (orange, log)")
    axes[-1].set_xlabel("global results-phase step")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def t24_stats(arm, d):
    steps = d["steps"]
    out = {"arm": arm, "segments": {"real": [], "clone": []}}
    for kind in ("real", "clone"):
        c = d[f"{kind}_c_t"]; xi = d[f"{kind}_xi"]
        for ci in range(c.shape[0]):
            out["segments"][kind].append(segment_stats(c[ci], xi[ci]))
    # pooled (real)
    cr = d["real_c_t"].reshape(-1); xr = d["real_xi"].reshape(-1)
    pooled_rho = float(spearmanr(cr, xr, nan_policy="omit").correlation)
    xr_sm = np.concatenate([moving_avg(d["real_xi"][ci], 5) for ci in range(d["real_xi"].shape[0])])
    pooled_rho_sm = float(spearmanr(cr, xr_sm, nan_policy="omit").correlation)
    pooled_med = float(np.nanmedian(cr))
    pooled_spike_rate = float(np.mean(cr > C.CENSUS_SPIKE_FACTOR * pooled_med))
    frac_xi = float(np.mean(xr > C.XI_SPIKE_THRESH))
    ref_frac = REF_XI_FRAC[arm]
    within_2x = bool(frac_xi > 0 and 0.5 <= (pooled_spike_rate / max(frac_xi, 1e-9)) <= 2.0)
    # clone pooled
    ccl = d["clone_c_t"].reshape(-1); xcl = d["clone_xi"].reshape(-1)
    clone_rho = float(spearmanr(ccl, xcl, nan_policy="omit").correlation)
    clone_med = float(np.nanmedian(ccl))
    clone_spike_rate = float(np.mean(ccl > C.CENSUS_SPIKE_FACTOR * clone_med))

    seg_rho = [s["rho_c_xi"] for s in out["segments"]["real"]]
    n_below = int(np.sum([(np.isfinite(r) and r < T24_RHO_FALSE) for r in seg_rho]))
    out["pooled"] = {
        "rho_c_xi": pooled_rho, "rho_c_xi_smooth5": pooled_rho_sm,
        "median_c": pooled_med, "spike_rate": pooled_spike_rate,
        "frac_xi_gt10": frac_xi, "ref_frac_xi_gt10": ref_frac,
        "spike_rate_within_2x_of_frac_xi": within_2x,
        "n_segments_rho_below_0.15": n_below,
        "clone_rho": clone_rho, "clone_spike_rate": clone_spike_rate,
        "clone_median_c": clone_med,
    }
    return out


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="T23+T24 login-node analysis (no jax)")
    ap.add_argument("--arm", choices=["old", "new", "both"], default="both")
    ap.add_argument("--smoke", action="store_true", help="read *_smoke.npz outputs")
    args = ap.parse_args(argv)
    arms = ["old", "new"] if args.arm == "both" else [args.arm]

    os.makedirs(C.T23_OUT, exist_ok=True)
    os.makedirs(C.T24_OUT, exist_ok=True)
    suf = "_smoke" if args.smoke else ""

    t23_doc = {"experiment": "T23 momentum-space xi accounting -- analysis",
               "status": "proposed (UNCERTIFIED)", "timestamp_utc": _now(),
               "xi_alignment": C.XI_ALIGNMENT, "metric_note": C.METRIC_NOTE,
               "thresholds": {"WALL_MET": WALL_MET, "WALL_FALSE": [WALL_FALSE_LO, WALL_FALSE_HI],
                              "STIFF_MET": STIFF_MET, "STIFF_FALSE": STIFF_FALSE},
               "arms": {}}
    t24_doc = {"experiment": "T24 encounter-geometry census -- analysis",
               "status": "proposed (UNCERTIFIED)", "timestamp_utc": _now(),
               "xi_alignment": C.XI_ALIGNMENT,
               "thresholds": {"T24_RHO_MET": T24_RHO_MET, "T24_RHO_FALSE": T24_RHO_FALSE,
                              "T24_SPIKE_BAND": T24_SPIKE_BAND,
                              "T24_SPIKE_MISMATCH": T24_SPIKE_MISMATCH},
               "arms": {}}

    for arm in arms:
        # ---- T23 ----
        try:
            d, man = load_t23(arm, args.smoke)
            rs_index = man["bounded"]["rs_index"]
            print(f"[analyze:{arm}] T23 plots FIRST ...")
            plot_xi_vs_zRs(arm, rs_index, os.path.join(C.T23_OUT, f"t23_{arm}{suf}_xi_vs_zRs.png"))
            plot_bounded_absz_ecdf(arm, d, man, os.path.join(C.T23_OUT, f"t23_{arm}{suf}_absz_ecdf.png"))
            plot_ct_stability_ecdf(arm, d, os.path.join(C.T23_OUT, f"t23_{arm}{suf}_ct_stability.png"))
            plot_turn_hist(arm, d, os.path.join(C.T23_OUT, f"t23_{arm}{suf}_turn_hist.png"))
            plot_loading_heatmap(arm, d, os.path.join(C.T23_OUT, f"t23_{arm}{suf}_loadings.png"))
            met = t23_metrics(arm, d, man)
            t23_doc["arms"][arm] = met
            v = met["verdicts"]; wq = met["registered_wall_coord"]
            print(f"[analyze:{arm}] T23 wall={wq['name']} q={wq['quantile']:.3f} "
                  f"stiff_ratio={met['stiff_ratio_real']:.3g} "
                  f"P_wall={v['P_T23_wall_MET']} F_wall={v['F_T23_wall_FIRES']} "
                  f"P_stiff={v['P_T23_stiff_MET']} F_stiff={v['F_T23_stiff_FIRES']} "
                  f"clone_clean={v['P_T23_clone_clean']}")
        except FileNotFoundError as e:
            print(f"[analyze:{arm}] T23 npz missing ({e}); skipping T23.")

        # ---- T24 ----
        try:
            d24, man24 = load_t24(arm, args.smoke)
            print(f"[analyze:{arm}] T24 per-segment plots ...")
            plot_t24_segments(arm, d24, os.path.join(C.T24_OUT, f"t24_{arm}{suf}_segments.png"))
            st = t24_stats(arm, d24)
            t24_doc["arms"][arm] = st
            p = st["pooled"]
            print(f"[analyze:{arm}] T24 pooled rho={p['rho_c_xi']:.3f} "
                  f"rho_sm={p['rho_c_xi_smooth5']:.3f} spike_rate={p['spike_rate']:.3f} "
                  f"frac_xi={p['frac_xi_gt10']:.3f} within2x={p['spike_rate_within_2x_of_frac_xi']} "
                  f"clone_rho={p['clone_rho']:.3f}")
        except FileNotFoundError as e:
            print(f"[analyze:{arm}] T24 npz missing ({e}); skipping T24.")

    # ---- F-T24 cross-arm evaluation ----
    if all(a in t24_doc["arms"] for a in arms) and len(arms) == 2:
        below = {a: t24_doc["arms"][a]["pooled"]["n_segments_rho_below_0.15"] for a in arms}
        # spike-rate mismatch factor per arm (max over-/under-shoot vs frac_xi)
        def _mm(a):
            p = t24_doc["arms"][a]["pooled"]
            fr = max(p["frac_xi_gt10"], 1e-9); sr = max(p["spike_rate"], 1e-9)
            return max(sr / fr, fr / sr)
        mismatch = {a: _mm(a) for a in arms}
        f_t24 = bool(all(below[a] >= 6 for a in arms) and all(mismatch[a] > T24_SPIKE_MISMATCH for a in arms))
        t24_doc["F_T24"] = {"n_segments_rho_below_0.15": below,
                            "spike_rate_mismatch_factor": mismatch,
                            "F_T24_FIRES": f_t24,
                            "note": "PROPOSED (UNCERTIFIED)"}

    p23 = os.path.join(C.T23_OUT, f"t23_analysis{suf}.json")
    p24 = os.path.join(C.T24_OUT, f"t24_analysis{suf}.json")
    json.dump(t23_doc, open(p23, "w"), indent=2)
    json.dump(t24_doc, open(p24, "w"), indent=2)
    print(f"[analyze] wrote {p23}\n[analyze] wrote {p24}\n[analyze] done (UNCERTIFIED).")


if __name__ == "__main__":
    main()
