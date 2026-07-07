"""T9 -- Step-size account: do per-step energy-error spikes track high-lambda1?

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T9, approved by the
human 2026-07-02). Chain of reasoning: T6/E1 found the stiffest GN eigenvalue
breathes x22 across the typical set; with ONE global MCLMC step size the per-step
energy error xi should spike wherever the chain visits high-lambda1 regions, and
the adapted eps is set by the worst wall -- a candidate quantitative account of the
ESS deficit. T9 tests it directly against the reference run's own history.

xi CONVENTION (read from src/gigalens_research/inference/mclmc.py):
  In the results phase (mode 0, adaptation OFF) xi is logged by `skip_adapt` at
  mclmc.py line 334:
      xi_val = jnp.square(info.energy_change) / (dim * desired_energy_var) + 1e-8
  i.e. xi = (per-step energy change)^2 / (dim * desired_energy_var) + 1e-8. It is a
  NORMALIZED squared per-step energy error (an EEVPD-style proxy; the additive 1e-8
  is a floor). SAME definition as the step-size adaptation (`step_size_adapt`, lines
  204/219), so burn-in and results xi are directly comparable. dim=22,
  desired_energy_var=0.0005 for this run (manifest). The OLD -1.0 sentinel is GONE
  (the code comment at line 327-333 says skip_adapt now logs the real xi instead of
  the old sentinel); empirically this run's results-phase xi has NO -1 and NO
  non-finite values (nonan mask is all-True over the results phase). A tiny number
  (~104/160000) sit exactly at the 1e-8 floor (energy_change ~ 0, a near-perfect
  step, not a NaN sentinel); we still EXCLUDE any non-finite and any value <=
  1e-8*(1+1e-6) as a safeguard before smoothing, and record the count.

ALIGNMENT: arrays.npz `samples_z` is (8, 20000, 22) = RESULTS ONLY. diagnostics.npz
`xi` is (8, 40000) = burnin(20000) + results(20000); results xi = the LAST 20000
columns. Position index k (results) <-> xi index k (results): xi at results-step k
reflects the (k-1)->k move; the +/-1 ambiguity is immaterial after +/-20-step RMS
smoothing (checkpoint). We index them 1:1.

METHOD: smooth |xi| per chain with a +/-20-step centered RMS window -> xi_s.
Stratified sample ~n_points (chain, step) positions: ~1/3 from the TOP decile of
xi_s, ~1/3 from the MIDDLE (40-60th pct), ~1/3 from the BOTTOM decile; seeded,
spread across chains, with a minimum step-separation of 50 within a chain to reduce
autocorrelation double-counting. At each sampled position compute lambda1(M_zhat)
in the SAME standardized units as T8/E1. Statistics: Spearman rho(lambda1, xi_s)
over all sampled points and within each stratum; median lambda1 top-decile /
bottom-decile ratio. Thresholds restated next to measured values.

The E1 chi^2 render-path gate is re-run at startup on sampled positions -- nothing
downstream is valid without it. jax is imported only inside functions so the module
imports cleanly under a plain conda python for offline smoke tests (--smoke) of the
smoothing / stratified-sampling / Spearman machinery.
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

from e1_fisher_survey import gn_metric, scale_cols, eig_desc, build_jax_ops, TINY

# --- pre-registered constants (T9 checkpoint) ------------------------------
RMS_HALFWIN = 20          # +/-20-step centered RMS smoothing window
N_POINTS_DEFAULT = 300
MIN_STEP_SEP = 50         # min within-chain step separation between sampled points
XI_FLOOR = 1e-8           # additive floor in the xi definition (mclmc.py)
T9_PREDICT_RHO = 0.5      # Spearman rho(lambda1, xi_s) > this => prediction
T9_FALSIFY_RHO = 0.2      # rho < this => account wrong
T9_PREDICT_DECILE_RATIO = 3.0   # median lambda1 top/bottom xi-decile >= this
T9_FALSIFY_DECILE_RATIO = 1.0   # ~1 => account wrong


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pure-numpy machinery (offline-testable)
# ===========================================================================

def rms_smooth(x, halfwin=RMS_HALFWIN):
    """Centered +/-halfwin RMS of a 1-D array (sqrt of the moving mean of x^2),
    edge-truncated (window shrinks at the ends). NaNs in x are ignored per window."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    x2 = x ** 2
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - halfwin); hi = min(n, i + halfwin + 1)
        w = x2[lo:hi]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = np.sqrt(np.mean(w))
    return out


def spearman_rho(a, b):
    """Spearman rank correlation (ties via average ranks), numpy-only. Returns nan
    if <3 finite paired points or zero variance in either ranking."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    ra = _rankdata(a); rb = _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(ra * rb) / denom)


def _rankdata(x):
    """Average-rank of x (1-based), ties averaged (matches scipy.stats.rankdata)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0          # average of 1-based positions i..j
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def valid_mask(xi, floor=XI_FLOOR):
    """Boolean mask of usable xi entries: finite AND above the additive floor
    (floor-value == NaN-zeroed / near-zero energy change; excluded per convention)."""
    xi = np.asarray(xi, dtype=np.float64)
    return np.isfinite(xi) & (xi > floor * (1.0 + 1e-6))


def stratified_sample(xi_s, valid, n_points, rng, min_sep=MIN_STEP_SEP):
    """Stratified (chain, step) sample from a (C, S) smoothed-xi grid.

    ~n_points/3 from the TOP decile of pooled xi_s, ~n_points/3 from the MIDDLE
    (40-60th pct), ~n_points/3 from the BOTTOM decile. Only entries with valid=True
    are eligible. Points are drawn spread across chains with a within-chain minimum
    step separation `min_sep` (greedy, seeded). Returns a list of dicts with
    chain, step, xi_s, stratum."""
    C, S = xi_s.shape
    pooled = xi_s[valid]
    if pooled.size < 30:
        raise ValueError(f"[T9] too few valid xi entries ({pooled.size}) to stratify.")
    p10, p40, p60, p90 = np.percentile(pooled, [10, 40, 60, 90])
    strata = {
        "top": lambda v: v >= p90,
        "middle": lambda v: (v >= p40) & (v <= p60),
        "bottom": lambda v: v <= p10,
    }
    per = int(round(n_points / 3.0))
    chosen = []
    # min_sep is enforced GLOBALLY across strata within a chain (shared record),
    # so no two sampled steps in the same chain fall within min_sep of each other.
    taken_by_chain = {c: [] for c in range(C)}
    for sname, pred in strata.items():
        # candidate (chain, step) indices in this stratum, valid only
        elig = valid & pred(xi_s)
        cc, ss = np.where(elig)
        if cc.size == 0:
            continue
        order = rng.permutation(cc.size)
        got = 0
        # round-robin over a shuffled candidate list, enforcing min_sep per chain
        for idx in order:
            if got >= per:
                break
            c = int(cc[idx]); s = int(ss[idx])
            if all(abs(s - s0) >= min_sep for s0 in taken_by_chain[c]):
                taken_by_chain[c].append(s)
                chosen.append({"chain": c, "step": s,
                               "xi_s": float(xi_s[c, s]), "stratum": sname})
                got += 1
        if got < per:
            print(f"[T9] stratum {sname}: only {got}/{per} points met min_sep={min_sep}")
    return chosen


# ===========================================================================
# chi^2 render-path gate (replicated from E1 Task A -- E1 does not factor it out)
# ===========================================================================

def chi2_gate(ops, z_points, tag="T9"):
    """Recompute chi^2 from the render path and reconcile with log_prob's reduced
    chi^2 aux. Convention IDENTICAL to E1: chi2_from_aux = aux * event_size. RAISE
    if relative match > 1e-6."""
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
# Offline smoke tests (numpy only)
# ===========================================================================

def smoke_spearman(verbose=True):
    """Plant a known monotone (rank) relation lambda1 ~ f(xi_s) + noise and recover
    a strong Spearman rho; check a shuffled control gives ~0."""
    rng = np.random.default_rng(3)
    n = 400
    x = rng.uniform(0, 1, n)
    y = np.exp(2.5 * x) + 0.25 * rng.standard_normal(n)   # monotone in x + noise
    rho = spearman_rho(x, y)
    rho_shuf = spearman_rho(x, rng.permutation(y))
    # cross-check against scipy if available
    scipy_rho = None
    try:
        from scipy.stats import spearmanr
        scipy_rho = float(spearmanr(x, y).statistic)
    except Exception:
        pass
    ok = rho > 0.85 and abs(rho_shuf) < 0.2
    if scipy_rho is not None:
        ok = ok and abs(rho - scipy_rho) < 1e-9
    if verbose:
        print(f"[smoke] spearman: planted rho={rho:.3f} (>0.85), shuffled "
              f"{rho_shuf:.3f} (~0)"
              + (f", scipy {scipy_rho:.3f} (match {abs(rho-scipy_rho):.1e})"
                 if scipy_rho is not None else "")
              + f" -> {'PASS' if ok else 'FAIL'}")
    return {"rho": rho, "rho_shuffled": rho_shuf, "scipy_rho": scipy_rho,
            "pass": bool(ok)}


def smoke_rms(verbose=True):
    """RMS window on a known signal: constant c -> RMS c; a lone spike is spread to
    a +/-halfwin plateau of the right height."""
    x = np.full(200, 2.0)
    s = rms_smooth(x, halfwin=5)
    ok_const = np.allclose(s, 2.0)
    x2 = np.zeros(200); x2[100] = 10.0
    s2 = rms_smooth(x2, halfwin=5)
    # window of 11 samples around center: RMS = sqrt(100/11)
    expected = np.sqrt(100.0 / 11.0)
    ok_spike = abs(s2[100] - expected) < 1e-9 and s2[100 - 5] > 0 and s2[100 - 6] == 0
    ok = ok_const and ok_spike
    if verbose:
        print(f"[smoke] rms: constant->{s[100]:.3f} (2.000); spike center "
              f"{s2[100]:.4f} (expect {expected:.4f}) -> {'PASS' if ok else 'FAIL'}")
    return {"const": float(s[100]), "spike": float(s2[100]),
            "spike_expected": float(expected), "pass": bool(ok)}


def smoke_stratify(verbose=True):
    """Stratified sampler on a synthetic (C,S) xi_s grid: strata means ordered
    top>middle>bottom; min-sep respected; all points valid."""
    rng = np.random.default_rng(5)
    C, S = 8, 20000
    xi_s = np.abs(rng.standard_normal((C, S))) + 0.01
    xi_s[:, ::500] += 20.0                       # planted spikes -> top decile
    valid = np.ones((C, S), dtype=bool)
    pts = stratified_sample(xi_s, valid, 300, rng, min_sep=50)
    by = {"top": [], "middle": [], "bottom": []}
    seps_ok = True
    from collections import defaultdict
    perchain = defaultdict(list)
    for p in pts:
        by[p["stratum"]].append(p["xi_s"]); perchain[p["chain"]].append(p["step"])
    for c, steps in perchain.items():
        ss = sorted(steps)
        seps_ok = seps_ok and all(ss[i + 1] - ss[i] >= 50 for i in range(len(ss) - 1))
    mt = np.mean(by["top"]); mm = np.mean(by["middle"]); mb = np.mean(by["bottom"])
    ok = mt > mm > mb and seps_ok and len(pts) > 200
    if verbose:
        print(f"[smoke] stratify: n={len(pts)} means top={mt:.2f} mid={mm:.2f} "
              f"bot={mb:.2f} min_sep_ok={seps_ok} -> {'PASS' if ok else 'FAIL'}")
    return {"n": len(pts), "mean_top": float(mt), "mean_middle": float(mm),
            "mean_bottom": float(mb), "min_sep_ok": bool(seps_ok), "pass": bool(ok)}


def run_smoke():
    print("=== T9 offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_rms(); b = smoke_spearman(); c = smoke_stratify()
    allpass = a["pass"] and b["pass"] and c["pass"]
    print(f"[smoke] overall: {'PASS' if allpass else 'FAIL'}")
    return {"rms": a, "spearman": b, "stratify": c, "pass": bool(allpass)}


# ===========================================================================
# Plots
# ===========================================================================

def plot_scatter(records, rho_all, out_path):
    """log lambda1 vs log xi_s, colored by stratum."""
    colors = {"top": "#8a3b3b", "middle": "#c08a3b", "bottom": "#3b6a8a"}
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for sname, col in colors.items():
        xs = [r["xi_s"] for r in records if r["stratum"] == sname]
        ys = [r["lambda1"] for r in records if r["stratum"] == sname]
        ax.scatter(xs, ys, s=22, c=col, alpha=0.6, label=f"{sname} (n={len(xs)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("smoothed |xi|  (+/-20-step RMS energy-error proxy)")
    ax.set_ylabel("lambda1(M_zhat)  (top GN eigenvalue, standardized)")
    ax.set_title(f"T9 lambda1 vs energy-error xi_s  (Spearman rho={rho_all:.3f}) "
                 "-- PROPOSED (UNCERTIFIED)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_traces(xi_s, records, out_path, max_chains=8):
    """Per-chain xi_s trace strip with sampled points marked, colored by lambda1."""
    C, S = xi_s.shape
    nshow = min(C, max_chains)
    fig, axes = plt.subplots(nshow, 1, figsize=(12, 1.7 * nshow), squeeze=False,
                             sharex=True)
    lam_all = np.array([r["lambda1"] for r in records if np.isfinite(r["lambda1"])])
    vmin = np.log10(lam_all.min()) if lam_all.size else 0.0
    vmax = np.log10(lam_all.max()) if lam_all.size else 1.0
    sc = None
    for c in range(nshow):
        ax = axes[c][0]
        ax.plot(np.arange(S), xi_s[c], color="0.7", lw=0.5)
        ax.set_yscale("log")
        rc = [r for r in records if r["chain"] == c and np.isfinite(r["lambda1"])]
        if rc:
            xs = [r["step"] for r in rc]; ys = [r["xi_s"] for r in rc]
            cs = [np.log10(r["lambda1"]) for r in rc]
            sc = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=vmin, vmax=vmax,
                            s=28, edgecolor="k", linewidth=0.3, zorder=3)
        ax.set_ylabel(f"chain {c}", fontsize=8); ax.tick_params(labelsize=7)
    axes[-1][0].set_xlabel("results-phase step", fontsize=9)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.01)
        cb.set_label("log10 lambda1(M_zhat)", fontsize=9)
    fig.suptitle("T9 per-chain xi_s traces with sampled points (color=lambda1) "
                 "-- PROPOSED (UNCERTIFIED)", fontsize=12)
    fig.savefig(out_path, dpi=130); plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T9 xi-vs-lambda1 (energy-error spikes "
                                            "track high-lambda1?)")
    p.add_argument("--data-dir")
    p.add_argument("--run-dir", help="MCLMC run dir with arrays.npz + diagnostics.npz")
    p.add_argument("--out-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--n-points", type=int, default=N_POINTS_DEFAULT)
    p.add_argument("--smoke", action="store_true",
                   help="run the offline numpy-only smoke tests and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        res = run_smoke()
        if not res["pass"]:
            raise SystemExit("[T9] smoke tests FAILED")
        return

    for req in ("data_dir", "run_dir", "out_dir", "seed"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    os.makedirs(args.out_dir, exist_ok=True)

    from common import assert_x64, load_target
    assert_x64()

    # --- load run arrays ----------------------------------------------------
    arrays = np.load(os.path.join(args.run_dir, "arrays.npz"))
    diag = np.load(os.path.join(args.run_dir, "diagnostics.npz"))
    samples_z = np.asarray(arrays["samples_z"], dtype=np.float64)   # (C, R, dim) results
    xi_full = np.asarray(diag["xi"], dtype=np.float64)              # (C, burnin+results)
    C, R, dim = samples_z.shape
    if xi_full.shape[0] != C:
        raise ValueError(f"xi chains {xi_full.shape[0]} != positions chains {C}")
    total = xi_full.shape[1]
    if total < R:
        raise ValueError(f"xi length {total} < results {R}")
    xi_res = xi_full[:, -R:]                                        # results-phase xi
    n_burn = total - R
    print(f"[T9] positions {samples_z.shape} (results-only); xi {xi_full.shape} "
          f"(burnin={n_burn}+results={R}); using last {R} xi columns")

    # xi validity + floor accounting (convention documented in the docstring/JSON)
    valid = valid_mask(xi_res)
    n_floor = int(np.sum(np.asarray(xi_res) <= XI_FLOOR * (1.0 + 1e-6)))
    n_nonfin = int(np.sum(~np.isfinite(xi_res)))
    print(f"[T9] xi results: min={np.nanmin(xi_res):.3e} max={np.nanmax(xi_res):.3e} "
          f"median={np.nanmedian(xi_res):.3e}; floor/sentinel excluded={n_floor}, "
          f"non-finite excluded={n_nonfin}, valid frac={valid.mean():.4f}")

    # --- smooth per chain ---------------------------------------------------
    xi_s = np.vstack([rms_smooth(xi_res[c], RMS_HALFWIN) for c in range(C)])
    # validity for smoothed grid: keep entries whose center xi was valid AND whose
    # smoothed value is finite (a fully-invalid window yields nan).
    valid_s = valid & np.isfinite(xi_s)

    # --- stratified sample --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    chosen = stratified_sample(xi_s, valid_s, args.n_points, rng, MIN_STEP_SEP)
    print(f"[T9] stratified sample: {len(chosen)} points "
          f"(top={sum(p['stratum']=='top' for p in chosen)}, "
          f"middle={sum(p['stratum']=='middle' for p in chosen)}, "
          f"bottom={sum(p['stratum']=='bottom' for p in chosen)})")

    # --- standardization: SAME as E1/T8 (pooled results positions) ----------
    flat = samples_z.reshape(-1, dim)
    std_z = np.std(flat, axis=0, ddof=1)

    # --- target + render ops + chi^2 gate -----------------------------------
    model_seq, qz, z_center, dim2, param_names = load_target(args.data_dir)
    if dim2 != dim:
        raise ValueError(f"data-dir dim {dim2} != run dim {dim}")
    ops = build_jax_ops(model_seq, param_names)
    W = ops["W"]
    gate_pts = [samples_z[chosen[i]["chain"], chosen[i]["step"]]
                for i in range(min(3, len(chosen)))]
    chi2_checks, worst_rel = chi2_gate(ops, gate_pts, tag="T9")

    # --- lambda1 at each sampled position -----------------------------------
    records = []
    for p in chosen:
        z = samples_z[p["chain"], p["step"]]
        Jz = np.asarray(ops["jac_render"](z), dtype=np.float64)
        M_zhat = gn_metric(scale_cols(Jz, std_z), W)
        w, _ = eig_desc(M_zhat)
        rec = dict(p)
        rec["lambda1"] = float(w[0])
        rec["lambda2"] = float(w[1])
        records.append(rec)

    lam = np.array([r["lambda1"] for r in records])
    xis = np.array([r["xi_s"] for r in records])
    strat = np.array([r["stratum"] for r in records])

    # --- statistics ---------------------------------------------------------
    rho_all = spearman_rho(lam, xis)
    rho_by = {s: spearman_rho(lam[strat == s], xis[strat == s])
              for s in ("top", "middle", "bottom")}
    med_top = float(np.median(lam[strat == "top"])) if np.any(strat == "top") else float("nan")
    med_bot = float(np.median(lam[strat == "bottom"])) if np.any(strat == "bottom") else float("nan")
    decile_ratio = med_top / med_bot if med_bot > 0 else float("inf")
    print(f"[T9] Spearman rho(lambda1, xi_s) all = {rho_all:.3f}  "
          f"(top {rho_by['top']:.3f} mid {rho_by['middle']:.3f} bot {rho_by['bottom']:.3f})")
    print(f"[T9] median lambda1 top-decile/bottom-decile = {med_top:.3e}/{med_bot:.3e} "
          f"= {decile_ratio:.2f}x")

    # --- plots --------------------------------------------------------------
    scatter_path = os.path.join(args.out_dir, "t9_scatter_lambda1_vs_xi.png")
    trace_path = os.path.join(args.out_dir, "t9_xi_traces.png")
    plot_scatter(records, rho_all, scatter_path)
    plot_traces(xi_s, records, trace_path)

    # --- JSON ---------------------------------------------------------------
    doc = {
        "experiment": "T9 -- do energy-error spikes track high-lambda1 regions?",
        "status": "proposed (UNCERTIFIED) -- grader inspects artifacts, not this summary",
        "timestamp_utc": _now(),
        "args": {k: v for k, v in vars(args).items()},
        "dim": dim, "param_names": param_names,
        "xi_convention": {
            "source": "src/gigalens_research/inference/mclmc.py line 334 (skip_adapt, "
                      "results phase) == line 204/219 (step_size_adapt)",
            "formula": "xi = energy_change^2 / (dim * desired_energy_var) + 1e-8",
            "meaning": "normalized squared per-step energy error (EEVPD-style proxy); "
                       "additive 1e-8 floor",
            "dim": dim, "desired_energy_var": 0.0005,
            "sentinel_note": "OLD -1.0 sentinel removed (mclmc.py comment lines 327-333); "
                             "results-phase xi has no -1 and no non-finite (nonan all-True). "
                             "Values at the 1e-8 floor (~near-zero energy change) are "
                             "excluded as a safeguard.",
            "floor_excluded_count": n_floor,
            "nonfinite_excluded_count": n_nonfin,
            "valid_fraction": float(valid.mean()),
        },
        "alignment": {
            "positions": "arrays.npz samples_z (C, R, dim) = RESULTS ONLY",
            "xi": f"diagnostics.npz xi (C, {total}) = burnin({n_burn})+results({R}); "
                  f"results xi = last {R} columns",
            "index_map": "position index k <-> results-xi index k (1:1); xi[k] reflects "
                         "the (k-1)->k move; +/-1 ambiguity immaterial after RMS smoothing",
        },
        "standardization": {
            "note": "SAME as E1/T8: zhat=z/std_z (per-coordinate ddof=1 over pooled "
                    "results positions); M_zhat = gn_metric(scale_cols(J_z, std_z), W); "
                    "lambda1 = top eigenvalue of M_zhat.",
            "std_z": std_z.tolist(),
        },
        "smoothing": {"method": "centered +/-%d-step RMS of |xi| per chain "
                                "(edge-truncated)" % RMS_HALFWIN},
        "sampling": {"n_requested": args.n_points, "n_used": len(records),
                     "strata": "top decile / 40-60th pct / bottom decile of pooled xi_s",
                     "min_step_separation": MIN_STEP_SEP,
                     "n_top": int(np.sum(strat == "top")),
                     "n_middle": int(np.sum(strat == "middle")),
                     "n_bottom": int(np.sum(strat == "bottom"))},
        "render_path_chi2_gate": {"checks": chi2_checks, "worst_relative_error": worst_rel},
        "records": records,
        "statistics": {
            "spearman_rho_all": rho_all,
            "spearman_rho_by_stratum": rho_by,
            "median_lambda1_top_decile": med_top,
            "median_lambda1_bottom_decile": med_bot,
            "decile_ratio_top_over_bottom": decile_ratio,
        },
        "prediction": {
            "text": f"Spearman rho > {T9_PREDICT_RHO:g} AND median lambda1 top/bottom "
                    f"xi-decile >= {T9_PREDICT_DECILE_RATIO:g}x",
            "rho_threshold": T9_PREDICT_RHO,
            "decile_ratio_threshold": T9_PREDICT_DECILE_RATIO,
            "measured_rho": rho_all, "measured_decile_ratio": decile_ratio,
        },
        "falsifier": {
            "text": f"rho < {T9_FALSIFY_RHO:g} OR decile ratio ~ {T9_FALSIFY_DECILE_RATIO:g} "
                    "=> the step-size/energy-error account is wrong",
            "rho_threshold": T9_FALSIFY_RHO,
            "decile_ratio_threshold": T9_FALSIFY_DECILE_RATIO,
        },
        "verdict_fields": {
            "T9_xi_tracks_lambda1": "proposed (UNCERTIFIED): compare spearman_rho_all "
                "and decile_ratio_top_over_bottom to the prediction/falsifier; read the "
                "scatter + trace plots.",
            "note": "this script does NOT adjudicate; a grader inspects plots + numbers.",
        },
        "plots": {"scatter": os.path.basename(scatter_path),
                  "traces": os.path.basename(trace_path)},
    }
    json_path = os.path.join(args.out_dir, "t9_results.json")
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[T9] wrote {json_path}, {scatter_path}, {trace_path}")

    print("\n=== T9 summary (PROPOSED / UNCERTIFIED) ===")
    print(f"  chi^2 render gate: PASSED, worst rel {worst_rel:.3e}")
    print(f"  n points = {len(records)} (top/mid/bot = "
          f"{int(np.sum(strat=='top'))}/{int(np.sum(strat=='middle'))}/"
          f"{int(np.sum(strat=='bottom'))})")
    print(f"  [T9] Spearman rho(lambda1, xi_s) = {rho_all:.3f} "
          f"(predict > {T9_PREDICT_RHO:g}; falsify < {T9_FALSIFY_RHO:g})")
    print(f"  [T9] median lambda1 top/bottom decile ratio = {decile_ratio:.2f}x "
          f"(predict >= {T9_PREDICT_DECILE_RATIO:g}; falsify ~ {T9_FALSIFY_DECILE_RATIO:g})")
    print("[T9] done.")


if __name__ == "__main__":
    main()
