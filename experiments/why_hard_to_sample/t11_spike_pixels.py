"""T11 -- Render-space spike localization: which pixels carry the stiffness?

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T11, approved by the
human 2026-07-02). Consumes T10's spike_list.json (top ~12 on-ridge lambda1 spikes
+ 12 matched same-segment median-lambda1 baselines) and dissects each point in
IMAGE space. Cause hypothesis: spike stiffness is carried by a SMALL set of image
pixels -- caustic-adjacent arc features whose positions respond violently to the
gamma-shear-source combination -- rather than by the whole image.

At each point z = samples_z[chain, step]:
  J        = jacfwd(render)(z)                       # (6400, 22), raw-z Jacobian
  J_zhat   = scale_cols(J, std_z)                    # standardized-units Jacobian
  M_zhat   = gn_metric(J_zhat, W) = J_zhat^T W J_zhat # SAME metric as E1/T8/T9/T10
  (lambda1, v1)                                       # top eigenpair of M_zhat
  c_i      = sqrt(W_i) * (J_zhat v1)_i               # pixel contribution field
SELF-CHECK: sum_i c_i^2 = v1^T M_zhat v1 = lambda1 (v1 unit eigenvector). We ASSERT
this to 1e-8 relative -- it certifies c is consistent with the metric M.

Localization metrics per point:
  L  = share of lambda1 in the top 1% of pixels (64 of 6400) -- the localization frac
  PR = (sum c^2)^2 / (N * sum c^4)  -- participation ratio (1 => uniform, ~1 => one px)
  centroid + spread of the c^2 map (image-plane pixel coordinates)
  v1 loadings with param NAMES (sorted-key order) -- same gamma-shear-src family as
     E1, or spike-specific? Plus |cos| of the spike's |v1| with E1's mean |loading|
     stiff-family direction (from e1_results.json if available; else loadings only).

PREDICTIONS (restated next to measured): L_spike >= 50% and L_spike/L_baseline >= 3.
FALSIFIER: spike localization ~ baseline (ratio <~ 1.5) => spikes are a global-image
  effect; the caustic story is wrong.

The chi^2 render-path gate (E1 Task A) is replicated VERBATIM and re-run at startup.
jax is imported only inside functions so the module imports cleanly under a plain
conda python for offline smoke tests (--smoke) of the localization metrics (a delta
c^2 map => L=1, PR=1/N; a uniform map => L=0.01, PR=1).
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
from matplotlib.colors import LogNorm

from e1_fisher_survey import gn_metric, scale_cols, eig_desc, build_jax_ops, TINY

# --- pre-registered constants (T11 checkpoint) -----------------------------
TOP_FRACTION = 0.01          # localization L = share of lambda1 in top 1% of pixels
SELF_CHECK_RTOL = 1e-8       # |sum c^2 - lambda1| / lambda1 must be below this
T11_PREDICT_L_SPIKE = 0.50   # L_spike >= 50%
T11_PREDICT_RATIO = 3.0      # L_spike / L_baseline >= 3
T11_FALSIFY_RATIO = 1.5      # ratio <~ 1.5 => spikes are a global-image effect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pure-numpy localization machinery (offline-testable)
# ===========================================================================

def localization(c2, top_fraction=TOP_FRACTION):
    """Localization metrics of a pixel contribution-squared field c2 (1-D, len N).

    L  = fraction of sum(c2) held by the top ceil(top_fraction*N) pixels.
    PR = participation ratio (sum c2)^2 / (N * sum c4). PR in [1/N, 1]:
         1   => perfectly uniform, ~1/N => a single pixel.
    Returns dict: L, n_top, PR, PR_inv (1/PR, ~ number of dominant pixels), total."""
    c2 = np.asarray(c2, dtype=np.float64).ravel()
    N = c2.size
    total = float(np.sum(c2))
    n_top = int(np.ceil(top_fraction * N))
    if total <= 0:
        return {"L": float("nan"), "n_top": n_top, "PR": float("nan"),
                "PR_inv": float("nan"), "total": total}
    top_sum = float(np.sum(np.sort(c2)[::-1][:n_top]))
    L = top_sum / total
    sum4 = float(np.sum(c2 ** 2))
    PR = (total ** 2) / (N * sum4) if sum4 > 0 else float("nan")
    return {"L": float(L), "n_top": n_top, "PR": float(PR),
            "PR_inv": float(1.0 / PR) if PR > 0 else float("nan"),
            "total": total}


def centroid_spread(c2, side):
    """Intensity-weighted centroid (col=x, row=y) and RMS radial spread (pixels) of
    a c^2 field reshaped to (side, side)."""
    c2 = np.asarray(c2, dtype=np.float64).reshape(side, side)
    w = c2
    tot = float(np.sum(w))
    ys, xs = np.mgrid[0:side, 0:side]
    if tot <= 0:
        return {"centroid_x": float("nan"), "centroid_y": float("nan"),
                "spread_px": float("nan")}
    cx = float(np.sum(w * xs) / tot)
    cy = float(np.sum(w * ys) / tot)
    spread = float(np.sqrt(np.sum(w * ((xs - cx) ** 2 + (ys - cy) ** 2)) / tot))
    return {"centroid_x": cx, "centroid_y": cy, "spread_px": spread}


def family_reference_from_e1(e1_json_path, dim):
    """Build E1's mean |loading| stiff-family direction (unit 22-vector) from the
    per-point signed top-8 loadings stored in e1_results.json. Returns (ref, names)
    or (None, None) if the file is unavailable/malformed.

    NOTE: E1's JSON stores only the top-8 |loadings| per point, so this reference has
    top-8 support per point averaged over the 32 points -- an APPROXIMATION of the
    full mean eigenvector, adequate for a param-FAMILY cosine (sign-robust: we use
    |loading|). Documented as such in the output."""
    if not e1_json_path or not os.path.isfile(e1_json_path):
        return None, None
    try:
        doc = json.load(open(e1_json_path))
        pts = doc.get("iid_points", [])
        names = doc.get("param_names")
        if not pts or names is None or len(names) != dim:
            return None, None
        acc = np.zeros(dim, dtype=np.float64)
        for p in pts:
            for ld in p.get("top_eigenvector_loadings", []):
                acc[int(ld["index"])] += abs(float(ld["loading"]))
        acc /= max(len(pts), 1)
        nrm = np.linalg.norm(acc)
        if nrm == 0:
            return None, None
        return acc / nrm, names
    except Exception:
        return None, None


def abs_cosine(u, v):
    """|u.v| / (|u||v|); nan if either is zero."""
    u = np.asarray(u, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    d = np.linalg.norm(u) * np.linalg.norm(v)
    if d == 0:
        return float("nan")
    return float(abs(u @ v) / d)


# ===========================================================================
# chi^2 render-path gate (replicated from E1 Task A -- E1 does not factor it out)
# ===========================================================================

def chi2_gate(ops, z_points, tag="T11"):
    """Recompute chi^2 from the render path and reconcile with log_prob's reduced
    chi^2 aux. Convention IDENTICAL to E1/T8/T9/T10: chi2_from_aux = aux*event_size.
    RAISE if relative match > 1e-6 (a wrong render invalidates all downstream)."""
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

def smoke_localization(verbose=True):
    """Delta map => L=1, PR=1/N; uniform map => L=top_frac, PR=1; plus centroid."""
    N = 6400; side = 80
    # delta
    c2 = np.zeros(N); c2[1234] = 7.0
    loc_d = localization(c2)
    ok_delta = abs(loc_d["L"] - 1.0) < 1e-12 and abs(loc_d["PR"] - 1.0 / N) < 1e-12
    # uniform
    c2u = np.full(N, 3.0)
    loc_u = localization(c2u)
    ok_uni = abs(loc_u["L"] - 64.0 / N) < 1e-12 and abs(loc_u["PR"] - 1.0) < 1e-12
    # centroid of a delta at (row=10, col=20)
    c2c = np.zeros((side, side)); c2c[10, 20] = 5.0
    cs = centroid_spread(c2c.ravel(), side)
    ok_cen = (abs(cs["centroid_x"] - 20) < 1e-9 and abs(cs["centroid_y"] - 10) < 1e-9
              and abs(cs["spread_px"]) < 1e-9)
    ok = ok_delta and ok_uni and ok_cen
    if verbose:
        print(f"[smoke] localization: delta L={loc_d['L']:.6f} (1) PR={loc_d['PR']:.3e} "
              f"(1/N={1.0/N:.3e}); uniform L={loc_u['L']:.6f} (0.01) PR={loc_u['PR']:.6f} "
              f"(1); centroid=({cs['centroid_x']:.1f},{cs['centroid_y']:.1f}) spread="
              f"{cs['spread_px']:.2e} -> {'PASS' if ok else 'FAIL'}")
    return {"delta": loc_d, "uniform": loc_u, "centroid": cs, "pass": bool(ok)}


def smoke_selfcheck(verbose=True):
    """Synthetic J,W,v: c=sqrt(W)*(J v), sum c^2 must equal v^T (J^T W J) v exactly."""
    rng = np.random.default_rng(7)
    m, n = 400, 6
    J = rng.standard_normal((m, n))
    W = np.abs(rng.standard_normal(m)) + 0.1
    M = gn_metric(J, W)
    w, V = eig_desc(M)
    v1 = V[:, 0]
    c = np.sqrt(W) * (J @ v1)
    sumc2 = float(np.sum(c ** 2))
    rel = abs(sumc2 - w[0]) / w[0]
    ok = rel < 1e-12
    if verbose:
        print(f"[smoke] selfcheck: sum c^2={sumc2:.6e} lambda1={w[0]:.6e} rel={rel:.2e} "
              f"-> {'PASS' if ok else 'FAIL'}")
    return {"sum_c2": sumc2, "lambda1": float(w[0]), "rel": rel, "pass": bool(ok)}


def run_smoke():
    print("=== T11 offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_localization(); b = smoke_selfcheck()
    allpass = a["pass"] and b["pass"]
    print(f"[smoke] overall: {'PASS' if allpass else 'FAIL'}")
    return {"localization": a, "selfcheck": b, "pass": bool(allpass)}


# ===========================================================================
# Per-point pixel dissection
# ===========================================================================

def dissect_point(z, ops, W, std_z, side, family_ref=None, param_names=None):
    """Compute the c^2 pixel field + localization metrics + v1 loadings at one z."""
    Jz = np.asarray(ops["jac_render"](z), dtype=np.float64)        # (6400, 22)
    Jzhat = scale_cols(Jz, std_z)                                  # standardized
    M = gn_metric(Jzhat, W)
    w, V = eig_desc(M)
    lam1 = float(w[0]); v1 = V[:, 0]
    c = np.sqrt(W) * (Jzhat @ v1)                                  # (6400,)
    c2 = c ** 2
    sumc2 = float(np.sum(c2))
    rel = abs(sumc2 - lam1) / max(abs(lam1), TINY)
    if rel > SELF_CHECK_RTOL:
        raise RuntimeError(
            f"[T11] self-check FAILED: sum c^2={sumc2:.6e} != lambda1={lam1:.6e} "
            f"(rel={rel:.2e} > {SELF_CHECK_RTOL:g}). c is inconsistent with M -- STOP.")
    loc = localization(c2)
    cs = centroid_spread(c2, side)
    order = np.argsort(-np.abs(v1))
    loadings = [{"param": (param_names[j] if param_names else str(j)),
                 "index": int(j), "loading": float(v1[j])} for j in order[:8]]
    fam_cos = abs_cosine(np.abs(v1), family_ref) if family_ref is not None else None
    return {
        "lambda1": lam1, "sum_c2": sumc2, "selfcheck_rel": rel,
        "L": loc["L"], "n_top": loc["n_top"], "PR": loc["PR"], "PR_inv": loc["PR_inv"],
        "centroid_x": cs["centroid_x"], "centroid_y": cs["centroid_y"],
        "spread_px": cs["spread_px"],
        "top_eigenvector_loadings": loadings,
        "e1_family_abs_cosine": fam_cos,
        "_c2": c2,                                # kept for plotting (not in JSON)
    }


# ===========================================================================
# Plots
# ===========================================================================

def _plot_c2_grid(points, dissections, ops, side, title, out_path):
    """<=4x3 panels: c^2 map per point (log color) with the model image at that z as
    a faint contour overlay; arc structure should be visible."""
    n = len(points)
    ncol = 4
    nrow = int(np.ceil(n / ncol)) if n else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow),
                             squeeze=False)
    for i, (pt, dd) in enumerate(zip(points, dissections)):
        ax = axes[i // ncol][i % ncol]
        c2 = dd["_c2"].reshape(side, side)
        vmax = float(c2.max())
        vmin = vmax * 1e-4 if vmax > 0 else 1e-12
        c2disp = np.clip(c2, vmin, None)
        im = ax.imshow(c2disp, origin="lower", cmap="magma",
                       norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
        model = np.asarray(ops["render"](
            np.asarray(pt["_z"], dtype=np.float64)), dtype=np.float64).reshape(side, side)
        try:
            ax.contour(model, levels=5, colors="cyan", alpha=0.35, linewidths=0.5)
        except Exception:
            pass
        ax.set_title(f"seg{pt['segment']} ch{pt['chain']} s{pt['step']}\n"
                     f"L={dd['L']:.2f} PR={dd['PR']:.3f} l1={dd['lambda1']:.1e}",
                     fontsize=7)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_paired_dots(pairs, out_path):
    """1x2: L and PR paired dot plot -- each spike connected to its matched baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, key, name, hline in (
            (axes[0], "L", "localization fraction L (share of lambda1 in top 1% px)",
             T11_PREDICT_L_SPIKE),
            (axes[1], "PR", "participation ratio PR (lower = more localized)", None)):
        for p in pairs:
            sb = p["baseline"][key]; sp = p["spike"][key]
            ax.plot([0, 1], [sb, sp], "-", color="0.6", lw=0.8, zorder=1)
            ax.scatter([0], [sb], color="#3b6a8a", s=32, zorder=2)
            ax.scatter([1], [sp], color="#8a3b3b", s=32, zorder=2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["baseline", "spike"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel(name, fontsize=9)
        if hline is not None:
            ax.axhline(hline, color="purple", ls="--", lw=1.0,
                       label=f"predict L_spike >= {hline:.0%}")
            ax.legend(fontsize=8)
        ax.set_title(key, fontsize=11)
    fig.suptitle("T11 spike vs matched-baseline localization -- PROPOSED (UNCERTIFIED)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T11 render-space spike localization")
    p.add_argument("--data-dir")
    p.add_argument("--run-dir", help="MCLMC run dir with arrays.npz (for z lookup)")
    p.add_argument("--spike-list", help="path to T10 spike_list.json")
    p.add_argument("--out-dir")
    p.add_argument("--e1-json", default=None,
                   help="optional e1_results.json for the family-cosine reference; "
                        "auto-detected as ../e1/e1_results.json if omitted")
    p.add_argument("--smoke", action="store_true",
                   help="run the offline numpy-only smoke tests and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        res = run_smoke()
        if not res["pass"]:
            raise SystemExit("[T11] smoke tests FAILED")
        return

    for req in ("data_dir", "run_dir", "spike_list", "out_dir"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    os.makedirs(args.out_dir, exist_ok=True)

    from common import assert_x64, load_target
    assert_x64()

    # --- inputs -------------------------------------------------------------
    sl = json.load(open(args.spike_list))
    spikes = sl["spikes"]; baselines = sl["baselines"]
    print(f"[T11] spike_list: {len(spikes)} spikes + {len(baselines)} baselines "
          f"(spike_factor={sl.get('spike_factor')})")

    arrays = np.load(os.path.join(args.run_dir, "arrays.npz"))
    samples_z = np.asarray(arrays["samples_z"], dtype=np.float64)   # (C, R, dim)
    C, Rr, dim = samples_z.shape
    flat = samples_z.reshape(-1, dim)
    std_z = np.std(flat, axis=0, ddof=1)                           # SAME as T10
    side = int(round(np.sqrt(6400)))                               # provisional; refined below

    model_seq, qz, z_center, dim2, param_names = load_target(args.data_dir)
    if dim2 != dim:
        raise ValueError(f"data-dir dim {dim2} != run dim {dim}")
    ops = build_jax_ops(model_seq, param_names)
    W = ops["W"]
    npix = W.size
    side = int(round(np.sqrt(npix)))
    if side * side != npix:
        raise ValueError(f"image not square: npix={npix} side={side}")
    print(f"[T11] image {side}x{side}={npix} px; top {TOP_FRACTION:.0%} = "
          f"{int(np.ceil(TOP_FRACTION*npix))} px; names[0..3]={param_names[:4]}")

    # attach z to each point (from run arrays)
    def _attach(pt):
        z = samples_z[int(pt["chain"]), int(pt["step"])]
        q = dict(pt); q["_z"] = z
        return q
    spikes = [_attach(p) for p in spikes]
    baselines = [_attach(p) for p in baselines]

    # E1 family reference (optional)
    e1_json = args.e1_json
    if e1_json is None:
        cand = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)),
                            "e1", "e1_results.json")
        e1_json = cand if os.path.isfile(cand) else None
    family_ref, fam_names = family_reference_from_e1(e1_json, dim)
    if family_ref is not None:
        print(f"[T11] E1 family reference loaded from {e1_json} (top-8-support "
              "mean |loading|; family cosine reported)")
    else:
        print("[T11] no E1 family reference available; reporting loadings only")

    # --- chi^2 gate at the first 3 available points -------------------------
    gate_src = (spikes + baselines)[:3]
    chi2_checks, worst_rel = chi2_gate(ops, [p["_z"] for p in gate_src], tag="T11")

    # --- dissect every point ------------------------------------------------
    def _dissect_all(points, label):
        out = []
        for i, pt in enumerate(points):
            dd = dissect_point(pt["_z"], ops, W, std_z, side, family_ref, param_names)
            print(f"[T11] {label}[{i}] seg{pt['segment']} ch{pt['chain']} "
                  f"s{pt['step']}: lambda1={dd['lambda1']:.3e} L={dd['L']:.3f} "
                  f"PR={dd['PR']:.4f} selfcheck_rel={dd['selfcheck_rel']:.2e}")
            out.append(dd)
        return out
    spike_dd = _dissect_all(spikes, "spike")
    base_dd = _dissect_all(baselines, "baseline")

    # --- paired comparison + summary ---------------------------------------
    base_by_rank = {b["matched_spike_rank"]: (b, d)
                    for b, d in zip(baselines, base_dd)}
    pairs = []
    for pt, dd in zip(spikes, spike_dd):
        rank = pt["rank"]
        if rank not in base_by_rank:
            continue
        bpt, bdd = base_by_rank[rank]
        pairs.append({"rank": rank, "spike": dd, "baseline": bdd,
                      "spike_pt": pt, "baseline_pt": bpt})

    L_spike = np.array([p["spike"]["L"] for p in pairs])
    L_base = np.array([p["baseline"]["L"] for p in pairs])
    PR_spike = np.array([p["spike"]["PR"] for p in pairs])
    PR_base = np.array([p["baseline"]["PR"] for p in pairs])
    med_L_spike = float(np.median(L_spike)) if L_spike.size else float("nan")
    med_L_base = float(np.median(L_base)) if L_base.size else float("nan")
    L_ratio = med_L_spike / med_L_base if med_L_base > 0 else float("inf")
    # per-pair ratio (median of the paired ratios, robust to a diffuse baseline)
    pair_ratios = [(p["spike"]["L"] / p["baseline"]["L"])
                   for p in pairs if p["baseline"]["L"] > 0]
    med_pair_ratio = float(np.median(pair_ratios)) if pair_ratios else float("nan")
    worst_selfcheck = max([d["selfcheck_rel"] for d in spike_dd + base_dd] or [0.0])

    print(f"[T11] median L: spike={med_L_spike:.3f} baseline={med_L_base:.3f} "
          f"ratio(medians)={L_ratio:.2f} median(pair-ratio)={med_pair_ratio:.2f}")
    print(f"[T11] median PR: spike={np.median(PR_spike):.4f} "
          f"baseline={np.median(PR_base):.4f}; worst selfcheck rel={worst_selfcheck:.2e}")

    # --- plots --------------------------------------------------------------
    c2_spike_path = os.path.join(args.out_dir, "t11_c2_spikes.png")
    c2_base_path = os.path.join(args.out_dir, "t11_c2_baselines.png")
    dots_path = os.path.join(args.out_dir, "t11_L_PR_paired.png")
    _plot_c2_grid(spikes, spike_dd, ops, side,
                  "T11 c^2 pixel-contribution maps -- SPIKES (log color, model image "
                  "cyan contours) -- PROPOSED (UNCERTIFIED)", c2_spike_path)
    _plot_c2_grid(baselines, base_dd, ops, side,
                  "T11 c^2 pixel-contribution maps -- BASELINES -- PROPOSED "
                  "(UNCERTIFIED)", c2_base_path)
    plot_paired_dots(pairs, dots_path)

    # --- JSON (drop the heavy _c2 / _z arrays) ------------------------------
    def _clean(dd, pt):
        q = {k: v for k, v in dd.items() if not k.startswith("_")}
        q.update({"segment": pt["segment"], "chain": pt["chain"],
                  "step": pt["step"], "census_lambda1": pt.get("lambda1")})
        for extra in ("rank", "matched_spike_rank", "segment_median",
                      "dist_to_nearest_spike"):
            if extra in pt:
                q[extra] = pt[extra]
        return q

    doc = {
        "experiment": "T11 -- render-space spike localization",
        "status": "proposed (UNCERTIFIED) -- grader inspects artifacts, not this summary",
        "timestamp_utc": _now(),
        "args": {k: v for k, v in vars(args).items()},
        "dim": dim, "param_names": param_names,
        "image_side": side, "n_pixels": npix,
        "top_fraction": TOP_FRACTION, "n_top_pixels": int(np.ceil(TOP_FRACTION * npix)),
        "spike_list_source": os.path.abspath(args.spike_list),
        "e1_family_reference": {
            "path": e1_json,
            "available": bool(family_ref is not None),
            "note": "mean |loading| over E1's 32 iid points, top-8-support per point "
                    "(APPROXIMATION of the full mean eigenvector); family cosine uses "
                    "|v1| (sign-robust). None => cosine omitted, loadings only.",
        },
        "c2_definition": {
            "formula": "c_i = sqrt(W_i) * (J_zhat v1)_i ; J_zhat = scale_cols(J_z, std_z); "
                       "M_zhat = J_zhat^T W J_zhat ; (lambda1, v1) top eigenpair.",
            "self_check": "sum_i c_i^2 = v1^T M_zhat v1 = lambda1 (asserted to "
                          f"{SELF_CHECK_RTOL:g} relative)",
            "worst_self_check_relative_error": worst_selfcheck,
        },
        "render_path_chi2_gate": {"checks": chi2_checks, "worst_relative_error": worst_rel},
        "standardization": {"std_z": std_z.tolist(),
                            "note": "SAME as T10 (per-coordinate ddof=1 over pooled "
                                    "results positions)"},
        "spikes": [_clean(d, p) for d, p in zip(spike_dd, spikes)],
        "baselines": [_clean(d, p) for d, p in zip(base_dd, baselines)],
        "summary": {
            "n_pairs": len(pairs),
            "median_L_spike": med_L_spike, "median_L_baseline": med_L_base,
            "L_ratio_medians": L_ratio, "median_pair_L_ratio": med_pair_ratio,
            "median_PR_spike": float(np.median(PR_spike)) if PR_spike.size else float("nan"),
            "median_PR_baseline": float(np.median(PR_base)) if PR_base.size else float("nan"),
        },
        "prediction": {
            "text": f"L_spike >= {T11_PREDICT_L_SPIKE:.0%} AND L_spike/L_baseline >= "
                    f"{T11_PREDICT_RATIO:g}",
            "L_spike_threshold": T11_PREDICT_L_SPIKE,
            "ratio_threshold": T11_PREDICT_RATIO,
            "measured_median_L_spike": med_L_spike,
            "measured_L_ratio_medians": L_ratio,
            "measured_median_pair_ratio": med_pair_ratio,
        },
        "falsifier": {
            "text": f"L_spike/L_baseline <~ {T11_FALSIFY_RATIO:g} => spikes are a "
                    "global-image effect; the caustic story is wrong",
            "ratio_threshold": T11_FALSIFY_RATIO,
        },
        "verdict_fields": {
            "T11_spike_pixel_localization": "proposed (UNCERTIFIED): compare median_L_"
                "spike to the 50% prediction and the spike/baseline L ratio to the >=3 "
                "prediction / <=1.5 falsifier; read the c^2 maps (arc structure) + the "
                "paired L/PR dot plot.",
            "note": "this script does NOT adjudicate; a grader inspects plots + numbers.",
        },
        "plots": {"c2_spikes": os.path.basename(c2_spike_path),
                  "c2_baselines": os.path.basename(c2_base_path),
                  "L_PR_paired": os.path.basename(dots_path)},
    }
    json_path = os.path.join(args.out_dir, "t11_results.json")
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[T11] wrote {json_path}, {c2_spike_path}, {c2_base_path}, {dots_path}")

    print("\n=== T11 summary (PROPOSED / UNCERTIFIED) ===")
    print(f"  chi^2 render gate: PASSED, worst rel {worst_rel:.3e}")
    print(f"  self-check (sum c^2 == lambda1): worst rel {worst_selfcheck:.2e} "
          f"(<= {SELF_CHECK_RTOL:g})")
    print(f"  n pairs = {len(pairs)}")
    print(f"  [L]     median spike = {med_L_spike:.3f} (predict >= "
          f"{T11_PREDICT_L_SPIKE:.0%}); baseline = {med_L_base:.3f}")
    print(f"  [ratio] L_spike/L_baseline (medians) = {L_ratio:.2f}; "
          f"median pair-ratio = {med_pair_ratio:.2f} "
          f"(predict >= {T11_PREDICT_RATIO:g}; falsifier <~ {T11_FALSIFY_RATIO:g})")
    print("[T11] done.")


if __name__ == "__main__":
    main()
