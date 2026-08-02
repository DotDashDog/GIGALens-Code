"""T10 -- On-ridge spike census: does the CHAIN actually meet the lambda1 spikes?

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T10, approved by the
human 2026-07-02). Chain of reasoning: T8 found the local Gauss-Newton metric's top
eigenvalue lambda1 = lambda1(M(zhat)) forms a field of SHARP NARROW spikes (width
~1e-3 in z, about one sampler step; up to x28 above a typical-set point HALF a step
away) along OFF-ridge straight transects. T10 asks the on-ridge question T8 could
not: does the real trajectory encounter such spikes at a rate and height sufficient
to pin the global step size (per T9's decile signal)? It walks CONTIGUOUS same-chain
segments of the reference run, computes lambda1 at EVERY step (batched jacfwd),
pairs it 1:1 with that step's smoothed energy error xi_s, and censuses the spikes.

PREDICTIONS (restated next to measured; from E1's tail + T8):
  spike rate  >= 5%  of on-ridge steps above 3x the SEGMENT median lambda1
  pooled max/median lambda1 >= 10
  spike width ~ 1-3 steps (consecutive steps above threshold)
  within-segment Spearman rho(lambda1, xi_s) >= 0.4
FALSIFIER: pooled max/median < 2 on ALL segments => spikes are an off-ridge
  phenomenon the chain never meets; the eps account collapses and T9's decile
  signal needs another source.

This script also stores v1 (stiffest eigenvector) at every step plus the
standardized chain-displacement direction and its |cos| with v1 -- collected now
because J is already in hand, feeding the FUTURE encounter model. It does NOT
adjudicate anything about the encounter geometry here (per the checkpoint).

REUSE (import, not duplicate): build_jax_ops / gn_metric / eig_desc / scale_cols
from e1_fisher_survey (verified render/W/aux builder + standardized-metric algebra);
rms_smooth / spearman_rho / valid_mask / XI_FLOOR from t9_xi_lambda (the xi
convention + smoothing, mclmc.py:334: xi = energy_change^2/(dim*5e-4)+1e-8; results
phase = last num_results columns of diagnostics.npz xi, 1:1 with arrays.npz
samples_z; floor/sentinel excluded). The chi^2 render-path gate (E1 Task A) is
replicated VERBATIM in convention and re-run at startup -- nothing downstream is
valid without it.

BATCHING: the expensive step is ~2048 forward-mode Jacobians of a 6400x22 map. We
jit jax.vmap(jax.jacfwd(render)) ONCE (reusing E1's jitted jac_render) and call it
over chunks of ~48 z-points, printing progress + a wall-time estimate per chunk.

jax is imported ONLY inside functions so the module imports cleanly under a plain
conda python (no jax) for offline smoke tests (--smoke) of the pure-numpy census
machinery: run-length spike widths, matched-baseline selection, Spearman reuse.
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
from t9_xi_lambda import rms_smooth, spearman_rho, valid_mask, XI_FLOOR, RMS_HALFWIN

# --- pre-registered constants (T10 checkpoint) -----------------------------
SEGMENT_LEN_DEFAULT = 256
N_SEGMENTS_DEFAULT = 8         # one per chain
CHUNK_DEFAULT = 48            # batched-jacfwd chunk size (checkpoint: ~32-64)
SPIKE_FACTOR = 3.0           # spike threshold = SPIKE_FACTOR x SEGMENT median lambda1
BASELINE_MIN_DIST = 20       # matched baseline must be >=20 steps from any spike
N_TOP_SPIKES = 12            # T11 input contract: top ~12 spikes + 12 baselines
# Pre-registered prediction / falsifier (restated next to measured numbers):
T10_PREDICT_RATE = 0.05          # >= 5% of on-ridge steps above 3x segment median
T10_PREDICT_MAXMED = 10.0        # pooled max/median lambda1 >= 10
T10_PREDICT_WIDTH = (1, 3)       # spike width ~ 1-3 steps
T10_PREDICT_RHO = 0.4            # within-segment Spearman rho(lambda1, xi_s) >= 0.4
T10_FALSIFY_MAXMED = 2.0         # max/median < 2 on ALL segments => falsified


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pure-numpy census machinery (offline-testable)
# ===========================================================================

def rle_runs(mask):
    """Run-length encode the True runs of a 1-D boolean array.

    Returns a list of (start_index, length) for each maximal run of consecutive
    True. Empty list if no True. This is the spike-width primitive: a spike is a
    contiguous run of steps above threshold, its WIDTH is the run length."""
    mask = np.asarray(mask, dtype=bool)
    runs = []
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((int(i), int(j - i + 1)))
            i = j + 1
        else:
            i += 1
    return runs


def spike_census(lam, threshold):
    """Census of lambda1 spikes in one segment given an absolute threshold.

    Returns dict with: above (bool mask), n_above, rate (fraction of steps above),
    runs (list of (start,length)), widths (run lengths, steps), spacings (diffs of
    run START indices, steps), n_spikes (number of runs)."""
    lam = np.asarray(lam, dtype=np.float64)
    above = lam > threshold
    runs = rle_runs(above)
    starts = np.array([r[0] for r in runs], dtype=np.int64)
    widths = [int(r[1]) for r in runs]
    spacings = np.diff(starts).astype(int).tolist() if starts.size >= 2 else []
    return {
        "above": above,
        "n_above": int(above.sum()),
        "rate": float(above.mean()) if lam.size else float("nan"),
        "runs": runs,
        "widths": widths,
        "spacings": spacings,
        "n_spikes": len(runs),
        "spike_start_indices": starts.tolist(),
    }


def select_baseline(lam, spike_steps, median, min_dist=BASELINE_MIN_DIST, used=None):
    """Pick a matched-baseline step index in this segment: the step whose lambda1 is
    NEAREST the segment median, subject to being >= min_dist steps from EVERY spike
    step and not already used.

    lam         : (L,) per-step lambda1 for the segment
    spike_steps : array of within-segment step indices flagged as spikes (above thr)
    median      : segment median lambda1 (the target value to match)
    used        : set of already-chosen within-segment indices to avoid duplicates
    Returns the chosen within-segment index (int) or None if no eligible step."""
    lam = np.asarray(lam, dtype=np.float64)
    used = used or set()
    spike_steps = np.asarray(spike_steps, dtype=np.int64)
    L = lam.size
    all_idx = np.arange(L)
    if spike_steps.size:
        # distance from each step to the NEAREST spike step
        d = np.min(np.abs(all_idx[:, None] - spike_steps[None, :]), axis=1)
    else:
        d = np.full(L, L, dtype=np.int64)
    eligible = (d >= min_dist)
    for u in used:
        if 0 <= u < L:
            eligible[u] = False
    if not eligible.any():
        return None
    cand = all_idx[eligible]
    j = cand[int(np.argmin(np.abs(lam[cand] - median)))]
    return int(j)


# ===========================================================================
# chi^2 render-path gate (replicated from E1 Task A -- E1 does not factor it out)
# ===========================================================================

def chi2_gate(ops, z_points, tag="T10"):
    """Recompute chi^2 from the render path and reconcile with log_prob's reduced
    chi^2 aux. Convention IDENTICAL to E1/T8/T9: chi2_from_aux = aux * event_size.
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

def smoke_rle(verbose=True):
    """Run-length spike widths on a synthetic thresholded series with KNOWN runs."""
    # steps above threshold at: [3], [7,8], [12,13,14] -> widths 1,2,3; starts 3,7,12
    mask = np.zeros(20, dtype=bool)
    mask[3] = True
    mask[7:9] = True
    mask[12:15] = True
    cen = spike_census(np.where(mask, 100.0, 1.0), threshold=10.0)
    widths_ok = cen["widths"] == [1, 2, 3]
    starts_ok = cen["spike_start_indices"] == [3, 7, 12]
    spac_ok = cen["spacings"] == [4, 5]          # 7-3, 12-7
    rate_ok = abs(cen["rate"] - 6.0 / 20.0) < 1e-12
    ok = widths_ok and starts_ok and spac_ok and rate_ok
    if verbose:
        print(f"[smoke] rle widths={cen['widths']} (expect [1,2,3]) starts="
              f"{cen['spike_start_indices']} (expect [3,7,12]) spacings="
              f"{cen['spacings']} (expect [4,5]) rate={cen['rate']:.3f} "
              f"(expect 0.300) -> {'PASS' if ok else 'FAIL'}")
    return {"widths": cen["widths"], "starts": cen["spike_start_indices"],
            "spacings": cen["spacings"], "rate": cen["rate"], "pass": bool(ok)}


def smoke_baseline(verbose=True):
    """Matched-baseline selection: honors the distance-from-spike constraint AND
    picks the nearest-to-median eligible step."""
    L = 100
    lam = np.full(L, 5.0)
    lam[50] = 500.0                    # a spike
    spike_steps = np.array([50])
    median = 5.0
    # nearest-to-median eligible step must be >=20 from step 50, i.e. outside (30,70)
    j = select_baseline(lam, spike_steps, median, min_dist=20, used=set())
    dist_ok = abs(j - 50) >= 20
    # value at j equals median (many ties -> argmin picks first eligible, fine)
    val_ok = abs(lam[j] - median) < 1e-9
    # a second call excluding j must return a DIFFERENT eligible index
    j2 = select_baseline(lam, spike_steps, median, min_dist=20, used={j})
    distinct_ok = (j2 is not None) and (j2 != j) and abs(j2 - 50) >= 20
    # a step INSIDE the forbidden band is never returned
    forbidden_ok = j not in range(31, 70) and j2 not in range(31, 70)
    ok = dist_ok and val_ok and distinct_ok and forbidden_ok
    if verbose:
        print(f"[smoke] baseline: j={j} (|j-50|={abs(j-50)}>=20, val={lam[j]:.1f}), "
              f"j2={j2} distinct={distinct_ok}, forbidden-band respected="
              f"{forbidden_ok} -> {'PASS' if ok else 'FAIL'}")
    return {"j": int(j), "j2": int(j2) if j2 is not None else None,
            "pass": bool(ok)}


def smoke_spearman_reuse(verbose=True):
    """Confirm the reused t9.spearman_rho recovers a planted monotone rho."""
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 1, 300)
    y = np.exp(3.0 * x) + 0.2 * rng.standard_normal(300)
    rho = spearman_rho(x, y)
    ok = rho > 0.85
    if verbose:
        print(f"[smoke] spearman reuse: rho={rho:.3f} (>0.85) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return {"rho": rho, "pass": bool(ok)}


def run_smoke():
    print("=== T10 offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_rle(); b = smoke_baseline(); c = smoke_spearman_reuse()
    allpass = a["pass"] and b["pass"] and c["pass"]
    print(f"[smoke] overall: {'PASS' if allpass else 'FAIL'}")
    return {"rle": a, "baseline": b, "spearman": c, "pass": bool(allpass)}


# ===========================================================================
# Batched Jacobian -> per-step lambda1 / v1 for one segment
# ===========================================================================

def segment_metrics(seg_z, batched_jac, W, std_z, chunk, tag=""):
    """Compute lambda1, lambda2, v1 at every step of a (L, dim) segment via a
    jitted, vmapped jacfwd called over chunks. Returns (lam1, lam2, V1) with
    lam1/lam2 shape (L,) and V1 shape (L, dim). Prints per-chunk progress and a
    wall-time estimate for the whole segment."""
    seg_z = np.asarray(seg_z, dtype=np.float64)
    L, dim = seg_z.shape
    lam1 = np.empty(L, dtype=np.float64)
    lam2 = np.empty(L, dtype=np.float64)
    V1 = np.empty((L, dim), dtype=np.float64)
    t_start = time.perf_counter()
    done = 0
    for lo in range(0, L, chunk):
        hi = min(L, lo + chunk)
        Z = seg_z[lo:hi]
        Js = np.asarray(batched_jac(Z), dtype=np.float64)     # (nb, 6400, 22)
        for k in range(Js.shape[0]):
            M = gn_metric(scale_cols(Js[k], std_z), W)
            w, Vv = eig_desc(M)
            lam1[lo + k] = w[0]
            lam2[lo + k] = w[1]
            V1[lo + k] = Vv[:, 0]
        done = hi
        elapsed = time.perf_counter() - t_start
        rate = done / max(elapsed, 1e-9)
        eta = (L - done) / max(rate, 1e-9)
        print(f"[T10]{tag} chunk {lo:4d}-{hi:4d}/{L}: {done}/{L} jacobians, "
              f"{elapsed:6.1f}s elapsed, {rate:5.1f}/s, ETA {eta:5.1f}s")
    return lam1, lam2, V1


# ===========================================================================
# Plots
# ===========================================================================

def plot_segment_traces(segments, out_path):
    """<=4x4 panels: per-segment lambda1(step) trace (log-y) with the 3x-median
    threshold line, spikes marked, and xi_s overlaid on a twin axis."""
    n = len(segments)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 2.6 * nrow),
                             squeeze=False)
    for i, S in enumerate(segments):
        ax = axes[i // ncol][i % ncol]
        steps = np.arange(S["lambda1"].size)
        lam = S["lambda1"]; thr = S["threshold"]; xis = S["xi_s"]
        ax.plot(steps, lam, color="#8a3b3b", lw=0.8, label="lambda1")
        ax.axhline(thr, color="purple", ls="--", lw=1.0,
                   label=f"3x median={thr:.2e}")
        above = lam > thr
        ax.scatter(steps[above], lam[above], s=10, color="k", zorder=3,
                   label="spike")
        ax.set_yscale("log")
        ax.set_ylabel(f"seg{S['segment']} (chain {S['chain']})\nlambda1",
                      fontsize=7)
        ax.tick_params(labelsize=6)
        axt = ax.twinx()
        axt.plot(steps, xis, color="#3b6a8a", lw=0.6, alpha=0.7)
        axt.set_yscale("log")
        axt.set_ylabel("xi_s", fontsize=6, color="#3b6a8a")
        axt.tick_params(labelsize=6, colors="#3b6a8a")
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
        ax.set_xlabel("step within segment", fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("T10 on-ridge lambda1(step) traces + xi_s (twin axis) "
                 "-- PROPOSED (UNCERTIFIED)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pooled_hists(lam_pooled, widths_pooled, maxmed, out_path):
    """1x2: pooled lambda1 histogram (log-x) with median + pooled max marked;
    pooled spike-width histogram (steps)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    lp = lam_pooled[np.isfinite(lam_pooled) & (lam_pooled > 0)]
    bins = np.geomspace(lp.min(), lp.max(), 60)
    ax.hist(lp, bins=bins, color="#8a3b3b", alpha=0.8)
    med = np.median(lp)
    ax.axvline(med, color="k", ls="--", lw=1.2, label=f"median={med:.2e}")
    ax.axvline(lp.max(), color="purple", ls=":", lw=1.2,
               label=f"max={lp.max():.2e} (max/median={maxmed:.1f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("lambda1(M_zhat)  (top GN eigenvalue, standardized)")
    ax.set_ylabel("count"); ax.set_title("pooled lambda1 histogram", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    if widths_pooled:
        wmax = max(widths_pooled)
        ax.hist(widths_pooled, bins=np.arange(0.5, wmax + 1.5, 1.0),
                color="#3b6a8a", alpha=0.85)
    ax.set_xlabel("spike width (consecutive steps above 3x segment median)")
    ax.set_ylabel("count")
    ax.set_title(f"pooled spike-width histogram (predict {T10_PREDICT_WIDTH[0]}-"
                 f"{T10_PREDICT_WIDTH[1]} steps)", fontsize=10)
    fig.suptitle("T10 pooled census distributions -- PROPOSED (UNCERTIFIED)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_lambda_vs_xi(lam_pooled, xis_pooled, rho, out_path):
    """Dense log-log scatter of lambda1 vs xi_s over ALL segment steps."""
    fig, ax = plt.subplots(figsize=(8, 6.2))
    m = (np.isfinite(lam_pooled) & (lam_pooled > 0)
         & np.isfinite(xis_pooled) & (xis_pooled > 0))
    ax.scatter(xis_pooled[m], lam_pooled[m], s=6, c="#555555", alpha=0.35)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("smoothed |xi|  (+/-%d-step RMS energy-error proxy)" % RMS_HALFWIN)
    ax.set_ylabel("lambda1(M_zhat)")
    ax.set_title(f"T10 on-ridge lambda1 vs xi_s (pooled, Spearman rho={rho:.3f}) "
                 "-- PROPOSED (UNCERTIFIED)", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T10 on-ridge spike census")
    p.add_argument("--data-dir")
    p.add_argument("--run-dir", help="MCLMC run dir with arrays.npz + diagnostics.npz")
    p.add_argument("--out-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--segment-len", type=int, default=SEGMENT_LEN_DEFAULT)
    p.add_argument("--n-segments", type=int, default=N_SEGMENTS_DEFAULT)
    p.add_argument("--chunk-size", type=int, default=CHUNK_DEFAULT,
                   help="batched-jacfwd chunk size (~32-64)")
    p.add_argument("--smoke", action="store_true",
                   help="run the offline numpy-only smoke tests and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        res = run_smoke()
        if not res["pass"]:
            raise SystemExit("[T10] smoke tests FAILED")
        return

    for req in ("data_dir", "run_dir", "out_dir", "seed"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    os.makedirs(args.out_dir, exist_ok=True)

    from common import assert_x64, load_target
    assert_x64()

    # --- load run arrays (same convention as T9) ----------------------------
    arrays = np.load(os.path.join(args.run_dir, "arrays.npz"))
    diag = np.load(os.path.join(args.run_dir, "diagnostics.npz"))
    samples_z = np.asarray(arrays["samples_z"], dtype=np.float64)   # (C, R, dim)
    xi_full = np.asarray(diag["xi"], dtype=np.float64)              # (C, burnin+results)
    C, R, dim = samples_z.shape
    if xi_full.shape[0] != C:
        raise ValueError(f"xi chains {xi_full.shape[0]} != positions chains {C}")
    total = xi_full.shape[1]
    if total < R:
        raise ValueError(f"xi length {total} < results {R}")
    xi_res = xi_full[:, -R:]                                        # results-phase xi
    n_burn = total - R
    print(f"[T10] positions {samples_z.shape} (results-only); xi {xi_full.shape} "
          f"(burnin={n_burn}+results={R}); using last {R} xi columns")

    # xi validity + floor accounting (identical convention to T9) + smoothing
    valid = valid_mask(xi_res)
    n_floor = int(np.sum(np.asarray(xi_res) <= XI_FLOOR * (1.0 + 1e-6)))
    n_nonfin = int(np.sum(~np.isfinite(xi_res)))
    xi_s = np.vstack([rms_smooth(xi_res[c], RMS_HALFWIN) for c in range(C)])
    print(f"[T10] xi results: min={np.nanmin(xi_res):.3e} max={np.nanmax(xi_res):.3e} "
          f"median={np.nanmedian(xi_res):.3e}; floor excluded={n_floor}, "
          f"non-finite excluded={n_nonfin}, valid frac={valid.mean():.4f}")

    # standardization: SAME as E1/T8/T9 (per-coordinate std over pooled positions)
    flat = samples_z.reshape(-1, dim)
    std_z = np.std(flat, axis=0, ddof=1)

    # --- segments: one seeded contiguous start per chain --------------------
    L = int(args.segment_len)
    nseg = int(args.n_segments)
    if L > R:
        raise ValueError(f"segment-len {L} > results length {R}")
    rng = np.random.default_rng(args.seed)
    seg_defs = []
    for i in range(nseg):
        chain = i % C
        start = int(rng.integers(0, R - L + 1))
        seg_defs.append({"segment": i, "chain": chain, "start": start})
    print(f"[T10] {nseg} segments x {L} steps (~{nseg*L} jacobians); "
          f"chunk={args.chunk_size}; starts="
          f"{[(d['chain'], d['start']) for d in seg_defs]}")

    # --- target + render ops + chi^2 gate -----------------------------------
    prob_model, qz, z_center, dim2, param_names = load_target(args.data_dir)
    if dim2 != dim:
        raise ValueError(f"data-dir dim {dim2} != run dim {dim}")
    ops = build_jax_ops(prob_model, param_names)
    W = ops["W"]

    import jax
    batched_jac = jax.jit(jax.vmap(ops["jac_render"]))   # (nb,dim)->(nb,6400,22)

    # gate at the first step of the first 3 segments
    gate_pts = [samples_z[seg_defs[i]["chain"], seg_defs[i]["start"]]
                for i in range(min(3, len(seg_defs)))]
    chi2_checks, worst_rel = chi2_gate(ops, gate_pts, tag="T10")

    # --- per-segment census -------------------------------------------------
    t0_all = time.perf_counter()
    segments = []          # rich per-segment records (arrays kept as np)
    for d in seg_defs:
        chain, start = d["chain"], d["start"]
        seg_z = samples_z[chain, start:start + L]              # (L, dim)
        lam1, lam2, V1 = segment_metrics(seg_z, batched_jac, W, std_z,
                                         args.chunk_size,
                                         tag=f" seg{d['segment']}(ch{chain})")
        seg_xi_s = xi_s[chain, start:start + L]
        seg_valid = valid[chain, start:start + L]

        # standardized chain-displacement direction + |cos| with v1 (per step)
        seg_z_hat = seg_z / std_z[None, :]
        disp = np.diff(seg_z_hat, axis=0)                      # (L-1, dim)
        dnorm = np.linalg.norm(disp, axis=1)
        disp_unit = disp / np.where(dnorm[:, None] > 0, dnorm[:, None], 1.0)
        cos_v1_disp = np.full(L, np.nan)                       # aligned to step k
        cos_v1_disp[:L - 1] = np.abs(np.sum(disp_unit * V1[:L - 1], axis=1))

        median = float(np.median(lam1))
        threshold = SPIKE_FACTOR * median
        cen = spike_census(lam1, threshold)
        rho_seg = spearman_rho(lam1, seg_xi_s)
        maxmed = float(np.max(lam1) / median) if median > 0 else float("inf")

        segments.append({
            "segment": d["segment"], "chain": chain, "start": start,
            "lambda1": lam1, "lambda2": lam2, "V1": V1,
            "xi_s": seg_xi_s, "valid": seg_valid,
            "cos_v1_disp": cos_v1_disp, "disp_norm_zhat": dnorm,
            "median": median, "threshold": threshold,
            "max": float(np.max(lam1)), "max_over_median": maxmed,
            "census": cen, "rho": rho_seg,
        })
        print(f"[T10] seg{d['segment']} ch{chain} start{start}: median={median:.3e} "
              f"max={np.max(lam1):.3e} max/med={maxmed:.2f} rate={cen['rate']:.3f} "
              f"n_spikes={cen['n_spikes']} widths={cen['widths'][:8]} "
              f"rho={rho_seg:.3f}")
    wall_metrics = time.perf_counter() - t0_all
    print(f"[T10] all-segment metric wall {wall_metrics:.1f}s "
          f"({nseg*L} jacobians, {nseg*L/max(wall_metrics,1e-9):.1f}/s)")

    # --- pooled statistics --------------------------------------------------
    lam_pooled = np.concatenate([S["lambda1"] for S in segments])
    xis_pooled = np.concatenate([S["xi_s"] for S in segments])
    pooled_median = float(np.median(lam_pooled))
    pooled_max = float(np.max(lam_pooled))
    pooled_maxmed = pooled_max / pooled_median if pooled_median > 0 else float("inf")
    pooled_rate = float(np.mean([S["census"]["above"].mean() for S in segments]))
    # pooled spike rate as total-above / total-steps (each seg's own threshold)
    pooled_n_above = int(sum(S["census"]["n_above"] for S in segments))
    pooled_rate_total = pooled_n_above / lam_pooled.size
    widths_pooled = [w for S in segments for w in S["census"]["widths"]]
    spacings_pooled = [s for S in segments for s in S["census"]["spacings"]]
    rho_pooled = spearman_rho(lam_pooled, xis_pooled)
    seg_maxmed = [S["max_over_median"] for S in segments]
    seg_rho = [S["rho"] for S in segments]

    median_width = float(np.median(widths_pooled)) if widths_pooled else float("nan")
    width_in_band = (widths_pooled and
                     T10_PREDICT_WIDTH[0] <= median_width <= T10_PREDICT_WIDTH[1])
    all_below_falsifier = all(mm < T10_FALSIFY_MAXMED for mm in seg_maxmed)

    print(f"[T10] POOLED: median={pooled_median:.3e} max={pooled_max:.3e} "
          f"max/median={pooled_maxmed:.2f} spike-rate(total)={pooled_rate_total:.3f} "
          f"median-width={median_width} rho={rho_pooled:.3f}")
    print(f"[T10] per-segment max/median = "
          f"{[round(x,2) for x in seg_maxmed]}")

    # --- spike_list.json (T11 input contract) -------------------------------
    # top N spikes pooled by lambda1; 12 matched same-segment baselines.
    pooled_records = []
    for S in segments:
        for k in range(S["lambda1"].size):
            pooled_records.append({
                "segment": S["segment"], "chain": S["chain"],
                "step": int(S["start"] + k), "seg_step": int(k),
                "lambda1": float(S["lambda1"][k]),
                "segment_median": S["median"],
                "segment_threshold": S["threshold"],
                "above_threshold": bool(S["lambda1"][k] > S["threshold"]),
            })
    order = np.argsort([-r["lambda1"] for r in pooled_records])
    top = [pooled_records[i] for i in order[:N_TOP_SPIKES]]
    spikes_out = []
    for rank, r in enumerate(top):
        rr = dict(r); rr["rank"] = rank
        spikes_out.append(rr)

    # matched baselines: one per spike, same segment, nearest-median, >=20 from
    # any spike step (spike = above threshold), distinct within a segment.
    seg_by_id = {S["segment"]: S for S in segments}
    used_by_seg = {}
    baselines_out = []
    for rank, r in enumerate(spikes_out):
        S = seg_by_id[r["segment"]]
        spike_steps = np.array(np.where(S["census"]["above"])[0], dtype=np.int64)
        used = used_by_seg.setdefault(r["segment"], set())
        j = select_baseline(S["lambda1"], spike_steps, S["median"],
                            BASELINE_MIN_DIST, used)
        if j is None:
            print(f"[T10] WARNING: no eligible baseline for spike rank {rank} "
                  f"(segment {r['segment']}); skipping.")
            continue
        used.add(j)
        d_near = int(np.min(np.abs(j - spike_steps))) if spike_steps.size else int(S["lambda1"].size)
        baselines_out.append({
            "matched_spike_rank": rank, "segment": r["segment"],
            "chain": S["chain"], "step": int(S["start"] + j), "seg_step": int(j),
            "lambda1": float(S["lambda1"][j]),
            "segment_median": S["median"],
            "segment_threshold": S["threshold"],
            "dist_to_nearest_spike": d_near,
        })
    spike_list = {
        "experiment": "T10 spike_list -- input contract for T11",
        "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(),
        "run_dir": os.path.abspath(args.run_dir),
        "data_dir": os.path.abspath(args.data_dir),
        "note": "z at each point is samples_z[chain, step] from run_dir/arrays.npz; "
                "std_z is per-coordinate std over pooled results positions (ddof=1); "
                "M_zhat = gn_metric(scale_cols(J_z, std_z), W).",
        "n_spikes": len(spikes_out), "n_baselines": len(baselines_out),
        "spike_factor": SPIKE_FACTOR, "baseline_min_dist": BASELINE_MIN_DIST,
        "spikes": spikes_out, "baselines": baselines_out,
    }
    spike_list_path = os.path.join(args.out_dir, "spike_list.json")
    with open(spike_list_path, "w") as fh:
        json.dump(spike_list, fh, indent=2)
    print(f"[T10] wrote {spike_list_path} "
          f"({len(spikes_out)} spikes + {len(baselines_out)} baselines)")

    # --- companion npz (full per-step v1 etc. for the encounter model) ------
    npz_path = os.path.join(args.out_dir, "t10_arrays.npz")
    np.savez(
        npz_path,
        seg_ids=np.array([S["segment"] for S in segments]),
        seg_chains=np.array([S["chain"] for S in segments]),
        seg_starts=np.array([S["start"] for S in segments]),
        lambda1=np.stack([S["lambda1"] for S in segments]),
        lambda2=np.stack([S["lambda2"] for S in segments]),
        V1=np.stack([S["V1"] for S in segments]),
        xi_s=np.stack([S["xi_s"] for S in segments]),
        cos_v1_disp=np.stack([S["cos_v1_disp"] for S in segments]),
        disp_norm_zhat=np.stack([S["disp_norm_zhat"] for S in segments]),
        std_z=std_z,
    )
    print(f"[T10] wrote {npz_path} (per-step v1 + displacement for encounter model)")

    # --- plots --------------------------------------------------------------
    traces_path = os.path.join(args.out_dir, "t10_segment_traces.png")
    hist_path = os.path.join(args.out_dir, "t10_pooled_hists.png")
    scatter_path = os.path.join(args.out_dir, "t10_lambda_vs_xi.png")
    plot_segment_traces(segments, traces_path)
    plot_pooled_hists(lam_pooled, widths_pooled, pooled_maxmed, hist_path)
    plot_lambda_vs_xi(lam_pooled, xis_pooled, rho_pooled, scatter_path)

    # --- JSON ---------------------------------------------------------------
    seg_json = []
    for S in segments:
        cen = S["census"]
        seg_json.append({
            "segment": S["segment"], "chain": S["chain"], "start": S["start"],
            "lambda1": S["lambda1"].tolist(),                 # per-step (checkpoint)
            "xi_s": S["xi_s"].tolist(),
            "cos_v1_disp": [None if not np.isfinite(x) else float(x)
                            for x in S["cos_v1_disp"]],
            "median": S["median"], "threshold": S["threshold"],
            "max": S["max"], "max_over_median": S["max_over_median"],
            "spike_rate": cen["rate"], "n_spikes": cen["n_spikes"],
            "spike_widths": cen["widths"], "spike_spacings": cen["spacings"],
            "spike_start_indices": cen["spike_start_indices"],
            "spike_steps_global": [int(S["start"] + s)
                                   for s in np.where(cen["above"])[0]],
            "spearman_rho_lambda1_xi": S["rho"],
        })

    doc = {
        "experiment": "T10 -- on-ridge spike census",
        "status": "proposed (UNCERTIFIED) -- grader inspects artifacts, not this summary",
        "timestamp_utc": _now(),
        "args": {k: v for k, v in vars(args).items()},
        "dim": dim, "param_names": param_names,
        "run_dir": os.path.abspath(args.run_dir),
        "xi_convention": {
            "source": "src/gigalens_research/inference/mclmc.py line 334 (skip_adapt)",
            "formula": "xi = energy_change^2 / (dim * desired_energy_var) + 1e-8",
            "dim": dim, "desired_energy_var": 0.0005,
            "results_phase": f"last {R} of {total} xi columns (burnin={n_burn})",
            "smoothing": f"centered +/-{RMS_HALFWIN}-step RMS of |xi| per chain",
            "floor_excluded_count": n_floor, "nonfinite_excluded_count": n_nonfin,
            "valid_fraction": float(valid.mean()),
        },
        "standardization": {
            "note": "SAME as E1/T8/T9: zhat=z/std_z (per-coordinate ddof=1 over pooled "
                    "results positions); M_zhat=gn_metric(scale_cols(J_z, std_z), W); "
                    "lambda1 = top eigenvalue of M_zhat; v1 its eigenvector.",
            "std_z": std_z.tolist(),
        },
        "spike_threshold_convention": f"{SPIKE_FACTOR:g} x SEGMENT median lambda1 "
                                      "(per-segment, not pooled)",
        "render_path_chi2_gate": {"checks": chi2_checks, "worst_relative_error": worst_rel},
        "displacement_note": "cos_v1_disp[k] = |unit(Delta zhat between step k and k+1) "
                             ". v1(step k)| (v1 sign arbitrary => abs). Collected for the "
                             "FUTURE encounter model; NOT adjudicated here. Full per-step "
                             "v1 + displacement in t10_arrays.npz.",
        "segments": seg_json,
        "pooled": {
            "n_steps": int(lam_pooled.size),
            "median_lambda1": pooled_median, "max_lambda1": pooled_max,
            "max_over_median": pooled_maxmed,
            "spike_rate_mean_over_segments": pooled_rate,
            "spike_rate_total": pooled_rate_total,
            "n_above_total": pooled_n_above,
            "spike_widths": widths_pooled, "median_spike_width": median_width,
            "spike_spacings": spacings_pooled,
            "spearman_rho_lambda1_xi": rho_pooled,
            "per_segment_max_over_median": seg_maxmed,
            "per_segment_spearman_rho": seg_rho,
        },
        "prediction": {
            "text": f"spike rate >= {T10_PREDICT_RATE:.0%}, pooled max/median >= "
                    f"{T10_PREDICT_MAXMED:g}, spike width {T10_PREDICT_WIDTH[0]}-"
                    f"{T10_PREDICT_WIDTH[1]} steps, within-segment rho >= {T10_PREDICT_RHO:g}",
            "rate_threshold": T10_PREDICT_RATE,
            "maxmed_threshold": T10_PREDICT_MAXMED,
            "width_band_steps": list(T10_PREDICT_WIDTH),
            "rho_threshold": T10_PREDICT_RHO,
            "measured_pooled_rate_total": pooled_rate_total,
            "measured_pooled_maxmed": pooled_maxmed,
            "measured_median_width": median_width,
            "measured_pooled_rho": rho_pooled,
            "measured_max_within_segment_rho": float(np.nanmax(seg_rho)) if seg_rho else float("nan"),
            "width_in_predicted_band": bool(width_in_band),
        },
        "falsifier": {
            "text": f"max/median < {T10_FALSIFY_MAXMED:g} on ALL segments => spikes are "
                    "off-ridge, the chain never meets them; the eps account collapses",
            "maxmed_threshold": T10_FALSIFY_MAXMED,
            "all_segments_below_threshold": bool(all_below_falsifier),
        },
        "verdict_fields": {
            "T10_chain_meets_spikes": "proposed (UNCERTIFIED): compare pooled max/median, "
                "spike rate, width band, and within-segment rho to the prediction; the "
                "falsifier fires only if ALL segments have max/median < 2. Read the "
                "per-segment trace + pooled histogram + lambda1-vs-xi scatter.",
            "note": "this script does NOT adjudicate; a grader inspects plots + numbers.",
        },
        "plots": {"segment_traces": os.path.basename(traces_path),
                  "pooled_hists": os.path.basename(hist_path),
                  "lambda_vs_xi": os.path.basename(scatter_path)},
        "spike_list": os.path.basename(spike_list_path),
        "arrays_npz": os.path.basename(npz_path),
    }
    json_path = os.path.join(args.out_dir, "t10_results.json")
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[T10] wrote {json_path}")

    print("\n=== T10 summary (PROPOSED / UNCERTIFIED) ===")
    print(f"  chi^2 render gate: PASSED, worst rel {worst_rel:.3e}")
    print(f"  segments: {nseg} x {L} steps ({nseg*L} jacobians in {wall_metrics:.1f}s)")
    print(f"  [rate]     pooled spike-rate(total) = {pooled_rate_total:.3f} "
          f"(predict >= {T10_PREDICT_RATE:.0%})")
    print(f"  [max/med]  pooled = {pooled_maxmed:.2f} "
          f"(predict >= {T10_PREDICT_MAXMED:g}; falsifier: ALL seg < {T10_FALSIFY_MAXMED:g} "
          f"=> {'FIRED' if all_below_falsifier else 'not fired'})")
    print(f"  [width]    pooled median = {median_width} steps "
          f"(predict {T10_PREDICT_WIDTH[0]}-{T10_PREDICT_WIDTH[1]})")
    print(f"  [rho]      pooled = {rho_pooled:.3f}; max within-segment = "
          f"{(float(np.nanmax(seg_rho)) if seg_rho else float('nan')):.3f} "
          f"(predict within-segment >= {T10_PREDICT_RHO:g})")
    print("[T10] done.")


if __name__ == "__main__":
    main()
