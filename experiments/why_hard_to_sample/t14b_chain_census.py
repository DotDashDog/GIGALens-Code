"""T14b -- Corrected last-link test: curvature along each posterior's OWN chains.

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T14b; run under the
human's standing "close the last link" mandate, conversation 2026-07-03; flagged for
explicit grading with the T14 post-mortem).

WHY (T14 post-mortem, verbatim from the checkpoint): T14 ran as registered and its
falsifier FIRED (NEW-data g_H teeth ~7x g_M along the dial), BUT the T14 dial measured
OFF-typical-set curvature: it exits the typical set within ~0.02 px (<< one tooth
spacing), and beyond that BOTH targets carry a ~75 sigma displaced-blob residual that
swamps the 14 sigma data difference. The dial measured curvature where the sampler
never goes. Registered design error (orchestrator's). The wash-out claim was left
UNRESOLVED, not refuted. T14b measures the SAME curvature quantities along each
posterior's OWN chains -- the locus the sampler actually visits.

DESIGN (T10-style census at the correct locus):
  For each of two targets --
    OLD : the self-consistent posterior. Model = common.load_target(systems/sys60)
          (RE-PINNED to the frozen stale reference; ss2 model on the ss2 data).
          Chains + xi from the OUR-harness npz results_t0t1/sys60/t0/t0_seed1.npz.
    NEW : the {d', ss2-model} arm. Model = t14.load_new_target(systems/sys60_ss16data,
          supersample=2) (ss2 model on the accurate ss128 data d'). Chains + xi from
          resim/sys60_ss16/arm_2/t0/t0_seed1.npz.
  -- walk 2 contiguous 128-step segments (seeded random chain+start within the
  RESULTS phase, >= 20 steps from the phase start). At EVERY step z:
    (i)   lambda1(M_zhat) and its top eigenvector e (the stiff/counter-image
          direction), M_zhat = gn_metric(scale_cols(J_z, std_z), W)  [census/T12 convention]
    (ii)  g_H = e^T H_zhat e, H_zhat = diag(std_z)(-hessian(prob_model.log_prob(z)[0]))
          diag(std_z)  [T14's exact-Hessian-in-zhat construction, along the LOCAL stiff e]
    (iii) render -> blob_stats -> counter-image sub-pixel centroid x_c (T12 machinery)
  and pair each step with its OWN run's smoothed xi (+/-20-step RMS; xi convention
  xi = energy_change^2/(dim*5e-4)+1e-8, floor entries excluded; T9/T10).

  ONE COMMON standardization std_z for BOTH targets: per-coordinate ddof=1 std over the
  FROZEN original reference results/testsys60/mclmc.stale-20260703T111618/arrays.npz
  pooled samples_z (so OLD/NEW curvatures are directly comparable, exactly as T14).
  The SAME seeded segment coordinates (chain, start) are applied to each run's own
  chains (a controlled comparison; both npz are (8, 4000, 22)).

PREDICTIONS (restated next to every measured number in the JSON):
  (1) lambda1(M) tooth statistics comparable across targets IF x_c supports comparable
      (the comb is position-set; M is data-independent up to the small W shift).
  (2) THE KEY METRIC -- H-form teeth experienced by the NEW chain SUPPRESSED >= 10x vs
      the OLD chain's (the wash-out claim at the right locus).
      FALSIFIER: NEW H-teeth >= 1/3 of OLD's AND xi stays calm on NEW => teeth don't
      cause slowness => the T9/T10 causal reading collapses (stop and rethink).
  (3) xi-lambda1 (and xi-g_H) coupling PRESENT on OLD (rho ~ 0.4, T9) and ABSENT on
      NEW (rho < 0.2).
  (4) x_c-spread alternative protector: NEW posterior's x_c spread >> OLD's (sub-pixel
      lock broken by the misfit floor) => "protection" = de-locking, not Hessian
      smoothing. Both are reported.

This script PRODUCES artifacts + PROPOSED (UNCERTIFIED) verdict fields; it does NOT
adjudicate (a grader inspects plots/JSON, not the printed summary).

REUSE (import, not duplicate):
  * t10_spike_census : rle_runs, spike_census (>3x-segment-median run-length census),
                       chi2_gate (the hard render-path gate), and the batched-jacfwd
                       segment pattern.
  * t9_xi_lambda     : rms_smooth, spearman_rho, valid_mask, XI_FLOOR, RMS_HALFWIN
                       (the +/-20-step RMS xi smoothing + Spearman).
  * e1_fisher_survey : build_jax_ops, gn_metric, scale_cols, eig_desc, TINY.
  * t12_flank_crossing: blob_stats (pedestal-subtracted counter-image centroid).
  * t14_hessian_dial : load_new_target (qz-free d' model builder) and its
                       H_zhat = diag(std_z)(-hessian(logp))diag(std_z) construction.
  * common           : assert_x64, load_target, compute_diagnostics.

jax is imported ONLY inside functions so the module imports cleanly under a plain conda
python (no jax) for offline smoke tests (--smoke) of the pure-numpy analysis.
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
from t10_spike_census import rle_runs, spike_census, chi2_gate
from t9_xi_lambda import rms_smooth, spearman_rho, valid_mask, XI_FLOOR, RMS_HALFWIN
from e1_fisher_survey import build_jax_ops, gn_metric, scale_cols, eig_desc, TINY
from t12_flank_crossing import blob_stats
from t14_hessian_dial import load_new_target

# --- pre-registered constants (T14b checkpoint) ----------------------------
N_SEGMENTS = 2          # 2 contiguous segments per target
SEGMENT_LEN = 128       # 128 steps each (=> 256 census points per target)
MIN_START = 20          # segment start >= 20 steps from the results-phase start
CHUNK = 32              # batched Jacobian + render chunk
SPIKE_FACTOR = 3.0      # lambda1 "spike" = > 3x the segment median (comparability)
TOOTH_SPACING_PX = 0.52  # ss=2 subgrid pitch (T12): the x_c-spread yardstick

# Pre-registered predictions / falsifiers (restated next to measured numbers).
NEW_SUPPRESS_MIN = 10.0     # (2) NEW H-teeth suppressed >= 10x vs OLD
FALSIFY_TEETH_FRAC = 1.0 / 3.0  # (2) NEW teeth >= 1/3 OLD (while xi calm) => wash-out wrong
OLD_RHO_MIN = 0.4           # (3) xi-lambda1 coupling on OLD rho >~ 0.4 (T9)
NEW_RHO_MAX = 0.2           # (3) xi coupling ABSENT on NEW rho < 0.2

# Default inputs (overridable via CLI).
DEFAULT_STD_REF = ("/global/homes/l/linusu/GIGALens-Code/results/testsys60/"
                   "mclmc.stale-20260703T111618/arrays.npz")
DEFAULT_OLD_RUN = "./results_t0t1/sys60/t0/t0_seed1.npz"
DEFAULT_NEW_RUN = "./resim/sys60_ss16/arm_2/t0/t0_seed1.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pure-numpy analysis machinery (offline-testable; operates on plain arrays)
# ===========================================================================

def lambda_tooth_stats(lam1, seg_id, spike_factor=SPIKE_FACTOR):
    """lambda1(M) tooth statistics for one target (comparability check, part 1).

    Spike threshold is PER-SEGMENT (3x that segment's median lambda1), matching T10.
    Returns pooled median, pooled max/median, and the fraction of steps above their
    OWN segment's 3x-median threshold, plus the per-segment run-length spike census."""
    lam1 = np.asarray(lam1, dtype=np.float64)
    seg_id = np.asarray(seg_id)
    seg_medians = {}
    above = np.zeros(lam1.size, dtype=bool)
    per_seg = []
    for s in np.unique(seg_id):
        m = seg_id == s
        med = float(np.median(lam1[m]))
        seg_medians[int(s)] = med
        thr = spike_factor * med
        above[m] = lam1[m] > thr
        cen = spike_census(lam1[m], thr)
        per_seg.append({"segment": int(s), "median": med, "threshold": thr,
                        "max": float(np.max(lam1[m])),
                        "max_over_median": float(np.max(lam1[m]) / med) if med > 0 else float("inf"),
                        "spike_rate": cen["rate"], "n_spikes": cen["n_spikes"],
                        "widths": cen["widths"]})
    pooled_median = float(np.median(lam1))
    pooled_max = float(np.max(lam1))
    return {
        "pooled_median": pooled_median,
        "pooled_max": pooled_max,
        "pooled_max_over_median": float(pooled_max / pooled_median) if pooled_median > 0 else float("inf"),
        "spike_rate_vs_segment_median": float(above.mean()),
        "per_segment": per_seg,
        "segment_medians": seg_medians,
    }


def gH_teeth_stats(g_H, seg_id):
    """H-form teeth for one target (THE KEY METRIC, part 2).

    Reports the raw distribution (median, p90, max, max/median) AND the detrended-by-
    segment-median excursion amplitude (the largest tooth above its LOCAL baseline),
    which is sign-robust (g_H need not be positive everywhere). The excursion is the
    primary amplitude for the cross-target suppression ratio; max/median is reported
    too (flagged non-robust when the median is non-positive)."""
    g_H = np.asarray(g_H, dtype=np.float64)
    seg_id = np.asarray(seg_id)
    median = float(np.median(g_H))
    p90 = float(np.percentile(g_H, 90))
    gmax = float(np.max(g_H))
    # per-segment-median detrend -> teeth above local baseline
    detr = np.empty_like(g_H)
    seg_med = {}
    for s in np.unique(seg_id):
        m = seg_id == s
        sm = float(np.median(g_H[m]))
        seg_med[int(s)] = sm
        detr[m] = g_H[m] - sm
    excursion_amp = float(np.max(detr))          # largest tooth above local baseline
    return {
        "median": median, "p90": p90, "max": gmax,
        "max_over_median": float(gmax / median) if median > 0 else float("nan"),
        "median_positive": bool(median > 0),
        "excursion_amp_over_segment_median": excursion_amp,
        "segment_medians": seg_med,
        "min": float(np.min(g_H)),
    }


def xc_spread_stats(x_c, tooth_spacing=TOOTH_SPACING_PX):
    """Posterior spread of the counter-image centroid x_c for one target (part 4).

    std and total range over the census points, plus the range as a fraction of one
    tooth spacing (0.52 px). Tests the de-locking alternative: NEW spread >> OLD's
    would mean 'protection' = lost sub-pixel lock, not Hessian smoothing."""
    x_c = np.asarray(x_c, dtype=np.float64)
    x_c = x_c[np.isfinite(x_c)]
    rng = float(np.max(x_c) - np.min(x_c)) if x_c.size else float("nan")
    return {
        "n": int(x_c.size),
        "std_px": float(np.std(x_c, ddof=1)) if x_c.size > 1 else float("nan"),
        "range_px": rng,
        "frac_tooth_spacing": float(rng / tooth_spacing) if np.isfinite(rng) else float("nan"),
        "min_px": float(np.min(x_c)) if x_c.size else float("nan"),
        "max_px": float(np.max(x_c)) if x_c.size else float("nan"),
    }


def couplings(lam1, g_H, xi_s, valid):
    """Spearman rho of the smoothed xi against lambda1(M) and against g_H, over the
    VALID (finite, above-floor) census points (part 3)."""
    lam1 = np.asarray(lam1, float); g_H = np.asarray(g_H, float)
    xi_s = np.asarray(xi_s, float); valid = np.asarray(valid, bool)
    m = valid & np.isfinite(lam1) & np.isfinite(g_H) & np.isfinite(xi_s)
    return {
        "n_valid": int(m.sum()),
        "rho_lambda1_xi": spearman_rho(lam1[m], xi_s[m]),
        "rho_gH_xi": spearman_rho(g_H[m], xi_s[m]),
    }


def per_target_metrics(res):
    """All single-target metrics, on a dict of per-point arrays
    (lam1, g_H, x_c, seg_id, xi_s, xi_valid)."""
    return {
        "lambda1_teeth": lambda_tooth_stats(res["lam1"], res["seg_id"]),
        "gH_teeth": gH_teeth_stats(res["g_H"], res["seg_id"]),
        "xc_spread": xc_spread_stats(res["x_c"]),
        "couplings": couplings(res["lam1"], res["g_H"], res["xi_s"], res["xi_valid"]),
    }


def analyze(old_m, new_m, old_xi_summary=None, new_xi_summary=None):
    """Cross-target comparisons + pre-registered verdict fields (proposed, UNCERTIFIED)."""
    # (1) lambda1 comparability
    lamO = old_m["lambda1_teeth"]; lamN = new_m["lambda1_teeth"]
    lam_cmp = {
        "OLD_pooled_max_over_median": lamO["pooled_max_over_median"],
        "NEW_pooled_max_over_median": lamN["pooled_max_over_median"],
        "OLD_spike_rate": lamO["spike_rate_vs_segment_median"],
        "NEW_spike_rate": lamN["spike_rate_vs_segment_median"],
        "note": "M = J^T W J is data-independent up to the small W (err-map) shift; the "
                "lambda1 comb is position-set. Comparable max/median + spike-rate => the "
                "two chains sample comparable x_c supports (checked directly in part 4).",
    }

    # (2) THE KEY METRIC: H-teeth suppression (excursion-above-segment-median, robust)
    excO = old_m["gH_teeth"]["excursion_amp_over_segment_median"]
    excN = new_m["gH_teeth"]["excursion_amp_over_segment_median"]
    suppression_excursion = float(excO / excN) if excN > 0 else float("inf")
    ratio_new_over_old_excursion = float(excN / excO) if excO > 0 else float("nan")
    mmO = old_m["gH_teeth"]["max_over_median"]; mmN = new_m["gH_teeth"]["max_over_median"]
    ratio_new_over_old_maxmed = (float(mmN / mmO)
                                 if (np.isfinite(mmO) and np.isfinite(mmN) and mmO != 0)
                                 else float("nan"))

    # xi-calm check for the falsifier conjunction: NEW xi coupling absent AND NEW's
    # smoothed-xi level below OLD's (documented heuristic; grader confirms "samples fast").
    new_rho_lam = new_m["couplings"]["rho_lambda1_xi"]
    new_xi_calm = None
    if old_xi_summary is not None and new_xi_summary is not None:
        new_xi_calm = bool(
            (not np.isfinite(new_rho_lam) or new_rho_lam < NEW_RHO_MAX)
            and (new_xi_summary["median_xi_s"] <= old_xi_summary["median_xi_s"])
        )

    teeth_ge_third = bool(np.isfinite(ratio_new_over_old_excursion)
                          and ratio_new_over_old_excursion >= FALSIFY_TEETH_FRAC)
    falsifier_fired = bool(teeth_ge_third and (new_xi_calm is True))

    key_metric = {
        "OLD_gH_excursion_amp": excO,
        "NEW_gH_excursion_amp": excN,
        "suppression_OLD_over_NEW_excursion": suppression_excursion,
        "ratio_NEW_over_OLD_excursion": ratio_new_over_old_excursion,
        "OLD_gH_max_over_median": mmO,
        "NEW_gH_max_over_median": mmN,
        "ratio_NEW_over_OLD_max_over_median": ratio_new_over_old_maxmed,
        "prediction": {
            "text": f"NEW H-teeth suppressed >= {NEW_SUPPRESS_MIN:g}x vs OLD "
                    "(suppression_OLD_over_NEW_excursion >= 10)",
            "threshold": NEW_SUPPRESS_MIN,
            "measured_suppression": suppression_excursion,
            "meets": bool(np.isfinite(suppression_excursion)
                          and suppression_excursion >= NEW_SUPPRESS_MIN),
        },
        "falsifier": {
            "text": f"NEW H-teeth >= 1/3 OLD (ratio_NEW_over_OLD_excursion >= "
                    f"{FALSIFY_TEETH_FRAC:.3f}) AND xi calm on NEW => teeth don't cause "
                    "slowness => T9/T10 causal reading collapses.",
            "teeth_condition_new_ge_third_old": teeth_ge_third,
            "new_xi_calm_heuristic": new_xi_calm,
            "fired": falsifier_fired,
            "note": "'xi calm on NEW' heuristic = (NEW xi-lambda1 rho < 0.2) AND "
                    "(NEW median smoothed-xi <= OLD's); the grader confirms 'samples "
                    "fast' against the run's own min bulk-ESS (context line).",
        },
    }

    # (3) xi couplings
    cmp_rho = {
        "OLD_rho_lambda1_xi": old_m["couplings"]["rho_lambda1_xi"],
        "NEW_rho_lambda1_xi": new_m["couplings"]["rho_lambda1_xi"],
        "OLD_rho_gH_xi": old_m["couplings"]["rho_gH_xi"],
        "NEW_rho_gH_xi": new_m["couplings"]["rho_gH_xi"],
        "prediction": {
            "text": f"OLD rho(lambda1,xi) >~ {OLD_RHO_MIN:g} (T9) AND NEW rho < {NEW_RHO_MAX:g}",
            "OLD_meets": bool(np.isfinite(old_m["couplings"]["rho_lambda1_xi"])
                              and old_m["couplings"]["rho_lambda1_xi"] >= OLD_RHO_MIN),
            "NEW_meets": bool(np.isfinite(new_m["couplings"]["rho_lambda1_xi"])
                              and new_m["couplings"]["rho_lambda1_xi"] < NEW_RHO_MAX),
        },
    }

    # (4) x_c spread (de-locking alternative)
    rangeO = old_m["xc_spread"]["range_px"]; rangeN = new_m["xc_spread"]["range_px"]
    xc_cmp = {
        "OLD_std_px": old_m["xc_spread"]["std_px"],
        "NEW_std_px": new_m["xc_spread"]["std_px"],
        "OLD_range_px": rangeO, "NEW_range_px": rangeN,
        "OLD_frac_tooth_spacing": old_m["xc_spread"]["frac_tooth_spacing"],
        "NEW_frac_tooth_spacing": new_m["xc_spread"]["frac_tooth_spacing"],
        "NEW_over_OLD_range_ratio": float(rangeN / rangeO) if rangeO > 0 else float("nan"),
        "note": "de-locking alternative: NEW x_c spread >> OLD's => 'protection' = lost "
                "sub-pixel lock (misfit floor), not Hessian smoothing. Both reported.",
    }

    return {
        "part1_lambda1_comparability": lam_cmp,
        "part2_KEY_gH_suppression": key_metric,
        "part3_xi_couplings": cmp_rho,
        "part4_xc_spread": xc_cmp,
    }


# ===========================================================================
# Offline smoke tests (numpy only; no jax/GPU)
# ===========================================================================

def _synth_target(rng, n_seg=N_SEGMENTS, seg_len=SEGMENT_LEN, tooth_amp=0.0,
                  xi_couple=0.0, xc_scale=0.02, xc_center=12.4):
    """A synthetic per-target census dict with a controllable planted H-tooth amplitude,
    a controllable xi<->lambda1 coupling, and a controllable x_c spread.

    seg baseline g_H = 100 + 10*s (per segment), plus one planted tooth of height
    tooth_amp on ONE step of each segment. lambda1 = smooth breathing + (xi_couple)*xi
    ranks so a nonzero xi_couple installs a monotone lambda1-xi relation. x_c ~ Normal
    centered at xc_center with std xc_scale (Gaussian, so std/range are analytic-ish)."""
    seg_ids = np.repeat(np.arange(n_seg), seg_len)
    N = seg_ids.size
    xi_s = np.abs(rng.standard_normal(N)) + 0.05
    xi_valid = np.ones(N, dtype=bool)
    # lambda1: smooth breathing + coupling to xi (monotone in xi if xi_couple>0)
    breathing = 1.0e4 * (1.0 + 0.2 * np.sin(np.linspace(0, 6, N)))
    lam1 = breathing * (1.0 + xi_couple * (xi_s - xi_s.mean()))
    lam1 = np.maximum(lam1, 1.0)
    # g_H: per-segment baseline + a single planted tooth per segment
    g_H = np.empty(N)
    for s in range(n_seg):
        m = seg_ids == s
        g_H[m] = 100.0 + 10.0 * s
    tooth_positions = [s * seg_len + 40 for s in range(n_seg)]
    for p in tooth_positions:
        g_H[p] += tooth_amp
    x_c = xc_center + xc_scale * rng.standard_normal(N)
    return {"lam1": lam1, "g_H": g_H, "x_c": x_c, "seg_id": seg_ids,
            "xi_s": xi_s, "xi_valid": xi_valid}, tooth_positions


def smoke_suppression(verbose=True):
    """Plant BIG teeth in OLD only, tiny teeth in NEW -> the excursion suppression ratio
    must recover OLD_amp / NEW_amp (here 3000 / 30 = 100x, well above the >=10x threshold),
    and NEW's max/median must be ~1 (no teeth)."""
    rng = np.random.default_rng(11)
    old, _ = _synth_target(rng, tooth_amp=3000.0)
    new, _ = _synth_target(rng, tooth_amp=30.0)
    om = per_target_metrics(old); nm = per_target_metrics(new)
    excO = om["gH_teeth"]["excursion_amp_over_segment_median"]
    excN = nm["gH_teeth"]["excursion_amp_over_segment_median"]
    supp = excO / excN
    # planted excursion amps are exactly the tooth heights (baseline is the seg median)
    ok = (abs(excO - 3000.0) < 1e-6 and abs(excN - 30.0) < 1e-6
          and abs(supp - 100.0) < 1e-6 and supp >= NEW_SUPPRESS_MIN)
    if verbose:
        print(f"[smoke] suppression: OLD exc={excO:.3f} (plant 3000), NEW exc={excN:.3f} "
              f"(plant 30), suppression={supp:.3f} (expect 100, >= {NEW_SUPPRESS_MIN:g}) "
              f"-> {'PASS' if ok else 'FAIL'}")
    return {"exc_old": excO, "exc_new": excN, "suppression": supp, "pass": bool(ok)}


def smoke_xi_coupling(verbose=True):
    """Plant a monotone lambda1<->xi coupling in OLD (xi_couple>0) and none in NEW
    (xi_couple=0) -> recover OLD rho high (> OLD_RHO_MIN), NEW rho ~ 0 (< NEW_RHO_MAX)."""
    rng = np.random.default_rng(7)
    old, _ = _synth_target(rng, xi_couple=0.6)
    new, _ = _synth_target(rng, xi_couple=0.0)
    ro = couplings(old["lam1"], old["g_H"], old["xi_s"], old["xi_valid"])["rho_lambda1_xi"]
    rn = couplings(new["lam1"], new["g_H"], new["xi_s"], new["xi_valid"])["rho_lambda1_xi"]
    ok = (ro >= OLD_RHO_MIN) and (abs(rn) < NEW_RHO_MAX)
    if verbose:
        print(f"[smoke] xi coupling: OLD rho={ro:.3f} (>= {OLD_RHO_MIN:g}), NEW rho="
              f"{rn:.3f} (< {NEW_RHO_MAX:g}) -> {'PASS' if ok else 'FAIL'}")
    return {"rho_old": ro, "rho_new": rn, "pass": bool(ok)}


def smoke_xc_spread(verbose=True):
    """x_c spread stats are EXACT on a known array: range = max-min, std = ddof=1 std,
    frac = range / 0.52."""
    x_c = np.array([12.0, 12.1, 12.3, 12.55, 12.9])
    st = xc_spread_stats(x_c)
    exp_range = 0.9
    exp_std = float(np.std(x_c, ddof=1))
    exp_frac = exp_range / TOOTH_SPACING_PX
    ok = (abs(st["range_px"] - exp_range) < 1e-12
          and abs(st["std_px"] - exp_std) < 1e-12
          and abs(st["frac_tooth_spacing"] - exp_frac) < 1e-12)
    if verbose:
        print(f"[smoke] x_c spread: range={st['range_px']:.4f} (expect {exp_range}), "
              f"std={st['std_px']:.4f} (expect {exp_std:.4f}), frac_tooth="
              f"{st['frac_tooth_spacing']:.4f} (expect {exp_frac:.4f}) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return {"range": st["range_px"], "std": st["std_px"],
            "frac": st["frac_tooth_spacing"], "pass": bool(ok)}


def run_smoke():
    print("=== T14b offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_suppression(); b = smoke_xi_coupling(); c = smoke_xc_spread()
    allpass = a["pass"] and b["pass"] and c["pass"]
    print(f"[smoke] overall: {'PASS' if allpass else 'FAIL'}")
    return {"suppression": a, "xi_coupling": b, "xc_spread": c, "pass": bool(allpass)}


# ===========================================================================
# Run-npz loader (OUR-harness format: position + xi + nb/nr scalars)
# ===========================================================================

def load_run(npz_path):
    """Load an OUR-harness run npz. Returns (position (C, nb+nr, dim), xi (C, nb+nr),
    nb, nr). Results phase = the last nr steps of both, aligned 1:1."""
    d = np.load(npz_path, allow_pickle=True)
    for k in ("position", "xi", "nb", "nr"):
        if k not in d.files:
            raise KeyError(f"[T14b] key {k!r} missing in {npz_path}; keys={d.files}")
    position = np.asarray(d["position"], dtype=np.float64)
    xi = np.asarray(d["xi"], dtype=np.float64)
    nb = int(d["nb"]); nr = int(d["nr"])
    if position.ndim != 3:
        raise ValueError(f"[T14b] position must be (C, steps, dim); got {position.shape}")
    C, total, dim = position.shape
    if total != nb + nr:
        raise ValueError(f"[T14b] position steps {total} != nb+nr = {nb+nr}")
    if xi.shape != (C, total):
        raise ValueError(f"[T14b] xi shape {xi.shape} != position (C, nb+nr) "
                         f"({C},{total})")
    return position, xi, nb, nr


def smoothed_xi(xi, nb, nr):
    """Per-chain +/-RMS_HALFWIN RMS smoothing of the RESULTS-phase xi (T9/T10). Returns
    (xi_s (C, nr), valid (C, nr), floor_count, nonfinite_count). Floor entries (~1e-8,
    near-zero energy change) and non-finite are excluded from the validity mask; the
    smoothing itself runs on the raw results xi (their squared contribution is
    negligible), exactly as T10."""
    xi_res = xi[:, -nr:]
    C = xi_res.shape[0]
    valid = valid_mask(xi_res)
    n_floor = int(np.sum(np.asarray(xi_res) <= XI_FLOOR * (1.0 + 1e-6)))
    n_nonfin = int(np.sum(~np.isfinite(xi_res)))
    xi_s = np.vstack([rms_smooth(xi_res[c], RMS_HALFWIN) for c in range(C)])
    valid = valid & np.isfinite(xi_s)
    return xi_s, valid, n_floor, n_nonfin


# ===========================================================================
# Segment selection (seeded; T10 pattern, applied to both runs' own chains)
# ===========================================================================

def make_seg_defs(C, nr, n_seg, seg_len, min_start, seed):
    """Seeded (chain, start) segment coordinates in RESULTS-phase indexing. start is
    >= min_start from the phase start and leaves room for seg_len. The SAME coordinates
    are applied to each run's own chains (a controlled comparison)."""
    if seg_len > nr - min_start:
        raise ValueError(f"[T14b] seg_len {seg_len} > nr-min_start {nr-min_start}")
    rng = np.random.default_rng(seed)
    defs = []
    for i in range(n_seg):
        chain = int(rng.integers(0, C))
        start = int(rng.integers(min_start, nr - seg_len + 1))
        defs.append({"segment": i, "chain": chain, "start": start})
    return defs


# ===========================================================================
# Per-target census: lambda1 + e, g_H, x_c along each segment (its own chains)
# ===========================================================================

def compute_target(model_seq, param_names, position, nb, seg_defs, seg_len,
                   std_z, gate_pts_z, tag, chunk=CHUNK):
    """Walk the seeded segments of THIS target's own results-phase chains, computing at
    every step: lambda1(M_zhat) + top eigenvector e; g_H = e^T H_zhat e (T14's exact
    Hessian-in-zhat, along the LOCAL stiff e); render -> blob_stats -> x_c. Batched
    Jacobians + renders per chunk; Hessians one-by-one (jitted once via ops)."""
    import jax

    ops = build_jax_ops(model_seq, param_names)
    W = np.asarray(ops["W"], dtype=np.float64)
    batched_jac = jax.jit(jax.vmap(ops["jac_render"]))
    render_batch = jax.jit(jax.vmap(ops["render"]))
    hess_logp = ops["hess_logp"]                    # jax.jit(jax.hessian(logp0))

    # hard chi^2 render-path gate against THIS target's OWN log_prob aux (obs/err differ)
    gate_checks, gate_worst = chi2_gate(ops, gate_pts_z, tag=f"T14b/{tag}")

    std_z = np.asarray(std_z, dtype=np.float64)
    lam1_l, gH_l, xc_l, yc_l, A_l, seg_l = [], [], [], [], [], []
    blob_fail = 0
    t0 = time.perf_counter()
    for d in seg_defs:
        chain, start = d["chain"], d["start"]
        seg_z = position[chain, nb + start: nb + start + seg_len]     # (L, dim) results phase
        L = seg_z.shape[0]
        for lo in range(0, L, chunk):
            hi = min(L, lo + chunk)
            Zc = seg_z[lo:hi]
            Js = np.asarray(batched_jac(Zc), dtype=np.float64)         # (nb, 6400, dim)
            Ims = np.asarray(render_batch(Zc), dtype=np.float64)       # (nb, 6400)
            for k in range(hi - lo):
                z = Zc[k]
                M = gn_metric(scale_cols(Js[k], std_z), W)             # standardized GN
                w, V = eig_desc(M)
                e = V[:, 0]                                            # stiff direction
                lam1_l.append(float(w[0]))                            # e^T M e = lambda1
                Hraw = np.asarray(hess_logp(z), dtype=np.float64)      # hessian(logp0)
                Hz = std_z[:, None] * (-Hraw) * std_z[None, :]         # H_zhat = -grad^2 logp
                gH_l.append(float(e @ Hz @ e))
                try:
                    bs = blob_stats(Ims[k])
                    xc_l.append(bs["x_c"]); yc_l.append(bs["y_c"]); A_l.append(bs["A"])
                except Exception as exc:                               # blob guard tripped
                    blob_fail += 1
                    xc_l.append(np.nan); yc_l.append(np.nan); A_l.append(np.nan)
                    if blob_fail <= 3:
                        print(f"[T14b/{tag}] blob_stats failed at seg{d['segment']} "
                              f"step {start+lo+k}: {exc}")
                seg_l.append(d["segment"])
        print(f"[T14b/{tag}] seg{d['segment']} ch{chain} start{start} done "
              f"({time.perf_counter()-t0:5.1f}s)")

    return {
        "tag": tag,
        "gate": {"checks": gate_checks, "worst_relative_error": gate_worst},
        "lam1": np.asarray(lam1_l), "g_H": np.asarray(gH_l),
        "x_c": np.asarray(xc_l), "y_c": np.asarray(yc_l), "A": np.asarray(A_l),
        "seg_id": np.asarray(seg_l), "blob_fail": blob_fail,
    }


def attach_xi(res, seg_defs, seg_len, xi_s, xi_valid):
    """Slice each target's smoothed-xi + validity along the SAME segments (results-phase
    indexing), in the SAME per-point order as compute_target."""
    xs_l, xv_l = [], []
    for d in seg_defs:
        chain, start = d["chain"], d["start"]
        xs_l.append(xi_s[chain, start:start + seg_len])
        xv_l.append(xi_valid[chain, start:start + seg_len])
    res["xi_s"] = np.concatenate(xs_l)
    res["xi_valid"] = np.concatenate(xv_l)
    if res["xi_s"].size != res["lam1"].size:
        raise ValueError(f"[T14b] xi/lambda length mismatch: {res['xi_s'].size} vs "
                         f"{res['lam1'].size}")
    return res


# ===========================================================================
# Plots (single figure, <= 4x4)
# ===========================================================================

def _seg_divider(ax, seg_id):
    """Vertical dividers between concatenated segments (x = cumulative step index)."""
    counts = [int(np.sum(seg_id == s)) for s in np.unique(seg_id)]
    x = 0
    for c in counts[:-1]:
        x += c
        ax.axvline(x - 0.5, color="0.6", ls="--", lw=0.8, zorder=0)


def _symlog_thresh(y):
    a = np.abs(np.asarray(y, float))
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.percentile(a, 10)) if a.size else 1.0


def plot_census(old, new, ana, out_path):
    """4x2 figure: rows = lambda1(M) traces, g_H traces (symlog), xi_s strips (log),
    and (row 4) x_c histogram overlay + xi-vs-gH scatter. cols = OLD, NEW for rows 0-2."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))
    cols = [("OLD (ss2 data)", old, "#8a3b3b"), ("NEW / d' (ss128 data)", new, "#2a6a8a")]

    for c, (title, res, color) in enumerate(cols):
        n = res["lam1"].size
        steps = np.arange(n)
        seg = res["seg_id"]
        # row 0: lambda1(M)
        ax = axes[0, c]
        ax.plot(steps, res["lam1"], "-o", ms=2.0, color=color, lw=0.8)
        ax.set_yscale("log"); _seg_divider(ax, seg)
        ax.set_title(f"{title}: lambda1(M_zhat)  (max/med="
                     f"{res['_lam_maxmed']:.1f})", fontsize=10)
        ax.set_ylabel("lambda1")
        # row 1: g_H (symlog)
        ax = axes[1, c]
        ax.plot(steps, res["g_H"], "-o", ms=2.0, color=color, lw=0.8)
        ax.set_yscale("symlog", linthresh=_symlog_thresh(res["g_H"]))
        _seg_divider(ax, seg)
        ax.set_title(f"{title}: g_H = e^T H_zhat e  (excursion amp="
                     f"{res['_gH_exc']:.3g})", fontsize=10)
        ax.set_ylabel("g_H (std curvature)")
        # row 2: xi_s (log)
        ax = axes[2, c]
        ax.plot(steps, res["xi_s"], "-", color=color, lw=0.8)
        ax.set_yscale("log"); _seg_divider(ax, seg)
        ax.set_title(f"{title}: smoothed xi_s  (rho_lam,xi="
                     f"{res['_rho_lam']:.3f})", fontsize=10)
        ax.set_ylabel("xi_s")
        ax.set_xlabel("census step (2 segments x 128)")

    # row 3 col 0: x_c histogram overlay with pixel-boundary lines
    ax = axes[3, 0]
    xo = old["x_c"][np.isfinite(old["x_c"])]; xn = new["x_c"][np.isfinite(new["x_c"])]
    allx = np.concatenate([xo, xn]) if (xo.size or xn.size) else np.array([0.0, 1.0])
    lo, hi = float(np.min(allx)), float(np.max(allx))
    pad = max(0.05, 0.1 * (hi - lo))
    bins = np.linspace(lo - pad, hi + pad, 40)
    ax.hist(xo, bins=bins, color="#8a3b3b", alpha=0.6,
            label=f"OLD (range={ana['part4_xc_spread']['OLD_range_px']:.3f}px)")
    ax.hist(xn, bins=bins, color="#2a6a8a", alpha=0.6,
            label=f"NEW (range={ana['part4_xc_spread']['NEW_range_px']:.3f}px)")
    for b in np.arange(np.floor(lo - pad) - 0.5, np.ceil(hi + pad) + 0.5, 1.0):
        ax.axvline(b, color="0.8", lw=0.6, zorder=0)
    ax.set_xlabel("counter-image centroid x_c (px; pixel boundaries dotted)")
    ax.set_ylabel("count")
    ax.set_title("x_c posterior spread (de-locking test)", fontsize=10)
    ax.legend(fontsize=7)

    # row 3 col 1: xi_s vs g_H scatter (both targets)
    ax = axes[3, 1]
    for title, res, color in cols:
        m = res["xi_valid"] & np.isfinite(res["g_H"]) & np.isfinite(res["xi_s"])
        ax.scatter(res["xi_s"][m], res["g_H"][m], s=10, c=color, alpha=0.5,
                   label=f"{title.split(' ')[0]} (rho_gH,xi={res['_rho_gH']:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=_symlog_thresh(
        np.concatenate([old["g_H"], new["g_H"]])))
    ax.set_xlabel("smoothed xi_s"); ax.set_ylabel("g_H")
    ax.set_title("xi_s vs g_H (coupling)", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("T14b: curvature along each posterior's OWN chains (OLD ss2 vs NEW/d') "
                 "-- PROPOSED (UNCERTIFIED)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T14b chain-locus curvature census")
    p.add_argument("--old-data-dir", help="systems/sys60 (RE-PINNED ss2 data; OLD model)")
    p.add_argument("--new-sys-dir", help="systems/sys60_ss16data (d'; NEW model via "
                                         "build_modelling_sequence)")
    p.add_argument("--old-run", default=DEFAULT_OLD_RUN,
                   help="OUR-harness npz for OLD chains+xi")
    p.add_argument("--new-run", default=DEFAULT_NEW_RUN,
                   help="OUR-harness npz for NEW chains+xi")
    p.add_argument("--std-ref", default=DEFAULT_STD_REF,
                   help="frozen reference arrays.npz (samples_z) for the COMMON std_z")
    p.add_argument("--out-dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=CHUNK)
    p.add_argument("--smoke", action="store_true",
                   help="run the offline numpy-only smoke tests and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        res = run_smoke()
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "t14b_smoke.json"), "w") as f:
                json.dump(res, f, indent=2)
        if not res["pass"]:
            raise SystemExit("[T14b] smoke tests FAILED")
        return

    for req in ("old_data_dir", "new_sys_dir", "out_dir"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    os.makedirs(args.out_dir, exist_ok=True)

    from common import assert_x64, load_target, compute_diagnostics
    assert_x64()

    # --- ONE common standardization std_z (frozen reference pooled samples) --
    ref = np.load(args.std_ref, allow_pickle=True)
    if "samples_z" not in ref.files:
        raise KeyError(f"[T14b] 'samples_z' missing in std-ref {args.std_ref}; "
                       f"keys={ref.files}")
    ref_sz = np.asarray(ref["samples_z"], dtype=np.float64)      # (C, R, dim)
    dim = ref_sz.shape[-1]
    std_z = np.std(ref_sz.reshape(-1, dim), axis=0, ddof=1)      # ONE common std
    print(f"[T14b] std_z from {args.std_ref} pooled samples_z {ref_sz.shape}; "
          f"std_z[:3]={std_z[:3]}")

    # --- load both runs (chains + xi) ---------------------------------------
    old_pos, old_xi, old_nb, old_nr = load_run(args.old_run)
    new_pos, new_xi, new_nb, new_nr = load_run(args.new_run)
    C, _, dim_o = old_pos.shape
    if dim_o != dim:
        raise ValueError(f"[T14b] OLD run dim {dim_o} != std-ref dim {dim}")
    if new_pos.shape[-1] != dim:
        raise ValueError(f"[T14b] NEW run dim {new_pos.shape[-1]} != std-ref dim {dim}")
    print(f"[T14b] OLD run {old_pos.shape} (nb={old_nb} nr={old_nr}); "
          f"NEW run {new_pos.shape} (nb={new_nb} nr={new_nr})")

    # smoothed xi per run (own columns)
    old_xis, old_xiv, old_nfloor, old_nnf = smoothed_xi(old_xi, old_nb, old_nr)
    new_xis, new_xiv, new_nfloor, new_nnf = smoothed_xi(new_xi, new_nb, new_nr)

    # --- seeded segments (same coordinates applied to both runs' own chains) -
    seg_defs = make_seg_defs(C, min(old_nr, new_nr), N_SEGMENTS, SEGMENT_LEN,
                             MIN_START, args.seed)
    print(f"[T14b] segments (chain,start): "
          f"{[(d['chain'], d['start']) for d in seg_defs]}")

    # --- targets ------------------------------------------------------------
    print("[T14b] === building OLD target (systems/sys60, ss2 data) ===")
    old_seq, _qz, _zc, dim2, names_o = load_target(args.old_data_dir)
    if dim2 != dim:
        raise ValueError(f"[T14b] OLD model dim {dim2} != {dim}")
    print("[T14b] === building NEW target (systems/sys60_ss16data, d') ===")
    new_seq, dim_n, names_n = load_new_target(args.new_sys_dir, supersample=2)
    if dim_n != dim:
        raise ValueError(f"[T14b] NEW model dim {dim_n} != {dim}")
    if list(names_n) != list(names_o):
        raise ValueError(f"[T14b] param-name order differs OLD vs NEW:\n{names_o}\n{names_n}")

    # gate points: first step of each segment + midpoint of segment 0 (3 points)
    def gate_pts(position, nb):
        pts = [position[d["chain"], nb + d["start"]] for d in seg_defs]
        d0 = seg_defs[0]
        pts.append(position[d0["chain"], nb + d0["start"] + SEGMENT_LEN // 2])
        return [np.asarray(z, float) for z in pts[:3]] + [np.asarray(pts[-1], float)]

    old = compute_target(old_seq, names_o, old_pos, old_nb, seg_defs, SEGMENT_LEN,
                         std_z, gate_pts(old_pos, old_nb)[:3], "OLD", args.chunk_size)
    new = compute_target(new_seq, names_n, new_pos, new_nb, seg_defs, SEGMENT_LEN,
                         std_z, gate_pts(new_pos, new_nb)[:3], "NEW", args.chunk_size)

    attach_xi(old, seg_defs, SEGMENT_LEN, old_xis, old_xiv)
    attach_xi(new, seg_defs, SEGMENT_LEN, new_xis, new_xiv)

    # --- per-target metrics + cross-target analysis -------------------------
    old_m = per_target_metrics(old)
    new_m = per_target_metrics(new)
    old_xi_summary = {"median_xi_s": float(np.nanmedian(old["xi_s"])),
                      "p90_xi_s": float(np.nanpercentile(old["xi_s"], 90)),
                      "max_xi_s": float(np.nanmax(old["xi_s"]))}
    new_xi_summary = {"median_xi_s": float(np.nanmedian(new["xi_s"])),
                      "p90_xi_s": float(np.nanpercentile(new["xi_s"], 90)),
                      "max_xi_s": float(np.nanmax(new["xi_s"]))}
    ana = analyze(old_m, new_m, old_xi_summary, new_xi_summary)

    # --- min bulk-ESS per run (context line) --------------------------------
    diag_old = compute_diagnostics(old_pos, old_nr, names_o)
    diag_new = compute_diagnostics(new_pos, new_nr, names_n)
    print(f"[T14b] OLD min bulk-ESS = {diag_old['min_ess']:.1f} "
          f"(param {diag_old['min_ess_param']}); NEW min bulk-ESS = "
          f"{diag_new['min_ess']:.1f} (param {diag_new['min_ess_param']})")

    # stash plotting scalars
    for res, m in ((old, old_m), (new, new_m)):
        res["_lam_maxmed"] = m["lambda1_teeth"]["pooled_max_over_median"]
        res["_gH_exc"] = m["gH_teeth"]["excursion_amp_over_segment_median"]
        res["_rho_lam"] = m["couplings"]["rho_lambda1_xi"]
        res["_rho_gH"] = m["couplings"]["rho_gH_xi"]

    # --- figure -------------------------------------------------------------
    fig_path = os.path.join(args.out_dir, "t14b_census.png")
    plot_census(old, new, ana, fig_path)

    # --- JSON ---------------------------------------------------------------
    def raw(res):
        return {k: np.asarray(res[k]).tolist() for k in
                ("lam1", "g_H", "x_c", "y_c", "A", "seg_id", "xi_s")}

    out = {
        "experiment": "T14b -- curvature along each posterior's OWN chains",
        "status": "proposed (UNCERTIFIED) -- grader inspects artifacts, not this summary",
        "timestamp_utc": _now(),
        "seed": args.seed,
        "inputs": {
            "old_data_dir": os.path.abspath(args.old_data_dir),
            "new_sys_dir": os.path.abspath(args.new_sys_dir),
            "old_run": os.path.abspath(args.old_run),
            "new_run": os.path.abspath(args.new_run),
            "std_ref": os.path.abspath(args.std_ref),
        },
        "design_notes": {
            "locus": "each target's OWN seed-1 results-phase chains (the sampler's actual "
                     "typical set) -- CORRECTS T14, which measured off-typical-set dial curvature.",
            "std_z": "ONE common standardization = per-coordinate ddof=1 std over the FROZEN "
                     "reference results/testsys60/mclmc.stale-20260703T111618 pooled samples_z "
                     "(so OLD/NEW curvatures are directly comparable, as T14).",
            "segments": f"{N_SEGMENTS} x {SEGMENT_LEN} steps; seeded (chain,start), start >= "
                        f"{MIN_START} from phase start; SAME coordinates applied to each run's "
                        "own chains (controlled comparison).",
            "lambda1": "lambda1(M_zhat), M_zhat = gn_metric(scale_cols(J_z, std_z), W); e = its "
                       "top eigenvector (stiff/counter-image direction). [census/T12 convention]",
            "g_H": "e^T H_zhat e, H_zhat = diag(std_z)(-hessian(prob_model.log_prob(z)[0]))"
                   "diag(std_z) -- INCLUDES prior+logdet (smooth at tooth scale). [T14 construction]",
            "x_c": "blob_stats counter-image sub-pixel centroid on the MODEL render (T12).",
            "xi": "xi = energy_change^2/(dim*5e-4)+1e-8; +/-20-step RMS smoothing per chain "
                  "(T9/T10); floor (~1e-8) + non-finite excluded from validity.",
            "blob_fail_handling": "a tripped blob guard records x_c=NaN (excluded from x_c "
                                  "spread + couplings) rather than aborting the census.",
        },
        "std_z": std_z.tolist(),
        "param_names": list(names_o),
        "segments": seg_defs,
        "xi_accounting": {
            "OLD": {"floor_excluded": old_nfloor, "nonfinite_excluded": old_nnf,
                    **old_xi_summary},
            "NEW": {"floor_excluded": new_nfloor, "nonfinite_excluded": new_nnf,
                    **new_xi_summary},
        },
        "gates": {"OLD": old["gate"], "NEW": new["gate"],
                  "note": "each gate reconciles against that target's OWN observed/err map; "
                          "both must pass."},
        "blob_failures": {"OLD": old["blob_fail"], "NEW": new["blob_fail"]},
        "min_bulk_ess_context": {
            "OLD": {"min_ess": diag_old["min_ess"], "min_ess_param": diag_old["min_ess_param"],
                    "max_rhat": diag_old["max_rhat"]},
            "NEW": {"min_ess": diag_new["min_ess"], "min_ess_param": diag_new["min_ess_param"],
                    "max_rhat": diag_new["max_rhat"]},
            "note": "context line only (part 5): the NEW arm samples ~80-110x faster than OLD "
                    "(T13'); confirms 'samples fast' for the falsifier's xi-calm clause.",
        },
        "per_target": {"OLD": old_m, "NEW": new_m},
        "analysis": ana,
        "raw": {"OLD": raw(old), "NEW": raw(new)},
        "figure": os.path.abspath(fig_path),
        "verdict": "proposed (UNCERTIFIED) -- no adjudication (a grader inspects the "
                   "plot/JSON, not this script).",
    }
    out_json = os.path.join(args.out_dir, "t14b_results.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # --- printed summary (restates prereg next to measured) -----------------
    key = ana["part2_KEY_gH_suppression"]
    print("\n=== T14b summary (PROPOSED, UNCERTIFIED) ===")
    print(f"  (1) lambda1 max/med  OLD={ana['part1_lambda1_comparability']['OLD_pooled_max_over_median']:.2f} "
          f"NEW={ana['part1_lambda1_comparability']['NEW_pooled_max_over_median']:.2f}; "
          f"spike-rate OLD={ana['part1_lambda1_comparability']['OLD_spike_rate']:.3f} "
          f"NEW={ana['part1_lambda1_comparability']['NEW_spike_rate']:.3f}")
    print(f"  (2) KEY g_H excursion  OLD={key['OLD_gH_excursion_amp']:.4g} "
          f"NEW={key['NEW_gH_excursion_amp']:.4g}  suppression(OLD/NEW)="
          f"{key['suppression_OLD_over_NEW_excursion']:.2f} "
          f"(predict >= {NEW_SUPPRESS_MIN:g}; meets={key['prediction']['meets']})")
    print(f"      falsifier fired={key['falsifier']['fired']} "
          f"(NEW>=1/3 OLD teeth={key['falsifier']['teeth_condition_new_ge_third_old']}, "
          f"xi_calm={key['falsifier']['new_xi_calm_heuristic']})")
    print(f"  (3) rho(lambda1,xi)  OLD={ana['part3_xi_couplings']['OLD_rho_lambda1_xi']:.3f} "
          f"(predict >= {OLD_RHO_MIN:g}) NEW={ana['part3_xi_couplings']['NEW_rho_lambda1_xi']:.3f} "
          f"(predict < {NEW_RHO_MAX:g})")
    print(f"  (4) x_c range  OLD={ana['part4_xc_spread']['OLD_range_px']:.4f}px "
          f"NEW={ana['part4_xc_spread']['NEW_range_px']:.4f}px "
          f"(NEW/OLD={ana['part4_xc_spread']['NEW_over_OLD_range_ratio']:.2f}; "
          f"tooth spacing {TOOTH_SPACING_PX}px)")
    print(f"  (5) min bulk-ESS  OLD={diag_old['min_ess']:.0f} NEW={diag_new['min_ess']:.0f}")
    print(f"  saved: {out_json}\n         {fig_path}")


if __name__ == "__main__":
    main()
