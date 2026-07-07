"""T13' Steps 4-6 -- the re-simulated sys60 (data ss16) model-fidelity 2x2 arms.

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T13'). Consumes d' produced
by t13_resim.py (resim/sys60_ss16/observed_ss16.npz) via the systems/sys60_ss16data
module. Four stages (one process each; standard-config sampler shared byte-identically
with T0/T1 via exp_config.STANDARD):

  --stage pipeline --arm {ss2,ss4}
      Fresh MAP->SVI on d' with the arm's model supersample (2 or 4), replicating
      TestSersic60.ipynb's pipeline config (MAPStage(num_steps=350, n_samples=200),
      SVIStage(num_steps=1000, n_vi=500), pipeline seed 0). Writes map/ svi/ under
      resim/sys60_ss16/arm_<ss>/. Also REPORTS the model-side misfit
      max|m_arm(truth) - m16(truth)|/err; if >= 0.3 sigma it prints a loud note
      recommending an ss8 model arm.

  --stage mclmc --arm {ss2,ss4} --seed N
      Standard 8/2000/2000 MCLMC (exp_config.STANDARD, common.run_standard_mclmc) using
      THIS arm's SVI qz (systems/sys60_ss16data/load_target(qz_arrays=<arm svi>)). Writes
      resim/sys60_ss16/arm_<ss>/t0/t0_seed<N>.npz + prints min bulk-ESS / max rank-Rhat
      (same conventions as run_t0_seed_variance.py, reusing its common.* helpers).

  --stage summary --arm {ss2,ss4}
      After seeds 1-4: writes the per-arm T0-style table + band and restates the T13'
      predictions/falsifiers next to the measured numbers. proposed (UNCERTIFIED).

  --stage comb   (Step 6; comb-identity check -- always the ss=2 model on d')
      One T12 dial scan of lambda1 vs measured counter-image displacement, ss2 model,
      d' data. J is data-independent, so the subgrid comb must be unchanged; a changed
      comb is itself a finding. REUSES t12_flank_crossing.run_scan / chi2_gate verbatim
      (the scan z-points come from the OLD t10 census -- they are just parameter points).
      Reports recovered peak frequency + peak-to-trough vs the T12 values 1.90/px, 30645.

Why Step 6 is implemented here (not by invoking t12 directly): t12_flank_crossing's
load_sys60 calls system.load_target(supersample=...) with NO qz_arrays, but the
sys60_ss16data module REQUIRES qz_arrays (the re-simulated data has no default qz) and
RAISES without it. Rather than weaken that guard or fork t12, --stage comb reuses t12's
scan machinery against the new module (passing the ss2 arm's own qz).

jax / pipeline imports are inside the stage functions so the module imports under a plain
conda python for --smoke (numpy-only checks of the summary aggregation).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

SS_MAP = {"ss2": 2, "ss4": 4}

# --- T13' pre-registered predictions / falsifiers (restated next to measured) ---
OLD_T0_BAND = (11.2, 15.3)          # sys60 standard-config min-ESS band (T0 log entry)
SS4_PREDICT_MIN_ESS = 130.0         # ss4 model: >= 10x the old band (~13 -> >=130)
CLONE_ESS = 1700.0                  # sys60 Gaussian clone reference (T1)
MISFIT_SIGMA_THRESHOLD = 0.3        # >= this => recommend an ss8 model arm
# T12 (original ss2 data) comb reference, for the Step-6 identity check:
T12_PEAK_FREQ = 1.90                # recovered dominant frequency (/px)
T12_PEAK_TO_TROUGH = 30645.0        # top-spike detrended peak-to-trough


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _import_system(data_dir):
    """Import systems/sys60_ss16data/system.py as a module."""
    data_dir = os.path.abspath(data_dir)
    sys_py = os.path.join(data_dir, "system.py")
    if not os.path.isfile(sys_py):
        raise FileNotFoundError(f"[T13-arms] no system.py in data-dir: {data_dir}")
    mod_name = "_t13arms_" + os.path.basename(data_dir).replace(".", "_")
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    spec = importlib.util.spec_from_file_location(mod_name, sys_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm_dir(out_dir, ss):
    return os.path.join(os.path.abspath(out_dir), f"arm_{ss}")


# ===========================================================================
# Stage: pipeline (MAP -> SVI on d') + model-side misfit report
# ===========================================================================

def stage_pipeline(args):
    ss = SS_MAP[args.arm]
    mod = _import_system(args.data_dir)
    from common import assert_x64, git_commit
    assert_x64()

    model_seq, dim, param_names = mod.build_modelling_sequence(supersample=ss)
    print(f"[T13-arms/pipeline] arm={args.arm} (model ss={ss}); dim={dim}")

    from gigalens_research.inference_utils.pipeline import (
        Pipeline, InferenceContext, MAPStage, SVIStage)
    # Pipeline default seed = 0 (== notebook); MAP/SVI settings == TestSersic60.ipynb.
    pipeline = Pipeline(InferenceContext.from_modelling_sequence(model_seq))
    # --map-samples/--svi-nvi default to the notebook values (200/500). The ss4
    # arm OOMs at the notebook batches (84.57 GiB = the same op as ss2's 21.26
    # GiB x4 pixels), so run_t13.sh passes reduced batches for ss4 ONLY. qz is
    # an initializer/adaptation anchor; each arm uses its own qz by design, and
    # the deviation is recorded in the manifest below.
    pipeline.add(MAPStage(num_steps=350, n_samples=args.map_samples))
    pipeline.add(SVIStage(num_steps=1000, n_vi=args.svi_nvi))

    arm_dir = _arm_dir(args.out_dir, ss)
    os.makedirs(arm_dir, exist_ok=True)
    print(f"[T13-arms/pipeline] running MAP->SVI, out_dir={arm_dir}")
    pipeline.run(out_dir=arm_dir, resume=True)
    svi_arrays = os.path.join(arm_dir, "svi", "arrays.npz")
    if not os.path.isfile(svi_arrays):
        raise RuntimeError(f"[T13-arms/pipeline] SVI did not produce {svi_arrays}")
    print(f"[T13-arms/pipeline] SVI qz at {svi_arrays}")

    # --- model-side misfit: max|m_arm(truth) - m16(truth)|/err ----------------
    from t13_resim import load_truth_nested, render_at_truth
    truth = load_truth_nested()
    m_arm, _, rt = render_at_truth(model_seq, param_names, truth)
    f = np.load(mod._RESIM_NPZ)   # the SAME d'/m16/err the module built its Dataset from
    key = "m_hi" if "m_hi" in f.files else "m16"   # ladder amendment: top render
    m16 = np.asarray(f[key], dtype=np.float64).reshape(-1)
    err = np.asarray(f["err_map"], dtype=np.float64).reshape(-1)
    misfit = np.abs(m_arm - m16) / err
    misfit_max = float(np.max(misfit))
    loc = np.unravel_index(int(np.argmax(misfit)), (80, 80))
    print(f"[T13-arms/pipeline] model-side misfit max|m{ss}(truth)-m16|/err = "
          f"{misfit_max:.4f} (at row,col {list(map(int, loc))}; truth round-trip {rt:.2e})")
    recommend_ss8 = misfit_max >= MISFIT_SIGMA_THRESHOLD
    if recommend_ss8:
        print("\n" + "!" * 72)
        print(f"[T13-arms/pipeline] NOTE: model-side misfit {misfit_max:.3f} sigma >= "
              f"{MISFIT_SIGMA_THRESHOLD} at row,col {list(map(int, loc))}. The ss={ss} "
              "MODEL cannot reproduce the ss16 truth to within the noise -- an ss=8 model "
              "arm is RECOMMENDED (per the checkpoint) so the 2x2 is not confounded by "
              "model-vs-data fidelity mismatch.")
        print("!" * 72 + "\n")

    doc = {"experiment": "T13' Step 4 -- MAP->SVI pipeline on d'",
           "status": "proposed (UNCERTIFIED)", "timestamp_utc": _now(),
           "git_commit": git_commit(), "arm": args.arm, "model_supersample": ss,
           "pipeline_config": {"MAPStage": {"num_steps": 350, "n_samples": args.map_samples},
                               "SVIStage": {"num_steps": 1000, "n_vi": args.svi_nvi},
                               "pipeline_seed": 0, "source": "TestSersic60.ipynb",
                               "deviation": (None if (args.map_samples, args.svi_nvi) == (200, 500)
                                             else "REDUCED batches vs notebook (OOM at ss4; logged)")},
           "svi_arrays": svi_arrays,
           "model_side_misfit": {
               "metric": "max|m_arm(truth) - m16(truth)| / err_map",
               "measured_sigma": misfit_max, "loc_row_col": list(map(int, loc)),
               "threshold_recommend_ss8": MISFIT_SIGMA_THRESHOLD,
               "recommend_ss8_arm": bool(recommend_ss8),
               "truth_roundtrip_err": rt}}
    out_json = os.path.join(arm_dir, "t13_pipeline_misfit.json")
    with open(out_json, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[T13-arms/pipeline] wrote {out_json}")


# ===========================================================================
# Stage: mclmc (standard config on d', arm's own SVI qz)
# ===========================================================================

def stage_mclmc(args):
    if args.seed is None:
        raise ValueError("--stage mclmc requires --seed N")
    ss = SS_MAP[args.arm]
    mod = _import_system(args.data_dir)
    from common import (assert_x64, compute_diagnostics, git_commit,
                        print_diagnostics, run_standard_mclmc)
    from exp_config import STANDARD
    assert_x64()

    arm_dir = _arm_dir(args.out_dir, ss)
    svi_arrays = os.path.join(arm_dir, "svi", "arrays.npz")
    model_seq, qz, z_center, dim, param_names = mod.load_target(
        supersample=ss, qz_arrays=svi_arrays)
    print(f"[T13-arms/mclmc] arm={args.arm} ss={ss} seed={args.seed}; qz={svi_arrays}")
    print(f"[T13-arms/mclmc] config = {STANDARD.to_dict()}")

    t0_dir = os.path.join(arm_dir, "t0")
    os.makedirs(t0_dir, exist_ok=True)
    out_npz = os.path.join(t0_dir, f"t0_seed{args.seed}.npz")
    pos = run_standard_mclmc(
        model_seq, qz, STANDARD, args.seed, out_npz,
        target_desc=f"T13' re-simulated sys60 (data ss16), model ss={ss}",
        provenance={"data_dir": os.path.abspath(args.data_dir),
                    "observed": "resim/sys60_ss16/observed_ss16.npz (d'=m16+noise)",
                    "qz": svi_arrays, "arm": args.arm})
    diag = compute_diagnostics(pos, STANDARD.num_results, param_names)
    print_diagnostics(diag, header=f"T13' arm {args.arm} seed {args.seed}")

    diag_json = os.path.join(t0_dir, f"t0_seed{args.seed}_diag.json")
    with open(diag_json, "w") as fh:
        json.dump({"experiment": "T13' Step 5 -- MCLMC arm", "arm": args.arm,
                   "model_supersample": ss, "seed": int(args.seed),
                   "status": "proposed (UNCERTIFIED)", "timestamp_utc": _now(),
                   "git_commit": git_commit(), "config": STANDARD.to_dict(),
                   "min_ess": diag["min_ess"], "min_ess_param": diag["min_ess_param"],
                   "max_rhat": diag["max_rhat"], "max_rhat_param": diag["max_rhat_param"],
                   "npz": os.path.abspath(out_npz), "param_names": param_names,
                   "table": diag["table"]}, fh, indent=2)
    print(f"[T13-arms/mclmc] wrote {out_npz} and {diag_json}")


# ===========================================================================
# Stage: summary (per-arm T0-style band + predictions/falsifiers)
# ===========================================================================

def _collect_seed_diags(t0_dir):
    per_seed = []
    if os.path.isdir(t0_dir):
        for fn in sorted(os.listdir(t0_dir)):
            if fn.startswith("t0_seed") and fn.endswith("_diag.json"):
                with open(os.path.join(t0_dir, fn)) as fh:
                    per_seed.append(json.load(fh))
    per_seed.sort(key=lambda d: d["seed"])
    return per_seed


def _band(per_seed):
    v = np.array([d["min_ess"] for d in per_seed], dtype=float)
    r = np.array([d["max_rhat"] for d in per_seed], dtype=float)
    return {"min_ess_min": float(v.min()), "min_ess_max": float(v.max()),
            "min_ess_ratio": float(v.max() / v.min()) if v.min() > 0 else float("inf"),
            "max_rhat_min": float(r.min()), "max_rhat_max": float(r.max())}


def stage_summary(args):
    ss = SS_MAP[args.arm]
    arm_dir = _arm_dir(args.out_dir, ss)
    per_seed = _collect_seed_diags(os.path.join(arm_dir, "t0"))
    if not per_seed:
        raise RuntimeError(f"[T13-arms/summary] no per-seed diagnostics in "
                           f"{os.path.join(arm_dir, 't0')} -- run --stage mclmc first")
    band = _band(per_seed)

    # Prediction/falsifier text keyed to the arm (restated next to measured numbers).
    if args.arm == "ss2":
        prediction = (f"{{d', model ss2}}: min-ESS within ~the old T0 band "
                      f"{OLD_T0_BAND} (the comb is model-side and survives the data fix)")
        falsifier = ("{d', ss2} fast (min-ESS >> old band) => the comb was data-side "
                     "after all; C-4 mechanism misassigned")
        within = OLD_T0_BAND[0] * 0.5 <= band["min_ess_min"] and \
            band["min_ess_max"] <= OLD_T0_BAND[1] * 3.0
        pred_note = (f"measured band [{band['min_ess_min']:.1f}, {band['min_ess_max']:.1f}]"
                     f" vs old T0 band {OLD_T0_BAND} -> "
                     f"{'consistent (comb survives data fix)' if within else 'OUTSIDE old band'}")
    else:  # ss4
        prediction = (f"{{d', model ss4}}: min-ESS >= 10x the old band (>= "
                      f"{SS4_PREDICT_MIN_ESS:.0f}, plausibly approaching the clone's "
                      f"~{CLONE_ESS:.0f})")
        falsifier = ("{d', ss4} unchanged (min-ESS still ~old band) => the comb was NOT "
                     "the ESS bottleneck; C-4's sampling relevance dies")
        met = band["min_ess_min"] >= SS4_PREDICT_MIN_ESS
        pred_note = (f"measured min-ESS band [{band['min_ess_min']:.1f}, "
                     f"{band['min_ess_max']:.1f}] vs predicted >= {SS4_PREDICT_MIN_ESS:.0f}"
                     f" -> {'MET (>=10x)' if met else 'NOT met (still near old band)'}")

    doc = {"experiment": "T13' Step 5 summary", "status": "proposed (UNCERTIFIED)",
           "timestamp_utc": _now(), "arm": args.arm, "model_supersample": ss,
           "n_seeds": len(per_seed),
           "per_seed": [{"seed": d["seed"], "min_ess": d["min_ess"],
                         "min_ess_param": d["min_ess_param"], "max_rhat": d["max_rhat"],
                         "max_rhat_param": d["max_rhat_param"]} for d in per_seed],
           "band": band, "old_T0_band": list(OLD_T0_BAND),
           "prediction": prediction, "falsifier": falsifier,
           "prediction_vs_measured": pred_note}
    out_json = os.path.join(arm_dir, f"t13_summary_{args.arm}.json")
    with open(out_json, "w") as fh:
        json.dump(doc, fh, indent=2)

    print(f"\n===== T13' arm {args.arm} (model ss={ss}) -- FULL SET (report this) =====")
    for d in per_seed:
        print(f"  seed {d['seed']:>3}: min-ESS={d['min_ess']:.4g} "
              f"({d['min_ess_param']})  max-Rhat={d['max_rhat']:.4f} "
              f"({d['max_rhat_param']})")
    print(f"  band: min-ESS {band['min_ess_min']:.4g} .. {band['min_ess_max']:.4g} "
          f"(ratio {band['min_ess_ratio']:.2f}); max-Rhat "
          f"{band['max_rhat_min']:.4f} .. {band['max_rhat_max']:.4f}")
    print(f"  PREDICTION: {prediction}")
    print(f"  FALSIFIER : {falsifier}")
    print(f"  MEASURED  : {pred_note}")
    print(f"[T13-arms/summary] wrote {out_json} (proposed, UNCERTIFIED)")


# ===========================================================================
# Stage: comb (Step 6 -- ss2 model on d'; reuse t12 scan machinery)
# ===========================================================================

def stage_comb(args):
    if not args.old_t10_dir or not args.old_run_dir:
        raise ValueError("--stage comb requires --old-t10-dir and --old-run-dir "
                         "(the original t10 census + reference-run arrays.npz; the "
                         "scan z-points are just parameter points)")
    mod = _import_system(args.data_dir)
    from common import assert_x64
    assert_x64()
    import jax
    from e1_fisher_survey import build_jax_ops
    from t12_flank_crossing import run_scan, chi2_gate

    # scan z-point = the OLD t10 top census spike (a parameter point near the counter-image)
    spike_list = json.load(open(os.path.join(args.old_t10_dir, "spike_list.json")))
    top = next(s for s in spike_list["spikes"] if s["rank"] == 0)
    arrays = np.load(os.path.join(args.old_run_dir, "arrays.npz"))
    samples_z = np.asarray(arrays["samples_z"], dtype=np.float64)   # (C, R, dim)
    flat = samples_z.reshape(-1, samples_z.shape[-1])
    std_z = np.std(flat, axis=0, ddof=1)                           # run property
    z_top = samples_z[int(top["chain"]), int(top["step"])].copy()
    print(f"[T13-arms/comb] top spike: seg{top['segment']} ch{top['chain']} "
          f"step{top['step']} old-lambda1={top['lambda1']:.3e}")

    # ss=2 model on d' (needs the ss2 arm's SVI qz, though the scan itself doesn't use qz)
    ss2_svi = os.path.join(_arm_dir(args.out_dir, 2), "svi", "arrays.npz")
    model_seq, qz, z_center, dim, param_names = mod.load_target(
        supersample=2, qz_arrays=ss2_svi)
    ops = build_jax_ops(model_seq, param_names)
    ops["render_batch"] = jax.jit(jax.vmap(ops["render"]))
    gate, worst = chi2_gate(ops, [z_top], tag="T13comb/ss2")

    scan = run_scan(model_seq, ops, z_top, std_z, ops["W"], "top spike (d' ss2)")
    fo = scan["oscillation_full"]["fourier"]
    pt = scan["oscillation_full"]["peak_to_trough"]
    peak_freq = fo["recovered_peak_freq"]
    ptr = pt["ratio"]

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # scan figure (reuse plotting-free arrays)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(scan["_disp"], scan["_lam1"], "-o", ms=2.5, color="#8a3b3b")
        ax.set_yscale("log")
        ax.set_xlabel("measured x_c displacement (px)")
        ax.set_ylabel("lambda1 (standardized)")
        ax.set_title(f"T13' Step 6 comb (d', ss2): peak-freq={peak_freq:.2f}/px "
                     f"pt={ptr:.1f} vs T12 {T12_PEAK_FREQ}/px, {T12_PEAK_TO_TROUGH:.0f} "
                     "(PROPOSED/UNCERTIFIED)", fontsize=8)
        fig_path = os.path.join(out_dir, "t13_comb_scan.png")
        fig.tight_layout(); fig.savefig(fig_path, dpi=130); plt.close(fig)
    except Exception as e:
        fig_path = None
        print(f"[T13-arms/comb] plot skipped: {e}")

    doc = {"experiment": "T13' Step 6 -- comb identity check (ss2 model on d')",
           "status": "proposed (UNCERTIFIED)", "timestamp_utc": _now(),
           "chi2_gate_worst_rel": worst,
           "measured": {"recovered_peak_freq_per_px": peak_freq,
                        "peak_to_trough": ptr,
                        "fourier_peak_ratio": fo["fourier_peak_ratio"]},
           "T12_reference_ss2_data": {"peak_freq_per_px": T12_PEAK_FREQ,
                                      "peak_to_trough": T12_PEAK_TO_TROUGH},
           "interpretation": ("J is data-independent, so the subgrid comb must be "
                              "UNCHANGED between the original ss2 data and d'; W shifts "
                              "only via the err map. A materially different peak "
                              "frequency or peak-to-trough is itself a finding."),
           "scan": {k: v for k, v in scan.items() if not k.startswith("_")},
           "plot": os.path.basename(fig_path) if fig_path else None}
    out_json = os.path.join(out_dir, "t13_comb.json")
    with open(out_json, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"\n=== T13' Step 6 comb (PROPOSED/UNCERTIFIED) ===")
    print(f"  recovered peak freq = {peak_freq:.3f}/px   (T12 ss2 data: {T12_PEAK_FREQ})")
    print(f"  peak-to-trough      = {ptr:.1f}          (T12 ss2 data: {T12_PEAK_TO_TROUGH:.0f})")
    print(f"[T13-arms/comb] wrote {out_json}")


# ===========================================================================
# Offline smoke (numpy-only: summary band aggregation + prediction classing)
# ===========================================================================

def run_smoke():
    print("=== T13-arms offline smoke (numpy only) ===")
    # synthetic ss2-like per-seed set inside the old band -> ratio + within check
    ps2 = [{"seed": s, "min_ess": v, "max_rhat": 1.5,
            "min_ess_param": "x", "max_rhat_param": "x"}
           for s, v in zip([1, 2, 3, 4], [12.8, 11.7, 11.2, 15.3])]
    b2 = _band(ps2)
    ok2 = abs(b2["min_ess_min"] - 11.2) < 1e-9 and abs(b2["min_ess_max"] - 15.3) < 1e-9 \
        and abs(b2["min_ess_ratio"] - 15.3 / 11.2) < 1e-6
    # synthetic ss4-like set (fast) -> min >= 130 predicted
    ps4 = [{"seed": s, "min_ess": v, "max_rhat": 1.01,
            "min_ess_param": "x", "max_rhat_param": "x"}
           for s, v in zip([1, 2, 3, 4], [1650, 1420, 1710, 1580])]
    b4 = _band(ps4)
    ok4 = b4["min_ess_min"] >= SS4_PREDICT_MIN_ESS
    okmap = SS_MAP == {"ss2": 2, "ss4": 4}
    ok = bool(ok2 and ok4 and okmap)
    print(f"[smoke] ss2 band {b2['min_ess_min']:.1f}-{b2['min_ess_max']:.1f} "
          f"ratio {b2['min_ess_ratio']:.3f} -> {'PASS' if ok2 else 'FAIL'}")
    print(f"[smoke] ss4 band min {b4['min_ess_min']:.0f} >= {SS4_PREDICT_MIN_ESS:.0f} "
          f"-> {'PASS' if ok4 else 'FAIL'}")
    print(f"[smoke] arm map {SS_MAP} -> {'PASS' if okmap else 'FAIL'}")
    print(f"[smoke] overall: {'PASS' if ok else 'FAIL'}")
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T13' arms (Steps 4-6)")
    p.add_argument("--map-samples", type=int, default=200)
    p.add_argument("--svi-nvi", type=int, default=500)
    p.add_argument("--arm", choices=list(SS_MAP.keys()),
                   help="model supersample arm (ss2=2, ss4=4)")
    p.add_argument("--stage", choices=["pipeline", "mclmc", "summary", "comb"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", help="resim/sys60_ss16 (arm dirs created under here)")
    p.add_argument("--data-dir", help="systems/sys60_ss16data")
    p.add_argument("--old-t10-dir", help="[comb] original t10 dir (spike_list.json)")
    p.add_argument("--old-run-dir",
                   default="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc",
                   help="[comb] original reference-run dir with arrays.npz (samples_z)")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        if not run_smoke():
            raise SystemExit("[T13-arms] smoke FAILED")
        return
    for req in ("stage", "out_dir", "data_dir"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required")
    if args.stage in ("pipeline", "mclmc", "summary") and args.arm is None:
        raise ValueError(f"--stage {args.stage} requires --arm")

    if args.stage == "pipeline":
        stage_pipeline(args)
    elif args.stage == "mclmc":
        stage_mclmc(args)
    elif args.stage == "summary":
        stage_summary(args)
    elif args.stage == "comb":
        stage_comb(args)


if __name__ == "__main__":
    main()
