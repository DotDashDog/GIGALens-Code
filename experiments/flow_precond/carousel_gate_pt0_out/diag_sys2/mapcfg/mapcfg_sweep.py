"""S2-MAPCFG: MAP optimizer-config sweep on system 2 (1_2_3_4_5_9), PLAIN
(archive-matched, non-adaptive) renderer. See design checkpoint in
docs/logs/carousel-mclmc-sampling.md ("S2-MAPCFG") for the pre-registered
hypothesis / prediction / falsifier.

Tests whether the archive's stored bad MAP (lp=-521150.351, z[37]=-1.8553,
OUTSIDE the sampled posterior range, 30.26 nats below the worst sharp-cluster
MCLMC draw) is a fixable optimizer-config artifact or a structural attractor.

Runs 4 arms of ModellingSequence.MAP directly (no Bridge/MCLMC stages, no
Pipeline/out_dir manifest machinery -- this is a bare MAP-only diagnostic, not
a pipeline run). Writes ONLY under this script's own directory (a NEW dir
under the worktree, never S2.S2_ROOT, the archive).

  A. baseline reproduction : adabelief lr=1e-2, 2000 steps, n_samples=128, seed 42
  B. lower LR              : adabelief lr=1e-3, 6000 steps, n_samples=128, seed 42
  C. decaying LR           : adabelief lr 1e-2 -> 1e-5 (cosine decay), 6000 steps,
                             n_samples=128, seed 42
  D. more restarts         : adabelief lr=1e-2 (archived), 2000 steps,
                             n_samples=512, seed 42

Run (inside the shifter/GPU env, PYTHONPATH set per the gpu-launch-recipe
memory; expects 4 visible GPUs):

    cd <worktree>/experiments/flow_precond/carousel_gate_pt0_out/diag_sys2/mapcfg
    /usr/bin/python3 mapcfg_sweep.py
"""
from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import numpy as np

_OUT_DIR = Path(__file__).resolve().parent

# carousel_model_s2 lives in experiments/flow_precond/; add it to sys.path
# so this script is runnable from its own subdirectory without relying on
# PYTHONPATH already containing that exact directory.
_FLOW_PRECOND_DIR = _OUT_DIR.parent.parent.parent
sys.path.insert(0, str(_FLOW_PRECOND_DIR))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import carousel_model_s2 as S2  # noqa: E402
from gigalens.jax.scene_prob_model import ImageData  # noqa: E402
import gigalens.jax.inference as ginf  # noqa: E402

# --- Archive-matched PLAIN renderer swap (per task instructions). ---
S2.AdaptiveImageData = ImageData

# --- Reference numbers (from the design checkpoint / established facts). ---
ARCHIVE_LP = -521150.351
ARCHIVE_Z37 = -1.8553
FALSIFIER_LP = -521120.07          # worst sharp-cluster MCLMC draw
SHARP_MEAN_LP = -521069.58
COMPACT_MEAN_LP = -520992.89
Z37_SHARP_COMPACT_BOUNDARY = -3.0920
Z37_POSTERIOR_RANGE = (-3.601, -2.262)

_ADABELIEF_HAS_NESTEROV = "nesterov" in inspect.signature(optax.adabelief).parameters


def _adabelief(lr, **kw):
    kwargs = dict(b1=0.95, b2=0.99)
    if _ADABELIEF_HAS_NESTEROV:
        kwargs["nesterov"] = True
    kwargs.update(kw)
    return optax.adabelief(lr, **kwargs)


def classify_z37(z37: float) -> str:
    if z37 < Z37_POSTERIOR_RANGE[0] or z37 > Z37_POSTERIOR_RANGE[1]:
        outside = True
    else:
        outside = False
    basin = "compact" if z37 < Z37_SHARP_COMPACT_BOUNDARY else "sharp"
    return f"{basin}" + (" (OUTSIDE sampled range!)" if outside else "")


ARMS = {
    "A_baseline_lr1e-2_2000": dict(
        optimizer=lambda: _adabelief(1e-2),
        num_steps=2000, n_samples=128, seed=42,
        desc="baseline reproduction: adabelief lr=1e-2, 2000 steps, n=128 (archive config)",
    ),
    "B_lowlr_1e-3_6000": dict(
        optimizer=lambda: _adabelief(1e-3),
        num_steps=6000, n_samples=128, seed=42,
        desc="lower LR: adabelief lr=1e-3, 6000 steps, n=128",
    ),
    "C_decay_1e-2to1e-5_6000": dict(
        optimizer=lambda: _adabelief(
            optax.cosine_decay_schedule(init_value=1e-2, decay_steps=6000, alpha=1e-3)
        ),
        num_steps=6000, n_samples=128, seed=42,
        desc="decaying LR: adabelief cosine-decay 1e-2 -> 1e-5 over 6000 steps, n=128",
    ),
    "D_restarts_n512_2000": dict(
        optimizer=lambda: _adabelief(1e-2),
        num_steps=2000, n_samples=512, seed=42,
        desc="more restarts: adabelief lr=1e-2 (archived), 2000 steps, n=512",
    ),
}


def main():
    selected = sys.argv[1:] or list(ARMS.keys())
    unknown = [t for t in selected if t not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choices are {list(ARMS.keys())}")
    print(f"Selected arms: {selected}")
    print("=" * 78)
    print("S2-MAPCFG -- MAP optimizer-config sweep -- MODEL CARD (pre-compute)")
    print("=" * 78)
    print(f"jax version        : {jax.__version__}")
    devices = jax.devices()
    print(f"jax.devices()       : {devices}")
    print(f"jax.device_count()  : {len(devices)}")
    print(f"jax_enable_x64      : {bool(jax.config.jax_enable_x64)}")
    print(f"out_dir (THIS run)  : {_OUT_DIR}  (NOT the archive)")
    print(f"archive (READ-ONLY) : {S2.S2_ROOT}")
    print(f"_ADABELIEF_HAS_NESTEROV: {_ADABELIEF_HAS_NESTEROV}")

    model_seq, prob_model = S2.build()
    ds0 = prob_model.datasets[0]
    print(f"dataset[0] class    : {type(ds0).__name__}")
    assert type(ds0).__name__ == "ImageData", (
        f"expected plain ImageData after the AdaptiveImageData swap, got "
        f"{type(ds0).__name__} -- refusing to proceed (archive comparability broken)")
    print("ASSERTED: dataset[0] is plain ImageData (archive-matched, non-adaptive).")
    print(f"model.num_free_params: {prob_model.model.num_free_params}")
    print("=" * 78)

    results = {}
    for tag in selected:
        cfg = ARMS[tag]
        n_dev = len(jax.devices())
        n_samples = (cfg["n_samples"] // n_dev) * n_dev
        if n_samples != cfg["n_samples"]:
            print(f"[{tag}] WARNING: n_samples {cfg['n_samples']} not divisible by "
                  f"{n_dev} devices -> library will floor-divide to {n_samples}")
        print(f"\n--- ARM {tag}: {cfg['desc']} ---")
        t0 = time.perf_counter()
        samples, lps, chisqs = model_seq.MAP(
            optimizer=cfg["optimizer"](),
            start=None,
            n_samples=cfg["n_samples"],
            num_steps=cfg["num_steps"],
            seed=cfg["seed"],
            output_type="best_step",
            pbar_interval=200,
        )
        wall_s = time.perf_counter() - t0

        lps_np = np.asarray(lps)
        chisqs_np = np.asarray(chisqs)
        samples_np = np.asarray(samples)
        n = len(lps_np)
        best = int(np.nanargmax(lps_np))
        best_lp = float(lps_np[best])
        best_chisq = float(chisqs_np[best])
        z_best = samples_np[best]
        z37 = float(z_best[37])

        tail_start = int(0.9 * n)
        last10_gain = float(lps_np[-1] - lps_np[tail_start])

        basin = classify_z37(z37)

        results[tag] = dict(
            desc=cfg["desc"], num_steps=cfg["num_steps"], n_samples=cfg["n_samples"],
            seed=cfg["seed"], wall_s=wall_s, n_steps_recorded=n,
            best_step=best, best_lp=best_lp, best_chisq=best_chisq,
            z37=z37, basin=basin, last10_gain=last10_gain,
            lp_hist=lps_np, chisq_hist=chisqs_np, z_best=z_best,
        )

        beats_falsifier = best_lp >= FALSIFIER_LP
        print(f"[{tag}] wall_time_s        = {wall_s:.1f}")
        print(f"[{tag}] best_step/n_steps  = {best}/{n}")
        print(f"[{tag}] best_lp            = {best_lp:.4f}")
        print(f"[{tag}] best_chisq         = {best_chisq:.6f}")
        print(f"[{tag}] last-10% net gain  = {last10_gain:+.4f} nats "
              f"({'still oscillating/negative-net' if last10_gain < 0.05 else 'rising tail'})")
        print(f"[{tag}] z[37]              = {z37:.4f}  -> basin = {basin}")
        print(f"[{tag}] vs archive MAP ({ARCHIVE_LP:.3f}): delta = {best_lp - ARCHIVE_LP:+.3f} nats")
        print(f"[{tag}] vs falsifier bar ({FALSIFIER_LP:.2f}, worst sharp MCLMC draw): "
              f"{'CLEARS (beats typical set)' if beats_falsifier else 'DOES NOT CLEAR'} "
              f"(delta = {best_lp - FALSIFIER_LP:+.3f})")
        print(f"[{tag}] vs sharp-cluster mean ({SHARP_MEAN_LP:.2f}): "
              f"delta = {best_lp - SHARP_MEAN_LP:+.3f}")
        print(f"[{tag}] vs compact-cluster mean ({COMPACT_MEAN_LP:.2f}): "
              f"delta = {best_lp - COMPACT_MEAN_LP:+.3f}")

    # --- Save raw arrays (never into the archive). ---
    npz_path = _OUT_DIR / "mapcfg_sweep_arrays.npz"
    save_dict = {}
    for tag, r in results.items():
        save_dict[f"{tag}__lp_hist"] = r["lp_hist"]
        save_dict[f"{tag}__chisq_hist"] = r["chisq_hist"]
        save_dict[f"{tag}__z_best"] = r["z_best"]
    np.savez(npz_path, **save_dict)
    print(f"\nSaved raw arrays -> {npz_path}")

    # --- Summary table ---
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    hdr = (f"{'arm':28s} {'wall_s':>8s} {'best_lp':>14s} {'z37':>9s} "
           f"{'basin':>10s} {'last10%gain':>12s} {'beats_falsifier':>16s}")
    print(hdr)
    for tag, r in results.items():
        print(f"{tag:28s} {r['wall_s']:8.1f} {r['best_lp']:14.4f} {r['z37']:9.4f} "
              f"{r['basin']:>10s} {r['last10_gain']:12.4f} "
              f"{'YES' if r['best_lp'] >= FALSIFIER_LP else 'no':>16s}")
    print(f"\narchive MAP lp = {ARCHIVE_LP:.4f}  z37 = {ARCHIVE_Z37:.4f}")
    print(f"falsifier bar (worst sharp MCLMC draw) = {FALSIFIER_LP:.4f}")
    print(f"sharp-cluster mean = {SHARP_MEAN_LP:.4f}   compact-cluster mean = {COMPACT_MEAN_LP:.4f}")

    # --- Plot: lp_hist overlay + zoom on last 10%. ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"A_baseline_lr1e-2_2000": "C0", "B_lowlr_1e-3_6000": "C1",
              "C_decay_1e-2to1e-5_6000": "C2", "D_restarts_n512_2000": "C3"}
    for tag, r in results.items():
        lp = r["lp_hist"]
        n = len(lp)
        ax[0].plot(lp, lw=0.9, color=colors.get(tag), label=tag)
        tail_start = int(0.9 * n)
        ax[1].plot(np.arange(tail_start, n), lp[tail_start:], lw=1.1,
                   color=colors.get(tag), label=tag)
    for a in ax:
        a.axhline(ARCHIVE_LP, color="k", ls="--", lw=1.2, label="archive MAP (-521150.351)")
        a.axhline(FALSIFIER_LP, color="red", ls=":", lw=1.4,
                  label="falsifier bar: worst sharp MCLMC draw (-521120.07)")
        a.axhline(SHARP_MEAN_LP, color="darkorange", ls=":", lw=1.0,
                  label="sharp-cluster mean (-521069.58)")
        a.axhline(COMPACT_MEAN_LP, color="purple", ls=":", lw=1.0,
                  label="compact-cluster mean (-520992.89)")
    ax[0].set(title="MAP optimizer-config sweep: lp_hist (system 2, 1_2_3_4_5_9)",
              xlabel="step", ylabel="log posterior (best-across-chains)")
    ax[1].set(title="zoom: last 10% of steps", xlabel="step")
    # dedupe legend
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower right")
    plt.tight_layout()
    plot_path = _OUT_DIR / "mapcfg_sweep_lp_hist.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nSaved plot -> {plot_path}")

    print("\nDONE. (UNCERTIFIED -- report per the design checkpoint, no pass/fail "
          "asserted here by design.)")


if __name__ == "__main__":
    main()
