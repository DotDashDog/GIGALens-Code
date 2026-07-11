#!/usr/bin/env python3
"""Run B analysis: crest-crossing statistic + traces for the arm-initialized
frozen-metric MCLMC run produced by `dspl_arm_init.py --run all`.

Pre-registered analysis (`docs/logs/sample-cosmology-dspl.md`, "Run B: arm-
initialized frozen-metric MCLMC (mechanism falsification)"):
  - crossing = passage between Om0 < 0.163 (arm side) and Om0 > 0.25 (bulk
    side), with >= 50-step dwell on the far side (de-jitters the count; 0.163
    is the baseline run's measured turnaround, 0.25 the T2 excursion
    threshold).
  - falsifier: mean >= 3 crossings/chain in 10000 steps is INCOMPATIBLE with a
    hard soft-barrier (the baseline's ~64 bulk-side approaches made ~8
    excursions/chain with 0 crossings each; >=3 implies per-approach crossing
    probability >~40%, versus the predicted <4.6%).

This script performs NO sampling; it only (a) transforms the saved
unconstrained z-space samples to physical (Om0, w0) via the model's own
bijector (rebuilding the FULL model from `dspl_arm_init.py::build_full_model`
purely for its `.bijector`/`.z_param_names` -- identical pattern to
`def_ratio_grid.py::load_mclmc_cosmo_samples`), and (b) computes/reports the
pre-registered statistic and a diagnostic trace figure.

Per this repo's rigor discipline: the verdict printed here is a PROPOSED
reading, marked UNCERTIFIED -- a grader must inspect `arm_init_traces.png`
and the printed table directly before treating it as a finding (never trust
this script's own summary judgement).

Usage:
    python3 dspl_arm_init_analysis.py [--samples-npz PATH] [--low-thr 0.163]
        [--high-thr 0.25] [--min-dwell 50]

Run in the same environment as `dspl_arm_init.py` (needs `gigalens`/
`gigalens_research` for the bijector only -- no GPU, no simulator, no
sampler -- so the CPU-only invocation in `run_dspl_arm_init.sh toy` style
works fine; see `docs/env_setup.md`).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dspl_arm_init import build_full_model, ARM_SAMPLES_NPZ, ARM_DIR

OUT_TRACES_PNG = os.path.join(ARM_DIR, "arm_init_traces.png")

CREST_OM0 = 0.2   # T2's measured crest location -- reference guide line only, not a threshold.
FALSIFIER_MEAN_CROSSINGS = 3.0


def load_samples(samples_npz=ARM_SAMPLES_NPZ):
    if not os.path.exists(samples_npz):
        raise FileNotFoundError(
            f"{samples_npz} does not exist -- dspl_arm_init.py's Stage 3 "
            "(`--run all`, after grader approval via --confirm-run-b-approved) "
            "has not produced samples yet; nothing to analyze."
        )
    with np.load(samples_npz) as d:
        samples_z = np.asarray(d["samples_z"])   # (n_chains, n_steps, dim)
        idx_om0_saved = int(d["idx_om0"])
        idx_w0_saved = int(d["idx_w0"])
        w0_arm = float(d["w0_arm"])
        nonan = np.asarray(d["nonan"]) if "nonan" in d.files else None
    return samples_z, idx_om0_saved, idx_w0_saved, w0_arm, nonan


def to_physical(samples_z, full_model):
    """z-space (n_chains, n_steps, dim) -> physical (Om0, w0) via the model's own
    bijector, identical pattern to def_ratio_grid.py::load_mclmc_cosmo_samples."""
    n_chains, n_steps, dim = samples_z.shape
    if dim != full_model.num_free_params:
        raise ValueError(
            f"samples_z dim={dim} != rebuilt full_model.num_free_params="
            f"{full_model.num_free_params}; the model construction in "
            "dspl_arm_init.py has drifted since this run's samples were produced."
        )
    flat = jnp.asarray(samples_z.reshape(-1, dim))
    phys = full_model.bijector.forward(list(flat.T))
    idx_om0 = full_model.z_param_names.index("cosmo/Om0")
    idx_w0 = full_model.z_param_names.index("cosmo/w0")
    om0 = np.asarray(phys["cosmo/Om0"]).reshape(n_chains, n_steps)
    w0 = np.asarray(phys["cosmo/w0"]).reshape(n_chains, n_steps)
    return om0, w0, idx_om0, idx_w0


def _zone_of(om0, low_thr, high_thr):
    """-1 = arm/low side, +1 = bulk/high side, 0 = mid/transition band (ambiguous)."""
    zone = np.zeros(om0.shape, dtype=int)
    zone[om0 < low_thr] = -1
    zone[om0 > high_thr] = 1
    return zone


def count_crossings_one_chain(om0_chain, low_thr, high_thr, min_dwell):
    """Pre-registered crossing count for one chain: a hysteresis state machine.

    Maintains a "confirmed side" (-1 arm / +1 bulk), initialized to the first
    definite (non-mid-band) zone the chain visits. Whenever the zone sequence
    departs the confirmed side (goes to the definite opposite side; mid-band
    samples do not by themselves start or continue a departure), it accumulates
    a running "candidate dwell" counter; if that counter reaches `min_dwell`
    consecutive opposite-side samples, ONE crossing is recorded and the
    confirmed side flips. A brief excursion that reverts to the confirmed side
    before `min_dwell` samples elapse resets the candidate counter to zero and
    is NOT counted -- exactly the pre-registered "passage ... with >= min_dwell
    -step dwell on the far side" definition, and de-jitters brief flicker across
    either threshold (this differs from naive run-length-encoding, which would
    also count the RETURN from an unqualified flicker as a separate "crossing"
    simply because the resuming side then dwells a long time -- verified against
    a synthetic bulk/arm/bulk/short-flicker/bulk trace during development).

    Returns (n_crossings, list_of_(side, start_index)) -- `side` is the newly
    confirmed side (+1/-1) and `start_index` is where that qualifying dwell began.
    """
    zone = _zone_of(om0_chain, low_thr, high_thr)
    nz = np.nonzero(zone)[0]
    if len(nz) == 0:
        return 0, []  # chain never entered either definite zone

    confirmed_side = int(zone[nz[0]])
    crossings = []
    candidate_side = 0
    candidate_len = 0
    candidate_start = None
    for i in range(int(nz[0]) + 1, len(zone)):
        z = int(zone[i])
        if z == 0 or z == confirmed_side:
            candidate_side, candidate_len, candidate_start = 0, 0, None
            continue
        if z == candidate_side:
            candidate_len += 1
        else:
            candidate_side, candidate_len, candidate_start = z, 1, i
        if candidate_len >= min_dwell:
            crossings.append((candidate_side, candidate_start))
            confirmed_side = candidate_side
            candidate_side, candidate_len, candidate_start = 0, 0, None
    return len(crossings), crossings


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-npz", default=ARM_SAMPLES_NPZ)
    parser.add_argument("--low-thr", type=float, default=0.163)
    parser.add_argument("--high-thr", type=float, default=0.25)
    parser.add_argument("--min-dwell", type=int, default=50)
    args = parser.parse_args()

    samples_z, idx_om0_saved, idx_w0_saved, w0_arm, nonan = load_samples(args.samples_npz)
    n_chains, n_steps, dim = samples_z.shape
    print(f"[dspl_arm_init_analysis] loaded {args.samples_npz}: "
          f"n_chains={n_chains} n_steps={n_steps} dim={dim} w0_arm={w0_arm:.4f}")
    if nonan is not None:
        print(f"[dspl_arm_init_analysis] nonan (non-rejected step) fraction: "
              f"{nonan.mean():.6f}")

    full_model, *_ = build_full_model()
    om0, w0, idx_om0, idx_w0 = to_physical(samples_z, full_model)
    if idx_om0 != idx_om0_saved or idx_w0 != idx_w0_saved:
        print(f"[dspl_arm_init_analysis] WARNING: recomputed (idx_om0,idx_w0)="
              f"({idx_om0},{idx_w0}) differ from those saved with the run "
              f"({idx_om0_saved},{idx_w0_saved}); z_param_names ordering may have "
              "changed since the run -- trusting the freshly recomputed indices.")

    print(f"\n[dspl_arm_init_analysis] crossing definition: passage between "
          f"Om0<{args.low_thr} (arm) and Om0>{args.high_thr} (bulk), "
          f">= {args.min_dwell}-step dwell on the far side.\n")

    all_crossings = np.zeros(n_chains, dtype=int)
    header = (f"{'chain':>5} {'n_crossings':>11} {'arm_occ_frac':>12} "
              f"{'Om0_min':>8} {'Om0_max':>8} {'w0_min':>8} {'w0_max':>8}")
    print(header)
    print("-" * len(header))
    for c in range(n_chains):
        n_cross, _runs = count_crossings_one_chain(
            om0[c], args.low_thr, args.high_thr, args.min_dwell)
        all_crossings[c] = n_cross
        arm_occ = float(np.mean(om0[c] < args.low_thr))
        print(f"{c:>5} {n_cross:>11} {arm_occ:>12.4f} {om0[c].min():>8.4f} "
              f"{om0[c].max():>8.4f} {w0[c].min():>8.4f} {w0[c].max():>8.4f}")

    mean_crossings = float(all_crossings.mean())
    overall_arm_occ = float(np.mean(om0 < args.low_thr))
    print(f"\n[dspl_arm_init_analysis] mean crossings/chain = {mean_crossings:.3f} "
          f"(per-chain: {all_crossings.tolist()})")
    print(f"[dspl_arm_init_analysis] overall arm-occupancy fraction "
          f"(Om0<{args.low_thr}): {overall_arm_occ:.4f}")

    print("\n[dspl_arm_init_analysis] PROPOSED reading against the pre-registered "
          f"falsifier (mean >= {FALSIFIER_MEAN_CROSSINGS} crossings/chain is "
          "incompatible with a hard soft-barrier) -- UNCERTIFIED, a grader must "
          "inspect the plot + table directly:")
    if mean_crossings >= FALSIFIER_MEAN_CROSSINGS:
        print("  -> FALSIFIER TRIGGERED: free bidirectional mixing across the crest is "
              "indicated; the frozen-metric soft-barrier hypothesis (T2) is NOT "
              "supported by this run as pre-registered.")
    else:
        print("  -> falsifier NOT triggered: consistent with (but does not by itself "
              "PROVE) the soft-barrier mechanism. Blind spots per the design checkpoint: "
              "this tests the BASELINE's frozen metric only (silent on whether arm-local "
              "adaptation would cross), and a failure to cross could also reflect an "
              "undiscovered genuine density defect in the new-API arm rather than a "
              "sampler artifact -- see Run A (free-r2 reparameterization) for an "
              "independent check of that possibility.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    steps = np.arange(n_steps)
    for c in range(n_chains):
        axes[0].plot(steps, om0[c], color=cmap(c % 10), lw=0.6, alpha=0.8, label=f"chain {c}")
        axes[1].plot(steps, w0[c], color=cmap(c % 10), lw=0.6, alpha=0.8)
    axes[0].axhline(args.low_thr, color="k", ls="--", lw=1,
                     label=f"Om0={args.low_thr} (arm bound)")
    axes[0].axhline(args.high_thr, color="k", ls=":", lw=1,
                     label=f"Om0={args.high_thr} (bulk bound)")
    axes[0].axhline(CREST_OM0, color="gray", ls="-", lw=1,
                     label=f"Om0={CREST_OM0} (T2 crest)")
    axes[0].set_ylabel(r"$\Omega_{m,0}$")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[0].set_title("Run B: arm-initialized frozen-metric MCLMC -- Om0/w0 traces "
                       f"({n_chains} chains)")
    axes[1].set_ylabel(r"$w_0$")
    axes[1].set_xlabel("step")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_TRACES_PNG), exist_ok=True)
    fig.savefig(OUT_TRACES_PNG, dpi=150)
    plt.close(fig)
    print(f"\n[dspl_arm_init_analysis] wrote {OUT_TRACES_PNG}")


if __name__ == "__main__":
    main()
