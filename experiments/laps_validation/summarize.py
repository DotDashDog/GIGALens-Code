r"""summarize.py -- per-preset cross-seed SUMMARY plots (mean +/- seed spread).

Usage
-----
    python summarize.py <preset>

Reads ``results/<preset>/summary.csv`` and writes ``results/<preset>/summary.png``.

  mscale  : b2_avg (and b2_max) vs M, log-log, with the 1/M reference line --
            the E-F unbiasedness test (slope -1 => pure finite-M noise).
  ksweep  : b2_*_at_switch and switch_index vs k -- the E-B ripeness/cost curve.
  schedC  : steps_to_floor grouped by (schedule, C) -- the cost-to-equilibrate
            discriminator.
  initmode: switch_index and b2_avg_at_switch by cell (warm/cold/off).
  offqz   : F1 off-qz robustness -- per-cell failure rate, b2_max box over seeds,
            b2_max vs (switch_index - steps_to_floor) margin scatter.
  phase2off: F3 -- b2_avg grouped bars, phase2 ON vs OFF, by (schedule, C).

Points are the cross-seed MEAN; error bars are the seed std (ddof=0).
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")


def _load(preset):
    path = os.path.join(RESULTS_DIR, preset, "summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no summary.csv for preset {preset!r}: {path}")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _agg(rows, key_fn, val_key):
    """mean+std of float(row[val_key]) grouped by key_fn(row); sorted by key."""
    g = defaultdict(list)
    for r in rows:
        g[key_fn(r)].append(_f(r[val_key]))
    keys = sorted(g)
    mean = np.array([np.nanmean(g[k]) for k in keys])
    std = np.array([np.nanstd(g[k]) for k in keys])
    return keys, mean, std


def summarize_mscale(rows, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, bed in zip(axes, ("T_iso", "T_ill")):
        br = [r for r in rows if r["target"] == bed]
        if not br:
            ax.set_title(f"{bed} (no runs)"); continue
        for val, c, lab in (("b2_avg", "C0", r"$b^2_{avg}$"),
                            ("b2_max", "C1", r"$b^2_{max}$")):
            Ms, m, s = _agg(br, lambda r: int(_f(r["M"])), val)
            Ms = np.array(Ms, dtype=float)
            ax.errorbar(Ms, m, yerr=s, marker="o", color=c, capsize=3, label=lab)
        Mline = np.array(sorted({int(_f(r["M"])) for r in br}), dtype=float)
        ax.plot(Mline, 1.0 / Mline, "k--", lw=1.0, label="1/M floor")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("M (chains)"); ax.set_ylabel(r"$b^2$")
        ax.set_title(f"E-F unbiasedness: {bed}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def summarize_ksweep(rows, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for val, c, lab in (("b2_avg_at_switch", "C0", r"$b^2_{avg}$@switch"),
                        ("b2_max_at_switch", "C1", r"$b^2_{max}$@switch"),
                        ("b2_avg", "C2", r"final $b^2_{avg}$")):
        ks, m, s = _agg(rows, lambda r: _f(r["switch_k"]), val)
        ax.errorbar(ks, m, yerr=s, marker="o", color=c, capsize=3, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("k"); ax.set_ylabel(r"$b^2$")
    ax.set_title("E-B: residual bias vs k"); ax.legend(fontsize=8)

    ax = axes[1]
    for val, c, lab in (("switch_index", "C3", "switch_index"),
                        ("steps_to_floor", "C4", "steps_to_floor")):
        ks, m, s = _agg(rows, lambda r: _f(r["switch_k"]), val)
        ax.errorbar(ks, m, yerr=s, marker="s", color=c, capsize=3, label=lab)
    ax.set_xlabel("k"); ax.set_ylabel("Phase-1 step")
    ax.set_title("E-B: switch timing / cost vs k"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def summarize_schedC(rows, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    keys, m, s = _agg(rows, lambda r: f"{r['schedule']}/C={_f(r['C']):g}",
                      "steps_to_floor")
    x = np.arange(len(keys))
    ax.bar(x, m, yerr=s, capsize=4, color="C0")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20)
    ax.set_ylabel("steps_to_floor (2/M)")
    ax.set_title("schedC: cost-to-equilibrate by (schedule, C)")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def summarize_initmode(rows, out_png):
    def cell(r):
        return f"{r['init_mode']}/qz={r['qz_mode']}"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, val, lab in ((axes[0], "switch_index", "switch_index"),
                         (axes[1], "b2_avg_at_switch", r"$b^2_{avg}$@switch")):
        keys, m, s = _agg(rows, cell, val)
        x = np.arange(len(keys))
        ax.bar(x, m, yerr=s, capsize=4, color="C0")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20)
        ax.set_title(f"E-C: {lab} by init cell"); ax.set_ylabel(lab)
        if val.startswith("b2"):
            ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def _b(x):
    """Parse a CSV bool field ('True'/'False')."""
    return str(x).strip().lower() in ("true", "1")


def _offqz_cell(r):
    """Cell label for the offqz preset: warm / cold / off / off+persist2."""
    if r["init_mode"] == "cold":
        return "cold"
    if r["qz_mode"] == "off":
        return "off+persist2" if int(_f(r.get("switch_persist", 1))) >= 2 else "off"
    return "warm"


def summarize_offqz(rows, out_png):
    r"""F1: off-qz robustness. (1) per-cell failure rate, (2) b2_max box over
    seeds, (3) b2_max vs margin (switch_index - steps_to_floor) scatter.

    The question: is the off-qz seed1 hard-fail a pattern (systematically higher
    failure rate / worse b2_max), and does persist=2 fix it (small margin => high
    b2 is the hypothesised mechanism, so the scatter is the load-bearing plot)?
    """
    order = ["warm", "cold", "off", "off+persist2"]
    cells = defaultdict(list)
    for r in rows:
        cells[_offqz_cell(r)].append(r)
    present = [c for c in order if c in cells]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) failure rate (fraction b2_success == False) per cell
    ax = axes[0]
    fail = [np.mean([0.0 if _b(r["b2_success"]) else 1.0 for r in cells[c]])
            for c in present]
    n = [len(cells[c]) for c in present]
    x = np.arange(len(present))
    ax.bar(x, fail, color="C3")
    for xi, fr, ni in zip(x, fail, n):
        ax.text(xi, fr, f"{fr:.0%}\n(n={ni})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=15)
    ax.set_ylabel("fraction b2_success = False"); ax.set_ylim(0, 1.05)
    ax.set_title("(1) F1: per-cell failure rate")

    # (2) b2_max distribution over seeds (box + raw points)
    ax = axes[1]
    data = [[_f(r["b2_max"]) for r in cells[c]] for c in present]
    ax.boxplot(data, labels=present, showfliers=False)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.08, 0.08, len(d)),
                   d, s=18, color="C0", alpha=0.6, zorder=3)
    ax.axhline(0.01, color="red", ls="--", lw=1.0, label="success bar 0.01")
    ax.set_yscale("log"); ax.set_ylabel(r"$b^2_{max}$")
    ax.set_title("(2) F1: b2_max over seeds"); ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15)

    # (3) b2_max vs margin = switch_index - steps_to_floor (mechanism test)
    ax = axes[2]
    colors = {"warm": "C0", "cold": "C1", "off": "C3", "off+persist2": "C2"}
    for c in present:
        marg = [_f(r["switch_index"]) - _f(r["steps_to_floor"]) for r in cells[c]]
        b2m = [_f(r["b2_max"]) for r in cells[c]]
        ax.scatter(marg, b2m, s=36, color=colors.get(c, "k"), label=c, alpha=0.8)
    ax.axhline(0.01, color="red", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("margin = switch_index - steps_to_floor")
    ax.set_ylabel(r"$b^2_{max}$")
    ax.set_title("(3) F1: small margin => high b2?"); ax.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def summarize_phase2off(rows, out_png):
    r"""F3: isolate the Phase-2 correction. Grouped b2_avg bars, phase2 ON vs OFF,
    by (schedule, C). Prediction: OFF >> ON, OFF is C/schedule-dependent (the
    unadjusted asymptotic bias), ON collapses to the 1/M floor.
    """
    groups = sorted({(r["schedule"], _f(r["C"])) for r in rows},
                    key=lambda t: (t[0], t[1]))
    labels = [f"{s}/C={c:g}" for s, c in groups]
    x = np.arange(len(groups))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (on, lab, col) in enumerate(((True, "phase2 ON", "C0"),
                                        (False, "phase2 OFF", "C3"))):
        m, s = [], []
        for sched, C in groups:
            vals = [_f(r["b2_avg"]) for r in rows
                    if r["schedule"] == sched and _f(r["C"]) == C
                    and _b(r["phase2_enabled"]) == on]
            m.append(np.nanmean(vals) if vals else np.nan)
            s.append(np.nanstd(vals) if vals else np.nan)
        ax.bar(x + (j - 0.5) * width, m, width, yerr=s, capsize=4,
               color=col, label=lab)
    M = _f(rows[0]["M"]) if rows else 512.0
    ax.axhline(1.0 / M, color="green", ls=":", lw=1.0, label=f"1/M floor ({1.0/M:.1e})")
    ax.axhline(0.01, color="red", ls="--", lw=1.0, label="success bar 0.01")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_yscale("log"); ax.set_ylabel(r"final $b^2_{avg}$")
    ax.set_title("F3: Phase-2 correction (ON vs OFF) by (schedule, C)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def summarize_stiff(rows, out_png):
    r"""P6a stiff beds. Per (target, init, k) cell, phase2 ON vs OFF:
      (1) final b2_avg bars (Phase-2 bias-reduction test) with 1/M + 0.01 lines,
      (2) Phase-2 acceptance vs the 0.70 target (the DIAGONAL-PRECOND test -- can
          bisection reach 0.70 on the rotated bed?),
      (3) k-discrimination: b2_max_at_switch and margin (switch_index -
          steps_to_floor) by k, split by target.
    Points are cross-seed mean; error bars are seed std.
    """
    def cell(r):
        return f"{r['target'].replace('T_','')}/{r['init_mode'][:1]}/k{_f(r['switch_k']):g}"

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # (1) final b2_avg, phase2 ON vs OFF, per cell
    ax = axes[0]
    cells = sorted({cell(r) for r in rows})
    x = np.arange(len(cells)); width = 0.38
    for j, (on, lab, col) in enumerate(((True, "phase2 ON", "C0"),
                                        (False, "phase2 OFF", "C3"))):
        m, s = [], []
        for c in cells:
            vals = [_f(r["b2_avg"]) for r in rows
                    if cell(r) == c and _b(r["phase2_enabled"]) == on]
            m.append(np.nanmean(vals) if vals else np.nan)
            s.append(np.nanstd(vals) if vals else np.nan)
        ax.bar(x + (j - 0.5) * width, m, width, yerr=s, capsize=3, color=col, label=lab)
    M = _f(rows[0]["M"]) if rows else 1024.0
    ax.axhline(1.0 / M, color="green", ls=":", lw=1.0, label=f"1/M ({1.0/M:.1e})")
    ax.axhline(0.01, color="red", ls="--", lw=1.0, label="0.01 bar")
    ax.set_xticks(x); ax.set_xticklabels(cells, rotation=35, ha="right", fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel(r"final $b^2_{avg}$")
    ax.set_title("(1) Phase-2 bias reduction"); ax.legend(fontsize=7)

    # (2) Phase-2 acceptance (ON only) vs 0.70 target -- diagonal-precond test
    ax = axes[1]
    on_rows = [r for r in rows if _b(r["phase2_enabled"])]
    keys, m, s = _agg(on_rows, cell, "accept_final")
    x = np.arange(len(keys))
    colors = ["C1" if "rot" in k else "C2" for k in keys]
    ax.bar(x, m, yerr=s, capsize=3, color=colors)
    ax.axhline(0.70, color="black", ls=":", lw=1.0, label="target 0.70")
    ax.axhspan(0.67, 0.73, color="green", alpha=0.15, label="+/-3% band")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Phase-2 accept (final)")
    ax.set_title("(2) diagonal-precond: accept->0.70?"); ax.legend(fontsize=7)

    # (3) k discrimination: b2_max_at_switch (ripeness at the switch step) by k,
    # per target. If equilibration exceeds the window, loose k=3.0 fires before
    # the floor => b2_max@switch rises with k; if the window still binds, flat.
    ax = axes[2]
    for bed, col in (("T_rot", "C1"), ("T_banana_hi", "C2")):
        br = [r for r in rows if r["target"] == bed]
        if not br:
            continue
        ks, m, s = _agg(br, lambda r: _f(r["switch_k"]), "b2_max_at_switch")
        ax.errorbar(ks, m, yerr=s, marker="o", color=col, capsize=3,
                    label=f"{bed.replace('T_','')} b2max@sw")
    ax.set_yscale("log"); ax.set_xlabel("k"); ax.set_ylabel(r"$b^2_{max}$@switch")
    ax.axhline(2.0 / M, color="green", ls=":", lw=1.0, label="2/M floor")
    ax.set_title("(3) k discrimination (ripeness@switch)"); ax.legend(fontsize=7)

    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


_DISPATCH = {
    "mscale": summarize_mscale, "ksweep": summarize_ksweep,
    "schedC": summarize_schedC, "initmode": summarize_initmode,
    "offqz": summarize_offqz, "phase2off": summarize_phase2off,
    "stiff": summarize_stiff,
}


def summarize(preset):
    rows = _load(preset)
    out_png = os.path.join(RESULTS_DIR, preset, "summary.png")
    fn = _DISPATCH.get(preset)
    if fn is None:
        raise ValueError(f"no summary plot defined for preset {preset!r}")
    fn(rows, out_png)
    print(f"[summarize] wrote {out_png}")
    return out_png


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else "mscale")
