"""E1b — float32 vs float64 GRADIENT audit of the lstsq-marginalized logp.

DIAGNOSIS ONLY. Follow-up to E1/E2 (see diagnosis_2026-06/REPORT.md):
E1 showed the float32 logp VALUE is ulp-accurate at all visited anchors
(chi^2 sits at a least-squares optimum, so solve/coeff error enters the value
only quadratically). E2 showed gram conditioning is benign at visited anchors.
The remaining unmeasured consumer path is the GRADIENT, which the MCLMC kernel
actually uses, and which by the envelope-theorem argument picks up coeff error
at FIRST order (cross term d2chi2/dtheta da . delta_a).

Pre-registered card (2026-06-12):
  Hypothesis: the catastrophic D3 floor (Var(dE)/dim ~ 1.4e5 at run_a_late,
    float32 only) enters through the float32 gradient of logp, not the value.
  Prediction: f32-vs-f64 gradient relative error >= O(1) (norm and/or
    direction) at run_a_late anchors; <= ~1e-3 at bootstrap.
  Falsifier: f32 gradient accurate (rel err <= 1e-2, cos ~ 1) at run_a_late
    => gradients innocent too; mechanism must be inside the kernel's energy
    bookkeeping (next: decompose dE of single kernel steps term by term).

Usage (x64 flag is process-global => one process per precision):
  python e1b_grad_audit.py dump --anchor-class run_a_late --n-max 25 --precision float32
  python e1b_grad_audit.py dump --anchor-class run_a_late --n-max 25 --precision float64
  ...
  python e1b_grad_audit.py analyze
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import e1_stage_noise as e1  # noqa: E402  (reuses sys.path setup, build_model, load_anchors)

DIM = e1.DIM
E1B_DIR = os.path.join(e1.DIAG_DIR, "e1b")


def cmd_dump(args):
    if args.precision == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

    os.makedirs(E1B_DIR, exist_ok=True)
    model_seq, lens_sim = e1.build_model(args.n_max)
    pm = model_seq.prob_model

    def logp(z):
        # pm.log_prob returns (log_prob, ...) with a leading chain axis of 1
        return jnp.asarray(pm.log_prob(lens_sim, z.reshape(1, DIM))[0]).reshape(())

    vg = jax.jit(jax.value_and_grad(logp))

    anchors = e1.load_anchors(args.anchor_class, args.n_max, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)

    for ai, anchor in enumerate(anchors):
        t0 = time.perf_counter()
        # anchor + K small perturbations (same delta scale as E1 mid-range)
        u = rng.standard_normal((args.K, DIM))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        zs = [anchor] + [anchor + 1e-6 * u[k] for k in range(args.K)]
        vals, grads = [], []
        for zv in zs:
            z = jnp.asarray(np.asarray(zv, dtype=np.float64))
            v, g = vg(z)
            vals.append(float(np.asarray(v)))
            grads.append(np.asarray(g, dtype=np.float64))
        fname = os.path.join(
            E1B_DIR,
            f"grads_{args.anchor_class}_nmax{args.n_max}_anchor{ai}_{args.precision}.npz",
        )
        np.savez(fname, anchor=anchor, vals=np.array(vals), grads=np.array(grads))
        g0 = grads[0]
        print(f"[E1b] {args.anchor_class} anchor {ai}: |g|={np.linalg.norm(g0):.6g} "
              f"logp={vals[0]:.8g}  ({time.perf_counter()-t0:.1f}s)", flush=True)


def cmd_analyze(args):
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {}
    rows = []
    for f32name in sorted(glob.glob(os.path.join(E1B_DIR, "grads_*_float32.npz"))):
        f64name = f32name.replace("_float32.npz", "_float64.npz")
        if not os.path.exists(f64name):
            print(f"[E1b] WARN no f64 match for {os.path.basename(f32name)}", flush=True)
            continue
        d32, d64 = np.load(f32name), np.load(f64name)
        g32, g64 = d32["grads"], d64["grads"]            # (K+1, 17)
        n32 = np.linalg.norm(g32, axis=1)
        n64 = np.linalg.norm(g64, axis=1)
        rel = np.linalg.norm(g32 - g64, axis=1) / n64
        cos = np.sum(g32 * g64, axis=1) / (n32 * n64)
        # gradient self-consistency across 1e-6 perturbations (noise within f32)
        g32_spread = np.linalg.norm(g32[1:] - g32[0], axis=1) / np.linalg.norm(g32[0])
        g64_spread = np.linalg.norm(g64[1:] - g64[0], axis=1) / np.linalg.norm(g64[0])
        key = os.path.basename(f32name)[len("grads_"):-len("_float32.npz")]
        summary[key] = {
            "norm_f32_anchor": float(n32[0]),
            "norm_f64_anchor": float(n64[0]),
            "norm_ratio_anchor": float(n32[0] / n64[0]),
            "rel_err_median": float(np.median(rel)),
            "rel_err_max": float(np.max(rel)),
            "cos_median": float(np.median(cos)),
            "cos_min": float(np.min(cos)),
            "f32_grad_spread_median": float(np.median(g32_spread)),
            "f64_grad_spread_median": float(np.median(g64_spread)),
            "logp_f64_anchor": float(d64["vals"][0]),
        }
        rows.append((key, summary[key]))
        print(f"[E1b] {key}: |g|f64={n64[0]:.4g} ratio={n32[0]/n64[0]:.4g} "
              f"rel_err_med={np.median(rel):.3g} cos_med={np.median(cos):.4f} "
              f"f32_spread={np.median(g32_spread):.3g} f64_spread={np.median(g64_spread):.3g}",
              flush=True)

    with open(os.path.join(E1B_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- plot: per-anchor grad rel error + direction error + norms ----
    keys = [k for k, _ in rows]
    colors = {"bootstrap": "tab:blue", "run_a_late": "tab:red",
              "run_b_late": "tab:purple", "prior_far": "tab:green"}

    def col(k):
        for c in colors:
            if k.startswith(c):
                return colors[c]
        return "gray"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(keys))
    axes[0].bar(x, [s["rel_err_median"] for _, s in rows], color=[col(k) for k in keys])
    axes[0].set_yscale("log")
    axes[0].axhline(1e-2, color="k", ls=":", label="falsifier 1e-2")
    axes[0].axhline(1.0, color="r", ls="--", label="prediction O(1)")
    axes[0].set_title("||g32-g64||/||g64|| (median)")
    axes[0].legend()
    axes[1].bar(x, [1 - s["cos_median"] for _, s in rows], color=[col(k) for k in keys])
    axes[1].set_yscale("log")
    axes[1].set_title("1 - cos(g32, g64) (median)")
    axes[2].bar(x - 0.2, [s["norm_f64_anchor"] for _, s in rows], 0.4, label="|g| f64")
    axes[2].bar(x + 0.2, [s["norm_f32_anchor"] for _, s in rows], 0.4, label="|g| f32")
    axes[2].set_yscale("log")
    axes[2].set_title("gradient norm at anchor")
    axes[2].legend()
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=75, fontsize=7, ha="right")
    fig.suptitle("E1b: float32 vs float64 gradient of marginalized logp")
    fig.tight_layout()
    out = os.path.join(E1B_DIR, "e1b_grad_audit.png")
    fig.savefig(out, dpi=130)
    print(f"[E1b] plot -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--anchor-class", required=True)
    d.add_argument("--n-max", type=int, default=25)
    d.add_argument("--precision", choices=["float32", "float64"], required=True)
    d.add_argument("--K", type=int, default=8)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_dump)
    a = sub.add_parser("analyze")
    a.set_defaults(fn=cmd_analyze)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
