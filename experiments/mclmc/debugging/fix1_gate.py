"""Fix 1 gate (G1') — does the float64 likelihood remove the ulp staircase and the
anchor-3 device cliff?

Three precision configs (x64 is process-global, so one process each):
  float32 : x64 OFF, high_precision OFF                  (baseline)
  full64  : x64 ON,  high_precision OFF                  (everything float64)
  mixed   : x64 ON,  high_precision ON  (float32 basis/state, float64 likelihood)  <- Fix 1

Per config: evaluate logp at bootstrap + the 4 frozen run_a anchors, and logp along a ray
through frozen anchor 0 (to expose the ~0.016 ulp staircase). Dump npz; `analyze` compares.

Usage (run each on the SAME device, serialized):
  python fix1_gate.py dump --config float32 --device gpu
  python fix1_gate.py dump --config full64  --device gpu
  python fix1_gate.py dump --config mixed   --device gpu
  python fix1_gate.py analyze
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import e1_stage_noise as e1  # build_model, load_anchors, DIM

OUT = os.path.join(e1.DIAG_DIR, "fix1")
ULP = 0.016  # float32 ulp of |logp|~1.6e5


def cmd_dump(args):
    import jax
    if args.config in ("full64", "mixed"):
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    os.makedirs(OUT, exist_ok=True)
    model_seq, lens_sim = e1.build_model(25)
    lens_sim.high_precision = (args.config == "mixed")
    pm = model_seq.prob_model
    print(f"[fix1] config={args.config} x64={jax.config.jax_enable_x64} "
          f"high_precision={lens_sim.high_precision} dev={jax.devices()[0]}", flush=True)

    def logp(zv):
        z = jnp.asarray(np.asarray(zv, dtype=np.float64)).reshape(1, e1.DIM)
        return float(np.asarray(pm.log_prob(lens_sim, z)[0]))

    boot = e1.load_anchors("bootstrap", 25)[0]
    frozen = e1.load_anchors("run_a_late", 25)  # (4,17)
    anchors = {"bootstrap": boot, **{f"run_a_late_{i}": frozen[i] for i in range(4)}}

    vals = {k: logp(v) for k, v in anchors.items()}
    for k, v in vals.items():
        print(f"    {k:16s} logp={v:.6f}", flush=True)

    # ray through frozen anchor 0
    rng = np.random.default_rng(0)
    u = rng.standard_normal(e1.DIM); u /= np.linalg.norm(u)
    ts = np.linspace(-1e-5, 1e-5, 400)
    ray = np.array([logp(frozen[0] + t * u) for t in ts])

    np.savez(os.path.join(OUT, f"dump_{args.config}.npz"),
             anchor_names=np.array(list(anchors.keys())),
             anchor_logp=np.array([vals[k] for k in anchors]),
             ts=ts, ray=ray)
    print(f"[fix1] dumped -> dump_{args.config}.npz  (ray unique={len(np.unique(ray))}/400)", flush=True)


def cmd_analyze(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cfgs = {}
    for c in ("float32", "full64", "mixed"):
        p = os.path.join(OUT, f"dump_{c}.npz")
        if os.path.exists(p):
            cfgs[c] = np.load(p, allow_pickle=True)
    if "full64" not in cfgs:
        print("[fix1] need full64 reference; abort"); return
    names = list(cfgs["full64"]["anchor_names"])
    ref = dict(zip(names, cfgs["full64"]["anchor_logp"]))

    summary = {"anchors": {}, "ray_unique_values": {}}
    print(f"\n{'anchor':16s} " + " ".join(f"{c:>14s}" for c in cfgs) + "   |mixed-full64|")
    for i, nm in enumerate(names):
        row = {c: float(cfgs[c]["anchor_logp"][i]) for c in cfgs}
        dmix = abs(row.get("mixed", np.nan) - ref[nm])
        summary["anchors"][nm] = {**row, "mixed_minus_full64": dmix}
        print(f"{nm:16s} " + " ".join(f"{row[c]:14.4f}" for c in cfgs) + f"   {dmix:.3e}")
    for c in cfgs:
        nuq = int(len(np.unique(cfgs[c]["ray"])))
        summary["ray_unique_values"][c] = nuq
    print("\nray unique logp values (400 pts through frozen anchor 0): "
          + ", ".join(f"{c}={summary['ray_unique_values'][c]}" for c in cfgs))

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"float32": "tab:red", "full64": "k", "mixed": "tab:green"}
    for c in cfgs:
        d = cfgs[c]
        ax[0].plot(d["ts"], d["ray"] - np.median(d["ray"]), ".-", ms=3,
                   color=colors.get(c), label=f"{c} (uniq={len(np.unique(d['ray']))})", alpha=0.8)
    ax[0].set_title("logp along ray through frozen anchor 0\n(staircase = few unique values)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("logp - median"); ax[0].legend(fontsize=8)
    x = np.arange(len(names))
    for c in cfgs:
        dd = [abs(float(cfgs[c]["anchor_logp"][i]) - ref[names[i]]) for i in range(len(names))]
        ax[1].bar(x + 0.25 * (list(cfgs).index(c) - 1), dd, 0.25, color=colors.get(c), label=c)
    ax[1].axhline(ULP, color="gray", ls=":", label="1 ulp (f32)")
    ax[1].set_yscale("log"); ax[1].set_title("|logp - full64 reference| per anchor")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fix1_gate.png"), dpi=130)
    print(f"[fix1] plot -> fix1/fix1_gate.png")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--config", required=True,
                                               choices=["float32", "full64", "mixed"])
    d.add_argument("--device", default="gpu"); d.set_defaults(fn=cmd_dump)
    a = sub.add_parser("analyze"); a.set_defaults(fn=cmd_analyze)
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
