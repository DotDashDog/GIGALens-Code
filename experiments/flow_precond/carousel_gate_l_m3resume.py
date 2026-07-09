#!/usr/bin/env python
"""GATE L M3 contingency runner: chunked multistart MAP with per-chunk saves/resume.

Same pinned M3 config as carousel_gate_l.py (GATE L checkpoint items vi + the
chunking amendment: 8x128 starts, seeds 0-7, adabelief 1e-2 b1 .95 b2 .99 nesterov,
4000 steps, output_type "all", classification on recomputed final lp). Used ONLY if
the main script's allocation expires mid-M3: completed chunks are loaded from
m3_chunk_<i>.npz, missing ones run, results merged into gate_l_summary.json (which
must already hold the M1/M2 blocks from the main run's early dump).
"""
import json
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carousel_model

OUT = os.path.join(HERE, "carousel_gate_l_out")
POCKET_COL, POCKET_THR = 6, -22.35
SEED, DIM = 0, 33
M3_N, M3_STEPS, M3_CHUNK = 1024, 4000, 128
IN_BASIN_NATS = 33.0

summary = json.load(open(os.path.join(OUT, "gate_l_summary.json")))
assert not summary["model_card"]["smoke"], "on-disk summary is a smoke artifact"
lp_star = (summary["M1"]["lp_star"]["M"], summary["M1"]["lp_star"]["P"])

model_seq, prob_model = carousel_model.build()
assert prob_model.z_param_names[POCKET_COL] == "planes/0/mass/1/center_x"
lp_fn = lambda z: prob_model.log_prob(z)[0]
_lp_vec = jax.jit(jax.vmap(lp_fn))


def lp_batch(z, chunk=128):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(len(z))
    for i in range(0, len(z), chunk):
        out[i:i + chunk] = np.asarray(_lp_vec(jnp.asarray(z[i:i + chunk])))
    return out


try:
    m3_opt = optax.adabelief(1e-2, b1=0.95, b2=0.99, nesterov=True)
except TypeError:
    m3_opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)

t0 = time.time()
finals_z, finals_lp_lib, bests_lp, bests_z = [], [], [], []
for ci in range(M3_N // M3_CHUNK):
    cpath = os.path.join(OUT, f"m3_chunk_{ci}.npz")
    if os.path.exists(cpath):
        c = np.load(cpath)
        finals_z.append(c["final_z"]); finals_lp_lib.append(c["final_lp_lib"])
        bests_lp.append(c["best_lp"]); bests_z.append(c["best_z"])
        print(f"[M3] chunk {ci}: loaded from cache", flush=True)
        continue
    s, l, _ = model_seq.MAP(m3_opt, n_samples=M3_CHUNK, num_steps=M3_STEPS,
                            seed=SEED + ci, output_type="all", pbar_interval=0)
    s, l = np.asarray(s), np.asarray(l)
    bs = np.nanargmax(np.where(np.isnan(l), -np.inf, l), axis=1)
    ck = dict(final_z=s[:, -1], final_lp_lib=l[:, -1],
              best_lp=l[np.arange(M3_CHUNK), bs], best_z=s[np.arange(M3_CHUNK), bs])
    np.savez(cpath, **ck)
    finals_z.append(ck["final_z"]); finals_lp_lib.append(ck["final_lp_lib"])
    bests_lp.append(ck["best_lp"]); bests_z.append(ck["best_z"])
    print(f"[M3] chunk {ci}: best lp {np.nanmax(ck['best_lp']):.3f} "
          f"({time.time()-t0:.0f}s elapsed)", flush=True)

final_z = np.concatenate(finals_z)
final_lp_lib = np.concatenate(finals_lp_lib)
best_lp = np.concatenate(bests_lp)
best_z = np.concatenate(bests_z)
final_lp = lp_batch(final_z)


def classify(z, lp):
    pocket = (z[:, POCKET_COL] > POCKET_THR) & (lp >= lp_star[1] - IN_BASIN_NATS)
    main = (z[:, POCKET_COL] <= POCKET_THR) & (lp >= lp_star[0] - IN_BASIN_NATS)
    return pocket, main


pk_f, mn_f = classify(final_z, final_lp)
pk_b, mn_b = classify(best_z, best_lp)
summary["M3"] = dict(
    wall_s=time.time() - t0, n=M3_N, steps=M3_STEPS, chunk=M3_CHUNK,
    chunk_seeds=list(range(SEED, SEED + M3_N // M3_CHUNK)),
    resumed_via="carousel_gate_l_m3resume.py",
    n_nan_final=int(np.isnan(final_lp).sum()),
    lp_recompute_vs_lib_max_abs=float(np.nanmax(np.abs(final_lp - final_lp_lib))),
    final=dict(pocket=int(pk_f.sum()), main=int(mn_f.sum()),
               straggler=int(M3_N - pk_f.sum() - mn_f.sum())),
    best_step_diagnostic_only=dict(pocket=int(pk_b.sum()), main=int(mn_b.sum()),
                                   straggler=int(M3_N - pk_b.sum() - mn_b.sum())),
    best_lp_overall=float(np.nanmax(best_lp)),
    lp_star_ref=dict(M=float(lp_star[0]), P=float(lp_star[1])),
    pipeline_map_manifest_best_lp=-291336.70477837865,
)
print(f"[M3] final: pocket {pk_f.sum()}, main {mn_f.sum()}, "
      f"straggler {M3_N - pk_f.sum() - mn_f.sum()}; "
      f"best lp overall {np.nanmax(best_lp):.3f}", flush=True)
with open(os.path.join(OUT, "gate_l_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
np.savez(os.path.join(OUT, "gate_l_m3_arrays.npz"),
         m3_final_z=final_z, m3_final_lp=final_lp, m3_final_lp_lib=final_lp_lib,
         m3_best_lp=best_lp, m3_best_z6=best_z[:, POCKET_COL])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(final_z[:, POCKET_COL], final_lp, ".", ms=3, alpha=0.5)
ax.axvline(POCKET_THR, color="r", lw=0.7)
for v, c in ((lp_star[0], "C1"), (lp_star[1], "C2")):
    ax.axhline(v - IN_BASIN_NATS, color=c, lw=0.7, ls="--")
ax.set_xlabel(f"final z[{POCKET_COL}]")
ax.set_ylabel("final lp")
ax.set_ylim(lp_star[1] - 400, lp_star[1] + 20)
ax.set_title("M3 multistart finals")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gate_l_m3.png"), dpi=120)
print("DONE (m3resume)", flush=True)
