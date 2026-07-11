#!/usr/bin/env python
"""PT-0 post-hoc ladder design: equal swap-cost spacing from Arm A's measured u draws.

Reads arrays_A_<path>.npz (u: (10, 2, 3000, 16) [beta, init-basin, step, chain]),
computes per-beta sd(u) over the retained window (pooled across chains/groups after
per-chain mean removal is NOT applied -- swap acceptance cares about the full
within-rung spread of u), then builds a ladder beta_0=BETA_MIN..1 with steps
d(beta) * sd(u|beta) ~= TARGET nats (target ~1 gives ~30-45%% adjacent acceptance).

Login-node safe (numpy only). Diagnostic-only: output feeds the PT-0b checkpoint.
"""
import json
import sys

import numpy as np

OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
TARGET = 1.0   # nats of d(beta)*sd(u) per adjacent pair
DISCARD = 1500

for path in ["power", "lik"]:
    try:
        d = np.load(f"{OUT}/arrays_A_{path}.npz")
    except FileNotFoundError:
        print(f"[{path}] arrays not found yet"); continue
    betas = np.asarray(d["betas"]) if "betas" in d else np.geomspace(0.01, 1.0, 10)
    u = np.asarray(d["u"])          # expect (nbeta, 2, steps, nch)
    if u.ndim != 4:
        print(f"[{path}] unexpected u shape {u.shape}"); continue
    uw = u[:, :, DISCARD:, :]
    sd = np.array([np.std(uw[k].reshape(-1)) for k in range(len(betas))])
    med = np.array([np.median(uw[k].reshape(-1)) for k in range(len(betas))])
    print(f"\n[{path}] per-beta sd(u) and median(u):")
    for b, s, m in zip(betas, sd, med):
        print(f"  beta={b:8.4f}  sd(u)={s:12.3f}  med(u)={m:14.2f}")
    # integrate d(beta)*sd(u) with log-linear sd interpolation; place knots at
    # cumulative cost multiples of TARGET
    bgrid = np.geomspace(betas[0], 1.0, 4000)
    sdg = np.exp(np.interp(np.log(bgrid), np.log(betas), np.log(np.maximum(sd, 1e-12))))
    cost = np.concatenate([[0.0], np.cumsum(0.5 * (sdg[1:] + sdg[:-1]) * np.diff(bgrid))])
    total = cost[-1]
    n_rungs = int(np.ceil(total / TARGET)) + 1
    knots = np.interp(np.linspace(0, total, n_rungs), cost, bgrid)
    knots[0], knots[-1] = betas[0], 1.0
    print(f"[{path}] total swap cost = {total:.1f} nats -> equal-cost ladder needs "
          f"{n_rungs} rungs at {TARGET} nat/pair:")
    print("  " + ",".join(f"{b:.6g}" for b in knots))
    json.dump(dict(path=path, betas_measured=betas.tolist(), sd_u=sd.tolist(),
                   med_u=med.tolist(), total_cost_nats=float(total),
                   target_nats_per_pair=TARGET, ladder=knots.tolist()),
              open(f"{OUT}/ladder_design_{path}.json", "w"), indent=1)
    print(f"  -> {OUT}/ladder_design_{path}.json")
