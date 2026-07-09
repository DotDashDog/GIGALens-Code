#!/usr/bin/env python
"""Carousel BENCHMARK (plan S5.4, human-approved 2026-07-08): flow-MAMS vs
vanilla MAMS on the carousel dPIE posterior.

Baseline: the EXISTING MAMS64 run (messy_tests/dpie/mams/, 64 x [2000 adapt +
1000 kept]) -- not re-run. Test arm (human-amended): 8 chains x 4000 kept,
burnin 2000 (pinned to the baseline's adaptation length), MAMS in u-space
through the Fv4 A+B flow (cache car_std_r10b14ts0lr0.003).

Win conditions (pre-registered in docs/logs/carousel-mclmc-sampling.md; all
computed on DECODED z, occupancy = z[:,6] > -22.35 post-decode):
  W1a occupancy-ESS >= 20/chain/1000-kept  (sd <= sqrt(p(1-p)/80) at 4000 kept)
  W1b switches/chain/1000-kept: median >= 60 AND min >= 12
  W1c occupancy-indicator plain split-Rhat <= 1.05   (baseline 1.719)
  W2  pooled occupancy in [2%, 15%]
  W3  split-Rhat <= 1.02 on all 33 decoded params; bulk-ESS >= baseline (direction)
  W4  ESS per kept-step per chain >= 3x baseline (fallback normalization --
      baseline gradient count/wall unrecoverable); test-arm wall + flow-eval
      fraction reported
Evidence standard for the W1-pass/W2-fail branch: indicator Rhat <= 1.02,
half-split pooled-occupancy agreement <= 1.5pp, Laplace (5.4%) comparison.

Timing pilot (pre-committed): short MAMS runs post-compile; ABORT if projected
total > 4 GPU-h.

GPU only. Outputs to carousel_benchmark_out/.
"""
import os
import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "carousel_benchmark_out")
os.makedirs(OUT_DIR, exist_ok=True)
FLOW_DIR = os.path.join(HERE, "carousel_gate_f_out")

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

sys.path.insert(0, HERE)
import carousel_model
from gigalens_research.inference import flows
from gigalens_research.inference.mams import MAMS_JIT
from demo_validation import (TransformedProbModel, FlowModelSeq, make_latent_qz,
                             decode_batch, split_rhat_ess)

SEED = 0
N_CHAINS = 8            # human-amended from 64 (2026-07-08)
NUM_BURNIN = 2000       # = baseline adaptation length (diagnostics.npz shapes)
NUM_RESULTS = 4000      # human-amended from 1000 (4x per-chain length)
BUDGET_H = 4.0

DPIE = os.path.join(HERE, "..", "sim_carousel", "messy_tests", "dpie")
POCKET_COL, POCKET_THR = 6, -22.35
FLOW_CACHE = os.path.join(FLOW_DIR, "flow_AB_car_std_r10b14ts0lr0.003.npz")
PHASE_A_LR = 3e-3  # only used to re-derive/verify the cache key


def occupancy_metrics(z, tag):
    """All W1/W2 statistics on decoded z of shape (chains, steps, dim)."""
    occ = z[..., POCKET_COL] > POCKET_THR
    n_chains, n_steps = occ.shape
    per = occ.mean(1)
    p = float(occ.mean())
    sd = float(per.std())
    ess_occ_chain = p * (1 - p) / sd**2 if sd > 0 else float("inf")
    ess_occ_per1000 = ess_occ_chain / (n_steps / 1000)
    sw = np.abs(np.diff(occ.astype(int), axis=1)).sum(1) / (n_steps / 1000)
    # plain split-Rhat on the binary indicator (rank-normalization vacuous)
    halves = occ.reshape(n_chains * 2, n_steps // 2).astype(float)
    m, v = halves.mean(1), halves.var(1, ddof=1)
    W, B = v.mean(), n_steps // 2 * m.var(ddof=1)
    rhat_ind = float(np.sqrt(((n_steps // 2 - 1) / (n_steps // 2) * W
                              + B / (n_steps // 2)) / W)) if W > 0 else float("inf")
    first, second = occ[:, :n_steps // 2].mean(), occ[:, n_steps // 2:].mean()
    out = dict(pooled_occupancy=p, per_chain_occ=per.tolist(),
               per_chain_occ_sd=sd, ess_occ_per_chain=ess_occ_chain,
               ess_occ_per_1000=ess_occ_per1000,
               switches_per_1000=dict(median=float(np.median(sw)),
                                      min=float(sw.min()), max=float(sw.max())),
               indicator_split_rhat=rhat_ind,
               half_split_occ=[float(first), float(second)],
               half_split_agreement_pp=abs(float(first) - float(second)) * 100)
    print(f"[{tag}] occupancy:", json.dumps(out, indent=1))
    return out


def main():
    model_seq, prob_model = carousel_model.build()
    dim = 33
    assert prob_model.z_param_names[POCKET_COL] == "planes/0/mass/1/center_x"

    svi = np.load(os.path.join(DPIE, "svi", "arrays.npz"))
    qz_loc, qz_scale_tril = svi["qz_loc"], svi["qz_scale_tril"]
    z_mams = np.load(os.path.join(DPIE, "mams", "arrays.npz"))["samples_z"]

    # Rebuild the Fv4 whitening EXACTLY as carousel_gate_f.py derived it
    w_all = np.linalg.solve(qz_scale_tril, (z_mams.reshape(-1, dim) - qz_loc).T).T
    sd_w = w_all.std(0)
    qz_scale_tril_std = qz_scale_tril @ np.diag(sd_w)
    spline_range = float(np.ceil(1.1 * np.abs(w_all / sd_w).max()))
    num_bins = int(min(512, np.ceil(spline_range * 48.0 / 35.0)))
    assert (spline_range, num_bins) == (10.0, 14), (spline_range, num_bins)

    key = jax.random.key(SEED + 10)
    sp_init, sp_make = flows.make_whitened_spline_flow(
        key, dim, jnp.asarray(qz_loc), jnp.asarray(qz_scale_tril_std),
        num_layers=6, num_bins=num_bins, spline_range=spline_range,
        hidden_dims=(64, 64), trainable_scale=False)
    cache = np.load(FLOW_CACHE, allow_pickle=True)
    leaves, treedef = jax.tree_util.tree_flatten(sp_init)
    params = jax.tree_util.tree_unflatten(
        treedef, [jnp.asarray(cache[f"leaf_{i}"]) for i in range(len(leaves))])
    assert not bool(cache["diverged"]), "cached flow marked diverged"
    bij = sp_make(params)
    # Flow identity check: reproduce the Fv4 pocket gate number before sampling
    basin = np.load("/pscratch/sd/l/linusu/carousel_diag/basin_slice/basin_slice.npz")
    zPM = jnp.stack([jnp.asarray(basin["zP"]), jnp.asarray(basin["zM"])])
    u_pm = bij.inverse(zPM)
    lq = np.asarray(flows._std_normal_logprob(u_pm)
                    + bij.inverse_log_det_jacobian(zPM, event_ndims=1))
    pocket_ratio = float(lq[0] - lq[1])
    print(f"flow identity check: pocket ratio {pocket_ratio:.3f} "
          f"(Fv4 recorded +1.003; must match to ~1e-3)")
    assert abs(pocket_ratio - 1.0029454473769874) < 1e-3, pocket_ratio

    wrapped = FlowModelSeq(model_seq, TransformedProbModel(prob_model, bij))
    qz_u = make_latent_qz(dim)

    model_card = dict(script=os.path.abspath(__file__), jax=jax.__version__,
                      devices=[str(d) for d in jax.devices()],
                      n_chains=N_CHAINS, burnin=NUM_BURNIN, results=NUM_RESULTS,
                      flow=os.path.basename(FLOW_CACHE), seed=SEED,
                      pocket_ratio_check=pocket_ratio)
    print("MODEL CARD:", json.dumps(model_card, indent=2))

    # PRE-COMMITTED TIMING PILOT: two short runs, difference cancels compile.
    if os.environ.get("BENCH_SKIP_PILOT") != "1":
        def timed(nb, nr):
            t = time.perf_counter()
            np.asarray(MAMS_JIT(wrapped, qz_u, n_hmc=N_CHAINS,
                                num_burnin_steps=nb, num_results=nr,
                                seed=SEED + 99, progress_bar=False))
            return time.perf_counter() - t
        t1 = timed(10, 10)
        t2 = timed(20, 20)
        per_step = max(0.0, (t2 - t1) / 20)
        proj_h = (NUM_BURNIN + NUM_RESULTS) * per_step / 3600 + 0.2
        print(f"PILOT: {per_step:.3f} s/step, projected {proj_h:.2f} h "
              f"(budget {BUDGET_H})")
        if proj_h > BUDGET_H:
            print("PILOT ABORT: projection exceeds budget; re-checkpoint required.")
            sys.exit(2)

    t0 = time.perf_counter()
    u = np.asarray(MAMS_JIT(wrapped, qz_u, n_hmc=N_CHAINS,
                            num_burnin_steps=NUM_BURNIN, num_results=NUM_RESULTS,
                            seed=SEED, progress_bar=False))
    wall = time.perf_counter() - t0
    print(f"flow-MAMS done in {wall:.0f}s")

    t0 = time.perf_counter()
    z = decode_batch(prob_model, bij, u)
    decode_s = time.perf_counter() - t0

    np.savez_compressed(os.path.join(OUT_DIR, "benchmark_arrays.npz"),
                        u=u, z=z, sd_w=sd_w)

    # ---- Metrics (decoded z only) -------------------------------------------
    occ_flow = occupancy_metrics(z, "flow-MAMS 8x4000")
    occ_base = occupancy_metrics(z_mams, "baseline MAMS64")

    rhat_f, ess_f = split_rhat_ess(z)
    rhat_b, ess_b = split_rhat_ess(z_mams)
    ess_f_per1000 = ess_f / (N_CHAINS * NUM_RESULTS / 1000)
    ess_b_per1000 = ess_b / (64 * 1000 / 1000)

    summary = dict(
        model_card=model_card, wall_s=wall, decode_s=decode_s,
        flow_arm=dict(occ=occ_flow, max_rhat=float(rhat_f.max()),
                      min_ess=float(ess_f.min()),
                      min_ess_per_1000_chain_steps=float(ess_f_per1000.min())),
        baseline=dict(occ=occ_base, max_rhat=float(rhat_b.max()),
                      min_ess=float(ess_b.min()),
                      min_ess_per_1000_chain_steps=float(ess_b_per1000.min())),
        gates=dict(
            W1a=dict(value=occ_flow["ess_occ_per_1000"], threshold=20,
                     pass_=bool(occ_flow["ess_occ_per_1000"] >= 20)),
            W1b=dict(median=occ_flow["switches_per_1000"]["median"],
                     min=occ_flow["switches_per_1000"]["min"],
                     pass_=bool(occ_flow["switches_per_1000"]["median"] >= 60
                                and occ_flow["switches_per_1000"]["min"] >= 12)),
            W1c=dict(value=occ_flow["indicator_split_rhat"], threshold=1.05,
                     baseline=occ_base["indicator_split_rhat"],
                     pass_=bool(occ_flow["indicator_split_rhat"] <= 1.05)),
            W2=dict(value=occ_flow["pooled_occupancy"],
                    pass_=bool(0.02 <= occ_flow["pooled_occupancy"] <= 0.15)),
            W3=dict(max_rhat=float(rhat_f.max()),
                    pass_=bool(rhat_f.max() <= 1.02),
                    ess_direction_ok=bool(ess_f_per1000.min()
                                          >= ess_b_per1000.min())),
            W4=dict(ratio=float(ess_f_per1000.min() / ess_b_per1000.min()),
                    threshold=3.0,
                    pass_=bool(ess_f_per1000.min()
                               >= 3.0 * ess_b_per1000.min())),
        ),
        evidence_standard=dict(
            indicator_rhat=occ_flow["indicator_split_rhat"],
            half_split_pp=occ_flow["half_split_agreement_pp"],
            laplace_proxy=0.054),
    )
    with open(os.path.join(OUT_DIR, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("GATES:", json.dumps(summary["gates"], indent=1))

    # ---- Plots before metrics (trace of the pocket column, both arms) --------
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for i in range(N_CHAINS):
        axes[0].plot(z[i, :, POCKET_COL], lw=0.4)
    axes[0].axhline(POCKET_THR, color="k", ls="--", lw=0.8)
    axes[0].set_title(f"flow-MAMS 8x{NUM_RESULTS}: z[:,6] traces")
    for i in range(0, 64, 8):
        axes[1].plot(z_mams[i, :, POCKET_COL], lw=0.4)
    axes[1].axhline(POCKET_THR, color="k", ls="--", lw=0.8)
    axes[1].set_title("baseline MAMS64 (every 8th chain): z[:,6] traces")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pocket_traces.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(N_CHAINS), occ_flow["per_chain_occ"], color="C0",
           label="flow-MAMS")
    ax.axhline(occ_flow["pooled_occupancy"], color="C0", ls="--")
    ax.axhline(0.054, color="k", ls=":", label="Laplace 5.4%")
    ax.set_xlabel("chain"); ax.set_ylabel("pocket occupancy"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "per_chain_occupancy.png"), dpi=140)


if __name__ == "__main__":
    main()
