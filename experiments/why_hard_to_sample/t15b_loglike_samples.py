"""T15b -- instrument correction for T15's F1: per-SAMPLE log-likelihoods.

T15 evaluated log_like at cloud MEANS (basin mean b, bulk mean m). Means of
curved thin-sheet posteriors are off-support points (the T14 lesson), so the
+1284-nat basin advantage could be an off-sheet artifact of the bulk mean.
Corrected instrument: the DISTRIBUTION of log_like over actual posterior draws
in each group. If the basin draws' log_like distribution sits above the bulk
draws', the basin is a genuinely higher-likelihood region (volume/entropy must
then explain why the posterior bulk ignores it); if the distributions overlap
and only the MEANS differ, the 1284 nats was the off-support artifact.

Groups (all thinned to N_PER pts):
  basin    : oldbij chain 0, draws [0:2500]           (old-arm z)
  old_bulk : oldbij chains 1-7, random draws          (old-arm z)
  new      : newbij all chains, random draws          (new-arm z; same physics)
"""
import json
import os
import sys

import numpy as np

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARNESS)
from common import assert_x64, load_target  # noqa: E402

OLD_RUN = ("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/"
           "messy_tests/minimal_case_oldbij/mclmc/arrays.npz")
NEW_RUN = ("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/"
           "messy_tests/minimal_case_newbij/mclmc/arrays.npz")
OUT = os.path.join(HARNESS, "results_carousel", "phaseA", "t15b_loglike_samples.json")
N_PER = 512
BATCH = 64
RNG_SEED = 20260703


def loglike_batched(pm, z):
    import jax.numpy as jnp
    out = np.empty(z.shape[0])
    for s in range(0, z.shape[0], BATCH):
        e = min(s + BATCH, z.shape[0])
        ll, _ = pm.log_like(jnp.asarray(z[s:e]))
        out[s:e] = np.asarray(ll, dtype=np.float64).reshape(-1)
    return out


def qtab(x):
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return {k: float(v) for k, v in zip(
        ["min", "p5", "p25", "p50", "p75", "p95", "max"], q)}


def main():
    assert_x64()
    rng = np.random.default_rng(RNG_SEED)

    old_ms, *_ = load_target(os.path.join(HARNESS, "systems", "carousel_min_old"))
    new_ms, *_ = load_target(os.path.join(HARNESS, "systems", "carousel_min_new"))

    old_sz = np.load(OLD_RUN)["samples_z"]  # (8, 10000, 14)
    new_sz = np.load(NEW_RUN)["samples_z"]

    basin = old_sz[0, 0:2500][rng.choice(2500, N_PER, replace=False)]
    bulk_pool = old_sz[1:8].reshape(-1, old_sz.shape[-1])
    old_bulk = bulk_pool[rng.choice(bulk_pool.shape[0], N_PER, replace=False)]
    new_pool = new_sz.reshape(-1, new_sz.shape[-1])
    new = new_pool[rng.choice(new_pool.shape[0], N_PER, replace=False)]

    res = {}
    res["basin"] = qtab(loglike_batched(old_ms.prob_model, basin))
    res["old_bulk"] = qtab(loglike_batched(old_ms.prob_model, old_bulk))
    res["new"] = qtab(loglike_batched(new_ms.prob_model, new))

    for k, v in res.items():
        print(f"[t15b] {k:9s} " + "  ".join(f"{kk}={vv:.1f}" for kk, vv in v.items()))
    d_med = res["basin"]["p50"] - res["old_bulk"]["p50"]
    overlap = res["basin"]["p5"] < res["old_bulk"]["p95"] and \
        res["old_bulk"]["p5"] < res["basin"]["p95"]
    print(f"[t15b] basin-vs-bulk median delta = {d_med:+.1f} nats; "
          f"5-95% overlap = {overlap}")
    res["delta_median_basin_minus_bulk"] = d_med
    res["overlap_5_95"] = bool(overlap)
    res["n_per_group"] = N_PER
    res["note"] = ("Corrected instrument for T15 F1 (means are off-support on "
                   "curved sheets). proposed (UNCERTIFIED).")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[t15b] wrote {OUT}")


if __name__ == "__main__":
    main()
