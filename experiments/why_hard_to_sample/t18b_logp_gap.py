"""T18b -- discriminator for the P_T18a failure + D3 instrument amendment.

Finding to explain: the 5000-step MAP is a certified local max of logp (D2
Newton gain 0.044 nats) yet its LOG-LIKELIHOOD sits ~6 nats below the bulk
samples' median. Two candidate explanations:
  (a) the z-space joint mode genuinely lives where likelihood is mediocre
      (flat valley + prior/Jacobian dominance) -> D3's loglike basis is the
      wrong instrument; the coordinate-consistent gap uses LOGP.
  (b) the optimizer converged to a non-global local max of logp.
Discriminator: logp (full joint, z-space) at the improved z_best vs the logp
distribution over actual posterior draws. Any draw beating the "mode" => (b).
Also reports the logp-based gap for all three MAPs (instrument amendment
candidate for D3, documented, not silently replacing the registered D3).
"""
import json
import os
import sys

import numpy as np

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARNESS)
from common import assert_x64, load_target  # noqa: E402

_SIM = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests"
REF_OLD_MAP = os.path.join(_SIM, "minimal_case_oldbij", "map", "arrays.npz")
REF_NEW_MAP = os.path.join(_SIM, "minimal_case_newbij", "map", "arrays.npz")
OLD_SAMPLES = os.path.join(_SIM, "minimal_case_oldbij", "mclmc", "arrays.npz")
NEW_SAMPLES = os.path.join(_SIM, "minimal_case_newbij", "mclmc", "arrays.npz")
T18_DIR = os.path.join(HARNESS, "results_carousel", "old", "t18")
IMPROVED_MAP = os.path.join(T18_DIR, "map_arrays.npz")

N_SUB = 512
BATCH = 64
RNG_SEED = 20260703


def logp_batched(pm, z):
    import jax.numpy as jnp
    out = np.empty(z.shape[0])
    for s in range(0, z.shape[0], BATCH):
        e = min(s + BATCH, z.shape[0])
        lp, _ = pm.log_prob(jnp.asarray(z[s:e]))
        out[s:e] = np.asarray(lp, dtype=np.float64).reshape(-1)
    return out


def main():
    assert_x64()
    import jax.numpy as jnp
    rng = np.random.default_rng(RNG_SEED)

    old_ms, *_ = load_target(os.path.join(HARNESS, "systems", "carousel_min_old"))
    new_ms, *_ = load_target(os.path.join(HARNESS, "systems", "carousel_min_new"))
    old_pm, new_pm = old_ms.prob_model, new_ms.prob_model

    res = {}
    for label, pm, map_p, samp_p in [
        ("ref_old_500", old_pm, REF_OLD_MAP, OLD_SAMPLES),
        ("ref_new_500", new_pm, REF_NEW_MAP, NEW_SAMPLES),
        ("improved_5000", old_pm, IMPROVED_MAP, OLD_SAMPLES),
    ]:
        zb = np.asarray(np.load(map_p)["z_best"], dtype=np.float64)
        lp_mode = float(np.asarray(pm.log_prob(jnp.asarray(zb)[None])[0]).reshape(-1)[0])
        sz = np.load(samp_p)["samples_z"].reshape(-1, 14)
        idx = rng.choice(sz.shape[0], N_SUB, replace=False)
        lp_s = logp_batched(pm, sz[idx])
        n_beat = int((lp_s > lp_mode).sum())
        gap_med = lp_mode - float(np.median(lp_s))
        print(f"[t18b] {label:14s} logp(z_best)={lp_mode:.3f}  sample logp "
              f"med={np.median(lp_s):.3f} max={lp_s.max():.3f}  "
              f"LOGP-gap(mode-med)={gap_med:+.2f}  draws beating mode: "
              f"{n_beat}/{N_SUB}")
        res[label] = {"logp_z_best": lp_mode,
                      "sample_logp_median": float(np.median(lp_s)),
                      "sample_logp_max": float(lp_s.max()),
                      "logp_gap_mode_minus_median": gap_med,
                      "n_draws_beating_mode": n_beat, "n_sub": N_SUB}

    imp = res["improved_5000"]
    verdict = ("(b) NON-GLOBAL LOCAL MAX: posterior draws beat the improved "
               "'mode' in logp" if imp["n_draws_beating_mode"] > 0 else
               "(a) GENUINE JOINT MODE: no draw beats it; D3's loglike basis "
               "is the wrong instrument for flat-valley+informative-prior "
               "targets; logp-gap is the coordinate-consistent replacement")
    print(f"[t18b] VERDICT: {verdict}")
    res["verdict"] = verdict
    res["status"] = "proposed (UNCERTIFIED)"
    out = os.path.join(T18_DIR, "t18b_logp_gap.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[t18b] wrote {out}")


if __name__ == "__main__":
    main()
