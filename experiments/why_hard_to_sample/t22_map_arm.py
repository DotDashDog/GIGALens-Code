"""T22 arm C -- reference-matched MAP (500 steps, n_samples=64, seed 42) under
whatever conv precision WHTS_CONV_PRECISION selects. Writes ONLY to
results_carousel/phaseC/t22/ (never touches t18 or reference artifacts).

Registered P-T22c: under conv float64 the MAP shows NO improvement (best logp
still below the posterior draws' median ~-119509.6) => the T18 trap is a
genuine local max / thin-ridge dynamics failure, not conv noise. If it DOES
reach the typical set, the noise-stall story was right after all.
"""
import json
import os
import sys
import time

import numpy as np

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HARNESS)
from common import assert_x64, load_target  # noqa: E402

OUT_DIR = os.path.join(HARNESS, "results_carousel", "phaseC", "t22")
SAMPLE_LOGP_MEDIAN_F32 = -119509.6   # T18b, conv-f32 values (context line)
REF_BEST_LP = -119514.83136782396    # reference 500-step MAP (conv f32)


def main():
    assert_x64()
    import jax
    from gigalens_research.inference_utils.pipeline import _default_map_optimizer

    conv = os.environ.get("WHTS_CONV_PRECISION", "float32")
    model_seq, _qz, _zc, dim, _names = load_target(
        os.path.join(HARNESS, "systems", "carousel_min_old"))
    print(f"[t22 map] conv_precision={conv} devices={len(jax.devices())}")

    t0 = time.perf_counter()
    samples, lps, chisqs = model_seq.MAP(
        optimizer=_default_map_optimizer(), n_samples=64, num_steps=500,
        seed=42, output_type="best_step")
    wall = time.perf_counter() - t0
    lp_hist = np.asarray(lps, dtype=np.float64).reshape(-1)
    best = int(np.nanargmax(lp_hist))
    z_best = np.asarray(samples, dtype=np.float64)[best]
    if z_best.shape != (dim,):
        raise ValueError(f"z_best shape {z_best.shape}")
    best_lp = float(lp_hist[best])

    import jax.numpy as jnp
    pm = model_seq.prob_model
    ll = float(np.asarray(pm.log_like(jnp.asarray(z_best)[None])[0]).reshape(-1)[0])
    print(f"[t22 map] best_lp={best_lp:.4f} (step {best}) loglike={ll:.4f} "
          f"wall={wall:.0f}s")
    print(f"[t22 map] reference conv-f32 best_lp={REF_BEST_LP:.4f}; posterior draws' "
          f"median logp (conv-f32 values) ~{SAMPLE_LOGP_MEDIAN_F32}")
    reached = best_lp > SAMPLE_LOGP_MEDIAN_F32
    print(f"[t22 map] P-T22c (NO improvement) -> {'VIOLATED: MAP reached typical set'
          if reached else 'holds: still below draws'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"map_conv_{conv}.npz"),
             z_best=z_best, lp_hist=lp_hist,
             chisq_hist=np.asarray(chisqs, dtype=np.float64).reshape(-1))
    with open(os.path.join(OUT_DIR, f"map_conv_{conv}.json"), "w") as f:
        json.dump({"conv_precision": conv, "best_lp": best_lp, "best_step": best,
                   "loglike_z_best": ll, "wall_s": wall,
                   "P_T22c_reached_typical_set": bool(reached),
                   "status": "proposed (UNCERTIFIED)"}, f, indent=2)
    print(f"[t22 map] wrote {OUT_DIR}/map_conv_{conv}.npz")


if __name__ == "__main__":
    main()
