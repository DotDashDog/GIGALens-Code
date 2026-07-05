"""Preflight probe: decide whether the NEW-arm chi2-gate miss (rel 2.6e-6) is
environment float noise or a model/prior mismatch.

Discriminators, per arm, at the stored MAP z_best:
  1. logp vs the map manifest's best_lp. A wrong prior shifts log-prior by
     O(0.1+) nats; env float noise shifts logp by ~|logp|*1e-6 at most.
  2. |grad chi2_red| in z: translates the chi2 discrepancy into an effective
     z-displacement, to compare against float-noise scales.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample")
from common import load_target  # noqa: E402

SYSDIRS = {
    "old": "systems/carousel_min_old",
    "new": "systems/carousel_min_new",
}
REFS = {
    "old": "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case_oldbij",
    "new": "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case_newbij",
}

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

for arm, sysdir in SYSDIRS.items():
    model_seq, qz, z_center, dim, names = load_target(
        "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/"
        "experiments/why_hard_to_sample/" + sysdir)
    pm = model_seq.prob_model
    man = json.load(open(REFS[arm] + "/map/manifest.json"))
    best_lp = float(man["metadata"]["best_chisq"]), float(man["metadata"]["best_lp"])
    z = jnp.asarray(z_center)[None, :]

    logp, chi2 = pm.log_prob(z)
    logp = float(np.ravel(logp)[0])
    chi2 = float(np.ravel(chi2)[0])
    exp_chi2, exp_lp = best_lp
    print(f"[{arm}] chi2_red={chi2!r} vs manifest {exp_chi2!r} rel={abs(chi2-exp_chi2)/exp_chi2:.3e}")
    print(f"[{arm}] logp    ={logp!r} vs manifest {exp_lp!r} "
          f"abs_diff={logp-exp_lp:+.6e} rel={abs(logp-exp_lp)/abs(exp_lp):.3e}")

    g = jax.grad(lambda zz: jnp.ravel(pm.log_prob(zz[None, :])[1])[0])(jnp.asarray(z_center))
    gn = float(jnp.linalg.norm(g))
    dchi = abs(chi2 - exp_chi2)
    print(f"[{arm}] |grad chi2_red|={gn:.3e}  -> equiv z-displacement {dchi/gn:.3e}")
print("DONE")
