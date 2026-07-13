#!/usr/bin/env python3
"""Pre-run gate for Run C (ratio-coordinates grouped prior) — cheap, CPU-safe.

Verifies, at N_DRAWS matched prior draws, that the grouped-prior model is the
SAME statistical model as the baseline scalar-prior model, differing only in
sampling coordinates (the structural claim behind the Run C design checkpoint
in `docs/logs/sample-cosmology-dspl.md`):

  gate 1 (matched likelihood): log_like at identical physical parameters
         agrees between the two ProbModels to rel <= 1e-8 (float64; Run A's
         gate standard — observed there at 1.5e-14).
  gate 2 (prior density): prior.log_prob at identical physical parameters
         agrees to abs <= 1e-10 (both are the same uniform box + nuisances).
  gate 3 (Jacobian): the grouped bijector's forward_log_det_jacobian matches
         the numeric slogdet of its jacrev to abs <= 1e-8 (real u_fn, not the
         unit tests' analytic stand-in).
  gate 4 (round-trip): forward(inverse(theta)) recovers (Om0, w0) draws to
         abs <= 1e-9 (the bisection is converged, not approximately so).

Writes `results/sample_cosmology/dspl_ratio_coords/ratio_coords_gate.json`;
`dspl_ratio_coords.py --run` refuses to launch unless it exists with
all_passed=true. Run inside the canonical container (docs/env_setup.md);
JAX_PLATFORMS=cpu is fine and keeps this off the GPU queue.
"""
from __future__ import annotations

import json
import os

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import dspl_arm_init as arm
import dspl_ratio_coords as rc

N_DRAWS = 64
SEED = 7

REL_TOL_LOGLIKE = 1e-8
ABS_TOL_LOGPRIOR = 1e-10
ABS_TOL_FLDJ = 1e-8
ABS_TOL_ROUNDTRIP = 1e-9

GROUP_KEY = "cosmo/Om0|cosmo/w0"


def build_models():
    base_model, base_lens, base_s1, base_s2 = arm.build_full_model()
    grp_model, grp_lens, grp_s1, grp_s2, grouped = rc.build_grouped_model()

    img1, img2, sim_config = arm.make_observed_images(
        base_model, base_lens, base_s1, base_s2, data_seed=0)

    base_pm = rc.make_prob_model(
        base_model, base_model.planes[1].light[0], base_model.planes[2].light[0],
        img1, img2, sim_config)
    grp_pm = rc.make_prob_model(
        grp_model, grp_model.planes[1].light[0], grp_model.planes[2].light[0],
        img1, img2, sim_config)
    return base_model, grp_model, base_pm, grp_pm, grouped


def matched_draws(base_model, n=N_DRAWS, seed=SEED):
    """Draw physical parameters from the BASELINE prior; return the same values
    as (baseline unique-key dict, grouped unique-key dict)."""
    theta = base_model.prior.sample(n, seed=jax.random.PRNGKey(seed))
    if "cosmo/Om0" not in theta or "cosmo/w0" not in theta:
        raise KeyError(f"unexpected baseline unique keys: {sorted(theta)}")
    theta_grp = {k: v for k, v in theta.items()
                 if k not in ("cosmo/Om0", "cosmo/w0")}
    theta_grp[GROUP_KEY] = jnp.stack(
        [theta["cosmo/Om0"], theta["cosmo/w0"]], axis=-1)
    return theta, theta_grp


def main():
    base_model, grp_model, base_pm, grp_pm, grouped = build_models()
    if grp_model.num_free_params != base_model.num_free_params:
        raise RuntimeError(
            f"free-param count mismatch: baseline {base_model.num_free_params} "
            f"vs grouped {grp_model.num_free_params}")

    theta_base, theta_grp = matched_draws(base_model)

    z_base = base_model.bijector.inverse(theta_base)
    z_grp = grp_model.bijector.inverse(theta_grp)

    # gate 1: matched likelihood ---------------------------------------------
    ll_base, _ = base_pm.log_like(jnp.asarray(z_base))
    ll_grp, _ = grp_pm.log_like(jnp.asarray(z_grp))
    rel_ll = float(jnp.max(jnp.abs(ll_grp - ll_base)
                           / jnp.maximum(jnp.abs(ll_base), 1.0)))
    pass1 = rel_ll <= REL_TOL_LOGLIKE

    # gate 2: matched prior density ------------------------------------------
    lp_base = base_model.prior.log_prob(base_model.cast_free_to_native(theta_base))
    lp_grp = grp_model.prior.log_prob(grp_model.cast_free_to_native(theta_grp))
    abs_lp = float(jnp.max(jnp.abs(lp_grp - lp_base)))
    pass2 = abs_lp <= ABS_TOL_LOGPRIOR

    # gate 3: grouped-bijector Jacobian vs numeric ----------------------------
    bij = grouped.experimental_default_event_space_bijector()
    z2 = bij.inverse(theta_grp[GROUP_KEY])
    fldj = bij.forward_log_det_jacobian(z2, event_ndims=1)
    jac = jax.vmap(jax.jacrev(bij.forward))(z2)
    _, logdet = jnp.linalg.slogdet(jac)
    abs_fldj = float(jnp.max(jnp.abs(fldj - logdet)))
    pass3 = abs_fldj <= ABS_TOL_FLDJ

    # gate 4: round-trip -------------------------------------------------------
    theta_rt = bij.forward(z2)
    abs_rt = float(jnp.max(jnp.abs(theta_rt - theta_grp[GROUP_KEY])))
    pass4 = abs_rt <= ABS_TOL_ROUNDTRIP

    result = dict(
        n_draws=N_DRAWS,
        seed=SEED,
        gate1_max_rel_loglike_diff=rel_ll, gate1_tol=REL_TOL_LOGLIKE, gate1_passed=pass1,
        gate2_max_abs_logprior_diff=abs_lp, gate2_tol=ABS_TOL_LOGPRIOR, gate2_passed=pass2,
        gate3_max_abs_fldj_diff=abs_fldj, gate3_tol=ABS_TOL_FLDJ, gate3_passed=pass3,
        gate4_max_abs_roundtrip=abs_rt, gate4_tol=ABS_TOL_ROUNDTRIP, gate4_passed=pass4,
        all_passed=bool(pass1 and pass2 and pass3 and pass4),
        validator_report=grouped.ratio_coords_report,
    )
    os.makedirs(rc.RESULTS_DIR, exist_ok=True)
    with open(rc.GATE_JSON, "w") as f:
        json.dump(result, f, indent=2)
    for k, v in result.items():
        if k != "validator_report":
            print(f"[gate] {k} = {v}")
    print(f"[gate] wrote {rc.GATE_JSON}")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
