#!/usr/bin/env python3
"""Pre-run gate for Run D (banded u-first ratio coordinates) — cheap, CPU-safe.

Run C's gate battery adapted for the banded support: prior draws with
u(theta) OUTSIDE the band (1.2867134, 1.3398900) are excluded from the matched
comparisons (the u-first model cannot represent them BY DESIGN — the support
amendment in the Run D checkpoint; the dropped fraction is reported and must
match the validator's ~11% excluded prior volume).

  gate 1: matched log-like at identical in-band physical params, rel <= 1e-8
  gate 2: matched prior.log_prob at in-band params, abs <= 1e-10
  gate 3: u-first bijector FLDJ vs numeric slogdet, abs <= 1e-8
  gate 4: (Om0, w0) round-trip, abs <= 1e-9
  gate 5 (new): u(forward(z)) is a function of z1 alone — moving z2 leaves
          u unchanged to abs <= 1e-12 (the zero-rotation construction claim)

Writes `results/sample_cosmology/dspl_ratio_ufirst/ratio_ufirst_gate.json`.
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
import dspl_ratio_ufirst as ru

N_DRAWS = 96          # pre-filter; ~11% get dropped as out-of-band
SEED = 7

REL_TOL_LOGLIKE = 1e-8
ABS_TOL_LOGPRIOR = 1e-10
ABS_TOL_FLDJ = 1e-8
ABS_TOL_ROUNDTRIP = 1e-9
ABS_TOL_U_INVARIANCE = 1e-12

GROUP_KEY = "cosmo/Om0|cosmo/w0"


def main():
    base_model, base_lens, base_s1, base_s2 = arm.build_full_model()
    grp_model, *_rest, grouped = ru.build_grouped_model_ufirst()

    img1, img2, sim_config = arm.make_observed_images(
        base_model, base_lens, base_s1, base_s2, data_seed=0)
    base_pm = rc.make_prob_model(
        base_model, base_model.planes[1].light[0], base_model.planes[2].light[0],
        img1, img2, sim_config)
    grp_pm = rc.make_prob_model(
        grp_model, grp_model.planes[1].light[0], grp_model.planes[2].light[0],
        img1, img2, sim_config)
    if grp_model.num_free_params != base_model.num_free_params:
        raise RuntimeError("free-param count mismatch")

    bij = grouped.experimental_default_event_space_bijector()
    u_fn = rc.build_u_fn()

    theta = base_model.prior.sample(N_DRAWS, seed=jax.random.PRNGKey(SEED))
    u_draws = jnp.vectorize(u_fn)(theta["cosmo/Om0"], theta["cosmo/w0"])
    in_band = np.asarray((u_draws > bij._u_lo_band) & (u_draws < bij._u_hi_band))
    n_kept = int(in_band.sum())
    keep = np.where(in_band)[0]
    theta_b = {k: jnp.asarray(np.asarray(v)[keep]) for k, v in theta.items()}
    theta_g = {k: v for k, v in theta_b.items()
               if k not in ("cosmo/Om0", "cosmo/w0")}
    theta_g[GROUP_KEY] = jnp.stack(
        [theta_b["cosmo/Om0"], theta_b["cosmo/w0"]], axis=-1)

    z_base = base_model.bijector.inverse(theta_b)
    z_grp = grp_model.bijector.inverse(theta_g)

    ll_base, _ = base_pm.log_like(jnp.asarray(z_base))
    ll_grp, _ = grp_pm.log_like(jnp.asarray(z_grp))
    rel_ll = float(jnp.max(jnp.abs(ll_grp - ll_base)
                           / jnp.maximum(jnp.abs(ll_base), 1.0)))

    lp_base = base_model.prior.log_prob(base_model.cast_free_to_native(theta_b))
    lp_grp = grp_model.prior.log_prob(grp_model.cast_free_to_native(theta_g))
    abs_lp = float(jnp.max(jnp.abs(lp_grp - lp_base)))

    z2c = bij.inverse(theta_g[GROUP_KEY])
    fldj = bij.forward_log_det_jacobian(z2c, event_ndims=1)
    jac = jax.vmap(jax.jacrev(bij.forward))(z2c)
    _, logdet = jnp.linalg.slogdet(jac)
    abs_fldj = float(jnp.max(jnp.abs(fldj - logdet)))

    theta_rt = bij.forward(z2c)
    abs_rt = float(jnp.max(jnp.abs(theta_rt - theta_g[GROUP_KEY])))

    # gate 5: u depends on z1 alone
    z_shift = z2c.at[:, 1].add(1.7)
    th1, th2 = bij.forward(z2c), bij.forward(z_shift)
    u1 = jnp.vectorize(u_fn)(th1[..., 0], th1[..., 1])
    u2 = jnp.vectorize(u_fn)(th2[..., 0], th2[..., 1])
    abs_uinv = float(jnp.max(jnp.abs(u1 - u2)))

    result = dict(
        n_draws=N_DRAWS, n_in_band=n_kept, seed=SEED,
        frac_dropped_out_of_band=1.0 - n_kept / N_DRAWS,
        gate1_max_rel_loglike_diff=rel_ll, gate1_tol=REL_TOL_LOGLIKE,
        gate1_passed=bool(rel_ll <= REL_TOL_LOGLIKE),
        gate2_max_abs_logprior_diff=abs_lp, gate2_tol=ABS_TOL_LOGPRIOR,
        gate2_passed=bool(abs_lp <= ABS_TOL_LOGPRIOR),
        gate3_max_abs_fldj_diff=abs_fldj, gate3_tol=ABS_TOL_FLDJ,
        gate3_passed=bool(abs_fldj <= ABS_TOL_FLDJ),
        gate4_max_abs_roundtrip=abs_rt, gate4_tol=ABS_TOL_ROUNDTRIP,
        gate4_passed=bool(abs_rt <= ABS_TOL_ROUNDTRIP),
        gate5_max_abs_u_invariance=abs_uinv, gate5_tol=ABS_TOL_U_INVARIANCE,
        gate5_passed=bool(abs_uinv <= ABS_TOL_U_INVARIANCE),
        validator_report=grouped.ratio_coords_report,
    )
    result["all_passed"] = bool(all(
        result[f"gate{i}_passed"] for i in range(1, 6)))
    os.makedirs(ru.RESULTS_DIR, exist_ok=True)
    with open(ru.GATE_JSON, "w") as f:
        json.dump(result, f, indent=2)
    for k, v in result.items():
        if k != "validator_report":
            print(f"[gate] {k} = {v}")
    print(f"[gate] wrote {ru.GATE_JSON}")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
