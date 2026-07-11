#!/usr/bin/env python3
"""PRE-REGISTERED equivalence gate for Run A (free-r2 reparameterization).

Design checkpoint: `docs/logs/sample-cosmology-dspl.md`, "Run A" -- falsifier
clause: "the pre-run equivalence gate fails -- log-prob [here: log-LIKELIHOOD,
since priors legitimately differ between the two parameterizations] of the
r2-model vs. the cosmology-model at ~64 matched prior draws (r2 set to
r2(Om0,w0)) differing by rel > 1e-8 (float64) => the structural claim is wrong
in the new API and everything upstream is re-opened."

This is cheap (CPU, no MAP, no MCLMC) and is meant to be RUN, unlike
`dspl_free_r2.py`'s sampler. It:

  1. Builds the baseline cosmology-model (verbatim `def_ratio_grid.build_model`,
     itself a verbatim copy of `dspl_cosmology_newapi.ipynb`'s model-construction
     cell) and the r2-model (`dspl_free_r2.build_r2_model`).
  2. Generates ONE shared noisy dataset (seed 0, via `dspl_free_r2.generate_dataset`)
     and builds a `ProbModel` for each model against that SAME dataset.
  3. Draws 64 samples from the cosmology-model's prior; for each draw, computes
     r2 = w0waCDM_Cosmo(z_lens=0.5, z_source_ref=1.0).deflection_ratio(1.5, ...)
     from (Om0, w0) (H0=70, k=0, wa=0), and builds the MATCHED r2-model
     unconstrained-z vector: same nuisance parameters (lens + both sources),
     plane-2 deflection_ratio = that r2.
  4. Compares `ProbModel.log_like(z)` (the pixel log-likelihood ONLY, not
     log_prob -- the priors differ between parameterizations by construction,
     so comparing log_prob would conflate the (irrelevant) prior mismatch with
     the (load-bearing) likelihood-equivalence claim) between the two models
     at each of the 64 matched draws.

PASS = max relative |diff| < 1e-8 (float64). This gate failing means the
structural claim underlying Run A ("cosmology enters the pixel likelihood
ONLY through the scalar r2") is WRONG in the new API, and the entire Run A
design (and everything built on it, including T1/T2 in the lab log) is
reopened -- report loudly either way, per the design checkpoint.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dspl_free_r2 as free_r2  # noqa: E402  (r2-model, dataset, constants)
import def_ratio_grid as drg    # noqa: E402  (verbatim cosmology-model builder)

from gigalens.jax.scene_prob_model import Dataset, ProbModel  # noqa: E402
from gigalens.jax.cosmo import w0waCDM_Cosmo  # noqa: E402

N_DRAWS = 64
GATE_SEED = 12345
REL_TOL = 1e-8

OUT_JSON = os.path.join(free_r2.RESULTS_DIR, "equivalence_gate.json")


def _cosmo_light_components(cosmo_model):
    source1 = cosmo_model.planes[1].light[0]
    source2 = cosmo_model.planes[2].light[0]
    return source1, source2


def build_prob_model(model, sim_config, observed_image1, observed_image2,
                      source1, source2):
    dataset1 = Dataset(observed_image1, sim_config,
                        background_rms=free_r2.BACKGROUND_RMS,
                        exp_time=free_r2.EXP_TIME, sees=[source1])
    dataset2 = Dataset(observed_image2, sim_config,
                        background_rms=free_r2.BACKGROUND_RMS,
                        exp_time=free_r2.EXP_TIME, sees=[source2])
    return ProbModel(model, [dataset1, dataset2], mode="forward")


def matched_r2_x(x_cosmo: dict, r2_values) -> dict:
    """Build the r2-model's unique-param dict from a cosmology-model draw.

    Copies every shared nuisance key verbatim (lens + both sources' free
    parameters use IDENTICAL path strings in both models, since both models
    order planes [lens-mass, source1-light, source2-light] the same way), and
    replaces the cosmology's (Om0, w0) with the free r2 parameter.
    """
    x_r2 = {k: v for k, v in x_cosmo.items() if not k.startswith("cosmo/")}
    x_r2["planes/2/geometry/deflection_ratio"] = r2_values
    return x_r2


def z_from_x(model, x: dict) -> jnp.ndarray:
    """Invert a model's bijector: constrained param dict -> flat unconstrained
    z array, shape (batch, n_params), in the SAME column order ``log_like``
    expects (``model.bijector.inverse`` returns the list of columns)."""
    cols = model.bijector.inverse(x)
    return jnp.stack(cols, axis=-1)


def main():
    os.makedirs(free_r2.RESULTS_DIR, exist_ok=True)

    # -- construction ---------------------------------------------------------
    cosmo_model = drg.build_model()
    r2_model = free_r2.build_r2_model()
    print(f"[gate] cosmo_model.num_free_params = {cosmo_model.num_free_params}")
    print(f"[gate] cosmo_model.z_param_names   = {cosmo_model.z_param_names}")
    print(f"[gate] r2_model.num_free_params    = {r2_model.num_free_params}")
    print(f"[gate] r2_model.z_param_names      = {r2_model.z_param_names}")
    idx_r2 = r2_model.z_param_names.index("planes/2/geometry/deflection_ratio")
    print(f"[gate] free deflection_ratio is z_param_names[{idx_r2}] "
          f"= 'planes/2/geometry/deflection_ratio'")

    sim_config = free_r2.build_sim_config()
    truth = free_r2.truth_params()

    # -- ONE shared dataset (seed 0) -------------------------------------------
    observed_image1, observed_image2, img1, img2 = free_r2.generate_dataset(
        r2_model, sim_config, truth, seed=0)

    source1_r2, source2_r2 = free_r2.build_light_components(r2_model)
    source1_cosmo, source2_cosmo = _cosmo_light_components(cosmo_model)

    prob_r2 = build_prob_model(r2_model, sim_config, observed_image1,
                                observed_image2, source1_r2, source2_r2)
    prob_cosmo = build_prob_model(cosmo_model, sim_config, observed_image1,
                                   observed_image2, source1_cosmo, source2_cosmo)

    # -- construction smoke test (one forward simulate + one log_prob eval) ---
    # NB: use batch=2 (not 1) -- SceneSimulator.simulate() does a bare
    # jnp.squeeze() on the rendered image, which for a batch-1 input silently
    # drops the batch axis too; batch=2 keeps everything unambiguous.
    print("\n[gate] --- construction smoke test ---")
    z0_cosmo = jnp.zeros((2, cosmo_model.num_free_params))
    x0_cosmo = cosmo_model.bijector.forward(list(z0_cosmo.T))
    params0_cosmo = cosmo_model.to_params(x0_cosmo)
    img0 = prob_cosmo.simulators[0].simulate(params0_cosmo)
    print(f"[gate] cosmo_model one forward simulate OK, image shape={np.asarray(img0).shape}")
    lp0, chisq0 = prob_cosmo.log_prob(z0_cosmo)
    print(f"[gate] cosmo_model.log_prob at z=0: {float(np.asarray(lp0)[0]):.6f} "
          f"(red_chi2={float(np.asarray(chisq0)[0]):.6f})")

    z0_r2 = jnp.zeros((2, r2_model.num_free_params))
    x0_r2 = r2_model.bijector.forward(list(z0_r2.T))
    params0_r2 = r2_model.to_params(x0_r2)
    img0_r2 = prob_r2.simulators[0].simulate(params0_r2)
    print(f"[gate] r2_model one forward simulate OK, image shape={np.asarray(img0_r2).shape}")
    lp0_r2, chisq0_r2 = prob_r2.log_prob(z0_r2)
    print(f"[gate] r2_model.log_prob at z=0: {float(np.asarray(lp0_r2)[0]):.6f} "
          f"(red_chi2={float(np.asarray(chisq0_r2)[0]):.6f})")
    print("[gate] --- smoke test OK ---\n")

    # -- the pre-registered equivalence gate ------------------------------------
    key = random.PRNGKey(GATE_SEED)
    x_cosmo = cosmo_model.prior.sample(N_DRAWS, seed=key)

    cosmo_lens = w0waCDM_Cosmo(z_lens=free_r2.Z_LENS, z_source_ref=free_r2.Z_SOURCE1)
    r2_values = cosmo_lens.deflection_ratio(
        free_r2.Z_SOURCE2, H0=70.0, Om0=x_cosmo["cosmo/Om0"], k=0.0,
        w0=x_cosmo["cosmo/w0"], wa=0.0,
    )
    r2_values = jnp.squeeze(jnp.asarray(r2_values), axis=0)  # drop z_lens's shape-(1,) axis
    r2_np = np.asarray(r2_values)
    print(f"[gate] r2 over the {N_DRAWS} matched draws: "
          f"min={r2_np.min():.6f} max={r2_np.max():.6f} "
          f"(model prior support [{free_r2.R2_PRIOR_LOW}, {free_r2.R2_PRIOR_HIGH}])")
    if r2_np.min() < free_r2.R2_PRIOR_LOW or r2_np.max() > free_r2.R2_PRIOR_HIGH:
        print("[gate] WARNING: some matched r2 draws fall OUTSIDE the r2-model's "
              "prior support box -- the r2-model's UniformBij prior does not fully "
              "cover the cosmology-model's induced r2 range. This affects the "
              "reconstruction step, not the gate itself (log_like is evaluated "
              "directly at these z's regardless of the r2-model's declared prior).")

    x_r2 = matched_r2_x(x_cosmo, r2_values)

    z_cosmo = z_from_x(cosmo_model, x_cosmo)
    z_r2 = z_from_x(r2_model, x_r2)

    log_like_cosmo, red_chi2_cosmo = prob_cosmo.log_like(z_cosmo)
    log_like_r2, red_chi2_r2 = prob_r2.log_like(z_r2)

    log_like_cosmo = np.asarray(log_like_cosmo)
    log_like_r2 = np.asarray(log_like_r2)

    abs_diff = np.abs(log_like_cosmo - log_like_r2)
    denom = np.maximum(np.abs(log_like_cosmo), np.abs(log_like_r2))
    rel_diff = abs_diff / denom

    max_abs_diff = float(abs_diff.max())
    max_rel_diff = float(rel_diff.max())
    passed = bool(max_rel_diff < REL_TOL)

    print("\n" + "=" * 72)
    print("EQUIVALENCE GATE RESULT")
    print("=" * 72)
    print(f"  n_draws          = {N_DRAWS}")
    print(f"  max |diff|       = {max_abs_diff:.3e}")
    print(f"  max rel |diff|   = {max_rel_diff:.3e}  (threshold {REL_TOL:.0e})")
    print(f"  log_like range (cosmo model) = [{log_like_cosmo.min():.6f}, "
          f"{log_like_cosmo.max():.6f}]")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  *** GATE FAILED: the 'cosmology enters the likelihood only via r2' ***")
        print("  *** structural claim is WRONG in the new API. Run A's design is  ***")
        print("  *** INVALID as pre-registered -- do NOT proceed to the sampler.  ***")
    print("=" * 72 + "\n")

    result = {
        "n_draws": N_DRAWS,
        "gate_seed": GATE_SEED,
        "rel_tol": REL_TOL,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "passed": passed,
        "log_like_cosmo_min": float(log_like_cosmo.min()),
        "log_like_cosmo_max": float(log_like_cosmo.max()),
        "log_like_r2_min": float(log_like_r2.min()),
        "log_like_r2_max": float(log_like_r2.max()),
        "r2_matched_min": float(r2_np.min()),
        "r2_matched_max": float(r2_np.max()),
        "r2_prior_low": free_r2.R2_PRIOR_LOW,
        "r2_prior_high": free_r2.R2_PRIOR_HIGH,
        "cosmo_z_param_names": cosmo_model.z_param_names,
        "r2_z_param_names": r2_model.z_param_names,
        "r2_free_index": idx_r2,
        "per_draw_abs_diff": abs_diff.tolist(),
        "per_draw_rel_diff": rel_diff.tolist(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[gate] wrote {OUT_JSON}")

    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
