#!/usr/bin/env python3
"""Run C: ratio-coordinates grouped prior for (Om0, w0) — single-run full forward model.

Pre-registered design: `docs/logs/sample-cosmology-dspl.md`, "Run C" checkpoint.
Established facts this builds on (same log): the pixel likelihood depends on
cosmology only through r2 (T1 + Run A's equivalence gate, rel diff 1.5e-14);
the (Om0, w0)+NormalCDF coordinates are the disease (Run A: free r2 samples
trivially; Run B: the frozen bulk-tuned metric makes the arm one-way); the
grid posterior carries mass 0.103 below Om0=0.146 (`def_ratio_grid.npz`).

Run A removed the pathology by DELETING cosmology from the model (free r2 +
offline reconstruction). Run C keeps the baseline's FULL forward model — free
(Om0, w0), redshift-carrying planes, identical priors — and changes ONLY the
sampling coordinates, via a grouped tuple-key prior on the cosmology Component
(`gigalens_research.priors.ratio_coords`): z1 -> Om0 (NormalCDF, as baseline);
z2 -> u = r2 squashed into its CONDITIONAL range at that Om0; w0 recovered by a
bracketed solve. The data-stiff scalar r2 is thereby a sampling coordinate
while (Om0, w0) remain model parameters — cosmology posterior samples come
straight out of the single run, no reconstruction step.

The prior DENSITY over (Om0, w0) is unchanged (uniform box, exactly the
baseline's), so the target posterior is identical to the baseline notebook's
by construction; the gate script (`dspl_ratio_coords_gate.py`) verifies the
matched-likelihood, Jacobian, and round-trip claims numerically and writes
`ratio_coords_gate.json`, which `--run` REQUIRES to exist with all checks
passing.

Conditioning direction and its measured degenerate sliver (validator numbers,
autodiff of the actual u_fn on a 201x201 grid, this workstation, 2026-07-11):
r2 is monotone-increasing in w0 at fixed Om0 except a genuine shallow
inversion where dark energy is dynamically irrelevant — worst signed
dr2/dw0 = -4.128e-5 at (Om0=0.88, w0=-2.0), flips confined to Om0 >= 0.785,
the whole region >= 15.5 sigma_r,eff from the data contour in u units
(posterior mass ~1e-53); max interior excursion beyond the w0-endpoint
bracket 8.785e-7 (~1.3e-3 sigma_r,eff). Derived tolerances passed to the
validator below: DU_DW_ATOL = 1e-4 (2.4x the measured worst inversion, 100x
below the median |dr2/dw0| ~ 1e-2), EXCURSION_ATOL = 3e-6 (3.4x measured).
On the zero-crossing curve of dr2/dw0 the log-det has an integrable
singularity; in float64 the reachable spike is <= ~46 nats vs a >= ~120-nat
likelihood penalty for even entering the region, so it cannot corrupt MAP or
attract chains — monitored anyway via the pre-registered 0-nonfinite-steps
diagnostic.

Import-safe: no work at import time; the expensive pipeline requires --run
AND --confirm-run-c-approved (grader approval per the design checkpoint) AND
a passing gate JSON. Dataset: the same seeded (seed 0) noise realization as
Runs A/B (`dspl_arm_init.make_observed_images`) — a different realization
from the (unseeded, unreproducible) baseline notebook run, as flagged there.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# Baseline system pieces (verbatim notebook constructions) — reused by import so
# Run C cannot drift from Runs A/B: constants, truth scene, dataset, ProbModel.
import dspl_arm_init as arm
from dspl_arm_init import (
    Z_LENS, Z_SOURCE1, Z_SOURCE2, BACKGROUND_RMS, EXP_TIME,
    TRUE_TRUTH_SCENE, GRID_NPZ,
)

from gigalens_research.priors.ratio_coords import (
    RatioCoordsUniform, deflection_ratio_u_fn,
)

HOME = os.path.expanduser("~")
RESULTS_DIR = os.path.join(HOME, "GIGALens-Code", "results", "sample_cosmology",
                           "dspl_ratio_coords")
GATE_JSON = os.path.join(RESULTS_DIR, "ratio_coords_gate.json")

OM0_BOUNDS = (0.0, 1.0)
W0_BOUNDS = (-2.0, -1.0 / 3.0)

# Derived validator tolerances — provenance in the module docstring above and
# the Run C design checkpoint (docs/logs/sample-cosmology-dspl.md).
DU_DW_ATOL = 1.0e-4
EXCURSION_ATOL = 3.0e-6


def build_u_fn():
    """The data-stiff scalar: plane-2 deflection ratio r2(Om0, w0) itself
    (single extra source plane -> weight 1.0). Fixed cosmology parameters are
    the baseline notebook's constants, passed explicitly."""
    from gigalens.jax.cosmo import w0waCDM_Cosmo

    cosmo_profile = w0waCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE1)
    return deflection_ratio_u_fn(cosmo_profile, (Z_SOURCE2,), (1.0,),
                                 fixed=dict(H0=70.0, k=0.0, wa=0.0))


def build_grouped_model():
    """The baseline FULL model with ONE change: the cosmology Component's two
    scalar UniformBij priors on Om0/w0 are replaced by the grouped tuple-key
    ratio-coordinates prior (same uniform box density, different sampling
    coordinates). Returns (model, lens, source1, source2, grouped_prior)."""
    from gigalens.jax.cosmo import w0waCDM_Cosmo
    from gigalens.jax.scene import Component, Plane, LensModel

    lens, source1, source2, _scalar_cosmo = arm.build_components()

    grouped = RatioCoordsUniform(
        build_u_fn(), OM0_BOUNDS, W0_BOUNDS,
        du_dw_atol=DU_DW_ATOL, excursion_atol=EXCURSION_ATOL,
    )
    cosmo = Component(
        w0waCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE1),
        {
            "H0": 70.0,
            "k": 0.0,
            "wa": 0.0,
            ("Om0", "w0"): grouped,
        },
    )
    model = LensModel(
        [
            Plane(redshift=Z_LENS, mass=[lens]),
            Plane(redshift=Z_SOURCE1, light=[source1]),
            Plane(redshift=Z_SOURCE2, light=[source2]),
        ],
        cosmo=cosmo,
    )
    return model, lens, source1, source2, grouped


def make_prob_model(model, comp1, comp2, img1, img2, sim_config):
    """Flat-z-migrated replacement for `dspl_arm_init.make_prob_model` (which
    predates the library's Dataset -> ImageData rename and no longer runs);
    same semantics: a forward-mode two-band ProbModel whose `sees` are the
    model's OWN light Component instances (identity-matched)."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel

    dataset1 = ImageData(img1, sim_config, background_rms=BACKGROUND_RMS,
                         exp_time=EXP_TIME, sees=[comp1])
    dataset2 = ImageData(img2, sim_config, background_rms=BACKGROUND_RMS,
                         exp_time=EXP_TIME, sees=[comp2])
    return ProbModel(model, [dataset1, dataset2], mode="forward")


def build_pipeline(prob_model, *, pipeline_seed: int = 42):
    """MAP -> diag-qz bridge -> MCLMC, stage-for-stage and seed-for-seed the
    same as Run A (`dspl_free_r2.build_pipeline`) so results are comparable."""
    from gigalens.jax.inference import ModellingSequence
    from gigalens_research.inference_utils import (
        InferenceContext, Pipeline, MAPStage, BridgeStage, MCLMCStage,
    )

    def _old_map_optimizer():
        import optax
        return optax.adabelief(1e-2, b1=0.95, b2=0.99)

    model_seq = ModellingSequence(prob_model)
    ctx = InferenceContext.from_modelling_sequence(model_seq)
    pipeline = Pipeline(ctx, seed=pipeline_seed)

    pipeline.add(MAPStage(
        num_steps=4000,
        n_samples=1000,
        pbar_interval=100,
        seed=1,
        optimizer_factory=_old_map_optimizer,
        optimizer_id="adabelief_1e-2_b1_0.95_b2_0.99_no_nesterov",
    ))

    def make_diag_qz(z_best):
        return tfd.MultivariateNormalDiag(
            loc=jnp.asarray(z_best),
            scale_diag=jnp.full(z_best.shape[-1], 1e-3),
        )

    pipeline.add(BridgeStage(
        name="diag_qz_from_map",
        version="v1",
        requires=("z_best",),
        produces=("qz",),
        fn=make_diag_qz,
    ))

    pipeline.add(MCLMCStage(
        n_chains=8,
        num_burnin_steps=10000,
        num_results=10000,
        desired_energy_variance=5e-4,
        seed=10,
        progress_bar=True,
        debug=True,
    ))
    return pipeline


def check_gate():
    """--run precondition: the gate script must have been run and passed."""
    if not os.path.exists(GATE_JSON):
        raise RuntimeError(
            f"gate file {GATE_JSON} not found — run dspl_ratio_coords_gate.py "
            "first (the equivalence/Jacobian/round-trip gate is a pre-run "
            "requirement of the Run C design checkpoint).")
    with open(GATE_JSON) as f:
        gate = json.load(f)
    if not gate.get("all_passed", False):
        raise RuntimeError(
            f"gate file {GATE_JSON} reports all_passed="
            f"{gate.get('all_passed')} — Run C must not launch on a failing "
            f"gate. Gate contents: {gate}")
    print(f"[dspl_ratio_coords] gate check OK: {GATE_JSON}")
    return gate


def main(run: bool, confirm_approved: bool):
    model, lens, source1, source2, grouped = build_grouped_model()
    print(f"[dspl_ratio_coords] num_free_params = {model.num_free_params}")
    print(f"[dspl_ratio_coords] z_param_names   = {model.z_param_names}")
    print(f"[dspl_ratio_coords] validator report = {grouped.ratio_coords_report}")

    img1, img2, sim_config = arm.make_observed_images(
        model, lens, source1, source2, data_seed=0)
    prob_model = make_prob_model(
        model, model.planes[1].light[0], model.planes[2].light[0],
        img1, img2, sim_config)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "dataset.npz"),
             observed_image1=np.asarray(img1), observed_image2=np.asarray(img2),
             noise_seed=0)
    with open(os.path.join(RESULTS_DIR, "validator_report.json"), "w") as f:
        json.dump(grouped.ratio_coords_report, f, indent=2)

    pipeline = build_pipeline(prob_model)

    if not run:
        print("[dspl_ratio_coords] --run not passed: model/data/pipeline "
              "constructed but MAP+MCLMC NOT launched (awaiting grader "
              "approval per the Run C design checkpoint).")
        return None
    if not confirm_approved:
        raise RuntimeError(
            "--run also requires --confirm-run-c-approved (grader approval of "
            "the Run C design checkpoint in docs/logs/sample-cosmology-dspl.md).")
    check_gate()
    artifacts = pipeline.run(out_dir=RESULTS_DIR, resume=True)
    print("[dspl_ratio_coords] pipeline.run complete.")
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Launch MAP + MCLMC (requires approval + gate).")
    parser.add_argument("--confirm-run-c-approved", action="store_true",
                        help="Assert the grader approved the Run C checkpoint.")
    args = parser.parse_args()
    main(run=args.run, confirm_approved=args.confirm_run_c_approved)
