#!/usr/bin/env python3
"""Run D: u-FIRST (banded) ratio-coordinates grouped prior — DSPL system.

Pre-registered design + amendment: `docs/logs/sample-cosmology-dspl.md`,
"Run D" checkpoint (approved 2026-07-11) and its banded-support amendment.
Run C established that the om-first ordering leaves a rotating likelihood band
(-8deg -> -84deg) that a frozen metric truncates at Om0~0.385 (63% of mass
unvisited). This run swaps the conditioning order so the likelihood depends on
z1 ALONE (axis-aligned slab, zero rotation):

  z1 -> u squashed over the BAND (u_a(0), min u_b) = (1.2867134, 1.3398900)
        [amendment: u has interior critical points on the full box — u_a dips
         before rising, u_b wiggles — so a full-box u-first map is
         topologically impossible; the band's edges sit 55.5 sigma and
         23.8 sigma from the data (u* = 1.3239203, sigma_r,eff = 6.7e-4), so
         the truncation excludes < 1e-125 of posterior mass but 11.1% of PRIOR
         box volume — an explicit, quantified support change]
  z2 -> Om0 squashed into [0, root of u_a(Om0) = u]
  w0 -> root of u_fn(Om0, w0) = u  (as Run C)

Everything else (system, dataset seed 0, priors, pipeline stages/seeds) is
identical to Run C (`dspl_ratio_coords.py`). Import-safe; the pipeline needs
--run AND --confirm-run-d-approved AND a passing gate JSON
(`dspl_ratio_ufirst_gate.py`).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import dspl_arm_init as arm
import dspl_ratio_coords as rc
from dspl_ratio_coords import Z_LENS, Z_SOURCE1, Z_SOURCE2

from gigalens_research.priors.ratio_coords import UFirstRatioCoordsUniform

HOME = os.path.expanduser("~")
RESULTS_DIR = os.path.join(HOME, "GIGALens-Code", "results", "sample_cosmology",
                           "dspl_ratio_ufirst")
GATE_JSON = os.path.join(RESULTS_DIR, "ratio_ufirst_gate.json")

# Derived validator tolerances. du_dw/excursion: same measurements as Run C
# (the w0-solve is shared). curve_atol: the real u_a curve passes STRICT 0.0
# (min signed slope above the band floor = +0.0250, measured 2026-07-11).
DU_DW_ATOL = rc.DU_DW_ATOL
EXCURSION_ATOL = rc.EXCURSION_ATOL
CURVE_ATOL = 0.0


def build_grouped_model_ufirst():
    """Run C's model with ONE change: the grouped prior's event-space bijector
    is the banded u-first map. Returns (model, lens, source1, source2, grouped)."""
    from gigalens.jax.cosmo import w0waCDM_Cosmo
    from gigalens.jax.scene import Component, Plane, LensModel

    lens, source1, source2, _scalar_cosmo = arm.build_components()

    grouped = UFirstRatioCoordsUniform(
        rc.build_u_fn(), rc.OM0_BOUNDS, rc.W0_BOUNDS,
        du_dw_atol=DU_DW_ATOL, excursion_atol=EXCURSION_ATOL,
        curve_atol=CURVE_ATOL,
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


def check_gate():
    if not os.path.exists(GATE_JSON):
        raise RuntimeError(
            f"gate file {GATE_JSON} not found — run dspl_ratio_ufirst_gate.py "
            "first (pre-run requirement of the Run D checkpoint).")
    with open(GATE_JSON) as f:
        gate = json.load(f)
    if not gate.get("all_passed", False):
        raise RuntimeError(
            f"gate reports all_passed={gate.get('all_passed')}; Run D must "
            f"not launch on a failing gate. Contents: {gate}")
    print(f"[dspl_ratio_ufirst] gate check OK: {GATE_JSON}")
    return gate


def main(run: bool, confirm_approved: bool):
    model, lens, source1, source2, grouped = build_grouped_model_ufirst()
    print(f"[dspl_ratio_ufirst] num_free_params = {model.num_free_params}")
    print(f"[dspl_ratio_ufirst] z_param_names   = {model.z_param_names}")
    print(f"[dspl_ratio_ufirst] validator report = {grouped.ratio_coords_report}")

    img1, img2, sim_config = arm.make_observed_images(
        model, lens, source1, source2, data_seed=0)
    prob_model = rc.make_prob_model(
        model, model.planes[1].light[0], model.planes[2].light[0],
        img1, img2, sim_config)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "dataset.npz"),
             observed_image1=np.asarray(img1), observed_image2=np.asarray(img2),
             noise_seed=0)
    with open(os.path.join(RESULTS_DIR, "validator_report.json"), "w") as f:
        json.dump(grouped.ratio_coords_report, f, indent=2)

    pipeline = rc.build_pipeline(prob_model)

    if not run:
        print("[dspl_ratio_ufirst] --run not passed: constructed only.")
        return None
    if not confirm_approved:
        raise RuntimeError(
            "--run also requires --confirm-run-d-approved (grader approval of "
            "the Run D checkpoint incl. the banded-support amendment).")
    check_gate()
    artifacts = pipeline.run(out_dir=RESULTS_DIR, resume=True)
    print("[dspl_ratio_ufirst] pipeline.run complete.")
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-run-d-approved", action="store_true")
    args = parser.parse_args()
    main(run=args.run, confirm_approved=args.confirm_run_d_approved)
