#!/usr/bin/env python3
"""Run A: free-r2 reparameterization of the DSPL cosmology system (new scene API).

Pre-registered design: `docs/logs/sample-cosmology-dspl.md` -- "Run A: free-r2
reparameterization + analytic (Om0, w0) reconstruction". Established fact
(T1/T2, same log): the pixel likelihood of `dspl_cosmology_newapi.ipynb`'s
system depends on cosmology ONLY through the scalar
`r2 = deflection_ratio(z_source2=1.5)` of the z=1.5 source plane (relative to
the z=1.0 reference plane, where the ratio is defined to be 1 by construction).
Hypothesis: the (Om0, w0) + NormalCDF sampling coordinates -- not the physics --
are what makes MCLMC truncate the low-Om0 arm (rotating, semi-infinite thin
ridge in unconstrained z-space). This script samples r2 directly as a free
`deflection_ratio` parameter on plane 2, removing the pathological cosmology
parameterization entirely, so the ONLY thing that changes relative to the
baseline notebook is which quantity is free at plane 2's geometry site.

This module performs NO WORK AT IMPORT TIME -- it only defines functions and
constants. Building the model, generating the dataset, and constructing the
pipeline are all explicit function calls; running the (expensive) MAP+MCLMC
pipeline additionally requires `--run` on the command line, so
`import dspl_free_r2` (e.g. from the equivalence gate script) never launches
anything. See `docs/logs/sample-cosmology-dspl.md` for the full pre-registered
design and falsifiers; do NOT run the pipeline before grader approval.

System (identical to the baseline notebook EXCEPT for the reparameterization):
  - z_lens = 0.5 (EPL mass plane, no geometry needed -- no light, no cosmology).
  - z_source1 = 1.0: Sersic source plane. Baseline notebook made this the
    cosmology's `z_source_ref` (deflection_ratio == 1 there BY DEFINITION).
    Here it is a plane with a FIXED `deflection_ratio = 1.0` constant --
    numerically identical fixed behavior, just expressed directly instead of
    via a cosmology object.
  - z_source2 = 1.5: Sersic source plane. Baseline notebook derived its
    deflection ratio from (Om0, w0) via `w0waCDM_Cosmo.deflection_ratio`. Here
    it is a plane with a FREE `deflection_ratio`, prior `UniformBij(1.2645,
    1.3432)` -- the exact range r2 spans over the WHOLE (Om0, w0) prior box
    (see `def_ratio_grid.py`'s `r2_grid.min()/max()`, computed at the same
    z_lens=0.5 / z_source1=1.0 / z_source2=1.5).

No `cosmo=` Component is built at all: `LensModel._validate_geometry` requires
EXACTLY ONE of {redshift (needs a cosmology), deflection_ratio (no cosmology)}
per plane, so dropping the cosmology means dropping every `redshift=` kwarg
too -- there is no in-between "some planes have redshift, no cosmology" state,
by design (`gigalens/src/gigalens/jax/scene.py::LensModel._validate_geometry`).
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

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.inference import ModellingSequence

from gigalens_research.inference_utils import (
    InferenceContext, Pipeline, MAPStage, BridgeStage, MCLMCStage,
)

# ---------------------------------------------------------------------------
# System constants -- MUST match dspl_cosmology_newapi.ipynb / def_ratio_grid.py.
# ---------------------------------------------------------------------------
Z_LENS = 0.5
Z_SOURCE1 = Z_LENS * 2  # == 1.0; fixed deflection_ratio = 1.0 (was the cosmology's
                        # z_source_ref, where the ratio is 1 by definition)
Z_SOURCE2 = Z_LENS * 3  # == 1.5; free deflection_ratio (was derived from Om0, w0)

# r2(truth) at the baseline notebook's truth cosmology (H0=70, Om0=0.3, k=0,
# w0=-1, wa=0), evaluated at z_lens=0.5 / z_source_ref=1.0 / z_source2=1.5.
# Hardcoded here (float64 literal) rather than recomputed, per the design
# checkpoint; provenance: `results/sample_cosmology/dspl_cosmology_newapi/
# def_ratio_grid.npz`'s `r2_truth` field, produced by `def_ratio_grid.py`,
# which computes exactly this quantity via `w0waCDM_Cosmo.deflection_ratio`.
R2_TRUTH = 1.3241652127

# Exact range r2 spans over the whole (Om0 in [0,1], w0 in [-2,-1/3]) prior
# box at this system's geometry -- from `def_ratio_grid.npz`'s `r2_grid.min()`
# / `r2_grid.max()` (n_grid=400 quadrature). This is the prior support for the
# free plane-2 deflection_ratio: it must cover the box exactly, or the r2-model
# excludes (or over-covers, changing the resulting reconstruction) part of the
# (Om0, w0) posterior support the baseline notebook's model would visit.
R2_PRIOR_LOW = 1.2645
R2_PRIOR_HIGH = 1.3432

RESULTS_DIR = os.path.join(
    os.path.expanduser("~"), "GIGALens-Code", "results",
    "sample_cosmology", "dspl_free_r2",
)


# ---------------------------------------------------------------------------
# UniformBij -- copied VERBATIM from dspl_cosmology_newapi.ipynb (cell fdecfa67).
# A tfd.Uniform with a Shift/Scale/NormalCDF event-space bijector in place of
# TFP's default (sigmoid-based) one for Uniform. Used here for the free r2
# parameter, exactly as the baseline notebook used it for Om0 and w0.
# ---------------------------------------------------------------------------
def tNCDF_bij(low, high):
    return tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.NormalCDF()])


class UniformBij(tfd.Uniform):
    def __init__(self, *args, event_space_bijector_class=tNCDF_bij, **kwargs):
        self._esb = event_space_bijector_class(*args)
        super().__init__(*args, **kwargs)

    def _default_event_space_bijector(self):
        return self._esb


# ---------------------------------------------------------------------------
# Model construction (no cosmology; plane-2 deflection_ratio is the free
# reparameterized cosmology scalar). Priors/values for the lens and both
# sources are copied verbatim from dspl_cosmology_newapi.ipynb -- only the
# geometry handling differs (see module docstring).
# ---------------------------------------------------------------------------
def build_r2_model() -> LensModel:
    lens = Component(
        EPL(),
        dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
            gamma=2.0,
            e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            center_x=tfd.Normal(0, 0.05),
            center_y=tfd.Normal(0, 0.05),
        ),
    )
    # NOTE (verbatim from the baseline notebook): the old notebook's `shear`
    # Prior/Component (mass.shear.Shear()) was drafted but commented out of the
    # model it actually built and ran; omitted here too, for the same reason.

    source1 = Component(
        SersicEllipse(use_lstsq=False),
        dict(
            center_x=tfd.Normal(0, 2),
            center_y=tfd.Normal(0, 2),
            e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
            n_sersic=tfd.Uniform(1, 10),
            R_sersic=tfd.LogNormal(jnp.log(1.), 0.15),
            Ie=tfd.LogNormal(jnp.log(150), 1),
        ),
    )
    source2 = Component(
        SersicEllipse(use_lstsq=False),
        dict(
            center_x=tfd.Normal(0, 2),
            center_y=tfd.Normal(0, 2),
            e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
            n_sersic=tfd.Uniform(1, 10),
            R_sersic=tfd.LogNormal(jnp.log(1.), 0.15),
            Ie=tfd.LogNormal(jnp.log(150), 1),
        ),
    )

    model = LensModel(
        [
            # Plane 0: the EPL deflector. No light, no cosmology present -> no
            # geometry field is required at all (LensModel._validate_geometry
            # only demands one of {redshift, deflection_ratio} for LENSED LIGHT
            # planes; a mass-only plane needs neither).
            Plane(mass=[lens]),
            # Plane 1 (z=1.0, the baseline's cosmology z_source_ref): FIXED
            # deflection_ratio=1.0 -- exactly the baseline's "ratio == 1 at the
            # reference plane by definition", expressed directly.
            Plane(deflection_ratio=1.0, light=[source1]),
            # Plane 2 (z=1.5): FREE deflection_ratio -- this replaces the
            # baseline's (Om0, w0) -> r2 mapping. Prior support is the exact
            # range r2 spans over the baseline's whole (Om0, w0) prior box.
            Plane(
                deflection_ratio=UniformBij(
                    jnp.float64(R2_PRIOR_LOW), jnp.float64(R2_PRIOR_HIGH)
                ),
                light=[source2],
            ),
        ],
        cosmo=None,
    )
    return model


# ---------------------------------------------------------------------------
# Ground truth -- verbatim physical values from the baseline notebook's
# `truth_scene`, restructured for the r2-model's geometry (deflection_ratio
# instead of redshift; no "cosmo" key at all).
# ---------------------------------------------------------------------------
def truth_params() -> dict:
    return {
        "planes": {
            0: {
                "mass": {
                    0: {"theta_E": 1.1, "gamma": 2.0, "e1": 0.05, "e2": 0.02,
                        "center_x": 0.0, "center_y": 0.0},
                },
            },
            1: {
                "geometry": {"deflection_ratio": 1.0},
                "light": {
                    0: {"R_sersic": 0.25, "n_sersic": 2., "e1": 0.05, "e2": 0.,
                        "center_x": 0.05, "center_y": 0., "Ie": 50.},
                },
            },
            2: {
                "geometry": {"deflection_ratio": R2_TRUTH},
                "light": {
                    0: {"R_sersic": 1., "n_sersic": 6., "e1": 0.0, "e2": 0.05,
                        "center_x": 0., "center_y": 0.05, "Ie": 15.},
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Simulation config, PSF -- verbatim from the baseline notebook.
# ---------------------------------------------------------------------------
def build_sim_config() -> SimulatorConfig:
    import photutils.psf as psf

    num_pix, delta_pix = 100, 0.065
    kernel = psf.GaussianPSF(x_fwhm=2, y_fwhm=2)
    yy, xx = np.mgrid[-7:8, -7:8]
    kernel = kernel(xx, yy)

    return SimulatorConfig(
        delta_pix=delta_pix,
        num_pix=num_pix,
        kernel=kernel,
        supersample=1,
        likelihood_precision="float64",
    )


NUM_PIX = 100
DELTA_PIX = 0.065
EXP_TIME = 1000
BACKGROUND_RMS = 0.1


def build_light_components(model: LensModel):
    """Return (source1, source2) Component objects (identity-matched, for
    ``sees=``) from a model built by :func:`build_r2_model`."""
    source1 = model.planes[1].light[0]
    source2 = model.planes[2].light[0]
    return source1, source2


# ---------------------------------------------------------------------------
# Data generation -- mirrors dspl_cosmology_newapi.ipynb's noise cell EXACTLY,
# except the noise draw is seeded (np.random.seed(0) once, immediately before
# the two add_noise calls) so the dataset is reproducible.
#
# NOTE: the baseline notebook's own noise draw is UNSEEDED (it relies on
# whatever numpy global RNG state happens to be active at cell-execution
# time), so this is a DIFFERENT noise realization from the baseline run --
# any comparison between this script's results and the baseline notebook's
# results is a comparison ACROSS noise realizations, not a byte-for-byte
# reproduction.
# ---------------------------------------------------------------------------
def add_noise(img, exp_time, sigma_bkd):
    from lenstronomy.Util import image_util
    poisson = image_util.add_poisson(img, exp_time=exp_time)
    bkg = image_util.add_background(img, sigma_bkd=sigma_bkd)
    return img + poisson + bkg


def generate_dataset(model: LensModel, sim_config: SimulatorConfig, truth: dict,
                      *, seed: int = 0):
    """Render the two noiseless "truth" bands and add seeded noise.

    Returns (observed_image1, observed_image2, img1_noiseless, img2_noiseless).
    """
    source1, source2 = build_light_components(model)
    sim1 = SceneSimulator(model, sim_config, sees=[source1])
    sim2 = SceneSimulator(model, sim_config, sees=[source2])

    img1 = np.asarray(sim1.simulate(truth))
    img2 = np.asarray(sim2.simulate(truth))

    np.random.seed(seed)
    observed_image1 = add_noise(img1, exp_time=EXP_TIME, sigma_bkd=BACKGROUND_RMS)
    observed_image2 = add_noise(img2, exp_time=EXP_TIME, sigma_bkd=BACKGROUND_RMS)
    return observed_image1, observed_image2, img1, img2


def _jsonable(d):
    """Recursively coerce a params-style nested dict to plain JSON-safe types."""
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    try:
        return float(d)
    except (TypeError, ValueError):
        return d


def save_dataset(results_dir: str, observed_image1, observed_image2,
                  img1_noiseless, img2_noiseless, truth: dict):
    os.makedirs(results_dir, exist_ok=True)
    np.savez(
        os.path.join(results_dir, "dataset.npz"),
        observed_image1=observed_image1,
        observed_image2=observed_image2,
        img1_noiseless=img1_noiseless,
        img2_noiseless=img2_noiseless,
        num_pix=NUM_PIX, delta_pix=DELTA_PIX,
        exp_time=EXP_TIME, background_rms=BACKGROUND_RMS,
        noise_seed=0,
        r2_truth=R2_TRUTH,
    )
    with open(os.path.join(results_dir, "truth.json"), "w") as f:
        json.dump(_jsonable(truth), f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline construction (MAP -> diag-qz bridge -> MCLMC), same stages/settings
# as dspl_cosmology_newapi.ipynb.
# ---------------------------------------------------------------------------
def _old_map_optimizer():
    import optax
    return optax.adabelief(1e-2, b1=0.95, b2=0.99)


def build_pipeline(model: LensModel, sim_config: SimulatorConfig,
                    observed_image1, observed_image2, *, pipeline_seed: int = 42):
    source1, source2 = build_light_components(model)
    dataset1 = Dataset(observed_image1, sim_config, background_rms=BACKGROUND_RMS,
                        exp_time=EXP_TIME, sees=[source1])
    dataset2 = Dataset(observed_image2, sim_config, background_rms=BACKGROUND_RMS,
                        exp_time=EXP_TIME, sees=[source2])

    prob_model = ProbModel(model, [dataset1, dataset2], mode="forward")
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
    return pipeline, prob_model


# ---------------------------------------------------------------------------
# Entry point. Building the model/data/pipeline is always cheap; ONLY the
# actual pipeline.run(...) call (MAP optimization + MCLMC sampling) is gated
# behind --run, so `python dspl_free_r2.py` with no flag is a pure
# construction/data-generation smoke test.
# ---------------------------------------------------------------------------
def main(run: bool):
    model = build_r2_model()
    print(f"[dspl_free_r2] num_free_params = {model.num_free_params}")
    print(f"[dspl_free_r2] z_param_names   = {model.z_param_names}")

    sim_config = build_sim_config()
    truth = truth_params()

    observed_image1, observed_image2, img1, img2 = generate_dataset(
        model, sim_config, truth, seed=0)
    save_dataset(RESULTS_DIR, observed_image1, observed_image2, img1, img2, truth)
    print(f"[dspl_free_r2] wrote dataset + truth to {RESULTS_DIR}")

    pipeline, prob_model = build_pipeline(model, sim_config, observed_image1,
                                           observed_image2)

    if run:
        artifacts = pipeline.run(out_dir=RESULTS_DIR, resume=True)
        print("[dspl_free_r2] pipeline.run complete.")
        return artifacts
    else:
        print("[dspl_free_r2] --run not passed: model/data/pipeline constructed "
              "but MAP+MCLMC NOT launched (awaiting grader approval per the "
              "design checkpoint in docs/logs/sample-cosmology-dspl.md).")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                         help="Actually launch MAP + MCLMC (pipeline.run). "
                              "Omit to only build/construct + smoke-test.")
    args = parser.parse_args()
    main(run=args.run)
