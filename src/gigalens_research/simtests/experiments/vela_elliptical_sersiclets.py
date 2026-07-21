"""Vela shapelets experiment: EPL + Shear + SersicEllipse lens + ShapeletsFast source.

This module registers:

- ``"vela_existing"`` generator — adapts the existing
  ``data/vela_sim_systems/`` directories into the framework's
  :class:`~system.System` format (reads ``lens_img.npy`` + pickled
  ``true_params``).  The generation notebook
  (``experiments/vela_sim_systems/lens_vela_system.ipynb``) remains the
  canonical simulation tool; this adapter brings existing results into the
  framework without re-simulating.
- ``"epl_shear_sersic_shapelets"`` inference builder — builds the Vela
  inference :class:`~gigalens.jax.inference.ModellingSequence` using
  ``BackwardProbModel`` (lstsq amplitudes) and ``ShapeletsFast`` source.
- ``"map_bootstrap_mclmc"`` pipeline builder (registered in ``pipelines.py``).

The sweep axis for this experiment is typically ``n_max`` (shapelet order).
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from gigalens_research.simtests.registry import (
    register_generator,
    register_inference_builder,
)


# ---------------------------------------------------------------------------
# Default Vela paths
# ---------------------------------------------------------------------------

_HOME = os.path.expanduser("~")
_DEFAULT_SYSTEM_DIR_ROOT = os.path.join(_HOME, "GIGALens-Code", "data", "vela_sim_systems")
_DEFAULT_SOURCE_DIR_ROOT = os.path.join(_HOME, "GIGALens-Code", "data", "vela_sources")

_DEFAULT_VELA_IDS = ["01", "03", "04", "07", "08", "10", "15", "21", "22", "23", "25", "26"]
_DEFAULT_CAM = "12"
_DEFAULT_FILTER_TAG = "a0.500_f814w"


# ---------------------------------------------------------------------------
# Vela inference prior
# ---------------------------------------------------------------------------


def vela_inference_prior():
    """Inference prior for the Vela elliptical shapelets experiment.

    Mirrors ``vela_utilities.vela_priors()``.
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    # New gigalens (dev refactor) expects the prior to emit the dict-keyed param
    # structure {'lens_mass': {'0': ..}, 'lens_light': {'0': ..}, 'source_light':
    # {'0': ..}} rather than the old 3-list JointDistributionSequential. Each
    # component is a JointDistributionNamed keyed by stringified profile index,
    # matching gigalens.prior.LensPrior / CompoundPrior output.
    return tfd.JointDistributionNamed({
        'lens_mass': tfd.JointDistributionNamed({
            '0': tfd.JointDistributionNamed(dict(
                theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
                e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
                center_x=tfd.Normal(0.0, 0.06),
                center_y=tfd.Normal(0.0, 0.06),
            )),
            '1': tfd.JointDistributionNamed(dict(
                gamma1=tfd.TruncatedNormal(0.0, 0.1, -0.5, 0.5),
                gamma2=tfd.Normal(0.0, 0.1),
            )),
        }),
        'lens_light': tfd.JointDistributionNamed({
            '0': tfd.JointDistributionNamed(dict(
                R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
                n_sersic=tfd.Uniform(0.5, 8.0),
                e1=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
                e2=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
                center_x=tfd.Normal(0.0, 0.02),
                center_y=tfd.Normal(0.0, 0.02),
            )),
        }),
        'source_light': tfd.JointDistributionNamed({
            '0': tfd.JointDistributionNamed(dict(
                beta=tfd.LogNormal(jnp.log(0.7), 0.4),
                e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
                n_sersic=tfd.Uniform(0.3, 8.0),
                center_x=tfd.Normal(0.0, 0.5),
                center_y=tfd.Normal(0.0, 0.5),
            )),
        }),
    })


# ---------------------------------------------------------------------------
# Inference builder
# ---------------------------------------------------------------------------


def _vela_scene_priors():
    """Per-parameter scene priors (free tfd.Distributions) for the sersiclets build.

    Same marginals as ``vela_inference_prior`` (the old 3-group prior), but emitted as
    flat per-param dicts the scene ``Component`` consumes. Each distribution is a FRESH
    object so nothing is linked by identity (independence is the scene default)."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    epl_p = dict(
        theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
        gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
        e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.06),
        center_y=tfd.Normal(0.0, 0.06),
    )
    shear_p = dict(
        gamma1=tfd.TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        gamma2=tfd.Normal(0.0, 0.1),
    )
    lens_light_p = dict(
        R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
        n_sersic=tfd.Uniform(0.5, 8.0),
        e1=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
        e2=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
        center_x=tfd.Normal(0.0, 0.02),
        center_y=tfd.Normal(0.0, 0.02),
    )
    source_p = dict(
        beta=tfd.LogNormal(jnp.log(0.7), 0.4),
        n_sersic=tfd.Uniform(0.3, 8.0),
        e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.5),
        center_y=tfd.Normal(0.0, 0.5),
    )
    return epl_p, shear_p, lens_light_p, source_p


@register_inference_builder("epl_shear_sersic_elliptical_sersiclets")
def build_epl_shear_sersic_elliptical_sersiclets(system: Any, **kwargs) -> Any:
    """Build the SCENE ModellingSequence for the Vela elliptical-sersiclets fit (G1).

    Migrated to the scene API: a :class:`gigalens.jax.scene.LensModel` (EPL+Shear mass +
    Sérsic lens light on plane 0; ``EllipticalSersiclets`` source on plane 1, lstsq amps)
    + a :class:`gigalens.jax.scene_prob_model.Dataset` + ``ProbModel(mode="lstsq")``,
    wrapped in a scene-backed ``ModellingSequence``. The public return
    type and role are unchanged; only the internals are scene objects.

    Kwargs: ``n_max`` (REQUIRED; no default — it sets the source model complexity).
    """
    import jax.numpy as jnp
    from gigalens.jax.inference import ModellingSequence
    from gigalens.jax.profiles.light import sersic
    from gigalens_research.simulations.sersiclets import EllipticalSersiclets
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ImageData, ProbModel

    if "n_max" not in kwargs:
        raise TypeError(
            "build_epl_shear_sersic_elliptical_sersiclets: 'n_max' is required "
            "(no default; it sets the source model complexity)."
        )
    n_max = int(kwargs["n_max"])

    epl_p, shear_p, lens_light_p, source_p = _vela_scene_priors()

    model = LensModel([
        Plane(mass=[Component(epl.EPL(50), epl_p), Component(shear.Shear(), shear_p)],
              light=[Component(sersic.SersicEllipse(use_lstsq=True), lens_light_p)]),
        Plane(deflection_ratio=1.0,
              light=[Component(EllipticalSersiclets(n_max=n_max, use_lstsq=True),
                               source_p)]),
    ])
    ds = ImageData(
        jnp.asarray(system.observed_image),
        system.sim_config,
        background_rms=system.background_rms,
        exp_time=system.exp_time,
        sees="all",
    )
    prob_model = ProbModel(model, ds, mode="lstsq")
    return ModellingSequence(prob_model)
