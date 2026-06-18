"""Vela SersicShapelets experiment: EPL + Shear + SersicEllipse lens +
SersicShapelets source.

This is the SersicShapelets counterpart to ``vela_shapelets.py``.  It reuses the
``vela_existing`` generator and the same Vela systems; only the *source* model
changes.  Where plain shapelets must raise ``n_max`` to fit a small, bright
central feature (leaking flexibility into the extended source and biasing the
recovered lens), ``SersicShapelets`` pins a Sersic profile to the shapelet
centre to absorb that cusp, so an adequate fit is expected at lower ``n_max``.

Registered here:

- ``"epl_shear_sersic_sersicshapelets"`` inference builder — ``BackwardProbModel``
  (lstsq amplitudes) with a ``SersicShapelets(n_max, use_lstsq=True)`` source.
- ``"map_bootstrap_mclmc_sersicshapelets"`` pipeline builder — fixed-lens MAP
  bootstrap (the profile-agnostic ``pipelines.PartialTruthBootstrapQzStage`` with
  the source left free) → MCLMC started at truth, isolating source-model
  misspecification as the only bias mechanism.

The sweep axis is ``n_max`` (shapelet order), set in the campaign YAML.

Single knob to tune
-------------------
:func:`_source_shape_priors` holds the SersicShapelets source *shape* priors
(``R_sersic``, ``n_sersic``, ``e1``, ``e2``, ``beta``).  Edit ``R_sersic`` there
to control how compact the pinned Sersic component is; the same priors feed both
the inference model and the bootstrap stage so there is one place to change.
"""
from __future__ import annotations

from typing import Any, Dict, List

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from gigalens_research.inference_utils.pipeline import (
    InferenceStage,
    MCLMCStage,
)
from gigalens_research.simtests.pipelines import PartialTruthBootstrapQzStage

from gigalens_research.simtests.registry import (
    register_inference_builder,
    register_pipeline_builder,
)


# ---------------------------------------------------------------------------
# Source SHAPE priors — the single tuning knob for this experiment
# ---------------------------------------------------------------------------


def _source_priors(tfd_mod: Any, jnp: Any) -> Dict[str, Any]:
    """SersicShapelets source shape priors (everything except the centre).

    These are shared by the inference prior and the bootstrap free prior so the
    source model is specified in exactly one place.

    EDIT ``R_sersic`` to set how compact the pinned central Sersic is (a small
    ``R_sersic`` lets the Sersic absorb the bright central cusp, freeing the
    shapelets to model the extended envelope at low ``n_max``).
    """
    return dict(
        R_sersic=tfd_mod.LogNormal(jnp.log(0.1), 0.3),   # <-- TUNE: compact central Sersic
        n_sersic=tfd_mod.Uniform(1.0, 10.0),
        e1=tfd_mod.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
        e2=tfd_mod.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
        beta=tfd_mod.LogNormal(jnp.log(0.7), 0.4),
        center_x=tfd.Normal(0.0, 0.5),
        center_y=tfd.Normal(0.0, 0.5),
    )


# ---------------------------------------------------------------------------
# Inference prior
# ---------------------------------------------------------------------------


def vela_sersicshapelets_inference_prior():
    """Inference prior for the Vela SersicShapelets experiment.

    Lens (EPL + Shear) and lens-light (Sersic) priors mirror
    ``vela_shapelets.vela_inference_prior``; the source is a SersicShapelets
    profile whose nonlinear params are ``{R_sersic, n_sersic, e1, e2,
    center_x, center_y, beta}`` (all amplitudes are solved by lstsq).
    """
    import jax.numpy as jnp

    lens_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
            gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
            e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.06),
            center_y=tfd.Normal(0.0, 0.06),
        )),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            gamma2=tfd.Normal(0.0, 0.1),
        )),
    ])
    lens_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
            e2=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
            center_x=tfd.Normal(0.0, 0.02),
            center_y=tfd.Normal(0.0, 0.02),
        )),
    ])
    source_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            **_source_priors(tfd, jnp),
        )),
    ])
    return tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_prior])


# ---------------------------------------------------------------------------
# Inference builder
# ---------------------------------------------------------------------------


@register_inference_builder("epl_shear_sersic_sersicshapelets")
def build_epl_shear_sersic_sersicshapelets(system: Any, **kwargs) -> Any:
    """Build the Vela BackwardProbModel + ModellingSequence with a
    ``SersicShapelets(n_max, use_lstsq=True)`` source.

    Kwargs: ``n_max`` (REQUIRED; no default — it sets the source model complexity).
    """
    import jax.numpy as jnp
    from gigalens.jax.inference import ModellingSequence
    from gigalens.jax.model import BackwardProbModel
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.light.sersic_shapelets import SersicShapelets
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.model import PhysicalModel

    if "n_max" not in kwargs:
        raise TypeError(
            "build_epl_shear_sersic_sersicshapelets: 'n_max' is required "
            "(no default; it sets the source model complexity)."
        )
    n_max = int(kwargs["n_max"])

    prior = vela_sersicshapelets_inference_prior()

    src_model = SersicShapelets(n_max=n_max, use_lstsq=True, interpolate=False)

    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=True)],
        [src_model],
    )
    prob_model = BackwardProbModel(
        prior,
        jnp.asarray(system.observed_image),
        background_rms=system.background_rms,
        exp_time=system.exp_time,
    )
    return ModellingSequence(phys_model, prob_model, system.sim_config)

