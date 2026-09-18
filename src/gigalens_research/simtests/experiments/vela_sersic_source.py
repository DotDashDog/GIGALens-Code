"""Vela single-Sersic *source* baseline: EPL + Shear + SersicEllipse lens +
SersicEllipse source.

This is the smooth-source baseline for the source-systematics comparison
(shapelets vs elliptical-shapelets vs single Sersic). It reuses the exact lens
and lens-light priors of ``vela_shapelets`` and the same Vela systems; only the
source model changes — here a single ``SersicEllipse`` with no shapelet
expansion, so there is no ``n_max`` axis.

Registered here:

- ``"epl_shear_sersic_sersic_source"`` inference builder — a thin wrapper over
  ``vela_shapelets.build_epl_shear_sersic_shapelets(use_shapelets=False)`` so the
  single-Sersic source and its prior have one source of truth. (The name differs
  from GL2's ``epl_shear_sersic_sersic`` builder, which is a different model.)

The source prior (``R_sersic``, ``n_sersic``, ``e1``, ``e2``, ``center_x``,
``center_y``) lives in ``vela_shapelets.vela_inference_prior(use_shapelets=False)``.
"""
from __future__ import annotations

from typing import Any

from gigalens_research.simtests.registry import register_inference_builder
# Importing this registers the shapelets/sersic builder + shared Vela priors.
from gigalens_research.simtests.experiments.vela_shapelets import (
    build_epl_shear_sersic_shapelets,
)


@register_inference_builder("epl_shear_sersic_sersic_source")
def build_epl_shear_sersic_sersic_source(system: Any, **kwargs) -> Any:
    """Vela single-``SersicEllipse`` source baseline.

    No ``n_max`` (smooth source). Delegates to the ``use_shapelets=False`` path
    of the shapelets builder so the source profile and prior are defined once.
    """
    kwargs.pop("use_shapelets", None)  # fixed for this builder
    return build_epl_shear_sersic_shapelets(system, use_shapelets=False, **kwargs)


# ---------------------------------------------------------------------------
# Broad-prior single-Sersic source (2026-09-17, vela_f140w_v3 first fit)
# ---------------------------------------------------------------------------

#: Broad, truth-agnostic source prior (user decision 2026-09-17): nothing here is
#: tuned to the VELA sources. R_sersic covers 0.06"-1.5" at 2 sigma; n_sersic is
#: flat over the physical range; the ellipticity is an exactly-truncated isotropic
#: Gaussian (|e| < e_max, i.e. q > (1-e_max)/(1+e_max)); the centre is wide enough
#: for any source offset the truth prior can draw (N(0, 0.25") per axis).
BROAD_SOURCE_PRIOR: dict = {
    "R_median_arcsec": 0.3,
    "R_log_sigma": 0.8,
    "n_low": 0.5,
    "n_high": 8.0,
    "e_max": 0.8,
    "e_scale": 0.4,
    "center_sigma_arcsec": 0.5,
}


@register_inference_builder("epl_shear_sersic_sersic_source_broad")
def build_epl_shear_sersic_sersic_source_broad(system: Any, **kwargs) -> Any:
    """EPL + Shear + Sersic lens light (the shared Vela lens priors) + ONE
    ``SersicEllipse`` source under the BROAD prior :data:`BROAD_SOURCE_PRIOR`, with
    an optional curvature-adaptive quadrature.

    Kwargs consumed (everything else is ignored, as for every builder):

    ``source_prior`` (dict, optional): overrides for :data:`BROAD_SOURCE_PRIOR`
    keys; unknown keys raise (a typo must not silently keep the default).

    ``adaptive`` (dict, optional): curvature/SNR-adaptive quadrature, see
    :func:`vela_shapelets.make_image_data` (shared by all Vela builders). Without it
    the dataset's ``inference_supersample`` (meta.json) is used uniformly.
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ProbModel
    from gigalens.jax.utils.grouped_priors import TruncatedDiskNormal
    from gigalens_research.simtests.experiments.vela_shapelets import (
        _vela_scene_lens_priors, make_image_data)
    tfd = tfp.distributions

    sp = dict(BROAD_SOURCE_PRIOR)
    overrides = dict(kwargs.get("source_prior") or {})
    unknown = set(overrides) - set(sp)
    if unknown:
        raise KeyError(
            f"build_epl_shear_sersic_sersic_source_broad: unknown source_prior keys "
            f"{sorted(unknown)}; allowed: {sorted(sp)}.")
    sp.update(overrides)

    epl_p, shear_p, lens_light_p = _vela_scene_lens_priors()
    source_p = {
        "R_sersic": tfd.LogNormal(jnp.log(float(sp["R_median_arcsec"])), float(sp["R_log_sigma"])),
        "n_sersic": tfd.Uniform(float(sp["n_low"]), float(sp["n_high"])),
        ("e1", "e2"): TruncatedDiskNormal(e_max=float(sp["e_max"]), scale=float(sp["e_scale"])),
        "center_x": tfd.Normal(0.0, float(sp["center_sigma_arcsec"])),
        "center_y": tfd.Normal(0.0, float(sp["center_sigma_arcsec"])),
    }
    model = LensModel([
        Plane(mass=[Component(epl.EPL(50), epl_p), Component(shear.Shear(), shear_p)],
              light=[Component(sersic.SersicEllipse(use_lstsq=True), lens_light_p)]),
        Plane(deflection_ratio=1.0,
              light=[Component(sersic.SersicEllipse(use_lstsq=True), source_p)]),
    ])

    ds = make_image_data(system, kwargs.get("adaptive"))
    return ProbModel(model, ds, mode="lstsq")
