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
