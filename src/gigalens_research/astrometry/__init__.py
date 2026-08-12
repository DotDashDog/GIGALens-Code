"""Measuring lensed point-source positions (and their covariance) from pixels.

gigalens cannot render point sources — :class:`gigalens.jax.scene.PointSource`
sets ``renders = False``, so the image positions that
:class:`gigalens.jax.point_source_position.PointSourcePositionData` consumes are
*data*, measured somewhere else. This subpackage is that "somewhere else": a
lenstronomy forward model of the cutout that reports image positions and, more
importantly, the ``(2n, 2n)`` astrometric covariance the position likelihood
now accepts as ``cov_img``.

- :mod:`~gigalens_research.astrometry.measure` — the measurement itself.
- :mod:`~gigalens_research.astrometry.validate` — the harness that decides
  whether the reported covariance may be believed.

The second module is not optional garnish. A covariance is a claim about
repeated experiments, and nothing inside a single fit can check that claim; see
that module's docstring for what has to be run before the numbers are used.
"""
from __future__ import annotations

from gigalens_research.astrometry.measure import (
    AstrometryResult,
    Frame,
    NoiseSpec,
    PSFSpec,
    SystematicsBudget,
    measure_astrometry,
)

__all__ = [
    "AstrometryResult",
    "Frame",
    "NoiseSpec",
    "PSFSpec",
    "SystematicsBudget",
    "measure_astrometry",
]
