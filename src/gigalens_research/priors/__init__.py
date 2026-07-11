"""Custom grouped (tuple-key) priors for gigalens scene models (research repo).

Library-quality grouped priors live in ``gigalens.jax.grouped_priors``; this
package holds research-stage ones under active investigation. Promote to the
library only after their validating run is certified in the lab log.
"""
from gigalens_research.priors.ratio_coords import (
    RatioCoordsBijector,
    RatioCoordsUniform,
    deflection_ratio_u_fn,
    validate_ratio_coords,
)

__all__ = [
    "RatioCoordsBijector",
    "RatioCoordsUniform",
    "deflection_ratio_u_fn",
    "validate_ratio_coords",
]
