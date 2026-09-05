"""Custom grouped (tuple-key) priors for gigalens scene models (research repo).

Library-quality grouped priors live in ``gigalens.jax.grouped_priors``; this
package holds research-stage ones under active investigation. Promote to the
library only after their validating run is certified in the lab log.
"""
from gigalens_research.priors.ratio_coords import (
    RatioCoordsBijector,
    RatioCoordsUniform,
    UFirstRatioCoordsBijector,
    UFirstRatioCoordsUniform,
    deflection_ratio_u_fn,
    validate_ratio_coords,
    validate_u_first_ratio_coords,
)
from gigalens_research.priors.ratio_pair_coords import (
    RatioPairBijector,
    RatioPairUniform,
    deflection_ratio_pair_fn,
    validate_ratio_pair,
)
from gigalens_research.priors.ratio_pair_wa import (
    RatioPairWaBijector,
    RatioPairWaUniform,
    deflection_ratio_pair_wa_fn,
    validate_ratio_pair_wa,
)

__all__ = [
    "RatioCoordsBijector",
    "RatioCoordsUniform",
    "RatioPairBijector",
    "RatioPairUniform",
    "RatioPairWaBijector",
    "RatioPairWaUniform",
    "UFirstRatioCoordsBijector",
    "UFirstRatioCoordsUniform",
    "deflection_ratio_pair_fn",
    "deflection_ratio_pair_wa_fn",
    "deflection_ratio_u_fn",
    "validate_ratio_coords",
    "validate_ratio_pair",
    "validate_ratio_pair_wa",
    "validate_u_first_ratio_coords",
]
