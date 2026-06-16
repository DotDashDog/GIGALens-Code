"""Alternate inference algorithms.

This namespace is for sampler / inference algorithm implementations, not the
pipeline wrappers around GIGA-Lens' MAP / SVI / HMC workflow. Those wrappers
live in :mod:`gigalens_research.inference_utils`.
"""

from .mclmc import MCLMC, MCLMC_JIT
from .laps import (
    LAPS_JIT,
)
from .laps_blackjax import (
    LAPS_blackjax,
)
from .hessian_surrogate import HessianSurrogate

__all__ = [
    "MCLMC",
    "MCLMC_JIT",
    "LAPS_JIT",
    "LAPS_blackjax",
    "HessianSurrogate",
]
