"""Alternate inference algorithms.

This namespace is for sampler / inference algorithm implementations, not the
pipeline wrappers around GIGA-Lens' MAP / SVI / HMC workflow. Those wrappers
live in :mod:`gigalens_research.inference_utils`.

The validated MCLMC and MAMS samplers have been consolidated into the core
``gigalens`` package. Both graduated out of ``experimental`` and now live at
:mod:`gigalens.jax.inference.mclmc` and :mod:`gigalens.jax.inference.mams`.
They are re-exported here for backward compatibility, so existing
``gigalens_research.inference`` imports (and the ``MCLMCStage`` / ``MAMSStage``
pipeline wrappers) keep working. Prefer importing them from their core modules
in new code.

Note both imports are from the ``mclmc`` / ``mams`` SUBMODULES, not the
:mod:`gigalens.jax.inference` package: the package ``__init__`` binds ``MCLMC``
and ``MAMS`` (aliased from ``MCLMC_JIT`` / ``MAMS_JIT``) but does not bind the
``_JIT`` names.
"""

from gigalens.jax.inference.mclmc import MCLMC, MCLMC_JIT
from gigalens.jax.inference.mams import MAMS, MAMS_JIT
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
    "MAMS",
    "MAMS_JIT",
    "LAPS_JIT",
    "LAPS_blackjax",
    "HessianSurrogate",
]
