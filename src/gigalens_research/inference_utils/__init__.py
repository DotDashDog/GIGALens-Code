"""Inference pipeline (MAP / SVI / HMC) and posterior diagnostics."""

from .pipeline import (
    PipelineConfig,
    MAPResults,
    SVIResults,
    HMCResults,
    run_pipeline,
    simulate_system,
)
from .diagnostics import (
    get_noise_image,
    get_chisq,
    log_prob_image,
    log_prob_image_patched,
    bridge_sampler,
)

__all__ = [
    "PipelineConfig",
    "MAPResults",
    "SVIResults",
    "HMCResults",
    "run_pipeline",
    "simulate_system",
    "get_noise_image",
    "get_chisq",
    "log_prob_image",
    "log_prob_image_patched",
    "bridge_sampler",
]
