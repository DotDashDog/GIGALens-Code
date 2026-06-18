"""Inference pipeline (MAP / SVI / HMC / MCLMC / ...) and posterior diagnostics.

The pipeline is a sequence of :class:`~.pipeline.InferenceStage` objects with
per-stage on-disk caching keyed by an input hash. See :mod:`.pipeline` for the
design and ``Pipeline.run`` for the user-facing entry point.
"""

from .pipeline import (
    BridgeStage,
    HessianSurrogateStage,
    HMCStage,
    InferenceContext,
    InferenceStage,
    MAPStage,
    MCLMCStage,
    Pipeline,
    PipelineMismatchError,
    StageDiagnostics,
    StageResult,
    SVIStage,
    diagnostics_from_disk,
    format_model_card,
    model_card,
    posterior_from_disk,
    register_stage,
    stable_hash,
)
from .posterior import (
    PointEstimate,
    Posterior,
    SamplerPosterior,
    SurrogatePosterior,
)
from .truth_diagnostics import (
    filter_labels_by_group,
    source_comparison,
    truth_source_from_light_model,
    z_scores,
)
from .diagnostics import (
    bridge_sampler,
    get_chisq,
    get_noise_image,
    log_prob_image,
    log_prob_image_patched,
)

__all__ = [
    "BridgeStage",
    "HessianSurrogateStage",
    "HMCStage",
    "InferenceContext",
    "InferenceStage",
    "MAPStage",
    "MCLMCStage",
    "Pipeline",
    "PipelineMismatchError",
    "PointEstimate",
    "Posterior",
    "SamplerPosterior",
    "StageDiagnostics",
    "StageResult",
    "SurrogatePosterior",
    "SVIStage",
    "diagnostics_from_disk",
    "format_model_card",
    "model_card",
    "filter_labels_by_group",
    "posterior_from_disk",
    "register_stage",
    "source_comparison",
    "stable_hash",
    "truth_source_from_light_model",
    "z_scores",
    "bridge_sampler",
    "get_chisq",
    "get_noise_image",
    "log_prob_image",
    "log_prob_image_patched",
]
