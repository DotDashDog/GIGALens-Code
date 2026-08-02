"""Extension-point registries for the simtests framework.

Four registries correspond to the four extension points in a campaign:

- **generator**: loads or creates a population of :class:`~system.System` objects.
  Signature: ``fn(spec: DatasetSpec, base_dir: str, seed: int) -> None``
  (writes systems to disk under ``base_dir/dataset/``).

- **inference_builder**: constructs the inference scene :class:`ProbModel`
  for one system, including the inference prior and probabilistic model.
  Signature: ``fn(system: System, **sweep_kwargs) -> ProbModel``.

- **pipeline_builder**: returns the ordered list of
  :class:`~gigalens_research.inference_utils.InferenceStage` objects for one run.
  Signature: ``fn(system: System, **kwargs) -> list[InferenceStage]``.

- **metric**: computes a scalar or dict metric from a finished posterior and the
  known truth.
  Signature: ``fn(posterior: Posterior, system: System) -> Any``.

Usage::

    from gigalens_research.simtests.registry import register_generator

    @register_generator("my_generator")
    def my_generator(spec, base_dir, seed):
        ...
"""
from __future__ import annotations

from typing import Any, Callable, Dict

_GENERATORS: Dict[str, Callable] = {}
_INFERENCE_BUILDERS: Dict[str, Callable] = {}
_PIPELINE_BUILDERS: Dict[str, Callable] = {}
_METRICS: Dict[str, Callable] = {}


def register_generator(name: str) -> Callable:
    """Decorator that registers a generator function under ``name``."""
    def decorator(fn: Callable) -> Callable:
        _GENERATORS[name] = fn
        return fn
    return decorator


def register_inference_builder(name: str) -> Callable:
    """Decorator that registers an inference-builder function under ``name``."""
    def decorator(fn: Callable) -> Callable:
        _INFERENCE_BUILDERS[name] = fn
        return fn
    return decorator


def register_pipeline_builder(name: str) -> Callable:
    """Decorator that registers a pipeline-builder function under ``name``."""
    def decorator(fn: Callable) -> Callable:
        _PIPELINE_BUILDERS[name] = fn
        return fn
    return decorator


def register_metric(name: str) -> Callable:
    """Decorator that registers a metric function under ``name``."""
    def decorator(fn: Callable) -> Callable:
        _METRICS[name] = fn
        return fn
    return decorator


def get_generator(name: str) -> Callable:
    if name not in _GENERATORS:
        raise KeyError(
            f"Generator {name!r} not found. "
            f"Available: {sorted(_GENERATORS)} "
            f"(did you import the relevant experiments module?)"
        )
    return _GENERATORS[name]


def get_inference_builder(name: str) -> Callable:
    if name not in _INFERENCE_BUILDERS:
        raise KeyError(
            f"Inference builder {name!r} not found. "
            f"Available: {sorted(_INFERENCE_BUILDERS)}"
        )
    return _INFERENCE_BUILDERS[name]


def get_pipeline_builder(name: str) -> Callable:
    if name not in _PIPELINE_BUILDERS:
        raise KeyError(
            f"Pipeline builder {name!r} not found. "
            f"Available: {sorted(_PIPELINE_BUILDERS)}"
        )
    return _PIPELINE_BUILDERS[name]


def get_metric(name: str) -> Callable:
    if name not in _METRICS:
        raise KeyError(
            f"Metric {name!r} not found. "
            f"Available: {sorted(_METRICS)}"
        )
    return _METRICS[name]


def list_registered() -> Dict[str, list]:
    """Return a summary dict of all registered names (useful for debugging)."""
    return {
        "generators": sorted(_GENERATORS),
        "inference_builders": sorted(_INFERENCE_BUILDERS),
        "pipeline_builders": sorted(_PIPELINE_BUILDERS),
        "metrics": sorted(_METRICS),
    }
