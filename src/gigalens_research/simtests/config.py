"""Campaign configuration dataclasses and YAML loading.

A *campaign* fully specifies a simulated-system inference test:

- :class:`DatasetSpec` — how to generate or load the system population.
- :class:`InferenceSpec` — which inference model, pipeline, and stage kwargs to use.
- The sweep — a list of dicts, one per sweep point (e.g. ``[{"n_max": 10}, {"n_max": 15}]``).
- :class:`ExecutionSpec` — sharding and memory budget hints.
- :class:`CampaignSpec` — top-level container.

YAML format example::

    name: hundred_sersic_v1
    seed: 0
    output_dir: ~/GIGALens-Code/simtests_results/hundred_sersic_v1
    plugins:
      - gigalens_research.simtests.experiments.gl2_sersic
    dataset:
      generator: gl2_existing
      npz_path: ~/GIGALens-Code/SystemSaves/100SystemsStandard80px.npz
      yaml_path: ~/GIGALens-Code/SystemSaves/100SystemsStandardParams.yaml
      delta_pix: 0.065
      num_pix: 80
      supersample: 2
      psf: gl2_hst_psf
      noise_kind: forward
      background_rms: 0.2
      exp_time: 100
    inference:
      builder: epl_shear_sersic_sersic
      pipeline: map_svi_hmc
      pipeline_kwargs:
        map_num_steps: 1000
        map_n_samples: 2000
        svi_num_steps: 5000
        svi_n_vi: 1000
        hmc_n_hmc: 64
        hmc_num_results: 1500
        hmc_num_burnin: 500
    sweep: [{}]
    metrics: [max_rhat, min_ess, all_zscores, wall_time, peak_gpu_bytes]
    execution:
      systems_per_task: 1
"""
from __future__ import annotations

import dataclasses
import itertools
import os
from typing import Any, Dict, List, Optional

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Specification for the simulated system population.

    ``generator`` is a key in the generator registry.  All other fields are
    passed verbatim to the generator function as keyword arguments.
    """
    generator: str
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        d = dict(d)
        generator = d.pop("generator")
        return cls(generator=generator, extra=d)


@dataclasses.dataclass(frozen=True)
class InferenceSpec:
    """Specification for the inference model and pipeline.

    ``builder`` and ``pipeline`` are keys in their respective registries.
    ``pipeline_kwargs`` are additional keyword arguments passed to the pipeline
    builder (merged with the sweep point, sweep point taking precedence).
    """
    builder: str
    pipeline: str
    pipeline_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceSpec":
        d = dict(d)
        builder = d.pop("builder")
        pipeline = d.pop("pipeline")
        pipeline_kwargs = d.pop("pipeline_kwargs", {}) or {}
        return cls(builder=builder, pipeline=pipeline, pipeline_kwargs=pipeline_kwargs)


@dataclasses.dataclass(frozen=True)
class ExecutionSpec:
    """Execution/sharding hints."""
    systems_per_task: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionSpec":
        d = dict(d) if d else {}
        return cls(systems_per_task=int(d.get("systems_per_task", 1)))


@dataclasses.dataclass
class CampaignSpec:
    """Top-level campaign specification.

    ``sweep_points`` is always a list of dicts; YAML shorthand is normalized
    on load.  Each dict contains keyword arguments that override
    ``inference.pipeline_kwargs`` for that run.
    """
    name: str
    seed: int
    output_dir: str
    dataset: DatasetSpec
    inference: InferenceSpec
    sweep_points: List[Dict[str, Any]]
    metrics: List[str]
    execution: ExecutionSpec
    plugins: List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "CampaignSpec":
        d = dict(d)
        name = d["name"]
        seed = int(d.get("seed", 0))
        output_dir = os.path.expanduser(
            str(d.get("output_dir", f"~/GIGALens-Code/simtests_results/{name}"))
        )
        dataset = DatasetSpec.from_dict(d["dataset"])
        inference = InferenceSpec.from_dict(d["inference"])
        metrics = list(d.get("metrics", ["max_rhat", "min_ess", "all_zscores",
                                          "wall_time", "peak_gpu_bytes"]))
        execution = ExecutionSpec.from_dict(d.get("execution", {}))
        plugins = list(d.get("plugins", []))
        sweep_raw = d.get("sweep", None)
        sweep_points = _normalize_sweep(sweep_raw)
        return cls(
            name=name,
            seed=seed,
            output_dir=output_dir,
            dataset=dataset,
            inference=inference,
            sweep_points=sweep_points,
            metrics=metrics,
            execution=execution,
            plugins=plugins,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "CampaignSpec":
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required to load campaign YAML files: pip install pyyaml")
        with open(os.path.expanduser(path)) as f:
            d = _yaml.safe_load(f)
        return cls.from_dict(d)

    def sweep_dir_name(self, sweep_point: Dict[str, Any]) -> str:
        """Return a filesystem-safe directory name for one sweep point."""
        if not sweep_point:
            return "default"
        parts = [f"{k}{v}" for k, v in sorted(sweep_point.items())]
        return "_".join(parts)

    @property
    def pipeline_kwargs_base(self) -> Dict[str, Any]:
        return dict(self.inference.pipeline_kwargs)

    def effective_pipeline_kwargs(self, sweep_point: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base pipeline_kwargs with a sweep point (sweep wins on conflict)."""
        return {**self.pipeline_kwargs_base, **sweep_point}


def _normalize_sweep(raw: Any) -> List[Dict[str, Any]]:
    """Normalize the raw YAML sweep spec to a list of kwarg dicts.

    Supported input forms:
    - ``null`` / ``[]`` / ``[{}]`` → ``[{}]`` (single default run)
    - list of dicts (direct) → as-is
    - dict ``{axis: k, values: [v1, v2, ...]}`` → ``[{k: v1}, {k: v2}, ...]``
    - dict ``{axes: [{k1: [v1a, v1b]}, {k2: [v2a]}]}`` → Cartesian product
    """
    if raw is None or raw == [] or raw == [{}] or raw == {}:
        return [{}]
    if isinstance(raw, list):
        normalized = []
        for item in raw:
            if item is None or item == {}:
                normalized.append({})
            elif isinstance(item, dict):
                normalized.append(item)
            else:
                raise ValueError(f"sweep list entries must be dicts; got {type(item)}")
        return normalized or [{}]
    if isinstance(raw, dict):
        if "axis" in raw and "values" in raw:
            axis = raw["axis"]
            return [{axis: v} for v in raw["values"]]
        if "axes" in raw:
            axis_lists = []
            for ax_spec in raw["axes"]:
                if not isinstance(ax_spec, dict) or len(ax_spec) != 1:
                    raise ValueError(f"axes entries must be single-key dicts; got {ax_spec!r}")
                (k, vals) = next(iter(ax_spec.items()))
                axis_lists.append([{k: v} for v in vals])
            product = []
            for combo in itertools.product(*axis_lists):
                merged: Dict[str, Any] = {}
                for d in combo:
                    merged.update(d)
                product.append(merged)
            return product
    raise ValueError(f"Cannot parse sweep spec: {raw!r}")
