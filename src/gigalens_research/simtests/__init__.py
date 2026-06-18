"""
gigalens_research.simtests
==========================

Model-agnostic framework for large simulated-system inference tests.

A *campaign* fully specifies a test:

- A **generator** creates or adapts a population of mock lensing systems
  (truth model + simulation prior + noise).
- An **inference builder** constructs the inference pipeline for each system
  (inference model + prior + probabilistic model — may differ from truth).
- A **pipeline builder** assembles the sequence of inference stages
  (MAP, SVI, HMC, MCLMC, custom bootstrap …).
- **Metrics** compute convergence and truth-recovery diagnostics per run.
- The **run driver** iterates over (system, sweep_point) pairs, caches each
  stage via ``InferenceContext.hash()`` + ``Pipeline.run(resume=True)``,
  and records results to ``run.json`` + ``index.csv``.
- The **aggregator** produces summary figures across all completed runs.

Typical usage::

    python -m gigalens_research.simtests generate campaign.yaml
    python -m gigalens_research.simtests run    campaign.yaml --shard 0/64
    python -m gigalens_research.simtests aggregate campaign.yaml
    python -m gigalens_research.simtests status   campaign.yaml

See ``experiments/hundred_systems_GL2/campaign.yaml`` and
``experiments/shapelets_systematics/campaign.yaml`` for reference campaigns.
"""

# gigalens defaults to float64 going forward (see docs/project-standards.md).
# float64/mixed likelihood precision requires jax_enable_x64, which JAX reads from
# this env var at import time. setdefault() so an explicit JAX_ENABLE_X64=0 still
# wins; importing the framework before jax (the CLI / Slurm path always does) opts
# the process into x64. Notebook/REPL users who import jax first should set
# jax.config.update("jax_enable_x64", True) themselves — the gigalens precision
# guard raises a clear error otherwise.
import os as _os
_os.environ.setdefault("JAX_ENABLE_X64", "1")

from .config import CampaignSpec, DatasetSpec, ExecutionSpec, InferenceSpec
from .registry import (
    get_generator,
    get_inference_builder,
    get_metric,
    get_pipeline_builder,
    list_registered,
    register_generator,
    register_inference_builder,
    register_metric,
    register_pipeline_builder,
)
from .system import System, from_gl2_npz_entry, from_vela_dir, load_manifest, write_manifest
from .generate import generate_campaign
from .run import enumerate_runs, run_campaign
from .aggregate import aggregate_campaign, register_campaign_metric
from .index import append_to_index, build_index, make_run_record, write_run_json

__all__ = [
    # Config
    "CampaignSpec",
    "DatasetSpec",
    "ExecutionSpec",
    "InferenceSpec",
    # Registry
    "register_generator",
    "register_inference_builder",
    "register_pipeline_builder",
    "register_metric",
    "register_campaign_metric",
    "get_generator",
    "get_inference_builder",
    "get_pipeline_builder",
    "get_metric",
    "list_registered",
    # System I/O
    "System",
    "from_gl2_npz_entry",
    "from_vela_dir",
    "load_manifest",
    "write_manifest",
    # Entry points
    "generate_campaign",
    "enumerate_runs",
    "run_campaign",
    "aggregate_campaign",
    # Index
    "append_to_index",
    "build_index",
    "make_run_record",
    "write_run_json",
]
