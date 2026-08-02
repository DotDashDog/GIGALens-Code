"""Batch inference driver for simtests campaigns.

:func:`run_campaign` enumerates all ``(system_id, sweep_point)`` pairs for a
campaign, takes the shard assigned to the current process, and for each run:

1. Loads the :class:`~system.System` from disk.
2. Calls the registered inference builder → scene :class:`ProbModel`.
3. Calls the registered pipeline builder → list of :class:`InferenceStage`.
4. Wraps them in a :class:`~gigalens_research.inference_utils.Pipeline` with
   ``resume=True`` (content-addressed per-stage caching).
5. Computes registered metrics over the finished posterior and the truth.
6. Records peak GPU memory and wall time.
7. Writes ``run.json`` to the run directory and appends to ``index.csv``.

Memory strategy
---------------
The default ``systems_per_task=1`` from :class:`~config.ExecutionSpec` means
each Slurm array task processes exactly one system sequentially, so per-GPU
memory is bounded by a single system's pipeline (MAP starts / sampler chains).
Multiple systems are never batched into one simulator call.
"""
from __future__ import annotations

import os
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def enumerate_runs(
    campaign_spec: Any,
    dataset_dir: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return all ``(system_id, sweep_point)`` pairs in stable order.

    Stable order: systems in manifest order × sweep points in YAML order.
    This ensures every process's ``--shard i/N`` slice is deterministic.
    """
    from .system import load_manifest
    manifest = load_manifest(dataset_dir)
    system_ids = manifest["system_ids"]
    sweep_points = campaign_spec.sweep_points  # already a list of dicts
    return [
        (sid, sp)
        for sid in system_ids
        for sp in sweep_points
    ]


def run_campaign(
    campaign_spec: Any,
    base_dir: str,
    *,
    shard_i: int = 0,
    shard_n: int = 1,
    verbose: bool = True,
    skip_existing: bool = True,
) -> None:
    """Run the inference loop for one shard of a campaign.

    Parameters
    ----------
    campaign_spec : CampaignSpec
    base_dir : str
        Campaign output root (contains ``dataset/``, ``runs/``, ``index.csv``).
    shard_i : int
        Zero-based shard index (Slurm ``$SLURM_ARRAY_TASK_ID``).
    shard_n : int
        Total number of shards (Slurm ``$SLURM_ARRAY_TASK_COUNT``).
    verbose : bool
    skip_existing : bool
        If True, skip runs whose ``run.json`` already records ``status=ok``.
    """
    from gigalens_research.inference_utils.pipeline import (
        InferenceContext, Pipeline,
    )
    from .registry import get_inference_builder, get_pipeline_builder, get_metric
    from .system import System
    from .metrics import peak_gpu_bytes as _peak_gpu
    from .index import write_run_json, append_to_index, make_run_record

    dataset_dir = os.path.join(base_dir, "dataset")
    runs_dir = os.path.join(base_dir, "runs")
    index_path = os.path.join(base_dir, "index.csv")

    all_runs = enumerate_runs(campaign_spec, dataset_dir)
    my_runs = all_runs[shard_i::shard_n]

    if verbose:
        print(
            f"[run] campaign={campaign_spec.name!r}  "
            f"shard {shard_i}/{shard_n}  "
            f"({len(my_runs)} of {len(all_runs)} runs)"
        )

    for run_idx, (system_id, sweep_point) in enumerate(my_runs):
        sweep_name = campaign_spec.sweep_dir_name(sweep_point)
        out_dir = os.path.join(runs_dir, system_id, sweep_name)
        run_json_path = os.path.join(out_dir, "run.json")

        if skip_existing and os.path.exists(run_json_path):
            try:
                import json
                with open(run_json_path) as f:
                    existing = json.load(f)
                if existing.get("status") == "ok":
                    if verbose:
                        print(f"[run] skip {system_id}/{sweep_name} (already ok)")
                    continue
            except Exception:
                pass  # re-run if unreadable

        if verbose:
            print(
                f"\n[run] {run_idx + 1}/{len(my_runs)}  "
                f"{system_id}/{sweep_name}"
            )

        status = "failed"
        record: Optional[Dict[str, Any]] = None
        try:
            record = _run_one(
                campaign_spec=campaign_spec,
                system_id=system_id,
                sweep_point=sweep_point,
                sweep_name=sweep_name,
                dataset_dir=dataset_dir,
                out_dir=out_dir,
                verbose=verbose,
            )
            status = record.get("status", "ok")
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[run] ERROR {system_id}/{sweep_name}: {exc}\n{tb}")
            record = {
                "campaign": campaign_spec.name,
                "system_id": system_id,
                "sweep_name": sweep_name,
                "sweep": sweep_point,
                "status": f"failed: {type(exc).__name__}: {exc}",
                "error_traceback": tb,
                "ctx_hash": "",
                "wall_time_s": 0,
                "peak_gpu_bytes": -1,
                "metrics": {},
                "system_meta": {},
            }

        if record is not None:
            write_run_json(out_dir, record)
            try:
                append_to_index(index_path, record)
            except Exception as exc:
                warnings.warn(f"[run] could not append to index.csv: {exc}", stacklevel=2)


def _run_one(
    *,
    campaign_spec: Any,
    system_id: str,
    sweep_point: Dict[str, Any],
    sweep_name: str,
    dataset_dir: str,
    out_dir: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Run inference for one (system_id, sweep_point) and return a run record."""
    from gigalens_research.inference_utils.pipeline import (
        InferenceContext, Pipeline,
    )
    from .registry import get_inference_builder, get_pipeline_builder, get_metric
    from .system import System
    from .metrics import peak_gpu_bytes as _peak_gpu

    # Load system
    system = System.load(dataset_dir, system_id)

    # Build inference context. The builder receives the MERGED kwargs
    # (pipeline_kwargs overlaid by the sweep point), same as the pipeline
    # builder, so model-level knobs (channel flags, solver settings, priors)
    # can live in ``inference.pipeline_kwargs`` and still be swept. Builders
    # take ``**kwargs`` and ignore what they don't consume; the run cache is
    # unaffected because it keys on the built context's content hash.
    effective_kwargs = campaign_spec.effective_pipeline_kwargs(sweep_point)
    inference_fn = get_inference_builder(campaign_spec.inference.builder)
    prob_model = inference_fn(system, **effective_kwargs)
    ctx = InferenceContext.from_prob_model(prob_model)
    ctx_hash = ctx.hash()

    # Build pipeline stages (pipeline_kwargs merged with sweep_point)
    pipeline_fn = get_pipeline_builder(campaign_spec.inference.pipeline)
    stages = pipeline_fn(system, **effective_kwargs)

    # Assemble and run
    pipeline = Pipeline(ctx, seed=campaign_spec.seed)
    for stage in stages:
        pipeline.add(stage)

    os.makedirs(out_dir, exist_ok=True)

    pre_peak = _peak_gpu()
    t0 = time.perf_counter()

    if verbose:
        print(f"  InferenceContext hash: {ctx_hash}")
        print(f"  Pipeline stages: {[s.instance_name for s in pipeline.stages]}")

    pipeline.run(out_dir=out_dir, resume=True, verbose=verbose)
    wall_time = time.perf_counter() - t0
    post_peak = _peak_gpu()
    peak_bytes = max(post_peak - pre_peak, 0) if pre_peak >= 0 else post_peak

    if verbose:
        print(f"  Wall time: {wall_time:.1f}s  peak GPU: {peak_bytes / 1e9:.2f} GB")

    # Compute metrics
    metric_values: Dict[str, Any] = {}
    try:
        posterior = pipeline.posterior()
        for metric_name in campaign_spec.metrics:
            if metric_name in ("wall_time", "peak_gpu_bytes"):
                continue
            try:
                fn = get_metric(metric_name)
                metric_values[metric_name] = fn(posterior, system)
            except Exception as exc:
                warnings.warn(
                    f"[run] metric {metric_name!r} failed for {system_id}: {exc}",
                    stacklevel=2,
                )
                metric_values[metric_name] = None
    except Exception as exc:
        warnings.warn(f"[run] could not build posterior for {system_id}: {exc}", stacklevel=2)

    # Preflight memory warning (best-effort, informational only)
    _maybe_warn_memory(system, effective_kwargs)

    system_meta = {
        "num_pix": system.num_pix,
        "supersample": system.supersample,
        "background_rms": system.background_rms,
        "exp_time": system.exp_time,
    }
    # Extract n_params / n_chains / n_results from pipeline results.
    for stage_name, result in pipeline.results.items():
        arr = result.arrays
        if "samples_z" in arr:
            system_meta["n_params"] = int(arr["samples_z"].shape[-1])
            system_meta["n_chains"] = int(arr["samples_z"].shape[0])
            system_meta["n_results"] = int(arr["samples_z"].shape[1])
            break
        if "qz_loc" in arr and "n_params" not in system_meta:
            system_meta["n_params"] = int(arr["qz_loc"].shape[-1])

    return {
        "campaign": campaign_spec.name,
        "system_id": system_id,
        "sweep_name": sweep_name,
        "sweep": sweep_point,
        "ctx_hash": ctx_hash,
        "status": "ok",
        "wall_time_s": round(wall_time, 2),
        "peak_gpu_bytes": int(peak_bytes),
        "metrics": metric_values,
        "system_meta": system_meta,
    }


def _maybe_warn_memory(system: Any, effective_kwargs: Dict[str, Any]) -> None:
    """Estimate peak memory footprint and warn if it looks large."""
    try:
        import jax
        num_pix = system.num_pix
        supersample = system.supersample
        n_chains = int(effective_kwargs.get("hmc_n_hmc",
                    effective_kwargs.get("n_chains", 64)))
        n_results = int(effective_kwargs.get("hmc_num_results",
                    effective_kwargs.get("num_results", 1500)))
        n_max = int(effective_kwargs.get("n_max", 0))
        n_params = max(22, (n_max + 1) * (n_max + 2) // 2 + 9 if n_max else 22)

        # Rough estimates (float32)
        img_bytes = (num_pix * supersample) ** 2 * 4
        map_bytes = img_bytes * int(effective_kwargs.get("map_n_samples", 2000))
        sample_bytes = n_chains * n_results * n_params * 4

        total_gb = (map_bytes + sample_bytes) / 1e9
        devices = jax.device_count()

        for dev in jax.local_devices():
            stats = dev.memory_stats()
            if stats:
                avail = stats.get("bytes_limit", stats.get("total_memory", 0))
                if avail > 0 and total_gb / devices > avail / 1e9 * 0.8:
                    warnings.warn(
                        f"[run] estimated memory {total_gb:.1f} GB / {devices} device(s) "
                        f"may exceed 80% of device capacity ({avail / 1e9:.1f} GB). "
                        f"Consider reducing n_samples/n_chains or increasing --shard N.",
                        stacklevel=3,
                    )
                    return
    except Exception:
        pass


def _dummy_key():
    import jax
    return jax.random.PRNGKey(0)
