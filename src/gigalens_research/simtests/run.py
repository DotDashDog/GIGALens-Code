"""Batch inference driver for simtests campaigns.

:func:`run_campaign` shards the campaign's systems across processes (all
sweep points of one system run sequentially inside one process) and for each
``(system, sweep_point)`` run:

1. Loads the :class:`~system.System` from disk.
2. Calls the registered inference builder → scene :class:`ProbModel`.
3. Calls the registered pipeline builder → list of :class:`InferenceStage`
   (the sweep point's reserved ``pipeline`` key selects the builder).
4. Splits the stage list into a *trunk* (the leading MAP/SVI stages) and a
   *tail* (bridge + sampler). The trunk runs once per system into a shared,
   config-addressed directory (``runs/<sid>/trunk/<digest>/``); every
   pipeline variant's tail is then seeded with the SAME trunk artifacts
   (``z_best``, ``qz``) via ``seed_artifacts`` — this is what guarantees all
   samplers start from the identical MAP/SVI, byte-for-byte.
5. Runs the tail with ``resume=True`` (content-addressed per-stage caching).
6. Computes registered metrics over the finished posterior and the truth.
7. Records peak GPU memory and wall time (tail-only, i.e. sampler cost;
   trunk cost is recorded separately as ``trunk_wall_time_s``).
8. Writes ``run.json`` to the run directory and appends to ``index.csv``.

Trunk sharing
-------------
The trunk directory digest hashes (ctx, trunk stage configs, seed), so sweep
points that share the model and MAP/SVI settings share one trunk, while a
sweep that varies them (e.g. an ``n_max`` or ``map_num_steps`` sweep) gets
separate trunk directories per configuration instead of cache churn. Because
sharding is by system, no two processes ever touch the same trunk directory.

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
    NOTE: sharding is over *systems* (see :func:`run_campaign`), not over
    these pairs; this enumeration is for status/accounting.
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
        Sharding is over SYSTEMS: every sweep point of a system runs
        sequentially inside the shard that owns it, so trunk (MAP/SVI)
        results are computed once and shared race-free.
    verbose : bool
    skip_existing : bool
        If True, skip runs whose ``run.json`` already records ``status=ok``.
    """
    from .system import load_manifest
    from .index import write_run_json, append_to_index

    dataset_dir = os.path.join(base_dir, "dataset")
    runs_dir = os.path.join(base_dir, "runs")
    index_path = os.path.join(base_dir, "index.csv")

    manifest = load_manifest(dataset_dir)
    my_systems = list(manifest["system_ids"])[shard_i::shard_n]
    sweep_points = campaign_spec.sweep_points
    n_runs = len(my_systems) * len(sweep_points)

    if verbose:
        print(
            f"[run] campaign={campaign_spec.name!r}  "
            f"shard {shard_i}/{shard_n}  "
            f"({len(my_systems)} systems x {len(sweep_points)} sweep points "
            f"= {n_runs} runs)"
        )

    run_idx = 0
    for system_id in my_systems:
        for sweep_point in sweep_points:
            run_idx += 1
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
                    f"\n[run] {run_idx}/{n_runs}  "
                    f"{system_id}/{sweep_name}"
                )

            record: Optional[Dict[str, Any]] = None
            try:
                record = _run_one(
                    campaign_spec=campaign_spec,
                    system_id=system_id,
                    sweep_point=sweep_point,
                    sweep_name=sweep_name,
                    dataset_dir=dataset_dir,
                    runs_dir=runs_dir,
                    out_dir=out_dir,
                    verbose=verbose,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[run] ERROR {system_id}/{sweep_name}: {exc}\n{tb}")
                record = {
                    "campaign": campaign_spec.name,
                    "system_id": system_id,
                    "sweep_name": sweep_name,
                    "sweep": sweep_point,
                    "pipeline": campaign_spec.pipeline_for(sweep_point),
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


def _split_trunk(stages: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Split a stage list into (trunk, tail) at the first non-MAP/SVI stage.

    The trunk — the leading run of :class:`MAPStage` / :class:`SVIStage`
    instances — is what pipeline variants share per system; everything after
    it (bridges, samplers) is variant-specific. A pipeline that starts with a
    custom stage (e.g. ``map_bootstrap_mclmc``) has an empty trunk and runs
    entirely in its own directory, exactly as before.
    """
    from gigalens_research.inference_utils.pipeline import MAPStage, SVIStage

    trunk: List[Any] = []
    for stage in stages:
        if isinstance(stage, (MAPStage, SVIStage)):
            trunk.append(stage)
        else:
            break
    return trunk, stages[len(trunk):]


def _trunk_digest(ctx_hash: str, trunk_stages: List[Any], seed: int) -> str:
    """Config-addressed digest naming the shared trunk directory.

    Deliberately hashes only the trunk ROOT — (ctx, seed, first-stage
    class/schema/config) — NOT the full trunk stage list. This way a
    ``[MAP]`` trunk (map_<sampler> variants) and a ``[MAP, SVI]`` trunk
    (map_svi_<sampler> variants) with the same MAP config land in the SAME
    directory and share one MAP computation byte-for-byte; the SVI stage
    simply adds its own input-hash-cached subdirectory inside it. Sweeps
    that change the model or MAP config (e.g. ``n_max``, ``map_num_steps``)
    change the digest and get cleanly separated trunk directories.

    (The one case that still churns the cache is two sweep points sharing a
    MAP config but sweeping the SVI config: the shared ``svi/`` dir is then
    moved aside and recomputed per pass. Correctness is unaffected — tails
    key on config-addressed artifact ids — it just forfeits SVI reuse.)
    """
    from gigalens_research.inference_utils.pipeline import stable_hash

    root = trunk_stages[0]
    return stable_hash({
        "ctx": ctx_hash,
        "seed": seed,
        "stage0": {
            "class": type(root).__name__,
            "schema": root.schema_version,
            "name": root.instance_name,
            "config": root.config_hash_data(),
        },
    })


def _run_trunk(
    ctx: Any,
    trunk_stages: List[Any],
    seed: int,
    trunk_dir: str,
    verbose: bool,
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any]]:
    """Run (or load) the shared trunk and return
    ``(artifacts, artifact_ids, trunk_meta)``.

    ``artifact_ids`` are config-addressed version strings
    (``<stage input_hash>:<artifact>``) handed to the tail pipeline as
    ``seed_artifact_ids`` — deterministic across reruns, so tail caches
    stay valid, and changing any trunk input invalidates all tails.
    """
    import json

    from gigalens_research.inference_utils.pipeline import Pipeline

    pipeline = Pipeline(ctx, seed=seed)
    for stage in trunk_stages:
        pipeline.add(stage)
    artifacts = pipeline.run(out_dir=trunk_dir, resume=True, verbose=verbose)

    # This run's pipeline.json (just written) records each stage's input hash
    # and cache status; reuse rather than re-deriving the hash logic.
    with open(os.path.join(trunk_dir, "pipeline.json")) as f:
        run_log = {e["stage"]: e for e in json.load(f)["stages"]}

    artifact_ids: Dict[str, str] = {}
    stage_meta: Dict[str, Any] = {}
    for stage in trunk_stages:
        entry = run_log[stage.instance_name]
        for key in stage.produces:
            artifact_ids[key] = f"{entry['input_hash']}:{key}"
        stage_meta[stage.instance_name] = {
            "status": entry.get("status"),
            # When loaded from cache this is the ORIGINAL compute time.
            "wall_time_s": pipeline.results[stage.instance_name].metadata.get("wall_time_s"),
        }

    trunk_meta = {"trunk_dir": trunk_dir, "stages": stage_meta}
    return artifacts, artifact_ids, trunk_meta


def _run_one(
    *,
    campaign_spec: Any,
    system_id: str,
    sweep_point: Dict[str, Any],
    sweep_name: str,
    dataset_dir: str,
    runs_dir: str,
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

    # Build pipeline stages (the sweep point's reserved ``pipeline`` key
    # selects the builder; kwargs are pipeline_kwargs merged with sweep_point)
    pipeline_name = campaign_spec.pipeline_for(sweep_point)
    pipeline_fn = get_pipeline_builder(pipeline_name)
    stages = pipeline_fn(system, **effective_kwargs)
    run_seed = campaign_spec.run_seed(system_id)

    # Shared trunk: run the leading MAP/SVI stages once per system (per trunk
    # config) and seed every variant's tail with the identical artifacts.
    trunk_stages, tail_stages = _split_trunk(stages)
    seed_artifacts: Dict[str, Any] = {}
    seed_artifact_ids: Dict[str, str] = {}
    trunk_meta: Optional[Dict[str, Any]] = None
    if trunk_stages and tail_stages:
        trunk_dir = os.path.join(
            runs_dir, system_id, "trunk",
            _trunk_digest(ctx_hash, trunk_stages, run_seed),
        )
        if verbose:
            print(f"  Trunk: {[s.instance_name for s in trunk_stages]} -> {trunk_dir}")
        trunk_artifacts, seed_artifact_ids, trunk_meta = _run_trunk(
            ctx, trunk_stages, run_seed, trunk_dir, verbose,
        )
        seed_artifacts = {k: trunk_artifacts[k] for k in seed_artifact_ids}
    else:
        tail_stages = stages  # nothing to share; run the pipeline whole

    # Assemble and run the (tail) pipeline
    pipeline = Pipeline(ctx, seed=run_seed)
    for stage in tail_stages:
        pipeline.add(stage)

    os.makedirs(out_dir, exist_ok=True)

    pre_peak = _peak_gpu()
    t0 = time.perf_counter()

    if verbose:
        print(f"  InferenceContext hash: {ctx_hash}")
        print(f"  Pipeline: {pipeline_name}  "
              f"stages: {[s.instance_name for s in pipeline.stages]}")

    pipeline.run(
        out_dir=out_dir,
        resume=True,
        verbose=verbose,
        seed_artifacts=seed_artifacts or None,
        seed_artifact_ids=seed_artifact_ids or None,
    )
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

    trunk_wall = None
    if trunk_meta is not None:
        stage_walls = [
            s.get("wall_time_s") for s in trunk_meta["stages"].values()
        ]
        if all(isinstance(w, (int, float)) for w in stage_walls):
            trunk_wall = round(sum(stage_walls), 2)

    return {
        "campaign": campaign_spec.name,
        "system_id": system_id,
        "sweep_name": sweep_name,
        "sweep": sweep_point,
        "pipeline": pipeline_name,
        "seed": run_seed,
        "ctx_hash": ctx_hash,
        "status": "ok",
        # Tail-only: the cost of the bridge + sampler, NOT MAP/SVI (which may
        # have been loaded from the shared trunk cache).
        "wall_time_s": round(wall_time, 2),
        # Sum of the trunk stages' compute times (original times if cached).
        "trunk_wall_time_s": trunk_wall,
        "trunk": trunk_meta,
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
