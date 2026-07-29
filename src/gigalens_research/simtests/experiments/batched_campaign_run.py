"""Campaign runner over the batched MAP -> SVI -> MCLMC pipeline (phase C).

Runs a simtests campaign (single sweep point) through
``batched_pipeline.batched_map_svi_mclmc`` in GPU waves, then persists each
system's results in the EXACT on-disk layout the solo ``run.py`` /
``inference_utils.pipeline.Pipeline`` produce:

    runs/<system_id>/<sweep_name>/
        model_card.json
        map/{manifest.json,arrays.npz}        z_best, lp_hist, chisq_hist
        svi/{manifest.json,arrays.npz}        qz_loc, qz_scale_tril, svi_loss_hist
        mclmc/{manifest.json,arrays.npz}      samples_z (C, N, P)
        mclmc/batched_extras.npz              adapted ss/L/mass matrix, nonan frac
        run.json                              metrics + status, as run.py writes

Fidelity notes:

- Stage manifests carry a REAL ``input_hash``, computed with the same recipe as
  ``Pipeline.run`` (ctx hash, stage class/schema/config, seed, upstream
  artifact hashes chained through the stages' own ``derive_artifacts``), using
  stage objects built by the campaign's registered pipeline builder. A later
  solo ``python -m gigalens_research.simtests run`` therefore RESUMES from
  these results instead of recomputing. Each stage's metadata records
  ``produced_by: batched_pipeline`` so provenance stays explicit — the arrays
  mirror the solo math/RNG streams but are not bitwise what a solo run would
  produce (certified equivalent: see docs/logs/batched-point-source.md).
- Seeds mirror solo semantics exactly: every stage of every system uses the
  campaign seed (``Pipeline(ctx, seed=spec.seed)`` + ``effective_seed``), which
  is also what the batched stages' RNG streams were certified against.
- Metrics are computed from ``posterior_from_disk`` — the same view metrics get
  in ``run.py`` — so the persisted layout is validated end-to-end in-run.
- ``index.csv`` is NOT appended (multi-process runs would race); run
  ``aggregate`` afterwards, which rebuilds it from run.json files.

Usage (one process per GPU; split systems with --offset/--n-systems):

    CUDA_VISIBLE_DEVICES=0 python -m gigalens_research.simtests.experiments.batched_campaign_run \
        experiments/hundred_point_sources/campaign_v2_quad.yaml \
        --offset 0 --n-systems 25 --chunk 13
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time
import warnings
from typing import Any, Dict, List

import numpy as np


def _persist_system(
    *,
    spec: Any,
    system: Any,
    system_id: str,
    sweep_name: str,
    model_seq: Any,
    stages: List[Any],
    out: Dict[str, np.ndarray],
    i: int,
    wall_time_s: float,
    peak_gpu_bytes: int,
    out_dir: str,
) -> Dict[str, Any]:
    """Write one system's stage dirs + run.json; return the run record."""
    from gigalens_research.inference_utils.pipeline import (
        InferenceContext, _save_stage, _write_manifest, model_card,
        _make_json_safe, stable_hash, posterior_from_disk,
    )
    from gigalens_research.simtests.registry import get_metric
    from gigalens_research.simtests.index import write_run_json, make_run_record

    ctx = InferenceContext.from_modelling_sequence(model_seq)
    ctx_hash = ctx.hash()
    os.makedirs(out_dir, exist_ok=True)
    _write_manifest(os.path.join(out_dir, "model_card.json"),
                    _make_json_safe(model_card(ctx)))

    # Per-stage arrays in the solo stages' exact formats.
    best_step = int(out["best_step"][i])
    lp_hist = np.asarray(out["lp_hist"][i])
    chisq_hist = np.asarray(out["chisq_hist"][i])
    stage_payloads = {
        "map": {
            "arrays": {
                "z_best": np.asarray(out["z_best"][i]),
                "lp_hist": lp_hist,
                "chisq_hist": chisq_hist,
            },
            "extra_meta": {
                "best_step": best_step,
                "best_lp": float(lp_hist[best_step]),
                "best_chisq": float(chisq_hist[best_step]),
            },
            "extras": None,
        },
        "svi": {
            "arrays": {
                "qz_loc": np.asarray(out["qz_loc"][i]),
                "qz_scale_tril": np.asarray(out["qz_scale_tril"][i]),
                "svi_loss_hist": np.asarray(out["svi_loss_hist"][i]),
            },
            "extra_meta": {
                "final_loss": float(np.asarray(out["svi_loss_hist"][i])[-1]),
            },
            "extras": None,
        },
        "mclmc": {
            "arrays": {"samples_z": np.asarray(out["samples_z"][i])},
            "extra_meta": {
                "num_chains": int(out["samples_z"][i].shape[0]),
                "num_steps": int(out["samples_z"][i].shape[1]),
                "n_params": int(out["samples_z"][i].shape[2]),
                "debug": False,
                "results_nonan_frac": float(out["results_nonan_frac"][i]),
                "final_L": float(out["final_L"][i]),
            },
            "extras": {
                "final_step_size": np.asarray(out["final_step_size"][i]),
                "final_inverse_mass_matrix":
                    np.asarray(out["final_inverse_mass_matrix"][i]),
            },
        },
    }

    # Hash chain exactly as Pipeline.run computes it (publish via the stages'
    # own derive_artifacts so a solo resume recomputes identical hashes from
    # the loaded arrays).
    artifact_hashes: Dict[str, str] = {}
    run_log = []
    for stage in stages:
        name = stage.instance_name
        payload = stage_payloads[name]
        seed = stage.effective_seed(int(spec.seed))
        upstream = {k: artifact_hashes[k] for k in stage.requires}
        input_hash = stable_hash({
            "ctx": ctx_hash,
            "class": type(stage).__name__,
            "schema": stage.schema_version,
            "config": stage.config_hash_data(),
            "seed": seed,
            "upstream": upstream,
        })
        per_stage_wall = wall_time_s * {"map": 0.15, "svi": 0.1,
                                        "mclmc": 0.75}[name]
        manifest = {
            "stage": name,
            "class": type(stage).__name__,
            "schema_version": stage.schema_version,
            "input_hash": input_hash,
            "upstream_hashes": upstream,
            "config": _make_json_safe(stage.config_hash_data()),
            "diagnostics_config": _make_json_safe(stage.diagnostics_config()),
            "seed": seed,
            "metadata": _make_json_safe({
                # Wave wall time apportioned by typical stage share —
                # informational only (the batch shares one program).
                "wall_time_s": per_stage_wall,
                "num_steps": getattr(stage, "num_steps", None),
                "produced_by": "batched_pipeline",
                **payload["extra_meta"],
            }),
            "arrays": sorted(payload["arrays"].keys()),
            "diagnostics": [],
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        stage_dir = os.path.join(out_dir, name)
        _save_stage(stage_dir, manifest, payload["arrays"], None)
        if payload["extras"]:
            np.savez(os.path.join(stage_dir, "batched_extras.npz"),
                     **payload["extras"])
        for k, v in stage.derive_artifacts(payload["arrays"]).items():
            artifact_hashes[k] = stable_hash(v)
        run_log.append({"stage": name, "class": type(stage).__name__,
                        "status": "ran", "input_hash": input_hash,
                        "wall_time_s": per_stage_wall})

    _write_manifest(os.path.join(out_dir, "pipeline_manifest.json"), {
        "ctx_hash": ctx_hash,
        "seed": int(spec.seed),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "stages": run_log,
        "produced_by": "batched_pipeline",
    })

    # Metrics over the persisted posterior — same view run.py's metrics get.
    metric_values: Dict[str, Any] = {}
    try:
        posterior = posterior_from_disk(out_dir, "mclmc", ctx)
        for metric_name in spec.metrics:
            if metric_name in ("wall_time", "peak_gpu_bytes"):
                continue
            try:
                metric_values[metric_name] = get_metric(metric_name)(
                    posterior, system)
            except Exception as exc:
                warnings.warn(f"[batched_run] metric {metric_name!r} failed "
                              f"for {system_id}: {exc}", stacklevel=2)
                metric_values[metric_name] = None
    except Exception as exc:
        warnings.warn(f"[batched_run] could not build posterior for "
                      f"{system_id}: {exc}", stacklevel=2)

    record = make_run_record(
        campaign=spec.name,
        system_id=system_id,
        sweep_name=sweep_name,
        sweep_point={},
        ctx_hash=ctx_hash,
        status="ok",
        metrics=metric_values,
        wall_time_s=wall_time_s,
        peak_gpu_bytes=int(peak_gpu_bytes),
        system_meta={
            "num_pix": system.num_pix,
            "supersample": system.supersample,
            "background_rms": system.background_rms,
            "exp_time": system.exp_time,
            "n_params": int(out["samples_z"][i].shape[-1]),
            "n_chains": int(out["samples_z"][i].shape[0]),
            "n_results": int(out["samples_z"][i].shape[1]),
        },
    )
    record["runner"] = "batched_pipeline"
    write_run_json(out_dir, record)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign", help="Path to campaign YAML.")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip this many (manifest-ordered) systems first.")
    ap.add_argument("--n-systems", type=int, default=-1,
                    help="How many systems to run (-1 = all remaining).")
    ap.add_argument("--chunk", type=int, default=13,
                    help="Systems per GPU wave. MAP's grad costs ~125 MB/system"
                         " on this likelihood; 13 fits an 11 GB card at "
                         "campaign budgets with margin.")
    ap.add_argument("--rerun", action="store_true",
                    help="Re-run systems whose run.json already says ok.")
    args = ap.parse_args()

    import jax

    from gigalens_research.simtests.config import CampaignSpec
    from gigalens_research.simtests.system import System, load_manifest
    from gigalens_research.simtests.registry import (
        get_inference_builder, get_pipeline_builder,
    )
    from gigalens_research.simtests.index import load_run_json
    from gigalens_research.simtests.metrics import peak_gpu_bytes as _peak_gpu
    import importlib

    spec = CampaignSpec.from_yaml(args.campaign)
    for mod in getattr(spec, "plugins", []) or []:
        importlib.import_module(mod)
    if len(spec.sweep_points) != 1:
        raise SystemExit("batched_campaign_run supports exactly one sweep "
                         f"point; campaign has {len(spec.sweep_points)}.")
    sweep_name = spec.sweep_dir_name(spec.sweep_points[0])
    kw = dict(spec.effective_pipeline_kwargs(spec.sweep_points[0]))

    base_dir = os.path.expanduser(spec.output_dir)
    dataset_dir = os.path.join(base_dir, "dataset")
    runs_dir = os.path.join(base_dir, "runs")
    dev = jax.devices()[0]
    print(f"[batched_run] backend: {dev.platform} ({dev.device_kind})")

    sids = list(load_manifest(dataset_dir)["system_ids"])
    stop = len(sids) if args.n_systems < 0 else args.offset + args.n_systems
    sids = sids[args.offset:stop]
    if not args.rerun:
        kept = []
        for sid in sids:
            rec = load_run_json(os.path.join(runs_dir, sid, sweep_name))
            if rec is not None and rec.get("status") == "ok":
                continue
            kept.append(sid)
        if len(kept) < len(sids):
            print(f"[batched_run] skipping {len(sids) - len(kept)} systems "
                  "with status=ok (use --rerun to force)")
        sids = kept
    if not sids:
        print("[batched_run] nothing to do.")
        return
    print(f"[batched_run] {len(sids)} systems: {sids[0]}..{sids[-1]} "
          f"(waves of <= {args.chunk})")

    from gigalens_research.simtests.experiments.batched_point_source import (
        BatchedPointSourceProb,
    )
    from gigalens_research.simtests.experiments.batched_pipeline import (
        batched_map_svi_mclmc,
    )

    t0 = time.perf_counter()
    systems = [System.load(dataset_dir, sid) for sid in sids]
    builder = get_inference_builder(spec.inference.builder)
    seqs = [builder(s, **kw) for s in systems]
    probs = [getattr(q, "prob_model", None) or q for q in seqs]
    pipeline_fn = get_pipeline_builder(spec.inference.pipeline)
    print(f"[batched_run] built {len(sids)} models in "
          f"{time.perf_counter() - t0:.1f} s")

    knobs = {k: kw[k] for k in
             ("map_num_steps", "map_n_samples", "map_lr", "map_clip_norm",
              "svi_num_steps", "svi_n_vi", "svi_init_scale", "svi_lr",
              "n_chains", "num_burnin_steps", "num_results",
              "desired_energy_variance", "frac_tune1", "frac_tune2",
              "frac_tune3", "mc_anneal_eps", "mc_anneal_steps",
              "mc_anneal_particles", "mc_anneal_lr", "mc_anneal_block")
             if k in kw}
    print(f"[batched_run] knobs: {knobs}")

    seed = int(spec.seed)
    waves = [list(range(a, min(a + args.chunk, len(sids))))
             for a in range(0, len(sids), args.chunk)]
    t0 = time.perf_counter()
    n_done = 0
    for w, idxs in enumerate(waves):
        wave_t0 = time.perf_counter()
        pre_peak = _peak_gpu()
        bp = BatchedPointSourceProb.from_probs([probs[i] for i in idxs])
        seeds = np.full(len(idxs), seed, dtype=np.int64)
        out = batched_map_svi_mclmc(
            bp, map_seeds=seeds, svi_seeds=seeds, mclmc_seeds=seeds, **knobs)
        wave_wall = time.perf_counter() - wave_t0
        post_peak = _peak_gpu()
        peak = max(post_peak - pre_peak, 0) if pre_peak >= 0 else post_peak

        for j, i in enumerate(idxs):
            sid = sids[i]
            rec = _persist_system(
                spec=spec, system=systems[i], system_id=sid,
                sweep_name=sweep_name, model_seq=seqs[i],
                stages=pipeline_fn(systems[i], **kw), out=out, i=j,
                wall_time_s=wave_wall / len(idxs), peak_gpu_bytes=peak,
                out_dir=os.path.join(runs_dir, sid, sweep_name))
            m = rec["metrics"]
            print(f"  [{sid}] R-hat {m.get('max_rhat')} ESS {m.get('min_ess')} "
                  f"nonan_frac {1.0 - (m.get('nan_rate') or 0.0):.3f}")
        n_done += len(idxs)
        print(f"[batched_run] wave {w + 1}/{len(waves)} done "
              f"({len(idxs)} systems, {wave_wall:.0f} s wave, "
              f"{n_done}/{len(sids)} total at "
              f"t={time.perf_counter() - t0:.0f} s)", flush=True)

    print(f"[batched_run] DONE: {len(sids)} systems in "
          f"{time.perf_counter() - t0:.0f} s "
          f"({(time.perf_counter() - t0) / len(sids):.1f} s/system). "
          f"Run 'aggregate' to refresh index.csv and figures.")


if __name__ == "__main__":
    main()
