"""Run index: per-run ``run.json`` records + merged campaign ``index.csv``.

Each completed run writes:
1. A ``run.json`` file inside the run's output directory (always written, no
   locking required — one file per run — this is the canonical record).
2. A per-run staging CSV in ``<base_dir>/index_rows/<sid>_<sweep>.csv`` (also
   lock-free, one unique file per run).  After writing its own staging file,
   the process does a best-effort merge of all staging files into
   ``index.csv``.

This avoids ``fcntl.flock``, which returns POSIX error 524 on Lustre/GPFS
filesystems such as those used on NERSC Perlmutter.

:func:`build_index` walks a campaign results directory and assembles a
:class:`pandas.DataFrame` from all ``run.json`` files.  Call it via
``python -m gigalens_research.simtests aggregate`` to refresh ``index.csv``
after a partial or complete run.

Column layout
-------------
``campaign``, ``system_id``, ``sweep_name``, ``ctx_hash``, ``status``,
``max_rhat``, ``min_ess``, ``nan_rate``, ``wall_time_s``, ``peak_gpu_bytes``,
one column per sweep-point key (e.g. ``n_max``),
one column per mass z-score (e.g. ``theta_E_z``, ``gamma_z``, …),
``zscores_json`` (full JSON blob for all_zscores),
``num_pix``, ``supersample``, ``background_rms``, ``exp_time``.
"""
from __future__ import annotations

import csv
import json
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-run JSON record
# ---------------------------------------------------------------------------


def write_run_json(out_dir: str, record: Dict[str, Any]) -> None:
    """Write ``<out_dir>/run.json`` with the run record."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, default=_json_safe)
    os.replace(tmp, path)


def load_run_json(out_dir: str) -> Optional[Dict[str, Any]]:
    """Load ``<out_dir>/run.json``, returning None if not present."""
    path = os.path.join(out_dir, "run.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        warnings.warn(f"[index] could not read {path}: {exc}", stacklevel=2)
        return None


# ---------------------------------------------------------------------------
# Campaign index CSV
# ---------------------------------------------------------------------------


def append_to_index(index_path: str, row: Dict[str, Any]) -> None:
    """Append one row to ``index_path``.

    Strategy: write the row to a per-run staging file in
    ``<index_dir>/index_rows/<system_id>_<sweep_name>.csv`` (no locking
    needed — each run writes a unique file), then attempt to merge all staging
    files into ``index_path``.  This avoids ``fcntl.flock`` which is
    unsupported on Lustre/GPFS (Perlmutter NERSC) and returns error 524.

    :func:`build_index` always walks ``run.json`` files and is the canonical
    way to refresh the index; this function provides a best-effort real-time
    update.
    """
    flat = _flatten_row(row)
    index_dir = os.path.dirname(os.path.abspath(index_path))
    rows_dir = os.path.join(index_dir, "index_rows")
    os.makedirs(rows_dir, exist_ok=True)

    # Write this run's row to its own staging file (lock-free, unique name).
    sid = str(flat.get("system_id", "unknown")).replace("/", "_")
    sweep = str(flat.get("sweep_name", "default")).replace("/", "_")
    row_file = os.path.join(rows_dir, f"{sid}_{sweep}.csv")
    tmp = row_file + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerow(flat)
    os.replace(tmp, row_file)

    # Best-effort: merge all staging rows into the main index.csv.
    try:
        _merge_index_rows(index_path, rows_dir)
    except Exception:
        pass  # aggregate step will rebuild properly


def _merge_index_rows(index_path: str, rows_dir: str) -> None:
    """Merge all per-run staging CSVs in ``rows_dir`` into ``index_path``.

    Reads each ``*.csv`` in ``rows_dir``, deduplicates by (system_id,
    sweep_name), and rewrites ``index_path``.  No file locking is required
    because this is a pure read-then-write operation and is called only
    best-effort (races are benign — the next aggregate call will clean up).
    """
    import glob

    row_files = sorted(glob.glob(os.path.join(rows_dir, "*.csv")))
    if not row_files:
        return

    all_rows: Dict[str, Dict[str, Any]] = {}

    # Load existing index rows first (so we preserve any manual edits).
    if os.path.exists(index_path):
        try:
            with open(index_path, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    key = f"{r.get('system_id','')}__{r.get('sweep_name','')}"
                    all_rows[key] = dict(r)
        except Exception:
            pass

    # Overlay staging rows (newer data wins).
    for rf in row_files:
        try:
            with open(rf, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    key = f"{r.get('system_id','')}__{r.get('sweep_name','')}"
                    all_rows[key] = dict(r)
        except Exception:
            pass

    if not all_rows:
        return

    all_keys: List[str] = sorted({k for row in all_rows.values() for k in row})
    tmp = index_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows.values())
    os.replace(tmp, index_path)


def build_index(runs_dir: str, campaign_name: str) -> "Any":
    """Walk ``runs_dir`` and assemble a DataFrame from run.json files.

    Only rows for ``campaign_name`` are included.  Returns a
    ``pandas.DataFrame`` (or an empty frame if pandas is unavailable).

    ``runs_dir`` is typically ``<base_dir>/runs/``.
    """
    try:
        import pandas as pd
    except ImportError:
        warnings.warn("[index.build_index] pandas not available; returning empty result.")
        return {}

    if not os.path.isdir(runs_dir):
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(runs_dir):
        if "run.json" in files:
            rec = load_run_json(root)
            if rec is not None and rec.get("campaign") == campaign_name:
                records.append(_flatten_row(rec))

    if not records:
        return pd.DataFrame()

    all_keys = sorted({k for r in records for k in r})
    return pd.DataFrame(records, columns=all_keys).sort_values(
        ["system_id", "sweep_name"], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_run_record(
    *,
    campaign: str,
    system_id: str,
    sweep_name: str,
    sweep_point: Dict[str, Any],
    ctx_hash: str,
    status: str,
    metrics: Dict[str, Any],
    wall_time_s: float,
    peak_gpu_bytes: int,
    system_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a complete run record dict suitable for both JSON and CSV."""
    return {
        "campaign": campaign,
        "system_id": system_id,
        "sweep_name": sweep_name,
        "ctx_hash": ctx_hash,
        "status": status,
        "wall_time_s": round(wall_time_s, 2),
        "peak_gpu_bytes": int(peak_gpu_bytes),
        "sweep": sweep_point,
        "metrics": metrics,
        "system_meta": system_meta,
    }


def _flatten_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a run record dict to a single-level dict for CSV."""
    flat: Dict[str, Any] = {}
    for k in ("campaign", "system_id", "sweep_name", "ctx_hash", "status",
               "wall_time_s", "peak_gpu_bytes", "pipeline", "seed",
               "trunk_wall_time_s"):
        flat[k] = record.get(k, "")

    for k, v in (record.get("sweep") or {}).items():
        flat[f"sweep_{k}"] = v

    metrics = record.get("metrics") or {}
    for metric_name, value in metrics.items():
        if isinstance(value, dict):
            # Bare ``<param>_z`` columns are reserved for z-score metrics.
            # Other dict metrics (percent_error, sbc_ranks, solver health, ...)
            # get namespaced ``<metric>.<key>`` columns — previously they ALL
            # wrote ``<param>_z``, so whichever dict metric ran last silently
            # overwrote the z-scores in the index (and in every figure built
            # from it).
            if metric_name.endswith("zscores"):
                for param, score in value.items():
                    flat[f"{param}_z"] = _safe_float(score)
            else:
                for param, score in value.items():
                    if isinstance(score, (int, float, np.integer, np.floating)):
                        flat[f"{metric_name}.{param}"] = _safe_float(score)
            flat[f"{metric_name}_json"] = json.dumps(
                {k: _safe_float(v) for k, v in value.items()}, default=str
            )
        elif isinstance(value, (int, float)):
            flat[metric_name] = _safe_float(value)
        else:
            flat[metric_name] = str(value)

    for k, v in (record.get("system_meta") or {}).items():
        flat[f"sys_{k}"] = v

    return flat


def _safe_float(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    try:
        return round(float(v), 6)
    except Exception:
        return str(v)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return repr(obj)
