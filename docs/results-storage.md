# Results storage: where runs are written, and how to keep them

**TL;DR** — Sampling output goes to **scratch** (`$PSCRATCH/gigalens` on NERSC),
not to `$HOME`. Scratch is fast and huge but **auto-purged**; copy anything worth
keeping to **CFS** with `scripts/archive_results_to_cfs.py`.

## Why

`$HOME` at NERSC is a 40 GiB, backed-up filesystem — the wrong place for
multi-GB sampling arrays, and it fills up fast (it hit 98.8% in July 2026). The
storage tiers we use:

| Tier | Path | Size | Backed up? | Purged? | Use for |
|---|---|---|---|---|---|
| home | `$HOME` | 40 GiB | yes | no | code only |
| scratch | `$PSCRATCH/gigalens` | ~20 TiB | no | **~180 days no-access** | active/scratch results |
| CFS | `$CFS/<project>/$USER/gigalens` | project quota | yes | no | results worth keeping |

## The single knob

All path resolution goes through `gigalens_research.paths`:

- `results_root()` — base for writing results. Resolves in order:
  1. `$GIGALENS_RESULTS_ROOT` (explicit override),
  2. `$PSCRATCH/gigalens` (or `$SCRATCH/gigalens`),
  3. `~/GIGALens-Code` (laptop fallback).
- `resolve_out_dir(path)` — `None` stays `None`; an **absolute** path is used
  verbatim; a **relative** path is joined onto `results_root()`.
- `cfs_archive_root()` — durable archive base: `$GIGALENS_ARCHIVE_ROOT`, else
  `$CFS/$GIGALENS_CFS_PROJECT/$USER/gigalens`, else `~/gigalens_archive`.

On NERSC, `$PSCRATCH` is always set, so **new code needs no configuration** —
results land on scratch by default. Check what will be used:

```bash
python -m gigalens_research.paths
```

## For new code

Pass a **relative** `out_dir` and let it resolve:

```python
pipeline.run(out_dir="results/my_system/sweep_a")   # -> $PSCRATCH/gigalens/results/my_system/sweep_a
```

`Pipeline.run` (`inference_utils/pipeline.py`) and the simtests campaign default
(`simtests/config.py`) already route through `resolve_out_dir`. Absolute paths
and explicit `output_dir:` YAML keys keep working unchanged.

## Existing `$HOME`-anchored scripts

Scripts that build `~/GIGALens-Code/results/...` (e.g. `experiments/sample_cosmology/`)
keep working: those trees were physically relocated to `$PSCRATCH/gigalens` and
replaced with symlinks (`~/GIGALens-Code/results -> $PSCRATCH/gigalens/results`,
same for `simtests_results`). Writes follow the symlink to scratch with no code
change. The symlinks are ignored via `.git/info/exclude` (the `.gitignore` rules
`results/`/`simtests_results/` are directory-only and don't match a symlink).

## Keeping results (archive to CFS)

Scratch is purged, so archive anything you care about:

```bash
export GIGALENS_CFS_PROJECT=m5362          # -> $CFS/m5362/$USER/gigalens
python scripts/archive_results_to_cfs.py results/sample_cosmology/dspl_cosmology_newapi            # preview
python scripts/archive_results_to_cfs.py results/sample_cosmology/dspl_cosmology_newapi --execute  # copy
```

It is `rsync -a` under the hood, so it is incremental and safe to re-run.
