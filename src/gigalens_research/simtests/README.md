# `gigalens_research.simtests`

A model-agnostic framework for large simulated-system inference tests.

## Concept

A **campaign** specifies a test from generation to diagnostics:

```
generate → fit (MAP / SVI / HMC / NUTS / MCLMC / MAMS) → aggregate
```

The framework is agnostic to model components by depending only on a small
**registry of Python builder functions** (for truth models, inference models,
pipelines, and metrics) referenced by name from a lightweight **campaign YAML**.

## Key design choices

| Choice | Rationale |
|---|---|
| **Truth and inference are independent** | Simulation prior/model is separate from the inference prior/model; the science lives in the mismatch. Per-system truth (e.g. a different Vela source per system) is supported via the generator abstraction. |
| **Content-addressed caching** | `InferenceContext.hash()` + `Pipeline.run(resume=True)` cache each stage on disk; changing any input (model, prior, data, seed) automatically invalidates downstream stages. |
| **One system per task** | Default `systems_per_task=1` bounds GPU memory. Never batch multiple systems into one `LensSimulator`. |
| **Memory-chunked generation** | `gen_chunk` controls how many systems are simulated per JAX batch during dataset generation. |
| **Honest diagnostics** | Reuses `z_scores` (scores only shared parameters), `rhat`, and `ess`. No statistical shortcuts. |
| **Pipeline-variant sweeps** | A sweep point may carry the reserved `pipeline` key to select a different registered pipeline builder per run (e.g. a sampler-comparison campaign). |
| **Per-system trunk sharing** | The leading MAP/SVI stages of every pipeline variant run ONCE per system into `runs/<sid>/trunk/<digest>/`; each variant's tail (bridge + sampler) is seeded with the identical trunk artifacts. Sharding is by system, so trunk access is race-free. |
| **Seed policy** | Top-level `seed_mode: per_system` derives a stable seed per system from `(seed, system_id)` (default `campaign` keeps the legacy one-seed-for-everything behavior). All variants of a system always share its seed. |

## Directory layout

```
<output_dir>/
  dataset/
    manifest.json          (generator, seed, n_systems, system_ids)
    systems/
      sys_000/
        observed_image.npy
        truth_x.pkl
        psf.npy            (optional)
        meta.json
      sys_001/ ...
  runs/
    sys_000/
      trunk/
        <digest>/          (shared MAP/SVI: map/ svi/ + pipeline.json)
      default/             (or n_max10/, map_svi_mclmc/, ...)
        pipeline.json      (stage manifest with hashes)
        map/ svi/ hmc/     (per-stage arrays.npz + manifest.json;
                            trunk-shared stages live under trunk/ instead)
        run.json           (metrics + timings)
    sys_001/ ...
  index.csv                (one row per run; refreshed by aggregate)
  aggregate/
    convergence.png
    zscores_scatter.png
    abs_zscore_vs_sweep.png
    percent_error.png
```

## CLI

```bash
# 1. Generate or adapt the dataset
python -m gigalens_research.simtests generate campaign.yaml

# 2. Run inference (single process)
python -m gigalens_research.simtests run campaign.yaml

# 3. Run inference (Slurm array: task i of N)
python -m gigalens_research.simtests run campaign.yaml --shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT

# 4. Aggregate figures
python -m gigalens_research.simtests aggregate campaign.yaml

# 5. Quick progress check
python -m gigalens_research.simtests status campaign.yaml
```

## Reference campaigns

| Campaign | Config | Purpose |
|---|---|---|
| `hundred_sersic_v1` | `experiments/hundred_systems_GL2/campaign.yaml` | Convergence test: 100 EPL+Shear+Sérsic systems, MAP→SVI→HMC |
| `shapelets_systematics_v1` | `experiments/shapelets_systematics/campaign.yaml` | Systematics: Vela sources, shapelet inference, sweep over `n_max` |
| `sampler_benchmark_v1` | `experiments/sampler_benchmark/campaign.yaml` | Sampler benchmark: 7 pipeline variants (`map_svi_{hmc,nuts,mclmc,mams}` + `map_{nuts,mclmc,mams}`) per system, all sharing one MAP/SVI trunk |

### Benchmark builder family

`pipelines.py` registers `map_svi_<sampler>` and `map_<sampler>` for each of
`hmc`, `nuts`, `mclmc`, `mams` (see `BENCHMARK_SAMPLERS`; LAPS joins when its
implementation is validated). `map_<sampler>` replaces SVI with a diagonal-qz
bridge around the MAP optimum (`bridge_qz_scale`, default 1e-2). Sampler knobs
are namespaced (`hmc_*`, `nuts_*`, `mclmc_*`, `mams_*`); chain counts default
to 16; `<sampler>_num_burnin` is **required** — burn-in budgets are never
defaulted.

## Extending the framework

### Add a new generator

```python
from gigalens_research.simtests import register_generator

@register_generator("my_generator")
def my_generator(spec, dataset_dir, seed):
    # Generate systems and call system.save(dataset_dir) for each,
    # then call write_manifest(...).
    ...
```

### Add a new inference builder

```python
from gigalens_research.simtests import register_inference_builder

@register_inference_builder("my_model")
def build_my_model(system, **kwargs):
    # Return a scene ProbModel
    ...
```

### Add a new metric

```python
from gigalens_research.simtests import register_metric

@register_metric("my_metric")
def my_metric(posterior, system):
    # Return a float or dict of floats
    ...
```

### Add a population-level (campaign) metric

```python
from gigalens_research.simtests.aggregate import register_campaign_metric

@register_campaign_metric("my_population_metric")
def my_population_metric(index_df, campaign_spec, agg_dir):
    # index_df is a pandas DataFrame with one row per completed run
    ...
```

## Slurm array submission

A template is provided at `src/gigalens_research/simtests/slurm/campaign_array.slurm`.

```bash
N=100  # total number of SYSTEMS (not systems × sweep points)
sbatch --array=0-$((N-1))%32 \
    --export=ALL,CAMPAIGN_YAML=/path/to/campaign.yaml \
    src/gigalens_research/simtests/slurm/campaign_array.slurm
```

Sharding is over systems: each array task owns a slice of the systems and
runs all sweep points of each owned system sequentially (so pipeline variants
share the per-system trunk MAP/SVI without races).
The Slurm template uses the canonical JAX-2026 shifter container + sidecar
`PYTHONPATH` (see `.cursor/rules/gigalens-runtime-environment.mdc`).
