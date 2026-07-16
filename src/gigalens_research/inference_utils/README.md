# `inference_utils` — GIGALens Inference Pipeline

A modular, cache-aware inference pipeline for strong gravitational lens
modeling. Stages (MAP, SVI, HMC, MCLMC, …) are composable, each
automatically cached to disk so re-runs only redo work that actually
changed. A companion Posterior view layer provides a uniform interface for
diagnostics and plotting across all inference algorithms.

---

## Contents

- [Quick start: the standard MAP → SVI → HMC run](#quick-start)
- [How the pipeline works](#how-the-pipeline-works)
- [InferenceContext](#inferencecontext)
- [Pipeline and stages reference](#stages-reference)
  - [MAPStage](#mapstage)
  - [SVIStage](#svistage)
  - [HMCStage](#hmcstage)
  - [MCLMCStage](#mclmcstage)
  - [BridgeStage](#bridgestage)
- [Posterior view layer](#posterior-view-layer)
  - [PointEstimate](#pointestimate)
  - [SurrogatePosterior](#surrogateposterior)
  - [SamplerPosterior](#sampleRposterior)
- [Non-standard recipes](#non-standard-recipes)
  - [Start SVI from truth (skip MAP)](#start-svi-from-truth-skip-map)
  - [MAP → MCLMC with a diagonal qz bridge](#map--mclmc-with-a-diagonal-qz-bridge)
  - [Warm-start HMC from a previous run's SVI result](#warm-start-hmc-from-a-previous-runs-svi-result)
  - [Two independent MAPs, pick the better one](#two-independent-maps-pick-the-better-one)
  - [Load a posterior from disk without re-running](#load-a-posterior-from-disk-without-re-running)
  - [Writing a custom stage](#writing-a-custom-stage)
- [Caching and invalidation](#caching-and-invalidation)
- [Truth-aware diagnostics](#truth-aware-diagnostics)
- [Debug diagnostics](#debug-diagnostics)

---

## Quick start

```python
from gigalens.jax.inference import ModellingSequence
from gigalens_research.inference_utils import (
    InferenceContext, Pipeline, MAPStage, SVIStage, HMCStage,
)

# phys_model, prob_model, sim_config come from your usual setup;
# see e.g. experiments/hundred_systems_GL2/setup.py
model_seq = ModellingSequence(phys_model, prob_model, sim_config)
ctx = InferenceContext.from_modelling_sequence(model_seq)

pipeline = Pipeline(ctx, seed=42)
pipeline.add(MAPStage(num_steps=500, n_samples=1000))
pipeline.add(SVIStage(num_steps=3000, n_vi=500))
pipeline.add(HMCStage(n_hmc=64, num_burnin_steps=500, num_results=1000))

artifacts = pipeline.run(out_dir="results/system_07", resume=True)

post = pipeline.posterior()   # picks HMC (richest terminal stage)
print(post.rhat)              # per-parameter R-hat, shape (n_params,)
```

On the second call with the same `out_dir`, stages whose inputs haven't
changed are loaded from disk in milliseconds. Change the prior, add a
lens-light component, or bump `num_results`, and only the affected stages
(and everything downstream) are recomputed.

---

## How the pipeline works

```
InferenceContext (model, prior, observed image, noise)
    │
    ▼
MAPStage.run()  →  {z_best, lp_hist, chisq_hist}  →  input_hash → disk
    │
    ▼
SVIStage.run()  →  {qz, svi_loss_hist}             →  input_hash → disk
    │
    ▼
HMCStage.run()  →  {samples_z}                     →  input_hash → disk
```

Each stage's input hash covers: model context + stage config + seed +
upstream artifact hashes. A stage is loaded from
`<out_dir>/<stage_name>/arrays.npz` if its hash matches; otherwise it runs
and saves. Downstream stages are automatically invalidated when any upstream
hash changes.

On-disk layout:
```
results/system_07/
  pipeline.json            # top-level run log
  map/
    manifest.json          # input_hash, config, timing
    arrays.npz             # z_best, lp_hist, chisq_hist
  svi/
    manifest.json
    arrays.npz             # qz_loc, qz_scale_tril, svi_loss_hist
  hmc/
    manifest.json
    arrays.npz             # samples_z  shape: (n_chains, n_steps, n_params)
```

---

## InferenceContext

```python
# Standard — wraps a ModellingSequence:
ctx = InferenceContext.from_modelling_sequence(model_seq)

# Manual — useful when you want to avoid importing ModellingSequence
# on the login node, or when model_seq is not yet available:
ctx = InferenceContext(
    phys_model=phys_model,
    prob_model=prob_model,
    sim_config=sim_config,
    model_seq=model_seq,   # can be None for Posterior-only use
)
```

`ctx.hash()` is the content-addressed key for the whole system, covering
the profile classes and their settings, the prior, the observed image, and
the noise convention (`background_rms`/`exp_time` for `ForwardProbModel`;
`err_map` for `BackwardProbModel`). Changing any of these invalidates all
cached stages.

---

## Stages reference

### MAPStage

Multi-start MAP optimization. Runs `n_samples` random prior draws for
`num_steps` gradient steps each and saves the globally best parameter
vector.

```python
MAPStage(
    num_steps=500,          # gradient steps per random start
    n_samples=1000,         # number of random starts
    optimizer_factory=None, # callable → optax optimizer; default: AdaBelief 1e-2
    optimizer_id=None,      # string hash key; set this when passing a custom
                            # optimizer_factory so the cache key is meaningful
    pbar_interval=5,        # print progress every N steps (0 to silence)
    name="map",             # directory name; override when using multiple MAPs
    seed=None,              # stage-specific seed (defaults to pipeline seed)
)
```

**Produces:** `z_best` (best z-vector), `lp_hist`, `chisq_hist`.  
**Posterior view:** `PointEstimate`.

---

### SVIStage

Gaussian mean-field variational inference initialized at `z_best`. Fits a
full-covariance Gaussian `q(z) = N(μ, LL^T)` by minimizing the ELBO over
`n_vi` Monte Carlo samples per gradient step.

```python
SVIStage(
    num_steps=3000,         # ELBO gradient steps
    n_vi=500,               # MC samples per gradient step
    init_scales=1e-3,       # initial diagonal scale of q(z)
    optimizer_factory=None, # default: AdaBelief 1e-4
    optimizer_id="adabelief_1e-4_b1_0.95_b2_0.99",
    pbar_interval=5,
    name="svi", seed=None,
)
```

**Requires:** `z_best`.  
**Produces:** `qz` (`tfd.MultivariateNormalTriL`), `svi_loss_hist`.  
**Posterior view:** `SurrogatePosterior`.

---

### HMCStage

Preconditioned HMC initialized from `n_hmc` draws from `qz`. The `qz`
covariance is used to precondition the leapfrog dynamics.

```python
HMCStage(
    n_hmc=64,               # number of parallel chains
    num_burnin_steps=500,   # adaptation + burn-in steps (discarded)
    num_results=1000,       # post-burnin samples kept per chain
    init_eps=0.3,           # initial leapfrog step size
    init_l=3,               # initial number of leapfrog steps
    max_leapfrog_steps=30,
    pbar_interval=0,
    name="hmc", seed=None,
)
```

**Requires:** `qz`.  
**Produces:** `samples_z` of shape `(n_hmc, num_results, n_params)`.  
**Posterior view:** `SamplerPosterior`.

---

### MCLMCStage

Microcanonical Langevin Monte Carlo (via `blackjax`). Generally more
efficient than HMC for high-dimensional posteriors; self-tunes step size
and trajectory length during burn-in.

```python
MCLMCStage(
    n_chains=16,
    num_burnin_steps=1000,
    num_results=2000,
    desired_energy_variance=5e-4,  # target for self-tuning
    init_L=None,            # initial trajectory length (auto-tuned if None)
    init_step_size=None,    # initial step size (auto-tuned if None)
    frac_tune1=0.2,         # burn-in fraction for stage-1 tuning
    frac_tune2=0.6,         # burn-in fraction for stage-2 tuning
    frac_tune3=0.2,         # burn-in fraction for stage-3 tuning
    progress_bar=False,
    debug=False,            # capture the tuning history for diagnostics
    name="mclmc", seed=None,
)
```

**Requires:** `qz`.  
**Produces:** `samples_z` of shape `(n_chains, num_results, n_params)`.  
**Posterior view:** `SamplerPosterior` (identical to HMC).

Run with `debug=True` to capture the per-step tuning history (step size, `L`,
inverse-mass-matrix spectrum, energy-error ratio `xi`, NaN mask) for the
[debug diagnostics](#debug-diagnostics) plot. `debug` is part of the cache
key, so toggling it re-runs the stage (the captured arrays are extra output).

---

### BridgeStage

A pure-function adapter for non-standard stitching. Transforms one set of
artifacts into another without running an optimizer or sampler. It
re-runs every pipeline invocation (assumed cheap and deterministic) but
participates fully in input-hash propagation, so downstream stages are
cached and invalidated correctly.

```python
BridgeStage(
    name="my_bridge",   # required; no default
    version="v1",       # bump this string whenever the fn logic changes;
                        # this is what invalidates downstream stages
    requires=("z_best",),
    produces=("qz",),
    fn=lambda z_best: ...,
    # fn receives requires as keyword args and should return either:
    #   - a dict {produces[0]: value, produces[1]: value, ...}
    #   - the value directly when len(produces) == 1
)
```

See the [non-standard recipes](#non-standard-recipes) section for concrete
examples.

---

## Posterior view layer

After running, get a posterior view with:

```python
post = pipeline.posterior()          # auto-picks richest terminal stage
post = pipeline.posterior("svi")     # request a specific stage by name
post = pipeline.posterior("hmc")

# Or load from a previous run without re-running anything:
from gigalens_research.inference_utils import posterior_from_disk
post = posterior_from_disk("results/system_07", "hmc", ctx)
```

All posterior types share a common interface:

| Property / method | Description |
|---|---|
| `post.ctx` | The `InferenceContext` |
| `post.scene` | The scene `LensModel` — the authority on what the parameters are |
| `post.n_params` | Number of free parameters |
| `post.z_to_x(z)` | Unconstrained → physical params (the scene's flat `{unique_key: array}` dict) |
| `post.median_x` | Physical params at the posterior median |
| `post.median_z` | Median in unconstrained z-space |
| `post.mean_z` | Mean in unconstrained z-space |
| `post.simulate(point="median")` | Render predicted PSF-convolved image |
| `post.source_plane(point, grid_pix=400, fov_arcsec=None)` | Intrinsic source surface brightness (no PSF, no lensing) |
| `post.err_map_at(predicted)` | Per-pixel noise σ (auto-selects Forward or Backward noise model) |
| `post.normalized_residual(observed, point)` | `(obs − pred) / σ` |
| `post.is_backward` | True for `BackwardProbModel` (lstsq amplitudes) |

Physical params come back as the scene's own flat dict, keyed by the parameter's
**scene path** — `planes/0/mass/0/theta_E`, `planes/1/geometry/redshift`,
`cosmo/H0`. Nothing here reconstructs parameter names: `post.scene` is the model
itself, and anything that needs the structure (which plane, which component,
which kind) reads it from there via `gigalens_research.param_index.param_sites`:

```python
from gigalens_research.param_index import param_sites, select_sites, sites_to_matrix

sites = param_sites(post)                        # one record per free parameter
mass = select_sites(sites, kind="mass", plane=0)
cols = sites_to_matrix(mass, post.flat_x)        # (n_samples, len(mass))
```

A `shared()` parameter is one free parameter feeding several sites, so it is one
record (and one column) here — matching `scene.num_free_params` — with every site
it feeds in `site.paths`.

### PointEstimate

From `MAPStage`. Single best-fit point; no posterior uncertainty.

```python
post.z_best       # best z-vector, shape (n_params,)
post.x            # physical params at z_best
post.lp_hist      # log-prob history over optimization steps
post.chisq_hist   # χ² history
```

### SurrogatePosterior

From `SVIStage`. A fitted full-covariance Gaussian `q(z)`.

```python
post.qz           # tfd.MultivariateNormalTriL
post.covariance   # (n_params, n_params) covariance matrix in z-space
post.loss_hist    # -ELBO history

# Sample a SamplerPosterior from the surrogate for plotting:
samples_post = post.draw(n=5000)   # returns SamplerPosterior

# Marginal quantiles (e.g. ±1σ):
lo = post.z_to_x(post.quantiles_z(0.159))
hi = post.z_to_x(post.quantiles_z(0.841))
```

### SamplerPosterior

From `HMCStage` or `MCLMCStage`. Holds the full chain; subsampled to
~5 000 samples by default for memory-intensive operations like corner plots.

```python
post.samples_z        # full chain: (n_chains, n_steps, n_params)
post.flat_z           # chain-flattened + subsampled: (N, n_params)
post.flat_x           # same in physical space: the scene's flat
                      # {unique_key: (N,) array} dict

# Convergence diagnostics. These are per *sampler (z) column*, so they are
# arrays, not path-keyed dicts; prob_model.z_param_names is the column→name map.
post.rhat             # rank-normalized split-R-hat, shape (n_params,)
post.ess              # rank-normalized bulk-ESS, shape (n_params,)
post.running_rhat()   # (schedule, rhat) — rhat is (n_windows, n_params)
post.running_ess()    # (schedule, ess)

# Marginal quantiles:
lo_z = post.quantiles_z(0.159)   # shape (n_params,)
hi_z = post.quantiles_z(0.841)
```

---

## Non-standard recipes

### Start SVI from truth (skip MAP)

Use `seed_artifacts` to inject a pre-computed artifact before stage 1.
The pipeline skips MAP and runs SVI → HMC directly from the truth point.

```python
import numpy as np

# truth_x is the scene-nested truth point ({"planes": {...}, "cosmo": {...}}),
# complete over the model's free parameters. The scene maps it to flat z:
z_truth = np.asarray(ctx.model_seq.scene_model.unconstrained(truth_x))

pipeline = Pipeline(ctx, seed=0)
pipeline.add(SVIStage(num_steps=3000, n_vi=500))
pipeline.add(HMCStage(n_hmc=64, num_results=1000))

artifacts = pipeline.run(
    out_dir="results/truth_init",
    seed_artifacts={"z_best": z_truth},
    # Give the injected artifact a stable string ID so the downstream
    # cache key doesn't depend on the exact floating-point values:
    seed_artifact_ids={"z_best": "truth_v1"},
)
```

If you re-run later with the same `z_truth` and the same
`seed_artifact_ids`, SVI and HMC load from cache. Change `"truth_v1"` →
`"truth_v2"` to force both to recompute.

---

### MAP → MCLMC with a diagonal qz bridge

Go straight from MAP to MCLMC without SVI, building a simple diagonal
`qz` centered at the MAP optimum.

```python
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

def make_diag_qz(z_best):
    return tfd.MultivariateNormalDiag(
        loc=jnp.asarray(z_best),
        scale_diag=jnp.full(z_best.shape[-1], 5e-2),
    )

pipeline = Pipeline(ctx, seed=0)
pipeline.add(MAPStage(num_steps=500, n_samples=1000))
pipeline.add(BridgeStage(
    name="diag_qz",
    version="v1",           # bump to 'v2' if you change the scale or logic
    requires=("z_best",),
    produces=("qz",),
    fn=make_diag_qz,
))
pipeline.add(MCLMCStage(n_chains=32, num_burnin_steps=1000, num_results=2000))
artifacts = pipeline.run("results/map_mclmc")
```

---

### Warm-start HMC from a previous run's SVI result

Re-use a `qz` from a previously saved run (e.g. the same system modeled
with a slightly different mass profile) to initialize HMC without
repeating the SVI optimization.

```python
from gigalens_research.inference_utils import posterior_from_disk

old = posterior_from_disk("results/system_06", "svi", ctx)

pipeline = Pipeline(ctx, seed=0)
pipeline.add(HMCStage(n_hmc=64, num_results=1000))
artifacts = pipeline.run(
    "results/system_07_warm",
    seed_artifacts={"qz": old.qz},
    seed_artifact_ids={"qz": "svi_from_system_06"},
)
```

---

### Two independent MAPs, pick the better one

Run two MAP pipelines to disk, then inject the better `z_best` as a
seed artifact into a joint SVI → HMC pipeline.

```python
import numpy as np

for run_id, seed in [("map_a", 7), ("map_b", 99)]:
    p = Pipeline(ctx, seed=seed)
    p.add(MAPStage(num_steps=500, n_samples=1000))
    p.run(f"results/{run_id}", resume=True)

from gigalens_research.inference_utils import posterior_from_disk
map_a = posterior_from_disk("results/map_a", "map", ctx)
map_b = posterior_from_disk("results/map_b", "map", ctx)

best_z = (map_a.z_best
          if map_a.lp_hist.max() >= map_b.lp_hist.max()
          else map_b.z_best)

pipeline = Pipeline(ctx, seed=0)
pipeline.add(SVIStage(num_steps=3000, n_vi=500))
pipeline.add(HMCStage(n_hmc=64, num_results=1000))
artifacts = pipeline.run(
    "results/best_start",
    seed_artifacts={"z_best": best_z},
    seed_artifact_ids={"z_best": "best_of_two_v1"},
)
```

---

### Load a posterior from disk without re-running

```python
from gigalens_research.inference_utils import posterior_from_disk, InferenceContext
from gigalens.jax.inference import ModellingSequence

model_seq = ModellingSequence(phys_model, prob_model, sim_config)
ctx = InferenceContext.from_modelling_sequence(model_seq)

post = posterior_from_disk("results/system_07", "hmc", ctx)
print(post.rhat)

# Load multiple stages for a PipelineReport without re-running:
from gigalens_research.plotting import PipelineReport
pr = PipelineReport.from_disk("results/system_07", ctx)
fig = pr.compound_corner()
```

---

### Writing a custom stage

Subclass `InferenceStage`, implement `run`, and decorate with
`@register_stage` so `posterior_from_disk` can find it by name.

```python
from gigalens_research.inference_utils import (
    InferenceStage, StageResult, register_stage,
)
from typing import ClassVar, Tuple
import numpy as np, time

@register_stage
class MyNUTSStage(InferenceStage):
    name: ClassVar[str] = "nuts"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ("qz",)
    produces: ClassVar[Tuple[str, ...]] = ("samples_z",)

    def __init__(self, *, n_chains: int = 32, num_steps: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.n_chains = int(n_chains)
        self.num_steps = int(num_steps)

    def run(self, ctx, artifacts, seed):
        t0 = time.perf_counter()
        # ... your sampler code here, using ctx.model_seq and artifacts["qz"] ...
        samples_np = np.zeros((self.n_chains, self.num_steps,
                               ctx.model_seq.n_params))
        return StageResult(
            arrays={"samples_z": samples_np},
            metadata={"wall_time_s": time.perf_counter() - t0},
        )

    @classmethod
    def to_posterior(cls, arrays, ctx):
        from gigalens_research.inference_utils import SamplerPosterior
        return SamplerPosterior(ctx, samples_z=arrays["samples_z"])
```

---

## Caching and invalidation

The pipeline hashes:
- The `InferenceContext` (model profiles + settings, prior, observed image,
  noise parameters).
- The stage class name and `schema_version`.
- All `config_hash_data()` entries (stage constructor kwargs).
- The pipeline-level and per-stage seed.
- The output hashes of all upstream artifacts the stage consumes.

**What triggers a re-run:**
- Changing the prior, any profile type or parameter (e.g. `EPL(niter=50)`
  → `EPL(niter=100)`), the observed image, `background_rms`, `exp_time`,
  or `err_map`.
- Changing any stage kwarg (`num_steps`, `n_samples`, `n_hmc`, etc.).
- Bumping a `BridgeStage`'s `version` string.
- Changing the seed.

**What does *not* trigger a re-run:**
- Changing code inside a `BridgeStage`'s `fn` without bumping `version`.
  Always bump `version` when the logic changes.
- Adding a new stage downstream of unchanged cached stages.

**`resume` modes:**

| Mode | Behaviour |
|---|---|
| `resume=True` (default) | Load from cache on hash match; re-run and save on mismatch. Stale dirs are renamed with a timestamp suffix. |
| `resume="strict"` | Raise `PipelineMismatchError` on any mismatch. Useful in batch scripts where silent re-runs are undesirable. |
| `resume=False` / `force=True` | Always re-run every stage; ignore any cached results. |

---

## Truth-aware diagnostics

When you have access to the ground truth (simulated systems), four
data-layer functions are provided. The matching plots live in
`gigalens_research.plotting`; see the plotting README or `PosteriorReport`
for the integrated report-level interface.

The truth is given the way the model names things — either **scene-nested**,

```python
truth_x = {
    "planes": {0: {"mass": {0: {"theta_E": 1.5, "gamma": 2.0, ...}}},
               1: {"light": {0: {"R_sersic": 0.3, ...}}}},
    "cosmo": {"H0": 70.0, "Om0": 0.3},
}
```

or **path-keyed** (`{"planes/0/mass/0/theta_E": 1.5, "cosmo/H0": 70.0}`). Both
locate a parameter by *where it lives*, so nothing is matched against a
reconstructed label. Either form may be partial; parameters it doesn't define are
skipped with a `UserWarning`.

```python
from gigalens_research.inference_utils import (
    z_scores,
    source_comparison,
    truth_source_from_light_model,
    filter_keys_by_kind,
)

# Per-parameter z-scores: (truth − median) / σ with asymmetric ±1σ quantiles.
# Returns a dict keyed by scene path. Parameters the truth is silent on (e.g. an
# ImageBasedLight truth fit with a shapelet source) are skipped with a
# UserWarning; the shared mass/center parameters are still scored.
zs = z_scores(post, truth_x)
# {"planes/0/mass/0/theta_E": 0.42, "cosmo/H0": -1.7, ...}

# Subset to one parameter class — "cosmology", "geometry", "mass", "light", or
# "all"/None. It classifies by scene path, so a new kind cannot leak into the
# mass panel the way the old prefix-by-negation rule let it.
mass_params = filter_keys_by_kind(list(zs.keys()), "mass")

# Source plane comparison — pre-rendered truth array:
truth, recovered, residual, extent = source_comparison(
    post, truth_source_image, extent=(-1.5, 1.5, -1.5, 1.5),
)

# Source plane comparison — continuous truth function (e.g. ImageBasedLight).
# Evaluated on the same grid as the recovered source, so no resampling.
from gigalens_research.simulations import load_vela_source
vela = load_vela_source("vela_sources/vela07_cam0_a0.500_f814w")

# The source's light params live at their own path in the truth — here plane 1,
# light component 0. Either wrap explicitly:
src_truth = truth_x["planes"][1]["light"][0]
truth_fn = lambda X, Y: vela.light.light(X, Y, **src_truth)

# Or use the helper (coerces params to static floats for JAX tracing):
truth_fn = truth_source_from_light_model(vela.light, src_truth)

truth, recovered, residual, extent = source_comparison(
    post, truth_fn, grid_pix=400, fov_arcsec=2.0,
)
```

---

## Debug diagnostics

Some stages can capture their *internal* run history for debugging failed
inference — distinct from posterior convergence diagnostics, which are about
the finished chain. Currently `MCLMCStage` supports it; the mechanism is
generic so other stages can opt in.

Enable it by constructing the stage with `debug=True`:

```python
pipeline.add(MCLMCStage(n_chains=32, num_burnin_steps=2000,
                        num_results=3000, debug=True))
pipeline.run(out_dir="run/")

# Pull the captured arrays (empty StageDiagnostics if debug was off):
diag = pipeline.diagnostics("mclmc")     # -> StageDiagnostics
diag.arrays.keys()    # step_size, L, xi, nonan, inverse_mass_matrix
diag.config           # tuning-stage boundaries needed by the plotter

# Or load them back later without an active pipeline:
from gigalens_research.inference_utils import diagnostics_from_disk
diag = diagnostics_from_disk("run/", "mclmc", ctx)
```

`StageDiagnostics` is pure data (arrays + plot config + ctx). The rendering
lives in `gigalens_research.plotting.plot_stage_diagnostics`, which dispatches
on the stage class — see the plotting README. Diagnostics are persisted to a
separate `diagnostics.npz` per stage so loading a posterior never pulls in the
(potentially large) debug arrays.

**Adding diagnostics to a new stage:**

1. In the stage's `run`, populate `StageResult.diagnostics` (only when a
   `debug` flag is set) and expose plot-relevant config via
   `diagnostics_config()`.
2. Register a plotter in `plotting/diagnostics.py` with
   `@register_diagnostic_plotter("YourStage")`.
