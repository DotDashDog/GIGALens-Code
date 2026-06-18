# Vela source-systematics suite

Three campaigns on the **same** Vela simulated systems, differing **only** in the
source light model, to measure how the source-model assumption biases the
recovered lens posteriors.

| Campaign | Source model | Builder | Sweep |
|---|---|---|---|
| `campaign_shapelets.yaml` | `ShapeletsFast` (circular) | `epl_shear_sersic_shapelets` | `n_max` |
| `campaign_elliptical_shapelets.yaml` | `EllipticalShapelets` (elliptical frame) | `epl_shear_sersic_elliptical_shapelets` | `n_max` |
| `campaign_sersic.yaml` | single `SersicEllipse` (smooth baseline) | `epl_shear_sersic_sersic_source` | none |

Everything else — dataset, lens & lens-light priors, bootstrap, MCLMC, metrics —
is identical across the three, so any difference in the results is attributable
to the source model. Content-addressed caching means the shared dataset is
generated once and reused by all three.

## How to run

```bash
cd ~/GIGALens-Code
# 1. generate the (shared) dataset — only needs to happen once
python -m gigalens_research.simtests generate experiments/vela_source_systematics/campaign_shapelets.yaml
# 2. test a single run before the full suite (see "Per-suite test" below)
# 3. run (single process, or --shard i/N for a Slurm array; see slurm/ template)
python -m gigalens_research.simtests run       experiments/vela_source_systematics/campaign_shapelets.yaml
python -m gigalens_research.simtests aggregate experiments/vela_source_systematics/campaign_shapelets.yaml
```

`N = n_systems × n_sweep_points`. With 12 vela_ids × 5 reps = 60 systems:
shapelets/elliptical = 60 × 6 n_max = **360 runs each**; sersic = **60 runs**.
(n_max=50 to be added to the shapelet sweeps once verified to function.)

---

## Physics choices

### A. Exposed directly in each campaign YAML

| Choice | Key | Current | Notes |
|---|---|---|---|
| Random seed | `seed` | 0 | seeds dataset + pipeline |
| Systems | `dataset.vela_ids`, `dataset.reps` | 12 ids × 5 reps | which Vela systems are in the suite |
| Camera / filter | `dataset.cam`, `dataset.filter_tag` | `12`, `a0.500_f814w` | |
| Pixel grid | `dataset.num_pix`, `dataset.supersample` | 200, 1 | must match how systems were simulated |
| Noise | `dataset.noise_kind`, `dataset.background_rms`, `dataset.exp_time` | forward, 0.002, 2000 | `BackwardProbModel` builds the error map from these |
| Bootstrap MAP | `bootstrap_map_steps`, `bootstrap_map_n_samples` | 200, 50 | recovers the source geometry at the truth lens |
| **Init tightness** | `bootstrap_diag_scale` | 1e-6 | variance of the `qz` the chains start from. **Relevant to the chains-stuck-at-init convergence issue seen earlier — loosen this to let chains explore.** |
| Truth pinning | `bootstrap_pin_eps` | 1e-6 | half-width of the `Uniform` pinning truth-constrained params |
| MCLMC | `n_chains`, `num_burnin_steps`, `num_results` | 8, 4000, 4000 | |
| MCLMC energy var | `desired_energy_variance` | 5e-4 | tuning target |
| MCLMC tuning split | `frac_tune1/2/3` | 0.2 / 0.6 / 0.2 | fractions of burn-in for step-size / mass-matrix / L |
| MCLMC debug | `mclmc_debug` | true | captures the tuning history for the diagnostics plot |
| Shapelet order | `sweep.values` | `[5, 10, 15, 20, 30, 40]` | shapelets & elliptical only (n_max=50 pending verification) |
| Metrics | `metrics` | rhat/ess/nan/zscores/timing | |

### B. In Python — priors and profile settings (the rest of the physics)

These are **not** in YAML (TFP distributions don't serialize cleanly). Edit them
in the builder modules. Lens and lens-light priors are **shared by all three
suites**; only the source prior differs.

**Lens mass (EPL + Shear) — shared.**
`src/gigalens_research/simtests/experiments/vela_shapelets.py:60`
- `theta_E` ~ LogNormal(log 1.25, 0.4); `gamma` ~ TruncN(2.0, 0.5, [1,3])
- `e1,e2` ~ TruncN(0, 0.2, [-0.5,0.5]); `center_x,y` ~ Normal(0, 0.06)
- shear `gamma1` ~ TruncN(0, 0.1, [-0.5,0.5]); `gamma2` ~ Normal(0, 0.1)

**Lens light (SersicEllipse) — shared.** same file `:74`
- `R_sersic` ~ LogNormal(log 1.6, 0.25); `n_sersic` ~ Uniform(0.5, 8)
- `e1,e2` ~ TruncN(0, 0.1, [-0.2,0.2]); `center_x,y` ~ Normal(0, 0.02)
- `Ie` solved by least-squares (not sampled)

**Source priors — suite-specific:**

| Suite | File:line | Source prior |
|---|---|---|
| shapelets | `vela_shapelets.py:85` | `beta`~LogNormal(log 0.7, 0.4); `center_x,y`~Normal(0, 0.5); amplitudes lstsq |
| sersic | `vela_shapelets.py:93` | `R_sersic`~LogNormal(log 0.25, 0.25); `n_sersic`~Uniform(0.5,8); `e1,e2`~TruncN(0,0.3,[-0.5,0.5]); `center_x,y`~Normal(0,0.5); `Ie` lstsq |
| elliptical | `vela_elliptical_shapelets.py:84` | `beta`~LogNormal(log 0.7, 0.4); `e1,e2`~TruncN(0,0.3,[-0.5,0.5]); `center_x,y`~Normal(0,0.5); amplitudes lstsq |

**Profiles / model settings:**
- Mass: `EPL(niter=50)` + `Shear()` — `vela_shapelets.py:138`
- Lens light: `SersicEllipse(use_lstsq=True)`
- Source: `ShapeletsFast(n_max, use_lstsq=True, interpolate=False)` /
  `EllipticalShapelets(n_max, use_lstsq=True)` / `SersicEllipse(use_lstsq=True)`
- Bootstrap optimizer: `optax.adabelief(1e-2, b1=0.95, b2=0.99)` —
  `simtests/pipelines.py:270` (hardcoded; tell me if you want it exposed)

### C. What the z-score metrics actually compare

The Vela truth source is a pixel image with **only** `center_x, center_y` as
parameters, so `mass_zscores` / `all_zscores` score the **shared** lens + lens-light
parameters. The source's `beta`/`n_max`/`R_sersic`/amplitudes are nuisance
flexibility and are not scored. The headline comparison is the lens-parameter
bias as a function of source model (and `n_max`).

---

## Per-suite test (before launching the full Slurm arrays)

Run one system × one sweep point with `mclmc_debug: true` to confirm the model,
bootstrap, and sampler work end-to-end and to eyeball convergence. We will do
this together for each suite before spawning the arrays.
