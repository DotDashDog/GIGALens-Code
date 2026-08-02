# GigaLens JAX Implementation Reference

## Package Location

Source: `~/gigalens/src/gigalens/`  
Helpers: `~/GIGALens-Code/src/gigalens_research/`

Most scripts add `~/gigalens/src` and `~/GIGALens-Code` to `sys.path`.

## Core Abstractions (in `gigalens/`)

| Class | Module | Purpose |
|---|---|---|
| `PhysicalModel` | `model.py` | Holds lists of `MassProfile`, lens `LightProfile`, source `LightProfile` |
| `ProbabilisticModel` | `model.py` | Abstract. Holds `prior`, `bij` (bijector). Defines `log_prob(simulator, z)` |
| `SimulatorConfig` | `simulator.py` | Dataclass: `delta_pix`, `num_pix`, `supersample`, `kernel` (PSF), `transform_pix2angle` |
| `LensSimulatorInterface` | `simulator.py` | Abstract. `simulate(params)` and `lstsq_simulate(params, observed_image, err_map)` |
| `ModellingSequenceInterface` | `inference.py` | Legacy ABC. The JAX side no longer implements it — `MAP`/`SVI`/`HMC` are free functions in `jax/inference/` |
| `Parameterized` | `profile.py` | Base for profiles. Has `name: str`, `params: list[str]` |
| `LightProfile` | `profile.py` | Has `light(x, y, **kwargs)`. Optional `use_lstsq` for linear amplitude solving |
| `MassProfile` | `profile.py` | Has `deriv(x, y, **kwargs)` returning `(alpha_x, alpha_y)` deflection angles |

## JAX Implementations (in `gigalens/jax/`)

### ForwardProbModel (`jax/model.py`)

Standard forward-modeling probabilistic model. Simulates image, compares to data.

```python
ForwardProbModel(prior, observed_image, background_rms, exp_time)
```

- `prior`: `tfd.JointDistributionSequential` (nested: `[lens_prior, lens_light_prior, source_light_prior]`)
- `bij`: auto-constructed bijector chain that maps unconstrained z -> physical parameters x
  - `bij.forward(z)` = unconstrained -> physical
  - `bij.inverse(x)` = physical -> unconstrained
- `log_prob(simulator, z)` returns `(log_posterior, chi_squared)` both shaped `(batch,)`
  - `z` is `(batch, n_params)` in unconstrained space
  - Error map: `sqrt(background_rms^2 + im_sim / exp_time)` (Poisson + Gaussian noise)

### BackwardProbModel (`jax/model.py`)

Inverse modeling: uses `lstsq_simulate` to solve for linear amplitudes.

```python
BackwardProbModel(prior, observed_image, background_rms, exp_time)
```

- Error map is computed once at init from the observed image (clipped to non-negative)
- `log_prob` calls `simulator.lstsq_simulate(x, observed_image, err_map)` instead of `simulate`

### LensSimulator (`jax/simulator.py`)

```python
LensSimulator(phys_model, sim_config, bs)
```

- `bs`: batch size (number of parameter sets to simulate in parallel on one device)
- `simulate(params)` returns `(bs, num_pix, num_pix)` images
  - `params` is a nested structure: `(lens_params_list, lens_light_params_list, source_light_params_list)` where each element is a list of dicts, one per profile
- `lstsq_simulate(params, observed_image, err_map)` returns `(images, coefficients)`
- Uses lenstronomy's `subgrid_kernel` for PSF, objax's `average_pool_2d` for downsampling

### Inference Functions (`jax/inference/`)

The inference stages are **free functions**, one per module under
`gigalens/jax/inference/`, all re-exported from the package:

```python
from gigalens.jax.inference import MAP, SVI, HMC, HMC_alt_multi
from gigalens.jax.inference.mclmc import MCLMC, MCLMC_JIT   # see .claude/mclmc.md
```

Each takes the `prob_model` as its **first positional argument**; there is no
wrapper object to build first. Everything the old `ModellingSequence` class
exposed is reachable from the prob model directly: `prob_model.model` is the
scene `LensModel` (the old `phys_model` / `scene_model`), and the
`sim_config` lives on `prob_model.datasets[i].sim_config`.

Note: the package `__init__` binds `MCLMC` (aliased from `MCLMC_JIT`) but not
the name `MCLMC_JIT` — import that one from the `mclmc` submodule.

**MAP** — multi-start gradient descent via `shard_map` across devices:
```python
map_samples, map_lps, map_chisqs = MAP(
    prob_model,
    optimizer,          # optax optimizer (default: adabelief 1e-2)
    start=None,         # (n_samples, n_params) or None to sample from prior
    n_samples=500,      # total across all devices (rounded to multiple of device count)
    num_steps=350,
    seed=0,
    output_type="best", # "all" | "best_step" | "best"
)
```
- `output_type="best"` returns `(1, n_params)`, single best-fit z across all samples and steps
- `output_type="best_step"` returns `(num_steps, n_params)`, best sample at each step
- `output_type="all"` returns `(n_samples, num_steps, n_params)`
- Returns: `(z_unconstrained, log_posteriors, chi_squareds)`

**SVI** — fits multivariate Gaussian surrogate by minimizing ELBO:
```python
qz, loss_hist = SVI(
    prob_model,
    start,              # (1, n_params) unconstrained, typically MAP best
    optimizer,          # optax optimizer (default: adabelief 1e-4)
    n_vi=250,           # samples per ELBO estimate (across all devices)
    init_scales=1e-3,   # initial diagonal scale for covariance
    num_steps=500,
    seed=0,
)
```
- `qz`: `tfd.MultivariateNormalTriL` in unconstrained space
- Tracks best params by ELBO across all steps

**HMC** — Preconditioned HMC with trajectory-length and step-size adaptation:
```python
samples = HMC(
    prob_model,
    q_z,                    # SVI surrogate (MultivariateNormalTriL)
    init_eps=0.3,           # initial step size
    init_l=3,               # initial leapfrog steps
    n_hmc=50,               # total chains across all devices
    num_burnin_steps=250,
    num_results=750,
    max_leapfrog_steps=30,
    seed=0,
)
```
- Mass matrix = inverse of SVI covariance
- Uses TFP's `PreconditionedHamiltonianMonteCarlo` + `GradientBasedTrajectoryLengthAdaptation` + `DualAveragingStepSizeAdaptation`
- Returns `(num_results, n_hmc, n_params)` in unconstrained space (after allgather across nodes)

**HMC_alt_multi** — Two-phase HMC: separate burn-in adapts mass matrix from samples, then fixed-kernel sampling:
```python
samples = HMC_alt_multi(
    prob_model,
    q_z, init_eps=0.3, init_l=3, n_hmc=50, n_vi=1000,
    num_burnin_steps=250, proportion_burnin_to_use=0.9,
    num_results=750, max_leapfrog_steps=30, seed=0,
    force_use_burnin=False,
)
```
- Compares burn-in covariance vs SVI covariance by ELBO; uses whichever is better (unless `force_use_burnin=True`)

## Available Profiles

### Mass Profiles (`gigalens.jax.profiles.mass`)

| Class | Params | Notes |
|---|---|---|
| `epl.EPL` | `theta_E, gamma, e1, e2, center_x, center_y` | Elliptical power law. `niter=18` controls iterative precision |
| `shear.Shear` | `gamma1, gamma2` | External shear |
| `sie.SIE` | `theta_E, e1, e2, center_x, center_y` | Singular isothermal ellipsoid |
| `sis.SIS` | `theta_E, center_x, center_y` | Singular isothermal sphere |
| `tnfw.TNFW` | `Rs, alpha_Rs, r_trunc, center_x, center_y` | Truncated NFW (spherical) |
| `tnfw_ellipse.TNFWEllipse` | (not exported in `__init__`) | Elliptical TNFW |

### Light Profiles (`gigalens.jax.profiles.light`)

| Class | Params | Notes |
|---|---|---|
| `sersic.Sersic` | `R_sersic, n_sersic, center_x, center_y` + `Ie` | Spherical Sersic. `Ie` is amplitude |
| `sersic.SersicEllipse` | `R_sersic, n_sersic, e1, e2, center_x, center_y` + `Ie` | Elliptical Sersic with numerical safety |
| `sersic.CoreSersic` | `R_sersic, n_sersic, Rb, alpha, gamma, e1, e2, center_x, center_y` + `Ie` | Core-Sersic |
| `shapelets.Shapelets` | `beta, center_x, center_y` + `amp00..ampNN` | Hermite shapelets. `n_max` controls order. Supports `use_lstsq` and `interpolate` |

All light profiles: when `use_lstsq=True`, `light()` returns un-scaled basis images (amplitude solved via least squares in `lstsq_simulate`).

## Prior Construction

Priors are `tfd.JointDistributionSequential` of `tfd.JointDistributionSequential`s of `tfd.JointDistributionNamed`s. The nesting matches the physical model structure: `[lens_prior, lens_light_prior, source_light_prior]`.

The default prior for simple simulated systems in the original GIGA-Lens paper (22 params for EPL+Shear, SersicEllipse lens light, SersicEllipse source light) is in `helpers.make_default_prior()`. Key distributions:
- `theta_E`: `LogNormal(log(1.25), 0.4)`
- `gamma`: `TruncatedNormal(2, 0.5, 1, 3)`
- `e1, e2` (EPL): `Normal(0, 0.2)`
- `Ie` (lens): `LogNormal(log(300), 0.5)`
- Source center: `Normal(0, 0.5)` (wider than lens center `Normal(0, 0.06)`)

## Parameter Spaces

All inference methods operate in **unconstrained space** (z). Bijectors handle the transform:
- `prob_model.bij.forward(z_list)` -> physical params (nested dicts)
- `prob_model.bij.inverse(x)` -> unconstrained flat array

`z` is flat `(batch, n_params)`. Physical params `x` are nested: `([{lens0_dict}, {lens1_dict}], [{lens_light0_dict}], [{source_light0_dict}])`.

## Pipeline Utilities (`gigalens_research.inference_utils`)

The research-side orchestration layer wraps the free functions above in
cache-aware stages. Full reference:
`src/gigalens_research/inference_utils/README.md`.

### Standard MAP -> SVI -> HMC run

```python
from gigalens_research.inference_utils import (
    InferenceContext, Pipeline, MAPStage, SVIStage, HMCStage,
)

ctx = InferenceContext.from_prob_model(prob_model)
pipeline = Pipeline(ctx, seed=42)
pipeline.add(MAPStage(num_steps=350, n_samples=500))
pipeline.add(SVIStage(num_steps=1500, n_vi=1000))
pipeline.add(HMCStage(n_hmc=50, num_burnin_steps=250, num_results=750))

artifacts = pipeline.run(out_dir="results/system_07", resume=True)
```

- `InferenceContext` carries `phys_model`, `prob_model`, `sim_config` — all
  derived from the prob model by `from_prob_model`
- Each stage hashes its inputs and reloads from `<out_dir>/<stage>/arrays.npz`
  when nothing upstream changed
- `pipeline.posterior()` returns a uniform Posterior view (`.rhat`,
  `.samples`, …) over whichever terminal stage ran

### Plotting

```python
from gigalens_research.plotting import plot_image_results, cornerplot_results, display_results
```

`display_results(r, true_img, lens_sim, true_params)` renders the image /
residual / loss / corner suite; pass `sim_config=` alongside
`plot_caustics=True` to overlay caustics.

## Typical Setup (EPL + Shear + SersicEllipse)

```python
import sys; sys.path.insert(0, '/global/homes/l/linusu/gigalens/src')
sys.path.insert(0, '/global/homes/l/linusu/GIGALens-Code')

from gigalens.jax.inference import MAP, SVI, HMC
from gigalens.jax.model import ForwardProbModel
from gigalens.model import PhysicalModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear

phys_model = PhysicalModel(
    lenses=[EPL(), Shear()],
    lens_light=[SersicEllipse(use_lstsq=False)],
    source_light=[SersicEllipse(use_lstsq=False)],
)
sim_config = SimulatorConfig(delta_pix=0.08, num_pix=80, supersample=2, kernel=psf_kernel)
prior = make_default_prior()
prob_model = ForwardProbModel(prior, observed_image, background_rms=0.2, exp_time=100)

best, lps, chisqs = MAP(prob_model, optax.adabelief(1e-2))
qz, loss_hist     = SVI(prob_model, best, optax.adabelief(1e-4))
samples           = HMC(prob_model, qz)
```

> **Stale:** the `ForwardProbModel` / `PhysicalModel` imports above predate the
> scene API; `gigalens.jax.model` and `gigalens.model` now live under
> `gigalens/old_api/`. The current path builds a scene `LensModel` and wraps it
> in `gigalens.jax.scene_prob_model.ProbModel(model, datasets, mode="lstsq")` —
> see the builders in `gigalens_research/simtests/`. Everything below the
> `prob_model =` line is already correct for both.

## Multi-Device / Multi-Node

- `n_samples`, `n_vi`, `n_hmc` are automatically rounded to multiples of `jax.device_count()`
- MAP uses `shard_map` with `P('device')` partitioning
- SVI uses `shard_map` for gradient averaging across devices
- HMC uses `pmap` with per-device seeds and `process_allgather` for multinode
- For multinode: call `jax.distributed.initialize()` before any JAX operations
