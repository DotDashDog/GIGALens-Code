# MCLMC Sampler Reference (mclmc_alt.py)

**Location**: `~/GIGALens-Code/alternate_inference/mclmc_alt.py`  
**Status**: Work in progress — not a final product.

MCLMC (Microcanonical Langevin Monte Carlo) is an alternative to HMC for posterior sampling. It is based on BlackJAX's MCLMC implementation, extended here with multi-chain support and non-diagonal mass matrix adaptation.

## Top-Level API

### MCLMC (three-phase: burnin then sampling)

```python
from alternate_inference.mclmc_alt import MCLMC

samples = MCLMC(
    model_seq,              # ModellingSequence instance (provides phys_model, sim_config, prob_model)
    qz,                     # SVI surrogate (MultivariateNormalTriL), used for init positions and mass matrix
    n_hmc=16,               # number of parallel chains
    num_burnin_steps=1000,  # total adaptation steps
    num_results=2000,       # sampling steps after adaptation
    mass_matrix_adapt=True, # adapt mass matrix from sample covariance during burnin
    continuous_adaptation=True,  # continuously update mass matrix vs. just track
    desired_energy_variance=5e-4,# target energy variance for step size tuning
    init_L=None,            # initial trajectory length (default: sqrt(dim))
    init_step_size=None,    # initial step size (default: sqrt(dim)*0.25)
    progress_bar=False,
    seed=0,
)
```

**Returns**: `(num_chains, num_results, n_params)` in unconstrained space.

Internally:
1. Builds `LensSimulator` with `bs=1` from `model_seq`
2. Defines `log_prob(z) = model_seq.prob_model.log_prob(lens_sim, z)[0]`
3. Initializes chain positions from `qz.sample((n_chains,))`
4. Runs `mclmc_find_L_and_step_size_smart` for burn-in adaptation
5. Runs fixed-parameter sampling via `blackjax.util.run_inference_algorithm`

### MCLMC_JIT (single-scan adaptation + sampling)

```python
from alternate_inference.mclmc_alt import MCLMC_JIT

samples = MCLMC_JIT(
    model_seq, qz, n_hmc=16,
    num_burnin_steps=1000, num_results=2000,
    desired_energy_variance=5e-4,
    frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
    progress_bar=False, seed=0,
    debug_output=False,       # if True, returns full history namedtuple
    step_size_adapt_use_psmile=False,  # use pSMILE step size adaptation
    use_shard_map=False,      # multi-GPU: distribute chains across devices
    windowed_mass_matrix=False, # STAN-style expanding-window mass matrix adaptation
    mass_matrix_num_effective_samples=1000, # EMA decay for continuous mass matrix
)
```

**Returns**: `(num_chains, num_results, n_params)` (or full `Hist` namedtuple if `debug_output=True`).

Uses `full_mclmc_with_adapt` (single-device) or `full_mclmc_with_adapt_sharded` (multi-GPU) which runs adaptation and sampling in a single `jax.lax.scan`.

### Multi-GPU via shard_map (`use_shard_map=True`)

Distributes chains evenly across devices. Architecture:
- `shard_map(axis_name='device')` wraps `jax.lax.scan`
- `jax.vmap` (**without** `axis_name`) for per-chain kernel calls inside the scan body
- Local cross-chain reductions via `jnp.sum/min` on batch dim
- Cross-device reductions via `lax.psum/pmin('device')`

Uses custom shard_map-compatible replacements for BlackJAX functions that have VMA issues (see `jax-shard-map.md`):
- `_build_kernel_shardmap`: replaces `blackjax.mcmc.mclmc.build_kernel` (`jnp.where` instead of `lax.cond`)
- `_ess_shardmap`: replaces `blackjax.diagnostics.effective_sample_size` (`associative_scan` instead of `scan`)
- Wilson-Hilferty approximation for gamma CDF in pSMILE adapter (avoids `igamma`'s internal `while_loop`)

### Windowed mass matrix adaptation (`windowed_mass_matrix=True`)

STAN-style expanding-window scheme (only with `use_shard_map=True`). Instead of continuously updating the mass matrix with EMA decay:
1. Mode 2 phase is divided into **3 doubling windows** (size ratio 1:2:4)
2. Within each window: Welford accumulates samples, mass matrix is NOT updated
3. At window boundaries: mass matrix is estimated from that window's samples, Welford resets, step size adaptation state resets (preserving `step_size_max`)

When `windowed_mass_matrix=False` (default), mass matrix is updated every step during mode 2 with EMA decay.

## Adaptation Stages

Both APIs use a three-stage burn-in:

1. **Stage 1** (`frac_tune1`, mode=1): Step-size-only adaptation. Uses energy variance to tune step size via `Var[E] = O(eps^6)` relationship.

2. **Stage 2** (`frac_tune2`, mode=2): Step size + mass matrix adaptation. Collects sample covariance across chains using Welford's online algorithm. The mass matrix is a weighted average of the SVI covariance and the sample covariance: `(n_smp * C_smp + n_svi * C_svi) / (n_smp + n_svi)`.

3. **Stage 3** (`frac_tune3`, mode=3): Trajectory length (L) adaptation based on effective sample size (ESS). `L = Lfactor * step_size * num_steps3 / min(ESS)`.

After adaptation, step sizes are synchronized across chains via `pmean`.

## Step Size Adaptation Variants

**Default** (`step_size_adapt`): Exponential moving average of `xi = E_change^2 / (dim * desired_energy_var)`, weighted by how close `xi` is to 1. Step size derived from `Var[E] = O(eps^6)`.

**pSMILE** (`step_size_adapt_psmile_continuous`, enabled via `step_size_adapt_use_psmile=True`): Fits a Gamma distribution to `|dE|` via moment matching, adjusts step size based on CDF position relative to median.

## Multi-Chain Infrastructure

### init_multi

```python
state = init_multi(positions, rng_key, logdensity_fn, map_factory=None)
```
- `positions`: `(n_chains, n_params)` — initial positions
- Returns: `IntegratorState` with batched fields (position, momentum, logdensity, logdensity_grad)
- Default: `jax.vmap`. Custom `map_factory` for device-level parallelism.

### build_kernel_multi / mclmc_multi

```python
alg = mclmc_multi(
    logdensity_fn, L, step_size, num_chains,
    integrator=isokinetic_mclachlan_smart,
    inverse_mass_matrix=qz.covariance(),
)
```
- Returns a `SamplingAlgorithm` with `init_fn` and `update_fn` (compatible with `blackjax.util.run_inference_algorithm`)
- `L` and `step_size` are broadcast to all chains (scalar -> replicated)

### Single-chain kernel (_single_kernel)

Wraps the integrator step with energy-error rejection: if `|energy_error| > sqrt(dim * eev_max)`, the step is rejected and the previous state is returned.

## Non-Diagonal Mass Matrix Integrators

The stock BlackJAX integrators only support diagonal mass matrices. `mclmc_alt.py` defines "smart" variants that work with full covariance matrices:

```python
isokinetic_mclachlan_smart  # McLachlan coefficients (default, recommended)
isokinetic_yoshida_smart    # Yoshida coefficients
isokinetic_omelyan_smart    # Omelyan coefficients
```

These use `esh_dynamics_momentum_update_one_step_smart`, which:
1. Cholesky-decomposes the inverse mass matrix once at construction
2. Transforms gradients by `chol.T @ grad` before the ESH momentum update
3. Transforms momentum by `chol @ momentum` after normalization

## Welford Utilities for Cross-Chain Covariance

- `welford_combine(state1, state2)`: Merges two Welford states (mean, m2, sample_size) using parallel Welford algorithm
- `aggregate_chain_welford(welford_state, chain_axis)`: Aggregates across a named JAX axis via `lax.psum/pmean`
- `aggregate_m2`, `aggregate_covariance`: Batch covariance pooling from multiple groups

## full_mclmc_with_adapt (Single-Scan Version)

Used by `MCLMC_JIT`. Runs the entire adaptation + sampling in one `jax.lax.scan`:

- Mode schedule: `[1]*n1 + [2]*n2_a + [1]*n2_b + [3]*n3 + [0]*(total-tuning)` where mode 0 = no adaptation
- At `step_size_sync_step` (end of stage 2): synchronizes step sizes across chains
- At `L_adaptation_step` (end of stage 3): computes ESS-based L
- Tracks full `Hist` namedtuple: `(position, step_size, L, inverse_mass_matrix, nonan, xi)`
- Uses `jax.vmap(..., axis_name='chain')` for cross-chain communication in mass matrix adaptation

## Key Differences from GigaLens HMC

| | GigaLens HMC | MCLMC |
|---|---|---|
| Algorithm | TFP Preconditioned HMC | BlackJAX MCLMC (microcanonical Langevin) |
| Acceptance | Metropolis-Hastings | Unadjusted (energy-error rejection only) |
| Mass matrix | Fixed from SVI covariance | Adapted during burn-in (SVI + sample covariance blend) |
| Parallelism | `pmap` across devices | `vmap` (single device) or `shard_map` (multi-GPU via `use_shard_map=True`) |
| Trajectory length | Adapted by TFP's GBTLA | ESS-based L adaptation |
| Integration | Leapfrog | Isokinetic McLachlan/Yoshida/Omelyan |

## Caveats (Work in Progress)

- Mass matrix adaptation can produce ill-conditioned matrices if there aren't enough effective samples
- The SVI covariance weighting (`svi_mass_matrix_weight`) acts as a regularizer; default is `10 * n_chains` in `MCLMC_JIT`
- `num_chains=1` may cause errors in the multi-chain path; use the original BlackJAX API for single chains
- `continuous_adaptation=True` is described in code as "mostly untested"
- The pSMILE step size adapter is experimental; uses Wilson-Hilferty approximation for gamma CDF in shard_map mode
- `windowed_mass_matrix` only affects the sharded version (`use_shard_map=True`)
- When editing the sharded code path, see `jax-shard-map.md` for VMA pitfall patterns
