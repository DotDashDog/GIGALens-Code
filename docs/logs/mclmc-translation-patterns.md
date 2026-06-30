# Translating a blackjax sampler into a production GIGALens sampler — patterns from MCLMC

Catalogue of the concrete, reusable patterns by which this repo turns a stock
blackjax MCMC kernel (here: unadjusted MCLMC) into a multi-device, x64-aware,
GIGALens-wired production sampler. Worked example only — no LAPS speculation.

Sources (line numbers are for the **uncommitted local** versions):
- `src/gigalens_research/inference/mclmc.py` — the driver (`MCLMC_JIT`, `full_mclmc_with_adapt_sharded`).
- `src/gigalens_research/inference/blackjax_updated_utils.py` — the shard_map-safe kernel/util/integrator layer.
- `src/gigalens_research/inference/__init__.py` — public surface.
- `src/gigalens_research/inference_utils/pipeline.py` — `MCLMCStage` (the calling convention).

Everything the kernel/adaptation touches is forced into a **single dtype**, all
branching that runs *inside* `shard_map` is done with `jnp.where`/`lax.cond`
(never anything with a data-dependent `while_loop`), and chains are a real
batch axis sharded across devices with `psum`/`pmin` for cross-chain reductions.

---

## 1. Sharding / multi-device

All sharding lives in `full_mclmc_with_adapt_sharded` (`mclmc.py:107`).

- **shard_map import shim** (`mclmc.py:7-10`): `jax.shard_map` if present, else
  `jax.experimental.shard_map.shard_map`, bound as `_shard_map`.
- **Mesh + device math** (`mclmc.py:134-138`, `488`): one 1-D mesh
  `jax.make_mesh((num_devices,), ('device',))`; `num_chains` is floored to a
  multiple of `num_devices`; `chains_per_device = num_chains // num_devices`.
- **Two-level batching.** Chains are split into a *sharded* `'device'` axis
  (via shard_map `in_specs`/`out_specs`) and an *inner* `vmap` over
  `chains_per_device`. The per-chain kernel work is `jax.vmap(per_chain)(...)`
  (`mclmc.py:351-354`); cross-chain reductions are local `jnp` ops followed by a
  collective on `'device'` only.
- **Cross-chain collectives** use `axis_name='device'` exclusively:
  - mass-matrix Welford: `jax.lax.psum` of local sum and local `M2`
    (`mclmc.py:377-385`), then `welford_combine`;
  - step-size sync: `psum` of summed step sizes / total chains (`mclmc.py:418-426`);
  - L adaptation: `jax.lax.pmin` of per-chain min-ESS (`mclmc.py:435-437`).
- **in/out specs** (`mclmc.py:490-501`): carry specs
  `(P('device'), P(), P('device'), P('device'), P(), P('device'))` — state /
  step_sizes / adapt_states / l_buffers are chain-sharded; `params` (L,
  step_size, inverse_mass_matrix) and the Welford accumulator are replicated
  `P()`. Keys come in as `P(None,'device')`.
- **The driver is `@jax.jit` over `@functools.partial(_shard_map, ...)`**
  (`mclmc.py:495-510`); inside, `pbar_scan_fn` runs `step_batched`, then a
  `moveaxis(0,1)` puts samples back to `(chain, step, ...)`.
- **VMA / manual-axis workarounds** (the central translation tax):
  - The kernel uses `jnp.where` instead of `jax.lax.cond` so it passes
    shard_map's varying-manual-axis (VMA) check — `_build_kernel_shardmap`
    docstring + body (`blackjax_updated_utils.py:61-108`).
  - `_ess_shardmap` reimplements blackjax `effective_sample_size` with
    `jax.lax.associative_scan` + `jnp.where` in place of `lax.scan`/`lax.cond`
    (`blackjax_updated_utils.py:310-387`).
  - The PSMILE step-size controller replaces `igamma` (whose internal
    `while_loop` trips VMA) with a Wilson–Hilferty normal approximation
    (`mclmc.py:232-236`).
  - **JAX 0.10 resharding**: 0.10 no longer auto-reshards shard_map inputs;
    inputs with a `PartitionSpec` are pre-sharded via `jax.reshard`
    (fallback `device_put`), and `None`-spec inputs are deliberately *left*
    replicated to avoid the "Closing over inputs sharded on Explicit axes"
    error (`mclmc.py:512-529`). Outputs are gathered back to replicated `P()`
    afterward so innocuous indexing like `samples.step_size[0,-1]` doesn't
    raise `ShardingTypeError` (`mclmc.py:537-544`).

## 2. Precision & platform

- **Single-dtype invariant, energy dtype is canonical** (`mclmc.py:166-186`).
  `_canon = jnp.asarray(state_init.logdensity).dtype` — float64 when the
  likelihood runs under `jax_enable_x64`, float32 otherwise. The whole initial
  state, `svi_mean`, and every adaptation param (inverse_mass_matrix, step_size,
  L) are cast to `_canon` so the scan carry is uniformly one dtype.
- **Why**: `qz.sample()`, `qz.mean()`, `qz.covariance()`, and
  `generate_unit_vector` can yield float32 even under x64; mixing float32 state
  with float64 energy trips `lax.select`/`lax.cond` dtype checks inside
  blackjax `handle_nans` and the mass-matrix cond.
- **Momentum cast** at single-chain init: `generate_unit_vector` draws float32
  normals; `_single_init` casts momentum to the position dtype
  (`blackjax_updated_utils.py:466-487`). Same cast appears in the MAMS kernel
  (`blackjax_updated_utils.py:187-191`).
- **Forward model stays float32 regardless** — the gigalens likelihood
  (`gigalens.jax.model.BackwardProbModel.log_prob`) is float32; only the
  log-density/energy reduction is promoted (`mclmc.py:172-173`).
- No explicit CPU/GPU switch in this file; device count is read from
  `jax.devices()` and the code works for 1..N devices uniformly. x64 itself is
  enabled *upstream* (env / session), not here.

## 3. The logdensity interface

- **The entire gigalens coupling is two lines** (`mclmc.py:42-45`):
  ```
  def log_prob(z):
      return model_seq.prob_model.log_prob(z)[0]
  ```
  `model_seq.prob_model` is a gigalens `ProbModel` (scene-only: it owns
  batch-flexible per-dataset `SceneSimulator`s, so `log_prob(z)` renders through
  them directly — no separately-built `LensSimulator` is passed). The `[0]`
  selects the scalar log-density from gigalens' `(log_prob, aux)` return.
- **Space**: `z` is the **unconstrained** parameter vector. gigalens'
  `BackwardProbModel.log_prob` applies the bijectors and includes the
  change-of-variables log-det internally, so the sampler sees an unconstrained
  Euclidean target and never handles bijectors itself.
- **Batching convention**: `z` is a single flat position of shape `(dim,)`.
  Per-chain batching is added by the sampler (`vmap`/shard_map), not by
  `log_prob`. `dim = state.position.shape[-1]` (`mclmc.py:62`,`140`).
- `log_prob` is passed straight into the kernel builder as `logdensity_fn`
  (`mclmc.py:52-55`) and into `init_multi` for value-and-grad init.

## 4. Initialization

- **Chains are seeded from the VI surrogate `qz`** (a `tfd.MultivariateNormalTriL`
  fit by SVI, see `MCLMCStage`/pipeline; also producible by `HessianSurrogate`
  Laplace approx). Three things come from `qz`:
  1. **Positions**: `qz.sample((n_chains,), seed=init_key)` → `init_multi`
     (`mclmc.py:61`).
  2. **Initial dense inverse mass matrix**: `qz.covariance()` seeds the
     `MCLMCAdaptationState` (`mclmc.py:66-68`).
  3. **SVI-mean reference** for the windowed mass-matrix prior:
     `svi_mean=qz.mean()` (`mclmc.py:83`), folded into the Welford prior
     `WelfordAlgorithmState(svi_mean, svi_inverse_mass_matrix*svi_mass_matrix_weight, svi_mass_matrix_weight)` (`mclmc.py:473`).
- **Dimensionality is inferred** from the sampled positions:
  `dim = state_multi.position.shape[-1]` (`mclmc.py:62`).
- **Per-chain init**: `init_multi(positions, rng_keys, logdensity_fn)`
  (`blackjax_updated_utils.py:489-505`) vmaps `_single_init`
  (`:466-487`), which does `value_and_grad(logdensity_fn)` and a momentum draw
  (with the float dtype cast). `_single_init` raises if `dim < 2`.
- **Default L / step_size** when not supplied: `init_L = sqrt(dim)`,
  `init_step_size = 0.25*sqrt(dim)` (`mclmc.py:64-65`).
- **Prior fallback**: there is no explicit prior-sampling fallback in this file;
  initialization always assumes a usable `qz`. `HessianSurrogate` exists as an
  alternative `qz` provider when SVI is unavailable.

## 5. Step-size / hyperparameter tuning added on top of stock blackjax

All custom adaptation lives **inside `full_mclmc_with_adapt_sharded`** as
closures over `dim`, `desired_energy_var`, `decay_rate`, etc. — they are not
free functions, but the algorithms are the reusable substance.

- **Stage schedule** (`mclmc.py:190-196`, `462-468`): burn-in is split into
  `num_steps1/2/3 = round(num_burnin*frac_tune{1,2,3})`; a per-step integer
  `mode` array drives behaviour — mode 1 = step-size adapt only, mode 2 =
  step-size + mass-matrix adapt, mode 3 = L-tuning buffer fill, mode 0 = results.
  (Mode-2 occupies the first 0.67 of stage 2, mode-1 the last 0.33.)
- **EEVPD step-size controller** `step_size_adapt(previous_state, next_state,
  info, params, adaptive_state, nan_key)` (`mclmc.py:199-212`): the blackjax MCLMC
  controller — `xi = energy_change^2 / (dim * desired_energy_var) + 1e-8`,
  exponential trust weighting, decayed Welford-style running average, and
  `step_size = (x_average/time)^(-1/6)`, clamped to `step_size_max` returned by
  blackjax `handle_nans`. `desired_energy_var` (the EEVPD target) defaults
  `5e-4`.
- **PSMILE alternative controller** `step_size_adapt_psmile_continuous(...)`
  (same signature, `mclmc.py:214-240`): adaptive-moment energy-error controller
  using a Wilson–Hilferty gamma-CDF approximation (the VMA-safe `igamma`
  replacement). Selected by `step_size_adapt_use_psmile` (default False →
  EEVPD).
- **Step-size sync** (`mclmc.py:417-426`): at `step_size_sync_step =
  num_steps1+num_steps2`, all chains are set to the cross-device mean step size
  (a one-shot averaging to remove per-chain drift).
- **Trajectory-length (L) adaptation** (`mclmc.py:428-445`): at
  `L_adaptation_step = tuning_steps`, fill an `l_stage_bufs` buffer during mode 3
  (`mclmc.py:357-366`), compute per-chain ESS via `_ess_shardmap`, take the
  global min ESS (`pmin`), and set `L = Lfactor * num_steps3 * synced_ss /
  global_min_ess` (`Lfactor=0.4`).
- **Windowed inverse-mass-matrix adaptation** (`mclmc.py:244-270`, `370-415`):
  STAN-style expanding windows (`n_windows=3`, geometric `2**k` sizes spanning
  ~0.67 of stage 2), with a precomputed `window_end_mask`. A **dense**
  (non-diagonal) covariance is accumulated with a numerically-stable parallel
  Welford: local mean/`M2` reduced by `psum`, combined into the running
  `WelfordAlgorithmState` by `welford_combine`
  (`blackjax_updated_utils.py:517-533`). At each window boundary the sample
  covariance becomes the new `inverse_mass_matrix`, and the Welford + step-size
  adapt states are reset (`_make_adapt_reset`).
- **Optional Stan-style regularization** `_regularize_cov(cov, n)`
  (`mclmc.py:154-164`, opt-in `regularize_mass_matrix`): symmetrize → `n/(n+5)`
  scale + `1e-3·shrink·I` floor → eigenvalue clip to guarantee PSD, so the
  downstream Cholesky in the dense integrator never sees a non-PSD metric.
  Default off ⇒ byte-identical to baseline.

## 6. Diagnostics & outputs

- **Return contract** (`mclmc.py:99-104`): `MCLMC_JIT` returns
  `result_samples = all_samples[:, -num_results:, :]` of shape
  `(num_chains, num_results, dim)` by default; with `debug_output=True` it
  returns the **full `Hist` namedtuple** instead.
- **`full_mclmc_with_adapt_sharded` returns `(samples, params_final)`**
  (`mclmc.py:546`) — `samples` is the per-step `Hist`, `params_final` the final
  `MCLMCAdaptationState`.
- **Per-step `Hist` history** (`mclmc.py:274-277`, `447-457`), shape
  `(num_chains, total_steps, ...)`: `position`, `step_size`, `L`,
  `inverse_mass_matrix`, `nonan` (step-size-adapt success), `xi` (EEVPD energy
  ratio, logged even in non-adapt modes — `mclmc.py:325-341`),
  `energy_change_raw`, `kernel_nonan`, `step_norm` (‖Δposition‖).
- **Kernel-level diagnostics** are threaded out of the kernel via
  `MCLMCInfoWithExtras` + `KernelExtras`
  (`blackjax_updated_utils.py:29-58`,`92-105`): `energy_change_raw` is the raw
  integrator energy error *before* NaN-zeroing; `kernel_nonan = ~nan_reject`.
  These are instrumentation-only and do **not** affect sampling.
- The pipeline `MCLMCStage` (`pipeline.py:1641-`) declares
  `diagnostics_config()` exposing the stage fractions so
  `plotting.diagnostics.plot_mclmc_diagnostics` can draw tuning-stage
  boundaries.

## 7. GIGALens-specific gotchas (from code + comments)

- **Mixed-dtype trap** (the big one): `qz.sample()/mean()/covariance()` and
  `generate_unit_vector` emit float32 even under x64; without the wholesale cast
  to `_canon`, float32 state meets float64 energy inside blackjax `handle_nans`
  / mass-matrix `cond` and raises dtype errors (`mclmc.py:166-186`,
  `blackjax_updated_utils.py:472-481`).
- **Non-PSD dense metric → Cholesky NaN cascade**: windows 2/3 accumulate from
  an empty float32 Welford with no shrinkage, so a window built from
  frozen/correlated/multi-modal chains can be rank-deficient or non-PSD →
  `cholesky` NaN → rejection cascade. `_regularize_cov` is the documented
  remedy (F3/F4 diagnosis note, `mclmc.py:148-164`).
- **NaN/high-energy handling**: the kernel rejects (reverts state) only on
  NaN/Inf energy error and zeroes the reported `energy_change` so it stays out
  of the step-size controller; the raw value is preserved in
  `extras.energy_change_raw` (`blackjax_updated_utils.py:79-105`). Downstream
  blackjax `handle_nans` provides the `step_size_max` clamp.
- **VMA / `while_loop` landmines under shard_map**: any blackjax routine with a
  data-dependent `while_loop` or `lax.cond` (igamma, `effective_sample_size`,
  the kernel's NaN branch) must be reimplemented with `jnp.where` /
  `associative_scan` / closed-form approximations, or shard_map's manual-axis
  check fails (`mclmc.py:232-236`; `blackjax_updated_utils.py:61-68`,`310-313`).
- **JAX 0.10 sharding strictness**: must pre-`reshard` PartitionSpec inputs but
  *not* `None`-spec ones, and must gather outputs back to replicated before any
  Python-side indexing (`mclmc.py:512-544`).
- **Optional heavy dependency**: blackjax is imported lazily inside
  `MCLMCStage.run` so MAP/SVI/HMC-only users don't need it (`pipeline.py:1697-1699`).
- **Editable-install note**: pyproject declares **no runtime deps** on purpose
  (pinned Shifter container + conda env per `docs/env_setup.md`); `pip install -e .`
  must not be allowed to upgrade JAX/TFP/NumPy out from under the JAX-0.10-nightly
  stack. gigalens is not importable outside that container.

## 8. Reusable APIs (call these, don't reinvent)

From `blackjax_updated_utils.py`:
- `_build_kernel_shardmap(logdensity_fn, inverse_mass_matrix, integrator)` →
  `kernel(rng_key, state, L, step_size)` — shard_map-safe unadjusted MCLMC kernel
  returning `(IntegratorState, MCLMCInfoWithExtras)`. (`:61`)
- `_build_adjusted_kernel_shardmap(logdensity_fn, inverse_mass_matrix, integrator)`
  → `kernel(rng_key, state, step_size, num_integration_steps)` — shard_map-safe
  MAMS (Metropolis-adjusted) variant. (`:151`)
- `generate_isokinetic_integrator_smart(coefficients)` and the prebuilt
  `isokinetic_mclachlan_smart` (+ `_velocity_verlet_/_yoshida_/_omelyan_smart`)
  — isokinetic integrators that accept a **dense 2-D** inverse mass matrix.
  (`:392`, `:461-464`)
- `esh_dynamics_momentum_update_one_step_smart(inverse_mass_matrix)` — the dense-metric
  ESH momentum update (Cholesky-based), building block of the above. (`:412`)
- `_single_init(position, logdensity_fn, rng_key)` → `IntegratorState` —
  value-and-grad single-chain init with dtype-safe momentum. (`:466`)
- `init_multi(positions, rng_keys, logdensity_fn, map_factory=None)` — vmapped
  multi-chain initializer. (`:489`)
- `welford_combine(wa_state1, wa_state2)` → `WelfordAlgorithmState` — parallel
  (chunk-merge) Welford combine for cross-device covariance. (`:517`)
- `_ess_shardmap(input_array, chain_axis=0, sample_axis=1)` — shard_map-safe
  effective sample size. (`:310`)
- `_gen_scan_fn_one_bar(num_samples, progress_bar, print_rate=None, axis_name=None)`
  — drop-in `lax.scan` with a single io_callback fastprogress bar (renders only
  on `axis_index(axis_name)==0`). (`:251`)
- `KernelExtras`, `MCLMCInfoWithExtras` (and `AdjustedKernelExtras`,
  `AdjustedMCLMCInfo`) — diagnostic-carrying info namedtuples. (`:29`,`:46`,`:111`,`:125`)

From `mclmc.py`:
- `full_mclmc_with_adapt_sharded(kernel, num_burnin_steps, num_results,
  state_init, params_init, svi_mean, rng_key, frac_tune1/2/3, desired_energy_var,
  ..., num_chains, svi_mass_matrix_weight, step_size_adapt_use_psmile,
  windowed_mass_matrix, regularize_mass_matrix, progress_bar)` →
  `(samples Hist, params_final)` — the full sharded adapt+sample driver. (`:107`)
- `MCLMC_JIT(model_seq, qz, n_hmc, num_burnin_steps, num_results,
  desired_energy_variance, init_L, init_step_size, frac_tune1/2/3, progress_bar,
  seed, debug_output, regularize_mass_matrix)` (alias `MCLMC`) — the
  GIGALens-facing entry point: builds `log_prob` from `model_seq.prob_model`,
  seeds from `qz`, runs the driver. (`:39`, `:551`)

From blackjax, reused as-is (imported via `blackjax_updated_utils`):
`blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState` (the L/step_size/IMM
container), `handle_nans` (NaN reject + step_size_max clamp), `welford_algorithm`
/ `WelfordAlgorithmState`, `generate_unit_vector`, `pytree_size`, the
`mclachlan/velocity_verlet/yoshida/omelyan_coefficients`, and the integrator
plumbing (`with_isokinetic_maruyama`, `generalized_two_stage_integrator`,
`euclidean_position_update_fn`, `format_isokinetic_state_output`, `ravel_pytree`,
`_normalized_flatten_array`).
