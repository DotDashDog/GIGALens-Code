# JAX shard_map: Pitfalls and Workarounds

`jax.experimental.shard_map.shard_map` (JAX ≤ 0.9) and `jax.shard_map` (JAX ≥ 0.10) enforce strong axis-type rules. JAX 0.6 surfaced these as **VMA** (Varying Manual Axis) errors; JAX 0.10 reorganized them around `AxisType.{Explicit, Manual, Auto}`. The two regimes share recurring failure patterns; the ones below are all things that have actually broken and been fixed in this workspace (`mclmc_alt.py`, `gigalens.jax.inference`).

## Forward-compatible imports & primitives

Use these aliases in any module that has to support both kernels (`gigalens.jax.inference` and `alternate_inference.mclmc_alt` already do):

```python
# shard_map moved from jax.experimental to top-level in JAX 0.10.
try:
    _shard_map = jax.shard_map  # type: ignore[attr-defined]
except AttributeError:
    from jax.experimental.shard_map import shard_map as _shard_map

# jax.reshard is the JAX 0.10 API for changing an array's sharding outside
# jit; jax.device_put with the same NamedSharding is the older equivalent.
_reshard = getattr(jax, 'reshard', jax.device_put)

# jax.lax.pvary was removed in JAX 0.10; replacement is
# jax.lax.pcast(..., to='varying'). pcast also errors if the source is
# already 'varying' (whereas pvary was idempotent), so swallow that case.
def _pvary(x, axis_name):
    pcast = getattr(jax.lax, 'pcast', None)
    if pcast is not None:
        try:
            return pcast(x, axis_name, to='varying')
        except ValueError as e:
            if 'from=varying' in str(e):
                return x
            raise
    return jax.lax.pvary(x, axis_name)  # type: ignore[attr-defined]
```

## JAX 0.10 specific patterns (Explicit / Manual / Auto axes)

The default mesh built with `jax.make_mesh((n,), ('device',))` has `AxisType.Explicit` axes. Inputs created in this context get tagged `({Explicit: ('device',)})`. Inside a `shard_map` body the axes are switched to `Manual`, and JAX 0.10 has not yet implemented the bridge between these two modes for arbitrary inputs.

### A. Strict `in_specs` matching — pre-shard before calling

JAX 0.10 no longer auto-reshards `shard_map` inputs:
```
ValueError: in_specs passed to shard_map: P(None, 'device') does not match
the specs of the input: P(None, None) for arg: key<fry>[4000,8].
```
**Fix**: pre-shard *only* the args whose `in_specs` is a `PartitionSpec`; leave `None`-spec args alone (pre-sharding them tags them Explicit and triggers pattern B):
```python
sharded = NamedSharding(mesh, P('device'))
keys = _reshard(keys, sharded)            # match P('device')
state = _reshard(state, sharded)          # match P('device')
# Do NOT reshard args with in_specs=None (e.g. mode/iter counters).
out = run_sharded(state, keys, mode, ...)
```

### B. "Closing over inputs to shard_map where the input is sharded on Explicit axes"

Anything Explicit-tagged that flows into a `shard_map` body — whether captured as a closure, sliced inside an inner trace (e.g. `value_and_grad`), or even threaded through `scan` carries — can trip:
```
NotImplementedError: Closing over inputs to shard_map where the input is sharded on
`Explicit` axes is not implemented. ... Got input with shape f32[779]({Explicit: ('device',)})
```

Fixes that have actually worked, in order of preference:

1. **Don't pre-shard `None`-spec inputs.** Replicated arrays without a `NamedSharding` don't pick up the Explicit tag, so leave scalars/iter counters/mode flags as plain `jnp` arrays.
2. **Strip the Explicit tag at the source via a numpy round-trip** when the array is inherited from an upstream sharded computation (e.g. an SVI starting point coming from MAP):
   ```python
   replicated_params = jnp.asarray(np.concatenate([np.asarray(a), np.asarray(b)]))
   ```
   This is what `ModellingSequence.SVI` does to break the chain from MAP's reshard-to-replicated output.
3. **Put `scan` *inside* `shard_map`, not the other way around.** A `@jit` → `scan` → `@shard_map` body forces an Explicit→Manual hop on every carry; flipping it to `@jit @shard_map` containing the `scan` keeps every per-step op in Manual mode. This is the structure used by both `mclmc_alt.full_mclmc_with_adapt_sharded` and the JAX-0.10 rewrite of `ModellingSequence.SVI`.
4. **Avoid slicing Explicit-tagged operands inside an inner `value_and_grad`/`jit` trace.** Slice in the outer Manual frame and pass the slices in as separate args:
   ```python
   # one_step_sharded is the scan body inside shard_map (Manual mode)
   mean = params[:n_params]
   cov_chol_raw = params[n_params:]
   val, (g_mean, g_cov) = jax.value_and_grad(neg_elbo, argnums=(0, 1))(mean, cov_chol_raw, key)
   ```

`jax.lax.pcast(x, axis_name, to='manual')` is the JAX-0.10-blessed way to flip a value into Manual mode, but in practice it does **not** rescue an Explicit-tagged carry; treat it as a documentation marker only.

### C. `ShardingTypeError` on output indexing

`shard_map` outputs declared with `out_specs=P('device')` (or any partitioned spec) carry the chain axis sharded across devices. Indexing them in user code raises:
```
ShardingTypeError: Use `.at[...].get(out_sharding=)` to provide output PartitionSpec
... operand=ShapedArray(float32[8@device,4000])
```
**Fix**: reshard outputs to fully-replicated **before returning** them across the API boundary:
```python
_replicated = NamedSharding(mesh, P())
samples = jax.tree.map(lambda x: _reshard(x, _replicated), samples)
params_final = jax.tree.map(lambda x: _reshard(x, _replicated), params_final)
return samples, params_final
```
Both `MCLMC_JIT` (`mclmc_alt.full_mclmc_with_adapt_sharded`) and `ModellingSequence.MAP` do this on every sharded output.

### D. `pvary` removal & idempotency

`jax.lax.pvary(x, 'device')` is gone in JAX 0.10. Replacement is `jax.lax.pcast(x, 'device', to='varying')`, but that errors with `Unsupported pcast from=varying, to='varying'` when the source is already varying (e.g. an `optimizer.init(...)` output once params are sharded). Use the `_pvary` wrapper above.

## Pre-existing patterns (still valid on JAX 0.6, also useful guidance on 0.10)

These were originally diagnosed during the MCLMC parallelization on the JAX 0.6 kernel. They still apply on 0.10; the language has just shifted from "VMA" to "Manual axis varying-ness".

### 1. `jax.lax.cond` → use `jnp.where`

`lax.cond` requires both branches to produce identical varying-axis annotations. Inside `shard_map`, one branch may produce varying values while the other produces replicated values, causing:
```
TypeError: cond branches must have equal output types but they differ
```

**Fix**: Replace `lax.cond` with `jnp.where`-based selection:
```python
result = jax.tree.map(lambda a, b: jnp.where(flag, a, b), true_val, false_val)
```

### 2. `jax.lax.scan` carry type mismatches → `_pvary`

When `scan` initial carry values are replicated (e.g., `jnp.zeros(...)`) but the body produces varying outputs:
```
TypeError: scan body function carry input and carry output must have equal types
```
**Fix**: Mark initial carry values as varying before passing to `scan` (use the `_pvary` wrapper for cross-version safety):
```python
init = jax.tree.map(lambda x: _pvary(x, 'device'), init)
carry, ys = jax.lax.scan(body, init, xs)
```

### 3. `vmap(axis_name=...)` inside `shard_map` → remove `axis_name`

Nesting `jax.vmap(..., axis_name='local')` inside `shard_map` and then using collectives like `lax.psum` over `'local'` causes:
```
ValueError: Collective psum_invariant must be applied to a device-varying type
```

**Fix**: Use `jax.vmap` **without** `axis_name` for within-device parallelism. Perform local reductions with standard `jnp.sum/min/mean` on the batch axis, and use `lax.psum/pmin` only on the `shard_map` axis:
```python
results = jax.vmap(per_chain_fn)(batched_states, batched_keys)
local_sum = jnp.sum(results, axis=0)
global_sum = jax.lax.psum(local_sum, axis_name='device')
```

### 4. JAX library `while_loop` internals → replace with non-iterative alternatives

Some JAX/scipy functions (e.g., `jax.scipy.stats.gamma.cdf` via `igamma`, `jax.scipy.special.gammainc`) use internal `while_loop`s whose carry mixes replicated constants with varying data. This triggers the same carry-type error as #2, but you can't apply `pvary` since the loop is inside JAX.

**Fix**: Replace with a non-iterative approximation. Example for `gamma.cdf`:
```python
# Wilson-Hilferty normal approximation (no while_loop)
x_std = x / (scale * shape + eps)
z = (jnp.cbrt(x_std) - (1.0 - 1.0 / (9.0 * shape))) / jnp.sqrt(1.0 / (9.0 * shape + eps))
cdf_value = jax.scipy.special.ndtr(z)  # ndtr uses erfc, no while_loop
```

### 5. BlackJAX functions with VMA / Manual-mode issues

These BlackJAX functions are incompatible with `shard_map` and have custom replacements in `mclmc_alt.py`:

| BlackJAX function | Issue | Replacement |
|---|---|---|
| `blackjax.mcmc.mclmc.build_kernel` | `lax.cond` | `_build_kernel_shardmap` (uses `jnp.where`) |
| `blackjax.diagnostics.effective_sample_size` | `lax.scan` carry mismatch | `_ess_shardmap` (uses `lax.associative_scan`) |

## General shard_map pattern (current canonical layout)

```python
mesh = jax.make_mesh((num_devices,), ('device',))  # Explicit axes by default in JAX 0.10
sharded = NamedSharding(mesh, P('device'))
replicated = NamedSharding(mesh, P())

# Pre-shard partitioned inputs (skip None-spec inputs).
sharded_arr = _reshard(sharded_arr, sharded)

@jax.jit
@functools.partial(_shard_map, mesh=mesh,
    in_specs=(P('device'), P()),  # use P()/None as appropriate
    out_specs=P())
def run(sharded_data, replicated_data):
    # Manual mode: do all per-step work here, including any scan.
    # - jax.vmap WITHOUT axis_name for per-item parallelism
    # - jnp.sum/mean for local reductions
    # - jax.lax.pmean/psum('device') for cross-device only
    # - jnp.where instead of lax.cond
    # - _pvary on scan initial carries if needed
    ...

out = run(sharded_data, replicated_data)
out = jax.tree.map(lambda x: _reshard(x, replicated), out)  # avoid C above
return out
```
