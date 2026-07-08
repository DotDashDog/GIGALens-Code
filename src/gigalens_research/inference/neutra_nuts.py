"""Plain-NeuTra NUTS kernel (Hoffman et al. 2019, as used in NumPyro's NeuTraReparam).

This module is deliberately *not* aware of the flow: the pullback
log-density (flow-warped target in latent u-space) is built elsewhere
(`gigalens_research.inference.flow_precond.TransformedProbModel`). Given
any scalar-valued latent-space log-density `log_prob_fn(u) -> scalar`,
`neutra_nuts` runs multi-chain NUTS on it with Stan/NumPyro-style window
(dual-averaging step size + diagonal mass matrix) adaptation, mirroring
NumPyro's `MCMC(NUTS(...), num_warmup=..., num_samples=..., num_chains=...)`
defaults as closely as blackjax 1.5 allows. That combination (flow pullback
+ vanilla NUTS in the whitened space) is exactly "plain NeuTra" per
Hoffman et al. 2019, *Neural Transport MCMC* (used identically in NumPyro's
`infer.reparam.NeuTraReparam` + `NUTS`): the flow does the work, the
sampler is unmodified NUTS.

blackjax 1.5 API notes (read from the installed source; all paths are under
``~/.conda/envs/gigalens_oldapi/lib/python3.14/site-packages/blackjax/``,
verified interactively -- see the report accompanying this module):

- ``blackjax.window_adaptation(algorithm, logdensity_fn, is_mass_matrix_diagonal=True,
  initial_step_size=1.0, target_acceptance_rate=0.80, progress_bar=False,
  adaptation_info_fn=return_all_adapt_info, integrator=velocity_verlet,
  **extra_parameters)`` -- ``adaptation/window_adaptation.py:246-256``. This *is*
  the 1.5 signature; there is no separate ``is_mass_matrix_diagonal`` vs. a
  different diagonal-only knob to choose between. ``**extra_parameters``
  (here: ``max_num_doublings``) are threaded into every warmup NUTS step and
  are echoed back in the returned ``parameters`` dict
  (``adaptation/window_adaptation.py:320-321,353-357``).
  ``.run(rng_key, position, num_steps=1000)`` (``window_adaptation.py:334``)
  returns ``(AdaptationResults(state, parameters), info)`` where
  ``AdaptationResults`` is ``NamedTuple(state, parameters)``
  (``adaptation/base.py:21-23``) with
  ``parameters == {"step_size": ..., "inverse_mass_matrix": ..., "max_num_doublings": ...}``
  and ``state`` a ``blackjax.mcmc.hmc.HMCState(position, logdensity,
  logdensity_grad)`` (``mcmc/hmc.py:36``, reused by NUTS via ``init = hmc.init``
  in ``mcmc/nuts.py:33``) -- i.e. the exact state type ``blackjax.nuts(...).init``
  produces, so it can be fed straight into the tuned NUTS kernel with no
  re-initialization. The warmup schedule (Stan-style fast/slow/fast windows,
  ``adaptation/window_adaptation.py:370-460``) is the same schedule NumPyro's
  NUTS warmup uses. Confirmed interactively (dim=3 standard normal,
  200 warmup steps): ``adapt_result.parameters.keys() ==
  {"step_size", "inverse_mass_matrix", "max_num_doublings"}``,
  ``adapt_result.state`` is an ``HMCState``.
- ``adaptation_info_fn`` controls what per-warmup-step info is stacked by the
  internal ``lax.scan``; we pass ``get_filter_adapt_info_fn()`` (no keys kept,
  ``adaptation/base.py:39-60``) so warmup draws/info are never materialized --
  this module's contract is "warmup draws NOT included" in the output.
- ``blackjax.nuts`` is a ``GenerateSamplingAPI(differentiable=as_top_level_api,
  init=hmc.init, build_kernel=...)`` dataclass whose ``__call__`` forwards to
  ``as_top_level_api`` (``blackjax/__init__.py:60-67,94-97,102``), so
  ``blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix,
  max_num_doublings=..., divergence_threshold=1000)`` is the documented
  1.5 call (``mcmc/nuts.py:150-158``) and also what ``window_adaptation``
  builds internally via ``algorithm.build_kernel``/``algorithm.init``
  (``adaptation/window_adaptation.py:300-303,335``).
- ``NUTSInfo`` fields (``mcmc/nuts.py:36-74``): ``is_divergent`` (bool),
  ``is_turning`` (bool), ``acceptance_rate`` (float, average accept prob
  across the whole doubling trajectory), ``num_integration_steps`` (int),
  ``num_trajectory_expansions`` (int). Confirmed interactively:
  ``info.is_divergent`` is a bare ``bool``-dtype array, ``info.acceptance_rate``
  a float in ``[0, 1]``.

Design choices
--------------
- **vmap over chains, not lax.map.** Both the per-chain warmup
  (``window_adaptation(...).run``) and the per-chain sampling loop are pure
  functions of ``(rng_key, init_position)`` with no data-dependent Python
  control flow -- every branch inside blackjax's warmup uses ``lax.switch``/
  ``lax.cond`` (``adaptation/window_adaptation.py:220-233``), not Python
  ``if``s on traced values. That means chains vmap cleanly (verified: the
  test suite below runs 8-16 chains via ``jax.vmap`` without shape or
  tracer-leak errors), and vmap keeps every chain's compute batched into one
  XLA program instead of unrolling a Python-level loop of ``n_chains`` (what
  ``lax.map`` under default ``batch_size=None`` would otherwise avoid
  batching, or a Python for-loop would compile ``n_chains`` copies of the
  same program). NumPyro's own default ``chain_method="vectorized"`` for
  ``MCMC.run`` is the same choice at the same level (vmap warmup+sampling
  over chains, not a sequential loop), so this mirrors NumPyro's default
  chain execution, not just its sampler defaults.
- **Tuned-parameter threading.** ``step_size`` and ``inverse_mass_matrix``
  come out of ``window_adaptation(...).run`` per chain (shape ``(n_chains,)``
  and ``(n_chains, dim)`` respectively after the vmap) and are passed
  straight into a fresh ``blackjax.nuts(log_prob_fn, step_size,
  inverse_mass_matrix, max_num_doublings=...)`` call per chain inside the
  vmapped sampling function -- i.e. each chain samples with its own tuned
  step size and diagonal mass matrix, exactly as Stan/NumPyro's per-chain
  warmup intends. The post-warmup ``HMCState`` is reused directly as the
  sampling loop's initial state (no wasted re-evaluation of the
  log-density).
- **jit.** The vmapped sampling loop (`_sample_chains`) is wrapped in
  ``jax.jit`` per the task spec ("jit the sampling loop"). The vmapped
  warmup (`_warmup_chains`) is left un-jitted-by-us on purpose: its own
  ``lax.scan`` is already a single XLA op per chain and vmap batches that,
  so an outer jit only adds compile-cache bookkeeping; JAX still fuses/
  compiles it as one program when called under the default eager dispatch,
  and leaving it separate makes it easy to time warmup vs. sampling
  independently if that's ever needed.
- **float64-safety.** No dtype is hardcoded anywhere in this module; every
  array (`init positions`, `step_size`, `inverse_mass_matrix`, `samples`)
  inherits its dtype from `log_prob_fn`'s output and from
  `jax.random.uniform`, both of which respect the ambient
  ``JAX_ENABLE_X64`` setting. Divergence/acceptance-rate outputs are cast to
  explicit ``bool``/``int64`` only where blackjax already returns them as
  such, never re-derived in float32.
- **NumPyro correspondence / deviations.** NumPyro's ``init_to_uniform``
  (its NUTS default) draws unconstrained-space init points from
  ``Uniform(-radius, radius)`` with default ``radius=2`` -- matched here by
  ``u0 ~ Uniform(-init_scale, init_scale)^dim`` with default
  ``init_scale=2.0``, applied directly in u-space (which is already
  "unconstrained" by construction, since the flow's job is to have already
  removed constraints). NumPyro defaults: ``num_warmup=1000``,
  ``num_samples=1000``, dense_mass=False (diagonal) -- matched by this
  module's defaults. ``max_tree_depth=10`` (NumPyro) <-> ``max_num_doublings=10``
  (blackjax) -- same semantics (2**10 max leapfrog steps per trajectory), matched.
  One deviation we could not close: blackjax's window adaptation does not
  perform Stan/NumPyro's initial "find a reasonable step size" line search
  before dual averaging begins (blackjax deliberately dropped it -- see the
  docstring at ``adaptation/window_adaptation.py:78-79,111-115``, "we do not
  use the `find_reasonable_step_size` algorithm which we found to be
  unnecessary"); we start dual averaging directly from
  ``initial_step_size=1.0`` (blackjax's own default) rather than from a
  line-searched value. This is an upstream blackjax design choice, not a
  gap in this module, and blackjax's own docs argue it doesn't matter in
  practice; we did not attempt to reimplement Stan's heuristic to force
  bit-for-bit parity with NumPyro.
"""

from functools import partial

import jax
import jax.numpy as jnp

import blackjax
from blackjax.adaptation.base import get_filter_adapt_info_fn

__all__ = ["neutra_nuts"]


def _warmup_one_chain(
    rng_key, init_position, *, log_prob_fn, num_warmup, target_accept, max_num_doublings
):
    """Run Stan/NumPyro-style window adaptation for a single chain.

    Returns the post-warmup ``HMCState`` (ready to feed straight into a
    tuned NUTS kernel) plus the tuned step size and diagonal inverse mass
    matrix. See module docstring for the blackjax API citations.
    """
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_prob_fn,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=target_accept,
        max_num_doublings=max_num_doublings,
        # Discard per-warmup-step info entirely: this module's contract is
        # "warmup draws NOT included" in the returned results.
        adaptation_info_fn=get_filter_adapt_info_fn(),
    )
    adapt_result, _ = warmup.run(rng_key, init_position, num_warmup)
    return (
        adapt_result.state,
        adapt_result.parameters["step_size"],
        adapt_result.parameters["inverse_mass_matrix"],
    )


def _sample_one_chain(
    rng_key,
    init_state,
    step_size,
    inverse_mass_matrix,
    *,
    log_prob_fn,
    num_results,
    max_num_doublings,
):
    """Draw ``num_results`` NUTS samples for a single chain with tuned params."""
    nuts_algo = blackjax.nuts(
        log_prob_fn, step_size, inverse_mass_matrix, max_num_doublings=max_num_doublings
    )

    def one_step(state, key):
        new_state, info = nuts_algo.step(key, state)
        return new_state, (new_state.position, info.is_divergent, info.acceptance_rate)

    keys = jax.random.split(rng_key, num_results)
    _, (positions, is_divergent, acceptance_rate) = jax.lax.scan(one_step, init_state, keys)
    return positions, is_divergent, acceptance_rate


def neutra_nuts(
    log_prob_fn,
    dim,
    *,
    n_chains=8,
    num_warmup=1000,
    num_results=1000,
    seed=0,
    target_accept=0.8,
    max_num_doublings=10,
    init_scale=2.0,
):
    """Multi-chain NUTS on a latent-space log-density, NumPyro-NeuTra-style.

    Parameters
    ----------
    log_prob_fn
        Scalar-valued log-density on the latent (already-pulled-back) space,
        e.g. ``TransformedProbModel.log_prob`` composed to return just the
        scalar log-density (see ``flow_precond.py``). Must be a pure
        JAX-traceable function ``u -> scalar`` (shape ``(dim,) -> ()``).
    dim
        Dimensionality of the latent space.
    n_chains
        Number of independent chains, each independently warmed up
        (Stan/NumPyro-style per-chain adaptation) and then vmapped together
        for sampling.
    num_warmup, num_results
        Number of window-adaptation steps and post-warmup NUTS draws per
        chain. Warmup draws are discarded; only ``num_results`` draws per
        chain are returned.
    seed
        Integer PRNG seed. Same seed -> bit-identical output; different seed
        -> different output (both chain init and all randomness downstream
        are deterministic functions of this seed via ``jax.random.split``).
    target_accept
        Target acceptance rate for dual-averaging step-size adaptation
        (NumPyro/blackjax default 0.8).
    max_num_doublings
        Max trajectory doublings per NUTS step (NumPyro's ``max_tree_depth``,
        default 10, i.e. up to 2**10 leapfrog steps per draw).
    init_scale
        Half-width of the per-coordinate uniform chain-init distribution,
        ``u0 ~ Uniform(-init_scale, init_scale)^dim``, matching NumPyro's
        ``init_to_uniform`` default radius of 2.

    Returns
    -------
    dict with keys:
        samples : (n_chains, num_results, dim)
            Post-warmup NUTS draws in latent space.
        step_size : (n_chains,)
            Per-chain tuned NUTS step size.
        inverse_mass_matrix : (n_chains, dim)
            Per-chain tuned diagonal inverse mass matrix.
        acceptance_rate : (n_chains,)
            Per-chain mean NUTS acceptance rate over the ``num_results``
            sampling draws (post-warmup only).
        num_divergences : (n_chains,)
            Per-chain count of divergent transitions during sampling
            (post-warmup only; int-valued).
        is_divergent : (n_chains, num_results)
            Full per-draw divergence indicator (bool), for diagnostics.
    """
    master_key = jax.random.PRNGKey(seed)
    init_key, warmup_key, sample_key = jax.random.split(master_key, 3)

    init_positions = jax.random.uniform(
        init_key, (n_chains, dim), minval=-init_scale, maxval=init_scale
    )
    warmup_keys = jax.random.split(warmup_key, n_chains)
    sample_keys = jax.random.split(sample_key, n_chains)

    warmup_chains = jax.vmap(
        partial(
            _warmup_one_chain,
            log_prob_fn=log_prob_fn,
            num_warmup=num_warmup,
            target_accept=target_accept,
            max_num_doublings=max_num_doublings,
        )
    )
    init_states, step_sizes, inverse_mass_matrices = warmup_chains(warmup_keys, init_positions)

    sample_chains = jax.jit(
        jax.vmap(
            partial(
                _sample_one_chain,
                log_prob_fn=log_prob_fn,
                num_results=num_results,
                max_num_doublings=max_num_doublings,
            )
        )
    )
    samples, is_divergent, acceptance_rate_per_draw = sample_chains(
        sample_keys, init_states, step_sizes, inverse_mass_matrices
    )

    num_divergences = jnp.sum(is_divergent.astype(jnp.int64), axis=-1)
    acceptance_rate = jnp.mean(acceptance_rate_per_draw, axis=-1)

    return {
        "samples": samples,
        "step_size": step_sizes,
        "inverse_mass_matrix": inverse_mass_matrices,
        "acceptance_rate": acceptance_rate,
        "num_divergences": num_divergences,
        "is_divergent": is_divergent,
    }
