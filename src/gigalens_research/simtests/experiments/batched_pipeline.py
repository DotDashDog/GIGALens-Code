"""System-batched MAP -> SVI -> MCLMC over SBC point-source systems (phase B).

Companion to :mod:`batched_point_source` (phase A, the batched log-prob) —
see that module's docstring for scope (SIMULATED SBC systems only) and the
attribute-swap mechanism. This module vmaps the entire per-system inference
pipeline over the system axis: every system gets its own MAP particles, SVI
surrogate, and MCLMC chains — including its own adapted step size, L, and
mass matrix, which fall out of the outer vmap with no segmented-adaptation
surgery.

Fidelity contract
-----------------
Each stage mirrors the solo implementation it replaces, including the RNG
stream structure, so a batched run with per-system seed s is the same
computation as the solo run with seed s (up to XLA reassociation roundoff,
which the chaotic samplers amplify — certification is therefore statistical,
not bitwise; see ``simtests/tests/batched_pipeline_test.py``):

- ``batched_map`` mirrors ``ModellingSequence.MAP`` (adam behind
  zero_nans+clip, loss ``-mean(lp)/loss_normalization``, per-step best particle
  recorded BEFORE the update — the C-6 fix — then argmax over steps, i.e. the
  MAPStage ``output_type="best_step"`` path).
- ``batched_svi`` mirrors ``ModellingSequence.SVI`` (MVN-TriL surrogate via
  FillScaleTriL(Exp, diag_shift=1e-6), n_vi-sample ELBO, best-loss parameter
  tracking, adabelief) with the solo per-step key chain (split parent, split
  1 device, take device 0).
- ``batched_mclmc`` reimplements ``full_mclmc_with_adapt_sharded`` WITHOUT
  shard_map: chains are an explicit vmapped axis and the cross-device
  collectives become plain reductions over it (psum -> sum, pmin -> min,
  axis_size -> n_chains). The kernel, NaN handling, Welford accumulator, and
  ESS come from the same gigalens/blackjax modules the solo sampler uses
  (``_build_kernel_shardmap`` is shard_map-COMPATIBLE, not -dependent: plain
  jnp ops, no collectives). Tuning schedule, windowed mass matrix, step-size
  sync and L adaptation replicate the solo code path line for line; the one
  intended difference is memory layout — burn-in positions are not stored,
  and ``thin_every`` can subsample the kept draws at emission time (the solo
  sampler stores every position of every step).

All stage entry points take a :class:`BatchedPointSourceProb` plus per-system
seed arrays and return system-stacked artifacts matching the solo stages'
outputs (``z_best``/``lp_hist``/``chisq_hist``; ``qz_loc``/``qz_scale_tril``/
``svi_loss_hist``; ``samples_z`` ``(S, C, N, P)``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _check_seeds(seeds: Any, n_systems: int, who: str):
    import jax.numpy as jnp
    seeds = jnp.asarray(seeds)
    if seeds.shape != (n_systems,):
        raise ValueError(
            f"{who}: seeds must be one integer per system, shape ({n_systems},); "
            f"got {tuple(seeds.shape)}. Distinct per-system seeds are what keeps "
            "the batched campaign's randomness equivalent to solo runs.")
    if not jnp.issubdtype(seeds.dtype, jnp.integer):
        raise ValueError(f"{who}: seeds must be integers (got dtype {seeds.dtype}).")
    return seeds


# --------------------------------------------------------------------------- MAP
def batched_map(bp, seeds, *, num_steps: int = 1500, n_samples: int = 1000,
                map_lr: float = 3e-3, map_clip_norm: float = 1.0
                ) -> Dict[str, np.ndarray]:
    """Multi-start MAP for every system at once (solo ``MAPStage`` semantics).

    Returns ``z_best (S, P)``, ``lp_hist (S, num_steps)``,
    ``chisq_hist (S, num_steps)``, ``best_step (S,)``.
    """
    import jax
    import jax.numpy as jnp
    import optax

    seeds = _check_seeds(seeds, bp.n_systems, "batched_map")
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(float(map_clip_norm)),
        optax.adam(float(map_lr)),
    )
    prior = bp.prob.prior          # shared across systems by SBC construction
    loss_norm = float(bp.prob.loss_normalization)

    def one(row, seed):
        p = bp._swapped(row)
        start = prior.sample(int(n_samples), seed=jax.random.PRNGKey(seed))
        z0 = p.bij.inverse(start)

        def loss(z):
            lp, chisq = p.log_prob(z)
            return -jnp.mean(lp) / loss_norm, (lp, chisq)

        vg = jax.value_and_grad(loss, has_aux=True)
        opt_state = optimizer.init(z0)

        def one_step(carry, _):
            z, opt_state = carry
            (_, (lp, chisq)), grads = vg(z)
            # Record BEFORE the update (the C-6 MAP pairing fix): the stored
            # particle must be the one the scores were evaluated at.
            i = jnp.nanargmax(lp)
            b = (z[i], lp[i], chisq[i])
            updates, opt_state = optimizer.update(grads, opt_state)
            z = optax.apply_updates(z, updates)
            return (z, opt_state), b

        (z_fin, _), (bz, blp, bchi) = jax.lax.scan(
            one_step, (z0, opt_state), None, length=int(num_steps))
        j = jnp.nanargmax(blp)     # MAPStage: globally best step
        lp_fin, _ = p.log_prob(z_fin)   # final-pool scores (anneal-phase seed)
        return bz[j], blp, bchi, j, z_fin, lp_fin

    z_best, lp_hist, chisq_hist, best_step, z_final, lp_final = jax.jit(
        jax.vmap(one))(bp.data, seeds)
    return {"z_best": np.asarray(z_best), "lp_hist": np.asarray(lp_hist),
            "chisq_hist": np.asarray(chisq_hist),
            "best_step": np.asarray(best_step),
            "z_final": np.asarray(z_final), "lp_final": np.asarray(lp_final)}


# ------------------------------------------------------------- MAP anneal (mc)
def batched_map_anneal(bp, z_pool, lp_pool, *, ladder, n_particles: int = 64,
                       num_steps: int = 300, lr: float = 1e-3,
                       clip_norm: float = 1.0, block: int = 16
                       ) -> Dict[str, np.ndarray]:
    """Multiplicity-constraint eps-annealing MAP refinement, all systems at once.

    Phase 2 of the annealed-MAP design (docs/logs/point-source-sbc.md P-9):
    bulk MAP runs WITHOUT the multiplicity term (1000 particles x quadrature
    would be ruinous, and most particles are discarded anyway); this phase takes
    the final phase-1 particle pool, keeps the top ``n_particles`` per system by
    phase-1 log-prob, and Adam-descends them for ``num_steps`` per ladder rung
    ``(eps, lam, grid_n)`` with the term ACTIVE — coarse rungs reach phantom
    modes (capture radius ~3 eps), fine rungs sharpen the wall at the caustic.
    Stage knobs are trace-time constants, so each rung re-jits; the term is left
    at the FINAL rung's configuration (the measured MCLMC target) on return.

    Returns ``z_best (S, P)`` scored at the final rung, plus the refined pool
    ``anneal_z (S, K, P)`` / ``anneal_lp (S, K)`` for diagnostics.

    ``block`` bounds GPU memory: the multiplicity-term gradient's working set
    scales with (systems x particles), and all 64 particles at once OOMs an
    11 GB card beyond ~5 systems at the final 384^2 rung (P-10 launch 1,
    job 24307807: 13 systems x 64 = 832 lanes asked for 9.4 GiB in one op).
    Particles are therefore Adam-descended in sequential blocks of ``block``
    per system (lax.map), cutting peak memory by K/block at the same total
    FLOPs. Note global-norm clipping then acts per block rather than across
    all K particles; per-particle semantics are otherwise unchanged.
    """
    import jax
    import jax.numpy as jnp
    import optax

    term = bp.prob.terms[0]
    if not getattr(term, "mc_on", False):
        raise ValueError(
            "batched_map_anneal requires the multiplicity constraint on the "
            "dataset (mc_on); without it there is no term to anneal — drop the "
            "phase instead of running it as an expensive no-op.")
    if not ladder:
        raise ValueError("batched_map_anneal: empty ladder.")

    z_pool = np.asarray(z_pool)
    lp_pool = np.asarray(lp_pool)
    S, N, P = z_pool.shape
    k = min(int(n_particles), N)
    top = np.argsort(-np.nan_to_num(lp_pool, nan=-np.inf), axis=1)[:, :k]
    z = jnp.asarray(np.take_along_axis(z_pool, top[:, :, None], axis=1))

    blk = max(1, min(int(block), k))
    while k % blk:
        blk -= 1
    nb = k // blk

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(float(clip_norm)),
        optax.adam(float(lr)),
    )
    loss_norm = float(bp.prob.loss_normalization)

    for eps, lam, grid_n in ladder:
        term._mc_set_stage(eps, lam, grid_n)
        term.mc_active = True

        def one(row, z0):
            p = bp._swapped(row)

            def loss(zz):
                lp, _ = p.log_prob(zz)
                return -jnp.mean(lp) / loss_norm

            vg = jax.value_and_grad(loss)

            def run_block(zb):
                opt_state = optimizer.init(zb)

                def one_step(carry, _):
                    zz, st = carry
                    _, grads = vg(zz)
                    updates, st = optimizer.update(grads, st)
                    return (optax.apply_updates(zz, updates), st), None

                (zf, _), _ = jax.lax.scan(
                    one_step, (zb, opt_state), None, length=int(num_steps))
                return zf

            return jax.lax.map(run_block, z0.reshape(nb, blk, P)).reshape(k, P)

        z = jax.jit(jax.vmap(one))(bp.data, z)

    def score(row, zz):
        p = bp._swapped(row)
        return jax.lax.map(lambda zb: p.log_prob(zb)[0],
                           zz.reshape(nb, blk, P)).reshape(k)

    lp = np.asarray(jax.jit(jax.vmap(score))(bp.data, z))
    z = np.asarray(z)
    j = np.nanargmax(np.nan_to_num(lp, nan=-np.inf), axis=1)
    return {"z_best": z[np.arange(S), j], "anneal_z": z, "anneal_lp": lp,
            "anneal_best_particle": j.astype(np.int64)}


# --------------------------------------------------------------------------- SVI
def batched_svi(bp, z_best, seeds, *, num_steps: int = 1500, n_vi: int = 500,
                init_scales: float = 1e-3, svi_lr: float = 1e-4
                ) -> Dict[str, np.ndarray]:
    """Gaussian VI for every system at once (solo ``SVIStage`` semantics).

    Returns ``qz_loc (S, P)``, ``qz_scale_tril (S, P, P)``,
    ``svi_loss_hist (S, num_steps)``.
    """
    import jax
    import jax.numpy as jnp
    import optax
    import tensorflow_probability.substrates.jax as tfp
    tfd, tfb = tfp.distributions, tfp.bijectors

    seeds = _check_seeds(seeds, bp.n_systems, "batched_svi")
    z_best = jnp.asarray(z_best)
    S, D = z_best.shape
    if S != bp.n_systems:
        raise ValueError(f"batched_svi: z_best has {S} rows for {bp.n_systems} systems.")

    optimizer = optax.adabelief(float(svi_lr), b1=0.95, b2=0.99)
    if not bp.prob.high_precision:
        raise ValueError(
            "batched_svi: the point-source SBC pipeline is float64-only "
            "(prob.high_precision is False here); a float32 surrogate would "
            "silently diverge from the solo float64 path.")
    cov_bij = tfb.FillScaleTriL(diag_bijector=tfb.Exp(),
                                diag_shift=jnp.asarray(1e-6, jnp.float64))
    scale0 = jnp.eye(D, dtype=jnp.float64) * float(init_scales)
    cov_raw0 = cov_bij.inverse(scale0)     # shared init, same for all systems

    def one(row, z0, seed):
        p = bp._swapped(row)
        params0 = jnp.concatenate([z0, cov_raw0])
        opt_state = optimizer.init(params0)

        def neg_elbo(mean, cov_chol_raw, k):
            qz = tfd.MultivariateNormalTriL(
                loc=mean, scale_tril=cov_bij.forward(cov_chol_raw))
            z = qz.sample(int(n_vi), seed=k)
            return jnp.mean(qz.log_prob(z) - p.log_prob(z)[0])

        vg = jax.value_and_grad(neg_elbo, argnums=(0, 1))

        def one_step(carry, _):
            params, opt_state, key, best_params, best_loss = carry
            # Solo key chain: split parent, split into dev_cnt=1, take dev 0.
            key, curr = jax.random.split(key)
            my_key = jax.random.split(curr, 1)[0]
            loss, (g_mean, g_cov) = vg(params[:D], params[D:], my_key)
            grad = jnp.concatenate([g_mean, g_cov])
            better = loss < best_loss
            best_params = jnp.where(better, params, best_params)
            best_loss = jnp.where(better, loss, best_loss)
            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, key, best_params, best_loss), loss

        init = (params0, opt_state, jax.random.PRNGKey(seed), params0, jnp.inf)
        (_, _, _, best_params, _), loss_hist = jax.lax.scan(
            one_step, init, None, length=int(num_steps))
        return best_params[:D], cov_bij.forward(best_params[D:]), loss_hist

    loc, tril, loss_hist = jax.jit(jax.vmap(one))(bp.data, z_best, seeds)
    return {"qz_loc": np.asarray(loc), "qz_scale_tril": np.asarray(tril),
            "svi_loss_hist": np.asarray(loss_hist)}


# ------------------------------------------------------------------------- MCLMC
def batched_mclmc(bp, qz_loc, qz_scale_tril, seeds, *, n_chains: int = 8,
                  num_burnin_steps: int = 5000, num_results: int = 10000,
                  desired_energy_variance: float = 5e-4,
                  frac_tune1: float = 0.2, frac_tune2: float = 0.6,
                  frac_tune3: float = 0.2,
                  init_L: Optional[float] = None,
                  init_step_size: Optional[float] = None,
                  Lfactor: float = 0.4, num_effective_samples: int = 100,
                  svi_mass_matrix_weight: Optional[float] = None,
                  regularize_mass_matrix: bool = True,
                  thin_every: int = 1) -> Dict[str, np.ndarray]:
    """MCLMC with per-system adaptation for every system at once.

    Faithful non-shard_map port of ``full_mclmc_with_adapt_sharded`` (see the
    module docstring for the fidelity contract), vmapped over systems.

    ``thin_every`` keeps every k-th post-burn-in draw at emission time (memory:
    the full ``(S, C, num_results, P)`` float64 block is ~8 GB at campaign
    defaults with S=100). ``num_results`` must be divisible by it. Downstream
    ESS estimates on thinned chains are per-kept-draw — divide gates
    accordingly or keep ``thin_every=1`` for gate-comparable runs.

    Returns ``samples_z (S, C, num_results//thin_every, P)``, the final
    adaptation state (``step_size (S, C)``, ``L (S,)``,
    ``inverse_mass_matrix (S, P, P)``), and per-system NaN-rejection
    diagnostics over the kept phase (``results_nonan_frac (S,)``).
    """
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from blackjax.adaptation.mclmc_adaptation import (
        MCLMCAdaptationState, handle_nans,
    )
    from blackjax.adaptation.mass_matrix import WelfordAlgorithmState
    from jax.flatten_util import ravel_pytree as _rp  # noqa: F401 (parity import)
    from blackjax.mcmc.integrators import ravel_pytree
    from gigalens.jax.experimental.blackjax_updated_utils import (
        _build_kernel_shardmap, _ess_shardmap, init_multi, welford_combine,
        isokinetic_mclachlan_smart,
    )
    from blackjax.adaptation.mass_matrix import welford_algorithm

    tfd = tfp.distributions
    seeds = _check_seeds(seeds, bp.n_systems, "batched_mclmc")
    qz_loc = jnp.asarray(qz_loc)
    qz_scale_tril = jnp.asarray(qz_scale_tril)
    S, D = qz_loc.shape
    if S != bp.n_systems or qz_scale_tril.shape != (S, D, D):
        raise ValueError(
            f"batched_mclmc: qz_loc {tuple(qz_loc.shape)} / qz_scale_tril "
            f"{tuple(qz_scale_tril.shape)} inconsistent with {bp.n_systems} systems.")

    C = int(n_chains)
    num_burnin_steps, num_results = int(num_burnin_steps), int(num_results)
    thin_every = int(thin_every)
    if thin_every < 1 or num_results % thin_every:
        raise ValueError(
            f"batched_mclmc: thin_every={thin_every} must be >= 1 and divide "
            f"num_results={num_results} (silent remainder-dropping would "
            "misreport the kept-draw count).")
    dev = float(desired_energy_variance)
    weight = float(svi_mass_matrix_weight if svi_mass_matrix_weight is not None
                   else 10.0 * C)
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    # --- tuning schedule (identical arithmetic to the solo driver) ----------
    total_steps = num_burnin_steps + num_results
    s1 = round(num_burnin_steps * frac_tune1)
    s2 = round(num_burnin_steps * frac_tune2)
    s3 = round(num_burnin_steps * frac_tune3)
    tuning_steps = s1 + s2 + s3
    if tuning_steps > num_burnin_steps:
        raise ValueError(
            f"batched_mclmc: frac_tune1+2+3 rounds to {tuning_steps} steps > "
            f"num_burnin_steps={num_burnin_steps}.")
    step_size_sync_step = s1 + s2
    L_adaptation_step = tuning_steps
    l_buffer_start = L_adaptation_step - s3

    mode = np.concatenate([
        np.ones(s1, np.int32),
        2 * np.ones(round(0.67 * s2), np.int32),
        np.ones(round(0.33 * s2), np.int32),
        3 * np.ones(s3, np.int32),
        np.zeros(total_steps - tuning_steps, np.int32),
    ])
    if mode.shape[0] != total_steps:
        raise ValueError("batched_mclmc: internal tuning-schedule length mismatch "
                         f"({mode.shape[0]} != {total_steps}).")

    # Stan-style expanding mass-matrix windows inside the mode-2 region.
    num_mm_steps = round(0.67 * s2)
    ratios = [2 ** k for k in range(3)]
    w_sizes = [max(1, round(num_mm_steps * r / sum(ratios))) for r in ratios]
    w_sizes[-1] = num_mm_steps - sum(w_sizes[:-1])
    _mask = np.zeros(total_steps, bool)
    _pos = s1
    for _ws in w_sizes:
        _pos += _ws
        if 0 <= _pos - 1 < total_steps:
            _mask[_pos - 1] = True
    window_end_mask = jnp.asarray(_mask)

    # Same covariance estimator object the solo driver builds (formula parity).
    _, _, welford_cov = welford_algorithm(is_diagonal_matrix=False)

    def _regularize_cov(cov, n):
        if not regularize_mass_matrix:
            return cov
        cov = 0.5 * (cov + jnp.swapaxes(cov, -1, -2))
        n = jnp.asarray(n, cov.dtype)
        eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
        shrink = 1e-3 * (5.0 / (n + 5.0))
        reg = (n / (n + 5.0)) * cov + shrink * eye
        w, V = jnp.linalg.eigh(reg)
        w = jnp.clip(w, shrink, None)
        return (V * w[..., jnp.newaxis, :]) @ jnp.swapaxes(V, -1, -2)

    init_L_v = float(np.sqrt(D)) if init_L is None else float(init_L)
    init_ss_v = (float(np.sqrt(D)) * 0.25 if init_step_size is None
                 else float(init_step_size))

    def one(row, loc, tril, seed):
        p = bp._swapped(row)
        log_prob = lambda z: p.log_prob(z)[0]
        kernel = lambda imm: _build_kernel_shardmap(
            logdensity_fn=log_prob, integrator=isokinetic_mclachlan_smart,
            inverse_mass_matrix=imm)

        rng_key = jax.random.key(seed)
        init_key, tune_key, run_key = jax.random.split(rng_key, 3)
        del run_key  # solo MCLMC_JIT never consumes it — keep streams aligned

        qz = tfd.MultivariateNormalTriL(loc=loc, scale_tril=tril)
        states = init_multi(qz.sample((C,), seed=init_key), init_key, log_prob)
        _canon = jnp.asarray(states.logdensity).dtype
        cast = lambda a: (jnp.asarray(a).astype(_canon)
                          if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating)
                          else jnp.asarray(a))
        states = jax.tree_util.tree_map(cast, states)
        params = MCLMCAdaptationState(
            L=jnp.asarray(init_L_v, _canon),
            step_size=jnp.asarray(init_ss_v, _canon),
            inverse_mass_matrix=qz.covariance().astype(_canon))

        welford_state = WelfordAlgorithmState(
            qz.mean().astype(_canon),
            (qz.covariance() * weight).astype(_canon),
            jnp.asarray(weight, _canon))
        welford_empty = WelfordAlgorithmState(
            jnp.zeros(D, _canon), jnp.zeros((D, D), _canon),
            jnp.asarray(0.0, _canon))

        adapt_single = (jnp.asarray(0.0, _canon), jnp.asarray(0.0, _canon),
                        jnp.asarray(jnp.inf, _canon))
        tile = lambda x: jnp.broadcast_to(x[jnp.newaxis], (C,) + x.shape)
        adapt_states = jax.tree.map(tile, adapt_single)
        step_sizes = jnp.full((C,), params.step_size, _canon)
        l_bufs = jnp.zeros((C, s3, D), _canon)

        keys = jax.random.split(tune_key, (C, total_steps))
        keys = jnp.moveaxis(keys, 0, 1)              # (total_steps, C, ...)

        def step_size_adapt(previous_state, next_state, info, prm, adaptive_state,
                            nan_key):
            time_, x_average, step_size_max = adaptive_state
            success, state, step_size_max, energy_change = handle_nans(
                previous_state, next_state, prm.step_size, step_size_max,
                info.energy_change, nan_key)
            xi = jnp.square(energy_change) / (D * dev) + 1e-8
            w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * 1.5)))
            x_average = decay_rate * x_average + w * (xi / jnp.power(prm.step_size, 6.0))
            time_ = decay_rate * time_ + w
            step_size = jnp.power(x_average / time_, -1.0 / 6.0)
            step_size = jnp.minimum(step_size, step_size_max)
            return state, prm._replace(step_size=step_size), \
                (time_, x_average, step_size_max), success, xi

        def _make_adapt_reset(cur):
            return (jnp.zeros_like(cur[0]), jnp.zeros_like(cur[1]), cur[2])

        _sel = lambda c, a, b: jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)

        def step(carry, xs):
            i, mode_i, keys_c = xs
            states, params, step_sizes, adapt_states, welford_state, l_bufs = carry
            do_ssa = jnp.logical_or(mode_i == 1, mode_i == 2)
            do_mm = mode_i == 2

            pairs = jax.vmap(jax.random.split)(keys_c)
            chain_keys, nan_keys = pairs[:, 0], pairs[:, 1]
            kernel_fn = kernel(params.inverse_mass_matrix)

            def per_chain(prev_state, rk, nk, ss, ast):
                new_state, info = kernel_fn(rng_key=rk, state=prev_state,
                                            L=params.L, step_size=ss)

                def adapt_one(_):
                    pseudo = params._replace(step_size=ss)
                    a_state, a_params, a_adapt, a_success, a_xi = step_size_adapt(
                        prev_state, new_state, info, pseudo, ast, nk)
                    return a_state, a_params.step_size, a_adapt, a_success, a_xi

                def skip_adapt(_):
                    ok = jnp.isfinite(new_state.position.reshape(-1)[0])
                    xi = (jnp.square(info.energy_change) / (D * dev) + 1e-8
                          ).astype(ss.dtype)
                    return new_state, ss, ast, ok, xi

                return jax.lax.cond(do_ssa, adapt_one, skip_adapt, operand=None)

            new_states, new_ss, new_ast, successes, xis = jax.vmap(per_chain)(
                states, chain_keys, nan_keys, step_sizes, adapt_states)

            l_bufs = jax.lax.cond(
                mode_i == 3,
                lambda buf: buf.at[:, i - l_buffer_start].set(new_states.position),
                lambda buf: buf, l_bufs)

            def run_mm(_):
                xs_pos = jax.vmap(lambda s: ravel_pytree(s.position)[0])(new_states)
                x_mean = jnp.mean(xs_pos, axis=0)
                deltas = xs_pos - x_mean[jnp.newaxis, :]
                m2 = jnp.einsum("ci,cj->ij", deltas, deltas)
                update = WelfordAlgorithmState(x_mean, m2, jnp.asarray(float(C), _canon))
                new_welford = welford_combine(welford_state, update)
                updated_welford = _sel(do_mm, new_welford, welford_state)
                at_boundary = window_end_mask[i]
                update_mm = jnp.logical_and(do_mm, at_boundary)
                sample_cov = _regularize_cov(
                    welford_cov(updated_welford)[0], updated_welford.sample_size)
                mm_params = params._replace(inverse_mass_matrix=sample_cov)
                updated_params = _sel(update_mm, mm_params, params)
                updated_welford = _sel(update_mm, welford_empty, updated_welford)
                updated_ast = _sel(update_mm, _make_adapt_reset(new_ast), new_ast)
                return updated_params, updated_welford, updated_ast

            params, welford_state, new_ast = jax.lax.cond(
                do_mm, run_mm,
                lambda _: (params, welford_state, new_ast), operand=None)

            synced_ss = jnp.mean(new_ss)
            new_ss = jnp.where(i == step_size_sync_step,
                               jnp.full_like(new_ss, synced_ss), new_ss)

            def calc_new_L(_):
                per_chain_ess = jax.vmap(lambda buf: _ess_shardmap(
                    buf[jnp.newaxis, :, :], chain_axis=0, sample_axis=1))(l_bufs)
                return Lfactor * s3 * synced_ss / jnp.min(per_chain_ess)

            new_L = jax.lax.cond(i == L_adaptation_step, calc_new_L,
                                 lambda _: params.L, operand=None)
            params = params._replace(L=new_L)

            carry = (new_states, params, new_ss, new_ast, welford_state, l_bufs)
            return carry, successes

        # --- burn-in: no position storage ---------------------------------
        carry0 = (states, params, step_sizes, adapt_states, welford_state, l_bufs)
        xs_burn = (jnp.arange(num_burnin_steps, dtype=jnp.int32),
                   jnp.asarray(mode[:num_burnin_steps]), keys[:num_burnin_steps])
        carry1, _ = jax.lax.scan(step, carry0, xs_burn)

        # --- results: emit the position every `thin_every` steps ----------
        n_kept = num_results // thin_every
        xs_res = (jnp.arange(num_burnin_steps, total_steps, dtype=jnp.int32),
                  jnp.asarray(mode[num_burnin_steps:]), keys[num_burnin_steps:])
        xs_res = jax.tree.map(
            lambda a: a.reshape((n_kept, thin_every) + a.shape[1:]), xs_res)

        def res_chunk(carry, xs_chunk):
            carry, succ = jax.lax.scan(step, carry, xs_chunk)
            states_c = carry[0]
            return carry, (states_c.position, jnp.mean(succ))

        carry2, (positions, nonan) = jax.lax.scan(res_chunk, carry1, xs_res)
        _, params_final, ss_final, _, _, _ = carry2
        samples = jnp.moveaxis(positions, 0, 1)      # (C, n_kept, D)
        return samples, ss_final, params_final.L, \
            params_final.inverse_mass_matrix, jnp.mean(nonan)

    samples, ss_final, L_final, imm_final, nonan = jax.jit(
        jax.vmap(one))(bp.data, qz_loc, qz_scale_tril, seeds)
    return {"samples_z": np.asarray(samples),
            "final_step_size": np.asarray(ss_final),
            "final_L": np.asarray(L_final),
            "final_inverse_mass_matrix": np.asarray(imm_final),
            "results_nonan_frac": np.asarray(nonan)}


# ---------------------------------------------------------------- full pipeline
def batched_map_svi_mclmc(bp, *, map_seeds, svi_seeds, mclmc_seeds,
                          **kwargs: Any) -> Dict[str, np.ndarray]:
    """MAP -> SVI -> MCLMC for every system, one call.

    Accepts the same knob names as the solo ``map_svi_mclmc`` pipeline builder
    (``map_num_steps``, ``map_n_samples``, ``map_lr``, ``map_clip_norm``,
    ``svi_num_steps``, ``svi_n_vi``, ``svi_init_scale``, ``svi_lr``,
    ``n_chains``, ``num_burnin_steps``, ``num_results``,
    ``desired_energy_variance``, ``frac_tune1/2/3``) plus ``thin_every``.
    Unknown knobs raise — a typo'd knob must never silently fall back.
    """
    known = {"map_num_steps", "map_n_samples", "map_lr", "map_clip_norm",
             "svi_num_steps", "svi_n_vi", "svi_init_scale", "svi_lr",
             "n_chains", "num_burnin_steps", "num_results",
             "desired_energy_variance", "frac_tune1", "frac_tune2",
             "frac_tune3", "thin_every",
             "mc_anneal_eps", "mc_anneal_steps", "mc_anneal_particles",
             "mc_anneal_lr", "mc_anneal_block"}
    unknown = set(kwargs) - known
    if unknown:
        raise ValueError(f"batched_map_svi_mclmc: unknown knobs {sorted(unknown)}; "
                         f"known: {sorted(known)}.")

    # Multiplicity-constraint staging (P-9 design): the term defines the MEASURED
    # target (MCLMC, and any later log_like scoring) but is disabled for the bulk
    # 1000-particle MAP and for SVI — SVI only fits the local Gaussian
    # preconditioner, and in the correct basin the term is inert (N_eff ~ n_obs,
    # penalty ~ 0), so skipping it there changes efficiency, not correctness.
    term = bp.prob.terms[0]
    mc_on = bool(getattr(term, "mc_on", False))
    if mc_on:
        ds = term.dataset
        mc_final = (float(ds.mc_eps), float(ds.mc_lam), int(ds.mc_grid_n))
    for key in ("mc_anneal_eps", "mc_anneal_steps", "mc_anneal_particles",
                "mc_anneal_lr", "mc_anneal_block"):
        if key in kwargs and not mc_on:
            raise ValueError(
                f"batched_map_svi_mclmc: {key} was given but the dataset has no "
                "multiplicity constraint — the knob would be silently dead.")

    try:
        if mc_on:
            term.mc_active = False
        out = batched_map(
            bp, map_seeds,
            num_steps=int(kwargs.get("map_num_steps", 1500)),
            n_samples=int(kwargs.get("map_n_samples", 1000)),
            map_lr=float(kwargs.get("map_lr", 3e-3)),
            map_clip_norm=float(kwargs.get("map_clip_norm", 1.0)))

        if mc_on:
            eps_f, lam_f, grid_f = mc_final
            coarse = [float(e) for e in
                      kwargs.get("mc_anneal_eps", (0.3, 0.15, 0.075))
                      if float(e) > eps_f * 1.05]
            epss = sorted(coarse, reverse=True) + [eps_f]
            ladder = []
            for i, e in enumerate(epss):
                if e == eps_f:
                    g = grid_f
                else:
                    # step <= eps sqrt(mu_min): coarse rungs may shrink the grid
                    # proportionally to eps (multiple of 4 keeps 16 tiles even).
                    g = min(grid_f, max(64, -(-int(grid_f * eps_f / e) // 4) * 4))
                ladder.append((e, lam_f * (i + 1) / len(epss), g))
            ladder[-1] = (eps_f, lam_f, grid_f)
            ann = batched_map_anneal(
                bp, out["z_final"], out["lp_final"], ladder=ladder,
                n_particles=int(kwargs.get("mc_anneal_particles", 64)),
                num_steps=int(kwargs.get("mc_anneal_steps", 300)),
                lr=float(kwargs.get("mc_anneal_lr", 1e-3)),
                block=int(kwargs.get("mc_anneal_block", 16)))
            out["z_best_premc"] = out["z_best"]
            out["z_best"] = ann["z_best"]
            out["mc_anneal_lp"] = ann["anneal_lp"]
            out["mc_anneal_best_particle"] = ann["anneal_best_particle"]
            out["mc_anneal_ladder"] = np.asarray(ladder)
            term.mc_active = False       # SVI: preconditioner only

        svi = batched_svi(
            bp, out["z_best"], svi_seeds,
            num_steps=int(kwargs.get("svi_num_steps", 1500)),
            n_vi=int(kwargs.get("svi_n_vi", 500)),
            init_scales=float(kwargs.get("svi_init_scale", 1e-3)),
            svi_lr=float(kwargs.get("svi_lr", 1e-4)))
        out.update(svi)

        if mc_on:
            term._mc_set_stage(*mc_final)
            term.mc_active = True        # MCLMC samples the FULL target
        mclmc = batched_mclmc(
            bp, svi["qz_loc"], svi["qz_scale_tril"], mclmc_seeds,
            n_chains=int(kwargs.get("n_chains", 8)),
            num_burnin_steps=int(kwargs.get("num_burnin_steps", 5000)),
            num_results=int(kwargs.get("num_results", 10000)),
            desired_energy_variance=float(kwargs.get("desired_energy_variance", 5e-4)),
            frac_tune1=float(kwargs.get("frac_tune1", 0.2)),
            frac_tune2=float(kwargs.get("frac_tune2", 0.6)),
            frac_tune3=float(kwargs.get("frac_tune3", 0.2)),
            thin_every=int(kwargs.get("thin_every", 1)))
        out.update(mclmc)
    finally:
        if mc_on:
            # The template term is SHARED with system 0's solo ProbModel: always
            # restore the full measured target for downstream log_like scoring.
            term._mc_set_stage(*mc_final)
            term.mc_active = True
    return out
