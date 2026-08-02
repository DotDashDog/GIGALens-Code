r"""LAPS — Late-Adjusted Parallel Sampler (Robnik & Seljak 2026), GIGALens driver.

Paper-faithful two-phase, sharded LAPS sampler composing the in-tree, already
hardened MCLMC building blocks with the validated ensemble reductions in
``laps_core.py``. Canonical spec: ``docs/logs/laps_spec.md``; build plan:
``docs/logs/laps_gigalens_translation.md``.

Public entries
--------------
``LAPS_late_adjusted(logdensity_fn, qz, ...)``
    Core driver. Mirrors the MCLMC gigalens interface: takes the gigalens
    ``log_prob(z)`` of an unconstrained ``z`` (shape ``(dim,)``) plus a chain
    initializer (``qz`` exposing ``.sample/.mean/.covariance`` for the warm
    start, or ``dim`` for the cold start) and the LAPS hyperparameters.
``LAPS_late_adjusted_JIT(prob_model, qz, ...)``
    Thin gigalens wrapper, mirrors ``MCLMC_JIT``: builds
    ``log_prob(z) = prob_model.log_prob(z)[0]`` and calls the core.

The two phases (spec §A)
------------------------
Phase 1 (unadjusted MCLMC): isotropic (identity) mass matrix; isokinetic
Leapfrog (velocity-Verlet, 1 grad/step). Each step, from the cross-chain
ensemble, update ``L = alpha*sqrt(sum Var[x_i])`` (alpha=2, Eq. 9), the
equipartition divergence ``D-tilde`` (Eqs. 4/6/18), the ensemble EEVPD
(Eq. 7), and the step size via ``eps <- eps*(F(C*D-tilde)/EEVPD_obs)^{1/6}``
(C=0.025; ``schedule="emaus"`` selects the predecessor ``C*D-tilde^{3/8}``,
C=0.1). A windowed ``x_i^2`` relative-fluctuation detector (Eqs. 10-11) decides
the Phase-1 -> Phase-2 switch; it is NOT allowed to fire before a full window
``T = save_frac*num_unadjusted_steps`` has accumulated. The fire criterion is
``switch_mode`` (DEFAULT ``"self_calibrated"``: fire when ``max_i delta_i`` is
within ``switch_k`` (1.5) of the per-dim equilibrium noise floor
``sqrt(Var_rho[f]/M)/E_rho[f]``, computed ONLINE from the current ensemble).
The literal-paper ``"absolute"`` (``max_i delta_i < 0.01``) is unreachable below
``sqrt(2/M)`` (0.063 at M=512) and RAISES at small M rather than silently burning
the budget; ``"m_scaled"`` uses ``k_m/sqrt(M)``. Phase 1 stops when the switch
fires OR the ``num_unadjusted_steps`` budget is hit. The observable flag
(``switch``: paper x_i^2 vs emaus x_i) is orthogonal to ``switch_mode``.

Phase boundary: diagonal preconditioner ``inverse_mass_matrix = diag(Var[x_i])``
from the SECOND HALF of Phase-1 positions (pooled over chains and the
second-half steps); integrator switches to MN2 (mclachlan) if d<=200 else MN4
(omelyan); kernel switches to the adjusted MAMS kernel with N=15 steps/traj.

Phase 2 (MAMS adjusted): STAGED per-chunk bracketing-bisection tune of the step
size to the target acceptance (0.7 for MN2, 0.9 for MN4). The Phase-2 step is
INITIALIZED from the trajectory scale (FIX A): ``eps0 = L / N`` (the latest
Phase-1 decoherence length ``L``, Eq. 9, over ``N = steps_per_trajectory``), so
the adjusted phase starts in the right ballpark instead of the tiny sub-EEVPD
Phase-1 handoff step (which pinned acceptance ~1.0); overridable via
``p2_eps_init``. The step is HELD CONSTANT within each Phase-2 chunk (the device
scan never mutates eps); between chunks the host computes the chunk's SETTLED
acceptance (the mean over the LATTER HALF of the chunk's per-step ensemble
accepts, dropping the within-chunk/post-switch transient) and calls
``laps_core.bisection_step`` ONCE to update ``(eps, lo, hi)`` for the NEXT chunk
-- exactly mirroring how Phase 1 makes host-side decisions between fixed-length
scan chunks. The update is a true two-phase BRACKET-then-bisect (FIX B): while the
target acceptance is not yet straddled it EXPANDS eps geometrically by ``p2_growth``
in the correct direction (a>target => step too small => grow eps UP; a<target =>
shrink), then once bracketed it bisects at the geometric midpoint. eps freezes
(one-way latch, bracket frozen) once the settled acceptance is in-band
(``|a_settled - a_target| <= tol``) for ``p2_freeze_persist`` CONSECUTIVE chunks;
remaining chunks sample at the frozen eps. A WIDE safety-rail clamp
(``eps_min``/``eps_max``, 3 decades around ``eps0 = L/N``) bounds the step so the
bracketing walk cannot run away while leaving it ample room.

CHANGE A (reach the target within budget): Phase 2 has its OWN chunk size
``p2_chunk_size`` (default 8, smaller than Phase-1's ``chunk_size``) so the fixed
``num_adjusted_steps`` adaptation budget yields MANY bisection updates (~25 chunks
at the defaults), and the bracketing expansion factor ``p2_growth`` is raised to
2.5 so the flat ``accept~1.0`` plateau (which can span ~10x in eps from a
too-small handoff) is crossed in a few chunks, leaving budget to refine + latch.

CHANGE B (post-freeze multi-sample collection): after the freeze, a COLLECTION
sub-phase runs at the FROZEN hyperparameters and keeps a thinned time series per
chain -- ``p2_keep_per_chain`` samples, one every ``p2_thin`` frozen steps (valid
MCMC at fixed hyperparameters). ``res.samples`` is ``(M, p2_keep_per_chain, dim)``
so ``.reshape((-1, dim))`` flattens to ``(M*keep, dim)``; ``keep=1`` reproduces the
old one-sample-per-chain final ensemble (Eq. 3) with 0 extra steps.
``res.n_samples_total = M*p2_keep_per_chain``.

WHY STAGED (regression fix): a prior design drove the bracket from a per-step
ROLLING-WINDOW ensemble acceptance. On the 22-D lens the post-switch transient
made that window lag near 1.0, so the bracket kept doubling eps, the step ran
away, and acceptance collapsed to 0 for the rest of Phase 2 (never froze). The
bracket MUST be updated from a SETTLED (equilibrium) acceptance at a HELD step,
not an instantaneous or lagging-rolling signal -- hence the per-chunk staging.

Control-flow choice (REQUIRED documentation, spec/design open-Q1)
----------------------------------------------------------------
The Phase-1 ``delta`` switch and the Phase-2 bisection are DATA-DEPENDENT stops.
A ``while_loop`` with data-dependent trip count is VMA-forbidden under
``shard_map`` (the same landmine ``mclmc.py`` avoids: it uses a single
FIXED-length ``lax.scan`` over a precomputed ``mode`` array, never a
data-dependent loop). We adopt the design doc's option (i): a **Python-side
outer loop over fixed-size jitted ``shard_map``+``scan`` chunks**, evaluating
the scalar switch criterion on the HOST between chunks. This keeps every traced
region a fixed-length scan (compiled once, reused across chunks), lets us stop
Phase 1 early (closer to the paper's ``while``), and keeps the data-dependent
decision off the device. The cost is switch-detection granularity of one chunk
(``chunk_size`` steps); ``switch_index`` records the step at which the trailing
window first satisfied the threshold. The Phase-2 bisection is likewise a
host-side between-chunks decision (``laps_core.bisection_step`` called ONCE per
chunk on the chunk's settled acceptance): the device scan runs at a FIXED eps and
only records the per-step ensemble acceptance, so no data-dependent control lives
inside the traced region. eps simply freezes (one-way latch) and the remaining
chunks sample at the frozen step size.

Spec deviations (honest reporting)
----------------------------------
* The in-tree ``_build_adjusted_kernel_shardmap`` is the *Hamiltonian* MAMS
  variant (full momentum refresh once per trajectory, NO intra-trajectory
  partial refresh — equivalent to ``L_proposal_factor = inf``). The spec's
  ``L_proposal = 1.25*L_full`` partial-refresh is therefore NOT applied by this
  kernel; ``L_full = N*eps`` and ``L_proposal`` are recorded in diagnostics but
  do not influence Phase 2. Composing the validated kernel was preferred over
  reimplementing partial refresh (task instruction).
* Velocity init: by default ``_single_init`` draws a RANDOM unit momentum; the
  paper (and blackjax ``laps_burn_in.initialize``) align the initial velocity
  with the gradient, sign-flipped per coordinate by the ensemble equipartition
  (overdispersed coordinate -> velocity toward the mode). The reference
  mechanism is available via ``velocity_init="gradient"`` (cold-start lever,
  stocktake F1); the default stays "random" pending its empirical evaluation.
  Its bj companion — first-step L=inf, so step 1 applies NO Maruyama refresh
  and the aligned velocity survives one full step — is ``L0_inf=True``
  (default off = paper Eq. 9 L0, the documented paper-over-bj choice).
* NaN -> step-halving: blackjax halves the Phase-1 step whenever ANY chain
  NaN-rejects (``laps_burn_in.py:333-335``); our default controller instead
  zeroes the rejected chains' energy change only (which biases EEVPD_obs LOW,
  i.e. GROWS the step). The reference brake is available via
  ``nan_eps_halving=True`` (stocktake F2). Default off = historical behaviour.
* Phase-boundary preconditioner: default pools Var[x_i] over the SECOND HALF of
  Phase-1 steps; both references use the END-STATE (final-step instantaneous)
  ensemble Var. ``precond_source="final"`` selects the reference behaviour
  (stocktake F3). Default stays "pooled_half" = historical behaviour.
* ``D-tilde`` uses the diagonal estimator (Eq. 18); the full-rank Hutchinson
  estimator (App. B.1) is not implemented (App. D: diagonal ~ full-rank).
"""

import functools
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

try:
    _shard_map = jax.shard_map  # type: ignore[attr-defined]
except AttributeError:  # older JAX
    from jax.experimental.shard_map import shard_map as _shard_map

from .blackjax_updated_utils import (
    _build_kernel_shardmap,
    _build_adjusted_kernel_shardmap,
    isokinetic_velocity_verlet_smart,
    isokinetic_mclachlan_smart,
    isokinetic_omelyan_smart,
    init_multi,
)
from . import laps_core


# Diagnostics bundle. Every field is exposed so an external grader can check
# each internal against the spec (task: "Expose enough that an external grader
# can check each internal against the spec").
LAPSResults = namedtuple(
    "LAPSResults",
    [
        "samples",            # (num_chains, p2_keep_per_chain, dim) post-freeze
                              #   thinned samples per chain (.reshape((-1, dim))
                              #   flattens to (M*keep, dim); keep=1 -> the final
                              #   ensemble, one sample/chain, as before)
        "n_samples_total",    # int: M * p2_keep_per_chain (total posterior draws)
        # ---- Phase 1 histories (length = phase1_len) ----
        "p1_D_tilde",         # (T1,) equipartition divergence per step
        "p1_eevpd_wanted",    # (T1,) target EEVPD = F(C*D-tilde) (or emaus law)
        "p1_eevpd_obs",       # (T1,) observed ensemble EEVPD
        "p1_step_size",       # (T1,) step size per step
        "p1_L",               # (T1,) decoherence length per step
        "p1_obs_sq",          # (T1, dim) ensemble E[x_i^2] (paper switch obs)
        "p1_obs_mean",        # (T1, dim) ensemble E[x_i]   (emaus switch obs)
        "p1_delta_max",       # (T1,) trailing-window max delta (NaN before window)
        "phase1_len",         # int: steps actually run in Phase 1
        "switch_index",       # int: step the ACTIVE switch fired (or phase1_len)
        "switch_index_paper", # int: post-hoc x_i^2 switch step (or T1 = never)
        "switch_index_emaus", # int: post-hoc x_i   switch step (or T1 = never)
        "switched",           # bool: did the active switch fire before budget
        # ---- phase boundary ----
        "precond_var",        # (dim,) diag preconditioner Var[x_i] (2nd half)
        "integrator_order",   # 2 (MN2/mclachlan) or 4 (MN4/omelyan)
        "target_accept",      # 0.7 or 0.9
        # ---- Phase 2 histories (length = num_adjusted_steps) ----
        "p2_accept",          # (T2,) ensemble-mean acceptance per step
        "p2_step_size",       # (T2,) step size per step (held within each chunk)
        "p2_frozen",          # (T2,) bool frozen flag per step
        "p2_settled_accept",  # (n_chunks,) per-chunk settled accept (latter-half mean)
        "p2_final_step_size", # float: final (frozen) step size
        "p2_L_full",          # float: N*eps_final (recorded; see deviations)
        "p2_L_proposal",      # float: 1.25*L_full (recorded; NOT applied)
        "p1_nan_frac",        # (T1,) fraction of chains NaN-rejected per step
                              #   (diagnostic stream, mirrors bj rejection_rate_nans)
        "chain_traj",         # None, or (track_chains=True) dict with per-chunk
                              #   per-chain snapshots: p1_pos (n_ck1, M, d),
                              #   p1_logp (n_ck1, M), p2_pos, p2_logp
        "resample_info",      # None, or (p2_resample_at_chunk set) dict: chunk,
                              #   skipped, n_survivors/n_stragglers, cut, dlogp,
                              #   stragglers/donors index arrays, eps0_rs, rs_var,
                              #   snap_index (post-resample chain_traj p2 index)
    ],
)


def _canon_dtype(state):
    return jnp.asarray(state.logdensity).dtype


def _gradient_equipartition_momentum(position, momentum, grad, canon):
    """F1 lever math (bj ``laps_burn_in.initialize``, lines 77-156).

    Each chain's velocity = grad/|grad| (unit norm), then ONE shared
    per-coordinate sign from the ensemble's UNCENTERED equipartition
    E_ii = E_rho[-x_i g_i] (bj's init-time convention, NOT the centered Eq. 4):
    +1 where E_ii >= 1 (overdispersed -> move toward the mode), -1 where < 1.
    The flip preserves unit norm. Safety rail beyond bj: rows with a non-finite
    gradient keep the supplied (random) momentum and are excluded from E_ii
    (no-op when, as measured on the lens, 0% of prior draws have non-finite
    grads).
    """
    gnorm = jnp.linalg.norm(grad, axis=-1, keepdims=True)
    v = grad / jnp.maximum(gnorm, jnp.asarray(1e-30, canon))
    row_ok = jnp.all(jnp.isfinite(grad), axis=-1, keepdims=True)
    n_ok = jnp.maximum(jnp.sum(row_ok.astype(canon)), jnp.asarray(1.0, canon))
    E_ii = jnp.sum(jnp.where(row_ok, -position * grad, 0.0), axis=0) / n_ok
    signs = jnp.where(E_ii < 1.0, -1.0, 1.0).astype(canon)
    return jnp.where(row_ok, v * signs, momentum).astype(canon)


def _traj_pack(traj):
    """Stack per-chunk chain snapshots into arrays (None passthrough)."""
    if traj is None:
        return None
    return {k: (np.stack(v, axis=0) if v else None) for k, v in traj.items()}


def _chunk_sizes(total, chunk):
    """Split ``total`` steps into chunks of at most ``chunk`` (final chunk short).

    Robustness (grader #5): a budget not divisible by ``chunk`` keeps its
    remainder as a final short chunk (never dropped); a budget smaller than
    ``chunk`` becomes a single chunk of size ``total`` (never ``n_chunks=0``).
    The jitted scan length = ``keys.shape[0]`` is dynamic, so a short final chunk
    simply triggers one extra compile -- correct, not a crash.
    """
    chunk = max(1, int(chunk))
    sizes = []
    done = 0
    while done < total:
        sizes.append(min(chunk, total - done))
        done += sizes[-1]
    return sizes


def _switch_index_host(obs_hist, switch, threshold, T,
                       switch_mode="absolute", k=1.5, k_m=2.0,
                       floor_hist=None, M=None):
    """Host-side: first step at which the trailing-T window fires (or len = never).

    Mirrors the spec switch exactly via ``laps_core.phase1_switch`` on each
    trailing window. Eligibility honored: a window is only evaluated once T
    steps have accumulated (``t >= T-1``), so a warm start cannot fire on step 0.
    ``switch_mode`` selects the fire criterion; ``floor_hist`` (per-step (n, d)
    noise floors) supplies the current-ensemble floor for self_calibrated.
    """
    n = obs_hist.shape[0]
    if n < T:
        return n
    for t in range(T - 1, n):
        window = jnp.asarray(obs_hist[t - T + 1 : t + 1])
        floor = None if floor_hist is None else floor_hist[t]
        _, _, fired = laps_core.phase1_switch(
            window, switch=switch, threshold=threshold,
            switch_mode=switch_mode, k=k, k_m=k_m, floor=floor, M=M)
        if bool(fired):
            return t
    return n


def LAPS_late_adjusted(
    logdensity_fn,
    qz=None,
    *,
    dim=None,
    num_chains=512,
    num_unadjusted_steps=300,
    num_adjusted_steps=200,
    chunk_size=25,
    init_mode="warm",          # "warm" (qz surrogate) | "cold" (N(0,I))
    init_positions=None,       # (num_chains, dim) unconstrained: overrides warm/cold
    schedule="paper",          # step law: "paper" F(C*D) | "emaus" C*D^{3/8}
    switch="paper",            # switch obs: "paper" x_i^2 | "emaus" x_i
    switch_mode="self_calibrated",  # fire rule: self_calibrated | absolute | m_scaled
    switch_k=1.5,              # self_calibrated: fire when delta < k*floor
    switch_k_m=2.0,            # m_scaled: threshold = k_m / sqrt(M)
    switch_persist=1,          # OPTIONAL guard: require N consecutive ripe chunks
    phase2_enabled=True,       # False -> return the unadjusted Phase-1 ensemble
    alpha=2.0,
    C=None,
    switch_threshold=0.01,
    save_frac=0.2,
    early_stop=True,
    steps_per_trajectory=15,    # N
    L_proposal_factor=1.25,
    bisection_tol=0.03,
    p2_eps_init=None,           # FIX A: Phase-2 eps0 (None -> L/N from trajectory L)
    p2_chunk_size=8,            # CHANGE A: Phase-2 chunk size (own, < Phase-1 chunk)
    p2_growth=2.5,              # FIX B/CHANGE A: geometric bracketing factor (>1)
    p2_freeze_persist=2,        # STAGED: freeze after N consecutive in-band CHUNKS
    p2_min_adapt_chunks=0,      # STAGED: no freeze before this many Phase-2 chunks
    p2_keep_per_chain=1,        # CHANGE B: post-freeze samples kept per chain
    p2_thin=5,                  # CHANGE B: frozen steps between kept samples
    p2_precond_var=None,        # DIAGNOSTIC: override Phase-2 diag preconditioner
                                # (dim,) with a known metric (e.g. qz var); None=from ensemble
    velocity_init="random",     # F1 lever: "random" (baseline) | "gradient"
                                # (bj-faithful grad-aligned + equipartition sign flip)
    nan_eps_halving=False,      # F2 lever: bj-faithful eps*0.5 when any chain NaN-rejects
    precond_source="pooled_half",  # F3 lever: "pooled_half" (baseline) | "final"
                                # (bj/paper end-state ensemble Var)
    L0_inf=False,               # F1-companion lever: bj-faithful FIRST-step L=inf
                                # (laps_burn_in.py:261: no Maruyama refresh on step 1,
                                # protecting a gradient-aligned init velocity; L is
                                # then re-set from the ensemble as usual)
    track_chains=False,         # DIAGNOSTIC: record per-chain (position, logdensity)
                                # at every ADAPTATION chunk boundary (both phases;
                                # p2_keep_per_chain>1 collection sub-chunks are NOT
                                # snapshotted) into results.chain_traj; default off
    p2_resample_at_chunk=None,  # RESAMPLE lever (DC-7.3): after this many Phase-2
                                # adaptation chunks, replace straggler STATES with
                                # uniform draws from survivor states (survivor iff
                                # logp > max - p2_resample_dlogp), REBUILD the diag
                                # metric from the resampled ensemble, and RESTART the
                                # eps bisection (eps0 = L/N of resampled ensemble).
                                # Pre-resample states are burn-in (never averaged);
                                # validity = MH re-initialization, the warm-init
                                # class. None = off (baseline).
    p2_resample_dlogp=None,     # survivor cut below ensemble max logp
                                # (None -> d/2 + 4*sqrt(d/2), ~7 posterior logp
                                # spreads: generous, misclassification only affects
                                # which INIT a chain restarts from)
    p2_resample_min_survivors=32,  # guard: skip resampling (warn) below this pool
    p2_resample_mode="replace",    # "replace" = move straggler states to survivor
                                   # draws; "retune_only" = KEEP all positions, only
                                   # rebuild metric+eps from the SURVIVOR subset and
                                   # restart bisection (control: isolates the
                                   # poisoned-metric fix from position replacement =
                                   # the pre-committed "retuned adjusted phase")
    cold_scale=1.0,
    seed=0,
):
    """Run the two-phase LAPS sampler. See module docstring for the algorithm.

    Returns a ``LAPSResults`` namedtuple of samples + per-phase diagnostics.
    """
    if init_positions is None and init_mode not in ("warm", "cold"):
        raise ValueError(f"init_mode must be 'warm' or 'cold', got {init_mode!r}")
    if init_positions is None and qz is None and dim is None:
        raise ValueError(
            "Provide qz (warm/cold dim), an explicit dim, or init_positions.")
    if int(p2_freeze_persist) < 1:
        raise ValueError(f"p2_freeze_persist must be >= 1, got {p2_freeze_persist}.")
    if int(p2_min_adapt_chunks) < 0:
        raise ValueError(
            f"p2_min_adapt_chunks must be >= 0, got {p2_min_adapt_chunks}.")
    if int(p2_chunk_size) < 1:
        raise ValueError(f"p2_chunk_size must be >= 1, got {p2_chunk_size}.")
    if int(p2_keep_per_chain) < 1:
        raise ValueError(f"p2_keep_per_chain must be >= 1, got {p2_keep_per_chain}.")
    if int(p2_thin) < 1:
        raise ValueError(f"p2_thin must be >= 1, got {p2_thin}.")
    if float(p2_growth) <= 1.0:
        raise ValueError(f"p2_growth must be > 1, got {p2_growth}.")
    if switch_mode not in ("self_calibrated", "absolute", "m_scaled"):
        raise ValueError(
            "switch_mode must be 'self_calibrated'|'absolute'|'m_scaled', "
            f"got {switch_mode!r}")
    if num_unadjusted_steps < 1 or num_adjusted_steps < 1:
        raise ValueError(
            "num_unadjusted_steps and num_adjusted_steps must each be >= 1 "
            f"(got {num_unadjusted_steps}, {num_adjusted_steps}).")
    if int(switch_persist) < 1:
        raise ValueError(f"switch_persist must be >= 1, got {switch_persist}.")
    if velocity_init not in ("random", "gradient"):
        raise ValueError(
            f"velocity_init must be 'random' or 'gradient', got {velocity_init!r}")
    if p2_resample_at_chunk is not None and int(p2_resample_at_chunk) < 1:
        raise ValueError(
            f"p2_resample_at_chunk must be >= 1 or None, got {p2_resample_at_chunk}")
    if int(p2_resample_min_survivors) < 2:
        raise ValueError("p2_resample_min_survivors must be >= 2")
    if p2_resample_mode not in ("replace", "retune_only"):
        raise ValueError(
            f"p2_resample_mode must be 'replace' or 'retune_only', "
            f"got {p2_resample_mode!r}")
    if precond_source not in ("pooled_half", "final"):
        raise ValueError(
            f"precond_source must be 'pooled_half' or 'final', got {precond_source!r}")

    num_devices = len(jax.devices())
    num_chains = (num_chains // num_devices) * num_devices
    if num_chains == 0:
        raise ValueError(f"num_chains must be >= num_devices ({num_devices})")
    cpd = num_chains // num_devices

    # GUARD (spec switch-resolution): the literal absolute threshold is unreachable
    # below the equilibrium noise floor sqrt(2/M); fire it loudly rather than burn
    # maxiter on a silent no-op (project-standards: no silent scientific default).
    if switch_mode == "absolute":
        floor_sqrt2M = float(np.sqrt(2.0 / num_chains))
        if float(switch_threshold) < floor_sqrt2M:
            raise ValueError(
                f"switch_mode='absolute' with threshold={switch_threshold} is below "
                f"the equilibrium noise floor sqrt(2/M)={floor_sqrt2M:.4g} at "
                f"M={num_chains}: Phase 1 can never switch on delta and would "
                f"silently run to the num_unadjusted_steps budget. Use the default "
                f"switch_mode='self_calibrated' (or 'm_scaled') for small M, or set "
                f"M >~ 2e4 to reproduce the paper's literal 0.01.")

    rng = jax.random.key(seed)
    k_init, k_p1, k_p2 = jax.random.split(rng, 3)

    # ---- chain initialization ----
    # init_positions (if provided) OVERRIDES warm/cold generation entirely: it is the
    # explicit unconstrained initial ensemble (shape (>=num_chains, dim)). This is the
    # plumbing for the wrapper's robust prior cold-start (Fix 1). "warm" (qz surrogate)
    # and "cold" (N(0,I)*cold_scale) remain available when init_positions is None.
    if init_positions is not None:
        positions = jnp.asarray(init_positions)
        if positions.ndim != 2:
            raise ValueError(
                f"init_positions must be 2-D (num_chains, dim); got {positions.shape}.")
        if positions.shape[0] < num_chains:
            raise ValueError(
                f"init_positions has {positions.shape[0]} rows but num_chains rounded "
                f"to {num_chains} (multiple of {num_devices} devices); provide at least "
                f"that many initial positions.")
        if dim is not None and positions.shape[1] != dim:
            raise ValueError(
                f"init_positions has dim {positions.shape[1]} != requested dim {dim}.")
        positions = positions[:num_chains]      # device-multiple slice (extras dropped)
    else:
        if dim is None:
            dim = int(np.asarray(qz.mean()).shape[-1])
        if init_mode == "warm":
            if qz is None:
                raise ValueError("init_mode='warm' requires qz.")
            positions = qz.sample((num_chains,), seed=k_init)
        else:  # cold: unconstrained N(0, I) * cold_scale (no silent target default)
            positions = cold_scale * jax.random.normal(k_init, (num_chains, dim))
        positions = jnp.asarray(positions)

    state = init_multi(positions, k_init, logdensity_fn)
    canon = _canon_dtype(state)
    _cast = lambda a: (
        jnp.asarray(a).astype(canon)
        if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating)
        else jnp.asarray(a)
    )
    state = jax.tree_util.tree_map(_cast, state)
    dim = state.position.shape[-1]

    # F1 lever (stocktake): blackjax-faithful gradient-aligned velocity init with
    # equipartition sign flip (see _gradient_equipartition_momentum).
    if velocity_init == "gradient":
        state = state._replace(momentum=_gradient_equipartition_momentum(
            state.position, state.momentum, state.logdensity_grad, canon))

    mesh = jax.make_mesh((num_devices,), ("device",))
    _resh = getattr(jax, "reshard", jax.device_put)
    sh_chain = NamedSharding(mesh, P("device"))
    sh_keys = NamedSharding(mesh, P(None, "device"))
    sh_repl = NamedSharding(mesh, P())

    # window guard: T = save_frac * num_unadjusted_steps (spec: 20% of total)
    T_window = max(2, round(save_frac * num_unadjusted_steps))

    # =====================================================================
    # PHASE 1 — unadjusted MCLMC, isotropic identity metric, Leapfrog
    # =====================================================================
    eye = jnp.eye(dim, dtype=canon)
    p1_kernel = _build_kernel_shardmap(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=eye,
        integrator=isokinetic_velocity_verlet_smart,
    )

    eevpd_want_fn = lambda D: laps_core.eevpd_wanted(D, schedule=schedule, C=C)

    # init eps0 = 0.01*sqrt(d) (spec Alg.1); L0 from initial ensemble (Eq. 9).
    eps0 = jnp.asarray(0.01 * np.sqrt(dim), canon)
    L0 = laps_core.decoherence_length(state.position, alpha=alpha).astype(canon)
    # F1-companion lever: bj initializes L = inf so the FIRST step applies no
    # Maruyama refresh (nu = exp(-2*eps/L) -> 1, zero noise), preserving the
    # init velocity through step 1; the per-step L adaptation immediately
    # replaces it from the ensemble stats afterwards. Default off = paper Eq. 9
    # from the initial ensemble (the documented paper-over-bj choice).
    if L0_inf:
        L0 = jnp.asarray(jnp.inf, canon)

    @jax.jit
    @functools.partial(
        _shard_map,
        mesh=mesh,
        in_specs=(P(None, "device"), P("device"), P(), P()),
        out_specs=(
            (P("device"), P(), P()),
            (P(), P(), P(), P(), P(), P(), P(), P(), P()),
        ),
    )
    def run_p1_chunk(keys, states, step_size, L):
        def body(carry, key_row):
            states, ss, L = carry

            def per_chain(st, k):
                return p1_kernel(rng_key=k, state=st, L=L, step_size=ss)

            new_states, infos = jax.vmap(per_chain)(states, key_row)

            xs = new_states.position                 # (cpd, d)
            gs = new_states.logdensity_grad          # (cpd, d)
            dE = infos.energy_change                 # (cpd,)
            n_total = cpd * jax.lax.axis_size("device")

            loc = jnp.concatenate(
                [jnp.sum(xs, 0), jnp.sum(xs * xs, 0),
                 jnp.sum(xs * gs, 0), jnp.sum(gs, 0),
                 jnp.sum(jnp.square(xs * xs), 0)]
            )
            glob = jax.lax.psum(loc, "device") / n_total
            s_x, s_xx, s_xg, s_g, s_xxxx = jnp.split(glob, 5)

            loc2 = jnp.stack([jnp.sum(dE), jnp.sum(dE * dE)])
            s_d, s_dd = jax.lax.psum(loc2, "device") / n_total

            # laps_core math, sharded: equipartition diag, EEVPD, L (Eqs.4/6/7/9)
            E_ii = -s_xg + s_x * s_g
            D_tilde = jnp.mean(jnp.square(jnp.ones((), canon) - E_ii))
            eevpd_obs = (s_dd - jnp.square(s_d)) / jnp.asarray(dim, canon)
            var_i = jnp.maximum(s_xx - jnp.square(s_x), jnp.asarray(0.0, canon))
            L_new = jnp.asarray(alpha, canon) * jnp.sqrt(jnp.sum(var_i))

            eevpd_w = eevpd_want_fn(D_tilde)
            ss_new = laps_core.step_size_update(ss, eevpd_w, eevpd_obs)

            # per-step NaN-rejection fraction (diagnostic stream; mirrors bj's
            # rejection_rate_nans). F2 lever: bj-faithful brake — when ANY chain
            # NaN-rejected this step, the controller factor is REPLACED by 0.5
            # (laps_burn_in.py:333-335: eps_factor = nan_reject(1-nans, 0.5, .)).
            nonans = infos.nonans.astype(canon)          # (cpd,)
            nan_frac = jax.lax.psum(
                jnp.sum(jnp.ones_like(nonans) - nonans), "device") / n_total
            if nan_eps_halving:
                ss_new = jnp.where(nan_frac > 0,
                                   ss * jnp.asarray(0.5, canon), ss_new)

            # self-calibrated noise floor of the ACTIVE switch observable from the
            # CURRENT ensemble: floor_i = sqrt(Var_rho[f]/M)/E_rho[f]. paper f=x_i^2
            # -> Var = E[x^4]-E[x^2]^2; emaus f=x_i -> Var = E[x^2]-E[x]^2.
            M_tot = jnp.asarray(n_total, canon)
            tiny = jnp.asarray(1e-30, canon)
            if switch == "paper":
                e_obs = s_xx
                var_obs = jnp.maximum(s_xxxx - jnp.square(s_xx),
                                      jnp.asarray(0.0, canon))
            else:  # emaus observable x_i
                e_obs = s_x
                var_obs = var_i
            floor = jnp.sqrt(var_obs / M_tot) / jnp.maximum(jnp.abs(e_obs), tiny)

            y = (D_tilde, eevpd_w, eevpd_obs, ss_new, L_new, s_xx, s_x, floor,
                 nan_frac)
            return (new_states, ss_new, L_new), y

        carry, ys = jax.lax.scan(body, (states, step_size, L), keys)
        return carry, ys

    # histories accumulated on host across chunks
    H = {k: [] for k in
         ["D_tilde", "ew", "eo", "ss", "L", "s_xx", "s_x", "floor", "nan",
          "delta_max"]}
    eps, L = eps0, L0
    steps_done = 0
    switch_index = num_unadjusted_steps
    switched = False
    consecutive_fires = 0       # persistence guard: consecutive ripe chunks
    p1_sizes = _chunk_sizes(num_unadjusted_steps, chunk_size)  # final short chunk ok

    traj = ({"p1_pos": [], "p1_logp": [], "p2_pos": [], "p2_logp": []}
            if track_chains else None)

    def _snap(phase):
        if traj is not None:
            traj[f"{phase}_pos"].append(np.asarray(_resh(state.position, sh_repl)))
            traj[f"{phase}_logp"].append(np.asarray(_resh(state.logdensity, sh_repl)))

    state = _resh(state, sh_chain)
    _snap("p1")                       # chunk-0 snapshot = the initial ensemble
    for c, sz in enumerate(p1_sizes):
        ck = jax.random.fold_in(k_p1, c)
        keys = jax.random.split(ck, (sz, num_chains))
        (state, eps, L), ys = run_p1_chunk(
            _resh(keys, sh_keys), state,
            _resh(eps, sh_repl), _resh(L, sh_repl),
        )
        _snap("p1")
        D_t, ew, eo, ss, Lh, s_xx, s_x, fl, nf = (
            np.asarray(_resh(a, sh_repl)) for a in ys)
        H["D_tilde"].append(D_t); H["ew"].append(ew); H["eo"].append(eo)
        H["ss"].append(ss); H["L"].append(Lh); H["s_xx"].append(s_xx)
        H["s_x"].append(s_x); H["floor"].append(fl); H["nan"].append(nf)
        steps_done += sz

        # host-side switch eval on the trailing window (active observable + mode)
        obs_active = H["s_xx"] if switch == "paper" else H["s_x"]
        obs_cat = np.concatenate(obs_active, axis=0)         # (steps_done, d)
        dmax_chunk = np.full(sz, np.nan)
        if steps_done >= T_window:
            window = jnp.asarray(obs_cat[-T_window:])
            floor_cur = H["floor"][-1][-1]                   # current-ensemble floor
            _, dmx, fired = laps_core.phase1_switch(
                window, switch=switch, threshold=switch_threshold,
                switch_mode=switch_mode, k=switch_k, k_m=switch_k_m,
                floor=floor_cur, M=num_chains)
            dmax_chunk[-1] = float(dmx)
            # OPTIONAL persistence guard (default switch_persist=1 = fire-on-first,
            # i.e. do_switch == fired -> identical to the pre-guard behaviour).
            do_switch, consecutive_fires = laps_core.persistence_update(
                bool(fired), consecutive_fires, switch_persist)
            if do_switch and early_stop:
                switch_index = steps_done
                switched = True
                H["delta_max"].append(dmax_chunk)
                break
        H["delta_max"].append(dmax_chunk)

    phase1_len = steps_done
    p1_D = np.concatenate(H["D_tilde"])[:phase1_len]
    p1_ew = np.concatenate(H["ew"])[:phase1_len]
    p1_eo = np.concatenate(H["eo"])[:phase1_len]
    p1_ss = np.concatenate(H["ss"])[:phase1_len]
    p1_L = np.concatenate(H["L"])[:phase1_len]
    p1_sq = np.concatenate(H["s_xx"], axis=0)[:phase1_len]
    p1_mn = np.concatenate(H["s_x"], axis=0)[:phase1_len]
    p1_floor = np.concatenate(H["floor"], axis=0)[:phase1_len]
    p1_nan = np.concatenate(H["nan"])[:phase1_len]
    p1_dmax = np.concatenate(H["delta_max"])[:phase1_len]

    # post-hoc: where would the LITERAL paper/emaus absolute switch fire over the
    # recorded history (diagnostic contrast; no M passed -> guard disabled so the
    # "never fires at small M" outcome is observable rather than an exception).
    sidx_paper = _switch_index_host(p1_sq, "paper", switch_threshold, T_window,
                                    switch_mode="absolute")
    sidx_emaus = _switch_index_host(p1_mn, "emaus", switch_threshold, T_window,
                                    switch_mode="absolute")

    # ---- phase boundary: diagonal preconditioner Var[x_i] ----
    # Default "pooled_half": Var from per-step ensemble moments POOLED over the
    # 2nd half of Phase 1 (historical). F3 lever "final": END-STATE instantaneous
    # ensemble Var (what BOTH references use: bj laps.py:250-251 consumes the
    # final-step carried Var; paper Eq. y_i = x_i/sqrt(Var_rho_t[x_i]) at switch).
    if precond_source == "final":
        e_xx = p1_sq[-1]
        e_x = p1_mn[-1]
    else:
        half = phase1_len // 2
        e_xx = p1_sq[half:].mean(axis=0)
        e_x = p1_mn[half:].mean(axis=0)
    precond_var = np.maximum(e_xx - e_x ** 2, 1e-12)        # Var[x_i]
    if p2_precond_var is not None:                          # DIAGNOSTIC override:
        # replace the (broad-ensemble) metric with a known-correct one (e.g. qz var)
        precond_var = np.maximum(np.asarray(p2_precond_var, dtype=precond_var.dtype), 1e-12)
    imm_diag = jnp.asarray(np.diag(precond_var), canon)

    # =====================================================================
    # PHASE 2 — adjusted MAMS, diagonal precond, MN2 (d<=200) / MN4 (d>200)
    # =====================================================================
    order = 2 if dim <= 200 else 4
    p2_integrator = isokinetic_mclachlan_smart if order == 2 else isokinetic_omelyan_smart
    a_target = laps_core.target_accept(order)
    N = int(steps_per_trajectory)

    # PHASE-2-OFF CONTROL (spec follow-up F3): isolate the Phase-2 correction.
    # When disabled, skip the adjusted MAMS phase entirely and return the
    # UNADJUSTED Phase-1 ensemble as the result, so b^2 reflects the unadjusted
    # asymptotic (discretization) bias. Phase-2 diagnostics become single-element
    # placeholders (no adjusted steps were run).
    if not phase2_enabled:
        # (M, 1, dim): the unadjusted Phase-1 ensemble as one sample/chain, in the
        # same 3-D layout as the adjusted path (.reshape((-1, dim)) -> (M, dim)).
        samples = np.asarray(_resh(state.position, sh_repl))[:, None, :]
        eps_final = float(eps)
        L_full = N * eps_final
        return LAPSResults(
            samples=samples,
            n_samples_total=int(samples.shape[0]),
            p1_D_tilde=p1_D, p1_eevpd_wanted=p1_ew, p1_eevpd_obs=p1_eo,
            p1_step_size=p1_ss, p1_L=p1_L, p1_obs_sq=p1_sq, p1_obs_mean=p1_mn,
            p1_delta_max=p1_dmax, phase1_len=phase1_len, switch_index=switch_index,
            switch_index_paper=sidx_paper, switch_index_emaus=sidx_emaus,
            switched=switched, precond_var=precond_var, integrator_order=order,
            target_accept=a_target,
            p2_accept=np.array([np.nan]), p2_step_size=np.array([eps_final]),
            p2_frozen=np.array([False]), p2_settled_accept=np.array([np.nan]),
            p2_final_step_size=eps_final, p2_L_full=L_full,
            p2_L_proposal=L_proposal_factor * L_full,
            p1_nan_frac=p1_nan,
            chain_traj=_traj_pack(traj),
            resample_info=None,
        )

    def _make_p2_runner(imm):
        """Build the adjusted kernel + jitted chunk runner for a given diagonal
        metric. A factory so the DC-7.3 resample step can REBUILD both from the
        resampled ensemble's Var mid-phase (the boundary metric is poisoned by
        the broad mixture)."""
        kern = _build_adjusted_kernel_shardmap(
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=imm,
            integrator=p2_integrator,
        )

        @jax.jit
        @functools.partial(
            _shard_map,
            mesh=mesh,
            in_specs=(P(None, "device"), P("device"), P()),
            out_specs=(P("device"), P()),
        )
        def _run(keys, states, step_size):
            # eps is HELD CONSTANT across the whole chunk: it is a scan CONSTANT,
            # not a carry, so the kernel cannot mutate it within the traced region.
            def body(states, key_row):
                def per_chain(st, k):
                    return kern(rng_key=k, state=st, step_size=step_size,
                                num_integration_steps=N)

                new_states, infos = jax.vmap(per_chain)(states, key_row)
                n_total = cpd * jax.lax.axis_size("device")
                accept = jax.lax.psum(
                    jnp.sum(infos.acceptance_rate), "device") / n_total
                return new_states, accept             # per-step ensemble accept

            states, accepts = jax.lax.scan(body, states, keys)
            return states, accepts

        return _run

    # STAGED per-chunk bisection (regression fix): the bisection lives ENTIRELY on
    # the host between chunks. The device scan runs at a FIXED ``step_size``
    # (never mutates eps) and only records the per-step ensemble acceptance; the
    # bracket/freeze are updated ONCE per chunk on the SETTLED acceptance (see the
    # host loop). ``persist_need`` counts CONSECUTIVE in-band CHUNKS;
    # ``min_chunks`` defers any freeze until that many chunks have elapsed.
    persist_need = int(p2_freeze_persist)
    min_chunks = int(p2_min_adapt_chunks)
    run_p2_chunk = _make_p2_runner(imm_diag)

    # FIX A: initialize the Phase-2 step from the TRAJECTORY SCALE, not the tiny
    # Phase-1 EEVPD handoff eps (which pins acceptance ~1.0 and leaves the bracket
    # nowhere to start). The adjusted MAMS kernel integrates N steps per trajectory
    # with total length L_full = N*eps (see module "Spec deviations"); matching that
    # to the decoherence/trajectory length L (Eq. 9, the latest Phase-1 estimate)
    # gives eps0 = L / N -- the right BALLPARK for the adjusted phase. This replaces
    # the regression where Phase 2 began at the sub-EEVPD Phase-1 step, acceptance
    # stayed pinned near 1.0, and the (then arithmetic, narrowly clamped) bisection
    # could not bracket UPWARD to the much larger step the phase needs. Overridable
    # via ``p2_eps_init``.
    L_traj = float(eps)            # fallback if L degenerate (see guard below)
    try:
        L_traj = float(L)          # latest Phase-1 decoherence length (Eq. 9)
    except Exception:
        L_traj = float(eps)
    if p2_eps_init is not None:
        eps0_p2 = float(p2_eps_init)
    else:
        eps0_p2 = L_traj / max(N, 1)
    if not np.isfinite(eps0_p2) or eps0_p2 <= 0.0:
        eps0_p2 = float(eps)       # degenerate L -> fall back to the handoff eps
    eps2 = jnp.asarray(eps0_p2, canon)
    # Start the bracket UNSET (NaN/NaN): the controller then BRACKETS by geometric
    # expansion (FIX B) from eps0 in whichever direction the settled acceptance
    # indicates, in EITHER direction. (We deliberately do NOT pre-seed a fixed
    # [eps0/30, eps0*30] bracket: a pre-seeded bracket is a HARD box the geometric
    # bisection can only narrow inward, so if the true step lies outside it -- e.g.
    # the start-far-below case, eps* > 30*eps0 -- the search converges to the box
    # edge with acceptance still off target and NEVER reaches it. NaN-start
    # expansion has no such trap and reaches the target from any starting offset
    # within the clamp.)
    lo = jnp.asarray(jnp.nan, canon)
    hi = jnp.asarray(jnp.nan, canon)
    frozen = False
    persist = 0                                       # consecutive in-band CHUNKS
    # Safety rail bounding the step so the bracketing walk cannot run away: WIDE
    # (3 decades each side of eps0 = L/N) so a legitimate upward/downward bracketing
    # search has ample room and is never blocked by the clamp.
    eps_min = eps0_p2 * 1e-3
    eps_max = eps0_p2 * 1e3
    P2 = {"accept": [], "ss": [], "frozen": []}
    settled_hist = []
    resample_info = None
    # CHANGE A: Phase 2 gets its OWN (smaller) chunk size so a FIXED
    # ``num_adjusted_steps`` adaptation budget yields MANY more bisection updates
    # (one per chunk). At the defaults (num_adjusted_steps=200, p2_chunk_size=8)
    # that is ~25 chunks -- ample room to BRACKET (a handful of geometric-expansion
    # chunks, sped up by p2_growth=2.5) AND REFINE (>= ~6 bisection chunks) AND hold
    # ``p2_freeze_persist`` consecutive in-band chunks before the latch. The old
    # design reused the Phase-1 chunk_size (25) -> only ~8 Phase-2 chunks, of which
    # ~7 were eaten just crossing the flat accept~1.0 plateau, so it overshot 0.70
    # with no chunks left to refine and never latched.
    p2_sizes = _chunk_sizes(num_adjusted_steps, p2_chunk_size)  # final short chunk ok
    for c, sz in enumerate(p2_sizes):
        ck = jax.random.fold_in(k_p2, c)
        keys = jax.random.split(ck, (sz, num_chains))
        state, accepts = run_p2_chunk(
            _resh(keys, sh_keys), state, _resh(eps2, sh_repl))
        _snap("p2")
        accepts = np.asarray(_resh(accepts, sh_repl))          # (sz,) per-step accept
        # Record the per-step ensemble accept (real trajectory for the diagnose
        # plot), the eps HELD during this chunk, and the (pre-update) frozen flag.
        P2["accept"].append(accepts)
        P2["ss"].append(np.full(sz, float(eps2)))
        P2["frozen"].append(np.full(sz, bool(frozen)))

        # SETTLED acceptance = mean over the LATTER HALF of the chunk's per-step
        # accepts (drop the first half: the within-chunk / post-switch transient
        # that lagged near 1.0 and ran the old per-step bracket away).
        half = sz // 2
        a_settled = float(np.mean(accepts[half:]))
        settled_hist.append(a_settled)

        # Host-side bracket/freeze update: ONCE per chunk, on the settled accept.
        # Skip once frozen (one-way latch: eps + bracket held, remaining chunks
        # sample at the frozen eps).
        if not frozen:
            in_band = abs(a_settled - a_target) <= bisection_tol
            persist = persist + 1 if in_band else 0
            freeze_now = (persist >= persist_need) and ((c + 1) >= min_chunks)
            eps2_new, lo, hi, fr = laps_core.bisection_step(
                eps2, jnp.asarray(a_settled, canon), a_target, lo=lo, hi=hi,
                tol=bisection_tol, frozen=False, freeze_enable=freeze_now,
                eps_min=eps_min, eps_max=eps_max, growth=p2_growth)
            eps2 = jnp.asarray(eps2_new, canon)
            frozen = bool(fr)

        # ---- DC-7.3 RESAMPLE step (flag-gated; default off) ----
        # Replace straggler STATES with survivor draws (an MH re-initialization:
        # pre-resample states are burn-in, never averaged), REBUILD the diagonal
        # metric from the resampled ensemble (the boundary metric is poisoned by
        # the broad mixture), and RESTART the eps bisection at eps0 = L/N of the
        # resampled ensemble (FIX A applied to the clean ensemble).
        if (p2_resample_at_chunk is not None
                and (c + 1) == int(p2_resample_at_chunk)):
            pos = np.asarray(_resh(state.position, sh_repl))
            lp = np.asarray(_resh(state.logdensity, sh_repl))
            dlp = (float(p2_resample_dlogp) if p2_resample_dlogp is not None
                   else dim / 2.0 + 4.0 * np.sqrt(dim / 2.0))
            cut = float(np.nanmax(lp)) - dlp
            surv = np.where(lp > cut)[0]
            strag = np.where(~(lp > cut))[0]        # includes NaN-logp chains
            if surv.size < int(p2_resample_min_survivors) or strag.size == 0:
                resample_info = dict(
                    chunk=c + 1, skipped=True, n_survivors=int(surv.size),
                    n_stragglers=int(strag.size), cut=cut, dlogp=dlp)
                print(f"[LAPS] resample SKIPPED at chunk {c + 1}: "
                      f"{surv.size} survivors / {strag.size} stragglers "
                      f"(min_survivors={int(p2_resample_min_survivors)})",
                      flush=True)
            else:
                if p2_resample_mode == "replace":
                    kr = jax.random.fold_in(k_p2, 0x5E5A)
                    donors = surv[np.asarray(
                        jax.random.randint(kr, (strag.size,), 0, surv.size))]
                    idx = np.arange(num_chains)
                    idx[strag] = donors
                    take = lambda a: jnp.asarray(
                        np.asarray(_resh(a, sh_repl))[idx])
                    state = state._replace(
                        position=take(state.position),
                        momentum=take(state.momentum),
                        logdensity=take(state.logdensity),
                        logdensity_grad=take(state.logdensity_grad))
                    state = _resh(jax.tree_util.tree_map(_cast, state), sh_chain)
                    pos_rs = pos[idx]
                else:                       # retune_only: positions untouched;
                    donors = None           # metric/eps from the SURVIVOR subset
                    pos_rs = pos[surv]
                rs_var = np.maximum(pos_rs.var(axis=0), 1e-12)
                run_p2_chunk = _make_p2_runner(jnp.asarray(np.diag(rs_var), canon))
                L_rs = float(laps_core.decoherence_length(
                    jnp.asarray(pos_rs), alpha=alpha))
                eps0_rs = L_rs / max(N, 1)
                if not np.isfinite(eps0_rs) or eps0_rs <= 0.0:
                    eps0_rs = float(eps2)           # degenerate rail
                eps2 = jnp.asarray(eps0_rs, canon)
                lo = jnp.asarray(jnp.nan, canon)
                hi = jnp.asarray(jnp.nan, canon)
                frozen = False
                persist = 0
                eps_min = eps0_rs * 1e-3
                eps_max = eps0_rs * 1e3
                _snap("p2")     # post-resample snapshot (duplicate-decorr t0)
                resample_info = dict(
                    chunk=c + 1, skipped=False, mode=p2_resample_mode,
                    n_survivors=int(surv.size),
                    n_stragglers=int(strag.size), cut=cut, dlogp=dlp,
                    stragglers=strag, donors=donors, eps0_rs=eps0_rs,
                    rs_var=rs_var,
                    snap_index=(len(traj["p2_pos"]) - 1
                                if traj is not None else None))
                print(f"[LAPS] resample mode={p2_resample_mode}: "
                      f"{strag.size} stragglers / {surv.size} survivors at chunk "
                      f"{c + 1} (cut={cut:.2f}, eps0_rs={eps0_rs:.3e})", flush=True)

    p2_accept = np.concatenate(P2["accept"])
    p2_ss = np.concatenate(P2["ss"])
    p2_frozen = np.concatenate(P2["frozen"])
    p2_settled = np.asarray(settled_hist)

    # =====================================================================
    # CHANGE B — post-freeze COLLECTION sub-phase (multi-sample per chain)
    # =====================================================================
    # The adaptation loop above (the ``num_adjusted_steps`` budget) tunes + freezes
    # eps. Now, at the FROZEN hyperparameters (eps2 held, diagonal precond, N
    # steps/traj), we keep a THINNED time series per chain: ``p2_keep_per_chain``
    # samples, one every ``p2_thin`` frozen Phase-2 steps. These are valid MCMC at
    # FIXED hyperparameters, so the within-chain (chain, sample) structure makes
    # R-hat/ESS meaningful while low chain counts still yield a dense posterior.
    #
    # BUDGET SPLIT (documented): ``num_adjusted_steps`` is the ADAPTATION budget;
    # the collection adds ``p2_keep_per_chain * p2_thin`` further Phase-2 steps AT
    # the frozen eps (0 when keep == 1). Total Phase-2 steps = num_adjusted_steps +
    # (keep*thin if keep > 1 else 0).
    keep = int(p2_keep_per_chain)
    thin = max(1, int(p2_thin))
    if keep <= 1:
        # Backward-compatible: keep == 1 takes the final adaptation ensemble
        # directly (Eq. 3), 0 extra steps, numerically identical to the
        # pre-CHANGE-B behaviour. Shape (M, 1, dim) so .reshape((-1, dim)) ->
        # (M, dim) exactly as the old (M, dim) ensemble.
        samples = np.asarray(_resh(state.position, sh_repl))[:, None, :]
    else:
        # keep > 1: run ``keep`` collection sub-chunks of ``thin`` frozen steps
        # each, recording the ensemble position after each (thinning decorrelates
        # successive kept samples and drops the adaptation-end transient).
        collected = []
        for j in range(keep):
            ck = jax.random.fold_in(k_p2, 10_000 + j)
            keys = jax.random.split(ck, (thin, num_chains))
            state, _ = run_p2_chunk(
                _resh(keys, sh_keys), state, _resh(eps2, sh_repl))
            collected.append(np.asarray(_resh(state.position, sh_repl)))  # (M, dim)
        samples = np.stack(collected, axis=1)                              # (M,keep,dim)
    n_samples_total = int(samples.shape[0] * samples.shape[1])

    # eps_final reflects the step COLLECTION actually sampled at (the held eps2);
    # for a frozen run eps2 == p2_ss[-1], so this matches the old report.
    eps_final = float(eps2)
    L_full = N * eps_final

    return LAPSResults(
        samples=samples,
        n_samples_total=n_samples_total,
        p1_D_tilde=p1_D, p1_eevpd_wanted=p1_ew, p1_eevpd_obs=p1_eo,
        p1_step_size=p1_ss, p1_L=p1_L, p1_obs_sq=p1_sq, p1_obs_mean=p1_mn,
        p1_delta_max=p1_dmax, phase1_len=phase1_len, switch_index=switch_index,
        switch_index_paper=sidx_paper, switch_index_emaus=sidx_emaus,
        switched=switched, precond_var=precond_var, integrator_order=order,
        target_accept=a_target,
        p2_accept=p2_accept, p2_step_size=p2_ss, p2_frozen=p2_frozen,
        p2_settled_accept=p2_settled,
        p2_final_step_size=eps_final, p2_L_full=L_full,
        p2_L_proposal=L_proposal_factor * L_full,
        p1_nan_frac=p1_nan,
        chain_traj=_traj_pack(traj),
        resample_info=resample_info,
    )


def LAPS_late_adjusted_JIT(prob_model, qz=None, *, init_mode="warm",
                           num_chains=512, seed=0, **kwargs):
    """gigalens wrapper mirroring ``MCLMC_JIT``: builds ``log_prob`` then runs LAPS.

    Scene-only: the ProbModel renders ``log_prob(z)`` through its per-dataset
    SceneSimulators directly (no separately-built simulator), exactly as
    ``MCLMC_JIT`` does.

    ``init_mode`` (Fix 1)
    ---------------------
    * ``"warm"``  : qz surrogate sample (requires ``qz``).
    * ``"cold"``  : unconstrained N(0, I) * cold_scale -- empirically OVER-DISPERSED
                    on the 22-D lens (chains never equilibrate); kept for ablation.
    * ``"prior"`` : RECOMMENDED robust cold-start. Sample the model's prior
                    (constrained) and map each draw to UNCONSTRAINED space, then hand
                    the (num_chains, dim) ensemble to the core via ``init_positions``.
                    Needs no qz. The constrained->unconstrained map is the bijector
                    INVERSE (``prob_model.bij.inverse``), exactly the map gigalens
                    uses to seed its own optimizers; ``bij.forward`` (used inside
                    ``log_prob``) is its inverse.
    """
    def log_prob(z):
        return prob_model.log_prob(z)[0]

    if init_mode == "prior":
        if prob_model.prior is None or prob_model.bij is None:
            raise ValueError(
                "init_mode='prior' requires the model to expose a prior + bijector "
                "(prob_model.prior / prob_model.bij are None for a constants-only "
                "model).")
        # Sample num_chains prior draws (constrained), map to UNCONSTRAINED via the
        # bijector inverse (flat-z convention: (n, dim) in, (n, dim) out). Same
        # idiom as gigalens' gigalens/src/gigalens/jax/inference.py:
        #     start = prob_model.prior.sample(n, seed=key)   # constrained
        #     params = prob_model.bij.inverse(start)         # unconstrained
        ndev = len(jax.devices())
        n_init = (num_chains // ndev) * ndev
        if n_init < 1:
            raise ValueError(
                f"num_chains ({num_chains}) must be >= num_devices ({ndev}).")
        k_prior = jax.random.fold_in(jax.random.key(seed), 0x9E3779B9)
        start = prob_model.prior.sample(n_init, seed=k_prior)     # constrained draws
        init_positions = prob_model.bij.inverse(start)            # (n_init, dim)
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        init_positions = init_positions.astype(dtype)             # sampler dtype
        return LAPS_late_adjusted(
            log_prob, qz, init_positions=init_positions,
            num_chains=num_chains, seed=seed, **kwargs)

    return LAPS_late_adjusted(
        log_prob, qz, init_mode=init_mode, num_chains=num_chains, seed=seed, **kwargs)
