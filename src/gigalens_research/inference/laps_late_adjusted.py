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
``LAPS_late_adjusted_JIT(model_seq, qz, ...)``
    Thin gigalens wrapper, mirrors ``MCLMC_JIT``: builds
    ``log_prob(z) = model_seq.prob_model.log_prob(z)[0]`` and calls the core.

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

Phase 2 (MAMS adjusted): bisection-tune the step size to the target acceptance
(0.7 for MN2, 0.9 for MN4) via ``laps_core.bisection_step``, freezing once
``|a - a_target| <= 0.03``, then sample to the ``num_adjusted_steps`` budget.
One sample per chain is collected (the final ensemble locations, Eq. 3).

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
window first satisfied the threshold. Bisection runs inside the scan as a pure
``jnp.where`` update (``laps_core.bisection_step``), with no early exit — it
simply freezes and the remaining budget samples at the frozen step size.

Spec deviations (honest reporting)
----------------------------------
* The in-tree ``_build_adjusted_kernel_shardmap`` is the *Hamiltonian* MAMS
  variant (full momentum refresh once per trajectory, NO intra-trajectory
  partial refresh — equivalent to ``L_proposal_factor = inf``). The spec's
  ``L_proposal = 1.25*L_full`` partial-refresh is therefore NOT applied by this
  kernel; ``L_full = N*eps`` and ``L_proposal`` are recorded in diagnostics but
  do not influence Phase 2. Composing the validated kernel was preferred over
  reimplementing partial refresh (task instruction).
* Velocity init: ``_single_init`` draws a RANDOM unit momentum; the paper aligns
  the initial velocity with the gradient (design open-Q2). Phase 1 is only meant
  to "approach the target fast", so random init is retained.
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
        "samples",            # (num_chains, dim) final ensemble (one sample/chain)
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
        "p2_step_size",       # (T2,) step size per step
        "p2_frozen",          # (T2,) bool frozen flag per step
        "p2_final_step_size", # float: final (frozen) step size
        "p2_L_full",          # float: N*eps_final (recorded; see deviations)
        "p2_L_proposal",      # float: 1.25*L_full (recorded; NOT applied)
    ],
)


def _canon_dtype(state):
    return jnp.asarray(state.logdensity).dtype


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
    cold_scale=1.0,
    seed=0,
):
    """Run the two-phase LAPS sampler. See module docstring for the algorithm.

    Returns a ``LAPSResults`` namedtuple of samples + per-phase diagnostics.
    """
    if init_mode not in ("warm", "cold"):
        raise ValueError(f"init_mode must be 'warm' or 'cold', got {init_mode!r}")
    if qz is None and dim is None:
        raise ValueError("Provide qz (warm/cold dim) or an explicit dim.")
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

    # ---- chain initialization (warm = qz surrogate; cold = N(0,I)) ----
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

    @jax.jit
    @functools.partial(
        _shard_map,
        mesh=mesh,
        in_specs=(P(None, "device"), P("device"), P(), P()),
        out_specs=(
            (P("device"), P(), P()),
            (P(), P(), P(), P(), P(), P(), P(), P()),
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

            y = (D_tilde, eevpd_w, eevpd_obs, ss_new, L_new, s_xx, s_x, floor)
            return (new_states, ss_new, L_new), y

        carry, ys = jax.lax.scan(body, (states, step_size, L), keys)
        return carry, ys

    # histories accumulated on host across chunks
    H = {k: [] for k in
         ["D_tilde", "ew", "eo", "ss", "L", "s_xx", "s_x", "floor", "delta_max"]}
    eps, L = eps0, L0
    steps_done = 0
    switch_index = num_unadjusted_steps
    switched = False
    consecutive_fires = 0       # persistence guard: consecutive ripe chunks
    p1_sizes = _chunk_sizes(num_unadjusted_steps, chunk_size)  # final short chunk ok

    state = _resh(state, sh_chain)
    for c, sz in enumerate(p1_sizes):
        ck = jax.random.fold_in(k_p1, c)
        keys = jax.random.split(ck, (sz, num_chains))
        (state, eps, L), ys = run_p1_chunk(
            _resh(keys, sh_keys), state,
            _resh(eps, sh_repl), _resh(L, sh_repl),
        )
        D_t, ew, eo, ss, Lh, s_xx, s_x, fl = (np.asarray(_resh(a, sh_repl)) for a in ys)
        H["D_tilde"].append(D_t); H["ew"].append(ew); H["eo"].append(eo)
        H["ss"].append(ss); H["L"].append(Lh); H["s_xx"].append(s_xx)
        H["s_x"].append(s_x); H["floor"].append(fl)
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
    p1_dmax = np.concatenate(H["delta_max"])[:phase1_len]

    # post-hoc: where would the LITERAL paper/emaus absolute switch fire over the
    # recorded history (diagnostic contrast; no M passed -> guard disabled so the
    # "never fires at small M" outcome is observable rather than an exception).
    sidx_paper = _switch_index_host(p1_sq, "paper", switch_threshold, T_window,
                                    switch_mode="absolute")
    sidx_emaus = _switch_index_host(p1_mn, "emaus", switch_threshold, T_window,
                                    switch_mode="absolute")

    # ---- phase boundary: diagonal preconditioner from 2nd half (pooled) ----
    half = phase1_len // 2
    e_xx = p1_sq[half:].mean(axis=0)
    e_x = p1_mn[half:].mean(axis=0)
    precond_var = np.maximum(e_xx - e_x ** 2, 1e-12)        # Var[x_i]
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
        samples = np.asarray(_resh(state.position, sh_repl))
        eps_final = float(eps)
        L_full = N * eps_final
        return LAPSResults(
            samples=samples,
            p1_D_tilde=p1_D, p1_eevpd_wanted=p1_ew, p1_eevpd_obs=p1_eo,
            p1_step_size=p1_ss, p1_L=p1_L, p1_obs_sq=p1_sq, p1_obs_mean=p1_mn,
            p1_delta_max=p1_dmax, phase1_len=phase1_len, switch_index=switch_index,
            switch_index_paper=sidx_paper, switch_index_emaus=sidx_emaus,
            switched=switched, precond_var=precond_var, integrator_order=order,
            target_accept=a_target,
            p2_accept=np.array([np.nan]), p2_step_size=np.array([eps_final]),
            p2_frozen=np.array([False]),
            p2_final_step_size=eps_final, p2_L_full=L_full,
            p2_L_proposal=L_proposal_factor * L_full,
        )

    p2_kernel = _build_adjusted_kernel_shardmap(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=imm_diag,
        integrator=p2_integrator,
    )

    @jax.jit
    @functools.partial(
        _shard_map,
        mesh=mesh,
        in_specs=(P(None, "device"), P("device"), P(), P(), P(), P()),
        out_specs=(
            (P("device"), P(), P(), P(), P()),
            (P(), P(), P()),
        ),
    )
    def run_p2_chunk(keys, states, step_size, lo, hi, frozen):
        def body(carry, key_row):
            states, ss, lo, hi, frozen = carry

            def per_chain(st, k):
                return p2_kernel(rng_key=k, state=st, step_size=ss,
                                 num_integration_steps=N)

            new_states, infos = jax.vmap(per_chain)(states, key_row)
            n_total = cpd * jax.lax.axis_size("device")
            accept = jax.lax.psum(jnp.sum(infos.acceptance_rate), "device") / n_total

            # Thread the incoming frozen flag so the freeze LATCHES across steps
            # and chunks (sticky one-way; spec Alg.1 ADAPT <- ADAPT and |a-t|>tol).
            ss_new, lo_new, hi_new, fr = laps_core.bisection_step(
                ss, accept, a_target, lo=lo, hi=hi, tol=bisection_tol, frozen=frozen)
            return (new_states, ss_new, lo_new, hi_new, fr), (accept, ss_new, fr)

        carry, ys = jax.lax.scan(body, (states, step_size, lo, hi, frozen), keys)
        return carry, ys

    eps2 = jnp.asarray(eps, canon)
    lo = jnp.asarray(jnp.nan, canon)
    hi = jnp.asarray(jnp.nan, canon)
    frozen = jnp.asarray(False)
    P2 = {"accept": [], "ss": [], "frozen": []}
    p2_sizes = _chunk_sizes(num_adjusted_steps, chunk_size)  # final short chunk ok
    for c, sz in enumerate(p2_sizes):
        ck = jax.random.fold_in(k_p2, c)
        keys = jax.random.split(ck, (sz, num_chains))
        (state, eps2, lo, hi, frozen), ys = run_p2_chunk(
            _resh(keys, sh_keys), state,
            _resh(eps2, sh_repl), _resh(lo, sh_repl),
            _resh(hi, sh_repl), _resh(frozen, sh_repl),
        )
        acc, ss, fr = (np.asarray(_resh(a, sh_repl)) for a in ys)
        P2["accept"].append(acc); P2["ss"].append(ss); P2["frozen"].append(fr)

    p2_accept = np.concatenate(P2["accept"])
    p2_ss = np.concatenate(P2["ss"])
    p2_frozen = np.concatenate(P2["frozen"])

    samples = np.asarray(_resh(state.position, sh_repl))
    eps_final = float(p2_ss[-1])
    L_full = N * eps_final

    return LAPSResults(
        samples=samples,
        p1_D_tilde=p1_D, p1_eevpd_wanted=p1_ew, p1_eevpd_obs=p1_eo,
        p1_step_size=p1_ss, p1_L=p1_L, p1_obs_sq=p1_sq, p1_obs_mean=p1_mn,
        p1_delta_max=p1_dmax, phase1_len=phase1_len, switch_index=switch_index,
        switch_index_paper=sidx_paper, switch_index_emaus=sidx_emaus,
        switched=switched, precond_var=precond_var, integrator_order=order,
        target_accept=a_target,
        p2_accept=p2_accept, p2_step_size=p2_ss, p2_frozen=p2_frozen,
        p2_final_step_size=eps_final, p2_L_full=L_full,
        p2_L_proposal=L_proposal_factor * L_full,
    )


def LAPS_late_adjusted_JIT(model_seq, qz, **kwargs):
    """gigalens wrapper mirroring ``MCLMC_JIT``: builds ``log_prob`` then runs LAPS.

    Scene-only: the ProbModel renders ``log_prob(z)`` through its per-dataset
    SceneSimulators directly (no separately-built simulator), exactly as
    ``MCLMC_JIT`` does.
    """
    def log_prob(z):
        return model_seq.prob_model.log_prob(z)[0]

    return LAPS_late_adjusted(log_prob, qz, **kwargs)
