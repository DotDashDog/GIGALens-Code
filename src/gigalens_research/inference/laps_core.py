"""LAPS core ensemble reductions (paper-faithful, pure & jit-able).

Late-Adjusted Parallel Sampler (LAPS), Robnik & Seljak 2026 (arXiv:2601.16696v1).
Canonical spec: ``docs/logs/laps_spec.md``; build plan:
``docs/logs/laps_gigalens_translation.md``.

These are the small, independently-testable ensemble reductions that the LAPS
driver composes on top of the in-tree MCLMC machinery. They are deliberately
**decoupled** from sharding / gigalens / the lens model: each is a pure function
of arrays, so it unit-tests without a mesh, a kernel, or ``prob_model``. In the
production sharded driver the cross-chain means/variances here become
``psum``/``pmin('device')`` reductions over the ``'device'`` axis; the math is
identical.

Dtype discipline (mirrors ``mclmc.py``): the *energy* dtype is canonical. The
likelihood/positions may arrive as float32 while energies/step sizes are float64
under ``jax_enable_x64``; mixing them trips ``lax.select``/``cond`` dtype checks.
Each reduction casts its floating inputs to the energy dtype ``_canon`` (the
dtype of the gradient / energy-change input) so the computation is uniformly one
dtype. See ``mclmc.py`` lines ~176-188.
"""

import math

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# dtype helper                                                                 #
# --------------------------------------------------------------------------- #
def _canon_cast(energy_like, *arrays):
    """Return (canon_dtype, *arrays-cast-to-it).

    ``energy_like`` is the array that carries the canonical *energy* dtype
    (gradients, or the per-chain energy change Delta). Floating inputs are cast
    to it; non-floating inputs (e.g. integer ``d``) are passed through. This is
    the ``_canon`` invariant from ``mclmc.py``.
    """
    canon = jnp.asarray(energy_like).dtype
    if not jnp.issubdtype(canon, jnp.floating):
        canon = jnp.result_type(canon, jnp.float32)
    out = []
    for a in arrays:
        a = jnp.asarray(a)
        out.append(a.astype(canon) if jnp.issubdtype(a.dtype, jnp.floating) else a)
    return (canon, *out)


# --------------------------------------------------------------------------- #
# 1. Diagonal equipartition divergence  D-tilde   (spec A, Eqs. 4/6/18)        #
# --------------------------------------------------------------------------- #
def equipartition_diagonal(positions, grads):
    r"""Diagonal equipartition matrix and divergence ``D-tilde`` (spec Eqs. 4/6/18).

    MECHANISM. ``D-tilde`` is LAPS's **total-bias proxy** in Phase 1: it measures
    how far the current ensemble distribution ``rho`` is from the target ``p`` via
    the equipartition identity. The equipartition matrix (Eq. 4)

        V_ij = E_rho[ -(x_i - E[x_i]) * d_j log p(x) ]

    satisfies ``V = I`` iff ``rho = p`` (Eq. 5, an integration-by-parts identity).
    Its diagonal is ``E_ii = V_ii = E_rho[ -(x_i - xbar_i) * g_i ]`` where
    ``g = grad log p``. The scalar divergence (Eq. 6, diagonal form Eq. 18) is

        D-tilde = (1/d) * sum_i (1 - V_ii)^2 = mean_i( (1 - E_ii)^2 ).

    UNITS: dimensionless. ``D-tilde -> 0`` at equilibrium; ``> 0`` otherwise.
    WHAT BREAKS IF WRONG: this drives the Phase-1 step-size target ``F(C*D-tilde)``
    and (via the switch) Phase-1 termination. An over-dispersed ensemble gives
    ``V_ii > 1`` (D-tilde grows); an under-dispersed one gives ``V_ii < 1``.

    Parameters
    ----------
    positions : (M, d) array of ensemble chain locations x^m.
    grads     : (M, d) array of gradients ``grad log p(x^m)`` (energy dtype).

    Returns
    -------
    E_ii      : (d,) per-dim equipartition diagonal V_ii.
    D_tilde   : scalar, ``mean_i((1 - E_ii)^2)``.
    """
    canon, positions, grads = _canon_cast(grads, positions, grads)
    xbar = jnp.mean(positions, axis=0)                       # (d,)
    # V_ii = E_rho[ -(x_i - xbar_i) * g_i ]
    E_ii = jnp.mean(-(positions - xbar) * grads, axis=0)     # (d,)
    D_tilde = jnp.mean(jnp.square(jnp.ones((), canon) - E_ii))
    return E_ii, D_tilde


# --------------------------------------------------------------------------- #
# 2. Phase-1 step-size target law F(.) + step update   (spec A, Eq. 8 + p.4)   #
# --------------------------------------------------------------------------- #
def F_bound(D):
    r"""Asymptotic-bias bound function ``F(D) = 4 D^{3/2} / (1 + D^{1/2})^2`` (Eq. 8).

    MECHANISM. Eq. 8 bounds the asymptotic (discretization) bias by an increasing
    function of EEVPD: ``D-tilde(p, p_eps) <= F^{-1}(EEVPD)``. Inverting, the
    *wanted* EEVPD for a target asymptotic bias ``D`` is ``F(D)``. ``F`` is
    monotonically increasing, ``F(0) = 0``; small-D it ~ ``4 D^{3/2}`` (slope
    steepens like D^{1/2}), large-D it ~ ``4 D^{1/2}``. The exact exponents 3/2
    and 1/2 are verified from the rendered PDF (spec headline + Eq. 8).
    """
    (canon, D) = _canon_cast(jnp.zeros((), jnp.float64) if jax.config.jax_enable_x64
                             else jnp.zeros((), jnp.float32), D)
    sqrtD = jnp.sqrt(D)
    return 4.0 * jnp.power(D, 1.5) / jnp.square(1.0 + sqrtD)


def eevpd_wanted(D_tilde, schedule="paper", C=None):
    r"""Phase-1 target EEVPD from the equipartition divergence (spec A, step 3).

    MECHANISM. The step-size controller drives observed EEVPD toward this target
    so that asymptotic bias stays a fixed fraction ``C`` of total bias (governing
    principle, App. A Thm 1).

    schedule="paper"  (DEFAULT, verified law, spec C rows 1-2):
        ``EEVPD_wanted = F(C * D-tilde)``,  C = 0.025.
    schedule="emaus"  (predecessor-paper variant the blackjax code ships):
        ``EEVPD_wanted = C * D-tilde^{3/8}``,  C = 0.1.

    The 3/8 exponent lies OUTSIDE F's effective range [1/2, 3/2]; the two are
    structurally different laws, not a tuning constant (spec headline finding).
    """
    if C is None:
        C = 0.025 if schedule == "paper" else 0.1
    (canon, D_tilde) = _canon_cast(
        jnp.zeros((), jnp.float64) if jax.config.jax_enable_x64 else jnp.zeros((), jnp.float32),
        D_tilde)
    C = jnp.asarray(C, canon)
    if schedule == "paper":
        return F_bound(C * D_tilde)
    elif schedule == "emaus":
        return C * jnp.power(D_tilde, 0.375)
    raise ValueError(f"schedule must be 'paper' or 'emaus', got {schedule!r}")


def step_size_update(step_size, eevpd_want, eevpd_obs, clip=(0.3, 3.0), eps=1e-30):
    r"""Phase-1 step-size update ``eps <- eps * (wanted/obs)^{1/6}`` (spec A, step 5).

    MECHANISM. Empirically ``EEVPD ~ eps^6`` (p.4), so the 1/6 power gives a
    one-step Newton-like correction toward ``eevpd_want``. The multiplicative
    factor is clipped to ``clip`` (code-only safety rail [0.3, 3.0], spec C row 8)
    to keep one bad ensemble estimate from blowing eps up/down by orders of
    magnitude. UNITS: ``step_size`` in position units; the ratio is dimensionless.
    """
    canon, step_size, eevpd_want, eevpd_obs = _canon_cast(
        step_size, step_size, eevpd_want, eevpd_obs)
    ratio = (eevpd_want + eps) / (eevpd_obs + eps)
    factor = jnp.power(ratio, 1.0 / 6.0)
    lo, hi = clip
    factor = jnp.clip(factor, jnp.asarray(lo, canon), jnp.asarray(hi, canon))
    return step_size * factor


# --------------------------------------------------------------------------- #
# 3. Ensemble EEVPD     (spec A, Eq. 7)                                        #
# --------------------------------------------------------------------------- #
def ensemble_eevpd(energy_change, d):
    r"""Ensemble EEVPD = ``Var_rho[Delta] / d`` (spec Eq. 7).

    MECHANISM. EEVPD (Energy Error Variance Per Dimension) is LAPS's
    **asymptotic-bias proxy**. Eq. 7: ``EEVPD = Var_rho[Delta(x,u|p,eps)] / d``,
    the variance ACROSS THE ENSEMBLE of the per-chain integrator energy error
    Delta, divided by dimension. NOTE this is the paper's *ensemble-instantaneous*
    cross-chain variance, NOT the in-tree per-chain decayed running average in
    ``mclmc.py:step_size_adapt`` (which the translation doc flags as a different
    estimator, §C / open-Q4). UNITS: (energy)^2 per dim.

    Parameters
    ----------
    energy_change : (M,) per-chain energy errors Delta (energy dtype).
    d             : int, problem dimension.

    Returns
    -------
    scalar EEVPD estimate ``var(Delta) / d`` (population variance, ddof=0,
    matching ``Var_rho`` as a distribution moment over the M-chain ensemble).
    """
    canon, energy_change = _canon_cast(energy_change, energy_change)
    var_delta = jnp.var(energy_change, ddof=0)
    return var_delta / jnp.asarray(d, canon)


# --------------------------------------------------------------------------- #
# 4. Decoherence / trajectory length L   (spec A, Eq. 9)                       #
# --------------------------------------------------------------------------- #
def decoherence_length(positions, alpha=2.0):
    r"""Trajectory length ``L = alpha * sqrt(sum_i Var_rho[x_i])`` (spec Eq. 9).

    MECHANISM. Sets the velocity decoherence scale (partial-refresh
    ``c1 = exp(-eps/L)``). Eq. 9 with ``alpha = 2`` (p.5: "a larger value alpha=2
    works better"; alpha=1 is the MCLMC default). Recomputed each Phase-1 step
    from the ensemble's per-coordinate variance. UNITS: position units.
    WHAT BREAKS IF WRONG: too-small L over-refreshes (diffusive, slow); too-large
    L under-refreshes (chains decohere slowly, correlated).
    """
    canon, positions = _canon_cast(
        jnp.zeros((), jnp.float64) if jax.config.jax_enable_x64 else jnp.zeros((), jnp.float32),
        positions)
    var_i = jnp.var(positions, axis=0, ddof=0)              # (d,)  Var_rho[x_i]
    return jnp.asarray(alpha, canon) * jnp.sqrt(jnp.sum(var_i))


# --------------------------------------------------------------------------- #
# 5. Phase-1 -> Phase-2 switch detector   (spec A, Eqs. 10-11)                 #
# --------------------------------------------------------------------------- #
def ensemble_mean_observable(positions, switch="paper"):
    r"""Per-step ensemble mean of the switch observable f (spec Eq. 10 inner mean).

    paper : f(x) = x_i^2  -> returns E_rho[x_i^2]  (mean of squares).
    emaus : f(x) = x_i    -> returns E_rho[x_i].

    This is the cross-CHAIN reduction (a ``psum`` over devices in production) that
    is then buffered across STEPS for the windowed sigma/mu in ``phase1_switch``.
    Note E_rho[x^2] != (E_rho[x])^2 -- the observable choice is load-bearing.
    """
    canon, positions = _canon_cast(
        jnp.zeros((), jnp.float64) if jax.config.jax_enable_x64 else jnp.zeros((), jnp.float32),
        positions)
    if switch == "paper":
        return jnp.mean(jnp.square(positions), axis=0)      # E[x_i^2], (d,)
    elif switch == "emaus":
        return jnp.mean(positions, axis=0)                  # E[x_i],   (d,)
    raise ValueError(f"switch must be 'paper' or 'emaus', got {switch!r}")


def phase1_switch(obs_mean_window, switch="paper", threshold=0.01,
                  switch_mode="self_calibrated", k=1.5, k_m=2.0,
                  floor=None, M=None):
    r"""Windowed relative-fluctuation switch test (spec Eqs. 10-11, §4).

    MECHANISM. Over a trailing window of T steps (T = 20% of total sampling time)
    of the per-step ensemble-mean observable ``mu_t[f] = (1/T) sum E_rho_s[f]``,
    compute the relative fluctuation

        delta_i = sigma_i / mu_i,   sigma_i^2 = (1/(T-1)) sum (E_rho_s[f] - mu_i)^2.

    EQUILIBRIUM NOISE FLOOR (spec switch-resolution doc). At stationarity delta_i
    does NOT vanish: it floors at ``floor_i = sqrt(Var_p[f]/M)/E_p[f]``, which for
    a zero-mean unit-variance coordinate with f=x_i^2 is ``sqrt(2/M)`` -- about
    0.022 at M=4096 and 0.063 at M=512. ``max_i`` inflates this a further ~8-20%
    over d. The paper's literal ``delta<0.01`` is therefore only reachable for
    M >~ 2e4; at the M~1e2-1e3 GIGALens uses it can NEVER fire (Phase 1 would
    silently burn maxiter). The paper writes the floor as "~ M^{-1/2}", dropping
    the ``sqrt(Var_p[f])/E_p[f]`` prefactor; the operative paper behaviour is
    "fire on the transient as delta decays through 0.01, else hit maxiter".

    Observable flag (orthogonal to ``switch_mode``):
      paper : f = x_i^2, statistic ``delta = sigma/mu`` (well-conditioned, mu~Var>0).
      emaus : f = x_i,   statistic ``r = (sigma/mu)^2`` (ill-conditioned on a
              mean-zero coord, mu~0 -> blows up -> NEVER fires; spec C rows 3-4).
    The caller must build ``obs_mean_window`` with the MATCHING observable via
    ``ensemble_mean_observable(..., switch)``.

    Fire criterion (``switch_mode``):
      "self_calibrated" (DEFAULT): fire when ``max_i(delta_i / floor_i) < k`` --
          "fluctuations within k x of their irreducible Monte-Carlo floor". M-,
          observable- and posterior-invariant; reduces to the paper's M^{-1/2}
          heuristic with the prefactor kept. Default ``k=1.5``. Requires ``floor``
          (per-dim, ``sqrt(Var_rho[f]/M)/E_rho[f]`` from the CURRENT ensemble).
      "absolute": literal ``max_i delta_i < threshold`` (paper-reproduction /
          large-M). Default ``threshold=0.01``. If ``M`` is supplied and
          ``threshold < sqrt(2/M)`` this RAISES -- the config can never fire and
          would silently burn maxiter (project-standards: no silent no-op default).
      "m_scaled": ``max_i delta_i < k_m * M^{-1/2}`` (explicit M-aware constant).
          Default ``k_m=2.0`` (~1.4x above floor). Requires ``M``.

    Parameters
    ----------
    obs_mean_window : (T, d) buffered per-step ensemble means of the observable.
    switch          : "paper" | "emaus".
    threshold       : absolute-mode fire threshold (0.01).
    switch_mode     : "self_calibrated" | "absolute" | "m_scaled".
    k               : self_calibrated floor multiple (fire when delta < k*floor).
    k_m             : m_scaled coefficient (threshold = k_m / sqrt(M)).
    floor           : (d,) per-dim noise floor (required for self_calibrated).
    M               : ensemble size (required for m_scaled; enables absolute guard).

    Returns
    -------
    delta     : (d,) per-dim statistic.
    delta_max : scalar max_i delta_i.
    fired     : bool.
    """
    canon, obs_mean_window = _canon_cast(
        jnp.zeros((), jnp.float64) if jax.config.jax_enable_x64 else jnp.zeros((), jnp.float32),
        obs_mean_window)
    mu = jnp.mean(obs_mean_window, axis=0)                  # (d,) Eq. 10
    sigma = jnp.std(obs_mean_window, axis=0, ddof=1)        # (d,) Eq. 11 (1/(T-1))
    ratio = sigma / mu                                      # sigma/mu (can blow up)
    if switch == "paper":
        delta = jnp.abs(ratio)
    elif switch == "emaus":
        delta = jnp.square(ratio)
    else:
        raise ValueError(f"switch must be 'paper' or 'emaus', got {switch!r}")
    delta_max = jnp.max(delta)

    if switch_mode == "self_calibrated":
        if floor is None:
            raise ValueError(
                "switch_mode='self_calibrated' requires a per-dim `floor` "
                "(= sqrt(Var_rho[f]/M)/E_rho[f] from the current ensemble).")
        floor = jnp.asarray(floor, canon)
        # guard floor -> 0: a zero floor cannot gate the switch (set crit -> +inf).
        safe_floor = jnp.where(floor > 0, floor, jnp.asarray(jnp.inf, canon))
        crit = jnp.max(delta / safe_floor)
        fired = crit < jnp.asarray(k, canon)
    elif switch_mode == "absolute":
        if M is not None and float(threshold) < math.sqrt(2.0 / float(M)):
            raise ValueError(
                f"absolute switch threshold {threshold} is below the equilibrium "
                f"noise floor sqrt(2/M)={math.sqrt(2.0/float(M)):.4g} at M={M}: the "
                f"switch can never fire and Phase 1 would silently burn maxiter. Use "
                f"switch_mode='self_calibrated' (default) or 'm_scaled' for small M, "
                f"or raise M >~ 2e4 to reproduce the paper's literal 0.01.")
        fired = delta_max < jnp.asarray(threshold, canon)
    elif switch_mode == "m_scaled":
        if M is None:
            raise ValueError("switch_mode='m_scaled' requires M.")
        thr = jnp.asarray(k_m, canon) * jnp.asarray(float(M), canon) ** (-0.5)
        fired = delta_max < thr
    else:
        raise ValueError(
            "switch_mode must be 'self_calibrated'|'absolute'|'m_scaled', "
            f"got {switch_mode!r}")
    return delta, delta_max, fired


def persistence_update(fired, consecutive, switch_persist=1):
    r"""Consecutive-fire counter for the OPTIONAL switch persistence guard.

    MECHANISM. ``phase1_switch`` is a *stateless* per-window ripeness test; a single
    lucky window can fire it even when the ensemble is not yet equilibrated (the
    off-qz early-switch failure mode). The persistence guard requires the ripeness
    criterion ``max_i(delta_i/floor_i) < k`` to hold for ``switch_persist``
    CONSECUTIVE switch-evaluations (chunks) before the Phase-1 -> Phase-2 switch is
    allowed to fire. This is a pure, host-side integer reducer threaded by the
    driver across chunks.

    ``consecutive`` is incremented on a ``fired`` evaluation and RESET to 0 on a
    miss (the fires must be back-to-back). The switch is permitted once the count
    reaches ``switch_persist``.

    DEFAULT-OFF INVARIANT: ``switch_persist=1`` returns ``do_switch=fired`` on every
    call (fire-on-first), exactly reproducing the pre-guard behaviour.

    Parameters
    ----------
    fired         : bool, the current window's ripeness test result.
    consecutive   : int, consecutive prior fires (0 at Phase-1 start).
    switch_persist: int >= 1, required consecutive fires (1 = no guard).

    Returns
    -------
    do_switch        : bool, whether the switch is now permitted.
    consecutive_next : int, updated consecutive-fire count.
    """
    consecutive_next = consecutive + 1 if fired else 0
    return (consecutive_next >= int(switch_persist)), consecutive_next


# --------------------------------------------------------------------------- #
# 6. Phase-2 bisection step tuner   (spec A, §5; B target-accept rows)         #
# --------------------------------------------------------------------------- #
def target_accept(integrator_order):
    """Target acceptance: 0.7 for 2nd-order (MN2), 0.9 for 4th-order (MN4) (spec p.6)."""
    return 0.7 if int(integrator_order) == 2 else 0.9


def bisection_step(step_size, accept, target, lo=jnp.nan, hi=jnp.nan, tol=0.03,
                   frozen=False, freeze_enable=True, eps_min=None, eps_max=None,
                   growth=2.5):
    r"""One Phase-2 bracketing-bisection step toward target acceptance (spec §5, p.6).

    MECHANISM. Phase 2 tunes eps to a target average acceptance by a two-phase
    **bracket-then-bisect** on ``a(eps) - a_target`` (no dual averaging -- low noise
    at large M). Acceptance is monotonically DECREASING in eps (bigger step ->
    larger energy error -> lower accept), so the search direction is fixed by sign:

      * a > target  (accept too HIGH)  => eps too SMALL  => INCREASE eps.
      * a < target  (accept too LOW)   => eps too BIG    => DECREASE eps.

    PHASE i -- BRACKETING (target not yet straddled, i.e. we lack EITHER a known
        a>target eps or a known a<target eps): EXPAND geometrically by ``growth``
        in the correct direction -- ``eps*growth`` when a>target (grow toward the
        larger step the phase needs), ``eps/growth`` when a<target. This is the fix
        for the warm-start failure where eps started far BELOW the needed step and
        a one-sided search never reached it: the expansion walks eps UP (or down)
        by a constant geometric factor until the target acceptance is bracketed.
    PHASE ii -- REFINE (target bracketed: we hold one eps with a>target = ``lo``
        and one with a<target = ``hi``): bisect at the GEOMETRIC midpoint
        ``sqrt(lo*hi)`` (scale-invariant; appropriate for a multiplicative step).

    **Freeze when |a - a_target| <= tol** (3% default) -- freezing removes ECA bias.

    ONE-WAY LATCH (spec Algorithm 1: ``ADAPT <- ADAPT and |a - a_target| > tol``).
    The freeze is STICKY: once within tol it stays frozen FOREVER, even if the
    ensemble's acceptance later drifts back outside tol. This is what "freezing
    removes ECA bias" means -- eps becomes a hard constant, fully decoupled from
    the ongoing ensemble. The incoming ``frozen`` flag is threaded by the caller
    and OR'd in: ``frozen_out = frozen_in or (|a - target| <= tol)``. While
    ``frozen_in`` is True, eps and the bracket (lo, hi) are HELD unchanged (no
    further bisection or bracket updates), so eps stops being a function of the
    ensemble.

    Bracket convention: ``lo`` is the largest eps known to give a >= target (a
    lower bound on the solution); ``hi`` the smallest eps known to give a < target.
    NaN means "not yet established". Pure / jit-able (jnp.where, no python branch
    on traced values), so it composes into a fixed-maxiter scan.

    Parameters
    ----------
    step_size : current eps.
    accept    : observed ensemble-mean acceptance a(eps) in [0, 1].
    target    : a_target (0.7 MN2 / 0.9 MN4).
    lo, hi    : current bracket (NaN if unset).
    tol       : freeze band (0.03).
    frozen    : incoming latch (bool); once True, eps/bracket are held.
    freeze_enable : external gate on a NEWLY firing freeze (bool, default True).
        The latch may fire on this step only if ``freeze_enable`` is True. This is
        how the driver imposes the windowed + persistent + min-step freeze
        conditions (Fix 2): the in-band test is still computed here, but committing
        the freeze is deferred until the caller's gate opens. ``freeze_enable=True``
        (the default) reproduces the original "freeze on first in-band" behaviour,
        so every existing caller is unchanged. An ALREADY-set ``frozen`` latch is
        sticky regardless of ``freeze_enable`` (one-way latch preserved).
    eps_min, eps_max : OPTIONAL safety rail on the ADAPTING step (default None =
        no clamp, so every existing caller is unchanged). When supplied, the next
        step is clamped into ``[eps_min, eps_max]`` BEFORE the freeze-hold, so a
        transient or over-confident bracket cannot run eps away to 0 or infinity
        during the geometric bracketing search. A frozen eps is still held EXACTLY
        (the clamp applies only to the adapting branch), preserving the one-way
        latch / hold-eps guarantee. The rail must be WIDE enough not to block a
        legitimate bracketing walk (the driver sets ~3 decades each side of eps0).
    growth : geometric expansion factor for the bracketing phase (default 2.5).
        Each unbracketed step multiplies/divides eps by this factor toward the
        target; must be > 1. The default was raised from 1.7 to 2.5 (CHANGE A) so
        the bracketing walk CROSSES a flat ``accept~1.0`` plateau -- which can
        span ~10x in eps when Phase 2 starts from a too-small handoff -- in a few
        chunks (log_2.5(10) ~ 2.5) instead of the ~7 doublings 1.7 needed, leaving
        the remaining Phase-2 chunk budget for the bisection refine + freeze.

    Returns
    -------
    step_size_next, lo_next, hi_next, frozen_out(bool). ``frozen_out`` is sticky.
    """
    canon, step_size, accept, target, lo, hi = _canon_cast(
        step_size, step_size, accept, target, lo, hi)
    frozen_in = jnp.asarray(frozen, jnp.bool_)
    freeze_enable = jnp.asarray(freeze_enable, jnp.bool_)
    a_high = accept > target                     # accept too high => eps too small
    # Update bracket: if a_high, eps is a lower bound (lo, the largest eps known to
    # give a>=target); else an upper bound (hi, the smallest eps known to give
    # a<target). DIRECTION (monotone-decreasing a): a_high => eps too small => the
    # solution lies ABOVE step_size, so step_size is a lower bound; confirmed below.
    lo_upd = jnp.where(a_high, step_size, lo)
    hi_upd = jnp.where(a_high, hi, step_size)
    have_lo = ~jnp.isnan(lo_upd)
    have_hi = ~jnp.isnan(hi_upd)
    both = have_lo & have_hi
    # PHASE ii (refine): geometric midpoint of the VERIFIED bracket [lo, hi].
    eps_bisect = jnp.sqrt(lo_upd * hi_upd)
    # PHASE i (bracketing): target not yet straddled -> EXPAND geometrically by
    # ``growth`` in the correct direction. a_high (eps too small) GROWS eps toward
    # the larger step the adjusted phase needs; a<target SHRINKS it. This walks eps
    # UP from a too-small handoff until a<target is finally observed (the fix), and
    # symmetrically DOWN from a too-large start.
    g = jnp.asarray(growth, canon)
    eps_expand = jnp.where(a_high, step_size * g, step_size / g)
    eps_step = jnp.where(both, eps_bisect, eps_expand)
    # OPTIONAL safety rail: clamp the ADAPTING step into [eps_min, eps_max] so a
    # transient / over-confident bracket cannot run eps away during the geometric
    # bracketing walk. Applied to eps_step only; a frozen eps is held below.
    if eps_min is not None:
        eps_step = jnp.maximum(eps_step, jnp.asarray(eps_min, canon))
    if eps_max is not None:
        eps_step = jnp.minimum(eps_step, jnp.asarray(eps_max, canon))
    # Sticky one-way latch: freeze once in band, stay frozen forever. A NEW freeze
    # is gated by ``freeze_enable`` (driver's windowed+persistent+min-step test);
    # an already-set latch (frozen_in) stays sticky regardless of the gate.
    in_band = jnp.abs(accept - target) <= jnp.asarray(tol, canon)
    frozen_out = frozen_in | (in_band & freeze_enable)
    # Hold eps when (already or newly) frozen; hold the bracket once already frozen.
    eps_next = jnp.where(frozen_out, step_size, eps_step)
    lo_next = jnp.where(frozen_in, lo, lo_upd)
    hi_next = jnp.where(frozen_in, hi, hi_upd)
    return eps_next, lo_next, hi_next, frozen_out
