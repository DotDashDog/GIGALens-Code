"""Unit tests for laps_core.py with PRE-REGISTERED expected values.

Each test states the expected value/direction (derived analytically) BEFORE the
assert, and PRINTS observed-vs-expected. Run as a script in the CPU container:

    JAX_ENABLE_X64=1 python3 test_laps_core.py

Rigor note (method discipline): D-tilde / EEVPD recovery are finite-sample
STOCHASTIC-ESTIMATOR claims -- tested at the large-M limit with tolerances
derived from the estimator's own sampling variance, not arbitrary bands. F(.) is
a DETERMINISTIC IDENTITY -- tested to solver tolerance. The D1 falsifier is a
CONDITIONING claim about the switch statistic -- tested by construction.
"""

import sys
import math
import jax
import jax.numpy as jnp
import numpy as np

from gigalens_research.inference.laps_core import (
    equipartition_diagonal,
    F_bound,
    eevpd_wanted,
    step_size_update,
    ensemble_eevpd,
    decoherence_length,
    ensemble_mean_observable,
    phase1_switch,
    bisection_step,
    target_accept,
)

_FAILS = []


def check(name, cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}: {msg}")
    if not cond:
        _FAILS.append(name)


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
def test_equipartition():
    hdr("TEST 1  Diagonal equipartition D-tilde  (spec Eqs. 4/6/18)")
    key = jax.random.PRNGKey(0)
    M, d = 400_000, 8
    # (a) ensemble drawn EXACTLY from target N(0, I); grad log p = -x.
    # PRE-REGISTERED: at equilibrium V_ii -> 1, D-tilde -> 0. The estimator
    # V_ii = (1/M) sum (x_i - xbar_i) x_i is a sample variance; its sampling
    # std ~ sqrt(2/M) (Gaussian), so E[D-tilde] ~ Var(V_ii) ~ 2/M.
    expected_Dtilde_eq = 2.0 / M
    tol_eq = 5e-4  # ~100x the 2/M=5e-6 expectation; comfortable upper bound
    x = jax.random.normal(key, (M, d))
    g = -x
    E_ii, D = equipartition_diagonal(x, g)
    print(f"  (a) equilibrium  N(0,I): E_ii(mean over dim)={float(jnp.mean(E_ii)):.6f} "
          f"(expect ~1.0)")
    print(f"      D-tilde observed = {float(D):.3e}  | pre-registered ~2/M = "
          f"{expected_Dtilde_eq:.3e}  | tol < {tol_eq:.0e}")
    check("D-tilde@equilibrium->0", float(D) < tol_eq,
          f"{float(D):.3e} < {tol_eq:.0e}")

    # (b) DELIBERATELY under-equilibrated: variance off by 2x (over-dispersed),
    # grad still for target N(0,I). PRE-REGISTERED: V_ii -> Var = 2, so
    # (1 - V_ii)^2 -> 1, D-tilde -> ~1.0 (materially > 0). DIRECTION: V_ii > 1.
    x2 = jnp.sqrt(2.0) * jax.random.normal(jax.random.PRNGKey(1), (M, d))
    g2 = -x2
    E_ii2, D2 = equipartition_diagonal(x2, g2)
    print(f"  (b) over-dispersed (Var=2): E_ii(mean)={float(jnp.mean(E_ii2)):.4f} "
          f"(expect ~2.0, i.e. >1)")
    print(f"      D-tilde observed = {float(D2):.4f}  | pre-registered ~1.0  "
          f"| require > 0.5")
    check("D-tilde under-equilibrated material", float(D2) > 0.5,
          f"{float(D2):.4f} > 0.5 in predicted direction (V_ii~2>1)")


# --------------------------------------------------------------------------- #
def test_F_and_step_laws():
    hdr("TEST 2  F(.) closed form + paper-vs-EMAUS step law  (spec Eq. 8, C r1-2)")
    # F(D) = 4 D^1.5 / (1+sqrt(D))^2. Analytic anchors:
    #   F(0) = 0; F(1) = 4/(2^2) = 1; F(4) = 4*8/(3^2) = 32/9 = 3.55556
    f0, f1, f4 = float(F_bound(0.0)), float(F_bound(1.0)), float(F_bound(4.0))
    print(f"  F(0)={f0:.3e} (expect 0)   F(1)={f1:.6f} (expect 1.0)   "
          f"F(4)={f4:.6f} (expect {32/9:.6f})")
    check("F(0)=0", abs(f0) < 1e-12, f"{f0:.3e}")
    check("F(1)=1", abs(f1 - 1.0) < 1e-12, f"{f1:.12f}")
    check("F(4)=32/9", abs(f4 - 32 / 9) < 1e-12, f"{f4:.12f}")
    grid = jnp.array([0.0, 1e-4, 1e-2, 0.1, 1.0, 4.0, 16.0, 100.0])
    Fg = F_bound(grid)
    mono = bool(jnp.all(jnp.diff(Fg) > 0))
    print(f"  F strictly increasing on {list(map(float, grid))} : {mono}")
    check("F strictly increasing", mono, "all diffs > 0")

    # paper vs EMAUS near convergence (small D-tilde): paper EEVPD_wanted ∝ D^1.5
    # must be SMALLER than EMAUS ∝ D^0.375 -> faster eps shrink.
    print("\n  EEVPD_wanted: paper F(0.025 D) vs EMAUS 0.1 D^0.375")
    print("   D-tilde     paper           emaus          paper<emaus")
    cross_prev = None
    crossover = None
    for D in [1e-3, 1e-2, 0.1, 1.0, 10.0, 20.0, 50.0]:
        p = float(eevpd_wanted(D, "paper"))
        e = float(eevpd_wanted(D, "emaus"))
        rel = "paper<emaus" if p < e else "paper>emaus"
        print(f"   {D:<9.3g}  {p:.6e}   {e:.6e}   {rel}")
        cur = p < e
        if cross_prev is not None and cur != cross_prev:
            crossover = D
        cross_prev = cur
    p_small = float(eevpd_wanted(1e-3, "paper"))
    e_small = float(eevpd_wanted(1e-3, "emaus"))
    print(f"  At D=1e-3 (near convergence): paper={p_small:.3e} << emaus="
          f"{e_small:.3e}  (paper gives smaller target -> faster eps shrink)")
    print(f"  Crossover (paper overtakes emaus) near D-tilde ~ {crossover}")
    check("paper<emaus at small D", p_small < e_small,
          f"{p_small:.3e} < {e_small:.3e}")

    # step update direction: smaller wanted -> smaller eps (since eps ∝ ratio^1/6).
    eps_new = float(step_size_update(1.0, eevpd_want=1e-7, eevpd_obs=1e-3))
    print(f"  step_size_update(eps=1, wanted<obs) -> {eps_new:.4f} (expect <1, "
          f"clipped at 0.3)")
    check("step shrinks when wanted<obs", eps_new < 1.0, f"{eps_new:.4f} < 1")


# --------------------------------------------------------------------------- #
def test_ensemble_eevpd():
    hdr("TEST 3  Ensemble EEVPD = Var[Delta]/d  (spec Eq. 7)")
    # PRE-REGISTERED: Delta ~ N(0, s2), EEVPD -> s2/d. Sampling rel-err on the
    # variance ~ sqrt(2/M); use 5% band (>> sqrt(2/5e5)=0.002).
    M, d, s2 = 500_000, 10, 0.5
    delta = jnp.sqrt(s2) * jax.random.normal(jax.random.PRNGKey(2), (M,))
    obs = float(ensemble_eevpd(delta, d))
    exp = s2 / d
    print(f"  Var[Delta]={s2}, d={d} -> EEVPD expected {exp:.5f}; observed "
          f"{obs:.5f}; rel-err {abs(obs-exp)/exp:.2%} (tol 5%)")
    check("EEVPD recovers Var/d", abs(obs - exp) / exp < 0.05,
          f"{obs:.5f} vs {exp:.5f}")


# --------------------------------------------------------------------------- #
def test_decoherence_length():
    hdr("TEST 4  Decoherence length L = alpha sqrt(sum Var[x_i])  (spec Eq. 9)")
    # PRE-REGISTERED: positions ~ N(0, I) in d dims -> sum Var = d, L = alpha*sqrt(d).
    M, d = 200_000, 9
    x = jax.random.normal(jax.random.PRNGKey(3), (M, d))
    L = float(decoherence_length(x, alpha=2.0))
    exp = 2.0 * math.sqrt(d)
    print(f"  N(0,I) d={d}: L expected alpha*sqrt(d)={exp:.4f}; observed {L:.4f}; "
          f"rel-err {abs(L-exp)/exp:.2%}")
    check("L = alpha sqrt(sum Var)", abs(L - exp) / exp < 0.01,
          f"{L:.4f} vs {exp:.4f}")


# --------------------------------------------------------------------------- #
def test_switch_and_D1_falsifier():
    hdr("TEST 5  D1 FALSIFIER: switch conditioning on a mean-zero coordinate "
        "(spec C r3-4)")
    # Mechanism: a mean-zero coordinate at equilibrium has time-averaged ensemble
    # mean mu ~ 0 (the grand mean wanders through 0). The EMAUS switch r=(sigma/mu)^2
    # on the IDENTITY observable x_i is then ill-conditioned (mu in the denominator)
    # and to FIRE needs r<0.01 i.e. |mu|>10*sigma -- impossible for a mean-zero
    # coord. The PAPER switch delta=sigma/mu on x_i^2 has mu = E[x^2] ~ Var > 0,
    # well-conditioned, and fires at delta ~ M^{-1/2} < 0.01.
    #
    # Construct the near-cancellation regime the spec pre-registers (|mu| ~ 1e-3*sigma):
    M, T = 50_000, 40
    sx = 1.0  # coordinate std
    s_mean = sx / math.sqrt(M)          # std of per-step ensemble mean E[x]
    s_sq = math.sqrt(2.0) * sx**2 / math.sqrt(M)  # std of per-step E[x^2]

    def standardize(v):
        return (v - jnp.mean(v)) / jnp.std(v, ddof=1)

    z = standardize(jax.random.normal(jax.random.PRNGKey(7), (T,)))

    # EMAUS window: identity observable E[x] time-series, mean = 1e-3*sigma (deep
    # but realistic cancellation for a mean-zero coord), std = s_mean.
    mu_emaus = 1e-3 * s_mean
    emaus_win = (mu_emaus + s_mean * z)[:, None]            # (T, 1)
    # switch_mode="absolute": this test isolates the absolute-threshold CONDITIONING
    # of the emaus (x_i) observable, orthogonal to the self_calibrated default.
    d_em, dmax_em, fired_em = phase1_switch(
        emaus_win, switch="emaus", switch_mode="absolute")
    # expected (sigma/mu)^2 = (s_mean / (1e-3 s_mean))^2 = (1e3)^2 = 1e6
    print(f"  EMAUS  (identity x_i): mu={float(jnp.mean(emaus_win)):.3e}, "
          f"sigma={float(jnp.std(emaus_win,ddof=1)):.3e}")
    print(f"         statistic (sigma/mu)^2 = {float(dmax_em):.3e}  (pre-reg ~1e6) "
          f"-> fired={bool(fired_em)} (threshold 0.01) => NEVER FIRES")
    check("EMAUS statistic blows up >=1e5", float(dmax_em) >= 1e5,
          f"{float(dmax_em):.3e} >= 1e5")
    check("EMAUS never fires on mean-zero coord", not bool(fired_em),
          f"fired={bool(fired_em)}")

    # PAPER window: x_i^2 observable E[x^2] time-series, mean = Var = sx^2 = 1,
    # std = s_sq. delta = sigma/mu = s_sq / 1 = sqrt(2/M).
    paper_win = (sx**2 + s_sq * z)[:, None]                 # (T, 1)
    d_pa, dmax_pa, fired_pa = phase1_switch(
        paper_win, switch="paper", switch_mode="absolute")
    exp_delta = math.sqrt(2.0 / M)
    print(f"  PAPER  (x_i^2): mu={float(jnp.mean(paper_win)):.4f}, "
          f"sigma={float(jnp.std(paper_win,ddof=1)):.3e}")
    print(f"         statistic delta=sigma/mu = {float(dmax_pa):.3e}  (pre-reg "
          f"sqrt(2/M)={exp_delta:.3e}) -> fired={bool(fired_pa)} => FIRES (<0.01)")
    check("PAPER statistic stays O(small) <0.01", float(dmax_pa) < 0.01,
          f"{float(dmax_pa):.3e} < 0.01")
    check("PAPER fires at equilibrium", bool(fired_pa), f"fired={bool(fired_pa)}")

    # Sanity: the SAME observable construction, identity reduction from raw positions.
    raw = sx * jax.random.normal(jax.random.PRNGKey(8), (M, 1))
    obs_paper = ensemble_mean_observable(raw, "paper")
    obs_emaus = ensemble_mean_observable(raw, "emaus")
    print(f"  reducer sanity: E[x^2]={float(obs_paper[0]):.4f} (~1), "
          f"E[x]={float(obs_emaus[0]):.3e} (~0)")


# --------------------------------------------------------------------------- #
def test_bisection():
    hdr("TEST 6  Phase-2 bisection step tuner toward target accept (spec §5)")
    # Synthetic MONOTONE decreasing accept(eps) = exp(-eps). target 0.7 (2nd-order).
    # PRE-REGISTERED: converges to |a-target|<=0.03 in a BOUNDED number of steps
    # (<50). Solution eps* = -ln(0.7) = 0.35667 (a freeze can occur earlier in band).
    tgt = target_accept(2)
    print(f"  target_accept(order=2) = {tgt} (expect 0.7); order=4 -> "
          f"{target_accept(4)} (expect 0.9)")
    check("target accept values", tgt == 0.7 and target_accept(4) == 0.9, "0.7 / 0.9")

    accept_fn = lambda eps: math.exp(-eps)
    eps = 1.0
    lo, hi = float("nan"), float("nan")
    frozen = False
    bstep = jax.jit(lambda e, a, lo, hi: bisection_step(e, a, tgt, lo, hi))
    traj = []
    for i in range(50):
        a = accept_fn(eps)
        e_next, lo, hi, fr = bstep(eps, a, lo, hi)
        traj.append((i, eps, a, float(lo), float(hi), bool(fr)))
        frozen = bool(fr)
        if frozen:
            break
        eps = float(e_next)
    print("   step   eps        accept     lo         hi        frozen")
    for (i, e, a, l, h, fr) in traj:
        print(f"   {i:<4d}  {e:.6f}   {a:.5f}   {l:<9.4g}  {h:<9.4g}  {fr}")
    final_eps, final_a = traj[-1][1], traj[-1][2]
    print(f"  converged: eps={final_eps:.5f}, accept={final_a:.5f}, "
          f"|a-target|={abs(final_a-tgt):.4f} (<=0.03), steps={len(traj)}")
    check("bisection freezes within 3% band", frozen and abs(final_a - tgt) <= 0.03,
          f"|{final_a:.4f}-{tgt}|={abs(final_a-tgt):.4f} in {len(traj)} steps")
    check("bisection bounded steps", len(traj) < 50, f"{len(traj)} < 50")


# --------------------------------------------------------------------------- #
def test_bisection_latch():
    hdr("TEST 7  Phase-2 freeze is a STICKY one-way latch (spec Alg.1; Fix 1)")
    # PRE-REGISTERED (grader Faithfulness-Gap #1): with the latch, the sequence
    # accept=[0.71, 0.55, 0.95] at target 0.7, tol 0.03 must freeze on step 0
    # (|0.71-0.7|=0.01<=0.03) and STAY frozen with eps CONSTANT, even though steps
    # 1-2 drift outside the band. FALSIFIER: any un-freeze (frozen flips back to
    # False) or any eps change after the first freeze.
    tgt = 0.7
    eps0 = 1.0
    seq = [0.71, 0.55, 0.95]
    eps = eps0
    lo, hi = float("nan"), float("nan")
    frozen = False
    rows = []
    eps_after_freeze = []
    for a in seq:
        e_next, lo, hi, fr = bisection_step(eps, a, tgt, lo=lo, hi=hi, tol=0.03,
                                            frozen=frozen)
        fr = bool(fr)
        rows.append((a, eps, float(e_next), fr))
        if frozen:                       # was already frozen entering this step
            eps_after_freeze.append((eps, float(e_next)))
        frozen = fr
        eps = float(e_next)
    print("   accept   eps_in     eps_out    frozen_out")
    for (a, ein, eout, fr) in rows:
        print(f"   {a:.3f}    {ein:.6f}   {eout:.6f}   {fr}")
    all_frozen = all(r[3] for r in rows)
    # eps must equal eps0 at every output once frozen (held constant)
    eps_const = all(abs(eo - eps0) < 1e-12 for (_, eo) in
                    [(None, r[2]) for r in rows])
    print(f"  frozen stays True for all 3 steps : {all_frozen}")
    print(f"  eps held == eps0={eps0} after freeze: {eps_const} "
          f"(outputs {[round(r[2],6) for r in rows]})")
    check("freeze latches (stays True)", all_frozen, f"{[r[3] for r in rows]}")
    check("eps constant after freeze", eps_const,
          f"outputs {[round(r[2],6) for r in rows]} all == {eps0}")


# --------------------------------------------------------------------------- #
def test_bisection_freeze_gate():
    hdr("TEST 7b  Phase-2 freeze GATE: freeze_enable defers the latch (Fix 2)")
    # PRE-REGISTERED: the freeze must NOT latch while freeze_enable=False, even with
    # accept exactly in-band (this is how the driver enforces windowed+persistent+
    # min-step conditions so a transient cannot freeze a too-large eps). FALSIFIER:
    # any freeze (frozen_out True) while freeze_enable=False on an unfrozen state.
    tgt = 0.7
    eps0 = 1.0
    lo, hi = float("nan"), float("nan")
    # (a) in-band but gate CLOSED -> no latch, eps keeps adapting.
    e1, lo1, hi1, fr1 = bisection_step(eps0, 0.71, tgt, lo=lo, hi=hi, tol=0.03,
                                       frozen=False, freeze_enable=False)
    print(f"  gate CLOSED, accept=0.71 in-band: frozen_out={bool(fr1)} "
          f"(expect False), eps {eps0}->{float(e1):.4f} (adapts)")
    check("freeze_enable=False blocks the latch", not bool(fr1), f"fr={bool(fr1)}")
    check("eps still adapts when gate closed", abs(float(e1) - eps0) > 1e-9,
          f"eps {eps0}->{float(e1)}")
    # (b) gate OPEN + in-band -> latch fires.
    e2, lo2, hi2, fr2 = bisection_step(eps0, 0.71, tgt, lo=lo, hi=hi, tol=0.03,
                                       frozen=False, freeze_enable=True)
    print(f"  gate OPEN, accept=0.71 in-band: frozen_out={bool(fr2)} (expect True), "
          f"eps held at {float(e2):.4f}")
    check("freeze_enable=True latches in-band", bool(fr2), f"fr={bool(fr2)}")
    check("eps held on the latch step", abs(float(e2) - eps0) < 1e-12, f"{float(e2)}")
    # (c) one-way latch: already frozen stays frozen and held even with gate closed
    # AND accept out-of-band (sticky regardless of freeze_enable).
    e3, lo3, hi3, fr3 = bisection_step(e2, 0.40, tgt, lo=lo2, hi=hi2, tol=0.03,
                                       frozen=True, freeze_enable=False)
    print(f"  already frozen, accept=0.40 out-of-band, gate CLOSED: "
          f"frozen_out={bool(fr3)} (expect True), eps {float(e2):.4f}->{float(e3):.4f}")
    check("latch sticky regardless of gate", bool(fr3), f"fr={bool(fr3)}")
    check("eps held while frozen", abs(float(e3) - float(e2)) < 1e-12,
          f"{float(e3)} == {float(e2)}")


# --------------------------------------------------------------------------- #
def test_self_calibrated_switch():
    hdr("TEST 8  Self-calibrated vs absolute switch at small M (D2; Fix 2)")
    # PRE-REGISTERED (switch-resolution doc): on an EXACTLY-equilibrium N(0,I)
    # ensemble at M=512, the per-dim noise floor is sqrt(2/M)=0.0625 (max_i ~0.068).
    #   * absolute@0.01 must NOT fire (0.068 > 0.01 -- unreachable below the floor).
    #   * self_calibrated@k=1.5 MUST fire (delta/floor ~ 1.0 < 1.5).
    #   * absolute@0.01 with M=512 supplied must RAISE (config can never fire).
    # FALSIFIER: absolute fires, OR self_calibrated does not fire, OR no raise.
    key = jax.random.PRNGKey(11)
    M, d, T = 512, 6, 100
    theory_floor = math.sqrt(2.0 / M)
    print(f"  M={M}  theory floor sqrt(2/M) = {theory_floor:.4f}  (>> 0.01)")

    # Build a window of T per-step ensemble means of x^2 from independent
    # equilibrium ensembles (decorrelated lower bound on a real run's window).
    keys = jax.random.split(key, T)
    obs_rows = []
    for kk in keys:
        x = jax.random.normal(kk, (M, d))
        obs_rows.append(jnp.mean(jnp.square(x), axis=0))       # E[x_i^2], (d,)
    window = jnp.stack(obs_rows)                                # (T, d)

    # Floor from a single current ensemble: sqrt(Var_rho[x^2]/M)/E_rho[x^2].
    xc = jax.random.normal(jax.random.PRNGKey(12), (M, d))
    xsq = jnp.square(xc)
    floor = jnp.sqrt(jnp.var(xsq, axis=0, ddof=0) / M) / jnp.mean(xsq, axis=0)
    print(f"  online floor per-dim (mean) = {float(jnp.mean(floor)):.4f} "
          f"(expect ~{theory_floor:.4f})")

    d_abs, dmax_abs, fired_abs = phase1_switch(
        window, switch="paper", switch_mode="absolute", threshold=0.01)  # no M -> no guard
    print(f"  absolute@0.01: delta_max={float(dmax_abs):.4f} -> fired={bool(fired_abs)} "
          f"(expect NOT fired; 0.068 > 0.01)")
    check("absolute@0.01 does NOT fire at M=512", not bool(fired_abs),
          f"delta_max {float(dmax_abs):.4f} >= 0.01")

    d_sc, dmax_sc, fired_sc = phase1_switch(
        window, switch="paper", switch_mode="self_calibrated", k=1.5, floor=floor)
    crit = float(jnp.max(d_sc / floor))
    print(f"  self_calibrated@k=1.5: max(delta/floor)={crit:.4f} -> "
          f"fired={bool(fired_sc)} (expect FIRED; ~1.0 < 1.5)")
    check("self_calibrated@k=1.5 FIRES at M=512", bool(fired_sc),
          f"max(delta/floor) {crit:.4f} < 1.5")
    check("delta/floor ratio is O(1)", 0.3 < crit < 1.5,
          f"{crit:.4f} in (0.3, 1.5)")

    # Guard: absolute@0.01 with M=512 supplied must raise (can never fire).
    raised = False
    try:
        phase1_switch(window, switch="paper", switch_mode="absolute",
                      threshold=0.01, M=M)
    except ValueError as e:
        raised = True
        print(f"  guard raised as expected: {str(e)[:70]}...")
    check("absolute@0.01 guard RAISES at M=512", raised,
          "ValueError on unreachable threshold")


# --------------------------------------------------------------------------- #
def _run_staged_controller(eps0, *, tgt=0.70, chunk_size=10, n_chunks=25,
                           persist_need=2, tol=0.03, growth=1.7, seed=123):
    """Reproduce the production driver's STAGED per-chunk Phase-2 host loop on a TOY.

    Ensemble accept is a known monotone-DECREASING function of eps,
    ``accept(eps) = exp(-eps)`` (so eps* = -ln(tgt)). eps is HELD CONSTANT within a
    chunk; the bracket/freeze update runs ONCE per chunk on the SETTLED accept
    (mean over the LATTER HALF of the chunk's per-step accepts). The FIRST HALF is
    deliberately spiked toward ~1.0 to emulate the post-switch transient that fooled
    the old per-step rolling window -- the staged controller must IGNORE it.

    Mirrors driver: eps0 = handoff (here we pass it directly to test BOTH the
    too-SMALL/too-LARGE start), NaN/NaN start bracket (expansion brackets in either
    direction), WIDE rail [eps0*1e-3, eps0*1e3]. Returns the per-chunk rows and the
    post-freeze eps list.
    """
    accept_fn = lambda e: math.exp(-e)
    rng = jax.random.PRNGKey(seed)
    eps = float(eps0)
    eps_min, eps_max = eps0 * 1e-3, eps0 * 1e3      # driver's WIDE safety rail
    lo = float("nan"); hi = float("nan")
    frozen = False; persist = 0
    rows = []
    eps_after_freeze = []
    bstep = lambda e, a, lo, hi, fe: bisection_step(
        e, a, tgt, lo=lo, hi=hi, tol=tol, frozen=False, freeze_enable=fe,
        eps_min=eps_min, eps_max=eps_max, growth=growth)
    for c in range(n_chunks):
        rng, k = jax.random.split(rng)
        base = accept_fn(eps)
        noise = 0.005 * jax.random.normal(k, (chunk_size,))
        steps = jnp.clip(base + noise, 0.0, 1.0)
        # spike the first half toward ~1.0 (lagging post-switch transient)
        steps = steps.at[: chunk_size // 2].set(
            jnp.clip(steps[: chunk_size // 2] + 0.5, 0.0, 1.0))
        steps = np.asarray(steps)
        half = chunk_size // 2
        a_settled = float(np.mean(steps[half:]))    # latter-half settled accept
        rows.append((c, eps, a_settled, lo, hi, frozen))
        if frozen:
            eps_after_freeze.append(eps)
            continue
        in_band = abs(a_settled - tgt) <= tol
        persist = persist + 1 if in_band else 0
        freeze_now = persist >= persist_need
        e_next, lo_, hi_, fr = bstep(eps, a_settled, lo, hi, freeze_now)
        lo, hi = float(lo_), float(hi_)
        eps = float(e_next)
        frozen = bool(fr)
    return rows, eps_after_freeze, (eps_min, eps_max)


def _grade_staged(label, eps0, rows, eps_after_freeze, rail, tgt=0.70, tol=0.03):
    eps_min, eps_max = rail
    eps_star = -math.log(tgt)
    accept_fn = lambda e: math.exp(-e)
    print(f"\n  --- {label}: eps0={eps0:g} (accept0={accept_fn(eps0):.3f}), "
          f"target={tgt}, eps*={eps_star:.5f}, rail=[{eps_min:.3g},{eps_max:.3g}] ---")
    print("   chunk  eps         a_settled  lo         hi         frozen")
    for (c, e, a, l, h, fr) in rows:
        print(f"   {c:<5d}  {e:.6f}   {a:.4f}     {l:<9.4g}  {h:<9.4g}  {fr}")
    froze = any(r[5] for r in rows)
    # eps at which the latch was armed: first chunk that was unfrozen-in but whose
    # update froze it -> the held eps is the last unfrozen row's eps.
    frozen_eps = next((e for (_, e, _, _, _, fr) in rows if fr), rows[-1][1])
    final_a = accept_fn(frozen_eps)
    in_band_chunks = [a for (_, _, a, _, _, fr) in rows if not fr and abs(a - tgt) <= tol]
    eps_latched = len(set(round(x, 12) for x in eps_after_freeze)) <= 1
    print(f"  froze={froze}; frozen eps={frozen_eps:.5f}; accept(frozen eps)="
          f"{final_a:.4f} (|.-tgt|={abs(final_a-tgt):.4f}); "
          f"in-band settled chunks={len(in_band_chunks)}; latched={eps_latched}")
    check(f"[{label}] FREEZES within {len(rows)} chunks", froze, f"froze={froze}")
    check(f"[{label}] frozen accept on-target (<=0.05)", abs(final_a - tgt) <= 0.05,
          f"|{final_a:.4f}-{tgt}|={abs(final_a-tgt):.4f} <= 0.05")
    check(f"[{label}] eps near eps* (no runaway)", abs(frozen_eps - eps_star) < 2 * eps_star,
          f"{frozen_eps:.5f} vs eps*={eps_star:.5f}")
    check(f"[{label}] eps stays inside rail (not pinned)",
          eps_min * 1.001 < frozen_eps < eps_max * 0.999,
          f"{frozen_eps:.5f} in ({eps_min:.3g}, {eps_max:.3g})")
    check(f"[{label}] eps LATCHES (held after freeze)", eps_latched,
          f"post-freeze eps: {sorted(set(round(x,6) for x in eps_after_freeze))}")
    return froze, frozen_eps


def test_staged_phase2_bisection():
    hdr("TEST 7c  STAGED per-chunk Phase-2 bracketing-bisection: BOTH directions "
        "(FIX A eps0=L/N + FIX B bracketing)")
    # MECHANISM under test: the production driver HOLDS eps constant within a
    # Phase-2 chunk and updates the bracket ONCE per chunk on the SETTLED accept
    # (latter-half mean). With the fixes it (FIX A) starts eps0 in the right
    # ballpark and (FIX B) BRACKETS by geometric expansion in EITHER direction
    # before bisecting -- so it reaches the target whether eps0 is too SMALL
    # (accept pinned high, the case that was broken) or too LARGE.
    #
    # accept(eps) = exp(-eps), target 0.70, eps* = -ln(0.70) = 0.35667.
    #
    # PRE-REGISTERED (both cases):
    #   * the controller brackets, refines, and the latch FIRES within the budget;
    #   * accept at the frozen eps is on target (|a - 0.70| <= 0.05);
    #   * eps LATCHES one-way and stays inside the WIDE rail (no runaway, not pinned).
    # FALSIFIER: no freeze, OR frozen accept off target, OR eps changes post-freeze,
    # OR eps pinned at a rail edge.
    tgt = 0.70

    # CASE 1 -- START ABOVE TARGET: tiny eps0=0.01 (accept ~0.99). eps* = 0.357 is
    # ~36x ABOVE eps0 (deliberately OUTSIDE any eps0*30 box), so this REQUIRES the
    # upward geometric expansion to walk eps UP past 30x. THIS is the case the old
    # one-sided/narrow controller could not solve (acceptance pinned ~1.0).
    rows1, eaf1, rail1 = _run_staged_controller(0.01, tgt=tgt)
    froze1, _ = _grade_staged("CASE1 start-ABOVE (eps0=0.01)", 0.01, rows1, eaf1, rail1, tgt)
    # explicit evidence the controller EXPANDED eps UPWARD before bracketing:
    eps_seq1 = [r[1] for r in rows1]
    expanded_up = max(eps_seq1) > 5 * 0.01 and eps_seq1[1] > eps_seq1[0]
    check("[CASE1] controller EXPANDS eps upward (a>target => grow)", expanded_up,
          f"eps went 0.01 -> max {max(eps_seq1):.4f} (>5x), step1>{eps_seq1[0]:.4f}")

    # CASE 2 -- START BELOW TARGET: large eps0=3 (accept ~0.05). eps* = 0.357 is
    # ~8x BELOW eps0, so the controller must SHRINK eps to bracket downward.
    rows2, eaf2, rail2 = _run_staged_controller(3.0, tgt=tgt)
    froze2, _ = _grade_staged("CASE2 start-BELOW (eps0=3)", 3.0, rows2, eaf2, rail2, tgt)
    eps_seq2 = [r[1] for r in rows2]
    shrank_down = min(eps_seq2) < 0.5 * 3.0 and eps_seq2[1] < eps_seq2[0]
    check("[CASE2] controller SHRINKS eps downward (a<target => shrink)", shrank_down,
          f"eps went 3 -> min {min(eps_seq2):.4f} (<0.5x), step1<{eps_seq2[0]:.4f}")


# --------------------------------------------------------------------------- #
def test_changeA_reaches_target_within_budget():
    hdr("TEST 7e  CHANGE A: cross flat accept~1 plateau + REACH/LATCH 0.70 within "
        "the chunk budget")
    # MECHANISM under test (CHANGE A, the production DIAGNOSIS): the real lens gave
    # Phase 2 only ~8 chunks; starting from eps0=L/N the acceptance was PINNED ~1.0
    # and the upward expansion (the flat plateau) ate ~7 chunks, then OVERSHOT 0.70
    # straight to ~0.05 with no chunks left to refine -> never latched. The fix is
    # (i) a smaller Phase-2 chunk -> MANY more chunks for a fixed budget (~25), and
    # (ii) a faster expansion factor p2_growth=2.5 so the plateau is crossed in a
    # FEW chunks, leaving budget to refine + freeze.
    #
    # SYNTHETIC ACCEPT (as pre-registered): a logistic accept(eps) that is ~1.0
    # until eps crosses ~10x eps0, then drops STEEPLY through 0.70 to ~0.05:
    #     accept(eps) = 1 / (1 + (eps/eps_half)^s),  eps_half = 12*eps0, s = 6.
    # At eps0: (1/12)^6 ~ 3e-7 -> accept ~1.0 (the flat plateau the old controller
    # could not cross in budget). accept = 0.70 at eps ~ 10.4x eps0 (steep region).
    #
    # PRE-REGISTERED: with p2_chunk_size=8 (-> 25 chunks at num_adjusted_steps=200)
    # and p2_growth=2.5, the controller BRACKETS, REFINES, and LATCHES at
    # accept 0.70 +/- 0.05 WITHIN the 25-chunk budget. FALSIFIER: no freeze within
    # 25 chunks, OR frozen accept off target by > 0.05, OR eps changes post-freeze.
    tgt = 0.70
    eps0 = 0.01
    eps_half, s = 12.0 * eps0, 6.0
    accept_fn = lambda e: 1.0 / (1.0 + (e / eps_half) ** s)
    n_chunks, chunk_size, growth, persist_need, tol = 25, 8, 2.5, 2, 0.03
    eps_min, eps_max = eps0 * 1e-3, eps0 * 1e3
    print(f"  eps0={eps0} (accept0={accept_fn(eps0):.4f}, plateau ~1.0), "
          f"eps(accept=0.70)~{eps_half*(1/tgt-1)**(1/s):.4f} (~{eps_half*(1/tgt-1)**(1/s)/eps0:.1f}x eps0)")
    print(f"  budget: {n_chunks} chunks x {chunk_size} steps, growth={growth}, "
          f"persist_need={persist_need}")

    rng = jax.random.PRNGKey(7)
    eps = eps0
    lo = float("nan"); hi = float("nan")
    frozen = False; persist = 0; froze_at = None
    rows = []
    for c in range(n_chunks):
        rng, k = jax.random.split(rng)
        base = accept_fn(eps)
        steps = np.array(jnp.clip(base + 0.005 * jax.random.normal(k, (chunk_size,)),
                                  0.0, 1.0))
        # spike the FIRST HALF toward ~1.0 (the post-switch transient the staged
        # controller must IGNORE by using the latter-half settled accept).
        steps[: chunk_size // 2] = np.clip(steps[: chunk_size // 2] + 0.5, 0.0, 1.0)
        a_settled = float(np.mean(steps[chunk_size // 2:]))
        rows.append((c, eps, a_settled, lo, hi, frozen))
        if frozen:
            continue
        in_band = abs(a_settled - tgt) <= tol
        persist = persist + 1 if in_band else 0
        freeze_now = persist >= persist_need
        e_next, lo_, hi_, fr = bisection_step(
            eps, a_settled, tgt, lo=lo, hi=hi, tol=tol, frozen=False,
            freeze_enable=freeze_now, eps_min=eps_min, eps_max=eps_max, growth=growth)
        lo, hi = float(lo_), float(hi_)
        if bool(fr) and froze_at is None:
            froze_at = c
        eps = float(e_next)
        frozen = bool(fr)

    print("   chunk  eps         a_settled  lo         hi         frozen")
    for (c, e, a, l, h, fr) in rows:
        print(f"   {c:<5d}  {e:.6f}   {a:.4f}     {l:<9.4g}  {h:<9.4g}  {fr}")
    frozen_eps = eps                              # held constant once frozen
    final_a = accept_fn(frozen_eps)
    eps_seq = [r[1] for r in rows]
    expanded_up = max(eps_seq) > 5 * eps0         # crossed the plateau upward
    post = [r[1] for r in rows if froze_at is not None and r[0] > froze_at]
    latched = (len({round(x, 12) for x in post}) <= 1) if post else True
    print(f"  froze_at chunk={froze_at} (< {n_chunks}); frozen eps={frozen_eps:.6f}; "
          f"accept(frozen)={final_a:.4f} (|.-0.70|={abs(final_a-tgt):.4f}); "
          f"crossed plateau (max eps {max(eps_seq):.4f} > 5x eps0)={expanded_up}; "
          f"latched={latched}")
    check("[CHANGE A] LATCHES within the 25-chunk budget",
          frozen and froze_at is not None and froze_at < n_chunks,
          f"froze_at={froze_at} < {n_chunks}, frozen={frozen}")
    check("[CHANGE A] frozen accept on target (|a-0.70|<=0.05)",
          abs(final_a - tgt) <= 0.05, f"{final_a:.4f} vs {tgt}")
    check("[CHANGE A] expansion CROSSED the accept~1 plateau", expanded_up,
          f"max eps {max(eps_seq):.4f} > {5*eps0}")
    check("[CHANGE A] eps LATCHES (held after freeze)", latched,
          f"post-freeze eps {sorted({round(x,6) for x in post})}")


# --------------------------------------------------------------------------- #
def test_changeB_core_shape():
    hdr("TEST 7f  CHANGE B: core run returns (M, keep, dim) + flattens correctly")
    # Small synthetic core smoke (CPU): a standard-normal target. PRE-REGISTERED:
    # with p2_keep_per_chain=4, res.samples is (M, 4, dim), res.n_samples_total =
    # M*4, and res.samples.reshape((-1, dim)) flattens to (M*4, dim). keep=1 stays
    # (M, 1, dim). FALSIFIER: wrong shape, wrong n_samples_total, or a flatten that
    # does not give (M*keep, dim). Pure shape/contract test (NOT a recovery claim).
    try:
        from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted
    except Exception as e:                          # pragma: no cover
        check("[CHANGE B] core import", False, f"import failed: {e}")
        return
    dim = 4
    logdensity_fn = lambda z: -0.5 * jnp.sum(z ** 2)
    M = 16
    res = LAPS_late_adjusted(
        logdensity_fn, qz=None, dim=dim, init_mode="cold", num_chains=M,
        num_unadjusted_steps=40, num_adjusted_steps=40, chunk_size=20,
        p2_chunk_size=8, p2_keep_per_chain=4, p2_thin=3, seed=0)
    smp = np.asarray(res.samples)
    flat = smp.reshape((-1, dim))
    Mr = smp.shape[0]
    print(f"  samples shape={smp.shape} (expect (M,4,{dim})); n_samples_total="
          f"{res.n_samples_total} (expect {Mr*4}); flat={flat.shape} "
          f"(expect ({Mr*4},{dim})); finite={bool(np.all(np.isfinite(smp)))}")
    check("[CHANGE B] samples shape (M,keep,dim)",
          smp.ndim == 3 and smp.shape[1] == 4 and smp.shape[2] == dim,
          f"{smp.shape}")
    check("[CHANGE B] n_samples_total == M*keep", res.n_samples_total == Mr * 4,
          f"{res.n_samples_total} == {Mr*4}")
    check("[CHANGE B] reshape((-1,dim)) -> (M*keep, dim)",
          flat.shape == (Mr * 4, dim), f"{flat.shape}")
    check("[CHANGE B] samples finite", bool(np.all(np.isfinite(smp))),
          "all finite")

    # keep=1 backward-compat: (M, 1, dim), n_samples_total == M, flatten -> (M, dim).
    res1 = LAPS_late_adjusted(
        logdensity_fn, qz=None, dim=dim, init_mode="cold", num_chains=M,
        num_unadjusted_steps=40, num_adjusted_steps=40, chunk_size=20,
        p2_chunk_size=8, p2_keep_per_chain=1, seed=0)
    smp1 = np.asarray(res1.samples)
    print(f"  keep=1: shape={smp1.shape} (expect (M,1,{dim})); n_total="
          f"{res1.n_samples_total}; flat={smp1.reshape((-1,dim)).shape}")
    check("[CHANGE B] keep=1 -> (M,1,dim), flatten (M,dim)",
          smp1.ndim == 3 and smp1.shape[1] == 1
          and smp1.reshape((-1, dim)).shape == (smp1.shape[0], dim),
          f"{smp1.shape}")


# --------------------------------------------------------------------------- #
def test_chunk_sizes():
    hdr("TEST 9  Chunk-divisibility robustness (grader #5; Fix 3)")
    # PRE-REGISTERED: _chunk_sizes never drops a remainder and never returns [].
    #   total < chunk        -> [total]            (no n_chunks=0 / phase1_len=0)
    #   total not divisible   -> sum == total, final short chunk kept
    #   total divisible       -> all == chunk
    from gigalens_research.inference.laps_late_adjusted import _chunk_sizes
    cases = [(20, 25), (310, 25), (300, 25), (1, 25), (50, 50)]
    okall = True
    for total, chunk in cases:
        sizes = _chunk_sizes(total, chunk)
        s = sum(sizes)
        nonempty = len(sizes) > 0
        covers = (s == total)
        bounded = all(0 < x <= chunk for x in sizes)
        ok = nonempty and covers and bounded
        okall = okall and ok
        print(f"  total={total:<4d} chunk={chunk:<3d} -> {sizes}  sum={s} "
              f"(cover={covers}, bounded={bounded})")
    check("_chunk_sizes covers budget, no empty, final short kept", okall,
          "all cases sum==total, nonempty, <=chunk")


# --------------------------------------------------------------------------- #
def main():
    print(f"jax {jax.__version__}  x64={jax.config.jax_enable_x64}")
    test_equipartition()
    test_F_and_step_laws()
    test_ensemble_eevpd()
    test_decoherence_length()
    test_switch_and_D1_falsifier()
    test_bisection()
    test_bisection_latch()
    test_bisection_freeze_gate()
    test_staged_phase2_bisection()
    test_changeA_reaches_target_within_budget()
    test_changeB_core_shape()
    test_self_calibrated_switch()
    test_chunk_sizes()
    hdr("SUMMARY")
    if _FAILS:
        print(f"  FAILED: {_FAILS}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
