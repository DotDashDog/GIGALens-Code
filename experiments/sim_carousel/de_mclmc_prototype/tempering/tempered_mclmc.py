"""Standalone TEMPERING wrapper around the real MCLMC kernel.

WHY (carousel lab notebook C-9..C-16): the production blocker is DISCOVERY, not
equilibration. A short MAP lands chains in a negligible-mass (~1e-5) SECONDARY
basin; vanilla MCLMC cannot cross/find the dominant GLOBAL basin (structural,
C-10), and EVERY cross-chain ensemble move (linear-DE, snooker, kernel-hop,
SA-MCMC) UNIVERSALLY PINS a lone chain in a tiny mode and provides NO discovery
(C-16). TEMPERING is the tool that CAN: flattening the target (beta<1) lets
MCLMC cross barriers via its GRADIENT dynamics (curvature-robust -- it follows
the ridge, unlike an affine chord jump), and the cold (beta=1) chain naturally
UNDER-weights / drains a tiny mode instead of pinning it.

THE TEMPERING IS EXACT IN THE MCLMC SENSE
-----------------------------------------
MCLMC (Robnik et al. 2023, arXiv:2303.18221) integrates
    dx = u dt,   du = P(u) f(x) dt + eta P(u) dW,   f(x) = -grad S(x)/(d-1),
with S = -log p and |u| = 1; the configuration-space stationary distribution is
    rho(x) ∝ e^{-S(x)} = p(x).                            [VERIFIED from the paper]
Replacing the log-density by  beta * log p  (i.e. S -> beta S) scales the force
by beta and makes the stationary distribution
    rho_beta(x) ∝ e^{-beta S(x)} = p(x)^beta.             [own-knowledge: direct
algebraic consequence of the verified equation]
That is precisely the tempered target. We build a SEPARATE real MCLMC kernel for
each beta on the ladder (the existing mclmc.py / de_mclmc.py are untouched; we
only import the kernel primitives read-only).

SIMPLEST-FIRST MECHANISM: TEMPERED BURN-IN (annealed burn-in)
-------------------------------------------------------------
Anneal beta from beta_min (flat) up to 1 over the burn-in; at each beta run K
real MCLMC steps (momentum carried within a stage, refreshed at stage
boundaries -- an auxiliary variable, refresh never biases the position
marginal). Then DISCARD the tempered samples and sample only at beta=1. Because
the tempered samples are discarded, NO importance weights are needed: the cold
phase is plain (unbiased) MCLMC; the anneal only RELOCATES the ensemble into the
correct basins with the correct between-mode occupancy (the cooling tracks the
tempered weight w_i^beta / sum_j w_j^beta, which sharpens to the true weight as
beta -> 1). This is the minimal discovery + tiny-mode-drain mechanism; escalate
to replica-exchange PT only if this is insufficient (and say why, with numbers).

MCLMC-UNDER-TEMPERING SUBTLETY (flagged):
The integrator step size that hits a target energy-error variance per dim
(EEVPD = mean(energy_change^2)/D = DESIRED_EVAR) DEPENDS on the target, because
the force scales by beta and the tempered target's width changes. At smaller
beta the force is SMALLER and the target BROADER, so the beta=1 EEVPD-tuned step
is CONSERVATIVE (finer than needed -> smaller EEVPD) at every beta<=1. The
dangerous direction (too-coarse a step -> huge energy error -> spurious
bias/collapse, the C-15 lesson) therefore cannot arise from reusing the beta=1
step across the cooling ladder. We tune the step ONCE at beta=1 via EEVPD and
MEASURE the realized EEVPD at every beta to confirm it stays <= DESIRED_EVAR.
(L, the momentum-decoherence length, is left fixed; flagged as a possible
efficiency knob -- a broader tempered target could use a larger L.)
"""
import os, sys
import numpy as np
import jax
import jax.numpy as jnp

# --- import the REAL MCLMC kernel primitives, read-only ----------------------
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap,
    isokinetic_mclachlan_smart,
    _single_init,
)

DESIRED_EVAR = 5e-4   # project EEVPD target = mean(energy_change^2)/D


def _build_kernel(logdensity_fn, beta, imm, integrator):
    """Real MCLMC kernel for the beta-tempered target (beta*log p)."""
    tempered = lambda z: beta * logdensity_fn(z)
    kern = _build_kernel_shardmap(
        logdensity_fn=tempered, inverse_mass_matrix=imm, integrator=integrator)
    return kern, tempered


# ============================================================ EEVPD step tuner
def measure_eevpd(logdensity_fn, D, positions, beta, L, step_size, imm=None,
                  n_steps=40, seed=0, integrator=isokinetic_mclachlan_smart):
    """Realized energy-error variance per dim = mean(energy_change^2)/D for the
    beta-tempered MCLMC kernel at the given step, started at `positions`."""
    if imm is None:
        imm = jnp.eye(D)
    n = positions.shape[0]
    kern, tf = _build_kernel(logdensity_fn, beta, imm, integrator)
    ikeys = jax.random.split(jax.random.key(seed + 1), n)
    states = jax.vmap(lambda p, k: _single_init(p, tf, k))(jnp.asarray(positions), ikeys)
    skeys = jax.random.split(jax.random.key(seed), n_steps * n).reshape(n_steps, n)

    def step(carry, k_t):
        def per_chain(s, k):
            ns, info = kern(rng_key=k, state=s, L=L, step_size=step_size)
            return ns, info.energy_change
        return jax.vmap(per_chain)(carry, k_t)

    _, ec = jax.lax.scan(step, states, skeys)
    ec = np.asarray(ec)
    return float(np.mean(ec ** 2) / D)


def tune_step_eevpd(logdensity_fn, D, positions, beta, L, imm=None,
                    step_grid=None, target=DESIRED_EVAR, n_steps=40, seed=0,
                    integrator=isokinetic_mclachlan_smart, verbose=True):
    """Sweep step sizes, measure EEVPD, and pick (log-log interpolate) the step
    where EEVPD == target. Returns (step_tuned, grid_steps, grid_eevpd)."""
    if step_grid is None:
        step_grid = np.geomspace(0.01, 1.0, 13)
    evs = []
    for s in step_grid:
        ev = measure_eevpd(logdensity_fn, D, positions, beta, L, float(s), imm,
                           n_steps=n_steps, seed=seed, integrator=integrator)
        evs.append(ev)
        if verbose:
            print(f"   [tune beta={beta:.4g}] step={s:.5f}  EEVPD={ev:.3e}  "
                  f"(target {target:.1e})", flush=True)
    evs = np.asarray(evs)
    lg_s, lg_e = np.log(step_grid), np.log(evs)
    # EEVPD increases with step; interpolate log-log for the crossing
    order = np.argsort(lg_e)
    step_tuned = float(np.exp(np.interp(np.log(target), lg_e[order], lg_s[order])))
    step_tuned = float(np.clip(step_tuned, step_grid.min(), step_grid.max()))
    if verbose:
        print(f"   [tune beta={beta:.4g}] EEVPD-tuned step ~= {step_tuned:.5f}", flush=True)
    return step_tuned, step_grid, evs


# ===================================================== tempered-burn-in sampler
def make_tempered_sampler(logdensity_fn, D, n_chains, betas, L, step_size,
                          imm=None, integrator=isokinetic_mclachlan_smart):
    """Build tempered-burn-in + cold-sampling functions around the real kernel.

    Parameters
    ----------
    betas      : 1-D array of ascending betas for the cooling ladder; the LAST
                 must be 1.0 (the cold target). e.g. geomspace(0.04, 1.0, 15).
    L, step_size : FIXED MCLMC knobs (step_size MUST be the EEVPD-tuned beta=1
                 step; it is conservative across the ladder -- see module docs).
    imm        : honest within-mode inverse mass matrix (default identity). DO
                 NOT pass a MM that already straddles both modes.

    Returns dict with:
      'init_states'  : fn(positions(n,D), key, beta) -> vmapped IntegratorState
      'anneal'       : fn(init_positions(n,D), key, steps_per_beta) ->
                         (final_positions(n,D), trace) where trace carries
                         per-stage positions(n_betas,n,D) and EEVPD(n_betas,).
      'sample_cold'  : jitted fn(positions(n,D), keys(T,n)) ->
                         (final_states, positions(T,n,D), ec(T,n)) at beta=1.
      'vanilla'      : same as sample_cold but used as the beta=1 baseline
                         (identical kernel; provided for naming clarity).
    """
    assert abs(float(betas[-1]) - 1.0) < 1e-12, "last beta must be 1.0 (cold target)"
    if imm is None:
        imm = jnp.eye(D)
    betas = [float(b) for b in betas]

    kernels, tfns = {}, {}
    for b in betas:
        k, tf = _build_kernel(logdensity_fn, b, imm, integrator)
        kernels[b], tfns[b] = k, tf

    def init_states(positions, key, beta):
        tf = tfns[float(beta)]
        keys = jax.random.split(key, positions.shape[0])
        return jax.vmap(lambda p, k: _single_init(p, tf, k))(positions, keys)

    # one jitted K-step scan per beta (kernel differs per beta) ----------------
    scans = {}
    for b in betas:
        kern_b = kernels[b]

        def _make(kern_b):
            @jax.jit
            def scan_fn(states, keys):  # keys: (K, n)
                def step(carry, k_t):
                    def per_chain(s, k):
                        ns, info = kern_b(rng_key=k, state=s, L=L, step_size=step_size)
                        return ns, (ns.position, info.energy_change)
                    return jax.vmap(per_chain)(carry, k_t)
                final, (pos, ec) = jax.lax.scan(step, states, keys)
                return final, pos, ec
            return scan_fn
        scans[b] = _make(kern_b)

    def anneal(init_positions, key, steps_per_beta):
        """Run the cooling ladder; carry positions stage-to-stage, refresh
        momentum at each beta change. Returns final positions + a trace."""
        positions = jnp.asarray(init_positions)
        n = positions.shape[0]
        stage_pos = np.empty((len(betas), n, D))
        stage_eevpd = np.empty(len(betas))
        key_iter = key
        for i, b in enumerate(betas):
            key_iter, ki, ks = jax.random.split(key_iter, 3)
            states = init_states(positions, ki, b)        # fresh momentum @ beta_i
            skeys = jax.random.split(ks, steps_per_beta * n).reshape(steps_per_beta, n)
            states, pos, ec = scans[b](states, skeys)
            positions = states.position
            stage_pos[i] = np.asarray(positions)
            stage_eevpd[i] = float(np.mean(np.asarray(ec) ** 2) / D)
        trace = dict(betas=np.asarray(betas), stage_pos=stage_pos,
                     stage_eevpd=stage_eevpd)
        return positions, trace

    cold_scan = scans[1.0]

    def sample_cold(positions, key, n_steps):
        n = positions.shape[0]
        ki, ks = jax.random.split(key)
        states = init_states(positions, ki, 1.0)
        skeys = jax.random.split(ks, n_steps * n).reshape(n_steps, n)
        states, pos, ec = cold_scan(states, skeys)
        return states, pos, ec

    return {
        "init_states": init_states,
        "anneal": anneal,
        "sample_cold": sample_cold,
        "betas": betas,
        "config": dict(D=D, n_chains=n_chains, L=L, step_size=step_size,
                       n_betas=len(betas)),
    }
