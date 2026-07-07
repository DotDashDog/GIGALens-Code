"""Replica-exchange PARALLEL TEMPERING around the real MCLMC kernel.

WHY (measured, drill_schedule.py): one-shot tempered burn-in DISCOVERS the
dominant basin but the comparable-mass between-mode WEIGHT freezes out at the
decoupling temperature (~0.57-0.59 vs truth 0.70; plateaus under 12x more steps
and denser ladders) because the cold barrier is uncrossable and the cold
ensemble is frozen. PT removes the freeze-out: the cold (beta=1) replica
CONTINUOUSLY swaps with hotter replicas that cross freely, so it receives
correctly-weighted configurations and its TIME-AVERAGED weight converges to the
truth, with a genuinely-mixing (non-frozen) cold series -> a block-bootstrap SE
is now well defined.

CONSTRUCTION
------------
R replicas at ascending inverse temperatures beta_1<...<beta_R=1, n_systems
independent ladders run in parallel (positions shape (R, n_systems, D)). One
round = [K real MCLMC steps at each level r using a beta_r-tempered kernel] then
[even/odd adjacent-pair replica-exchange swaps].

SWAP ACCEPTANCE (VERIFIED, standard PT; see arXiv physics/0508111): exchanging
configurations x_r (at beta_r) and x_{r+1} (at beta_{r+1}) is accepted with
    alpha = min(1, exp[(beta_r - beta_{r+1}) * (logp(x_{r+1}) - logp(x_r))])
where logp is the BASE (untempered) log-density. This is detailed-balance exact
for the joint product target prod_r p(x_r)^{beta_r}; the MCLMC level updates are
(approximately) p^{beta_r}-invariant. Momentum is refreshed each round (auxiliary
-> no position bias). Only the cold replica (beta=1) is kept as the sample.

Step size: pass the EEVPD-tuned step (scalar reused across levels is CONSERVATIVE
-- the force scales by beta<=1 so hotter levels have SMALLER energy error; same
argument as tempered_mclmc.py), or a per-level array.
"""
import os, sys
import numpy as np
import jax
import jax.numpy as jnp

from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, isokinetic_mclachlan_smart, _single_init)


def make_pt_sampler(logdensity_fn, D, n_systems, betas, L, step_size, K=20,
                    imm=None, integrator=isokinetic_mclachlan_smart):
    """Build a PT round function. betas ascending, betas[-1]==1.0.
    step_size: scalar or length-R array (per level)."""
    assert abs(float(betas[-1]) - 1.0) < 1e-12, "last beta must be 1.0"
    R = len(betas)
    betas = np.asarray([float(b) for b in betas])
    if imm is None:
        imm = jnp.eye(D)
    steps = np.full(R, float(step_size)) if np.isscalar(step_size) else np.asarray(step_size, float)

    # per-level tempered kernel + jitted K-step scan
    tfns, scans = [], []
    for r in range(R):
        b = betas[r]
        tf = (lambda bb: (lambda z: bb * logdensity_fn(z)))(b)
        kern = _build_kernel_shardmap(logdensity_fn=tf, inverse_mass_matrix=imm,
                                      integrator=integrator)
        tfns.append(tf)
        sr = float(steps[r])

        def _make(kern, sr):
            @jax.jit
            def scan_fn(states, keys):  # keys (K, n_sys)
                def step(carry, k_t):
                    def per_chain(s, k):
                        ns, info = kern(rng_key=k, state=s, L=L, step_size=sr)
                        return ns, info.energy_change
                    return jax.vmap(per_chain)(carry, k_t)
                final, ec = jax.lax.scan(step, states, keys)
                return final, ec
            return scan_fn
        scans.append(_make(kern, sr))

    logp_v = jax.jit(jax.vmap(logdensity_fn))   # base logp over a (n,D) batch

    def init_level(positions, key, r):
        keys = jax.random.split(key, positions.shape[0])
        return jax.vmap(lambda p, k: _single_init(p, tfns[r], k))(positions, keys)

    def mclmc_all_levels(positions, key):
        # positions (R, n_sys, D) -> updated positions, eevpd per level
        n = positions.shape[1]
        new_pos = np.empty((R, n, D))
        eevpd = np.empty(R)
        keys = jax.random.split(key, R)
        for r in range(R):
            ki, ks = jax.random.split(keys[r])
            states = init_level(jnp.asarray(positions[r]), ki, r)
            skeys = jax.random.split(ks, K * n).reshape(K, n)
            states, ec = scans[r](states, skeys)
            new_pos[r] = np.asarray(states.position)
            eevpd[r] = float(np.mean(np.asarray(ec) ** 2) / D)
        return new_pos, eevpd

    def swap_sweep(positions, lp, rng, parity):
        # host-side even/odd adjacent swaps. positions (R,n,D), lp (R,n).
        n = positions.shape[1]
        acc = np.zeros(R - 1)
        for r in range(parity, R - 1, 2):
            dbeta = betas[r] - betas[r + 1]              # < 0
            log_alpha = dbeta * (lp[r + 1] - lp[r])      # (n,)
            u = rng.random(n)
            accept = np.log(u) < log_alpha               # (n,)
            # swap accepted systems between levels r and r+1
            pr, pr1 = positions[r].copy(), positions[r + 1].copy()
            positions[r] = np.where(accept[:, None], pr1, pr)
            positions[r + 1] = np.where(accept[:, None], pr, pr1)
            lpr, lpr1 = lp[r].copy(), lp[r + 1].copy()
            lp[r] = np.where(accept, lpr1, lpr)
            lp[r + 1] = np.where(accept, lpr, lpr1)
            acc[r] = accept.mean()
        return positions, lp, acc

    def run(init_positions, key, n_rounds, seed_swap=0):
        """init_positions: (R, n_sys, D). Returns dict of arrays."""
        positions = np.asarray(init_positions).copy()
        rng = np.random.default_rng(seed_swap)
        cold = np.empty((n_rounds, positions.shape[1], D))
        swap_acc = np.zeros(R - 1)
        eevpd_acc = np.zeros(R)
        keys = jax.random.split(key, n_rounds)
        for t in range(n_rounds):
            positions, eevpd = mclmc_all_levels(positions, keys[t])
            lp = np.array(logp_v(jnp.asarray(positions.reshape(R * positions.shape[1], D)))
                          ).reshape(R, positions.shape[1])
            positions, lp, acc = swap_sweep(positions, lp, rng, parity=t % 2)
            cold[t] = positions[-1]
            swap_acc += acc
            eevpd_acc += eevpd
        return dict(cold=cold, swap_acc=swap_acc / n_rounds,
                    eevpd_mean=eevpd_acc / n_rounds, betas=betas, R=R)

    return dict(run=run, betas=betas, R=R,
                config=dict(D=D, n_systems=n_systems, R=R, L=L, K=K))
