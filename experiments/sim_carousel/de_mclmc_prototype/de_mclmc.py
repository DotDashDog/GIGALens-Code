"""Standalone composite sampler: MCLMC local moves interleaved with DE-MC
(Differential Evolution) ensemble cross-chain jumps.

PURPOSE
-------
Vanilla MCLMC cannot cross/discover separated basins (established fact): with an
honest within-mode (identity) inverse-mass-matrix it stays trapped in the basin
it starts in. The DE-MC ensemble move proposes jumps along the *difference
vectors* between chains; once chains populate both basins (or the inner tail
toward the barrier) these difference vectors are large enough to hop modes. The
DE move is an EXACT Metropolis update for the true target, so the composite
leaves the target (approximately, modulo MCLMC's own approximate invariance)
invariant.

DESIGN (what makes it unbiased)
-------------------------------
Composite step = K MCLMC steps (real kernel, fixed L/step_size/identity MM) then
ONE DE-MC ensemble move. The DE move uses a TWO-GROUP (split-ensemble) update:

  * split the n chains into halves A and B.
  * update A using ONLY B's current (frozen) states; THEN update B using A's
    (now-updated) states. A chain is NEVER updated with a pair drawn from its own
    group's current/proposed states -> no self-reference, valid in parallel.
  * proposal for chain i (in the group being updated):
        z_i' = z_i + gamma*(z_a - z_b) + eps,   a != b drawn from complementary group,
        eps ~ N(0, b0^2 I),  b0 small.
    gamma = 2.38/sqrt(2D) most steps; gamma = 1.0 a fraction (p_jump) of the time
    for mode jumps.
  * the proposal density is SYMMETRIC (ordered pairs (a,b) and (b,a) are drawn
    with equal probability, eps is symmetric, b0 fixed, complementary states
    frozen) => Metropolis acceptance is min(1, exp(logp(z_i') - logp(z_i))),
    applied per chain.

WHY IT IS UNBIASED
------------------
Target over the ensemble Z=(z_1,...,z_n) is the product pi(Z)=prod_i pi(z_i).
Conditional on the frozen complementary group, the chains in the updated group
are independent draws from pi, and each gets its own symmetric-proposal
Metropolis step -> this is a valid Metropolis-within-Gibbs block update that
preserves the product target. Swapping roles (A|B then B|A) preserves it too.
MCLMC is (approximately) pi-invariant by construction. Composition of
pi-invariant kernels is pi-invariant. The DE move adds NO bias.

After a DE move the position can change discontinuously, so the MCLMC
IntegratorState is rebuilt at the new position with a freshly sampled (unit)
momentum via the real `_single_init`. Momentum is an auxiliary variable that may
be refreshed at any time without biasing the position marginal.

This module is GENERIC in the target: it takes a logdensity_fn. The analytic
bimodal validation target lives in validate_analytic.py.
"""
import os, sys
import jax
import jax.numpy as jnp

# --- import the REAL MCLMC kernel primitives, read-only ----------------------
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap,
    isokinetic_mclachlan_smart,
    _single_init,
)


def make_composite(logdensity_fn, D, n_chains, L, step_size, K,
                   b0=0.05, p_jump=0.2, inverse_mass_matrix=None,
                   eps_scale=None, integrator=isokinetic_mclachlan_smart):
    """Build a jitted composite-round function and helpers.

    Parameters
    ----------
    logdensity_fn : callable z(D,) -> scalar logdensity
    D             : dimension
    n_chains      : number of ensemble chains (MUST be even; split into halves)
    L, step_size  : MCLMC knobs (FIXED, no adaptation)
    K             : number of MCLMC steps per composite round
    b0            : std of the additive Gaussian jitter eps in the DE proposal
    p_jump        : per-chain probability of using gamma=1.0 (mode-jump scale)
    inverse_mass_matrix : honest within-mode preconditioner. Default = identity
                          (None -> ones(D)). DO NOT pass a MM that already covers
                          both modes; that would defeat the test.

    Returns
    -------
    dict with:
      'init_states'  : fn(positions(n,D), key) -> vmapped IntegratorState
      'round'        : jitted fn(states, key) -> (states, (positions(n,D),
                        mclmc_energy_change(K,n), de_accept(n,)))
      'mclmc_only'   : jitted fn(states, keys(steps,n)) -> (states, positions(steps,n,D))
                        pure MCLMC scan, NO DE, NO forced momentum refresh
                        (the honest vanilla baseline).
    """
    assert n_chains % 2 == 0, "n_chains must be even for the two-group DE update"
    if inverse_mass_matrix is None:
        # honest identity / within-mode scale. The smart integrator requires a
        # FULL (2D) inverse mass matrix, so pass identity (NOT a diagonal vector).
        inverse_mass_matrix = jnp.eye(D)

    kernel = _build_kernel_shardmap(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inverse_mass_matrix,
        integrator=integrator,
    )

    gamma_big = 2.38 / jnp.sqrt(2.0 * D)
    # Jitter scaling. eps_scale may be:
    #   None  -> isotropic ones(D)            : eps = b0 * xi
    #   1-D   -> per-dim std (diagonal)        : eps = b0 * sigma * xi   (axis anisotropy only)
    #   2-D   -> lower-Cholesky of within-mode cov : eps = b0 * (L @ xi) ~ N(0, b0^2 cov)
    #            (respects within-mode CORRELATIONS, not just axis scales)
    # In all cases eps is a fixed-covariance zero-mean Gaussian -> symmetric proposal
    # -> unbiasedness unchanged.
    eps_scale_v = jnp.ones(D) if eps_scale is None else jnp.asarray(eps_scale)

    # ---- init vmapped states ------------------------------------------------
    def init_states(positions, key):
        keys = jax.random.split(key, positions.shape[0])
        return jax.vmap(lambda p, k: _single_init(p, logdensity_fn, k))(positions, keys)

    # ---- K MCLMC steps, vmapped over chains, scanned over steps -------------
    def mclmc_k_steps(states, keys):
        # keys: (K, n)
        def scan_step(carry, k_t):
            def per_chain(s, k):
                ns, info = kernel(rng_key=k, state=s, L=L, step_size=step_size)
                return ns, info.energy_change
            return jax.vmap(per_chain)(carry, k_t)
        final, ec = jax.lax.scan(scan_step, states, keys)
        return final, ec  # ec: (K, n)

    # ---- DE-MC update of ONE group using a FROZEN complementary group -------
    def update_group(grp, comp, key):
        # grp:  (g, D)  positions to update
        # comp: (gc, D) frozen complementary positions used for difference vectors
        g = grp.shape[0]
        gc = comp.shape[0]
        keys = jax.random.split(key, g)

        def one(z_i, k):
            ka, kb, ke, kg, ku = jax.random.split(k, 5)
            # two DISTINCT indices a != b, uniform over ordered pairs of comp
            a = jax.random.randint(ka, (), 0, gc)
            b = jax.random.randint(kb, (), 0, gc - 1)
            b = jnp.where(b >= a, b + 1, b)          # remap to skip a -> a!=b
            use_jump = jax.random.uniform(kg) < p_jump
            gamma = jnp.where(use_jump, 1.0, gamma_big)
            xi = jax.random.normal(ke, (D,))
            eps = b0 * (eps_scale_v @ xi if eps_scale_v.ndim == 2 else eps_scale_v * xi)
            prop = z_i + gamma * (comp[a] - comp[b]) + eps   # SYMMETRIC proposal
            lp_cur = logdensity_fn(z_i)
            lp_prop = logdensity_fn(prop)
            log_alpha = lp_prop - lp_cur
            accept = jnp.log(jax.random.uniform(ku)) < log_alpha
            newz = jnp.where(accept, prop, z_i)
            return newz, accept.astype(jnp.float64)

        return jax.vmap(one)(grp, keys)

    # ---- one full DE ensemble move (two-group, RE-RANDOMIZED partition) ------
    def de_move(positions, key):
        # Re-draw the A/B partition every call (independent of state). Each step is
        # still a valid block update (A | frozen B, then B | updated A), so the move
        # is target-invariant; a mixture over random partitions of invariant kernels
        # is invariant. Re-randomizing is strictly more robust: as long as the whole
        # ensemble straddles the modes, a random half almost always contains both, so
        # difference vectors span the inter-mode gap (a FIXED partition can trap a
        # half in a single mode and make gap-jumps structurally impossible).
        n = positions.shape[0]
        half = n // 2
        kperm, kA, kB = jax.random.split(key, 3)
        perm = jax.random.permutation(kperm, n)
        idxA, idxB = perm[:half], perm[half:]
        posA, posB = positions[idxA], positions[idxB]
        newA, accA = update_group(posA, posB, kA)      # A | frozen B
        newB, accB = update_group(posB, newA, kB)      # B | updated A
        new_positions = positions.at[idxA].set(newA).at[idxB].set(newB)
        acc = jnp.zeros(n, jnp.float64).at[idxA].set(accA).at[idxB].set(accB)
        return new_positions, acc                       # scattered back to original chain order

    # ---- one composite round: K MCLMC steps then one DE move ----------------
    @jax.jit
    def round_fn(states, key):
        kmc, kde, kinit = jax.random.split(key, 3)
        mc_keys = jax.random.split(kmc, K * n_chains).reshape(K, n_chains)
        states, ec = mclmc_k_steps(states, mc_keys)
        positions = states.position                      # (n, D)
        new_positions, acc = de_move(positions, kde)
        # rebuild MCLMC state at the (possibly moved) position: fresh momentum,
        # recomputed logdensity/grad. Momentum refresh never biases positions.
        new_states = init_states(new_positions, kinit)
        return new_states, (new_positions, ec, acc)

    # ---- pure MCLMC baseline (NO DE, NO forced refresh) ---------------------
    @jax.jit
    def mclmc_only(states, keys):
        # keys: (steps, n) -- continuous kernel scan, momentum carried naturally
        def scan_step(carry, k_t):
            def per_chain(s, k):
                ns, info = kernel(rng_key=k, state=s, L=L, step_size=step_size)
                return ns, ns.position
            return jax.vmap(per_chain)(carry, k_t)
        final, pos = jax.lax.scan(scan_step, states, keys)
        return final, pos  # pos: (steps, n, D)

    return {
        "init_states": init_states,
        "round": round_fn,
        "mclmc_only": mclmc_only,
        "gamma_big": float(gamma_big),
        "config": dict(D=D, n_chains=n_chains, L=L, step_size=step_size, K=K,
                       b0=b0, p_jump=p_jump),
    }
