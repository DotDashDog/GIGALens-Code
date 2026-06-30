"""Standalone composite sampler: MCLMC local moves interleaved with a
KERNEL-HOP ("Normal Kernel Coupler" / KDE-independence) cross-chain move.

================================================================================
DESIGN NOTE  (literature + the exact construction implemented here)
================================================================================

MOTIVATION (established finding, trust)
---------------------------------------
The linear DE-MC mode jump  z_i' = z_i + gamma*(z_a - z_b) + eps  FAILS on the
carousel: the within-mode geometry is a CURVED thin ridge, so a chord/difference
vector between two on-ridge points lands OFF the ridge (measured ~7x off-ridge
for the secondary mode). Every affine variant (snooker, mode-matching) inherits
this because they all extrapolate a straight difference vector.

THE KERNEL-HOP IDEA
-------------------
Instead of extrapolating a difference vector, propose chain i NEAR ANOTHER
chain's current position -- that other position is itself ON the manifold, so a
small Gaussian blur around it stays (approximately) on-manifold regardless of
curvature. This is exactly the Normal Kernel Coupler (Warnes 2001 thesis; KDE /
kernel-density ensemble MCMC) and is an INDEPENDENCE Metropolis-Hastings sampler
whose proposal is a Gaussian kernel-density estimate built from the other chains.

EXACT PROPOSAL + ACCEPTANCE (this is what the code implements)
-------------------------------------------------------------
Two-group, FROZEN-COMPLEMENT update (the cleanly-exact "red-black" form, as in
ter Braak & Vrugt 2008 DE-MC and Goodman & Weare 2010 affine-invariant ensemble):

  * Partition the n chains into groups A and B (re-randomized every round).
  * Update every chain in A using ONLY the FROZEN states of B; then update B
    using the (now-updated, frozen) A. A chain is never in its own proposal.
  * Kernel-density proposal for a chain at z (z in the group being updated),
    built from the frozen complement C = {z_j : j in other group}:

        q_C(.) = (1/|C|) * sum_{j in C} N(. ; z_j , eps^2 * M)

    DRAW: pick j ~ Uniform(C),  z' = z_j + eps * chol(M) @ xi ,  xi ~ N(0,I_D).
  * Because q_C does NOT depend on the current state z (only on the frozen C),
    this is an INDEPENDENCE sampler. Metropolis-Hastings acceptance:

        alpha = min(1,  [ pi(z') q_C(z) ] / [ pi(z) q_C(z') ] )

    with  q_C(z) = (1/|C|) sum_j N(z; z_j, eps^2 M)  evaluated at BOTH the
    current z and the proposal z' using the SAME frozen C.
  * Then resample MCLMC momentum (auxiliary) and continue MCLMC local moves.

WHY IT IS REVERSIBLE / UNBIASED
-------------------------------
Conditional on the frozen complement C, each updated chain runs a standard
INDEPENDENCE Metropolis-Hastings step with target pi and a fixed proposal
density q_C(.) that does not depend on the chain's current state. The
independence-sampler acceptance min(1, pi(z')q_C(z)/[pi(z)q_C(z')]) satisfies
detailed balance w.r.t. pi for ANY fixed q_C with q_C>0 wherever pi>0 (here q_C
is a sum of Gaussians, strictly positive everywhere, so this holds for ANY
eps>0 and any M). Detailed balance:
      pi(z) q_C(z') alpha(z->z')  =  min( pi(z)q_C(z'), pi(z')q_C(z) )
is symmetric under z<->z'. Hence each A-chain update preserves pi; conditional
on C the A-chains are independent so the block preserves the product target
prod_i pi(z_i); B is untouched during the A-update so the joint target is
preserved; swapping roles (B | updated A) preserves it too. A mixture over
random A/B partitions of pi-invariant kernels is pi-invariant. MCLMC is
(approximately) pi-invariant. Composition of pi-invariant kernels is
pi-invariant. The kernel-hop adds NO bias. eps>0 is REQUIRED (a delta kernel is
not absolutely continuous / not reversible); eps is small but nonzero.

This differs from the linear DE move in de_mclmc.py in TWO ways: (1) the proposal
is a point near an existing on-manifold sample (curvature-robust) rather than a
chord extrapolation; (2) the proposal is ASYMMETRIC (KDE), so the acceptance
carries the Hastings q-ratio q_C(z)/q_C(z') -- this q-ratio is essential and is
exactly what keeps the move unbiased despite the proposal being "adaptive" to the
current ensemble occupancy.

BANDWIDTH (eps, M) -- recommendation + the tension
--------------------------------------------------
M = a fixed within-mode covariance (honest preconditioner; identity if the space
is already whitened). eps = a SMALL multiple of a within-mode std. Tension
(own-knowledge, standard KDE reasoning):
  * eps too large -> proposals land off-manifold / overshoot a TIGHT mode -> low
    acceptance; also a wide kernel from one mode bleeds density into the gap.
  * eps too small -> proposals hug the chosen center (on-manifold, high
    acceptance WITHIN a covered mode) BUT q_C has vanishing density at any mode
    the complement does NOT currently occupy -> such modes can be neither entered
    nor (for a lone chain already there) left -> slow mixing for that mode.
A single global bandwidth is mismatched when modes have very different scales
(the carousel secondary mode is ~5x tighter than the global mode): pick eps
small enough that eps*sigma_global < sigma_secondary, i.e. eps ~ 0.1-0.3 here.
We SWEEP eps in validation rather than assert a value.

THE TWO HUMAN WORRIES -- how the construction (does / does not) handle them
---------------------------------------------------------------------------
(a) DRAINING a significant mode: cannot happen at stationarity. pi is the EXACT
    invariant distribution (detailed balance above), so a mode with appreciable
    mass keeps its mass in expectation; the Hastings q-ratio exactly cancels the
    fact that the proposal favors whichever mode the complement currently
    over-populates. The proposal need not "know" the weights. (Validated
    empirically by weight recovery + invariance-from-truth at 0.6/0.4.)
(b) A stuck chain OVER-representing a TINY mode: this is the genuine risk and the
    KNOWN weakness of KDE-independence couplers (own-knowledge). If the complement
    does not cover a tiny, isolated mode, the reverse proposal density q_C(z_i)
    at the lone chain's position is exponentially small, so the Hastings ratio
    suppresses LEAVING -> the chain can be PINNED -> the tiny mode is
    over-represented in finite time (even though stationarity is still pi). This
    is exactly why the validation pre-registers a TIME-AVERAGED tiny-mode
    occupancy test (must match the true weight within MC error; must NOT pin at
    1/n_chains; must NOT drain to 0). A config that pins or drains is REJECTED.

CITATIONS
  * Independence-sampler acceptance min(1, pi(y)q(x)/[pi(x)q(y)]) and detailed
    balance: standard MH theory (e.g. Tierney 1994; textbook).  [verified via
    secondary sources during this work].
  * Normal Kernel Coupler / KDE-proposal ensemble MCMC: Warnes (2001), Univ.
    Washington PhD thesis / tech report "The Normal Kernel Coupler".  [own-
    knowledge of the construction; primary PDF not machine-readable here].
  * Two-group / red-black frozen-complement exactness for ensemble difference
    proposals: ter Braak & Vrugt (2008), Stat. Comput. 18:435-446; Goodman &
    Weare (2010), CAMCoS 5:65-80.  [construction reproduced from own-knowledge].
  * Mode-jump context: Tjelmeland & Hegstad (2001); darting MC (Sminchisescu &
    Welling); Wormhole HMC (Lan et al. 2014).  [own-knowledge].

This module ONLY adds the kernel-hop. It does NOT modify de_mclmc.py / mclmc.py.
It is GENERIC in the target (takes a logdensity_fn), mirroring make_composite.
================================================================================
"""
import jax
import jax.numpy as jnp

# --- import the REAL MCLMC kernel primitives, read-only (same as de_mclmc.py) --
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap,
    isokinetic_mclachlan_smart,
    _single_init,
)


def make_kernel_hop_composite(logdensity_fn, D, n_chains, L, step_size, K,
                              eps=0.2, kernel_cov=None, p_hop=1.0,
                              inverse_mass_matrix=None,
                              integrator=isokinetic_mclachlan_smart):
    """Build a jitted composite-round fn: K MCLMC steps then ONE kernel-hop move.

    Parameters
    ----------
    logdensity_fn : callable z(D,) -> scalar logdensity (the target, pi)
    D             : dimension
    n_chains      : ensemble size (MUST be even; split into halves A/B)
    L, step_size  : MCLMC knobs (FIXED, no adaptation)
    K             : number of MCLMC steps per composite round
    eps           : kernel bandwidth multiplier. Kernel covariance = eps^2 * M.
                    SMALL but >0 (reversibility needs eps>0; NOT a delta).
    kernel_cov    : M, the (D,D) within-mode covariance shaping the kernel.
                    Default = identity (use when the space is whitened).
    p_hop         : per-chain probability of ATTEMPTING a hop each round (else
                    identity). Default 1.0. Does not affect correctness.
    inverse_mass_matrix : honest within-mode preconditioner for MCLMC (default
                    identity). DO NOT pass an MM covering both modes.

    Returns dict with: 'init_states', 'round', 'mclmc_only' (same API as
    de_mclmc.make_composite so the validation harness can swap movers).
    """
    assert n_chains % 2 == 0, "n_chains must be even for the two-group update"
    if inverse_mass_matrix is None:
        inverse_mass_matrix = jnp.eye(D)
    if kernel_cov is None:
        kernel_cov = jnp.eye(D)
    kernel_cov = jnp.asarray(kernel_cov)

    kernel = _build_kernel_shardmap(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inverse_mass_matrix,
        integrator=integrator,
    )

    # Cholesky of M (lower). Kernel draw: z' = center + eps * Lm @ xi.
    Lm = jnp.linalg.cholesky(kernel_cov)
    # log-normalizer of a single N(.; m, eps^2 M); it is an ADDITIVE constant in
    # log q, so it cancels in the log_q(z)-log_q(z') ratio -- kept for a correct
    # standalone log_q (harmless to the ratio).
    logdet_M = 2.0 * jnp.sum(jnp.log(jnp.diagonal(Lm)))
    lognorm = -0.5 * (D * jnp.log(2.0 * jnp.pi) + 2.0 * D * jnp.log(eps) + logdet_M)
    inv_eps2 = 1.0 / (eps * eps)

    # ---- init vmapped states ------------------------------------------------
    def init_states(positions, key):
        keys = jax.random.split(key, positions.shape[0])
        return jax.vmap(lambda p, k: _single_init(p, logdensity_fn, k))(positions, keys)

    # ---- K MCLMC steps, vmapped over chains, scanned over steps -------------
    def mclmc_k_steps(states, keys):
        def scan_step(carry, k_t):
            def per_chain(s, k):
                ns, info = kernel(rng_key=k, state=s, L=L, step_size=step_size)
                return ns, info.energy_change
            return jax.vmap(per_chain)(carry, k_t)
        final, ec = jax.lax.scan(scan_step, states, keys)
        return final, ec  # ec: (K, n)

    # ---- log q_C(z) : KDE density over the frozen complement C --------------
    def log_q(z, comp):
        # z:(D,), comp:(gc,D).  q_C(z) = (1/gc) sum_j N(z; comp_j, eps^2 M)
        diff = z[None, :] - comp                                   # (gc, D)
        w = jax.vmap(lambda d: jax.scipy.linalg.solve_triangular(Lm, d, lower=True))(diff)
        quad = jnp.sum(w * w, axis=1) * inv_eps2                   # (gc,)  Mahalanobis/eps^2
        logN = lognorm - 0.5 * quad                               # (gc,)
        return jax.scipy.special.logsumexp(logN) - jnp.log(comp.shape[0])

    # ---- kernel-hop update of ONE group using a FROZEN complement -----------
    def update_group(grp, comp, key):
        g = grp.shape[0]
        gc = comp.shape[0]
        keys = jax.random.split(key, g)

        def one(z_i, k):
            kpick, kxi, ku, kh = jax.random.split(k, 4)
            j = jax.random.randint(kpick, (), 0, gc)
            xi = jax.random.normal(kxi, (D,))
            z_prop = comp[j] + eps * (Lm @ xi)                    # ~ q_C
            # independence-sampler MH ratio with Hastings q-ratio q_C(z)/q_C(z')
            log_pi_ratio = logdensity_fn(z_prop) - logdensity_fn(z_i)
            log_q_ratio = log_q(z_i, comp) - log_q(z_prop, comp)
            log_alpha = log_pi_ratio + log_q_ratio
            attempt = jax.random.uniform(kh) < p_hop
            accept = jnp.logical_and(attempt, jnp.log(jax.random.uniform(ku)) < log_alpha)
            newz = jnp.where(accept, z_prop, z_i)
            return newz, accept.astype(jnp.float64)

        return jax.vmap(one)(grp, keys)

    # ---- one full kernel-hop move (two-group, RE-RANDOMIZED partition) ------
    def hop_move(positions, key):
        n = positions.shape[0]
        half = n // 2
        kperm, kA, kB = jax.random.split(key, 3)
        perm = jax.random.permutation(kperm, n)
        idxA, idxB = perm[:half], perm[half:]
        posA, posB = positions[idxA], positions[idxB]
        newA, accA = update_group(posA, posB, kA)                 # A | frozen B
        newB, accB = update_group(posB, newA, kB)                 # B | updated A
        new_positions = positions.at[idxA].set(newA).at[idxB].set(newB)
        acc = jnp.zeros(n, jnp.float64).at[idxA].set(accA).at[idxB].set(accB)
        return new_positions, acc

    # ---- one composite round: K MCLMC steps then one kernel-hop move --------
    @jax.jit
    def round_fn(states, key):
        kmc, khop, kinit = jax.random.split(key, 3)
        mc_keys = jax.random.split(kmc, K * n_chains).reshape(K, n_chains)
        states, ec = mclmc_k_steps(states, mc_keys)
        positions = states.position
        new_positions, acc = hop_move(positions, khop)
        new_states = init_states(new_positions, kinit)            # momentum refresh
        return new_states, (new_positions, ec, acc)

    # ---- pure MCLMC baseline (NO hop, NO forced refresh) --------------------
    @jax.jit
    def mclmc_only(states, keys):
        def scan_step(carry, k_t):
            def per_chain(s, k):
                ns, info = kernel(rng_key=k, state=s, L=L, step_size=step_size)
                return ns, ns.position
            return jax.vmap(per_chain)(carry, k_t)
        final, pos = jax.lax.scan(scan_step, states, keys)
        return final, pos

    return {
        "init_states": init_states,
        "round": round_fn,
        "mclmc_only": mclmc_only,
        "config": dict(D=D, n_chains=n_chains, L=L, step_size=step_size, K=K,
                       eps=eps, p_hop=p_hop),
    }
