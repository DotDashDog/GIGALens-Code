"""Standalone composite sampler: MCLMC local moves interleaved with a cross-chain
ensemble move, with THREE selectable move types -- the literature constructions
the user asked about, wrapping the REAL MCLMC kernel with minimal change.

This COPIES the de_mclmc.py two-group frozen-complement scaffolding (NOT imported /
modified) and swaps the proposal. It modifies no shared module; it imports the REAL
MCLMC kernel primitives read-only exactly as de_mclmc.py does.

MOVE TYPES (move=...)
=====================
'gamma1'  -- ter Braak (2006) DE-MC.  z' = z_i + gamma*(z_a - z_b) + eps,
             gamma = gamma_big = 2.38/sqrt(2D) with prob (1-p_jump), gamma = 1.0
             with prob p_jump (the periodic gamma=1 mode-jump). Ordered pairs (a,b)
             drawn uniformly => SYMMETRIC proposal => log-Hastings = 0. UNBIASED by
             the same Metropolis-within-Gibbs argument as de_mclmc.py. [VERIFIED
             construction: ter Braak 2006, every-10th-gen gamma=1.]

'near'    -- gamma=1 NEAR-TELEPORT (the user's idea).  Partner z_b chosen as the
             NEAREST complement chain to z_i, partner z_a uniform over the rest:
                 z' = z_a + (z_i - z_b*) + eps,   b* = argmin_{j in C} ||z_i - z_j||.
             Because z_b* ~ z_i, z' ~ z_a (a near-teleport ONTO on-manifold chain a).
             The nearest-neighbour selection is ASYMMETRIC, so the move carries a
             Hastings q-ratio. eps>0 (absolute continuity). The proposal density is
                 q(z'|z_i) = (1/(|C|-1)) sum_{a != b*(z_i)} N(z'; z_a + z_i - z_b*(z_i), eps^2 I)
             and the EXACT MH acceptance is
                 alpha = min(1, pi(z') q(z_i|z') / [pi(z_i) q(z'|z_i)]),
             with b*(z') the nearest complement chain to z'. This is computed
             exactly (|C| small). UNBIASED for any eps>0 (detailed balance of a
             state-dependent-proposal MH step conditional on the frozen complement).

'snooker' -- ter Braak & Vrugt (2008) snooker updater [Jacobian VERIFIED from the
             DREAM-Suite source, Calc_proposal.m / Metropolis_rule.m]:
                 c,a,b ~ distinct complement;  F = z_i - z_c;
                 zp = F * ((z_a - z_b).F / F.F);   gamma_s ~ U(1.2, 2.2);
                 z' = z_i + gamma_s * zp + eps.
             Acceptance carries the radial Jacobian (d-1):
                 alpha = min(1, (||z'-z_c|| / ||z_i-z_c||)^(D-1) * pi(z')/pi(z_i)).
             OMITTING the (d-1) factor BIASES the sampler -- exposed via
             drop_jacobian=True for the bias demonstration.

All three use the de_mclmc.py two-group, RE-RANDOMIZED-partition red/black update
(A | frozen B, then B | updated A), which preserves the product target; MCLMC
momentum is refreshed after each move (auxiliary, no position bias).
"""
import jax
import jax.numpy as jnp

from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap,
    isokinetic_mclachlan_smart,
    _single_init,
)


def make_teleport_composite(logdensity_fn, D, n_chains, L, step_size, K,
                            move="gamma1", b0=0.05, p_jump=0.5, eps=0.2,
                            drop_jacobian=False, inverse_mass_matrix=None,
                            eps_scale=None, integrator=isokinetic_mclachlan_smart):
    """move in {'gamma1','near','snooker'}. Returns dict with the de_mclmc API:
    'init_states', 'round', 'mclmc_only', 'config'."""
    assert n_chains % 2 == 0
    assert move in ("gamma1", "near", "snooker")
    if move == "near":
        # the 'near' KDE Hastings ratio below assumes ISOTROPIC jitter N(0, b0^2 I);
        # a non-isotropic eps_scale would make log_q wrong.
        assert eps_scale is None, "move='near' requires isotropic jitter (eps_scale=None)"
    if inverse_mass_matrix is None:
        inverse_mass_matrix = jnp.eye(D)
    kernel = _build_kernel_shardmap(logdensity_fn=logdensity_fn,
                                    inverse_mass_matrix=inverse_mass_matrix,
                                    integrator=integrator)
    gamma_big = 2.38 / jnp.sqrt(2.0 * D)
    eps_scale_v = jnp.ones(D) if eps_scale is None else jnp.asarray(eps_scale)

    def _eps(xi):
        return b0 * (eps_scale_v @ xi if eps_scale_v.ndim == 2 else eps_scale_v * xi)

    def init_states(positions, key):
        keys = jax.random.split(key, positions.shape[0])
        return jax.vmap(lambda p, k: _single_init(p, logdensity_fn, k))(positions, keys)

    def mclmc_k_steps(states, keys):
        def scan_step(carry, k_t):
            def per_chain(s, k):
                ns, info = kernel(rng_key=k, state=s, L=L, step_size=step_size)
                return ns, info.energy_change
            return jax.vmap(per_chain)(carry, k_t)
        return jax.lax.scan(scan_step, states, keys)

    # ----------------------------- proposal kernels --------------------------
    def _gamma1(z_i, comp, k):
        gc = comp.shape[0]
        ka, kb, ke, kg, ku = jax.random.split(k, 5)
        a = jax.random.randint(ka, (), 0, gc)
        b = jax.random.randint(kb, (), 0, gc - 1)
        b = jnp.where(b >= a, b + 1, b)
        use_jump = jax.random.uniform(kg) < p_jump
        gamma = jnp.where(use_jump, 1.0, gamma_big)
        prop = z_i + gamma * (comp[a] - comp[b]) + _eps(jax.random.normal(ke, (D,)))
        log_alpha = logdensity_fn(prop) - logdensity_fn(z_i)        # symmetric
        return prop, log_alpha, ku

    def _near(z_i, comp, k):
        gc = comp.shape[0]
        ka, ke, ku = jax.random.split(k, 3)
        # b* = nearest complement to z_i
        d2 = jnp.sum((z_i[None, :] - comp) ** 2, axis=1)
        bstar = jnp.argmin(d2)
        # a uniform over comp \ {b*}
        a = jax.random.randint(ka, (), 0, gc - 1)
        a = jnp.where(a >= bstar, a + 1, a)
        prop = comp[a] + (z_i - comp[bstar]) + _eps(jax.random.normal(ke, (D,)))

        # log q(y | x) = logsumexp_a' N(y; comp[a'] + x - comp[b*(x)], eps^2 I) - log(gc-1)
        def log_q(y, x):
            bx = jnp.argmin(jnp.sum((x[None, :] - comp) ** 2, axis=1))
            centers = comp + (x - comp[bx])[None, :]               # (gc,D); a'=bx -> x itself
            diff = y[None, :] - centers
            # exclude a' = bx (center == x) : set its logN to -inf
            quad = jnp.sum(diff * diff, axis=1) / (b0 * b0)
            logN = -0.5 * quad                                     # drop const (cancels in ratio)
            logN = jnp.where(jnp.arange(gc) == bx, -jnp.inf, logN)
            return jax.scipy.special.logsumexp(logN)               # -log(gc-1) cancels
        log_hast = log_q(z_i, prop) - log_q(prop, z_i)
        log_alpha = (logdensity_fn(prop) - logdensity_fn(z_i)) + log_hast
        return prop, log_alpha, ku

    def _snooker(z_i, comp, k):
        gc = comp.shape[0]
        kabc, ks, ke, ku = jax.random.split(k, 4)
        idx = jax.random.permutation(kabc, gc)[:3]
        za, zb, zc = comp[idx[0]], comp[idx[1]], comp[idx[2]]
        F = z_i - zc
        FF = jnp.maximum(jnp.dot(F, F), 1e-300)
        zp = F * (jnp.dot(za - zb, F) / FF)
        gs = 1.2 + jax.random.uniform(ks)
        prop = z_i + gs * zp + _eps(jax.random.normal(ke, (D,)))
        XpZ = jnp.linalg.norm(prop - zc) + 1e-300
        XZ = jnp.linalg.norm(z_i - zc) + 1e-300
        log_jac = (D - 1) * jnp.log(XpZ / XZ)
        log_jac = jnp.where(drop_jacobian, 0.0, log_jac)
        log_alpha = (logdensity_fn(prop) - logdensity_fn(z_i)) + log_jac
        return prop, log_alpha, ku

    proposer = {"gamma1": _gamma1, "near": _near, "snooker": _snooker}[move]

    def update_group(grp, comp, key):
        keys = jax.random.split(key, grp.shape[0])
        def one(z_i, k):
            prop, log_alpha, ku = proposer(z_i, comp, k)
            accept = jnp.log(jax.random.uniform(ku)) < log_alpha
            newz = jnp.where(accept, prop, z_i)
            return newz, accept.astype(jnp.float64)
        return jax.vmap(one)(grp, keys)

    def de_move(positions, key):
        n = positions.shape[0]; half = n // 2
        kperm, kA, kB = jax.random.split(key, 3)
        perm = jax.random.permutation(kperm, n)
        idxA, idxB = perm[:half], perm[half:]
        posA, posB = positions[idxA], positions[idxB]
        newA, accA = update_group(posA, posB, kA)
        newB, accB = update_group(posB, newA, kB)
        new_positions = positions.at[idxA].set(newA).at[idxB].set(newB)
        acc = jnp.zeros(n, jnp.float64).at[idxA].set(accA).at[idxB].set(accB)
        return new_positions, acc

    @jax.jit
    def round_fn(states, key):
        kmc, kde, kinit = jax.random.split(key, 3)
        mc_keys = jax.random.split(kmc, K * n_chains).reshape(K, n_chains)
        states, ec = mclmc_k_steps(states, mc_keys)
        positions = states.position
        new_positions, acc = de_move(positions, kde)
        new_states = init_states(new_positions, kinit)
        return new_states, (new_positions, ec, acc)

    @jax.jit
    def mclmc_only(states, keys):
        def scan_step(carry, k_t):
            def per_chain(s, k):
                ns, info = kernel(rng_key=k, state=s, L=L, step_size=step_size)
                return ns, ns.position
            return jax.vmap(per_chain)(carry, k_t)
        return jax.lax.scan(scan_step, states, keys)

    return {"init_states": init_states, "round": round_fn, "mclmc_only": mclmc_only,
            "gamma_big": float(gamma_big),
            "config": dict(D=D, n_chains=n_chains, L=L, step_size=step_size, K=K,
                           move=move, b0=b0, p_jump=p_jump, eps=eps,
                           drop_jacobian=drop_jacobian)}
