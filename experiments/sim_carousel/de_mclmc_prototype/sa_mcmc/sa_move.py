"""Standalone composite sampler: MCLMC local moves interleaved with a
SAMPLE-ADAPTIVE MCMC (Zhu, NeurIPS 2019) self-INCLUSIVE cross-chain move.

================================================================================
THE ALGORITHM  (Michael H. Zhu, "Sample Adaptive MCMC", NeurIPS 2019)
================================================================================
VERIFIED from the readable NeurIPS proceedings PDF (Algorithm 1, Section 3, and
Proposition 1).  Source:
  https://papers.nips.cc/paper/2019/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf

State = N points  S = (theta_1, ..., theta_N).  Target over the ensemble is the
PRODUCT  pi^{otimes N}(S) = prod_n pi(theta_n).  Define
    mu(S)    = (1/N) sum_n theta_n              (sample mean of the N points)
    Sigma(S) = sample covariance of the N points  (optionally diagonal)
and a proposal family q(. | mu(S), Sigma(S)) (Zhu uses a Gaussian or Gaussian
scale-mixture).  ONE SA-MCMC iteration (Algorithm 1, verbatim structure):

  1. S = (theta_1..theta_N)  <- current state.
  2. Sample a proposed point         theta_{N+1} ~ q(. | mu(S), Sigma(S)).
  3. Form the (N+1)-point augmented set T = (theta_1..theta_N, theta_{N+1}).
     For n = 1..N:   S_{-n}    = T with theta_n removed  (= S with theta_n
                                  replaced by theta_{N+1}).
     For n = N+1:    S_{-(N+1)} = S  (= T with the proposal removed).
     i.e.  S_{-n} = T minus {theta_n}   for ALL n = 1..N+1.
  4. SUBSTITUTION WEIGHTS (the self-inclusive acceptance):
         lambda_n = q(theta_n | mu(S_{-n}), Sigma(S_{-n})) / p(theta_n),
                                                       n = 1..N+1.
     (an INVERSE importance weight: low-target / high-proposal points get
      deleted preferentially.)
  5. Sample j with  P[J=n] = lambda_n / sum_i lambda_i.
  6. Next state  S(k) = S_{-j}.   (j<=N -> substitute proposal into slot j;
     j=N+1 -> reject, keep S.)

Only ONE new target evaluation per iteration (p(theta_{N+1}); the other p(theta_n)
are cached). The lambda's use only CHEAP proposal-density evaluations.

WHY IT IS UNBIASED  (Proposition 1: detailed balance w.r.t. pi^{otimes N})
--------------------------------------------------------------------------
[Statement VERIFIED in the paper; the one-line balance below is reconstructed
own-knowledge, but it reproduces the paper's claim exactly.]  Consider a move
S -> S' that replaces theta_a by a new point y.  It requires: proposed
theta_{N+1}=y, then deleted j=a.  The augmented set is T = S u {y} = Sprime u {theta_a}
-- IDENTICAL in the forward and reverse directions.  Let
    Z(T) = sum_{t in T} q(t | T-minus-t) / p(t)  (symmetric in the set T).
Forward:  pi^{xN}(S) P(S->S')
        = [pi(theta_a) prod_{n!=a} pi(theta_n)] * q(y|S) * [q(theta_a|S')/p(theta_a)]/Z(T).
Using pi(theta_a)/p(theta_a) = 1/C (C = normalizer), this is
        (1/C) prod_{n!=a} pi(theta_n) * q(y|S) * q(theta_a|S') / Z(T),
which is SYMMETRIC under (S,y,a) <-> (S',theta_a,a) because q(y|S) and
q(theta_a|S') swap and Z(T) is unchanged.  Hence detailed balance holds and
pi^{otimes N} is stationary: at stationarity each of the N points is marginally
~ pi.  *** The proof uses NOTHING about q being a fitted Gaussian -- only that
q(.|S) depends on S through the unordered SET. So ANY set-symmetric proposal is
admissible, including a self-inclusive KDE/mixture over the points. ***  (Zhu
proves ergodicity, Theorem 1, specifically for a DIAGONAL covariance proposal.)

SELF-INCLUSIVE vs the kernel-hop's SELF-EXCLUSION
-------------------------------------------------
kernel_hop.py builds an INDEPENDENCE-MH proposal from the COMPLEMENT (excludes
chain i); the reverse density q_C(z_i) is evaluated at i's own location with i
EXCLUDED -> a valley -> rejection at small bandwidth (knife edge).  SA-MCMC is
self-inclusive: theta_n's keep/delete weight lambda_n uses q evaluated on the set
S_{-n} that INCLUDES the proposed point and all other current points, and the
proposal that drew theta_{N+1} also includes theta_n.  There is no "valley at your
own location"; the deletion mechanism (not a plain Hastings ratio) is what keeps
it reversible.

TWO PROPOSAL FAMILIES IMPLEMENTED
---------------------------------
proposal="gaussian"  (the literal Zhu variant, SIMPLEST FIRST): a single Gaussian
    N(mu(S), prop_scale^2 Sigma(S)) fitted to the N points.  For a bimodal set
    this is a broad proposal covering both modes -> can hop, but is OVER-DISPERSED
    relative to a thin curved within-mode ridge.
proposal="mixture"  (curvature-aware complexity, added only if/when needed): a
    self-inclusive KDE  q(.|S) = (1/N) sum_m N(.; theta_m, bandwidth^2 kernel_cov).
    Proposes NEAR an existing on-manifold point -> curvature robust.  Still exact
    by the same Proposition-1 argument (set-symmetric proposal).

This module ONLY adds the SA move + composite. It imports the REAL MCLMC kernel
primitives READ-ONLY (as de_mclmc.py / kernel_hop.py do) and modifies nothing.
================================================================================
"""
import jax
import jax.numpy as jnp

from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap,
    isokinetic_mclachlan_smart,
    _single_init,
)


def make_sa_composite(logdensity_fn, D, n_chains, L, step_size, K,
                      n_sa=None, proposal="gaussian", prop_scale=1.0,
                      diag_cov=False, bandwidth=0.3, kernel_cov=None,
                      rho_broad=0.1, cov_reg=1e-6, inverse_mass_matrix=None,
                      integrator=isokinetic_mclachlan_smart):
    """Build a jitted composite-round fn: K MCLMC steps then n_sa SA-MCMC moves.

    Parameters
    ----------
    logdensity_fn : callable z(D,) -> scalar  (log of UNNORMALIZED target p)
    D, n_chains   : dimension, ensemble size N (any N>=3; need not be even)
    L, step_size  : MCLMC knobs (FIXED, no adaptation)
    K             : MCLMC steps per composite round (K=0 -> pure SA, for isolating)
    n_sa          : number of SA-MCMC substitution iterations per round
                    (default N).  Each changes AT MOST one point.
    proposal      : "gaussian" (Zhu single fitted Gaussian) or "mixture"
                    (self-inclusive KDE over the points).
    prop_scale    : scalar multiplier on the fitted covariance (gaussian only).
    diag_cov      : use diagonal of the fitted covariance (gaussian only; this is
                    the variant Zhu's ergodicity Theorem 1 covers).
    bandwidth     : KDE bandwidth multiplier (mixture only).
    kernel_cov    : (D,D) local cov shaping the KDE kernel (mixture only;
                    default identity).
    cov_reg       : ridge added to every proposal covariance for numerical PD.

    Returns dict: 'init_states', 'round', 'mclmc_only' (same swap-in API as
    de_mclmc.make_composite / kernel_hop).
    """
    N = int(n_chains)
    assert N >= 3, "SA-MCMC needs N>=3 points"
    if n_sa is None:
        n_sa = N
    if inverse_mass_matrix is None:
        inverse_mass_matrix = jnp.eye(D)
    if kernel_cov is None:
        kernel_cov = jnp.eye(D)
    kernel_cov = jnp.asarray(kernel_cov)
    Lk = jnp.linalg.cholesky(kernel_cov)            # KDE kernel chol (mixture)
    reg = cov_reg * jnp.eye(D)

    kernel = _build_kernel_shardmap(
        logdensity_fn=logdensity_fn,
        inverse_mass_matrix=inverse_mass_matrix,
        integrator=integrator,
    )

    # ---------- multivariate-normal logpdf with explicit (mean, cov) ----------
    def _logmvn(x, mean, cov):
        Lc = jnp.linalg.cholesky(cov)
        d = x - mean
        y = jax.scipy.linalg.solve_triangular(Lc, d, lower=True)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(Lc)))
        return -0.5 * (jnp.dot(y, y) + D * jnp.log(2.0 * jnp.pi) + logdet)

    # ---------- init vmapped MCLMC states -------------------------------------
    def init_states(positions, key):
        keys = jax.random.split(key, positions.shape[0])
        return jax.vmap(lambda p, k: _single_init(p, logdensity_fn, k))(positions, keys)

    # ---------- K MCLMC steps, vmapped over chains, scanned over steps ---------
    def mclmc_k_steps(states, keys):
        def scan_step(carry, k_t):
            def per_chain(s, k):
                ns, info = kernel(rng_key=k, state=s, L=L, step_size=step_size)
                return ns, info.energy_change
            return jax.vmap(per_chain)(carry, k_t)
        final, ec = jax.lax.scan(scan_step, states, keys)
        return final, ec

    # =================== ONE SA-MCMC substitution iteration ====================
    # carry: (S (N,D), logp_S (N,)).  Returns new carry + substitution flag.
    # ---- component log-densities of a set-symmetric proposal, evaluated for the
    #      DRAW (point x given set S) and for ALL leave-one-out sets (vectorized).
    def _gauss_logq_at(x, Sset):
        mu = jnp.mean(Sset, axis=0); covS = jnp.cov(Sset.T)
        if diag_cov:
            covS = jnp.diag(jnp.diagonal(covS))
        return _logmvn(x, mu, prop_scale * prop_scale * covS + reg)

    def _gauss_logq_loo(T, M):
        sum1 = jnp.sum(T, axis=0); sum2 = T.T @ T
        def loo(n):
            t_n = T[n]; s1 = sum1 - t_n; mean_n = s1 / N
            s2 = sum2 - jnp.outer(t_n, t_n)
            cov_n = (s2 - jnp.outer(s1, s1) / N) / (N - 1)
            if diag_cov:
                cov_n = jnp.diag(jnp.diagonal(cov_n))
            return _logmvn(t_n, mean_n, prop_scale * prop_scale * cov_n + reg)
        return jax.vmap(loo)(jnp.arange(M))

    def _kde_logq_loo(T, M):                                   # (1/N) sum_{m!=n} N(T_n;T_m,H)
        Hc = bandwidth * Lk
        logdetH = 2.0 * jnp.sum(jnp.log(jnp.diagonal(Hc)))
        lognorm = -0.5 * (D * jnp.log(2.0 * jnp.pi) + logdetH)
        def pair(n, mm):
            d = T[n] - T[mm]
            w = jax.scipy.linalg.solve_triangular(Hc, d, lower=True)
            return lognorm - 0.5 * jnp.dot(w, w)
        G = jax.vmap(lambda n: jax.vmap(lambda mm: pair(n, mm))(jnp.arange(M)))(jnp.arange(M))
        G_off = G + jnp.diag(jnp.full((M,), -jnp.inf))         # exclude self (diagonal)
        return jax.scipy.special.logsumexp(G_off, axis=1) - jnp.log(N)

    def sa_step(carry, key):
        S, logp_S = carry
        kpick, kxi, kj, kc = jax.random.split(key, 4)
        lr = jnp.log(rho_broad); l1r = jnp.log1p(-rho_broad)

        # ---- (2) draw proposed point theta_{N+1} ~ q(.|S) --------------------
        if proposal == "gaussian":
            mu = jnp.mean(S, axis=0); covS = jnp.cov(S.T)
            if diag_cov:
                covS = jnp.diag(jnp.diagonal(covS))
            y = mu + jnp.linalg.cholesky(prop_scale * prop_scale * covS + reg) @ jax.random.normal(kxi, (D,))
        else:  # "mixture" (KDE) or "combined" (KDE + rho*broad-Gaussian)
            m = jax.random.randint(kpick, (), 0, N)
            y_kde = S[m] + bandwidth * (Lk @ jax.random.normal(kxi, (D,)))
            if proposal == "combined":
                mu = jnp.mean(S, axis=0); covS = jnp.cov(S.T)
                if diag_cov:
                    covS = jnp.diag(jnp.diagonal(covS))
                y_broad = mu + jnp.linalg.cholesky(prop_scale * prop_scale * covS + reg) @ jax.random.normal(kxi, (D,))
                y = jnp.where(jax.random.uniform(kc) < rho_broad, y_broad, y_kde)
            else:
                y = y_kde
        logp_y = logdensity_fn(y)

        # ---- (3) augmented set T (N+1, D) and cached logp --------------------
        T = jnp.concatenate([S, y[None, :]], axis=0)
        logp_T = jnp.concatenate([logp_S, logp_y[None]])
        M = N + 1

        # ---- (4) lambda_n = q(theta_n | S_{-n}) / p(theta_n) -----------------
        if proposal == "gaussian":
            logq = _gauss_logq_loo(T, M)
        elif proposal == "mixture":
            logq = _kde_logq_loo(T, M)
        else:  # combined: logq = logaddexp(log(1-rho)+kde, log(rho)+gauss)
            logq = jnp.logaddexp(l1r + _kde_logq_loo(T, M), lr + _gauss_logq_loo(T, M))

        log_lambda = logq - logp_T                             # (M,)

        # ---- (5) sample j ~ Categorical(lambda) ------------------------------
        j = jax.random.categorical(kj, log_lambda)             # in 0..N
        substituted = (j < N)

        # ---- (6) S(k) = S_{-j}.  j<N: replace row j by y; j==N: reject. -------
        newS = jnp.where(substituted, S.at[jnp.clip(j, 0, N - 1)].set(y), S)
        new_logp = jnp.where(substituted,
                             logp_S.at[jnp.clip(j, 0, N - 1)].set(logp_y), logp_S)
        return (newS, new_logp), substituted.astype(jnp.float64)

    def sa_phase(S, key):
        logp_S = jax.vmap(logdensity_fn)(S)
        keys = jax.random.split(key, n_sa)
        (Sf, _), subs = jax.lax.scan(sa_step, (S, logp_S), keys)
        return Sf, subs                                        # subs: (n_sa,)

    # ---------- one composite round: K MCLMC steps then n_sa SA moves ----------
    @jax.jit
    def round_fn(states, key):
        kmc, ksa, kinit = jax.random.split(key, 3)
        mc_keys = jax.random.split(kmc, K * N).reshape(K, N) if K > 0 else None
        if K > 0:
            states, ec = mclmc_k_steps(states, mc_keys)
        else:
            ec = jnp.zeros((1, N))
        S = states.position
        Sf, subs = sa_phase(S, ksa)
        new_states = init_states(Sf, kinit)                    # momentum refresh
        return new_states, (Sf, ec, subs)

    # ---------- pure MCLMC baseline (NO SA, NO forced refresh) -----------------
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
        "config": dict(D=D, n_chains=N, L=L, step_size=step_size, K=K, n_sa=n_sa,
                       proposal=proposal, prop_scale=prop_scale, diag_cov=diag_cov,
                       bandwidth=bandwidth, rho_broad=rho_broad),
    }
