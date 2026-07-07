"""Diagnostic: map vanilla-MCLMC crossing vs DE-MCLMC behaviour as a function of
mode separation m (free-energy barrier = m^2/2). Find m where vanilla is TRAPPED
within the run (so the contrast is meaningful) and check DE behaviour there.
Hypothesis: barrier ~ m^2/2; vanilla crosses readily for m=3.5 (barrier 6.1) but
should be trapped by m~5 (barrier 12.5, ~600x rarer). DE-from-truth should mix at
ALL m (cross-mode difference vectors ~2m are exactly the jump needed)."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from de_mclmc import make_composite

D, N_CHAINS = 10, 64
L, STEP, K = 2.0, 0.5, 20

def make_logp(m):
    W = jnp.log(jnp.array([0.7, 0.3])); MU = jnp.array([+m, -m])
    c = -0.5 * D * jnp.log(2*jnp.pi)
    def lp(z):
        z0 = z[0]; qr = jnp.sum(z[1:]**2)
        a = W[0] + c - 0.5*((z0-MU[0])**2 + qr)
        b = W[1] + c - 0.5*((z0-MU[1])**2 + qr)
        return jax.scipy.special.logsumexp(jnp.stack([a, b]))
    return lp

def vanilla_tail(comp, m, total_steps, seed):
    st = comp["init_states"](jnp.zeros((N_CHAINS, D)).at[:, 0].set(m), jax.random.key(seed))
    chunk = 2500; done = 0; tail = None
    while done < total_steps:
        nthis = min(chunk, total_steps - done)
        ck = jax.random.split(jax.random.fold_in(jax.random.key(seed+1), done), nthis*N_CHAINS).reshape(nthis, N_CHAINS)
        st, posv = comp["mclmc_only"](st, ck)
        tail = np.asarray(posv[:, :, 0]); done += nthis
    return float((tail > 0).mean())

def de_run(comp, init_pos, rounds, seed):
    st = comp["init_states"](init_pos, jax.random.key(seed))
    keys = jax.random.split(jax.random.key(seed+1), rounds)
    fr = np.empty(rounds)
    for r in range(rounds):
        st, (p, ec, acc) = comp["round"](st, keys[r])
        fr[r] = (np.asarray(p)[:, 0] > 0).mean()
    return fr

for m in [4.0, 4.5, 5.0, 5.5]:
    lp = make_logp(m)
    comp = make_composite(lp, D, N_CHAINS, L=L, step_size=STEP, K=K, b0=0.05, p_jump=0.2)
    t0 = time.time()
    vt = vanilla_tail(comp, m, total_steps=2000*K, seed=10)
    # DE from single mode (discovery test)
    fr_single = de_run(comp, jnp.zeros((N_CHAINS, D)).at[:, 0].set(m), 1500, seed=20)
    # DE from truth (mixing/unbiasedness test)
    rng = np.random.default_rng(7)
    cl = (rng.random(N_CHAINS) >= 0.7).astype(int); zt = rng.standard_normal((N_CHAINS, D)); zt[:,0]+= np.where(cl==0, m, -m)
    fr_truth = de_run(comp, jnp.asarray(zt), 1500, seed=30)
    print(f"m={m} barrier={m*m/2:.1f} | vanilla_tail_+frac={vt:.3f} | "
          f"DE_single_final200={fr_single[-200:].mean():.3f} (min {fr_single.min():.3f}) | "
          f"DE_truth_final200={fr_truth[-200:].mean():.3f}  [{time.time()-t0:.0f}s]")
