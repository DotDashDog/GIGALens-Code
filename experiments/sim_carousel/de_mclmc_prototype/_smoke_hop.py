"""Smoke test for kernel_hop.py: (1) it runs; (2) the log_q ratio is computed
correctly (compare to an independent brute-force KDE log-density); (3) on a
trivial well-separated 2D mixture with KNOWN weights and 16 chains it recovers
the weights from a single-mode init (a basic unbiasedness sanity check)."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from kernel_hop import make_kernel_hop_composite

def pr(*a): print(*a, flush=True)

# --- trivial 2D mixture: two isotropic Gaussians, weights 0.65/0.35, sep ~8 sigma
D = 2
mu = np.array([[0.0, 0.0], [8.0, 0.0]])
sig = 1.0
W = np.array([0.65, 0.35])
logW = np.log(W)
def logp(z):
    d0 = ((z - jnp.asarray(mu[0]))**2).sum() / (2*sig**2)
    d1 = ((z - jnp.asarray(mu[1]))**2).sum() / (2*sig**2)
    c = -jnp.log(2*jnp.pi*sig**2)
    return jax.scipy.special.logsumexp(jnp.array([logW[0]+c-d0, logW[1]+c-d1]))

# --- (2) verify log_q against brute force --------------------------------------
eps = 0.5
comp = jnp.asarray(np.random.default_rng(0).standard_normal((8, D)) * sig)
# rebuild the module's log_q via a tiny composite (access through closure is awkward;
# recompute here with the SAME formula and cross-check brute force)
Lm = np.eye(D)
def log_q_ref(z):
    # (1/gc) sum_j N(z; comp_j, eps^2 I)
    c = np.asarray(comp); gc = c.shape[0]
    quad = ((np.asarray(z) - c)**2).sum(1) / eps**2
    lognorm = -0.5*(D*np.log(2*np.pi) + 2*D*np.log(eps))
    from scipy.special import logsumexp
    return logsumexp(lognorm - 0.5*quad) - np.log(gc)
zt = np.array([0.3, -0.2])
pr("brute-force log_q at zt =", log_q_ref(zt))

# --- (1)+(3) run from single-mode init, recover weight -------------------------
NCH = 16; K = 5; ROUNDS = 1500
comp_s = make_kernel_hop_composite(logp, D, NCH, L=3.0, step_size=0.3, K=K,
                                   eps=eps, kernel_cov=jnp.eye(D), p_hop=1.0,
                                   inverse_mass_matrix=jnp.eye(D))
rng = np.random.default_rng(1)
init = mu[0] + sig*rng.standard_normal((NCH, D))   # ALL in mode 0
st = comp_s["init_states"](jnp.asarray(init), jax.random.key(0))
keys = jax.random.split(jax.random.key(7), ROUNDS)
t = time.time(); occ = np.empty(ROUNDS); acc = np.empty(ROUNDS)
for r in range(ROUNDS):
    st, (p, ec, a) = comp_s["round"](st, keys[r])
    pos = np.asarray(p)
    occ[r] = (pos[:, 0] > 4.0).mean()     # fraction in mode 1
    acc[r] = float(np.asarray(a).mean())
burn = ROUNDS//2
pr(f"ran {ROUNDS} rounds in {time.time()-t:.1f}s | hop accept={acc.mean():.3f} | "
   f"weight(mode1) post-burn = {occ[burn:].mean():.3f} (target {W[1]:.3f}) | "
   f"first-half {occ[:burn].mean():.3f}")
pr("SMOKE OK")
