"""Quick smoke: SA-MCMC mover runs, hops modes, recovers weight ~0.70 on the
EASY analytic D=10 mixture (modes +/-5 on axis0, weights [0.7,0.3]). Small run."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_move import make_sa_composite

D = 10; m = 5.0
W = np.array([0.7, 0.3]); MU = np.array([+m, -m])
_logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU)
_c = -0.5 * D * jnp.log(2 * jnp.pi)

def logdensity_fn(z):
    z0 = z[0]; qr = jnp.sum(z[1:] ** 2)
    c0 = _logW[0] + _c - 0.5 * ((z0 - _MU[0]) ** 2 + qr)
    c1 = _logW[1] + _c - 0.5 * ((z0 - _MU[1]) ** 2 + qr)
    return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))

def run(comp, init_pos, n_rounds, key, N):
    st = comp["init_states"](init_pos, jax.random.fold_in(key, 0))
    keys = jax.random.split(key, n_rounds)
    pos = np.empty((n_rounds, N, D)); subs = []
    for r in range(n_rounds):
        st, (p, ec, sub) = comp["round"](st, keys[r])
        pos[r] = np.asarray(p); subs.append(float(np.asarray(sub).mean()))
    return pos, np.array(subs)

if __name__ == "__main__":
    N = 32
    for prop in ["gaussian", "mixture"]:
        comp = make_sa_composite(logdensity_fn, D, N, L=2.0, step_size=0.5, K=5,
                                 n_sa=N, proposal=prop, prop_scale=1.0,
                                 bandwidth=1.0)
        init = jnp.zeros((N, D)).at[:, 0].set(m)        # ALL in +mode
        t0 = time.time()
        pos, subs = run(comp, init, 400, jax.random.key(1), N)
        z0 = pos[200:, :, 0]
        wp = float((z0 > 0).mean())
        print(f"[{prop:8s}] 400 rounds in {time.time()-t0:.1f}s | "
              f"weight(+mode)={wp:.3f} (truth 0.70) | sub-rate={subs.mean():.3f} | "
              f"any -mode chains={int((z0<0).sum())}")
