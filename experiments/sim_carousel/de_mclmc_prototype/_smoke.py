import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(__file__))
from de_mclmc import make_composite

D = 10
m = 3.5
W = jnp.array([0.7, 0.3])
MU = jnp.array([+m, -m])  # along axis 0
logW = jnp.log(W)
c = -0.5 * D * jnp.log(2 * jnp.pi)

def logdensity_fn(z):
    z0 = z[0]
    rest = z[1:]
    quad_rest = jnp.sum(rest**2)
    comp0 = logW[0] + c - 0.5 * ((z0 - MU[0])**2 + quad_rest)
    comp1 = logW[1] + c - 0.5 * ((z0 - MU[1])**2 + quad_rest)
    return jax.scipy.special.logsumexp(jnp.stack([comp0, comp1]))

# single-mode MCLMC test: does the kernel mix WITHIN a mode and stay (no cross)?
n = 8
comp = make_composite(logdensity_fn, D, n, L=2.0, step_size=0.5, K=1)
key = jax.random.key(0)
pos0 = jnp.zeros((n, D)).at[:, 0].set(m)  # all in +mode
st = comp["init_states"](pos0, key)
steps = 500
keys = jax.random.split(jax.random.key(1), steps * n).reshape(steps, n)
t0 = time.time()
st, pos = comp["mclmc_only"](st, keys)
pos = np.asarray(pos)  # (steps, n, D)
print("mclmc_only ran", steps, "steps in", round(time.time() - t0, 2), "s")
z0 = pos[:, :, 0]
print("axis0 mean", z0.mean(), "std", z0.std(), "min", z0.min(), "max", z0.max())
frac_pos = (z0 > 0).mean()
print("fraction in +mode (should stay ~1.0):", frac_pos)
print("within-mode axis1 mean/std", pos[:, :, 1].mean(), pos[:, :, 1].std())

# one composite round timing
comp2 = make_composite(logdensity_fn, D, 64, L=2.0, step_size=0.5, K=20)
st2 = comp2["init_states"](jnp.zeros((64, D)).at[:, 0].set(m), jax.random.key(2))
t0 = time.time()
st2, (p, ec, acc) = comp2["round"](st2, jax.random.key(3))
p.block_until_ready()
print("one composite round (compile+run):", round(time.time() - t0, 2), "s")
t0 = time.time()
for i in range(10):
    st2, (p, ec, acc) = comp2["round"](st2, jax.random.key(100 + i))
p.block_until_ready()
print("10 rounds (run):", round(time.time() - t0, 2), "s; DE accept mean", float(np.asarray(acc).mean()),
      "mclmc |energy_change| mean", float(np.abs(np.asarray(ec)).mean()))
