import os
os.environ["JAX_PLATFORMS"]="cpu"
os.environ["JAX_ENABLE_X64"]="0"  # float32, like production
import jax, jax.numpy as jnp, functools
from collections import namedtuple

# --- Structural test 1: does kernel(traced_metric) called inside a jitted scan
#     trigger one compilation, or one-per-step? Count compilations of the inner fn. ---

compile_counter = {"n":0}

def logdensity_fn(z):
    # toy curved logdensity
    return -0.5*jnp.sum(z**2) - 0.1*jnp.sum(jnp.sin(z))

def build_kernel(inverse_mass_matrix):
    # mimic esh_..._smart: cholesky of a TRACED metric, used inside step
    chol = jnp.linalg.cholesky(inverse_mass_matrix)
    vg = jax.value_and_grad(logdensity_fn)
    def kernel(z):
        compile_counter["n"] += 1   # increments at TRACE time, once per compilation
        l, g = vg(z)
        g2 = chol.T @ g
        return l, jnp.sum(g2)
    return kernel

def step(carry, x):
    z, metric = carry
    # rebuild closure each step with TRACED metric (mirrors line 256 kernel(params.inverse_mass_matrix))
    k = build_kernel(metric)
    l, gs = k(z)
    znew = z + 1e-3*jnp.sin(z) + 1e-3*gs
    # mimic a "window swap": occasionally replace metric with a new traced matrix
    metric_new = jnp.where(x[0] == 3, metric*2.0, metric)
    return (znew, metric_new), l

@jax.jit
def run(z0, metric0, modes):
    return jax.lax.scan(step, (z0, metric0), modes)

dim=5
z0=jnp.ones(dim)
metric0=jnp.eye(dim)
modes=jnp.stack([jnp.array([3,0,0,3,0,0,0,0],dtype=jnp.int32)],axis=-1)
carry,ls = run(z0, metric0, modes)
jax.block_until_ready(ls)
print("TEST1 compilations of inner kernel build:", compile_counter["n"], "(expect 1 => metric-as-arg, single XLA program)")

# --- Structural test 2: standalone jitted value_and_grad vs an in-scan recompute,
#     on CPU float32 -> agree to <= a few ulp (seam offset is a GPU phenomenon) ---
standalone = jax.jit(jax.value_and_grad(logdensity_fn))
@jax.jit
def in_kernel(z):
    # different jit context / different surrounding ops than standalone
    l,g = jax.value_and_grad(logdensity_fn)(z)
    return l+0.0*jnp.sum(g), g

import numpy as np
rng=np.random.default_rng(0)
maxdiff=0.0
for _ in range(200):
    z=jnp.asarray(rng.standard_normal(dim).astype(np.float32))*30.0
    l1,_=standalone(z); l2,_=in_kernel(z)
    ulp=np.spacing(np.float32(np.abs(np.asarray(l1))))
    d=abs(float(l1)-float(l2))
    maxdiff=max(maxdiff, d/max(ulp,1e-30))
print(f"TEST2 max |logp_standalone - logp_inkernel| in ulps over 200 pts: {maxdiff:.2f} (CPU)")
