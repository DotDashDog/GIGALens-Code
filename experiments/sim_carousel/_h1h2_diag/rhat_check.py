import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import tensorflow_probability.substrates.jax as tfp
d=np.load('/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/mclmc.stale-20260626T115536/arrays.npz')
sz=d['samples_z']  # (chains, steps, params)
print("samples_z shape", sz.shape)
import jax.numpy as jnp
sz_t=jnp.transpose(jnp.asarray(sz),(1,0,2))  # (steps, chains, params)
for split in (False, True):
    r=np.asarray(tfp.mcmc.potential_scale_reduction(sz_t, split_chains=split))
    print(f"split_chains={split}: max R-hat={np.nanmax(r):.3f}, max(R-1)={np.nanmax(r)-1:.3f}")
# default
r=np.asarray(tfp.mcmc.potential_scale_reduction(sz_t))
print(f"DEFAULT: max R-hat={np.nanmax(r):.3f}")
