import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
d=np.load('/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/mclmc.stale-20260626T115536/arrays.npz')
x=d['samples_z']  # (chains, draws, params)
C,N,P=x.shape
# hand-rolled (my earlier formula)
m=x.mean(1); v=x.var(1,ddof=1)
B=N*m.var(0,ddof=1); W=v.mean(0)
varhat=(N-1)/N*W+B/N
rhat_hand=np.sqrt(varhat/W)
# tfp
sz_t=jnp.transpose(jnp.asarray(x),(1,0,2))
rhat_tfp=np.asarray(tfp.mcmc.potential_scale_reduction(sz_t))
print("max R-hat hand:", np.nanmax(rhat_hand), " tfp:", np.nanmax(rhat_tfp))
worst=np.argsort(rhat_tfp)[::-1][:6]
print("param  hand    tfp     within-var W   betw-var(means)")
for p in worst:
    print(f"{p:3d}  {rhat_hand[p]:7.2f} {rhat_tfp[p]:7.2f}   {W[p]:.3e}   {m[:,p].var(ddof=1):.3e}")
# Is it a within-chain near-zero variance (frozen) issue?
print("\nmin within-chain std per param (any frozen chains?):")
wc_std=x.std(2).min() if False else None
sd=x.std(1)  # (chains,params)
print("min per-chain std over all:", sd.min(), " at param", np.unravel_index(np.argmin(sd),sd.shape))
print("for worst param", worst[0], "per-chain std:", np.round(sd[:,worst[0]],4))
print("for worst param", worst[0], "per-chain mean:", np.round(m[:,worst[0]],3))
