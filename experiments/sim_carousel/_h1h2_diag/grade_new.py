import numpy as np, sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
import tensorflow_probability.substrates.jax as tfp
names=list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
d=np.load(RUN+"mclmc/diagnostics.npz"); sz=d['samples_z']  # (8,2000,32)
xi=d['xi']; nb=10000; nr=2000
zb=np.asarray(np.load(RUN+"map/arrays.npz")['z_best']).reshape(-1)
def rhat(s):
    t=jnp.transpose(jnp.asarray(s),(1,0,2)); return np.asarray(tfp.mcmc.potential_scale_reduction(t))
rh=rhat(sz); print(f"max R-hat (tfp) = {np.nanmax(rh):.2f}   (old run was ~71)")
print("worst 6 params:")
for i in np.argsort(rh)[::-1][:6]: print(f"  {i:2d} {names[i]:28s} R-hat={rh[i]:.2f}")
# drift
C,N,P=sz.shape; fh=sz[:,:N//4].mean(1); lh=sz[:,-N//4:].mean(1); s=sz.std(1)
dr=np.abs((lh-fh)/s); print(f"drift |Δmean|/std: median={np.median(dr):.2f} max={np.max(dr):.2f}  (old ~1.8/2.6)")
# lp gap
lp=jax.jit(jax.vmap(lambda z: prob_model.log_prob(z)[0]))
flat=sz.reshape(-1,32); idx=np.random.default_rng(0).choice(len(flat),3000,replace=False)
lpv=np.asarray(lp(jnp.asarray(flat[idx]))); lpb=float(prob_model.log_prob(jnp.asarray(zb))[0])
print(f"new MAP lp={lpb:.1f}; chains lp 5/50/95={np.percentile(lpv,[5,50,95])}")
print(f"lp gap (chain median - MAP) = {np.median(lpv)-lpb:+.1f}   (old gap was +1600)")
# xi results phase
rxi=xi[:,-nr:]
print(f"results-phase xi: median={np.median(rxi):.2e} frac>1={np.mean(rxi>1):.3f} max={rxi.max():.2e}  (old frac>1=0.042)")
