import numpy as np, sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
d=np.load(RUN+"mclmc/diagnostics.npz"); sz=d['samples_z']; xi=d['xi']; nr=2000
zb=np.asarray(np.load(RUN+"map/arrays.npz")['z_best']).reshape(-1)
jlp=jax.jit(jax.vmap(lambda z: prob_model.log_prob(z)[0]))
def blp(Z,bs=12):
    o=[]
    for k in range(0,len(Z),bs): o.append(np.asarray(jlp(jnp.asarray(Z[k:k+bs]))))
    return np.concatenate(o)
flat=sz.reshape(-1,32); idx=np.random.default_rng(0).choice(len(flat),1200,replace=False)
lpv=blp(flat[idx]); lpb=float(prob_model.log_prob(jnp.asarray(zb))[0])
print(f"new MAP lp={lpb:.1f}")
print(f"chains lp 5/50/95 = {np.percentile(lpv,[5,50,95])}")
print(f"lp gap (chain median - MAP) = {np.median(lpv)-lpb:+.1f}   (OLD run gap was +1600)")
rxi=xi[:,-nr:]
print(f"results-phase xi: median={np.median(rxi):.2e} frac>1={np.mean(rxi>1):.3f} max={rxi.max():.2e}  (OLD frac>1=0.042, median 5.4e-4)")
