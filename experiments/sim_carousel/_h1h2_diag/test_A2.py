import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
import build_model as bm
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, isokinetic_mclachlan_smart, init_multi)
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
D=np.load(RUN+"mclmc/diagnostics.npz")
flat=D['samples_z'].reshape(-1,32)
imm=jnp.asarray(D['inverse_mass_matrix'][0,-1])
Lrun=float(np.median(D['L'][:,-1])); eps_op=float(np.median(D['step_size'][:,-1]))
rng=np.random.default_rng(2)
pts=jnp.asarray(flat[rng.choice(len(flat),10,replace=False)])
keys=jax.random.split(jax.random.key(0),10)
epss=np.geomspace(0.5,1e-5,16)

def dE_curve(conv):
    pm=bm.make_prob_model(conv); lp=lambda z: pm.log_prob(z)[0]
    state=init_multi(pts, keys, lp)
    kernel=_build_kernel_shardmap(logdensity_fn=lp, inverse_mass_matrix=imm,
                                  integrator=isokinetic_mclachlan_smart)
    kfun=jax.jit(jax.vmap(lambda k,s,e: kernel(k,s,Lrun,e)[1].extras.energy_change_raw,
                          in_axes=(0,0,None)))
    out=[]
    for e in epss:
        out.append(float(np.median(np.abs(np.asarray(kfun(keys,state,jnp.float64(e)))))))
    return np.array(out)

d32=dE_curve("float32"); print("f32 done",flush=True)
d64=dE_curve("float64"); print("f64 done",flush=True)
print(f"operating eps~{eps_op:.3f} L~{Lrun:.1f}; controller |dE| target ~{np.sqrt(32*5e-4):.3f}")
print(f"{'eps':>10}{'|dE|_f32':>12}{'|dE|_f64':>12}")
for i in range(16): print(f"{epss[i]:10.2e}{d32[i]:12.3e}{d64[i]:12.3e}")
def slope(d,lo,hi):
    m=(epss>lo)&(epss<hi)&(d>0); return np.polyfit(np.log(epss[m]),np.log(d[m]),1)[0] if m.sum()>2 else np.nan
print(f"\nslope in [1e-3,1e-1]: f32={slope(d32,1e-3,1e-1):.2f} f64={slope(d64,1e-3,1e-1):.2f}  (smooth->~3)")
print(f"|dE| floor at eps=1e-5: f32={d32[-1]:.2e} f64={d64[-1]:.2e}")
plt.figure(figsize=(7,5))
plt.loglog(epss,d32,'o-',label="conv=float32 (notebook)")
plt.loglog(epss,d64,'s-',label="conv=float64 (control)")
plt.axvline(eps_op,color='k',ls=':',label=f"operating eps={eps_op:.2f}")
plt.axhline(np.sqrt(32*5e-4),color='r',ls='--',label="controller dE target")
ref=epss**3*(d64[6]/epss[6]**3); plt.loglog(epss,ref,'k:',alpha=0.4,label="slope-3 ref")
plt.xlabel("eps"); plt.ylabel("median |energy change| / step"); plt.legend()
plt.title("A2: MCLMC single-step energy error vs eps"); plt.savefig(WD+"A2_energy_scaling.png",dpi=95)
print("saved A2_energy_scaling.png")
