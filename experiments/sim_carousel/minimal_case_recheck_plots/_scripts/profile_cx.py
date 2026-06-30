import os, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
import optax
import sys; sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts")
from build_model import build
def pr(*a): print(*a,flush=True)
EXP="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel"; OUT=f"{EXP}/minimal_case_recheck_plots"
pm=build(); pr("devices",jax.devices())
zmap=jnp.asarray(np.load(f"{EXP}/messy_tests/minimal_case/map/arrays.npz")['z_best'].reshape(-1))
CX=9; free_idx=[i for i in range(14) if i!=CX]
fi=jnp.asarray(free_idx)
def assemble(free, cx):
    z=jnp.zeros(14).at[fi].set(free).at[CX].set(cx); return z
@jax.jit
def neglogp(free, cx):
    z=assemble(free,cx); lp,_=pm.log_prob(z[None,:]); return -lp[0]
vg=jax.jit(jax.value_and_grad(neglogp))
def optimize(free0, cx, steps=800, lr=3e-3):
    opt=optax.adam(lr); st=opt.init(free0); free=free0
    for _ in range(steps):
        v,g=vg(free,cx); upd,st=opt.update(g,st); free=optax.apply_updates(free,upd)
    v,_=vg(free,cx); return free,-float(v)
free_map=zmap[fi]
# cold-start profile: each cx independently from MAP free params
cxs=np.linspace(4.27,4.53,27)
pr("cold-start profile...")
t=time.time(); cold=[]
for cx in cxs:
    _,lp=optimize(free_map, float(cx)); cold.append(lp)
cold=np.array(cold); pr("cold done %.1fs"%(time.time()-t))
# continuation profile: sweep up from MAP, then down from MAP, warm-start
pr("continuation profile...")
cx_map=float(zmap[CX])
up_cx=cxs[cxs>=cx_map]; dn_cx=cxs[cxs<cx_map][::-1]
cont={}; free=free_map
for cx in up_cx:
    free,lp=optimize(free,float(cx),steps=600); cont[float(cx)]=lp
free=free_map
for cx in dn_cx:
    free,lp=optimize(free,float(cx),steps=600); cont[float(cx)]=lp
contarr=np.array([cont[float(c)] for c in cxs])
# barrier analysis on cold profile (true profiled curve)
def barrier(curve,xv):
    # find interior local maxima
    mx=[i for i in range(1,len(curve)-1) if curve[i]>=curve[i-1] and curve[i]>=curve[i+1]]
    pr("  profiled local maxima (cx,logp):",[(round(xv[i],3),round(curve[i],2)) for i in mx])
    if len(mx)>=2:
        i1,i2=mx[0],mx[-1]; saddle=curve[i1:i2+1].min()
        b=min(curve[i1],curve[i2])-saddle; return b,(xv[i1],xv[i2]),saddle
    return -1,None,None
pr("COLD profile barrier:"); bc,pk,sd=barrier(cold,cxs)
pr("  barrier=%.3f logp (peaks at %s, saddle %.2f)"%(bc,pk,sd) if pk else "  single peak (no barrier)")
pr("CONT profile barrier:"); bk,pk2,sd2=barrier(contarr,cxs)
pr("  barrier=%.3f logp (peaks %s)"%(bk,pk2) if pk2 else "  single peak (no barrier)")
pr("cold max logp %.2f at cx=%.3f ; MAP logp ref -119514.93"%(cold.max(),cxs[cold.argmax()]))
np.savez(f"{OUT}/R_profile_cx.npz",cxs=cxs,cold=cold,cont=contarr,cx_map=cx_map,
         barrier_cold=bc,barrier_cont=bk)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(cxs,cold-cold.max(),'o-',label='cold-start (each cx from MAP)')
plt.plot(cxs,contarr-contarr.max(),'s--',label='continuation (warm)',alpha=.7)
plt.axvline(4.339,color='b',ls=':',label='MAP/lower mode cx=4.34')
plt.axvline(4.449,color='g',ls=':',label='upper mode cx=4.45')
plt.axhline(-3,color='r',ls='--',alpha=.5,label='-3 (real-mode threshold)')
plt.xlabel('src4 center_x (profiled)'); plt.ylabel('profiled logp - max')
plt.title('PROFILE likelihood over src4 cx (other 13 re-optimized)\nDECISIVE barrier test')
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/R_profile_cx.png",dpi=120)
pr("saved R_profile_cx.npz/png ; ALLDONE")
