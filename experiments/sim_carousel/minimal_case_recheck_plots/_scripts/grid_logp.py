import os, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
import sys; sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts")
from build_model import build
def pr(*a): print(*a, flush=True)
EXP="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel"
OUT=f"{EXP}/minimal_case_recheck_plots"
pr("devices:", jax.devices())
pm=build()
zmap=np.asarray(np.load(f"{EXP}/messy_tests/minimal_case/map/arrays.npz")['z_best']).reshape(-1)
P=zmap.shape[0]
def logp_batch(Z, chunk=16):
    Z=np.asarray(Z); out=np.empty(len(Z)); rc=np.empty(len(Z))
    for i in range(0,len(Z),chunk):
        zb=jnp.asarray(Z[i:i+chunk]); lp,r=pm.log_prob(zb)
        out[i:i+chunk]=np.asarray(lp).ravel(); rc[i:i+chunk]=np.asarray(r).ravel()
    return out,rc
t=time.time(); lp0,rc0=logp_batch(zmap[None,:])
pr("MAP logp %.4f red_chi2 %.5f (warmup %.1fs)"%(lp0[0],rc0[0],time.time()-t))
sz=np.load(f"{EXP}/messy_tests/minimal_case/mclmc/arrays.npz")['samples_z'].reshape(-1,P)
thr=4.40
cL=sz[sz[:,9]<thr].mean(0); cU=sz[sz[:,9]>=thr].mean(0)
pr("centroid L col9=%.4f U col9=%.4f |cU-cL|=%.4f"%(cL[9],cU[9],np.linalg.norm(cU-cL)))
# DECISIVE 1-D path
ts=np.linspace(-0.4,1.4,73)
Zpath=np.array([(1-t)*cL+t*cU for t in ts])
t=time.time(); lp_path,rc_path=logp_batch(Zpath); pr("path done %.1fs"%(time.time()-t))
lpL,_=logp_batch(cL[None,:]); lpU,_=logp_batch(cU[None,:])
seg=(ts>=0)&(ts<=1); lp_min_seg=lp_path[seg].min(); t_min=ts[seg][np.argmin(lp_path[seg])]
barrier=min(lpL[0],lpU[0])-lp_min_seg
pr("logp cL=%.3f cU=%.3f MAP=%.3f"%(lpL[0],lpU[0],lp0[0]))
pr("path min on [0,1]=%.3f at t=%.3f ; BARRIER=%.3f logp"%(lp_min_seg,t_min,barrier))
# SAVE decisive result immediately
np.savez(f"{OUT}/R_logp_path.npz",ts=ts,lp_path=lp_path,rc_path=rc_path,cL=cL,cU=cU,
    zmap=zmap,lpL=lpL[0],lpU=lpU[0],lp_map=lp0[0],barrier=barrier,t_min=t_min,thr=thr)
pr("saved R_logp_path.npz")
# 2-D conditional grids (25x25)
def grid2d(ci,cj,vi,vj):
    G=np.meshgrid(vi,vj,indexing='ij'); Z=np.tile(zmap,(vi.size*vj.size,1))
    Z[:,ci]=G[0].ravel(); Z[:,cj]=G[1].ravel(); lp,rc=logp_batch(Z)
    return lp.reshape(vi.size,vj.size),rc.reshape(vi.size,vj.size)
v9=np.linspace(4.20,4.60,25); v10=np.linspace(3.70,4.10,25); v2=np.linspace(4.92,5.16,25)
t=time.time(); g_910,rc_910=grid2d(9,10,v9,v10); pr("grid910 done %.1fs max %.3f"%(time.time()-t,g_910.max()))
t=time.time(); g_92,rc_92=grid2d(9,2,v9,v2); pr("grid92 done %.1fs max %.3f"%(time.time()-t,g_92.max()))
np.savez(f"{OUT}/R_logp_geometry.npz",ts=ts,lp_path=lp_path,rc_path=rc_path,cL=cL,cU=cU,
    zmap=zmap,lpL=lpL[0],lpU=lpU[0],lp_map=lp0[0],barrier=barrier,t_min=t_min,thr=thr,
    v9=v9,v10=v10,g_910=g_910,rc_910=rc_910,v2=v2,g_92=g_92,rc_92=rc_92)
pr("saved R_logp_geometry.npz ; ALLDONE")
