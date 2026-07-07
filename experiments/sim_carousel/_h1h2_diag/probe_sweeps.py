import numpy as np, sys
sys.path.insert(0, "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
z_best = np.load(WD+"z_best.npy"); dim=z_best.shape[0]
names = list(np.load(WD+"names.npy"))
lp = lambda z: prob_model.log_prob(z)[0]
vg = jax.jit(jax.value_and_grad(lp))
def batched_vg(Z, bs=8):
    vs=[]; gs=[]
    for k in range(0,len(Z),bs):
        v,g=jax.vmap(vg)(jnp.asarray(Z[k:k+bs])); vs.append(np.asarray(v)); gs.append(np.asarray(g))
    return np.concatenate(vs), np.concatenate(gs)
targets = {"NFW_Rs":31,"NFW_alphaRs":30,"NFW_cx":29,"NFW_cy":28,"NFW_e1":27,"NFW_e2":26,
 "src9_n_sersic":0,"src9_R_sersic":5,"src9_e1":2,"src9_e2":1,"EPLlf_e1":17,"EPLlf_cx":19}
grid = np.linspace(-6,6,201)
fig,axes=plt.subplots(4,3,figsize=(16,14)); res={}
for ax,(nm,i) in zip(axes.ravel(), targets.items()):
    Z=np.tile(z_best,(grid.size,1)); Z[:,i]=z_best[i]+grid
    vs,gs=batched_vg(Z); gi=gs[:,i]; gn=np.linalg.norm(gs,axis=1)
    res[nm]=(i,vs,gi,gn)
    nfin=int(np.sum(~np.isfinite(vs)))
    # detect kinks: 2nd difference spikes in finite region
    fin=np.isfinite(vs); d2=np.abs(np.diff(vs,2))
    ax.plot(grid,vs-np.nanmax(vs),'b-',lw=0.9); ax.axvline(0,color='k',ls=':',lw=0.7)
    ax.set_ylim(-300,5); ax.set_title(f"{nm}(i{i}) NaN={nfin} max|gi|={np.nanmax(np.abs(gi)):.1e} max d2lp={np.nanmax(d2) if d2.size else 0:.1e}")
plt.tight_layout(); plt.savefig(WD+"sweeps.png",dpi=85)
print("idx name  NaNcount  lp_curvature(max|d2|)  max|grad_i|  max|grad_full|  lp@MAP_offset")
for nm,(i,vs,gi,gn) in res.items():
    d2=np.abs(np.diff(vs[np.isfinite(vs)],2))
    print(f"{i:2d} {nm:14s} NaN={int(np.sum(~np.isfinite(vs))):3d}  d2max={np.nanmax(d2) if d2.size else 0:.2e}  "
          f"max|gi|={np.nanmax(np.abs(gi)):.2e}  max|gfull|={np.nanmax(gn):.2e}")
np.savez(WD+"sweeps.npz", grid=grid, **{nm:res[nm][1] for nm in res})
print("saved sweeps.png + npz")
