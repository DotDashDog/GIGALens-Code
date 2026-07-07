import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
names=list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; flat=sz.reshape(-1,32); mu=flat.mean(0)
vstar=np.load(WD+"CB.npz")['vstar']; s=(flat-mu)@vstar
# ridge from per-coordinate quadratic fit (cheap; samples-only)
Xd=np.vstack([np.ones_like(s),s,s**2]).T
coef,_,_,_=np.linalg.lstsq(Xd,flat-mu,rcond=None); blin,cvec=coef[1],coef[2]
u=cvec-(cvec@vstar)*vstar; u/=np.linalg.norm(u)
def zfit(a): return mu+blin*a+cvec*a**2          # ridge point at slow-coord a
lp=lambda z: prob_model.log_prob(z)[0]
def blp(Z,bs=6):
    out=[]
    for i in range(0,len(Z),bs): out.append(np.asarray(jax.vmap(lp)(jnp.asarray(Z[i:i+bs]))))
    return np.concatenate(out)
ag=np.linspace(s.min()*1.05,s.max()*1.05,33); yU=(flat-mu)@u
bg=np.linspace(yU.min()*1.12,yU.max()*1.12,29)
A,Bp=np.meshgrid(ag,bg)
# condition off-plane dims on the quadratic ridge at s=a; set in-plane coords to (a,b)
base=np.array([zfit(a) for a in A.ravel()])      # (Npts,32)
z=base.copy()
z=z+(A.ravel()-(base-mu)@vstar)[:,None]*vstar[None]
z=z+(Bp.ravel()-(base-mu)@u)[:,None]*u[None]
L=blp(z).reshape(A.shape); L=L-np.nanmax(L)
p_u=(np.array([zfit(a) for a in ag])-mu)@u        # ridge crest in (slow,u)
np.savez(WD+"banana3.npz",A=A,Bp=Bp,L=L,ag=ag,p_u=p_u,s=s,yU=yU,u=u,vstar=vstar)
print(f"slow top: "+", ".join(f'{names[i].split("/")[-1]}({vstar[i]:+.2f})' for i in np.argsort(np.abs(vstar))[::-1][:3]))
print(f"bend(u) top: "+", ".join(f'{names[i].split("/")[-1]}({u[i]:+.2f})' for i in np.argsort(np.abs(u))[::-1][:3]))
print(f"ridge crest u: min={p_u.min():.2f} max={p_u.max():.2f} (bend={p_u.max()-p_u.min():.2f})")
fig,ax=plt.subplots(1,2,figsize=(15,6))
for k,(t,ss) in enumerate([("ridge-conditioned logp contour",False),("+ MCMC samples",True)]):
    cf=ax[k].contourf(A,Bp,np.clip(L,-25,0),levels=np.linspace(-25,0,26),cmap="viridis")
    ax[k].plot(ag,p_u,'r-',lw=2.5,label="ridge crest (quadratic)")
    if ss: ax[k].scatter(s,yU,s=2,c='w',alpha=0.12)
    ax[k].set_xlabel("slow direction (eigdir9): e1/shear/e2"); ax[k].set_ylabel("bend direction u: NFW center/scale")
    ax[k].set_title(t); ax[k].legend(loc="upper center")
plt.colorbar(cf,ax=ax[1],label="logp - max"); plt.tight_layout()
plt.savefig(WD+"banana3.png",dpi=95); print("saved banana3.png")
