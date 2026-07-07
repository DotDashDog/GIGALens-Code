import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
names=list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; flat=sz.reshape(-1,32); mu=flat.mean(0)
Sig=np.cov(flat.T); lam,V=np.linalg.eigh(Sig); o=np.argsort(lam)[::-1]; lam=lam[o]; V=V[:,o]
vstar=np.load(WD+"CB.npz")['vstar']
s_samp=(flat-mu)@vstar
kw=int(np.argmin(np.abs((V*0+1).sum(0)*0)+0))  # unused
# complement basis = eigvecs orthogonal to vstar (use all V cols, drop the one most parallel to vstar)
par=np.abs(V.T@vstar); kdrop=int(np.argmax(par)); cidx=[j for j in range(32) if j!=kdrop]
Vc=V[:,cidx]; sc=np.sqrt(np.maximum(lam[cidx],1e-12))
lp=lambda z: prob_model.log_prob(z)[0]
_vg=jax.jit(jax.vmap(jax.value_and_grad(lp)))
def vg(Z,bs=12):
    vs=[];gs=[]
    for i in range(0,len(Z),bs):
        v,g=_vg(jnp.asarray(Z[i:i+bs])); vs.append(np.asarray(v)); gs.append(np.asarray(g))
    return np.concatenate(vs), np.concatenate(gs)
sgrid=np.linspace(s_samp.min()*1.05, s_samp.max()*1.05, 41); G=len(sgrid)
# 1-D profile: maximize logp over complement at each s, save the ridge path
w=np.zeros((G,len(cidx))); m=np.zeros_like(w); vv=np.zeros_like(w); b1,b2,lr=0.9,0.999,0.05
def path_z(w): return mu[None]+sgrid[:,None]*vstar[None]+(w*sc[None])@Vc.T
for t in range(500):
    z=path_z(w); val,g=vg(z)
    gw=g@Vc*sc[None]
    m=b1*m+(1-b1)*gw; vv=b2*vv+(1-b2)*gw**2
    w=w+lr*(m/(1-b1**(t+1)))/(np.sqrt(vv/(1-b2**(t+1)))+1e-8)
zpath=path_z(w)                                   # ridge path (G,32)
dev=zpath-(mu[None]+sgrid[:,None]*vstar[None])     # deviation from straight line
# bend direction = top PCA of deviation
Ud,Sd,_=np.linalg.svd(dev-dev.mean(0),full_matrices=False)
u=_= (np.linalg.svd(dev-dev.mean(0),full_matrices=False)[2][0]); u=u-(u@vstar)*vstar; u/=np.linalg.norm(u)
explained=Sd[0]**2/np.sum(Sd**2)
p_u=(zpath-mu)@u                                  # ridge u-coord vs s
width=np.std(((flat-mu)@u)-np.polyval(np.polyfit((flat-mu)@vstar,(flat-mu)@u,1),(flat-mu)@vstar))
bend=p_u.max()-p_u.min()
print(f"profile climbed: logp range on ridge = {val.min():.1f}..{val.max():.1f}")
print(f"bend dir u top params: "+", ".join(f'{names[i].split("/")[-1]}({u[i]:+.2f})' for i in np.argsort(np.abs(u))[::-1][:3]))
print(f"bend-direction explains {explained:.2f} of ridge-path deviation")
print(f"ridge u-bend (max-min of path) = {bend:.3f};  within-ridge width(u) = {width:.3f};  bend/width = {bend/width:.1f}")
# ridge-conditioned 2-D contour: off-plane dims follow the path
bg=np.linspace(((flat-mu)@u).min()*1.1, ((flat-mu)@u).max()*1.1, 41)
def blp(Z,bs=25):
    out=[]
    for i in range(0,len(Z),bs): out.append(np.asarray(jax.vmap(lp)(jnp.asarray(Z[i:i+bs]))))
    return np.concatenate(out)
A,Bp=np.meshgrid(sgrid,bg)
# for each (a,b): take ridge point at s=a, then set its u-coord to b
import numpy as _np
zA=path_z(w)                                       # (G,32), indexed by sgrid
def zgrid():
    Z=np.empty((Bp.size,32))
    ai=np.searchsorted(sgrid,A.ravel()); ai=np.clip(ai,0,G-1)
    base=zA[ai]                                     # ridge point at nearest s
    cur_u=(base-mu)@u
    Z=base+(Bp.ravel()-cur_u)[:,None]*u[None]
    # also set v* coord exactly to a
    cur_v=(Z-mu)@vstar
    Z=Z+(A.ravel()-cur_v)[:,None]*vstar[None]
    return Z
L=blp(zgrid()).reshape(A.shape); L=L-np.nanmax(L)
fig,ax=plt.subplots(1,2,figsize=(15,6))
for k,(title,showsamp) in enumerate([("ridge-conditioned logp contour",False),("+ MCMC samples",True)]):
    cf=ax[k].contourf(A,Bp,np.clip(L,-30,0),levels=np.linspace(-30,0,25),cmap="viridis")
    ax[k].plot(sgrid,p_u,'r-',lw=2.5,label="ridge crest (profile)")
    if showsamp: ax[k].scatter(s_samp,(flat-mu)@u,s=2,c='w',alpha=0.10)
    ax[k].set_xlabel("slow direction (eigdir 9): e1/shear/e2"); ax[k].set_ylabel("bend direction u: NFW center/scale")
    ax[k].set_title(title); ax[k].legend(loc='upper center')
plt.colorbar(cf,ax=ax[1],label="logp - max")
plt.tight_layout(); plt.savefig(WD+"banana2.png",dpi=95); print("saved banana2.png")
