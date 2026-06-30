import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
names=list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; flat=sz.reshape(-1,32); mu=flat.mean(0)
vstar=np.load(WD+"CB.npz")['vstar']                       # slow direction (eigdir 9)
s=(flat-mu)@vstar                                         # slow coordinate per sample
# per-coordinate quadratic fit z ~ a + b s + c s^2  -> c = curvature (bend) vector
Xd=np.vstack([np.ones_like(s), s, s**2]).T
coef,_,_,_=np.linalg.lstsq(Xd, flat-mu, rcond=None)       # (3,32)
blin, cvec = coef[1], coef[2]
u = cvec - (cvec@vstar)*vstar; u/=np.linalg.norm(u)       # bend direction, orth to vstar
# curvature magnitude: ridge u-offset from quadratic term vs within-ridge scatter in u
yU=(flat-mu)@u
yU_detrend = yU - (Xd@(np.linalg.lstsq(Xd,yU,rcond=None)[0]))   # residual scatter in u
bend = (cvec@u)*(s.max()**2)                              # u-shift of ridge at s_max from quadratic term
print(f"slow dir (eigdir9) top params: "+", ".join(f"{names[i].split('/')[-1]}({vstar[i]:+.2f})" for i in np.argsort(np.abs(vstar))[::-1][:3]))
print(f"bend dir (u) top params:        "+", ".join(f"{names[i].split('/')[-1]}({u[i]:+.2f})" for i in np.argsort(np.abs(u))[::-1][:3]))
print(f"ridge quadratic u-bend over s-range = {bend:.4f};  within-ridge scatter(u) std = {yU_detrend.std():.4f};  bend/scatter = {abs(bend)/yU_detrend.std():.1f}")

# 2-D logp contour in (vstar,u) plane through mean (other 30 dims at mean)
lp=lambda z: prob_model.log_prob(z)[0]
def blp(Z,bs=25):
    o=[]
    for i in range(0,len(Z),bs): o.append(np.asarray(jax.vmap(lp)(jnp.asarray(Z[i:i+bs]))))
    return np.concatenate(o)
ag=np.linspace(s.min()*1.05, s.max()*1.05, 45)
bg=np.linspace(yU.min()*1.15, yU.max()*1.15, 41)
A,B=np.meshgrid(ag,bg)
Z=mu[None,:]+A.ravel()[:,None]*vstar[None,:]+B.ravel()[:,None]*u[None,:]
L=blp(Z).reshape(A.shape); L=L-np.nanmax(L)
ridge_b=(blin@u)*ag+(cvec@u)*ag**2                        # fitted ridge in (vstar,u)
fig,ax=plt.subplots(1,2,figsize=(15,6))
cf=ax[0].contourf(A,B,np.clip(L,-60,0),levels=np.linspace(-60,0,25),cmap="viridis")
ax[0].plot(ag,ridge_b,'r-',lw=2,label="fitted ridge (quadratic)")
plt.colorbar(cf,ax=ax[0],label="logp - max")
ax[0].set_xlabel("slow direction (eigdir 9)"); ax[0].set_ylabel("bend direction u")
ax[0].set_title("logp contour in (slow, bend) plane  [other 30 dims = mean]"); ax[0].legend()
# overlay samples on a second copy
cf2=ax[1].contour(A,B,np.clip(L,-60,0),levels=np.linspace(-60,0,13),cmap="viridis",alpha=0.7)
ax[1].scatter(s, yU, s=2, c='k', alpha=0.12)
ax[1].plot(ag,ridge_b,'r-',lw=2,label="ridge")
ax[1].set_xlabel("slow direction (eigdir 9)"); ax[1].set_ylabel("bend direction u")
ax[1].set_title("same contour + MCMC samples (black)"); ax[1].legend()
plt.tight_layout(); plt.savefig(WD+"banana.png",dpi=95); print("saved banana.png")
