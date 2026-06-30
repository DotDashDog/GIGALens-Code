import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
import arviz as az
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
names=list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; C,N,P=sz.shape
flat=sz.reshape(-1,P); mu=flat.mean(0)
Sig=np.cov(flat.T); lam,V=np.linalg.eigh(Sig)            # ascending
order=np.argsort(lam)[::-1]; lam=lam[order]; V=V[:,order] # descending eigenvalues
# ---- Test C: ESS / rank-Rhat per eigen-direction ----
proj=(sz-mu)@V                                           # (8,2000,32) in eigenbasis
def rE(a): return float(az.ess(a,method="bulk"))
def rR(a): return max(float(az.rhat(a,method="rank")),float(az.rhat(a,method="folded")))
essK=np.array([rE(proj[:,:,k]) for k in range(P)])
rhK =np.array([rR(proj[:,:,k]) for k in range(P)])
kw=int(np.argmin(essK))                                  # slowest-mixing eigendir
print("=== TEST C: slowest eigen-directions (by bulk-ESS) ===")
for k in np.argsort(essK)[:5]:
    load=V[:,k]; top=np.argsort(np.abs(load))[::-1][:3]
    desc=", ".join(f"{names[t].split('/')[-1]}({load[t]:+.2f})" for t in top)
    print(f"  eigdir {k:2d}: bulkESS={essK[k]:5.0f} rankRhat={rhK[k]:.2f} eigval={lam[k]:.2e} | top: {desc}")
print(f"  -> ridge-tracing eigendir {kw} (ESS={essK[kw]:.0f}, Rhat={rhK[kw]:.2f})")

vstar=V[:,kw]
s_samp=(flat-mu)@vstar                                   # projection of samples onto slow dir
# complement basis (other 31 eigvecs) + their std scale, for well-conditioned profiling
cidx=[j for j in range(P) if j!=kw]
Vc=V[:,cidx]; sc=np.sqrt(np.maximum(lam[cidx],1e-12))    # (32,31),(31,)
lp=lambda z: prob_model.log_prob(z)[0]
sgrid=np.linspace(s_samp.min()*1.1, s_samp.max()*1.1, 31)
z0=mu.copy()
# conditional slice (other dims fixed at mean)
def blp(Z,bs=31):
    o=[]; 
    for i in range(0,len(Z),bs): o.append(np.asarray(jax.vmap(lp)(jnp.asarray(Z[i:i+bs]))))
    return np.concatenate(o)
slice_z=np.array([z0+s*vstar for s in sgrid]); slice_lp=blp(slice_z)
# profile: maximize logp over complement w (O(1) scale) at each s, batched Adam ascent
def z_of(s,w): return z0[None]+s[:,None]*vstar[None]+ (w*sc[None])@Vc.T   # (G,32)
G=len(sgrid); w=np.zeros((G,31))
val_and_grad=jax.jit(jax.vmap(jax.value_and_grad(lambda zz: lp(zz))))
sg=jnp.asarray(sgrid)
def lp_grad_w(w):
    z=z0[None]+sgrid[:,None]*vstar[None]+(w*sc[None])@Vc.T
    v,g=val_and_grad(jnp.asarray(z))                      # g wrt z (G,32)
    gw=(np.asarray(g))@Vc*sc[None]                        # chain rule to w (G,31)
    return np.asarray(v), gw
m=np.zeros_like(w); vv=np.zeros_like(w); b1,b2,lr,eps=0.9,0.999,0.05,1e-8
prof_hist=[]
for t in range(400):
    val,gw=lp_grad_w(w)
    m=b1*m+(1-b1)*gw; vv=b2*vv+(1-b2)*gw**2
    mh=m/(1-b1**(t+1)); vh=vv/(1-b2**(t+1))
    w=w+lr*mh/(np.sqrt(vh)+eps)                           # ASCENT on logp
    if t%50==0: prof_hist.append(val.copy())
prof_lp,_=lp_grad_w(w)
print("\n=== TEST B: ridge-trace along slow eigendir ===")
print(f"  conditional-slice logp range: {slice_lp.min():.1f}..{slice_lp.max():.1f}")
print(f"  PROFILE logp range:           {prof_lp.min():.1f}..{prof_lp.max():.1f}")
# barrier metrics on the PROFILE (the decisive curve): dip between the two ends
imax=np.argmax(prof_lp)
left=prof_lp[:imax+1].min() if imax>0 else prof_lp[0]
right=prof_lp[imax:].min() if imax<G-1 else prof_lp[-1]
# count local maxima in profile (smoothed)
from numpy import convolve
ps=convolve(prof_lp,np.ones(3)/3,mode='same')
nmax=sum(1 for i in range(1,G-1) if ps[i]>ps[i-1] and ps[i]>ps[i+1])
print(f"  profile #local maxima (smoothed): {nmax}")
print(f"  profile barrier depth (max - min over interior): {prof_lp.max()-prof_lp.min():.1f}")
print(f"  optimizer climbed: median(profile - slice) = {np.median(prof_lp-slice_lp):.1f} (should be >0)")

# ---- plots ----
fig,ax=plt.subplots(1,3,figsize=(18,5))
ax[0].plot(np.argsort(essK)[::-1]*0+np.arange(P), np.sort(essK)[::-1],'o-')  # placeholder
ax[0].clear(); ax[0].bar(range(P), essK[np.argsort(essK)]); ax[0].set_yscale('log')
ax[0].set_xlabel("eigen-direction (sorted)"); ax[0].set_ylabel("bulk-ESS"); ax[0].set_title("C: ESS per eigen-direction")
ax[0].axhline(C, color='r', ls='--', label=f"n_chains={C}"); ax[0].legend()
# 2D scatter on the two slowest eigendirs, colored by chain
ks=np.argsort(essK)[:2]
for c in range(C):
    pc=(sz[c]-mu)@V
    ax[1].scatter(pc[:,ks[0]], pc[:,ks[1]], s=4, alpha=0.4, label=f"ch{c}")
ax[1].set_xlabel(f"eigdir {ks[0]} (slowest)"); ax[1].set_ylabel(f"eigdir {ks[1]}")
ax[1].set_title("C: samples in 2 slowest eigendirs (color=chain)"); ax[1].legend(fontsize=6,ncol=2)
# ridge trace: slice vs profile
ax[2].plot(sgrid, slice_lp-prof_lp.max(), 'o-', color='tab:gray', label="conditional slice (others=mean)")
ax[2].plot(sgrid, prof_lp-prof_lp.max(), 's-', color='tab:red', label="PROFILE (others re-optimized)")
ax[2].axhline(0,color='k',ls=':',lw=0.7)
ax[2].set_xlabel(f"position along slow eigendir {kw}"); ax[2].set_ylabel("logp - max")
ax[2].set_ylim(-200,5)
ax[2].set_title(f"B: ridge-trace eigendir {kw} (barrier in PROFILE => multimodal)"); ax[2].legend()
plt.tight_layout(); plt.savefig(WD+"CB_ridge.png",dpi=95); print("saved CB_ridge.png")
np.savez(WD+"CB.npz", sgrid=sgrid, slice_lp=slice_lp, prof_lp=prof_lp, essK=essK, rhK=rhK, kw=kw, vstar=vstar)
