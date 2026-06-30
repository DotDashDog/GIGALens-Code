import numpy as np, sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
import tensorflow_probability.substrates.jax as tfp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
names=list(np.load(WD+"names.npy"))
U=np.load(WD+"run_under.npz"); O=np.load(WD+"run_over.npz")
nrU=int(U['nr']); nrO=int(O['nr'])
lp=lambda z: prob_model.log_prob(z)[0]
jlp=jax.jit(jax.vmap(lp))
def batched_lp(Z,bs=64):
    out=[]
    for k in range(0,len(Z),bs): out.append(np.asarray(jlp(jnp.asarray(Z[k:k+bs]))))
    return np.concatenate(out)

# --- H2 mechanism: is high xi associated with low lp (on the steep walls)? ---
xi=U['xi'][:,-nrU:].ravel()
pos=U['position'][:,-nrU:,:].reshape(-1,32)
# subsample for lp eval
rng=np.random.default_rng(0); idx=rng.choice(len(xi),6000,replace=False)
lpv=batched_lp(pos[idx]); xiv=xi[idx]
fin=np.isfinite(lpv)
lpv=lpv[fin]; xiv=xiv[fin]
lxi=np.log10(np.clip(xiv,1e-12,None))
c=np.corrcoef(lxi, lpv)[0,1]
print(f"corr(log10 xi, log_prob) = {c:+.3f}  (negative => blowups at LOW lp / on walls)")
hi=xiv>1; 
print(f"lp at xi>1 steps: median={np.median(lpv[hi]):.1f} (n={hi.sum()})")
print(f"lp at xi<=1 steps: median={np.median(lpv[~hi]):.1f} (n={(~hi).sum()})")
print(f"lp(MAP)~ -293643; best chains sit near {np.median(lpv):.1f}")
print(f"lp spread results phase: 5/50/95 pct = {np.percentile(lpv,[5,50,95])}")

# --- consolidated plots ---
fig,ax=plt.subplots(1,3,figsize=(18,5))
# running rhat
def rhat(sz):
    t=jnp.transpose(jnp.asarray(sz),(1,0,2)); return float(np.nanmax(np.asarray(tfp.mcmc.potential_scale_reduction(t))))
for tag,run,nr in [("under",U,nrU),("over",O,nrO)]:
    res=run['position'][:,-nr:,:]; N=res.shape[1]
    sched=np.unique(np.geomspace(50,N,15).astype(int))
    rr=[rhat(res[:,:n,:]) for n in sched]
    ax[0].loglog(sched,rr,'o-',label=tag)
ax[0].axhline(1.1,color='k',ls=':'); ax[0].set_xlabel("# result samples"); ax[0].set_ylabel("max R-hat (tfp)")
ax[0].set_title("H1: running max R-hat vs samples"); ax[0].legend()
# xi hist
ax[1].hist(np.log10(np.clip(U['xi'][:,-nrU:].ravel(),1e-12,None)),bins=80,alpha=0.5,label="under",density=True)
ax[1].hist(np.log10(np.clip(O['xi'][:,-nrO:].ravel(),1e-12,None)),bins=80,alpha=0.5,label="over",density=True)
ax[1].axvline(np.log10(5e-4),color='k',ls='--',label="target 5e-4"); ax[1].axvline(0,color='r',ls=':',label="xi=1")
ax[1].set_xlabel("log10 xi (results phase)"); ax[1].set_title("H2: energy-error distribution"); ax[1].legend()
# xi vs lp scatter
ax[2].scatter(lpv,lxi,s=3,alpha=0.2)
ax[2].axhline(0,color='r',ls=':'); ax[2].set_xlabel("log_prob"); ax[2].set_ylabel("log10 xi")
ax[2].set_title(f"H2: xi vs lp (corr={c:+.2f})"); ax[2].set_xlim(np.percentile(lpv,1),np.percentile(lpv,99.5))
plt.tight_layout(); plt.savefig(WD+"h1h2_summary.png",dpi=90); print("saved h1h2_summary.png")
