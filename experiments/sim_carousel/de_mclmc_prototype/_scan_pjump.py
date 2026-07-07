"""Diagnostic 2: cross-mode DE mixing rate depends on p_jump (gamma=1 difference-
vector jumps land directly on the other mode, independent of the barrier), while
vanilla crossing depends only on the barrier m^2/2. Test p_jump=0.5 at m=4.5 and
m=5.0: want (i) vanilla trapped over the run, (ii) DE-from-truth tau SMALL enough
that the run is powered, (iii) DE discovery from single mode.
Prediction: higher p_jump sharply lowers DE-truth tau (more cross attempts);
vanilla unaffected; discovery improves."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from de_mclmc import make_composite

D, N_CHAINS = 10, 64
L, STEP, K = 2.0, 0.5, 20

def make_logp(m):
    W = jnp.log(jnp.array([0.7, 0.3])); MU = jnp.array([+m, -m])
    c = -0.5 * D * jnp.log(2*jnp.pi)
    def lp(z):
        z0 = z[0]; qr = jnp.sum(z[1:]**2)
        a = W[0] + c - 0.5*((z0-MU[0])**2 + qr); b = W[1] + c - 0.5*((z0-MU[1])**2 + qr)
        return jax.scipy.special.logsumexp(jnp.stack([a, b]))
    return lp

def iat(x, c=5.0):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if np.allclose(x, 0): return 1.0
    f = np.fft.fft(x, n=2*n); acf = np.fft.ifft(f*np.conjugate(f))[:n].real; acf/=acf[0]
    tau = 1.0
    for k in range(1, n):
        tau += 2*acf[k]
        if k > c*tau: break
    return max(tau, 1.0)

def vanilla_tail(comp, m, total_steps, seed):
    st = comp["init_states"](jnp.zeros((N_CHAINS, D)).at[:, 0].set(m), jax.random.key(seed))
    chunk=2500; done=0; tail=None
    while done<total_steps:
        nthis=min(chunk,total_steps-done)
        ck=jax.random.split(jax.random.fold_in(jax.random.key(seed+1),done),nthis*N_CHAINS).reshape(nthis,N_CHAINS)
        st,posv=comp["mclmc_only"](st,ck); tail=np.asarray(posv[:,:,0]); done+=nthis
    return float((tail>0).mean())

def de_frac(comp, init_pos, rounds, seed):
    st=comp["init_states"](init_pos, jax.random.key(seed))
    keys=jax.random.split(jax.random.key(seed+1),rounds); fr=np.empty(rounds); accs=[]
    for r in range(rounds):
        st,(p,ec,acc)=comp["round"](st,keys[r]); fr[r]=(np.asarray(p)[:,0]>0).mean(); accs.append(float(np.asarray(acc).mean()))
    return fr, np.mean(accs)

for m in [4.5, 5.0]:
  for pj in [0.5]:
    lp=make_logp(m); comp=make_composite(lp,D,N_CHAINS,L=L,step_size=STEP,K=K,b0=0.05,p_jump=pj)
    t0=time.time()
    R=3000
    vt=vanilla_tail(comp,m,total_steps=R*K,seed=10)
    rng=np.random.default_rng(7); cl=(rng.random(N_CHAINS)>=0.7).astype(int); zt=rng.standard_normal((N_CHAINS,D)); zt[:,0]+=np.where(cl==0,m,-m)
    fr_t,acc_t=de_frac(comp,jnp.asarray(zt),R,seed=30)
    fr_s,acc_s=de_frac(comp,jnp.zeros((N_CHAINS,D)).at[:,0].set(m),R,seed=20)
    tau=iat(fr_t)
    print(f"m={m} pj={pj} barrier={m*m/2:.1f} | vanilla_tail(+,{R*K}steps)={vt:.3f} | "
          f"DE_truth mean={fr_t.mean():.3f} tau={tau:.0f} acc={acc_t:.3f} | "
          f"DE_single final500={fr_s[-500:].mean():.3f} min={fr_s.min():.3f} | [{time.time()-t0:.0f}s]")
