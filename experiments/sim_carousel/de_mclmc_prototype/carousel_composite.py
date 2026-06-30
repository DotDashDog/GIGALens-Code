"""Carousel double-well confirmation of the DE-MCLMC composite (orchestrator-run, GPU).
Tests whether the composite produces the round-trips + correct (~all-global) occupancy
that vanilla MCLMC could not. Honest within-mode preconditioner (upper-mode cov), NOT pooled.
Controls: vanilla MCLMC (no DE), refresh-only (momentum refresh every K, no DE).
Saves raw arrays only -> plotted separately. Incremental save after each config.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
SCR_DIAG="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts"
HERE="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/de_mclmc_prototype"
sys.path.insert(0, SCR_DIAG); sys.path.insert(0, HERE)
from build_model import build
import de_mclmc
from gigalens_research.inference.blackjax_updated_utils import _build_kernel_shardmap, isokinetic_mclachlan_smart, _single_init

def pr(*a): print(*a, flush=True)
OUT=HERE; NPZ=os.path.join(OUT,"C_carousel_data.npz")
pm=build()
def logp(z): return pm.log_prob(z)[0]
D=14; THR=4.40; COL=9

prep=np.load(os.path.join(SCR_DIAG,"basin_prep.npz"))
upper_cov=jnp.asarray(prep["upper_cov"]); L0=float(prep["L_final"]); ss0=float(prep["ss_final"])
z_best=np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/map/arrays.npz")["z_best"].reshape(-1)
sec_center=jnp.asarray(z_best)

# --- on-ridge GLOBAL-basin inits: use REAL upper-cluster posterior samples (guaranteed
#     on-ridge, unlike the cluster mean which is off-ridge by ~1150 logp) ---
sz=np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/mclmc/arrays.npz")["samples_z"]
flat=sz.reshape(-1,D); up_samples=flat[flat[:,COL]>THR]      # upper-cluster draws
rng0=np.random.default_rng(123); glob_pool=jnp.asarray(up_samples[rng0.choice(up_samples.shape[0], 64, replace=False)])
glob_center=glob_pool[0]
pr("sec logp=%.1f (col9=%.3f) | glob-sample logp=%.1f (col9=%.3f) | n_up_samples=%d"%(
    float(logp(sec_center)),float(sec_center[COL]),float(logp(glob_center)),float(glob_center[COL]),up_samples.shape[0]))

NCH=16; K=20; ROUNDS=500  # 16 chains (better DE pairs; flat in D). 500*20=10000 MCLMC steps
# Cholesky-whitened jitter: eps ~ N(0, b0^2 * upper_cov) respects within-mode CORRELATIONS
# (the carousel has tight correlated degeneracies; diagonal scaling would be many-sigma along
# the stiff eigendirections). b0=0.1 -> jitter ~0.1 of a within-mode draw (small ergodicity term).
L_within=jnp.linalg.cholesky(upper_cov)
comp=de_mclmc.make_composite(logp, D, NCH, L=L0, step_size=ss0, K=K,
                             b0=0.1, p_jump=0.3, inverse_mass_matrix=upper_cov,
                             eps_scale=L_within)

def ball(center,n,scale,seed):
    rng=np.random.default_rng(seed)
    return jnp.asarray(np.asarray(center)[None,:]+scale*rng.standard_normal((n,D)))

def balanced_init(seed):
    # INTERLEAVE modes so the two-group split [:half],[half:] each straddles BOTH
    # modes -> difference vectors z_a-z_b can span the inter-mode gap (required for
    # DE mode-jumps; a mode-segregated split makes gap jumps structurally impossible).
    a=ball(sec_center,NCH//2,1e-3,seed)                       # secondary draws
    rng=np.random.default_rng(seed+99)
    b=glob_pool[rng.choice(glob_pool.shape[0],NCH//2,replace=False)]  # on-ridge global draws
    out=np.empty((NCH,D)); out[0::2]=np.asarray(a); out[1::2]=np.asarray(b)  # [sec,glob,sec,glob,...]
    return jnp.asarray(out)

store={"thr":THR,"L0":L0,"ss0":ss0,"names":[]}
def save(): np.savez(NPZ,**store)

def crossings(c9):  # c9 (T,C): per-chain up+down crossings of THR, and round-trips
    above=c9>THR; T,C=c9.shape
    ncross=np.zeros(C,int); nround=np.zeros(C,int)
    for ci in range(C):
        a=above[:,ci].astype(int); ch=np.abs(np.diff(a)); ncross[ci]=ch.sum()
        # round trip = at least one up AND one down crossing
        ups=np.sum(np.diff(a)==1); dns=np.sum(np.diff(a)==-1); nround[ci]=min(ups,dns)
    return ncross,nround

def run_composite(name, init, seed):
    t=time.time(); st=comp["init_states"](init, jax.random.key(seed)); key=jax.random.key(seed+7)
    pos=[]
    rk=jax.random.split(key, ROUNDS)
    def body(carry, k):
        s=carry; s2,(p,ec,acc)=comp["round"](s,k); return s2,(p[:,COL],acc.mean())
    _,(c9,acc)=jax.lax.scan(body, st, rk)
    c9=np.array(c9); acc=np.array(acc)        # c9: (ROUNDS, NCH)
    ncross,nround=crossings(c9)
    occ=float((c9>THR).mean())
    store[name+"__c9"]=c9.astype(np.float32); store[name+"__acc"]=acc
    store[name+"__ncross"]=ncross; store[name+"__nround"]=nround; store[name+"__occ"]=occ
    store["names"]=list(store["names"])+[name]; save()
    pr(f"[{name}] {time.time()-t:.0f}s  round-trips/chain={nround}  total_cross={ncross.sum()}  "
       f"occ(global)={occ:.3f}  DEacc={acc.mean():.3f}")

def run_vanilla(name, init, seed):
    t=time.time(); st=comp["init_states"](init, jax.random.key(seed))
    keys=jax.random.split(jax.random.key(seed+7),(K*ROUNDS,NCH))
    _,pos=comp["mclmc_only"](st,keys); c9=np.array(pos[:,:,COL])
    # subsample to ROUNDS for fair crossing stats
    c9s=c9[::K]; ncross,nround=crossings(c9s); occ=float((c9>THR).mean())
    store[name+"__c9"]=c9s.astype(np.float32); store[name+"__ncross"]=ncross
    store[name+"__nround"]=nround; store[name+"__occ"]=occ; store["names"]=list(store["names"])+[name]; save()
    pr(f"[{name}] {time.time()-t:.0f}s  round-trips/chain={nround}  total_cross={ncross.sum()}  occ(global)={occ:.3f}")

def run_refresh_only(name, init, seed):
    # K MCLMC steps then momentum refresh (no DE), repeated ROUNDS times
    t=time.time()
    kernel=_build_kernel_shardmap(logdensity_fn=logp, inverse_mass_matrix=upper_cov, integrator=isokinetic_mclachlan_smart)
    st=comp["init_states"](init, jax.random.key(seed))
    def kstep(s,kk):
        def per(ss,k): ns,info=kernel(rng_key=k,state=ss,L=L0,step_size=ss0); return ns,ns.position[COL]
        return jax.vmap(per)(s,kk)
    def rnd(carry,k):
        s=carry; mk,rk=jax.random.split(k)
        mks=jax.random.split(mk,(K,NCH)); s2,c=jax.lax.scan(kstep,s,mks)
        pos=s2.position; s3=comp["init_states"](pos, rk)   # momentum refresh
        return s3, pos[:,COL]
    _,c9=jax.lax.scan(rnd, st, jax.random.split(jax.random.key(seed+7),ROUNDS))
    c9=np.array(c9); ncross,nround=crossings(c9); occ=float((c9>THR).mean())
    store[name+"__c9"]=c9.astype(np.float32); store[name+"__ncross"]=ncross
    store[name+"__nround"]=nround; store[name+"__occ"]=occ; store["names"]=list(store["names"])+[name]; save()
    pr(f"[{name}] {time.time()-t:.0f}s  round-trips/chain={nround}  total_cross={ncross.sum()}  occ(global)={occ:.3f}")

pr("\n=== composite, balanced init (clean equilibration test) ===")
run_composite("comp_balanced", balanced_init(0), 0)
pr("\n=== controls, balanced init ===")
run_vanilla("vanilla_balanced", balanced_init(0), 0)
run_refresh_only("refresh_balanced", balanced_init(0), 0)
pr("\n=== composite, ALL-secondary init (production scenario) ===")
run_composite("comp_allsec", ball(sec_center,NCH,1e-3,0), 0)
pr("\nDONE ->",NPZ)
