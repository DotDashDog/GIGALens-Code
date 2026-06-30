# A1: autodiff vs finite-difference gradient self-consistency at TYPICAL-SET points.
import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; flat=sz.reshape(-1,32)
rng=np.random.default_rng(0)
pts=flat[rng.choice(len(flat),15,replace=False)]            # typical set (posterior draws)
lp=lambda z: prob_model.log_prob(z)[0]
jlp=jax.jit(jax.vmap(lp)); jgrad=jax.jit(jax.vmap(jax.grad(lp)))
def blp(Z,bs=32):
    o=[]; Z=np.asarray(Z)
    for k in range(0,len(Z),bs): o.append(np.asarray(jlp(jnp.asarray(Z[k:k+bs]))))
    return np.concatenate(o)
G=np.asarray(jgrad(jnp.asarray(pts)))                       # autodiff grads at typical pts
print(f"typical-set |grad| (autodiff): median={np.median(np.linalg.norm(G,axis=1)):.2e} "
      f"(cf. bad-MAP ~4e4)  lp range={blp(pts).min():.0f}..{blp(pts).max():.0f}")
hs=np.geomspace(1e-2,1e-9,16)
ndir=4
# build all perturbed points
P=[]; meta=[]
for pi,z0 in enumerate(pts):
    for di in range(ndir):
        v=rng.normal(size=32); v/=np.linalg.norm(v)
        Dad=float(G[pi]@v)
        for h in hs:
            P.append(z0+h*v); P.append(z0-h*v); meta.append((pi,di,h,Dad))
vals=blp(np.array(P))
vals=vals.reshape(-1,2)   # (nmeta, [+,-])
best=[]
for k,(pi,di,h,Dad) in enumerate(meta):
    pass
# recompute grouped: meta has len = 15*4*16
import collections
rel_by_pd=collections.defaultdict(list)
for k,(pi,di,h,Dad) in enumerate(meta):
    Dfd=(vals[k,0]-vals[k,1])/(2*h)
    rel=abs(Dfd-Dad)/(abs(Dad)+1e-30)
    rel_by_pd[(pi,di)].append((h,rel,Dad))
minrel=[]
for key,lst in rel_by_pd.items():
    rels=[r for _,r,_ in lst]; minrel.append(min(rels))
minrel=np.array(minrel)
print(f"\nA1 RESULT: min-over-h relative error |FD-AD|/|AD| across {len(minrel)} (point,direction) probes")
print(f"  median={np.median(minrel):.2e}  90th pct={np.quantile(minrel,.9):.2e}  max={minrel.max():.2e}")
print(f"  fraction with min-rel-err > 1e-3 (would indicate non-smoothness): {np.mean(minrel>1e-3):.2f}")
# show the h-convergence for the WORST probe
worst=max(rel_by_pd.items(), key=lambda kv: min(r for _,r,_ in kv[1]))
print(f"\n  worst probe (point {worst[0][0]}, dir {worst[0][1]}), Dad={worst[1][0][2]:.3e}: rel-err vs h")
for h,r,_ in worst[1]: print(f"    h={h:.1e}  rel={r:.2e}")
