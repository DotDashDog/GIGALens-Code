import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import tensorflow_probability.substrates.jax as tfp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
names=list(np.load(WD+"names.npy"))
U=np.load(WD+"run_under.npz"); O=np.load(WD+"run_over.npz")
nb=int(U['nb']); nr=int(U['nr'])
def rhat(sz):  # sz (chains,steps,params)
    t=jnp.transpose(jnp.asarray(sz),(1,0,2))
    return np.asarray(tfp.mcmc.potential_scale_reduction(t))
def results(run): return run['position'][:, -int(run['nr']):, :]

print("="*70); print("H1: EQUILIBRATION (under vs over-dispersed init)"); print("="*70)
for tag,run in [("under",U),("over",O)]:
    res=results(run); C,N,P=res.shape
    rh=rhat(res); 
    # running rhat over results
    sched=np.unique(np.geomspace(max(50,N//40),N,8).astype(int))
    rr=[np.nanmax(rhat(res[:,:n,:])) for n in sched]
    # drift
    fh=res[:,:N//4,:].mean(1); lh=res[:,-N//4:,:].mean(1); sd=res.std(1)
    drift=np.abs((lh-fh)/sd)
    print(f"\n[{tag}] results shape {res.shape}  max R-hat(tfp)={np.nanmax(rh):.2f}")
    print(f"  running max-Rhat over results: " + " ".join(f"{n}:{v:.1f}" for n,v in zip(sched,rr)))
    print(f"  drift |Δmean|/std: median={np.median(drift):.2f} max={np.max(drift):.2f}")
# do under & over reach the SAME marginals?
ru=results(U).reshape(-1,32); ro=results(O).reshape(-1,32)
print("\n[H1 agreement] per-param median diff in pooled-std units (worst 8):")
pooled=np.sqrt(0.5*(ru.var(0)+ro.var(0)))
mdiff=np.abs(np.median(ru,0)-np.median(ro,0))/np.maximum(pooled,1e-12)
for i in np.argsort(-mdiff)[:8]:
    print(f"  {i:2d} {names[i]:30s} |Δmed|/pooledstd={mdiff[i]:.2f}  "
          f"under_std={ru[:,i].std():.3f} over_std={ro[:,i].std():.3f}")
print(f"  median over all params: {np.median(mdiff):.2f}")

print("\n"+"="*70); print("H2: xi (energy error) vs POSITION  [under run, results phase]"); print("="*70)
xi=U['xi'][:, -nr:]              # (chains, nr)
pos=U['position'][:, -nr:, :]    # (chains, nr, dim)
print(f"results-phase xi: median={np.median(xi):.2e} frac>1={np.mean(xi>1):.3f} "
      f"frac==-1={np.mean(xi==-1.0):.3f} max={xi.max():.2e}")
lxi=np.log10(np.clip(xi,1e-12,None)).ravel()
# correlate log xi with |z_i - median_i| per param
flat=pos.reshape(-1,32); med=np.median(flat,0); sd=flat.std(0)+1e-12
dev=np.abs(flat-med)/sd
corr=np.array([np.corrcoef(lxi, dev[:,i])[0,1] for i in range(32)])
print("\nparams whose extremeness most predicts high xi (corr log10 xi vs |z-med|/std):")
for i in np.argsort(-corr)[:10]:
    print(f"  {i:2d} {names[i]:30s} corr={corr[i]:+.3f}")
# also step displacement vs xi
print(f"\nfrac of results-phase steps with xi>1: {np.mean(xi>1):.3f}  (target ~ controlled)")
np.save(WD+"h2_corr.npy", corr)
print("\nDONE")
