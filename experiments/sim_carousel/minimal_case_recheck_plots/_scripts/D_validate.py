import os, sys, numpy as np, jax, jax.numpy as jnp, time
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts")
import fixed_knob_mclmc as H
def pr(*a): print(*a, flush=True)
pr("devices", jax.devices())
prep = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts/basin_prep.npz")
lower = prep["lower_center"]; upper = prep["upper_center"]
pooled = jnp.asarray(prep["pooled_cov"]); L0=float(prep["L_final"]); ss0=float(prep["ss_final"])
pr("L0",L0,"ss0",ss0)
# sanity: logp at the two basin centers
pr("logp lower_center", float(H.log_prob_single(jnp.asarray(lower))))
pr("logp upper_center", float(H.log_prob_single(jnp.asarray(upper))))

runner = H.make_runner(pooled)
# quick validation: secondary init, 8 chains, 2000 steps
init = H.init_ball(lower, 8, 1e-3, seed=0)
t=time.time()
c9, ec, logp, nonan = runner(init, jnp.asarray(L0), jnp.asarray(ss0), 2000, 0)
c9=np.array(c9); ec=np.array(ec); logp=np.array(logp); nonan=np.array(nonan)
pr("compile+run 8ch x2000 took %.1fs"%(time.time()-t))
xi = ec**2/(H.DIM*H.DESIRED_EVAR)+1e-8
pr("col9 start mean", c9[0].mean(), "end mean", c9[-1].mean())
pr("col9 min/max over run", c9.min(), c9.max())
pr("frac steps above thresh (per chain)", np.round((c9>H.THRESH).mean(0),3))
pr("xi median", np.median(xi), "xi 90pct", np.percentile(xi,90), "max", xi.max())
pr("nonan frac", nonan.mean())
pr("logp median", np.median(logp), "logp range", logp.min(), logp.max())
mfpt, comm = H.analyze(c9)
pr("MFPT (first up-cross)", mfpt)
pr("committed escape step", comm)
# timing for a 10k run estimate
t=time.time()
c9b,_,_,_ = runner(init, jnp.asarray(L0), jnp.asarray(ss0), 2000, 1)
np.array(c9b).block_until_ready() if hasattr(np.array(c9b),'block_until_ready') else None
pr("second run (already compiled) 2000 steps %.1fs"%(time.time()-t))
pr("VALIDATE DONE")
