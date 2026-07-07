"""Lean barrier-crossing sweep: decisive L-dependence + one MM contrast + basin-finding.
All configs 8 chains x 10k steps, secondary-basin init (except basin-finding = global init).
Saves incrementally to D_lean_data.npz after EACH config so partial results survive a kill.
No matplotlib (plots made separately from the npz). Reuses fixed_knob_mclmc harness.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
SCR = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts"
sys.path.insert(0, SCR)
import fixed_knob_mclmc as H

OUT = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots"
NPZ = os.path.join(OUT, "D_lean_data.npz")
def pr(*a): print(*a, flush=True)

prep = np.load(os.path.join(SCR, "basin_prep.npz"))
upper = prep["upper_center"]   # upper-cluster mean (off-ridge) — used only for basin-finding init
pooled = jnp.asarray(prep["pooled_cov"]); upper_cov = jnp.asarray(prep["upper_cov"])
L0 = float(prep["L_final"]); ss0 = float(prep["ss_final"])
ident = jnp.asarray(np.eye(H.DIM))
NSTEP = 10000; NCH = 8
# Secondary-basin init = the on-ridge MAP (z_best, col9~4.339) = exactly where the real run
# started via diag_qz. The cluster MEAN lands ~2000 logp off-ridge, so do NOT use it as the init.
sec_center = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/"
                     "messy_tests/minimal_case/map/arrays.npz")["z_best"].reshape(-1)
pr(f"L0={L0:.3f} ss0={ss0:.4f}  thresh={H.THRESH}  nstep={NSTEP} nch={NCH}")
pr("logp sec_center(z_best)=%.1f col9=%.4f | upper_mean=%.1f" % (
    float(H.log_prob_single(jnp.asarray(sec_center))), float(sec_center[9]),
    float(H.log_prob_single(jnp.asarray(upper)))))
store_sec = sec_center

store = {"L0": L0, "ss0": ss0, "thresh": H.THRESH, "names": []}
def save():
    np.savez(NPZ, **store)

def run(name, runner, center, Lmult, seed=0):
    t = time.time()
    init = H.init_ball(center, NCH, 1e-3, seed)
    c9, ec, logp, nonan = runner(init, jnp.asarray(L0*Lmult), jnp.asarray(ss0), NSTEP, seed)
    c9 = np.array(c9); ec = np.array(ec); nonan = np.array(nonan)
    mfpt, comm = H.analyze(c9)
    ess = H.col9_ess(c9)
    xi = ec**2 / (H.DIM*H.DESIRED_EVAR) + 1e-8
    dt = time.time()-t
    store[name+"__col9"] = c9.astype(np.float32)
    store[name+"__mfpt"] = mfpt; store[name+"__committed"] = comm
    store[name+"__ess"] = ess
    store[name+"__xiq"] = np.percentile(xi, [50,90,99,100])
    store[name+"__nonan"] = float(nonan.mean())
    store["names"] = list(store["names"]) + [name]
    save()
    nesc = int(np.sum(~np.isnan(mfpt)))
    pr(f"[{name}] {dt:.0f}s  escaped {nesc}/{NCH}  MFPT={np.round(mfpt,0)}  "
       f"ESS(col9)med={np.median(ess):.1f}  xi(med/99/max)="
       f"{np.percentile(xi,50):.1f}/{np.percentile(xi,99):.0f}/{xi.max():.0f} nonan={nonan.mean():.3f}")
    return mfpt

# --- PRIMARY: pooled MM, L sweep, secondary init (one compiled runner; L is a runtime arg) ---
pr("\n=== pooled MM, secondary init, L sweep ===")
pooled_runner = H.make_runner(pooled)
for Lm in [1, 2, 4, 8]:
    run(f"sec_pooled_L{Lm}x", pooled_runner, sec_center, Lm)

# --- BASIN-FINDING: pooled MM, GLOBAL init (reuse runner, free): does it ever fall to secondary? ---
pr("\n=== basin-finding: pooled MM, global init, L1x ===")
run("glob_pooled_L1x", pooled_runner, upper, 1)

# --- MM CONTRAST at L1x, secondary init ---
pr("\n=== MM contrast (identity, upper-mode) secondary init L1x ===")
run("sec_ident_L1x", H.make_runner(ident), sec_center, 1)
run("sec_upperMM_L1x", H.make_runner(upper_cov), sec_center, 1)

pr("\nALL DONE; data ->", NPZ)
