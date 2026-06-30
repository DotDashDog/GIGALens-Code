"""GPU carousel PARALLEL TEMPERING test (C-19 follow-up): does a continuously-mixing
cold replica DRAIN the ~1e-5 secondary to occ(global)->~1.0, where tempered-burn-in
froze out at ~0.6? All replicas init in the SECONDARY basin (production scenario).

- R replicas at ascending betas (geom 0.02->1.0); per-level tempered MCLMC kernel.
- Per-(level,chain) EEVPD step adaptation = the USER's mclmc.step_size_adapt (defaults
  DEVAR=5e-4/TRUST=1.5/decay=149/151), carried across rounds within a level (fixes the
  C-19 fixed-step confound; the cold replica visits both basins).
- Swap math VERIFIED-faithful to tempering/parallel_tempering.py: even/odd adjacent pairs,
  alpha=min(1,exp[(beta_r-beta_{r+1})(logp_base(x_{r+1})-logp_base(x_r))]).
- INCREMENTAL save of per-round cold occ(global) + swap acc + per-level EEVPD.

PRE-REG: prediction cold occ(global) -> ~1.0 (secondary drains) with faithful per-level
EEVPD (~5e-4) and non-trivial swap acceptance. falsifier: cold occ stays ~0.6 (freeze-out
NOT removed -> PT insufficient on the real posterior) or swaps ~0 (ladder disconnected).
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
SCR_DIAG = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts"
HERE = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/de_mclmc_prototype"
sys.path.insert(0, SCR_DIAG); sys.path.insert(0, HERE)
from build_model import build
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, isokinetic_mclachlan_smart, _single_init, handle_nans)
def pr(*a): print(*a, flush=True)

pm = build()
def logp(z): return pm.log_prob(z)[0]
logp_v = jax.jit(jax.vmap(logp))
D = 14; THR = 4.40; COL = 9
prep = np.load(os.path.join(SCR_DIAG, "basin_prep.npz"))
upper_cov = jnp.asarray(prep["upper_cov"]); L0 = float(prep["L_final"]); ss0 = float(prep["ss_final"])
z_best = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/map/arrays.npz")["z_best"].reshape(-1)
sec_center = np.asarray(z_best)
_sz = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/mclmc/arrays.npz")["samples_z"].reshape(-1, 14)
up_samples = _sz[_sz[:, 9] > 4.40]                       # real global-basin draws
NPZ = os.path.join(HERE, "pt_carousel.npz")

DEVAR = 5e-4; TRUST = 1.5; DECAY = (150.0 - 1.0) / (150.0 + 1.0)
R = 12; NSYS = 16; K = 10; ROUNDS = 90
betas = np.geomspace(0.005, 1.0, R)   # lower beta_min: barrier ~415 nats needs beta~0.005 for K=10 to cross

def adapt_one(prev_state, next_state, info, step_size, adapt_state, nan_key):
    time_, x_avg, ss_max = adapt_state
    success, state, ss_max, ec = handle_nans(prev_state, next_state, step_size, ss_max, info.energy_change, nan_key)
    xi = jnp.square(ec) / (D * DEVAR) + 1e-8
    weight = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * TRUST)))
    x_avg = DECAY * x_avg + weight * (xi / jnp.power(step_size, 6.0))
    time_ = DECAY * time_ + weight
    new_step = jnp.minimum(jnp.power(x_avg / time_, -1.0 / 6.0), ss_max)
    return state, new_step, (time_, x_avg, ss_max), jnp.square(info.energy_change)

def occ_global(P): return float((np.asarray(P)[..., COL] > THR).mean())

# per-level tempered kernels + jitted adaptive K-step runners
kernels, tfns, runners = [], [], []
for b in betas:
    tf = (lambda bb: (lambda z: bb * logp(z)))(float(b))
    kern = _build_kernel_shardmap(logdensity_fn=tf, inverse_mass_matrix=upper_cov, integrator=isokinetic_mclachlan_smart)
    tfns.append(tf); kernels.append(kern)
    def _mk(kern):
        @jax.jit
        def run(states, ss, ast, keys):  # ss (n,), ast tuple of (n,), keys (K,n)
            def sstep(carry, kt):
                states, ss, ast = carry
                def per(s, step, a, k):
                    kk, kn = jax.random.split(k)
                    ns, info = kern(rng_key=kk, state=s, L=L0, step_size=step)
                    s2, new, a2, ec2 = adapt_one(s, ns, info, step, a, kn)
                    return s2, new, a2, ec2
                s2, new, a2, ec2 = jax.vmap(per)(states, ss, ast, kt)
                return (s2, new, a2), ec2
            (states, ss, ast), ec2 = jax.lax.scan(sstep, (states, ss, ast), keys)
            return states, ss, ast, ec2
        return run
    runners.append(_mk(kern))

def init_level(positions, tf, key):
    keys = jax.random.split(key, positions.shape[0])
    return jax.vmap(lambda p, k: _single_init(p, tf, k))(jnp.asarray(positions), keys)

def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # ALL-SECONDARY init (production bad-MAP scenario) + beta_min=0.005: tests whether PT now
    # both DISCOVERS (hot level crosses) AND DRAINS (cold->~1.0) in one run.
    pos = (np.broadcast_to(sec_center[None, None, :], (R, NSYS, D)) + 1e-3 * rng.standard_normal((R, NSYS, D))).copy()
    pr(f"PT: R={R} NSYS={NSYS} K={K} ROUNDS={ROUNDS}; ALL-SECONDARY init occ(global)={occ_global(pos):.3f} "
       f"(truth ~1.0 since secondary ~1e-5)")
    pr(f"betas={[round(float(b),3) for b in betas]}")
    steps = [jnp.full(NSYS, ss0) for _ in range(R)]
    adapt = [(jnp.zeros(NSYS), jnp.zeros(NSYS), jnp.full(NSYS, 1.0)) for _ in range(R)]
    key = jax.random.key(7); swap_rng = np.random.default_rng(1)
    cold_occ = np.zeros(ROUNDS); hot_occ = np.zeros(ROUNDS); mid_occ = np.zeros(ROUNDS)
    swap_acc_run = np.zeros(R - 1); ev_last = np.zeros(R)

    for t in range(ROUNDS):
        key, *lks = jax.random.split(key, R + 1)
        lp = np.empty((R, NSYS))
        for r in range(R):
            st = init_level(pos[r], tfns[r], lks[r])            # momentum refresh
            kk = jax.random.split(jax.random.fold_in(lks[r], 1), K * NSYS).reshape(K, NSYS)
            st, steps[r], adapt[r], ec2 = runners[r](st, steps[r], adapt[r], kk)
            pos[r] = np.asarray(st.position); ev_last[r] = float(np.mean(np.asarray(ec2)) / D)
            lp[r] = np.asarray(logp_v(jnp.asarray(pos[r])))     # BASE logp, 16-wide (avoids 160-wide FFT OOM)
        # even/odd adjacent swaps
        parity = t % 2
        for r in range(parity, R - 1, 2):
            dbeta = betas[r] - betas[r + 1]
            accept = np.log(swap_rng.random(NSYS)) < dbeta * (lp[r + 1] - lp[r])
            pr_, pr1 = pos[r].copy(), pos[r + 1].copy()
            pos[r] = np.where(accept[:, None], pr1, pr_); pos[r + 1] = np.where(accept[:, None], pr_, pr1)
            l_, l1 = lp[r].copy(), lp[r + 1].copy()
            lp[r] = np.where(accept, l1, l_); lp[r + 1] = np.where(accept, l_, l1)
            swap_acc_run[r] += accept.mean()
        cold_occ[t] = occ_global(pos[-1]); hot_occ[t] = occ_global(pos[0]); mid_occ[t] = occ_global(pos[R//2])
        if t % 5 == 0 or t == ROUNDS - 1:
            pr(f"  round {t:3d}  occ(global) hot/mid/cold={hot_occ[t]:.3f}/{mid_occ[t]:.3f}/{cold_occ[t]:.3f}  "
               f"EEVPD[h/m/c]={ev_last[0]:.1e}/{ev_last[R//2]:.1e}/{ev_last[-1]:.1e}")
            np.savez(NPZ, betas=betas, cold_occ=cold_occ[:t+1], hot_occ=hot_occ[:t+1], mid_occ=mid_occ[:t+1],
                     swap_acc=swap_acc_run/(t+1)*2, ev_last=ev_last, cold_col9=pos[-1][:, COL].astype(np.float32))
    burn = ROUNDS // 3
    occ_mean = float(cold_occ[burn:].mean())
    pr(f"\n========== PT RESULT (PROPOSED/UNCERTIFIED) ==========")
    pr(f"  cold occ(global) time-avg (post-burn) = {occ_mean:.3f}  (truth ~1.0; tempered-burnin froze ~0.6)")
    pr(f"  swap acceptance per adjacent pair = {np.round(swap_acc_run/ROUNDS*2,3)}")
    pr(f"  per-level EEVPD (last) = {np.array2string(ev_last, precision=1)}  (target 5e-4)")
    pr(f"  wall {time.time()-t0:.0f}s -> {NPZ}")

if __name__ == "__main__":
    main()
