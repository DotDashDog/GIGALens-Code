"""GPU carousel test of the self-inclusive SA-MCMC mode-hop (C-15) on the REAL
minimal-carousel posterior, at the REAL adapted (EEVPD-tuned) step.  Compares
SA-mixture vs vanilla MCLMC vs linear-DE on the actual curved lens posterior.

Honest within-mode preconditioner (upper_cov); KDE kernel shaped by upper_cov
(on-ridge proposals).  Balanced init = equilibration test (should DRAIN the ~1e-5
secondary toward global, occ(global)->~1.0; tiny-mode pinning may leave ~1 chain).
All-secondary init = discovery test (expected to FAIL for any ensemble move).
Saves raw arrays only; plotted separately. Incremental save per config.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
SCR_DIAG = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts"
HERE = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/de_mclmc_prototype"
sys.path.insert(0, SCR_DIAG); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "sa_mcmc"))
from build_model import build
import de_mclmc
from sa_move import make_sa_composite
from gigalens_research.inference.blackjax_updated_utils import _build_kernel_shardmap, isokinetic_mclachlan_smart, _single_init
def pr(*a): print(*a, flush=True)

OUT = HERE; NPZ = os.path.join(OUT, "SA_carousel_data.npz")
pm = build()
def logp(z): return pm.log_prob(z)[0]
D = 14; THR = 4.40; COL = 9

prep = np.load(os.path.join(SCR_DIAG, "basin_prep.npz"))
upper_cov = jnp.asarray(prep["upper_cov"]); L0 = float(prep["L_final"]); ss0 = float(prep["ss_final"])
z_best = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/map/arrays.npz")["z_best"].reshape(-1)
sec_center = jnp.asarray(z_best)
sz = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/mclmc/arrays.npz")["samples_z"]
flat = sz.reshape(-1, D); up_samples = flat[flat[:, COL] > THR]
rng0 = np.random.default_rng(123); glob_pool = jnp.asarray(up_samples[rng0.choice(up_samples.shape[0], 64, replace=False)])
pr("sec logp=%.1f (col9=%.3f) | glob logp=%.1f (col9=%.3f) | L0=%.3f ss0=%.4f (EEVPD-tuned)" % (
    float(logp(sec_center)), float(sec_center[COL]), float(logp(glob_pool[0])), float(glob_pool[0][COL]), L0, ss0))

NCH = 16; K = 20; ROUNDS = 500
def ball(center, n, scale, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.asarray(center)[None, :] + scale * rng.standard_normal((n, D)))
def balanced_init(seed):
    a = ball(sec_center, NCH // 2, 1e-3, seed)
    rng = np.random.default_rng(seed + 99)
    b = glob_pool[rng.choice(glob_pool.shape[0], NCH // 2, replace=False)]
    out = np.empty((NCH, D)); out[0::2] = np.asarray(a); out[1::2] = np.asarray(b)
    return jnp.asarray(out)

def crossings(c9):
    above = c9 > THR; T, C = c9.shape
    ncross = np.zeros(C, int); nround = np.zeros(C, int)
    for ci in range(C):
        a = above[:, ci].astype(int); ncross[ci] = np.abs(np.diff(a)).sum()
        ups = np.sum(np.diff(a) == 1); dns = np.sum(np.diff(a) == -1); nround[ci] = min(ups, dns)
    return ncross, nround

store = {"thr": THR, "L0": L0, "ss0": ss0, "names": []}
def save(): np.savez(NPZ, **store)

# --- SA-mixture composite (on-ridge KDE: kernel_cov = within-mode upper_cov) ---
sa = make_sa_composite(logp, D, NCH, L=L0, step_size=ss0, K=K, n_sa=NCH,
                       proposal="mixture", bandwidth=0.2, kernel_cov=upper_cov,
                       inverse_mass_matrix=upper_cov)
de = de_mclmc.make_composite(logp, D, NCH, L=L0, step_size=ss0, K=K, b0=0.1, p_jump=0.3,
                             inverse_mass_matrix=upper_cov,
                             eps_scale=jnp.linalg.cholesky(upper_cov))

def run_round_based(comp, name, init, seed, get_extra):
    t = time.time(); st = comp["init_states"](init, jax.random.key(seed))
    rk = jax.random.split(jax.random.key(seed + 7), ROUNDS)
    def body(carry, k):
        s = carry; s2, (p, ec, extra) = comp["round"](s, k); return s2, (p[:, COL], get_extra(extra))
    _, (c9, ex) = jax.lax.scan(body, st, rk)
    c9 = np.array(c9); ex = np.array(ex)
    ncross, nround = crossings(c9); occ = float((c9 > THR).mean())
    store[name + "__c9"] = c9.astype(np.float32); store[name + "__extra"] = ex
    store[name + "__ncross"] = ncross; store[name + "__nround"] = nround; store[name + "__occ"] = occ
    store["names"] = list(store["names"]) + [name]; save()
    pr(f"[{name}] {time.time()-t:.0f}s  round-trips/chain={nround}  total_cross={ncross.sum()}  "
       f"occ(global)={occ:.3f}  extra(mean)={ex.mean():.3f}")

def run_vanilla(name, init, seed):
    t = time.time(); st = sa["init_states"](init, jax.random.key(seed))
    keys = jax.random.split(jax.random.key(seed + 7), (K * ROUNDS, NCH))
    _, pos = sa["mclmc_only"](st, keys); c9 = np.array(pos[:, :, COL])
    c9s = c9[::K]; ncross, nround = crossings(c9s); occ = float((c9 > THR).mean())
    store[name + "__c9"] = c9s.astype(np.float32); store[name + "__ncross"] = ncross
    store[name + "__nround"] = nround; store[name + "__occ"] = occ; store["names"] = list(store["names"]) + [name]; save()
    pr(f"[{name}] {time.time()-t:.0f}s  round-trips/chain={nround}  total_cross={ncross.sum()}  occ(global)={occ:.3f}")

pr("\n=== SA-mixture, balanced init (equilibration: should drain secondary -> occ~1.0) ===")
run_round_based(sa, "sa_balanced", balanced_init(0), 0, lambda sub: sub.mean())
pr("\n=== controls, balanced init ===")
run_vanilla("vanilla_balanced", balanced_init(0), 0)
run_round_based(de, "de_balanced", balanced_init(0), 0, lambda acc: acc.mean())
pr("\n=== SA-mixture, ALL-secondary init (discovery test; expected to FAIL) ===")
run_round_based(sa, "sa_allsec", ball(sec_center, NCH, 1e-3, 0), 0, lambda sub: sub.mean())
pr("\nDONE ->", NPZ)
