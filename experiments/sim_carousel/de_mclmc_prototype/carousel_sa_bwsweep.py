"""Is SA's carousel freeze just a TOO-BROAD kernel? Sweep the KDE bandwidth on the
real posterior and see if a SMALL bw lets the secondary DRAIN (balanced init).

Mechanism under test (corrected): SA drains by proposing a near-copy of an existing
GLOBAL chain (on-manifold => kept) and deleting a SECONDARY chain (lambda_sec high).
At bw=0.2 the proposal lands -314 nats off-manifold => proposal itself deleted => frozen.
PREDICTION: at small bw (~0.02-0.05) proposals near a global chain stay on-manifold
=> occ(global) DRAINS from 0.5 toward 1.0. FALSIFIER: still frozen at all bw => the
freeze is NOT bandwidth (structural after all).
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
SCR_DIAG = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts"
HERE = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/de_mclmc_prototype"
sys.path.insert(0, SCR_DIAG); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "sa_mcmc"))
from build_model import build
from sa_move import make_sa_composite
def pr(*a): print(*a, flush=True)

pm = build()
def logp(z): return pm.log_prob(z)[0]
D = 14; THR = 4.40; COL = 9
prep = np.load(os.path.join(SCR_DIAG, "basin_prep.npz"))
upper_cov = jnp.asarray(prep["upper_cov"]); L0 = float(prep["L_final"]); ss0 = float(prep["ss_final"])
z_best = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/map/arrays.npz")["z_best"].reshape(-1)
sec_center = np.asarray(z_best)
sz = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/mclmc/arrays.npz")["samples_z"]
flat = sz.reshape(-1, D); up = flat[flat[:, COL] > THR]
rng0 = np.random.default_rng(123); glob_pool = jnp.asarray(up[rng0.choice(up.shape[0], 64, replace=False)])

NCH = 16; K = 20; ROUNDS = 200
def balanced_init(seed):
    rng = np.random.default_rng(seed)
    a = sec_center[None, :] + 1e-3 * rng.standard_normal((NCH//2, D))
    b = np.asarray(glob_pool[np.random.default_rng(seed+99).choice(glob_pool.shape[0], NCH//2, replace=False)])
    out = np.empty((NCH, D)); out[0::2] = a; out[1::2] = b
    return jnp.asarray(out)

def run_occ(bw):
    sa = make_sa_composite(logp, D, NCH, L=L0, step_size=ss0, K=K, n_sa=NCH,
                           proposal="mixture", bandwidth=bw, kernel_cov=upper_cov,
                           inverse_mass_matrix=upper_cov)
    st = sa["init_states"](balanced_init(0), jax.random.key(0))
    rk = jax.random.split(jax.random.key(7), ROUNDS)
    def body(carry, k):
        s2, (p, ec, sub) = sa["round"](carry, k); return s2, ((p[:, COL] > THR).mean(), sub.mean())
    _, (occ, sub) = jax.lax.scan(body, st, rk)
    occ = np.array(occ); sub = np.array(sub)
    return occ, sub

def main():
    t0 = time.time()
    pr(f"balanced init occ(global)=0.500; truth~1.0 (secondary ~1e-5). ROUNDS={ROUNDS}")
    pr("bw      occ_first10  occ_last50  sub_rate   -> DRAINS?")
    for bw in [0.01, 0.02, 0.05, 0.1, 0.2]:
        occ, sub = run_occ(bw)
        of, ol = occ[:10].mean(), occ[-50:].mean()
        pr(f"{bw:5.2f}   {of:.3f}        {ol:.3f}       {sub.mean():.3f}      {'DRAINS' if ol>of+0.05 else 'frozen'}")
    pr(f"\nwall {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
