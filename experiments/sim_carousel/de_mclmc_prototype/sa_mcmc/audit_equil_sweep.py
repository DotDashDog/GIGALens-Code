"""GRADER audit: does SA-mixture's COMPARABLE-MASS re-equilibration survive to
carousel-level curvature, where the affine DE baseline truly floors?

C2 showed SA (0.30->0.605, 7.6 flips/rd) beats DE (0.30->0.547, 0.56 flips/rd) on
the b=0.70 testbed -- but DE there still partially moved (8.5% composite acc), i.e.
that testbed is dynamically MILDER than the real carousel (DE ~0.6%, 0 round-trips).
This sweep cranks curvature b and checks, at each, whether SA still re-equilibrates
a WRONG populated weight (0.30 -> 0.60) while DE flips -> 0. Weight/flips depend on
mode assignment (dim1), NOT the dim0 ridge collapse, so this metric is valid at high b.

Pre-reg: prediction = SA flips/rd stays high (on-ridge proposals are curvature-robust)
and SA end-weight -> 0.60 across all b; DE flips/rd -> 0 and DE end-weight stays near
the 0.30 init as b grows. Falsifier of the SA win = SA also floors (flips->0, end~0.30)
at the b where DE floors => SA's advantage was only on mild curvature.
"""
import os, sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from curved_testbed import build_target, static_linear_de_acceptance
from sa_move import make_sa_composite
from de_mclmc import make_composite
def pr(*a): print(*a, flush=True)

N, L, STEP, K = 64, 2.0, 0.2, 20
R = int(os.environ.get("R", "800")); SEED = 20260628
BS = [0.7, 1.5, 3.0, 6.0]

def classify_frac(tgt, pos): return (tgt.classify(np.asarray(pos)) == 0).mean(axis=-1)
def flips(tgt, pos):
    lab = tgt.classify(pos); return float((lab[1:] != lab[:-1]).sum(axis=1).mean())

def run(comp, init, nr, key):
    st = comp["init_states"](jnp.asarray(init), jax.random.fold_in(key, 0))
    keys = jax.random.split(key, nr); pos = np.empty((nr, N, init.shape[1])); acc = np.empty(nr)
    for r in range(nr):
        st, (p, ec, a) = comp["round"](st, keys[r]); pos[r] = np.asarray(p); acc[r] = float(np.asarray(a).mean())
    return pos, acc

def main():
    t0 = time.time(); rows = []
    for b in BS:
        tgt = build_target(b=b, n_curve=4); D = tgt.D
        imm = jnp.asarray(tgt.within_mode_cov(mode=1))
        g1 = static_linear_de_acceptance(tgt, n=40000)
        # wrong populated init: w_A = 0.30 (truth 0.60)
        rngw = np.random.default_rng(SEED + 7)
        nA = int(round(0.30 * N)); compl = np.ones(N, int); compl[:nA] = 0
        Xw = rngw.standard_normal((N, D)) * np.sqrt(tgt.var)[None, :] + tgt.mu[compl]
        init = tgt.f_np(Xw)
        sa = make_sa_composite(tgt.logp, D, N, L=L, step_size=STEP, K=K, n_sa=N,
                               proposal="mixture", bandwidth=0.20, kernel_cov=jnp.eye(D),
                               inverse_mass_matrix=imm)
        de = make_composite(tgt.logp, D, N, L=L, step_size=STEP, K=K, b0=0.05, p_jump=0.5,
                            inverse_mass_matrix=imm)
        psa, _ = run(sa, init, R, jax.random.key(SEED + 1))
        pde, ade = run(de, init, R, jax.random.key(SEED + 2))
        half = R // 2
        sa_end = float(classify_frac(tgt, psa)[half:].mean()); sa_fl = flips(tgt, psa)
        de_end = float(classify_frac(tgt, pde)[half:].mean()); de_fl = flips(tgt, pde)
        row = dict(b=b, g1_static_pct=g1*100, sa_end=sa_end, sa_flips=sa_fl,
                   de_end=de_end, de_flips=de_fl, de_acc_pct=float(ade.mean())*100)
        rows.append(row)
        pr(f"b={b:4.1f} | GATE1stat={g1*100:5.2f}% | SA end={sa_end:.3f} flips/rd={sa_fl:5.2f} "
           f"| DE end={de_end:.3f} flips/rd={de_fl:5.2f} acc={ade.mean()*100:5.2f}%  (truth 0.60, init 0.30)")
        with open(os.path.join(HERE, "audit_equil_sweep.json"), "w") as f:
            json.dump(rows, f, indent=2, default=float)
    pr(f"\ntotal wall {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
