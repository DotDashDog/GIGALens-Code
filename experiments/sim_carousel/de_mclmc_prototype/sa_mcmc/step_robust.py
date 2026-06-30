"""STEP-ROBUSTNESS check (orchestrator): find an MCLMC step where the SA-mixture
COMPOSITE is invariant in BOTH the cross-mode weight AND the within-ridge marginal,
at carousel-level curvature (b=1.5) -- and stress at b=3.0.

The SA move is exact (Prop 1); the composite's residual bias is the UNADJUSTED MCLMC
kernel, which should approach the true target as step -> 0. From a TRUTH init (w_A=0.60,
dim0 std~3.0) the composite should PRESERVE both. We sweep step and measure drift
(1st vs 2nd half over a long run distinguishes BIAS from slow mixing).

Pre-reg: prediction = finer step monotonically reduces BOTH the weight drift and the
dim0 collapse, converging to (w_A=0.60, dim0=3.0). Falsifier (the b>=3 step-0.1-worse
anomaly): if a finer step does NOT reduce the weight drift at b=1.5, step alone is not
the fix and Metropolized MCLMC (MAMS) is needed.
"""
import os, sys, time, json
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from curved_testbed import build_target
from sa_move import make_sa_composite
def pr(*a): print(*a, flush=True)

N, K, BW = 64, 20, 0.20
R = int(os.environ.get("R", "1500"))
CONFIGS = [(1.5, 0.2), (1.5, 0.1), (1.5, 0.05), (1.5, 0.025),
           (3.0, 0.05), (3.0, 0.025)]

def cfrac(t, p): return (t.classify(np.asarray(p)) == 0).mean(axis=-1)

def main():
    t0 = time.time(); rows = []
    for b, step in CONFIGS:
        t = build_target(b=b, n_curve=4); D = t.D
        imm = jnp.asarray(t.within_mode_cov(mode=1))
        init, lbl = t.exact_draws_balanced(N, np.random.default_rng(7))   # truth w_A=0.60
        sa = make_sa_composite(t.logp, D, N, L=2.0, step_size=step, K=K, n_sa=N,
                               proposal="mixture", bandwidth=BW, kernel_cov=jnp.eye(D),
                               inverse_mass_matrix=imm)
        st = sa["init_states"](jnp.asarray(init), jax.random.key(1))
        keys = jax.random.split(jax.random.key(2), R)
        fa = np.empty(R); pos_last = None
        for r in range(R):
            st, (p, ec, sub) = sa["round"](st, keys[r]); fa[r] = cfrac(t, p)
            if r >= R - 200: pos_last = np.asarray(p) if pos_last is None else pos_last
        h = R // 2
        w1, w2 = float(fa[:h].mean()), float(fa[h:].mean())
        # ridge dim0 std + per-mode dim1 mean from the last block (unwarp to base)
        Xun = pos_last - t._cmask_np[None, :] * (t.b * (pos_last[:, 0:1] ** 2 - t._s2))
        lab = t.classify(pos_last)
        d0 = float(pos_last[:, 0].std())
        d1A = float(Xun[lab == 0][:, 1].mean()) if (lab == 0).any() else float("nan")
        row = dict(b=b, step=step, w1=w1, w2=w2, drift=abs(w2 - w1), d0_std=d0, dim1A=d1A)
        rows.append(row)
        pr(f"b={b:3.1f} step={step:.3f} | w_A 1st={w1:.3f} 2nd={w2:.3f} drift={abs(w2-w1):.3f} "
           f"(truth .60) | dim0 std={d0:.2f}(truth 3.0) | dim1A={d1A:+.2f}(truth +4.0)")
        with open(os.path.join(HERE, "step_robust.json"), "w") as f:
            json.dump(rows, f, indent=2, default=float)
    pr(f"\ntotal wall {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
