"""Does the NORMAL energy-variance (EEVPD) tuning pick a step in the UNBIASED regime
on the curved testbed?  (Catch: the step_robust / attrib steps were hand-set, NOT
EEVPD-tuned. The 'biased step 0.2' is only production-relevant if the tuning would
actually choose ~0.2.)

This project's MCLMC target (fixed_knob_mclmc.py): per-step EEVPD = E[ec^2]/D with
DESIRED_EVAR = 5e-4, i.e. the adaptation drives step so that mean(energy_change^2)/D ~ 5e-4.
Here: measure EEVPD vs step on the curved testbed, locate the EEVPD-tuned step, and
compare it to the unbiased-weight threshold (step<=0.1 @ b=1.5) and the dim0 recovery.
"""
import os, sys, time, json
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from curved_testbed import build_target
from sa_move import make_sa_composite
def pr(*a): print(*a, flush=True)

DESIRED_EVAR = 5e-4           # project target (fixed_knob_mclmc.DESIRED_EVAR)
N, K = 64, 20
R = int(os.environ.get("R", "400"))
STEPS = [0.20, 0.10, 0.05, 0.025, 0.0125]

def cfrac(t, p): return (t.classify(np.asarray(p)) == 0).mean(axis=-1)

def main():
    t0 = time.time(); rows = []
    for b in [1.5, 3.0]:
        t = build_target(b=b, n_curve=4); D = t.D
        imm = jnp.asarray(t.within_mode_cov(mode=1))
        init, lbl = t.exact_draws_balanced(N, np.random.default_rng(7))   # truth init
        pr(f"\n=== b={b} (D={D}, DESIRED_EVAR={DESIRED_EVAR}, target mean(ec^2)/D) ===")
        for step in STEPS:
            sa = make_sa_composite(t.logp, D, N, L=2.0, step_size=step, K=K, n_sa=N,
                                   proposal="mixture", bandwidth=0.20, kernel_cov=jnp.eye(D),
                                   inverse_mass_matrix=imm)
            st = sa["init_states"](jnp.asarray(init), jax.random.key(1))
            keys = jax.random.split(jax.random.key(2), R)
            ecs = []; fa = np.empty(R)
            for r in range(R):
                st, (p, ec, sub) = sa["round"](st, keys[r])
                ecs.append(np.asarray(ec).ravel()); fa[r] = cfrac(t, p)
            ec = np.concatenate(ecs[R//4:])              # drop transient
            eevpd = float(np.mean(ec**2) / D)            # the tuned quantity
            xi = eevpd / DESIRED_EVAR                     # >1 => step too big (tuning shrinks it)
            w2 = float(fa[R//2:].mean())
            row = dict(b=b, step=step, eevpd=eevpd, xi=xi, w_A=w2)
            rows.append(row)
            pr(f"  step={step:.4f} | EEVPD=mean(ec^2)/D={eevpd:.2e}  xi=EEVPD/target={xi:6.2f} "
               f"({'TOO BIG' if xi>1.5 else ('tuned~here' if xi>0.5 else 'finer than needed')}) | w_A={w2:.3f}")
        with open(os.path.join(HERE, "evar_tune_check.json"), "w") as f:
            json.dump(rows, f, indent=2, default=float)
    pr(f"\ntotal wall {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
