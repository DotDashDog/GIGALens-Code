"""DRILL (protocol rule 1): at a FAITHFUL step (anneal EEVPD<=5e-4) the curved-
barrier discovery occ_A fell with step at FIXED 300 steps/stage (0.27->0.094).
Is this (a) BUDGET (a finer step covers less ground per stage -> need more steps,
the documented EEVPD-fine-step cost) or (b) STRUCTURAL (tempering cannot cross
the curved barrier faithfully)?

Hold the step FIXED & FAITHFUL (0.05, anneal EEVPD~1e-8) and scale steps/stage.
PREDICTION (budget): occ_A RISES and SATURATES toward the freeze-out weight
(~0.5-0.6) as steps/stage grows. FALSIFIER (structural): occ_A stays ~0 (near the
all-modeB init) regardless of budget -> tempering fails to cross faithfully.
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sa_mcmc"))
from tempered_mclmc import make_tempered_sampler, DESIRED_EVAR
from curved_testbed import CurvedBimodal

N_CHAINS, L = 64, 2.0
STEP = 0.05                      # faithful (anneal EEVPD ~1.4e-8 << 5e-4)
BETAS = np.geomspace(0.006, 1.0, 22)
COLD_STEPS = 600
SEED = 20260628
EFFORTS = [300, 900, 2700, 6000]

def main():
    t0 = time.time()
    tgt = CurvedBimodal(); D = tgt.D
    samp = make_tempered_sampler(tgt.logp, D, N_CHAINS, BETAS, L=L, step_size=STEP)
    rng = np.random.default_rng(SEED)
    yb, _ = tgt.exact_draws_balanced(N_CHAINS, rng)
    initB = yb[tgt.classify(yb) == 1]
    reps = int(np.ceil(N_CHAINS/max(1, len(initB)))); initB = np.tile(initB, (reps, 1))[:N_CHAINS]

    occA = []; amax = []
    for eff in EFFORTS:
        ka, kc = jax.random.split(jax.random.key(SEED + 1))
        fp, tr = samp["anneal"](jnp.asarray(initB), ka, eff)
        _, cp, _ = samp["sample_cold"](fp, kc, COLD_STEPS)
        cp = np.asarray(cp)
        lbl = tgt.classify(cp.reshape(-1, D)).reshape(cp.shape[:2])
        oA = float((lbl == 0).mean()); am = float(tr["stage_eevpd"].max())
        occA.append(oA); amax.append(am)
        print(f"   steps/stage={eff:5d}: occ_A={oA:.4f}  anneal_max_EEVPD={am:.2e} "
              f"({'faithful' if am<=DESIRED_EVAR else 'COARSE'})", flush=True)
    occA = np.array(occA)
    a = tgt.w[0]**1; b = tgt.w[1]**1
    print(f"\n   true w_A={tgt.w[0]}; vanilla=0 (trapped). occ_A trend: {np.array2string(occA, precision=3)}")

    np.savez(os.path.join(HERE, "drill_curved.npz"),
             efforts=np.array(EFFORTS), occA=occA, amax=np.array(amax),
             step=STEP, wA=tgt.w[0])
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.plot(EFFORTS, occA, "o-", label="tempered occ_A (faithful step 0.05)")
    ax.axhline(tgt.w[0], color="k", ls="--", label=f"true w_A={tgt.w[0]}")
    ax.axhline(0.0, color="C3", ls=":", label="vanilla (trapped)")
    ax.set_xscale("log"); ax.set_xlabel("steps per beta stage"); ax.set_ylabel("cold occ_A")
    ax.set_ylim(-0.02, 0.7)
    ax.set_title("Gate D drill: faithful-step curved discovery is BUDGET-limited (saturates>0)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "drill_curved.png"), dpi=110); plt.close(fig)
    print(f"   wall {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
