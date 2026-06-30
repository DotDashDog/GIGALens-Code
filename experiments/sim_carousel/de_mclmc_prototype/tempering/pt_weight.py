"""GATE A via PARALLEL TEMPERING (the escalation justified by measured freeze-out
in drill_schedule.py / drill_tiny.py): a continuously-swapping cold (beta=1)
replica recovers the comparable-mass weight 0.70 WITHOUT freeze-out bias and with
a genuinely-mixing (non-frozen) cold series.

EASY mixture (D=10, modes +/-5, weights 0.7/0.3, barrier 12.5). All replicas of
all systems init in the WRONG (-mode, 0.30) basin -> tests discovery + unbiased
weight. PT cold-replica time+system-averaged occ_+ must reach 0.70.

PRE-REGISTRATION
PREDICTION: cold occ_+ -> 0.70 (from 0.0). THRESHOLD |occ_+ - 0.70| < 3*SE with
SE a block-bootstrap SE over the per-round cold occupancy series (valid now: the
cold replica MIXES via swaps, so the series varies). No drift: |w2nd-w1st|<3SE.
Swap acceptance per adjacent pair reported (healthy 0.2-0.6). CONTRAST: vanilla
MCLMC stays trapped (occ_+ ~ 0). FALSIFIER: occ_+ frozen away from 0.70, or swap
acceptance ~0 (ladder disconnected -> no cold mixing).
EEVPD note: benign-Gaussian, step=0.5 conservative (EEVPD 1.5e-9, see tune_easy).
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from parallel_tempering import make_pt_sampler
from validate_analytic import integrated_autocorr_time, block_bootstrap_se

D, m = 10, 5.0
W = np.array([0.7, 0.3]); MU = np.array([+m, -m])
_logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU); _c = -0.5*D*jnp.log(2*jnp.pi)
def logdensity_fn(z):
    z0 = z[0]; qr = jnp.sum(z[1:]**2)
    c0 = _logW[0]+_c-0.5*((z0-_MU[0])**2+qr); c1 = _logW[1]+_c-0.5*((z0-_MU[1])**2+qr)
    return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))

N_SYS, L, STEP, K = 16, 2.0, 0.5, 20
BETAS = np.geomspace(0.03, 1.0, 10)
N_ROUNDS = 3000
BURN = 600
SEED = 20260628

def main():
    t0 = time.time()
    pt = make_pt_sampler(logdensity_fn, D, N_SYS, BETAS, L=L, step_size=STEP, K=K)
    print("PT config:", pt["config"], "R=", pt["R"], flush=True)

    # init ALL replicas of ALL systems in the WRONG (-mode) basin
    rng = np.random.default_rng(SEED)
    init = rng.standard_normal((pt["R"], N_SYS, D)); init[:, :, 0] += MU[1]
    out = pt["run"](jnp.asarray(init), jax.random.key(SEED+1), N_ROUNDS, seed_swap=SEED+2)
    cold = out["cold"]                      # (n_rounds, n_sys, D)
    print(f"swap acceptance per pair: {np.array2string(out['swap_acc'], precision=3)}", flush=True)
    print(f"mean EEVPD per level: {np.array2string(out['eevpd_mean'], precision=2)}", flush=True)

    frac = (cold[:, :, 0] > 0).mean(axis=1)     # per-round cold occ_+
    fk = frac[BURN:]
    occ = float(fk.mean())
    se, blk = block_bootstrap_se(fk, rng=np.random.default_rng(1))
    half = len(fk)//2
    w1, w2 = float(fk[:half].mean()), float(fk[half:].mean())
    se1, _ = block_bootstrap_se(fk[:half], rng=np.random.default_rng(2))
    se2, _ = block_bootstrap_se(fk[half:], rng=np.random.default_rng(3))
    se_d = np.hypot(se1, se2)
    print(f"\nPT cold occ_+ = {occ:.4f} +/- {se:.4f} (block={blk})", flush=True)
    print(f"   |occ_+ - 0.70| = {abs(occ-0.70):.4f}  (3SE={3*se:.4f})")
    print(f"   drift w1={w1:.4f} w2={w2:.4f} |diff|={abs(w2-w1):.4f} (3SE_d={3*se_d:.4f})")

    # within-mode moments (cold)
    thin = cold[BURN::3].reshape(-1, D)
    plus = thin[thin[:, 0] > 0]; minus = thin[thin[:, 0] < 0]
    print(f"   +mode mean {plus[:,0].mean():.3f} var {plus[:,0].var():.3f}; "
          f"-mode mean {minus[:,0].mean():.3f} var {minus[:,0].var():.3f}; "
          f"axis1 var {thin[:,1].var():.3f}")

    np.savez(os.path.join(HERE, "pt_weight.npz"),
             betas=BETAS, swap_acc=out["swap_acc"], eevpd=out["eevpd_mean"],
             frac=frac, occ=occ, se=se, w1=w1, w2=w2, se_d=se_d, burn=BURN)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frac, lw=0.5); axes[0].axhline(0.70, color="k", ls="--", label="truth 0.70")
    axes[0].axhline(occ, color="C1", label=f"post-burn mean {occ:.3f}")
    axes[0].axvline(BURN, color="r", ls=":", label="burn")
    axes[0].set_xlabel("PT round"); axes[0].set_ylabel("cold occ_+")
    axes[0].set_title("Gate A (PT): cold replica discovers + recovers weight, MIXES")
    axes[0].legend(fontsize=8)
    axes[1].bar(range(len(out["swap_acc"])), out["swap_acc"], color="C0")
    axes[1].axhspan(0.2, 0.6, color="green", alpha=0.15, label="healthy 0.2-0.6")
    axes[1].set_xlabel("adjacent pair index (hot->cold)"); axes[1].set_ylabel("swap acceptance")
    axes[1].set_title("PT swap acceptance per rung"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "A_pt_weight.png"), dpi=110); plt.close(fig)

    print("\n========== PRE-REGISTERED (PROPOSED/UNCERTIFIED) ==========")
    print(f" A weight within 3SE  |{occ-0.70:.4f}| < {3*se:.4f} -> {'PASS' if abs(occ-0.70)<3*se else 'FAIL'}")
    print(f" A no drift           |{abs(w2-w1):.4f}| < {3*se_d:.4f} -> {'PASS' if abs(w2-w1)<3*se_d else 'FAIL'}")
    print(f" ladder connected     min swap acc {out['swap_acc'].min():.3f} > 0.05 -> {'PASS' if out['swap_acc'].min()>0.05 else 'FAIL'}")
    print(f" wall {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
