"""GATE B (tiny-mode DRAIN) via PARALLEL TEMPERING -- the robust drain that the
one-shot anneal cannot deliver (drill_tiny.py: one-shot freezes occupancy at
integer chains/16, cannot represent 1e-3) and that EVERY ensemble mode-hop FAILS
(C-16: pin at 1/16 = 0.0625). PT's continuously-swapping cold replica visits the
tiny mode a fraction ~ w_minor of the time -> correct drained occupancy.

EASY mixture, MINOR (-mode) true weight w_minor in {1e-3, 1e-5}; ALL replicas of
all systems SEEDED in the tiny mode. Cold (beta=1) replica time+system-averaged
occ_minor must DRAIN toward w_minor, NOT pin at 0.0625.

PRE-REGISTRATION
PREDICTION: cold occ_minor -> ~w_minor (1e-3 resolvable with N_SYS*rounds cold
samples; 1e-5 -> ~0). THRESHOLD: occ_minor < 0.5/N_SYS = 0.0156 (drained, < half
the pin) AND for 1e-3 |occ_minor - 1e-3| within a few block-bootstrap SE.
CONTRAST: ensemble hops pin at 0.0625 (C-16). FALSIFIER: occ_minor ~ 0.0625
(pinned) -> PT failed to drain.
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
from validate_analytic import block_bootstrap_se

D, m = 10, 5.0
N_SYS, L, STEP, K = 24, 2.0, 0.5, 20
BETAS = np.geomspace(0.03, 1.0, 10)
N_ROUNDS = 4000
BURN = 800
SEED = 20260628
W_MINORS = [1e-3, 1e-5]

def make_logp(w_minor):
    W = np.array([1.0 - w_minor, w_minor]); MU = np.array([+m, -m])
    _logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU); _c = -0.5*D*jnp.log(2*jnp.pi)
    def logp(z):
        z0 = z[0]; qr = jnp.sum(z[1:]**2)
        c0 = _logW[0]+_c-0.5*((z0-_MU[0])**2+qr); c1 = _logW[1]+_c-0.5*((z0-_MU[1])**2+qr)
        return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))
    return logp

def main():
    t0 = time.time(); out = {}; pin = 1.0/N_SYS
    for wm in W_MINORS:
        pt = make_pt_sampler(make_logp(wm), D, N_SYS, BETAS, L=L, step_size=STEP, K=K)
        rng = np.random.default_rng(SEED)
        init = rng.standard_normal((pt["R"], N_SYS, D)); init[:, :, 0] += -m   # seed tiny mode
        r = pt["run"](jnp.asarray(init), jax.random.key(SEED+1), N_ROUNDS, seed_swap=SEED+3)
        cold = r["cold"]
        frac = (cold[:, :, 0] < 0).mean(axis=1)
        fk = frac[BURN:]; occ = float(fk.mean())
        se, blk = block_bootstrap_se(fk, rng=np.random.default_rng(1))
        drained = occ < 0.5/N_SYS
        out[wm] = dict(frac=frac, occ=occ, se=se, swap=r["swap_acc"])
        print(f"w_minor={wm:.0e}: cold occ_minor = {occ:.5f} +/- {se:.5f} "
              f"(truth {wm:.0e}, pin {pin:.4f}) -> {'DRAINED' if drained else 'PINNED/OFF'}",
              flush=True)
        print(f"   swap acc: {np.array2string(r['swap_acc'], precision=2)}", flush=True)

    np.savez(os.path.join(HERE, "pt_drain.npz"), betas=BETAS, w_minors=np.array(W_MINORS),
             pin=pin, **{f"frac_{wm:.0e}": out[wm]["frac"] for wm in W_MINORS},
             **{f"occ_{wm:.0e}": out[wm]["occ"] for wm in W_MINORS},
             **{f"se_{wm:.0e}": out[wm]["se"] for wm in W_MINORS})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for wm in W_MINORS:
        axes[0].plot(out[wm]["frac"], lw=0.4, label=f"w={wm:.0e}")
    axes[0].axhline(pin, color="r", ls=":", label=f"ensemble-hop pin {pin:.3f}")
    axes[0].axhline(1e-3, color="C2", ls="--", label="truth 1e-3")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[0].set_xlabel("PT round"); axes[0].set_ylabel("cold occ_minor")
    axes[0].set_title("Gate B (PT): cold replica DRAINS tiny mode (mixing)"); axes[0].legend(fontsize=8)
    xs = [f"{wm:.0e}" for wm in W_MINORS]; occs = [out[wm]["occ"] for wm in W_MINORS]
    ses = [3*out[wm]["se"] for wm in W_MINORS]
    axes[1].bar(xs, occs, yerr=ses, color="C0", capsize=4, label="PT cold occ_minor")
    axes[1].axhline(pin, color="r", ls=":", label=f"ensemble-hop PIN {pin:.4f}")
    axes[1].axhline(0.5/N_SYS, color="k", ls="--", label=f"half-chain {0.5/N_SYS:.4f}")
    axes[1].set_ylabel("cold minor occupancy"); axes[1].set_ylim(0, 0.07)
    axes[1].set_title("Tempering DRAINS where ensemble hops PIN"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "B_pt_drain.png"), dpi=110); plt.close(fig)
    print(f"wall {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
