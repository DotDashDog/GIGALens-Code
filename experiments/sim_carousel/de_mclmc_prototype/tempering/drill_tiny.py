"""DRILL (protocol rule 1): the one-shot tempered-burn-in tiny-mode DRAIN is
freeze-out-limited (w=1e-3 froze at 0.4375 hot-end occupancy in tiny_drain.py).
Characterize: multi-seed + slower/denser cooling through the decoupling region.
If the drain is seed-dependent / needs heroic schedules, the robust tool is PT.

PREDICTION: one-shot occ_minor for w=1e-3 is high-variance across seeds and only
drains with a slow dense cool through beta in [0.05,0.6]; expected (correct)
occupancy is 16*1e-3=0.016 chains ~ 0. FALSIFIER: occ_minor ~ 0 robustly across
seeds with the default schedule (then no freeze-out drain problem).
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tempered_mclmc import make_tempered_sampler

D, m = 10, 5.0
N_CHAINS, L, STEP = 16, 2.0, 0.5
WM = 1e-3
SEED = 20260628

def make_logp(w_minor):
    W = np.array([1.0 - w_minor, w_minor]); MU = np.array([+m, -m])
    _logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU); _c = -0.5*D*jnp.log(2*jnp.pi)
    def logp(z):
        z0 = z[0]; qr = jnp.sum(z[1:]**2)
        c0 = _logW[0]+_c-0.5*((z0-_MU[0])**2+qr); c1 = _logW[1]+_c-0.5*((z0-_MU[1])**2+qr)
        return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))
    return logp

SCHEDULES = {
    "default(18g,300)": (np.geomspace(0.03, 1.0, 18), 300),
    "dense-decouple":   (np.concatenate([np.geomspace(0.03, 0.05, 4),
                                         np.linspace(0.06, 0.6, 30),
                                         np.linspace(0.62, 1.0, 8)]), 600),
}
N_SEEDS = 6

def main():
    t0 = time.time()
    logp = make_logp(WM)
    res = {}
    for sname, (betas, spb) in SCHEDULES.items():
        samp = make_tempered_sampler(logp, D, N_CHAINS, betas, L=L, step_size=STEP)
        occs = []
        for s in range(N_SEEDS):
            z = np.random.default_rng(SEED+s).standard_normal((N_CHAINS, D)); z[:, 0] += -m
            ka, kc = jax.random.split(jax.random.key(SEED+s))
            fp, tr = samp["anneal"](jnp.asarray(z), ka, spb)
            _, cp, _ = samp["sample_cold"](fp, kc, 600)
            cp = np.asarray(cp)
            occs.append(float((cp[:, :, 0] < 0).mean()))
        occs = np.asarray(occs)
        res[sname] = occs
        print(f"{sname:18s} (nbeta={len(betas)},spb={spb}): occ_minor per seed = "
              f"{np.array2string(occs, precision=4)}  mean={occs.mean():.4f}", flush=True)

    pin = 1.0/N_CHAINS
    print(f"\n   truth w_minor={WM} (16*w={16*WM:.3f} chains); pin 1/16={pin:.4f}")
    np.savez(os.path.join(HERE, "drill_tiny.npz"),
             **{f"occ_{k}": res[k] for k in res}, wm=WM, pin=pin)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, k in enumerate(res):
        ax.scatter([i]*N_SEEDS, res[k], s=40, alpha=0.7, label=k)
    ax.axhline(pin, color="r", ls=":", label="ensemble-hop pin 0.0625")
    ax.axhline(WM, color="k", ls="--", label=f"truth {WM}")
    ax.set_xticks(range(len(res))); ax.set_xticklabels(list(res.keys()), fontsize=8)
    ax.set_ylabel("one-shot cold occ_minor"); ax.legend(fontsize=8)
    ax.set_title("One-shot tempered-burn-in tiny-drain is freeze-out / seed dependent")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "drill_tiny.png"), dpi=110); plt.close(fig)
    print(f"   wall {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
