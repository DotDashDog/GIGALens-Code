"""GATE B (T3): tiny-mode occupancy. The user's explicit worry -- a config must
NOT over-represent a tiny mode (pin a chain at 1/n_chains) NOR drain it; the
TIME-AVERAGED minor-mode occupancy must match the true weight within MC error.

Uses the EASY analytic mixture geometry (modes +/-5 on axis0) but with the minor
mode weight set to w_minor in {0.03, 0.001}; 16 chains.  Both modes are POPULATED
at init (the realistic regime for an equilibration move; cold discovery of a
0.001 mode is a separate capability the mixture lacks, as documented).

Pre-registered: time-averaged frac(minor) within 3 SE (block bootstrap) of
w_minor.  REJECT if it pins at 1/16=0.0625 (over-rep) or drains to 0.
Run:  python tiny_mode_test.py <w_minor> [bandwidth]
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from sa_move import make_sa_composite
from validate_analytic import integrated_autocorr_time, block_bootstrap_se

W_MINOR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
PROP = sys.argv[2] if len(sys.argv) > 2 else "mixture"   # mixture|gaussian
BW = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0     # bandwidth (mix) or prop_scale (gauss)
D, m = 10, 5.0
N_CHAINS, L, STEP, K, N_SA = 16, 2.0, 0.5, 20, 16
R = int(os.environ.get("R","6000"))
SEED = 20260628

W = np.array([1.0 - W_MINOR, W_MINOR]); MU = np.array([+m, -m])  # minor = -mode
_logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU); _c = -0.5 * D * jnp.log(2 * jnp.pi)

def logdensity_fn(z):
    z0 = z[0]; qr = jnp.sum(z[1:] ** 2)
    c0 = _logW[0] + _c - 0.5 * ((z0 - _MU[0]) ** 2 + qr)
    c1 = _logW[1] + _c - 0.5 * ((z0 - _MU[1]) ** 2 + qr)
    return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))

def main():
    t0 = time.time()
    kw = dict(bandwidth=BW, kernel_cov=jnp.eye(D)) if PROP == "mixture" else dict(prop_scale=BW)
    comp = make_sa_composite(logdensity_fn, D, N_CHAINS, L=L, step_size=STEP, K=K,
                             n_sa=N_SA, proposal=PROP, **kw)
    # init: at least one chain in the minor mode so it is POPULATED (we test
    # occupancy maintenance, not cold discovery). Put round(w*N) but >=1 in minor.
    n_minor = max(1, int(round(W_MINOR * N_CHAINS)))
    comp_lbl = np.zeros(N_CHAINS, int); comp_lbl[:n_minor] = 1
    z = np.random.default_rng(SEED).standard_normal((N_CHAINS, D)); z[:, 0] += MU[comp_lbl]
    st = comp["init_states"](jnp.asarray(z), jax.random.key(SEED))
    keys = jax.random.split(jax.random.key(SEED + 1), R)
    frac_minor = np.empty(R)
    for r in range(R):
        st, (p, ec, sub) = comp["round"](st, keys[r])
        frac_minor[r] = (np.asarray(p)[:, 0] < 0).mean()
    burn = R // 5
    fm = frac_minor[burn:]
    est = float(fm.mean()); se, blk = block_bootstrap_se(fm, rng=np.random.default_rng(1))
    pin = 1.0 / N_CHAINS
    print(f"w_minor={W_MINOR}  bw={BW}  est_occupancy={est:.5f} +/- {se:.5f} (block={blk})")
    print(f"   truth={W_MINOR}  |est-truth|={abs(est-W_MINOR):.5f} (3SE={3*se:.5f})")
    print(f"   pin level 1/16={pin:.4f}; drained=0.  est is {'NEAR TRUTH' if abs(est-W_MINOR)<3*se else 'OFF'}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frac_minor, lw=0.5); ax.axhline(W_MINOR, color="k", ls="--", label=f"truth {W_MINOR}")
    ax.axhline(est, color="C1", label=f"time-avg {est:.4f}")
    ax.axhline(pin, color="r", ls=":", label=f"pin 1/16={pin:.3f}")
    ax.set_title(f"T3 tiny-mode occupancy w_minor={W_MINOR} bw={BW}"); ax.set_xlabel("round"); ax.set_ylabel("minor-mode frac"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"T3_{PROP}_w{W_MINOR}_s{BW}.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(HERE, f"T3_{PROP}_w{W_MINOR}_s{BW}.npz"),
             frac_minor=frac_minor, est=est, se=se, w_minor=W_MINOR, pin=pin, bw=BW)
    verdict = "PASS (near truth, no pin/drain)" if abs(est - W_MINOR) < 3 * se else \
              ("PIN" if est > 1.5 * pin else ("DRAIN" if est < 0.3 * W_MINOR else "OFF"))
    print(f"   PROPOSED: {verdict}  | wall {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
