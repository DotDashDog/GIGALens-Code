"""DIAGNOSTIC (not a fix): why does the Gaussian testbed NOT reproduce the carousel's
~0.6% DE acceptance? Two candidate causes:
  (A) the Gaussian fit removed the carousel's curved / non-Gaussian within-mode ridges;
  (B) the prompt-mandated weight change (real secondary mass ~1e-5 -> 0.30) made jumps
      INTO the secondary mode ~30000x more acceptable.

This script isolates (B): sweep the secondary weight w_sec in {0.30, 1e-2, 1e-3, 1e-5}
keeping the SAME (Gaussian) geometry, MCLMC knobs, MM, jitter, p_jump, init. If DE
acceptance collapses toward ~0.6% as w_sec -> 1e-5, the testbed-vs-carousel gap is
LARGELY the weight, and the carousel's low acceptance is (partly) CORRECT behaviour:
the sampler rightly rejects jumps into a near-empty mode. "Fixing" acceptance upward
would then BIAS the run by over-populating the secondary mode.

Pre-registration:
  hypothesis: DE acceptance is dominated by the density at the proposal landing site;
    lowering w_sec lowers logp at the secondary mode, so jumps global->secondary are
    rejected and round-trips vanish.
  prediction: acceptance falls monotonically with w_sec; at w_sec=1e-5 it is O(0.1-1%)
    (same order as the carousel's 0.64%), and round-trips -> 0 (as in the carousel).
  falsifier: acceptance stays high (>5%) and round-trips persist even at w_sec=1e-5
    -> weight is NOT the explanation; curvature (A) is implicated.
CPU, synchronous. PROPOSED / UNCERTIFIED.
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carousel_testbed as T
from carousel_testbed import Mixture
from de_mclmc import make_composite

NCH = 16; K = 20; SEED = 20260627; R = 2000


def pr(*a): print(*a, flush=True)


mu, cov, sec, glob = T.fit_clusters()
Cg = jnp.asarray(cov[1]); Lg = jnp.linalg.cholesky(Cg)


def round_trips(modes):
    nrt = 0
    for c in range(modes.shape[1]):
        d = np.diff(modes[:, c]); nrt += min(int(np.sum(d == 1)), int(np.sum(d == -1)))
    return nrt


def init_balanced(mix, seed):
    rng = np.random.default_rng(seed)
    a = mu[0] + rng.standard_normal((NCH // 2, T.D)) @ np.asarray(mix._chol_np[0]).T
    b = mu[1] + rng.standard_normal((NCH // 2, T.D)) @ np.asarray(mix._chol_np[1]).T
    out = np.empty((NCH, T.D)); out[0::2] = a; out[1::2] = b
    return out


def run(w_sec, seed=SEED):
    w = np.array([w_sec, 1.0 - w_sec])
    mix = Mixture(mu, cov, w)
    comp = make_composite(mix.logp, T.D, NCH, L=T.L0, step_size=T.SS0, K=K,
                          b0=0.1, p_jump=0.3, inverse_mass_matrix=Cg, eps_scale=Lg)
    st = comp["init_states"](jnp.asarray(init_balanced(mix, seed)), jax.random.key(seed))
    keys = jax.random.split(jax.random.key(seed + 7), R)
    pos = np.empty((R, NCH, T.D)); acc = np.empty((R, NCH))
    s = st
    for r in range(R):
        s, (p, ec, a) = comp["round"](s, keys[r]); pos[r] = np.asarray(p); acc[r] = np.asarray(a)
    modes = mix.classify(pos)
    return float(acc.mean()), round_trips(modes), float(modes[R // 2:].mean())


def main():
    t = time.time()
    ws = [0.30, 1e-2, 1e-3, 1e-5]
    res = []
    pr("w_sec      DE_acc%   round-trips   w(glob) post-burn")
    for w_sec in ws:
        a, rt, wg = run(w_sec)
        res.append((w_sec, a, rt, wg))
        pr(f"{w_sec:.0e}    {a*100:7.3f}    {rt:6d}      {wg:.3f}")
    pr(f"\n(carousel anchor: DE acc 0.637%, round-trips 0, at real w_sec~1e-5)")
    res = np.array(res)
    np.savez(os.path.join(HERE, "E_weight_diag.npz"), res=res,
             carousel_acc=0.00637, carousel_rt=0)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].loglog(res[:, 0], res[:, 1] * 100, "o-", label="testbed (Gaussian)")
    ax[0].axhline(0.637, color="r", ls="--", label="carousel 0.637%")
    ax[0].set_xlabel("secondary weight w_sec"); ax[0].set_ylabel("DE acceptance %")
    ax[0].set_title("DE acceptance vs secondary weight"); ax[0].legend(fontsize=8); ax[0].invert_xaxis()
    ax[1].semilogx(res[:, 0], res[:, 2], "s-", color="C2")
    ax[1].axhline(0, color="r", ls="--", label="carousel round-trips=0")
    ax[1].set_xlabel("secondary weight w_sec"); ax[1].set_ylabel("round-trips (16 chains)")
    ax[1].set_title("mode-mixing vs secondary weight"); ax[1].legend(fontsize=8); ax[1].invert_xaxis()
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "E_weight_diag.png"), dpi=110); plt.close(fig)
    pr(f"\nDONE {time.time()-t:.1f}s -> E_weight_diag.png / .npz")


if __name__ == "__main__":
    main()
