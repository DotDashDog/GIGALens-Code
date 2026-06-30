"""GATE B / T3: tiny-mode occupancy (the user's explicit worry -- "draining a
significant mode or leaving one of 16 chains pinned in a tiny mode that doesn't
deserve it").

Run on the SEPARATED (non-curved) analytic mixture ON PURPOSE: there, cross-mode
jumps DO succeed, so any failure to represent the tiny mode's weight is a
DRAIN/PIN pathology of the MOVE, cleanly isolated from the curvature wall (curvature
is tested separately in curved_gates). D=10, modes +-5 axis 0; TRUE minor weight
w_tiny in {0.03, 0.001}; 16 chains.

Operational test: seed EXACTLY 1 chain in the minor mode (occupancy 1/16=0.0625 >
both truths) and run. The TIME-AVERAGED minor occupancy must relax to w_tiny within
MC error. Pinned (stuck ~0.0625) or drained (->0) => REJECTED however fast.

PRE-REGISTRATION (cause/prediction/falsifier):
  Cause: a difference-vector / teleport move can only propose INTO a mode that some
  complement chain currently occupies. Once the lone minor chain leaves, the minor
  mode is unpopulated and no chain can be proposed back in -> DRAIN. The KDE 'near'
  move additionally suppresses LEAVING a lone-occupied mode (reverse density tiny)
  -> PIN.
  Prediction: gamma1 -> occupancy DRAINS below w_tiny toward 0 (single seeded chain
  leaves, mode then unreachable). near -> occupancy PINS near 0.0625.
  Falsifier: time-averaged occupancy within 3*SE of w_tiny (neither drains nor pins).
Plots before metrics. Verdicts PROPOSED / UNCERTIFIED. Usage: tiny_mode_T3.py [Rscale]
"""
import os, sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from de_teleport import make_teleport_composite
from validate_analytic import integrated_autocorr_time, block_bootstrap_se

D = 10; m = 5.0; NCH = 16; L, STEP, K = 2.0, 0.5, 20; SEED = 20260627
RS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
R = int(3000 * RS); BURN = R // 3


def make_logp(w_tiny):
    W = jnp.asarray([1.0 - w_tiny, w_tiny]); logW = jnp.log(W); MU = jnp.asarray([+m, -m])
    c = -0.5 * D * jnp.log(2 * jnp.pi)
    def logp(z):
        z0 = z[0]; qr = jnp.sum(z[1:] ** 2)
        c0 = logW[0] + c - 0.5 * ((z0 - MU[0]) ** 2 + qr)
        c1 = logW[1] + c - 0.5 * ((z0 - MU[1]) ** 2 + qr)
        return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))
    return logp


def seeded_init(seed):
    """15 chains in major (+m), 1 chain in minor (-m)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((NCH, D)); z[:, 0] += m       # all major
    z[-1] = rng.standard_normal(D); z[-1, 0] += -m         # one minor
    return z


def run(comp, init, seed):
    st = comp["init_states"](jnp.asarray(init), jax.random.key(seed))
    keys = jax.random.split(jax.random.key(seed + 7), R)
    occ = np.empty(R)                                       # frac chains in minor (z0<0)
    for r in range(R):
        st, (p, ec, a) = comp["round"](st, keys[r])
        occ[r] = float((np.asarray(p)[:, 0] < 0).mean())
    return occ


def main():
    t0 = time.time(); boot = np.random.default_rng(SEED); summary = {}
    moves = ["gamma1", "near"]
    w_tinys = [0.03, 0.001]
    fig, axes = plt.subplots(len(w_tinys), 1, figsize=(11, 4.0 * len(w_tinys)))
    for wi, w_tiny in enumerate(w_tinys):
        logp = make_logp(w_tiny)
        print(f"\n##### w_tiny = {w_tiny} (true occupancy {w_tiny}; 1/16={1/16:.4f}) #####", flush=True)
        for move in moves:
            comp = make_teleport_composite(logp, D, NCH, L=L, step_size=STEP, K=K,
                                           move=move, b0=0.05, p_jump=0.5, eps=0.05,
                                           eps_scale=None)
            occ = run(comp, seeded_init(SEED + wi), SEED + 1)
            occ_avg = float(occ[BURN:].mean())
            se, _ = block_bootstrap_se(occ[BURN:], rng=boot)
            pinned = occ_avg > 0.045                         # ~ stuck near 1/16=0.0625
            drained = occ_avg < 0.2 * w_tiny                 # collapsed well below truth
            within = abs(occ_avg - w_tiny) < max(3 * se, 0.005)
            verdict = ("PIN" if pinned else "DRAIN" if drained else "OK" if within else "OFF")
            print(f"  {move:8s} time-avg minor occ = {occ_avg:.4f} +/- {se:.4f}  "
                  f"(truth {w_tiny}) -> {verdict}", flush=True)
            summary[f"{move}_{w_tiny}"] = dict(occ=occ_avg, se=se, truth=w_tiny,
                pinned=bool(pinned), drained=bool(drained), within=bool(within), verdict=verdict)
            axes[wi].plot(occ, lw=0.5, label=f"{move} (avg {occ_avg:.4f}, {verdict})")
        axes[wi].axhline(w_tiny, color='k', ls='--', label=f"truth {w_tiny}")
        axes[wi].axhline(1/16, color='r', ls=':', label="pin 1/16=0.0625")
        axes[wi].set_xlabel("round"); axes[wi].set_ylabel("frac chains in minor")
        axes[wi].set_title(f"T3 tiny-mode occupancy, w_tiny={w_tiny}"); axes[wi].legend(fontsize=8)
        axes[wi].set_ylim(-0.005, 0.10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "tiny_mode_T3.png"), dpi=110); plt.close(fig)
    with open(os.path.join(HERE, "tiny_mode_T3.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsaved tiny_mode_T3.png/.json  total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
