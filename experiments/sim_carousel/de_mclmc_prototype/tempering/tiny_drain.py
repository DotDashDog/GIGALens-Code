"""GATE B (tiny-mode DRAIN) for TEMPERED BURN-IN -- the key differentiator from
every ensemble mode-hop (C-16: linear-DE/snooker/kernel-hop/SA-MCMC ALL PIN a
lone tiny-mode chain at ~1/n_chains = 0.0625, over-representing a ~1e-5 mode by
60-6000x). Tempering DRAINS it instead.

Setup: EASY mixture geometry (modes +/-5 axis0, barrier 12.5) with the MINOR
(-mode) true weight set to w_minor in {1e-3, 1e-5}; 16 chains; ALL chains SEEDED
in the tiny (-mode) basin. Tempered burn-in (anneal beta small->1), then cold
sample at beta=1; measure the TIME-AVERAGED cold minor-mode occupancy.

PRE-REGISTRATION
CAUSE: at small beta the tiny mode is INFLATED (w_minor^beta/(w_minor^beta+
(1-w_minor)^beta), e.g. w=1e-5,beta=0.04 -> ~0.39), so chains can leave it across
the (then-low) barrier; as beta->1 the tiny mode's tempered weight COLLAPSES to
~0 and the dominant basin captures the chains -> cold minor occupancy ~ 0.
PREDICTION: cold occ_minor ~ 0 (n_chains * w_minor << 1, so the unbiased # of
chains in the tiny mode is ~0). It must NOT pin at 1/16 = 0.0625.
THRESHOLD (derived): the correct expected occupancy is w_minor (=1e-3 or 1e-5);
with 16 chains the binomial expectation is 16*w_minor << 1 chain, so the unbiased
outcome is 0 chains -> occ_minor consistent with 0. DRAIN PASS if
occ_minor < 0.5/16 = 0.03125 (less than half a chain on average) AND
occ_minor << pin 0.0625. CONTRAST: ensemble hops give ~0.0625 (documented C-16).
FALSIFIER: occ_minor pins near 0.0625 (tempering failed to drain) OR stays high.
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
BETAS = np.geomspace(0.03, 1.0, 18)
STEPS_PER_BETA = 300
COLD_STEPS = 1500
SEED = 20260628
W_MINORS = [1e-3, 1e-5]

def make_logp(w_minor):
    W = np.array([1.0 - w_minor, w_minor]); MU = np.array([+m, -m])
    _logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU); _c = -0.5*D*jnp.log(2*jnp.pi)
    def logp(z):
        z0 = z[0]; qr = jnp.sum(z[1:]**2)
        c0 = _logW[0]+_c-0.5*((z0-_MU[0])**2+qr); c1 = _logW[1]+_c-0.5*((z0-_MU[1])**2+qr)
        return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))
    return logp, W

def tempered_minor(beta, w_minor):
    a = (1-w_minor)**beta; b = w_minor**beta; return b/(a+b)

def main():
    t0 = time.time()
    out = {}
    for wm in W_MINORS:
        logp, W = make_logp(wm)
        samp = make_tempered_sampler(logp, D, N_CHAINS, BETAS, L=L, step_size=STEP)
        # seed ALL chains in tiny (-mode) basin
        z = np.random.default_rng(SEED).standard_normal((N_CHAINS, D)); z[:, 0] += MU_minor()
        key = jax.random.key(SEED + hash(str(wm)) % 1000)
        ka, kc = jax.random.split(key)
        final_pos, trace = samp["anneal"](jnp.asarray(z), ka, STEPS_PER_BETA)
        _, cold_pos, _ = samp["sample_cold"](final_pos, kc, COLD_STEPS)
        cold_pos = np.asarray(cold_pos)              # (T,n,D)
        frac_minor_round = (cold_pos[:, :, 0] < 0).mean(axis=1)
        occ_minor = float(frac_minor_round.mean())
        stage_minor = (trace["stage_pos"][:, :, 0] < 0).mean(axis=1)
        pin = 1.0/N_CHAINS
        drained = occ_minor < 0.5/N_CHAINS
        print(f"w_minor={wm:.0e}: cold occ_minor = {occ_minor:.5f} "
              f"(truth {wm:.0e}, pin {pin:.4f}, half-chain {0.5/N_CHAINS:.4f}) "
              f"-> {'DRAINED' if drained else 'NOT DRAINED'}", flush=True)
        out[wm] = dict(occ_minor=occ_minor, frac_round=frac_minor_round,
                       stage_minor=stage_minor, pin=pin)

    np.savez(os.path.join(HERE, "tiny_drain.npz"),
             betas=BETAS, w_minors=np.array(W_MINORS),
             **{f"occ_{wm:.0e}": out[wm]["occ_minor"] for wm in W_MINORS},
             **{f"fracround_{wm:.0e}": out[wm]["frac_round"] for wm in W_MINORS},
             **{f"stage_{wm:.0e}": out[wm]["stage_minor"] for wm in W_MINORS})

    # PLOTS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # anneal: minor frac vs beta, with tempered-minor-weight overlay
    for wm in W_MINORS:
        axes[0].plot(BETAS, out[wm]["stage_minor"], "o-", ms=3, label=f"w={wm:.0e} ensemble")
        axes[0].plot(BETAS, [tempered_minor(b, wm) for b in BETAS], "--", alpha=0.5,
                     label=f"w={wm:.0e} tempered weight")
    axes[0].set_xscale("log"); axes[0].set_xlabel("beta"); axes[0].set_ylabel("minor-mode frac")
    axes[0].set_title("Anneal: tiny mode inflated when hot, DRAINS as beta->1"); axes[0].legend(fontsize=7)
    # cold occupancy bars vs pin
    xs = [f"{wm:.0e}" for wm in W_MINORS]
    occs = [out[wm]["occ_minor"] for wm in W_MINORS]
    axes[1].bar(xs, occs, color="C0", label="tempered cold occ_minor")
    axes[1].axhline(1.0/N_CHAINS, color="r", ls=":", label="ensemble-hop PIN 1/16=0.0625")
    axes[1].axhline(0.5/N_CHAINS, color="k", ls="--", label="half-chain 0.03125")
    axes[1].set_ylabel("cold minor-mode occupancy"); axes[1].set_ylim(0, 0.08)
    axes[1].set_title("Gate B: tempering DRAINS (vs ensemble-hop pin)"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "B_tiny_drain.png"), dpi=110); plt.close(fig)
    print(f"wall {time.time()-t0:.1f}s")

def MU_minor():
    return -m

if __name__ == "__main__":
    main()
