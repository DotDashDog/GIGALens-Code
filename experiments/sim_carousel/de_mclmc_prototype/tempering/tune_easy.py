"""Smoke test + EEVPD step tuning for the EASY analytic bimodal mixture
(D=10, modes +/-5 along axis0, weights [0.7,0.3], barrier m^2/2 = 12.5).

Tunes the MCLMC step at beta=1 so EEVPD = mean(energy_change^2)/D ~= 5e-4
(the project target), then MEASURES the realized EEVPD at every beta on the
cooling ladder with that fixed step to confirm it stays <= target (the
conservative-step argument). Pure tuning; cheap; one file-logged job.
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tempered_mclmc import tune_step_eevpd, measure_eevpd, DESIRED_EVAR

D, m = 10, 5.0
W = np.array([0.7, 0.3]); MU = np.array([+m, -m])
_logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU); _c = -0.5 * D * jnp.log(2 * jnp.pi)

def logdensity_fn(z):
    z0 = z[0]; qr = jnp.sum(z[1:] ** 2)
    c0 = _logW[0] + _c - 0.5 * ((z0 - _MU[0]) ** 2 + qr)
    c1 = _logW[1] + _c - 0.5 * ((z0 - _MU[1]) ** 2 + qr)
    return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))

def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # representative typical-set positions: exact draws from the mixture
    comp = (rng.random(64) >= W[0]).astype(int)
    z = rng.standard_normal((64, D)); z[:, 0] += MU[comp]
    L = 2.0

    print("=== EEVPD step tuning at beta=1 (easy mixture) ===", flush=True)
    step1, grid, evs = tune_step_eevpd(logdensity_fn, D, jnp.asarray(z), beta=1.0,
                                       L=L, target=DESIRED_EVAR, n_steps=40)
    print(f"\nbeta=1 EEVPD-tuned step = {step1:.5f}\n", flush=True)

    # realized EEVPD across the cooling ladder at the FIXED beta=1 step
    betas = np.geomspace(0.04, 1.0, 15)
    print("=== realized EEVPD across ladder at fixed beta=1 step ===", flush=True)
    ev_ladder = []
    for b in betas:
        ev = measure_eevpd(logdensity_fn, D, jnp.asarray(z), float(b), L, step1,
                           n_steps=40)
        ev_ladder.append(ev)
        flag = "OK" if ev <= DESIRED_EVAR * 1.5 else "ABOVE TARGET"
        print(f"   beta={b:.4f}  EEVPD={ev:.3e}  ({flag})", flush=True)
    ev_ladder = np.asarray(ev_ladder)

    np.savez(os.path.join(HERE, "tune_easy.npz"),
             step1=step1, grid=grid, evs=evs, betas=betas, ev_ladder=ev_ladder,
             target=DESIRED_EVAR, L=L)
    print(f"\nmax EEVPD across ladder = {ev_ladder.max():.3e} "
          f"(target {DESIRED_EVAR:.1e}); conservative-step holds = "
          f"{ev_ladder.max() <= DESIRED_EVAR*1.5}")
    print(f"wall {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
