#!/usr/bin/env python
"""Archive the arm-D (faithful NeuTra) divergence-mechanism evidence (2 legs).

Leg A — demo target gradient finiteness: the lens log-prob is finite at ordinary
unconstrained draws z ~ N(0, s*I), but its GRADIENT is non-finite for a large
fraction of them. The untrained (unwhitened) NumPyro-faithful IAF proposes exactly
there, so its first 1-sample ELBO steps consume non-finite grad(lp).

Leg B — recipe control on an equally-sharp but finite-gradient target: the same
faithful recipe (3-block IAF, 1-sample reverse-KL, Adam 3e-3) on a 22-dim sharp
Gaussian (mu=1.17, sigma=0.01; starting neg-ELBO ~1e5 regime) is violently noisy
but NEVER produces non-finite loss/params — so the flow cannot self-NaN; the NaN
must enter through the target.

(Leg C — numerical faithfulness of the IAF to numpyro source — is pinned by
test_flows.py: test_iaf_init_reproduces_numpyro_scheme + the Jacobian tests.)

CPU-runnable (renders ~a minute); writes armD_mechanism_out/{summary.json, arrays.npz}.
"""
import os
import json
import sys

sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "armD_mechanism_out")
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import optax

sys.path.insert(0, HERE)
from gigalens_research.inference import flows

SCALES = [1.0, 3.0, 5.0, 8.0]
N_DRAWS = 16
SEED = 0
TOY_STEPS = 400
TOY_SEEDS = (0, 1)
TOY_MU, TOY_SIGMA = 1.17, 0.01
IAF_LR = 3e-3


def leg_a():
    import demo_validation as dv
    model_seq, prob_model, qz, *_ , dim = dv.build_model_and_fit()
    lp_fn = lambda z: prob_model.log_prob(z)[0]
    g_fn = jax.grad(lp_fn)
    rng = np.random.default_rng(SEED)
    rows = {}
    grads_all = {}
    for s in SCALES:
        zs = jnp.asarray(rng.normal(size=(N_DRAWS, dim)) * s)
        lps = np.asarray(jax.vmap(lp_fn)(zs))
        grads = np.asarray(jax.vmap(g_fn)(zs))
        rows[str(s)] = dict(
            lp_finite=int(np.isfinite(lps).sum()), n=N_DRAWS,
            grad_finite_rows=int(np.isfinite(grads).all(1).sum()),
            lp_min=float(np.nanmin(lps)), lp_max=float(np.nanmax(lps)),
            max_abs_grad=float(np.nanmax(np.abs(grads))),
        )
        grads_all[f"grads_s{s}"] = grads
        print(f"leg A scale {s}: {rows[str(s)]}")
    return rows, grads_all, dim


def leg_b(dim):
    mu = jnp.full(dim, TOY_MU)

    def target_lp(z):  # batched (n, dim) -> (n,)
        return -0.5 * jnp.sum(((z - mu) / TOY_SIGMA) ** 2, axis=-1) \
            - dim * jnp.log(TOY_SIGMA) - 0.5 * dim * jnp.log(2 * jnp.pi)

    hists = {}
    for seed in TOY_SEEDS:
        init, make = flows.make_numpyro_iaf_flow(jax.random.key(seed), dim)
        opt = optax.adam(IAF_LR)

        @jax.jit
        def step(params, opt_state, key):
            loss, g = jax.value_and_grad(flows.neg_elbo_loss)(
                params, make, target_lp, key, 1, dim)
            updates, opt_state = opt.update(g, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        params, opt_state = init, opt.init(init)
        keys = jax.random.split(jax.random.key(100 + seed), TOY_STEPS)
        hist = []
        for i in range(TOY_STEPS):
            params, opt_state, loss = step(params, opt_state, keys[i])
            hist.append(float(loss))
        params_finite = all(bool(jnp.isfinite(l).all())
                            for l in jax.tree_util.tree_leaves(params))
        hists[seed] = dict(hist=np.asarray(hist),
                           any_nonfinite_loss=bool(~np.isfinite(hist).all()),
                           params_finite_at_end=params_finite,
                           first=float(hist[0]), last=float(hist[-1]))
        print(f"leg B seed {seed}: first {hist[0]:.3g} last {hist[-1]:.3g} "
              f"nonfinite_loss={hists[seed]['any_nonfinite_loss']} "
              f"params_finite={params_finite}")
    return hists


def main():
    rows, grads_all, dim = leg_a()
    hists = leg_b(dim)
    summary = {
        "leg_a_grad_finiteness": rows,
        "leg_b_toy_control": {str(s): {k: v for k, v in h.items() if k != "hist"}
                              for s, h in hists.items()},
        "config": dict(scales=SCALES, n_draws=N_DRAWS, seed=SEED,
                       toy=dict(steps=TOY_STEPS, seeds=list(TOY_SEEDS),
                                mu=TOY_MU, sigma=TOY_SIGMA, lr=IAF_LR)),
    }
    np.savez(os.path.join(OUT_DIR, "arrays.npz"),
             **grads_all, **{f"toy_hist_{s}": h["hist"] for s, h in hists.items()})
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY:", json.dumps(summary["leg_a_grad_finiteness"], indent=2))


if __name__ == "__main__":
    main()
