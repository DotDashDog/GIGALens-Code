"""Synthetic-target reproduction of the range-357/bins-490 flow instability.

Target: diagonal Gaussian in dim=33 with per-dim sds geomspace(1.7, 50)
(mimics the carousel's SVI-whitened geometry). Flow: make_whitened_spline_flow
with qz_loc=0, qz_scale_tril=I (grid a-c) or diag(sds) (grid d-e).
Training: adam(lr), value_and_grad(flows.neg_elbo_loss), n_draws=128,
fresh key per step (same recipe as demo_validation.train_flow Phase A).
Diagnosis only -- no fixes.
"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jr

from gigalens_research.inference import flows

DIM = 33
SDS = np.geomspace(1.7, 50.0, DIM)
N_DRAWS = 128
NUM_STEPS = 200
SEED = 0
FLOW_KEY = jr.key(123)

_LOG2PI = np.log(2 * np.pi)


def make_target(sds):
    sds = jnp.asarray(sds, dtype=jnp.float64)
    const = -0.5 * DIM * _LOG2PI - jnp.sum(jnp.log(sds))

    def lp(z):  # (n, dim) -> (n,)
        return const - 0.5 * jnp.sum((z / sds) ** 2, axis=-1)

    return lp


def run_config(name, num_bins, spline_range, lr, prestandardized,
               num_steps=NUM_STEPS, diagnostics=False):
    print(f"\n{'='*72}\nCONFIG {name}: bins={num_bins} range={spline_range} "
          f"lr={lr} prestd={prestandardized}", flush=True)
    if prestandardized:
        scale_tril = jnp.diag(jnp.asarray(SDS, dtype=jnp.float64))
    else:
        scale_tril = jnp.eye(DIM, dtype=jnp.float64)
    init_params, make_bij = flows.make_whitened_spline_flow(
        FLOW_KEY, DIM, jnp.zeros(DIM, dtype=jnp.float64), scale_tril,
        num_bins=num_bins, spline_range=spline_range)
    lp = make_target(SDS)

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state, key):
        loss, g = jax.value_and_grad(flows.neg_elbo_loss)(
            params, make_bij, lp, key, N_DRAWS, DIM)
        updates, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, g

    # fixed eval key for a low-noise loss decomposition
    eval_key = jr.key(999)

    @jax.jit
    def eval_terms(params):
        bij = make_bij(params)
        u = jr.normal(eval_key, (512, DIM), dtype=jnp.float64)
        z = bij.forward(u)
        fldj = bij.forward_log_det_jacobian(u, event_ndims=1)
        base = -0.5 * jnp.sum(u * u + _LOG2PI, axis=-1)
        return (jnp.mean(base - fldj - lp(z)),
                jnp.mean(-fldj), jnp.mean(-lp(z)), jnp.max(jnp.abs(z)))

    def grad_report(g, label):
        print(f"  --- grad norms {label} ---", flush=True)
        for li, layer in enumerate(g["couplings"]):
            parts = []
            for si, (W, b) in enumerate(layer):
                parts.append(f"L{si}: |gW|={float(jnp.linalg.norm(W)):.3e} "
                             f"|gb|={float(jnp.linalg.norm(b)):.3e} "
                             f"max|gW|={float(jnp.max(jnp.abs(W))):.3e}")
            print(f"  coupling{li}: " + " | ".join(parts), flush=True)

    def unused_bin_check(g, params):
        # grad of loss wrt final-layer BIAS of coupling 0, reshaped to raw
        # per-dim spline params (dim, 3*nb-1). Bias adds directly to raw, so
        # this is exactly d loss / d raw (summed over batch pathways).
        gb = np.asarray(g["couplings"][0][-1][1]).reshape(DIM, 3 * num_bins - 1)
        grw = gb[:, :num_bins]
        grh = gb[:, num_bins:2 * num_bins]
        grs = gb[:, 2 * num_bins:]
        # hit bins at identity init: uniform width 2R/nb; u~N(0,1) -> |u|<~4.5
        w0 = 2 * spline_range / num_bins
        lo = int((spline_range - 4.5) / w0)
        hi = int(np.ceil((spline_range + 4.5) / w0))
        hit = np.zeros(num_bins, bool)
        hit[max(lo, 0):min(hi + 1, num_bins)] = True
        print(f"  hit-bin window: bins [{lo},{hi}] of {num_bins} "
              f"({hit.sum()} hit, {num_bins - hit.sum()} unused)", flush=True)
        for tag, arr in [("rw", grw), ("rh", grh)]:
            h = np.abs(arr[:, hit]).mean()
            u_ = np.abs(arr[:, ~hit]).mean() if (~hit).any() else 0.0
            umax = np.abs(arr[:, ~hit]).max() if (~hit).any() else 0.0
            print(f"  grad wrt {tag} logits: mean|hit|={h:.3e} "
                  f"mean|unused|={u_:.3e} max|unused|={umax:.3e}", flush=True)
        print(f"  grad wrt slope raw: mean|all|={np.abs(grs).mean():.3e}",
              flush=True)

    params, opt_state = init_params, opt.init(init_params)
    keys = jr.split(jr.key(SEED), num_steps)
    hist = []
    t0 = time.time()
    g0 = None
    p0 = jax.tree_util.tree_map(lambda x: x, params)
    for i in range(num_steps):
        params_new, opt_state, loss, g = step(params, opt_state, keys[i])
        hist.append(float(loss))
        if diagnostics and i == 0:
            g0 = g
            grad_report(g, "step 0")
            unused_bin_check(g, params)
        if i % 10 == 0:
            l, nf, nl, zmax = eval_terms(params)
            print(f"  step {i:4d}: loss={hist[-1]:14.4f}  "
                  f"eval={float(l):12.4f} (-fldj={float(nf):10.3f} "
                  f"-lp={float(nl):10.3f}) max|z|={float(zmax):.1f}", flush=True)
        if not np.isfinite(hist[-1]):
            print(f"  DIVERGED (non-finite loss) at step {i}", flush=True)
            break
        params = params_new
        if diagnostics and i == 10:
            grad_report(g, "step 10")
            # param drift after 10 steps
            print("  --- param drift after 10 steps (max|delta|) ---", flush=True)
            for li in range(len(params["couplings"])):
                parts = []
                for si in range(len(params["couplings"][li])):
                    dW = jnp.max(jnp.abs(params["couplings"][li][si][0]
                                         - p0["couplings"][li][si][0]))
                    db = jnp.max(jnp.abs(params["couplings"][li][si][1]
                                         - p0["couplings"][li][si][1]))
                    parts.append(f"L{si}: dW={float(dW):.3e} db={float(db):.3e}")
                print(f"  coupling{li}: " + " | ".join(parts), flush=True)
    l, nf, nl, zmax = eval_terms(params)
    print(f"  final ({len(hist)} steps): loss={hist[-1]:14.4f}  "
          f"eval={float(l):12.4f} max|z|={float(zmax):.1f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    h = np.array(hist)
    inc = int(np.sum(np.diff(h) > 0))
    print(f"  summary: loss[0]={h[0]:.2f} min={h.min():.2f} "
          f"last={h[-1]:.2f} #increases={inc}/{len(h)-1}", flush=True)
    return h


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    grid = {
        "a":  dict(num_bins=490, spline_range=357.0, lr=3e-3,
                   prestandardized=False, diagnostics=True),
        "b1": dict(num_bins=490, spline_range=357.0, lr=3e-4,
                   prestandardized=False),
        "b2": dict(num_bins=490, spline_range=357.0, lr=1e-4,
                   prestandardized=False),
        "c":  dict(num_bins=48, spline_range=357.0, lr=3e-3,
                   prestandardized=False),
        "d":  dict(num_bins=16, spline_range=11.0, lr=3e-3,
                   prestandardized=True),
        "e":  dict(num_bins=48, spline_range=11.0, lr=3e-3,
                   prestandardized=True),
    }
    names = list(grid) if which == "all" else which.split(",")
    for n in names:
        run_config(n, **grid[n])
