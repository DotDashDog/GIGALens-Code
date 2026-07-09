"""Mechanism probes at init (no training).

P1: output displacement caused by ONE adam step, vs (range, bins).
    Adam's first update is ~ lr * sign(g) in every coordinate with g != 0,
    so displacement per step should scale with the parameterization's
    output-sensitivity, hypothesized ~ 2R (softmax*2R + cumsum lever arm),
    roughly independent of bin count.

P2: decompose that first step: apply only the final-layer-bias update
    restricted to (hit-bin logits) vs (unused-bin logits) vs (slope raws)
    vs (hidden-layer W,b) and measure max|dz| and dloss for each group.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random as jr

from gigalens_research.inference import flows

DIM = 33
SDS = np.geomspace(1.7, 50.0, DIM)
_LOG2PI = np.log(2 * np.pi)
FLOW_KEY = jr.key(123)
LR = 3e-3


def make_target():
    sds = jnp.asarray(SDS, dtype=jnp.float64)
    const = -0.5 * DIM * _LOG2PI - jnp.sum(jnp.log(sds))
    return lambda z: const - 0.5 * jnp.sum((z / sds) ** 2, axis=-1)


lp = make_target()
u_eval = jr.normal(jr.key(7), (512, DIM), dtype=jnp.float64)


def setup(num_bins, spline_range):
    init_params, make_bij = flows.make_whitened_spline_flow(
        FLOW_KEY, DIM, jnp.zeros(DIM, dtype=jnp.float64),
        jnp.eye(DIM, dtype=jnp.float64),
        num_bins=num_bins, spline_range=spline_range)
    loss_fn = lambda p: flows.neg_elbo_loss(
        p, make_bij, lp, jr.key(0), 128, DIM)
    return init_params, make_bij, loss_fn


def one_adam_step(params, loss_fn):
    opt = optax.adam(LR)
    st = opt.init(params)
    l0, g = jax.value_and_grad(loss_fn)(params)
    upd, _ = opt.update(g, st, params)
    return optax.apply_updates(params, upd), l0, g, upd


def zdisp(make_bij, p0, p1):
    z0 = make_bij(p0).forward(u_eval)
    z1 = make_bij(p1).forward(u_eval)
    d = jnp.abs(z1 - z0)
    return float(jnp.max(d)), float(jnp.mean(d))


print("=== P1: displacement from ONE adam step (lr 3e-3), by (range, bins) ===")
for R, K in [(357.0, 490), (357.0, 48), (357.0, 16), (89.0, 490), (89.0, 122),
             (11.0, 490), (11.0, 16), (6.0, 8)]:
    params, make_bij, loss_fn = setup(K, R)
    p1, l0, g, upd = one_adam_step(params, loss_fn)
    dmax, dmean = zdisp(make_bij, params, p1)
    l1 = float(loss_fn(p1))
    print(f"R={R:6.1f} K={K:4d}: loss {float(l0):9.3f} -> {l1:9.3f}  "
          f"max|dz|={dmax:8.3f} mean|dz|={dmean:8.4f}  dz/R={dmax/R:.4f}",
          flush=True)

print("\n=== P2: group decomposition of first step, R=357 K=490 ===")
K, R = 490, 357.0
params, make_bij, loss_fn = setup(K, R)
p1, l0, g, upd = one_adam_step(params, loss_fn)
pcount = 3 * K - 1
w0 = 2 * R / K
lo = int((R - 4.5) / w0)
hi = int(np.ceil((R + 4.5) / w0))
hit = np.zeros(K, bool)
hit[lo:hi + 1] = True
hitmask = np.concatenate([hit, hit, np.zeros(pcount - 2 * K, bool)])  # rw,rh hit
unusedmask = np.concatenate([~hit, ~hit, np.zeros(pcount - 2 * K, bool)])
slopemask = np.concatenate([np.zeros(2 * K, bool), np.ones(pcount - 2 * K, bool)])


def apply_group(upd, which):
    """Zero the update everywhere except the selected group."""
    def sel(path_layer, sub, arr, u):
        return u  # placeholder
    newp = jax.tree_util.tree_map(lambda x: x, params)
    coup = []
    for li, layer in enumerate(params["couplings"]):
        nl = []
        for si, (W, b) in enumerate(layer):
            uW, ub = upd["couplings"][li][si]
            last = si == len(layer) - 1
            if which == "hidden" and not last:
                nl.append((W + uW, b + ub))
            elif which in ("bias_hit", "bias_unused", "bias_slope") and last:
                m = {"bias_hit": hitmask, "bias_unused": unusedmask,
                     "bias_slope": slopemask}[which]
                mfull = jnp.asarray(np.tile(m, DIM))
                nl.append((W, b + ub * mfull))
            elif which == "finalW" and last:
                nl.append((W + uW, b))
            else:
                nl.append((W, b))
        coup.append(nl)
    return {"couplings": coup}


print(f"full step: loss {float(l0):.3f} -> {float(loss_fn(p1)):.3f}, "
      f"max|dz|={zdisp(make_bij, params, p1)[0]:.3f}")
for which in ["bias_hit", "bias_unused", "bias_slope", "finalW", "hidden"]:
    pg = apply_group(upd, which)
    dmax, dmean = zdisp(make_bij, params, pg)
    lg = float(loss_fn(pg))
    print(f"group {which:12s}: loss -> {lg:9.3f}  max|dz|={dmax:8.4f} "
          f"mean|dz|={dmean:8.5f}", flush=True)

print("\n=== P2b: coherence of bias update (rh logits), coupling0, dim with "
      "largest sd ===")
ub = np.asarray(upd["couplings"][0][-1][1]).reshape(DIM, pcount)
gb = np.asarray(g["couplings"][0][-1][1]).reshape(DIM, pcount)
for d in [0, 16, 32]:
    rh_u = ub[d, K:2 * K]
    print(f"dim {d} (sd={SDS[d]:5.1f}): rh update: "
          f"frac|u|>0.9lr={np.mean(np.abs(rh_u) > 0.9 * LR):.2f}, "
          f"sum(update)={rh_u.sum():+.4f}, "
          f"sum(left half)={rh_u[:K//2].sum():+.4f}, "
          f"sum(hit)={rh_u[hit].sum():+.4f}, "
          f"sum(unused)={rh_u[~hit].sum():+.4f}", flush=True)
