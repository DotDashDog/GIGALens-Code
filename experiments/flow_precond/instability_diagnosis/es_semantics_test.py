#!/usr/bin/env python
"""Early-stopping semantics test (Fv6; grader item 1 -- the producer's artifact).

Adversarial setup: Phase-B fkl trained on 3x-inflated data so it actively
damages the ELBO metric; verifies on CPU that
  (1) the stop fires on the metric schedule (patience 2 with the 4-nat floor),
  (2) the diag channel has ZERO influence on the decision (fed ratio_diag=1e9),
  (3) the returned params are exactly the best-metric checkpoint (metric of the
      returned flow == running best, not the stop-time value),
  (4) the es_trace round-trips through the cache.

Run:
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=...:src:experiments/flow_precond \
      python es_semantics_test.py
"""
import sys
import tempfile

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jax import numpy as jnp

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import demo_validation as dv
from gigalens_research.inference import flows

dv.OUT_DIR = tempfile.mkdtemp()
dim = 3
init, make = flows.make_whitened_spline_flow(
    jax.random.key(0), dim, jnp.zeros(dim), jnp.eye(dim),
    num_layers=2, num_bins=4, trainable_scale=False)
lp = lambda z: -0.5 * jnp.sum(z ** 2)
zs = np.random.default_rng(0).normal(size=(64, dim)) * 3.0  # mismatched on purpose


def ev(params):
    m = float(flows.neg_elbo_loss(params, make, jax.vmap(lp),
                                  jax.random.key(7), 64, dim))
    return m, 0.01, dict(ratio_diag=1e9)  # adversarial diag: must be ignored


params, hist = dv.train_flow(
    "es_semantics", init, make, lp, dim, n_draws=16, num_steps=0, lr=1e-3,
    seed=0, phase_b_samples=zs, phase_b_steps=800, phase_b_lr=5e-3,
    n_chunks=1, phase_b_eval_fn=ev, phase_b_eval_every=50)

trace = hist["es"]
metrics = [e["metric"] for e in trace]
best = min(metrics)
final = float(flows.neg_elbo_loss(params, make, jax.vmap(lp),
                                  jax.random.key(7), 64, dim))
stopped_early = len(hist["b"]) < 800
print("trace metrics:", [round(m, 3) for m in metrics])
assert stopped_early, "stop never fired on an actively damaging Phase B"
assert abs(final - best) < 1e-9, (final, best)
# patience 2: at least two violating checks after the best
viol = [m for m in metrics if m - best > max(0.02, 4.0)]
assert len(viol) >= 2, f"stopped with <2 violations: {metrics}"
# cache round-trip
params2, hist2 = dv.train_flow(
    "es_semantics", init, make, lp, dim, n_draws=16, num_steps=0, lr=1e-3,
    seed=0, phase_b_samples=zs, phase_b_steps=800, phase_b_lr=5e-3,
    n_chunks=1, phase_b_eval_fn=ev, phase_b_eval_every=50)
assert hist2["es"] is not None and len(hist2["es"]) == len(trace)
print(f"ES SEMANTICS PASS: stopped early, final==best ({final:.3f}), "
      f"diag ignored, {len(viol)} violations recorded, cache round-trip OK")
