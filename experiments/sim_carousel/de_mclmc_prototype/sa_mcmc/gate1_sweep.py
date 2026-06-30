"""GATE-1 calibration: sweep curvature b (and n_curve), measure STATIC linear-DE
WITHIN-mode acceptance (target ~0.5-1.5%, carousel ~0.6%) plus a curvature proxy
and the per-curved-dim off-ridge factor (carousel ~7x)."""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curved_testbed import build_target, static_linear_de_acceptance

NC = int(os.environ.get("NCURVE", "2"))
BS = [float(x) for x in (sys.argv[1:] or [0.1, 0.2, 0.3, 0.4, 0.6])]
print(f"n_curve={NC}")
print("b      offridge_x(per-dim)  curv_drop(nats)  static_linDE_acc")
for b in BS:
    tgt = build_target(b=b, n_curve=NC)
    # off-ridge factor: chord midpoint perpendicular offset / s_thin, per curved dim
    # offset = b*(u-v)^2/4 ; typical (u-v)^2 = 2 s_ridge^2 -> b*s_ridge^2/2
    offx = b * tgt.s_ridge ** 2 / 2.0 / tgt.s_thin
    rngc = np.random.default_rng(2); cB = np.ones(400, int)
    Y = tgt.f_np(tgt._draw_x(cB, rngc)); a, bb = Y[:200], Y[200:400]; mid = 0.5 * (a + bb)
    lp = jax.vmap(tgt.logp)
    drop = float((np.asarray(lp(jnp.asarray(mid))) - 0.5 * (np.asarray(lp(jnp.asarray(a))) + np.asarray(lp(jnp.asarray(bb))))).mean())
    acc = static_linear_de_acceptance(tgt, n=40000)
    print(f"{b:.3f}      {offx:6.2f}            {drop:9.2f}        {acc*100:6.3f}%")
