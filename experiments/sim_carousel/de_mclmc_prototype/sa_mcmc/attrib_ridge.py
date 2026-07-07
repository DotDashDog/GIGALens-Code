"""ORCHESTRATOR/GRADER attribution of the dim0 (along-ridge) std COLLAPSE seen in
C1 (truth init, dim0 std -> 0.65 vs target 3.0).  Invariance must hold from a truth
init, so a collapse is non-invariance SOMEWHERE.  Decisive question: is it the
MCLMC KERNEL (curved ridge under a single linear preconditioner at this step size
-> SA exonerated, an orthogonal tuning issue) or the SA MOVE itself (-> real bias,
disqualifying)?

Test (all from a TRUTH init = exact draws of curved mode B, dim0 std ~ 3.0; the
sampler should PRESERVE it):
  (a) vanilla MCLMC ONLY (no SA), step in {0.2, 0.1, 0.05}: does dim0 std collapse,
      and does a FINER step recover it toward 3.0?  (finer-step recovery => the
      collapse is MCLMC curvature-resolution, fixable, not SA.)
  (b) pure SA (K=0, mixture bw=0.20): does the SA move ALONE preserve dim0 std=3.0?
      (preserve => SA is invariant on the ridge; collapse => SA bias.)
  (c) the actual composite (K=20 step0.2 + SA): reproduces C1's collapse for ref.

Single process, sequential, file-logged (avoids the concurrent-job login-node kill).
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from curved_testbed import build_target
from sa_move import make_sa_composite

def pr(*a): print(*a, flush=True)

def main():
    t0 = time.time()
    tgt = build_target(b=0.70, n_curve=4); D = tgt.D; N = 64
    imm = jnp.asarray(tgt.within_mode_cov(mode=1))
    # TRUTH init: exact draws of curved mode B only (dim0 std ~ s_ridge=3.0)
    rng = np.random.default_rng(0)
    comp_lbl = np.full(N, 1, int)
    Xb = rng.standard_normal((N, D)) * np.sqrt(tgt.var)[None, :] + tgt.mu[comp_lbl]
    initB = tgt.f_np(Xb)
    pr(f"truth-init dim0 std = {initB[:,0].std():.3f} (target s_ridge={tgt.s_ridge})  N={N} D={D}")

    # build one composite to reuse mclmc_only + init_states (knobs match C1)
    base = make_sa_composite(tgt.logp, D, N, L=2.0, step_size=0.2, K=20, n_sa=N,
                             proposal="mixture", bandwidth=0.20, kernel_cov=jnp.eye(D),
                             inverse_mass_matrix=imm)

    # ---- (a) vanilla MCLMC only, step sweep ---------------------------------
    pr("\n[a] vanilla MCLMC ONLY (no SA) from truth init, dim0 std after 3000 steps:")
    for step in [0.2, 0.1, 0.05]:
        comp = make_sa_composite(tgt.logp, D, N, L=2.0, step_size=step, K=20, n_sa=N,
                                 proposal="mixture", bandwidth=0.20, kernel_cov=jnp.eye(D),
                                 inverse_mass_matrix=imm)
        st = comp["init_states"](jnp.asarray(initB), jax.random.key(5))
        mk = jax.random.split(jax.random.key(7), 3000 * N).reshape(3000, N)
        st, pos = comp["mclmc_only"](st, mk); pos = np.asarray(pos)
        d0 = pos[-500:, :, 0].std()
        # also mean logp of last-step positions (off-ridge => very negative)
        lp = float(np.asarray(jax.vmap(tgt.logp)(jnp.asarray(pos[-1]))).mean())
        pr(f"   step={step:.2f}: dim0 std last500={d0:.3f} (truth 3.0)  mean logp last={lp:.1f}")

    # ---- (b) pure SA (K=0) from truth init ----------------------------------
    pr("\n[b] pure SA (K=0, mixture bw=0.20) from truth init, dim0 std vs round:")
    saK0 = make_sa_composite(tgt.logp, D, N, L=2.0, step_size=0.2, K=0, n_sa=N,
                             proposal="mixture", bandwidth=0.20, kernel_cov=jnp.eye(D),
                             inverse_mass_matrix=imm)
    st = saK0["init_states"](jnp.asarray(initB), jax.random.key(5))
    keys = jax.random.split(jax.random.key(8), 800)
    d0s = []
    for r in range(800):
        st, (p, ec, sub) = saK0["round"](st, keys[r]); d0s.append(np.asarray(p)[:, 0].std())
    d0s = np.array(d0s)
    pr(f"   pure-SA dim0 std: init={initB[:,0].std():.3f} round100={d0s[100]:.3f} "
       f"round400={d0s[400]:.3f} last200mean={d0s[-200:].mean():.3f} (truth 3.0)")

    # ---- (b2) pure SA with a LARGER along-ridge bandwidth --------------------
    # if dim0 collapses under bw=0.20 (kernel std 0.20 << ridge 3.0) but NOT under a
    # ridge-matched bandwidth, the collapse is a bw-efficiency (mixing) effect, still
    # exact at stationarity but slow; test bw=1.0 to see dim0 mobility.
    pr("\n[b2] pure SA (K=0) with bw=1.0 (better ridge coverage):")
    saK0b = make_sa_composite(tgt.logp, D, N, L=2.0, step_size=0.2, K=0, n_sa=N,
                              proposal="mixture", bandwidth=1.0, kernel_cov=jnp.eye(D),
                              inverse_mass_matrix=imm)
    st = saK0b["init_states"](jnp.asarray(initB), jax.random.key(5))
    keys = jax.random.split(jax.random.key(9), 800); d0s2 = []
    for r in range(800):
        st, (p, ec, sub) = saK0b["round"](st, keys[r]); d0s2.append(np.asarray(p)[:, 0].std())
    d0s2 = np.array(d0s2)
    pr(f"   pure-SA bw=1.0 dim0 std: round100={d0s2[100]:.3f} last200mean={d0s2[-200:].mean():.3f} (truth 3.0)")

    # ---- (c) actual composite (K=20 step0.2 + SA bw0.20) --------------------
    pr("\n[c] composite (K=20 step0.2 + SA bw0.20) from truth init (reproduce C1):")
    st = base["init_states"](jnp.asarray(initB), jax.random.key(5))
    keys = jax.random.split(jax.random.key(10), 800); d0c = []
    for r in range(800):
        st, (p, ec, sub) = base["round"](st, keys[r]); d0c.append(np.asarray(p)[:, 0].std())
    d0c = np.array(d0c)
    pr(f"   composite dim0 std: round100={d0c[100]:.3f} last200mean={d0c[-200:].mean():.3f} (truth 3.0)")

    np.savez(os.path.join(HERE, "attrib_ridge.npz"), d0s=d0s, d0s2=d0s2, d0c=d0c)
    pr(f"\ntotal wall {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
