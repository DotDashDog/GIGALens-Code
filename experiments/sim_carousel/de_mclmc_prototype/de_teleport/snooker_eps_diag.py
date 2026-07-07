"""DRILL-DOWN (protocol rule 1): snooker w@truth=0.66 != 0.70 with the Jacobian.
Hypothesis: the isotropic eps jitter (b0) pushes the proposal OFF the snooker LINE,
so the radial (d-1) Jacobian -- exact only for an ON-LINE move -- mismeasures the
proposal density -> small bias. Prediction: b0=0 (pure on-line snooker) -> w@truth
~ 0.70 (unbiased); b0 growing -> bias grows. Falsifier: b0=0 still gives 0.66 ->
the Jacobian itself is wrong.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from de_teleport import make_teleport_composite
from validate_analytic import block_bootstrap_se

D = 10; m = 5.0; NCH = 32; L, STEP, K = 2.0, 0.5, 20; SEED = 20260627
W = jnp.asarray([0.7, 0.3]); logW = jnp.log(W); MU = jnp.asarray([+m, -m]); c = -0.5*D*jnp.log(2*jnp.pi)
def logp(z):
    z0 = z[0]; qr = jnp.sum(z[1:]**2)
    return jax.scipy.special.logsumexp(jnp.stack([logW[0]+c-0.5*((z0-MU[0])**2+qr),
                                                  logW[1]+c-0.5*((z0-MU[1])**2+qr)]))
def truth_init(seed):
    rng = np.random.default_rng(seed); n0 = int(round(0.7*NCH))
    comp = np.ones(NCH, int); comp[:n0] = 0
    z = rng.standard_normal((NCH, D)); z[:, 0] += MU.__array__()[comp]; return z

R = 2500; boot = np.random.default_rng(SEED)
print("snooker invariance-from-truth vs eps jitter b0 (truth weight(+)=0.70):", flush=True)
for b0 in [0.0, 0.005, 0.02, 0.05]:
    comp = make_teleport_composite(logp, D, NCH, L=L, step_size=STEP, K=K,
                                   move="snooker", b0=b0, eps_scale=None)
    st = comp["init_states"](jnp.asarray(truth_init(SEED)), jax.random.key(SEED))
    keys = jax.random.split(jax.random.key(SEED+5), R)
    frac = np.empty(R)
    for r in range(R):
        st, (p, ec, a) = comp["round"](st, keys[r]); frac[r] = float((np.asarray(p)[:,0] > 0).mean())
    w = float(frac.mean()); se, _ = block_bootstrap_se(frac, rng=boot)
    print(f"  b0={b0:5.3f}  w(+)@truth={w:.4f} +/- {se:.4f}  |w-0.70|={abs(w-0.70):.4f} ({abs(w-0.70)/max(se,1e-9):.1f} SE)", flush=True)
