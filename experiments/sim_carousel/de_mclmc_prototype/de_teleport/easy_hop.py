"""Gate A 'basic mode-hopping on the EASY case': from a BOTH-MODES-POPULATED init,
do the moves actually round-trip between the two SEPARATED unit-Gaussian modes?
(V2 invariance alone does not prove hopping -- a frozen ensemble also holds weight.)
Separated geometry => chords ARE on-manifold => hopping SHOULD work; this is the
benign control for the curved-target failure. Reports round-trips per move + the
vanilla-MCLMC (no move) contrast (must be ~0 round-trips: barrier 12.5 trapped).
"""
import os, sys, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from de_teleport import make_teleport_composite

D = 10; m = 5.0; NCH = 32; L, STEP, K = 2.0, 0.5, 20; SEED = 20260627; R = 1500
W = jnp.asarray([0.7, 0.3]); logW = jnp.log(W); MU = jnp.asarray([+m, -m]); c = -0.5*D*jnp.log(2*jnp.pi)
def logp(z):
    z0 = z[0]; qr = jnp.sum(z[1:]**2)
    return jax.scipy.special.logsumexp(jnp.stack([logW[0]+c-0.5*((z0-MU[0])**2+qr),
                                                  logW[1]+c-0.5*((z0-MU[1])**2+qr)]))
def balanced_init(seed):
    rng = np.random.default_rng(seed); n0 = int(round(0.7*NCH))
    comp = np.ones(NCH, int); comp[:n0] = 0
    z = rng.standard_normal((NCH, D)); z[:, 0] += np.asarray(MU)[comp]; return z
def round_trips(modes):
    tot = cross = 0
    for cc in range(modes.shape[1]):
        d = np.diff(modes[:, cc]); ups = int((d==1).sum()); dns = int((d==-1).sum())
        tot += min(ups, dns); cross += int(np.abs(d).sum())
    return tot, cross

print(f"easy-case mode-hopping from populated init (R={R}, {NCH} chains):", flush=True)
for move in ["gamma1", "near", "snooker"]:
    comp = make_teleport_composite(logp, D, NCH, L=L, step_size=STEP, K=K,
                                   move=move, b0=(0.0 if move=="snooker" else 0.05),
                                   p_jump=0.5, eps_scale=None)
    st = comp["init_states"](jnp.asarray(balanced_init(SEED)), jax.random.key(SEED))
    keys = jax.random.split(jax.random.key(SEED+5), R)
    modes = np.empty((R, NCH), int); acc = np.empty(R)
    for r in range(R):
        st, (p, ec, a) = comp["round"](st, keys[r])
        modes[r] = (np.asarray(p)[:,0] < 0).astype(int); acc[r] = float(np.asarray(a).mean())
    rt, cross = round_trips(modes)
    w_plus = (modes[R//3:]==0).mean()
    print(f"  {move:8s} round-trips={rt:4d} crossings={cross:4d} move-acc={acc.mean()*100:5.2f}% w(+)={w_plus:.3f}", flush=True)
# vanilla contrast
st = comp["init_states"](jnp.asarray(balanced_init(SEED)), jax.random.key(SEED+1))
modes = np.empty((R, NCH), int)
ck = jax.random.split(jax.random.key(SEED+2), R*K*NCH).reshape(R, K, NCH)
for r in range(R):
    st, pv = comp["mclmc_only"](st, ck[r]); modes[r] = (np.asarray(pv[-1])[:,0] < 0).astype(int)
rt, cross = round_trips(modes)
print(f"  {'vanilla':8s} round-trips={rt:4d} crossings={cross:4d} (barrier 12.5 -> expect ~0)", flush=True)
