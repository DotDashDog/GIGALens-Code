import jax, jax.numpy as jnp, numpy as np
from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted
np.set_printoptions(precision=2, suppress=False, linewidth=160)
print("devices", len(jax.devices()), "total")
d = 8
STEPS = 2000
def run(logp, init, tag, true_std=None):
    res = LAPS_late_adjusted(logp, dim=init.shape[1], num_chains=init.shape[0],
                             num_unadjusted_steps=STEPS, num_adjusted_steps=1,
                             init_positions=init, early_stop=False,
                             phase2_enabled=False, chunk_size=100, seed=0)
    var = np.asarray(res.p1_obs_sq) - np.asarray(res.p1_obs_mean)**2
    std = np.sqrt(np.clip(var, 0, None))
    init_std = std[0]; final_std = std[-1]; peak = std.max(0)
    lp = np.asarray(jax.vmap(lambda z: logp(z[None])[0])(jnp.asarray(res.samples).reshape(-1, d)))
    nan_frac = float(np.mean(~np.isfinite(lp)))
    infl = final_std / (init_std + 1e-30)
    print(f"\n### {tag}: STEPS={STEPS}  final-ensemble nonfinite-logp frac={nan_frac:.3f}")
    hdr = f"{'dim':>3} {'init_std':>9} {'final_std':>9} {'peak_std':>9} {'final/init':>10}"
    if true_std is not None: hdr += f" {'final/true':>10}"
    print(hdr)
    for i in range(d):
        line = f"{i:>3} {init_std[i]:9.2e} {final_std[i]:9.2e} {peak[i]:9.2e} {infl[i]:10.2f}"
        if true_std is not None: line += f" {final_std[i]/true_std[i]:10.2f}"
        print(line)
    verdict = "INFLATES (lens-like FAIL)" if np.any(infl > 1.5) else "contracts (healthy)"
    print(f"   => {verdict}   (max final/init = {infl.max():.1f})")
    return infl.max(), nan_frac

key = jax.random.key(0)
init_wide = 0.05 * jax.random.normal(key, (512, d))

# --- A: ROTATED anisotropic Gaussian (cond ~1e4, correlated) ---
sig = jnp.asarray(np.geomspace(1e-2, 1.0, d))          # cond 1e4
Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(7), (d, d)))
Prec = Q @ jnp.diag(1.0/sig**2) @ Q.T                  # precision = Sigma^-1
def logp_rot(x): return -0.5 * jnp.sum((x @ Prec) * x, axis=-1)
true_rot = np.sqrt(np.diag(np.asarray(Q @ jnp.diag(sig**2) @ Q.T)))
run(logp_rot, init_wide, "A rotated-aniso-Gaussian(cond1e4)", true_rot)

# --- B: BANANA (curved) in dims 0,1; isotropic N(0,0.1) elsewhere ---
b, s0, s1 = 3.0, 0.3, 0.1
def logp_ban(x):
    x0, x1 = x[..., 0], x[..., 1]
    curved = ((x1 - b*(x0**2 - s0**2))/s1)**2
    rest = jnp.sum((x[..., 2:]/0.1)**2, axis=-1)
    return -0.5*((x0/s0)**2 + curved + rest)
run(logp_ban, init_wide, "B banana(b=3)")

# --- C: HARD-BOUNDARY Gaussian: -inf outside box |x_i|<0.1 (unphysical-region proxy) ---
sig_c = 0.03
init_c = 0.15 * jax.random.normal(jax.random.key(1), (512, d))   # ~half start outside box
def logp_box(x):
    inside = jnp.all(jnp.abs(x) < 0.1, axis=-1)
    g = -0.5*jnp.sum((x/sig_c)**2, axis=-1)
    return jnp.where(inside, g, -jnp.inf)
frac_bad_init = float(np.mean(~np.isfinite(np.asarray(jax.vmap(lambda z: logp_box(z[None])[0])(init_c)))))
print(f"\n[C init] fraction of prior draws in forbidden region (logp=-inf): {frac_bad_init:.3f}")
run(logp_box, init_c, "C hard-boundary-box(|x|<0.1)")
print("\nESCALATE DONE")
