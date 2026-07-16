"""Is z[37] (src5 log-beta) in prior-data conflict? Uses the harness's exact prior-draw path."""
import numpy as np, jax
import carousel_model_s2 as S2
_, pm = S2.build()
names = list(pm.z_param_names); j = names.index('planes/3/light/1/beta')
x = pm.prior.sample(200000, seed=jax.random.PRNGKey(0))
zs = np.asarray(pm.bij.inverse(x), dtype=np.float64)
print("prior z-sample shape:", zs.shape, "| src5 beta index:", j)
p = zs[:, j]
print(f"PRIOR      z[{j}]: mean={p.mean():+.4f} sd={p.std():.4f} range=[{p.min():+.3f},{p.max():+.3f}]")
print(f"   expected if bijector==exp: N(log(0.1)={np.log(0.1):+.4f}, 0.15)")
post = np.load(S2.MCLMC_NPZ_S2)["samples_z"].reshape(-1, len(names))[:, j]
print(f"POSTERIOR  z[{j}]: mean={post.mean():+.4f} sd={post.std():.4f} range=[{post.min():+.3f},{post.max():+.3f}]")
for label, v in (("sharp mode", -2.78), ("compact mode", -3.40), ("post min", float(post.min()))):
    print(f"   {label:12s} z={v:+.3f} -> {(v-p.mean())/p.std():+6.2f} prior-sigma  (beta={np.exp(v):.4f} arcsec = {np.exp(v)/0.2:.3f} pix)")
print(f"\nP(prior     < -3.0920) = {(p < -3.0920).mean():.3e}")
print(f"P(posterior < -3.0920) = {(post < -3.0920).mean():.4f}")
