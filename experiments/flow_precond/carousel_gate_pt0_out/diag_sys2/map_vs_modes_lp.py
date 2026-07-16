"""Verify: is the stored MAP actually the posterior mode? Plain (archive-matched) model."""
import numpy as np, jax, jax.numpy as jnp
import carousel_model_s2 as S2
from gigalens.jax.scene_prob_model import ImageData
S2.AdaptiveImageData = ImageData          # archive-matched renderer
_, pm = S2.build()
assert type(pm.datasets[0]).__name__ == "ImageData", type(pm.datasets[0]).__name__
print("renderer:", type(pm.datasets[0]).__name__)
names = list(pm.z_param_names); j = names.index("planes/3/light/1/beta")
zb = np.load(S2.MAP_NPZ_S2)["z_best"]
Z = np.load(S2.MCLMC_NPZ_S2)["samples_z"].reshape(-1, 46)
THR = -3.0920
def lp_of(A, chunk=64):
    o=[]
    for i in range(0,len(A),chunk):
        v,_ = pm.log_prob(jnp.asarray(A[i:i+chunk])); o.append(np.asarray(v,dtype=np.float64))
    return np.concatenate(o)
lp_map = float(lp_of(zb[None])[0])
rng = np.random.default_rng(0)
out = {}
for tag, pool in (("sharp", Z[Z[:,j] > THR]), ("compact", Z[Z[:,j] < THR])):
    sub = pool[rng.choice(len(pool), 3000, replace=False)]
    lp = lp_of(sub)
    out[tag] = (lp.max(), lp.mean(), lp.min(), sub[np.argmax(lp), j])
    print(f"{tag:8s}: n={len(pool):6d}  lp max={lp.max():.3f}  mean={lp.mean():.3f}  min={lp.min():.3f}  z[37]@max={out[tag][3]:+.4f}")
print(f"\nstored MAP lp(z_best) = {lp_map:.3f}   z[37]_MAP = {zb[j]:+.4f}")
print(f"  best SHARP draw   - MAP = {out['sharp'][0]-lp_map:+.2f} nats")
print(f"  best COMPACT draw - MAP = {out['compact'][0]-lp_map:+.2f} nats")
print(f"  MEAN SHARP draw   - MAP = {out['sharp'][1]-lp_map:+.2f} nats")
print(f"  MEAN COMPACT draw - MAP = {out['compact'][1]-lp_map:+.2f} nats")
print(f"\n  best COMPACT - best SHARP = {out['compact'][0]-out['sharp'][0]:+.2f} nats")
print("\nIf draws exceed the MAP, the MAP optimizer did not find the mode (in 46-d the")
print("typical set sits BELOW the mode, so draws should be LOWER, never higher).")
