"""CONTROL: is the archive reproduced by PLAIN ImageData (non-adaptive)?
If yes, the 1394-nat gap is notebook-vs-archive version drift, not a port bug."""
import numpy as np, jax.numpy as jnp
import carousel_model_s2 as S2
from gigalens.jax.scene_prob_model import ImageData
# swap the adaptive class for the plain one, change NOTHING else
S2.AdaptiveImageData = ImageData
print("patched AdaptiveImageData ->", S2.AdaptiveImageData.__name__)
_, pm = S2.build()
z = jnp.asarray(np.load(S2.MAP_NPZ_S2)["z_best"])
lp, chi2 = pm.log_prob(z)
BEST_LP, BEST_CHI2 = -521149.3527310094, 1.148182863187924
print(f"PLAIN ImageData: lp        = {float(lp):.6f}")
print(f"                 manifest  = {BEST_LP:.6f}")
print(f"                 abs diff  = {abs(float(lp)-BEST_LP):.6f} nats")
print(f"PLAIN ImageData: red_chi2  = {float(chi2):.9f}")
print(f"                 manifest  = {BEST_CHI2:.9f}")
print(f"                 abs diff  = {abs(float(chi2)-BEST_CHI2):.3e}")
