import numpy as np, sys
sys.path.insert(0, "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model, model_seq, param_names
import jax, jax.numpy as jnp
# cached MAP
z_best = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/map/arrays.npz")['z_best']
z_best = np.asarray(z_best).reshape(-1)
dim = z_best.shape[0]
names = param_names(dim)
print("dim =", dim, " n_names =", len(names))
lp = lambda z: prob_model.log_prob(jnp.asarray(z))[0]
val = float(lp(z_best))
print("log_prob(MAP) =", val)
# constrained values at MAP
struct = prob_model.bij.forward(list(np.asarray(z_best).reshape(-1,1)))
print("\nidx | name | z_MAP | constrained_MAP")
flat_constr = None
try:
    fp = dict(struct)  # maybe returns dict name->val
except Exception as e:
    fp = None
for i,n in enumerate(names):
    print(f"{i:2d} | {n:35s} | {z_best[i]:9.4f}")
np.save("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy", np.array(names))
np.save("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/z_best.npy", z_best)
