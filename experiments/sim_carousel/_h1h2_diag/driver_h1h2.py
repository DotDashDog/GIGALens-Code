import numpy as np, sys, os, time
sys.path.insert(0, "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model, model_seq
import jax, jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from gigalens_research.inference import MCLMC_JIT

WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
mode = sys.argv[1]          # "under" or "over"
nb   = int(sys.argv[2])
nr   = int(sys.argv[3])
outname = sys.argv[4]
seed = int(sys.argv[5]) if len(sys.argv)>5 else 42

z_best = jnp.asarray(np.load(WD+"z_best.npy"))
dim = z_best.shape[0]
print(f"[{mode}] devices:", jax.devices())

if mode == "under":
    # notebook default: tight isotropic ball
    qz = tfd.MultivariateNormalDiag(loc=z_best, scale_diag=jnp.full(dim, 1e-2))
elif mode == "over":
    # over-dispersed ALONG the estimated posterior ridge: 2x the empirical std
    # (4x covariance) of the cached (under-converged) chains, centered at MAP.
    scov = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/"
                   "messy_tests/just_map/mclmc.stale-20260626T115536/diagnostics.npz")['samples_cov']
    scov = 0.5*(scov+scov.T) + 1e-8*np.eye(dim)
    L = np.linalg.cholesky(4.0*scov)   # 4x cov => 2x std
    qz = tfd.MultivariateNormalTriL(loc=z_best, scale_tril=jnp.asarray(L))
else:
    raise SystemExit("mode must be under|over")

t0=time.perf_counter()
hist = MCLMC_JIT(model_seq=model_seq, qz=qz, n_hmc=8,
                 num_burnin_steps=nb, num_results=nr,
                 desired_energy_variance=5e-4, regularize_mass_matrix=True,
                 progress_bar=False, seed=seed, debug_output=True)
print(f"[{mode}] sampling wall {time.perf_counter()-t0:.1f}s")
pos = np.asarray(hist.position)          # (chains, total, dim)
xi  = np.asarray(hist.xi)
ss  = np.asarray(hist.step_size)
L_  = np.asarray(hist.L)
imm = np.asarray(hist.inverse_mass_matrix[:1])
np.savez(WD+outname, position=pos, xi=xi, step_size=ss, L=L_, inverse_mass_matrix=imm,
         nb=nb, nr=nr, mode=mode)
# quick xi sanity: results-phase no longer sentinel?
res_xi = xi[:, -nr:]
print(f"[{mode}] results-phase xi: median={np.median(res_xi):.3e}, "
      f"frac>1={np.mean(res_xi>1):.3f}, frac==-1(sentinel)={np.mean(res_xi==-1.0):.3f}, max={res_xi.max():.2e}")
print(f"[{mode}] saved {outname}")
