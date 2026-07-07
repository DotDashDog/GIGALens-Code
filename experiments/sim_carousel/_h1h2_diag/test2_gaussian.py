import numpy as np, sys, jax, jax.numpy as jnp, warnings, time
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
import arviz as az
import tensorflow_probability.substrates.jax as tfp; tfd=tfp.distributions
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from gigalens_research.inference import MCLMC_JIT
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; C,N,P=sz.shape
mu=jnp.asarray(sz.reshape(-1,P).mean(0))
Sig=np.cov(sz.reshape(-1,P).T); Sig=0.5*(Sig+Sig.T)+1e-10*np.eye(P)
w=np.linalg.eigvalsh(Sig); print(f"Gaussian target: dim={P} cond(Sigma)={w[-1]/w[0]:.2e} (matches real posterior)")
cho=jax.scipy.linalg.cholesky(jnp.asarray(Sig),lower=True)
class FakeProb:
    def log_prob(self,z):
        d=z-mu; sol=jax.scipy.linalg.cho_solve((cho,True),d)
        return (-0.5*jnp.dot(d,sol),)
class FakeSeq: prob_model=FakeProb()
# same init style as real run (tight ball at mu), same sampler settings
qz=tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.full(P,1e-2))
t0=time.perf_counter()
hist=MCLMC_JIT(model_seq=FakeSeq(), qz=qz, n_hmc=8, num_burnin_steps=10000, num_results=2000,
    desired_energy_variance=5e-4, regularize_mass_matrix=True, progress_bar=False, seed=42, debug_output=True)
print(f"Gaussian run wall {time.perf_counter()-t0:.1f}s")
gz=np.asarray(hist.position[:,-2000:,:]); gxi=np.asarray(hist.xi[:,-2000:])
# diagnostics
def tfp_psrf(a):
    Cc,Nn=a.shape; m=a.mean(1); W=a.var(1,ddof=1).mean(); Bn=m.var(ddof=1)
    return (Nn-1)/Nn+(Cc+1)/Cc*Bn/max(W,1e-300)
rrank=np.array([float(az.rhat(gz[:,:,i],method="rank")) for i in range(P)])
psrf=np.array([tfp_psrf(gz[:,:,i]) for i in range(P)])
bess=np.array([float(az.ess(gz[:,:,i],method="bulk")) for i in range(P)])
tess=np.array([float(az.ess(gz[:,:,i],method="tail")) for i in range(P)])
print("\n=== GAUSSIAN REFERENCE (same cond, same settings) ===")
print(f"  max PSRF(z)   = {psrf.max():.2f}   (real run: 11.6)")
print(f"  max rank-Rhat = {rrank.max():.2f}   (real run: 1.73)")
print(f"  min bulk-ESS  = {bess.min():.0f}    (real run: 12)")
print(f"  min tail-ESS  = {tess.min():.0f}    (real run: 23)")
print(f"  xi: median={np.median(gxi):.2e} frac>1={np.mean(gxi>1):.3f} max={gxi.max():.2e}  (real: med 0.41, frac>1 0.42)")
