import numpy as np, warnings; warnings.filterwarnings("ignore")
import jax; jax.config.update("jax_enable_x64", True)
import arviz as az
from gigalens_research.inference_utils.posterior import SamplerPosterior
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']  # (8,2000,32)
C,N,P=sz.shape
mine_r=SamplerPosterior._rhat(sz)
mine_e=SamplerPosterior._ess(sz)
# arviz reference
az_r=np.array([max(float(az.rhat(sz[:,:,i],method="rank")),float(az.rhat(sz[:,:,i],method="folded"))) for i in range(P)])
az_e=np.array([float(az.ess(sz[:,:,i],method="bulk")) for i in range(P)])
print(f"R-hat: my max={mine_r.max():.3f}  arviz max={az_r.max():.3f}  max|diff|={np.abs(mine_r-az_r).max():.4f}")
print(f"ESS:   my min={mine_e.min():.0f}  arviz min={az_e.min():.0f}  max rel-diff={np.abs(mine_e-az_e).max()/az_e.mean():.3f}")
print("per-param worst-5 by my R-hat (mine vs arviz):")
for i in np.argsort(mine_r)[::-1][:5]:
    print(f"  p{i:2d}  Rhat mine={mine_r[i]:.3f} az={az_r[i]:.3f}   ESS mine={mine_e[i]:.0f} az={az_e[i]:.0f}")
