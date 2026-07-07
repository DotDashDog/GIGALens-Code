import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
import arviz as az
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
names=list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; C,N,P=sz.shape
# physical
s=prob_model.bij.forward(list(jnp.asarray(sz.reshape(-1,P)).T))
con=np.stack([np.asarray(s[n]).reshape(C,N) for n in names],axis=-1)
def tfp_psrf(a):
    Cc,Nn=a.shape; m=a.mean(1); W=a.var(1,ddof=1).mean(); Bn=m.var(ddof=1)
    return (Nn-1)/Nn+(Cc+1)/Cc*Bn/max(W,1e-300)
rows=[]
for i in range(P):
    z=sz[:,:,i]; ph=con[:,:,i]
    rank_z = float(az.rhat(z, method="rank"))
    rank_ph= float(az.rhat(ph, method="rank"))
    fold   = float(az.rhat(z, method="folded"))
    bess=float(az.ess(z, method="bulk")); tess=float(az.ess(z, method="tail"))
    rows.append((i,names[i],tfp_psrf(z),tfp_psrf(ph),max(rank_z,fold),rank_ph,bess,tess))
print(f"{'idx':>3} {'param':26} {'PSRF_z':>7} {'PSRF_ph':>7} {'rankRhat':>8} {'rankRh_ph':>9} {'bulkESS':>7} {'tailESS':>7}")
for r in sorted(rows,key=lambda r:-r[4]):
    print(f"{r[0]:3d} {r[1]:26} {r[2]:7.2f} {r[3]:7.2f} {r[4]:8.2f} {r[5]:9.2f} {r[6]:7.0f} {r[7]:7.0f}")
arr=np.array([(r[2],r[3],r[4],r[6],r[7]) for r in rows])
print(f"\nMAX  PSRF_z={arr[:,0].max():.2f}  PSRF_phys={arr[:,1].max():.2f}  rank-Rhat={arr[:,2].max():.2f}")
print(f"MIN  bulkESS={arr[:,3].min():.0f}  tailESS={arr[:,4].min():.0f}   (n_chains*draws={C*N})")
print(f"rank-Rhat invariance check (z vs phys identical): max|diff|={max(abs(r[4]-r[5]) for r in rows):.4f}")
