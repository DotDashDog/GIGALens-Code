import numpy as np, sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
names=list(np.load(WD:="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"+"names.npy")) if False else list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; C,N,P=sz.shape
def psrf(s):
    Cc,Nn,Pp=s.shape; m=s.mean(1); W=s.var(1,ddof=1).mean(0); Bn=m.var(0,ddof=1)
    return (Nn-1)/Nn + (Cc+1)/Cc*Bn/np.maximum(W,1e-300)
# unconstrained
ru=psrf(sz)
# constrained
s=prob_model.bij.forward(list(jnp.asarray(sz.reshape(-1,P)).T))
con=np.stack([np.asarray(s[n]).reshape(C,N) for n in names],axis=-1)  # (C,N,P) same name order
rc=psrf(con)
print(f"max R-hat  UNCONSTRAINED z = {np.nanmax(ru):.2f}   CONSTRAINED phys = {np.nanmax(rc):.2f}")
print("\nparam                         R-hat(z)   R-hat(phys)")
for i in np.argsort(ru)[::-1][:8]:
    print(f"  {i:2d} {names[i]:26s}  {ru[i]:7.2f}    {rc[i]:7.2f}")
