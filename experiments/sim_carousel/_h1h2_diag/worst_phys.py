import numpy as np, sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
names=list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; C,N,P=sz.shape
def psrf(s):
    Cc,Nn,Pp=s.shape; m=s.mean(1); W=s.var(1,ddof=1).mean(0); Bn=m.var(0,ddof=1)
    return (Nn-1)/Nn+(Cc+1)/Cc*Bn/np.maximum(W,1e-300)
s=prob_model.bij.forward(list(jnp.asarray(sz.reshape(-1,P)).T))
con=np.stack([np.asarray(s[n]).reshape(C,N) for n in names],axis=-1)
rc=psrf(con); ru=psrf(sz)
print("worst params by CONSTRAINED (physical) R-hat:")
for i in np.argsort(rc)[::-1][:6]:
    cmean=con[:,:,i].mean(1); wstd=np.median(con[:,:,i].std(1))
    print(f"  {i:2d} {names[i]:26s} Rhat_phys={rc[i]:6.2f} Rhat_z={ru[i]:6.2f}")
    print(f"       chain means (phys): {np.array2string(np.sort(cmean),precision=4,floatmode='fixed')}  within-std {wstd:.4f}")
