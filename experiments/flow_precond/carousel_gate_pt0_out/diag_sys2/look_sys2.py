import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/debug_carousel/1_2_3_4_5_9/mclmc/arrays.npz"
z=np.load(D)["samples_z"]; print("samples_z",z.shape,z.dtype)
j=37
fig,ax=plt.subplots(2,1,figsize=(11,7))
for c in range(z.shape[0]):
    ax[0].plot(z[c,:,j],lw=.4,alpha=.8,label=f"ch{c}")
ax[0].set(title=f"1_2_3_4_5_9 MCLMC: z[{j}] traces (8 chains x 10000)",xlabel="draw",ylabel=f"z[{j}]")
ax[0].legend(ncol=8,fontsize=7)
for c in range(z.shape[0]):
    ax[1].hist(z[c,:,j],bins=60,histtype="step",lw=1.2,label=f"ch{c}")
ax[1].set(title=f"per-chain marginals z[{j}] (segregation => non-overlapping)",xlabel=f"z[{j}]")
ax[1].legend(ncol=8,fontsize=7)
plt.tight_layout(); plt.savefig("sys2_z37.png",dpi=110)
# per-chain split at the pooled 2-means boundary, computed transparently
from sklearn.cluster import KMeans
x=z[:,:,j].reshape(-1,1)
km=KMeans(2,n_init=20,random_state=0).fit(x)
cen=np.sort(km.cluster_centers_.ravel()); thr=cen.mean()
print(f"pooled 2-means centers={cen}, boundary={thr:.4f}")
print("per-chain fraction ABOVE boundary:")
for c in range(z.shape[0]):
    f=(z[c,:,j]>thr).mean(); print(f"  chain{c}: {f:.4f}   range=[{z[c,:,j].min():.3f},{z[c,:,j].max():.3f}]")
print("pooled fraction above:", (z[:,:,j]>thr).mean().round(4))
