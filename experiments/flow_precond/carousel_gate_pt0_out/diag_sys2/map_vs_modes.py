import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/debug_carousel/1_2_3_4_5_9"
z=np.load(f"{R}/mclmc/arrays.npz")["samples_z"]; mz=np.load(f"{R}/map/arrays.npz")["z_best"]
print("z_best",mz.shape,"z[37]_MAP=",round(float(mz[37]),4))
thr=-3.092; lab=(z[:,:,37]>thr)   # right(sharp)=True
print("MAP is in", "RIGHT/sharp mode" if mz[37]>thr else "LEFT/broad mode")
# which dims separate the two modes? standardized center gap using pooled draws
flat=z.reshape(-1,46); m=lab.reshape(-1)
gap=np.abs(flat[m].mean(0)-flat[~m].mean(0))/flat.std(0)
top=np.argsort(gap)[::-1][:8]
print("\ntop dims by standardized between-mode center gap:")
for j in top: print(f"  z[{j:2d}]  gap={gap[j]:.3f} sd   MAP={mz[j]:+.4f}  right_mean={flat[m][:,j].mean():+.4f} left_mean={flat[~m][:,j].mean():+.4f}")
j2=int(top[1]) if int(top[0])==37 else int(top[0])
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(flat[m][::20,37],flat[m][::20,j2],'.',ms=1.5,alpha=.3,label="right/sharp")
ax[0].plot(flat[~m][::20,37],flat[~m][::20,j2],'.',ms=1.5,alpha=.3,label="left/broad")
ax[0].plot(mz[37],mz[j2],'r*',ms=22,mec='k',label="MAP")
ax[0].set(xlabel="z[37]",ylabel=f"z[{j2}]",title=f"two modes in (z[37], z[{j2}]) + MAP"); ax[0].legend()
ax[1].hist(flat[:,37],bins=120,color='0.7'); ax[1].axvline(mz[37],color='r',lw=2,label="MAP"); ax[1].axvline(thr,color='b',ls='--',label="2-means boundary")
ax[1].set(xlabel="z[37]",title="pooled marginal + MAP"); ax[1].legend()
plt.tight_layout(); plt.savefig("sys2_map_vs_modes.png",dpi=110)
