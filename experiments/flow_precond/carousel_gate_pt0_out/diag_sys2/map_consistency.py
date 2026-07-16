import numpy as np
R="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/debug_carousel/1_2_3_4_5_9"
z=np.load(f"{R}/mclmc/arrays.npz")["samples_z"].reshape(-1,46)
mz=np.load(f"{R}/map/arrays.npz")["z_best"]
lo,hi,mu,sd=z.min(0),z.max(0),z.mean(0),z.std(0)
inside=(mz>=lo)&(mz<=hi)
print(f"MAP inside pooled MCLMC per-dim range: {inside.sum()}/46 dims")
print(f"outside dims: {np.where(~inside)[0].tolist()}")
zs=(mz-mu)/sd
print(f"\n|z-score| of MAP vs pooled draws: median={np.median(np.abs(zs)):.2f} max={np.abs(zs).max():.2f}")
print("\ndims where MAP is >3 sd from pooled mean:")
for j in np.where(np.abs(zs)>3)[0]:
    print(f"  z[{j:2d}] MAP={mz[j]:+10.4f}  pooled mean={mu[j]:+10.4f} sd={sd[j]:8.4f}  zscore={zs[j]:+7.2f}  range=[{lo[j]:+.3f},{hi[j]:+.3f}]")
print("\n--- per-dim table (all 46) ---")
for j in range(46):
    flag="  <-- OUTSIDE" if not inside[j] else ""
    print(f"z[{j:2d}] MAP={mz[j]:+11.4f} range=[{lo[j]:+9.3f},{hi[j]:+9.3f}] z={zs[j]:+7.2f}{flag}")
