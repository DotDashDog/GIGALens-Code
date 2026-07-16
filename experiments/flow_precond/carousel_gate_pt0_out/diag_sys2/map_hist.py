import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel"
sets = {"system-2 (1_2_3_4_5_9)": f"{R}/debug_carousel/1_2_3_4_5_9/map/arrays.npz",
        "dPIE carousel":          f"{R}/messy_tests/dpie/map/arrays.npz"}
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
for k,(tag,p) in enumerate(sets.items()):
    d = np.load(p); lp = d["lp_hist"]; ch = d["chisq_hist"]
    n = len(lp); best = int(np.argmax(lp))
    ax[k,0].plot(lp, lw=.8); ax[k,0].axvline(best, color='r', ls='--', label=f"best step {best}/{n}")
    ax[k,0].set(title=f"{tag}: lp_hist", xlabel="step", ylabel="log posterior"); ax[k,0].legend()
    tail = lp[int(.9*n):]
    ax[k,1].plot(np.arange(int(.9*n), n), tail, lw=.9)
    ax[k,1].set(title=f"{tag}: last 10% (still rising?)", xlabel="step")
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0]
    gain_last10 = lp[-1]-lp[int(.9*n)]
    print(f"{tag}:")
    print(f"   steps={n}  best_step={best}  lp_best={lp[best]:.3f}  lp_final={lp[-1]:.3f}")
    print(f"   last-10% gain = {gain_last10:+.3f} nats   tail slope = {slope:+.4f} nats/step")
    print(f"   => extrapolated further gain if 2x steps: ~{slope*n:+.1f} nats")
    print(f"   chisq final={ch[-1]:.6f}")
plt.tight_layout(); plt.savefig("map_hist_convergence.png", dpi=110)
print("\nsaved map_hist_convergence.png")
