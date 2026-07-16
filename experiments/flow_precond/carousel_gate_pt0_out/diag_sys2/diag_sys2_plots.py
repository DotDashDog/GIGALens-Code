"""Supporting plots for the sys2 ss=1 falsifier attempt (CPU-only, no jax/GPU needed).

compare_to_reference FAILED on every (endpoint, psf_mode) combination (see
diag_sys2_ss1_falsifier.py run log) with a shape-mismatch TypeError before ever
returning an 'adaptive'/'reference'/'delta' image -- so there is NO |delta|/sigma
map to plot. These two plots instead show (a) the all-factor-1 grid really is
uniform plain ss=1 (the thing we tried to falsify with), and (b) the z[37]
draw pools / representative points used, for the record.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out/diag_sys2"
ARCHIVE = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/debug_carousel/1_2_3_4_5_9/mclmc/arrays.npz"

# --- (a) factor map: all-1 grid for band 4-5, 300x300 -----------------------
# (plot_factor_map itself is trivial -- reimplemented inline to avoid importing
# the adaptive_supersample module, which pulls in jax and this CPU-only conda
# env has no jax; the AdaptiveGrid/tier-histogram construction was already
# verified numerically on-GPU in diag_sys2_ss1_falsifier.py's STEP 5 output:
# tiers = {1.0: 90000}, i.e. every one of the 300x300 pixels is factor 1.0.)
fm = np.ones((300, 300), dtype=np.float64)
fig, ax = plt.subplots(figsize=(5.5, 5.0))
im = ax.imshow(fm, cmap="gray", vmin=0, vmax=2, origin="lower", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("sampling factor (uniform 1.0 everywhere)")
ax.set_title("band 4-5 (300x300): all-factor-1 grid used for the (failed)\n"
              "falsifier attempt -- verified on-GPU tiers={1.0: 90000}")
ax.set_xlabel("x [pix]"); ax.set_ylabel("y [pix]")
fig.savefig(f"{OUT}/factor_map_all1_band4-5.png", dpi=110)
plt.close(fig)
print("wrote factor_map_all1_band4-5.png; tiers (from GPU run): {1.0: 90000}")

# --- (b) z[37] traces with THR, stationary segments, and representative draws
z = np.load(ARCHIVE)["samples_z"]  # (8, 10000, 46)
THR = -3.0920
J = 37
arrivals = {3: 3222, 5: 3755, 6: 5579}
z_sharp_rep_37 = -2.784449
z_compact_rep_37 = -3.435570

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
ax = axes[0]
for c in range(8):
    ax.plot(z[c, :, J], lw=0.5, alpha=0.8, label=f"ch{c}")
ax.axhline(THR, color="k", ls="--", lw=1, label=f"THR={THR}")
ax.axhline(z_sharp_rep_37, color="tab:red", ls=":", lw=1.5, label="z_sharp_rep[37]")
ax.axhline(z_compact_rep_37, color="tab:blue", ls=":", lw=1.5, label="z_compact_rep[37]")
ax.set(title="system-2 MCLMC z[37] (planes/3/light/1/beta) traces, all 8 chains",
       xlabel="draw", ylabel="z[37]")
ax.legend(ncol=6, fontsize=7, loc="upper right")

ax = axes[1]
colors = {3: "tab:green", 5: "tab:orange", 6: "tab:purple"}
for c, a in arrivals.items():
    ax.plot(np.arange(a, z.shape[1]), z[c, a:, J], lw=0.6, color=colors[c],
            label=f"ch{c} post-arrival (draw {a}-9999, n={z.shape[1]-a})")
v1 = z[1, :, J]
ax.plot(np.arange(z.shape[1]), v1, lw=0.6, color="gray", alpha=0.6,
        label="ch1 (EXCLUDED -- still drifting)")
ax.axhline(THR, color="k", ls="--", lw=1)
ax.set(title="z_compact stationary segments actually used (chains 3,5,6 "
              "post-arrival) vs excluded chain 1",
       xlabel="draw", ylabel="z[37]")
ax.legend(ncol=2, fontsize=7, loc="upper right")
plt.tight_layout()
fig.savefig(f"{OUT}/z37_pools_and_representatives.png", dpi=110)
plt.close(fig)
print("wrote z37_pools_and_representatives.png")
