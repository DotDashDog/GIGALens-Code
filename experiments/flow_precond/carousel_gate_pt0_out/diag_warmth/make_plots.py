import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
OUT = os.path.join(BASE, "diag_warmth")

FILES = {
    "PT4_G1": ("arrays_D2_G1pt4.npz", "PT4"),
    "PT4_G2": ("arrays_D2_G2pt4.npz", "PT4"),
    "PT4_G3": ("arrays_D2_G3pt4.npz", "PT4"),
    "PT4_G4": ("arrays_D1_G4pt4.npz", "PT4"),
    "PR1": ("arrays_PR_PR1pt5ar2.npz", "PR"),
    "PR2": ("arrays_PR_PR2pt5ar2.npz", "PR"),
    "PR3": ("arrays_PR_PR3pt5ar2.npz", "PR"),
}

data = {}
for k, (fn, fam) in FILES.items():
    d = np.load(os.path.join(BASE, fn))
    data[k] = dict(betas=d["betas"], att=d["swap_attempts"], acc=d["swap_accepts"],
                    cold_ind=d["cold_ind"], family=fam)

# fixed categorical colors: PT4 family = blue shades, PR family = orange/red shades
pt4_colors = ["#1f5fa8", "#3f7fc2", "#5f9fdc", "#7fbff6"]
pr_colors = ["#c2410c", "#e2662c", "#f4874c"]
pt4_keys = [k for k in FILES if FILES[k][1] == "PT4"]
pr_keys = [k for k in FILES if FILES[k][1] == "PR"]

# ---------- Plot (a): realized swap acceptance vs boundary ----------
fig, ax = plt.subplots(figsize=(7.5, 5))
for i, k in enumerate(pt4_keys):
    att, acc = data[k]["att"], data[k]["acc"]
    p = acc.sum(axis=1) / att.sum(axis=1)
    x = np.arange(len(p))
    ax.plot(x, p, "-o", color=pt4_colors[i], lw=2, ms=5, label=f"{k} (PT-4)")
for i, k in enumerate(pr_keys):
    att, acc = data[k]["att"], data[k]["acc"]
    p = acc.sum(axis=1) / att.sum(axis=1)
    x = np.arange(len(p))
    ax.plot(x, p, "--s", color=pr_colors[i], lw=2, ms=5, label=f"{k} (PR)")

# mark cold-end boundaries (last boundary index for 5-boundary and 6-boundary arms)
ax.axvspan(3.5, 5.5, color="0.85", zorder=0, label="cold-end boundaries (nearest beta=1)")
ax.set_xlabel("adjacent-rung boundary index (0 = hottest pair)")
ax.set_ylabel("realized swap acceptance\n(cumulative over full run)")
ax.set_title("Realized swap acceptance per boundary: PT-4 vs PR arms")
ax.set_ylim(0.35, 0.75)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "swap_accept_vs_boundary.png"), dpi=150)
plt.close(fig)

# ---------- Plot (b): occupancy trace vs round ----------
fig, ax = plt.subplots(figsize=(8, 5))


def rolling(x, w):
    n = len(x) // w * w
    return x[:n].reshape(-1, w).mean(axis=1)


W = 25
for i, k in enumerate(pt4_keys):
    ci = data[k]["cold_ind"].mean(axis=1)  # per-round occupancy over 16 systems
    y = rolling(ci, W)
    xr = np.arange(len(y)) * W + W / 2
    ax.plot(xr, y, "-", color=pt4_colors[i], lw=1.6, label=f"{k} (PT-4)")
for i, k in enumerate(pr_keys):
    ci = data[k]["cold_ind"].mean(axis=1)
    y = rolling(ci, W)
    xr = np.arange(len(y)) * W + W / 2
    ax.plot(xr, y, "--", color=pr_colors[i], lw=1.6, label=f"{k} (PR)")

ax.axvline(500, color="k", lw=1.2, ls=":", label="metric freeze (round 500)")
ax.set_xlabel("round")
ax.set_ylabel(f"cold-rung pocket occupancy\n(rolling mean, w={W} rounds)")
ax.set_title("Pocket occupancy vs round: PT-4 vs PR arms")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "occupancy_vs_round.png"), dpi=150)
plt.close(fig)

print("wrote:")
print(os.path.join(OUT, "swap_accept_vs_boundary.png"))
print(os.path.join(OUT, "occupancy_vs_round.png"))
