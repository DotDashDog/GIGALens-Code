"""
Read-only plotting script for ss_max=5-vs-1 ablation (PR1/PR2/PR3).
NO GPU / login-node only. Reads pre-computed .npz arrays and produces
three comparison figures into diag_ssmax/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
OUT = f"{BASE}/diag_ssmax"

FREEZE_ROUND = 500
PRE_FREEZE = (0, 250)
NEAR_FREEZE = (250, 500)

SEEDS = {
    "PR1": dict(
        seed=60,
        ss5=f"{BASE}/arrays_PR_PR1pt5ar2_ssmax5.npz",
        ss1=f"{BASE}/arrays_PR_PR1pt5ar2.npz",
        ss1chk=f"{BASE}/arrays_PR_PR1pt5ar2_ssmax1chk.npz",
    ),
    "PR2": dict(
        seed=61,
        ss5=f"{BASE}/arrays_PR_PR2pt5ar2_ssmax5.npz",
        ss1=f"{BASE}/arrays_PR_PR2pt5ar2.npz",
        ss1chk=None,
    ),
    "PR3": dict(
        seed=62,
        ss5=f"{BASE}/arrays_PR_PR3pt5ar2_ssmax5.npz",
        ss1=f"{BASE}/arrays_PR_PR3pt5ar2.npz",
        ss1chk=None,
    ),
}

SEED_COLORS = {"PR1": "tab:blue", "PR2": "tab:orange", "PR3": "tab:green"}

def rolling_mean(x, window):
    """Simple centered-ish rolling mean via cumulative sum, min_periods=1 at edges."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + (window - half))
        out[i] = np.nanmean(x[lo:hi])
    return out


def load(path):
    return np.load(path)


# ------------------------------------------------------------------
# FIGURE 1: cold-rung step_mean vs round
# ------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5), dpi=110, sharey=True)

for ax, (tag, info) in zip(axes1, SEEDS.items()):
    color = SEED_COLORS[tag]
    d5 = load(info["ss5"])
    d1 = load(info["ss1"])

    n5 = int(d5["rounds_done"])
    n1 = int(d1["rounds_done"])

    cold5 = d5["step_mean"][:n5, -1]
    cold1 = d1["step_mean"][:n1, -1]

    rounds5 = np.arange(n5)
    rounds1 = np.arange(n1)

    ax.plot(rounds5, cold5, color=color, linestyle="-", linewidth=1.6, label="ss_max=5")
    ax.plot(rounds1, cold1, color=color, linestyle="--", linewidth=1.6, label="ss_max=1 (archived)")

    if info["ss1chk"] is not None:
        dchk = load(info["ss1chk"])
        nchk = int(dchk["rounds_done"])
        coldchk = dchk["step_mean"][:nchk, -1]
        roundschk = np.arange(nchk)
        ax.plot(roundschk, coldchk, color=color, linestyle=":", linewidth=1.8, label="ss_max=1 (chk)")

    ax.axvspan(PRE_FREEZE[0], PRE_FREEZE[1], color="grey", alpha=0.15, zorder=0)
    ax.axvline(FREEZE_ROUND, color="black", linestyle="-", linewidth=1.0, alpha=0.7)

    ax.set_title(f"{tag} (seed{info['seed']})")
    ax.set_xlabel("round")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

axes1[0].set_ylabel("cold-rung step_mean")
fig1.suptitle(
    "Realized cold-rung step size: ss_max=5 releases the pre-freeze cap (ss1 pinned ~1.0)",
    fontsize=12,
)
fig1.tight_layout(rect=[0, 0, 1, 0.94])
fig1.savefig(f"{OUT}/ssmax_stepmean.png", dpi=110)
plt.close(fig1)

# ------------------------------------------------------------------
# FIGURE 2: cold-rung pocket occupancy vs round (rolling mean w=25)
# ------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), dpi=110, sharey=True)

for ax, (tag, info) in zip(axes2, SEEDS.items()):
    color = SEED_COLORS[tag]
    d5 = load(info["ss5"])
    d1 = load(info["ss1"])

    n5 = int(d5["rounds_done"])
    n1 = int(d1["rounds_done"])

    occ5 = d5["cold_ind"][:n5].mean(axis=1)
    occ1 = d1["cold_ind"][:n1].mean(axis=1)

    occ5_roll = rolling_mean(occ5, 25)
    occ1_roll = rolling_mean(occ1, 25)

    rounds5 = np.arange(n5)
    rounds1 = np.arange(n1)

    ax.plot(rounds5, occ5_roll, color=color, linestyle="-", linewidth=1.6, label="ss_max=5")
    ax.plot(rounds1, occ1_roll, color=color, linestyle="--", linewidth=1.6, label="ss_max=1 (archived)")

    ax.axvspan(NEAR_FREEZE[0], NEAR_FREEZE[1], color="orange", alpha=0.15, zorder=0)
    ax.axvline(FREEZE_ROUND, color="black", linestyle="-", linewidth=1.0, alpha=0.7)

    ax.set_title(f"{tag} (seed{info['seed']})")
    ax.set_xlabel("round")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

axes2[0].set_ylabel("cold-rung pocket occupancy (rolling mean, w=25)")
fig2.suptitle(
    "Cold-rung occupancy: ss_max=5 did NOT raise occupancy-at-freeze (mechanism is not transport)",
    fontsize=12,
)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig(f"{OUT}/ssmax_occupancy.png", dpi=110)
plt.close(fig2)

# ------------------------------------------------------------------
# FIGURE 3: grouped bar chart of scored W-G max gen-eig
# ------------------------------------------------------------------
GENEIG = {
    "PR1": dict(ss5=32.85, ss1=34.69, n_ss5=2, n_ss1=2),
    "PR2": dict(ss5=25.85, ss1=42.70, n_ss5=2, n_ss1=2),
    "PR3": dict(ss5=28.37, ss1=43.40, n_ss5=2, n_ss1=2),
}
PT4_BAND = (19.7, 27.6)
WG_THRESHOLD = 30

fig3, ax3 = plt.subplots(figsize=(8, 5.5), dpi=110)

tags = list(GENEIG.keys())
x = np.arange(len(tags))
width = 0.35

ss5_vals = [GENEIG[t]["ss5"] for t in tags]
ss1_vals = [GENEIG[t]["ss1"] for t in tags]

bars5 = ax3.bar(x - width / 2, ss5_vals, width, label="ss_max=5", color="tab:blue")
bars1 = ax3.bar(x + width / 2, ss1_vals, width, label="ss_max=1 (archived)", color="tab:gray")

ax3.axhspan(PT4_BAND[0], PT4_BAND[1], color="lightgreen", alpha=0.5, zorder=0, label="PT-4 reference band")
ax3.axhline(WG_THRESHOLD, color="red", linestyle="--", linewidth=1.5, label="W-G threshold (30)")

for bars, tag_key in [(bars5, "n_ss5"), (bars1, "n_ss1")]:
    for bar, tag in zip(bars, tags):
        n_axes = GENEIG[tag][tag_key]
        ax3.annotate(
            f"N axes>10\n= {n_axes}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax3.set_xticks(x)
ax3.set_xticklabels(tags)
ax3.set_ylabel("max generalized eigenvalue (cold rung vs Sigma_ref)")
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(alpha=0.3, axis="y")

fig3.suptitle(
    "Frozen-metric W-G max gen-eig: ss_max=5 cuts worst-axis magnitude on 2/3 (PR2/PR3) into\n"
    "PT-4's band, but axis-count stays 2>10 (gate needs <=1) => no arm passes",
    fontsize=11,
)
fig3.tight_layout(rect=[0, 0, 1, 0.90])
fig3.savefig(f"{OUT}/ssmax_geneig.png", dpi=110)
plt.close(fig3)

print("Wrote:")
print(f"  {OUT}/ssmax_stepmean.png")
print(f"  {OUT}/ssmax_occupancy.png")
print(f"  {OUT}/ssmax_geneig.png")
