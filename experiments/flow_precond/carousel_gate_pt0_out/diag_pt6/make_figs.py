"""PT-6 discovery-vs-propagation diagnostic figures + numeric table.
Read-only analysis of completed arrays_D2_pt6_{tag}.npz. No new runs.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
DIAG = OUTDIR + "/diag_pt6"

TAGS = ["Lcert", "La", "Lb", "Lc"]
LABELS = {"Lcert": "L-cert (certified, 6 rungs)",
          "La": "L-a (geomspace 0.3594-1, 6 rungs)",
          "Lb": "L-b (geomspace 0.10-1, 8 rungs)",
          "Lc": "L-c (geomspace 0.05-1, 10 rungs)"}
COLORS = {"Lcert": "#1b9e77", "La": "#7570b3", "Lb": "#d95f02", "Lc": "#e7298a"}

data = {t: np.load(f"{OUTDIR}/arrays_D2_pt6_{t}.npz") for t in TAGS}

def roll_mean(x, w):
    if len(x) < w:
        return np.array([])
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0))
    return (c[w:] - c[:-w]) / w

# ---------- Figure 1: cold-rung pocket occupancy vs round ----------
fig, ax = plt.subplots(figsize=(8, 5))
w = 25
for t in TAGS:
    cold_ind = data[t]["cold_ind"]  # (rounds, nsys)
    occ = cold_ind.mean(axis=1).astype(np.float64)
    rm = roll_mean(occ, w)
    rounds = np.arange(w - 1, w - 1 + len(rm))
    ax.plot(rounds, rm, label=LABELS[t], color=COLORS[t], lw=1.6)
ax.axhline(0.2, color="k", ls="--", lw=1, label="transport threshold 0.2")
ax.axhspan(0.32, 0.49, color="gray", alpha=0.15, label="certified band 0.32-0.49")
ax.set_xlabel("round")
ax.set_ylabel(f"cold-rung pocket occupancy (rolling mean, w={w})")
ax.set_title("PT-6: cold-rung pocket occupancy vs round (all arms, D2)")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(-0.02, 0.65)
fig.tight_layout()
fig.savefig(f"{DIAG}/pt6_occupancy.png", dpi=110)
plt.close(fig)

# ---------- Figure 2: per-boundary cumulative swap acceptance ----------
fig, ax = plt.subplots(figsize=(8, 5))
for t in TAGS:
    att = data[t]["swap_attempts"]  # (nrung-1, 2)
    acc = data[t]["swap_accepts"]
    rate = acc.sum(axis=1) / att.sum(axis=1)
    bidx = np.arange(len(rate))
    ax.plot(bidx, rate, marker="o", label=LABELS[t], color=COLORS[t], lw=1.6)
ax.axhline(0.23, color="k", ls="--", lw=1, label="typical PT target ~0.23")
ax.set_xlabel("boundary index (0 = hottest pair)")
ax.set_ylabel("cumulative swap acceptance rate")
ax.set_title("PT-6: per-boundary cumulative swap acceptance (all arms, D2)")
ax.legend(fontsize=8, loc="center right")
ax.set_ylim(0, 0.65)
fig.tight_layout()
fig.savefig(f"{DIAG}/pt6_swap_accept.png", dpi=110)
plt.close(fig)

# ---------- Figure 3: per-rung pocket occupancy vs beta (window [500:1000]) ----------
# ind_thin: (n_thin, nrung, nsys), thinned every THIN_B=5 rounds. Confirmed
# (verified numerically) ind_thin[:, -1, :] == cold_ind at thinned rounds,
# ind_thin[:, 0, :] == the harness's own "hot occ" printed quantity.
# A true per-rung indicator EXISTS (ind_thin) -> use it directly; no log
# fallback needed.
fig, ax = plt.subplots(figsize=(8, 5))
WIN_LO, WIN_HI = 500, 1000
report_perrung = {}
for t in TAGS:
    betas = data[t]["betas"]
    ind_thin = data[t]["ind_thin"]  # (n_thin, nrung, nsys)
    n_thin = ind_thin.shape[0]
    thin_rounds = np.arange(0, 5 * n_thin, 5)
    sel = (thin_rounds >= WIN_LO) & (thin_rounds < WIN_HI)
    occ_per_rung = ind_thin[sel].astype(np.float64).mean(axis=(0, 2))  # (nrung,)
    report_perrung[t] = occ_per_rung
    ax.plot(betas, occ_per_rung, marker="o", label=LABELS[t], color=COLORS[t], lw=1.6)
ax.set_xlabel(r"rung $\beta$")
ax.set_ylabel(f"pocket occupancy, time-avg over rounds [{WIN_LO}:{WIN_HI})")
ax.set_title("PT-6: per-rung pocket occupancy vs beta\n(discovery-without-propagation check; ind_thin per-rung indicator)")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(f"{DIAG}/pt6_discovery.png", dpi=110)
plt.close(fig)

# ---------- Numeric report table ----------
print(f"{'arm':8s} {'n_rungs':7s} {'cold_occ[500:1000]':20s} {'hot_occ[500:1000]':18s} "
      f"{'RT_pocket':10s} {'swap_acc_min':13s} {'swap_acc_mean':14s} {'n_revert':9s}")
rows = []
for t in TAGS:
    d = data[t]
    betas = d["betas"]
    nrung = len(betas)
    cold_ind = d["cold_ind"]
    cold_occ_win = cold_ind[WIN_LO:WIN_HI].mean()
    hot_rung_occ_series = report_perrung[t][0]  # rung 0 (hottest), time-avg [500:1000)
    rt_pocket = int(d["round_trips_pocket"].sum())
    att = d["swap_attempts"]; acc = d["swap_accepts"]
    rate = acc.sum(axis=1) / att.sum(axis=1)
    n_revert = int(d["n_revert"])
    print(f"{t:8s} {nrung:7d} {cold_occ_win:20.4f} {hot_rung_occ_series:18.4f} "
          f"{rt_pocket:10d} {rate.min():13.4f} {rate.mean():14.4f} {n_revert:9d}")
    rows.append((t, nrung, cold_occ_win, hot_rung_occ_series, rt_pocket,
                 rate.min(), rate.mean(), n_revert, report_perrung[t].tolist(), betas.tolist()))

print()
print("Full per-rung occupancy [500:1000) vectors (rung 0 = hottest .. rung -1 = coldest):")
for t in TAGS:
    print(f"  {t:8s} betas={np.round(data[t]['betas'],4).tolist()}")
    print(f"  {'':8s} occ  ={np.round(report_perrung[t],4).tolist()}")

import json
with open(f"{DIAG}/pt6_report_table.json", "w") as f:
    json.dump({r[0]: dict(n_rungs=r[1], cold_occ_500_1000=r[2], hot_rung_occ_500_1000=r[3],
                           RT_pocket=r[4], swap_acc_min=r[5], swap_acc_mean=r[6],
                           n_revert=r[7], per_rung_occ_500_1000=r[8], betas=r[9])
               for r in rows}, f, indent=1)

print("\nFigures written:")
print(f"  {DIAG}/pt6_occupancy.png")
print(f"  {DIAG}/pt6_swap_accept.png")
print(f"  {DIAG}/pt6_discovery.png")
