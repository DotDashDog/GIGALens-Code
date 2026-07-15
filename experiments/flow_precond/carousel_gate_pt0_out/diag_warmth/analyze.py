import numpy as np
import json
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
    data[k] = dict(
        betas=d["betas"],
        att=d["swap_attempts"],
        acc=d["swap_accepts"],
        cold_ind=d["cold_ind"],
        rounds_done=int(d["rounds_done"]),
        metric_windows=d["metric_windows"].tolist(),
        family=fam,
    )

print("=== Per-arm ladder + boundary count ===")
for k, v in data.items():
    R = len(v["betas"])
    print(f"{k:8s} fam={v['family']:3s} R={R} boundaries={R-1} "
          f"rounds_done={v['rounds_done']} freeze_windows={v['metric_windows']} "
          f"betas={np.round(v['betas'],4).tolist()}")

print()
print("=== Realized swap acceptance per boundary (CUMULATIVE over full run: "
      "att/acc are running totals accumulated from round 0 through the last "
      "saved round -- there is no per-round or windowed breakdown stored in "
      "these npz files, so this is the only 'realized swap acceptance' "
      "available; see caveat in report) ===")
table = {}
for k, v in data.items():
    att, acc = v["att"], v["acc"]  # (n_boundary, 2) [same, cross]
    att_tot = att.sum(axis=1)
    acc_tot = acc.sum(axis=1)
    p = acc_tot / att_tot
    table[k] = p
    print(f"{k:8s} " + " ".join(f"{x:.4f}" for x in p))

print()
print("=== Cold-end vs warm-end ratio (last boundary / first boundary; "
      "last boundary = nearest beta=1.0) ===")
for k, p in table.items():
    print(f"{k:8s} cold(last)={p[-1]:.4f} warm(first)={p[0]:.4f} "
          f"ratio(cold/warm)={p[-1]/p[0]:.3f} "
          f"ratio(cold/boundary-mean)={p[-1]/p.mean():.3f}")

# save table to json for reuse in plotting
np.savez(os.path.join(OUT, "swap_table.npz"),
          **{k: v for k, v in table.items()})

print()
print("=== Occupancy (cold_ind mean over all 16 systems) last-500-round avg ===")
for k, v in data.items():
    ci = v["cold_ind"]
    last = min(500, ci.shape[0])
    occ = ci[-last:].mean()
    print(f"{k:8s} occ_last{last}={occ:.4f} occ_at_round500="
          f"{ci[:500].mean(axis=1)[-1] if ci.shape[0]>=500 else float('nan'):.4f}")
