"""Posterior summary for the vela22 Sersic fit: per-parameter mean/std/z vs truth (constrained space),
trace plots of the 4 worst-|z| mass params, and a 4x4 corner of (theta_E, gamma, e1, gamma2) with truth."""
import sys, json, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.system import System
from gigalens_research.simtests.registry import get_inference_builder
R = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/runs/vela22_cam12_a0.400_rep00/fitsersic_v1"
out = sys.argv[1]
spec = _load_campaign("experiments/vela_f140w_v3/fit_sersic_vela22.yaml")
system = System.load("/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset", "vela22_cam12_a0.400_rep00")
prob = get_inference_builder(spec.inference.builder)(system, **spec.effective_pipeline_kwargs({"fit": "sersic_v1"}))
S = np.load(R + "/mclmc/arrays.npz")["samples_z"]           # (8, 5000, 20)
nch, nst, D = S.shape
lab = prob.labeled_samples(jnp.asarray(S.reshape(-1, D)))
names = list(lab.keys()); X = np.stack([np.asarray(lab[k]).reshape(nch, nst) for k in names], axis=-1)  # (8, 5000, P)
truth = system.truth_x
def T(name):
    p = name.split("/"); g = {"mass": 0, "light": 1 if p[1] == "0" else 2}[p[2]]
    try: return float(truth[g][int(p[3])][p[4]])
    except Exception: return np.nan
rows = []
print(f"{'param':36s} {'mean':>9s} {'std':>9s} {'truth':>9s} {'z':>7s}")
for i, n in enumerate(names):
    v = X[..., i].ravel(); m, s, t = v.mean(), v.std(), T(n); z = (m - t) / s if np.isfinite(t) else np.nan
    rows.append((n, float(m), float(s), float(t), float(z))); print(f"{n:36s} {m:9.4f} {s:9.4f} {t:9.4f} {z:7.2f}" if np.isfinite(t) else f"{n:36s} {m:9.4f} {s:9.4f} {'-':>9s} {'-':>7s}")
json.dump({n: dict(mean=m, std=s, truth=t, z=z) for n, m, s, t, z in rows}, open(out.replace(".png", ".json"), "w"), indent=1)
# per-chain means spread vs within-chain std (a cheap visual R-hat check)
sel = [n for n in names if "/mass/" in n]; zs = {n: r[4] for n, r in zip(names, rows)}
worst = sorted(sel, key=lambda n: -abs(zs[n]))[:4]
fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
for a, n in zip(ax, worst):
    i = names.index(n)
    for c in range(nch): a.plot(X[c, :, i], lw=0.3, alpha=0.7)
    a.axhline(T(n), color="k", ls="--", lw=1); a.set_ylabel(n.split("/")[-1]); a.set_title(f"{n}: z = {zs[n]:+.1f}, truth dashed", fontsize=9)
ax[-1].set_xlabel("MCLMC step (after burn-in)"); plt.tight_layout(); plt.savefig(out.replace(".png", "_traces.png"), dpi=90)
# 4x4 corner with truth
keys = ["planes/0/mass/0/theta_E", "planes/0/mass/0/gamma", "planes/0/mass/0/e1", "planes/0/mass/1/gamma2"]
idx = [names.index(k) for k in keys]; F = X.reshape(-1, X.shape[-1])[:, idx]; tr = [T(k) for k in keys]
fig, ax = plt.subplots(4, 4, figsize=(11, 11))
for i in range(4):
    for j in range(4):
        a = ax[i, j]
        if j > i: a.axis("off"); continue
        if i == j:
            a.hist(F[:, i], bins=60, color="C0", histtype="stepfilled", alpha=0.6); a.axvline(tr[i], color="r", lw=1.5)
            a.set_yticks([])
        else:
            a.hist2d(F[:, j], F[:, i], bins=60, cmap="Blues"); a.plot(tr[j], tr[i], "r*", ms=12)
        if i == 3: a.set_xlabel(keys[j].split("/")[-1] + ("" if "mass/0" in keys[j] else " (shear)"))
        if j == 0 and i > 0: a.set_ylabel(keys[i].split("/")[-1])
        a.tick_params(labelsize=7)
fig.suptitle("vela22 Sersic-source fit: lens-mass posterior (8 x 5000 draws), red = truth", y=0.995)
plt.tight_layout(); plt.savefig(out, dpi=90); print("wrote", out)
