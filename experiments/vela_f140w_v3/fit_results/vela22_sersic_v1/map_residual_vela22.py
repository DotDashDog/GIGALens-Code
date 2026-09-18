"""MAP residual plot + parameter table for the vela22 Sersic fit (CPU; rebuilds the prob model from the campaign)."""
import sys, json, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.system import System
from gigalens_research.simtests.registry import get_inference_builder
R = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/runs/vela22_cam12_a0.400_rep00/fitsersic_v1"
out = sys.argv[1]
spec = _load_campaign("experiments/vela_f140w_v3/fit_sersic_vela22.yaml")
ds = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"
system = System.load(ds, "vela22_cam12_a0.400_rep00")
kw = spec.effective_pipeline_kwargs({"fit": "sersic_v1"})
prob = get_inference_builder(spec.inference.builder)(system, **kw)
z = np.load(R + "/map/arrays.npz")["z_best"]
params = prob.constrained(jnp.asarray(z))
lab = prob.labeled_samples(jnp.asarray(z)[None])
sim = prob.simulators[0]
obs = np.asarray(prob.observed_image); err = np.asarray(prob.error_map)
model = np.asarray(sim.lstsq_simulate(params, jnp.asarray(obs), jnp.asarray(err)))
coeffs = np.asarray(sim.lstsq_simulate(params, jnp.asarray(obs), jnp.asarray(err), return_coeffs=True)).ravel()
stacked = np.asarray(sim.lstsq_simulate(params, jnp.asarray(obs), jnp.asarray(err), return_stacked=True)).squeeze()
lens_light = stacked[..., 0] * coeffs[0]; source = stacked[..., 1] * coeffs[1]
res = (obs - model) / err
chi2 = float((res ** 2).sum()); nu = obs.size - 20
print(f"MAP chi2 = {chi2:.1f}  chi2/nu = {chi2 / nu:.4f}  (nu = {nu}; noise expectation 1.000 +- {np.sqrt(2 / nu):.4f})")
print(f"amplitudes (lstsq): lens light {coeffs[0]:.4g}, source {coeffs[1]:.4g}; source flux in frame {source.sum():.1f} cps (truth 418.1), lens {lens_light.sum():.1f} (truth 1421.2)")
print(f"|res| > 3: {(np.abs(res) > 3).sum()} px (noise expectation {obs.size * 0.0027:.0f}); > 4: {(np.abs(res) > 4).sum()} px; > 5: {(np.abs(res) > 5).sum()} px; max |res| {np.abs(res).max():.2f}")
# annulus statistics: where are the residuals?
yy, xx = np.indices(obs.shape); r = np.hypot(yy - 79.5, xx - 79.5) * 0.065
for lo, hi in [(0, 0.3), (0.3, 0.8), (0.8, 1.4), (1.4, 2.5), (2.5, 6)]:
    m = (r >= lo) & (r < hi); print(f"  r {lo:.1f}-{hi:.1f}\": chi2/n = {(res[m]**2).mean():.3f} over {m.sum()} px, max |res| {np.abs(res[m]).max():.2f}")
# parameter table vs truth
truth = system.truth_x
def T(g, c, p):
    try: return float(truth[g][c][p])
    except Exception: return np.nan
rows = []
for name, val in lab.items():
    v = float(np.asarray(val).ravel()[0])
    parts = name.split("/")  # planes/0/mass/0/theta_E
    g = {"mass": 0, "light": 1 if parts[1] == "0" else 2}[parts[2]]; c = int(parts[3]); p = parts[4]
    rows.append((name, v, T(g, c, p)))
print(f"{'param':40s} {'MAP':>10s} {'truth':>10s} {'diff':>10s}")
for n, v, t in rows: print(f"{n:40s} {v:10.4f} {t:10.4f} {v - t:10.4f}" if np.isfinite(t) else f"{n:40s} {v:10.4f} {'(no truth counterpart)':>21s}")
json.dump({"chi2": chi2, "nu": nu, "chi2_nu": chi2 / nu, "params": {n: [v, t] for n, v, t in rows}, "coeffs": coeffs.tolist()}, open(out.replace(".png", ".json"), "w"), indent=1)
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
norm = PowerNorm(0.5, vmin=0, vmax=obs.max())
ax[0, 0].imshow(obs, origin="lower", cmap="inferno", norm=norm); ax[0, 0].set_title("observed")
ax[0, 1].imshow(model, origin="lower", cmap="inferno", norm=norm); ax[0, 1].set_title(f"MAP model (Sersic source)  chi2/nu = {chi2 / nu:.4f}")
im = ax[0, 2].imshow(res, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5); ax[0, 2].set_title("(obs - model)/sigma, full frame"); plt.colorbar(im, ax=ax[0, 2], fraction=0.046)
ax[1, 0].imshow(source, origin="lower", cmap="inferno", norm=PowerNorm(0.5, vmin=0)); ax[1, 0].set_title("MAP lensed source component")
ax[1, 1].imshow(obs - lens_light, origin="lower", cmap="inferno", norm=PowerNorm(0.5, vmin=0, vmax=(obs - lens_light).max())); ax[1, 1].set_title("observed - MAP lens light")
im2 = ax[1, 2].imshow(res[40:120, 40:120], origin="lower", cmap="RdBu_r", vmin=-5, vmax=5, extent=[40, 120, 40, 120]); ax[1, 2].set_title("residual, central 5.2\""); plt.colorbar(im2, ax=ax[1, 2], fraction=0.046)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(out, dpi=100); print("wrote", out)
