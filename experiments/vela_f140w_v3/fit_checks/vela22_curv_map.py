"""Proposed curvature-adaptive supersampling map for vela22 (CPU, no fit)."""
import sys, json, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from gigalens.jax.experimental.adaptive_supersample import AdaptiveImageData, estimate_psf_sigma, plot_factor_map
from gigalens_research.simtests.system import System
ds_dir = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"
sid = "vela22_cam12_a0.400_rep00"
out = sys.argv[1]
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
system = System.load(ds_dir, sid)
cfg = system.sim_config
psf_sigma = estimate_psf_sigma(system.psf)
print("psf_sigma (px, moment) =", round(psf_sigma, 3), "; FWHM-equivalent =", round(2.355 * psf_sigma * 0.065, 3), '"')
print("sim_config supersample (uniform, from meta) =", cfg.supersample, "; psf_convention =", getattr(cfg, "psf_convention", None))
data = AdaptiveImageData(system.observed_image, cfg, driver="curvature", curvature_kwargs={"psf_sigma": psf_sigma, **kw},
                         background_rms=system.background_rms, exp_time=system.exp_time, sees="all")
g = data.adaptive_grid
fm = np.asarray(g.factor_map)
print(repr(g))
vals, counts = np.unique(fm, return_counts=True)
n_pix = fm.size
print("tier | pixels | frac | points")
tot = 0
for v, c in zip(vals, counts):
    pts = c * (v ** 2 if v >= 1 else 1.0 / (1 / v) ** 2)
    tot += pts
    print(f"{v:>5g} | {c:>6d} | {c / n_pix:6.1%} | {pts:>9.0f}")
print(f"total points {g.n_points:.0f}  (uniform ss=4: {n_pix * 16}, ss=8: {n_pix * 64}); ratio to ss=4 = {g.n_points / (n_pix * 16):.2f}")
img = np.asarray(system.observed_image); err = np.asarray(data.error_map)
snr = img / err
for v in vals:
    m = fm == v
    print(f"tier {v:g}: SNR median {np.median(snr[m]):6.1f}, max {snr[m].max():7.1f}; image max {img[m].max():.3f}")
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img, origin="lower", cmap="inferno", norm=PowerNorm(0.5, vmin=0)); ax[0].set_title(f"{sid} observed")
ax[1].imshow(snr, origin="lower", cmap="inferno", norm=PowerNorm(0.5, vmin=0, vmax=50)); ax[1].set_title("SNR per pixel (clipped at 50)")
plot_factor_map(g, ax=ax[2], title=f"curvature factor map {kw or '(defaults)'}\n{g.n_points:.0f} pts = {g.n_points / (n_pix * 16):.2f} x uniform ss4")
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(out, dpi=110); print("wrote", out)
