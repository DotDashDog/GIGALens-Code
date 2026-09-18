"""Falsifier for the vela22 factor map: render the TRUE scene (VELA image source + true lens)
on the adaptive grid and compare with (a) the ss=32 noiseless truth on disk and (b) a uniform
ss=8 bin-first reference. Also the uniform ss=4 render (the stock inference quadrature)."""
import sys, json, dataclasses, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import yaml
from gigalens.jax.experimental.adaptive_supersample import (AdaptiveImageData, AdaptiveSceneSimulator,
    compare_to_reference, estimate_psf_sigma, plot_factor_map)
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens_research.simtests.system import System
from gigalens_research.simtests.experiments.vela_simulated import (_load_pristine_source, preprocess_source,
    resolve_truth_prior_spec, _build_truth_scene_model)
from gigalens_research.simulations.image_based_light import ImageBasedLight
from gigalens_research.inference_utils.params import truth_x_to_scene_params

ds_dir = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"
sid = "vela22_cam12_a0.400_rep00"; out = sys.argv[1]
variants = json.loads(sys.argv[2])
system = System.load(ds_dir, sid); cfg = system.sim_config
d = f"{ds_dir}/systems/{sid}/"
noiseless = np.load(d + "noiseless_image.npy").astype(np.float64)
camp = yaml.safe_load(open("experiments/vela_f140w_v3/campaign.yaml"))["dataset"]
prior_spec = resolve_truth_prior_spec(camp.get("truth_prior"))
sb_raw, src_scale, _, _ = _load_pristine_source(json.load(open(d + "meta.json"))["truth_assets"]["vela_source_dir"], False)
sb, _ = preprocess_source(sb_raw, src_scale, crop_radius_arcsec=None, recenter=True, smooth_sigma_pix=None)
model = _build_truth_scene_model(prior_spec, ImageBasedLight(sb, src_scale))
params = truth_x_to_scene_params(system.truth_x, model)
psf_core = 0.1722767 / 0.065 / 2.3548
print(f"psf sigma: moment {estimate_psf_sigma(system.psf):.3f} px, core (FWHM) {psf_core:.3f} px; truth cfg psf_convention={cfg.psf_convention}, ss={cfg.supersample}")
base = AdaptiveImageData(system.observed_image, cfg, driver="curvature", curvature_kwargs={"psf_sigma": psf_core},
                         background_rms=system.background_rms, exp_time=system.exp_time, sees="all")
err = np.asarray(base.error_map)
cfg1 = dataclasses.replace(cfg, supersample=1)
# uniform ss=4 (stock inference quadrature) vs ss32 truth
u4 = np.asarray(SceneSimulator(model, dataclasses.replace(cfg, supersample=4)).simulate(params)).squeeze()
d4 = (u4 - noiseless) / err
print(f"uniform ss=4 vs ss32 truth: max|d|/sigma {np.abs(d4).max():.3f}, rms {np.sqrt((d4**2).mean()):.4f}, sum d^2 {np.sum(d4**2):.2f}")
res = {}
fig, axes = plt.subplots(2, len(variants) + 1, figsize=(4.2 * (len(variants) + 1), 8))
axes[0, 0].imshow(noiseless, origin="lower", cmap="inferno", norm=PowerNorm(0.5, vmin=0)); axes[0, 0].set_title("ss=32 truth (noiseless)")
im = axes[1, 0].imshow(d4, origin="lower", cmap="RdBu_r", vmin=-0.3, vmax=0.3); axes[1, 0].set_title(f"uniform ss4 - truth, /sigma  max {np.abs(d4).max():.2f}")
for j, (name, kw) in enumerate(variants.items(), start=1):
    kw = dict(kw); kw.setdefault("psf_sigma", psf_core)
    data = AdaptiveImageData(system.observed_image, cfg, driver="curvature", curvature_kwargs=kw,
                             background_rms=system.background_rms, exp_time=system.exp_time, sees="all")
    g = data.adaptive_grid
    asim = AdaptiveSceneSimulator(model, cfg1, g)
    a = np.asarray(asim.simulate(params)).squeeze()
    da = (a - noiseless) / err
    cmp8 = compare_to_reference(asim, params, reference_supersample=8, error_map=err, psf_mode="bin_first")
    d8 = np.asarray(cmp8["delta_over_sigma"]).squeeze()
    iy, ix = np.unravel_index(np.argmax(np.abs(da)), da.shape)
    fm = np.asarray(g.factor_map)
    res[name] = dict(n_points=int(g.n_points), max_vs_truth32=float(np.abs(da).max()), rms_vs_truth32=float(np.sqrt((da**2).mean())),
                     sumsq_vs_truth32=float(np.sum(da**2)), max_vs_ref8=float(np.abs(d8).max()), worst_pix=[int(iy), int(ix)],
                     worst_tier=float(fm[iy, ix]), worst_snr=float(system.observed_image[iy, ix] / err[iy, ix]),
                     tiers={str(k): int(v) for k, v in zip(*np.unique(fm, return_counts=True))})
    # worst residual per tier
    per_tier = {float(t): float(np.abs(da[fm == t]).max()) for t in np.unique(fm)}
    res[name]["max_vs_truth32_per_tier"] = per_tier
    print(name, json.dumps(res[name]))
    plot_factor_map(g, ax=axes[0, j], title=f"{name}\n{g.n_points} pts")
    axes[1, j].imshow(da, origin="lower", cmap="RdBu_r", vmin=-0.3, vmax=0.3); axes[1, j].set_title(f"adaptive - truth, /sigma  max {np.abs(da).max():.2f}")
for a_ in axes.ravel(): a_.set_xticks([]); a_.set_yticks([])
plt.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, label="(render - ss32 truth) / sigma")
plt.savefig(out, dpi=100, bbox_inches="tight"); print("wrote", out)
json.dump(res, open(out.replace(".png", ".json"), "w"), indent=1)
