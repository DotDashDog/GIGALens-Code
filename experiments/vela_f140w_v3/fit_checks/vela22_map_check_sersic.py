"""Quadrature check of candidate factor maps on a SMOOTH Sersic-source fiducial (the inference model class):
true lens mass + true lens light + Sersic source at the true source position. Reference: uniform ss=16 bin-first."""
import sys, json, dataclasses, copy, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import yaml, tensorflow_probability.substrates.jax as tfp; tfd = tfp.distributions
from scipy import ndimage as ndi
from gigalens.jax.experimental.adaptive_supersample import AdaptiveImageData, AdaptiveGrid, AdaptiveSceneSimulator, plot_factor_map
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens_research.simtests.system import System
from gigalens_research.simtests.experiments.vela_simulated import resolve_truth_prior_spec, _make_dist
from gigalens_research.inference_utils.params import truth_x_to_scene_params

ds_dir = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"
sid = "vela22_cam12_a0.400_rep00"; out = sys.argv[1]
system = System.load(ds_dir, sid); cfg = system.sim_config; cfg1 = dataclasses.replace(cfg, supersample=1)
camp = yaml.safe_load(open("experiments/vela_f140w_v3/campaign.yaml"))["dataset"]
spec = resolve_truth_prior_spec(camp.get("truth_prior"))
def comp_params(group, comp):
    return {p: (float(d["value"]) if d["dist"] == "Fixed" else _make_dist(d, f"{group}.{comp}.{p}")) for p, d in spec[group][comp].items()}
src_p = {k: tfd.Normal(0.0, 1.0) for k in ["Ie", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]}
model = LensModel([
    Plane(mass=[Component(epl.EPL(50), comp_params("lens_mass", "0")), Component(shear.Shear(), comp_params("lens_mass", "1"))],
          light=[Component(sersic.SersicEllipse(use_lstsq=False), comp_params("lens_light", "0"))]),
    Plane(deflection_ratio=1.0, light=[Component(sersic.SersicEllipse(use_lstsq=False), src_p)])])
truth = copy.deepcopy(system.truth_x)
sx, sy = truth[2][0]["center_x"], truth[2][0]["center_y"]
psf_core = 0.1722767 / 0.065 / 2.3548
base = AdaptiveImageData(system.observed_image, cfg, driver="curvature", curvature_kwargs={"psf_sigma": psf_core, "nsig": 4.0},
                         background_rms=system.background_rms, exp_time=system.exp_time, sees="all")
err = np.asarray(base.error_map); img = np.asarray(system.observed_image)
snr_s = ndi.gaussian_filter(img, 1.0) / err
floor2 = np.where(ndi.binary_dilation(snr_s > 3, iterations=2), 2.0, 0.25)
floor4 = np.where(ndi.binary_dilation(snr_s > 3, iterations=2), 4.0, 0.25)
grids = {
  "core_nsig4": base.adaptive_grid,
  "core_nsig4_max8": AdaptiveGrid.from_curvature(img, err, psf_sigma=psf_core, nsig=4.0, max_factor=8.0),
}
REF = 32
ref_sim = SceneSimulator(model, dataclasses.replace(cfg, supersample=REF))
u4_sim = SceneSimulator(model, dataclasses.replace(cfg, supersample=4))
sims = {"uniform_ss4": u4_sim, **{k: AdaptiveSceneSimulator(model, cfg1, g) for k, g in grids.items()}}
res = {}
fids = {"n1_R0.2": dict(R_sersic=0.2, n_sersic=1.0), "n4_R0.2": dict(R_sersic=0.2, n_sersic=4.0), "n4_R0.1": dict(R_sersic=0.1, n_sersic=4.0)}
fig, axes = plt.subplots(len(fids), len(sims), figsize=(3.6 * len(sims), 3.6 * len(fids)))
for i, (fname, fkw) in enumerate(fids.items()):
    t = copy.deepcopy(truth); t[2][0] = dict(Ie=1.0, e1=0.1, e2=-0.1, center_x=sx, center_y=sy, **fkw)
    t_src = copy.deepcopy(t); t_src[1][0]["Ie"] = 0.0          # source only at Ie=1
    t_ll = copy.deepcopy(t); t_ll[2][0]["Ie"] = 0.0            # lens light only
    ref_src = np.asarray(ref_sim.simulate(truth_x_to_scene_params(t_src, model))).squeeze()
    ref_ll = np.asarray(ref_sim.simulate(truth_x_to_scene_params(t_ll, model))).squeeze()
    k = 418.0 / ref_src.sum()                                   # lensed source flux ~ vela22's 418 cps
    ref = ref_ll + k * ref_src
    for j, (mname, sim) in enumerate(sims.items()):
        a = np.asarray(sim.simulate(truth_x_to_scene_params(t_ll, model))).squeeze() + k * np.asarray(sim.simulate(truth_x_to_scene_params(t_src, model))).squeeze()
        d = (a - ref) / err
        iy, ix = np.unravel_index(np.argmax(np.abs(d)), d.shape)
        fm = np.asarray(getattr(sim, "grid", None).factor_map) if hasattr(sim, "grid") else np.full(d.shape, 4.0)
        res[f"{fname}/{mname}"] = dict(max=float(np.abs(d).max()), rms=float(np.sqrt((d**2).mean())), sumsq=float((d**2).sum()),
                                       worst=[int(iy), int(ix)], worst_tier=float(fm[iy, ix]), n_points=int(getattr(sim, "n_points", 160*160*16)))
        pt = {float(t_): float(np.abs(d[fm == t_]).max()) for t_ in np.unique(fm)}; print("   per-tier max:", {k_: round(v_, 3) for k_, v_ in pt.items()})
        print(f"{fname:9s} {mname:22s} max|d|/sig {np.abs(d).max():.3f}  rms {np.sqrt((d**2).mean()):.4f}  sum d^2 {(d**2).sum():7.2f}  worst px {iy},{ix} tier {fm[iy,ix]:g}  pts {res[f'{fname}/{mname}']['n_points']}")
        ax = axes[i, j]; ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=-0.3, vmax=0.3); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{fname} | {mname}\nmax {np.abs(d).max():.2f}", fontsize=8)
plt.tight_layout(); plt.savefig(out, dpi=90); print("wrote", out)
json.dump(res, open(out.replace(".png", ".json"), "w"), indent=1)
fig2, ax2 = plt.subplots(1, len(grids), figsize=(4.5 * len(grids), 4.2))
for a_, (k_, g) in zip(ax2, grids.items()): plot_factor_map(g, ax=a_, title=f"{k_}\n{g.n_points} pts"); a_.set_xticks([]); a_.set_yticks([])
plt.tight_layout(); plt.savefig(out.replace(".png", "_maps.png"), dpi=90)
