"""Is the 0.46 sigma at the lens centre under factor 8 a quadrature-convergence property of the n=5.2 cusp, or an adaptive tier-8 geometry bug?
Render LENS LIGHT ONLY at the truth: uniform ss in {1,2,4,8,16,32,64} and adaptive all-8 / all-4 maps; report centre pixels relative to ss=64."""
import sys, dataclasses, copy, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import yaml, tensorflow_probability.substrates.jax as tfp; tfd = tfp.distributions
from gigalens.jax.experimental.adaptive_supersample import AdaptiveGrid, AdaptiveSceneSimulator
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens_research.simtests.system import System
from gigalens_research.simtests.experiments.vela_simulated import resolve_truth_prior_spec, _make_dist
from gigalens_research.inference_utils.params import truth_x_to_scene_params
ds_dir = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"; sid = "vela22_cam12_a0.400_rep00"
system = System.load(ds_dir, sid); cfg = system.sim_config
spec = resolve_truth_prior_spec(yaml.safe_load(open("experiments/vela_f140w_v3/campaign.yaml"))["dataset"].get("truth_prior"))
def cp(g, c): return {p: (float(d["value"]) if d["dist"] == "Fixed" else _make_dist(d, p)) for p, d in spec[g][c].items()}
src_p = {k: tfd.Normal(0.0, 1.0) for k in ["Ie", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]}
model = LensModel([Plane(mass=[Component(epl.EPL(50), cp("lens_mass", "0")), Component(shear.Shear(), cp("lens_mass", "1"))],
                         light=[Component(sersic.SersicEllipse(use_lstsq=False), cp("lens_light", "0"))]),
                   Plane(deflection_ratio=1.0, light=[Component(sersic.SersicEllipse(use_lstsq=False), src_p)])])
t = copy.deepcopy(system.truth_x); t[2][0] = dict(Ie=0.0, R_sersic=0.2, n_sersic=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
params = truth_x_to_scene_params(t, model)
err = np.sqrt(system.background_rms ** 2 + np.clip(np.asarray(system.observed_image), 0, None) / system.exp_time)
imgs = {}
for ss in [1, 2, 4, 8, 16, 32, 64]:
    imgs[f"u{ss}"] = np.asarray(SceneSimulator(model, dataclasses.replace(cfg, supersample=ss)).simulate(params)).squeeze()
cfg1 = dataclasses.replace(cfg, supersample=1)
for f in [4.0, 8.0]:
    g = AdaptiveGrid(np.full((160, 160), f))
    imgs[f"a{int(f)}"] = np.asarray(AdaptiveSceneSimulator(model, cfg1, g).simulate(params)).squeeze()
ref = imgs["u64"]; cy, cx = np.unravel_index(np.argmax(ref), ref.shape)
print("lens light only; centre pixel", (cy, cx), "ref ss=64 value", ref[cy, cx], "sigma", err[cy, cx])
print(f"{'render':>6} {'centre (d/sig)':>15} {'max|d|/sig':>11} {'where':>10}  3x3 block around centre (d/sig)")
for k, im in imgs.items():
    d = (im - ref) / err; iy, ix = np.unravel_index(np.argmax(np.abs(d)), d.shape)
    print(f"{k:>6} {d[cy, cx]:>15.3f} {np.abs(d).max():>11.3f} {str((iy, ix)):>10}  ", np.round(d[cy-1:cy+2, cx-1:cx+2].ravel(), 3))
print("adaptive all-8 vs uniform ss=8: max |diff|/sig", np.abs(imgs["a8"] - imgs["u8"]).max() / err.min(), "; all-4 vs u4:", np.abs(imgs["a4"] - imgs["u4"]).max() / err.min())
