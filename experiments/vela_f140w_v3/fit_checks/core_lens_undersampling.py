"""Undersampling diagnostics for the CORED lens lights (regenerated set, 2026-09-18), shown
before any fit is re-run. For every system, lens light only at the truth (core-Sersic with the
drawn R_b, gamma, alpha = 5): uniform ss 4 / 8 / 16 / 32 (= the generator's truth quadrature)
vs uniform 64, and the fits' curvature-adaptive map (max 8) vs 64 and vs 32. Reports the
lens-centre pixel and the worst pixel in sigma units, and the 3x3 block, plus the old
pure-Sersic numbers for the same lens (R_b = 0) for contrast. Also the generator's own
32-vs-64 convergence (is the truth render converged?)."""
import os, sys, json, copy, dataclasses, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import yaml, tensorflow_probability.substrates.jax as tfp; tfd = tfp.distributions
from gigalens.jax.experimental.adaptive_supersample import AdaptiveGrid, AdaptiveSceneSimulator
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens_research.simtests.system import System
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.experiments import vela_simulated as V
from gigalens_research.simulations.image_based_light import ImageBasedLight
from gigalens_research.inference_utils.params import truth_x_to_scene_params
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"
DS = sys.argv[1] if len(sys.argv) > 1 else BASE + "/dataset"
cy_ = yaml.safe_load(open("experiments/vela_f140w_v3/campaign.yaml"))["dataset"]
spec = V.resolve_truth_prior_spec(cy_.get("truth_prior"), cy_.get("lens_light_profile", "sersic"))
model = V._build_truth_scene_model(spec, ImageBasedLight(np.ones((16, 16)), 0.05))
fit = _load_campaign("experiments/vela_f140w_v3/fit_sersic_vela22.yaml"); curv = {k: v for k, v in fit.effective_pipeline_kwargs({"fit": "x"})["adaptive"].items() if k != "driver"}
man = json.load(open(DS + "/manifest.json")); sids = man["system_ids"] if isinstance(man, dict) and "system_ids" in man else [s if isinstance(s, str) else s["system_id"] for s in (man["systems"] if isinstance(man, dict) else man)]
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None   # one system per process on GPU: the ss=64 simulators exhaust device memory across systems
if ONLY: sids = [x for x in sids if any(x.startswith(o) for o in ONLY)]
JS = sys.argv[3] if len(sys.argv) > 3 else f"{HERE}/core_lens_undersampling.json"   # argv[3]: output JSON (2026-09-19: tied-core set -> tied_core_lens_undersampling.json)
out = json.load(open(JS)) if (ONLY and os.path.exists(JS)) else {}
print("lens light only at truth, d/sigma vs uniform ss=64. 'a8' = the fits' curvature map (max 8).")
print(f"{'system':8s} {'n':>4s} {'Rb/Re':>6s} {'Rb px':>5s} {'gam':>4s} | centre px: {'u4':>6s} {'u8':>6s} {'u16':>6s} {'u32':>6s} {'a8':>6s} | worst px: {'a8':>6s} {'a8 vs u32':>9s} {'u32':>6s} | pure-Sersic same lens: a8 centre, worst")
for sid in sids:
    sy = System.load(DS, sid); cfg = sy.sim_config; cfg1 = dataclasses.replace(cfg, supersample=1)
    t = copy.deepcopy(sy.truth_x); t[2][0] = dict(t[2][0]); t[2][0]["amp"] = 0.0
    LL = t[1][0]; params = truth_x_to_scene_params(t, model)
    obs = np.asarray(sy.observed_image); err = np.sqrt(sy.background_rms ** 2 + np.clip(obs, 0, None) / sy.exp_time)
    imgs = {f"u{s}": np.asarray(SceneSimulator(model, dataclasses.replace(cfg, supersample=s)).simulate(params)).squeeze() for s in (4, 8, 16, 32, 64)}
    g8 = AdaptiveGrid.from_curvature(obs, err, **{**curv, "max_factor": 8.0}); imgs["a8"] = np.asarray(AdaptiveSceneSimulator(model, cfg1, g8).simulate(params)).squeeze()
    ref = imgs["u64"]; cyx = np.unravel_index(np.argmax(ref), ref.shape)
    d = {k: (v - ref) / err for k, v in imgs.items()}; d["a8_vs_u32"] = (imgs["a8"] - imgs["u32"]) / err
    # the same lens as a pure Sersic (Rb -> 0, gamma 0) for contrast
    t0 = copy.deepcopy(t); t0[1][0]["Rb"] = 1e-12; t0[1][0]["gamma"] = 0.0; p0 = truth_x_to_scene_params(t0, model)
    ref0 = np.asarray(SceneSimulator(model, dataclasses.replace(cfg, supersample=64)).simulate(p0)).squeeze(); a80 = np.asarray(AdaptiveSceneSimulator(model, cfg1, g8).simulate(p0)).squeeze(); d0 = (a80 - ref0) / err
    r = {k: dict(centre=float(v[cyx]), worst=float(np.abs(v).max()), block3=np.round(v[cyx[0]-1:cyx[0]+2, cyx[1]-1:cyx[1]+2], 3).tolist()) for k, v in d.items()}
    r["pure_sersic_a8"] = dict(centre=float(d0[cyx]), worst=float(np.abs(d0).max()))
    out[sid] = dict(n=float(LL["n_sersic"]), Rb_over_Re=float(LL["Rb"] / LL["R_sersic"]), Rb_px=float(LL["Rb"] / cfg.delta_pix), gamma=float(LL["gamma"]), n_points_a8=int(g8.n_points), results=r)
    print(f"{sid[:6]:8s} {LL['n_sersic']:4.2f} {100*LL['Rb']/LL['R_sersic']:5.2f}% {LL['Rb']/cfg.delta_pix:5.2f} {LL['gamma']:4.2f} | {r['u4']['centre']:+6.3f} {r['u8']['centre']:+6.3f} {r['u16']['centre']:+6.3f} {r['u32']['centre']:+6.3f} {r['a8']['centre']:+6.3f} | {r['a8']['worst']:6.3f} {r['a8_vs_u32']['worst']:9.3f} {r['u32']['worst']:6.3f} | {r['pure_sersic_a8']['centre']:+6.3f}, {r['pure_sersic_a8']['worst']:6.3f}")
json.dump(out, open(JS, "w"), indent=1); print("done")
