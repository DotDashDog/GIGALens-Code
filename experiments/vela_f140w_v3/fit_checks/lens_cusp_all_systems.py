"""Lens-light cusp quadrature across ALL 10 systems, and what raising the adaptive ceiling to 16
would buy. Lens light only at the truth, vs the generator's uniform ss=32 (= the data):
uniform 8 / 16, the fits' curvature map (max 8) and the same map with max 16 (ALLOWED_FACTORS
extended in-process; factors >= 1 are plain uniform sub-grids so 16 needs no new mechanism —
verified here: all-16 adaptive == uniform 16). Fisher projection of each residual onto the lens
mass with the broad-Sersic lstsq model at a crude source point (truth lens + light, source at
the truth centre with R 0.2", n 1.5, round); bias in units of the Fisher sigma (on vela22 the
exact-point version gave < 0.07 sigma and Fisher sigma = MCLMC sigma to 10%)."""
import os, json, copy, dataclasses, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import yaml, tensorflow_probability.substrates.jax as tfp; tfd = tfp.distributions
import gigalens.jax.experimental.adaptive_supersample as ads
ads.ALLOWED_FACTORS = ads.ALLOWED_FACTORS + (16.0, 32.0)
from gigalens.jax.experimental.adaptive_supersample import AdaptiveSceneSimulator, AdaptiveGrid
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.system import System
from gigalens_research.simtests.registry import get_inference_builder
from gigalens_research.simtests.experiments.vela_simulated import resolve_truth_prior_spec, _make_dist
from gigalens_research.inference_utils.params import truth_x_to_scene_params
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"
spec = _load_campaign("experiments/vela_f140w_v3/fit_sersic_vela22.yaml"); kw = spec.effective_pipeline_kwargs({"fit": "x"})
adaptive = dict(kw["adaptive"]); curv = {k: v for k, v in adaptive.items() if k != "driver"}
tps = resolve_truth_prior_spec(yaml.safe_load(open("experiments/vela_f140w_v3/campaign.yaml"))["dataset"].get("truth_prior"))
def cp(g, c): return {p: (float(d["value"]) if d["dist"] == "Fixed" else _make_dist(d, p)) for p, d in tps[g][c].items()}
src_p = {k: tfd.Normal(0.0, 1.0) for k in ["Ie", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]}
model_nl = LensModel([Plane(mass=[Component(epl.EPL(50), cp("lens_mass", "0")), Component(shear.Shear(), cp("lens_mass", "1"))],
                            light=[Component(sersic.SersicEllipse(use_lstsq=False), cp("lens_light", "0"))]),
                      Plane(deflection_ratio=1.0, light=[Component(sersic.SersicEllipse(use_lstsq=False), src_p)])])
sids = json.load(open(BASE + "/dataset/manifest.json"))
sids = sids["system_ids"] if isinstance(sids, dict) and "system_ids" in sids else [s if isinstance(s, str) else s["system_id"] for s in (sids["systems"] if isinstance(sids, dict) else sids)]
out = {}
print("centre-pixel d/sig for uniform 8, 16, the fits' map (a8), a8 with the r<=4 disk at 16 / 32 | max|d| a8, +16, +32 | a8 max outside disk | chi2 a8, +16, +32 | points | px the driver itself would set to 16 | Fisher mass bias max |d|/sig: a8, +16disk")
print(f"{'system':8s} {'n':>4s} {'R\"':>5s} {'u8':>7s} {'u16':>7s} {'a8':>7s} {'+16':>7s} {'+32':>7s}")
for sid in sids:
    system = System.load(BASE + "/dataset", sid); cfg = system.sim_config; cfg1 = dataclasses.replace(cfg, supersample=1)
    t = copy.deepcopy(system.truth_x); t[2][0] = dict(Ie=0.0, R_sersic=0.2, n_sersic=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    p_nl = truth_x_to_scene_params(t, model_nl); LL = system.truth_x[1][0]
    obs = np.asarray(system.observed_image); err = np.sqrt(system.background_rms ** 2 + np.clip(obs, 0, None) / system.exp_time)
    ref = np.asarray(SceneSimulator(model_nl, dataclasses.replace(cfg, supersample=32)).simulate(p_nl)).squeeze()
    yy, xx = np.indices(obs.shape); row = (obs.shape[0] - 1) / 2 + float(LL["center_y"]) / cfg.delta_pix; col = (obs.shape[1] - 1) / 2 + float(LL["center_x"]) / cfg.delta_pix
    disk = np.hypot(yy - row, xx - col) <= 4.0; cy, cx = int(round(row)), int(round(col))
    g8 = AdaptiveGrid.from_curvature(obs, err, **{**curv, "max_factor": 8.0})
    g16drv = AdaptiveGrid.from_curvature(obs, err, **{**curv, "max_factor": 16.0})
    driver_asks_16 = int((g16drv.factor_map > 8).sum())
    grids = {"a8": g8}
    for f in (16.0, 32.0):   # the fits' map with the lens-centre disk (r <= 4 px) forced to f
        fm = np.array(g8.factor_map); fm[disk] = f; grids[f"a8+{int(f)}disk"] = AdaptiveGrid(fm)
    imgs = {f"u{s}": np.asarray(SceneSimulator(model_nl, dataclasses.replace(cfg, supersample=s)).simulate(p_nl)).squeeze() for s in (8, 16)}
    for k, g in grids.items(): imgs[k] = np.asarray(AdaptiveSceneSimulator(model_nl, cfg1, g).simulate(p_nl)).squeeze()
    if sid.startswith("vela22"):
        g16 = AdaptiveGrid(np.full(obs.shape, 16.0)); a16u = np.asarray(AdaptiveSceneSimulator(model_nl, cfg1, g16).simulate(p_nl)).squeeze()
        print("  [mechanism check, vela22] adaptive all-16 vs uniform 16: max |diff|/sigma =", np.abs(a16u - imgs["u16"]).max() / err.min())
    d = {k: (v - ref) / err for k, v in imgs.items()}
    r = {k: dict(centre=float(v[cy, cx]), max_abs=float(np.abs(v).max()), max_outside_disk4=float(np.abs(v[~disk]).max()), chi2=float((v ** 2).sum())) for k, v in d.items()}
    # Fisher projection (crude source point) for the two adaptive maps
    fb = {}
    for k in ("a8", "a8+16disk"):
        try:
            from gigalens.jax.experimental.adaptive_supersample import AdaptiveImageData
            ds = AdaptiveImageData(jnp.asarray(obs), cfg1, adaptive_grid=grids[k], background_rms=system.background_rms, exp_time=system.exp_time, sees="all")
            from gigalens.jax.scene_prob_model import ProbModel
            builder_model = get_inference_builder(spec.inference.builder)(system, **{**kw, "adaptive": None}).model  # same priors/model, dataset swapped below
            prob = ProbModel(builder_model, [ds], mode="lstsq")
            params = truth_x_to_scene_params(system.truth_x, prob.model)
            pk1 = prob.model.plane_key(1); ck = prob.model.component_key(1, "light", 0)
            params["planes"][pk1]["light"][ck].update(dict(R_sersic=0.2, n_sersic=1.5, e1=0.05, e2=-0.03))  # e1=e2=0 exactly gives NaN AD gradients (known)
            z0 = jnp.asarray(prob.unconstrained(jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), params))).reshape(-1)
            keys = list(prob.labeled_samples(z0[None]).keys())
            o, e = jnp.asarray(prob.observed_image), jnp.asarray(prob.error_map); en = np.asarray(e)
            sim = prob.simulators[0]
            J = np.moveaxis(np.asarray(jax.jacfwd(lambda z: jnp.reshape(sim.lstsq_simulate(prob.constrained(z[None]), o, e), obs.shape))(z0)), -1, 0)
            Jt = np.asarray(jax.jacfwd(lambda z: jnp.stack([jnp.asarray(prob.labeled_samples(z[None])[q], jnp.float64).reshape(()) for q in keys]))(z0))
            Jm = J.reshape(len(z0), -1) / en.ravel(); F = Jm @ Jm.T; Fi = np.linalg.inv(F)
            dz = -Fi @ (Jm @ ((imgs[k] - ref) / en).ravel()); dth = Jt @ dz; sig = np.sqrt(np.diag(Jt @ Fi @ Jt.T))
            fb[k] = {q: float(dth[i] / sig[i]) for i, q in enumerate(keys)}
        except Exception as ex:
            fb[k] = {"error": repr(ex)[:200]}
    def mx(k): return (max(abs(v) for q, v in fb[k].items() if "/mass/" in q) if "error" not in fb[k] else float("nan"))
    out[sid] = dict(lens_light=dict(n=float(LL["n_sersic"]), R=float(LL["R_sersic"]), Ie=float(LL["Ie"])), residuals=r, n_points={k: int(g.n_points) for k, g in grids.items()}, driver_asks_16_px=driver_asks_16, fisher=fb)
    a16, a32 = r["a8+16disk"], r["a8+32disk"]
    print(f"{sid[:6]:8s} {float(LL['n_sersic']):4.2f} {float(LL['R_sersic']):5.2f} {r['u8']['centre']:7.3f} {r['u16']['centre']:7.3f} {r['a8']['centre']:7.3f} {a16['centre']:7.3f} {a32['centre']:7.3f} | {r['a8']['max_abs']:6.3f} {a16['max_abs']:6.3f} {a32['max_abs']:6.3f} | {r['a8']['max_outside_disk4']:6.3f} | {r['a8']['chi2']:6.2f} {a16['chi2']:6.2f} {a32['chi2']:6.2f} | {grids['a8'].n_points:6d} {grids['a8+16disk'].n_points:6d} {grids['a8+32disk'].n_points:6d} | drv16 {driver_asks_16:3d} | {mx('a8'):.3f}, {mx('a8+16disk'):.3f}" + ("" if "error" not in fb["a8"] else "  ERR " + fb["a8"]["error"]))
    json.dump(out, open(f"{HERE}/lens_cusp_all_systems.json", "w"), indent=1)
print("done")
