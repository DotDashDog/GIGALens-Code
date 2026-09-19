"""vela22: compare the truth-free Sersic fit (DC-1), the truth-started Sersic fit and the
truth-started shapelet fits (n_max 5..20). Corner plots via the repo's corner utilities
(gigalens_research.plotting: plot_corner / plot_corner_overlay, corner package underneath)."""
import os, sys, json, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.system import System
from gigalens_research.simtests.registry import get_inference_builder
from gigalens_research.inference_utils.pipeline import InferenceContext, posterior_from_disk
from gigalens_research.plotting import plot_corner, plot_corner_overlay
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"; SID = "vela22_cam12_a0.400_rep00"
system = System.load(BASE + "/dataset", SID)
RUNS = [("sersic_free", "experiments/vela_f140w_v3/fit_sersic_vela22.yaml", {"fit": "sersic_v1"}, "fitsersic_v1"),
        ("sersic_truth", "experiments/vela_f140w_v3/fit_sersic_truthinit_vela22.yaml", {"fit": "sersic_truthinit_v1"}, "fitsersic_truthinit_v1"),
        ("sersic_truth_seed1", "experiments/vela_f140w_v3/fit_sersic_truthinit_seed1_vela22.yaml", {"fit": "sersic_truthinit_seed1"}, "fitsersic_truthinit_seed1"),
        ("sersic_truth_maskcusp", "experiments/vela_f140w_v3/fit_sersic_truthinit_maskcusp_vela22.yaml", {"fit": "sersic_truthinit_maskcusp"}, "fitsersic_truthinit_maskcusp")]
RUNS += [(f"shapelets_n{n}", "experiments/vela_f140w_v3/fit_shapelets_truthinit_vela22.yaml", {"fit": "shapelets_truthinit_v1", "n_max": n}, f"fitshapelets_truthinit_v1_n_max{n}") for n in (5, 10, 15, 20)]
if only: RUNS = [r for r in RUNS if r[0] in only]
def path_truth(model):
    t = system.truth_x; out = {}
    for i, plane in enumerate(model.planes):
        for j, c in enumerate(plane.mass):
            for p, v in t[0][j].items(): out[f"planes/{model.plane_key(i)}/mass/{model.component_key(i, 'mass', j)}/{p}"] = float(v)
        for j, c in enumerate(plane.light):
            g = 2 if i == 1 else 1
            for p, v in t[g][j].items(): out[f"planes/{model.plane_key(i)}/light/{model.component_key(i, 'light', j)}/{p}"] = float(v)
    return out
results = {}; posteriors = {}
for label, yaml_path, sp, rdir in RUNS:
    run_dir = f"{BASE}/runs/{SID}/{rdir}"
    if not os.path.exists(run_dir + "/mclmc/arrays.npz"): print("missing", label); continue
    spec = _load_campaign(yaml_path); kw = spec.effective_pipeline_kwargs(sp)
    prob = get_inference_builder(spec.inference.builder)(system, **kw); ctx = InferenceContext.from_prob_model(prob)
    post = posterior_from_disk(run_dir, "mclmc", ctx); posteriors[label] = post
    S = np.load(run_dir + "/mclmc/arrays.npz")["samples_z"]; nch, nst, D = S.shape
    lab = prob.labeled_samples(jnp.asarray(S.reshape(-1, D))); truth = path_truth(prob.model)
    stats = {}
    for k, v in lab.items():
        v = np.asarray(v).reshape(-1); m, s = float(v.mean()), float(v.std())
        stats[k] = dict(mean=m, std=s, truth=truth.get(k), z=(m - truth[k]) / s if k in truth else None)
    # chi2/nu of the posterior-mean-z model (lstsq amplitudes re-solved)
    zbar = jnp.asarray(S.reshape(-1, D).mean(0)); params = prob.constrained(zbar); sim = prob.simulators[0]
    obs = np.asarray(prob.observed_image); err = np.asarray(prob.error_map)
    model_img = np.asarray(sim.lstsq_simulate(params, jnp.asarray(obs), jnp.asarray(err))).squeeze()
    res = (obs - model_img) / err; nu = obs.size - D
    yy, xx = np.indices(obs.shape); r = np.hypot(yy - 79.5, xx - 79.5) * 0.065; ring = (r >= 0.8) & (r < 1.4)
    run_json = json.load(open(run_dir + "/run.json")) if os.path.exists(run_dir + "/run.json") else {}
    met = run_json.get("metrics", run_json)
    uc = {}
    if os.path.exists(run_dir + "/undersampling_check/manifest.json"):
        um = json.load(open(run_dir + "/undersampling_check/manifest.json")).get("metadata", {})
        rep = (um.get("reports") or [{}])[0]
        uc = {k: rep.get(k) for k in ("configured_max_abs_delta_over_sigma", "configured_frac_above_tolerance", "configured_max_in_excluded", "reference_self_error_max", "reference_converged", "configured_delta_chi2")}
        uc["rungs"] = [(rr["label"], round(rr["max_abs_delta_over_sigma"], 3), rr["passes"]) for rr in rep.get("rungs", [])]
    results[label] = dict(n_params=D, max_rhat=met.get("max_rhat"), min_ess=met.get("min_ess"), nan_rate=met.get("nan_rate"),
                          chi2_nu_mean_model=float((res ** 2).sum() / nu), ring_chi2_n=float((res[ring] ** 2).mean()),
                          mass_z={k.split("/")[-1] + ("_shear" if "/mass/1/" in k else ""): round(v["z"], 2) for k, v in stats.items() if "/mass/" in k},
                          stats=stats, undersampling=uc, wall_time=run_json.get("wall_time_s", met.get("wall_time")))
    print(f"{label:14s} D={D:2d} R-hat {met.get('max_rhat', float('nan')):.4f} ESS {met.get('min_ess', float('nan')):8.0f} chi2/nu(mean z) {results[label]['chi2_nu_mean_model']:.4f} ring {results[label]['ring_chi2_n']:.3f} | mass z: {results[label]['mass_z']} | undersampling: {uc.get('configured_max_abs_delta_over_sigma')}")
    # per-run mass corner via the library, truth crosshairs
    fig = plot_corner(post, kind="mass", truth=truth, color="C0", truth_color="red")
    fig.suptitle(f"vela22 {label}: lens-mass posterior (8 x 5000), truth = red", y=1.0); fig.savefig(f"{OUT}/corner_mass_{label}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # residual of the mean-z model
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2)); im = ax[0].imshow(res, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5); ax[0].set_title(f"{label}: (obs - mean-z model)/sigma, chi2/nu {results[label]['chi2_nu_mean_model']:.4f}")
    ax[1].imshow(res[40:120, 40:120], origin="lower", cmap="RdBu_r", vmin=-5, vmax=5); ax[1].set_title(f"central 5.2\"  ring annulus chi2/n {results[label]['ring_chi2_n']:.3f}")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=ax.tolist(), fraction=0.03); fig.savefig(f"{OUT}/residual_{label}.png", dpi=100, bbox_inches="tight"); plt.close(fig)
json.dump(results, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
# overlays via the library
if "sersic_free" in posteriors and "sersic_truth" in posteriors:
    tr = path_truth(get_inference_builder("epl_shear_sersic_sersic_source_broad")(system).model)
    ov = {"truth-free MAP start, seed 0 (DC-1)": posteriors["sersic_free"], "started at truth, seed 0": posteriors["sersic_truth"]}
    if "sersic_truth_seed1" in posteriors: ov["started at truth, seed 1"] = posteriors["sersic_truth_seed1"]
    fig = plot_corner_overlay(ov, kind="mass", truth=tr)
    fig.suptitle("vela22 Sersic source: lens-mass posteriors, two chain starts", y=1.0); fig.savefig(f"{OUT}/overlay_mass_sersic_free_vs_truth.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    comp = {}
    for other in ("sersic_truth", "sersic_truth_seed1"):
        if other not in results: continue
        d = {k: (results["sersic_free"]["stats"][k]["mean"] - results[other]["stats"][k]["mean"]) / results[other]["stats"][k]["std"] for k in results[other]["stats"]}
        w = {k: results["sersic_free"]["stats"][k]["std"] / results[other]["stats"][k]["std"] for k in results[other]["stats"]}
        print("sersic_free vs %s: max |dmean|/sigma = %.3f (%s); width ratio range %.3f-%.3f" % (other, max(abs(v) for v in d.values()), max(d, key=lambda k: abs(d[k])), min(w.values()), max(w.values())))
        comp[other] = {"dmean_over_sigma": d, "width_ratio": w}
    json.dump(comp, open(f"{OUT}/sersic_start_comparison.json", "w"), indent=1)
shap = [l for l in ("shapelets_n5", "shapelets_n10", "shapelets_n15", "shapelets_n20") if l in posteriors]
if shap:
    tr = path_truth(posteriors[shap[0]].ctx.prob_model.model) if hasattr(posteriors[shap[0]], "ctx") else path_truth(get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=5).model)
    fig = plot_corner_overlay({l.replace("shapelets_n", "n_max "): posteriors[l] for l in shap}, kind="mass", truth=tr)
    fig.suptitle("vela22 shapelet sources started at truth: lens-mass posteriors vs n_max", y=1.0); fig.savefig(f"{OUT}/overlay_mass_shapelets_nmax.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # z-scores vs n_max
    labels = ["sersic_truth"] + shap if "sersic_truth" in results else shap
    fig, ax = plt.subplots(figsize=(9, 4.5)); names = list(results[labels[0]]["mass_z"].keys())
    xs = np.arange(len(labels))
    for i, n in enumerate(names): ax.plot(xs, [results[l]["mass_z"][n] for l in labels], "o-", label=n)
    ax.axhspan(-3, 3, color="0.9"); ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20); ax.set_ylabel("(posterior mean - truth) / posterior sigma"); ax.legend(ncol=4, fontsize=8); ax.set_title("vela22: lens-mass z-scores vs source model (chains started at truth)")
    fig.savefig(f"{OUT}/mass_z_vs_model.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT)
