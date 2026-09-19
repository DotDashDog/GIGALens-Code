"""DC-6 (2026-09-19): vela22 cored set, Sersic source, truth start — R_b prior truncated at 0.01 px and
gamma fixed at truth (`fitsersic_truthinit_core_v2_rbfloor_gfix`) vs the baseline core-Sersic fit
(`fitsersic_truthinit_core_v1`) and the pure-Sersic-lens fit on the old set (`fitsersic_truthinit_v1`,
reference efficiency). Per-parameter rank R-hat / ESS (arviz), ESS per second, invariance of the
posterior (means in baseline-sigma units, width ratios), R_b tail fractions, traces, mass overlay."""
import os, sys, json, numpy as np, jax, jax.numpy as jnp, arviz as az
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.system import System
from gigalens_research.simtests.registry import get_inference_builder
from gigalens_research.inference_utils.pipeline import InferenceContext, posterior_from_disk
from gigalens_research.plotting import plot_corner_overlay
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"; SID = "vela22_cam12_a0.400_rep00"
system_scatter = System.load(BASE + "/dataset_20260918_core_scatter", SID); system_tied = System.load(BASE + "/dataset", SID)
system = system_scatter; DP = system.delta_pix
RUNS = [
    ("baseline", "experiments/vela_f140w_v3/fit_sersic_truthinit_core_vela22.yaml", {"fit": "sersic_truthinit_core_v1"}, "fitsersic_truthinit_core_v1"),
    ("dc6", "experiments/vela_f140w_v3/fit_sersic_truthinit_core_rbfloor_vela22.yaml", {"fit": "sersic_truthinit_core_v2_rbfloor_gfix"}, "fitsersic_truthinit_core_v2_rbfloor_gfix"),
    ("rbfix", "experiments/vela_f140w_v3/fit_sersic_truthinit_core_rbfix_vela22.yaml", {"fit": "sersic_truthinit_core_v2_rbfix_gfix"}, "fitsersic_truthinit_core_v2_rbfix_gfix"),
    # tied-core SET (regenerated 2026-09-19; vela22's core is 0.41 px there, not 0.11): the fit has no core params
    ("tied", "experiments/vela_f140w_v3/fit_sersic_truthinit_tied_vela22.yaml", {"fit": "sersic_truthinit_tied_v1"}, "fitsersic_truthinit_tied_v1"),
]
RUNS = [r for r in RUNS if os.path.exists(f"{BASE}/runs/{SID}/{r[3]}/mclmc/arrays.npz")]
# the old-set pure-Sersic lens fit: same yaml family but the dataset differs; only its run.json numbers are used
OLD = BASE + "/runs/" + SID + "/fitsersic_truthinit_v1"
res = {}; post = {}; lab = {}; truthd = {}
def path_truth(model):
    t = system.truth_x; out = {}
    for i, plane in enumerate(model.planes):
        for j, c in enumerate(plane.mass):
            for p, v in t[0][j].items(): out[f"planes/{model.plane_key(i)}/mass/{model.component_key(i, 'mass', j)}/{p}"] = float(v)
        for j, c in enumerate(plane.light):
            g = 2 if i == 1 else 1
            for p, v in t[g][j].items(): out[f"planes/{model.plane_key(i)}/light/{model.component_key(i, 'light', j)}/{p}"] = float(v)
    return out
for label, yaml_path, sp, rdir in RUNS:
    run_dir = f"{BASE}/runs/{SID}/{rdir}"
    system = system_tied if label == "tied" else system_scatter
    spec = _load_campaign(yaml_path); kw = spec.effective_pipeline_kwargs(sp)
    prob = get_inference_builder(spec.inference.builder)(system, **kw); ctx = InferenceContext.from_prob_model(prob)
    post[label] = posterior_from_disk(run_dir, "mclmc", ctx)
    S = np.load(run_dir + "/mclmc/arrays.npz")["samples_z"]; nch, nst, D = S.shape
    L = prob.labeled_samples(jnp.asarray(S.reshape(-1, D)))
    L = {k: np.asarray(v).reshape(nch, nst, -1) for k, v in L.items()}
    flat = {}
    for k, v in L.items():
        if v.shape[-1] == 1: flat[k] = v[..., 0]
        else:
            for i, kk in enumerate(k.split("|")): flat[kk] = v[..., i]
    lab[label] = flat; truthd[label] = path_truth(prob.model)
    rj = json.load(open(run_dir + "/run.json")); met = rj["metrics"]
    rhat = {k: float(az.rhat(v)) for k, v in flat.items()}; ess = {k: float(az.ess(v)) for k, v in flat.items()}
    spread = {k: float(v.mean(1).std() / v.std()) for k, v in flat.items()}   # chain-mean spread / posterior sigma
    rb = flat["planes/0/light/0/Rb"].reshape(-1) / DP if "planes/0/light/0/Rb" in flat else np.full(1, np.nan)
    uc = {}
    if os.path.exists(run_dir + "/undersampling_check/manifest.json"):
        rep = (json.load(open(run_dir + "/undersampling_check/manifest.json")).get("metadata", {}).get("reports") or [{}])[0]
        uc = {k: rep.get(k) for k in ("configured_max_abs_delta_over_sigma", "reference_self_error_max")}
    res[label] = dict(D=D, wall_s=rj.get("wall_time_s"), max_rhat=max(rhat.values()), min_ess=min(ess.values()),
                      argmin_ess=min(ess, key=ess.get), argmax_rhat=max(rhat, key=rhat.get),
                      ess_per_s=min(ess.values()) / rj["wall_time_s"], mean_ess=float(np.mean(list(ess.values()))),
                      max_chain_spread=max(spread.values()), rhat=rhat, ess=ess,
                      rb_px=dict(median=float(np.median(rb)), mean=float(rb.mean()), std=float(rb.std()), p05=float(np.percentile(rb, 5)), p95=float(np.percentile(rb, 95)),
                                 frac_below_0p02=float(np.mean(rb < 0.02)), frac_below_0p05=float(np.mean(rb < 0.05))),
                      mass_z=met.get("mass_zscores"), nan_rate=met.get("nan_rate"), undersampling=uc,
                      stats={k: dict(mean=float(v.mean()), std=float(v.std()), truth=truthd[label].get(k)) for k, v in flat.items()})
    print(f"{label:9s} D={D} wall {rj.get('wall_time_s')} s | R-hat max {res[label]['max_rhat']:.4f} ({res[label]['argmax_rhat']}) | ESS min {res[label]['min_ess']:.0f} ({res[label]['argmin_ess']}) mean {res[label]['mean_ess']:.0f} | ESS/s {res[label]['ess_per_s']:.1f} | spread {res[label]['max_chain_spread']:.3f} | Rb px {res[label]['rb_px']}")
old = json.load(open(OLD + "/run.json")); res["old_set_pure_sersic_lens"] = dict(max_rhat=old["metrics"]["max_rhat"], min_ess=old["metrics"]["min_ess"], wall_s=old.get("wall_time_s"), ess_per_s=old["metrics"]["min_ess"] / old["wall_time_s"])
print("old-set pure-Sersic lens:", res["old_set_pure_sersic_lens"])
# invariance: each variant vs baseline, in baseline sigma
b = res["baseline"]["stats"]
for lbl in [r[0] for r in RUNS if r[0] != "baseline"]:
    d = res[lbl]["stats"]; inv = {}
    for k in d:
        if k in b: inv[k] = dict(dmean_over_sigma_base=(d[k]["mean"] - b[k]["mean"]) / b[k]["std"], width_ratio=d[k]["std"] / b[k]["std"],
                                 z_base=(b[k]["mean"] - b[k]["truth"]) / b[k]["std"] if b[k]["truth"] is not None else None,
                                 z_var=(d[k]["mean"] - d[k]["truth"]) / d[k]["std"] if d[k]["truth"] is not None else None)
    res[f"invariance_{lbl}_vs_baseline"] = inv
    print(f"\ninvariance ({lbl} - baseline)/sigma_base, width ratio:")
    for k, v in inv.items(): print(f"  {k:32s} dmean {v['dmean_over_sigma_base']:+.3f}  width x{v['width_ratio']:.2f}  z {v['z_base'] if v['z_base'] is None else round(v['z_base'],2)} -> {v['z_var'] if v['z_var'] is None else round(v['z_var'],2)}")
    print(f"\nper-parameter ESS baseline -> {lbl}:")
    for k in res[lbl]["ess"]: print(f"  {k:32s} {res['baseline']['ess'].get(k, float('nan')):8.0f} -> {res[lbl]['ess'][k]:8.0f}   R-hat {res['baseline']['rhat'].get(k, float('nan')):.4f} -> {res[lbl]['rhat'][k]:.4f}")
json.dump(res, open(OUT + "/dc6_comparison.json", "w"), indent=1)
# figure: ESS per parameter + Rb / n / R_e traces (chain 0) + Rb histograms
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
names = list(res["dc6"]["ess"]); short = [n.replace("planes/0/mass/0/", "m:").replace("planes/0/mass/1/", "sh:").replace("planes/0/light/0/", "ll:").replace("planes/1/light/0/", "src:") for n in names]
x = np.arange(len(names)); labs = {"baseline": "baseline (R_b free tail, γ free)", "dc6": "DC-6 (R_b ≥ 0.01 px, γ fixed)", "rbfix": "ablation (R_b and γ fixed at truth)", "tied": "tied-core set: R_b = rule(R_e, n), γ = 0"}
present = [r[0] for r in RUNS]; w = 0.8 / len(present)
for j, lbl in enumerate(present): ax[0, 0].bar(x + (j - (len(present) - 1) / 2) * w, [res[lbl]["ess"].get(n, 0) for n in names], w, label=labs[lbl])
ax[0, 0].axhline(old["metrics"]["min_ess"], ls="--", c="k", lw=0.8, label="old set, pure-Sersic lens: min ESS")
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(short, rotation=90, fontsize=7); ax[0, 0].set_ylabel("ESS (8 x 5000)"); ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("ESS per parameter")
for lbl, c in (("baseline", "C0"), ("dc6", "C1"), ("rbfix", "C2"), ("tied", "C3")):
    if lbl not in lab: continue
    if "planes/0/light/0/Rb" in lab[lbl]:
        rb = lab[lbl]["planes/0/light/0/Rb"] / DP
        ax[0, 1].plot(rb[0], lw=0.4, c=c, alpha=0.8, label=lbl); ax[1, 0].hist(np.log10(rb.reshape(-1)), bins=80, histtype="step", color=c, density=True, label=lbl)
    ax[1, 1].plot(lab[lbl]["planes/0/light/0/n_sersic"][0], lw=0.4, c=c, alpha=0.8, label=lbl)
ax[0, 1].set_yscale("log"); ax[0, 1].axhline(truthd["dc6"]["planes/0/light/0/Rb"] / DP, c="r", lw=0.8); ax[0, 1].set_title("R_b [px], chain 0"); ax[0, 1].legend(fontsize=8)
ax[1, 0].axvline(np.log10(truthd["dc6"]["planes/0/light/0/Rb"] / DP), c="r", lw=0.8); ax[1, 0].axvline(-2, c="k", ls=":", lw=0.8); ax[1, 0].set_xlabel("log10 R_b [px]"); ax[1, 0].set_title("R_b marginal (truth red, 0.01 px floor dotted)"); ax[1, 0].legend(fontsize=8)
ax[1, 1].axhline(truthd["dc6"]["planes/0/light/0/n_sersic"], c="r", lw=0.8); ax[1, 1].set_title("lens-light n, chain 0"); ax[1, 1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "/dc6_ess_traces.png", dpi=110); plt.close(fig)
ov = {"baseline core-Sersic fit": post["baseline"], "DC-6: R_b >= 0.01 px, gamma fixed": post["dc6"]}
if "rbfix" in post: ov["ablation: R_b, gamma fixed at truth"] = post["rbfix"]
if "tied" in post: ov["tied-core set: R_b = rule(R_e, n), gamma 0"] = post["tied"]
fig = plot_corner_overlay(ov, kind="mass", truth=truthd["dc6"]); fig.suptitle("vela22 Sersic source, lens mass: baseline vs DC-6 variants", y=1.0)
fig.savefig(OUT + "/overlay_mass_dc6.png", dpi=90); plt.close(fig)
print("wrote", OUT)
