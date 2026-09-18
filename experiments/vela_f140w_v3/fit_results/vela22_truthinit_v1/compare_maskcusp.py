"""DC-3: the truth-start Sersic fit with the 49-px lens-cusp disk masked vs the unmasked one
(and the truth-free DC-1 run). Pre-registered prediction (log, DC-3): mass |dmean|/sigma < 0.3,
z-scores within 1 sigma of C-2's, lens-light n width x1.5, centre x1.5-1.7, R x1.1."""
import os, sys, json, numpy as np, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.system import System
from gigalens_research.simtests.registry import get_inference_builder
from gigalens_research.inference_utils.pipeline import InferenceContext, posterior_from_disk
from gigalens_research.plotting import plot_corner_overlay
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"; SID = "vela22_cam12_a0.400_rep00"
system = System.load(BASE + "/dataset", SID)
RUNS = {"truth-free (DC-1)": ("experiments/vela_f140w_v3/fit_sersic_vela22.yaml", {"fit": "sersic_v1"}, "fitsersic_v1"),
        "truth start": ("experiments/vela_f140w_v3/fit_sersic_truthinit_vela22.yaml", {"fit": "sersic_truthinit_v1"}, "fitsersic_truthinit_v1"),
        "truth start, cusp masked (DC-3)": ("experiments/vela_f140w_v3/fit_sersic_truthinit_maskcusp_vela22.yaml", {"fit": "sersic_truthinit_maskcusp"}, "fitsersic_truthinit_maskcusp")}
truth = json.load(open(f"{OUT}/summary.json"))["sersic_free"]["stats"]; truth = {k: v["truth"] for k, v in truth.items()}
stats, posts, chi2 = {}, {}, {}
for label, (yaml_path, sp, rdir) in RUNS.items():
    run_dir = f"{BASE}/runs/{SID}/{rdir}"; spec = _load_campaign(yaml_path); kw = spec.effective_pipeline_kwargs(sp)
    prob = get_inference_builder(spec.inference.builder)(system, **kw); ctx = InferenceContext.from_prob_model(prob)
    posts[label] = posterior_from_disk(run_dir, "mclmc", ctx)
    S = np.load(run_dir + "/mclmc/arrays.npz")["samples_z"]; D = S.shape[-1]; lab = prob.labeled_samples(jnp.asarray(S.reshape(-1, D)))
    stats[label] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in lab.items()}
    zbar = jnp.asarray(S.reshape(-1, D).mean(0)); sim = prob.simulators[0]; obs = np.asarray(prob.observed_image); err = np.asarray(prob.error_map)
    ds = prob.datasets[0]; mask = np.asarray(ds.mask, bool) if ds.mask is not None else np.ones(obs.shape, bool)
    m = np.asarray(sim.lstsq_simulate(prob.constrained(zbar), jnp.asarray(obs), jnp.asarray(err), mask=jnp.asarray(mask))).squeeze()
    res = (obs - m) / err; chi2[label] = dict(chi2_nu_kept=float((res[mask] ** 2).sum() / (mask.sum() - D)), n_kept=int(mask.sum()))
    met = json.load(open(run_dir + "/run.json")).get("metrics", {}); chi2[label].update(max_rhat=met.get("max_rhat"), min_ess=met.get("min_ess"))
    print(f"{label:34s} chi2/nu on kept px {chi2[label]['chi2_nu_kept']:.4f} ({chi2[label]['n_kept']} px)  R-hat {met.get('max_rhat'):.4f} ESS {met.get('min_ess'):.0f}")
ref, new = stats["truth start"], stats["truth start, cusp masked (DC-3)"]
print(f"\n{'parameter':30s} {'z unmasked':>10s} {'z masked':>9s} {'dmean/sig':>10s} {'width ratio':>11s}")
rows = {}
for k in ref:
    m0, s0 = ref[k]; m1, s1 = new[k]; t = truth.get(k)
    z0 = (m0 - t) / s0 if t is not None else None; z1 = (m1 - t) / s1 if t is not None else None
    rows[k] = dict(z_unmasked=z0, z_masked=z1, dmean_over_sigma_masked=(m1 - m0) / s1, width_ratio=s1 / s0, toward_truth_sigma=(None if t is None else (abs(m0 - t) - abs(m1 - t)) / s1))
    print(f"{k:30s} {('%.2f' % z0) if z0 is not None else '-':>10s} {('%.2f' % z1) if z1 is not None else '-':>9s} {(m1 - m0) / s1:10.3f} {s1 / s0:11.3f}")
mass = {k: v for k, v in rows.items() if "/mass/" in k}
print("\nmax mass |dmean|/sigma:", round(max(abs(v["dmean_over_sigma_masked"]) for v in mass.values()), 3),
      "; max movement toward truth (sigma):", round(max(v["toward_truth_sigma"] for v in mass.values()), 3),
      "; max mass |z| masked:", round(max(abs(v["z_masked"]) for v in mass.values()), 2))
json.dump(dict(rows=rows, fits=chi2), open(f"{OUT}/maskcusp_comparison.json", "w"), indent=1)
tr = {k: v for k, v in truth.items() if v is not None}
for kind, fn in (("mass", "overlay_mass_maskcusp.png"), ("light", "overlay_light_maskcusp.png")):
    fig = plot_corner_overlay(posts, kind=kind, truth=tr); fig.suptitle(f"vela22 Sersic source, {kind} parameters: lens-cusp disk masked vs not", y=1.0)
    fig.savefig(f"{OUT}/{fn}", dpi=100, bbox_inches="tight"); plt.close(fig)
print("wrote overlays")
