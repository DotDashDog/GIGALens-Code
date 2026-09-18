"""Can the lens-light cusp quadrature error (adaptive tier 8 vs the generator's uniform ss=32)
explain the 3-9 sigma lens-mass bias of the vela22 Sersic fit?

A. Measure the mismatch image D = LL_adaptive8(truth) - LL_ss32(truth) (lens light only; the
   generator's lens_light_only.npy IS the ss=32 truth render, reproduced here to float32).
B. Fisher linear response at the truth start z0 (truth lens + bootstrapped Sersic source):
   E[dz] = -F^-1 J^T W D, F = J^T W J, J = d(lstsq model image)/dz at the fit's quadrature.
   Convert to parameter space and divide by the MCLMC posterior sigma of the DC-1 fit.
   Also: Fisher sigma vs MCLMC sigma (is the linearisation trustworthy?).
C. Same mismatch at the DC-1 posterior-mean model (lens light + fitted Sersic source, both at
   adaptive-8 vs uniform ss=32 with the same lstsq coefficients) -> the 'undersampling at the
   MAP' doubt, and its Fisher bias.
D. Predictions for a refit with the r<=4 px lens-centre disk masked: residual quadrature bias
   and the Fisher width inflation of the mass parameters.
Outputs: vela22_cusp_bias.png / .json in this directory."""
import sys, os, json, copy, dataclasses, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import yaml, tensorflow_probability.substrates.jax as tfp; tfd = tfp.distributions
from gigalens.jax.experimental.adaptive_supersample import AdaptiveSceneSimulator
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens_research.simtests.cli import _load_campaign
from gigalens_research.simtests.system import System
from gigalens_research.simtests.registry import get_inference_builder
from gigalens_research.simtests.experiments.vela_simulated import resolve_truth_prior_spec, _make_dist
from gigalens_research.inference_utils.params import truth_x_to_scene_params
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"; SID = "vela22_cam12_a0.400_rep00"
system = System.load(BASE + "/dataset", SID); cfg = system.sim_config; cfg1 = dataclasses.replace(cfg, supersample=1)
spec = _load_campaign("experiments/vela_f140w_v3/fit_sersic_truthinit_vela22.yaml"); kw = spec.effective_pipeline_kwargs({"fit": "sersic_truthinit_v1"})
prob = get_inference_builder(spec.inference.builder)(system, **kw)
ds = prob.datasets[0]; grid = ds.adaptive_grid; sim_ad = prob.simulators[0]
obs = jnp.asarray(prob.observed_image); err = jnp.asarray(prob.error_map); errn = np.asarray(err)
mask = np.asarray(ds.mask, bool) if ds.mask is not None else np.ones(errn.shape, bool)
z0 = np.load(f"{BASE}/runs/{SID}/fitsersic_truthinit_v1/undersampling_check/arrays.npz")["z0"]
Sfree = np.load(f"{BASE}/runs/{SID}/fitsersic_v1/mclmc/arrays.npz")["samples_z"]; D = Sfree.shape[-1]
zmean = Sfree.reshape(-1, D).mean(0); zstd = Sfree.reshape(-1, D).std(0)
post = json.load(open("experiments/vela_f140w_v3/fit_results/vela22_truthinit_v1/summary.json"))["sersic_free"]["stats"]
keys = list(prob.labeled_samples(jnp.asarray(z0)[None]).keys())
def theta(z):
    lab = prob.labeled_samples(jnp.asarray(z)[None]); return jnp.stack([jnp.asarray(lab[k], jnp.float64).reshape(()) for k in keys])
yy, xx = np.indices(errn.shape)
def disk(cy_arc, cx_arc, r_pix):
    row = (errn.shape[0] - 1) / 2 + cy_arc / cfg.delta_pix; col = (errn.shape[1] - 1) / 2 + cx_arc / cfg.delta_pix
    return np.hypot(yy - row, xx - col) <= r_pix
LL = system.truth_x[1][0]; disk4 = disk(float(LL["center_y"]), float(LL["center_x"]), 4.0)
out = {}
# ---------------- A. lens-light-only mismatch at the truth (twin non-lstsq model, as vela22_cusp_conv.py)
tps = resolve_truth_prior_spec(yaml.safe_load(open("experiments/vela_f140w_v3/campaign.yaml"))["dataset"].get("truth_prior"))
def cp(g, c): return {p: (float(d["value"]) if d["dist"] == "Fixed" else _make_dist(d, p)) for p, d in tps[g][c].items()}
src_p = {k: tfd.Normal(0.0, 1.0) for k in ["Ie", "R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]}
model_nl = LensModel([Plane(mass=[Component(epl.EPL(50), cp("lens_mass", "0")), Component(shear.Shear(), cp("lens_mass", "1"))],
                            light=[Component(sersic.SersicEllipse(use_lstsq=False), cp("lens_light", "0"))]),
                      Plane(deflection_ratio=1.0, light=[Component(sersic.SersicEllipse(use_lstsq=False), src_p)])])
t = copy.deepcopy(system.truth_x); t[2][0] = dict(Ie=0.0, R_sersic=0.2, n_sersic=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
p_nl = truth_x_to_scene_params(t, model_nl)
LL_ad = np.asarray(AdaptiveSceneSimulator(model_nl, cfg1, grid).simulate(p_nl)).squeeze()
LL_32 = np.asarray(SceneSimulator(model_nl, dataclasses.replace(cfg, supersample=32)).simulate(p_nl)).squeeze()
LL_file = np.load(f"{BASE}/dataset/systems/{SID}/lens_light_only.npy").astype(np.float64)
repro = np.abs(LL_32 - LL_file).max() / errn.min()
print(f"A. ss=32 twin render reproduces lens_light_only.npy to {repro:.2e} sigma (float32 file)")
dA = (LL_ad - LL_32) / errn
def summarize(d, tag):
    iy, ix = np.unravel_index(np.argmax(np.abs(d)), d.shape)
    s = dict(max_abs=float(np.abs(d).max()), argmax_rowcol=[int(iy), int(ix)], max_abs_outside_disk4=float(np.abs(d[~disk4]).max()),
             chi2_total=float((d ** 2).sum()), chi2_disk4=float((d[disk4] ** 2).sum()), chi2_outside_disk4=float((d[~disk4] ** 2).sum()),
             n_above_0p1_outside=int((np.abs(d[~disk4]) > 0.1).sum()))
    print(f"{tag}: max|D|/sig {s['max_abs']:.3f} at {s['argmax_rowcol']}, outside r<=4px disk {s['max_abs_outside_disk4']:.3f}; "
          f"chi2 total {s['chi2_total']:.3f} (disk {s['chi2_disk4']:.3f}, outside {s['chi2_outside_disk4']:.3f}); pixels >0.1sig outside: {s['n_above_0p1_outside']}")
    return s
out["A_truth_lens_light"] = summarize(dA, "A. adaptive8 - ss32, lens light only, truth"); out["A_repro_sigma"] = float(repro)
cy, cx = np.unravel_index(np.argmax(LL_32), LL_32.shape); out["A_centre_3x3"] = np.round(dA[cy-1:cy+2, cx-1:cx+2], 3).tolist()
print("   centre 3x3 (D/sig):", out["A_centre_3x3"])
# ---------------- B. Fisher linear response at z0
def model_img(z): return jnp.reshape(sim_ad.lstsq_simulate(prob.constrained(z[None]), obs, err), errn.shape)
J = np.moveaxis(np.asarray(jax.jacfwd(model_img)(jnp.asarray(z0))), -1, 0)  # (D, ny, nx)
Jt = np.asarray(jax.jacfwd(theta)(jnp.asarray(z0)))                # (K, D) d theta / d z
sig_post = np.array([post[k]["std"] for k in keys]); mean_post = np.array([post[k]["mean"] for k in keys])
def fisher(m):
    Jm = J[:, m] / errn[m]; F = Jm @ Jm.T; Fi = np.linalg.inv(F); return Jm, F, Fi
def bias(Jm, Fi, d_img, m):
    dz = -Fi @ (Jm @ d_img[m]); dth = Jt @ dz; return dz, dth
Jm, F, Fi = fisher(mask)
cov_th = Jt @ Fi @ Jt.T; sig_fisher = np.sqrt(np.diag(cov_th))
dz, dth = bias(Jm, Fi, dA, mask)
print("B. Fisher sigma / MCLMC sigma (linearisation check) and predicted cusp-quadrature bias, DC-1 fit:")
print(f"   {'param':32s} {'sigF/sigMC':>10s} {'pred bias/sig':>14s} {'observed z':>11s}")
rows = {}
for i, k in enumerate(keys):
    zobs = post[k]["z"]; rows[k] = dict(sig_ratio=float(sig_fisher[i] / sig_post[i]), pred_bias_sigma=float(dth[i] / sig_post[i]), observed_z=zobs)
    if "/mass/" in k or "planes/0/light" in k: print(f"   {k:32s} {sig_fisher[i]/sig_post[i]:10.2f} {dth[i]/sig_post[i]:14.3f} {('%.1f' % zobs) if zobs is not None else '-':>11s}")
out["B_fisher_truth"] = rows
# ---------------- C. mismatch at the DC-1 posterior mean (lens light + Sersic source, same lstsq coefficients)
pm = prob.constrained(jnp.asarray(zmean)[None])
coef = np.asarray(sim_ad.lstsq_simulate(pm, obs, err, return_coeffs=True)).ravel()
B_ad = np.asarray(sim_ad.lstsq_simulate(pm, obs, err, return_stacked=True)).squeeze()      # (ny, nx, ncomp)
sim32 = SceneSimulator(prob.model, dataclasses.replace(cfg, supersample=32))
B_32 = np.asarray(sim32.lstsq_simulate(pm, obs, err, return_stacked=True)).squeeze()
m_ad = B_ad @ coef; m_32 = B_32 @ coef; dC = (m_ad - m_32) / errn
dC_LL = (B_ad[..., 0] - B_32[..., 0]) * coef[0] / errn; dC_src = (B_ad[..., 1] - B_32[..., 1]) * coef[1] / errn
out["C_mean_full"] = summarize(dC, "C. adaptive8 - ss32 at the DC-1 posterior mean (lens light + Sersic source)")
out["C_mean_lens_light"] = summarize(dC_LL, "   ... lens-light column only"); out["C_mean_source"] = summarize(dC_src, "   ... source column only")
dzC, dthC = bias(Jm, Fi, dC, mask); out["C_fisher_mean"] = {k: float(dthC[i] / sig_post[i]) for i, k in enumerate(keys)}
print("   predicted bias/sig from the posterior-mean mismatch:", {k.split('/')[-1] + ("_sh" if "/mass/1" in k else ""): round(float(dthC[i] / sig_post[i]), 3) for i, k in enumerate(keys) if "/mass/" in k})
# ---------------- D. masked-disk refit predictions
mask2 = mask & ~disk4; Jm2, F2, Fi2 = fisher(mask2); sig2 = np.sqrt(np.diag(Jt @ Fi2 @ Jt.T))
dz2, dth2 = bias(Jm2, Fi2, dA, mask2)
out["D_masked"] = {k: dict(width_ratio=float(sig2[i] / sig_fisher[i]), residual_bias_sigma=float(dth2[i] / sig_post[i])) for i, k in enumerate(keys)}
print(f"D. r<=4 px disk masked ({int(disk4.sum())} px): Fisher width ratio (masked/full) and residual quadrature bias/sig:")
for i, k in enumerate(keys):
    if "/mass/" in k or "planes/0/light" in k: print(f"   {k:32s} {sig2[i]/sig_fisher[i]:10.2f} {dth2[i]/sig_post[i]:14.3f}")
# ---------------- figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
im = ax[0].imshow(dA, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5); ax[0].set_title("A: (adaptive8 - ss32)/sigma\nlens light only, at the truth")
ax[1].imshow(dA[cy-10:cy+11, cx-10:cx+11], origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5, extent=[cx-10.5, cx+10.5, cy-10.5, cy+10.5])
ax[1].add_patch(plt.Circle((disk4.shape[1] and ((errn.shape[1]-1)/2 + float(LL["center_x"])/cfg.delta_pix), (errn.shape[0]-1)/2 + float(LL["center_y"])/cfg.delta_pix), 4.0, fill=False, color="k", ls="--"))
ax[1].set_title(f"zoom, lens centre; dashed = r<=4 px\nmax |D|/sig {out['A_truth_lens_light']['max_abs']:.2f}, outside disk {out['A_truth_lens_light']['max_abs_outside_disk4']:.3f}")
ax[2].imshow(dC, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5); ax[2].set_title(f"C: at the DC-1 posterior mean (LL + src)\nmax {out['C_mean_full']['max_abs']:.2f}, outside disk {out['C_mean_full']['max_abs_outside_disk4']:.3f}")
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.colorbar(im, ax=ax.tolist(), fraction=0.02, label="quadrature mismatch / sigma")
fig.savefig(f"{HERE}/vela22_cusp_bias.png", dpi=110, bbox_inches="tight")
json.dump(out, open(f"{HERE}/vela22_cusp_bias.json", "w"), indent=1)
print("wrote", f"{HERE}/vela22_cusp_bias.png")
