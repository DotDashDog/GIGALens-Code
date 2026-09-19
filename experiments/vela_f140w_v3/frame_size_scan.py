"""How large a cutout removes the cut-off SELECTION part of the size/magnification confound?

For every source of the smoothed v3 set, draw K lens + source-position + brightness
configurations from the campaign prior (independent of the source, as in the
generator), render the lensed source once on a large canvas (3 x 120 px, low
supersample: the cut-off rule is a 10%-flux / 1-sigma-border test), and evaluate the
generator's cut-off rule for every candidate frame size N from that one render:

    accept(N) = flux outside the central N x N <= max_flux_outside
                and max border SB (outer 3 px) <= max_border_sb_sigma * background_rms

Then, per N: acceptance fraction per source (expected redraws = 1/p - 1) and the
Spearman correlation across accepted draws of the source half-light radius with
theta_E (pure selection: the prior does not know the source) and with the
flux-weighted magnification (selection + geometry). The all-draws correlation is
the geometric floor that no frame size can remove.

Usage: frame_size_scan.py DATASET_DIR OUT_DIR [K] [canvas_supersample]
"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
from jax import random
from gigalens_research.simtests.experiments import vela_simulated as V
from gigalens_research.simtests.system import load_manifest
from gigalens.simulator import SimulatorConfig
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens_research.simulations.image_based_light import ImageBasedLight

jax.config.update("jax_enable_x64", True)
ds, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 100
SS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(out, exist_ok=True)
man = load_manifest(ds); ex = man["extra"]
pre = ex["source_preprocessing"]
bkg = ex["noise"]["background_rms"]
cut = ex["cutoff"]; calib = ex["calibration"]
src_root = ex.get("source_root") or "/pscratch/sd/l/linusu/gigalens/vela_sources_pristine"
sf, cam, filt = ex["scale_factor"], ex["cam"], ex["filter"]
spec = ex["truth_prior"]
num_pix, delta_pix = int(ex["num_pix"]), float(ex["delta_pix"])
ids = man["system_ids"]
psf = np.load(os.path.join(ds, "systems", ids[0], "psf.npy"))
NC = 3 * num_pix                       # 360 px = 23.4" canvas
FRAMES = [120, 140, 160, 180, 200, 240, 280, 320]
sb_target = calib["peak_sb"]["sb_mag_arcsec2"]; n_peak = int(calib["peak_sb"]["n_brightest_pix"])
assert sb_target["dist"] == "Normal"
prior = V.build_truth_prior(spec)
rng = np.random.default_rng(0)
key = random.PRNGKey(12345)
print(f"canvas {NC} px at supersample {SS}; K={K} draws per source; cutoff {cut}; peak-SB target {sb_target}", flush=True)


def crop(img, n):
    c = (img.shape[0] - n) // 2
    return img[c:c + n, c:c + n]


res = {}
for si, sid in enumerate(ids):
    sim = sid.split("_")[0].replace("vela", "")
    src_dir = os.path.join(src_root, V._source_dir_name(sim, cam, sf, filt))
    sb_raw0, src_scale, _, src_meta = V._load_pristine_source(src_dir, bool(ex.get("transpose_image", False)))
    sb, _ = V.preprocess_source(sb_raw0, src_scale, crop_radius_arcsec=pre["crop_radius_arcsec"], recenter=pre["recenter"],
                                crop_taper_arcsec=pre.get("crop_taper_arcsec"), smooth_sigma_pix=pre["smooth_sigma_pix"])
    zp_ab = V.ab_zeropoint_from_photfnu(float(src_meta["photfnu_Jy"]))
    unlensed = float(sb.sum()) * src_scale ** 2
    # half-light radius of the (smoothed, recentred) source about its flux centroid
    n = sb.shape[0]; yy, xx = np.mgrid[:n, :n]
    cy, cx = (sb * yy).sum() / sb.sum(), (sb * xx).sum() / sb.sum()
    r = np.hypot(yy - cy, xx - cx).ravel() * src_scale; f = sb.ravel()
    o = np.argsort(r); cf = np.cumsum(f[o]) / f.sum()
    r50 = float(r[o][np.searchsorted(cf, 0.5)]); r90 = float(r[o][np.searchsorted(cf, 0.9)])
    model = V._build_truth_scene_model(spec, ImageBasedLight(sb, src_scale))
    cfg = SimulatorConfig(delta_pix=delta_pix, num_pix=NC, supersample=SS, kernel=psf, likelihood_precision="float64")
    simc = SceneSimulator(model, cfg)
    draws = []
    for k in range(K):
        tkey = random.fold_in(random.fold_in(key, si), k)
        truth = V._sample_to_legacy(prior.sample(seed=tkey))
        img = V._render(simc, model, V._with(V._with(truth, 1, 0, Ie=0.0), 2, 0, amp=1.0))
        target = float(rng.normal(sb_target["loc"], sb_target["scale"]))
        # the generator measures the peak SB on the cutout render: use the central 120 px
        amp = V.calibrate_amp_peak_sb(sb_mag_arcsec2=target, zeropoint_ab=zp_ab,
                                      peak_sb_unit_amp=V.peak_surface_brightness(crop(img, num_pix), n_peak, delta_pix))
        tot = float(img.sum())
        row = {"theta_E": truth[0][0]["theta_E"], "beta": float(np.hypot(truth[2][0]["center_x"], truth[2][0]["center_y"])),
               "target": target, "amp": amp, "mu_canvas": tot / unlensed, "mu_cutout": float(crop(img, num_pix).sum()) / unlensed}
        for N in FRAMES:
            c = crop(img, N)
            row[f"out_{N}"] = 1.0 - float(c.sum()) / tot
            row[f"border_{N}"] = amp * V.border_max(c, int(cut["border_width_pix"])) / bkg
            row[f"ok_{N}"] = bool(row[f"out_{N}"] <= cut["max_flux_outside"] and row[f"border_{N}"] <= cut["max_border_sb_sigma"])
        draws.append(row)
    acc = {N: float(np.mean([d[f"ok_{N}"] for d in draws])) for N in FRAMES}
    res[sid] = {"r50_arcsec": r50, "r90_arcsec": r90, "acceptance": acc, "draws": draws}
    print(f"{sid.split('_')[0]}: R50 {r50:.3f}\" R90 {r90:.3f}\"; acceptance " +
          " ".join(f"{N}:{acc[N]:.2f}" for N in FRAMES), flush=True)

# ---- correlations across accepted draws ------------------------------------------
summary = {"frames": FRAMES, "K": K, "canvas_supersample": SS, "canvas_pix": NC}
R50 = np.array([res[s]["r50_arcsec"] for s in ids])
allrows = [(res[s]["r50_arcsec"], d) for s in ids for d in res[s]["draws"]]
r50_all = np.array([a for a, _ in allrows]); tE_all = np.array([d["theta_E"] for _, d in allrows])
mu_all = np.array([d["mu_canvas"] for _, d in allrows]); muc_all = np.array([d["mu_cutout"] for _, d in allrows])
summary["no_selection"] = {"rho_r50_thetaE": float(spearmanr(r50_all, tE_all)[0]),
                           "rho_r50_mu_canvas": float(spearmanr(r50_all, mu_all)[0]),
                           "rho_r50_mu_cutout": float(spearmanr(r50_all, muc_all)[0]),
                           "rho_r50_median_mu_per_source": float(spearmanr(R50, [np.median([d["mu_canvas"] for d in res[s]["draws"]]) for s in ids])[0])}
per_N = {}
for N in FRAMES:
    ok = np.array([d[f"ok_{N}"] for _, d in allrows])
    accs = np.array([res[s]["acceptance"][N] for s in ids])
    med_mu = [np.median([d["mu_canvas"] for d in res[s]["draws"] if d[f"ok_{N}"]] or [np.nan]) for s in ids]
    med_tE = [np.median([d["theta_E"] for d in res[s]["draws"] if d[f"ok_{N}"]] or [np.nan]) for s in ids]
    per_N[N] = {"min_acceptance": float(accs.min()), "acceptance_per_source": dict(zip(ids, accs.tolist())),
                "expected_redraws_max": float(1 / max(accs.min(), 1e-9) - 1),
                "rho_r50_thetaE_accepted": float(spearmanr(r50_all[ok], tE_all[ok])[0]),
                "rho_r50_mu_accepted": float(spearmanr(r50_all[ok], mu_all[ok])[0]),
                "rho_r50_median_thetaE_per_source": float(spearmanr(R50, med_tE, nan_policy="omit")[0]),
                "rho_r50_median_mu_per_source": float(spearmanr(R50, med_mu, nan_policy="omit")[0]),
                "median_thetaE_accepted_per_source": dict(zip(ids, [float(x) for x in med_tE])),
                "rho_r50_acceptance": float(spearmanr(R50, accs)[0])}
    print(f"N={N:3d} ({N*delta_pix:.1f}\"): min acceptance {accs.min():.2f} (max E[redraws] {per_N[N]['expected_redraws_max']:.1f}); "
          f"rho(R50, theta_E | accepted) {per_N[N]['rho_r50_thetaE_accepted']:+.2f}; rho(R50, mu | accepted) {per_N[N]['rho_r50_mu_accepted']:+.2f}; "
          f"per-source medians: theta_E {per_N[N]['rho_r50_median_thetaE_per_source']:+.2f}, mu {per_N[N]['rho_r50_median_mu_per_source']:+.2f}", flush=True)
print(f"no selection: rho(R50, theta_E) {summary['no_selection']['rho_r50_thetaE']:+.2f}; rho(R50, mu_canvas) {summary['no_selection']['rho_r50_mu_canvas']:+.2f}; "
      f"rho(R50, mu_cutout) {summary['no_selection']['rho_r50_mu_cutout']:+.2f}; per-source median mu {summary['no_selection']['rho_r50_median_mu_per_source']:+.2f}")
summary["per_frame"] = per_N
summary["sources"] = {s: {k: v for k, v in res[s].items() if k != "draws"} for s in ids}
json.dump({"summary": summary, "draws": {s: res[s]["draws"] for s in ids}}, open(os.path.join(out, "frame_size_scan.json"), "w"), indent=1)

# ---- figure ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
order = np.argsort(R50)
cm = plt.get_cmap("viridis")
ax = axes[0]
for j, i in enumerate(order):
    s = ids[i]; ax.plot(FRAMES, [res[s]["acceptance"][N] for N in FRAMES], "o-", color=cm(j / max(len(ids) - 1, 1)),
                        label=f"{s.split('_')[0]} (R50 {R50[i]:.2f}\")")
ax.axhline(0.95, color="k", ls=":", lw=0.8); ax.set_xlabel("frame size N (px, 0.065\"/px)"); ax.set_ylabel("cut-off acceptance fraction of prior draws")
ax.set_title("acceptance vs frame size, per source (colour = R50)"); ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 1.02)
ax = axes[1]
ax.plot(FRAMES, [per_N[N]["rho_r50_thetaE_accepted"] for N in FRAMES], "s-", label=r"$\rho$(R50, $\theta_E$) accepted draws (selection only)")
ax.plot(FRAMES, [per_N[N]["rho_r50_mu_accepted"] for N in FRAMES], "o-", label=r"$\rho$(R50, $\mu$) accepted draws")
ax.axhline(summary["no_selection"]["rho_r50_mu_canvas"], color="C1", ls="--", lw=1, label=r"$\rho$(R50, $\mu$) all draws: geometric floor")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("frame size N (px)"); ax.set_ylabel("Spearman rho (pooled draws)")
ax.set_title("size–lens correlations among accepted draws"); ax.legend(fontsize=8)
ax = axes[2]
for j, i in enumerate(order):
    s = ids[i]; d = res[s]["draws"]
    ax.scatter([R50[i]] * len(d), [x["theta_E"] for x in d], s=6, color="0.8")
    okd = [x for x in d if x["ok_120"]]
    ax.scatter([R50[i]] * len(okd), [x["theta_E"] for x in okd], s=8, color=cm(j / max(len(ids) - 1, 1)))
ax.set_xlabel("source R50 (arcsec)"); ax.set_ylabel(r"$\theta_E$ (arcsec)"); ax.set_title("draws accepted at N = 120 (colour) vs all (grey)")
fig.suptitle(f"Frame-size scan: {K} prior draws per source on a {NC}-px canvas (supersample {SS}); cut-off rule of the campaign", y=1.0)
plt.tight_layout(); plt.savefig(os.path.join(out, "frame_size_scan.png"), dpi=90)
print("wrote", os.path.join(out, "frame_size_scan.png"))
