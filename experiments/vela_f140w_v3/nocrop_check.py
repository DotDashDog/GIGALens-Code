"""Uncropped sources through the current truths: cut-off metrics on the 2x canvas + frame-edge numbers."""
import json, os, pickle, numpy as np
from gigalens_research.simtests.experiments import vela_simulated as vs
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.simulator import SimulatorConfig
from gigalens_research.simulations.image_based_light import ImageBasedLight
R = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"
man = json.load(open(R + "/manifest.json")); ex = man["extra"]; cut = ex["cutoff"]
prior_spec = vs.resolve_truth_prior_spec(ex.get("truth_prior"))
num_pix, dp, bkg = ex["num_pix"], ex["delta_pix"], ex["background_rms"]
n_canvas = num_pix * cut["canvas_factor"]
print(f"cutoff thresholds: outside <= {cut['max_flux_outside']:.0%}, border < {cut['max_border_sb_sigma']} sigma (outer {cut['border_width_pix']} px)")
print(f"{'src':<7}{'frame':>6}{'edge0.15':>9}{'edge0.3':>8}{'maxSBedge':>10} | uncropped: {'outside':>8}{'border':>8}{'pass':>6} | hard crop: {'outside':>8}{'border':>7}")
for sid in man["system_ids"]:
    short = sid.split("_")[0]; sd = os.path.join(R, "systems", sid)
    truth = pickle.load(open(os.path.join(sd, "truth_x.pkl"), "rb")); psf = np.load(os.path.join(sd, "psf.npy"))
    gen = json.load(open(os.path.join(sd, "generation.json")))["metrics"]
    raw, s, _, _ = vs._load_pristine_source(f"/pscratch/sd/l/linusu/gigalens/vela_sources_pristine/{short}_cam12_a0.400_f140w", False)
    raw = np.asarray(raw, float); n = raw.shape[0]
    yy, xx = np.indices(raw.shape); d = np.minimum.reduce([yy, xx, n-1-yy, n-1-xx]) * s
    amp = float(truth[2][0]["amp"]); sig_sb = bkg / dp**2
    f15, f30 = raw[d <= 0.15].sum()/raw.sum(), raw[d <= 0.3].sum()/raw.sum()
    # per-image-pixel SB at the frame edge: bin the outer 0.15" band to 0.065" (conservative: mu>1 shrinks the footprint)
    k = max(1, int(round(dp / s))); edge = raw * (d <= 0.15) * amp
    m = (n // k) * k; binned = edge[:m, :m].reshape(m//k, k, m//k, k).mean(axis=(1, 3))
    max_sb_edge = binned.max() / sig_sb
    full, _ = vs.preprocess_source(raw, s, crop_radius_arcsec=None, recenter=True)
    cfg = SimulatorConfig(delta_pix=dp, num_pix=n_canvas, supersample=cut["canvas_supersample"], kernel=psf,
                          likelihood_precision=ex["likelihood_precision"], conv_precision=ex["conv_precision"])
    model = vs._build_truth_scene_model(prior_spec, ImageBasedLight(full, s))
    img = vs._render(SceneSimulator(model, cfg), model, vs._with(truth, 1, 0, Ie=0.0))
    outside = vs.flux_outside_cutout(img, num_pix)
    o = (n_canvas - num_pix) // 2; cutout = img[o:o+num_pix, o:o+num_pix]
    border = vs.border_max(cutout, cut["border_width_pix"]) / bkg
    ok = outside <= cut["max_flux_outside"] and border < cut["max_border_sb_sigma"]
    print(f"{short:<7}{n*s:>5.1f}\"{100*f15:>8.2f}%{100*f30:>7.2f}%{max_sb_edge:>9.1f}σ | {100*outside:>17.2f}%{border:>7.2f}σ{'yes' if ok else 'NO':>6} | {100*gen['flux_outside_frac']:>18.3f}%{gen['border_sb_sigma']:>6.2f}σ")
