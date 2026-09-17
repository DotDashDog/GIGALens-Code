"""Raw-vs-smoothed source residuals and their lensed, noise-normalised counterparts.

For every system of a vela_simulated dataset generated with ``source_smooth_sigma_pix``:
  * re-render the noiseless image at the stored truth from the SMOOTHED source (must
    reproduce the stored ``noiseless_image.npy``) and from the RAW source;
  * source residual: raw - smoothed source map (cps/arcsec^2, at the calibrated amp),
    shown transposed to the image axes like the gallery panel;
  * lens residual: (raw - smoothed) / sigma_tot with sigma_tot^2 = bkg^2 + smoothed/exp_time,
    plus delta chi^2 = sum of its square -- the chi^2 floor the raw set would carry.

Usage: smoothing_residuals.py DATASET_DIR OUT_DIR
"""
import os, sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from gigalens_research.simtests.experiments import vela_simulated as V
from gigalens_research.simtests.system import load_manifest
from gigalens.simulator import SimulatorConfig
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens_research.simulations.image_based_light import ImageBasedLight

ds, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
man = load_manifest(ds); ex = man["extra"]
pre = ex["source_preprocessing"]; sig_pix = pre["smooth_sigma_pix"]
assert sig_pix is not None, "dataset was generated without source_smooth_sigma_pix"
bkg, exp_t = ex["noise"]["background_rms"], ex["noise"]["exp_time"]
src_root = ex.get("source_root") or "/pscratch/sd/l/linusu/gigalens/vela_sources_pristine"
sf, cam, filt = ex["scale_factor"], ex["cam"], ex["filter"]
prior_spec = ex["truth_prior"] if "truth_prior" in ex else None
ids = man["system_ids"]
res = {}
fig, axes = plt.subplots(len(ids), 4, figsize=(20, 4.6 * len(ids)))
WIN = 1.2  # half-window of the source panels, arcsec
for row, sid in enumerate(ids):
    sdir = os.path.join(ds, "systems", sid)
    truth = pickle.load(open(f"{sdir}/truth_x.pkl", "rb"))
    meta = json.load(open(f"{sdir}/meta.json"))
    psf = np.load(f"{sdir}/psf.npy"); stored = np.load(f"{sdir}/noiseless_image.npy").astype(np.float64)
    sim = sid.split("_")[0].replace("vela", "")
    src_dir = os.path.join(src_root, V._source_dir_name(sim, cam, sf, filt))
    sb_raw0, src_scale, _, _ = V._load_pristine_source(src_dir, bool(ex.get("transpose_image", False)))
    kw = dict(crop_radius_arcsec=pre["crop_radius_arcsec"], recenter=pre["recenter"],
              crop_taper_arcsec=pre.get("crop_taper_arcsec"))
    sb_sm, _ = V.preprocess_source(sb_raw0, src_scale, smooth_sigma_pix=sig_pix, **kw)
    sb_raw, _ = V.preprocess_source(sb_raw0, src_scale, smooth_sigma_pix=None, **kw)
    amp = float(truth[2][0]["amp"])
    spec = V.resolve_truth_prior_spec(ex.get("truth_prior_overrides")) if prior_spec is None else prior_spec
    cfg = SimulatorConfig(delta_pix=meta["delta_pix"], num_pix=meta["num_pix"], supersample=int(ex["supersample"]),
                          kernel=psf, likelihood_precision=meta.get("likelihood_precision", "float64"))
    imgs = {}
    for name, sb in [("smoothed", sb_sm), ("raw", sb_raw)]:
        model = V._build_truth_scene_model(spec, ImageBasedLight(sb, src_scale))
        imgs[name] = V._render(SceneSimulator(model, cfg), model, truth)
    check = float(np.abs(imgs["smoothed"] - stored).max())
    sig_tot = np.sqrt(bkg ** 2 + np.clip(imgs["smoothed"], 0, None) / exp_t)
    nres = (imgs["raw"] - imgs["smoothed"]) / sig_tot
    lensed_src = imgs["smoothed"]  # full image incl. lens light; arcs dominate the residual
    res[sid] = {"render_check_max_abs": check, "delta_chi2": float((nres ** 2).sum()),
                "max_abs_nres": float(np.abs(nres).max()), "rms_nres_detected": float(nres[imgs["smoothed"] > 3 * sig_tot].std()),
                "n_pix_gt_1sig": int((np.abs(nres) > 1).sum()), "n_pix_gt_3sig": int((np.abs(nres) > 3).sum()),
                "source_flux_change_frac": float(sb_sm.sum() / sb_raw.sum() - 1.0)}
    print(f"{sid}: render check {check:.1e}; dchi2 {res[sid]['delta_chi2']:.0f}; max |res| {res[sid]['max_abs_nres']:.2f} sigma; "
          f">1sig {res[sid]['n_pix_gt_1sig']} px, >3sig {res[sid]['n_pix_gt_3sig']} px", flush=True)
    # ---- source panels (transposed to image axes; window WIN") ----
    n = sb_sm.shape[0]; c = (n - 1) / 2; w = int(WIN / src_scale)
    sl = slice(int(c - w), int(c + w) + 1)
    s_sm = (sb_sm * amp).T[sl, sl]; s_res = ((sb_raw - sb_sm) * amp).T[sl, sl]
    ext = [-WIN, WIN, -WIN, WIN]
    ax = axes[row, 0]; im = ax.imshow(s_sm, cmap="inferno", norm=PowerNorm(0.5, vmin=0), origin="lower", extent=ext)
    ax.set_title(f"{sid.split('_')[0]}: smoothed source (sigma {sig_pix:g} cells), x amp {amp:.2f}", fontsize=10); plt.colorbar(im, ax=ax, fraction=0.046, label="cps/arcsec$^2$")
    v = np.percentile(np.abs(s_res), 99.9)
    ax = axes[row, 1]; im = ax.imshow(s_res, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower", extent=ext)
    ax.set_title("raw - smoothed source (99.9% scale)", fontsize=10); plt.colorbar(im, ax=ax, fraction=0.046, label="cps/arcsec$^2$")
    # ---- lens panels ----
    fov = meta["num_pix"] * meta["delta_pix"] / 2; ext2 = [-fov, fov, -fov, fov]
    ax = axes[row, 2]; im = ax.imshow(lensed_src, cmap="inferno", norm=PowerNorm(0.5, vmin=0), origin="lower", extent=ext2)
    ax.set_title("noiseless image, smoothed source", fontsize=10); plt.colorbar(im, ax=ax, fraction=0.046, label="cps/px")
    vv = max(1.0, np.abs(nres).max())
    ax = axes[row, 3]; im = ax.imshow(nres, cmap="RdBu_r", vmin=-vv, vmax=vv, origin="lower", extent=ext2)
    ax.set_title(f"(raw - smoothed) / sigma_tot   dchi2 = {res[sid]['delta_chi2']:.0f}, max {res[sid]['max_abs_nres']:.1f} sigma", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, label="sigma")
    for a in axes[row]: a.set_xlabel("arcsec")
fig.suptitle(f"Raw vs smoothed VELA sources (Gaussian sigma = {sig_pix:g} source cells = {sig_pix * src_scale:.4f}\"): "
             f"source residuals and noise-normalised lensed residuals at the same truth", y=1.0)
plt.tight_layout(); plt.savefig(os.path.join(out, "smoothing_residuals.png"), dpi=80)
json.dump(res, open(os.path.join(out, "smoothing_residuals.json"), "w"), indent=1)
print("wrote", os.path.join(out, "smoothing_residuals.{png,json}"))
