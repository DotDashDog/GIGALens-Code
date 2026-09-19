"""Boundary treatments for the v3 F140W sources, measured in the IMAGE plane.

For each review-set system (same truth, same amp, same PSF/noise as the dataset) lens three
versions of the source: the hard 1.5" crop the dataset uses, a cosine taper (1 inside r0, 0
beyond r1, centred on the same 1.5"), and the uncropped VELA frame. The light each treatment
removes relative to the uncropped frame is expressed in noise units: per-pixel peak and the
quadrature sum over the image (the sqrt of the chi^2 the truncation would contribute).
Also lenses the outermost 0.15" band of the VELA frame (what an uncropped source truncates).
"""
import json, os, pickle, sys
import numpy as np
from gigalens_research.simtests.experiments import vela_simulated as vs

R = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
R0, R1 = 1.2, 1.8        # taper: weight 1 inside R0, cosine to 0 at R1 (half weight at 1.5")
BAND = 0.15              # outer band of the VELA frame, arcsec

man = json.load(open(os.path.join(R, "manifest.json")))
ex = man["extra"]
bkg, t_exp, dp, num_pix, ss = ex["background_rms"], ex["exp_time"], ex["delta_pix"], ex["num_pix"], ex["supersample"]
prior_spec = vs.resolve_truth_prior_spec(ex.get("truth_prior"))
crop = ex["source_preprocessing"]["crop_radius_arcsec"] if isinstance(ex["source_preprocessing"], dict) and "crop_radius_arcsec" in ex["source_preprocessing"] else 1.5

from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.simulator import SimulatorConfig
from gigalens_research.simulations.image_based_light import ImageBasedLight


def shift_int(img, dy, dx):
    n = img.shape[0]
    out = np.zeros_like(img)
    ys, yd = (slice(0, n - dy), slice(dy, n)) if dy >= 0 else (slice(-dy, n), slice(0, n + dy))
    xs, xd = (slice(0, n - dx), slice(dx, n)) if dx >= 0 else (slice(-dx, n), slice(0, n + dx))
    out[yd, xd] = img[ys, xs]
    return out


def snr(delta, sig):
    z = delta / sig
    return float(np.sqrt((z ** 2).sum())), float(z.max())


rows, images = {}, {}
for sid in man["system_ids"]:
    short = sid.split("_")[0]
    sd = os.path.join(R, "systems", sid)
    truth = pickle.load(open(os.path.join(sd, "truth_x.pkl"), "rb"))
    psf = np.load(os.path.join(sd, "psf.npy"))
    lens_only = np.load(os.path.join(sd, "lens_light_only.npy")).astype(np.float64)
    noiseless = np.load(os.path.join(sd, "noiseless_image.npy")).astype(np.float64)
    gen = json.load(open(os.path.join(sd, "generation.json")))
    src_dir = f"/pscratch/sd/l/linusu/gigalens/vela_sources_pristine/{short}_cam12_a0.400_f140w"
    raw, s, _, _ = vs._load_pristine_source(src_dir, False)
    raw = np.asarray(raw, dtype=np.float64)
    n = raw.shape[0]
    # same geometry as the generator: crop about the raw centroid, recenter on the cropped centroid
    hard, info = vs.preprocess_source(raw, s, crop_radius_arcsec=crop, recenter=True)
    dx_a, dy_a = info["recenter_shift_arcsec"]
    dy, dx = int(round(dy_a / s)), int(round(dx_a / s))
    cy, cx = vs._flux_centroid(raw)
    yy, xx = np.indices(raw.shape)
    r = np.hypot(yy - cy, xx - cx) * s
    w = np.where(r <= R0, 1.0, np.where(r >= R1, 0.0, 0.5 * (1 + np.cos(np.pi * (r - R0) / (R1 - R0)))))
    taper = shift_int(raw * w, dy, dx)
    full = shift_int(raw, dy, dx)
    d_edge = np.minimum.reduce([yy, xx, n - 1 - yy, n - 1 - xx]) * s
    band = shift_int(raw * (d_edge <= BAND), dy, dx)
    assert np.allclose(hard, shift_int(raw * (r <= crop), dy, dx)), short  # geometry reproduced

    cfg = SimulatorConfig(delta_pix=dp, num_pix=num_pix, supersample=ss, kernel=psf,
                          likelihood_precision=ex["likelihood_precision"], conv_precision=ex["conv_precision"])
    t_src = vs._with(truth, 1, 0, Ie=0.0)
    im = {}
    for name, src in (("hard", hard), ("taper", taper), ("full", full), ("band", band)):
        model = vs._build_truth_scene_model(prior_spec, ImageBasedLight(src, s))
        im[name] = vs._render(SceneSimulator(model, cfg), model, t_src)
    # sanity: the hard-crop render must reproduce the dataset's lensed source
    ds_src = noiseless - lens_only
    rel = float(np.abs(im["hard"] - ds_src).max() / ds_src.max())
    sig = np.sqrt(bkg ** 2 + np.clip(im["full"] + lens_only, 0, None) / t_exp)
    rem_hard, rem_taper = im["full"] - im["hard"], im["full"] - im["taper"]
    sn_h, pk_h = snr(rem_hard, sig)
    sn_t, pk_t = snr(rem_taper, sig)
    sn_b, pk_b = snr(im["band"], sig)
    tot = full.sum()
    rows[short] = dict(theta_E=float(truth[0][0]["theta_E"]), amp=float(truth[2][0]["amp"]), fov=n * s,
                       recon_rel_err=rel,
                       f_removed_hard=1 - hard.sum() / tot, f_removed_taper=1 - taper.sum() / tot, f_band=band.sum() / tot,
                       snr_removed_hard=sn_h, peak_removed_hard=pk_h,
                       snr_removed_taper=sn_t, peak_removed_taper=pk_t,
                       snr_band=sn_b, peak_band=pk_b,
                       src_lensed_snr=snr(im["full"], sig)[0])
    images[short] = dict(hard=im["hard"], taper=im["taper"], full=im["full"], band=im["band"], sig=sig,
                         src_hard=hard, src_taper=taper, src_full=full, s=s, lens=lens_only)
    R_ = rows[short]
    print(f"{short} thE={R_['theta_E']:.2f} recon_err={rel:.1e} | hard: removed {100*R_['f_removed_hard']:.1f}% "
          f"S/N={sn_h:.1f} peak={pk_h:.2f}σ | taper: removed {100*R_['f_removed_taper']:.1f}% S/N={sn_t:.1f} "
          f"peak={pk_t:.2f}σ | frame band {BAND}\": {100*R_['f_band']:.2f}% S/N={sn_b:.1f} peak={pk_b:.2f}σ | src S/N={R_['src_lensed_snr']:.0f}")

json.dump(dict(R0=R0, R1=R1, band=BAND, crop=crop, rows=rows), open(os.path.join(OUT, "boundary_treatments.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, "boundary_treatments_images.npz"),
                    **{f"{k}__{kk}": v for k, d in images.items() for kk, v in d.items() if isinstance(v, np.ndarray)})
print("saved")
