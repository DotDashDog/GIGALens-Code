"""Gallery + diagnostic table for a `vela_simulated` (v2) dataset.

Display standard for these systems: ``inferno`` colormap with a square-root
stretch floored at 0 (``PowerNorm(gamma=0.5, vmin=0)``), for every panel.

Two figures:

* full gallery (one row per system): observed image, noiseless lensed source
  only (shows where the arcs sit relative to the cutout edge), the lensed source
  only with an independent realisation of the same noise model (so edge / crop
  artefacts can be judged against their SNR), and the true (unlensed, preprocessed)
  source at its calibrated amplitude, with the calibration / cut-off numbers
  from generation.json in the titles;
* compact grid of the observed images only (for slides / the design page).

Usage (numpy/json/matplotlib + the generator's preprocess_source; no JAX)::

    python experiments/vela_plot_dataset.py DATASET_DIR OUT_DIR

Works for any vela_simulated dataset (any filter / pixel scale / calibration
mode). Writes OUT_DIR/dataset_gallery.png and OUT_DIR/dataset_grid.png and
prints a per-system table.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

CMAP = "inferno"


def sqrt_norm(img):
    """Square-root stretch floored at 0: the display standard for these systems."""
    return PowerNorm(gamma=0.5, vmin=0.0, vmax=float(np.max(img)))


def show(ax, img, extent, label=None):
    im = ax.imshow(np.clip(img, 0.0, None), origin="lower", extent=extent, cmap=CMAP,
                   norm=sqrt_norm(img))
    return im


def load_source_sb(source_dir, amp, pre):
    """Source SB (cps/arcsec^2) exactly as the generator lensed it: same unit
    conversion, then the manifest's crop / recenter preprocessing, then x amp."""
    from gigalens_research.simtests.experiments.vela_simulated import preprocess_source
    m = json.load(open(os.path.join(source_dir, "metadata.json")))
    img = np.load(os.path.join(source_dir, "source_image.npy"))
    s = float(m["source_pixel_scale_arcsec"])
    sb = np.asarray(img / (s ** 2) * 1e-9 / float(m["photfnu_Jy"]))
    sb, _info = preprocess_source(sb, s, crop_radius_arcsec=pre.get("crop_radius_arcsec"),
                                  crop_taper_arcsec=pre.get("crop_taper_arcsec"),
                                  recenter=bool(pre.get("recenter", False)))
    return sb * amp, s


def crop_source(sb, sscale, win=1.6):
    ny, nx = sb.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    tot = sb.sum()
    cy = (yy * sb).sum() / tot
    cx = (xx * sb).sum() / tot
    wpx = int(win / sscale)
    y0, y1 = max(0, int(cy) - wpx), min(ny, int(cy) + wpx)
    x0, x1 = max(0, int(cx) - wpx), min(nx, int(cx) + wpx)
    return sb[y0:y1, x0:x1], win


def add_noise(img, background_rms, exp_time, seed):
    """Same model as simtests.generate._add_noise (Poisson var I/exp_time with gain
    1 e-/count, plus Gaussian background_rms), independent realisation."""
    rng = np.random.default_rng(seed)
    var = np.clip(img, 0.0, None) / exp_time + background_rms ** 2
    return img + rng.normal(0.0, np.sqrt(var))


def load_dataset(dataset_dir):
    man = json.load(open(os.path.join(dataset_dir, "manifest.json")))
    systems = []
    for sid in man["system_ids"]:
        sd = os.path.join(dataset_dir, "systems", sid)
        meta = json.load(open(os.path.join(sd, "meta.json")))
        gen = json.load(open(os.path.join(sd, "generation.json")))
        noiseless = np.load(os.path.join(sd, "noiseless_image.npy"))
        lens_only = np.load(os.path.join(sd, "lens_light_only.npy"))
        src_only = noiseless - lens_only
        systems.append(dict(
            sid=sid, meta=meta, m=gen["metrics"],
            img=np.load(os.path.join(sd, "observed_image.npy")),
            src_only=src_only,
            src_noisy=add_noise(src_only, float(meta["background_rms"]), float(meta["exp_time"]),
                                seed=abs(hash(sid)) % (2 ** 32)),
            fov=meta["num_pix"] * meta["delta_pix"],
        ))
    return man, systems


def suptitle(man):
    ex = man["extra"]
    psf = ex.get("psf", {})
    noise = ex.get("noise", {})
    cal = ex.get("calibration", {})
    mode = cal.get("mode", "ratio")
    if mode == "ratio":
        cal_txt = f"calib ratio {cal.get('source_to_lens_flux_ratio')}"
    elif mode == "peak_sb":
        d = cal["peak_sb"]["sb_mag_arcsec2"]
        cal_txt = (f"calib peak SB {d.get('value', d.get('loc', d.get('median')))} mag/arcsec2 "
                   f"({d['dist']}) over {cal['peak_sb']['n_brightest_pix']} px")
    elif mode == "unlensed_ab_mag":
        d = cal["unlensed_ab_mag"]
        cal_txt = f"calib unlensed AB {d.get('value', d.get('loc', d.get('median')))} ({d['dist']})"
    else:
        cal_txt = f"calib {mode}"
    return (f"{man.get('generator')} v{ex.get('generator_version')} | {ex.get('filter', '?').upper()} "
            f"{ex.get('delta_pix', float('nan')):.3f}\"/px {ex.get('num_pix')}px | z=1.5 ({ex['scale_factor']}) | "
            f"PSF {psf.get('kind')} FWHM {psf.get('fwhm_arcsec_measured', float('nan')):.3f}\" | "
            f"noise {noise.get('kind')} bkg_rms={noise.get('background_rms', float('nan')):.4f} "
            f"exp={noise.get('exp_time')}s | {cal_txt} | "
            f"crop {ex.get('source_preprocessing', {}).get('crop_radius_arcsec')}\" | "
            f"inferno, sqrt stretch floored at 0")


def make_gallery(man, systems, out_png):
    n = len(systems)
    fig, axs = plt.subplots(n, 4, figsize=(19, 4.4 * n),
                            gridspec_kw={"width_ratios": [4, 4, 4, 3]}, squeeze=False)
    for i, s in enumerate(systems):
        m = s["m"]
        ext = [-s["fov"] / 2, s["fov"] / 2, -s["fov"] / 2, s["fov"] / 2]

        ax = axs[i, 0]
        im = show(ax, s["img"], ext)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / pixel", fontsize=7)
        ax.set_title(f"{s['sid']}\nobserved  theta_E={m['theta_E']:.2f}\"  "
                     f"lens AB={m.get('lens_ab_mag_cutout', float('nan')):.2f}", fontsize=9)

        ax = axs[i, 1]
        im = show(ax, s["src_only"], ext)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / pixel", fontsize=7)
        ax.set_title(f"lensed source only (noiseless)  AB={m.get('source_ab_mag_lensed_cutout', float('nan')):.2f}  "
                     f"peak SB={m.get('peak_sb_mag_arcsec2', float('nan')):.2f}/{m.get('peak_sb_n_pix', '?')}px\n"
                     f"src/lens={m['source_to_lens_ratio_cutout']:.2f}  mu={m['magnification_cutout']:.1f}  "
                     f"outside={100 * m.get('flux_outside_frac', float('nan')):.2f}%  "
                     f"border={m['border_sb_sigma']:.2f}sig  redraws={m['n_redraws']}", fontsize=8)

        ax = axs[i, 2]
        im = show(ax, s["src_noisy"], ext)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / pixel", fontsize=7)
        ax.set_title(f"lensed source only + noise (independent realisation)\n"
                     f"sigma_bkg={s['meta']['background_rms']:.4f} cps/px  exp={s['meta']['exp_time']:g} s", fontsize=8)

        ax = axs[i, 3]
        sb, sscale = load_source_sb(s["meta"]["truth_assets"]["vela_source_dir"], float(m["amp"]),
                                    man["extra"].get("source_preprocessing", {}))
        crop_r = man["extra"].get("source_preprocessing", {}).get("crop_radius_arcsec")
        crop, win = crop_source(sb, sscale, win=(crop_r + 0.1) if crop_r else 2.9)  # no crop: show the 5.8" half-frame
        im = show(ax, crop, [-win, win, -win, win])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / arcsec$^2$", fontsize=6)
        ax.set_title(f"true source (as lensed, recentered) x amp={m['amp']:.2f}\n"
                     f"unlensed AB={m['source_ab_mag_unlensed']:.2f}", fontsize=8)
        for a in axs[i]:
            a.set_xlabel("arcsec", fontsize=7)
            a.tick_params(labelsize=6)
    parts = suptitle(man).split(" | ")
    fig.suptitle(" | ".join(parts[:5]) + "\n" + " | ".join(parts[5:]), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print("wrote", out_png)


def make_grid(man, systems, out_png, ncol=4):
    n = len(systems)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.15 * nrow), squeeze=False)
    for k, ax in enumerate(axs.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        s = systems[k]
        m = s["m"]
        ext = [-s["fov"] / 2, s["fov"] / 2, -s["fov"] / 2, s["fov"] / 2]
        show(ax, s["img"], ext)
        short = s["sid"].split("_")[0]
        ax.set_title(f"{short}  θE={m['theta_E']:.2f}\"  src/lens={m['source_to_lens_ratio_cutout']:.2f}\n"
                     f"src AB={m['source_ab_mag_unlensed']:.1f}  μ={m['magnification_cutout']:.1f}  "
                     f"redraws={m['n_redraws']}", fontsize=8)
        t = float(np.floor(s["fov"] / 2))
        ax.set_xticks([-t, 0, t])
        ax.set_yticks([-t, 0, t])
        ax.tick_params(labelsize=7)
    parts = suptitle(man).split(" | ")
    fig.suptitle("observed images | " + " | ".join(parts[1:5]) + "\n" + " | ".join(parts[5:]), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print("wrote", out_png)


def print_table(systems):
    print(f"{'system':<28}{'thE':>6}{'amp':>7}{'ratio':>7}{'mu':>6}{'outside%':>10}{'border':>8}"
          f"{'redraw':>7}{'srcAB':>7}{'arcAB':>7}{'lensAB':>7}{'peakSB':>8}")
    for s in systems:
        m = s["m"]
        print(f"{s['sid']:<28}{m['theta_E']:>6.2f}{m['amp']:>7.2f}{m['source_to_lens_ratio_cutout']:>7.3f}"
              f"{m['magnification_cutout']:>6.1f}{100 * m.get('flux_outside_frac', float('nan')):>10.2f}"
              f"{m['border_sb_sigma']:>8.2f}{m['n_redraws']:>7d}{m['source_ab_mag_unlensed']:>7.2f}"
              f"{m.get('source_ab_mag_lensed_cutout', float('nan')):>7.2f}"
              f"{m.get('lens_ab_mag_cutout', float('nan')):>7.2f}{m.get('peak_sb_mag_arcsec2', float('nan')):>8.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: vela_plot_dataset.py DATASET_DIR OUT_DIR")
    ds, out_dir = sys.argv[1], sys.argv[2]
    man, systems = load_dataset(ds)
    make_gallery(man, systems, os.path.join(out_dir, "dataset_gallery.png"))
    make_grid(man, systems, os.path.join(out_dir, "dataset_grid.png"))
    print_table(systems)
