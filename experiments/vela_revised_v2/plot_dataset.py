"""Gallery + diagnostic table for a `vela_simulated` (v2) dataset.

Display standard for these systems: ``inferno`` colormap with a square-root
stretch floored at 0 (``PowerNorm(gamma=0.5, vmin=0)``), for every panel.

Two figures:

* full gallery (one row per system): observed image, noiseless lensed source
  only (shows where the arcs sit relative to the cutout edge), and the true
  (unlensed, cropped) source at its calibrated amplitude, with the calibration /
  cut-off numbers from generation.json in the titles;
* compact grid of the observed images only (for slides / the design page).

Usage (pure numpy/json/matplotlib; no JAX)::

    python experiments/vela_revised_v2/plot_dataset.py [DATASET_DIR] [OUT_DIR]

DATASET_DIR defaults to the campaign's resolved output (.../vela_revised_v2/dataset);
OUT_DIR defaults to this experiment directory. Writes dataset_gallery.png and
dataset_grid.png.
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


def _default_dataset_dir():
    from gigalens_research.paths import resolve_out_dir
    return os.path.join(resolve_out_dir("simtests_results/vela_revised_v2"), "dataset")


def load_source_sb(source_dir, amp, pre):
    """Source SB (cps/arcsec^2) exactly as the generator lensed it: same unit
    conversion, then the manifest's crop / recenter preprocessing, then x amp."""
    from gigalens_research.simtests.experiments.vela_simulated import preprocess_source
    m = json.load(open(os.path.join(source_dir, "metadata.json")))
    img = np.load(os.path.join(source_dir, "source_image.npy"))
    s = float(m["source_pixel_scale_arcsec"])
    sb = np.asarray(img / (s ** 2) * 1e-9 / float(m["photfnu_Jy"]))
    sb, _info = preprocess_source(sb, s, crop_radius_arcsec=pre.get("crop_radius_arcsec"),
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


def load_dataset(dataset_dir):
    man = json.load(open(os.path.join(dataset_dir, "manifest.json")))
    systems = []
    for sid in man["system_ids"]:
        sd = os.path.join(dataset_dir, "systems", sid)
        meta = json.load(open(os.path.join(sd, "meta.json")))
        gen = json.load(open(os.path.join(sd, "generation.json")))
        noiseless = np.load(os.path.join(sd, "noiseless_image.npy"))
        lens_only = np.load(os.path.join(sd, "lens_light_only.npy"))
        systems.append(dict(
            sid=sid, meta=meta, m=gen["metrics"],
            img=np.load(os.path.join(sd, "observed_image.npy")),
            src_only=noiseless - lens_only,
            fov=meta["num_pix"] * meta["delta_pix"],
        ))
    return man, systems


def suptitle(man):
    ex = man["extra"]
    psf = ex.get("psf", {})
    noise = ex.get("noise", {})
    return (f"{man.get('generator')} v{ex.get('generator_version')} | z=1.5 ({ex['scale_factor']}) | "
            f"PSF {psf.get('kind')} FWHM {psf.get('fwhm_arcsec_measured', float('nan')):.3f}\" | "
            f"noise {noise.get('kind')} bkg_rms={noise.get('background_rms', float('nan')):.4f} "
            f"exp={noise.get('exp_time')}s | ratio target "
            f"{ex.get('calibration', {}).get('source_to_lens_flux_ratio')} | "
            f"crop {ex.get('source_preprocessing', {}).get('crop_radius_arcsec')}\" | "
            f"inferno, sqrt stretch floored at 0")


def make_gallery(man, systems, out_png):
    n = len(systems)
    fig, axs = plt.subplots(n, 3, figsize=(15, 4.4 * n),
                            gridspec_kw={"width_ratios": [4, 4, 3]}, squeeze=False)
    for i, s in enumerate(systems):
        m = s["m"]
        ext = [-s["fov"] / 2, s["fov"] / 2, -s["fov"] / 2, s["fov"] / 2]

        ax = axs[i, 0]
        im = show(ax, s["img"], ext)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / pixel", fontsize=7)
        ax.set_title(f"{s['sid']}\nobserved  theta_E={m['theta_E']:.2f}\"", fontsize=9)

        ax = axs[i, 1]
        im = show(ax, s["src_only"], ext)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / pixel", fontsize=7)
        ax.set_title(f"lensed source only (noiseless)\n"
                     f"src/lens={m['source_to_lens_ratio_cutout']:.2f}  mu={m['magnification_cutout']:.1f}  "
                     f"outside={100 * m.get('flux_outside_frac', float('nan')):.2f}%  "
                     f"border={m['border_sb_sigma']:.2f}sig  redraws={m['n_redraws']}", fontsize=8)

        ax = axs[i, 2]
        sb, sscale = load_source_sb(s["meta"]["truth_assets"]["vela_source_dir"], float(m["amp"]),
                                    man["extra"].get("source_preprocessing", {}))
        crop, win = crop_source(sb, sscale)
        im = show(ax, crop, [-win, win, -win, win])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / arcsec$^2$", fontsize=6)
        ax.set_title(f"true source (as lensed: cropped, recentered) x amp={m['amp']:.2f}\n"
                     f"unlensed AB={m['source_ab_mag_unlensed']:.2f}", fontsize=8)
        for a in axs[i]:
            a.set_xlabel("arcsec", fontsize=7)
            a.tick_params(labelsize=6)
    fig.suptitle(suptitle(man), fontsize=11)
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
        ax.set_title(f"{short}  theta_E={m['theta_E']:.2f}\"  amp={m['amp']:.1f}  "
                     f"mu={m['magnification_cutout']:.1f}  redraws={m['n_redraws']}", fontsize=9)
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        ax.tick_params(labelsize=7)
    fig.suptitle("observed images, " + suptitle(man).split(" | ", 1)[1], fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print("wrote", out_png)


def print_table(systems):
    print(f"{'system':<28}{'thE':>6}{'amp':>7}{'ratio':>7}{'mu':>6}{'outside%':>10}{'border':>8}"
          f"{'redraw':>7}{'srcAB':>7}{'Flens':>7}")
    for s in systems:
        m = s["m"]
        print(f"{s['sid']:<28}{m['theta_E']:>6.2f}{m['amp']:>7.2f}{m['source_to_lens_ratio_cutout']:>7.3f}"
              f"{m['magnification_cutout']:>6.1f}{100 * m.get('flux_outside_frac', float('nan')):>10.2f}"
              f"{m['border_sb_sigma']:>8.2f}{m['n_redraws']:>7d}{m['source_ab_mag_unlensed']:>7.2f}"
              f"{m['lens_flux_cutout']:>7.0f}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else _default_dataset_dir()
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    man, systems = load_dataset(ds)
    make_gallery(man, systems, os.path.join(out_dir, "dataset_gallery.png"))
    make_grid(man, systems, os.path.join(out_dir, "dataset_grid.png"))
    print_table(systems)
