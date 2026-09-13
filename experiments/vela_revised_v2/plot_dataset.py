"""Gallery + diagnostic table for a `vela_simulated` (v2) dataset.

Per system: observed image (log), noiseless lensed source only (linear, shows
where the arcs sit relative to the cutout edge), and the true (unlensed, cropped)
source. Titles carry the calibration / cut-off numbers from generation.json.

Usage (pure numpy/json/matplotlib; no JAX)::

    python experiments/vela_revised_v2/plot_dataset.py [DATASET_DIR] [OUT.png]

DATASET_DIR defaults to the campaign's resolved output (.../vela_revised_v2/dataset).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


def _default_dataset_dir():
    from gigalens_research.paths import resolve_out_dir
    return os.path.join(resolve_out_dir("simtests_results/vela_revised_v2"), "dataset")


def load_source_sb(source_dir, amp):
    m = json.load(open(os.path.join(source_dir, "metadata.json")))
    img = np.load(os.path.join(source_dir, "source_image.npy"))
    s = float(m["source_pixel_scale_arcsec"])
    return np.asarray(img / (s ** 2) * 1e-9 / float(m["photfnu_Jy"])) * amp, s


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


def main(dataset_dir, out_png):
    man = json.load(open(os.path.join(dataset_dir, "manifest.json")))
    ids = man["system_ids"]
    ex = man["extra"]
    rows = []
    fig, axs = plt.subplots(len(ids), 3, figsize=(15, 4.4 * len(ids)),
                            gridspec_kw={"width_ratios": [4, 4, 3]}, squeeze=False)
    for i, sid in enumerate(ids):
        sd = os.path.join(dataset_dir, "systems", sid)
        img = np.load(os.path.join(sd, "observed_image.npy"))
        meta = json.load(open(os.path.join(sd, "meta.json")))
        gen = json.load(open(os.path.join(sd, "generation.json")))
        m = gen["metrics"]
        noiseless = np.load(os.path.join(sd, "noiseless_image.npy"))
        lens_only = np.load(os.path.join(sd, "lens_light_only.npy"))
        src_only = noiseless - lens_only
        bkg = float(meta["background_rms"])
        fov = meta["num_pix"] * meta["delta_pix"]
        ext = [-fov / 2, fov / 2, -fov / 2, fov / 2]

        ax = axs[i, 0]
        vmin = max(bkg, 1e-4)
        im = ax.imshow(np.clip(img, vmin, None), origin="lower", extent=ext, cmap="inferno",
                       norm=LogNorm(vmin=vmin, vmax=float(img.max())))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / pixel", fontsize=7)
        ax.set_title(f"{sid}\nobserved (log)  theta_E={m.get('theta_E', float('nan')):.2f}" if "theta_E" in m
                     else f"{sid}\nobserved (log)", fontsize=9)

        ax = axs[i, 1]
        im = ax.imshow(src_only, origin="lower", extent=ext, cmap="inferno",
                       norm=Normalize(vmin=0.0, vmax=float(src_only.max())))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / pixel", fontsize=7)
        ax.set_title(f"lensed source only (noiseless, linear)\n"
                     f"src/lens={m['source_to_lens_ratio_cutout']:.2f}  mu={m['magnification_cutout']:.1f}  "
                     f"outside={100 * m.get('flux_outside_frac', float('nan')):.2f}%  "
                     f"border={m['border_sb_sigma']:.2f}sig  redraws={m['n_redraws']}", fontsize=8)

        ax = axs[i, 2]
        sb, sscale = load_source_sb(meta["truth_assets"]["vela_source_dir"], float(m["amp"]))
        crop, win = crop_source(sb, sscale)
        speak = float(sb.max())
        im = ax.imshow(np.clip(crop, speak * 1e-3, None), origin="lower",
                       extent=[-win, win, -win, win], cmap="inferno",
                       norm=LogNorm(vmin=speak * 1e-3, vmax=speak))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("cps / arcsec$^2$", fontsize=6)
        ax.set_title(f"true source x amp={m['amp']:.2f}\n"
                     f"unlensed AB={m['source_ab_mag_unlensed']:.2f}", fontsize=8)
        for a in axs[i]:
            a.set_xlabel("arcsec", fontsize=7)
            a.tick_params(labelsize=6)
        rows.append((sid, m))

    psf = ex.get("psf", {})
    noise = ex.get("noise", {})
    fig.suptitle(
        f"{man.get('generator')} v{ex.get('generator_version')} | z=1.5 ({ex['scale_factor']}) | "
        f"PSF {psf.get('kind')} FWHM {psf.get('fwhm_arcsec_measured', float('nan')):.3f}\" | "
        f"noise {noise.get('kind')} bkg_rms={noise.get('background_rms', float('nan')):.4f} "
        f"exp={noise.get('exp_time')}s | ratio target "
        f"{ex.get('calibration', {}).get('source_to_lens_flux_ratio')} | "
        f"crop {ex.get('source_preprocessing', {}).get('crop_radius_arcsec')}\"",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=100)
    print("wrote", out_png)

    print(f"{'system':<28}{'amp':>7}{'ratio':>7}{'mu':>6}{'outside%':>10}{'border':>8}{'redraw':>7}"
          f"{'srcAB':>7}{'Flens':>7}")
    for sid, m in rows:
        print(f"{sid:<28}{m['amp']:>7.2f}{m['source_to_lens_ratio_cutout']:>7.3f}"
              f"{m['magnification_cutout']:>6.1f}{100 * m.get('flux_outside_frac', float('nan')):>10.2f}"
              f"{m['border_sb_sigma']:>8.2f}{m['n_redraws']:>7d}{m['source_ab_mag_unlensed']:>7.2f}"
              f"{m['lens_flux_cutout']:>7.0f}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else _default_dataset_dir()
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset_gallery.png")
    main(ds, out)
