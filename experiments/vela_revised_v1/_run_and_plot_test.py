"""One-off exploratory plotter for the vela_revised_v1 test set: lensed image +
separate true-source panel per system, each with its own colorbar. Emits both a
log-scaled and a linear-scaled version. Prints a brightness-diagnostic table.

NOT part of the framework. Reads the already-generated dataset (generate first);
pure numpy/json/matplotlib.
"""
import os, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

OUT = os.path.expanduser("~/GIGALens-Code/simtests_results/vela_revised_v1/dataset")
PNG_LOG = os.path.expanduser("~/GIGALens-Code/experiments/vela_revised_v1/lensed_systems_test.png")
PNG_LIN = os.path.expanduser("~/GIGALens-Code/experiments/vela_revised_v1/lensed_systems_test_linear.png")

man = json.load(open(os.path.join(OUT, "manifest.json")))
ids = man["system_ids"]
flux_scale = float(man["extra"]["source_flux_scale"])


def load_source_sb(source_dir):
    m = json.load(open(os.path.join(source_dir, "metadata.json")))
    img = np.load(os.path.join(source_dir, "source_image.npy"))
    s = float(m["source_pixel_scale_arcsec"])
    sb = img / (s ** 2) * 1e-9 / float(m["photfnu_Jy"])   # cps/arcsec^2
    return np.asarray(sb) * flux_scale, s, m


def crop_source(sb, sscale, win=1.4):
    ny, nx = sb.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    tot = sb.sum()
    cy = (yy * sb).sum() / tot; cx = (xx * sb).sum() / tot
    wpx = int(win / sscale)
    y0, y1 = max(0, int(cy) - wpx), min(ny, int(cy) + wpx)
    x0, x1 = max(0, int(cx) - wpx), min(nx, int(cx) + wpx)
    return sb[y0:y1, x0:x1], win


rows, data = [], []
for sid in ids:
    sd = os.path.join(OUT, "systems", sid)
    img = np.load(os.path.join(sd, "observed_image.npy"))
    meta = json.load(open(os.path.join(sd, "meta.json")))
    truth = pickle.load(open(os.path.join(sd, "truth_x.pkl"), "rb"))
    Ie = float(np.squeeze(truth[1][0]["Ie"]))
    sb, sscale, smeta = load_source_sb(meta["truth_assets"]["vela_source_dir"])
    rows.append((sid, smeta.get("pristine_ABMAG"), sb.shape[0] * sscale, float(sb.max()),
                 Ie, float(np.percentile(img, 99.9)), float(img.max())))
    data.append((sid, img, meta, sb, sscale))

# ---- diagnostic table ----
print(f"{'system':<26}{'ABMAG':>7}{'srcFOVas':>9}{'peakSrcSB':>11}{'lensIe':>9}"
      f"{'img99.9':>9}{'imgmax':>8}")
for r in rows:
    abm = f"{r[1]:.2f}" if r[1] is not None else "  n/a"
    print(f"{r[0]:<26}{abm:>7}{r[2]:>9.2f}{r[3]:>11.3g}{r[4]:>9.2f}{r[5]:>9.3g}{r[6]:>8.3g}")


def make_fig(scale, png):
    """scale: 'log' or 'linear'."""
    n = len(ids); sys_per_row = 3; nrow = int(np.ceil(n / sys_per_row))
    fig, axs = plt.subplots(nrow, sys_per_row * 2,
                            figsize=(4.2 * sys_per_row * 1.5, 4.2 * nrow),
                            gridspec_kw={"width_ratios": [4, 2] * sys_per_row})
    for idx, (sid, img, meta, sb, sscale) in enumerate(data):
        r, c = idx // sys_per_row, idx % sys_per_row
        axL, axS = axs[r, 2 * c], axs[r, 2 * c + 1]

        fov = meta["num_pix"] * meta["delta_pix"]
        ext = [-fov / 2, fov / 2, -fov / 2, fov / 2]
        vmax = float(img.max())
        if scale == "log":
            vmin = max(meta["background_rms"], 1e-4)
            norm = LogNorm(vmin=vmin, vmax=vmax); show = np.clip(img, vmin, None)
        else:
            norm = Normalize(vmin=0.0, vmax=vmax); show = img
        im = axL.imshow(show, origin="lower", extent=ext, cmap="inferno", norm=norm)
        cb = fig.colorbar(im, ax=axL, fraction=0.046, pad=0.03)
        cb.set_label("cps / pixel", fontsize=7); cb.ax.tick_params(labelsize=6)
        axL.set_title(sid, fontsize=9); axL.set_xlabel("arcsec", fontsize=8)

        crop, win = crop_source(sb, sscale)
        sext = [-win, win, -win, win]; speak = float(sb.max())
        if scale == "log":
            snorm = LogNorm(vmin=speak * 1e-3, vmax=speak); scrop = np.clip(crop, speak * 1e-3, None)
        else:
            snorm = Normalize(vmin=0.0, vmax=speak); scrop = crop
        ims = axS.imshow(scrop, origin="lower", extent=sext, cmap="inferno", norm=snorm)
        cbs = fig.colorbar(ims, ax=axS, fraction=0.046, pad=0.03)
        cbs.set_label("src cps/arcsec$^2$", fontsize=6); cbs.ax.tick_params(labelsize=5)
        axS.set_title("true source", fontsize=8); axS.set_xlabel("arcsec", fontsize=7)
        axS.tick_params(labelsize=6)

    for idx in range(n, nrow * sys_per_row):
        r, c = idx // sys_per_row, idx % sys_per_row
        axs[r, 2 * c].axis("off"); axs[r, 2 * c + 1].axis("off")

    fig.suptitle(
        f"vela_revised_v1 TEST ({scale} scale) | z=1.5 (a0.400), flux_scale={flux_scale}, "
        f"PSF FWHM={man['extra']['psf_fwhm_arcsec']}\", bkg_rms={man['extra']['background_rms']}, "
        f"exp={man['extra']['exp_time']}s, supersample={man['extra']['supersample']} "
        f"| left: lensed cps/pixel; right: unlensed source SB",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png, dpi=110)
    print("wrote", png)


make_fig("log", PNG_LOG)
make_fig("linear", PNG_LIN)
