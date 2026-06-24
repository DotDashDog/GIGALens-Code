"""Display candidate replacement Vela sources (pristine, unlensed) at z=1.5 so
the user can reject double galaxies / messy morphology before swapping one in.

Extracts each pristine source via the generator's verified-EXTNAME path (reusing
pre-staged FITS on scratch), then plots the unlensed source surface brightness.
Run inside the jax-2026 Shifter container.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import gigalens_research.simtests.experiments.vela_simulated as vs

CANDIDATES = ["02", "05", "09", "11", "13", "20"]
STAGED_DATADIR = "/pscratch/sd/l/linusu/vela_probe"
SRCROOT = os.path.expanduser("~/GIGALens-Code/data/vela_sources_pristine")
PNG = os.path.expanduser("~/GIGALens-Code/experiments/vela_revised_v1/candidate_sources.png")


def load_sb(source_dir):
    m = json.load(open(os.path.join(source_dir, "metadata.json")))
    img = np.load(os.path.join(source_dir, "source_image.npy"))
    s = float(m["source_pixel_scale_arcsec"])
    sb = img / (s ** 2) * 1e-9 / float(m["photfnu_Jy"])   # cps/arcsec^2
    return np.asarray(sb), s, m


fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()

for ax, sim in zip(axs, CANDIDATES):
    sdir = vs.ensure_pristine_source(sim, "12", "a0.400", "f814w",
                                     source_root=SRCROOT, datadir=STAGED_DATADIR,
                                     version="v3")
    sb, sscale, m = load_sb(sdir)
    # crop to a centered window around the flux centroid for a fair look
    ny, nx = sb.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    tot = sb.sum()
    cy = (yy * sb).sum() / tot; cx = (xx * sb).sum() / tot
    win = 2.0  # arcsec half-window (wider, to reveal companions/double galaxies)
    wpx = int(win / sscale)
    y0, y1 = max(0, int(cy) - wpx), min(ny, int(cy) + wpx)
    x0, x1 = max(0, int(cx) - wpx), min(nx, int(cx) + wpx)
    crop = sb[y0:y1, x0:x1]
    speak = float(sb.max())
    im = ax.imshow(np.clip(crop, speak * 1e-3, None), origin="lower",
                   extent=[-win, win, -win, win], cmap="inferno",
                   norm=LogNorm(vmin=speak * 1e-3, vmax=speak))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("cps / arcsec$^2$", fontsize=8)
    abm = m.get("pristine_ABMAG")
    abm_s = f"{abm:.2f}" if abm is not None else "n/a"
    ax.set_title(f"vela{sim}  (z=1.5, ABMAG={abm_s}, FOV shown {2*win:.0f} arcsec)",
                 fontsize=11)
    ax.set_xlabel("arcsec"); ax.set_ylabel("arcsec")

fig.suptitle("Candidate replacement sources (pristine, unlensed) at z=1.5 / a0.400, cam12, f814w "
             "— log surface brightness", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(PNG, dpi=110)
print("wrote", PNG)
