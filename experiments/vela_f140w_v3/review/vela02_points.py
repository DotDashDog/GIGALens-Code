"""vela02: are the scattered bright points resolved clumps or packet spikes, and can the
lensed data tell the raw map from the 2-cell-smoothed one?"""
import numpy as np, json
from scipy.ndimage import gaussian_filter, maximum_filter, label
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
S = "/pscratch/sd/l/linusu/gigalens/vela_sources_pristine/vela02_cam12_a0.400_f140w/source_image.npy"
img = np.load(S).astype(np.float64)
CELL = 0.007271; MU = 10.1; PSF_SIG = 0.073          # vela02 in the current set
res_cells = PSF_SIG / np.sqrt(MU) / CELL              # tangential source-plane resolution, sigma in cells
print(f"vela02: mu={MU}, data source-plane resolution sigma = {res_cells:.2f} cells = {res_cells*CELL:.4f}\"")
# --- 1. bright points in the raw map: local maxima above 5% of the global peak, within 1.5" of centre
cy, cx = 399.5, 399.5
yy, xx = np.mgrid[:800, :800]; r = np.hypot(yy - cy, xx - cx) * CELL
peak = img.max()
lm = (img == maximum_filter(img, size=5)) & (img > 0.05 * peak) & (r < 1.5)
ys, xs = np.nonzero(lm)
# neighbour statistics: max of 8 neighbours / peak, and flux in 5x5 vs central pixel
rows = []
for y, x in zip(ys, xs):
    nb = img[y-1:y+2, x-1:x+2].copy(); c = nb[1, 1]; nb[1, 1] = 0
    box5 = img[y-2:y+3, x-2:x+3].sum()
    rows.append((c, nb.max() / c, c / box5))
rows = np.array(rows)
single = rows[:, 1] < 0.2           # brightest neighbour < 20% of the peak = single-pixel spike
print(f"  local maxima > 5% of peak within 1.5\": {len(rows)}; single-pixel spikes (neighbour max < 20%): {single.sum()}; "
      f"median neighbour/peak ratio {np.median(rows[:,1]):.2f}; median fraction of 5x5 flux in the central pixel {np.median(rows[:,2]):.2f}")
print(f"  flux in those maxima's central pixels: {rows[:,0].sum()/img.sum():.2%} of the galaxy; in their 5x5 boxes: {sum(img[y-2:y+3, x-2:x+3].sum() for y,x in zip(ys,xs))/img.sum():.1%}")
# pixel-to-pixel fractional scatter inside the brightest 1% (how trustworthy is a 'point' at all?)
loc = gaussian_filter(img, 2.5); bright = loc >= np.percentile(loc, 99)
print(f"  fractional pixel scatter about a 2.5-cell local mean in the brightest 1%: {((img-loc)/loc)[bright].std():.2f}")
# --- 2. what the data can see: convolve raw and smoothed to the data's source-plane resolution
sm2 = gaussian_filter(img, 2.0); sm1 = gaussian_filter(img, 1.0)
at_data = {k: gaussian_filter(v, res_cells) for k, v in [("raw", img), ("sm1", sm1), ("sm2", sm2)]}
d = at_data["raw"] - at_data["sm2"]
ref = at_data["raw"]
print(f"  at data resolution: max |raw - sm2| / peak = {np.abs(d).max()/ref.max():.3f}; rms over pixels above 5% of peak: "
      f"{d[ref > 0.05*ref.max()].std()/ref.max():.4f} of peak; effective resolution broadening sqrt(res^2+2^2)/res = {np.sqrt(res_cells**2+4)/res_cells:.2f}")
# --- 3. figure: 1.2" window around the centre
w = int(0.6 / CELL); sl = slice(int(cy - w), int(cy + w) + 1); ext = [-0.6, 0.6, -0.6, 0.6]
panels = [("raw map", img), ("smoothed sigma=1 cell", sm1), ("smoothed sigma=2 cells", sm2),
          ("raw, at data resolution (sigma %.1f cells)" % res_cells, at_data["raw"]),
          ("sigma=2, at data resolution", at_data["sm2"]), ("difference at data resolution", d)]
fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
for ax, (t, a) in zip(axes.ravel(), panels):
    if t.startswith("difference"):
        v = np.abs(a[sl, sl]).max(); im = ax.imshow(a.T[sl, sl], cmap="RdBu_r", vmin=-v, vmax=v, origin="lower", extent=ext)
        ax.set_title(f"{t}\nmax |diff| = {v/ref.max():.3f} of the peak", fontsize=10)
    else:
        vmax = a[sl, sl].max(); im = ax.imshow(a.T[sl, sl], cmap="inferno", norm=PowerNorm(0.5, vmin=0, vmax=vmax), origin="lower", extent=ext)
        ax.set_title(t, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("arcsec")
axes[0, 0].plot(((xs - cx) * CELL)[single], ((ys - cy) * CELL)[single], "c+", ms=6, mew=0.8, label="single-pixel spikes")
axes[0, 0].plot(((xs - cx) * CELL)[~single], ((ys - cy) * CELL)[~single], "wx", ms=5, mew=0.8, label="multi-pixel maxima")
axes[0, 0].legend(fontsize=8, loc="lower left"); axes[0, 0].set_xlim(-0.6, 0.6); axes[0, 0].set_ylim(-0.6, 0.6)
fig.suptitle("vela02 source (transposed to image axes, 1.2\" window): raw vs smoothed, and what survives at the data's source-plane resolution", y=0.995)
plt.tight_layout(); plt.savefig("/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/vela-generator-v2/experiments/vela_f140w_v3/vela02_smoothing_check.png", dpi=85)
# profile through the brightest single-pixel spike
if single.any():
    i = np.argmax(rows[single, 0]); y, x = ys[single][i], xs[single][i]
    print("  brightest single-pixel spike at offset (%.3f, %.3f)\": raw profile (px -3..3):" % ((x-cx)*CELL, (y-cy)*CELL),
          np.round(img[y, x-3:x+4] / img[y, x], 3), " smoothed-2 profile:", np.round(sm2[y, x-3:x+4] / sm2[y, x], 3),
          " peak raw/sm2 = %.1f" % (img[y, x] / sm2[y, x]))
