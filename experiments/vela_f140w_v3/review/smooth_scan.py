"""How the Sunrise MC signature decays with Gaussian smoothing of the source map."""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
S = "/pscratch/sd/l/linusu/gigalens/vela_sources_pristine"
CELL_ARCSEC = 0.007271
print("sigma [cells] | 3x3-median flux loss | frac scatter (bright 0.5%) | frac scatter (5-20% of peak) | sigma in arcsec | kpc")
for sim in ["25", "02", "22"]:
    img = np.load(f"{S}/vela{sim}_cam12_a0.400_f140w/source_image.npy").astype(np.float64)
    for s in [0, 0.5, 1, 1.5, 2, 3, 4]:
        sm = gaussian_filter(img, s) if s > 0 else img
        med = median_filter(sm, 3)
        loss = 1 - med.sum() / sm.sum()
        # fractional pixel-to-pixel scatter about a 5x5 local mean, in two SB bands
        loc = gaussian_filter(sm, 2.5)
        frac = (sm - loc) / np.maximum(loc, 1e-30)
        bright = loc >= np.percentile(loc, 99.5)
        mid = (loc > 0.05 * loc.max()) & (loc < 0.2 * loc.max())
        print(f"vela{sim} s={s:<4} {loss:8.3%} {frac[bright].std():10.3f} {frac[mid].std():10.3f}   {s*CELL_ARCSEC:.4f}\"  {s*0.0625:.3f} kpc")
    print()
print("source-plane tangential PSF sigma (0.073\"/sqrt(mu)) for mu = 3, 5, 10, 22:", " ".join(f"{0.073/np.sqrt(m):.4f}\"" for m in [3, 5, 10, 22]), "=", " ".join(f"{0.073/np.sqrt(m)/CELL_ARCSEC:.1f} cells" for m in [3, 5, 10, 22]))
