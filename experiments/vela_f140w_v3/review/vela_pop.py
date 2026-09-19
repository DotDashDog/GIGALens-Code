import numpy as np
from astropy.io import fits
for sim in ["25", "02", "22"]:
    f = f"/pscratch/sd/l/linusu/gigalens/vela_downloads/cam12/hst/wfc3/f140w/hlsp_vela_hst_wfc3_vela{sim}-cam12-a0.400_f140w_v3_sim.fits"
    with fits.open(f) as h:
        img = np.asarray(h["IMAGE_PRISTINE"].data, dtype=np.float64); sbf = h["IMAGE_PRISTINE"].header["SBFACTOR"]
    raw = img / sbf; nz = raw[raw > 0]
    hist, edges = np.histogram(np.log10(nz), bins=60)
    lo = int(np.argmax(hist[:30])); hi = 30 + int(np.argmax(hist[30:]))
    gap_i = lo + int(np.argmin(hist[lo:hi])); gap = edges[gap_i]
    faint = raw[(raw > 0) & (np.log10(raw) < gap)]
    print(f"vela{sim}: modes at 10^{edges[lo]:.1f} and 10^{edges[hi]:.1f} nJy(raw), gap at 10^{gap:.1f} "
          f"(min count {hist[gap_i]}); faint population = {faint.size} px = {faint.size/raw.size:.1%} of the map, "
          f"holding {faint.sum()/raw.sum():.2e} of the flux; zeros {np.mean(raw==0):.1%}")
