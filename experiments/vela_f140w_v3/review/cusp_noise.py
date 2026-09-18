"""Is the cell-scale scatter in the raw VELA map noise or structure, as a function of brightness?
High-pass = raw / (3x3 mean of the 8 neighbours) - 1. Noise -> scatter ~ 1/sqrt(packets) and
lag-1 autocorrelation ~ 0 (white); structure -> correlated neighbours. Also: the flux quantum
(smallest non-zero value) gives the expected packet noise 1/sqrt(value/quantum)."""
import numpy as np, sys
from scipy.ndimage import uniform_filter
R = "/pscratch/sd/l/linusu/gigalens/vela_sources_pristine/vela%s_cam12_a0.400_f140w/source_image.npy"
for sim in sys.argv[1:]:
    img = np.load(R % sim).astype(np.float64)
    nz = img[img > 0]; q = np.percentile(nz, 0.5)  # a low quantile of the non-zero values ~ single-packet level
    nb = (uniform_filter(img, 3) * 9 - img) / 8.0   # mean of the 8 neighbours
    ok = nb > 0
    hp = np.where(ok, img / np.where(ok, nb, 1) - 1, 0)
    peak = img.max()
    print(f"vela{sim}: peak {peak:.3g}, quantum ~ {q:.2g} (0.5th pct of non-zero), peak/quantum = {peak/q:.2g} packets -> expected MC scatter at the peak {1/np.sqrt(peak/q):.1%}")
    edges = [1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0]
    print(f"  {'value/peak bin':>18}{'cells':>8}{'frac scatter':>14}{'lag-1 corr':>12}{'expected MC':>13}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (nb >= lo * peak) & (nb < hi * peak)
        if m.sum() < 50: continue
        v = hp[m]; sc = v.std()
        # lag-1 autocorrelation of the high-pass field within the bin (x direction)
        mx = m[:, :-1] & m[:, 1:]
        a, b = hp[:, :-1][mx], hp[:, 1:][mx]
        r = np.corrcoef(a, b)[0, 1] if mx.sum() > 50 else np.nan
        exp = 1 / np.sqrt(np.sqrt(lo * hi) * peak / q)
        print(f"  {lo:>8.0e}-{hi:<8.0e}{m.sum():>8d}{sc:>14.2f}{r:>12.2f}{exp:>13.1%}")
    # the nucleus: radial profile of the brightest pixel's surroundings, raw vs 2-cell smoothed
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(img, 2.0); y, x = np.unravel_index(np.argmax(img), img.shape)
    print("  nucleus profile (cells -4..4 through the peak), raw/peak:", np.round(img[y, x-4:x+5] / peak, 3))
    print("                                          smoothed/peak:", np.round(sm[y, x-4:x+5] / peak, 3))
