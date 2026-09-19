import numpy as np
from astropy.io import fits
for sim in ["25", "02"]:
    f = f"/pscratch/sd/l/linusu/gigalens/vela_downloads/cam12/hst/wfc3/f140w/hlsp_vela_hst_wfc3_vela{sim}-cam12-a0.400_f140w_v3_sim.fits"
    with fits.open(f) as h:
        p = h[0].header
        if sim == "25":
            print("REFERENC:", p.get("REFERENC")); print("HLSPNAME:", p.get("HLSPNAME"), "| HLSPVER:", p.get("HLSPVER"), "| REF:", p.get("REF"))
            for c in p["HISTORY"]: print("HISTORY:", c)
        img = np.asarray(h["IMAGE_PRISTINE"].data, dtype=np.float64); sbf = h["IMAGE_PRISTINE"].header["SBFACTOR"]
        dm = p["DISTMOD"]; tot = h["IMAGE_PRISTINE"].header["FLUX_NJY"]
    orig = img / sbf                       # undo the SB scaling to get the raw Sunrise output
    nz = orig[orig > 0]
    print(f"\nvela{sim}: total {tot:.0f} nJy; pixels {img.size}, exactly zero {np.mean(img==0):.3f}")
    print(f"  raw non-zero values: min {nz.min():.4g}, 0.1% {np.percentile(nz,0.1):.4g}, 1% {np.percentile(nz,1):.4g}, 10% {np.percentile(nz,10):.4g}, median {np.median(nz):.4g}")
    # is there a floor / quantum? fraction of non-zero pixels within a factor 2 of the minimum, and histogram of log values
    print(f"  fraction of non-zero px within 2x of min: {np.mean(nz < 2*nz.min()):.4f}; within 10x: {np.mean(nz < 10*nz.min()):.4f}")
    hist, edges = np.histogram(np.log10(nz), bins=40)
    print("  log10 histogram (counts per 0.1-dex-ish bin):", " ".join(f"{e:.1f}:{c}" for e, c in zip(edges[:-1], hist) if c))
    # outer region (r > 2") only
    yy, xx = np.mgrid[:800, :800]; r = np.hypot(yy-399.5, xx-399.5) * 0.007271
    o = orig[r > 2.0]; onz = o[o > 0]
    print(f"  r>2\": zero {np.mean(o==0):.3f}; non-zero min {onz.min():.4g} median {np.median(onz):.4g} max {onz.max():.4g}; N non-zero {onz.size}")
    # isolated spikes: pixels > 20x the max of their 8 neighbours
    from scipy.ndimage import maximum_filter
    nb = maximum_filter(orig, size=3, mode="constant"); 
    # neighbour max excluding self:
    import itertools
    nmax = np.zeros_like(orig)
    for dy, dx in itertools.product([-1,0,1],[-1,0,1]):
        if dy==0 and dx==0: continue
        nmax = np.maximum(nmax, np.roll(np.roll(orig, dy, 0), dx, 1))
    spikes = (orig > 20*nmax) & (orig > 0)
    sv = orig[spikes]
    print(f"  isolated spikes (>20x all 8 neighbours): {spikes.sum()} px, holding {sv.sum()/orig.sum():.3%} of flux; values min/median/max {sv.min():.3g}/{np.median(sv):.3g}/{sv.max():.3g}")
    # luminosity of the brightest spike
    top = sv.max() * sbf
    mab = -2.5*np.log10(top*1e-9/3631); Mab = mab - dm
    print(f"  brightest spike {top:.2f} nJy -> m_AB {mab:.2f}, M_AB {Mab:.2f} (rest ~5550 A, no K-corr) -> L ~ {10**(-0.4*(Mab-4.81)):.2e} Lsun(V)")
