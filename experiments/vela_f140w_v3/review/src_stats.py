import json, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter

ROOT='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
IDS=["02","03","04","08","09","21","22","23","25","26"]
out={}
rows=[]
for sid in IDS:
    d=f'{ROOT}/vela{sid}_cam12_a0.400_f140w'
    meta=json.load(open(d+'/metadata.json'))
    img=np.load(d+'/source_image.npy')  # nJy per px
    s=meta['source_pixel_scale_arcsec']; skpc=meta['source_pixel_scale_kpc']
    tot=img.sum()
    # centroid & peak
    yy,xx=np.indices(img.shape)
    cy=(yy*img).sum()/tot; cx=(xx*img).sum()/tot
    # main-galaxy peak from smoothed image (smooth to 0.05" ~ 7 px)
    sm=gaussian_filter(img, 7)
    py,px=np.unravel_index(np.argmax(sm), img.shape)
    r=np.hypot(yy-cy,xx-cx)*s
    # curve of growth about centroid
    order=np.argsort(r.ravel()); fr=np.cumsum(img.ravel()[order])/tot
    rr=r.ravel()[order]
    def rad(f): return float(np.interp(f, fr, rr))
    r20,r50,r80,r90=rad(.2),rad(.5),rad(.8),rad(.9)
    # about the peak instead
    rp=np.hypot(yy-py,xx-px)*s
    op=np.argsort(rp.ravel()); frp=np.cumsum(img.ravel()[op])/tot; rrp=rp.ravel()[op]
    r50p=float(np.interp(.5,frp,rrp))
    # negative pixels / zero fraction
    negfrac=float((img<0).sum())/img.size
    # MC noise estimate: high-freq residual vs 3x3 median, in annuli by SB
    med=median_filter(img,3)
    res=img-med
    # SB bins
    prof=[]
    for lo,hi in [(1e-6,1e-5),(1e-5,1e-4),(1e-4,1e-3),(1e-3,1e-2),(1e-2,1e-1),(1e-1,1e0)]:
        msk=(med>=lo)&(med<hi)
        if msk.sum()>200:
            prof.append((np.sqrt(lo*hi), float(np.std(res[msk])), float(np.median(med[msk])), int(msk.sum())))
    # flux fractions
    f1=float(img[r<1.0].sum()/tot); f25=float(img[r<2.5].sum()/tot)
    rows.append(dict(sid=sid, kpc=skpc, arcsec=s, tot_nJy=float(tot), mag=meta['mock_AB_mag_apparent'],
        r20=r20,r50=r50,r80=r80,r90=r90, r50_kpc=r50/s*skpc, r50p=r50p,
        cen_peak_off=float(np.hypot(cy-py,cx-px)*s), negfrac=negfrac,
        f_in_1as=f1, f_in_25as=f25, mcprof=prof))
    out[sid]=dict(img_shape=img.shape)
np.save('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/src_rows.npy', np.array(rows,dtype=object))
print(f"{'sid':>4}{'kpc/px':>8}{'as/px':>9}{'mag':>7}{'R20\"':>8}{'R50\"':>8}{'R80\"':>8}{'R90\"':>8}{'R50kpc':>8}{'R50pk\"':>8}{'cen-pk\"':>9}{'f<1\"':>7}{'f<2.5\"':>8}{'neg%':>7}")
for r in rows:
    print(f"{r['sid']:>4}{r['kpc']:>8.4f}{r['arcsec']:>9.5f}{r['mag']:>7.2f}{r['r20']:>8.3f}{r['r50']:>8.3f}{r['r80']:>8.3f}{r['r90']:>8.3f}{r['r50_kpc']:>8.2f}{r['r50p']:>8.3f}{r['cen_peak_off']:>9.3f}{r['f_in_1as']:>7.3f}{r['f_in_25as']:>8.3f}{100*r['negfrac']:>7.3f}")
print()
print('MC/high-freq residual: sqrt(SBbin)  std(img-med3)  median(med3)  npix')
for r in rows:
    print('vela'+r['sid'])
    for p in r['mcprof']:
        print('   %9.2e %9.2e %9.2e %8d   ratio=%6.3f'%(p[0],p[1],p[2],p[3],p[1]/max(p[2],1e-30)))
