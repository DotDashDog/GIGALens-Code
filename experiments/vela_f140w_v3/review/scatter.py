from astropy.io import fits
import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
P='/pscratch/sd/l/linusu/gigalens/vela_downloads/cam12/hst/wfc3/f140w/hlsp_vela_hst_wfc3_vela%s-cam12-a0.400_f140w_v3_sim.fits'
fig,axs=plt.subplots(3,4,figsize=(17,12))
for k,sid in enumerate(['02','22','25']):
    with fits.open(P%sid) as h:
        names=[x.header.get('EXTNAME') for x in h]
        if k==0: print(names); print(repr(h[2].header))
        a=np.asarray(h[names.index('IMAGE_PRISTINE')].data,float)
        b=np.asarray(h[names.index('IMAGE_PRISTINE_NONSCATTER')].data,float)
    yy,xx=np.indices(a.shape); t=a.sum(); cy=int((yy*a).sum()/t); cx=int((xx*a).sum()/t)
    s=0.007271301218058634 if sid not in ('21','26','07') else 0.01454260243611725
    r=np.hypot(yy-cy,xx-cx)*s
    print(f'vela{sid}: sum P={a.sum():.4g} NS={b.sum():.4g} ratio={a.sum()/b.sum():.4f}')
    for lo,hi in [(0.2,0.4),(1.0,1.5)]:
        m=(r>lo)&(r<hi)
        for nm,im in (('PRIST',a),('NOSCAT',b)):
            v=im[m]; sm=gaussian_filter(im,2)[m]
            print(f'   {nm} r={lo}-{hi}: mean={v.mean():.3e} med={np.median(v):.3e} zero={float((v<=0).mean()):.3f} '
                  f'std(res2px)/mean={np.std(v-sm)/v.mean():7.3f} p99/med={np.percentile(v,99)/max(np.median(v),1e-12):8.1f}')
    w=int(round(0.6/s))
    axs[k,0].imshow(np.arcsinh(a[cy-w:cy+w,cx-w:cx+w]/1e-3),origin='lower',cmap='magma'); axs[k,0].set_title(f'vela{sid} PRISTINE central 1.2"')
    axs[k,1].imshow(np.arcsinh(b[cy-w:cy+w,cx-w:cx+w]/1e-3),origin='lower',cmap='magma'); axs[k,1].set_title('NONSCATTER central 1.2"')
    o=int(round(1.2/s))
    axs[k,2].imshow(a[cy+o-40:cy+o+40,cx-40:cx+40],origin='lower',cmap='magma'); axs[k,2].set_title('PRISTINE outskirt (lin)')
    axs[k,3].imshow(b[cy+o-40:cy+o+40,cx-40:cx+40],origin='lower',cmap='magma'); axs[k,3].set_title('NONSCATTER outskirt (lin)')
plt.tight_layout(); plt.savefig('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/fig_scatter.png',dpi=105)
