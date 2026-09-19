import numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
R='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
sids=['02','22','25']
fig,axs=plt.subplots(len(sids),4,figsize=(17,4.2*len(sids)))
for k,sid in enumerate(sids):
    meta=json.load(open(f'{R}/vela{sid}_cam12_a0.400_f140w/metadata.json'))
    img=np.load(f'{R}/vela{sid}_cam12_a0.400_f140w/source_image.npy')
    s=meta['source_pixel_scale_arcsec']
    n=img.shape[0]; c=n//2
    # centroid
    yy,xx=np.indices(img.shape); t=img.sum()
    cy=int((yy*img).sum()/t); cx=int((xx*img).sum()/t)
    w=int(round(0.6/s))  # 1.2" box
    sub=img[cy-w:cy+w, cx-w:cx+w]
    ax=axs[k,0]; ax.imshow(np.arcsinh(sub/1e-3), origin='lower', cmap='magma'); ax.set_title(f'vela{sid} central 1.2" (asinh)')
    # zoom 0.15" on a faint outskirt region ~1.5" away
    o=int(round(1.2/s))
    sub2=img[cy+o-40:cy+o+40, cx-40:cx+40]
    ax=axs[k,1]; im=ax.imshow(sub2, origin='lower', cmap='magma'); plt.colorbar(im,ax=ax)
    ax.set_title(f'outskirt 1.2" away, 80px=0.58" (linear nJy)')
    # histogram of pixel values in an annulus 1.0-1.5"
    r=np.hypot(yy-cy,xx-cx)*s
    ann=img[(r>1.0)&(r<1.5)]
    ax=axs[k,2]
    ax.hist(np.log10(np.clip(ann,1e-10,None)), bins=120)
    ax.set_yscale('log'); ax.set_title(f'log10 SB, annulus 1.0-1.5": zero-frac={float((ann<=0).mean()):.3f}\nmean={ann.mean():.3e} med={np.median(ann):.3e} max={ann.max():.3e}')
    # radial profile: mean vs median (MC spikes inflate the mean)
    bins=np.arange(0,3.0,0.05); idx=np.digitize(r.ravel(),bins)
    v=img.ravel()
    mn=np.array([v[idx==i].mean() if (idx==i).sum()>10 else np.nan for i in range(1,len(bins))])
    md=np.array([np.median(v[idx==i]) if (idx==i).sum()>10 else np.nan for i in range(1,len(bins))])
    p99=np.array([np.percentile(v[idx==i],99) if (idx==i).sum()>10 else np.nan for i in range(1,len(bins))])
    ax=axs[k,3]; b=0.5*(bins[1:]+bins[:-1])
    ax.semilogy(b,mn,label='mean'); ax.semilogy(b,md,label='median'); ax.semilogy(b,p99,label='99th pct')
    # 1 sigma_bkg per OUTPUT pixel converted to source SB (nJy per source px), amp=1
    photfnu=meta['photfnu_Jy']
    # SB[cps/arcsec2] = nJy/s^2*1e-9/photfnu ; 1 sigma per 0.065 px => SB = 0.0076/0.065^2
    sb1 = 0.0076/0.065**2
    njy1 = sb1*(s**2)*photfnu/1e-9
    ax.axhline(njy1, color='r', ls='--', label='1 sigma_bkg/output px (amp=1)')
    ax.legend(fontsize=7); ax.set_xlabel('r [arcsec]'); ax.set_title(f'vela{sid} radial profile (nJy/src px)')
plt.tight_layout(); plt.savefig('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/fig_mcnoise.png',dpi=110)
print('saved')
# quantitative: spike statistics in the outskirts
for sid in ['02','03','04','08','09','21','22','23','25','26']:
    meta=json.load(open(f'{R}/vela{sid}_cam12_a0.400_f140w/metadata.json'))
    img=np.load(f'{R}/vela{sid}_cam12_a0.400_f140w/source_image.npy'); s=meta['source_pixel_scale_arcsec']
    yy,xx=np.indices(img.shape); t=img.sum(); cy=(yy*img).sum()/t; cx=(xx*img).sum()/t
    r=np.hypot(yy-cy,xx-cx)*s
    for lo,hi in [(0.2,0.4),(1.0,1.5)]:
        m=(r>lo)&(r<hi); v=img[m]
        sm=gaussian_filter(img,2)[m]
        print(f"vela{sid} r={lo}-{hi}: mean={v.mean():.3e} med={np.median(v):.3e} mean/med={v.mean()/max(np.median(v),1e-12):8.2f} "
              f"frac_zero={float((v<=0).mean()):.3f} p99/med={np.percentile(v,99)/max(np.median(v),1e-12):8.2f} "
              f"std(res_g2)/mean={np.std(v-sm)/v.mean():6.2f}")
