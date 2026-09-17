import numpy as np, json
from scipy.ndimage import gaussian_filter, median_filter
R='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
for sid in ['02','22','25','21','04']:
    meta=json.load(open(f'{R}/vela{sid}_cam12_a0.400_f140w/metadata.json'))
    img=np.load(f'{R}/vela{sid}_cam12_a0.400_f140w/source_image.npy'); s=meta['source_pixel_scale_arcsec']
    yy,xx=np.indices(img.shape); t=img.sum(); cy=(yy*img).sum()/t; cx=(xx*img).sum()/t
    r=np.hypot(yy-cy,xx-cx)*s
    v=img[(r>1.5)&(r<2.5)]
    pos=np.sort(v[v>0])
    u=np.unique(pos)
    print(f'--- vela{sid}: annulus 1.5-2.5": n={v.size} nonzero={pos.size} unique={u.size} zero_frac={(v<=0).mean():.3f}')
    print('    smallest 12 positive values:', np.array2string(pos[:12], precision=3, formatter={'float':lambda x:f'{x:.4e}'}))
    if u.size>3:
        q=u[0]
        print(f'    ratios of 12 smallest UNIQUE values to the smallest: {np.array2string(u[:12]/q, precision=3)}')
    # flux carried by single-pixel spikes
    m3=median_filter(img,3)
    spike=img-m3
    print(f'    total flux={img.sum():.1f} nJy;  flux removed by median3={100*(1-m3.sum()/img.sum()):.1f}%; '
          f'flux in pixels where img>5*med3={100*img[(img>5*np.maximum(m3,1e-12))].sum()/img.sum():.1f}% '
          f'(n={(img>5*np.maximum(m3,1e-12)).sum()} px, {(100.*(img>5*np.maximum(m3,1e-12)).sum()/img.size):.2f}% of pixels)')
    # brightest local peaks: measure FWHM in source pixels
    sm=gaussian_filter(img,0)  # raw
    idx=np.argsort(img.ravel())[::-1][:2000]
    ys,xs=np.unravel_index(idx,img.shape)
    # take top 5 well-separated peaks
    peaks=[]
    for y,x in zip(ys,xs):
        if all((y-py)**2+(x-px)**2>100 for py,px in peaks): peaks.append((y,x))
        if len(peaks)>=5: break
    for (y,x) in peaks:
        cut=img[y, max(0,x-8):x+9]
        pk=img[y,x]; half=pk/2
        prof=img[y-6:y+7, x-6:x+7]
        # radial mean profile around the peak
        d=np.array([img[y,x], np.mean([img[y+1,x],img[y-1,x],img[y,x+1],img[y,x-1]]),
                    np.mean([img[y+2,x],img[y-2,x],img[y,x+2],img[y,x-2]]),
                    np.mean([img[y+3,x],img[y-3,x],img[y,x+3],img[y,x-3]]),
                    np.mean([img[y+5,x],img[y-5,x],img[y,x+5],img[y,x-5]])])
        print(f'    peak at ({y},{x}) val={pk:.3f}: profile r=0,1,2,3,5 px -> {np.array2string(d/pk, precision=3)}')
