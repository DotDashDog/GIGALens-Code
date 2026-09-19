import numpy as np, json, os
D='/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems'
IDS=["02","03","04","08","09","21","22","23","25","26"]
bkg=0.0076; exp=1197.7; dpix=0.065
man=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))['extra']['per_system']
psf=np.load(D+'/vela02_cam12_a0.400_rep00/psf.npy')
# PSF area in pixels (1/sum(p^2))
npsf=1.0/ (psf**2).sum()
print('PSF effective area (pix) =', npsf, ' sum', psf.sum())
print(f"{'sys':>6}{'S/N_tot':>9}{'f>3sig':>8}{'A>3sig':>9}{'Nres':>7}{'f>10s':>8}{'src/lens':>9}{'mu':>7}{'peakSNR':>9}")
res={}
for sid in IDS:
    d=f'{D}/vela{sid}_cam12_a0.400_rep00'
    nl=np.load(d+'/noiseless_image.npy').astype(np.float64)
    ll=np.load(d+'/lens_light_only.npy').astype(np.float64)
    src=nl-ll
    sig=np.sqrt(bkg**2 + np.maximum(nl,0)/exp)   # total noise incl. lens-light Poisson
    snr=src/sig
    m3=snr>3; m10=snr>10
    tot=src.sum()
    sn_tot=tot/np.sqrt((sig**2)[src>0.0].sum())
    nres=m3.sum()/npsf
    v=man[f'vela{sid}_cam12_a0.400_rep00']
    res[sid]=dict(f3=float(src[m3].sum()/tot), a3=int(m3.sum()), nres=float(nres), sn=float(sn_tot),
                  f10=float(src[m10].sum()/tot), peak=float(snr.max()))
    print(f"vela{sid:>2}{sn_tot:>9.1f}{src[m3].sum()/tot:>8.3f}{m3.sum():>9d}{nres:>7.1f}{src[m10].sum()/tot:>8.3f}{v['source_to_lens_ratio_cutout']:>9.3f}{v['magnification_cutout']:>7.1f}{snr.max():>9.1f}")
json.dump(res, open('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/imgsn.json','w'), indent=1)
