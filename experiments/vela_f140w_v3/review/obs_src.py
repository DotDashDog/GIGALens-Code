import numpy as np, json, pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from gigalens_research.simtests.experiments.vela_simulated import _load_pristine_source, preprocess_source
R='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
geo=json.load(open('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/lensgeom.json'))
man=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))['extra']['per_system']
IDS=["02","03","04","08","09","21","22","23","25","26"]
PSF=0.1723
fig,axs=plt.subplots(3,10,figsize=(30,9.6))
print(f"{'sys':>6}{'mu':>7}{'res_eff\"':>9}{'res_kpc':>8}{'Nclump':>8}{'f_clump':>9}{'sig/mean@bright':>16}")
for k,sid in enumerate(IDS):
    sb,s,_,meta=_load_pristine_source(f'{R}/vela{sid}_cam12_a0.400_f140w',False)
    sb,_=preprocess_source(sb,s,crop_radius_arcsec=None,recenter=True)
    mu=man[f'vela{sid}_cam12_a0.400_rep00']['magnification_cutout']
    res_eff=PSF/np.sqrt(mu)          # isotropic-equivalent source-plane resolution
    sig_px=res_eff/2.355/s
    n=sb.shape[0]; c=n//2; w=int(round(1.2/s))
    cut=sb[c-w:c+w,c-w:c+w]
    blur=gaussian_filter(cut, sig_px)
    # count clumps above 20% of peak in the blurred map, separated
    mx=maximum_filter(blur, size=int(max(3,2*sig_px)))
    pk=(blur==mx)&(blur>0.2*blur.max())
    nclump=int(pk.sum())
    # fraction of flux in "clumps" = above 2x the azimuthally smoothed (5x resolution) baseline
    base=gaussian_filter(cut, sig_px*5)
    fcl=float(cut[blur>2*base].sum()/cut.sum())
    # fractional pixel scatter in bright region
    m=gaussian_filter(cut,4.0); br=m>np.percentile(m,99.5)
    fs=float(np.std((cut-m)[br])/np.mean(m[br]))
    print(f"vela{sid:>2}{mu:>7.1f}{res_eff:>9.3f}{res_eff/s*meta['source_pixel_scale_kpc']:>8.2f}{nclump:>8d}{fcl:>9.3f}{fs:>16.3f}")
    ext=[-1.2,1.2,-1.2,1.2]
    axs[0,k].imshow(np.arcsinh(cut.T/ (0.01*cut.max())),origin='lower',cmap='magma',extent=ext); axs[0,k].set_title(f'vela{sid} native (asinh)',fontsize=9)
    axs[1,k].imshow(np.arcsinh(blur.T/(0.01*blur.max())),origin='lower',cmap='magma',extent=ext); axs[1,k].set_title(f'blurred to {res_eff:.3f}" (mu={mu:.0f})',fontsize=9)
    axs[2,k].imshow(blur.T,origin='lower',cmap='magma',extent=ext,vmax=np.percentile(blur,99.9)); axs[2,k].set_title('blurred, linear',fontsize=9)
    for a in axs[:,k]: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/fig_obs_src.png',dpi=85)
print('saved')
