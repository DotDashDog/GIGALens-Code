import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
a=np.load('/global/homes/l/linusu/GIGALens-Code/data/desi238/cutout238b.npy')
print('shape',a.shape,'sum',a.sum(),'max',a.max())
sig=0.0076
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
im=ax[0].imshow(np.arcsinh(a/sig/3),origin='lower',cmap='magma');plt.colorbar(im,ax=ax[0]);ax[0].set_title('DESI-238 F140W (asinh S/N)')
im=ax[1].imshow(a/sig,origin='lower',cmap='magma',vmin=0,vmax=10);plt.colorbar(im,ax=ax[1]);ax[1].set_title('S/N clip 10')
# radial profile about the brightest pixel
n=a.shape[0]; cy,cx=np.unravel_index(np.argmax(a+0),a.shape)
yy,xx=np.indices(a.shape); r=np.hypot(yy-cy,xx-cx)*0.065
bins=np.arange(0,3.9,0.065)
prof=[np.median(a[(r>=bins[i])&(r<bins[i+1])]) for i in range(len(bins)-1)]
mx=[np.max(a[(r>=bins[i])&(r<bins[i+1])]) if ((r>=bins[i])&(r<bins[i+1])).sum() else np.nan for i in range(len(bins)-1)]
ax[2].semilogy(bins[:-1],np.array(prof)/sig,label='median/sig');ax[2].semilogy(bins[:-1],np.array(mx)/sig,label='max/sig')
ax[2].axhline(1,color='k',ls=':');ax[2].legend();ax[2].set_xlabel('r ["]');ax[2].set_title('radial profile, sigma units')
plt.tight_layout();plt.savefig('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/fig_desi238.png',dpi=110)
print('npix>3sig',(a>3*sig).sum(),'npix>10sig',(a>10*sig).sum())
print('flux in r>1.0" and >3sig:', a[(r>1.0)&(a>3*sig)].sum(), ' total flux r<1.0":', a[r<1.0].sum())
print('peak', a.max()/sig, 'sigma')
