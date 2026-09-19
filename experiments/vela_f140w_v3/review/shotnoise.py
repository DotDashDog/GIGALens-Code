import numpy as np, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
R='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
sids=['02','04','08','22','25','21']
fig,ax=plt.subplots(1,2,figsize=(13,5))
for sid in sids:
    img=np.load(f'{R}/vela{sid}_cam12_a0.400_f140w/source_image.npy').astype(float)
    m=gaussian_filter(img,4.0)
    r=img-m
    lo=np.log10(np.clip(m,1e-8,None))
    bins=np.arange(-7,1.0,0.25); idx=np.digitize(lo.ravel(),bins)
    rv=r.ravel(); mv=m.ravel()
    xs=[];v_over_m=[];fracsig=[]
    for i in range(1,len(bins)):
        s=idx==i
        if s.sum()<500: continue
        mm=mv[s].mean(); vv=rv[s].var()
        xs.append(mm); v_over_m.append(vv/mm); fracsig.append(np.sqrt(vv)/mm)
    ax[0].loglog(xs,v_over_m,'o-',ms=3,label=f'vela{sid}')
    ax[1].loglog(xs,fracsig,'o-',ms=3,label=f'vela{sid}')
    print(f'vela{sid}: Var/mean over 4 decades: min={min(v_over_m):.3e} max={max(v_over_m):.3e} ratio={max(v_over_m)/min(v_over_m):.1f}')
ax[0].set_xlabel('local mean SB [nJy/src px]'); ax[0].set_ylabel('Var(residual)/mean  [nJy]')
ax[0].set_title('flat => pure shot noise with fixed packet weight w\n(rising ∝ mean => real structure)')
ax[1].set_xlabel('local mean SB [nJy/src px]'); ax[1].set_ylabel('sigma/mean (fractional pixel noise)')
ax[1].axhline(1.0,color='k',ls=':')
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/fig_shotnoise.png',dpi=110)
print('saved')
