import numpy as np, json, pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from gigalens.jax.profiles.mass import epl, shear
from gigalens_research.simtests.experiments.vela_simulated import _load_pristine_source, preprocess_source
DS='/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems'
SR='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
IDS=["02","03","04","08","09","21","22","23","25","26"]
E=epl.EPL(50); SH=shear.Shear()
N=2400; H=4.5   # image-plane grid +-4.5" (cutout is +-3.9")
g=np.linspace(-H,H,N); X,Y=np.meshgrid(g,g,indexing='ij'); dx=g[1]-g[0]
res={}
fig,axs=plt.subplots(2,5,figsize=(24,9.5))
for k,sid in enumerate(IDS):
    d=f'{DS}/vela{sid}_cam12_a0.400_rep00'
    tr=pickle.load(open(d+'/truth_x.pkl','rb'))
    lp={kk:float(v) for kk,v in tr[0][0].items()}; sp={kk:float(v) for kk,v in tr[0][1].items()}
    ax_,ay_=E.deriv(jnp.array(X),jnp.array(Y),**lp); bx_,by_=SH.deriv(jnp.array(X),jnp.array(Y),**sp)
    AX=np.asarray(ax_+bx_); AY=np.asarray(ay_+by_)
    BX=X-AX; BY=Y-AY
    # jacobian
    dBXdx,dBXdy=np.gradient(BX,dx,dx); dBYdx,dBYdy=np.gradient(BY,dx,dx)
    det=dBXdx*dBYdy-dBXdy*dBYdx
    mu=1.0/np.abs(det)
    # source-plane magnification map by ray-shooting histogram
    M=600; S=1.5
    sb_edges=np.linspace(-S,S,M+1)
    hist,_,_=np.histogram2d(BX.ravel(),BY.ravel(),bins=[sb_edges,sb_edges])
    cell_s=(2*S/M)**2
    mu_src=hist*dx*dx/cell_s
    # source image on the same grid: SB(x,y)=img[ix,iy] with ImageBasedLight convention
    sb,s,_,meta=_load_pristine_source(f'{SR}/vela{sid}_cam12_a0.400_f140w',False)
    sb,_=preprocess_source(sb,s,crop_radius_arcsec=None,recenter=True)
    cx=float(tr[2][0]['center_x']); cy=float(tr[2][0]['center_y'])
    sc=0.5*(sb_edges[1:]+sb_edges[:-1])
    SX,SY=np.meshgrid(sc,sc,indexing='ij')
    n=sb.shape[0]; half=(n-1)/2*s
    ii=np.clip(np.round((SX-cx+half)/s).astype(int),0,n-1)
    jj=np.clip(np.round((SY-cy+half)/s).astype(int),0,n-1)
    srcmap=sb[ii,jj]*(np.abs(SX-cx)<half)*(np.abs(SY-cy)<half)
    w=srcmap*cell_s
    tot_in=w.sum()
    # total source flux (whole 800px frame) in cps
    tot_all=float(sb.sum())*s*s
    mu_eff=(w*mu_src).sum()/max(tot_in,1e-30)
    q=lambda th: float((w*(mu_src>th)).sum()/tot_in)
    # brightest clumps: top source pixels within +-1.5"
    flat=np.argsort(srcmap.ravel())[::-1]
    pk=[]
    for f in flat:
        i,j=np.unravel_index(f,srcmap.shape)
        if all((i-a)**2+(j-b)**2> (0.08/(2*S/M))**2 for a,b in pk): pk.append((i,j))
        if len(pk)>=6: break
    mus=[float(mu_src[i,j]) for i,j in pk]
    res[sid]=dict(mu_eff=float(mu_eff), f_mu3=q(3), f_mu5=q(5), f_mu10=q(10), f_mu20=q(20),
                  frac_in_window=float(tot_in/tot_all), clump_mus=mus,
                  mu_at_center=float(mu_src[np.argmin(abs(sc)),np.argmin(abs(sc))]),
                  theta_E=lp['theta_E'])
    ax=axs[k//5,k%5]
    ax.imshow(np.log10(np.maximum(mu_src,0.3)).T,origin='lower',extent=[-S,S,-S,S],cmap='Greys',vmin=-0.2,vmax=1.6)
    ax.contour(sc,sc,np.log10(np.maximum(srcmap,1e-6)).T,levels=np.log10(srcmap.max())-np.array([3,2,1,0.3]),colors='r',linewidths=0.7)
    ax.contour(sc,sc,mu_src.T,levels=[3,10],colors=['c','b'],linewidths=0.8)
    ax.plot(cx,cy,'g+',ms=10)
    ax.set_title(f"vela{sid}  mu_eff={mu_eff:.1f}  f(mu>10)={q(10):.2f}",fontsize=9)
plt.tight_layout(); plt.savefig('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/fig_lensgeom.png',dpi=95)
print(f"{'sys':>6}{'thetaE':>8}{'mu_eff':>8}{'mu_rec':>8}{'f(mu>3)':>9}{'f(mu>5)':>9}{'f(mu>10)':>9}{'f(mu>20)':>9}{'f_in3\"':>8}  clump mus")
man=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))['extra']['per_system']
for sid in IDS:
    r=res[sid]; mrec=man[f'vela{sid}_cam12_a0.400_rep00']['magnification_cutout']
    print(f"vela{sid:>2}{r['theta_E']:>8.2f}{r['mu_eff']:>8.2f}{mrec:>8.2f}{r['f_mu3']:>9.3f}{r['f_mu5']:>9.3f}{r['f_mu10']:>9.3f}{r['f_mu20']:>9.3f}{r['frac_in_window']:>8.3f}  "+
          ' '.join(f'{m:.1f}' for m in r['clump_mus']))
json.dump(res,open('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/lensgeom.json','w'),indent=1)
