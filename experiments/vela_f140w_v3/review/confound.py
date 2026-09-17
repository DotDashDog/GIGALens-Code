import numpy as np, json
from scipy.stats import spearmanr, pearsonr
D='/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems'
man=json.load(open('/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json'))['extra']['per_system']
rows=np.load('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/src_rows.npy',allow_pickle=True)
geo=json.load(open('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/lensgeom.json'))
sn=json.load(open('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/imgsn.json'))
IDS=[r['sid'] for r in rows]
R50=np.array([r['r50'] for r in rows]); R90=np.array([r['r90'] for r in rows])
tE=np.array([man[f'vela{s}_cam12_a0.400_rep00']['theta_E'] for s in IDS])
mu=np.array([man[f'vela{s}_cam12_a0.400_rep00']['magnification_cutout'] for s in IDS])
nred=np.array([man[f'vela{s}_cam12_a0.400_rep00']['n_redraws'] for s in IDS])
amp=np.array([man[f'vela{s}_cam12_a0.400_rep00']['amp'] for s in IDS])
mag_un=np.array([man[f'vela{s}_cam12_a0.400_rep00']['source_ab_mag_unlensed'] for s in IDS])
nres=np.array([sn[s]['nres'] for s in IDS]); f3=np.array([sn[s]['f3'] for s in IDS])
snt=np.array([sn[s]['sn'] for s in IDS])
ratio=np.array([man[f'vela{s}_cam12_a0.400_rep00']['source_to_lens_ratio_cutout'] for s in IDS])
def rep(a,b,na,nb):
    r,p=spearmanr(a,b); print(f'  spearman({na},{nb}) = {r:+.3f} (p={p:.3f})')
print('n=10 systems')
rep(R50,tE,'R50_src','theta_E'); rep(R50,mu,'R50_src','mu'); rep(R90,mu,'R90_src','mu')
rep(R50,nres,'R50_src','N_res'); rep(mu,nres,'mu','N_res'); rep(R50,mag_un,'R50','unlensed_mag')
rep(R50,amp,'R50','amp'); rep(mu,snt,'mu','S/N_tot'); rep(R50,snt,'R50','S/N_tot')
rep(R50,ratio,'R50','src/lens ratio')
print()
print(f"{'sys':>6}{'R50src':>8}{'thetaE':>8}{'mu':>7}{'amp':>7}{'m_unl':>8}{'Nres':>7}{'redraw':>7}{'glare':>8}{'glare_sn':>9}")
for i,s in enumerate(IDS):
    nl=np.load(f'{D}/vela{s}_cam12_a0.400_rep00/noiseless_image.npy').astype(float)
    ll=np.load(f'{D}/vela{s}_cam12_a0.400_rep00/lens_light_only.npy').astype(float)
    src=nl-ll
    sig=np.sqrt(0.0076**2+np.maximum(nl,0)/1197.7)
    det=src>3*sig
    glare=float(src[(ll>src)&det].sum()/max(src[det].sum(),1e-30))
    # S/N loss from lens-light Poisson: compare sigma with and without lens light
    sig0=np.sqrt(0.0076**2+np.maximum(src,0)/1197.7)
    gs=float(np.sqrt(((src/sig0)**2).sum())/np.sqrt(((src/sig)**2).sum()))
    print(f"vela{s:>2}{R50[i]:>8.3f}{tE[i]:>8.2f}{mu[i]:>7.1f}{amp[i]:>7.2f}{mag_un[i]:>8.2f}{nres[i]:>7.1f}{nred[i]:>7d}{glare:>8.3f}{gs:>9.3f}")
