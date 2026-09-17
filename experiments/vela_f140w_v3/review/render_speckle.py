import os, json, pickle, time, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens_research.simulations.image_based_light import ImageBasedLight
from gigalens.simulator import SimulatorConfig
from gigalens_research.simtests.experiments.vela_simulated import _load_pristine_source, preprocess_source

DS='/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems'
SR='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
SS=int(os.environ.get('SS','16')); BKG=0.0076; DP=0.065; NP=120
SIDS=os.environ.get('SIDS','02,22,25').split(',')
psf=np.load(DS+'/vela02_cam12_a0.400_rep00/psf.npy')

def build(src_img, src_scale, truth):
    light=ImageBasedLight(src_img, src_scale)
    planes=[Plane(mass=[Component(epl.EPL(50), {k:float(v) for k,v in truth[0][0].items()}),
                        Component(shear.Shear(), {k:float(v) for k,v in truth[0][1].items()})],
                  light=[Component(sersic.SersicEllipse(use_lstsq=False), {k:float(v) for k,v in truth[1][0].items()})]),
            Plane(deflection_ratio=1.0, light=[Component(light, {k:float(v) for k,v in truth[2][0].items()})])]
    return LensModel(planes)

out={}
for sid in SIDS:
    d=f'{DS}/vela{sid}_cam12_a0.400_rep00'
    truth=pickle.load(open(d+'/truth_x.pkl','rb'))
    sb,src_scale,_,meta=_load_pristine_source(f'{SR}/vela{sid}_cam12_a0.400_f140w', False)
    sb,_=preprocess_source(sb, src_scale, crop_radius_arcsec=None, recenter=True)
    amp=float(truth[2][0]['amp'])
    res={}
    for tag, img in (('orig', sb), ('smooth2', gaussian_filter(sb,2.0)), ('smooth4', gaussian_filter(sb,4.0))):
        m=build(img, src_scale, truth)
        cfg=SimulatorConfig(delta_pix=DP, num_pix=NP, supersample=SS, kernel=psf, likelihood_precision='float64')
        sim=SceneSimulator(m,cfg)
        # lens light off (Ie=0) -> source only
        params=[[dict(truth[0][0]),dict(truth[0][1])],[{**{k:float(v) for k,v in truth[1][0].items()},'Ie':0.0}],[{k:float(v) for k,v in truth[2][0].items()}]]
        from gigalens_research.inference_utils.params import truth_x_to_scene_params
        t0=time.time(); im=np.asarray(jnp.squeeze(sim.simulate(truth_x_to_scene_params(params,m))),dtype=np.float64)
        res[tag]=im
        print(f'vela{sid} {tag}: sum={im.sum():.5f} max={im.max():.4f} ({time.time()-t0:.1f}s)')
    ref=np.load(d+'/noiseless_image.npy').astype(float)-np.load(d+'/lens_light_only.npy').astype(float)
    print(f'  ref(ss32) sum={ref.sum():.5f}; orig(ss{SS}) vs ref: max|d|/sig={np.abs(res["orig"]-ref).max()/BKG:.3f}, rms/sig={np.std(res["orig"]-ref)/BKG:.4f}')
    for tag in ('smooth2','smooth4'):
        d2=res['orig']-res[tag]
        print(f'  orig-{tag}: flux change={100*(res[tag].sum()/res["orig"].sum()-1):+.3f}%  max|d|/sig={np.abs(d2).max()/BKG:.2f}  rms/sig={np.std(d2)/BKG:.3f}  '
              f'rms over pix>3sig region /sig={np.std(d2[res["orig"]>3*BKG])/BKG:.3f}  Nchi2={np.sum((d2/BKG)**2):.0f}')
    out[sid]=res
np.save('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/speckle_imgs.npy', np.array(out,dtype=object))
# figure
fig,axs=plt.subplots(len(SIDS),4,figsize=(17,4.2*len(SIDS)))
if len(SIDS)==1: axs=axs[None,:]
for k,sid in enumerate(SIDS):
    r=out[sid]; d2=r['orig']-r['smooth2']
    a=axs[k,0].imshow(r['orig'],origin='lower',cmap='magma'); plt.colorbar(a,ax=axs[k,0]); axs[k,0].set_title(f'vela{sid} lensed source (ss{SS})')
    a=axs[k,1].imshow(r['orig']/BKG,origin='lower',cmap='magma',vmin=0,vmax=10); plt.colorbar(a,ax=axs[k,1]); axs[k,1].set_title('S/N (bkg only), clip 10')
    a=axs[k,2].imshow(d2/BKG,origin='lower',cmap='coolwarm',vmin=-2,vmax=2); plt.colorbar(a,ax=axs[k,2]); axs[k,2].set_title('(orig - smooth2px)/sigma_bkg')
    a=axs[k,3].imshow((r['orig']-r['smooth4'])/BKG,origin='lower',cmap='coolwarm',vmin=-2,vmax=2); plt.colorbar(a,ax=axs[k,3]); axs[k,3].set_title('(orig - smooth4px)/sigma_bkg')
plt.tight_layout(); plt.savefig('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/fig_speckle.png',dpi=105)
print('saved fig')
