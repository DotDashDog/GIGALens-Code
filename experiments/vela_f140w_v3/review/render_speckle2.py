import os, json, pickle, time, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens_research.simulations.image_based_light import ImageBasedLight
from gigalens.simulator import SimulatorConfig
from gigalens_research.simtests.experiments.vela_simulated import _load_pristine_source, preprocess_source
from gigalens_research.inference_utils.params import truth_x_to_scene_params

DS='/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems'
SR='/pscratch/sd/l/linusu/gigalens/vela_sources_pristine'
SS=16; BKG=0.0076; DP=0.065; NP=120
psf=np.load(DS+'/vela02_cam12_a0.400_rep00/psf.npy')
SIDS=['02','22','25','04','21']
def render(src_img, src_scale, truth):
    light=ImageBasedLight(src_img, src_scale)
    m=LensModel([Plane(mass=[Component(epl.EPL(50), {k:float(v) for k,v in truth[0][0].items()}),
                        Component(shear.Shear(), {k:float(v) for k,v in truth[0][1].items()})],
                  light=[Component(sersic.SersicEllipse(use_lstsq=False), {k:float(v) for k,v in truth[1][0].items()})]),
            Plane(deflection_ratio=1.0, light=[Component(light, {k:float(v) for k,v in truth[2][0].items()})])])
    cfg=SimulatorConfig(delta_pix=DP, num_pix=NP, supersample=SS, kernel=psf, likelihood_precision='float64')
    sim=SceneSimulator(m,cfg)
    p=[[dict(truth[0][0]),dict(truth[0][1])],[{**{k:float(v) for k,v in truth[1][0].items()},'Ie':0.0}],[{k:float(v) for k,v in truth[2][0].items()}]]
    return np.asarray(jnp.squeeze(sim.simulate(truth_x_to_scene_params(p,m))),dtype=np.float64)

print(f"{'sys':>6} {'variant':>9} {'dflux%':>8} {'max|d|/s':>9} {'rms/s':>7} {'rms_det/s':>10} {'chi2':>8} {'Npix>3s':>8}")
store={}
for sid in SIDS:
    d=f'{DS}/vela{sid}_cam12_a0.400_rep00'
    truth=pickle.load(open(d+'/truth_x.pkl','rb'))
    sb,s,_,meta=_load_pristine_source(f'{SR}/vela{sid}_cam12_a0.400_f140w', False)
    sb,_=preprocess_source(sb, s, crop_radius_arcsec=None, recenter=True)
    base=render(sb,s,truth); det=base>3*BKG
    store[sid]={'orig':base}
    for tag,img in (('med3',median_filter(sb,3)),('gauss1',gaussian_filter(sb,1.0)),('gauss2',gaussian_filter(sb,2.0))):
        im=render(img,s,truth); store[sid][tag]=im; dd=base-im
        print(f"vela{sid:>2} {tag:>9} {100*(im.sum()/base.sum()-1):+8.3f} {np.abs(dd).max()/BKG:9.2f} {np.std(dd)/BKG:7.3f} "
              f"{np.std(dd[det])/BKG:10.3f} {np.sum((dd/BKG)**2):8.0f} {det.sum():8d}")
np.save('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/speckle2.npy', np.array(store,dtype=object))
fig,axs=plt.subplots(len(SIDS),4,figsize=(17,4.0*len(SIDS)))
for k,sid in enumerate(SIDS):
    r=store[sid]
    a=axs[k,0].imshow(r['orig']/BKG,origin='lower',cmap='magma',vmin=0,vmax=10);plt.colorbar(a,ax=axs[k,0]);axs[k,0].set_title(f'vela{sid} S/N')
    for j,tag in enumerate(['med3','gauss1','gauss2']):
        a=axs[k,j+1].imshow((r['orig']-r[tag])/BKG,origin='lower',cmap='coolwarm',vmin=-1,vmax=1);plt.colorbar(a,ax=axs[k,j+1])
        axs[k,j+1].set_title(f'(orig-{tag})/sigma')
plt.tight_layout();plt.savefig('/global/homes/l/linusu/.claude/jobs/41490d07/tmp/review/fig_speckle2.png',dpi=100)
