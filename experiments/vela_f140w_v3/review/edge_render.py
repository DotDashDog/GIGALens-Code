import numpy as np, pickle, json
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
psf=np.load(DS+'/vela02_cam12_a0.400_rep00/psf.npy'); BKG=0.0076
def render(img,s,truth):
    light=ImageBasedLight(img,s)
    m=LensModel([Plane(mass=[Component(epl.EPL(50),{k:float(v) for k,v in truth[0][0].items()}),
                 Component(shear.Shear(),{k:float(v) for k,v in truth[0][1].items()})],
                 light=[Component(sersic.SersicEllipse(use_lstsq=False),{k:float(v) for k,v in truth[1][0].items()})]),
            Plane(deflection_ratio=1.0,light=[Component(light,{k:float(v) for k,v in truth[2][0].items()})])])
    sim=SceneSimulator(m,SimulatorConfig(delta_pix=0.065,num_pix=120,supersample=16,kernel=psf,likelihood_precision='float64'))
    p=[[dict(truth[0][0]),dict(truth[0][1])],[{**{k:float(v) for k,v in truth[1][0].items()},'Ie':0.0}],[{k:float(v) for k,v in truth[2][0].items()}]]
    return np.asarray(jnp.squeeze(sim.simulate(truth_x_to_scene_params(p,m))),dtype=np.float64)
for sid,shrink in [('25',0.3),('03',0.3),('09',0.3),('02',0.3)]:
    d=f'{DS}/vela{sid}_cam12_a0.400_rep00'; tr=pickle.load(open(d+'/truth_x.pkl','rb'))
    sb,s,_,_=_load_pristine_source(f'{SR}/vela{sid}_cam12_a0.400_f140w',False)
    sb,info=preprocess_source(sb,s,crop_radius_arcsec=None,recenter=True)
    n=sb.shape[0]; c=(n-1)/2; k=int(round(shrink/s))
    sb2=sb.copy(); sb2[:k,:]=0; sb2[-k:,:]=0; sb2[:,:k]=0; sb2[:,-k:]=0
    a=render(sb,s,tr); b=render(sb2,s,tr); dd=a-b
    nl=np.load(d+'/noiseless_image.npy').astype(float)
    sig=np.sqrt(BKG**2+np.maximum(nl,0)/1197.7)
    m=np.abs(dd)>0
    sn=dd.sum()/np.sqrt((sig**2)[m].sum())
    print(f"vela{sid}: outer {shrink}\" shell of the SOURCE FRAME contributes flux={dd.sum():.4f} cps "
          f"({100*dd.sum()/a.sum():.3f}% of arc), over {m.sum()} px, coherent S/N={sn:.2f}, max/sig={np.abs(dd).max()/BKG:.2f}")
