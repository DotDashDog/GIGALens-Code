import jax
jax.config.update("jax_enable_x64", True)
import os, numpy as np, jax.numpy as jnp
from astropy.io import fits
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.light.shapelets import Shapelets
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig

NFW0={'Rs':37.439292907714844,'alpha_Rs':16.946298599243164,'center_x':5.350801467895508,'center_y':3.905395984649658,'e1':-0.0517449676990509,'e2':0.030034080147743225}
EPL_Le={'center_x':-22.150279998779297,'center_y':-24.73190689086914,'e1':0.05808083713054657,'e2':0.29447436332702637,'gamma':2.199589967727661,'theta_E':2.3904967308044434}
EPL_Lf={'center_x':-14.754530906677246,'center_y':-4.7380170822143555,'e1':0.2831626236438751,'e2':-0.20151233673095703,'gamma':2.59816312789917,'theta_E':0.9524290561676025}
shear={'gamma1':0.027523696422576904,'gamma2':-0.012137502431869507}
src4=Component(Shapelets(n_max=12,use_lstsq=True),{'beta':0.2498,'center_x':3.273,'center_y':3.131})
src5=Component(Shapelets(n_max=6,use_lstsq=True),{'beta':0.2033,'center_x':3.623,'center_y':-0.206})
src9=Component(SersicEllipse(n_max=12,use_lstsq=True),{'R_sersic':0.4022407829761505,'center_x':-10.395180702209473,'center_y':-16.07061767578125,'e1':0.279153972864151,'e2':0.29133689403533936,'n_sersic':2.795565128326416})

z4_5=1.432; z9=1.506; z_lens=0.49
# FIX: z_lens matches the mass plane (0.49); z_source_ref = the redshift the theta_E were
# referenced to (z4_5 = 1.432), now REQUIRED.
cosmo=Component(wCDM_Cosmo(z_lens=z_lens, z_source_ref=z4_5), dict(H0=70.0,Om0=0.3,k=0.0,w0=-1.0))
model=LensModel([
    Plane(redshift=z_lens, mass=[Component(NFW_ELLIPSE(),NFW0),Component(EPL(50),EPL_Le),Component(EPL(50),EPL_Lf),Component(Shear(),shear)]),
    Plane(redshift=z4_5, light=[src4,src5]),
    Plane(redshift=z9, light=[src9]),
], cosmo=cosmo)

path=os.path.join(os.path.dirname(__file__),"fwdnewcutouts")
def ds(ext, sees):
    with fits.open(os.path.join(path,f"source{ext}.fits")) as h:
        oi=jnp.array(h['DATA'].data.astype("float64")); bkg=h['DATA'].header['BKG_RMS']; et=h['PRIMARY'].header['EXPTIME']
        em=jnp.sqrt(bkg**2+jnp.clip(oi,0.0)/et)
    psf=jnp.load(os.path.join(path,f"psf{ext}.npy")); mask=jnp.load(os.path.join(path,"hot_pix.npy"))
    cfg=SimulatorConfig(delta_pix=0.2,num_pix=300,supersample=1,kernel=psf,likelihood_precision="float64")
    return Dataset(oi,cfg,error_map=em,mask=mask,sees=sees), bkg
d4_5,_=ds("4-5",[src4,src5]); d9,bkg9=ds("9",[src9])
pm=ProbModel(model,[d4_5,d9],mode="lstsq"); sim9=pm.simulators[1]
p=model.to_params({})
print("trace_mode:",sim9.trace_mode)
pos=sim9.trace(p); print("max|src9_pos-grid| (lensing applied):",float(jnp.max(jnp.abs(pos[2][0]-sim9.img_X))))
im=np.asarray(sim9.lstsq_simulate(p,d9.image,err_map=d9.error_map,mask=d9.mask))
tgt=np.asarray(d9.image); m=np.asarray(d9.mask).astype(bool)
disp=lambda a: np.where(m,a,np.nan)
norm=AsinhNorm(linear_width=5*bkg9, vmin=-2*bkg9, vmax=float(np.nanmax(disp(tgt))))
fig,ax=plt.subplots(1,3,figsize=(16,5.2))
for a,(d,t) in zip(ax,[(disp(tgt),"TARGET source9"),(disp(im),"MODEL (library fix, auto deflection_ratio)"),(disp(tgt)-disp(im),"residual")]):
    im0=a.imshow(d,origin="lower",norm=norm,cmap="viridis"); a.set_title(t,fontsize=10); fig.colorbar(im0,ax=a,fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"diag_src9_libfix.png"),dpi=95)
print("target peak xy:",np.unravel_index(np.nanargmax(disp(tgt)),tgt.shape)[::-1])
print("model  peak xy:",np.unravel_index(np.nanargmax(disp(im)),tgt.shape)[::-1])
print("saved diag_src9_libfix.png")
