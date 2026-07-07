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
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig

NFW0={'Rs':37.439292907714844,'alpha_Rs':16.946298599243164,'center_x':5.350801467895508,'center_y':3.905395984649658,'e1':-0.0517449676990509,'e2':0.030034080147743225}
EPL_Le={'center_x':-22.150279998779297,'center_y':-24.73190689086914,'e1':0.05808083713054657,'e2':0.29447436332702637,'gamma':2.199589967727661,'theta_E':2.3904967308044434}
EPL_Lf={'center_x':-14.754530906677246,'center_y':-4.7380170822143555,'e1':0.2831626236438751,'e2':-0.20151233673095703,'gamma':2.59816312789917,'theta_E':0.9524290561676025}
shear={'gamma1':0.027523696422576904,'gamma2':-0.012137502431869507}
src9_p={'R_sersic':0.4022407829761505,'center_x':-10.395180702209473,'center_y':-16.07061767578125,'e1':0.279153972864151,'e2':0.29133689403533936,'n_sersic':2.795565128326416}
mass=[Component(NFW_ELLIPSE(),NFW0),Component(EPL(50),EPL_Le),Component(EPL(50),EPL_Lf),Component(Shear(),shear)]

path=os.path.join(os.path.dirname(__file__),"fwdnewcutouts")
with fits.open(os.path.join(path,"source9.fits")) as h:
    oi=jnp.array(h['DATA'].data.astype("float64")); bkg=h['DATA'].header['BKG_RMS']; et=h['PRIMARY'].header['EXPTIME']
    em=jnp.sqrt(bkg**2+jnp.clip(oi,0.0)/et)
psf=jnp.load(os.path.join(path,"psf9.npy")); mask=jnp.load(os.path.join(path,"hot_pix.npy"))
cfg=SimulatorConfig(delta_pix=0.2,num_pix=300,supersample=1,kernel=psf,likelihood_precision="float64")

def run(model, src):
    d=Dataset(oi,cfg,error_map=em,mask=mask,sees=[src]); pm=ProbModel(model,[d],mode="lstsq")
    s=pm.simulators[0]; p=model.to_params({})
    return np.asarray(s.lstsq_simulate(p,d.image,err_map=d.error_map,mask=mask))

# (A) broken: cosmology + redshift geometry, single mass plane
src_b=Component(SersicEllipse(n_max=12,use_lstsq=True),src9_p)
cosmo=Component(wCDM_Cosmo(z_lens=0.962),dict(H0=70.0,Om0=0.3,k=0.0,w0=-1.0))
model_b=LensModel([Plane(redshift=0.49,mass=mass),Plane(redshift=1.432),Plane(redshift=1.506,light=[src_b])],cosmo=cosmo)
im_b=run(model_b,src_b)

# (C) deflection_ratio from correct lens z=0.49
cp=dict(H0=70.0,Om0=0.3,k=0.0,w0=-1.0)
dr=float(np.asarray(wCDM_Cosmo(z_lens=0.49).deflection_ratio(jnp.array([1.506]),**cp)).ravel()[0])
src_c=Component(SersicEllipse(n_max=12,use_lstsq=True),src9_p)
model_c=LensModel([Plane(mass=mass),Plane(deflection_ratio=dr,light=[src_c])])
im_c=run(model_c,src_c)

tgt=np.asarray(oi); m=np.asarray(mask).astype(bool)
disp=lambda a: np.where(m,a,np.nan)
norm=AsinhNorm(linear_width=5*bkg, vmin=-2*bkg, vmax=float(np.nanmax(np.where(m,tgt,np.nan))))
fig,ax=plt.subplots(1,3,figsize=(16,5.2))
for a,(d,t) in zip(ax,[(disp(tgt),"TARGET source9 (hot px masked)"),(disp(im_b),"BROKEN: cosmo+redshift, 1 mass plane\n(no deflection applied)"),(disp(im_c),f"deflection_ratio={dr:.3f} (z_lens=0.49)")]):
    im0=a.imshow(d,origin="lower",norm=norm,cmap="viridis"); a.set_title(t,fontsize=10); fig.colorbar(im0,ax=a,fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"diag_src9_compare.png"),dpi=95)
print("dr(z9,z_lens=0.49)=",dr)
print("target peak xy:", np.unravel_index(np.nanargmax(disp(tgt)),tgt.shape)[::-1])
print("broken peak xy:", np.unravel_index(np.nanargmax(disp(im_b)),tgt.shape)[::-1])
print("dr-model peak xy:", np.unravel_index(np.nanargmax(disp(im_c)),tgt.shape)[::-1])
print("saved diag_src9_compare.png")
