import jax
jax.config.update("jax_enable_x64", True)
import os, numpy as np, jax.numpy as jnp
from astropy.io import fits
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig

NFW0 = {'Rs':37.439292907714844,'alpha_Rs':16.946298599243164,'center_x':5.350801467895508,
        'center_y':3.905395984649658,'e1':-0.0517449676990509,'e2':0.030034080147743225}
EPL_Le = {'center_x':-22.150279998779297,'center_y':-24.73190689086914,'e1':0.05808083713054657,
          'e2':0.29447436332702637,'gamma':2.199589967727661,'theta_E':2.3904967308044434}
EPL_Lf = {'center_x':-14.754530906677246,'center_y':-4.7380170822143555,'e1':0.2831626236438751,
          'e2':-0.20151233673095703,'gamma':2.59816312789917,'theta_E':0.9524290561676025}
shear = {'gamma1':0.027523696422576904,'gamma2':-0.012137502431869507}
src9_p = {'R_sersic':0.4022407829761505,'center_x':-10.395180702209473,'center_y':-16.07061767578125,
          'e1':0.279153972864151,'e2':0.29133689403533936,'n_sersic':2.795565128326416}

z9=1.506; z_lens=0.49
cp=dict(H0=70.0,Om0=0.3,k=0.0,w0=-1.0)
dr9 = float(np.asarray(wCDM_Cosmo(z_lens=z_lens).deflection_ratio(jnp.array([z9]),**cp)).ravel()[0])
print("deflection_ratio(z9) with z_lens=0.49 :", dr9)

src9 = Component(SersicEllipse(n_max=12, use_lstsq=True), src9_p)
# NO cosmology; explicit deflection_ratio on the source plane (what the trace supports)
model = LensModel([
    Plane(mass=[Component(NFW_ELLIPSE(), NFW0), Component(EPL(50), EPL_Le),
                Component(EPL(50), EPL_Lf), Component(Shear(), shear)]),
    Plane(deflection_ratio=dr9, light=[src9]),
])

path = os.path.join(os.path.dirname(__file__), "fwdnewcutouts")
with fits.open(os.path.join(path, "source9.fits")) as hdul:
    oi = jnp.array(hdul['DATA'].data.astype("float64"))
    bkg = hdul['DATA'].header['BKG_RMS']; exptime = hdul['PRIMARY'].header['EXPTIME']
    em = jnp.sqrt(bkg**2 + jnp.clip(oi,0.0)/exptime)
psf = jnp.load(os.path.join(path,"psf9.npy")); mask = jnp.load(os.path.join(path,"hot_pix.npy"))
cfg = SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1, kernel=psf, likelihood_precision="float64")
d9 = Dataset(oi, cfg, error_map=em, mask=mask, sees=[src9])
pm = ProbModel(model, [d9], mode="lstsq")
sim9 = pm.simulators[0]
params = model.to_params({})
print("trace_mode:", sim9.trace_mode, " geometry:", params["planes"][1].get("geometry"))
pos = sim9.trace(params); bx,by = pos[1]
print("max|src_pos-grid| (should be >0 now):", float(jnp.max(jnp.abs(bx-sim9.img_X))+jnp.max(jnp.abs(by-sim9.img_Y))))

im = np.asarray(sim9.lstsq_simulate(params, d9.image, err_map=d9.error_map, mask=d9.mask))
tgt = np.asarray(d9.image); m = np.asarray(mask)
fig, ax = plt.subplots(1,3, figsize=(15,5))
panels=[(np.where(m,tgt,0),"masked target source9"),(im,f"lstsq w/ deflection_ratio={dr9:.3f}"),
        (np.where(m,tgt,0)-im,"residual (target-model)")]
for a,(d,t) in zip(ax,panels):
    v=np.nanpercentile(np.abs(d),99); im0=a.imshow(d,origin="lower",vmin=-v if "resid" in t else np.nanpercentile(d,5),vmax=v)
    a.set_title(t); fig.colorbar(im0,ax=a,fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"diag_src9_fix.png"),dpi=90)
print("saved diag_src9_fix.png  model range:", np.nanmin(im), np.nanmax(im))
