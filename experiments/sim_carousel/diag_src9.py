import jax
jax.config.update("jax_enable_x64", True)
import os
import numpy as np
import jax.numpy as jnp
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.light.shapelets import Shapelets
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

src9 = Component(SersicEllipse(n_max=12, use_lstsq=True),{
    'R_sersic':0.4022407829761505,'center_x':-10.395180702209473,'center_y':-16.07061767578125,
    'e1':0.279153972864151,'e2':0.29133689403533936,'n_sersic':2.795565128326416})

z1=0.962; z4_5=1.432; z9=1.506; z_lens=0.49
cosmo = Component(wCDM_Cosmo(z_lens=z1), dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
src4 = Component(Shapelets(n_max=12, use_lstsq=True), {'beta':0.2498,'center_x':3.273,'center_y':3.131})
src5 = Component(Shapelets(n_max=6, use_lstsq=True), {'beta':0.2033,'center_x':3.623,'center_y':-0.206})

model = LensModel([
    Plane(redshift=z_lens, mass=[Component(NFW_ELLIPSE(), NFW0), Component(EPL(50), EPL_Le),
                                 Component(EPL(50), EPL_Lf), Component(Shear(), shear)]),
    Plane(redshift=z4_5, light=[src4, src5]),
    Plane(redshift=z9, light=[src9]),
], cosmo=cosmo)

path = os.path.join(os.path.dirname(__file__), "fwdnewcutouts")
def dataset_from_dir(ext):
    with fits.open(os.path.join(path, f"source{ext}.fits")) as hdul:
        observed_image = jnp.array(hdul['DATA'].data.astype("float64"))
        bkg = hdul['DATA'].header['BKG_RMS']; exptime = hdul['PRIMARY'].header['EXPTIME']
        error_map = jnp.sqrt(bkg**2 + jnp.clip(observed_image,0.0)/exptime)
    psf = jnp.load(os.path.join(path, f"psf{ext}.npy"))
    mask = jnp.load(os.path.join(path, f"hot_pix.npy"))
    return observed_image, error_map, psf, mask

def ds(ext, sees):
    oi, em, psf, mask = dataset_from_dir(ext)
    cfg = SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1, kernel=psf, likelihood_precision="float64")
    return Dataset(oi, cfg, error_map=em, mask=mask, sees=sees)

d4_5 = ds("4-5", sees=[src4, src5])
d9 = ds("9", sees=[src9])
prob_model = ProbModel(model, [d4_5, d9], mode="lstsq")
sim9 = prob_model.simulators[1]

params = model.to_params({})
print("trace_mode:", sim9.trace_mode)
print("sim9._light planes:", [(i,j) for i,j,_,_ in sim9._light])
print("geometry keys per plane:", [list(params["planes"][i].get("geometry",{}).keys()) for i in range(3)])

# --- Is the deflection actually applied to the source plane (index 2)? ---
positions = sim9.trace(params)
bx, by = positions[2]
gridx, gridy = sim9.img_X, sim9.img_Y
diff = float(jnp.max(jnp.abs(bx - gridx)) + jnp.max(jnp.abs(by - gridy)))
print("max|source_pos - image_grid| (0 => NO lensing applied):", diff)

# deflection field magnitude, to show it IS computed but not used
fx, fy = sim9._alpha(params, gridx, gridy)
print("max deflection |alpha|:", float(jnp.max(jnp.hypot(fx, fy))))

im = sim9.lstsq_simulate(params, d9.image, err_map=d9.error_map, mask=d9.mask)
im = np.asarray(im)
print("output finite:", np.isfinite(im).all(), " range:", np.nanmin(im), np.nanmax(im))

tgt = np.asarray(d9.image); mask = np.asarray(d9.mask)
fig, ax = plt.subplots(1,3, figsize=(15,5))
for a,(d,t) in zip(ax, [(tgt,"target source9"),(im,"lstsq_simulate"),(np.where(mask,tgt,0),"masked target")]):
    vmax = np.nanpercentile(d, 99.5) if np.isfinite(d).any() else 1
    im0=a.imshow(d, origin="lower", vmin=np.nanpercentile(d,5), vmax=vmax); a.set_title(t); fig.colorbar(im0,ax=a,fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"diag_src9.png"), dpi=90)
print("saved diag_src9.png")
