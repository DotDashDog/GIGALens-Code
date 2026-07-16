"""Rebuild the carousel model EXACTLY as prelim_sim_carousel.ipynb (cells 0-8),
expose prob_model / model_seq / ctx / param names. Imported by probe scripts."""
import os, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from astropy.io import fits
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.light.shapelets import Shapelets
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.inference import ModellingSequence

EXPDIR = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel"

NFW0 = Component(NFW_ELLIPSE(), dict(
   Rs = tfd.Uniform(20,100), alpha_Rs = tfd.Uniform(10,40),
   e1 = tfd.TruncatedNormal(0,0.05,-0.2,0.2), e2 = tfd.TruncatedNormal(0,0.05,-0.2,0.2),
   center_x = tfd.Normal(5.344,0.05), center_y = tfd.Normal(3.805,0.05)))
EPL_Le = Component(EPL(50), dict(
   theta_E = tfd.TruncatedNormal(2.4,0.1,1,3), gamma = tfd.TruncatedNormal(2.2,0.5,1,3),
   e1 = tfd.TruncatedNormal(0,0.1,-0.3,0.3), e2 = tfd.TruncatedNormal(0,0.1,-0.3,0.3),
   center_x = tfd.Normal(-22.1,0.1), center_y = tfd.Normal(-24.7,0.1)))
EPL_Lf = Component(EPL(50), dict(
   center_x = tfd.Normal(-15.10088063,0.1), center_y = tfd.Normal(-4.66657821,0.1),
   e1 = tfd.TruncatedNormal(0,0.1,-0.3,0.3), e2 = tfd.TruncatedNormal(0,0.1,-0.3,0.3),
   theta_E = tfd.TruncatedNormal(0.8151327,0.05,0.2,1.5), gamma = tfd.TruncatedNormal(2.2266,0.5,1,3)))
shear = Component(Shear(), dict(
    gamma1 = tfd.TruncatedNormal(0.,0.1,-0.3,0.3), gamma2 = tfd.TruncatedNormal(0.,0.1,-0.3,0.3)))
src4 = Component(Shapelets(n_max=8, use_lstsq=True), dict(
    center_x = tfd.Normal(3.7,1), center_y = tfd.Normal(3.2,1), beta = tfd.LogNormal(jnp.log(0.4),0.15)))
src5 = Component(Shapelets(n_max=6, use_lstsq=True), dict(
    center_x = tfd.Normal(3.0,1), center_y = tfd.Normal(0.,1), beta = tfd.LogNormal(jnp.log(0.1),0.15)))
src9 = Component(SersicEllipse(use_lstsq=True), dict(
    center_x = tfd.Normal(-10,1), center_y = tfd.Normal(-16,1),
    n_sersic = tfd.Uniform(0.1,10), R_sersic = tfd.LogNormal(jnp.log(0.4),0.15),
    e1 = tfd.TruncatedNormal(0,0.1,-0.3,0.3), e2 = tfd.TruncatedNormal(0,0.1,-0.3,0.3)))

z4_5=1.432; z9=1.506; z_lens=0.49
cosmo = Component(wCDM_Cosmo(z_lens=z_lens, z_source_ref=z4_5), dict(H0=70.0,Om0=0.3,k=0.0,w0=-1.0))
model = LensModel([
    Plane(redshift=z_lens, mass=[NFW0, EPL_Le, EPL_Lf, shear]),
    Plane(redshift=z4_5, light=[src4, src5]),
    Plane(redshift=z9, light=[src9]),
], cosmo=cosmo)

def dataset_from_dir(path, ext):
    with fits.open(os.path.join(path, f"source{ext}.fits")) as hdul:
        observed_image = jnp.array(hdul['DATA'].data.astype("float64"))
        error_map = jnp.array(np.sqrt(hdul['STAT'].data.astype("float64")))
        psf = hdul['PSF'].data.astype(jnp.float64)
        mask = hdul['MASK'].data.astype(jnp.bool)
    return observed_image, error_map, psf, mask

path = os.path.join(EXPDIR, "newnewcutouts/")
def ds(ext, sees):
    observed_image, error_map, psf, mask = dataset_from_dir(path, ext)
    cfg = SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1, kernel=psf,
                          likelihood_precision="float64", conv_precision="float32")
    return Dataset(observed_image, cfg, error_map=error_map, mask=mask, sees=sees)

d4_5 = ds("4-5", sees=[src4, src5])
d9 = ds("9", sees=[src9])
prob_model = ProbModel(model, [d4_5, d9], mode="lstsq")
model_seq = ModellingSequence(prob_model)

def make_prob_model(conv_precision="float32"):
    """Rebuild the prob_model with a chosen conv_precision (default = notebook's
    float32). Used to isolate convolution-precision noise in diagnostics."""
    def _ds(ext, sees):
        oi, em, psf, mask = dataset_from_dir(path, ext)
        cfg = SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1, kernel=psf,
                              likelihood_precision="float64", conv_precision=conv_precision)
        return Dataset(oi, cfg, error_map=em, mask=mask, sees=sees)
    return ProbModel(model, [_ds("4-5", sees=[src4, src5]), _ds("9", sees=[src9])], mode="lstsq")

def param_names(dim):
    probe = np.zeros((1, dim))
    return list(prob_model.bij.forward(list(probe.T)).keys())
