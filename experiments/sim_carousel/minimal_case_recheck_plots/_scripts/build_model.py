import os, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE
from gigalens.jax.profiles.light.shapelets import Shapelets
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig
from astropy.io import fits
EXP="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel"
def build():
    NFW0=Component(NFW_ELLIPSE(),dict(Rs=tfd.Uniform(20,100),alpha_Rs=tfd.Uniform(10,40),
        e1=tfd.TruncatedNormal(0,0.05,-0.2,0.2),e2=tfd.TruncatedNormal(0,0.05,-0.2,0.2),
        center_x=tfd.Normal(5.344,0.05),center_y=tfd.Normal(3.805,0.05)))
    shear=Component(Shear(),dict(gamma1=tfd.TruncatedNormal(0.,0.1,-0.3,0.3),gamma2=tfd.TruncatedNormal(0.,0.1,-0.3,0.3)))
    src4=Component(Shapelets(n_max=8,use_lstsq=True),dict(center_x=tfd.Normal(3.7,1),center_y=tfd.Normal(3.2,1),beta=tfd.LogNormal(jnp.log(0.4),0.15)))
    src5=Component(Shapelets(n_max=6,use_lstsq=True),dict(center_x=tfd.Normal(3.0,1),center_y=tfd.Normal(0.,1),beta=tfd.LogNormal(jnp.log(0.1),0.15)))
    cosmo=Component(wCDM_Cosmo(z_lens=0.49,z_source_ref=1.432),dict(H0=70.0,Om0=0.3,k=0.0,w0=-1.0))
    model=LensModel([Plane(redshift=0.49,mass=[NFW0,shear]),Plane(redshift=1.432,light=[src4,src5])],cosmo=cosmo)
    with fits.open(os.path.join(EXP,"newnewcutouts/source4-5.fits")) as h:
        oi=jnp.array(h['DATA'].data.astype("float64")); em=jnp.array(np.sqrt(h['STAT'].data.astype("float64")))
        psf=h['PSF'].data.astype(jnp.float64); mask=h['MASK'].data.astype(jnp.bool)
    cfg=SimulatorConfig(delta_pix=0.2,num_pix=300,supersample=1,kernel=psf,likelihood_precision="float64",conv_precision="float32")
    prob_model=ProbModel(model,[Dataset(oi,cfg,error_map=em,mask=mask,sees=[src4,src5])],mode="lstsq")
    return prob_model
if __name__=="__main__":
    pm=build()
    z=jnp.asarray(np.load(f"{EXP}/messy_tests/minimal_case/map/arrays.npz")['z_best']).reshape(-1)
    lp,rc=pm.log_prob(z[None,:])
    print("MAP logp",float(lp[0]),"red_chi2",float(rc[0]))
    t=time.time(); 
    for _ in range(3):
        lp,rc=pm.log_prob(z[None,:]); lp.block_until_ready()
    print("per single eval (s):",(time.time()-t)/3)
    # batch test
    zb=jnp.repeat(z[None,:],8,axis=0)
    t=time.time(); lp,rc=pm.log_prob(zb); lp.block_until_ready(); print("batch8 (s):",time.time()-t, "logp[0]",float(lp[0]))
