"""Cost of the multiplicity term inside log_prob on the current device.

Prints ms per jitted value_and_grad of the campaign-like ProbModel (positions +
fluxes + delays) WITH and WITHOUT the multiplicity term, at batch 8 (a MAMS
step's chains) and 1000 (a MAP step's particles). Run inside the container on
a GPU node; also runs on CPU.
"""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import ProbModel
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.point_source_position import PointSourceObsData
from gigalens.jax.point_source_multiplicity import PointSourceMultiplicityData
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.point_source import PointSourcePosition

print("devices:", jax.devices(), "jax", jax.__version__)
xs = np.array([0.4488, -0.4076, -1.1854, 1.1466]); ys = np.array([1.0185, -1.0216, 0.3591, -0.4417])
flux = np.array([15.1, 16.9, 16.9, 17.5]); td = np.array([0.0, 3.1, 11.0, 12.4])
src = Component(PointSourcePosition(absolute=True, with_amp=True),
                dict(center_x=tfd.Normal(0.0, 0.25), center_y=tfd.Normal(0.0, 0.25),
                     amp=tfd.LogNormal(np.log(2.0), 0.5)))
epl_c = Component(EPL(niter=18), dict(
    theta_E=tfd.LogNormal(np.log(1.25), 0.4), gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
    e1=tfd.Normal(0.0, 0.2), e2=tfd.Normal(0.0, 0.2),
    center_x=tfd.Normal(0.0, 0.06), center_y=tfd.Normal(0.0, 0.06)))
shr_c = Component(Shear(), dict(gamma1=tfd.Normal(0.0, 0.1), gamma2=tfd.Normal(0.0, 0.1)))
cosmo = Component(wCDM_Cosmo(z_lens=0.5, z_source_ref=1.5),
                  dict(H0=tfd.Uniform(20.0, 100.0), Om0=0.3, k=0.0, w0=-1.0))
model = LensModel([Plane(redshift=0.5, mass=[epl_c, shr_c]), Plane(redshift=1.5, light=[src])],
                  cosmo=cosmo)
pos = PointSourceObsData(src, xs, ys, 0.004, flux_obs=flux, sigma_flux=0.005 * flux,
                         td_obs=td, sigma_td=np.ones(3), src_anchor_sigma=0.004)
mult = PointSourceMultiplicityData(pos, flux_min=0.2)
pm0 = ProbModel(model, [pos]); pm1 = ProbModel(model, [pos, mult])
z = model.bijector.inverse(model.prior.sample(1000, seed=jax.random.PRNGKey(0)))
for label, pm in (("without term", pm0), ("with term", pm1)):
    f = jax.jit(jax.value_and_grad(lambda q: -jnp.mean(pm.log_prob(q)[0])))
    for n in (8, 1000):
        zz = z[:n]
        t0 = time.perf_counter(); v = f(zz); jax.block_until_ready(v); tc = time.perf_counter() - t0
        ts = []
        for k in range(5):
            t0 = time.perf_counter(); v = f(zz + 1e-4 * k); jax.block_until_ready(v); ts.append(time.perf_counter() - t0)
        print(f"{label:13s} batch {n:5d}: {1e3*np.median(ts):8.2f} ms/step  (compile {tc:.1f}s)")
