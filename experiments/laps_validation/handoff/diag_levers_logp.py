#!/usr/bin/env python
"""Verify the logp-separation claim (§6): core vs straggler logp for A0/A1.
Model build (no MAP/SVI; log_prob only) verbatim from diag_levers.py."""
import os, json
import jax
jax.config.update("jax_enable_x64", True)
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
import tensorflow_probability.substrates.jax as tfp
import numpy as np
from jax import numpy as jnp
tfd = tfp.distributions
HERE = os.path.dirname(os.path.abspath(__file__))
epl_p = dict(theta_E=tfd.LogNormal(jnp.log(1.25), 0.25), gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
    e1=tfd.Normal(0, 0.1), e2=tfd.Normal(0, 0.1), center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05))
shear_p = dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
lens_light_p = dict(R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15), n_sersic=tfd.Uniform(2, 6),
    e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3), e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
    center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05), Ie=tfd.LogNormal(jnp.log(500.0), 0.3))
source_light_p = dict(R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15), n_sersic=tfd.Uniform(0.5, 4),
    e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5), e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
    center_x=tfd.Normal(0, 0.25), center_y=tfd.Normal(0, 0.25), Ie=tfd.LogNormal(jnp.log(150.0), 0.5))
ASSETS = "/global/u1/l/linusu/gigalens/src/gigalens/assets"
kernel = np.load(f"{ASSETS}/psf.npy").astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=2, kernel=kernel)
model = LensModel([
    Plane(mass=[Component(epl.EPL(50), epl_p), Component(shear.Shear(), shear_p)],
          light=[Component(sersic.SersicEllipse(use_lstsq=False), lens_light_p)]),
    Plane(deflection_ratio=1.0, light=[Component(sersic.SersicEllipse(use_lstsq=False), source_light_p)])])
ds = Dataset(jnp.asarray(np.load(f"{ASSETS}/demo.npy")), sim_config, background_rms=0.2, exp_time=100, sees="all")
prob_model = ProbModel(model, ds, mode="forward")
DIM = int(model.num_free_params)
logp_v = jax.jit(jax.vmap(lambda z: prob_model.log_prob(z)[0]))
hmc = np.load(os.path.join(HERE, "hmc_ref", "hmc_mass.npy")); hm, hs = hmc.mean(0), hmc.std(0)
out = {}
for arm in ["A0_baseline", "A1_f1"]:
    z = np.load(os.path.join(HERE, "diag_levers", f"{arm}_samples_z.npy"))
    mass = np.load(os.path.join(HERE, "diag_levers", f"{arm}_mass.npy"))
    core = np.all(np.abs((mass - hm) / hs) < 6, axis=1)
    lp = np.asarray(logp_v(jnp.asarray(z)))
    q = lambda a: [float(x) for x in np.percentile(a, [5, 25, 50, 75, 95])]
    out[arm] = dict(n_core=int(core.sum()),
                    core_logp_q=q(lp[core]), strag_logp_q=q(lp[~core]),
                    core_logp_min=float(lp[core].min()), strag_logp_max=float(lp[~core].max()),
                    overlap_stragglers_above_core_min=int((lp[~core] > lp[core].min()).sum()))
    print(arm, json.dumps(out[arm]), flush=True)
json.dump(out, open(os.path.join(HERE, "diag_levers", "logp_separation.json"), "w"), indent=2)
print("DIAG DONE")
