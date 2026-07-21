import jax
import os
import time
from datetime import datetime

import numpy as np
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

# --- new (scene) API ---------------------------------------------------------
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.nfw_ellipse_slope import NFW_ELLIPSE_SLOPE
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.cosmo import w0waCDM_Cosmo
from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.inference import ModellingSequence

# --- research-side pipeline / diagnostics (gigalens_research) ----------------
from gigalens_research.inference_utils import (
    InferenceContext, Pipeline, MAPStage, BridgeStage, MCLMCStage,
)
from gigalens_research.plotting import PosteriorReport, PipelineReport
import photutils.psf as psf


def tNCDF_bij(low, high):
    return tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.NormalCDF()])


# Verbatim from the old notebook: a tfd.Uniform with a Shift/Scale/NormalCDF
# event-space bijector in place of TFP's default (sigmoid-based) one for Uniform.
class UniformBij(tfd.Uniform):
    def __init__(self, *args, event_space_bijector_class=tNCDF_bij, **kwargs):
        self._esb = event_space_bijector_class(*args)
        super().__init__(*args, **kwargs)

    def _default_event_space_bijector(self):
        return self._esb

def make_5spl_bullseye(sample_wa):
    z_lens = 0.5
    z_source1 = 1.0
    z_source2 = 1.5
    z_source3 = 4.0
    z_source4 = 7.0
    z_source5 = 15.0

    s = 13.2/0.9

    s2 = s**2
    
    lens = Component(
        NFW_ELLIPSE_SLOPE(),
        dict(
            theta_E=tfd.LogNormal(jnp.log(1.25*s), 0.25),
            s_E = tfd.Uniform(0,0.75),# gamma=tfd.TruncatedNormal(2, 0.4, 1, 3),
            e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            center_x=tfd.Normal(0, 0.05*s),
            center_y=tfd.Normal(0, 0.05*s),
        ),
    )
    # NOTE: the old notebook also drafted a `shear` Prior/Component (mass.shear.Shear())
    # but commented it out of the model it actually built and ran; omitted here too.
    def sourceprior_dict():
        return dict(
                center_x=tfd.Normal(0*s, 2*s),
                center_y=tfd.Normal(0*s, 2*s),
                e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                n_sersic=tfd.Uniform(1, 10),
                R_sersic=tfd.LogNormal(jnp.log(1.*s), 0.15),
                Ie=tfd.LogNormal(jnp.log(150/s2), 1),
            )
    
    source1 = Component(
        SersicEllipse(use_lstsq=False),
        sourceprior_dict(),
    )
    source2 = Component(
        SersicEllipse(use_lstsq=False),
        sourceprior_dict(),
    )
    
    
    source3 = Component(
        SersicEllipse(use_lstsq=False),
        sourceprior_dict(),
    )
    
    source4 = Component(
        SersicEllipse(use_lstsq=False),
        sourceprior_dict(),
    )
    
    source5 = Component(
        SersicEllipse(use_lstsq=False),
        sourceprior_dict(),
    )
    
    
    def tNCDF_bij(low, high):
        return tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.NormalCDF()])
    
    
    # Verbatim from the old notebook: a tfd.Uniform with a Shift/Scale/NormalCDF
    # event-space bijector in place of TFP's default (sigmoid-based) one for Uniform.
    class UniformBij(tfd.Uniform):
        def __init__(self, *args, event_space_bijector_class=tNCDF_bij, **kwargs):
            self._esb = event_space_bijector_class(*args)
            super().__init__(*args, **kwargs)
    
        def _default_event_space_bijector(self):
            return self._esb
    
    from gigalens_research.priors.ratio_pair_coords import RatioPairUniform,deflection_ratio_pair_fn
    
    cos = w0waCDM_Cosmo(z_lens, z_source2)
    cosmo = Component(cos, {
                "H0": 70.0, "k": 0.0, "wa": 0.0, 
                "Om0":UniformBij(jnp.float64(0.0), jnp.float64(1.0)),
                "w0":UniformBij(jnp.float64(-2.0), jnp.float64(-1 / 3)),
                "wa": UniformBij(jnp.float64(-3.0), jnp.float64(2.0)) if sample_wa else 0.0,
                # ("Om0", "w0"): RatioPairUniform(
                #     deflection_ratio_pair_fn(cos, (z_source2, z_source3),
                #                              fixed=dict(H0=70.0, k=0.0, wa=0.0)),
                #     (0.044, 0.994), (-2.0, -1.0 / 3.0)),
            })
    
    
    # "wa": 
    
    model = LensModel(
        [
            Plane(redshift=z_lens, mass=[lens]),
            Plane(redshift=z_source1, light=[source1]),
            Plane(redshift=z_source2, light=[source2]),
            Plane(redshift=z_source3, light=[source3]),
            Plane(redshift=z_source4, light=[source4]),
            Plane(redshift=z_source5, light=[source5]),
        ],
        cosmo=cosmo,
    )
    
    print(f"num_free_params = {model.num_free_params}")
    print("z_param_names   =", model.z_param_names)

    Ie = 15.
    truth_scene = {
        "planes": {
            0: {
                "geometry": {"redshift": z_lens},
                "mass": {
                    0: {"theta_E": 0.9*s, "s_E":0.4, "e1": 0.05, "e2": 0.02, # "gamma": 1.6, 
                        "center_x": 0.0*s, "center_y": 0.0*s},
                },
            },
            1: {
                "geometry": {"redshift": z_source1},
                "light": {
                    0: {"R_sersic": 1.*s, "n_sersic": 2., "e1": 0.05, "e2": 0.,
                        "center_x": 0.15*s, "center_y": 0.*s, "Ie": Ie/s2},
                },
            },
            2: {
                "geometry": {"redshift": z_source2},
                "light": {
                    0: {"R_sersic": 1.*s, "n_sersic": 2., "e1": 0.0, "e2": 0.05,
                        "center_x": 0.*s, "center_y": 0.01*s, "Ie": Ie/s2},
                },
            },
            3: {
                "geometry": {"redshift": z_source3},
                "light": {
                    0: {"R_sersic": 1.*s, "n_sersic": 2., "e1": 0.0, "e2": 0.05,
                        "center_x": 0.*s, "center_y": 0.05*s, "Ie": Ie/s2},
                },
            },
            4: {
                "geometry": {"redshift": z_source4},
                "light": {
                    0: {"R_sersic": 1.*s, "n_sersic": 2., "e1": 0.0, "e2": 0.05,
                        "center_x": 0.05*s, "center_y": 0.05*s, "Ie": Ie/s2},
                },
            },
            5: {
                "geometry": {"redshift": z_source5},
                "light": {
                    0: {"R_sersic": 1.*s, "n_sersic": 2., "e1": 0.0, "e2": 0.05,
                        "center_x": 0.1*s, "center_y": 0.0*s, "Ie": Ie/s2},
                },
            },
        },
        "cosmo": dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0, wa=0.0),
    }



    numPix, deltaPix, exp_time, background_rms = 80, 0.065*s, 1000, 0.1
    extent = (-numPix / 2 * deltaPix, numPix / 2 * deltaPix,
              -numPix / 2 * deltaPix, numPix / 2 * deltaPix)
    
    kernel = psf.GaussianPSF(x_fwhm=2, y_fwhm=2)
    yy, xx = np.mgrid[-7:8, -7:8]
    kernel = kernel(xx, yy)
    
    sim_config = SimulatorConfig(
        delta_pix=deltaPix,
        num_pix=numPix,
        kernel=kernel,
        supersample=1,
        likelihood_precision="float64",
    )
    
    sim_config_truth = SimulatorConfig(
        delta_pix=deltaPix,
        num_pix=numPix,
        kernel=kernel,
        supersample=1,
        likelihood_precision="float64",
    )


    from lenstronomy.Util import image_util


    def add_noise(img, exp_time, sigma_bkd):
        poisson = image_util.add_poisson(img, exp_time=exp_time)
        bkg = image_util.add_background(img, sigma_bkd=sigma_bkd)
        return img + poisson + bkg

    sim1 = SceneSimulator(model, sim_config_truth, sees=[source1])
    sim2 = SceneSimulator(model, sim_config_truth, sees=[source2])
    sim3 = SceneSimulator(model, sim_config_truth, sees=[source3])
    sim4 = SceneSimulator(model, sim_config_truth, sees=[source4])
    sim5 = SceneSimulator(model, sim_config_truth, sees=[source5])
    
    print("trace mode (both simulators, same single-mass-plane model):",
      sim1.trace_mode, sim2.trace_mode)
    
    img1 = np.asarray(sim1.simulate(truth_scene))
    img2 = np.asarray(sim2.simulate(truth_scene))
    img3 = np.asarray(sim3.simulate(truth_scene))
    img4 = np.asarray(sim4.simulate(truth_scene))
    img5 = np.asarray(sim5.simulate(truth_scene))
    
    observed_image1 = add_noise(img1, exp_time=exp_time, sigma_bkd=background_rms)
    observed_image2 = add_noise(img2, exp_time=exp_time, sigma_bkd=background_rms)
    observed_image3 = add_noise(img3, exp_time=exp_time, sigma_bkd=background_rms)
    observed_image4 = add_noise(img4, exp_time=exp_time, sigma_bkd=background_rms)
    observed_image5 = add_noise(img5, exp_time=exp_time, sigma_bkd=background_rms)
    
    plt.figure(figsize=(16, 3))
    ax = plt.subplot(151)
    plt.imshow(observed_image1, extent=extent)
    plt.colorbar()
    ax = plt.subplot(152)
    plt.imshow(observed_image2, extent=extent)
    plt.colorbar()
    ax = plt.subplot(153)
    plt.imshow(observed_image3, extent=extent)
    plt.colorbar()
    ax = plt.subplot(154)
    plt.imshow(observed_image4, extent=extent)
    plt.colorbar()
    ax = plt.subplot(155)
    plt.imshow(observed_image5, extent=extent)
    plt.colorbar()
    plt.show()
    
    dataset1 = ImageData(observed_image1, sim_config, background_rms=background_rms,
                    exp_time=exp_time, sees=[source1])
    dataset2 = ImageData(observed_image2, sim_config, background_rms=background_rms,
                        exp_time=exp_time, sees=[source2])
    dataset3 = ImageData(observed_image3, sim_config, background_rms=background_rms,
                        exp_time=exp_time, sees=[source3])
    dataset4 = ImageData(observed_image4, sim_config, background_rms=background_rms,
                        exp_time=exp_time, sees=[source4])
    dataset5 = ImageData(observed_image5, sim_config, background_rms=background_rms,
                        exp_time=exp_time, sees=[source5])
    
    prob_model = ProbModel(model, [dataset1, dataset2, dataset3, dataset4, dataset5], mode="forward")
    model_seq = ModellingSequence(prob_model)
    ctx = InferenceContext.from_modelling_sequence(model_seq)


    import copy
    model_def_ratio = LensModel(
        [
            Plane(mass=[lens]),
            Plane(deflection_ratio=tfd.Uniform(0.5,1), light=[source1]),
            Plane(deflection_ratio=1.0, light=[source2]),
            Plane(deflection_ratio=tfd.Uniform(1.0,1.7), light=[source3]),
            Plane(deflection_ratio=tfd.Uniform(1.0,1.7), light=[source4]),
            Plane(deflection_ratio=tfd.Uniform(1.0,1.7), light=[source5]),
        ],
    )
    
    tc = copy.deepcopy(truth_scene)
    cos_truth = tc.pop("cosmo")
    for k in truth_scene["planes"]:
        tc["planes"][k]["geometry"]["deflection_ratio"] = float(jnp.squeeze(cos.deflection_ratio(tc["planes"][k]["geometry"].pop("redshift"), **cos_truth)))
    truth_def_ratio = tc
    
    prob_model_dr = ProbModel(model_def_ratio, [dataset1, dataset2, dataset3, dataset4,dataset5], mode="forward")
    model_seq_dr = ModellingSequence(prob_model_dr)
    ctx_dr = InferenceContext.from_modelling_sequence(model_seq_dr)


    return prob_model, prob_model_dr, truth_scene, truth_def_ratio


def standard_pipeline(ctx_in, seed=42):

    pipeline = Pipeline(ctx_in, seed=seed)
    pipeline.add(MAPStage(
        num_steps=4000,
        n_samples=1000,
        pbar_interval=100,
        seed=1,
    ))
    
    def make_diag_qz(z_best):
        return tfd.MultivariateNormalDiag(
            loc=jnp.asarray(z_best),
            scale_diag=jnp.full(z_best.shape[-1], 1e-3),
        )
    
    
    pipeline.add(BridgeStage(
        name="diag_qz_from_map",
        version="v1",
        requires=("z_best",),
        produces=("qz",),
        fn=make_diag_qz,
    ))
    
    pipeline.add(MCLMCStage(
        n_chains=8,
        num_burnin_steps=10000,
        num_results=10000,
        desired_energy_variance=5e-4,
        seed=10,
        progress_bar=True,
        debug=True,
    ))
    return pipeline



#* PLOTTING STUFF

cosmo_prior_wCDM = tfd.JointDistributionNamed(
    dict(
        Om0=tfd.Uniform(0.0, 1.0),
        w0=tfd.Uniform(-2.0, -1/3),
    )
)
cosmo_prior_waCDM = tfd.JointDistributionNamed(
    dict(
        Om0=tfd.Uniform(0.0, 1.0),
        w0=tfd.Uniform(-2.0, -1/3),
        wa= UniformBij(jnp.float64(-3.0), jnp.float64(2.0)),
    )
)


def lens_cosmo_loglike(Om0, w0, wa, z_source,
                    deflect_ratio_obs, deflect_ratio_cov_det, deflect_ratio_cov_inv,
                    cosmo):
    def mapped_dr(Om0, w0, wa):
        return cosmo.deflection_ratio(z_source, Om0=Om0, w0=w0, wa=wa, H0=70., k=0.0)
    deflect_ratio = mapped_dr(Om0, w0, wa)
    delta = deflect_ratio - deflect_ratio_obs
    chi2 = delta * deflect_ratio_cov_inv.dot(delta)
    normalization = jnp.log(2 * jnp.pi * deflect_ratio_cov_det)
    log_like = -1 / 2 * (chi2 + normalization)
    return jnp.sum(log_like, axis=0)  # sum over redshifts

# log_prob
def lens_cosmo_logprob(state, observation, cosmo, cosmo_prior):
    log_prior = cosmo_prior.log_prob(state)
    log_like = lens_cosmo_loglike(cosmo=cosmo, **state, **observation)
    return log_like + log_prior


def logsumexp_norm(log_prob):
    c = jnp.max(log_prob)
    logsumexp = c + jnp.log(jnp.nansum(jnp.exp(log_prob - c)))
    return log_prob - logsumexp

    
def make_prob_grid(observation, cosmo, sample_wa=False, plane_indexes=None, n_grid=400):
    Om0_grid, w0_grid = jnp.linspace(0, 1, n_grid), jnp.linspace(-2, -1/3, n_grid),

    if plane_indexes is None:
        plane_indexes = np.array(range(len(observation['z_source'])))
        # print(plane_indexes)
    

    if sample_wa:
        wa_grid = jnp.linspace(-3, 2.0, n_grid)
        Om0_grid, w0_grid, wa_grid = jnp.meshgrid(Om0_grid, w0_grid, wa_grid, indexing='ij')
        states = {
            'Om0': Om0_grid.flatten(), 
            'w0': w0_grid.flatten(),
            'wa': wa_grid.flatten()
        }
    else:
        Om0_grid, w0_grid = jnp.meshgrid(Om0_grid, w0_grid, indexing='ij')
        states = {
                'Om0': Om0_grid.flatten(), 
                'w0': w0_grid.flatten(),
                'wa': 0.0
            }

    selected_med = observation['deflect_ratio_obs'][plane_indexes]
    selected_cov = observation['deflect_ratio_cov'][plane_indexes][:, plane_indexes]
    # print(selected_cov)
    obs_p = {}
    obs_p["deflect_ratio_obs"] = selected_med
    obs_p["deflect_ratio_cov_inv"] = jnp.linalg.inv(selected_cov)
    obs_p["deflect_ratio_cov_det"] = jnp.abs(jnp.linalg.det(selected_cov))
    obs_p["z_source"] = observation["z_source"][plane_indexes]
    
    log_prob = lens_cosmo_logprob(states, obs_p, cosmo=cosmo, cosmo_prior=cosmo_prior_waCDM if sample_wa else cosmo_prior_wCDM)
    log_prob = log_prob.reshape(Om0_grid.shape)
    log_prob = np.array(log_prob).astype(np.float64)
    log_prob = logsumexp_norm(log_prob)
    prob = np.exp(log_prob)
    if sample_wa:
        return prob, (np.array(Om0_grid), np.array(w0_grid), np.array(wa_grid))
    else: 
        return prob, (np.array(Om0_grid), np.array(w0_grid))