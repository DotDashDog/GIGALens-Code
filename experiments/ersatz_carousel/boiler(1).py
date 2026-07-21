#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./src')
sys.path.append('/global/homes/s/seanjx/.conda/envs/gigajax5.0/lib/python3.12/site-packages')

import time
import json
from datetime import datetime
from copy import deepcopy

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.prob_model import ForwardProbModel, BackwardProbModel, ForwardMultiModel, BackwardMultiModel
from gigalens.model import PhysicalModelBase
from gigalens.jax.physical_model import PhysicalModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles import mass, light
from gigalens.multiband import SourcePlane, get_from_source_list

import tensorflow_probability.substrates.jax as tfp
import jax
from jax import random
import numpy as np
import optax
from jax import numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
import optax

from astropy.table import Table
from astropy.visualization import simple_norm

from corner import corner
tfd = tfp.distributions
tfe = tfp.experimental
    
numPix = 300
deltaPix = 0.2
exp_time = 9920 #7715.77464545289
extent = (-numPix/2*deltaPix, numPix/2*deltaPix, -numPix/2*deltaPix, numPix/2*deltaPix)   

def settings():
    return numPix, deltaPix, extent

# import photutils.psf as psf
# psf = psf.GaussianPSF(x_fwhm=0.717821594376891/deltaPix, y_fwhm=0.717821594376891/deltaPix)
# yy, xx = np.mgrid[-7:8, -7:8]
# return psf(xx, yy)

    
from gigalens.jax.cosmo import wCDM_Cosmo, w0waCDM_Cosmo
from gigalens.jax.prior import Prior, make_prior_and_model
with open('models/EvanNFW459.json', 'r') as file:
    best_model = json.load(file)
    
halo_model = Prior(
    #mass.epl.EPL(),
    mass.nfw.NFW_ELLIPSE_EINSTEIN(),
    # mass.bpl.BPL(),
    # mass.piemd.PIEMD(),
    # best_model['lens_mass']['0'] | 
    dict(
        center_x = tfd.Normal(6.69965551, 1),
        center_y = tfd.Normal(4.80431651, 1), 
        e1 = tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        Rs = tfd.Uniform(20, 100),
        theta_E = tfd.Uniform(12, 14),
        
        # # #gamma = tfd.TruncatedNormal(1.6, 0.5, 1 , 3),
        # gamma = tfd.Uniform(1,3)
        # b = tfd.Uniform(10,30), #tfd.TruncatedNormal(13, 2, 10, 20), #tfd.Normal(13, 1),
        # alpha = tfd.Uniform(1,3),
        # alpha_c = lambda alpha: tfd.Uniform(0, alpha),
        # r_core = lambda theta_E: tfd.TruncatedNormal(theta_E/2, theta_E/4, 0, theta_E)
    )
)


# fixed position, PIEMD
ld_free_ellip_model = Prior(
    mass.piemd.DPIE(),
    # mass.epl.EPL(),
    # best_model['lens_mass']['1'],
    dict(
        center_x = tfd.Normal(11.80977389, 0.1),
        center_y = tfd.Normal(23.0283886, 0.1),
        # theta_E = 1.6730331,
        # r_cut = 2.749177,
        # r_core = 0.2986106,
        # e1 = 0.41781854,
        # e2 = 0.07367268,
        theta_E = tfd.TruncatedNormal(1.6730331, 0.1, 1, 2.5),
        r_cut = tfd.LogNormal(jnp.log(10), 1),
        r_core = 0.05, #tfd.LogNormal(jnp.log(0.5), 0.1),
        # gamma = tfd.TruncatedNormal(2., 0.5, 1, 3),
        e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3), #lambda e1: tfd.TruncatedNormal(0, 0.05, -np.sqrt(0.09-e1**2), np.sqrt(0.09-e1**2)),
    ) #| best_model['lens_mass']['1']
)

shear_model = Prior(mass.shear.Shear(),
                    # best_model['lens_mass']['3'] | 
                    dict(
                        gamma1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                        gamma2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                    ))

le_free_model = Prior(mass.epl.EPL(),
                      # best_model['lens_mass']['1'] |
                   dict(
                       center_x = tfd.Normal(-21.17580938, 0.1),
                       center_y = tfd.Normal(-24.25810504, 0.1),
                       # e1 = 0.07383415,
                       # e2 = 0.03570823, 
                       e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       theta_E = tfd.TruncatedNormal(1.6711541, 0.1, 1, 2.5),
                       gamma = tfd.TruncatedNormal(2.007689, 0.5, 1, 3)
                   ) #| best_model['lens_mass']['3']
                     )

group_halo_free_model = Prior(#mass.epl.EPL(),
                              mass.piemd.DPIE(),
                              # best_model['lens_mass']['2'] |
                   dict(
                       center_x = tfd.Normal(-15.10088063, .1),
                       center_y = tfd.Normal(-4.66657821, .1),
                       # center_x = -15.10088063,
                       # center_y = -4.66657821,
                       # e1 = -0.00561034,
                       # e2 = -0.3605992,
                       e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       theta_E = tfd.TruncatedNormal(0.8151327, 0.05, 0.2, 1.5),
                       r_cut = tfd.LogNormal(jnp.log(10), 1),
                       r_core = 0.05, #tfd.LogNormal(jnp.log(0.5), 0.1),
                       # gamma = tfd.TruncatedNormal(2.2266, 0.5, 1, 3)
                   ) #| best_model['lens_mass']['4']
)

upper_right_halo = Prior(mass.epl.EPL(),
                   dict(
                       center_x = tfd.Normal(32, .5),
                       center_y = tfd.Normal(22, .5),
                       e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
                       theta_E = tfd.LogNormal(jnp.log(1),0.05),
                       gamma = tfd.TruncatedNormal(2, 0.5, 1, 3)
                   )
)

source1_prior = Prior(
    light.combined_profile.CombinedProfile(
        profiles=[
            light.sersic_shapelets.SersicShapelets(6, use_lstsq=True, interpolate=False), #10
            light.sersic.SersicEllipse(use_lstsq=True)
        ],
        shared_params=[],
        use_lstsq=True
    ),
    dict(
        # deflection_ratio = tfd.Uniform(0.5,1),
        z_source = 0.962,
        center_x_0 = tfd.Normal(7.67187389, 2),
        center_y_0 = tfd.Normal(3.31911655, 2),
        e1_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3), 
        n_sersic_0 = tfd.Uniform(1,10),
        R_sersic_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        beta_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),

        center_x_1 = tfd.Normal(0, 2),
        center_y_1 = tfd.Normal(0, 2),
        e1_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_1 = tfd.Uniform(1,10),
        R_sersic_1 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
    )
)

source3_prior = Prior(
    light.combined_profile.CombinedProfile(
        profiles=[
            light.sersic_shapelets.SersicShapelets(10, use_lstsq=True, interpolate=False), #15
            light.sersic.SersicEllipse(use_lstsq=True)
        ],
        shared_params=['center_x', 'center_y'],
        use_lstsq=True
    ),
    dict(
        # deflection_ratio = tfd.Uniform(0.5,1),
        z_source = 1.166,
        center_x = tfd.Normal(6.79821086, 2),
        center_y = tfd.Normal(7.91570776, 2),
        e1_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e1_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_0 = tfd.Uniform(1,10),
        R_sersic_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        n_sersic_1 = tfd.Uniform(1,10),
        R_sersic_1 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.05), 0.05),
        beta_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source45_prior = Prior(
    light.combined_profile.CombinedProfile(
        profiles=[
            # light.sersic.SersicEllipse(),
            light.sersic_shapelets.SersicShapelets(8, use_lstsq=True, interpolate=False), #12
            light.sersic.SersicEllipse(use_lstsq=True),
            # light.sersic_shapelets.SersicShapelets(4, use_lstsq=True, interpolate=True),
            # light.shapelets.Shapelets(4, use_lstsq=True, interpolate=False),
        ],
        shared_params=[],
        use_lstsq=True,
    ),
    # best_model['source_light']['0'] | 
    dict(
        # deflection_ratio = 1,
        z_source = 1.432,
        center_x_0 = tfd.Normal(4.63, 2),
        center_y_0 = tfd.Normal(3.79, 2),
        center_x_1 = tfd.Normal(4.78792923, 2),
        center_y_1 = tfd.Normal(0.84347007, 2),
        e1_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        
        e1_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_0 = tfd.Uniform(1,10),
        n_sersic_1 = tfd.Uniform(1,10),
        R_sersic_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        R_sersic_1 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        beta_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # beta_1 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie_0 = tfd.LogNormal(jnp.log(40), 1),
        # Ie_1 = tfd.LogNormal(jnp.log(40), 1),
    )
)
# source 9
source9_prior = Prior(
    # light.sersic_shapelets.SersicShapelets(4, use_lstsq=True),
    light.sersic.SersicEllipse(use_lstsq=True),
    # best_model['source_light']['1'] | 
    dict(
        # deflection_ratio = tfd.Uniform(0.75, 1.25),
        z_source = 1.506,
        center_x = tfd.Normal(-7, 2),
        center_y = tfd.Normal(-13, 2),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3), 
        n_sersic = tfd.Uniform(1,10),
        R_sersic = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        
        # beta = tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
        # point source
# source9_prior = Prior(
#     light.point_source.PointSource(use_lstsq=True),
#     dict(
#         # deflection_ratio = cosmo_model.deflection_ratio(1.506, H0=70, Om0=0.3, k=0, w0=-1)
#         z_source = 1.506
#     )
# )
source7_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True),
    dict(
        # deflection_ratio = tfd.Uniform(1.,1.5),
        z_source = 1.628,
        center_x = tfd.Normal(0, 2),
        center_y = tfd.Normal(0, 2),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(1,10),
        R_sersic = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source6_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True),
    dict(
        # deflection_ratio = tfd.Uniform(1.,1.5),
        z_source = 1.656,
        center_x = tfd.Normal(0, 2),
        center_y = tfd.Normal(0, 2),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(1,10),
        R_sersic = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source1213_prior = Prior(
    light.combined_profile.CombinedProfile(
        profiles=[light.sersic_shapelets.SersicShapelets(6, use_lstsq=True, interpolate=False), #8
                  # light.sersic.SersicEllipse(use_lstsq=True),
                  light.sersic.SersicEllipse(use_lstsq=True),],
        shared_params=[],
        use_lstsq=True,
    ),
    dict(
        # deflection_ratio = 1.2674489,
        # deflection_ratio = tfd.Uniform(1,1.5),
        z_source = 3.086,
        center_x_0 = tfd.Normal(2.906817674636841, 2),
        center_y_0 = tfd.Normal(4.523301601409912, 2),
        center_x_1 = tfd.Normal(6.670745849609375, 2),
        center_y_1 = tfd.Normal(5.6769537925720215, 2),
        e1_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_0 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e1_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2_1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic_0 = tfd.Uniform(1,10),
        n_sersic_1 = tfd.Uniform(1,10),
        R_sersic_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        R_sersic_1 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        beta_0 = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie_0 = tfd.LogNormal(jnp.log(40), 1),
        # Ie_1 = tfd.LogNormal(jnp.log(40), 1),
    )
)
source8_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True),
    # light.sersic_shapelets.SersicShapelets(6, use_lstsq=True),
    dict(
        # deflection_ratio = tfd.Uniform(1,1.5),
        z_source = 3.549,
        center_x = tfd.Normal(6.1898108, 2),
        center_y = tfd.Normal(6.8792906, 2),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(1,10),
        R_sersic = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # beta = tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)
source11_prior = Prior(
    light.sersic.SersicEllipse(use_lstsq=True),
    dict(
        # deflection_ratio = tfd.Uniform(1.,1.5),
        z_source = 4.090,
        center_x = tfd.Normal(4.4566746, 2),
        center_y = tfd.Normal(2.0770147, 2),
        e1 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2 = tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic = tfd.Uniform(1,10),
        R_sersic = tfd.Uniform(1e-3,1),#tfd.LogNormal(jnp.log(0.4), 0.15),
        # Ie = tfd.LogNormal(jnp.log(40), 1),
    )
)

cosmo_model = w0waCDM_Cosmo(0.49, 1.432)
cosmo_prior = Prior(
        cosmo_model,  # you need to set the redshifts for the cosmology to work, theta_E is relative to z_source_ref
    dict(
        H0=70.,
        Om0=tfd.Uniform(0,1),
        w0=tfd.Uniform(-2,1/3),
        # wa=tfd.Uniform(-3.0, 1.),
        wa=0.0,
        k=0.0,
    )
)

source1 = SourcePlane(prior=source1_prior, path='model_data/muse/cutouts/source1.fits')
source3 = SourcePlane(prior=source3_prior, path='model_data/muse/cutouts/source3.fits')
source45 = SourcePlane(prior=source45_prior, path='model_data/muse/cutouts/source4-5.fits')
source6 = SourcePlane(prior=source6_prior, path='model_data/muse/cutouts/source6.fits')
source7 = SourcePlane(prior=source7_prior, path='model_data/muse/cutouts/source7.fits')
source8 = SourcePlane(prior=source8_prior, path='model_data/muse/cutouts/source8.fits')
source9 = SourcePlane(prior=source9_prior, path='model_data/muse/cutouts/source9.fits')
source11 = SourcePlane(prior=source11_prior, path='model_data/muse/cutouts/source11.fits')
source1213 = SourcePlane(prior=source1213_prior, path='model_data/muse/cutouts/source12-13.fits')

names = ('1', '3', '4-5', '6', '7', '8', '9', '11', '12-13')
sources = [source1, source3, source45, source6, source7, source8, source9, source11, source1213]
for name, source in zip(names, sources):
    source.observed_image = np.load(f'model_data/simulated_images/simulated{name}.npy')

# mask11 = np.ones((300,300)).astype('bool')
# mask11[150:250,150:250] = 0
# source11.mask &= mask11

# mask7 = np.ones((300,300)).astype('bool')
# mask7[150:250,150:250] = 0
# source7.mask &= mask7

sources = [source1, source3, source45, source6, source7, source8, source9, source11, source1213]

def prior_from_sources(source_priors):
    prior, phys_model = make_prior_and_model(
        lenses=[
            halo_model,
            ld_free_ellip_model,
            shear_model,
            le_free_model,
            group_halo_free_model,
            # upper_right_halo
        ],
        sources=source_priors,
        foreground=[],
        cosmo=cosmo_prior
    )
    return prior, phys_model

def translate(x, y):
    vector = np.array([x,y])
    angle = 54.49
    rotation = np.array([[np.cos(angle*np.pi/180), -np.sin(angle*np.pi/180)],[np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])
    return rotation @ (vector - np.array([0, 0.5])) + np.array([4.5,4])
    
def etranslate(e1, e2):
    vector = np.array([e1, e2])
    angle = 55*2
    rotation = np.array([[np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)],[-np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])
    return rotation @ vector
    
def convertEllipticity(e1, e2):
    phi = jnp.arctan2(e2, e1) / 2
    c = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
    q = (1 - c) / (1 + c)
    return float(q), float(phi)

def num_amps_shapelets(n):
     return (n + 1) * (n + 2) // 2
    
def load_result(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, list):
            new_d[k] = jnp.squeeze(jnp.array(v))
        # elif isinstance(v, float):
        #     new_d[k] = jnp.array([v])
        elif isinstance(v, dict):
            new_d[k] = load_result(v)
        else:
            new_d[k] = v
    return new_d

def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1.
    Values from dict2 will overwrite values in dict1.
    If both values are dictionaries, it merges them recursively.

    AI GENERATED >:(
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            # If both values are dicts, merge them recursively
            deep_merge(dict1[key], value)
        else:
            # Otherwise, use the value from dict2 (overwrites existing or adds new)
            dict1[key] = value
    return dict1
    
from functools import reduce
def add_model_scatter(best_model, prior=None, num=100, scatter=0.1, seed=0):
    dist = tfd.Normal(0, scatter)
    if prior is None:
        return jax.tree.map_with_path(lambda path, _: jnp.repeat(jnp.array(reduce(lambda x,y: x[y.key], path, best_model)), num) + dist.sample(num,jax.random.key(seed)), best_model)
    else:
        return jax.tree.map_with_path(lambda path, _: jnp.repeat(jnp.array(reduce(lambda x,y: x[y.key], path, best_model)), num) + dist.sample(num,jax.random.key(seed)), prior.sample(num, jax.random.key(0)))

        
def multiband_source_simulate(simulator: LensSimulator, params):
    no_mass_model = deepcopy(simulator.phys_model)
    no_mass_model.lenses = []
    no_mass_simulator = LensSimulator(no_mass_model, simulator.sim_config, 1)
    sourcesimulateds = no_mass_simulator.multiband_simulate({'lens_mass':{}, 'lens_light':{}, 'source_light': params.get('source_light', {}), 'cosmo': params.get('cosmo', {})})
    return sourcesimulateds

def multiband_lstsq_source_simulate(simulator: LensSimulator, params, prob_model):
    sourcesimulateds = []
    coeffs_list = simulator.multiband_lstsq_simulate(params, prob_model.observed_images, prob_model.error_maps, prob_model.masks, return_coeffs=True)
    
    no_mass_model = deepcopy(simulator.phys_model)
    no_mass_model.lenses = []
    no_mass_simulator = LensSimulator(no_mass_model, simulator.sim_config, simulator.bs)
    stacked = no_mass_simulator.multiband_lstsq_simulate({'lens_mass':{}, 
                                                           'lens_light':{}, 
                                                           'source_light': params.get('source_light', {}),
                                                           'cosmo': params.get('cosmo', {})
                                                          }, 
                                                          prob_model.observed_images, 
                                                          prob_model.error_maps, 
                                                          prob_model.masks,
                                                          return_stacked=True
                                                          )
    counter = 0
    source_list = [lightModel for lightModel in no_mass_model.source_light if lightModel.depth !=0]
    for coeffs, lightModel in zip(coeffs_list, source_list):
        # if lightModel.depth == 0: continue
        new_counter = counter + lightModel.depth
        ret = stacked[...,counter:new_counter]
        counter = new_counter
        
        sourcesimulateds.append(jnp.squeeze(jnp.sum(ret * coeffs[:, jnp.newaxis, jnp.newaxis, :], axis=-1)))
    return sourcesimulateds

def _critical_and_caustic_curves(
    lens_params,
    simulator,
    supersample: int = 20,
    deflection_ratio: float = 1.0,
):
    """Compute (critical, caustic) curves natively from the gigalens lens model.

    For a source plane with deflection ratio ``r`` the lens map is
    ``beta = theta - r * alpha(theta)`` with Jacobian ``A = I - r * d alpha/d theta``.
    Critical curves are the ``det(A) = 0`` contours (image plane); each is mapped
    through the lens equation to its caustic (source plane). Returns matched lists
    of ``(x_array, y_array)`` polylines, one entry per disjoint segment.

    At ``deflection_ratio=1`` this reduces exactly to lenstronomy's
    ``critical_curve_caustics`` (``det(A) = 1/magnification``,
    ``beta = theta - alpha``) — the parity check used to certify the native path.
    """
    if simulator.bs != 1: raise Exception(f'Batch size bs must be 1! Currently {simulator.bs}.')
        
    from skimage.measure import find_contours
    dr = float(deflection_ratio)

    numpify = lambda x: np.squeeze(np.array(x))
    
    img_X, img_Y = np.array(simulator.img_X), np.array(simulator.img_Y)
    f_xx, f_xy, f_yx, f_yy = np.array(simulator.hessian(img_X, img_Y, lens_params,))*dr
    det_A = np.squeeze((1 - f_xx) * (1 - f_yy) - f_xy * f_yx)

    critical, caustics = [], []
    for v in find_contours(det_A, 0.0):
        dec, ra = simulator.wcs.pix2angle(*v.T)
        critical.append((ra, dec))
        ra_c, dec_c = [numpify(coord) for coord in simulator.beta(ra, dec, lens_params, dr)]
        caustics.append((ra_c, dec_c))
    return critical, caustics
    
def plot_caustics_critical(
    ax,
    lens_params,
    simulator,
    color_critical: str = "red",
    color_caustic: str = "green",
    supersample: int = 20,
    deflection_ratio: float = 1.0,
) -> None:
    """Overlay both caustic *and* critical curves on a single ``ax``.

    Kept for convenience, but note the two curves live in *different* planes
    (caustics: source; critical: image). The reports draw them separately —
    critical on the image/lens panel, caustics on the source panel. Prefer
    :func:`plot_caustics` / :func:`plot_critical_curves` for a single plane.
    """
    critical, caustics = _critical_and_caustic_curves(
        lens_params, simulator, supersample=supersample,
        deflection_ratio=deflection_ratio,
    )
    for xs, ys in critical:
        ax.plot(xs, ys, color=color_critical, linewidth=1.5)
    for xs, ys in caustics:
        ax.plot(xs, ys, color=color_caustic, linewidth=1.5)