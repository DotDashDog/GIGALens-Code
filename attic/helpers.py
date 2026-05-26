# """
# Helper functions for GigaLens inference and visualization.

# This module provides utility functions for:
# 1. Running inference pipelines (MAP, SVI, HMC) on gravitational lensing systems
# 2. Visualizing results through various plots:
#    - Image comparisons (true vs predicted)
#    - Residual analysis and Gaussianity tests
#    - Loss histories for optimization
#    - Corner plots for parameter distributions
# 3. Computing diagnostics like chi-squared statistics and noise maps
# 4. Parameter manipulation and indexing

# The functions here streamline the workflow of fitting lens models and analyzing their results.


# NOTES FOR ELDEN:
# - The precision_parameterization parameter I use for SVI means that gradient descent is done on the precision matrix rather than the covariance matrix.
#     It's implemented in my inference.py file
#     I'll ask about whether we should use it at the next meeting. If we decide not to, just don't pass it to SVI
# - I have my ModellingSequence objects named ModellingSequence (the default GIGALens one) and HarryModellingSequence (Harry's multinode one). 
#     If you do things differently, you'll have to change the references to the ModellingSequence class names in the results objects
# - The way residualplot_params (used for hundred systems) works is very dependent on the way I store the modelling results. So probably don't use it for now.
# """

# from jax import numpy as jnp
# import numpy as np
# import matplotlib.pyplot as plt
# import optax
# # from gigalens.jax.inference import ModellingSequence
# from gigalens.jax.model import ForwardProbModel, BackwardProbModel
# from gigalens.jax.simulator import LensSimulator
# import jax
# import tensorflow_probability.substrates.jax as tfp
# tfd = tfp.distributions

# from scipy.stats import norm, kstest
# import corner
# import matplotlib
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import time
# import os
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D

# import lenstronomy.Util.simulation_util as sim_util
# from lenstronomy.LensModel.lens_model import LensModel
# from lenstronomy.Plots import lens_plot
# from lenstronomy.Data.imaging_data import ImageData

# def params_jax_to_lists(params_jax):
#     """Convert nested parameter structure of JAX arrays to nested Python lists"""
#     # params_list = []
#     # for i in range(len(params_jax)):
#     #     params_list.append([])
#     #     for j in range(len(params_jax[i])):
#     #         params_list[i].append({})
#     #         for key in params_jax[i][j]:
#     #             params_list[i][j][key] = params_jax[i][j][key].tolist()
#     # return params_list
#     return jax.tree.map(lambda a : list(a), params_jax)

# def params_lists_to_jax(params_list):
#     """Convert nested parameter structure of Python lists back to JAX arrays"""
#     params = []
#     for i in range(len(params_list)):
#         params.append([])
#         for j in range(len(params_list[i])):
#             params[i].append({})
#             for key in params_list[i][j]:
#                 params[i][j][key] = jnp.array(params_list[i][j][key])
#     return params
#     # return jax.tree.map(lambda a : jnp.array(a), params_list, is_leaf=lambda x : isinstance(x, list) and isinstance(x[0], int))

# def index_params(params, i):
#     #* gets the params just for the ith system
            
#     return jax.tree.map(lambda a : a[i], params)
