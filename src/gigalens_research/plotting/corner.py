import corner
import jax
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

def cornerplot_labels(example_params, latex=False):
    """
    Generate the labels for the cornerplot based on the tree structure of the parameters.
    """
    
    return list(flatten_params_to_labeled_dict(example_params).keys())

def latex_label(label):

    latex_label_map = {
        # Mass
        "theta_E": r"$\theta_E$",
        "gamma": r"$\gamma_{epl}$",
        "e1": r"$\epsilon_{epl,1}$",
        "e2": r"$\epsilon_{epl,2}$",
        "center_x": r"$x_{epl}$",
        "center_y": r"$y_{epl}$",
        "gamma1": r"$\gamma_{ext,1}$",
        "gamma2": r"$\gamma_{ext,2}$",

        # Lens Light
        "lens_R_sersic": r"$R_{l}$",
        "lens_n_sersic": r"$n_{l}$",
        "lens_e1": r"$\epsilon_{l,1}$",
        "lens_e2": r"$\epsilon_{l,2}$",
        "lens_center_x": r"$x_{l}$",
        "lens_center_y": r"$y_{l}$",
        "lens_Ie": r"$I_l$",

        # Source Light
        "src_R_sersic": r"$R_{s}$",
        "src_n_sersic": r"$n_{s}$",
        "src_e1": r"$\epsilon_{s,1}$",
        "src_e2": r"$\epsilon_{s,2}$",
        "src_center_x": r"$x_{s}$",
        "src_center_y": r"$y_{s}$",
        "src_Ie": r"$I_s$",
    }

    return latex_label_map[label]

def flatten_params_to_labeled_dict_sim(params):
    tups = [(0, 0), (0, 1), (1, 0), (2, 0)]
    label_prefixes = ['', '', 'lens_', 'src_']

    flat_dict = {}
    for (i, j), label_prefix in zip(tups, label_prefixes):
        flat_dict.update({label_prefix + key: params[i][j][key] for key in params[i][j].keys()})
    return flat_dict

def flatten_params_to_labeled_dict(params):
    tups = []
    prefix = ['mass', 'lens', 'src']
    label_prefixes = []
    for i in range(len(params)):
        for j in range(len(params[i])):
            tups.append((i, j))
            label_prefixes.append(f"{prefix[i]}_{str(j)}")

    flat_dict = {}
    for (i, j), label_prefix in zip(tups, label_prefixes):
        flat_dict.update({label_prefix + key: params[i][j][key] for key in params[i][j].keys()})
    return flat_dict

def cornerplot_posterior(raw_samples, fig=None, truth=None, overplots=None, color='black', truth_color='black', overplot_color='red', plot_params=None, latex=False):
    """
    Create a cornerplot of a set of samples in the physical space.
    Option to overplot a single point, such as the MAP best fit
    Can also overplot a second point as crossed vertical and horizontal lines (most often the truth or median of the samples)
    """
    flat_samples = flatten_params_to_labeled_dict(raw_samples)
    if plot_params is None:
        plot_params = list(flat_samples.keys())
        # flat_samples = {k:flat_samples[k] for k in plot_params}
        

    if overplots is not None:
        flat_overplots = flatten_params_to_labeled_dict(overplots)
            #flat_overplots = {k:flat_overplots[k] for k in plot_params}
        overplot_pts = np.squeeze(np.stack([flat_overplots[key] for key in plot_params]))

    if truth is not None:
        flat_truth = flatten_params_to_labeled_dict(truth)
        # if plot_params is not None:
        #     flat_truth = {k:flat_truth[k] for k in plot_params}
        truth_overplot_pts = np.squeeze(np.stack([flat_truth[key] for key in plot_params]))
    else:
        truth_overplot_pts = None

    samples = np.vstack([flat_samples[key] for key in plot_params]).T
    histargs = {'density': True, 'color': color}
    if latex:
        labels = [latex_label(label) for label in plot_params]
    else:
        labels = plot_params
        
    fig = corner.corner(samples, fig=fig, truths=truth_overplot_pts, truth_color=truth_color, 
        show_titles=True, title_fmt='.3f',
        labels=labels, hist_kwargs=histargs, color=color)

    if overplots is not None:
        corner.overplot_points(fig, overplot_pts[np.newaxis], marker='*', markersize=20, mfc=overplot_color, mec=overplot_color)
    
    return fig

def cornerplot_results(map_best, svi_samples=None, HMC_samples=None, true_params=None, hmc_median=None, plot_params=None, svi_label='SVI', hmc_label='HMC', legend_loc='upper right', legend_kwargs=None, truth_label='Truth', map_label='MAP'):
    """
    Cornerplot of the results of the inference pipeline, including MAP, SVI, and HMC.
    """

    svi_color = 'blue'
    hmc_color = 'black'

    fig = cornerplot_posterior(svi_samples, truth=true_params, overplots=map_best, color=svi_color, truth_color='black', overplot_color='red', plot_params=plot_params)
    cornerplot_posterior(HMC_samples, fig=fig, color=hmc_color, plot_params=plot_params)

    # Build a single consolidated legend
    handles = []
    if (svi_label is not None):
        handles.append(Patch(facecolor=svi_color, edgecolor='none', alpha=0.6, label=svi_label))
    if (hmc_label is not None):
        handles.append(Patch(facecolor=hmc_color, edgecolor='none', alpha=0.6, label=hmc_label))
    if (true_params is not None) and (truth_label is not None):
        handles.append(Line2D([0], [0], color='black', lw=1.5, label=truth_label))
    if (map_best is not None) and (map_label is not None):
        handles.append(Line2D([0], [0], marker='*', markersize=12, linestyle='none', markerfacecolor='red', markeredgecolor='red', label=map_label))

    if legend_kwargs is None:
        legend_kwargs = {}
    if 'fontsize' not in legend_kwargs:
        legend_kwargs['fontsize'] = 12
    prev_leg = getattr(fig, "_corner_legend_obj", None)
    if prev_leg is not None:
        try:
            prev_leg.remove()
        except Exception:
            pass
    new_leg = fig.legend(handles=handles, loc=legend_loc, frameon=False, **legend_kwargs)
    setattr(fig, "_corner_legend_obj", new_leg)