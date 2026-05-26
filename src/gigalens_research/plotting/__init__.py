"""Plotting helpers: images, residuals, loss curves, and corner plots."""

from .systems import (
    plot_image,
    plot_image_results,
    add_caustics,
    histogram_residuals,
)
from .corner import (
    cornerplot_labels,
    cornerplot_posterior,
    cornerplot_results,
    latex_label,
    flatten_params_to_labeled_dict,
    flatten_params_to_labeled_dict_sim,
)
from .suites import (
    plot_loss_histories,
    display_results,
)

__all__ = [
    "plot_image",
    "plot_image_results",
    "add_caustics",
    "histogram_residuals",
    "cornerplot_labels",
    "cornerplot_posterior",
    "cornerplot_results",
    "latex_label",
    "flatten_params_to_labeled_dict",
    "flatten_params_to_labeled_dict_sim",
    "plot_loss_histories",
    "display_results",
]
