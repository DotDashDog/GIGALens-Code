"""Plotting helpers for GIGALens inference outputs.

Organized so each module does one thing:

- :mod:`.image` — image-plane primitives (``plot_image``, residuals, histogram).
- :mod:`.source_plane` — source plane + caustic/critical curve overlays.
- :mod:`.convergence` — chain traces, running R-hat / ESS, loss histories.
- :mod:`.corner` — corner plots from :class:`Posterior` objects.
- :mod:`.diagnostics` — stage-specific debug run histories (e.g. MCLMC tuning).
- :mod:`.labels` — LaTeX label registry and parameter flatteners.
- :mod:`.reports` — compound multi-panel figures (single posterior or pipeline).
"""

from .convergence import (
    plot_chain_traces,
    plot_loss_history,
    plot_running_ess,
    plot_running_rhat,
)
from .corner import plot_corner, plot_corner_overlay
from .diagnostics import (
    has_diagnostic_plotter,
    plot_mclmc_diagnostics,
    plot_stage_diagnostics,
    register_diagnostic_plotter,
)
from .image import normalized_residual, plot_image, plot_residual_histogram
from .labels import (
    LATEX_LABELS,
    flatten_param_names,
    flatten_params,
    latex_label,
)
from .reports import PipelineReport, PosteriorReport
from .source_plane import (
    LENS_MODEL_NAME_MAP,
    plot_caustics,
    plot_caustics_critical,
    plot_critical_curves,
    plot_image_border,
    plot_source_plane,
)
from .truth import plot_source_comparison, plot_z_scores

__all__ = [
    "LATEX_LABELS",
    "LENS_MODEL_NAME_MAP",
    "PipelineReport",
    "PosteriorReport",
    "flatten_params",
    "flatten_param_names",
    "has_diagnostic_plotter",
    "latex_label",
    "normalized_residual",
    "plot_caustics",
    "plot_caustics_critical",
    "plot_chain_traces",
    "plot_corner",
    "plot_critical_curves",
    "plot_corner_overlay",
    "plot_image",
    "plot_image_border",
    "plot_loss_history",
    "plot_mclmc_diagnostics",
    "plot_residual_histogram",
    "plot_running_ess",
    "plot_running_rhat",
    "plot_source_comparison",
    "plot_source_plane",
    "plot_stage_diagnostics",
    "plot_z_scores",
    "register_diagnostic_plotter",
]
