"""
gigalens_research

Research-side scripts and utilities built on top of the `gigalens` library
for strong gravitational lens modeling.

Subpackages
-----------
- inference      : Alternate inference algorithms / samplers.
- inference_utils: MAP / SVI / HMC pipeline orchestration and diagnostics.
- plotting       : Image, residual, loss, and corner-plot helpers.
- voronoi_src    : Experimental pixelized (Delaunay/Voronoi) source reconstruction.

This package is intentionally light at the top level. Import from the
subpackages explicitly, e.g.

    from gigalens_research.inference import MCLMC
    from gigalens_research.inference_utils import PipelineConfig, run_pipeline
    from gigalens_research.plotting import plot_image_results, cornerplot_results
"""

__version__ = "0.1.0"
