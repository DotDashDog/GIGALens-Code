"""
Experimental pixelized source reconstruction utilities.

This package is intentionally kept under GIGALens-Code (not gigalens/src)
because it is research / prototype code.
"""

from .delaunay_mesh import (
    DelaunayMesh,
    build_brightness_adaptive_imageplane_delaunay_from_truth,
    build_brightness_adaptive_sourceplane_delaunay_from_truth,
    build_frozen_sourceplane_delaunay_from_truth,
    build_regular_imageplane_mesh,
)
from .pixelized_regularization import REGULARIZATION_BUILDERS, build_regularization_matrix
from .pixelized_simulator import PixelizedSourceSimulator
from .pixelized_prob_model import PixelizedSourceProbModel

__all__ = [
    "DelaunayMesh",
    "build_brightness_adaptive_imageplane_delaunay_from_truth",
    "build_brightness_adaptive_sourceplane_delaunay_from_truth",
    "build_frozen_sourceplane_delaunay_from_truth",
    "build_regular_imageplane_mesh",
    "REGULARIZATION_BUILDERS",
    "build_regularization_matrix",
    "PixelizedSourceSimulator",
    "PixelizedSourceProbModel",
]

