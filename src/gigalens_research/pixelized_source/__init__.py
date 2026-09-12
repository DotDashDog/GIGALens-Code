"""Pixelized (mesh / grid) source models on the gigalens scene API.

Three separable pieces (see ``docs/plans/pixelized-source-regularizer-options.md``):

1. **A frozen source-plane domain** with a differentiable interpolation operator
   (:mod:`.domain`). Built once from a pilot model, outside inference; the
   tessellation never changes inside an inference algorithm. Two implementations of one
   interface: :class:`RegularGridDomain` (bilinear) and :class:`TriangleMeshDomain`
   (Delaunay, bucketed point location).
2. **Light profiles** (:mod:`.profiles`) that plug into ``gigalens.jax.scene`` like any
   other light: :class:`MeshSource` (an lstsq basis, one function per vertex, carrying
   its regularizer's hyperparameters as its only parameters) and
   :class:`CorrelatedFieldSource` (a forward-mode Gaussian random field whose latent
   excitations are ordinary sampled parameters).
3. **A swappable coefficient prior**: :mod:`.regularizers` builds the quadratic
   penalty ``H`` for :class:`MeshSource`; :mod:`.likelihood` holds the research-side
   evidence term that consumes it (Option A in the plan doc — the gigalens hook is
   deliberately NOT implemented here).

The old ``voronoi_src`` package (parallel simulator + prob model on the pre-scene API)
is retired to ``attic/voronoi_src``.
"""

from .domain import (
    RegularGridDomain,
    SourceDomain,
    TriangleMeshDomain,
    adaptive_delaunay_domain,
    kmeans_weights,
)
from .likelihood import RegularizedImageData, RegularizedImageLikelihoodTerm
from .profiles import CorrelatedFieldSource, MeshSource
from .regularizers import GraphLaplacian, QuadraticRegularizer

__all__ = [
    "SourceDomain",
    "RegularGridDomain",
    "TriangleMeshDomain",
    "adaptive_delaunay_domain",
    "kmeans_weights",
    "MeshSource",
    "CorrelatedFieldSource",
    "QuadraticRegularizer",
    "GraphLaplacian",
    "RegularizedImageData",
    "RegularizedImageLikelihoodTerm",
]
