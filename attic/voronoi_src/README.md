# voronoi_src (retired, September 2026)

The pre-scene-API pixelized-source prototype: a parallel `PixelizedSourceSimulator` +
`PixelizedSourceProbModel` + hand-written MAP loop on the old gigalens API, with
image-plane-frozen triangle lookup tables. Superseded by
`src/gigalens_research/pixelized_source/`, which puts the mesh source on the scene API
as a light profile with a swappable coefficient prior.

What moved and where:

| here | now |
|---|---|
| `pixelized_regularization.py` (graph-Laplacian builders) | `pixelized_source/regularizers.py` (`GraphLaplacian`) |
| evidence formula in `pixelized_prob_model.py` | `pixelized_source/likelihood.py` (`RegularizedImageLikelihoodTerm`) |
| `delaunay_mesh.py` builders | `pixelized_source/domain.py` (`TriangleMeshDomain`, `adaptive_delaunay_domain`) |
| `diagnostics/quality_metrics.alternating_pattern_score`, evidence scan | `pixelized_source/diagnostics.py` |

Not carried over: the Stage 0 / Phase 1 / Phase 2 driver scripts under `tests/` (they
target the old API and the removed `source_modeling/` layout and were never run on this
machine), the non-negative `lsq_linear` solver, and the "moving vertices with frozen
connectivity" design itself — the tessellation is now frozen in the source plane by
decision (see `docs/logs/pixelized-source.md`).

Nothing in this directory imports.
