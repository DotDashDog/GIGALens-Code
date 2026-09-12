# Lab notebook: pixelized (mesh / grid) sources

Newest first. Claims register at the bottom. Design decisions live in
`docs/plans/pixelized-source-regularizer-options.md`.

## 2026-09-12 — restructure onto the scene API (branch `worktree-pixelized-source-restructure`)

**Decision (user):** the tessellation never changes inside an inference algorithm;
differentiability in the mass parameters is a must. Connectivity and source-plane vertex
positions are frozen from a pilot model. Consequence: no folds, no degenerate-triangle
guards, the likelihood is C0 and a.e. differentiable in the mass (piecewise-linear
interpolant; derivative jumps at cell edges).

**What was built** (`src/gigalens_research/pixelized_source/`), replacing
`voronoi_src` (moved to `attic/voronoi_src`, see its README):

- `domain.py`: `RegularGridDomain` (bilinear), `TriangleMeshDomain` (Delaunay,
  bucketed point location, CCW-oriented, degenerate triangles raise), `adaptive_delaunay_domain`
  (weighted-KMeans image-plane seeds from a pilot image, ray-traced with
  `LensModel.trace_to_plane`, optional padding ring).
- `profiles.py`: `MeshSource` (lstsq basis, carries its regularizer's hyperparameters),
  `CorrelatedFieldSource` (forward-mode log-normal GRF on the regular grid; excitations
  as a grouped tuple-key prior).
- `regularizers.py`: `GraphLaplacian` (gradient / curvature; analytic log det).
- `likelihood.py`: `RegularizedImageData` → `RegularizedImageLikelihoodTerm`, the
  research-side evidence term (Option A). gigalens untouched.

**Verification (tests/test_pixelized_source.py, 17 tests, login-node CPU, float64):**
point location vs scipy on 4000 queries; exact linear reproduction by both interpolants;
partition of unity through the scene simulator under an EPL lens; evidence pieces
(chi2, sHs, log det A, log det H, log Z) vs an independent float64 numpy evaluation to
rtol 1e-9; batched == unbatched; weak-prior limit == plain lstsq chi2 to 1e-6; analytic
log det H == slogdet; correlated field white limit exact and unit variance at slope 1.5;
forward ProbModel finite with finite gradients in all 68 params; adaptive builder end to
end.

**Finding: chi²/ν floor is set by grid resolution at high SNR, not by the regularizer.**
Mock: EPL θE=0.9 + Sersic (R=0.25", n=1.5, Ie=20), 24×24 px at 0.1"/px, no PSF.
Unregularized (lam→0) reduced chi-square on a `RegularGridDomain` of half-size 0.8":

| grid | spacing | noise 0.05 (peak SNR 400) | noise 0.2 (SNR 100) | noise 0.5 |
|---|---|---|---|---|
| 16×16 | 0.107" | 3.08 | 0.88 | 0.77 |
| 24×24 | 0.070" | 1.68 | 0.65 | 0.59 |
| 32×32 | 0.052" | 1.11 | — | — |

Cause hypothesis: bilinear interpolation bias on the n=1.5 cusp (second-derivative
error ~h²·Ie/R² ≈ 0.5 SB units at the peak vs σ=0.05). Predicted direction: excess falls
with spacing; observed 2.08 → 0.68 → 0.11, confirmed. At SNR 100 the floor is below 1
(over-fit: 256 basis functions for 576 pixels) and the evidence-optimal lambda brings
it to 1.02 (16×16) / 0.88 (24×24). The evidence-optimal `lam` was 10 (decade grid) in
every configuration — expected scaling with σ⁻² was NOT seen at decade resolution;
**open**, see below.

Plot (data / model / (data−model)/σ / reconstructed source / truth), n=24, noise 0.2,
lam=10: residual structureless, ring reproduced, source peak at the truth centre
(0.05, −0.02). Reconstruction is in per-pixel flux units (peak 1.6 ≈ SB 200 × 0.01
arcsec²). Shallow negative ringing (−0.25) around the cusp — gradient regularization has
no positivity. Not saved to the record (scratch); regenerate with
`tests/test_pixelized_source.py::test_regularized_prob_model_gradients_and_evidence_optimal_lambda`
machinery + `domain.render`.

**Test threshold derivation:** the regularizer test uses the 16×16 / noise 0.2
configuration where the floor over-fits (0.88) and the evidence optimum is expected at
chi²/ν = 1 ± sqrt(2/576) = 0.06; asserted window 0.85–1.2. The noise-0.05 configuration
first tried (floor 3.08) is NOT a regularizer test and was withdrawn.

**Traps recorded:** (F1) a parameterless light component has no params-tree node in
gigalens (`MeshSource` therefore requires a regularizer; flat prior = `lam` fixed tiny);
(F2) `z = 0` under a Normal prior is `theta_E = 0` (NaN EPL gradient), build `z` with
`LensModel.unconstrained`.

## Claims register

| claim | status | evidence |
|---|---|---|
| Frozen-domain mesh source is differentiable in the mass through the scene API | **verified** (unit) | finite gradients, 17 tests |
| Research-side evidence term equals the Suyu/WD03 formula | **verified** (unit, rtol 1e-9) | independent numpy |
| Evidence picks an interior lambda that yields chi²/ν ≈ 1 at SNR 100 | **UNCERTIFIED**, one mock, one seed | scan table above |
| Evidence-optimal lambda scales with σ⁻² | **open** — not seen at decade resolution; needs a fine lam grid and a normalization check | — |
| Correlated field recovers a source / samples with MCLMC | **not attempted** | — |
| Adaptive Delaunay domain on real data (Vela / carousel) | **not attempted** | — |
