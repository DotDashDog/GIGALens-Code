# Pixelized sources on the scene API: where the regularizer should live

**Status:** decision doc, September 2026. The restructure itself is done in
`src/gigalens_research/pixelized_source/` (frozen domains, `MeshSource`,
`CorrelatedFieldSource`, `GraphLaplacian`). What is *not* decided is the permanent home of
the quadratic-prior / evidence machinery, because it changes what "lstsq mode" means in
gigalens. This doc lays out the options with concrete code so the choice can be made on
the diff, not on prose. The research repo ships **Option A** as a working prototype so
everything is exercisable today; nothing in `~/gigalens` was touched.

Lab log: `docs/logs/pixelized-source.md`.

## 0. What the restructure settled

- **Tessellation never changes inside inference.** Connectivity *and* source-plane vertex
  positions are frozen from a pilot model (`adaptive_delaunay_domain`, or a
  `RegularGridDomain`). The only mass dependence is the ray-traced evaluation point, so
  the likelihood is continuous and a.e. differentiable in the mass; there are no folds and
  no degenerate-triangle guards.
- **A pixelized source is a light profile.** `MeshSource(domain, regularizer)` is an
  lstsq profile whose `light(x, y)` returns the `(I, ...)` basis; `CorrelatedFieldSource`
  is a forward-mode profile whose latent excitations are ordinary sampled parameters. Both
  ride the scene simulator's PSF, pooling, multi-dataset, multiplane and inference paths
  unchanged.
- **The coefficient prior is the swappable piece.** For the correlated field it is
  *already* just a prior on ordinary parameters (nothing to decide). For the lstsq mesh it
  is a quadratic prior whose analytic marginalization needs a place to add
  `s^T H s + log det A − log det H` to the likelihood. That is the decision.

## 1. The math the hook has to carry

For one imaging dataset with weighted design `X (npix, ncomp)` (the rendered basis stack,
divided by sigma, masked) and target `y`, with a block-diagonal prior precision `H`
(zero blocks for unregularized components):

```
A   = XᵀX + H
s*  = A⁻¹ Xᵀ y
-2 log Z = |y − X s*|² + s*ᵀ H s* + log det A − log det H_reg + Σ log 2πσ²   (+ const)
```

`log det H_reg` runs over the regularized blocks only (a flat prior on the other blocks is
an additive constant). Note the **profile-vs-marginal** difference: today's `lstsq` mode
returns `|y − X s*|²` only (the *profile* likelihood, maximized over amplitudes). The
evidence adds `log det A`, which depends on the mass through `X`. With `H = 0` the two
differ by exactly that term, so a hook cannot be "byte-identical when no prior is
declared" *and* Bayesian at the same time — see decision D1.

`H` is per component: `MeshSource.regularizer.matrix(**hyper)` with the hyperparameters
(`lam`) read from the component's params. Its log-determinant is analytic
(`GraphLaplacian.logdet`: `I log lam + const`), no `slogdet` in the hot path.

## 2. Options

### A. Research-side likelihood term (implemented: `pixelized_source/likelihood.py`)

`RegularizedImageData(ImageData).build_term()` returns
`RegularizedImageLikelihoodTerm(ImageLikelihoodTerm)`, which calls
`sim.lstsq_simulate(..., return_stacked=True)` for the basis stack and does its own
normal-equation solve with `H` added.

- **+** Zero gigalens change; usable now; tested against an independent numpy evaluation
  of the formula (`tests/test_pixelized_source.py`).
- **−** Reaches into `SceneSimulator._light` to find which columns belong to which
  component (private attribute; will break silently if the stack order changes).
- **−** Duplicates the solve tail of `lstsq_simulate` (jitter policy, LU-vs-Cholesky
  rationale) — two copies to keep in sync.
- **−** The dataset class, not the source, decides that a prior exists: a `MeshSource`
  with a regularizer seen by a plain `ImageData` is silently fit with a flat prior.

### B. gigalens hook on the profile + likelihood term (recommended)

Regularization is a property of the light component, so declare it there and let the
two existing seams consume it. Sketch (≈60 lines):

```python
# gigalens/profile.py
class LightProfile:
    def linear_prior(self, **params):
        """(H, logdetH) for this component's lstsq coefficients, or None (flat)."""
        return None

# gigalens/jax/scene_simulator.py  — lstsq_simulate, after building X (bs, npix, ncomp)
def _linear_prior(self, params, ncomp, bs):
    H = jnp.zeros((bs, ncomp, ncomp)); logdet = jnp.zeros((bs,)); off = 0
    for i, j, comp, d in self._light:
        lp = comp.profile.linear_prior(**self.model.component_params(params, i, "light", j))
        if lp is not None:
            Hk, ld = lp                       # (bs, d, d), (bs,)   [vmap over hyper]
            H = H.at[:, off:off+d, off:off+d].add(Hk); logdet = logdet + ld
        off += d
    return H, logdet

    ...
    H, logdetH = self._linear_prior(params, self.depth, X.shape[0])
    coeffs = _solve_normal_eq_with_fallback(Xt @ X + H, Xt @ Y)[..., 0]
    if return_evidence_terms:
        return coeffs, dict(sHs=..., logdetA=slogdet(Xt @ X + H), logdetH=logdetH)

# gigalens/jax/scene_prob_model.py — ImageLikelihoodTerm.log_like
    if self.has_linear_prior:            # any seen profile declares one
        chi2, extra = ...; ll = -0.5 * (chi2 + extra["sHs"] + extra["logdetA"] - extra["logdetH"] + norm)
```

`MeshSource.linear_prior` then returns `(regularizer.matrix(lam), regularizer.logdet(lam))`
and `RegularizedImageData` disappears. The hyperparameter already lives on the profile
(the restructure did that part), so no params-tree change is needed.

- **+** The prior travels with the source: every dataset that sees the component gets it.
- **+** One solve, one jitter policy, one place to remat.
- **+** Any lstsq profile can declare a prior later (e.g. a ridge on high-order shapelets).
- **−** Touches the two most load-bearing files in gigalens; needs the byte-identity gate
  for models with no prior (D1 makes that exact).

### C. Explicit coefficients, no marginalization

Make the vertex values ordinary sampled parameters (`MeshSource` in forward mode with
`I` params and a `tfd.MultivariateNormal*` prior with precision `H`), i.e. treat the
mesh exactly like `CorrelatedFieldSource`.

- **+** Zero gigalens change and zero new math: "a prior is a prior".
- **+** The only route for **non-Gaussian** coefficient priors (positivity, sparsity,
  log-normal fields) and for hierarchical `lam` sampled jointly.
- **−** Gives up the Rao-Blackwellization that makes lstsq fast: `I` extra dimensions
  (hundreds to thousands) in the sampler, and the linear–nonlinear coupling is exactly the
  stiff geometry MCLMC struggles with.
- Recommended as the *second* path, not the default: the correlated field already
  exercises it.

### D. Hook at the dataset/term level in gigalens

`ImageData(linear_prior=...)`: like A but upstream. Rejected — it puts the source's prior
on the observation; with two bands seeing one source you would declare it twice.

## 3. Decisions the chosen option needs

- **D1 — profile vs marginal likelihood when a prior is declared.** Recommendation:
  *no* component declares a prior → today's profile likelihood, byte-identical (no
  `log det A`); *any* component declares one → full marginalization over **all** linear
  coefficients of that dataset (flat on the undeclared blocks), i.e. `log det A` over the
  full matrix. This is the Suyu convention (lens-light amplitudes marginalized alongside
  the source) and it is what Option A does now.
- **D2 — hyperparameter parameterization.** `lam` as a positive param with a LogNormal
  prior (the chart is the log). Scale is set by the basis normalization (basis values in
  [0, 1] per pixel, design divided by σ): on the 24-px mocks the evidence optimum sat at
  `lam ≈ 10` for noise 0.05–0.5 alike on a decade grid — the expected σ⁻² scaling was
  not resolved; treat the scale as unverified and scan (see the log).
- **D3 — evidence terms in the diagnostics channel.** `(log_like, red_chi2)` stays: the
  reduced chi-square reported is `|y − X s*|²/N` at the regularized `s*`, which is what
  the operating card's chi²/ν table expects.

## 4. Two small gigalens gaps found on the way (independent of the regularizer)

- **F1 — parameterless light components are unaddressable.** `LensModel._derive`
  creates params-tree sites only for declared parameters, so a component with no params
  has no node and `component_params()` raises `KeyError` at render time. Fix: create the
  `planes/<i>/light/<key>` node for every component in `_derive`. Until then `MeshSource`
  requires a regularizer and a flat-prior basis is `lam` fixed to a tiny constant
  (tested: chi² agrees with plain lstsq to 1e-6).
- **F2 — `z = 0` is not "the prior mean".** For a Normal prior the unconstrained
  coordinate is the raw value, so `z = 0` puts `theta_E = 0`, where the EPL gradient is
  NaN. Not a bug, but a trap for test authors; `LensModel.unconstrained(params)` is the
  safe way to build a `z`.
