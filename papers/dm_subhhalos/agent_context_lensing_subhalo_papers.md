# Strong-Lensing Subhalo Literature Context

Purpose: reusable notes for future agents working on dark-matter subhalo inference in GIGALens, especially for scientifically rigorous re-analyses and for building believable simulated systems with and without perturbers.

Scope: papers currently in `papers/dm_subhhalos/`:

- `Vegetti2009.pdf`
- `Vegetti2012.pdf`
- `Sengul2022.pdf`
- `Minor2017.pdf`
- `Li2026.pdf`

This file emphasizes:

- what each paper actually modeled
- what evidence/statistical claims were made
- where the main scientific and Bayesian weak points are
- what ingredients are most useful for forward simulations

## Executive Takeaways

- The classic "gravitational imaging" detections are not simple direct detections of a subhalo in an unconstrained model class. They are strong model-comparison results inside a relatively specific pipeline: smooth macro model, flexible but regularized source, non-parametric local potential correction, then parametric perturber fit and evidence comparison.
- In the Vegetti papers, the most rigorous result is a Bayes-factor preference for `smooth + perturber` over `smooth-only` within the chosen prior family. The translated `sigma` numbers are heuristic, not primary statistical outputs.
- The strongest recurring failure modes are source-model flexibility, regularization dependence, restricted smooth-mass families, PSF / lens-light subtraction systematics, and incomplete accounting of line-of-sight perturbers.
- `Sengul2022` is especially important because it directly reopens a headline detection and argues that the B1938+666 perturber is better interpreted as a line-of-sight halo than as a same-plane subhalo.
- `Minor2017` is not a detection paper but is essential context: total perturber mass can be badly biased under profile/truncation mismatch; projected mass near the perturbation scale is much more robust.
- `Li2026` is also not a detection paper, but it is highly relevant for building a physically motivated smooth baseline for `J0946+1006` with multi-component stellar mass, a gNFW halo, and flexible pixelated sources.

## Cross-Paper Pattern

The broad analysis template across the detection papers is:

1. subtract lens light and choose an image mask
2. fit a smooth macro lens and flexible source
3. search for localized residuals or potential corrections
4. insert an analytic perturber model
5. compare `smooth` versus `smooth + perturber`

That template is scientifically useful, but each step can absorb or create false evidence for substructure if:

- the smooth macro model is too restrictive
- the source prior is too stiff or too permissive
- the noise model ignores pixel correlations or preprocessing uncertainty
- the prior on perturber position/mass is treated as exhaustive when the real analyst choice space is larger

## Paper-by-Paper Notes

### `Vegetti2009.pdf`

Title: `Detection of a Dark Substructure through Gravitational Imaging`

Status in this context:

- first flagship claimed detection using gravitational imaging
- system: `SDSSJ0946+1006` ("Double Einstein Ring")

Scientific claim:

- a dark substructure is detected in `HST/ACS F814W` imaging of `J0946+1006`
- quoted perturber mass: `M_sub = (3.51 +/- 0.15) x 10^9 Msun`
- quoted location: about `(-0.651, 1.040)` arcsec relative to the lens
- quoted model preference: `Delta log E ~ -128`
- the paper calls this roughly a `16 sigma` detection, but that conversion is heuristic

Data and setup:

- lens redshift `z_l = 0.222`
- source redshift for the inner ring `z_s1 = 0.609`
- faint outer source/ring exists but is excluded from the main inference
- modeling concentrates on a narrow annulus around the bright inner ring

Main-lens model:

- elliptical power-law mass distribution plus external shear
- most runs fix the mass centroid to the light peak
- this is a relatively compact macro family

Subhalo model:

- tidally truncated pseudo-Jaffe perturber
- free parameters effectively reduced to total mass plus image-plane position

Source model:

- pixelated source on an adaptive Delaunay grid
- adaptive curvature regularization weighted by inverse image S/N
- no compact analytic source basis for the final science fit

Inference/statistics:

- smooth fit first
- then linear potential corrections on a Cartesian grid to localize anomalies
- then parametric pseudo-Jaffe perturber fit
- then evidence comparison with `MULTINEST`
- priors on common macro parameters are centered on the smooth best fit and chosen to contain the bulk of the evidence

Main pitfalls:

- the evidence contest is largely against one narrow smooth family, not against a broad class of realistic smooth alternatives
- the source prior is not benign: the paper explicitly notes that some potential structure can be absorbed by the source when source regularization changes
- common-parameter priors are partly data-informed, so the quoted evidence values are not from purely pre-data priors
- masking out the outer ring simplifies the problem but discards potentially useful constraints on the global potential
- system selection and subhalo-position search complicate any population-level interpretation
- the quoted mass uncertainties are statistical only; systematics are explored but not fully marginalized

Why it matters for GIGALens:

- this is the key historical positive-claim target for re-analysis
- it is also directly relevant for Jackpot-like simulations, but only if you separate:
  - the observable local perturbation in the image
  - the inferred total pseudo-Jaffe mass, which is much more model-dependent

Simulation-facing takeaways:

- smooth baseline near a nearly round power-law lens plus moderate external shear
- bright inner ring is the main constraining feature
- a perturber near the arc with local aperture-scale mass signature is the important observable
- null simulations should explicitly stress-test source flexibility, PSF mismatch, and lens-light subtraction residuals because those are exactly the channels through which false positives can arise

### `Vegetti2012.pdf`

Title: `Gravitational detection of a low-mass dark satellite at cosmological distance`

Status in this context:

- second flagship gravitational-imaging claim
- system: `JVAS B1938+666`

Scientific claim:

- reports a `1.9 +/- 0.1 x 10^8 Msun` dark satellite
- quoted evidence preference: `Delta log E = -65.10`
- often described as a `12 sigma` detection, again only through a heuristic Bayes-factor-to-sigma translation
- combines this system with `J0946+1006` to infer a substructure mass fraction and mass-function slope beyond the Local Group

Data and setup:

- Keck adaptive-optics imaging in `K'` and `H`
- archival `HST/NICMOS F160W`
- nearly complete Einstein ring of diameter about `0.9"`
- high angular resolution and multi-dataset consistency are part of the robustness argument

Main-lens model:

- ellipsoidal power-law density plus external shear
- the working parameterization includes a softened power-law elliptical surface density
- core radius is fixed very small in the nominal smooth stage

Subhalo model:

- truncated pseudo-Jaffe perturber
- position plus mass are the key fitted perturber parameters
- the paper also quotes projected mass estimates from the pixelized potential-correction map itself

Source model:

- pixelated source on an adaptive Delaunay tessellation
- gradient regularization in the main analysis
- the source-grid resolution is itself chosen using Bayesian evidence

Inference/statistics:

- fit smooth model
- hold that smooth component fixed and solve for source plus pixelized potential corrections
- identify a localized positive density correction
- replace it with an analytic pseudo-Jaffe perturber
- compare `smooth` and `smooth + perturber` by marginalized evidence

Main pitfalls:

- the same broad concerns as `Vegetti2009` remain: restricted macro family, dependence on source regularization/basis, and heuristic conversion of evidence to `sigma`
- the quoted total mass depends strongly on the assumption that the perturber sits in the lens plane and is tidally truncated there
- the paper itself notes a much safer quantity: a projected mass near the perturbation scale from the potential-correction map
- population inference from two detections is strongly prior-sensitive, especially in the mass-function slope
- the detectability function is simplified relative to a full position-dependent selection model

Why it matters for GIGALens:

- B1938+666 is the canonical "low-mass" dark perturber benchmark in the literature
- but it should not be treated as settled proof of a same-plane subhalo because `Sengul2022` reinterprets it

Simulation-facing takeaways:

- a nearly complete ring is useful because it gives more local arc constraints than a short arc segment
- for realistic simulations, store both:
  - a projected/aperture mass near the perturber
  - any model-specific total mass only as a derived, profile-dependent quantity
- if trying to reproduce the literature detection setup, test both pseudo-Jaffe and non-pseudo-Jaffe truth models

### `Sengul2022.pdf`

Title: `Substructure Detection Reanalyzed: Dark Perturber shown to be a Line-of-Sight Halo`

Status in this context:

- not a fresh detection
- reanalysis of the `B1938+666` perturber
- scientifically important because it attacks the interpretation, not the existence, of the perturbation

Scientific claim:

- the perturber is better fit as a line-of-sight halo than as a same-plane subhalo
- quoted preferred redshift: `z_int ~ 1.42` for a main lens at `z_lens = 0.881`
- quoted preference for NFW interloper over NFW subhalo: `Delta BIC = -17.2` and `log10 K = 3.4`

Data and setup:

- reanalysis uses `HST/NICMOS 1.6 micron` imaging
- masked annulus around the ring after lens-light subtraction

Main-lens model:

- `PEMD + external shear`
- simpler and more standard than a full composite baryon+DM decomposition

Perturber models:

- compares `SIS`, `pseudo-Jaffe`, and `NFW`
- crucially frees the perturber redshift in a multi-plane lensing model
- NFW interloper is the favored interpretation in the real-system analysis

Source model:

- shapelets, not a pixelated inversion
- source complexity controlled by shapelet order `nmax`
- linear coefficients solved analytically for fixed nonlinear lens parameters

Inference/statistics:

- dynamic nested sampling with `dynesty`
- `BIC` used to pick source complexity and for some model-class comparisons
- evidences are also reported

Main pitfalls:

- source flexibility is still conditional on a selected basis and `nmax`, not fully marginalized across richer source-model families
- lens-light subtraction is fixed from an earlier fit rather than jointly propagated through the final posterior
- the macro model remains compact; unresolved macro complexity can masquerade as subtle perturber signatures
- evidence ratios do not automatically include the full analyst choice space across masks, source families, or perturber profile families

Why it matters for GIGALens:

- it is the strongest in-folder argument that "subhalo detection" and "dark perturber detection" are not identical statements
- any rigorous GIGALens program should treat line-of-sight structure as a first-class nuisance or alternative hypothesis

Simulation-facing takeaways:

- generate both same-plane subhalos and interlopers
- if you test redshift recovery, include multi-plane lensing in the forward model
- the inferred total mass can change by about an order of magnitude when the interpretation changes from subhalo to interloper

### `Minor2017.pdf`

Title: `A Robust Mass Estimator for Dark Matter Subhalo Perturbations in Strong Gravitational Lenses`

Status in this context:

- methods paper, not an observational detection paper
- one of the most practically useful papers in this folder for simulation design and for defining what quantity should be compared to theory

Scientific claim:

- total subhalo mass inferred from lensing can be badly biased if the profile shape or tidal radius is wrong
- a more robust quantity is the projected mass within the perturbation scale, scaled by the host log-slope
- the paper recommends reporting an effective subhalo lensing mass rather than only a total mass

Setup:

- simulated strong-lens images, roughly SDP.81-like
- smooth lens: power-law ellipsoid plus external shear
- source: two-Gaussian truth model, then pixel-based source reconstruction in fitting

Perturber truth and fit:

- truth uses a more general cuspy halo family with varying inner slope and tidal radius
- fits use a pseudo-Jaffe perturber with the common assumption tying truncation scale to the host geometry

Source model:

- pixelated adaptive source grid
- curvature regularization
- several algorithmic choices are fixed rather than marginalized

Inference/statistics:

- adaptive `T-Walk` MCMC
- evidence is used in model comparison
- but this is still a simulation study designed to test robustness, not a full observational selection-function treatment

Main pitfalls and lessons:

- the common assumption that perturber truncation is set by projected distance to the host center can badly bias total mass
- macro parameters, especially the host slope, can shift to absorb perturber-model mismatch
- source adaptation/regularization choices remain part of the inference pipeline and can affect model comparison
- the robust lensing observable is local and projected, not the global bound mass

Why it matters for GIGALens:

- if you want a simulation campaign that is scientifically honest, this paper argues strongly that the injected and recovered comparison target should include a local projected mass observable, not just `M_200` or pseudo-Jaffe total mass

Simulation-facing takeaways:

- record multiple mass summaries for every injected perturber:
  - true halo mass definition used by the simulator
  - projected mass within a physically motivated aperture near the perturbation scale
  - any model-specific fitted mass
- profile mismatch tests are mandatory, not optional

### `Li2026.pdf`

Title: `A Salpeter IMF and an NFW halo: Disentangling the dark and stellar mass through precise lens modelling of a double-source-plane system...`

Status in this context:

- not a subhalo-detection paper
- still highly relevant because it provides a modern, high-fidelity smooth model for the Jackpot lens

Scientific claim:

- strong-lensing and kinematic analysis favor a stellar-plus-dark-matter decomposition with:
  - multi-Gaussian stellar mass
  - possible `M/L` gradient
  - elliptical gNFW halo
- the paper explicitly says it does not model dark substructure

Data and setup:

- `HST ACS F814W`
- double-source-plane geometry with additional source-plane mass
- MUSE AO kinematics added by importance reweighting

Main-lens model:

- stars represented by a non-concentric multi-Gaussian expansion
- dark matter as elliptical gNFW
- alternative comparison against a star+EPL model
- source-plane-1 mass modeled as an `SIS`

Source model:

- first a parametric Gaussian stage
- then a pixelated Gaussian-process source with Matérn power spectrum
- positivity enforced by `softplus`

Inference/statistics:

- modern JAX/Herculens pipeline
- SVI initialization followed by `NUTS` / `HMC-within-Gibbs`
- about `3000` parameters

Main pitfalls:

- lens light is fixed after SVI rather than jointly marginalized in the final HMC
- source-plane-1 mass is a major structural uncertainty and partially mimics mass-sheet-like freedom
- kinematic information is folded in by importance reweighting, not full joint sampling
- substructure is treated only as a systematic possibility, not a sampled nuisance component

Why it matters for GIGALens:

- if you want to simulate believable `J0946+1006` systems with or without subhalos, this is the best in-folder baseline for the smooth mass model
- it is a much better starting point than a single power-law if your goal is to stress-test whether subhalo claims survive a more realistic macro/source model

Simulation-facing takeaways:

- use this paper as the default smooth Jackpot baseline
- then add subhalos or interlopers on top of it
- explicitly test whether flexible source-plane mass and source priors can absorb or fake local perturbations

## Cross-Cutting Methodological Vulnerabilities

### 1. Source flexibility is inseparable from subhalo significance

- In the Vegetti pipeline, the source is flexible but strongly regularized.
- In `Sengul2022`, the source is a shapelet basis with BIC-selected complexity.
- In `Li2026`, the source is a GP field with structured hyperpriors.

Implication:

- "detected substructure" always means "preferred once source complexity has been controlled in a specific way."
- A rigorous program should treat source-model misspecification as a primary systematic, not a nuisance footnote.

### 2. Restricted macro families can overstate perturber evidence

- Early papers use compact macro models like power-law ellipsoid plus shear.
- That is computationally sensible, but a local residual can reflect:
  - true perturbers
  - line-of-sight structure
  - unresolved macro asymmetry
  - multipoles
  - stellar/dark decomposition mismatch

Implication:

- Bayes factors are conditional on the smooth family. They are not universal evidence for a subhalo in all plausible macro models.

### 3. Total perturber mass is not the cleanest observable

- `Minor2017` is the clearest statement of this.
- `Vegetti2012` also effectively supports this by quoting projected mass from the potential-correction map.
- `Sengul2022` shows the total mass can shift strongly when the perturber is reinterpreted as an interloper.

Implication:

- compare theories to local projected mass-like observables whenever possible
- do not over-interpret pseudo-Jaffe or NFW total masses as directly measured facts

### 4. Line-of-sight structure must be modeled explicitly

- `Sengul2022` turns this from a generic caveat into a concrete case study.
- A program aimed at dark-matter inference should not assume every perturbation is a subhalo.

### 5. Selection functions and look-elsewhere effects are often simplified

- Positive systems are often chosen because they are bright, structured, and favorable for detection.
- Searches over perturber position are usually included in the formal model, but searches over datasets, masks, preprocessing, and model families are not always fully encoded in the evidence.

## Practical Guidance For Future GIGALens Work

### Recommended baseline simulation philosophy

- Build a realistic smooth baseline first, especially for `J0946+1006`.
- Inject perturbers only after the smooth model can already reproduce the data class without obvious residual pathologies.
- Generate both:
  - same-plane subhalos
  - line-of-sight halos
- Compare recovery in terms of:
  - local projected mass / perturbation-strength proxy
  - inferred model-specific total mass
  - posterior support for "no perturber"

### Minimum stress tests for believable simulations

- vary source complexity and source prior family
- vary macro family from simple power-law to composite stellar+DM
- vary PSF mismatch and lens-light subtraction residuals
- vary noise model, including correlated-noise approximations when relevant
- vary perturber profile and truncation assumptions
- vary perturber redshift, not just its in-plane position

### Recommended reference quantities to save for each injection

- true 3D halo mass definition
- projected aperture mass near the perturbation radius
- apparent image-plane position
- true redshift relative to the main lens
- truth profile family and concentration/truncation parameters
- local arc S/N and magnification at the perturber location

## Bottom Line

- `Vegetti2009` and `Vegetti2012` are essential benchmarks, but their claims should be read as conditional model-comparison successes, not final word on subhalo demographics.
- `Sengul2022` is a required companion paper because it shows that at least one canonical "subhalo" may be better interpreted as a line-of-sight halo.
- `Minor2017` should shape what quantity you try to recover in simulations.
- `Li2026` should shape what you mean by a realistic smooth Jackpot baseline.

