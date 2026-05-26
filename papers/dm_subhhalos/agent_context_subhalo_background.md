# Dark Matter Subhalo Background Context

Purpose: reusable background notes for future agents on what is currently understood about dark-matter subhalos outside the narrow context of strong-lensing detections.

This file is meant to support scientifically careful reasoning about:
- what CDM generically predicts for subhalo populations
- how baryons modify those predictions
- what observations currently do and do not establish
- which uncertainties matter when connecting lensing detections to cosmological inference

## Scope And Source Basis
This note combines:
- classic review context from Bullock & Boylan-Kolchin 2017
- classic DMO subhalo results from Aquarius
- recent baryonic-Milky-Way subhalo results from `The dark side of FIRE` (Barry et al. 2023)
- recent large-volume subhalo-structure trends from Moliné et al. 2023
- recent numerical-caution context from Xu 2025 on subhalo convergence/orphan treatment
- observational context from dwarf satellites, stellar streams, indirect searches, and high-redshift structure probes

Use this as a working scientific summary, not as a substitute for reading the cited papers when a future task depends on exact equations or quantitative limits.

## Short Version
- In CDM, subhalos are an unavoidable consequence of hierarchical assembly.
- The dark-matter-only prediction is qualitatively robust: subhalos are numerous, their abundance rises steeply to low mass, and lower-mass structure should continue far below galaxy-forming scales.
- Baryons matter enormously in the inner halo. A central galaxy can destroy or deplete many subhalos on plunging orbits, so dark-matter-only counts overpredict the inner subhalo population.
- The cleanest observationally secure low-mass structures are still luminous satellites. Truly dark subhalos remain hard to establish unambiguously.
- Strong lensing, stellar streams, and indirect detection are complementary because they probe different mass ranges, radii, and assumptions.
- For inference, the biggest scientific danger is comparing a lensing-derived model-specific "subhalo mass" to a simulation prediction that is defined using a different mass concept, radial selection, or baryonic environment.

## 1. What CDM Predicts In Broad Terms

### 1.1 Subhalos are generic, not optional
In Lambda-CDM, structure forms hierarchically: small halos collapse first and later merge into larger hosts. The dense cores of many infalling halos survive tidal stripping, leaving bound subhalos inside host halos.

This qualitative statement is not controversial. The details that matter are:
- how many subhalos survive
- how that number depends on host mass and radius
- how concentrated they are
- how baryons and numerical resolution alter the answer

### 1.2 Halo and subhalo abundances are steep
Bullock & Boylan-Kolchin summarize the basic DMO expectation as:
- halo mass function `dn/dM ~ M^-1.9`
- subhalo mass function rising steeply as well, with `dN/dm ~ m^alpha_s` and `alpha_s ~ -1.8`

Interpretation:
- there are many more low-mass halos than high-mass halos
- each mass decade contributes non-negligibly
- naive extrapolation of DMO CDM implies abundant low-mass substructure down to the free-streaming cutoff

### 1.3 The minimum mass scale can be tiny in canonical CDM
For standard WIMP-like CDM, Bullock & Boylan-Kolchin note a free-streaming / damping cutoff near roughly Earth-mass scales, `~10^-6 Msun`, though the exact value depends on the particle model.

Implication:
- the physically existing subhalo hierarchy in CDM extends far below the mass scale currently observable in lensing or streams
- observed subhalos are therefore an extremely biased, high-mass, high-signal subset of the full population

## 2. What Dark-Matter-Only Simulations Established

### 2.1 Aquarius remains canonical background
The Aquarius simulations resolved nearly `300,000` bound subhalos in a Milky-Way-sized halo and established several enduring qualitative points:
- substructure is abundant
- subhalo counts are approximately self-similar across host mass
- subhalos are typically more concentrated than comparable isolated halos
- inner density profiles are better described by gently curving Einasto-like forms than by a single asymptotic power-law

Aquarius also emphasized something that is still easy to forget in lensing discussions:
- the local substructure mass fraction depends strongly on radius
- even if total halo substructure is abundant, the inner galaxy can contain a much smaller surviving fraction

### 2.2 Large-volume modern simulations refine the scaling relations
Moliné et al. 2023 use the `Phi-4096` and `Uchuu` suites to characterize subhalo abundance and structure across a very wide dynamic range. Their abstract-level takeaways are:
- subhalo mass and velocity functions are well fit by power laws
- the slopes show little redshift dependence
- subhalo abundance depends only weakly on host halo mass
- concentration depends strongly on distance to the host center
- at fixed subhalo mass, subhalos inside more massive hosts are more concentrated

Implication:
- one should expect radial trends in detectability even before adding baryonic physics
- subhalos near the host center are not just fewer; the survivors can also be structurally different

## 3. What Baryons Change

### 3.1 Baryons are not a perturbation in the inner halo
The biggest qualitative correction to DMO expectations is the presence of the central galaxy. Disk and bulge potentials increase tidal shocking and destruction for subhalos on small-pericenter orbits.

Barry et al. 2023 summarize this clearly for Milky-Way-mass hosts:
- within `50 kpc`, a typical baryonic FIRE-2 host contains about `16` subhalos above `10^7 Msun`
- and only about `1` above `10^8 Msun` at `z = 0`
- the corresponding DMO runs overpredict counts by about `2x-10x`, especially at smaller radii

This is one of the most important facts for lensing interpretation:
- DMO predictions cannot be compared directly to observationally inferred inner-halo substructure without baryonic corrections

### 3.2 The radial distribution is especially sensitive
The baryonic depletion is strongest near the host center, exactly where galaxy-galaxy lensing is most sensitive.

Consequences:
- inner-halo substructure fractions from DMO-only predictions are generally too high
- line-of-sight halos become relatively more important compared with surviving same-plane subhalos
- the host's assembly history and the presence of massive satellites like the LMC can materially affect present-day subhalo counts

Barry et al. also find:
- subhalo counts were about `10x` higher at `z = 1` than at `z = 0`
- LMC-like satellite passages can enhance local subhalo populations by factors of about `1.4-2.7`

So the subhalo population is not just a function of host mass. It depends on dynamical history.

### 3.3 Baryons also affect which low-mass halos host galaxies
The classic "missing satellites" issue is not just "CDM predicts too many subhalos." It is more specifically:
- CDM predicts many low-mass halos and subhalos
- only a subset of them are expected to host visible galaxies because star formation becomes inefficient at low mass
- reionization, feedback, gas stripping, and environment all matter

Current consensus is not that the small-scale tensions have vanished, but that several are substantially alleviated once:
- survey incompleteness
- baryonic suppression of star formation
- baryonic destruction of inner subhalos
are modeled more realistically.

## 4. What Is Observationally Established

### 4.1 Luminous satellites are the cleanest confirmed low-mass structures
Observed dwarf satellites of the Milky Way and M31 are the clearest secure examples of low-mass dark-matter-dominated substructure.

What they establish:
- low-mass halos exist
- galaxy formation becomes extremely inefficient at low mass
- there is large scatter between halo mass and stellar content at the faint end

What they do not establish on their own:
- the abundance of truly dark subhalos
- the exact inner-halo subhalo mass function around massive ellipticals
- a direct one-to-one mapping between luminous satellites and the full subhalo population relevant for lensing

### 4.2 Stellar streams are one of the cleanest probes of dark subhalos in the Milky Way
Cold stellar streams can be perturbed by passing subhalos, producing gaps, spur-like features, and velocity perturbations.

Banik et al. 2021 show that stream-based measurements of the subhalo mass function can already be used to constrain alternative dark-matter models. Their abstract reports:
- `m_WDM > 3.6 keV` from streams alone
- strengthening to `m_WDM > 6.2 keV` when combined with dwarf counts

Interpretation:
- stream perturbations are now competitive as a low-mass structure probe
- but stream inference depends strongly on the Galactic potential, stream history, baryonic perturbers, and the treatment of subhalo depletion in the inner Milky Way

### 4.3 Strong lensing is unique because it responds to mass, not light
Lensing can detect perturbers that have no observed stars and lie at cosmological distance.

Strength:
- sensitive to gravitational potential directly
- accesses halos/subhalos in different hosts and redshifts than the Local Group

Weakness:
- sensitive only through model-dependent perturbations of an image
- strongly conditioned on source structure, macro-model choice, noise, PSF, and line-of-sight confusion

For this reason, lensing and streams are complementary:
- streams probe the Milky Way inner halo over time
- strong lensing probes individual distant systems in projection

### 4.4 Indirect searches have not produced a secure dark-subhalo detection
Bullock & Boylan-Kolchin already emphasized that Fermi dwarf analyses showed no conclusive dark-matter signal. More recent Fermi-LAT searches for starless subhalos among unassociated sources also continue to yield null results rather than a secure detection.

Safe summary:
- indirect searches currently provide constraints, not confirmed dark-subhalo discoveries
- interpretation remains highly particle-model-dependent because the signal depends on annihilation or decay physics, not just gravitational mass

### 4.5 High-redshift structure also constrains small-scale power
Lyman-alpha forest and high-redshift galaxy counts constrain suppression of small-scale structure. These are not subhalo detections, but they matter because they limit how different the low-mass halo spectrum can be from CDM.

Bullock & Boylan-Kolchin summarize older Lyman-alpha constraints around thermal relic `m_WDM > 3.3 keV`; modern limits are model- and analysis-dependent, but the broad lesson remains:
- the power spectrum cannot be strongly suppressed on dwarf-galaxy scales without conflicting with other data

## 5. What Remains Uncertain Or Actively Debated

### 5.1 Inner-halo subhalo counts are numerically difficult
Recent work like Xu 2025 highlights that numerical convergence of inner-halo substructure remains hard, especially where tidal forces are strongest. The abstract-level takeaway is:
- subhalo abundance and phase-space structure near the host center can require orphan modeling and careful convergence treatment
- the inner `<< 1 Mpc` regime remains especially challenging

Practical implication:
- even simulation-based "ground truth" for subhalo abundance near the center of a galaxy is not perfectly settled
- future comparisons between lensing detections and theory should track whether the simulation result includes:
  - baryons or not
  - orphan subhalos or not
  - what halo finder / mass definition is used

### 5.2 The correct mass definition depends on the observable
This is a major source of confusion across the literature.

Possible mass-like quantities include:
- `M_200`
- `M_vir`
- bound mass after stripping
- `M_peak`
- `V_max`
- projected mass within a small aperture
- model-specific pseudo-Jaffe or truncated-NFW total mass

These are not interchangeable.

For lensing:
- local projected mass or local deflection perturbation is often closer to the actual observable
- total halo mass is usually inferred through a model family and environmental assumptions

### 5.3 Line-of-sight halos can rival or exceed same-plane subhalos
This matters especially for strong lensing.

Even if CDM predicts abundant subhalos in host halos, the effective perturber population in a lens image is a mixture of:
- surviving host subhalos
- foreground halos
- background halos

Because baryons deplete the inner host subhalo population, line-of-sight contributions can become comparatively more important than a naive DMO-only intuition suggests.

### 5.4 Small-scale tensions are not one single problem
The "missing satellites," "too-big-to-fail," and "cusp-core" problems are related but distinct.

Current cautious consensus:
- none is best treated as a one-line falsification of CDM
- baryonic physics appears capable of easing much of the tension
- but the exact degree of resolution is still debated and depends on the system, mass scale, and comparison method

## 6. Implications For A Strong-Lensing Subhalo Program

### 6.1 Why realistic simulations matter
If you want to infer dark-matter physics from lensing perturbers, you need simulated systems that are believable in all of the following senses:
- believable smooth macro lens
- believable source morphology
- believable PSF and noise
- believable perturber population
- believable host-versus-line-of-sight mixture

Otherwise a statistically impressive fit can still answer the wrong scientific question.

### 6.2 What theory-to-data comparison should look like
The safest comparison pipeline is usually:
1. simulate a host population with baryons or baryon-calibrated subhalo depletion
2. generate both subhalos and line-of-sight halos
3. propagate them through a full lensing forward model
4. recover observables that are as local and model-independent as possible
5. only then map those observables back to halo-mass-function parameters

### 6.3 Suggested summary quantities for future agents
When discussing or storing subhalo predictions, always state:
- host mass scale
- redshift
- radial range inside the host
- whether baryons are included
- whether line-of-sight halos are included
- what mass definition is used
- whether the quantity refers to all subhalos, luminous satellites only, or dark subhalos only

## 7. Practical Heuristics For Future Agents
- Never compare a pseudo-Jaffe lensing mass directly to a simulation `M_200` without an explicit mapping.
- Never compare an observational inner-halo perturber count to DMO subhalo counts without a baryonic correction.
- Treat line-of-sight halos as part of the signal model, not as an afterthought.
- When possible, use projected/aperture masses or perturbation-strength summaries as the interface between inference and theory.
- Be explicit about whether a statement concerns:
  - all halos
  - host subhalos
  - luminous satellites
  - dark satellites

## 8. Bottom Line
- CDM robustly predicts abundant substructure.
- The broad existence of subhalos is not in doubt.
- The detailed abundance of observable perturbers in the inner regions of real galaxies depends strongly on baryons, environment, line-of-sight structure, and mass definition.
- Truly dark subhalos are still difficult to identify securely with any single observational method.
- For strong lensing, the scientifically honest target is not "does CDM predict some subhalos?" but rather "does a fully specified population model, with baryons and line-of-sight structure included, reproduce the distribution of image perturbations we actually observe?"

## References Mentioned Explicitly In This Note
- Bullock, J. S. and Boylan-Kolchin, M. 2017, `Small-Scale Challenges to the LambdaCDM Paradigm`
- Springel et al. 2008, `The Aquarius Project: the subhalos of galactic halos`
- Moliné et al. 2023, `LambdaCDM halo substructure properties revealed with high-resolution and large-volume cosmological simulations`
- Barry et al. 2023, `The dark side of FIRE: predicting the population of dark matter subhaloes around Milky Way-mass galaxies`
- Banik et al. 2021, `Novel constraints on the particle nature of dark matter from stellar streams`
- Xu 2025, `Abundance and phase-space distribution of subhalos in cosmological N-body simulations: testing numerical convergence and correction methods`
