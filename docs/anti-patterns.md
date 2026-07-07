# Anti-Patterns Playbook (negative knowledge)

This file holds the project's **tempting-but-wrong moves**: the things an agent (or human)
will reach for by default that experience here says are not fruitful. This is the knowledge a
model *cannot* infer from general principles — it exists only because someone here already went
down the path. It is the highest-value file for the human to keep expanding.

**Entry format** — every entry needs all four fields; an entry without a *why* or a *correct
first move* is a superstition, not a rule:

- **The tempting move:** what an agent will do by default.
- **Why it fails here:** the mechanism, in model/algorithm/data terms.
- **Correct first move:** what to do instead, concretely.
- **Real instance:** pointer to the log/artifact where this actually happened. Entries whose
  supporting diagnosis is still `proposed (UNCERTIFIED)` say so — verify against the cited
  artifact before leaning on them hard.

**Maintenance:** entries are *earned* — one failure → one post-mortem → at most one entry
(AGENTS.md structural rule 4). The top ~3 headlines are mirrored on the operating card
(`docs/agent-operating-card.md`); when you add or retire an entry, re-check the card. Retire
entries whose underlying cause is fixed structurally (e.g. by a lint or a raise).

---

## AP-1 — Sampler-knobs-first on slow MCLMC

- **The tempting move:** MCLMC mixes slowly or R̂ is terrible → tune step size, L, integrator,
  adaptation schedule.
- **Why it fails here:** in the cases diagnosed so far, slow mixing was dominated by the
  *initialization*, not the sampler: an under-converged MAP start means burn-in is spent
  migrating toward the typical set (long winding cornerplot tracks, huge R̂), and no knob fixes
  that. Curved/nonlinear degeneracies in the posterior can then limit ESS even after the start is
  fixed.
- **Correct first move:** verify MAP convergence (loss plateaued? χ²/ν at MAP sane? particles
  actually moving?) and look at the degeneracy geometry in a cornerplot *before* touching any
  sampler hyperparameter. `/diagnose-sampling` encodes this order.
- **Real instance:** `docs/logs/carousel-mclmc-sampling.md` — original "R̂≈70, chains crawling"
  was dominated by an under-converged MAP; a 4000-step MAP took the run from broken to merely
  ESS-limited. Diagnosis status: proposed (UNCERTIFIED); artifacts in
  `experiments/sim_carousel/_h1h2_diag/`.

## AP-2 — Trusting a good χ²/ν as evidence of correct model specification

- **The tempting move:** the fit converged and χ²/ν ≈ 1–1.3 → declare the model fine and move on.
- **Why it fails here:** the source models are flexible (high-`n_max` shapelets, hundreds of
  lstsq amplitudes); they will absorb a misspecified forward model — wrong PSF, wrong noise,
  wrong grid — into an unphysical source and still fit the data well. Goodness-of-fit is blind
  to exactly the failure that matters.
- **Correct first move:** read the **model card** printed/saved by `Pipeline.run`
  (PSF / noise / grid / precision) before trusting any run; where ground truth exists, check
  parameter recovery, not fit quality; look at the source-plane plot for physicality (does it
  look like a galaxy?).
- **Real instance:** run silently modeled with `psf=None` after a wrong source-dir path hit a
  loader fallback; χ²/ν≈1.25 hid it. Led to `tools/lint_silent_defaults.py` +
  `tests/test_no_silent_scientific_defaults.py`. See `docs/logs/`.

## AP-3 — Blaming noise, conditioning, or multimodality for slow mixing before checking initialization

- **The tempting move:** sampling struggles → hypothesize an exotic cause (likelihood noise
  floor, ill-conditioning, multimodal or highly curved posterior) and design experiments around it.
- **Why it fails here:** the exotic causes are *sometimes* real (see AP-4 for a genuine noise
  floor) but they are rarer than the mundane ones, and each costs an expensive run to chase.
  In the carousel diagnosis, noise / conditioning / multimodality were each explicitly ruled
  out; the causes were an under-converged start plus slight curvature exacerbated by strong degeneracy
  from a poorly chosen NFW parameterization.
- **Correct first move:** exhaust the cheap mundane checks (init convergence, model card,
  precision flags, trace plots) before designing an experiment around an exotic hypothesis —
  and when you do test one, pre-register the falsifier so the hypothesis can actually die.
- **Real instance:** `docs/logs/carousel-mclmc-sampling.md` (status: proposed/UNCERTIFIED).

## AP-4 — Running float32 at high `n_max` because it's faster

- **The tempting move:** halve memory / speed up on Ampere by running the likelihood in float32.
- **Why it fails here:** float32 leaves a basis/convolution noise floor that breaks
  high-`n_max` shapelet sampling — MCLMC adaptation collapses. The failure is not loud; it
  shows up as mysterious sampler behavior far downstream of the actual cause.
- **Correct first move:** float64 is the project default and needs *both* settings
  (`jax_enable_x64` + `SimulatorConfig(likelihood_precision="float64")`) — see
  `docs/project-standards.md` §8. If speed is needed, move only the PSF convolution to
  `conv_precision: float32`. Datasets generated before precision was persisted load as float32
  — regenerate them.
- **Real instance:** MCLMC adaptation-collapse diagnosis (June 2026); standards §8.

## AP-5 — Reporting aggregate convergence statistics

- **The tempting move:** report mean R̂ or total ESS across parameters (they look better).
- **Why it fails here:** convergence claims are gated by the *worst* parameter — one
  unconverged degeneracy direction invalidates the joint posterior. Aggregates are a blind spot
  by construction.
- **Correct first move:** report max(R̂) (rank-R̂ where available) and min(ESS), and name which
  parameter is worst — the identity of the worst parameter is itself diagnostic (it usually
  points at the degeneracy).
- **Real instance:** When implementing LAPS, an agent cited a median sample spread of ~18x spread relative 
  to the true posterior, when the maximum was ~300x, the true width of the prior

## AP-6 — Suggesting Exotic Sampling Algorithms as a Solution to Sampling Issues

- **The tempting move:** when a given sampling algorithm (usually MCLMC), suggest switching to a
  more complex, not-yet-implements sampling algorithm (normalizing flows, annealing, parallel tempering)
- **Why it fails here:** it's very rarely truly an insurmountable issue with the sampler. MCLMC is very efficient
  and well-validated, and it's most often a preconditioning issue, or something with the bijectors. Additionally, 
  getting a new sampling algorithm to be at all functional on lensing posteriors is an involved and difficult process. 
  It's a rabbit hole that's not worth going down to get a single system to sample.
- **Correct first move:** invoke `/diagnose-sampling` for more details on how to improve sampling performance.
- **Real instance:** Much of the Carousel minimal example sampling attempts inolved making substantive modifications to
  MCLMC in order to attempt to get it to sample. None succeded, and most made things worse.

---

## [FILL IN: your entries — the highest-leverage writing you can do here]

Seed questions to mine your own experience (delete once used):

- **What do agents keep suggesting that you always veto?** Each veto you've issued twice is an
  entry. (E.g.: switching samplers instead of diagnosing; adding SVI stages; shrinking the
  model to make it converge; masking "problem" pixels.)
- **What diagnostic do you always run first that agents run last (or never)?** That ordering
  *is* an anti-pattern entry, written from the other side.
- **Which parameter degeneracies are known and expected** (e.g. Einstein-radius/slope,
  shear/ellipticity, source-size/magnification)? An agent that doesn't know them will
  misread a healthy posterior as pathological — or design a run to "fix" physics.
- **What has burned you in data handling?** Units, pixel-scale conventions, PSF normalization,
  noise-map definitions, mask semantics — each past incident is an entry.
- **Which quick fixes are scientifically unacceptable here** even though they'd make the metric
  move (e.g. widening priors to fix divergences, dropping chains, thinning to hide
  autocorrelation)?
