---
name: diagnose-sampling
description: Ordered diagnostic workflow for a misbehaving inference run — MCMC/MCLMC/HMC mixing slowly, bad R-hat or ESS, frozen chains, suspicious posteriors, or a fit that "converged" but looks wrong. Use when diagnosing sampling or fit quality in gigalens. Encodes the project's cheap-mundane-checks-first ordering; produces UNCERTIFIED findings for the lab log, not fixes.
---

# Diagnose a sampling / inference run

You are diagnosing, not fixing. The deliverable is a **cause diagnosis with evidence**, written
to the area's lab-notebook log as `proposed (UNCERTIFIED)` findings. Do not change sampler
hyperparameters, model structure, or code as part of this workflow — propose changes at the
end, each with the pre-registration triplet (hypothesis / predicted direction+magnitude /
falsifier), for the grader to approve.

**Ordering is the point.** The steps go from cheap-and-usually-guilty to
expensive-and-usually-innocent. Do not skip ahead to an exotic hypothesis (noise floor,
ill-conditioning, multimodality) before the mundane checks are done — see `docs/anti-patterns.md`
AP-1 and AP-3 for real instances where that wasted significant compute.

Read first if not already in context: `docs/inference-diagnostics.md` (plot reading),
`docs/anti-patterns.md`, the area's log in `docs/logs/`, and — for MCLMC internals —
`.claude/mclmc.md`.

For the deeper *why* — a distilled catalog of the posterior pathologies this project has actually
characterized (the self-consistency/subgrid-comb likelihood defect, the init trap, float32-conv
numerics, curved-valley coordinates, and the marginal-vs-conditional funnel), each instrument's
plot-reading guide, and the registered-metric design lessons — see
`docs/playbooks/sampling-diagnosis-playbook.md` (status: proposed / UNCERTIFIED). Its clone-gap /
D3 / seed-band triage extends Steps 1–2 here; its disease catalog is where a diagnosis lands once
these ordered steps localize the fault.

## Step 0 — Model card and precision audit (minutes, catches the worst failures)

- Read the **model card** printed/saved by `Pipeline.run` (`inference_utils.model_card`):
  PSF present and correct? Noise model? Mask? Pixel grid? Units?
- Precision: `jax_enable_x64` on **and** `likelihood_precision="float64"`? (Both are required;
  one without the other silently degrades or raises — `docs/project-standards.md` §8.) Datasets
  generated before precision was persisted in `meta.json` load as float32 — check.
- If anything here is wrong, **stop**: everything downstream is diagnosis of a misspecified run.

## Step 1 — Numbers, classified

Compute and report, in this exact form: **max(R̂)** (rank-R̂ where available), **min(ESS)**
(bulk and tail), **χ²/ν** at the relevant stage, and *which parameter is worst* (its identity
is diagnostic — it usually points at the issue or degeneracy). Classify each against the decision table
on the operating card (`docs/agent-operating-card.md`). State the gap from target in meaningful
units and the regime: **structurally wrong vs. fine-tuning**. Never report aggregate/mean
convergence stats (AP-5).

## Step 2 — Initialization / MAP convergence (the usual culprit)

Before any sampler hypothesis: is the chain *start* any good?
- MAP loss plateaued, or still descending when it stopped? χ²/ν at MAP sane?
- Any NaN-frozen particles (particles not moving at all)?

## Step 3 — Preconditioning/Inverse Mass Matrix

Does the inverse mass matrix (interpreted as a covariance matrix) match the sample spread?
- If it's too broad in a dimension, samples likely get slung out of the typical set, or the step size will plummet to compensate.
- If it's much too narrow, samples will move of their initialization region slowly, and mixing will be poor.

## Step 4 — Trace plots

Classify the failure visually: **frozen** (chains barely move) vs. **slow mixing** (chains
separated, blending poorly) vs. **mode-hopping** (discrete jumps with good within-mode mixing).
Each points to a different cause family; say which you see and attach the plot paths.

## Step 5 — Cornerplot geometry

- Only ever plot cornerplots with at max 4 selected parameters, (worst R̂, worst ESS, or parameters of interest). 
  Cornerplots with all parameters are too dense for your vision model to read.
- Reference `inference-diagnostics.md` for how to read cornerplots and what to look for.
- Overplot inference stages (MAP → surrogate → MCMC). Multiple-σ migration between stages
  means the earlier stage is untrustworthy, not that the later one is broken.
- Compare the inverse mass matrix (as a Gaussian at the sample mean) against the sample spread —
  a large mismatch implicates adaptation.
- Multimodality: check 2-D contours (easier than 1-D histograms), per-chain histograms, and
  Step-3 hop signatures *before* claiming it.

## Step 6 — Physicality

Source-plane plot (unconvolved): does the source look like a galaxy? Parameter values
physically plausible (masses, sizes, no net-negative intensities)? An unphysical source with a
good χ² means the mass model or forward model is absorbing misspecification (AP-2) — that is a
finding about the *model*, not the sampler.

## Step 7 — Write the record

Append findings to the area's log: what was checked, what was ruled out (with evidence paths),
the proposed cause(s) marked **UNCERTIFIED**, and proposed next runs as design-checkpoint
drafts. Negative results (checks that came back clean) are results — log them so the next
agent doesn't repeat them.

---

## Standard steps for diagnosing
 
1. The first thing I look at for a run is Rhat. It tells me whether I'm in the converged regime or not, and which parameters are the worst-mixing. 
  These are typically the parameters that I focus my attention on in later steps.
2. Then, I look at the model specifications (prior, model components, simulator config, etc.), checking for any obvious mis-specifications or unphysicality.
  A bad model is one of the most common causes of poor sampling performance. And if the model isn't valid, then the sampling has no meaning, even if converged.
3. If Rhat is bad and further diagnosis is needed, I then look at a cornerplot (see  `inference-diagnostics.md` for how to read cornerplots), focusing on the worst-mixing parameters and those I care about scientifically.
4. To confirm either migration or multimodality, I look at trace plots for the parameters I've identified. This gives a time series history that the cornerplot doesn't.
5. If my chains are frozen, I check the MCLMC step size to make sure there wasn't a catastrophic drop in step size that caused the chains to freeze.
6. If all this fails to identify the cause, I look at the modeled image, source-plane plot, and normalized residuals. Gross model misfitting will show up here.

