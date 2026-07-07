# Project Standards (project-specific) — GIGALens Inference and Modeling

This file holds the **project-specific** experimental standards. The general "don't fool yourself" method discipline lives in `method-discipline.md`; both are required reading. Fill each section with the standards for this domain, delete any that don't apply, and add domain-specific ones. Keep general method discipline *out* of this file so it stays portable.

---

## 1. A variety of diagnostics should be applied to every claim

Using a wide suite of relevant diagnostics (convergence statistics, cornerplots, resiudals, simulations, trace plots) ensures that you don't miss any glaring shortcomings in the modeling. Commonly used diagnostics are listed in `inference-diagnostics.md`. 

## 2. Statistical convergence is a must for any strong modeling claims
MCMC methods are the gold standard for inference in this project. Point estimates and variational surrogates have no guarantees and in my experience, can sometimes be highly biased. Claims derived from MCMC methods are only valid if the chains have converged (R-hat is the primary metric for this). See `inference-diagnostics.md` for more details on methods of determining convergence and diagnosing convergence failures.

## 3. Physicality
This is a physics project, so the models we use don't exist in a vaccuum. Physicality is a valuable check on whether modeling results are sane. 
Examples:
- If the source plane for a lensing simulation looks nothing like a galaxy, you are mismodeling the source.
- If you get physical parameters (halo masses, angular sizes, etc) that are obviously too high or too low, your model is also breaking down.
- Negative net light intensities, though mathematically valid, are inherently unphysical.

## 4. Uncertainty is reported, not hidden

- Every reported metric carries an uncertainty estimate (bootstrap CI, posterior credible interval, or equivalent). Point estimates alone are insufficient. This includes MAP on its own.
- SVI is not a valid measure of a posterior. It carries no convergence guarantees and no real metrics to verify its success.

## 5. Reproducibility

- **Seeds** recorded (all RNGs). **Versions** pinned (code, data snapshot, key dependencies). **Inputs addressable** (commit hash / checksum / dataset-version tag — not "the data at `/some/path` as of today"). **Configs saved next to results.**

## 6. Negative and null results are recorded

- A failed experiment is a result; file it in the relevant lab-notebook log. A control/baseline that beats your method is the *most important* thing to report, not to hide. If you abandon an approach, record *why* so it isn't repeated.

## 7. Failure modes to actively watch for - Update as needed

List this project's known traps — if any occurs, stop and reassess. The *tempting-but-wrong
moves* (what an agent will reach for by default and shouldn't) live in the companion playbook
`docs/anti-patterns.md`; the top headlines from both are mirrored on the always-injected
operating card (`docs/agent-operating-card.md`) — keep the card in sync when editing here.

- **Silent scientific defaults.** A missing/empty model input (PSF, noise model, mask,
  units, pixel grid, regularization, priors, `n_max`) must **raise**, never default to a
  degraded-but-plausible model. A flexible model (e.g. high-`n_max` shapelets, ~hundreds of
  lstsq amplitudes) then fits the data to a good chi² and the misspecification is invisible.
  *(Real instance: a run modeled with no PSF because the source dir path was wrong and the
  loader fell back to `psf=None`; chi²/ν≈1.25 hid it. See `docs/logs/`.)* Guards in place:
  (1) `tools/lint_silent_defaults.py` + `tests/test_no_silent_scientific_defaults.py` fail on
  any **new** silently-defaulting fallback in model-construction code (burn down the baseline
  in `tools/silent_defaults_baseline.txt`); (2) `Pipeline.run` prints and saves a **model card**
  (`inference_utils.model_card`) of the effective forward model — check PSF/noise/grid/precision
  there before trusting any run.
- **A good chi²/converged fit is not evidence the model is correctly specified** — flexible
  source models absorb misspecification. Verify against ground truth (recovery of known
  lens/source params), not just fit quality.
- [FILL IN: failure mode — e.g. "the result is driven entirely by <confounder/nuisance parameter>"]

## 8. Numerics: float64 is the gigalens default

gigalens runs in **float64** going forward. Two coupled settings are required — one
without the other silently degrades precision or raises:

1. **`jax_enable_x64` must be on.** JAX reads `JAX_ENABLE_X64` at import time. The
   `gigalens_research.simtests` package sets `os.environ.setdefault("JAX_ENABLE_X64","1")`
   on import, so the CLI / Slurm path (`python -m gigalens_research.simtests …`) gets it
   automatically. **Notebook/REPL users who `import jax` before importing the framework
   must call `jax.config.update("jax_enable_x64", True)` themselves** — otherwise the
   gigalens precision guard (`gigalens/jax/simulator.py`) raises.
2. **`SimulatorConfig(likelihood_precision="float64")`.** In simtests this is the default
   in the `vela_existing` generator and is **persisted to each system's `meta.json`**, so
   `run`/`plot` (which load via `System.load`) honour it. Override per-campaign with
   `dataset.likelihood_precision: float32 | mixed | float64` in the YAML.

Why: float32 leaves a basis/convolution noise floor that breaks high-`n_max` shapelet
sampling (MCLMC adaptation collapses). Cost: float64 is ~2× memory and slower on Ampere
FP64 units — use `conv_precision: float32` to move only the PSF convolution off the slow
FP64 path if needed (basis/solve/reduction stay float64).

Pre-change datasets (generated before precision was persisted) lack the `meta.json` keys
and load as float32 — regenerate them to get float64.

## 9. Other Standards - Update as needed

[FILL IN: standards specific to this field — e.g. physical sanity checks, units/conventions, instrument or noise models, known identifiability quirks, symmetries to respect. General method discipline does not go here; it lives in `method-discipline.md`.]
