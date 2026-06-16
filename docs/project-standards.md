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

- Every reported metric carries an uncertainty estimate (bootstrap CI, posterior credible interval, or equivalent). Point estimates alone are insufficient.
- [FILL IN: domain conventions — e.g. prefer fitting with a likelihood/NLL so you get calibrated per-result uncertainty for free.]

## 5. Reproducibility

- **Seeds** recorded (all RNGs). **Versions** pinned (code, data snapshot, key dependencies). **Inputs addressable** (commit hash / checksum / dataset-version tag — not "the data at `/some/path` as of today"). **Configs saved next to results.**
- [FILL IN: any domain-specific reproducibility needs — hardware, solver tolerances, pipeline/instrument versions.]

## 6. Negative and null results are recorded

- A failed experiment is a result; file it in the relevant lab-notebook log. A control/baseline that beats your method is the *most important* thing to report, not to hide. If you abandon an approach, record *why* so it isn't repeated.

## 7. Failure modes to actively watch for - Update as needed

List this project's known traps — if any occurs, stop and reassess:

- [FILL IN: failure mode 1 — e.g. "the result is driven entirely by <confounder/nuisance parameter>"]
- [FILL IN: failure mode 2 — e.g. "performance correlates with <something that shouldn't matter>"]
- [FILL IN: …]

## 8. Other Standards - Update as needed

[FILL IN: standards specific to this field — e.g. physical sanity checks, units/conventions, instrument or noise models, known identifiability quirks, symmetries to respect. General method discipline does not go here; it lives in `method-discipline.md`.]
