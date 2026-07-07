# Open Topics — parked ideas, not active work

A parking lot for ideas that were considered, judged not ready, and deliberately tabled — with
enough context that a future agent (or the human) can pick one up without re-deriving the
reasoning. This is *not* a research-area lab notebook: nothing here is active, and nothing here
should be started without the human explicitly re-opening it.

**Entry format:** what the idea is · why it was tabled (the actual objection, not a
euphemism) · what "ready to revisit" looks like · provenance.

---

## OT-1 — Numerical cornerplot sidecar ("corner card")

- **The idea:** emit a machine-readable stats sidecar alongside every cornerplot, so agents
  read numbers as their primary channel and use (chunked, ≤4×4) plot crops only for
  verification. Candidate contents: correlation matrix with top-|r| pairs named; per-marginal
  skew/excess kurtosis; per-chain mean separation in σ (multimodality proxy); GMM k=1 vs k=2
  BIC on the worst pairs; a quadratic-ridge coefficient on the top-correlated pairs
  ("banana index" — would give a definitive number for the recurring curvature debate);
  inter-stage Mahalanobis migration (MAP→SVI→MCMC); inverse-mass-matrix vs sample-covariance
  generalized-eigenvalue ratios.
- **Why tabled (2026-07-01, human):** direct bad experience with numerical diagnostics
  substituting for eyeballing cornerplots — in high-dimensional and/or under-converged cases,
  agent-invented numerics were unhelpful and repeatedly led to *wrong* conclusions. Trust has
  to be earned per-statistic, not assumed. Until the human has a good sense of how each number
  corresponds to what they see, numbers-as-primary is a fooling-yourself risk, not a rigor
  improvement.
- **Ready to revisit looks like:** a calibration study, not a tool build. Assemble a library
  of cornerplots the human has already read and graded (converged and unconverged, low- and
  high-D, including known past misreads); compute the candidate statistics on all of them;
  adopt *only* the statistics whose verdicts match the human's readings across the library,
  and record each adopted statistic's blind spot (method-discipline §4). Statistics that
  disagree with the human's eye on under-converged cases are rejected regardless of how
  principled they look. Until then: agents use chunked ≤4×4 cornerplots per
  `docs/inference-diagnostics.md` and the honesty rules for plots they can't read.
- **Provenance:** conversation 2026-07-01 (cornerplot-legibility discussion; the ≤4×4 chunking
  rule was adopted immediately, this half was parked). Related: vision-input downsampling
  makes >4×4 cornerplots unreadable by agents — that constraint is permanent context for
  whoever revisits this.

---

*(Add new entries above this line. When an entry is re-opened, move it to a proper lab
notebook log and leave a one-line tombstone here pointing to it.)*
