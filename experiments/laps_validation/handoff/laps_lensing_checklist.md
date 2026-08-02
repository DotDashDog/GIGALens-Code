# LAPS real-lens operator checklist

Concise operator guide for running the validated GIGALens LAPS sampler on a real
lens posterior from a notebook, and reading whether it converged **without ground
truth**. Companion code: `laps_handoff.py` (`run_laps`, `diagnose`,
`compare_warm_cold`); runnable mock demo: `laps_handoff_demo.py`.

Provenance of the defaults and guidance below: the CPU known-answer validation
(`docs/logs/laps_validation_report.md`) + the P6a robustness study.

## 1. The recommended call (validated defaults baked in)

```python
from laps_handoff import run_laps, diagnose, compare_warm_cold

# prob_model : gigalens scene ProbModel; prob_model.log_prob(z)[0] is log p(z)
# qz        : SVI surrogate (.sample((n,), seed) / .mean() / .covariance())

res    = run_laps(prob_model, qz, init_mode="cold")    # primary run
health = diagnose(res, out_png="laps_diag.png")        # ground-truth-free health
cmp    = compare_warm_cold(prob_model, qz, out_png="laps_warm_cold.png")  # cross-check
```

`run_laps` bakes in the validated config and you should NOT change it without
re-validating: `schedule="paper"` (F(C·D̃)), `switch="paper"` (x_i² observable),
`switch_mode="self_calibrated"`, `switch_k=1.5`, `C=0.025`, `alpha=2`, `N=15`,
target accept 0.70 (MN2, d≤200) / 0.90 (MN4, d>200). Tune only `num_chains`
(scale to your GPUs; more chains → lower 1/M floor), `num_unadjusted_steps`,
`num_adjusted_steps`, `seed`. **Start with `init_mode="cold"`** (see §4).

## 2. What each diagnostic should look like — PASS criteria

`diagnose(res)` prints a PASS/FLAG per check and writes a 6-panel PNG.

| check | PASS looks like | panel |
|---|---|---|
| `D_reached_floor` | D̃(t) decays and flattens at a late-time floor; final ≈ floor | (a) |
| `eevpd_tracking` | EEVPD_obs/EEVPD_wanted settles to ≈1 (within ~0.3–3×) after the transient | (b) |
| `switch_fired` | the x_i² switch fires **before** the Phase-1 budget (`switched=True`) | (a) red line |
| `switch_margin_ok` | `switch_index − steps_to_floor(D̃)` is comfortably positive (≳ phase1_len/20) | (a)/(f) |
| `p2_accept_on_target` | Phase-2 acceptance lands near target (0.70/0.90), within ~±0.05 | (c) |
| `p2_freeze_latched` | the bisection froze and **stays** frozen (sticky ε) by the end | (c) purple line |
| `rhat_ok` | between-subensemble split-R̂ < 1.01 on every param | (d) |

ESS panel (e): for LAPS the ensemble is **one sample per chain**, so the chains
are independent and ESS ≈ `num_chains` (M) when sub-ensembles agree; the operative
cross-chain check is **R̂ ≈ 1**, not an autocorrelation ESS. ESS here = M / R̂²
(a between-group efficiency), reported only to flag heterogeneity.

## 3. RED FLAGS (and what they mean)

- **Switch never fires / `switched=False`** → Phase 1 burned its whole budget; the
  returned ensemble is the **unadjusted** Phase-1 state (biased). On a real lens
  this is the classic broken-LAPS symptom. Check that the x_i² observable is in use
  (it is, by default) and that `num_unadjusted_steps` is large enough to equilibrate.
- **Switch margin ≈ 0 or negative** → the switch fired *before* D̃ reached its floor
  = **early switch**; Phase 2 then starts from a non-equilibrated ensemble. Increase
  `num_unadjusted_steps`; prefer cold init; consider `switch_persist=2` (costs
  Phase-2 steps — not a free fix, see report §5).
- **Acceptance far from target** (e.g. <0.5 or pinned at 1.0), or **freeze never
  latches** → Phase-2 step-size tuning failed; ε is still chasing the ensemble (ECA
  bias). Give Phase 2 more steps; check the integrator order matches d.
- **D̃ never reaches a floor** (still descending at the end) → not equilibrated;
  extend Phase 1.
- **R̂ > 1.01 / warm and cold DISAGREE** → not converged. Two independent inits
  landing on different posteriors is the strongest ground-truth-free failure signal.
- **Joint structure off even if marginals look fine** → marginal means/stds can match
  while the *joint* shape (correlations / curvature) is wrong. `compare_warm_cold`
  checks cross-moments E[x_i x_j] (as a sampling z-score) for exactly this; on the
  CPU beds a banana-curvature diagnostic (`metrics.joint_shape_metrics`,
  E[x_0²x_1]=2b) is available if you have a known-answer surrogate.

## 4. Warm vs cold init (P6a guidance)

- **Cold is the robust default**: 0/10 failures in P6a, tightest b²_max, and
  **no early-switch is possible** (the ensemble starts diffuse and equilibrates into
  the switch). Use it for the carousel and any uncertain/hard posterior.
- **Warm is faster for a GOOD qz** (fewer Phase-1 steps to equilibrate) but carries a
  **rare catastrophic early-switch tail** when qz is off (1/10 in P6a, margin ≈ 0).
- **`switch_k=1.5` protects the cold-stiff regime**: k=1.0 is too strict (can fail to
  fire); k∈[1.5,3] were not separated on the CPU beds → 1.5 is the conservative pick.
- **Operating rule**: run cold as primary; run `compare_warm_cold` — **agreement of
  the two independent inits is your convergence evidence**. If warm’s switch margin
  is small, trust the cold run.

## 5. Known efficiency caveats (correctness is unaffected)

- **Diagonal preconditioner.** Phase 2 uses a diagonal `1/Var` metric. On a strongly
  **rotated/dense** posterior it cannot whiten the geometry → effective step size can
  be ~20× smaller than an aligned metric would allow (slower, **but still exact MH**).
  Strongly-correlated lens posteriors may want a dense metric / full-rank Hutchinson
  D̃ (untested; App. D claims diagonal ≈ full-rank).
- **MAMS partial momentum refresh not applied.** The in-tree kernel is the
  Hamiltonian MAMS variant (full refresh per trajectory; `L_proposal=1.25·L_full`
  recorded but **not** used). Efficiency-only: the sampler is still exact, but the
  paper’s efficiency may not be fully reproduced.
- **Joint/curvature bias.** On `T_banana_hi` (b=0.5, d=12, M=1024) the **unadjusted**
  Phase-1 ensemble already reproduced the curvature E[x_0²x_1]=2b to the 1/M floor, so
  Phase-2 had no detectable joint bias to correct (ON ≈ OFF). Phase-2’s value scales
  with the unadjusted Phase-1 bias — negligible on easy/curved-but-equilibrated
  targets, larger on stiff ones. Watch the joint cross-moments on the real posterior.
- **Success bar at small M is noise-bound.** The b²<0.01 flag is noise-dominated at
  M≈512; judge against the 1/M floor and use warm/cold agreement + R̂, not a single
  pass/fail.
