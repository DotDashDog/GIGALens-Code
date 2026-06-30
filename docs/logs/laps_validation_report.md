# LAPS (Late-Adjusted Parallel Sampler) — GIGALens CPU validation report (Phase 5)

Implementation: `gigalens_research.inference.laps_late_adjusted` (+ `laps_core`). Validated on CPU/x64
known-answer targets. Predictions pre-registered in `laps-engagement.md` BEFORE the runs. proposer≠grader:
experiments built by one agent, adjudicated by the orchestrator, then **adversarially graded** — this report
is the post-grader revision; an earlier draft over-claimed and was corrected (see §7).

## 1. Internals (validated, equal weight to results)
Across T_iso / T_ill / T_corr / T_curve (`results/*/diagnostics.png`):
- Equipartition `D̃(t) → ~2/M` floor (equilibrates).
- EEVPD controller `EEVPD_obs/EEVPD_wanted → 1` after the transient.
- Switch ripeness logic fires as designed; window-min guard prevents firing before a full window.
- Phase-2 acceptance lands in **0.654–0.738** (target 0.70 MN2; not always inside ±3%, but the freeze
  LATCHES — sticky, ε held — verified).

## 2. Unbiasedness (the principled result test) — SUPPORTED, scoped
b² second-moment bias scales as the 1/M finite-ensemble floor with **no rising/plateau trend** over
M ∈ {512, 2048, 8192} on T_iso + T_ill (`results/mscale/summary.png`). b²_avg/(1/M) seed-means:
T_iso 1.45 / 1.55 / 0.92; T_ill 1.39 / 1.03 / 1.31 — flat, no plateau. b²_max runs ~3–11× the floor
(max-over-8-dim entitlement; the worst is 10.9× at T_iso M512 s0) with the same slope.
**Honest scope:** this shows *no detectable bias down to ~1e-4* over warm Gaussians at M ≤ 8192; it cannot
exclude a smaller residual. T_corr and T_curve were NOT M-scaled (single M). The lens posterior (higher-d,
non-Gaussian, stiff) is the real out-of-distribution test (Phase 6). We judge against the 1/M floor, NOT the
paper's 0.01 line — which at M=512 is only ~5× the floor and is crossed by pure noise (see §6).

## 3. The decisive correctness lever: the switch
- **EMAUS identity-`x_i` switch NEVER fires** on any bed with a near-zero-mean coordinate (`σ/μ` blows up):
  Phase 1 burns its full budget and returns biased *unadjusted* samples — exactly the real `laps.py`
  "mostly fails to converge" symptom. The paper `x_i²` switch fires promptly. Confirmed in unit tests, the
  smoke test, and every grid run.
- This is the single load-bearing fix for a broken LAPS.

## 4. Phase-2 corrects Phase-1 bias — EVIDENCED (F3, `results/phase2off/summary.png`)
Phase-2-off (unadjusted Phase-1 only) vs on, T_ill cold, by (schedule, C), final b²_avg:
| config | Phase-2 OFF | Phase-2 ON |
|---|---|---|
| emaus / C=0.1 | **1.6e-2** (worst) | 2.6e-3 |
| emaus / C=0.025 | 7.1e-3 | 2.3e-3 |
| paper / C=0.025 | 3.5e-3 (lowest OFF) | 1.7e-3 |
| paper / C=0.1 | 2.0e-3 | 2.9e-3 (≈, both at floor) |
**Phase-2 ON brings every config to the 1/M floor; OFF rises above it, most for EMAUS/large-C.** So:
- The Metropolis adjustment is what guarantees asymptotic exactness; its *value scales with the Phase-1
  unadjusted bias* (negligible on easy Gaussians, large on stiff/biased configs).
- **schedule/C are first-order for the unadjusted (Phase-1) bias** (paper/C=0.025 lowest OFF), second-order
  after Phase-2. ⇒ paper-faithful F(C·D̃) + C=0.025 is the choice that **minimizes the bias Phase-2 must
  correct** — margin for hard targets where Phase-2 may under-correct (few adjusted steps).

## 5. Warm-start safety (F1, `results/offqz/summary.png`) — nuanced
{warm, cold, off-qz, off-qz+persist2} × 10 seeds, T_ill, M=512:
- **Cold start is the robust choice: 0/10 failures, tightest b²_max, no early-switch possible.**
- Warm-good and off-qz are usually fine but off-qz has a **rare catastrophic early-switch** (1/10: b²_max
  2.5e-2 at switch margin ≈ 0). The self-calibrated switch + window guard *usually* waits for equilibration
  even from a bad qz, but a marginal (≈0-step) margin can slip through.
- **`switch_persist=2` is NOT a clean fix** (40% vs 10% by the noisy metric): it removes the one catastrophe
  but its later switch leaves fewer Phase-2 steps. Available, not recommended as default without higher-M
  confirmation.
- **Caveat (metric sanity):** at M=512 the `b²_max<0.01` success flag is NOISE-DOMINATED — even gold warm
  "fails" 30%. Fail-rate is not a reliable cross-cell discriminator at this M; use b² vs 1/M or larger M.
- **Handoff guidance:** for uncertain qz quality (carousel / hard posteriors) prefer **cold start**; warm is
  fine for good qz but monitor the switch margin `switch_index − steps_to_floor` (small margin = risk).

## 6. Defaults — mechanism + empirics
| param | default | justification (mechanism + which experiment) |
|---|---|---|
| switch observable | `x_i²` | §3: identity `x_i` never fires on mean-0 coords; `x_i²` is positive, mean≈Var → well-conditioned. DECISIVE. |
| switch_mode | `self_calibrated`, k=1.5 | D2: literal δ<0.01 unreachable at our M (floor √(2/M)); self-cal fires when drift ≈ M-noise floor. k=1.0 too strict (never fires); **k=3.0 PREMATURE on the stiff rotated cold bed (P6a: b²_max@switch 14× floor); k=1.5 ripe in both regimes → empirically justified.** |
| schedule | `paper` F(C·D̃) | F3: lowest unadjusted Phase-1 bias; ≈EMAUS after Phase-2. EMAUS flag retained. |
| `C` | 0.025 | F3: minimizes Phase-1 bias (emaus/C=0.1 is worst OFF); paper value, ablation-stable. |
| `α` | 2 | paper; EEVPD tracks, D̃→floor. |
| init | warm (default), cold (robust) | F1/E-C: warm fine for good qz; cold robust for uncertain qz. |
| precond | diagonal 1/Var @ phase boundary | T_corr (moderate corr): accept on target, b² at floor. Dense untested. |
| target accept | 0.70 MN2 / 0.90 MN4 | bisection lands 0.654–0.738; freeze latches. |
| N / L_proposal | 15 / 1.25·L_full | paper; partial-refresh recorded-not-applied (efficiency-only). |

## 7. Open items / limitations (status after P6a stiff-proxy validation)
- **RESOLVED — k.** P6a separated k on the stiff rotated cold bed: k=3.0 fires PREMATURELY (margin 0,
  b²_max@switch 14× floor), k=1.5 waits and is ripe. k=1.5 is justified at both ends.
- **RESOLVED (on correctness) — preconditioner.** On a rotated Σ (cond 1e4, max-corr 0.93), the DIAGONAL
  precond hits acceptance 0.70, marginal b² at floor, AND recovers the dense Σ off-diagonals to ~3.6–7.2%
  (noise level). The M-ensemble + exact MH make the metric an EFFICIENCY knob (≈20× smaller Phase-2 step on
  strong rotation), not a correctness one. Dense metric remains a pure-efficiency option for higher cond /
  tighter budgets.
- **Phase-2 correction — evidenced, conditional.** Phase-2 reduces bias exactly where Phase-1 is biased
  (EMAUS/large-C; premature-switch T_rot cold). On easy Gaussians AND the strong banana, Phase-1 already
  reaches the floor (incl. the banana's curvature, via a calibrated joint metric), so Phase-2 is correctly
  neutral. The lens on/off comparison is in the handoff checklist.
- **Still open (efficiency / for the real lens):** Phase-2 partial momentum refresh not applied (efficiency-
  only, still exact MH); dense precond as an efficiency upgrade; warm-start early-switch tail (rare, cold
  avoids it — monitor margin); multimodality OUT of scope. Beds were d≤12 Gaussians/banana — the higher-d,
  stiff, non-Gaussian lens posterior is the real test.

## 8. Verdict
A faithful, sharding-correct GIGALens LAPS, **unbiased to the finite-M floor on the CPU beds** (incl. a
rotated cond-1e4 target and a strong banana), with internals matching the spec and a well-posed self-
calibrated `x_i²` switch — the single most important correctness lever, confirmed mechanistically and
empirically. Phase-2's bias correction is evidenced; the diagonal preconditioner is correct (dense = pure
efficiency); k=1.5 is empirically justified. Handoff package (`experiments/laps_validation/handoff/`):
`run_laps` (validated defaults), ground-truth-free `diagnose` (D̃/EEVPD/switch-margin/freeze/split-R̂+ESS),
`compare_warm_cold` (two-init cross-validation), and an operator checklist. Ready for the user's real-lens
notebook run + final GPU validation.
