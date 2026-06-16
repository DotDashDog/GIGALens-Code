# MCLMC fix log (2026-06-14) — incremental, gated

Plan: `~/.claude/plans/valiant-hatching-kettle.md` (FIX PLAN section). Baseline = commit `71357bd`.
Decisions: mixed precision (float64 likelihood only), flag-gated in `gigalens`, model layer first.
Gates: G1 = d3/fix1 EEVPD + logp accuracy; G2 = d4_d5 synthetic gauntlet; G3 = repro runs a/b/c.

## Fix 0 — precision-mechanics spike (DONE, 2026-06-14)
Script: `fix0_precision_spike.py`. **Verdict: mixed precision feasible; falsifier did NOT fire.**
- dtype mechanics under x64: float32 ops stay float32; weak python floats don't promote;
  `.astype(float64)` gives TRUE float64; a sum after cast is float64. ✓
- simulator coordinate grids `img_X/img_Y` are float32 under x64. ✓
- BUT the unmodified log_prob is float64 under x64 — the basis promotes via the untyped
  `jnp.zeros((0, *img_X.shape))` seed (`simulator.py:163`); the shapelet light model already pins
  its heavy precompute to float32 (`shapelets.py:74-75`). One-line float32 pin on the seed keeps
  the basis float32. NOT an escalation.
- Memory: worst case (current unpinned float64 basis) at n_max=25, mem_fraction 0.75, ran on GPU
  with NO OOM (exit 0). The float32-pinned mixed version is lighter → comfortably in budget. ✓

## Fix 1 — float64 likelihood (flag-gated), IN PROGRESS
**Edits (step 1a, basis pin + float64 reduction):**
- `gigalens/simulator.py`: `SimulatorConfig.high_precision_likelihood: bool = False` (opt-in).
- `gigalens/jax/simulator.py`: `LensSimulator.high_precision` read from sim_config; basis seed
  pinned to `self.img_X.dtype` (unconditional, no-op without x64).
- `gigalens/jax/model.py`: `BackwardProbModel.log_prob` — when `simulator.high_precision`, compute
  the ~40000-pixel Gaussian log-lik in float64 (replicates tfd.Independent(Normal).log_prob).
- NOT yet done: 1b (float64 gram/solve in `_weighted_lstsq_reconstruct`) — add only if the
  anchor-3 device cliff survives reduction-only float64.
**Card.** Hypothesis: float64 logp removes the ulp staircase (benign floors) and, with 1b, the
anchor-3 cliff. Prediction (G1'/fix1_gate): mixed logp == full-float64 logp to ~f64 ulp at benign
anchors; ray staircase (few unique values in float32) becomes smooth (hundreds of unique values);
anchor-3 |mixed-full64| large under 1a-only (motivates 1b) → ≤ulp after 1b. Falsifier: mixed ==
float32 (reduction not actually float64 / formula wrong) or benign floor unchanged.
**Results (fix1_gate, GPU):**
- Formula check: my float64 Normal reduction == tfd.Independent(Normal).log_prob to diff=0.0. ✓
- **Reduction-only float64 (float64 forward + float64 reduction, "mixed v1"):** logp smooth, matches
  full-float64 to ~4e-3 at ALL anchors INCLUDING anchor 3 (3586 cliff → 4e-3). Staircase gone
  (ray 6→400 unique). BUT: under x64 the bijector emits float64 params → the WHOLE forward model
  is float64 → bootstrap MAP (bs=100) design matrix is f64[100,40000,352] = 11 GB → **OOM**. So
  "mixed v1" is really full-float64 forward; not memory-saving.
- **True mixed (float32 forward + float64 reduction, "mixed v2"):** memory-safe, but **the plot
  beats the metric** (method-discipline §5): ray "400/400 unique" looked smooth, but the figure
  shows the float32-forward logp is JITTERY (~±0.02) at the frozen anchors — float32 im_sim carries
  its own noise once the reduction is f64. Anchor-3 only partly recovered (156276→159565, still 297
  off). HOWEVER the bootstrap anchor (where a healthy chain lives) is clean (0.007). So the
  forward-precision noise is concentrated at the pathological frozen positions.
- **Decision:** the absolute fix needs float64 FORWARD too (reduction-only is insufficient at
  frozen anchors). But the memory hit only bites the bootstrap MAP (bs=100), not the sampler
  (8 chains/1 GPU fits in f64). Whether float32-forward is "good enough" depends on whether a
  healthy chain stays near the (clean) bootstrap → **decided by G3 (repro run-a).** [running]
- 1b (float64 gram/solve) confirmed UNNECESSARY for the value: reduction-only already moved anchor-3
  from 156276 to 159862 (v1); the solve is innocent (E2 vindicated).

Artifacts: `diagnosis_2026-06/fix1/` (fix1_gate.png, dump_{float32,full64,mixed}.npz, summary.json).

**End-to-end wiring (2026-06-14, for user notebook test):**
- `gigalens/jax/model.py`: high_precision → cast params to float32 for the forward model (basis/
  conv/solve stay float32, ~baseline memory/speed), reduction in float64.
- `System.high_precision_likelihood` (system.py) → flows through `system.sim_config` to BOTH the
  bootstrap (`VelaBootstrapQzStage` uses `system.sim_config`) and the sampler
  (`MCLMCStage→MCLMC_JIT→model_seq.sim_config`). One setting on the system enables everything.
- `pipelines.py derive_artifacts`: qz loc/scale_tril dtype-consistency (latent x64 bug, fixed).
- `gigalens/jax/simulator.py`: warns if high_precision set but x64 off (silent-truncation guard).
- Memory/compute: mixed keeps the forward float32 → ≈ baseline. Full float64 would be ≈2× memory
  AND ≈2× compute (A100 FP64 ≈ ½ FP32; bytes double) — confirmed by the 5.6→11 GB bootstrap matrix.
  Under x64 the sampler STATE/dynamics/reduction are float64 (dim-17, negligible); only the
  expensive forward is float32. So mixed = float64-clean signal at float32 cost.
- **Status: awaiting user's TestNewAPI.ipynb run** to answer the open science question (does a
  healthy chain stay in the clean region, or does float32-forward im_sim noise re-collapse it).
  First full-sampler run under x64 — known dtype landmine (qz) fixed; others may surface in the
  adaptation and would be quick fixes.

**x64 dtype-mismatch in the adapter (user notebook, 2026-06-14):** `lax.cond` in the mass-matrix
adaptation raised "branches must have equal output types" — Welford `.mean/.m2` float64 (true) vs
float32 (false). Root cause: the bootstrap qz is float32, but under x64 `qz.sample()` promotes the
chain positions to float64 while `qz.mean()`/`qz.covariance()` stay float32 → the Welford built
from `svi_mean` (float32) clashed with the Welford built from float64 positions. Fix
(`full_mclmc_with_adapt_sharded`, mclmc.py): canonicalize the state + params to one dtype.
**CORRECTED after CPU reproduction (`fix1_dtype_test.py`):** `qz.sample()` is unreliable — it can
return float32 positions even when the energy is float64 (and the toy test showed exactly that).
So the position dtype is the WRONG canonical; the **energy/logdensity dtype** must drive everything.
Final fix keys off `state_init.logdensity.dtype` and casts the WHOLE initial state (position,
momentum, grad) AND all params (inverse_mass_matrix, step_size, L) to it → scan carry uniformly the
energy dtype. The actually-failing leaf was `step_size_max` (float64, from jnp.inf) vs
`step_size*0.8` (float32) in blackjax handle_nans. Verified by `fix1_dtype_test.py` (toy float64
logdensity + float32 qz = the exact notebook mix): sampler runs to completion, ALL GOOD. No-op
without x64 (baseline unchanged).

**Second x64 dtype-mismatch — initial momentum (user notebook, 2026-06-14):** `lax.select` inside
blackjax `handle_nans` raised "same dtypes, got float32, float64". Root cause: `_single_init` sets
`momentum=generate_unit_vector(rng_key, position)`, and `jax.random.normal` defaults to float32
even under x64, so the initial momentum was float32 while the kernel produces float64 momentum.
Fix (`_single_init`, blackjax_updated_utils.py): cast the initial momentum to the position dtype.
Verified the per-step noise (`blackjax.partially_refresh_momentum`) samples with `dtype=m.dtype`,
so float64 propagates through every step once the init is float64 — class is closed: every scan-
carry leaf is now float64 under x64, and no in-loop random source reintroduces float32.

**Validation (2026-06-14):** (1) CPU toy `fix1_dtype_test.py` (float64 logdensity + float32 qz =
the exact notebook dtype mix) — sampler runs to completion, ALL GOOD. (2) End-to-end GPU smoke,
REAL model n_max=10, high_precision, nb=40/nr=10: no dtype errors, no OOM, healthy adaptation
(eps 1.4→2.2 stable, 0 rejections, MM windows PSD). Artifacts: `fix1/fix1_smoke_n10/`. Repro-script
fix: set `system.high_precision_likelihood` (not `model_seq.sim_config`) so the bootstrap — which
reads `system.sim_config` — also runs float32 forward (the notebook already does this). **Fix 1
plumbing is complete and error-free end-to-end; the n_max=25 science test (does float64 likelihood
stop the collapse) is the remaining G3 run.**

## Fix 1 G3 SCIENCE RESULT (2026-06-15) — mixed (float32-forward) FAILS; full float64 forward FIXES it
User ran n_max=25 high_precision in TestNewAPI.ipynb → **still collapses** (milder: eps→~1e-4 vs
1.67e-5; tune1 eps crash before MM adaptation; window-3 non-PSD). Falsifier branch of the Fix-1
card hit ("run-a still collapses with truthful EEVPD → controller alone → motivates Fix 2") — BUT
the diagnostic below shows it is NOT the controller; it is the **float32 forward model**.

**Diagnostic D-fix1 (diag_scale sweep under high_precision, GPU; `fix1_diag_sweep.sh` +
`analyze_diag_sweep.py`; dumps `fix1/diag_sweep/`):**
- n_max=25 high_precision (mixed: float32 forward + float64 reduction), diag_scale ∈
  {1e-4,1e-6,1e-8,1e-10}: **ALL freeze in tune1** (step_norm ~1e-12), then windows go **non-PSD**
  (n_neg up to 6 at window 2/3). Init metric swept over **6 orders of magnitude changes nothing**.
  → **H-A (init-metric mis-scaling) FALSIFIED.** diag_scale is not the lever.
- n_max=10 high_precision, diag_scale=1e-8 (control): **healthy** (eps 1.4→7.5→1.93, step_norm
  ~1e-2, all windows PSD). Same dim=17, same metric, same everything except n_max.
- **n_max=25 FULL float64 (forward+reduction, `--x64` only, diag_scale=1e-8): HEALTHY.** eps
  1.31→7.64→0.53, step_norm ~1e-2, 0 rejections, ALL windows PSD (n_neg=0, cond ~1e6),
  first_nonpsd_step=None. Identical quality to n10.

**Verdict.** The freeze driver is the **float32 FORWARD model** (`im_sim`), not the controller and
not the init metric. The float32-forward logp jitter (~±0.02 at frozen anchors, flagged in
fix1_gate) gives dE~0.02–0.09 → xi~1 at small eps (see n25 mixed tune1 xi₉₀=1.08) → seeds the eps
crash → freeze → degenerate frozen-chain Welford → non-PSD windows. **Fix 1's reduction-only
float64 is insufficient; the forward (basis/conv/solve) must be float64 too.** The n25 likelihood
is NOT intrinsically pathological in float64 (gram cond ~1e16 notwithstanding) — full f64 adapts
cleanly, so the controller defects F1–F8 and the non-PSD windows were DOWNSTREAM symptoms of frozen
chains here (they remain real for other models per D4, but are not what kills vela n25).

**STRATEGY SHIFT (needs user decision — the "mixed/float32-forward" decision is overturned):**
mixed precision cannot work for n25 because the forward IS the noise source. Options:
  (1) full float64 forward, flag-gated (proven fix; ~2× forward mem/compute; sampler fits at
      map_samples≤16; bootstrap MAP is the OOM risk → keep it float32 or modest map_samples);
  (2) float64 gram/solve ONLY, keep basis+conv float32 (cheap — solve is 352×352) — UNTESTED;
      viable only if the jitter is dominated by the ill-conditioned normal-equations solve
      (cond~1e16) rather than basis/conv float32 rounding. One probe run decides it.
  (3) regularize the gram (Tikhonov) / cap n_max so float32 forward suffices (more invasive).
Recommendation: probe (2) before committing to (1) — it could give the cheap mixed result the
user wanted. Artifacts: `fix1/diag_sweep/{*_panels.png, sweep_summary.json, *.log}`.

## Solve-only probe (2026-06-15, user chose "probe solve-only first") — CHEAP PATH RULED OUT
Script `fix1_solve_precision_probe.py` (GPU): faithfully replicates the lstsq logp at the frozen
anchor in 3 modes (replication validated vs real `lstsq_simulate`: f32 maxdiff=0, f64 maxdiff=3e-9).
Measures logp jitter (RMS residual from a degree-4 fit) along rays. dE_target(xi=1)=0.0922.

| window | f32all | f64solve | f64all |
|--------|--------|----------|--------|
| 1e-3   | 7.55e-3 | 7.54e-3 | **7.08e-6** |
| 1e-4   | 6.41e-3 | 6.47e-3 | **1.19e-10** |

**f64solve ≡ f32all (identical), f64all kills the jitter (1e3–1e7× lower).** → the float32 jitter
lives in the **BASIS/CONV, not the gram/solve**. **Falsifier of the cheap-path hypothesis FIRED:
float64-solve-only is INSUFFICIENT.** The floor is ~6e-3 and **eps-independent** (same at window
1e-3 and 1e-4) — the flat noise floor the eps^6 controller cannot satisfy; its gradient (≈6e-3 /
1e-4 spacing ≈ 60) is what kicks the integrator (matches D5 gradient-ripple → collapse). Only **full
float64 forward** (basis+conv+solve) removes it; this is the proven fix (n25 full-f64 run adapts
cleanly). Cost is affordable: the full-f64 sampler run already FIT at map_samples=16/MEM_FRACTION
0.75; the bootstrap MAP (the OOM risk) can stay float32. Artifacts: `fix1/solve_probe/`.
**Decision needed: implement full float64 forward (gated; bootstrap MAP stays f32) — overturns the
original "mixed" precision choice.**

## Fix 1 FINAL + Fix 2a (F4) implemented & gated (2026-06-15, user approved)
**Flag (user: "sounds good as a flag in sim_config"):** replaced the boolean
`high_precision_likelihood` with a 3-way `SimulatorConfig.likelihood_precision` ∈
{"float32"(default), "mixed", "float64"}. `gigalens/jax/model.py` log_prob branches on it
(float32/mixed → float32 forward; float64 → float64 forward; mixed/float64 → float64 reduction).
Old bool kept as deprecated alias → "mixed". Threaded through `System.likelihood_precision` and the
repro (`--likelihood-precision`). Default "float32" is byte-identical to baseline (the cast to f32
is a no-op without x64); only the *accidental* "x64+no-flag → f64 forward" path changes, now
explicit.

**F4 (user: "STAN-like F4, behind a default-false MCLMC argument"):** added
`regularize_mass_matrix=False` to `MCLMC_JIT` / `full_mclmc_with_adapt_sharded`. When True,
`_regularize_cov` applies to EVERY window's sample covariance (not just window-1): symmetrize +
Stan/blackjax shrinkage `(n/(n+5))·cov + 1e-3·(5/(n+5))·I` + an eigenvalue floor (clip to the
shrinkage value) so the downstream cholesky never sees a non-PSD metric. Default False ⇒
byte-identical to baseline.

**Gates (both PASS):**
- **G2 (synthetic cond=1e8 honest qz, CPU):** baseline COLLAPSES (eps→1.99e5, kern_rej 0.90,
  windows n_neg=2/6/6 non-PSD). **+F4: eps stable ~3e-3, kern_rej=0.000, windows n_neg=0/0/0 PSD,
  chains moving.** F4 converts the catastrophe into stable adaptation.
- **G3 (vela n25 float64, GPU, --no-bootstrap to dodge the live-notebook GPU-memory OOM in the f64
  bootstrap MAP):** likelihood_precision="float64" reproduces a HEALTHY run (eps 0.02→0.19→0.48,
  0 rej, windows PSD). **+F4: also HEALTHY (eps 0.02→0.19→0.12, 0 rej, PSD) — NO regression**, and
  F4 yields *better-conditioned* metrics (window cond 1.7/2e3/7e4 vs 2e3/1e6/4e5; eigenvalue floor
  ~1e-6 vs ~1e-9). 
Repro-script bugfix: the `--no-bootstrap` qz had the same loc/scale_tril dtype mismatch as
pipelines.py (now cast scale_tril to loc.dtype). Artifacts: `fix1/flag_f4/`, `d4_d5/f4_cond1e8_*`.
**Status: Fix 1 (float64 forward via the flag) + Fix 2a (F4) complete, gated, validated.**
