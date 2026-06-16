# MCLMC adaptation-collapse diagnosis — interim report

Date: 2026-06-12. Baseline commit: `71357bd` (REVERT-MARK tweaks deleted + instrumentation:
`energy_change_raw`, `kernel_nonan`, `step_norm` added to the debug Hist).
Plan: `~/.claude/plans/valiant-hatching-kettle.md`. Protocol: `docs/method-discipline.md` +
user CLAUDE.md debugging rules (pre-registered prediction/falsifier per experiment).

Failing case: vela shapelets (vela01_cam12_rep03), n_max=25, dim=17 (sampled; shapelet amplitudes
lstsq-marginalized — n_max changes basis size/conditioning only: 66 → 351 components).
xi=1 energy-error target: |dE| = sqrt(17·5e-4) ≈ **0.092**.

## Status

| Diagnostic | Status | Artifacts |
|---|---|---|
| Phase 0 repro (runs a/b/c) | DONE — collapse reproduced | `run_{a,b,c}_*/` (summary.json, run.log, hist_*.npy) |
| D1 finiteness audit | DONE — all clean, H2 dead | `d1_d2/d1_results.json`, `d1_d2/d1_log.txt` |
| D2 noise floor | PARTIAL/UNTRUSTED — to be redone | `d1_d2/d2_f32_log.txt` (fragment) |
| D3 EEVPD(eps) curve | RUNNING (4 variants chained) | `d3/` |
| D4+D5 synthetic controller | DONE — falsifiers fired | `d4_d5/<exp>/`, scripts `d4_d5_synthetic.py` |
| D6 code audit | DONE — findings F1–F8 | this file (§D6); verification plot `phase0_eps_xi_traces.png` |

## Phase 0 — deterministic repro (runs a/b/c; nb=2000, nr=500, 8 chains, seed 0, ~95 s each)

- run (b) n_max=10, ds=1e-8: HEALTHY. eps→0.793, 0 rejections, step_norm ~5e-3, MM windows PSD.
- run (a) n_max=25, ds=1e-8: COLLAPSE, **zero NaN/rejections**. eps 7.27→1.67e-5 (47,000× below
  healthy), xi_q90(tune2)=2.87, step_norm→2.7e-7 (frozen).
- run (c) n_max=25, ds=1e-6: COLLAPSE via NaN cascade. 33% rejections in tune2 (xi_q50 pinned at
  the 1e-8 floor = zeroed-energy pollution), window-3 metric non-PSD (5 negative eigenvalues,
  cond 1.4e9) → cholesky NaN → 100% rejection, step_norm = 0, energy NaN.
- Anomaly (open): tune1 energy_change_raw RMS at n_max=25/ds1e-8 is 0.029 — ~10× SMALLER than
  n_max=10 (0.339). Contradicts naive "more noise at high n_max".

### Main-agent trace verification (plot: `phase0_eps_xi_traces.png`)

run (a) per-step history: eps = 9.22 at step 514 (window-1 boundary) → **0.29 at step 515**
(32× one-step drop, exactly at the metric swap; D6-F2 prediction confirmed). eps recovers after
swaps 1–2 (0.89 by step 1200); the window-3 swap (step 1203) is terminal: eps < 0.01 by step
1263 → 1.67e-5 by 1600. During the terminal decay (steps 1300–1599, eps spanning 1e-2→6e-5),
25% of steps still have xi>1 (q90=8.3): **xi is eps-independent ≥1 in this regime** (eps^6 would
predict ~1e-12×) — the flat-EEVPD signature in vivo. At the end, median xi → 1e-8 floor:
eps·velocity ~1e-7 is below float32 position resolution; the chain physically cannot move.

## D1 — finiteness audit (DONE)

Across n_max ∈ {10,15,20,25,30} × anchors {bootstrap, qz ds1e-6 samples, frozen run-a positions,
healthy run-b positions, prior draws}: **fraction of non-finite logp = 0.000 and non-finite
gradient = 0.000 everywhere** ("all_ok" in all 25 cells). Pre-registered expectation (mostly
clean, given run (a) collapsed with zero rejections) CONFIRMED; **H2 (static NaN
landscape/gradients at high n_max) is DEAD.** run (c)'s NaNs are dynamic (integrator overflow at
excessive eps), not landscape poisoning.

## D6 — adversarial audit of the implementation (DONE; reference = installed blackjax)

- **F1 (confirmed, permissive):** kernel zeroes energy_change on NaN and pre-reverts state, so
  blackjax `handle_nans`' step_size_max backoff is **dead code** (ceiling provably inert all
  run), and a rejected step enters the EWMA as xi=1e-8 with weight 0.123 → pushes eps UP.
  Fraction-dependent (see D5): minority-NaN still nets eps down; majority-NaN → explosive eps-up.
- **F2 (confirmed in vivo, collapse trigger):** at window boundaries the metric is swapped (here
  spanning ~5 orders of magnitude) while eps is kept and the EWMA is reset → next step's xi can
  be ~1e4–1e5 → eps slammed (`eps ← eps·xi^(−1/6)`); verified at step 515. Blackjax never swaps
  metric mid-stream without re-tuning eps; structural deviation.
- **F3 (confirmed by isolated CPU repro):** float32 cancellation in `welford_combine` cross-term
  yields genuinely negative covariance eigenvalues on near-degenerate input (same input in
  float64: PSD) → `cholesky(M⁻¹)` = NaN → 100% rejection. Exactly run (c) window 3.
- **F4 (confirmed, enabling):** only window 1 carries the SVI prior; windows 2/3 accumulate from
  empty with NO regularization (blackjax always adds Stan-style shrinkage·I). Per-step
  cross-chain update has rank ≤ 7 (8 chains).
- **F5 (latent):** step-size sync fires at step n1+n2 (mode-3 region), one region late.
- **F6 (latent):** L formula uses min-ESS and num_steps3·eps (blackjax: eps·mean(steps/ess)) —
  systematically over-estimates L; moot at collapse.
- **F7 (cleared):** dense-metric isokinetic integrator math is a correct generalization of
  blackjax's diagonal version (chol/cholᵀ usage, (dims−1) factors, `gr` is the velocity output).
- **F8 (confirmed analytically + in vivo):** the controller math matches blackjax EXACTLY, but
  has **no positive fixed point when xi is eps-independent and ≥1** — eps→0 by construction.
  The brittleness is structural (assumes smooth EEVPD), not a transcription bug.

## D4+D5 — synthetic controller testbed (DONE; scripts `d4_d5_synthetic.py` = permanent regression gauntlet)

- **D4 pre-registered prediction (all clean Gaussians adapt) FAILED — falsifier fired.**
  cond ≤ 1e4: clean at every init (6/6). **cond = 1e8: run-(c)-type collapse on a clean
  Gaussian** — window-1 sample covariance swaps in non-PSD (float32 Welford) at step 514,
  cholesky NaN at 515, 100% rejection, zeroed-NaN xi=1e-8 floor → eps runs UP to 1.99e5.
  The HONEST (true-covariance) qz init is the WORST case; tiny diag-1e-8 init is *protective*
  (eps merely 16× low, no NaN cascade) — inverts the naive H3(e) story (anomaly, unexplained).
- **D5 ripple ladders reproduce run (a) exactly** (eps down, ZERO NaN, xi90≈2–4, step_norm
  collapse), for BOTH value-ripple and gradient-ripple (interchangeable within ~15%).
  **Pre-registered δ threshold (≈0.092) FAILED by ~100×: collapse onset at δ ≈ 1e-3 ≈ 1% of the
  energy target.** Caveat: a high-frequency ripple of amplitude δ induces |dE| ≫ δ at moderate
  eps; the transferable quantity is the induced dE(eps) curve → D3 measures the real one.
- NaN-shell (minority NaN ~13–23%): eps still nets DOWN; xi floor pollution confirmed; explosive
  eps-up requires NaN-majority (matches run (c) phases and D4 cond=1e8).

## D2 — noise floor (PARTIAL, UNTRUSTED, to be redone)

Fragment before the job died: float32 "mean noise over 4 rays" ≈ 119–142 at two qz points
(n_max=25) — naively ~1300× above the 0.092 target. NOT trusted: polynomial-fit residual may be
dominated by genuine posterior curvature over the large increment range; no controls, no plots,
no float64 counterpart. Redo with audited fit method + plots (method-discipline §5).

## D3 — measured EEVPD(eps) (float32 DONE — decisive; float64 rerun in progress)

Plots + raw arrays: `d3/nmax{10,25}_float32_K64_eps25/` (`eevpd_*.png`, `dE_*.npz`, summary.json).
**There is NO slope-6 region anywhere in float32.** Every measured curve is a flat,
eps-independent noise floor at small eps that meets a steeply rising branch at large eps. The
eps^6 controller assumption holds only on the rising branch; below it, xi does not respond to
eps at all. The controller is stable iff the floor sits below the 5e-4 target so a crossing
exists. Measured floors (Var(dE)/dim units; target 5e-4):

| n_max | metric | anchor | floor | floor/target | 5e-4 crossing |
|---|---|---|---|---|---|
| 10 | run-b final | run-b late (healthy, in vivo) | 2.7e-6 | **185× below** | eps ≈ 0.8 — **matches run (b)'s settled eps 0.793** |
| 25 | 1e-8·I | bootstrap | ~1e-5–1e-4 | 5–50× below | eps ≈ 1–2 (workable → tune1 looked fine) |
| 25 | 1e-8·I | **frozen run-a positions** | **1.42e5** | **3×10⁸ ABOVE** | none — unreachable |
| 25 | run-b final (sane) | frozen run-a positions | 1.4e5 | 3×10⁸ above | none — floor is an ANCHOR property, not a metric property |

Anchor logdensities are all ~1.5–1.6e5 (float32 ulp there ≈ 0.016 → predicted ulp-noise
Var/dim ≈ 7e-6 — matches the BENIGN floors, i.e. the base floor is exactly float32 rounding of
logp). ~~The catastrophic 1.4e5 floor at the frozen positions is ~10 orders above ulp noise —
consistent only with lstsq-solve instability (cond~1e16 basis) making logp effectively a random
function of amplitude ~1e3 under machine-eps input perturbations there.~~
**CORRECTION (E-series, same day — see "E-series" section below): this attribution was WRONG.**
The 1.4e5 number is the pooled variance of four *deterministic, momentum- and eps-independent
per-anchor offsets* (+3.86/+4.20/+2.67/−3581) injected by the D3 harness's own state
construction (init logdensity from a standalone compilation vs the kernel's in-step
recomputation). The offset-corrected stochastic floors at the frozen anchors are ~1e-3–1e-2
(≈1e3–1e4× above healthy, crossing the target at eps≈2e-4–1e-2) — still collapse-inducing, but
via a different mechanism and magnitude. The "grad_norm = 1.1e8 at one frozen anchor" note was
not reproduced by E1b on CPU (all frozen-anchor |g| are 4.7e3–1.1e4) and is reclassified as a
GPU-kernel-context artifact pending E1c.

**Verdict: H1 CONFIRMED with a position-dependent structure, H4 CONFIRMED at the positions that
matter.** The causal chain for run (a): bootstrap neighborhood has a workable crossing → tune1
adapts ("healthy"); chains drift during tune2 into regions where the solve-noise floor is
astronomically above target → xi pinned ≥1 at all eps → F8 no-fixed-point → eps→0 → frozen; the
F2 metric-swap slams accelerate/trigger the descent. n_max=10 is healthy not because noise is
absent but because its floor stays 2 orders below target everywhere the chains go.
**float64 result (DONE): the eps^6 law is restored EVERYWHERE.** Fitted slopes 4.3–5.95 in all
10 combos (vs 0.1–0.7 in float32). At the frozen run-a positions with a sane metric
(`nmax25_float64/eevpd_run_b_final__run_a_late.png`): floor 2.3e-21 (= float64 ulp of logp²),
clean slope-5.07 rising branch from eps~1e-3, target crossing at eps≈0.46. The same positions
floored at 1.4e+5 in float32 — a **26-order-of-magnitude** difference. The catastrophic floor is
therefore *entirely float32-precision-driven*; the model is perfectly MCLMC-able in float64 even
where the chains froze. (Niche regime note: `diag_1e-8 × run_a_late` in f64 has NO crossing —
with that tiny metric the dynamics can't generate dE ≥ target at any eps ≤ 10; an over-small
metric makes the target unreachable from BELOW, another way the controller can mis-adapt.)

## PHASE 3 VERDICT

**Root cause (two interacting layers, both necessary):**
1. **Model layer (H1+H4 CONFIRMED, position-dependent; amplification mechanism CORRECTED by the
   E-series, see below):** in float32, single-step energy errors bottom out at a
   position-dependent noise floor instead of falling as eps^6. The base floor is benign ulp
   rounding of |logp|≈1.5e5. ~~near the n_max=25 chains' tune2 trajectories the ill-conditioned
   (cond~1e16) lstsq marginalization amplifies it to Var(dE)/dim ~1e5~~ → E2/E1/E1b killed the
   conditioning/value/gradient explanations; the honest offset-corrected floors at the frozen
   positions are ~1e-3–1e-2 (ulp-staircase noise, ~1e3–1e4× healthy), crossing the target only
   at eps≈2e-4–1e-2 — `desired_energy_var=5e-4` is unreachable at any usable eps there (H4
   stands, with corrected magnitude). On top of this, mixing differently-compiled logdensity
   evaluations injects deterministic fake dE offsets up to ~3.6e3. In float64 both effects
   vanish (offsets ~1e-11, floors ~1e-21) and the controller's assumption holds again.
2. **Adaptation layer (H3 CONFIRMED standalone):** the controller has no positive fixed point
   when xi flattens ≥1 (F8) and no brakes: metric swaps slam eps without rescaling (F2, verified
   at step 515), `handle_nans` is dead code and zeroed-NaN steps read as xi=1e-8 "perfect" steps
   (F1), windows 2/3 are unregularized and the float32 Welford can go non-PSD → cholesky NaN →
   permanent 100% rejection (F3+F4, reproduced in isolation and on a clean cond=1e8 Gaussian).
   Synthetic δ-ripple at ~1% of the target already collapses it (D5) — the controller amplifies
   rather than contains model noise.

**Causal chain for the observed failure (run a):** workable crossing near bootstrap → tune1 fine
→ chains drift into high-floor territory in tune2 → xi pinned ≥1 at all eps → eps→0 (no fixed
point) accelerated by window-swap slams → chains freeze below float32 position resolution →
(diag-1e-6 variant) degenerate window covariance → NaN cascade. n_max=10 is healthy because its
floor stays ~2 orders below target everywhere its chains go — not because noise is absent.

**Hypothesis scoreboard:** H1 CONFIRMED (position-dependent float32 noise floor, lstsq-driven at
high n_max) · H2 DEAD (landscape everywhere finite) · H3 CONFIRMED standalone (controller/metric
brittleness, 4 verified mechanisms) · H3(e) DEAD as cause (tiny init even protective) · H4
CONFIRMED at the positions that matter, float32 only.

**Dead ends ruled out (do not revisit):** static NaN gradients; dense-metric integrator math;
ESS port; window indexing; small-start-init as trigger; controller arithmetic transcription.

**Open residuals (minor):** run (a) tune1 ecraw anomaly (plausibly floor-position interplay, not
load-bearing); why tiny init is protective at cond=1e8 (D4 anomaly); F5 sync off-by-region and
F6 L-formula deviation (latent, not collapse-causing).

**Fix directions implied (NOT executed — separate plan, with the D4/D5 gauntlet as regression):**
the diagnosis points at (a) restoring a truthful EEVPD signal (float64 accumulation in the
likelihood/solve path, or a better-conditioned marginalization), and (b) de-brittling the
adaptation (eps rescale or re-tune window across metric swaps; regularize/shrink all windows;
PSD-project or float64 the Welford; make rejected steps shrink — not grow — eps; floor/ceiling
sanity on eps). Each candidate must state its predicted effect on: the D3 curves, the D5 ripple
threshold, the D4 cond=1e8 gauntlet, and the three Phase-0 runs.

## E-series — floor-mechanism localization (2026-06-12, post-verdict follow-up)

Motivating question (user): *why* does the float32 EEVPD floor appear, and can anything short of
global float64 lower or remove it? Three experiments (E2, E1, E1b) + raw-dE forensics. Artifacts:
`e2/`, `e1/`, `e1b/` (scripts `e2_cond_map.py`, `e1_stage_noise.py`, `e1b_grad_audit.py`).

### E2 — condition-number map (prediction FAILED — falsifier fired)

Pre-registered: cond tracks the 10⁹–10¹⁰× floor gap between bootstrap and frozen anchors.
Measured (float64 SVD of the 40000×352 design matrix A): cond(A) = 156 (bootstrap), ~2.9e3
(frozen run-a median), 2.8–6.8e16 (prior draws ONLY). cond(gram) = cond(A)² exactly (parity
1.000±0.015); float32 gram = float64 gram within 1–7% with zero negative eigenvalues at all
visited anchors. **Gram conditioning gap (336×) cannot explain the floor gap (2.4e10×) — off by
~10⁷.** The simulator's "cond~1e16" comment is a property of unphysical far-field positions,
not of anywhere the chains go. Plots: `e2/e2_money_plot.png` (floor vs cond — frozen cluster
sits 10¹⁰ right of bootstrap at only ~10² the cond), `e2_sv_spectra.png`, `e2_parity_cond_sq.png`.

### E1 — stage-wise value-noise attribution (prediction FAILED — informatively)

Pre-registered: f32-vs-f64 discrepancy erupts AT the solve and stays large downstream, only at
frozen anchors. Harness validated against library logp (2.9e-11 in f64; ≤1 ulp in f32).
Observed (`e1/noise_waterfall.png`): the solve coefficients DO erupt (rel err ~1e-2 frozen,
~5e-2 run-b-through-n25, 2–8 prior) **but chi²/logp recover to ulp level** — logp abs err at
frozen anchors 0.005–0.012 = 0.3–0.8 ulp, statistically identical to bootstrap. Reason:
least-squares optimality — coefficient error perturbs the VALUE only at second order. Value
eruption survives to logp only at prior draws (err ~1.9e3, the cond~1e16 territory).
**The float32 value pipeline is clean at every visited anchor; a 1.4e5 floor from value noise is
excluded by 10 orders of magnitude.** Ray plot (`e1/ray_run_a_late_anchor0_overlay.png`): f32
logp through a frozen anchor is a ±0.02 ulp staircase around a smooth f64 curve.

### E1b — gradient audit (prediction FAILED — falsifier fired)

Pre-registered (envelope-theorem motivation: the gradient picks up coeff error at FIRST order):
f32 gradient rel error ≥ O(1) at frozen anchors. Observed: rel err 1.2e-4–7.3e-4 with
cos = 1.0000 at ALL visited anchors (frozen and healthy); catastrophe only at prior_far anchor 1
(rel 36.8, cos 0.25). **Gradients are innocent at the positions that matter.** Side findings:
|g| at frozen anchors = 4.7e3–1.1e4 vs 1.5e5 at bootstrap (the old "1.1e8" note not reproduced);
all measurements CPU — GPU-kernel-context gradients not yet independently audited (E1c).

### Raw-dE forensics (decisive; no new compute — re-analysis of `d3/*/dE_*.npz`)

At eps→0 in float32 the per-step dE is a **deterministic constant per anchor, identical across
all 64 momentum draws and the 8 smallest eps decades**: +3.859, +4.203, +2.672, **−3581**
(frozen run-a anchors 0–3); **+0.078125 exactly** (= 5 ulps) at bootstrap, where Var = 0 made
the old analysis report "floor 0.0" and the offset invisible. The pooled variance of the four
frozen-anchor constants is 1.423e5 — **the entire reported "noise floor" was the variance of
deterministic offsets, not stochastic noise.** float64: same structure at ~1e-11.
Mechanism: the D3 harness builds `IntegratorState.logdensity` from a standalone
`jit(value_and_grad)` while the kernel recomputes logdensity through its own shard_map
compilation — two XLA compilations of the same float32 function disagree by a position-dependent
constant c(x). All CPU paths (plain `log_prob`, `value_and_grad`, E1 harness) agree to ulps, so
the large offsets live in the GPU compilation seam; the −3581 magnitude (2.2% of |logp|) at one
anchor is unresolved (candidate: compilation-dependent f32 GPU solve at gram cond 8e6, where
cond·ε ≈ 0.5; second-order value effect of a large inter-compilation coeff jitter).

**Offset-corrected (honest) float32 EEVPD curves** (per-anchor c subtracted):

| anchors | corrected floor (Var/dim) | 5e-4 crossing | reading |
|---|---|---|---|
| n10 run-b late (healthy) | 2–5e-6 | eps ≈ 1 | matches settled eps 0.79 — unchanged ✓ |
| n25 bootstrap | ~6e-6 | eps ≈ 2.2 | unchanged ✓ |
| n25 frozen run-a | 1e-4 → 1e-2, slope ~0.4–0.8 | eps ≈ 2e-4 (worst) – 1e-2 | ~1e3–1e4× healthy; target reachable only at unusable eps |

### REVISED Layer-1 statement

In float32 the per-step energy error is never eps^6 at small eps; it is **ulp-staircase
evaluation noise** (logp quantized at ~0.016 for |logp|≈1.6e5; Var grows ~eps^{0.5–0.8} as steps
random-walk across the staircase), whose amplitude is position-dependent — ~30–1000× larger (in
Var) at the n_max=25 frozen positions than at healthy ones (consistent with solve-jitter
coarsening the staircase where residuals are large; the exact anchor-3 amplifier is unresolved).
Controller stability is set by where this noise crosses 5e-4: eps ≈ 1–2 at healthy positions vs
eps ≈ 2e-4–1e-2 at the frozen ones. Separately, **any code path that mixes differently-compiled
logdensity evaluations** (state init, adaptation resets vs in-kernel) injects deterministic fake
dE offsets from ulps up to thousands. Note the target itself is marginal in float32 for this
likelihood: dE_target = 0.092 ≈ 6 ulps of logp — healthy systems operate 2–3 ulps from the
noise; brittle by construction. The bottom line of the verdict (f32 EEVPD signal structurally
unusable at the failing positions; f64 restores eps^6 with enormous headroom) is **unchanged**;
the amplification mechanism is corrected: NOT conditioning (E2), NOT value noise (E1), NOT
gradient corruption (E1b).

### Production compilation-seam audit (DONE — sampler is clean in vivo)

Read-only audit of `full_mclmc_with_adapt_sharded` + adaptation paths against the question: does
the REAL sampler ever compare logdensities from two different compilations in vivo (the D3-harness
sin)? Verdict table (main-agent spot-verified at the cited lines):

| Path | Verdict | Evidence |
|---|---|---|
| (a) run-start init | SUSPECT — benign 1-step transient | `blackjax_updated_utils.py:330,365`, `mclmc.py:67` |
| (b) step-size adapt reset | SAFE | `mclmc.py:231-232,351-353` |
| (c) mass-matrix window swap | SAFE | `mclmc.py:256`, `blackjax_updated_utils.py:275` |
| (d) xi/EWMA dE | SAFE | `blackjax_updated_utils.py:74-77`, `mclmc.py:166` |
| re-jit vs arg | SAFE — metric-as-arg, 1 compilation (TEST1) | `mclmc.py:448-463,256` |

The invariant (verified): every step's `state.logdensity` is the previous kernel call's OWN
output (`blackjax_updated_utils.py:77` forms dE = kinetic_change − logdensity + state.logdensity;
:86/:93 store the kernel's own logdensity into the new state), so both logp terms in any in-vivo
dE share one compilation. The metric is a TRACED carry argument consumed via `cholesky`
(`:275`) inside a single jitted scan (`mclmc.py:448-463`) — TEST1: rebuilding
`kernel(traced_metric)` per step compiles the inner kernel exactly once → no compilation-N vs
N+1 straddle at window swaps. CPU smoke (TEST2): standalone-vs-in-kernel value_and_grad agree to
**0.00 ulp** over 200 f32 points (seam is GPU-compilation-specific, as hypothesized).
**Only path (a) differs:** `init_multi`/`_single_init` builds the first state's logdensity with a
standalone jitted `value_and_grad` → step 1's dE straddles init-compilation vs kernel-compilation
(one GPU seam offset c(x₀) in step-1 xi only), but the kernel overwrites state.logdensity on that
same step, so steps ≥2 are seam-free. Plausibly the source of the open **run-(a) tune1 ecraw
anomaly** (one corrupted step of ~400; washed out by the EWMA in tens of steps — non-load-bearing).
**Bottom line: the controller is NOT reading fake cross-compilation energy errors in real runs**
(except the overwritten step-1 transient). F2 (real metric-swap eps-slam) is a distinct, real
mechanism and is unaffected. Artifacts: `prod_audit/` (`prod_seam_smoke.py`, `smoke_results.txt`,
`verdict_table.png`). E1c (GPU) will confirm path (a)'s offset prediction directly.

### E1c — compilation/device-seam probe (DONE; mechanism CONFIRMED for the dominant term, reframed)

GPU+CPU probe at the 4 frozen run-a anchors + bootstrap (`e1c_compilation_seam.py`,
`run_e1c_on_gpu.sh`; artifacts `e1c/`, plot `e1c/e1c_device_seam.png`). The hypothesis (offsets =
within-process standalone-vs-kernel seam) FAILED as stated — within ONE process, standalone vs
kernel logdensity agree to ±1 ulp on BOTH CPU and GPU (and the honest single-compilation step
gives dE=0 everywhere). But the **decisive number is a CPU-vs-GPU logp discrepancy**:

| anchor | CPU logp | GPU logp | CPU−GPU | archived D3 dE | gnorm CPU | gnorm GPU |
|---|---|---|---|---|---|---|
| run_a_late_3 | 159862.77 | **156276.20** | **+3586.6** | **−3580.78** | 9835 | **1.11e8** |
| run_a_late_0 | 159881.08 | 159881.02 | +0.06 | +3.86 | 4.7e3 | 4.69e3 |
| run_a_late_1 | 159871.27 | 159870.98 | +0.28 | +4.20 | 7.3e3 | 7.44e3 |
| run_a_late_2 | 159894.58 | 159894.41 | +0.17 | +2.67 | 1.15e4 | 1.14e4 |
| bootstrap_0 | 154849.75 | 154849.77 | −0.02 | +0.08 | 1.47e5 | 1.47e5 |

**Anchor 3 — CONFIRMED and it is the whole ballgame.** The reported "1.42e5 floor" is the pooled
variance of the four offsets, and −3581² ≫ (+3.5)² — anchor 3 IS the floor. Its f32 logp swings
**3586.6 between CPU and GPU compilations** of the SAME function at the SAME point (matching the
archived −3580.78 to 0.2%), and its gradient explodes from 9835 (CPU) to **1.11e8 (GPU)** —
reproducing the long-open "1.1e8 grad" note (E1b on CPU couldn't, because the pathology is
GPU-compilation-specific). So D3's init `value_and_grad` evaluated anchor 3 with the CPU-like
compilation (159862) while the shard_map kernel used the GPU compilation (156276) → dE ≈ −3586.
Root: at this anchor the f32 lstsq logp sits on a near-singular cliff (grad 1e8) where two valid
float32 compilations land in different basins — the ultimate float32-precision failure, gone in
float64 (E1b/E1 f64 here: smooth, grad ~1e4).

**Residual (NOT chased, per anti-rabbit-hole rule):** the small benign offsets +3.86/+4.20/+2.67
at anchors 0–2 are reproduced by NEITHER the within-process seam (±1 ulp) NOR the CPU-GPU gap
(0.06–0.28); they are some D3-kernel-step-specific quantity (candidate: kinetic_change at eps→0
under the 1e-8·I metric) that E1c's probe did not replicate. They are ~40× the dE target so would
matter IF in vivo — but the production audit already showed the in-vivo step carries a single
self-consistent logdensity, so this residual is a D3-harness measurement artifact, not a sampler
mechanism. Logged, not load-bearing.

**Net:** strengthens the verdict — float32 logp at the high-n_max frozen positions is not merely
ulp-noisy, it is **compilation/device-unstable by thousands** at the gradient-singular anchor that
dominated the floor. float64 is the fix; the production sampler is clean in vivo (single
compilation carried; only the overwritten step-1 init differs).

### E-series open items

1. ~~**E1c**~~ DONE (above): anchor-3 −3580 CONFIRMED as a CPU-vs-GPU f32 logp swing (3586, grad
   1e8); within-process seam is only ±1 ulp; small benign offsets unreproduced (harness residual).
2. ~~Why anchor 3's offset is ~10³× its siblings.~~ ANSWERED: anchor 3 sits on a near-singular f32
   cliff (grad 1e8 on GPU) where compilations diverge; siblings have grad ~1e4 and stay stable.
3. ~~**Production audit:** do run-start init and `_make_adapt_reset`/window swaps recompute
   logdensity via a different compilation than the kernel?~~ **DONE — see "Production
   compilation-seam audit" above. Verdict: SAFE in vivo except a benign step-1 transient (path a).**
4. Whether the in-vivo flat xi≥1 during run (a)'s terminal decay is fully explained by the
   corrected staircase noise at the actual trajectory positions (the 4 archived anchors bracket
   but do not pin this).

### Fix implications (sharpened; still NOT executed — separate plan)

The dominant model-side lever is **killing the staircase, not re-conditioning the solve**:
cast only the residual/pixel-sum reduction (and optionally gram+solve) to float64 — O(N) adds at
negligible cost — predicted to shrink staircase noise by ~1e8 and restore a healthy crossing.
Orthonormalization/ridge/QR address solve jitter but NOT the |logp| ulp staircase → insufficient
alone (testable: E1-style staged measurement post-fix). Eliminate compilation-seam offsets by
initializing state logdensity through the kernel's own compiled function. Controller-side fixes
unchanged from the verdict. Every candidate must predict its effect on the offset-corrected D3
curves, the D5 ripple threshold, the D4 gauntlet, and the Phase-0 runs.

## Hypothesis ledger (current standing)

| Hypothesis | Standing |
|---|---|
| H1 model-side float32 noise floor | **CONFIRMED, mechanism revised by E-series**: position-dependent ulp-staircase noise (×30–1000 at frozen positions) + compilation-seam dE offsets; NOT conditioning (E2), NOT value noise (E1), NOT gradients (E1b) |
| H2 static NaN landscape/gradients | **DEAD** (D1: all finite everywhere incl. n_max=30) |
| H3 controller/metric brittleness | **CONFIRMED standalone** (D4: clean-Gaussian collapse at cond 1e8; D5: collapse at δ≈1% of target; F2/F3/F4/F8 mechanisms verified) |
| H3(e) small-start metric init | DEAD as cause; paradoxically protective at high cond (open anomaly) |
| H4 unreachable EEVPD target | OPEN — decided by D3's floor height vs 5e-4 |

## Open questions for Phase 3

1. D3: does the real model show an eps-independent dE floor at n_max=25, at what height, and does
   float64 remove it?
2. Why is the window-3 metric the terminal trigger in run (a) (metric quality? eigenvalue spread?)
3. The tune1 low-energy anomaly (run a) and the protective-tiny-init anomaly (D4 cond=1e8).
4. Where exactly the real model's noise lives (lstsq solve vs conv/pooling vs prior bijectors) —
   D2-redo localization.
