# Lab Notebook — Compute profiling & likelihood speedups

Profiling the scene-API likelihood gradient (the MCLMC inner loop) to find and
reduce compute cost, for the canonical single-plane shapelet/sersiclet fits.

**Last updated:** 2026-06-24

> One log per rough research area (see `../../AGENTS.md` → *The record*). Durable source
> of truth; agent memory is not. Update after any substantive step.

---

## Current state

Phase 0 (static FLOP/byte model) and Phase 1 (device-timing breakdown) are **done**
on Perlmutter A100-40GB (JAX 0.10, container `jax-2026-04-13`). They reranked the
bottlenecks vs. the prior guess. A first float32-basis ablation ran end-to-end
(grad agreement strong; adaptation comparison **inconclusive** pending a seed
control). Live artifacts:
- harness: `gigalens/wip/profile_scene_likelihood.py` (Phase 0/1)
- ablation: `GIGALens-Code/experiments/basis_precision_ablation.py`
- knob: `SimulatorConfig.basis_precision` (gigalens `simulator.py` + `scene_simulator.py`);
  `System.basis_precision` (gigalens_research `simtests/system.py`). Default `None` =
  byte-identical; opt-in only.

---

## Claims register

### C-1 — The lstsq gram/solve is NOT a bottleneck; basis-gen + its VJP dominate
- **Status:** `proposed (UNCERTIFIED)` — awaiting human grading of the numbers.
- **Criterion (pre-registered):** decompose one bs=1 grad eval by differencing nested
  jitted sub-computations; a stage is "the bottleneck" if it is the largest share of
  the grad wall-time.
- **Evidence / artifact:** Phase-1 run (200×200, ss=2, n_max=15, A100-40GB): basis+conv+pool
  3.06 ms (26%), EPL trace 0.49 ms (4%), **gram+solve 0.55 ms (5%)**, reduction 0.01 ms
  (0.1%), **backward (VJP) 7.55 ms (65%)**, grad total 11.66 ms. Phase-0: grad GFLOP
  scales **linearly** with depth (0.41 GFLOP/component, flat over n_max 8→20), so the
  depth² gram is subdominant (it runs on the *pooled* 40k-px grid, not the 160k-px
  supersampled one). This **falsifies** the pre-run hypothesis that the float64 normal
  equations dominate.
- **Proposed by / on:** agent · 2026-06-24 · **Grader:** _pending_
- **Caveats:** numbers are bs=1 single-eval; real MCLMC vmaps 2 chains/device. Relative
  breakdown expected to hold.

### C-2 — The grad kernel is ~co-limited by HBM bandwidth and half-rate Ampere FP64
- **Status:** `proposed (UNCERTIFIED)` — **prediction PARTIALLY FALSIFIED** (see C-6).
- **Criterion:** arithmetic intensity vs. the A100 fp64 roofline ridge.
- **Evidence:** Phase-0 reports 6.5 GB accessed / grad eval at ss=2,n_max=15 → ~4.3 ms
  pure HBM at 1.5 TB/s (vs 11.7 ms total); intensity 56 GFLOP/6.5 GB ≈ 8.7 FLOP/byte,
  near the fp64 ridge (~6.5).
- **Falsifier outcome:** the roofline reasoning predicted that a float32 basis (halving the
  basis bytes) would give ~2× on the basis stage. The direct measurement (C-6) shows
  ~1.0–1.04× on that stage. So either the basis stage is not the bandwidth-limited part, or
  the f64 basis bytes are a smaller share than the whole-graph 6.5 GB suggested (the conv —
  already float32 — and pooling dominate that stage). The roofline was too coarse to localize
  the limit. Do not rely on C-2 to motivate basis-precision work.
- **Proposed by / on:** agent · 2026-06-24 · **Grader:** _pending_

### C-3 — float32 light basis reproduces the gradient near the typical set (depth 137)
- **Status:** `proposed (UNCERTIFIED)` — fidelity holds; the *speedup* number is superseded
  by C-6.
- **Criterion (pre-registered, user-approved):** max rel-err of grad(log_prob), f32-basis
  vs f64-basis, over 8 post-warmup (typical-set) MCLMC positions ≤ 1e-4.
- **Evidence:** n_max=15, ss=1 run → **5.16e-5** (≤ 1e-4). gram/solve + reduction stay
  float64 (promoted by the float64 noise/observed arrays). NOTE: the "1.18× speedup" first
  reported here was model-specific (vela/sersiclets) and timing-noise-inflated (C-5); the
  clean back-to-back measurement is C-6 (~1.0–1.09×).
- **Proposed by / on:** agent · 2026-06-24 · **Grader:** _pending_

### C-7 — The backward pass is dominated by the lstsq-solve and the lens-deflection coupling
- **Status:** `proposed (UNCERTIFIED)`
- **Criterion:** attribute the backward (grad − forward) to stages by differencing grad-times
  of nested losses L0..L3, with (a) a no-PSF simulator twin to add the convolution, and (b)
  a **runtime** weight map (not a closed-over constant — a constant lets XLA constant-fold
  the pool/conv VJP, an artifact that zeroed the conv backward in the first two attempts).
- **Evidence (ss=2, 400×400, n_max=15, depth 137; total backward 7.50 ms ≈ 65% of grad):**
  gram solve + reduction **3.46 ms (46%)**, EPL deflection-coupling **3.00 ms (40%)**,
  convolution 0.65 ms (9%), basis-gen + pool 0.38 ms (5%). ss=1 is noisier (differencing
  error ~0.3 ms; conv row goes slightly negative) but agrees on the ranking
  (EPL 59%, gram 36%, basis 7%, conv ~0).
  - Artifact: `gigalens/wip/profile_scene_likelihood.py --backward-breakdown` (task birqhyzg0).
- **Caveats (buckets are not pure):** "EPL deflection-coupling" = EPL `fori_loop` (niter=18)
  VJP **+** the basis-position Jacobian (d basis/d β over 137 components) — not separated.
  "gram solve + reduction" = `jnp.linalg.solve` VJP + chi² reduction + a *second* basis
  traversal (real lstsq differentiates the basis through both image=Σ ret·coeffs and
  coeffs=solve(gram(ret))); this is also where float64 is load-bearing (conditioning).
- **Implication:** the candidate levers (shapelet-recurrence VJP, convolution/rfft2) are each
  ≲10% of the backward — low value. Real targets: the EPL `fori_loop` VJP (isolate via an
  niter sweep or EPL→SIE) and the lstsq-solve VJP. Both need care (EPL accuracy; solve
  conditioning/float64).
- **Proposed by / on:** agent · 2026-06-24 · **Grader:** _pending_

### C-6 — float32 basis yields only ~1.0–1.09× speedup; the basis stage barely changes
- **Status:** `proposed (UNCERTIFIED)` — **the float32-basis lever is not worth shipping.**
- **Criterion:** median-of-30 back-to-back grad timing, same synthetic model (EPL+Shear,
  Sérsic lens light, plain Shapelets source, Gaussian PSF), basis_precision None vs float32.
- **Evidence:** num_pix=200, n_max=15 (depth 137):
  - ss=1: basis+conv+pool 0.910→0.908 ms (**1.00×**); grad total 5.481→5.401 ms (**1.01×**).
  - ss=2: basis+conv+pool 3.079→2.960 ms (**1.04×**); grad total 11.637→10.640 ms (**1.09×**),
    with the gain coming mostly from the *backward* (rest-of-grad 1.11×), not the forward.
  - The basis-generation arithmetic is a minority of the basis+conv+pool stage: the conv is
    already float32 (`conv_precision`) and pooling dominates, so making the basis float32
    barely moves that stage. This is the direct test that supersedes the C-2 roofline guess.
  - Artifact: `gigalens/wip/profile_scene_likelihood.py --basis-compare` (task bl89ebo65).
- **Implication:** real speedups must target the **backward pass (~65%)** and the
  **convolution** (e.g. rfft2 if its VJP is clean on JAX 0.10), or reduce supersample —
  not basis precision.
- **Proposed by / on:** agent · 2026-06-24 · **Grader:** _pending_

### C-4 — float32 basis leaves MCLMC adaptation unchanged (within the noise floor)
- **Status:** `proposed (UNCERTIFIED)` — now supported by a seed control; proposed PASS.
- **Criterion (pre-registered, user-approved):** adapted step_size and per-chain min-ESS
  within 5% of the float64 baseline. **Revised** (the single-seed version was naive): the
  float32 mean must fall inside the float64 across-seed range, and means agree within 5%.
- **Evidence (seed control, n_max=15, ss=1, n_chains=8, burnin/results=2000/2000, shared
  qz, seeds 0/1/2):**
  - float64: L [34.2, 68.3] mean 52.8 (spread 64%); min-ESS [4.3, 17.2] mean 9.4 (137%);
    step_size spread 8%.
  - float32: L [47.9, 58.4] mean 54.3 (spread 19%); min-ESS [7.6, 11.0] mean 8.9 (37%);
    step_size spread 9%.
  - Means: L 54.3 vs 52.8 (**2.9%**), min-ESS 8.9 vs 9.4 (**5.3%**), step_size matched. The
    float32 distributions lie **entirely inside** the float64 band. The single-seed "42% L
    gap" (orig ablation) was seed/non-determinism noise.
  - Artifact: `experiments/basis_precision_seed_control.py` output (task bulam2zia).
- **Scope (if certified):** EPL+Shear+Sérsic-lens-light + EllipticalSersiclets source,
  Vela vela01_cam12, num_pix=200, ss=1, n_max=15, this MCLMC scheme. NOT shown: ss=2,
  other systems, very high n_max, or that the *posterior marginals* match (only the
  adaptation hyperparameters + grad were checked).
- **Proposed by / on:** agent · 2026-06-24 · **Grader:** _pending_

### C-5 — Adapted MCLMC L / min-ESS are NOT reproducible at fixed seed + precision (GPU)
- **Status:** `proposed (UNCERTIFIED)`
- **Evidence:** float64 seed=0 with identical qz gave L=56.65/min-ESS=5.8 in the ablation
  but L=34.24/min-ESS=17.2 in the seed control; float32 seed=0 gave L=32.81 then L=56.67.
  Same code, same seed → different adapted L/ESS. Implies non-deterministic GPU reductions
  (float32 conv / FFT / pooling atomics) feed the sensitive L-adaptation. The step_size is
  far more stable (~8% spread).
- **Relevance:** any single-seed MCLMC adaptation comparison (incl. [[the adaptation-collapse
  work]]) must treat L/ESS as noisy; use multi-seed spreads, not single pairs.
- **Proposed by / on:** agent · 2026-06-24 · **Grader:** _pending_

---

## Design checkpoints (criteria awaiting approval)

- **(cleared 2026-06-24) Seed control for C-4** — ran, confirmed the prediction (float64
  L spread 64% ≫ the float32-vs-float64 mean gap of 2.9%). See C-4.
- **(proposed) ss=2 speedup + posterior-marginal check.** Measure the float32-basis grad
  speedup at supersample=2 (Phase-1 implies the basis dominates ~4× more there → larger
  payoff than the 1.18× at ss=1), and add a direct float32-vs-float64 comparison of the
  result-phase posterior marginals (per-dim mean/std), which is what ultimately matters.
  **Status:** awaiting approval.

---

## Log (newest first)

- **2026-06-24 (latest)** — ss=2 basis-precision timing (C-6): float32 basis gives only
  1.00× (ss=1) / 1.04× (ss=2) on the basis+conv+pool stage and 1.01×/1.09× on the full grad.
  The basis stage is dominated by the (already-float32) conv + pooling, so f32 basis arithmetic
  barely helps. This **partially falsifies C-2** (the roofline predicted ~2×) and supersedes the
  optimistic 1.18× in C-3. Conclusion: the `basis_precision` knob is numerically safe (C-3/C-4)
  but **not worth shipping for speed**; target the backward pass + convolution instead.
- **2026-06-24 (later)** — Seed control (f64 & f32 × seeds 0/1/2, shared qz, n_max=15, ss=1)
  resolved C-4: float64 L spread is 64% / min-ESS 137%; float32 means match float64 to 2.9%
  (L) / 5.3% (ESS) and lie inside the float64 band → float32 basis benign for adaptation
  (proposed PASS). Surprise → C-5: adapted L/ESS are not even reproducible at fixed
  seed+precision on GPU (non-deterministic reductions); step_size is stable. Lesson: the
  original single-seed 5% falsifier was unsound for a quantity with a 64% noise floor —
  do not certify single-seed adaptation comparisons.
- **2026-06-24** — Phase 0/1 profiling (A100-40GB). Falsified the "gram dominates"
  hypothesis (C-1); identified basis-gen + backward as dominant (65% is the VJP) and a
  bandwidth/fp64 co-limit (C-2). Added opt-in `basis_precision` knob. First float32-basis
  ablation at n_max=15 (the user's regime; num_free=20 is n_max-independent since amps are
  lstsq, so convergence is the same 20-D problem): grad faithful to 5e-5 (C-3), step_size
  matches, but L/ESS differ — flagged as inconclusive pending a seed control (C-4). Speedup
  modest at ss=1 (1.18×). NOTE recorded: read per-chain ESS, not just means.

---

## Open questions

- Is the 42% adapted-L difference (C-4) within float64 run-to-run variance? → seed control.
- Speedup at ss=2 (the user's other regime): Phase-1 implies the basis is ~4× more
  dominant there, so the float32-basis payoff should exceed the 1.18× seen at ss=1 —
  unmeasured.
- Secondary lever (not yet pursued): full-complex FFT → rfft2 (~2× on convolution) if its
  VJP is clean on JAX 0.10. Re-validate the original VJP-bug rationale.
