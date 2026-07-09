# Lab Notebook — Compute profiling & likelihood speedups

Profiling the scene-API likelihood gradient (the MCLMC inner loop) to find and
reduce compute cost, for the canonical single-plane shapelet/sersiclet fits.

**Last updated:** 2026-07-09

> One log per rough research area (see `../../AGENTS.md` → *The record*). Durable source
> of truth; agent memory is not. Update after any substantive step.

---

## Current state

Phase 0 (static FLOP/byte model) and Phase 1 (device-timing breakdown) are **done**
on Perlmutter A100-40GB (JAX 0.10, container `jax-2026-04-13`). They reranked the
bottlenecks vs. the prior guess. A first float32-basis ablation ran end-to-end
(grad agreement strong; adaptation comparison **inconclusive** pending a seed
control). **Phase 2 (slow-regime matrix: n_max 30/40, ss=4, 300px, 2 datasets,
MAP-scale bs) is done 2026-07-09** → C-8/C-9/C-10 and the revised target ranking
in the 2026-07-09 log entry. Live artifacts:
- harness: `gigalens/wip/profile_scene_likelihood.py` (Phase 0/1)
- ablation: `GIGALens-Code/experiments/basis_precision_ablation.py`
- knob: `SimulatorConfig.basis_precision` (gigalens `simulator.py` + `scene_simulator.py`);
  `System.basis_precision` (gigalens_research `simtests/system.py`). Default `None` =
  byte-identical; opt-in only.

---

## Claims register

### C-8 — Slow-regime ranking: the supersampled per-component pipeline (basis+conv+pool,
### fwd + VJP) dominates every slow config; gram stays subdominant even at n_max=40
- **Status:** `proposed (UNCERTIFIED)` — Phase-2 matrix, 2026-07-09 (A100-40GB login node,
  niter=18 per user directive; artifacts `experiments/profiling/results_slow_regimes.json`,
  `profile_run_20260709.log`; driver `profile_slow_regimes.py`).
- **H-A outcome (depth scaling): CONFIRMED.** grad n_max=30 / anchor = 33.56/10.26 =
  **3.27×** (pre-registered band [2.7, 4.5], linear prediction 3.6×); n_max=40 = 63.91/10.26
  = **6.23×** (predicted ≈6.3×). GFLOP/component flat at 0.41–0.42 over n_max 15→40;
  gram+solve stage ≤ **12%** of grad even at depth 862 (7.56/63.91 ms). The depth² gram
  does NOT re-enter ⇒ mask-gather / gram-side levers stay LOW priority.
- **H-B outcome (ss scaling): prediction MISSED (middle zone, pre-registered rule).**
  ss=4/ss=2 at n_max=15 = 42.66/10.26 = **4.16×** (confirm band was 2.5–3.5; falsifier >5);
  at n_max=30: 148.35/33.56 = **4.42×**. Mechanism review: the ss-scaled stages are ~95%
  of the niter=18 anchor cost, not the ~75% inferred from June's niter=50 shares — the
  prediction under-counted the ss-scaled share; the ss² mechanism itself is supported
  (ratio ≈ 4). Practical: supersampling multiplies nearly the whole eval; vela regime
  (30, ss4, 200px) costs **148 ms/grad eval**.
- **H-C outcome (EPL loop VJP): not falsified; magnitude at niter=18 is 10–20%.**
  grad vs niter {5,18,50} = 8.83/9.76/14.40 ms median (mins 8.79/9.60/11.58; the niter=50
  cell had 24% median-vs-min contention — use mins there). Swing 5→50 = 2.8–5.6 ms ≫ the
  0.5 ms falsifier. Slope 0.06–0.12 ms/iter ⇒ niter-sensitive part at the June niter=50
  config ≈ 3–6 ms, consistent with C-7's 3.0 ms bucket being mostly the fori_loop VJP;
  at the user's niter=18 it is **~1–2 ms of a ~10 ms grad (10–20%)** ⇒ EPL unroll is a
  moderate, cheap target, no longer a top one.
- **Anchor note:** at niter=18 the backward is **~46%** of grad (4.77/10.26), vs 65% at
  June's EPL(50); basis+conv+pool forward alone is **40%**. Together with their VJP twins
  the supersampled per-component pipeline is ~70–85% of every slow config.

### C-9 — MAP batch throughput is memory-capped (~790 MB/sample at 200px ss2 n_max15 f64);
### batching buys 1.5–2.9× utilization before OOM
- **Status:** `proposed (UNCERTIFIED)` — H-E cells, same artifacts.
- Per-sample grad: 200px: 13.6 (bs=1) → 9.1 ms (bs=8, **1.49×**), **OOM at bs=32**
  (XLA-static peak 24.9 GB); 80px: 4.77 → 1.67 ms (bs=32, **2.86×**), OOM at bs=128
  (18.5 GB). Peak memory ~linear in bs (≈787 MB/sample at 200px, ≈147 MB at 80px).
  Direction of H-E confirmed (bs=1 underutilizes); the binding constraint for MAP-scale
  throughput is VJP residency ⇒ remat / residency-reducing changes (direct-χ²) are
  promoted; batching alone caps at ~1.5–3×.

### C-10 — XLA already CSE-shares the trace+basis across identical-grid datasets;
### the incremental cost of a second band is ~0.5–0.6× one eval
- **Status:** `proposed (UNCERTIFIED)` — H-D rerun with independent per-dataset noise
  (first attempt used byte-identical twin datasets: 2-ds min == 1-ds min, pure CSE
  artifact, recorded as a measurement-validity lesson; fixed driver seeds dataset k with
  seed+1000k; artifact `results_slow_regimes_hd_fixed.json`, `profile_run_20260709_hd_fixed.log`).
- 2 datasets / 1 dataset = **1.59×** (min-time; medians contended 22%) and XLA GFLOP
  84/56 = **1.50×**: with shared grid + shared light params the deflection trace and
  basis render are identical subgraphs and the compiler shares them; only the per-dataset
  weighting/gram/solve/reduction replicate. The pre-registered ≤1.5× falsifier fires (at
  threshold) ⇒ **deprioritize multi-dataset trace-sharing restructuring** for
  identical-grid bands; scope: same grid, same PSF, sees=all — different WCS/pixel scales
  would break the sharing.


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
- **Caveats (buckets are not pure):** "EPL deflection-coupling" = EPL `fori_loop`
  VJP **+** the basis-position Jacobian (d basis/d β over 137 components) — not separated.
  **Correction (2026-07-09, found at Phase-2 grading):** the June harness built `EPL(50)`
  (`wip/profile_scene_likelihood.py` build_problem); the "niter=18" originally written here
  was the EPL class default, not the measured config. The 3.0 ms deflection-coupling bucket
  is a **niter=50** number.
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

- **(2026-07-09) Direct-χ² lstsq + rfft2 convolution — equivalence & speedup gates
  (implementation of C-8 targets 1–2; user-approved direction, in-session).**
  **Classification:** two deterministic-identity claims (exact-math refactors), each
  with a stochastic timing claim on top. Code in gigalens worktree `lstsq-fast`
  (branch off linusu-dev-merge @ 4b8db1d); validation
  `wip/validate_fast_lstsq.py` there; results recorded here.
  - **Change 1 (direct-χ²):** χ²(c) = ‖Y‖² − 2cᵀ(XᵀY) + cᵀ(XᵀX)c with the
    UNREGULARIZED gram in the quadratic — an algebraic identity in the solved c
    (c itself still comes from the regularized solve, unchanged), equal to the
    pixel-space χ² of the reconstructed image for any c (mask ∈ {0,1} ⇒ mask² =
    mask). Cause hypothesis for speed: removes the second basis traversal (image
    = Σ ret·c) from the VJP — the dominant backward bucket (C-7/C-8) — plus the
    full-size model-image intermediates.
    Equivalence prediction: rel|Δχ²| ~1e-12 (f64 reassociation over ~4e4-term
    sums, ε√N ≈ 2e-14, headroom for cancellation); **falsifier: rel > 1e-8 on any
    of 8 prior draws (anchor AND one n_max=30 cell for conditioning) ⇒ bug, not
    shippable.** grad(log_prob) max rel err ≤ 1e-6 (predict ~1e-10).
    Speedup prediction: anchor grad total −15–35%; **falsifier: < 5% ⇒ the
    second-traversal model of the backward is wrong — stop, re-profile, don't
    ship complexity.**
  - **Change 2 (rfft2 + 5-smooth padded sizes; kernel FFT left to XLA
    constant-folding):** identical linear-convolution values (extra zero-padding
    beyond the linear length does not alter the cropped 'same' region); real FFTs
    halve FFT work/bytes. Equivalence prediction: conv out & grad rel ≤ 1e-13
    (f64; different FFT algorithm); **falsifier: rel > 1e-10 or any NaN in the
    VJP (the historical rfft-VJP-bug rationale) ⇒ revert to full-complex.**
    Speedup prediction: basis+conv+pool stage 1.2–1.6×, grad total −8–20%;
    falsifier: < 3% ⇒ conv share overestimated (keep only if exact, zero-cost).
  - **Combined:** ≥ 1.25× anchor grad; measure 2×2 variant matrix
    {complex-fft, rfft} × {image-χ², direct-χ²} at the anchor + the winning
    variant at (30,ss2) and (30,ss4,vela); XLA peak memory per variant (predict
    direct-χ² cuts peak ≥ 15%; informs C-9/remat). Blind spots: (i) equivalence
    at prior draws doesn't probe pathological gram conditioning beyond the
    n_max=30 cell; (ii) stage differencing impurity as before; (iii) library
    validation tests (`gigalens/tests/validation`) must also pass — regression
    gate, not just the new-path gate.
  - Metric/timing conventions and contention rule as Phase 2. Cost: ~10–20
    GPU-min, login A100. **Status: awaiting approval (rigor-grader).**

- **(2026-07-09) Slow-regime profiling matrix (Phase 2)** — extend Phase 0/1 to the
  user's actual slow regimes: high n_max (30/40), ss=4, 300-px cutouts, 2 datasets,
  MAP-scale batch. Purpose: rank the candidate speedup targets (direct-χ² lstsq VJP,
  EPL `fori_loop` unroll, rfft2 conv, mask gather, remat, multi-dataset sharing)
  before any implementation. **Classification:** diagnostic / measurement run
  (stochastic-estimator behaviour of wall-clock stage timings + deterministic XLA
  cost/memory analysis). It tests the *extrapolation links* of C-1/C-7 from
  (n_max=15, ss=2, 200 px, 1 dataset, bs=1) to the slow regimes; it does NOT test
  any fidelity claim (no sampling, no posterior).
  - **Metric:** median-of-30 (and min) device-synchronized wall ms per
    `grad(log_prob)` eval, stage-decomposed by differencing nested jits exactly as
    Phase 1; XLA `cost_analysis` FLOPs/bytes and `memory_analysis` peak per config.
    Known blind spot (inherited C-7 caveat): differencing attributes fused/shared
    work to whichever nested jit first includes it, so buckets are impure; treat
    bucket *ratios across configs* as the signal, not absolute purity. Shared
    login-node GPU adds contention noise: record median AND min; if
    (median−min)/min > 10% on the baseline anchor, move to a dedicated interactive
    node before trusting comparisons. Additional pre-registered blind spots
    (added at grading): (i) the nds=2 sub-stage rows are impure — the differencing
    jits time only `simulators[0]`, so dataset-1's forward lands in the
    "reduction" bucket; interpret ONLY grad_total for H-D. (ii) The driver sets
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` (shared-GPU citizenship), which the June
    harness did not — recorded env delta for the June-comparable niter=50 cell.
    (iii) H-A/H-B outcomes falling between the confirm band and the falsifier
    threshold = prediction MISSED → mechanism review, not a pass.
  - **H-A (depth scaling; tests whether the gram re-enters at high n_max).**
    Cause hypothesis: per-eval grad cost is linear in depth at fixed grid (C-1:
    0.41 GFLOP/component, flat over n_max 8→20); the depth² gram stays subdominant
    because it runs on the pooled grid. Prediction: at 200 px ss=2, grad time
    n_max=30 (depth 496) ≈ 3.6× the n_max=15 (depth 137) anchor, n_max=40
    (depth 861) ≈ 6.3×; gram+solve forward stage stays < 15% of grad. Derivation:
    linear extrapolation of the C-1 slope anchored at the re-measured n_max=15
    time; expected gram FLOPs 2·40k·496² ≈ 20 GFLOP fwd (~3× w/ VJP) vs ~203 GFLOP
    linear part → ≤ ~25% additive ⇒ ratio band 3.6×±25% ≈ [2.7, 4.5].
    Falsifier: ratio ≥ 5.5× at n_max=30 (or gram+solve stage ≥ 25% of grad) ⇒
    depth² term dominant ⇒ rerank toward gram/solve targets (mask gather, direct-χ²).
  - **H-B (supersample scaling).** Cause hypothesis: basis+conv+pool and their VJP
    scale ∝ ss² (supersampled-grid work), gram/solve fixed (pooled grid). Prediction:
    ss=4 vs ss=2 at 200 px n_max=15 → grad total 2.5–3.5× (the ss-scaled stages are
    ~70–80% of grad at ss=2, so 0.75·4 + 0.25·1 ≈ 3.25). Falsifier: < 1.8× or > 5×
    ⇒ cost structure mis-modeled (overhead- or FFT-size-dominated) — investigate
    before trusting any ss-targeted lever (adaptive supersampling).
  - **H-C (EPL `fori_loop` VJP share; decides the unroll target).** Cause
    hypothesis: the niter-sensitive part of C-7's "EPL deflection-coupling" bucket
    (3.0 ms, 40% of backward at the anchor, measured at **niter=50** — see the C-7
    correction above) is the `fori_loop` VJP; the remainder is the basis-position
    Jacobian. Prediction (restated at the measured anchor niter=50 per grading):
    grad time is roughly linear in niter over the 5→18→50 sweep, and the
    zero-intercept (niter-extrapolated) share of the 3.0 ms bucket is ≥ 50%.
    Falsifier: total grad swing < 0.5 ms over niter 5→50 ⇒ bucket is
    position-Jacobian-dominated ⇒ EPL unroll is low-value (kill target 2).
    (0.5 ms threshold ≈ 1.7× the observed differencing noise ~0.3 ms from C-7 ss=1.)
  - **H-D (multi-dataset overlap).** Cause hypothesis: the per-dataset Python loop
    in `ProbModel.log_like` serializes; nothing is shared, so 2 identical-grid
    datasets cost 2× one. Prediction: grad ratio 2 ds / 1 ds ∈ [1.8, 2.2].
    Falsifier: ≤ 1.5× ⇒ XLA already overlaps the per-dataset graphs ⇒ deprioritize
    multi-dataset restructuring (target 6).
  - **H-E (MAP-scale batch; diagnostic, threshold partially underivable).** Cause
    hypothesis: bs=1 kernels underutilize the A100, so per-sample grad cost falls
    with bs and the OOM frontier — not FLOPs — caps usable batch. Prediction
    (direction only): per-sample cost at bs=128–500 strictly below bs=1; peak
    memory ~linear in bs·depth·ss². A quantitative utilization threshold is **not
    derivable** without knowing the achieved occupancy per kernel — recorded as
    such; the deliverable is the measured per-sample-cost-vs-bs curve and the peak
    memory slope (informs remat target 5). Falsifier (direction): per-sample cost
    at bs≥128 ≥ bs=1 cost ⇒ kernels already saturated ⇒ remat/batching gains capped.
  - **EPL niter (user directive 2026-07-09):** all matrix / dataset / bs cells run
    at **niter=18** — the user's representative production value — NOT the June
    harness's EPL(50). Consequence: the (15,ss2,200px) anchor is *not* directly
    comparable to the June absolute numbers; June comparability is carried by the
    H-C niter sweep's niter=50 cell instead. H-A/H-B/H-D/H-E ratios are internally
    consistent (both sides at niter=18).
  - **Matrix (grad-eval timing + stage decomposition unless noted):** anchor
    (15,ss2,200px,niter=18); depth axis (30,ss2,200px),
    (40,ss2,200px); ss axis (15,ss4,200px), (30,ss4,200px) [vela regime]; cutout
    axis (15,ss2,300px), (30,ss2,300px) [carousel regime]; niter sweep {5,18,50}
    at anchor; 2-dataset twin of anchor (same grid, sees=all both); bs sweep
    {1,8,32,128,500} at anchor, forward+grad of the MAP loss (mean over batch),
    timing + peak memory only (no stage decomposition); **bs sweep also at
    (80 px, ss2, n_max=15) × {1,32,128,512,2000}** — registered pre-launch at
    grading; purpose: locate the utilization knee below the anchor's OOM frontier
    (hundred_systems-scale batch). Phase-0 cost/memory analysis over the full
    matrix (free, no GPU time). **Metric amendment (grading):** bs cells are
    median-of-10 (compile-dominated, large cells expensive); stage cells remain
    median-of-30. **Peak memory per bs cell** = XLA `memory_analysis`
    (compile-time temp+output), NOT device `peak_bytes_in_use`, which is
    process-cumulative and censored by earlier cells.
  - **Expected appearance:** stage-share table roughly constant across the depth
    axis if H-A holds (all stages linear in depth); if falsified, the gram+solve
    row grows toward ≥ 25% at n_max=30. If H-C holds, grad-vs-niter is a line with
    visible slope; flat line = falsified.
  - **Cost:** ~16 timed configs × (30–120 s compile + 30 × 10–200 ms timed) ≈
    30–60 GPU-min total, interactive/login GPU; big-memory cells (bs≥128, n_max=40,
    ss=4) on a dedicated interactive node (salloc) to avoid shared-GPU contention
    and OOM ambiguity. No Slurm batch queue. Seeds: harness synthetic seed=0.
    Code: gigalens @ 4b8db1d (read-only), driver
    `experiments/profiling/profile_slow_regimes.py` in worktree flow-precond
    (branch flow-precond-mams @ 3c6dc83, driver uncommitted pending user OK —
    this worktree hosts active MAMS work). No library-code changes in this run.
  - **Status:** user gave in-session go-ahead for profiling (2026-07-09); graded
    by rigor-grader 2026-07-09 → NEEDS-MORE with 4 minimal fixes (bs-cell memory
    via compile-time `memory_analysis`; register the 80px bs sweep; correct the
    C-7 niter record + restate H-C at niter=50; bs-cell metric median-of-10).
    All 4 applied above, plus the user's mid-grading directive to run the matrix
    at niter=18 (recorded in the "EPL niter" bullet). **Launched 2026-07-09** on
    the login-node A100-40GB (idle at launch). **CLEARED 2026-07-09:** ran same
    day (~15 min + H-D rerun); outcomes recorded as C-8/C-9/C-10 (H-A confirmed;
    H-B missed high per the middle-zone rule, mechanism reviewed; H-C not
    falsified, magnitude restated at niter=18; H-D falsified-at-threshold after
    the CSE-artifact fix; H-E direction confirmed, memory-capped). Contention:
    anchor 0.7% ⇒ login GPU acceptable; niter=50 / H-D medians contended 22–38%
    ⇒ min-times used there, as pre-registered.

- **(cleared 2026-06-24) Seed control for C-4** — ran, confirmed the prediction (float64
  L spread 64% ≫ the float32-vs-float64 mean gap of 2.9%). See C-4.
- **(proposed) ss=2 speedup + posterior-marginal check.** Measure the float32-basis grad
  speedup at supersample=2 (Phase-1 implies the basis dominates ~4× more there → larger
  payoff than the 1.18× at ss=1), and add a direct float32-vs-float64 comparison of the
  result-phase posterior marginals (per-dim mean/std), which is what ultimately matters.
  **Status:** awaiting approval.

---

## Log (newest first)

- **2026-07-09** — Phase-2 slow-regime matrix ran (pre-registered above; rigor-grader
  NEEDS-MORE fixes applied pre-launch; matrix at niter=18 per user). Outcomes → C-8
  (per-component supersampled pipeline dominates; gram ≤12% even at n_max=40 — H-A
  confirmed, H-B missed high with mechanism reviewed, H-C magnitude at niter=18 is
  10–20%), C-9 (MAP batching memory-capped, ~790 MB/sample at 200px; 1.5–2.9×
  utilization headroom), C-10 (XLA CSE already shares trace+basis across identical-grid
  datasets; 2nd band costs ~0.5–0.6×; first H-D attempt invalidated by byte-identical
  twin datasets — lesson: never profile duplicated constants). Revised target ranking
  for implementation: (1) direct-χ² lstsq (kills the second basis traversal + full-image
  reconstruction; also cuts VJP residency), (2) rfft2 + precomputed/pooled kernel FFT
  (conv+pool share of the dominant stage), (3) remat for MAP/ELBO batch throughput,
  (4) EPL unroll (~10–20% at niter=18), (5) adaptive supersampling (structural; the
  only ~4× lever for ss=4 vela), (6) mask gather + multi-dataset restructuring
  deprioritized (C-8/C-10). Artifacts: `experiments/profiling/profile_slow_regimes.py`,
  `results_slow_regimes*.json`, `profile_run_20260709*.log`.
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
