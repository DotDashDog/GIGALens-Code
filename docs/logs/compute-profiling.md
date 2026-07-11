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
in the 2026-07-09 log entry. **Shipped so far (gigalens branch `conv-pool-fold`,
pushed):** rfft2 conv (C-11), fused conv+pool at ss≥3 + scatter-free shapelet
recurrence (C-15); vela-cell gradient 148 → 94 ms (−37%) cumulative. Dead ends
recorded: direct-χ² (C-12), float32 basis (C-6), gram/mask work (C-8/H-A),
multi-dataset restructuring (C-10). Next candidates: remat for MAP batching
(C-9), layout cleanup, windowed supersampling. Live artifacts:
- harness: `gigalens/wip/profile_scene_likelihood.py` (Phase 0/1)
- ablation: `GIGALens-Code/experiments/basis_precision_ablation.py`
- knob: `SimulatorConfig.basis_precision` (gigalens `simulator.py` + `scene_simulator.py`);
  `System.basis_precision` (gigalens_research `simtests/system.py`). Default `None` =
  byte-identical; opt-in only.

---

## Claims register

### C-18 — MAP-scoped remat shipped: MAP auto-runs remat for ss<3 simulators
### (2× batch frontier, −46% peak), samplers untouched by construction
- **Status:** `proposed (UNCERTIFIED)` — 2026-07-09, branch map-throughput @
  934e400 (user chose this option after C-17). `ProbModel.with_map_remat()`
  (twin, remat only for ss < _FUSE_CONV_POOL_MIN_SS, identity no-op otherwise)
  + `ModellingSequence.MAP` wiring; MAP PRINTS its remat decision (§0), since
  the override is invisible to `d.sim_config` → `InferenceContext.hash` and
  the model card still record remat_basis=False while MAP runs True — the owed
  model-card follow-up must report the MAP-EFFECTIVE remat.
- Gates all PASS (`wip/validate_map_remat.py`): G1 twin identity 2.4e-15 f64 /
  1.4e-16 f32; G2 ss=4 no-op (returns self); G3 memory wiring at bs=16:
  9197 → 4934 MB, exactly the C-16 rows; G4 end-to-end
  MAP(n_samples=48, 3 steps) on a DEFAULT (remat_basis=False) config ran at
  19.3 GB device peak — below the derived 20 GB remat/unremat separator
  (unremat signature ~28 GB; the cumulative peak exceeds the 14.8 GB pure-MAP
  expectation because earlier gate executions and optimizer state count).
  pytest 60 passed + pre-existing C-13 only.
- Scope: ss<3 simulators; single-device end-to-end (multi-device shard_map
  path and SVI not exercised). Zero sampler cost by construction (samplers
  never see the twin).

### C-17 — remat_basis default-flip REJECTED by its own gates: at compute
### saturation the recompute cost emerges (+8.3% vmapped 8-chain ss2; +23%
### fused ss4 bs=1); remat stays opt-in, correct everywhere
- **Status:** `proposed (UNCERTIFIED)` — 2026-07-09, branch map-throughput
  (gates `wip/validate_remat_default.py` + results json). The pre-registered
  STOP rule (3–10% band exceeds the user-accepted ~3%) FIRED at corner (ii)
  and the >10% rule at corner (i); per pre-registration the default was NOT
  shipped and the user is being consulted.
- Corner (ii), vmap(grad) 8 chains (200,ss2,15): **+8.3% min-based** (median
  +8.4% — no contention ambiguity). Mechanism: at 8 chains the device is
  compute-saturated, so recompute no longer hides in idle HBM bandwidth (the
  bs=1 0–3% result was the bandwidth-offset regime). The doubt-report
  hypothesis that production chain counts sit "toward the favorable end" was
  WRONG — saturation is the unfavorable regime unless memory-bound.
- Corner (i), fused ss4 bs=1 (200,ss4,30): **+23%** (peak −23%: 5232 vs
  6818 MB). Recomputing the spectral-fold conv is genuinely expensive.
- Equivalence: ALL PASS — f64 grad L2 3.4e-13 (≤1e-12), f32 3.1e-6/1.2e-6
  (≤1e-5), chi² bitwise 0.0, no NaNs; pytest 60 passed + C-13 only, incl. the
  golden anchor under remat. Remat is CORRECT in every corner; only the
  default-on economics failed.
- Consequence: `remat_basis` stays default False (docstring updated to the
  measured numbers); recommended usage = enable per MAP/SVI stage
  (`dataclasses.replace(cfg, remat_basis=True)`), or implement MAP-scoped
  auto-remat in the library MAP loop (proposed to user). Model-card printing
  of remat_basis/conv_precision/basis_precision still owed (C-16 amendment).

### C-16 — MAP-throughput trio outcomes: remat_basis ships (2× MAP batch frontier,
### −40–46% peak, ~0 bs=1 cost); channels-first layout FALSIFIED; FFT-norm fold
### NOT IMPLEMENTABLE
- **Status:** `proposed (UNCERTIFIED)` — 2026-07-09; gigalens branch
  `map-throughput` @ origin (base c5014fd); gates
  `wip/validate_map_throughput.py` (+ results json), probe
  `wip/probe_irfft_norm.py`. rigor-grader NEEDS-MORE pre-launch (prediction (d)
  rederived to 40–50%; probe artifact committed; shipped-config rmt32 gate +
  free-memory recording added) — applied.
- **Change 1 (remat_basis, opt-in SimulatorConfig flag) SHIPS.** f64 identity
  6.4e-14 (≤1e-12); peak −40% (bs=1: 535→321 MB) / −46% (bs=8: 4609→2478 MB) ✓;
  **MAP OOM frontier 2×** (bs=32 runs at static 9.9 GB where base OOMs at
  bs=32/18.4 GB; measured with a co-tenant holding ~24 GB — bias conservative)
  ✓; best per-sample 7.299→6.990 ms (falsifier NOT fired; the ≥1.2× prediction
  MISSED — the A100 is ~saturated by bs=16 at this size, so batching headroom
  beyond H-E's bs≤32 regime was already banked); **bs=1 regression 0–3%** vs
  the rederived 40–50% expectation — missed in the GOOD direction; mechanism:
  the recompute replaces the stored-intermediate HBM writes/reads ~1:1.
  Practical: for real MAP (n_samples/device ≥ ~32 at 200px ss2) remat is a
  FEASIBILITY lever, not just a speed one. Scope: (200,ss2,15) MAP loss,
  production conv f32, single A100; remat×fused (ss≥3) unexercised.
  **Threshold correction (flagged for human review):** the rmt32 gate (1e-10,
  added at grading without derivation) FIRED at n_max=30 (grad L2 8.0e-7);
  derived bound for f32-recompilation reassociation × n_max=30 gram
  conditioning is the 1e-5 class (C-15/E4 precedent) — measured value passes
  it and is 60× below the C-3/C-4 benign-noise level (5e-5). f64 identity
  gates all passed; no NaNs anywhere incl. shipped config.
- **Change 2 (channels-first gram) FALSIFIED — removed.** Equivalence diff
  BITWISE 0.0 at both cells and both dtypes; speed 0.0% (mins) at (200,ss2,15)
  and (200,ss2,30): XLA already compiles the legacy layout to the identical
  executable — the gram transpose was never real work. Mechanism for C-14's
  ~14% copies/transposes: the FFT ops' internal separable lowering (IFFT+IRFFT
  + transposes, see probe), not user-level transposes. Reverted per the
  pre-registered <2% falsifier; variant retained at commit 889e913.
- **Change 3 (FFT 1/N fold) NOT IMPLEMENTABLE at the JAX/XLA-API level** —
  closed pre-run per its decision rule; optimized-HLO probe shows the
  normalization inside the fft op's backend lowering, no user-fusible multiply
  (`wip/probe_irfft_norm.py`).
- pytest on the shipped tree: 60 passed + pre-existing C-13 only.

### C-14 — Kernel-level attribution of the post-rfft2 gradient (perfetto traces)
- **Status:** `proposed (UNCERTIFIED)` — 2026-07-09, user-approved diagnostic (the
  in-session approval WAS the grader sign-off); cells (200,ss2,15), (200,ss2,30),
  (200,ss4,30), 20 traced grad evals each, kernel-time coverage ≈ wall (10.9/29.7/
  108.4 ms/eval). Artifacts: `$PSCRATCH/claude_perf/{trace_attribution.py,
  trace_run.log, trace_attribution_results.json}` (pscratch purges ~180d — the
  numbers below are the durable record).
- Vela cell (200,ss4,30): **FFT-conv pipeline ~42%** (transforms 27.5 ms + cuFFT
  scal normalization 9.2 + r2c/c2r pre/post 9.0; pad fusions ~8 more);
  **shapelet-recurrence scatter fusions 17.8%** (scale as depth ×3.3 for n_max
  15→30 and ss² ×3.97 for ss 2→4); copies/transposes ~14%; gram GEMM ~4% at ss4
  (~15% at (30,ss2); pooled-grid, ss-invariant); solve 1.2%; pool 1.0%; **no
  while-loop kernels** (EPL cost lives inside elementwise fusions, ≤1–2 ms per
  H-C). Pre-registered branch rules: "FFT>35%" FIRED (conv work re-enters top);
  GEMM/pool/while rules NOT fired; the "elementwise basis+pool>50%" expectation
  was WRONG (~25–30%). Consequence: targets re-ranked to (1) Fourier pool-fold,
  (2) scatter elimination, EPL unroll demoted to last.

### C-15 — Fused conv+pool (spectral fold, ships at ss≥3) + scatter-free shapelet
### recurrence (ships everywhere): exact, −13.3% at the vela cell; cumulative −37%
- **Status:** `proposed (UNCERTIFIED)` — 2026-07-09; gigalens branch
  `conv-pool-fold` (= lstsq-fast + f6d6115..da1bd64, pushed to origin). Full
  pre-registration + amendments + outcomes: `wip/c15_checkpoint_and_outcomes.md`
  ON THE BRANCH (written there while $HOME was at quota); gates
  `wip/validate_fold_stack.py`, brute-force `wip/test_fold_bruteforce.py`.
  rigor-grader NEEDS-MORE pre-launch (brute-force test committed; dead
  multi-kernel branches replaced with a loud raise; anchor/marginal/
  magnitude-miss readings pre-registered) — all applied.
- **Equivalence (all PASS):** fold vs conv→crop→pool: CPU brute force 2e-15
  (odd/even kernels, rect, ss∈{2,4}); in-pipeline f64 χ² ≤6.7e-15, grad L2
  ≤1.25e-12; production-f32 χ² ≤1.1e-6, no VJP NaNs. Stack vs buffer: χ²
  ≤5.9e-15, grad L2 ≤3.5e-13 (magnitude-miss recorded: concat-vs-scatter changes
  fusion boundaries → fp-level reassociation downstream; values of φ identical).
  pytest tests/validation: 60 passed + pre-existing C-13 only; golden anchor
  passes on the fused tree (pre-registered reading — tolerances.py §6b comment
  update owed on merge). f64 ss4 identity cell ran at 120px (200px f64 UNFUSED
  reference OOMs 40GB; amendment 7, feasibility not results).
- **Speed (200,ss4,30, production f32, pre-registered marginals):** fold +8.7%
  (falsifier <5% NOT fired; 15–25% prediction MISSED LOW — the fold
  gather+phase pass and its VJP scatter-add eat part of the c2r saving); stack
  +5.7% (in band, also −4% peak). Combined 108.4→94.1 ms. **ss=2: fold gains
  ~0–3% but REGRESSES peak memory ~20%** (gather intermediate ≻ small-stride
  savings; C-9 makes peak memory the MAP binding constraint) ⇒ deployment rule
  `_FUSE_CONV_POOL_MIN_SS = 3`; final config re-verified: ss2 −5% time/−5%
  memory at n_max=30 (stack alone), ss4 −13.3%/−3%. **Cumulative vela-cell
  gradient vs pre-rfft2 baseline: 148.0 → 93.9 ms (−36.6%).**

### C-11 — rfft2 convolution on 7-smooth padded sizes: exact, VJP-clean on JAX 0.10,
### −14% (anchor) to −27% (vela cell) grad time and −28–31% peak memory
- **Status:** `proposed (UNCERTIFIED)` — all pre-registered gates PASS (2026-07-09;
  gigalens branch `lstsq-fast` @ bfaa59b, pushed; artifacts
  `gigalens wip/validate_fast_lstsq{.py,_results.json}`).
- Gates: E2 (f64) chi² rel worst 2.1e-14 (≤1e-10), grad L2 worst 3.6e-13 (≤1e-10);
  E4 (production conv f32) chi² rel worst 1.5e-6 (≤1e-5); zero VJP NaNs (the
  historical rfft2-VJP-bug rationale is retired on JAX 0.10). Speed: anchor
  9.64→8.33 ms (−13.6%, band −8–20%); (200,ss2,30) −16%; (200,ss4,30) 148→108 ms
  (−27%). XLA peak −28–31% (10.1→7.0 GB at the vela cell) — also relieves the C-9
  memory cap. Full-complex path retained as `_manual_fftconvolve_same_complex`.

### C-12 — NEGATIVE: the direct-χ² lstsq path is exact but gives 0% speed / 0 MB
### memory; the C-7 "second basis traversal" backward interpretation was wrong
- **Status:** `proposed (UNCERTIFIED)` — pre-registered falsifier FIRED and the
  prescribed action was taken (not shipped; reverted in bfaa59b, implementation
  retained at a04be87).
- The identity χ²(c)=‖Y‖²−2cᵀr+cᵀGc was verified exact (E1 worst 4.7e-14, incl.
  n_max=30 conditioning cell; E3 combined grad worst 2e-11). Speed: anchor
  8.33→8.35 ms (0% vs ≥5% required); attribution cells at (200,ss4,30):
  rfft+image 108.34 ≈ rfft+direct 108.12 ms, peaks identical ⇒ 0% at scale too.
  Memory prediction (−15% peak) also failed (0%). Implication: XLA already
  optimizes the reconstruction+chi² path; C-7's 46% "gram solve + reduction"
  backward bucket must be dominated by something else (solve VJP / reduction /
  differencing impurity) — re-attribution needed before any further work there.
  Lesson: bucket-differencing attributions are hypotheses, not measurements.

### C-13 — Pre-existing golden-anchor LOG_PRIOR drift (env, not code)
- **Status:** `proposed (UNCERTIFIED)` — found while running the mandatory
  regression gate. `tests/validation/test_regression_anchor.py::test_regression_
  anchor_probmodel_free_params_vs_golden` fails with LOG_PRIOR max rel=1.258e-07
  IDENTICALLY on unmodified gigalens 4b8db1d and on the lstsq-fast branch (all
  other anchor checks incl. chi²/image/coeffs pass). f32-scale drift in a prior
  term under the current TFP-nightly sidecar vs the frozen golden. Needs an owner
  decision: re-freeze the golden under the pinned env, or pin the drifted TFP
  piece. Unrelated to the rfft2 change.

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

- **(2026-07-09) MAP-scoped remat (user-chosen option after C-17): ProbModel.
  with_map_remat() + inference.MAP wiring.** Branch map-throughput. MAP builds
  a twin prob model with remat_basis forced ON only for simulators with
  supersample < _FUSE_CONV_POOL_MIN_SS (the measured-benefit regime; ss≥3
  keeps the user config — C-17's +23% fused-recompute regime). Samplers read
  the untouched prob model: zero sampler cost by construction. Gates
  (`wip/validate_map_remat.py`, thresholds inherit graded precedents):
  G1 twin identity f64 ≤1e-12 / f32 ≤1e-5 + finiteness; G2 ss≥3 no-op
  (identity object); G3 memory wiring — bs=16 MAP-loss static peak twin
  < 0.7× base (C-16 expectation ~4934 vs ~9197 MB); G4 end-to-end —
  ModellingSequence.MAP(n_samples=48, num_steps=3) on a default config MUST
  run (analytically infeasible unremat: bs=32 OOM'd at 18.4 GB in C-16, bs=48
  base ≈ 28 GB static); falsifier: OOM with ≥20 GB free ⇒ wiring bug (<20 GB
  free ⇒ record contention, rely on G3). Companion: pytest (C-13 tolerated,
  NEW failure blocks). Blind spots: (i) MAP smoke is 3 steps/1 device — the
  multi-device shard_map path is exercised only at dev_cnt=1 here; (ii) the
  twin shares datasets/model by shallow copy — any future ProbModel mutable
  state would alias (none exists today). Cost ~10 GPU-min.
  **Amendments (pre-launch, at grading — rigor-grader NEEDS-MORE, applied):**
  (1) G4 gains a NUMERIC peak gate: device `peak_bytes_in_use` during MAP
  < 20 GB (derived: remat bs=48 ≈ 3×4934 ≈ 14.8 GB; base ≈ 3×9197 ≈ 27.6 GB;
  ≥5 GB margin both sides) — the original "MUST run" branch alone is NOT proof
  of remat, since base bs=48 fits an idle 40 GB device (the C-16 bs=32 OOM was
  under a ~24 GB co-tenant). Script asserts dev_cnt==1 (G4 is vacuous
  otherwise). (2) Doubt report: (a) every research-pipeline MAP now shifts by
  the f32-conv ~1e-6 class with ZERO config diff and an UNCHANGED
  `InferenceContext.hash` / model card (both read `d.sim_config.remat_basis =
  False` while MAP runs True) — intended and user-chosen, but the owed
  model-card addition must report the MAP-EFFECTIVE remat, not the config
  field; (b) `inference.MAP` now PRINTS its remat decision (§0 printed-never-
  silent; the hasattr guard would otherwise fail silent); (c) with_map_remat
  docstring scoped to MAP (SVI not wired). (3) On PASS, claim scope: MAP-scoped
  remat active for ss<3 simulators, single-device end-to-end at bs=48/3 steps;
  multi-device shard_map path and SVI not exercised.
  **Status: grader fixes applied; launched 2026-07-09. CLEARED 2026-07-09 —
  all four gates PASS, outcomes → C-18; shipped @ 934e400.**

- **(2026-07-09) remat_basis default → True (user-directed; ~3% worst-case
  sampler cost accepted in-session) + closure of the two C-16 unexercised
  corners.** Branch map-throughput, follow-up commit. Thresholds inherit from
  the graded C-15/C-16 precedents (same gate machinery,
  `wip/validate_remat_default.py`).
  - **Corner (i) remat×fused (ss≥3):** equivalence remat-on vs remat-off at
    (120,ss4,30): f64 chi² rel & grad L2 ≤ 1e-12 (exact-recompute identity,
    C-16 measured 6.4e-14 for the non-fused analogue); production-f32 grad L2
    ≤ 1e-5 (f32 recompile-reassociation class, C-16 correction) + finiteness.
    bs=1 latency at (200,ss4,30) remat on/off: prediction 0–5% penalty
    (HBM-offset mechanism as at ss2); **decision rule: penalty > 10% ⇒ do NOT
    default-on at ss≥3 (auto-off like _FUSE_CONV_POOL_MIN_SS) — flag stays
    honest for shared configs.**
  - **Corner (ii) vmapped-chains sampler shape:** vmap(grad(log_prob)) over 8
    chains at (200,ss2,15), remat on/off: prediction 0–3% penalty (bs=1
    analogue); falsifier > 10% ⇒ same decision rule (do not default-on for
    that shape; report to user before shipping).
  - **Default-flip regression reading (pre-registered):** with the default True,
    pytest tests/validation now exercises remat EVERYWHERE incl. the frozen
    golden anchor; remat is f64-exact (~1e-13) ≪ ANCHOR tolerances (1e-10) ⇒
    predicted pass; any NEW failure blocks; the C-13 pre-existing LOG_PRIOR
    failure is tolerated. Cost ~15 GPU-min, login A100.
  - **Amendments (pre-launch, at grading — rigor-grader NEEDS-MORE, applied):**
    (1) Decision metric: latency penalties are read on **min-of-30**
    (contention-robust per the C-8 precedent); medians reported alongside; if
    median and min disagree > 20%, note contention and prefer min. (2)
    Middle-zone rules: corner (ii) penalty ≤ 3% ⇒ in-band, ship; **3–10% ⇒
    magnitude miss AND exceeds the user-accepted ~3% worst case ⇒ STOP and
    report to the user before shipping default-on**; > 10% ⇒ do not default-on
    for that shape. Corner (i) analogous with a 5% band edge. (3) Doubt report:
    (a) 8 chains probes one point between bs=1 (0–3%) and saturation (remat
    favorable, C-16); production chain counts sit toward the favorable end —
    the vmap claim is scoped to "8 chains; interpolation argued, not measured".
    (b) The flip propagates SILENTLY through gigalens_research
    `System.sim_config` (no remat_basis kwarg anywhere downstream): executable
    change + f32-class ~1e-6 numeric shift for all runs with zero config diff —
    intended (that is what a default flip is for), but named; the model card
    does not print remat_basis/conv_precision/basis_precision — **card
    addition owed as a follow-up**. (c) The f64 equivalence cell at 120px is
    conservative (the C-15 OOM was the UNFUSED reference; both sides here are
    fused), justified by shared-GPU co-tenancy; f32 covers (200,ss4,30).
    (4) On outcome, update the simulator.py docstring to reflect ACTUAL gate
    results (it currently pre-asserts them); if the ss≥3 rule fires, auto-off
    needs a new code gate (none exists yet).
    **Status: grader fixes applied; launched 2026-07-09. CLEARED 2026-07-09 —
    the STOP rule fired (corner (ii) +8.3% > the 3% acceptance; corner (i)
    +23% > 10%): default flip NOT shipped, outcomes → C-17, user consulted.**

- **(2026-07-09) MAP-throughput trio: opt-in remat_basis + channels-first gram +
  scal-fold exploration (user-approved in-session; ss=2 focus, batched MAP is
  the stated pain point).** Code: gigalens worktree `perf3`, branch
  `map-throughput` off linusu-dev-merge @ c5014fd. Gates:
  `wip/validate_map_throughput.py`. **Classification:** two exact refactors
  (remat = identical ops recomputed; layout = same sums, GEMM-order fp
  reassociation) + timing claims; change 3 is an implementability probe.
  - **Change 1 (remat_basis, opt-in SimulatorConfig flag, default False):**
    cause: C-9 — MAP batch throughput is capped by the OOM frontier
    (supersampled/complex VJP residuals), with 1.5–2.9× per-sample headroom
    below it. Predictions at (200,ss2,15), production conv f32, MAP loss
    (mean/num_pixels as inference.py): (a) XLA static peak at fixed bs=8 drops
    ≥ 40%; (b) max feasible bs ≥ 2× base (base was bs=32-OOM pre-rfft2);
    (c) best per-sample grad ≥ 1.2× better than base best-bs; (d) bs=1 single
    eval regresses ≤ 25% (recompute cost — acceptable, flag is opt-in).
    **Falsifier: best per-sample (remat) ≤ best per-sample (base) ⇒ record
    negative, keep flag default-off, do not recommend.** Equivalence: remat
    grad L2 vs no-remat ≤ 1e-12 (identical ops; reassociation only via
    scheduling).
  - **Change 2 (channels-first gram, `_GRAM_VIA_CHANNELS`):** cause: C-14 —
    copies/transposes ~14% of grad; the (0,2,3,1) full-tensor transpose before
    the gram reshape is the largest movable piece. Prediction: bs=1 grad −5–10%
    at (200,ss2,15) and (200,ss2,30). **Falsifier: < 2% ⇒ revert** (transpose
    was already fused/free). Equivalence: f64 chi²/grad-L2 ≤ 1e-10 both cells
    (n_max=30 conditioning read as in C-15); production-f32 chi² ≤ 2e-5.
  - **Change 3 (FFT 1/N scal fold): implementability probe first.** Reasoning
    to test: the scal kernels (~7–8% at the anchor per C-14) are the IRFFT
    normalization, which lives INSIDE XLA's IRFFT lowering (and its VJP), not
    as a user-fusible op — and the algebra (K/N before the transform ≡ 1/N
    after) is not rewriting XLA will do across an fft op. Decision rule:
    inspect the optimized HLO around the fft custom-calls; if the 1/N is
    internal ⇒ verdict "not implementable at the JAX/XLA-API level", record the
    negative WITH the HLO evidence, no timing run. If it IS a fusible multiply
    ⇒ implement + gate like change 2 (predict −3–7% at ss2; falsifier < 1%).
  - **Protocol (ss=2 focus):** bs sweep {1, 8, 16, 32, 64, 128, 256}-to-OOM of
    the MAP-loss grad at (200,ss2,15) for variants {base = merged HEAD behavior
    (legacy layout, no remat), layout, layout+remat}; per-sample median-of-10 +
    min, XLA static peak per cell (compile-time memory_analysis, not
    process-cumulative). bs=1 latency at (200,ss2,15) and (200,ss2,30) for
    change 2 (median-of-30). Contention rule as Phase 2. Cost ~30 GPU-min,
    login A100.
    Mandatory companion: pytest tests/validation on the final tree (C-13
    pre-existing failure tolerated; any NEW failure blocks) — added 2026-07-09
    just after grader dispatch (omitted in the first write-up; the C-15
    precedent makes it standing policy).
    **Change-3 outcome (closed pre-run per its decision rule):** NOT
    IMPLEMENTABLE at the JAX/XLA-API level. HLO probe (minimal
    rfft2·K→irfft2 grad, jit-compiled, optimized HLO): jnp.fft.irfft2 lowers to
    separable IFFT+IRFFT fft ops with internal transposes and ZERO user-visible
    constant-multiply ops adjacent to the ffts — the 1/N normalization is
    inside the backend lowering of the fft op itself, so no user constant can
    absorb it, and XLA will not move K/N across an fft op algebraically.
    (Side finding: those internal separable-FFT transposes explain part of
    C-14's copy/transpose class — equally not user-foldable.)
  - **Amendments (pre-launch, at grading — rigor-grader NEEDS-MORE, applied):**
    (1) prediction (d)'s ≤25% was NOT derivable: the checkpointed region
    includes the EPL trace, so by C-8's own stage shares the expected bs=1
    regression is **~40–50%**; pre-registered reading: 25–50% = magnitude miss
    on (d) that does NOT block the opt-in ship if (c) holds; >50% ⇒ investigate
    before shipping. (2) The change-3 HLO probe is committed as
    `wip/probe_irfft_norm.py` (output snippet in its docstring); the log prose
    summarizes that artifact. (3) The SHIPPED MAP config (production f32 conv +
    remat) added to the equivalence gates (rmt32 grad L2 ≤ 1e-10 + finiteness).
    (4) Blind spots: (i) remat×fused conv+pool (ss≥3) inside jax.checkpoint is
    UNEXERCISED — the remat claim is scoped to the ss=2 non-fused path; (ii)
    the OOM frontier is measured with PREALLOCATE=false on a shared login GPU —
    device free memory is recorded before each variant sweep, and the static
    XLA peak is the preferred reading where they disagree; (iii) variants run
    sequentially in one process with remat LAST (bias direction conservative
    against remat). (5) Scope of the eventual claim: remat validated only at
    (200, ss2, n_max=15) MAP-loss batching, production conv f32, single A100;
    layout validated at ss2, n_max ∈ {15, 30}.
    **Status: grader fixes applied; launched 2026-07-09. CLEARED 2026-07-09 —
    outcomes → C-16 (remat ships opt-in; layout falsified bitwise; scal-fold
    not implementable). Branch `map-throughput` pushed.**

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
    GPU-min, login A100.
  - **Amendment (pre-launch, at grading — rigor-grader NEEDS-MORE, fixes
    applied):** (a) E2 equivalence runs at `conv_precision=None` (f64), matching
    the threshold derivation; a separately-derived **E4** check at the production
    `conv_precision="float32"` (chi2 rel ≤ 1e-5 = f32 eps × FFT growth; any VJP
    NaN ⇒ revert) covers the shipped dtype. (b) E2's grad gate metric is
    ‖Δg‖/‖g‖ (L2), threshold 1e-10 — the per-element-with-floor metric was
    underivable (effective abs tol below f64 eps on small elements). (c)
    Correction: padded sizes are **7-smooth** (2·3·5·7), not 5-smooth. (d)
    Pre-registered interpretation: E1 failure only at the n_max=30 cell ⇒
    investigate cross-graph solve rounding × gram conditioning before concluding
    "bug". (e) `python -m pytest tests/validation -q` (incl. the frozen golden
    anchor, the only pre-change reference) is a MANDATORY gate — E1/E3 alone
    cannot detect a bug shared by the refactored `lstsq_simulate` and
    `lstsq_chi2`. **Status: fixes applied; launched 2026-07-09.**
  - **Post-run amendment (2026-07-09, pre-registered before the follow-up
    cells):** the direct-χ² anchor speed falsifier FIRED (0% vs ≥5% required);
    rfft2 landed in-band (−13.6%). The big cells ran only new-vs-old (−16%,
    −27%), which cannot attribute the gain. Follow-up (wip/attrib_ss4.py): the
    two missing 2×2 variants at (200,ss4,n_max=30). Hypothesis: rfft2 accounts
    for essentially all of the −27%. Falsifier: rfft+image ≥ ~120 ms (i.e.
    direct-χ² contributes ≥10% at scale) ⇒ revisit the no-ship decision for
    direct-χ². Also: base-commit rerun of the golden-anchor test to classify the
    LOG_PRIOR 1.3e-7 pytest failure (my diff cannot touch the prior; suspected
    pre-existing environment drift vs frozen golden — if it fails identically on
    unmodified 4b8db1d, it is not caused by this change).
    **CLEARED 2026-07-09:** all equivalence gates PASS; rfft2 in-band; direct-χ²
    speed falsifier fired and attribution confirmed 0% at scale ⇒ rfft2 shipped
    alone, direct-χ² reverted (outcomes → C-11, C-12; pytest failure → C-13,
    pre-existing). Branch `lstsq-fast` @ bfaa59b pushed to origin
    (seanxuseanxu/gigalens); PR left for the user to open (permission layer
    declined an agent-created PR on the external-owner repo).

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

- **2026-07-09 (latest, MAP-scoped remat + default-flip rejection)** — The
  user-requested remat default flip was REJECTED by its own pre-registered
  gates (C-17: +8.3% on the vmapped 8-chain sampler shape at ss2 — the bs=1
  0–3% result was the bandwidth-offset regime, gone at compute saturation;
  +23% at fused ss4); per the STOP rule the user was consulted and chose
  MAP-scoped remat instead. Shipped as C-18: `ProbModel.with_map_remat()` +
  `ModellingSequence.MAP` wiring — MAP auto-runs remat for ss<3 simulators
  (2× batch frontier, −46% peak, gates G1–G4 all PASS incl. the
  grader-hardened numeric peak gate: 19.3 GB vs ~28 GB unremat signature;
  pytest green), samplers untouched by construction, decision PRINTED since
  sim_config/hash/model-card cannot see the override (card follow-up owed).
  One shared SimulatorConfig now does the right thing for both MAP and
  samplers. Shipped @ 934e400 on map-throughput.
- **2026-07-09 (MAP-throughput trio)** — C-16: `remat_basis` ships
  (opt-in SimulatorConfig flag; 2× MAP batch OOM frontier, −40–46% peak, best
  per-sample 7.30→6.99 ms, bs=1 cost 0–3% — the rederived 40–50% recompute
  expectation missed in the good direction, mechanism recorded). Channels-first
  gram layout FALSIFIED bitwise (XLA already elides the transpose; C-14's
  transpose class = FFT-internal separable lowering) and removed per its
  falsifier. FFT 1/N fold closed as NOT IMPLEMENTABLE (HLO probe committed).
  One threshold correction flagged for human review (rmt32 gate, C-16).
  Branch `map-throughput` @ origin. Recommended usage: set
  `SimulatorConfig(remat_basis=True)` for batched MAP/SVI; leave off for MCMC.
- **2026-07-09 (earlier)** — C-14 kernel-trace attribution (post-rfft2; ran from
  $PSCRATCH during the $HOME quota outage) re-ranked targets: FFT-conv pipeline
  ~42% at the vela cell, scatter fusions ~18%, EPL invisible (no while kernels).
  Then C-15: implemented + gated the two new targets on branch `conv-pool-fold`
  (contains the whole perf series incl. rfft2; pushed to origin). Fused
  conv+pool exact (2e-15 brute force; 1e-15/1e-12 in-pipeline) but ships at
  **ss≥3 only** (at ss=2 it regresses peak memory ~20% for ~0 gain);
  scatter-free recurrence exact and ships everywhere. Vela-cell gradient
  108.4→94.1 ms this round; **cumulative 148→94 ms (−37%) since profiling
  began**. Both C-15 speed-prediction outcomes recorded honestly: fold missed
  its 15–25% band low (+8.7%, mechanism noted), stack landed in band (+5.7%).
  Grader (rigor-grader) NEEDS-MORE items applied pre-launch. Deferred-from-quota
  drafts merged into this log; full checkpoint text lives on the branch
  (`wip/c15_checkpoint_and_outcomes.md`).
- **2026-07-09 (later)** — Implemented + gated C-8 targets 1–2 (gigalens worktree
  branch `lstsq-fast`, rigor-grader-reviewed checkpoint; grader caught an
  f32/f64 threshold mismatch and an underivable grad metric in the harness
  pre-launch — fixed, E4 production-dtype gate added). Outcomes: **rfft2 shipped**
  (C-11: exact, VJP-clean on JAX 0.10, −13.6% anchor / −16% n_max=30 / −27% vela
  grad time, −28–31% peak memory); **direct-χ² reverted** (C-12: identity exact
  to 4.7e-14 but 0% speed and 0 MB at anchor AND at scale — pre-registered
  falsifier fired; C-7's "second basis traversal" attribution falsified —
  re-attribute the backward before further work there); **C-13**: pre-existing
  golden-anchor LOG_PRIOR drift (1.258e-07, identical on unmodified 4b8db1d;
  TFP-nightly env drift) found by the mandatory regression gate — needs owner
  decision (re-freeze golden vs pin). Final tree re-verified: pytest 60 passed +
  only the pre-existing failure. Branch pushed; PR left to the user.
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
