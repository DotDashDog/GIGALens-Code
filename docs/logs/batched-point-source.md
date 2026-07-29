# Batched point-source inference (GPU batching for SBC campaigns)

**Started 2026-07-24.** Companion log to `point-source-sbc.md`. Goal: run the
~100-system SBC campaigns as ONE batched computation instead of ~100 sequential
solo runs. Motivation (measured): MCLMC over 16 chains × 12 params cannot
occupy a GPU — a solo system is kernel-launch-latency bound and runs no faster
(user-measured: slower) on GPU than CPU — but launch cost is nearly flat in
batch width, so evaluating all systems per step in one program should cost
roughly what one system costs. v1 campaign totals 21.1 CPU-serial hours
(median 12.8 min/system at 1× budget).

**Scope decision (Linus, 2026-07-24):** SBC on SIMULATED point-source systems
only — never real data, never imaging likelihoods. Everything lives in
GIGALens-Code (`simtests/experiments/batched_point_source.py`); gigalens is not
modified. The enabling property is SBC-specific: all systems share model
structure, free-parameter space, and prior (sim prior == inference prior from
one persisted config), so systems differ only through a small pytree of
observed-data constants.

## Phase A — batched log-prob (DONE, certified by equivalence tests)

**Mechanism.** One template `ProbModel`; inside a `jax.vmap`-ed function,
shallow-copy the likelihood term (and its dataset) and swap the data-derived
attributes for traced per-system rows. Attribute reads happen at trace time, so
every lane evaluates the exact solo float64 code path with a batched system
axis.

**Per-system data surface** (established by reading the gigalens term; the
enumeration is the module's `_ARRAY_FIELDS`/`_SCALAR_FIELDS`):

- arrays: `x`, `y`, `sig_x`, `sig_y`, `cap` (positions channel);
  `inv_flux_obs`, `sig_inv_flux` (flux — note `sig_inv_flux = sigma_F/F^2` is
  data-dependent, which also makes `_log_norm` per-system); `td_obs`, `sig_td`.
- scalars: `_log_norm`, `dataset.trust_radius_arcsec` (per-system via the
  minimum image separation).
- everything else the traced code reads is static and shared — asserted equal
  at construction (raise-never-default): `n_images` (all 100 v1 systems are
  quads by generator selection), `newton_steps`, `trust_region_frac`,
  `src_anchor_sigma`, channel flags, `event_size`, redshifts/cosmology mode
  (z fixed 0.5/1.5, only H0 sampled), and the z-param name list.

**API.** `BatchedPointSourceProb.from_probs(solo_probs)` /
`.from_systems(systems, **builder_kwargs)`; `log_prob(z)` and `log_like(z)`
take `(S, *batch, P)` and return `(S, *batch)` pairs `(value, reduced_chi2)`,
matching the solo `ProbModel` convention.

**Equivalence tests** (`simtests/tests/batched_ps_test.py`, all passing
2026-07-24, CPU float64):

- batched == solo for `log_prob`, `log_like`, reduced χ², and gradients, at
  per-system truth points and at hostile random z (including non-finite lanes,
  which match), with the anchor off and on; jit-compatibility checked.
- measured batched-vs-solo disagreement ≤ 2e-8 RELATIVE — not bitwise, because
  XLA compiles the vmapped program with different fusion/FMA reassociation and
  the iterated Levenberg solve amplifies roundoff (documented in the gigalens
  module). Far below dynamical relevance (MCLMC energy tolerance 5e-4). A
  data-swap bug would present as O(1) shifts; the test asserts a swapped-lane
  cross-check (system 1's lane must NOT reproduce system 0's likelihood).
- guards: mixed `newton_steps`, mixed anchor on/off, mixed channel config all
  refuse loudly.

**Tripwire:** if a future gigalens change adds a new data-derived Term
attribute, the equivalence test is what catches it — keep it in any CI run
that touches the point-source likelihood.

## Benchmark (phase A)

Sampler-realistic access pattern: jitted `lax.scan` of `value_and_grad` steps
at z_best + jitter (converged-solve region), 16 chains/system, on the real v1
systems. Script: job scratch `bench_batched.py`. Comparisons: solo 1-system,
batched-1 (vmap overhead), batched-100. Hardware caveat: the available node is
RTX 2080 Ti (fp64 at 1/32 rate) — absolute times are NOT representative of the
V100/A100/H100 target; the claim under test is the SCALING (batched-100 ≈
batched-1 per step).

RESULTS (2026-07-24, n0028.es1, RTX 2080 Ti, float64, 100 steps GPU / 30 CPU):

| config                  | GPU ms/step | CPU ms/step |
|-------------------------|------------:|------------:|
| solo 1 system × 16 ch   |       10.94 |        6.47 |
| batched 1 × 16          |       10.95 |        6.55 |
| batched 100 × 16        |       16.32 |      225.07 |

- **GPU: batched-100 costs 1.49× one solo system → 67× effective speedup**,
  on the WORST-case fp64 GPU (2080 Ti, 1/32 rate). vmap-machinery overhead is
  nil (batched-1 == solo, 1.00×). Confirms the launch-latency-bound hypothesis
  almost exactly (per-step cost nearly flat in batch width).
- CPU control: batched-100 costs 34.8× solo (4 cores) → only ~3× — CPU is
  compute-bound, so the win is GPU-specific, as predicted.
- Solo GPU (10.9 ms/step) is SLOWER than solo CPU (6.5 ms/step) — reproduces
  Linus's original observation that motivated this effort.
- One-time Python cost: building 100 solo models = 42 s (builder prior-probes
  dominate); compile ≈ 13 s.
- Projection at these rates: 1×-budget MCLMC (15k steps) for ALL 100 systems
  ≈ 4 min; 6×-budget ≈ 25 min — vs 21.1 h CPU-serial for v1. An fp64-capable
  GPU (V100/A100/H100) should do better still.

## Phase B — batched MAP -> SVI -> MCLMC (DONE, certified 2026-07-24)

**Module:** `simtests/experiments/batched_pipeline.py` — `batched_map`,
`batched_svi`, `batched_mclmc`, and `batched_map_svi_mclmc` (campaign knob
names, unknown-knob refusal). All take a `BatchedPointSourceProb` + per-system
seed arrays; outputs are system-stacked solo-stage artifacts.

**Design.** Rather than vmapping the solo implementations (shard_map + tqdm
callbacks don't compose with vmap), each stage is a faithful vmappable
reimplementation with the solo math and RNG-stream structure:

- MAP: adam behind zero_nans+clip, `-mean(lp)/loss_normalization`, per-step
  best particle recorded BEFORE the update (C-6 fix), argmax over steps.
- SVI: MVN-TriL via FillScaleTriL(Exp, 1e-6), n_vi-draw ELBO, best-loss
  tracking, adabelief, solo per-step key chain.
- MCLMC: line-for-line port of `full_mclmc_with_adapt_sharded` with the
  cross-device collectives replaced by chain-axis reductions (psum→sum,
  pmin→min); kernel/handle_nans/Welford/ESS imported from the SAME
  gigalens/blackjax modules the solo sampler uses (verified: the "shardmap"
  helpers are shard_map-compatible, not -dependent — no collectives), and the
  covariance estimator is blackjax's own `welford_algorithm` final fn (formula
  parity, m2/(n-1)). Per-system step size / L / mass matrix fall out of the
  outer vmap — no segmented adaptation needed. Memory: burn-in positions not
  stored; `thin_every` subsamples kept draws at emission (chunked scan).

**Certification** (`simtests/tests/batched_pipeline_test.py`, passing):
stage-isolated, same inputs both paths (chained comparison measures
predecessor divergence along flat degeneracies, not the stage under test —
learned the hard way, measured 0.4 z-units of MAP z_best drift at equal lp).

- MAP: solo-vs-batched best-lp gap 0.75 lp units on both test systems
  (different similar-quality particles win under trajectory divergence).
- SVI: identical to machine precision (|dloc| < 1e-3 qz-sd, sd ratio 1.000) —
  adabelief's contraction suppresses roundoff divergence.
- MCLMC: on the system where both paths converge, mean shift 0.099 sd, sd
  ratio [0.91, 1.07]. Moment comparison is R-hat-GATED (>1.2 either side →
  skipped with a printed note; ≥1 converged comparison required so the test
  cannot pass vacuously). Measured why: a hard generated system at toy budget
  hit solo R̂ 3.2 / ESS 6 — comparing moments there is noise vs noise.

**Real-data validation** (v1 campaign, FULL campaign budgets, fresh seeds,
2080 Ti, `diagnostics/batched_v1_check.json`): first 8 gate-passing systems —
every batched posterior matches the stored campaign posterior within MC error:
max |z-mean shift| 0.089 sd (median ~0.03), sd ratios within [0.94, 1.10],
all R̂ ≤ 1.041, min ESS 228, NaN-rejection 0 everywhere. Wall: **585.8 s for
8 systems = 73 s/system vs 768 s/system solo CPU** — a 10.5× per-system
speedup at only 8-wide batching on the worst-case fp64 GPU. A 94-system
campaign-scale run (thin_every=4 for the 11 GB card) is the definitive
benchmark: results appended below when complete.

**RESULTS (94-system campaign-scale run, 2026-07-24 night,
`diagnostics/batched_v1_check_94.json`):** all 94 gate-passing v1 systems,
full campaign budgets, fresh seeds, anchor off (v1 config), thin_every=4,
4× RTX 2080 Ti (one process per GPU after Linus freed the node — his Jupyter
kernel had been preallocating 8.4 GB on all four GPUs, which was the real
cause of the earlier OOM ceiling; the "125 MB/system" MAP-grad figure stands
but the pool was ~2.5 GB, not 8.25 GB):

- **Wall: 21.8 min for all 94 systems (13.9 s/system) — 55× the solo
  campaign's per-system throughput**; per-GPU processes ran 53–57 s/system in
  12-wide waves.
- **Posterior agreement: among the 90 gate-passers, median max-|shift| 0.046
  sd, max 0.220 sd — matching the MC-error expectation sqrt(2/ESS) ≈ 0.037
  median.** Larger shifts pair with lower ESS exactly as MC error predicts.
  sd ratios ~[0.88, 1.20]. NaN-rejection 0 everywhere.
- Gate failures: 4/94 (sys_20 R̂ 1.055; sys_60 2.02; sys_63 1.076; sys_65
  1.127) — the known borderline class under FRESH seeds (the campaign's own
  seeds produced 6 failures on different systems); visibly flagged by R̂, not
  silent, and expected to shrink under the anchor + intensive-budget v2
  config. Not a batching artifact: converged systems agree to MC error.
- ESS figures are on thinned (every-4th) draws; gate comparability caveat
  as documented in `batched_mclmc`.

## Pending phases

- **Phase C — campaign integration**: batched runner behind the campaign spec
  (per-stage seed derivation matching `Pipeline`, artifact persistence per
  system for resume/metrics), then campaign v2 (waits on Linus's prior
  update, 2026-07-24). Multi-GPU: currently one process per GPU (script-level
  split, works fine); the clean version shards the system axis across devices
  in one program (one compile, no per-process build cost). Also worth adding:
  `jax.checkpoint` (remat) around MAP's loss — exact, cuts the ~125 MB/system
  MAP-grad memory several-fold, raises the single-GPU wave ceiling.
- fp64 hardware: request V100/A100 (es1) or H100/H200 (es2) for production;
  A40/2080Ti fp64 is 1/64–1/32 rate. All timings above are the 2080 Ti floor.
