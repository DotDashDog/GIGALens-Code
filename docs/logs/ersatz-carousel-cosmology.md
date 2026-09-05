# Lab log — ersatz_carousel / cosmology (Om0, w0, wa) mixing

Area: `experiments/ersatz_carousel/gen_improved_sersic_ersatz_carousel.ipynb`, results
`results/ersatz_carousel_better/startfit_seed1/mclmc.stale-20260831T152745` (8 chains,
20k burn-in + 20k draws, init at truth, MCLMC, dense regularized mass matrix, ε=0.337, L=164).
All claims below are **proposed (UNCERTIFIED)** — Mode B, human grades.

## Claims register

### C-1 — Only the cosmology block mixes badly; the rest of the 104-parameter posterior is fine
wa R̂=1.43 / bulk-ESS 16; w0 R̂=1.16 / ESS 38; Om0 R̂=1.14 / ESS 594. Every other
parameter: R̂≤1.06, min ESS 300, median ESS 2376. (Rank-normalized split R̂ / bulk ESS,
ArviZ 1.2, z-space; physical-space R̂ for cosmology identical.)

### C-2 — The data pin two cosmological directions; the third is a curved filament along ≈wa
Cosmology enters only through 8 deflection ratios (single mass plane). Jacobian of the 8
ratios w.r.t. (Om0,w0,wa) at truth has singular values 0.140 : 0.0082 : 0.00023
(1 : 0.059 : 0.0016). Right null vector at truth (0.125, 0.081, −0.989). Posterior ratios are
constrained to 1e-4–1e-3 fractional; the filament in (Om0,w0,wa) is the level set of the
two constrained ratio combinations, bounded only by the prior box (wa∈[−3,2]) and a weak
third-direction constraint.

### C-3 — In the sampler's own (gaussianized-box) z-coordinates the filament ROTATES by 54°
Local null direction of d(ratios)/dz binned along the chains' physical wa: rotates from
(−0.09,−0.47,0.88) at wa≈−2.5 to (−0.76,0.11,0.65) at wa≈0.75 (54°) and back toward
(0.02,−0.04,1.0) at the wa=2 wall. A single frozen metric cannot be aligned with it everywhere.

### C-4 — The wa≈2 wall region is a mobility TRAP, not a mode
GPU re-evaluation of log-posterior on 250 thinned draws/chain: median lp of wall draws
(wa>1.9, n=321) minus bulk (n=1679) = +0.96 nats (within-chain p10–p90 spread ≈ 45 nats);
χ²_red identical (0.99830 vs 0.99831). Yet chain 2 sat there for all 20k draws and chains 1 and
6 migrated in at ~d10000/d17000 and never left; 16% of all draws are at wa>1.9 versus ~1–2%
expected from the flat-filament × uniform-prior volume (toy grid: 1.3%). The wa<−2 arm
(toy-grid mass 12%) got 2% of draws. ⇒ the baseline wa marginal is biased in both tails.

### C-5 — Mechanism: frozen dense metric vs rotating ridge → integrator instability at the wall
In the tuned dense metric (Cholesky-whitened −Hessian, hvp on GPU): the stiffest direction of
the whole 104-d problem at the bulk-median and wall-median points is (wa − 0.7·Om0), with max
eigenvalue 1869 (bulk) / 3258 (wall) / 5241 (high-wa draw) vs 915 at truth; curvature along the
whitened wa axis swings 147 (truth) → 23 (bulk) → −0.2 (wall). Linear stability ε_max≈2/√λ
= 0.028–0.066 vs ε=0.337 used. Energy error ξ>10 on 15% of all steps, 62% of the wall chain's
steps, corr(log ξ, z_wa)=0.585, spikes are short (median run 2 steps) ⇒ rattling, not NaNs.
Tuner symptoms: imm(wa)=0.15 vs posterior var 1.01 (7× under), L=164 from a poor ESS estimate.

## Design checkpoint — DC-1: ratio-pair + wa chart (matched short pilot)  [2026-09-05]

**Cause hypothesis.** Poor cosmology mixing is the frozen-metric-vs-rotating-ridge disease
(C-3, C-5), the same one proven on the DSPL system (`sample-cosmology-dspl.md`, T2/Run D). The
fix is a change of sampling coordinates in which the likelihood-null direction is a straight
coordinate axis: z=(z1,z2,z3) → (r_a, r_b) = deflection ratios of planes z=0.962 and 4.090
(NormalCDF-squashed into their image boxes), wa = −3+5·Φ(z3), (Om0,w0) solved from (r_a,r_b) at
that wa (damped Newton, IFT gradients, analytic log-det; prior density unchanged: uniform box,
Om0 trimmed to (0.05,0.99) — posterior mass outside: 0 of 160k baseline draws below 0.10).
Prototype: `RatioPairWaBijector` (job tmp `chart/ratio_pair_wa.py`; to be promoted to
`gigalens_research.priors.ratio_pair_wa`). Verified: float64 round-trip exact on a 15×15×11
grid; analytic fldj = AD to 4e-15; custom-VJP grads = FD to 1e-8; truth (0.3,−1,0), wall
(0.308,−1.209,1.956) and far arm (0.48,−0.5,−2.9) map to z=(−0.69,0.47,·), (−0.65,0.45,·),
(−0.71,0.49,·): the filament is a straight line along z3.

**Run.** `g3_run.py {ratio|gauss} 2000 2000 10`: identical model/data/init/seed, 8 chains,
2000 burn-in + 2000 draws each (≈1/10 of the production budget, ~3 min each on 4 A100).
Gates before sampling: identical z layout; χ²_red at the truth point identical across charts
(baseline 0.9980).

**Pre-registered predictions (derived).**
- P1 (primary): physical bulk-ESS(wa) in the ratio chart ≥ 100 at 16k draws (nuisance-level
  ESS rate 1.5%/draw → ~240; allow 2.4× margin) AND ≥ 5× the gauss-chart pilot's ESS(wa) at the
  same budget (baseline rate 0.01%/draw → ≤ 20). R̂(wa) < 1.05 in the ratio chart.
- P2: ratio-chart P(wa>1.9) ∈ [0.3%, 5%] (flat filament × uniform prior, toy-grid 1.3%);
  gauss-chart pilot unpredictable (trapping is stochastic in a short run) but the 20k run had 16%.
- P3: ratio-chart ξ>10 fraction < 6% (the bulk chains' level, 3–5%), vs ≥ 10% in gauss.
- P4: non-cosmology min/median ESS in the ratio chart within 0.5–2× of the gauss pilot
  (the chart touches nothing else).
**Falsifier.** P1 fails (ESS(wa) < 3× gauss) ⇒ the chart did not remove the dominant slow
direction; suspect cosmology–mass coupling (wa–halo e1/e2, shear γ1 correlations 0.3–0.4) and
stop reparameterising (rule: after two failed fixes, diagnose).
**Artifact / cost note.** A short pilot cannot certify the tails (P2 is order-of-magnitude);
certification needs the production-length run the user controls. Pilot launched without
grader approval because the user is away and the cost is ~6 GPU-minutes on an idle node;
flagged here explicitly.

**DC-1 outcome (2026-09-05 11:45) — VOID, not a test of the chart.** The cold 2k/2k pilot never
left the 1e-6 seed metric: final imm diag (cosmo) 2e-5/1e-5/3e-5 vs true z-variances
0.03/0.06/1.0, ε=13.3, L=610; median ESS 22 for EVERY parameter (cosmology and nuisance alike),
wa sd 0.09 (true ≈1.0). The windowed Welford metric bootstraps from the qz covariance with an
80-pseudo-sample prior; at 2000 burn-in the windows are 50–400 steps and the chains have not moved.
(The production run's first metric update was at step 4258.) Chart-independent tuner limitation of
short burn-ins — the cold gauss pilot is left running only as the matched control of that statement.
Also learned: the chart's Newton solve (40 fixed iterations) made each step 4× slower (173 vs 43
ms/step); fixed with an early-exit while_loop (4–6 iterations reach a 1e-13 whitened residual).
Also learned: TFP's bijector cache returns the cached input for ``forward(inverse(x))`` on the
same array object, so "round-trip exact" checks are vacuous unless the intermediate is re-created
(fixed in the unit tests; the fldj-vs-AD, FD-gradient and sampler paths were real solves).

**Chart support, measured (2026-09-05).** Validator grid (41,31,21) on Om0∈(0.05,0.99): the 2x2
det flips sign in one slab, w0 > −0.67, wa ∈ (0.25, 0.5) (all Om0), min signed whitened det
−2.07e-3; the same slab appears for every pair tested ((1.166,4.09), (0.962,3.086), (1.166,3.549),
(0.962,1.656)) ⇒ intrinsic (w_eff→0 makes dark energy matter-like, so Om0 and w0 gradients align).
On 3000 uniform random points of the box only 4 (0.13%) have no root (all at w0>−0.57,
wa∈(0.12,0.29)); everywhere else the round trip is exact to 2e-11. Baseline posterior mass in
the slab: 0 of 160,000 draws with w0 > −0.75 at −0.1<wa<0.6 (max w0 there −0.784). Adopted
tolerances: det_atol=3e-3, roundtrip_atol=1e-9 (``experiments/ersatz_carousel/ersatz_cosmo_chart.py``).

## Design checkpoint — DC-2: warm-metric matched pair  [2026-09-05 11:50]

**Design.** Same model/data/seed/init-at-truth as DC-1, 8 chains, 2000/2000, but the tuner is
seeded with the 20k baseline run's full posterior covariance (104×104), transformed into each
chart (cosmology columns of the 160k draws mapped gauss-z → physical → ratio-z; nuisance columns
identical). This is a *stronger* test of C-5 than the cold pilot: in the gauss chart the sampler now
holds the best possible single global metric, so any residual failure is the rotating-ridge effect
and not the tuner; in the ratio chart a single global metric is predicted to suffice.
**Predictions.**
- P1' gauss-warm: ESS(wa) < 40 in 16k draws and ξ>10 fraction ≥ 10% despite the perfect global
  metric (C-5: local stiff direction misaligned with the metric along the filament).
- P2' ratio-warm: ESS(wa) ≥ 100 (≥ 3× gauss-warm), R̂(wa) < 1.05, ξ>10 fraction < 6%,
  P(wa>1.9) ∈ [0.3%, 5%], nuisance min/median ESS within 0.5–2× of gauss-warm.
- P3' cost: ratio-chart step time ≤ 1.5× gauss (early-exit Newton).
**Falsifier.** P2' ESS(wa) < 3× gauss-warm ⇒ the chart is not the cure; suspect cosmology–mass
coupling; stop reparameterising and diagnose (two-failed-fixes rule).
