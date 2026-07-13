# LAPS prior-init on the real lens — diagnostic investigation (Phase 6)

Companion to `laps_validation_report.md` (CPU known-answer validation). This report covers the
**real lensing posterior** (the `jax-demo` scene-API lens) and answers one question the user posed:
*why does prior-seeded LAPS fail on the demo lens, and can it be fixed?*

**Discipline note.** Proposer != grader throughout: subagents built/ran experiments; the orchestrator
pre-registered hypotheses (cause + predicted magnitude + falsifier), adjudicated artifacts, and ran
controlled known-answer beds *before* proposing any fix. Five candidate mechanisms and a budget test were
falsified before the real cause was isolated. Metric discipline: primary comparison is **per-marginal
width-ratio + median/max vs a converged HMC reference**, NOT R-hat and NOT bias-sigma (both blind to
over-dispersion).

## 1. Ground truth
Basic HMC (not NUTS), VI-warm-started, doubled budget (burnin 500, 1500 results), on the demo lens:
`experiments/laps_validation/handoff/hmc_ref/hmc_mass.npy` (72000 x 8 mass params). Mass-param
R-hat <= 1.004 (global max 1.048 on a nuisance param; acceptable for mass posterior). This is the
reference for all width claims. Warm-init LAPS matches it to **0.95-1.03x** on all 8 mass params.

## 2. The failure
Prior-seeded LAPS (init from `prior.sample` -> `bij.inverse`) over-disperses vs HMC by
**~20x (median) to ~370x (tightest param, theta_E)**; over-dispersion is anti-correlated (Spearman -0.95)
with marginal tightness. Warm-init (SVI surrogate) does not. bias-sigma = |mean-truth|/std is small for
BOTH an accurate and an over-dispersed posterior — it hid this twice; width-ratio is the metric that
exposes it.

## 3. Six falsifications (cause hunt, controlled where possible)
| Hypothesis | Test | Result |
|---|---|---|
| Anisotropy (isotropic Phase-1 metric can't contract tight dirs) | 100x anisotropic Gaussian bed | FALSE — contracts to truth (0.9-1.0x) |
| NaN / bad-region freezing | lens per-chain finiteness | FALSE — 0% non-finite logp/grad at init/300/1500 |
| Step size eps too large | eps trajectory prior vs warm (6000 steps) | FALSE — prior eps (0.00145) ~= warm (0.00153) |
| Phase-2 acceptance collapse | full-LAPS p2_accept | FALSE — prior reaches 0.69 (target 0.70) |
| Just needs more budget | 13x budget (3000 unadj + 1500 adj) | FALSE — 22x/236x unchanged, tight dims pinned |
| Switch + preconditioner poisoning | correct qz metric override (`p2_precond_var`) + budget | FALSE — 20x/188x unchanged |

Controlled-bed corollary: rotated cond-1e4 Gaussian and a banana ALSO converge to truth — geometry,
correlation, and curvature do not break prior-init. Only a hard-boundary/NaN target freezes (chains stuck
at scattered starts). The lens is none of these.

## 4. The mechanism (what IS true)
Prior draws start catastrophically far from this tight posterior: ensemble **median logp -1.57e5 vs warm
+163** (the lens likelihood is far more informative than the prior). The unadjusted Phase-1 descends
enormously (equipartition D-tilde 1e9 -> 7e3) but **locks into a broad, self-consistent quasi-stationary
state**: because ECA hyperparameters (eps, L, diagonal preconditioner) are computed *from* the ensemble,
a broad ensemble sets hyperparameters that keep it broad. It is effectively **bistable** — warm-init sits
in the tight basin, prior-init in a broad one, and the dynamics do not cross at any practical budget.
Everything downstream is poisoned by this: the switch **false-fires** (delta goes small on the broad
quasi-stationary ensemble), the Phase-2 preconditioner is built from the broad variance (~100x too large
in tight dirs), so Phase-2 bisection forces a tiny eps (0.015 vs warm 0.62) whose local moves cannot
transport a broad ensemble to the mode. Phase-2 acceptance still hits 0.7 — not the problem.

**Root:** prior-init is a **scale-contraction** failure, not a warm-up/location failure. Why the paper's
prior-init works and ours doesn't: inference-gym priors are not this much broader than their posteriors;
this lens is outside LAPS's prior-init envelope.

## 5. Annealing bridge (v1) — partial, and it localized the obstacle
Chained short unadjusted Phase-1 over `beta = geomspace(0.003, 1, 10)`, 200 steps/level, tempered target
`logp_beta = log_prior + beta*log_like` (beta=0 is the prior; prior draws are already in equilibrium).
`experiments/laps_validation/handoff/diag_anneal.py`.
- **Location bridges:** median true logp climbs monotonically -1.68e5 -> -369 (beta=1); final adjusted
  LAPS from the annealed ensemble reaches +50 (warm +162). Best chains reach the mode.
- **Scale does NOT contract:** per-dim std stays ~0.21 at EVERY beta (warm 0.013); final width
  **18x median / 213x max** — unchanged from the no-anneal baseline. Annealing moves the ensemble's
  center, not its width. The scale-lock operates at every temperature, so unadjusted tempering inherits it.

## 5b. REVISED mechanism (2026-07-06, DC-7.1 run): a chain-level MIXTURE, not a broad ensemble

The DC-7.1 lever run exposed a plot-vs-metric conflict whose resolution corrects §4's mechanism:

- **39-46% of prior-seeded final states sit essentially inside the posterior**: the subset inside
  HMC+-6sigma (n~200-235/512 across arms + old baseline) is CLOSE TO but statistically
  distinguishable from HMC — core-width-ratio 1.23-1.38x median (max 1.55x, A1b), mean offsets
  ~1 sigma (5-11 SE at n~211). The remainder sits at rms distances of tens-to-hundreds of sigma
  (rms-z percentiles: p25=1.3, p50=29, p90=190, p99=678). NOTE (grader): with only final
  snapshots, "a fixed converged subpopulation + stranded chains" is NOT distinguished from "all
  chains sampling one heavy-tailed quasi-stationary law"; the per-chain reading below is
  inference, not measurement (open: save per-chain trajectories in the next run).
- **This was ALWAYS the structure**: the original early-stopped baseline (`hmc_ref/prior_mass.npy`)
  has 46% in-core with core ratio 1.16x. The reported "20-370x over-dispersion" was the std of this
  mixture — std is quadratic in outliers, so a ~55% straggler tail at ~100 sigma dominates it
  completely. **Metric blind spot, third strike** (after bias-sigma twice): the per-marginal
  std-ratio cannot distinguish "uniformly broad ensemble" from "converged core + straggler tail";
  MAD-ratio (8.7x median) is also mixture-contaminated. §4's "the ensemble locks into a broad,
  self-consistent quasi-stationary state" is therefore WRONG at the ensemble level: the failure is
  concentrated in a large sub-population far from the mode (per-chain-vs-heavy-tail reading open,
  see above). CAVEAT (grader): "the stranded fraction persists at 13x budget" is INFERRED from the
  mixture-blind std ratios (`diag_budget/` saved no per-chain data) — unchanged stds show the
  tail's second moment persists; the fraction itself was never measured at 13x. Open.
- What survives from §4: prior draws start catastrophically far; the switch false-fires; Phase-2
  cannot transport stragglers (its eps serves the core). What changes: the object needing rescue is
  the STRAGGLER SUBPOPULATION, not the whole ensemble; the tested levers (init velocity, first-step
  refresh, precond source) don't touch it.
- Annealing v1's "location bridges, scale doesn't" (§5) needs re-reading in mixture terms: its
  scale metric was the same mixture-blind std; whether annealing changed the CORE FRACTION was
  never measured (open).

## 6. Status / recommendations (revised 2026-07-06 after DC-7.1)
- **Warm-init LAPS is validated** and is the recommendation whenever a usable SVI/MAP surrogate exists.
- **Prior-seeded LAPS is a chain-level straggler problem (§5b), not ensemble scale-contraction.**
  All four reference-faithfulness levers falsified (DC-7.1). The natural next fix is **straggler
  rescue at the phase boundary**: ~40% of final states reach the posterior unaided, and logp
  separation is MEASURED but not perfectly clean (`diag_levers/logp_separation.json`, computed
  from A0/A1 samples_z on the GPU): core logp p5-p95 = [152, 165] (median 160 = the posterior
  typical set; warm ref +163) vs straggler median -5088 / p75 -738; a threshold at logp ~ 150 is
  near-clean in A0 (admits 0 stragglers, max 147.2; loses ~5% of core) but strict disjointness
  FAILS at the extremes (A0: 4/301 stragglers above the core MINIMUM logp 96; A1 core min 23.9).
  So logp-based selection is a supported HYPOTHESIS, not yet certified — the resampling design
  must pre-register the threshold + its misclassification cost, and address two grader-flagged
  risks: (i) the core itself is ~1sigma offset / 1.25-1.45x wide vs HMC, so a resampled ensemble
  INHERITS that bias while the in-box diagnostics read "fixed" — the adjusted re-diversification
  phase must be shown to erase it (compare against warm-init, not against the core); (ii) if the
  truth is a heavy-tailed stationary law rather than stranded chains, resampling is importance-
  truncation of a real tail — settle the per-chain-trajectory question first (cheap: save
  per-chain positions each chunk in the next run). SMC-style resampling then re-diversification
  with the validated Phase-2 kernel remains far cheaper than full tempering.
- Full tempering/SMC (§6 v1 recommendation) remains the fallback if boundary resampling fails.
- Secondary bug worth fixing regardless: the self-calibrated switch false-fires on broad
  quasi-stationary ensembles (fires when delta is small, which is not the same as converged).

## 7. Design checkpoints

### DC-7.1 — Reference-faithfulness cold-start levers (stocktake F1/F2/F3 + L0_inf) — status: CERTIFIED (design only), v2

*2026-07-06. Code: worktree-laps, `laps_late_adjusted.py` blob `be0ab48c` (F1/F2/F3 + `L0_inf`
levers, all default-off); suite 24/24 on GPU job 55600450 (jax-2026 shifter stack — the arms' exact
stack; transcript of the certified-time run not archived, step `.3` COMPLETED per sacct; the launch
run archives its own pytest transcript). Follows `laps_stocktake_2026-07-06.md` §5.1.*

*Grading scope (v2): rigor-grader CERTIFY-RECOMMENDED on the PRE-REGISTRATION ONLY — no empirical
claim about prior-init rescue is made or certified until the arms run and are independently graded
(plots viewed). Attribution rule (grader): differences between arms smaller than the A1-vs-A1b seed
spread are NOT attributed to any lever (F2's no-op is exact only in the step-size path; ULP-level
trajectory noise amplifies chaotically). If arms straddle a threshold by less than that spread, the
verdict is "no verdict", not "companion mechanism confirmed".*

**Claim + classification.** Distributional claim: prior-seeded LAPS with the blackjax-faithful
cold-start mechanisms reproduces HMC posterior widths on the demo lens. Chain of links: (i) levers
implement the reference mechanisms — TESTED (unit tests vs bj math: exact 0.5-halving, formula-exact
precond, sign/norm properties); (ii) the levers change cold-start basin selection on the lens —
THIS RUN; (iii) resulting widths match HMC — THIS RUN. Link (i) is not re-tested here.

**Arms** (all lens arms: 512 chains, 300 unadj + 200 adj, same model build as `laps_overlay_j26.py`,
and — grader amendment 1 — **`early_stop=False`** so every arm gets the FULL 300-step Phase-1 budget;
`switch_index_paper`/`p1_delta_max`/`switched` recorded post-hoc per arm. Rationale: the
self-calibrated switch false-fires on broad prior ensembles (measured: fired at 200/300,
`diag_p2accept/p2accept_summary.json`) and would truncate Phase 1 at an arm-dependent point,
confounding exactly the F1 falsifier's partial-effect regime; every Phase-6 diagnostic pinned
`early_stop=False` for the same reason):
- A0 baseline prior-init, levers off, seed 0 — same-code control / regression check of the edits.
- A1 `velocity_init="gradient"` only, seed 0 — THE hypothesis test (F1).
- A1b = A1 at seed 1 — single-seed guard on basin-selection stochasticity.
- A1c = A1 + `L0_inf=True` (grader amendment 2: bj's first-step L=inf, `laps_burn_in.py:261`, no
  Maruyama refresh on step 1 — the reference companion that protects the aligned velocity; ported
  as a fourth default-off lever so F1 is tested WITH its companion, not orphaned).
- A2 all four levers on (`velocity_init="gradient"`, `L0_inf=True`, `nan_eps_halving=True`,
  `precond_source="final"`), seed 0.
- A3 regression bed: 100x anisotropic Gaussian, cold start, F1 on — F1 must not break the
  already-working case. Script `handoff/aniso_bed_f1.py` (the F1-on variant of the preserved
  `handoff/aniso_bed.py` baseline; writes into `handoff/diag_levers/`).

**Pre-registration triplet (A1, primary).**
- *Cause hypothesis:* prior-init failure is (partly) basin selection at initialization — random
  initial velocities let the ensemble relax into the broad quasi-stationary basin; the reference's
  gradient-aligned velocities with per-coordinate equipartition signs are a coherent
  contraction-directed impulse that seeds the tight basin instead.
- *Prediction:* if H holds, width-ratio vs HMC collapses from ~20x median / ~340x max to O(1-3x).
  Honest mechanism doubt, stated up front: Maruyama refresh decorrelates velocity in ~L/eps = O(10)
  steps while the scale-lock develops over hundreds, so H may well be false — then ratios stay
  within ~2x of baseline. An intermediate outcome (max drops >3x to <~100x without reaching O(1))
  = partial mechanism; follow up before concluding.
- *Falsifier:* median width-ratio still >= 10x (and max >= 100x) under ALL of A1, A1c and A2 =>
  velocity init (with or without its L0=inf companion) is not the basin selector; with A1c/A2
  covering bj's first-step-refresh protection, the dropped-reference-mechanism explanations are
  then exhausted and prior-init work moves to the SMC/tempering track (stocktake §5.3).

**Secondary predictions.** A0 (grader amendment 1): full-length baseline — the two measured
baselines are 22.9x/341x (early-stopped at 200; `hmc_ref/overlay_summary.json`) and 21.8x/236x
(3000 full steps; `diag_budget/`), and the widths plateau in Phase 1, so predict median 18-28x, max
150-400x; additionally `p1_nan_frac == 0` at every step (edits didn't change baseline; lens has no
NaN region). A2 ~= A1c in widths (F2 inert at 0% NaN; F3 changes the metric but diag_precond
already falsified a metric-only rescue). A1b within noise of A1. A3 converges to truth (width
ratio ~1 vs analytic sigmas); falsifier: F1 breaks a working bed.

**Threshold derivation.** Primary metric: per-marginal width ratio sigma_LAPS/sigma_HMC over the 8
mass params (median and max), HMC ref = `handoff/hmc_ref/hmc_mass.npy`. Scales in hand: warm-init
achieves 0.92-1.04x; the failure sits at 20x/340x; at M=512 the width-estimate noise is a few
percent (user guidance: ignore <5%). SUCCESS: median < 1.5 AND max < 3 — 3x sits >100x below the
failure mode and far above metric noise, cleanly between the structurally-wrong and fine-tuning
regimes. FALSIFIED: median >= 10x. Between: partial effect, no verdict without follow-up.
(Derived from the measured failure magnitude, the measured warm-init band, and measured metric
noise — not from first principles.)
- *Blind spot:* per-marginal width ratio is blind to joint-shape/correlation error and location
  bias. Complement: median true logp of the final ensemble (warm reference +163; annealing v1
  reached +50 while still 18x wide) and a corner overlay vs HMC, VIEWED by the orchestrator before
  any convergence conclusion (user requirement).

**Expected plot.** If H holds: A1 corner contours coincide with HMC's like the warm overlay
(`hmc_ref/overlay_corner.png`). If the falsifier fires: A1 contours remain a diffuse cloud that
swallows the HMC posterior as a point, as in the baseline prior overlay.

**Cost.** 1 interactive GPU node (4x A100), one salloc: model build ~3 min + 5 lens arms x ~4 min +
bed ~2 min => ~30-40 min wall-clock inside a 4 h allocation.

**ADJUDICATION (run 2026-07-06, job 55600450, artifacts `handoff/diag_levers/`).**
- Suite 24/24 (transcript archived at top of `diag_levers/run.log`). A0 in-band: 24.02x median /
  335.9x max (predicted 18-28 / 150-400); `p1_nan_frac == 0` in all arms; A3 bed intact with F1 on
  (all 8 dims final/true 0.9-1.0, `diag_levers/aniso_bed_f1.npz`).
- **Falsifier FIRED**: std-ratio median 24.10 (A1), 23.96 (A1c), 24.41 (A2) — all >= 10x, and all
  within the A1-vs-A1b seed spread (A1b: 26.42) of the A0 baseline (24.02). Per the attribution
  rule, NO lever effect. Gradient/equipartition velocity init, first-step L=inf, NaN eps-halving,
  and end-state preconditioner are all falsified as prior-init rescues. The dropped-reference-
  mechanism explanations are exhausted.
- **Plot-vs-metric conflict, resolved into a NEW structural finding (see §5b):** the zoom corner
  (`levers_corner_zoom.png`, viewed; in-box samples only — read WITH `levers_corner_full.png`)
  shows arm contours only ~1.3x wider than HMC inside the HMC+-6sigma window — irreconcilable
  with a uniformly broad cloud (box-mass counterfactuals: uniformly-24x-wide ~2e-6; at the
  measured per-dim widths ~1e-7; measured 0.39-0.43). The prior-init ensemble is a MIXTURE.
  Status: adjudicated; lever hypothesis dead; mechanism section revised below.

**CERTIFIED CLAIMS (results grading, rigor-grader 2026-07-06):**
- C1 CERTIFIED: falsifier fired. On the demo lens (512 chains, 300+200, early_stop=False) the four
  levers produce no std-ratio change beyond the single-seed spread (2.33 median): |A1-A0|=0.08,
  |A1c-A0|=0.06, |A2-A0|=0.40. Falsified AS PRIOR-INIT RESCUES; does NOT exclude sub-10% effects;
  single seed per lever combination.
- C2 CERTIFIED: A0 in predicted band; p1_nan_frac==0 all arms; A3 bed intact under F1; suite 24/24
  (transcript archived in run.log).
- C3 CERTIFIED (descriptive, exploratory/post-hoc): final-SNAPSHOT prior-init ensemble is a
  core+tail mixture — 39-43% of chains inside HMC+-6sigma (robust at 4/10 sigma; boundary sits in
  a density gap), core widths 1.23-1.38x median / up to 1.55x max (A1b), core mean offsets ~1sigma
  (statistically DISTINGUISHABLE from HMC at 5-11 SE; grader-verified not a truncation artifact —
  a truncated uniform-broad cloud would give ~3.4x); rms-z tail p50~29, p99~678; the historical
  20-370x std-ratio was tail-dominated. NOT certified: the per-chain interpretation ("~40% of
  chains equilibrate" vs a heavy-tailed stationary law — needs per-chain TRAJECTORIES, only final
  snapshots saved); straggler-fraction persistence at 13x budget (diag_budget saved only
  mixture-blind stds — open).

**Grading.** rigor-grader verdict on v1: NEEDS-MORE, three defects — (1) early_stop confound
unstated in the arms (driver already ran early_stop=False; now stated + A0 prediction re-anchored
to the full-length baselines); (2) falsifier closure overbroad with bj's first-step L=inf unported
(now ported as the default-off `L0_inf` lever; arms A1c/A2); (3) the 22/22 unit-suite result was
producer-reported only. RESOLVED: full suite re-run on the GPU node (job 55600450, jax-2026 shifter
stack, the arms' exact stack): **24/24 passed** (two tests added with the L0_inf lever). One
GPU-only flake fixed en route: the F2 no-op test asserted BITWISE step-size equality; computing the
new nan_frac stream changes XLA fusion on GPU (measured 2e-15 rel. ULP drift), so the assertion is
now near-machine-tolerance on the brake's target quantity (step size) only — semantic no-op
confirmed (`p1_nan_frac == 0` exactly). All three amendments applied 2026-07-06; re-submitted to
grader.

### DC-7.2 — Mechanism-settling diagnostics for the resampling design — status: CERTIFIED (design only), v2

*Grader caveats binding the adjudication: (C-a) H_S may be declared only if late hysteresis
crossing is DECISIVELY below the realized positive-control (logp>160) crossing on the same run;
if the positive control is not clearly elevated, the churn estimator's sensitivity is unproven and
no H_S verdict may be rendered. (C-b) the D2 logp-membership/bimodality bound is
logp_typical - (d/2 + 2*sqrt(d/2)) = -28.63 for d=22 (a "-37.27" figure that appeared in one
agent report is arithmetically wrong and appears nowhere in this DC or the scripts).*

*2026-07-06, autonomous resampling engagement (user-authorized long-horizon task). Code: worktree,
`track_chains` diagnostic flag added to `laps_late_adjusted.py` (per-chunk per-chain position+logp
snapshots, default off; suite 25/25 CPU). These are DIAGNOSTIC runs feeding the DC-7.3 resampler
design — labeled as such; D2 is exploratory. v1 graded NEEDS-MORE (slow-mixing degeneracy in the
D1 rule; jitter floor under the churn band; D2 outcome criterion unpinned) — all amendments
applied below as v2; grader also VERIFIED the instrumentation (bitwise no-op on dynamics on CPU;
snapshot 0 = exact initial ensemble; logp snapshots = true logdensity in both phases; chain order
stable). Instrumentation caveat: `p2_keep_per_chain > 1` collection sub-chunks are NOT snapshotted
— "every chunk boundary" holds for keep=1 only (documented in the sampler docstring).*

**D1 — per-chain trajectories on the lens (settles C3's open per-chain question). v2 per grader.**
Claim to settle, REFRAMED (grader): a literal "heavy-tailed stationary law" is excluded a priori —
Phase 2 is Metropolis-adjusted (invariant law = the exact posterior) and the tail sits at
Delta-logp ~ -5e3, which cannot be invariant mass in d=22. The live alternatives are **H_S: FROZEN
tail** (stranded chains; resampling well-founded) vs **H_R: SLOWLY-RELAXING tail** (longer Phase 2
would recover it; resampling = premature truncation). Arms: A0 config (prior-init, 512 chains,
300+200, early_stop=False) + `track_chains=True`, **seeds 0 and 1** (seed spread is the
established yardstick).
- *Membership with HYSTERESIS* (kills typical-set boundary jitter, whose floor ~sqrt(d/2)=3.3 in
  logp sits at the same order as a naive <2%/chunk churn band): leave-core at logp < 0, enter-core
  at logp > 152 (measured gap: straggler p75 = -738 vs core p5 = 152). *Positive control:* the
  same crossing estimator at threshold logp > 160 (core median) on the same data MUST show high
  crossing — proves the estimator sees churn when churn exists.
- *Tie-breaker (the decisive statistic):* tail DYNAMICS over the last third of Phase 2 — median
  per-tail-chain Delta-logp/chunk (drift) and per-chunk rms displacement, with the CORE chains'
  per-chunk displacement as the mixing yardstick.
- *Adjudication:* declare **H_S** only if late hysteresis-crossing is low (<2%/chunk) AND tail
  chains are dynamically frozen (|Delta-logp| drift ~ 0 AND tail displacement << core
  displacement). Low churn + tail displacement comparable to core, or systematic upward logp
  drift => **H_R** (slowly-relaxing), NO H_S verdict. Ambiguous middle = no verdict.
- *Blind spots:* chunk snapshots (25/8 steps apart) alias within-chunk crossings — and this bias
  points TOWARD H_S, the outcome favorable to the planned resampler (named per grader);
  complement = rank stability + the dynamics tie-breaker (aliasing cannot fake a frozen tail:
  displacement is measured on the same snapshots).
- If H_S: resampling = re-initialization of stuck chains, well-founded (and the 13x-budget
  persistence caveat of section 5b is retired by the drift measurement). If H_R: design pivots to
  longer/retuned adjusted phase for the tail (or tempered moves), not discard-and-resample.

**D2 — controlled stranding bed (exploratory; known-answer testbed hunt).**
d=22 anisotropic Gaussian, per-dim sigma = geomspace(1e-3, 0.5, 22) (lens-z-like span), init
N(0,1) draws ("prior" 2x-1000x overdispersed per dim), 512 chains, 300+200 lens-like budget,
early_stop=False, seeds 0/1, `track_chains=True`. Measure: core fraction (all |z_i| < 6 sigma_i),
per-dim std ratio vs analytic, membership churn + dynamics as in D1 v2. Outcome A pre-pinned
(grader): final core fraction in [0.2, 0.7] with a tail beyond 6 sigma and bimodal logp
separation = lens-like mixture => cheap KNOWN-ANSWER testbed for the resampler loop; core
fraction in (0.7, 1) = partial stranding, NOT a testbed. Outcome B (bed converges) claims only:
stranding is NOT reproduced by Gaussian anisotropy up to 1000x per-dim overdispersion (not
"requires lens-specific structure"); resampler validation then runs on the lens vs HMC only.
Either outcome directs the design; no falsifier (exploratory).

**Cost.** ~10 min GPU total (2 lens-scale runs + 2 bed runs) on a fresh 4h interactive node.

**ADJUDICATION (run 2026-07-06, job 55615610, artifacts `handoff/diag_chaintraj/`; seeds agree on
every statistic; spaghetti plots VIEWED both seeds).**
- *Estimator validity:* positive control 12.0-12.6%/chunk >> hysteresis churn 1.0%/chunk — the
  binding caveat C-a is satisfied; the estimator sees churn when churn exists.
- *D1 verdict: **H_S REJECTED — no frozen tail.** Per the pre-registered rule the outcome is H_R
  (slowly-relaxing), with a measured refinement:* arrival is a JUMP process, not diffusive
  relaxation. Facts: (1) core entries happen ONLY in Phase 2 (core fraction = 0 for all 13
  Phase-1 snapshots, both seeds); (2) one-way flow (exits = 0 at every transition); (3) arrivals
  continue to the last snapshot (core 0.03 -> 0.40, still +0.4-0.6%/chunk at the end); (4) only
  66% of final-core chains were in-core at the Phase-2 midpoint (H_S needed >90%); (5) tail
  displacement ~= core displacement (ratio 1.04/1.12); (6) tail chains climbed a median +16.6k
  logp over Phase 2, but the LATE median drift is only +2.0-3.2 logp/chunk; (7) the spaghetti
  shows the tail parked in DISCRETE logp PLATEAU BANDS (dense band -5e3..-2e4; a distinct band at
  ~+40, i.e. only ~120 below the core; scattered flat lines), from which chains individually
  escape and then "zipline" to the core within 1-3 chunks.
- *Mechanism (measured):* depth-stratified escape — heavy-tailed escape times from quasi-stable
  plateau structures; shallow-tail chains arrive in O(10-100) adjusted steps, the deep tail
  escapes at rates that make budget-only fixes hopeless (remaining gap ~5e3 logp at +2.5/chunk
  ~ 16,000 adjusted steps). This RECONCILES the old 13x-budget falsification (1500 adjusted steps
  barely dents the deep tail) with the ongoing-arrival observation.
- *Invariant-mass check:* the +40 plateau sits at Delta-logp ~ -120 from the core; e^120 cannot be
  bought by volume in d=22 unless a secondary basin is ~1e50x wider — negligible posterior mass.
  Same a fortiori for the -5e3 band. The tail is transient non-equilibrium mass. CAVEAT for other
  systems: a logp threshold cannot distinguish "transient plateau" from "genuine wide secondary
  mode"; on this lens the Delta-logp magnitudes exclude the latter.
- *D2 verdict: Outcome B* — the graded-anisotropy Gaussian bed (2-1000x per-dim overdispersion)
  fully converges (core fraction 1.0 both defs, std ratio 1.00-1.05, zero tail): stranding is NOT
  reproduced by Gaussian anisotropy up to 1000x. No cheap testbed; resampler validation runs on
  the lens vs HMC. (Also retires "prior-init envelope" as a pure-scale story: the plateaus are a
  structural feature of the lens target.)
- Section 5b's open per-chain question is now SETTLED (mixture = late-arrivals + plateau-parked
  chains; not a stationary heavy tail; not permanently frozen either), and the 13x-budget
  persistence caveat is retired by the drift measurement.

### DC-7.3 — Phase-boundary/mid-phase straggler resampling (the fix) — status: CERTIFIED (design/pre-registration + instrumentation only), v2

*Re-grade 2026-07-06: all five v1 amendments verified applied; tests green in the CPU stack with
the ACTIVE resample path exercised (test_laps_resample.py 6/6; full suite 31/31). NO empirical
rescue claim certified — arms not yet run. BINDING for the arms grading: the dup_decorr
"saturated" boolean can read True while duplicates remain correlated in the tight (theta_E)
direction — the decorrelation CURVE and the corner-zoom must be VIEWED before any SUCCESS verdict;
the n_eff noise bar uses n_survivors (conservative lower bound on effective ancestors).*

*2026-07-06. Design informed by DC-7.2. NOTE ON THE PRE-REGISTERED PIVOT: DC-7.2 v2 pre-committed
"If H_R: design pivots to longer/retuned adjusted phase ... not discard-and-resample." That wording
assumed smooth relaxation. The measured escape-time structure changes the calculus: (i) "longer
adjusted phase" is quantitatively hopeless (~16k steps for the deep tail vs 200 baseline, from the
measured late drift); (ii) the tail carries negligible invariant mass (Delta-logp <= -120 =>
occupation probability < e^-120), so replacing straggler STATES mid-run is a re-INITIALIZATION of
an MH-invariant chain, not truncation of posterior mass — pre-resample states are discarded as
burn-in, never averaged. This is exactly the validity class of warm-init (init from an SVI
surrogate), which is CERTIFIED at 0.92-1.04x vs HMC. The grader is asked to attack this departure
explicitly.*

**Claim.** Prior-seeded LAPS + one mid-Phase-2 resampling step matches HMC on the lens at
~baseline compute (distributional claim; adjudicated vs HMC and vs a same-code warm-init arm).

**Mechanism (flag-gated, default off).** After a settling segment of Phase-2 adaptation (T2a
chunks), host-side: (1) classify chains by logp (available in the carried state, zero extra
likelihood evals): survivor iff logp > max_ensemble_logp - Delta, Delta = d/2 + 4*sqrt(d/2)
(~24.3 at d=22; posterior logp spread is ~sqrt(d/2)=3.3, so Delta is ~7 spreads below the max —
generous; misclassification cost is trivial because this only selects INITIAL STATES: a
"straggler" at Delta-logp -20 is a fine init). (2) Replace each straggler's state by a
uniform-with-replacement draw from the survivor states (positions only; the adjusted kernel does
a full momentum refresh each trajectory, so no momentum surgery). (3) REBUILD the diagonal
preconditioner from the resampled ensemble Var (the boundary metric was poisoned by the broad
mixture — measured ~100x too large in tight dims, forcing eps 0.012 vs warm 0.62) and rebuild the
Phase-2 kernel. (4) RESTART the step-size bisection (fresh bracket, eps0 = L/N from the resampled
ensemble, unfrozen) and run the remaining T2b chunks; samples come from the final ensemble as
usual. Guard: if survivors < s_min (default 32, floor for a meaningful resample pool + metric),
skip resampling entirely and warn (degenerate-Phase-1 rail).

**Pre-registration triplet.**
- *Cause hypothesis:* the residual prior-init failure is straggler chains parked on negligible-
  mass plateaus + a Phase-2 metric/step poisoned by the mixture; replacing straggler inits with
  core states and re-tuning metric+eps removes both.
- *Prediction:* final ensemble ~100% in-core by construction + MH evolution; per-marginal
  std-ratio vs HMC lands in the warm-init class (0.9-1.2x; warm measured 0.92-1.04x), mean
  offsets < ~0.3 sigma; Phase-2 eps after re-tune rises toward the warm scale (0.012 -> O(0.1-1)).
- *Adjudication bands (v2, grader-pinned; mirrors DC-7.1):* **SUCCESS** = every mass-param
  std-ratio in [0.9, 1.2] AND |mean offset| <= 0.3 sigma_HMC, in >= 2 of 3 seeds (the warm-class
  claim). **FALSIFIED** = any ratio outside [0.8, 1.5] OR |offset| > 0.5 sigma in 2+ seeds.
  **BETWEEN = partial, NO "matches-HMC" claim — iterate.** 0.5 sigma is a PROGRESS bar (halves the
  core's inherited 1-sigma offset), not a warm-class bar; a 0.45-sigma offset is a real
  mis-location and must not be reported as success.
- *Threshold derivation:* [0.9, 1.2] = warm band (0.92-1.04) + ratio noise; noise bar RECOMPUTED
  from effective sample size (n_eff gate below), NOT M=512. Blind spots: std-ratio is
  mixture-blind (covered: core fraction reported; corner overlays VIEWED) and joint-shape-blind
  (covered: corner pairwise panels); duplicate-induced variance deflation would UNDERSTATE widths.
- *n_eff GATE (v2):* the width claim is interpretable ONLY if the duplicate-decorrelation curve
  (rms straggler-donor distance vs chunk) SATURATES at the core's internal spread before the final
  snapshot, AND the ratio noise bar is recomputed from n_unique ancestors (~n_survivors), not 512.
  If the curve has not saturated: no width verdict at all.
- *Expected plot:* corner overlay of resampled-prior arms on HMC coincides like the warm overlay;
  duplicate-separation curve rises and saturates at the core's internal spread.

**Arms** (512 chains, 300 unadj, T2a=13 chunks (~104 traj) + T2b=18 chunks (~144 traj) ~= 248
total adjusted traj vs baseline 200; early_stop=False; track_chains=True):
- R1/R1b/R1c: resampled prior-init (`p2_resample_mode="replace"`), seeds 0/1/2.
- M1 (v2, grader-required control): `p2_resample_mode="retune_only"`, seed 0 — rebuilds metric+eps
  from the SURVIVOR subset and restarts bisection but MOVES NO POSITIONS. This (i) isolates the
  poisoned-metric fix from position replacement and (ii) gives the DC-7.2 pre-committed "retuned
  adjusted phase" alternative a fair shot at the SAME budget (the ~16k-step hopelessness estimate
  was computed at the poisoned eps). If M1 rescues, resampling is unnecessary — the simpler fix
  wins; if M1 fails while R1 passes, position replacement is established as necessary.
- W: warm-init, seed 0, same code+budget (fresh yardstick, same-day apparatus).
- A0-r control: prior-init, resampling OFF, seed 0 (mixture reproduces; also gives the
  no-resample core fraction at the same T2 for an arrival-rate check).
- T2a=13 chunks derived from D1: core fraction ~30% by chunk 13 (both seeds) — enough survivors
  (>=150) for a stable metric; earlier would resample from a thinner core.
**Doubt report (v2 additions, grader).** (i) The HMC reference is itself warm/VI-seeded into the
same basin: HMC agreement does NOT independently exclude a missed secondary mode; the
invariant-mass exclusion applies only to plateaus the finite ensemble VISITED (grader verified the
visited-plateau bound quantitatively: needed volume e^120 ~ 1e52 vs max geometrically available
~1e41.5 within the prior support — negligible with >10 orders margin). (ii) The committed unit
tests must be green in the CPU stack with the ACTIVE path exercised before launch (v1 fixtures
skipped it; the mechanism itself was independently probe-verified by the grader: donors disjoint,
all-clear-cut post-resample, duplicates decorrelate, width 0.91-1.17x on the probe).
**Cost.** ~30 min GPU (6 lens arms + build).

**ADJUDICATION (run 2026-07-06, job 55615610, artifacts `handoff/diag_resample/`; plots VIEWED:
dup_decorr.png + resample_corner_zoom.png + _full.png).**
- **SUCCESS band met in 3/3 seeds** (needed 2/3): per-param std-ratios R1 [0.969, 1.071],
  R1b [0.958, 1.029], R1c [0.946, 1.057] — all inside [0.9, 1.2]; |offsets| max 0.055/0.114/0.108
  sigma — all under 0.3. The same-day warm arm W spans [0.936, 1.021] / 0.068 sigma: the resampled
  arms are indistinguishable from warm-init within the n_eff noise bar (~0.057 at n_unique
  153-165). Core fraction 1.000 all R arms.
- **Binding visual gates pass:** straggler-donor rms rises steeply, reaches the final-core
  saturation reference by ~4 chunks post-resample and fluctuates around it for the remaining ~14
  (all seeds) — duplicates genuinely decorrelated, including no visible tight-direction anomaly in
  the zoom corner, where R1 contours coincide with HMC/W in every 1-D and pairwise panel.
- **M1 (retune-only) FAILS: 24.4x median / 323x max, core 0.441** — indistinguishable from the
  A0r no-resample control (24.8x/344x, core 0.424). Metric+eps re-tune alone does nothing at this
  budget: the pre-committed "retuned adjusted phase" alternative is now EMPIRICALLY falsified at
  matched budget, and position replacement is established as the operative mechanism.
- Secondary predictions: post-resample eps re-tuned 0.012 -> 0.215-0.233 (predicted O(0.1-1));
  accept 0.69-0.72 at target; survivors 153-165 at chunk 13 (predicted >=150).
- Verdict: **prior-seeded LAPS + mid-Phase-2 straggler resampling matches HMC in the warm-init
  class on the demo lens at ~1.2x baseline adjusted budget.**
- **RESULTS CERTIFIED (rigor-grader, 2026-07-07)** — every ratio/offset independently recomputed
  from the mass arrays (matches to 4 dp); plots re-viewed; theta_E deflation check passed (R-arm
  theta_E ratios 0.958-1.0005, at/above warm's 0.949 — no tight-direction undershoot); M1/A0r
  controls verified. SCOPE: single lens; ~1.2x baseline adjusted budget; the HMC reference is
  itself in-basin, so agreement does NOT exclude a missed global secondary mode (invariant-mass
  exclusion covers only visited plateaus); M<512 not covered (DC-7.4). Grader honesty credit:
  the MH phase demonstrably ERASED the core's inherited 1.25x/1sigma bias (not a by-construction
  pass). Bookkeeping: DC-7.2 and DC-7.3 both ran on allocation 55615610 (sequential steps,
  separate artifact dirs diag_chaintraj/ vs diag_resample/ — not conflated).

### DC-7.4 — Efficiency + 128-chain validation (user goal) — status: v2 (grader amendments applied verbatim; launched without a further design round-trip — the v1 grading supplied the exact amendment set [arm-to-arm gate with its band arithmetic; A0r128 control], both applied unchanged; results grading to re-verify band application)

*2026-07-07. The user's stated end goal includes efficient operation down to 128 chains with
512-chain controls. Claims: (Q1) the resampled prior-start pipeline works at M=128; (Q2) CHANGE-B
thinned collection (keep=4) recovers estimator precision at 128 chains; (Q3, exploratory) Phase-1
can be trimmed (arrivals happen in Phase 2 — D1 showed the ensemble reaches the plateau band by
~step 200 of Phase 1).*

**Arms** (all prior-init, early_stop=False, track_chains=True, p2_resample_at_chunk=13,
num_adjusted_steps=248 unless noted):
- R128a/b/c: num_chains=128, seeds 0/1/2, `p2_resample_min_survivors=24` (128-chain guard:
  D1 core fraction ~30% at chunk 13 predicts ~38 survivors; 24 = floor at which a diag-Var metric
  estimate has ~15%/dim relative error — acceptable; if survivors < 24 the guard skips and the arm
  fails informatively).
- W128: warm-init, num_chains=128, seed 0 (small-M yardstick).
- R128k: as R128a plus `p2_keep_per_chain=4, p2_thin=5` (512 samples from 128 chains; +20 frozen
  trajectories; NOTE: collection sub-chunks are not snapshotted — decorr curve uses the adaptation
  phase only, unaffected).
- A0r128 (v2, grader-required): prior-init, num_chains=128, resample OFF, seed 0 — same-M negative
  control; the M=128 apparatus must still reproduce the mixture pathology, else an R128 pass
  proves nothing.
- Rtrim: num_chains=512, num_unadjusted_steps=200 (Phase-1 trim probe), seed 0.
**Adjudication bands (v2, grader-amended).** At M=128 the absolute-vs-HMC band is noise-dominated
(ratio SE 0.117 at n_eff~38 ancestors + 0.063 from 128 samples), so the **load-bearing
quantitative gate is ARM-TO-ARM: per-param std(R128)/std(W128)** — W128 is the same-M realization
of the correct answer, cancelling the small-M systematics that absolute-HMC comparison cannot.
PRIMARY gate (valid CONDITIONAL on the duplicate-decorrelation saturation gate passing, which
licenses n_eff -> M): every mass-param R128/W128 width ratio in **[0.73, 1.27]** (= 1 +- 3 *
sqrt(2)*0.063, the paired 128-sample noise) in >= 2/3 seeds, AND |offset vs HMC| <= 0.5 sigma
(3 SE at n_eff 38). If saturation fails, the conservative ancestor bar applies ([0.6, 1.4]) and
only a WEAK verdict may be reported. Secondary (reported, not load-bearing): absolute-vs-HMC
ratios expected in [0.7, 1.5]; corner overlay VIEWED as complement (grader: the eye cannot
resolve 1.4x at M=128 — it must not be the discriminator). SUCCESS(R128k) = flattened (512-sample)
absolute ratios in [0.75, 1.45] + R-hat/ESS complement reported. FALSIFIED = arm-to-arm ratio
outside [0.6, 1.9] or |offset| > 0.8 sigma in 2+ seeds, or guard-skip (survivors < 24) in 2+
seeds, or A0r128 fails to reproduce the mixture (apparatus invalid — no verdict on R128 at all).
BETWEEN = no verdict. Rtrim is exploratory (Q3): report against the DC-7.3 bands, no hard
falsifier; if it passes, Phase-1 cost drops 33%.
**Blind spots.** Small-M ratios are noise-dominated (named above; visual gate load-bearing);
keep=4 within-chain samples are thinned but correlated (report R-hat/ESS over the (128,4) sample
array as complement).
**Cost.** ~20 min GPU (128-chain arms are ~4x cheaper per step; 6 arms + build).

**ADJUDICATION (run 2026-07-06 22:52, job 55615610, artifacts `handoff/diag_resample128/`;
corner128_zoom + dup_decorr128 VIEWED).**
- Saturation gate: R128a/b/c + Rtrim all saturated (final rms 0.367-0.407 at refs 0.384-0.421) —
  primary band licensed.
- **PRIMARY arm-to-arm gate PASSED 3/3 seeds** (needed 2/3): std(R128)/std(W128) per param —
  R128a [0.964, 1.100], R128b [0.834, 1.038], R128c [0.940, 1.101], all inside [0.73, 1.27].
  |offsets vs HMC| 0.154/0.207/0.168 sigma <= 0.5. Secondary absolute ratios [0.871, 1.150] within
  [0.7, 1.5]. Corner complement: R/W/keep-4/trim arms coincide with HMC in every panel; no
  width/offset anomaly visible.
- **A0r128 control reproduces the mixture at M=128** (25.3x median / 400x max, core 0.344) —
  apparatus valid; the rescue is doing the work at small M too.
- Guard: survivors 31/51/47 >= 24 (R128a at 31 — thin but above floor; seed-to-seed spread of the
  survivor count at M=128 is large, as predicted).
- **Q2 (keep=4): PASSED** — R128k flattened ratios [0.968, 1.089] (band [0.75, 1.45]); vs_W128
  [0.936, 1.035]; unsplit R-hat(m=128, n=4) max 1.64 with crude ESS 190-398 of 512 kept samples:
  thinned draws are partially correlated (honestly reported) but effective samples ~1.5-3x the
  128 single-snapshot baseline — precision recovered at fixed chain count.
- **Q3 (Rtrim): PASSED the full DC-7.3 512-chain bands** with Phase-1 at 200 steps: absolute
  ratios [0.977, 1.052], offsets <= 0.077 sigma, survivors 189. Phase-1 budget cut 33% with no
  degradation (exploratory arm; single seed — flag for any production default change).
- Verdict: **the resampled prior-start pipeline works at M=128 with 512-chain-control fidelity;
  keep=4 collection recovers estimator precision; Phase-1 can plausibly be trimmed to 200 steps.**

**RESULTS CERTIFIED (rigor-grader, 2026-07-07; all numbers independently recomputed, plots
re-viewed, no deflation, no theta_E undershoot). Grader-surfaced caveats (binding):**
(i) the M=128 width claim rests on exactly 3 independent seeds; survivor counts 31/51/47 with
seed 0 just 7 above the guard floor 24 — robustness to survivor variation below ~31 untested;
(ii) R128k shares R128a's seed-0 run (same 31 ancestors) — Q2 evidence only, NOT a 4th width
seed; (iii) the arm-to-arm denominator is a single warm seed (cross-checked by the passing
absolute-vs-HMC ratios); (iv) keep=4 precision gain is ~1.5-3x effective (R-hat(n=4) 1.64,
correlated draws), not 4x; (v) R-arm re-tuned eps ~2.4x smaller than warm's at matched accept —
benign for widths; per-gradient efficiency vs warm NOT benchmarked; (vi) model card inherited
from the DC-7.3 apparatus.

## 7b. ENGAGEMENT VERDICT — CERTIFIED (grader, 2026-07-07)

**Prior-start (no MAP/SVI) LAPS with mid-Phase-2 straggler resampling is WORKING and UNBIASED
against in-basin warm-seeded HMC on the demo lens for M in {128, 512}**: per-marginal widths in
the warm-init class, locations within <=0.21 sigma of HMC, at ~1.2x baseline adjusted budget.
Scope caveats that must survive into any downstream claim:
1. **Single lens** (the jax-demo scene); no multi-lens or real-data generalization claimed.
2. **"Unbiased" = matches the in-basin HMC reference** (itself warm/VI-seeded); agreement does NOT
   exclude a missed global secondary mode (invariant-mass exclusion covers only visited plateaus).
   Not unbiasedness vs truth (the posterior itself is offset from truth in gamma/eps2 — a
   posterior property shared by HMC).
3. **"Efficient" is scoped to:** 4x-cheaper 128-chain operation with keep=4 precision recovery
   (~1.5-3x effective, correlated draws), ~1.2x baseline adjusted budget, and a PLAUSIBLE 33%
   Phase-1 trim (single-seed, not a certified default). Per-gradient efficiency vs warm-init not
   benchmarked.
4. M=128 robustness bounded by small, seed-dependent survivor counts near the guard floor.
5. Warm-init LAPS remains the recommendation whenever a usable SVI/MAP surrogate exists;
   prior-start + resampling is the validated no-surrogate path.

## 8. Artifacts
- HMC reference: `handoff/hmc_ref/` (hmc_mass.npy, hmc_summary.json, overlay_corner.png).
- Contraction / stragglers: `handoff/diag_phase1/`. Step-size controller: `handoff/diag_stepsize/`.
- Finiteness: `handoff/diag_freeze/`. Phase-2 acceptance: `handoff/diag_p2accept/`.
- Budget: `handoff/diag_budget/`. Correct-precond: `handoff/diag_precond/`. Annealing v1: `handoff/diag_anneal/`.
- Controlled beds: `handoff/aniso_bed.py`, `handoff/escalate_bed.py`, `handoff/floor.py`
  (preserved in-tree 2026-07-06; formerly only in the ephemeral job dir).
- DC-7.1 lever run: `handoff/diag_levers/` (per-arm mass/z samples, p1 std trajectories,
  p1_delta_max, `levers_summary.json`, corner overlays full+zoom, `aniso_bed_f1.npz`,
  `run.log` incl. archived 24/24 pytest transcript). Driver `handoff/diag_levers.py`,
  bed `handoff/aniso_bed_f1.py`.
- Sampler changes (isolated, flag-gated, default off) in
  `src/gigalens_research/inference/laps_late_adjusted.py`: `p2_precond_var` override (diagnostic
  hook), and the stocktake F1/F2/F3 levers `velocity_init` / `nan_eps_halving` / `precond_source`
  (+ always-on `p1_nan_frac` diagnostic stream); unit tests
  `src/gigalens_research/inference/tests/test_laps_levers.py`.
