# LAPS engagement — lab notebook / pre-registration ledger

Goal: build & validate a robust GIGALens implementation of **LAPS** (Robnik 2026 microcanonical
adaptive sampler). Sources of truth: the paper (`papers/LAPSRobnik2026.pdf`) + blackjax 1.5
reference (`blackjax/adaptation/laps.py`, `laps_burn_in.py`). Method discipline (`docs/method-discipline.md`)
governs. proposer ≠ grader at every step. Internals validated against spec prediction with equal
weight to results.

## File-location conventions
- **Source-of-truth files** read by absolute path from the MAIN checkout (user's current, partly
  uncommitted state): `/global/u1/l/linusu/GIGALens-Code/...`
- **blackjax reference** (host path, no container needed to read):
  `/global/homes/l/linusu/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages/blackjax/adaptation/`
- **New deliverables** isolated in this worktree: `.claude/worktrees/laps/`
- **CPU env** (to run anything): shifter container, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, explicit
  PYTHONPATH (recipe in the `aps-mclmc-engagement` memory).

## Phase status
- [x] P1 ground-truth spec → `laps_spec.md` (+ paper/blackjax sub-specs)
- [x] P2 translation design → `laps_gigalens_translation.md` (rec: Option B, build on in-tree MCLMC APIs)
- [x] P3 audit → `laps_existing_audit.md` (+ per-file audits): laps.py = EMAUS-code, salvageable, switch-never-
      fires is dominant cause (CONFIRMS D1 cause hypothesis); COLDSTART3 = incomplete stub → rewrite
- [x] **CHECK-IN with user (2026-06-29)** — DECISIONS: (1) build fresh on in-tree MCLMC kernels (Option B,
      no laps.py salvage); (2) support both init modes, default warm from qz; (3) paper-faithful schedule
      default + EMAUS A/B flag. Defaults accepted: v1 minimal faithful core (diagonal D̃, no superchains/
      split-R̂/Hutchinson); lensing target TBD before P6.
- [→] P4 build — internals-first.
      - [x] P4a `laps_core.py` + tests: all 6 reductions match pre-registered analytic values; **D1 falsifier
        reproduced** (EMAUS (σ/μ)²-on-x_i = 1.0e6 never fires; paper δ-on-x_i² = 6.3e-3 fires). ACCEPTED.
      - [x] SUBSTRATE SYNC: worktree branched from origin/main `31cad04` (pre-MAMS) → synced
        blackjax_updated_utils.py / mams.py / mclmc.py / __init__.py from main working tree (`2bf8c0e` +
        uncommitted edits). `_build_adjusted_kernel_shardmap` now present; all imports clean CPU+x64.
      - [x] P4b `laps_late_adjusted.py` (`LAPS_late_adjusted` + `_JIT`): runs end-to-end CPU/mesh no NaN;
        D̃ 12.2→0.005; EEVPD_obs tracks wanted; **D1 confirmed dynamically** (paper x_i² switches @178/thr0.05,
        EMAUS x_i never); mean≈noise, var max rel err 8.4%; Phase-2 accept→0.677 vs 0.70. Control flow =
        Python outer loop over fixed-length shard_map+scan chunks (host-side switch eval). ACCEPTED-with-flags.
        Deviations: in-tree MAMS = Hamiltonian variant, no partial refresh (L_proposal recorded-not-applied);
        random velocity init; diagonal D̃ only (per scope).
      - [→] SWITCH-STATISTIC RESOLUTION (pre-reg D2 below) — blocks P5.
      - [x] P4c grader audit (`audit-laps-late-adjusted.md`): faithful + sharding-correct, NO correctness-
        breaking bug. Equipartition/step-law/EEVPD/L/switch/boundary/sharding/control-flow/diagnostics/init
        all CONFORM. **MAMS partial-refresh resolved = efficiency-only, NOT bias** (full refresh is a
        different-but-exact proposal). MUST-FIX before P5: (#1) Phase-2 freeze not latched — ε can un-freeze
        (verified 0.71→frozen, 0.55→un-frozen); spec wants one-way latch; risks mid-adaptation final ensemble.
        Robustness: (#5) crash when num_unadjusted_steps not divisible by chunk_size.
      - [x] CONSOLIDATED FIX ROUND DONE+VERIFIED: (a) freeze latched (sticky frozen, holds ε — test 7 pass;
        smoke froze step 11, ε const 5.61); (b) self-calibrated switch default (switch_mode∈{self_calibrated,
        absolute,m_scaled}, k=1.5; online floor_i; absolute guard raises when thr<√(2/M) — test 8 pass; smoke
        fired step 100 @M512 where absolute 0.01 never); (c) chunk-divisibility (`_chunk_sizes`, test 9 pass).
        All 9 unit tests pass; smoke var rel err 0.103, accept 0.717. NO regressions.
- [x] **P4 BUILD COMPLETE** — faithful, sharding-correct, internals-validated LAPS; well-posed switch.

### Pre-registration D2 — switch statistic δ vs ensemble size M (MUST resolve before P5)
Observation: windowed δ=σ/μ of ensemble-mean x_i² has an equilibrium floor ≈√(2/M) (measured 0.044 @ M=512).
Literal threshold 0.01 then needs M≳2e4 — yet paper uses M=4096 (floor ≈0.022>0.01 under this definition)
AND claims the switch fires. CONTRADICTION ⇒ our δ definition is likely wrong (internals mismatch = our
understanding or impl is wrong, per the discipline). 
- Hyp A: paper's σ is a standard ERROR (std/√T over the window) or detrended fluctuation → floor ≪√(2/M)
  → literal 0.01 fires at M=4096. Hyp B: our signal-std δ is right and 0.01 is effectively M-tuned / relies
  on the budget cap.
- Test (falsifier): measure the equilibrium floor of EACH candidate δ definition on an exactly-sampled
  Gaussian ensemble at M∈{512,4096,16384}. Correct definition = the one whose M=4096 floor < 0.01 (so the
  paper CAN fire) and matches the paper's stated δ~M^-1/2 (§D). Predicted under Hyp A: floor_4096 ≪ 0.01.
- Outcome gates the default threshold (literal 0.01 vs an M-aware k·M^-1/2) — a hyperparameter that must be
  justified by mechanism + empirics, not assumed.

**D2 RESOLVED (`laps-switch-resolution.md`).** Paper δ (p.5 Eqs 10-11) = σ/μ of windowed per-step ensemble-
mean of x_i², σ = plain sample std ddof=1 over window steps (NO /√T, NO detrend). Our `phase1_switch` is
FAITHFUL — no estimator bug. Measured equilibrium floor = √(Var_ρ[x²]/M)/E_ρ[x²] = √(2/M) Gaussian:
0.068@512, **0.024@4096**, 0.012@16384; max_i δ<0.01 needs M≳3e4. Autocorr doesn't lower it. The paper's
"∼M^-1/2" DROPS the √2 prefactor → its own M=4096 floor (0.024) exceeds 0.01, so the literal switch is
ill-posed at our M (silently burns maxiter). **GIGALens decision: SELF-CALIBRATED switch (default) =**
fire when `max_i(δ_i/floor_i) < k`, floor_i=√(Var_ρ[x_i²]/M)/E_ρ[x_i²] computed ONLINE, k=1.5 default
(exposed). Mechanism: "drift shrunk to ~the irreducible M-noise"; well-posed at any M; posterior-aware
(adapts prefactor for non-Gaussian marginals). Keep literal-0.01 + k·M^-1/2 as opt-in modes; ADD a guard
that ERRORS (not silent no-op) when a literal threshold < √(2/M). **This is a justified DEVIATION from the
paper's literal switch** (which can't fire at our M). Validate k in P5: switch must be "ripe" — residual
2nd-moment bias ≤ noise floor at the switch point; sweep k, check not-premature / not-wasteful.
- [ ] P5 CPU validation (internals AND results; pre-registered A/B paper-vs-EMAUS) — predictions below
- [ ] P6 lensing handoff

## Phase 5 PRE-REGISTRATION (written before any validation run)

**Testbed ladder (known-answer, CPU-fast M≈512, d≈8–16, seconds/run):** (T-iso) isotropic N(0,I);
(T-ill) ill-conditioned diagonal Gaussian (cond ~1e2–1e3); (T-corr) dense-correlated Gaussian; (T-curve)
mild banana/Rosenbrock (unimodal, curved — multimodality is paper-OOS, excluded). Each target has analytic
mean/Var → exact b² (paper success metric = 2nd-moment bias < 0.01).

**Two-tier diagnostics for EVERY run (plots BEFORE metrics):**
- INTERNALS (validate the machinery, equal weight to results): D̃(t)→0; EEVPD_obs/EEVPD_wanted →1; δ_i/floor_i
  trajectory + switch point; Phase-2 acceptance→target±3%; p2_frozen latches. Falsifier: any internal that
  does NOT match the spec prediction = our understanding or impl is wrong (regardless of result quality).
- RESULTS: per-dim mean/Var recovery; b²_max,b²_avg<0.01; coverage if cheap.

**Pre-registered experiments + predictions:**
- **E-A (D1 A/B, central):** grid schedule∈{paper,emaus} × switch_obs∈{x_i²,x_i}. PRED: x_i switch FAILS to
  fire on any bed with a near-zero-mean coord (T-iso has them by symmetry) → Phase-1 burns budget → biased;
  x_i² fires "ripe". Paper schedule F(C·D̃) reaches b²<0.01 in ≤ the gradient calls of EMAUS C·D̃^{3/8}.
  FALSIFIER: if x_i also fires and both reach b²<0.01 equally fast → D1 wrong, switch/schedule not the cause.
- **E-B (self-cal k sweep, justify the default):** k∈{1.0,1.5,2.0,3.0}. PRED: switch "ripeness" — at the
  switch point residual 2nd-moment bias should be ≤ ~k× the M-noise floor. Too-small k → premature switch →
  Phase-2 starts under-equilibrated → elevated b². Too-large k → wasted Phase-1 steps, same b². The justified
  default minimizes cost s.t. residual bias ≤ floor. PRED: k≈1.5 is at/just past the knee. FALSIFIER: if b²
  is flat across k, the switch timing doesn't matter (then default to the cheapest k).
- **E-C (warm vs cold, validates decision #2):** warm qz (near-posterior) vs cold (prior/N(0,I)). PRED: warm
  → switch fires early but result UNBIASED if qz≈posterior; with a deliberately-off qz (inflated/shifted),
  warm fires the switch BEFORE equilibration → biased Phase-2 = the "skipped warming" risk → motivates a
  minimum-Phase-1 guard. cold → longer Phase 1, same final b². FALSIFIER: if an off-qz warm start still gives
  correct b², the early-switch risk is not real and no guard is needed.
- **E-D (C sweep):** C∈{0.025(paper),0.05,0.1(emaus)}. PRED: smaller C → smaller asymptotic (discretization)
  bias → lower D̃ floor / lower b², at more gradient calls. Verify the paper C=0.025 reaches the D̃ floor.
- **E-E (precond, design Q5):** on T-corr, diagonal precond vs (if cheap) a dense metric. PRED: diagonal
  under-performs on strongly-correlated targets (Phase-2 acceptance harder to hit / more steps); quantify the
  gap to decide if dense precond is needed for correlated lens posteriors.

Orchestration: builder writes the harness (targets+diagnostics+b²+PLOTS+grid driver) and runs the fast core
grid → artifacts; I examine PLOTS first then adjudicate each prediction; an independent grader checks the
artifacts (proposer≠grader). I launch any longer grids myself via Bash bg+monitor.

- **E-F (M-scaling unbiasedness, the principled result test — added after seeing the b² floor):** the b²
  floor is 1/M (Ê[x²] sampling var = Var[x²]/M ⇒ E[b²_i]=1/M). Run T_iso + T_ill at M∈{512,2048,8192}. PRED:
  b²_avg ∝ 1/M (slope −1 in log-log), i.e. the sampler is unbiased and b² is pure finite-M noise. FALSIFIER:
  b²_avg plateaus above 1/M as M grows → residual algorithmic bias. This REPLACES anchoring on the arbitrary
  0.01 line — derive the threshold from M, don't assume it.

HARNESS SANITY (examined the PLOT, not just numbers): T_iso default clean+UNBIASED — D̃ at floor, EEVPD→1,
2nd-moment scatter SYMMETRIC about truth (not one-sided ⇒ noise not bias), b²_avg 1.8e-3≈1/M, accept→0.70,
freeze latched. KEY PLOT INSIGHT: on warm+easy targets the switch is WINDOW-limited (T=60), max δ/floor~1.1
≪ k=1.5 — so k/ripeness is only exercised by cold/hard beds (E-B/E-C). Harness `experiments/laps_validation/`.

### CORE GRID RESULT (11 runs, warm, M=512) — adjudicated
- **E-A switch half CONFIRMED:** every x_i (emaus) switch NEVER fires (switched=False, burns 300-step budget)
  on T_iso & T_ill; every x_i² (paper) switch fires @75. D1 mechanism reproduced on real runs.
- **Default config UNBIASED on all 4 beds:** plots show 2nd-moment recovery ON the y=x line (T_ill across 3
  orders of magnitude), symmetric scatter; b² at the 1/M floor. The lone "fail" (T_ill b²_max=1.08e-2>0.01)
  is finite-M noise in a LOW-VARIANCE dim (b² normalizes by Var[x²]→ amplified), NOT bias — PLOT beats the
  number; E-F settles it.
- **E-A schedule half INCONCLUSIVE at warm/M=512** (b² differences are within the ~50% single-seed floor
  spread; several runs sit below 1/M = sampling noise). Both schedules switch @75 (window-limited) ⇒ no
  discrimination. **Methodological finding: warm+easy is non-discriminating for schedule/k/C; even
  never-switching gives b²≈floor (unadjusted MCLMC samples easy targets fine) → the never-switch CONSEQUENCE
  (unadjusted bias) only manifests on HARD beds = exactly the real laps.py "fails to converge" symptom.**
- **Sweep re-orientation:** all sweeps now COLD init + MULTI-SEED (≥3); add b²-trajectory metric from the
  existing p1_obs_sq history → residual 2nd-moment bias AT the switch step (direct k-ripeness test) and
  steps-to-ripe (schedule/C efficiency, less noisy than final b²).

### SWEEP RESULTS (mscale/ksweep/initmode/schedC, 51 runs, 3 seeds) — adjudicated vs pre-registration
- **E-F UNBIASEDNESS CONFIRMED (plot mscale/summary.png):** b²_avg ∝ 1/M, no plateau, on T_iso+T_ill across
  M=512/2048/8192 (b²_avg sits 1.0–1.4× the 1/M floor throughout; b²_max ~3–4× = max-over-8-dim entitlement,
  same slope). Sampler is UNBIASED — b² is pure finite-M noise. Falsifier (plateau) did NOT occur.
- **E-C OFF-QZ SAFE (decision #2 validated):** off-qz warm (2× inflated+shifted) equilibrates slower
  (steps_to_floor 123 vs warm 0) but switch fires @150 AFTER equilibration → final b²_avg 2.24e-3 ≈ floor =
  UNBIASED. The self-calibrated switch keys off ACTUAL equilibration not a fixed step count, so a moderately-
  wrong warm start does NOT cause premature-switch bias. No extra min-window guard needed. SCOPED: moderate
  perturbation, unimodal (multimodality OOS).
- **E-B k JUSTIFIED (plot ksweep/summary.png):** steps_to_floor k-independent (105); switch fires AFTER
  equilibration for all k (ripe); final b² at floor for all k. k=1.0 too strict (switch@300, sits below the
  d=8 δ/floor entitlement ~1.2 → can't fire promptly). k=1.5 = smallest k clearing the entitlement with
  margin. HONEST LIMIT: this bed doesn't discriminate k∈[1.5,3] (equilibration 105 < window-eligibility 150,
  so even k=3 is ripe); preferring 1.5 over 3 is mechanism/conservatism (margin vs premature firing on
  harder/higher-d beds where the entitlement grows + equilibration may exceed the window), NOT empirically
  shown here. Flag for the lensing handoff.
- **D1 DECOMPOSED (schedC):** schedule (paper vs emaus) and C (0.025 vs 0.1) show NO clean efficiency
  separation (steps_to_floor 76–125, no monotone pattern, within noise) and ALL give unbiased final b².
  Mechanism: Phase-2 Metropolis adjustment CORRECTS Phase-1 unadjusted bias → schedule/C affect only
  equilibration speed, not final accuracy, and that effect is modest here. **⇒ the SWITCH (x_i² vs x_i) is
  the DECISIVE failure cause; schedule/C are second-order.** laps.py failed because it never switched (stuck
  in biased unadjusted phase on a hard target), not primarily its step law. This REFINES the original D1
  (which lumped switch+schedule+C); the empirics separate them. Default stays paper-faithful (cheap, correct,
  marginally better-conditioned) but the load-bearing fix is the switch.
- Internals validated across beds: D̃→floor, EEVPD_obs/wanted→1, acceptance→target±3%, freeze latches.

### ADVERSARIAL GRADE of the Phase-5 report (`audit-phase5-report.md`) — over-claims found, WALKING BACK
proposer≠grader caught real problems; corrections in flight before any user report:
1. **Off-qz robustness = CHERRY-PICK (most important).** off-qz **seed1**: b2_max=0.025, b2_success=False,
   max_var_rel_err=0.223 — WORST run in the dataset, equilibrated only ~4 steps before switch. 1/3 hard
   failure was HIDDEN. ⇒ warm-start is NOT unconditionally safe. Needs data (F1) + maybe a stricter ripeness
   guard. Bears on decision #2 + the lensing handoff (lens qz may be off).
2. **"No efficiency separation" = UNSUPPORTED + my plots-before-metrics VIOLATION.** schedC plot shows 4
   cleanly separated tight-error-bar clusters (steps_to_floor 75/78/105/125). Real structure, not noise.
   Walk back to "schedule/C measurably affect equilibration COST, not final unbiasedness."
3. **Unbiasedness over-certain:** trend holds (b2_avg/(1/M) flat 0.92–1.55, no plateau) but M≤8192 can't
   exclude residual bias ~1e-4. And "b2_max 3–4×" WRONG (hits 10.9×). Scope to "no detectable bias to ~1e-4
   over warm Gaussians M≤8192." T_curve/T_corr never M-scaled.
4. **"Phase-2 corrects Phase-1" asserted, no control** → run F3 (Phase-2-off).
5. **k:** k=1.0-too-strict solid (seed2 never fires); k=2/3 byte-identical (WINDOW guard binds, not k) →
   "k=1.5 prompt" overstates. 
6. Acceptance actually 0.654–0.738 (not ±3%); several quoted numbers don't match CSVs → fix all.

**FOLLOW-UPS:** F1 = {warm,cold,off-qz}×10 seeds on T_ill (is off-qz systematically risky? failure rate?);
F3 = Phase-2-off vs on × schedule/C (isolate the correction). Then revise `laps_validation_report.md` with
corrected numbers + honest scope, re-grade if needed, THEN user check-in. (F4 non-Gaussian M-scaling deferred
— lensing handoff is the real non-Gaussian test.)

### FOLLOW-UP RESULTS (F1 offqz 40 runs, F3 phase2off 24 runs) — P5 conclusions finalized
- **F3 Phase-2 correction EVIDENCED (clean):** Phase-2 ON → 1/M floor for ALL (schedule,C); OFF rises above,
  worst emaus/C=0.1 (b²_avg 1.6e-2 OFF → 2.6e-3 ON). ⇒ schedule/C are FIRST-order for unadjusted Phase-1
  bias (paper/C=0.025 lowest OFF=3.5e-3), second-order after Phase-2. **STRENGTHENS the paper+C=0.025 default
  with evidence** (minimizes bias Phase-2 must correct). Corrects my earlier "second-order/within-noise" claim.
- **F1 warm-start: COLD is robust (0/10 fail); off-qz has a RARE catastrophic early-switch (1/10, b²_max
  2.5e-2 @ margin≈0).** persist=2 NOT a clean fix (40% vs 10% by the metric — removes the catastrophe but
  fewer Phase-2 steps; metric is NOISE-BOUND at M=512: gold warm "fails" 30%). PLOT corrected me — I almost
  recommended persist=2; data says cold-start is the real mitigation. Handoff: prefer cold for uncertain qz;
  warm OK for good qz + monitor switch margin (switch_index−steps_to_floor).
- **Report over-claims all corrected** in `laps_validation_report.md` (unbiasedness scoped to ~1e-4/M≤8192;
  b²_max 10.9× not 3-4×; acceptance 0.654–0.738; schedule/C reframed; off-qz nuanced). 

## P5 COMPLETE. Net: faithful, unbiased-to-floor GIGALens LAPS; x_i² self-calibrated switch = decisive lever;
## Phase-2 correction evidenced; paper+C=0.025 default justified by F3; cold-start robust for uncertain qz.
- [→] P6 handoff (user decisions 2026-06-29): (1) validate on a STIFF SYNTHETIC PROXY first, then real lens;
      (2) USER runs the real lens in their own notebook and wants the OPTION to compare cold vs warm — so
      expose both toggleable, do NOT pick an init default; package a notebook-ready handoff.
      - [→] P6a STIFF-PROXY VALIDATION: rotated strongly-correlated ill-conditioned Gaussian (diagonal precond
        structurally can't capture rotation → tests diagonal-vs-dense need) + strong higher-d banana (Phase-1
        appreciably biased → Phase-2 correction + non-√2 switch floor exercised). Tests: Phase-2 on/off bias
        reduction; k stress (does equilibration exceed the window so k matters?); diagonal precond adequacy
        (acceptance, b²) → decide if a DENSE precond option must be added; warm vs cold. Pre-reg P6a below.
      - [x] P6a RESULTS (stiff preset, 48 cells, T_rot d=12 cond1e4 maxcorr0.93 + T_banana_hi d=12; all nan-free):
        **(c) k=1.5 EMPIRICALLY JUSTIFIED** — T_rot cold: k=3.0 PREMATURE (margin 0, b2_max@sw 2.8e-2=14×floor),
        k=1.5 ripe (switch 333, at floor). Warm/easy: window binds, k irrelevant. k=1.0 too strict + k=3 premature
        ⇒ 1.5 is the sweet spot, protective in the cold-stiff (carousel) regime.
        **(b) DIAGONAL PRECOND SUFFICIENT — pre-reg prediction FALSIFIED:** on rotated Σ, diagonal 1/Var hits
        accept 0.70, marginal b² at floor, AND recovers dense Σ off-diagonals to 3.6–7.2% (noise). Mechanism:
        M-ensemble + exact MH ⇒ precond is an EFFICIENCY knob (21× smaller Phase-2 step on rotation), NOT
        correctness. Open item RETIRED on correctness; dense = efficiency option for higher cond/tight budgets.
        **(a) Phase-2 correction evidenced** (T_rot cold k3: Phase-1 1.0e-2 → ON 1.1e-3) BUT marginal b² is BLIND
        to the banana's JOINT-SHAPE bias (Phase-1 already at marginal floor → ON≈OFF). ⇒ add a joint-shape
        diagnostic; Phase-2-on-curvature not yet shown via a joint metric (lensing-relevant: degeneracies).
      - [→] P6b NOTEBOOK HANDOFF: verify `LAPS_late_adjusted_JIT(model_seq,qz)` matches the gigalens prob_model
        interface (vs mclmc.py MCLMC_JIT); usage example + warm/cold compare harness + diagnostics/plots +
        checklist (switch margin, D̃→floor, EEVPD tracking, accept→target, Phase-2-reduces-bias) + a JOINT-SHAPE
        diagnostic (cross-moments/2D) and a quick banana Phase-2 on/off joint-bias check to close (a).

### P6a PRE-REGISTRATION
- **Rotated correlated Gaussian (cond ~1e3, dense rotation):** PRED — diagonal 1/Var precond CANNOT whiten a
  rotated covariance → Phase-2 bisection struggles to hit 0.70 and/or b² above floor; a dense precond would
  fix it. FALSIFIER: if diagonal precond hits acceptance + b² at floor on the rotated target, diagonal-only
  is sufficient and no dense metric is needed. (This decides the diagonal-vs-dense open item.)
- **Strong banana (higher d):** PRED — Phase-2 OFF shows appreciable bias (b²≫floor), Phase-2 ON reduces it
  toward floor = Phase-2 correction demonstrated on a target where it MATTERS (unlike easy Gaussians).
  FALSIFIER: Phase-2 ON ≈ OFF ⇒ either Phase-1 already unbiased here too, or Phase-2 not correcting.
- **k stress:** PRED — on a stiff target equilibration may exceed the window-eligibility, so loose k (3.0)
  fires BEFORE equilibration (residual b²_at_switch ≫ floor) while k=1.5 waits → k finally discriminated.
  FALSIFIER: all k still ripe ⇒ window still binds even here; k value remains non-critical.

### Pre-registration D1-test (moved into the build, since laps.py confirmation was skipped)
The D1 falsifier is now a UNIT TEST in P4a: on a target with a near-zero-mean coordinate, the EMAUS switch
statistic (σ/μ on identity x_i) must blow up (→ never switches) while the paper statistic (δ=σ/μ on x_i²)
stays O(1) and fires at convergence. Predicted: EMAUS r_max ≳ 1e6 for a coord with |mean|≲1e-3·σ; paper
δ ≲ 0.01 at equilibrium. Falsifier: if x_i statistic does NOT blow up for near-zero-mean coords, the
laps.py non-convergence diagnosis is wrong and must be revisited.
- [ ] P4 build
- [ ] P5 CPU validation (internals AND results)
- [ ] P6 lensing handoff

## Pre-registration ledger
(Each entry: cause hypothesis · predicted direction+magnitude · falsifier · structurally-wrong vs
fine-tuning · observed vs predicted.)

_P1 is pure reading/spec extraction — no method-evaluation claims yet, so no pre-registration entry
required until the first run/experiment._

## P1 findings (2026-06-29)

**Both independent readers (paper + blackjax code) converge on the same algorithm.** LAPS =
**"Late-Adjusted Parallel Sampler"** (Robnik & Seljak, *Faster parallel MCMC: Metropolis adjustment
is best served warm*, arXiv:2601.16696, Jan 2026). NOT adaptive path sampling. Mechanism:
- M independent parallel MCLMC chains (one sample/chain), ensemble-chain-adaptation (ECA): shared
  hyperparameters adapted from the ensemble.
- **Phase 1 (unadjusted MCLMC):** `L = α·√(ΣVar xᵢ)`; step size set each iter so the asymptotic
  (discretization) bias is a fixed fraction `C` below the *total* bias — total bias from
  equipartition divergence `D̃` (`bias_type=3` = diagonal equipartition), asymptotic from EEVPD;
  `EEVPD_wanted = C·bias^(3/8)`, `eps_factor = clip((EEVPD_wanted/EEVPD)^(1/6), 0.3, 3)`. Step starts
  large, shrinks. Early-stop when ensemble-mean second moments stop changing (`r_max ≤ r_end`).
- **Phase 2 (MAMS, Metropolis-adjusted MCLMC):** MN2 (mclachlan) if d≤200 else MN4 (omelyan),
  `L_proposal = 1.25·L_full`, `N=15` steps/trajectory; step bisected to target acceptance, then
  hyperparameters frozen → ensemble converges exactly.
- **Outputs samples/expectations ONLY. No evidence / normalizing constant. No annealing, SMC,
  tempering, resampling, or superchains** (named as compatible future work; `superchain_size` in code
  only affects split-R̂ + key-sharing).

**Headline discrepancies (paper vs blackjax code) → adjudication targets:**
- `C`: paper **0.025** vs code **0.1** (4×; load-bearing — controls step-shrink aggressiveness).
- `α`: paper **2** vs code **1.9** (minor).
- preconditioning: code estimates diag mass matrix in Ph1 but APPLIES it only in Ph2 (Ph1 = identity);
  confirm vs paper.
- Consistent across both: target acceptance 0.7 (MN2)/0.9 (MN4), N=15, L_factor 1.25, save_frac/window
  0.2, switch threshold r_end/δ = 0.01, init ε≈0.01√d.

**Code footguns flagged by Reader B (revisit in audit/build):** Ph1 identity-mass quirk above;
`num_adaptation_samples` dead code in Ph2; many bare constants with no in-code derivation
(`alpha`, `C`, `save_frac`, `L_proposal_factor`, 0.3/3 step clip, 0.7/0.9 accept).

**Paper-reading caveats (lossy ghostscript extraction by Reader A — re-verify from rendered PDF):**
`F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²` glyphs garbled; Algorithm-1 `until` condition possibly OCR-inverted;
total run length / `maxiter` never given numerically yet Ph1 window = "20% of total" (forward-ref);
no explicit Ph2 stopping rule beyond maxiter; multimodality explicitly OUT of scope.

## P1 RECONCILIATION — VERIFIED (rendered PDF + code docstrings). Canonical: `laps_spec.md`

**MAJOR FINDING: the blackjax `adaptation/laps.py` does NOT implement the published LAPS Phase-1
schedule.** Three structural paper-vs-code divergences, each verified twice:
1. **Step-size law:** paper (p.4 + Alg.1 p.16) `EEVPD_wanted = F(C·D̃)`, `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²`.
   Code `EEVPD_wanted = C·D̃^{3/8}` — its docstring attributes this to "eq (9) of the EMAUS paper" (a
   different predecessor). 3/8 ∉ [½,3/2] → structurally different, not an approximation.
2. **C:** paper 0.025 (p.4, "we will fix it to C=0.025", ablation-stable Fig.6); code 0.1 (EMAUS value).
3. **Phase-1→2 switch:** paper `δ=σ/μ<0.01` on observable **x_i²**; code `(σ/μ)²<0.01` (⟹ δ<0.1, 10×
   looser) on **x_i**. Paper warns early switching "dramatically slows convergence."
Benign/consistent: α 2(paper)/1.9(code) flat region; preconditioning Phase-2-only in both; accept
0.7/0.9; N=15; L_factor 1.25; init ε=0.01√d; output=samples only (no evidence); multimodality OOS.

### Pre-registration D1 — Phase-1 schedule, C, and switch rule
- **Cause hypothesis (existing-impl failure):** if `laps.py`/`COLDSTART3` were written against the
  blackjax code, they inherited the EMAUS step law (C·D̃^{3/8}, C=0.1) and the 10×-looser switch
  (δ<0.1 on x_i) → premature Phase-1→2 switch → paper-warned "dramatically slowed convergence" =
  the observed "mostly fails to converge." Classification: **STRUCTURAL** (not fine-tuning).
- **Build decision:** implement PAPER-faithful LAPS as primary default (F(C·D̃), C=0.025, switch
  δ=σ/μ<0.01 on x_i², α=2); expose the EMAUS/code variant as a switchable A/B alternative. Rationale:
  the paper is the definition of "LAPS"; its values are ablation-backed; the code's are inherited from
  a different paper without re-justification. NOT finalized unilaterally — A/B decides with evidence.
- **Prediction (Phase 5, falsifiable):** on known-answer (Gaussian / ill-conditioned) beds, (i) the
  code/EMAUS switch rule fires at a point where second-moment equipartition D̃ is still materially
  (~order 10×) from converged → measurable residual b² at switch; (ii) the paper schedule reaches
  b²<0.01 in ≤ the gradient calls of the EMAUS schedule. **Falsifier:** if both schedules switch at
  statistically indistinguishable points AND reach b²<0.01 equally fast, the discrepancy is NOT the
  failure cause (fine-tuning) → look elsewhere (bug in the existing impls, not the schedule).
- **Adjudication items carried to the user check-in:** D1 (this), plus confirm the A/B-then-default
  approach is acceptable.
