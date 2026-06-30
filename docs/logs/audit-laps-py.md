# Audit — `src/gigalens_research/inference/laps.py` vs canonical LAPS spec

**Target:** `/global/u1/l/linusu/GIGALens-Code/src/gigalens_research/inference/laps.py` (exports `LAPS_JIT`, 1380 lines).
**Standard:** `docs/logs/laps_spec.md` (reconciled paper + blackjax reader). Judged against the spec; the impl is treated as untrusted.
**Verdict up front:** the core sampler (`full_laps_sharded`) is a *blackjax/EMAUS-flavored* MCLMC→MAMS pipeline, not the verified LAPS paper. It diverges on all four headline axes (H1–H4) and adds an un-spec'd dense full-rank metric that is applied during Phase 1. The public entry point `LAPS_JIT` is **dead code** (raises `NotImplementedError` at line 976).

---

## 1. Structure map

| Lines | Component | Notes |
|---|---|---|
| 1–27 | Imports | Pulls blackjax integrator internals (`mclachlan_coefficients`, `omelyan_coefficients`, `velocity_verlet_coefficients`, `with_isokinetic_maruyama`, `generalized_two_stage_integrator`). |
| 30–80 | `namedtuple`s | `UnadjustedHist`, `AdjustedHist`, `LAPSCarry` (diagnostic histories + final carry). |
| 92–172 | **Full-rank isokinetic integrator** | `_require_full_rank_mass_matrix`, `_esh_dynamics_momentum_update_one_step_fullrank` (Cholesky of inverse mass matrix, lines 110/123/135), three integrators (LF / McLachlan / Omelyan). **Not in spec** — paper uses a *diagonal* metric. |
| 175–228 | Welford + mass-matrix helpers | `_welford_combine`, `_covariance_to_mass_matrix`, `_stabilize_mass_matrix`, eig summary. All forced 2D (dense). |
| 236–244 | `_initialize_like_laps` | Aligns initial momentum with gradient sign via equipartition diagonal. |
| 247–267 | History/contraction helpers | `_update_history`, **`_contract_history`** (the switch statistic, 255–263), `_equipartition_diagonal_loss`. |
| 283–310 | `_bisection_monotonic_update` | Phase-2 step-size bisection, 3% freeze (line 309). |
| 313–416 | Two Phase-2 proposal kernels | `_adjusted_mclmc_proposal_shardmap` (base-kernel path, 313–350) and `_adjusted_mclmc_proposal_integrator_shardmap` (direct-integrator path with finite checks, 353–416). LAPS_JIT wires the latter. |
| 419–733 | **`full_laps_sharded`** — the actual algorithm | Signature/defaults 419–448; setup 449–476; **Phase-1 step_fn** 510–708 (kernel call 535–544; cross-chain reductions 550–565; `new_L` 568; `observed_eevpd` 569; D̃ 570; switch statistic 572–574; **step law 585**; mass-matrix adaptation 593–643; switch bookkeeping 644–666); Phase-1 carry init 710–730; scan 731. |
| 742–770 | Phase-1→2 handoff | Unpacks carry; `adjusted_step_size_init = unadjusted_step_size`; builds `HMCState` from frozen positions. |
| 787–918 | **Phase-2 `run_adjusted_sharded`** | Proposal dispatch 819–854; cross-chain mean acceptance 856–857; `do_adapt = iteration < num_adjusted_steps` (858); bisection 859–872; history 877–892. |
| 920–939 | Finalize | `samples = last num_results positions` (928); returns `(unadjusted_hist, adjusted_hist, samples), LAPSCarry`. |
| 942–1066 | **`LAPS_JIT`** | **Raises `NotImplementedError` at 976** before any work. Below the raise: builds log_prob, draws from `qz`, sets init ε=0.01√d (1009–1013), integrator switch d>200 (1020–1022), target accept 0.9/0.7 (1023–1024), calls `full_laps_sharded`. |
| 1069–1380 | `plot_laps_diagnostics` | Plotting only. |

**Returns:** samples / expectations only (positions). **No evidence/logZ** — correct (spec §A).

---

## 2. Conformance table

| Spec element | What laps.py does (line) | PAPER / EMAUS-CODE / NEITHER / ABSENT | Severity |
|---|---|---|---|
| **H1** Phase-1 step law `EEVPD_wanted=F(C·D̃)`, `F=4D̃^{3/2}/(1+D̃^{1/2})²` | `target_eevpd = C * max(bias,1e-12)**(3/8)` (585) | **EMAUS-CODE** (`C·D̃^{3/8}`) | **High** |
| **H2** Switch `δ=σ/μ<0.01` on `x_i²` | `_contract_history` computes `r=(avg_sq−sq_avg)/sq_avg = (σ/μ)²` (255–263) on `history_obs=mean_x` (572); `continue = fluctuations[0] > 0.01` (590) | **EMAUS-CODE** ((σ/μ)² **and** observable `x_i`, identity) | **High** |
| **H3** `C=0.025` | `C=jnp.float32(0.1)` (431, 952) | **EMAUS-CODE** (0.1) | **High** |
| **H4** Diagonal precond applied at Phase-1→2 boundary, Phase-2 only; Phase 1 isotropic | Dense metric seeded from SVI covariance used in Phase 1 from step 0 (466–469, 714); re-adapted mid-Phase-1 and fed to the Phase-1 kernel next step (593–628, 535) | **NEITHER** (wrong phase + dense, not diagonal) | **High** |
| α=2 | `alpha=jnp.float32(1.9)` (432, 953); `new_L = alpha*sqrt(mean(var))*sqrt(d)` (568) | **EMAUS-CODE** (1.9) | Low |
| `L_t=α(Σ Var)^{1/2}` | `alpha*sqrt(mean(var))*sqrt(d)` = `alpha*sqrt(Σvar)` (568) | PAPER (algebraically equal) | none |
| Step update `(wanted/current)^{1/6}` | `(target/observed)**(1/6)`, clipped [0.3,3] (586–587) | PAPER (exponent) / code-only clip | Low |
| Phase-1 integrator = Leapfrog | `isokinetic_velocity_verlet_fullrank` (1029) | PAPER | none |
| Phase-2 integrator MN2 / MN4 (d>200) | `omelyan if dim>200 else mclachlan` (1020–1022) | PAPER | none |
| Init ε=0.01√d | `sqrt(dim)*0.01` (1009–1013) | PAPER | none |
| Init L from positions | passed `init_L` **discarded** (`del ... init_L`, 449); `running_L` starts `jnp.inf` (713) | **NEITHER** | Low–Med |
| N=15 steps/traj | `adj_n_steps=15` (436); scan over 15 (327) | PAPER | none |
| `L=1.25·L_full` | `partial_refresh_L = 1.25*(N*ε)` (326) | PAPER | none |
| Target accept 0.7/0.9 | `0.9 if dim>200 else 0.7` (1023–1024) | PAPER | none |
| Bisection freeze 3% | `tolerance=0.03`; `|a−a*|<tol` freezes (309) | PAPER | none |
| Full-rank D̃ via Hutchinson | `equi_full_loss = equi_diag_loss` (575) — Hutchinson never implemented | ABSENT (benign; spec says diag≈full) | Low |
| V centering `E[−(x−Ex)∂log p]` | `mean_equi_diag = mean(−x·∂log p)` uncentered (554, 560) | code-style; **equal at stationarity** (∫x∂p = −δ) | Low |
| No annealing/tempering/evidence | returns positions only (928, 1065–1066) | PAPER | none |
| Phase-2 metric re-adapt | `imm` carried unchanged (909, 899) — fixed | PAPER (App. D) | none |

---

## 3. Concrete bugs / defects (distinct from spec divergences)

1. **`LAPS_JIT` is unreachable.** Line 976 `raise NotImplementedError(...)` fires before any code. The exported entry point cannot run; whatever the user exercises is `full_laps_sharded` directly or a restored older revision. Flag first — it changes what "fails to converge" even refers to.

2. **Switch statistic is ill-conditioned for near-zero-mean coordinates (lines 255–263, 572, 590).** `history_obs` stores the ensemble mean `mean_x` (i.e. observable `f=x_i`, identity). `_contract_history` divides by `max(square_average, 1e-12)` where `square_average = (time-avg of mean_x)²`. For any coordinate whose posterior mean ≈ 0 (shear, centroid offsets, etc.), `square_average → 1e-12` while `average_square ~ O(Var)`, so `r → ~1e12`. `fluctuations[0]=max_i r` is then dominated by that coordinate, `continue_unadjusted` stays `True`, and **Phase 1 never switches** — it exhausts `num_unadjusted_steps`. The paper's observable `x_i²` is strictly positive and avoids exactly this. This is the prime suspect (see §4).

3. **First Phase-1 step runs with `L=inf` (713).** `running_L` is initialized to `jnp.inf` and only updated *after* the kernel call (535/568). On step 0 the partial-refresh coefficient `c₁=e^{−ε/L}=1`, `c₂=0` → no momentum decoherence on the first step. Minor (1 step), but combined with the discarded `init_L` (449) it means the spec'd L-initialization is simply absent.

4. **Dense SVI metric drives the isokinetic integrator (92–146).** The momentum update Cholesky-factors `inverse_mass_matrix` (110) and applies `chol.T @ grad` / `chol @ momentum` (123/135). If the SVI covariance is ill-conditioned (common for a VI fit to a lensing posterior), the Cholesky amplifies directions and the isokinetic step can blow up. `_stabilize_mass_matrix` only adds `~1e-6·trace/d` jitter (215) — insufficient for a badly-scaled metric. This is a structural change to the dynamics with no spec basis.

5. **Mass-matrix adaptation couples to the same broken statistic (594–597).** `mass_matrix_triggered` fires on `fluctuations[0] ≤ 0.05` OR `fluctuations[1] ≤ 0.05`. Because of bug #2 those fluctuations rarely drop, so the empirical metric re-estimation typically never triggers — Phase 1 just rides the fixed SVI metric. So the *intended* H4 behavior usually does not even occur; the realized behavior is "fixed dense SVI precond throughout Phase 1," which is also not the spec.

6. **`equi_full_loss = equi_diag_loss` (575).** `bias_type==2` ("full-rank") is silently identical to diagonal; the advertised Hutchinson path does not exist. Benign per spec, but dead branch.

7. **Wasted compute after switch (correct but inefficient).** `switched` is a single global scalar (644); once true, `active=False` freezes all chains (651) yet the scan still runs every remaining Phase-1 step as a no-op. Correctness-neutral; only relevant because bug #2 usually prevents switching at all.

8. **No NaN/early-stop in Phase 1.** Phase-1 kernel results aren't finite-checked (only Phase 2 is, 270–280, 374–377). A single non-finite gradient pollutes `mean_x`/`mean_equi_diag` via `psum` and corrupts the global step size and metric for all chains.

---

## 4. Ranked root-cause hypotheses for "mostly fails to converge"

**RC1 — Broken Phase-1→2 switch (observable `x_i` + `(σ/μ)²`, ill-conditioned at μ≈0).** Lines 255–263, 572, 590.
*Cause:* identity observable with division by `mean²` blows up for zero-mean coordinates, so `r_max` never falls below 0.01.
*Prediction if true:* `unadjusted_hist.active` stays 1 for the entire Phase 1; `carry.switch_index == num_unadjusted_steps`; `delta_x2` (r_max) trace is pinned at huge values dominated by one/few coords; Phase 2 starts from an under-equilibrated state regardless of `num_unadjusted_steps`. *Falsifier:* switching the observable to `x_i²` (and threshold to true δ<0.01) makes `switch_index < num_unadjusted_steps` and convergence appears. This is the single highest-value experiment.

**RC2 — Dense SVI metric applied in Phase 1 destabilizes the isokinetic dynamics (H4).** Lines 92–146, 466–469, 535.
*Cause:* Phase 1 is supposed to be isotropic; instead it runs a Cholesky-factored dense VI covariance from step 0, weakly stabilized. Ill-conditioned metric → exploding/NaN substeps → corrupted global reductions (bug #8).
*Prediction if true:* `mass_matrix_*_eig` spans many orders of magnitude; Phase-1 `EEVPD`/`step_size` traces show spikes or hit the `[0.3,3]` clip repeatedly; Phase-2 `invalid_substeps`/`invalid_trajectory` are high; acceptance can't reach target. *Falsifier:* forcing Phase 1 to identity metric (and applying a *diagonal* `1/Var` only at the boundary) stabilizes EEVPD and raises Phase-2 acceptance.

**RC3 — Wrong step-size law `C·D̃^{3/8}` with `C=0.1` (H1+H3).** Line 585, 431.
*Cause:* `D̃^{3/8} ≫ D̃^{3/2}` as D̃→0, so `EEVPD_wanted` (and hence ε) stays too *large* near convergence; `C=0.1` (4× paper) compounds it. Phase 1 hands Phase 2 an over-large step / over-biased state.
*Prediction if true:* Phase-1 `step_size` plateaus rather than decaying smoothly (spec Fig. 1); observed EEVPD sits well above the paper's `F(C·D̃)` curve; D̃ stalls at a floor instead of →0. *Falsifier:* replacing line 585 with `F(C·D̃)` and `C=0.025` makes D̃ and step size decay monotonically and lowers second-moment bias.

(Secondary contributors: discarded `init_L` + first-step `L=inf` (RC-minor); no Phase-1 finite checks (amplifies RC2).)

Ordering rationale: RC1 is a hard logic defect that can nullify the entire purpose of Phase 1 and is independent of tuning; RC2 changes the dynamics globally and produces NaNs; RC3 is a quantitatively-wrong-but-still-running schedule. RC1 and RC2 most plausibly produce "mostly fails to converge" while occasionally "sort of samples."

---

## 5. Salvageable vs rewrite

**Salvageable with targeted patches**, provided the dense-metric ambition is dropped. The integrator wiring, MAMS Phase-2 kernel, bisection (3% freeze), ECA cross-chain reductions, sharding, and diagnostics are sound and match the spec. Four scoped changes recover spec conformance:

1. **Switch (RC1/H2):** make the history observable `x_i²` (store `mean_xsq` or `mean(x²)` per coord, not `mean_x`) and test `δ=σ/μ<0.01` (sqrt the current `r`), not `(σ/μ)²`. Lines 572 + 255–263 + 590.
2. **Step law (H1/H3):** replace line 585 with `EEVPD_wanted = F(C·D̃)`, `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²`, `C=0.025`. Keep the EMAUS form behind a flag for A/B.
3. **Metric (H4):** Phase 1 isotropic (identity); estimate a **diagonal** `Var_ρ` and apply `y_i=x_i/Var^{1/2}` **only at the Phase-1→2 boundary**. Remove the dense Cholesky path (92–146) or gate it off by default.
4. **Init:** stop discarding `init_L` (449); seed `running_L` from Eq. 9 on initial positions instead of `jnp.inf` (713).

A full rewrite is **not** warranted — the structural skeleton is correct; the defects are localized to the adaptation math (one line for the step law), one helper (the switch statistic), and the metric phase/shape. The only thing that *must* be rebuilt regardless is the dead `LAPS_JIT` wrapper (976), which needs porting to the scene API before any of this is reachable.

> Method note: I did not run the code (LAPS_JIT raises; `full_laps_sharded` needs a GIGALens model). All findings are static. RC1/RC2/RC3 are stated as falsifiable predictions over the existing `debug_output=True` diagnostics so they can be graded against plots, not asserted as confirmed.
