# Audit — `laps-COLDSTART3.py` vs canonical LAPS spec

**Target:** `src/gigalens_research/inference/laps-COLDSTART3.py` (325 lines)
**Standard:** `docs/logs/laps_spec.md` (paper arXiv:2601.16696v1; blackjax 1.5 reconciliation)
**Verdict in one line:** This is **not an implementation of LAPS Phase 1 at all** — it is a vanilla blackjax-MCLMC warmup bolted to a single-step Metropolis kernel. The defining LAPS machinery (equipartition bias control, `F(C·D̃)` step law, `δ[x_i²]<0.01` switch, N=15 MAMS trajectories, acceptance bisection) is **absent**. Audited skeptically; treat as incorrect.

---

## 1. Structure map

| Block | Lines | What it is |
|---|---|---|
| Imports / `LAPSAdaptationState` | 1–37 | Carry = `(L, step_size, inverse_mass_matrix)`. Note `inverse_mass_matrix` is a **full 2D matrix**. |
| `_esh_momentum_update_smart` | 43–71 | Isokinetic momentum update with a **full** mass matrix via Cholesky of `inverse_mass_matrix`. |
| `_make_isokinetic_integrator` / `isokinetic_mclachlan_smart` | 74–88 | Wraps **McLachlan (2nd-order, MN2-equivalent)** coefficients. Used in **both** phases. |
| `_unadjusted_kernel` (Phase-1 kernel) | 95–115 | MCLMC step + **hard energy-error rail** (`|ΔE|<1000`, finite) that reverts position. Momentum always kept. |
| `_adjusted_kernel` (Phase-2 "MAMS") | 118–143 | **One** integrator step → accept/reject → partial refresh. **N=1, not N=15.** |
| `_single_init` / `_init_multi` | 150–163 | Per-chain init; momentum = **random unit vector** (not gradient-aligned). |
| `laps_find_hyperparams` (Phase 1) | 170–220 | Fixed-length `lax.scan`; bang-bang step control; α=2 `L`; end-of-phase full-cov preconditioner. |
| `_laps_adjusted_multi` (Phase 2) | 227–239 | Frozen `L`, `step_size`; **no adaptation**, no bisection. |
| `LAPS` wrapper (init scheme + driver) | 246–325 | Cold/warm init, runs Phase 1 then Phase 2, returns `samples` only. |

Return: `samples` of shape `(num_results, n_local, dim)` (line 325). **No evidence / logZ** — correct (LAPS has none).

---

## 2. Conformance table

| Spec element | What the impl does (lines) | Class | Severity |
|---|---|---|---|
| **H1** Phase-1 step law `EEVPD_wanted=F(C·D̃)` | `sq_energy=mean(ΔE²)`; `xi=sq_energy/(d·5e-4)`; `step×1.02 if xi<1 else ×0.97` (194–198). Fixed target `desired_energy_var=5e-4`; bang-bang P-controller. | **NEITHER** (not `F(C·D̃)`, not `C·D̃^{3/8}`; ≈ vanilla blackjax MCLMC fixed-EEVPD warmup) | **High** |
| **H3** Bias ratio `C` (0.025) | No `C` anywhere. | **ABSENT** | High |
| `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²` | Never computed. | **ABSENT** | High |
| Equipartition `D̃`, matrix `V=E[−Δx ∂log p]` (total-bias proxy) | Never computed. The entire bias-ratio principle is missing. | **ABSENT** | **High** |
| **H2** Phase-1→2 switch `δ=σ/μ<0.01` on `x_i²` | No switch test. Phase 1 = fixed `lax.scan(num_burnin_steps)` (203–208). No window `T`, no `δ`, no observable. | **ABSENT** | **High** (prime non-convergence suspect) |
| Step update exponent 1/6 | Replaced by multiplicative ×1.02/×0.97 (197). | NEITHER | High |
| Decoherence `L=α(ΣVar)^{1/2}`, **α=2** | `new_L=2.0·sqrt(sum(ensemble_var))` (191–192). | **PAPER** (α=2 ✓) | — |
| Init `ε=0.01√d` | `eps0=sqrt(dim)*0.25` = **0.25√d** (276), 25× too large. | NEITHER | Med-High |
| Init `L` (Eq.9) | `L0=√d` (275), then Eq.9 in-loop. | Partial | Low |
| **H4** Preconditioner: diagonal `y_i=x_i/Var^{1/2}`, set end-Phase-1, active Phase-2 | Set end-Phase-1 from 2nd-half positions, used Phase-2 (211–218, 309). But it is a **FULL empirical covariance**, shrunk 5% toward `target_covariance` (215–217) — not the diagonal transform; shrinkage not in paper. Timing ✓, form ✗. | NEITHER (form); timing PAPER | Med |
| MAMS: **N=15** integrator steps/trajectory | Single step `step(state, step_size)` (123). **N=1.** | **ABSENT** (N=1) | **High** |
| Partial-refresh `L=1.25·L_full`, `L_full=15ε` | `mix=exp(-ε/L)` with adapted `L` (135–136); no `L_proposal_factor=1.25`, no `L_full=15ε`. | **ABSENT** | Med |
| Phase-2 step tuning: bisection to accept 0.7/0.9, freeze `|a−a*|≤0.03` | None. `update_fn` runs frozen `L,step_size` (235–237); `accept_prob` returned but never used. | **ABSENT** | **High** |
| Target acceptance 0.7 (2nd) / 0.9 (4th) | No target. | ABSENT | High |
| Integrator MN2 default, **MN4 for d>200** | McLachlan (MN2) for **both** phases; no `d>200` switch (88, 310). Phase 1 should be Leapfrog. | Partial / NEITHER | Med |
| No annealing/tempering/SMC/evidence | None added. | **PAPER** (correct) | — |
| Output = samples only | `return samples` (325). | PAPER ✓ | — |
| Velocity init aligned with gradient | Random unit vector (155). | NEITHER | Low |

---

## 3. Concrete bugs (distinct from design divergences)

**B1 — MAMS proposal is a single integrator step (line 123).**
```python
(position, momentum, logdensity, logdensity_grad), kinetic_change = step(state, step_size)
```
`step = integrator(...)` is **one** McLachlan step. The MAMS kernel must take **N=15** steps accumulating energy error, then one accept/reject. As written the proposal moves ~1/15 of a trajectory. *Not formally incorrect* (still valid MCMC, exact target preserved), but efficiency-fatal: per-step displacement is tiny, so mixing is ~15× slower. Author's own TODO (lines 5–7) flags related incompleteness.

**B2 — COLDSTART init ignores the prior (lines 266–270).**
```python
one     = model_seq.prob_model.prior.sample(seed=init_key)
dim_map = len(model_seq.prob_model.bij.inverse(one))
init_positions = jax.random.normal(init_key, shape=(n_local, dim_map))
init_inv_mass  = jnp.eye(dim_map)
```
The prior is sampled **only to read the dimension**; all chains are then seeded from a **standard normal `N(0,I)` in mapped space**, not from the prior. This is the literal "COLDSTART": chains start in an arbitrary Gaussian blob unrelated to the actual prior/posterior geometry. (When `qz` is supplied, lines 263–265 give a proper warm start — so the cold path is the dangerous one.) See §3 warm-start note below.

**B3 — Adapted mass matrix is computed but unused in Phase 1 (lines 181–184 vs 218).** `kernel_v` is jitted at 181 with `init_params.inverse_mass_matrix`; the matrix updated at 218 only reaches Phase 2. This is *consistent with spec timing* (precondition takes effect in Phase 2), but it means the `params._replace(inverse_mass_matrix=…)` value never feeds Phase 1 — easy to misread as a live update. Flag: behavior-correct, readability-trap.

**B4 — `inverse_mass_matrix` is overloaded as a covariance (lines 215–217, 264).** `emp_cov` and `qz.covariance()` are stored directly into the field named `inverse_mass_matrix` and Cholesky-factored in `_esh_momentum_update_smart` (47, 53). Convention "inverse mass ≈ posterior covariance" is defensible, but the naming invites a transpose/inverse error and there is no test pinning the convention.

**B5 — Magic stability rails not in paper.** Phase-1 hard reject `|ΔE|<1000` reverting position (line 106); `step_size_cap=5√d`, floor `1e-3` (178, 198); shrink `0.05` toward target cov (216). All silent, undocumented; the `|ΔE|<1000` rail can **freeze chains** under the 0.25√d oversized init step (B-link to init ε).

**Warm-start premise — broken.** LAPS Phase 2 (Metropolis adjustment) is "best served warm": the unadjusted phase must equilibrate the ensemble (gated by `δ[x_i²]<0.01`) before adjustment turns on. Here Phase 1 (a) starts **cold** (`N(0,I)`, B2), (b) has **no equipartition/bias control** (§2), and (c) has **no `δ` convergence gate** — it stops after a fixed `num_burnin_steps` regardless of warmth. So Phase 2 can begin on an un-equilibrated ensemble. The premise the paper relies on is not enforced anywhere.

---

## 4. Ranked root-cause hypotheses for failure

**RC1 (most likely) — Phase 1 is not LAPS; no warmth guarantee.** No `D̃`, no `F(C·D̃)`, no `δ[x_i²]<0.01` switch (§2, lines 191–208). Phase 1 is a fixed-length vanilla-MCLMC bang-bang warmup targeting a constant `EEVPD=5e-4`. *Falsifiable prediction:* instrument `δ_t[x_i²]` at the end of Phase 1 — for non-trivial posteriors it will exceed 0.01 at `num_burnin_steps=1000`, and final-sample bias `b²` will depend strongly on `num_burnin_steps` (no plateau). If it did converge, the paper's warm-start machinery would be redundant — it is not.

**RC2 — Single-step MAMS (N=1) starves exploration (line 123).** *Prediction:* effective sample size per step ≈ 1/15 of a correct N=15 build; autocorrelation length ~15× longer; chains barely move per accepted step. Setting N=15 should raise ESS ~10–15× with acceptance roughly unchanged. Falsifier: if ESS is already high with N=1, this is not the bottleneck.

**RC3 — Cold `N(0,I)` init + no acceptance tuning compound into non-convergence within budget (B2; lines 235–237).** With chains cold, no Phase-1 bias control, no Phase-2 bisection (step inherited from a fixed-EEVPD target, not tuned to 0.7/0.9), the exact-target Phase 2 is asymptotically correct but will not converge in `num_results` steps. *Prediction:* `qz`-warm runs look fine; prior/cold runs are badly biased and show Phase-2 acceptance far from 0.7. Falsifier: cold and warm runs give equal posteriors.

Secondary: init `ε=0.25√d` (25× spec) tripping the `|ΔE|<1000` rail and freezing chains early (lines 276, 106); full-covariance preconditioner with 5% shrink (lines 215–217) introducing bias if `emp_cov` is rank-deficient from cold, unconverged Phase-1 positions.

---

## 5. Salvageable vs rewrite

**Mostly rewrite.** Salvage: the blackjax integrator plumbing (`_esh_momentum_update_smart`, `_make_isokinetic_integrator`, lines 43–88), the gigalens wrapper/multi-process scaffolding (246–272, 304–325), and the α=2 `L` update (191–192, the one correct LAPS piece). The scientific core must be rebuilt:

- **Phase 1:** add equipartition `D̃` (Eq. 6/18), `EEVPD_wanted=F(C·D̃)` with `C=0.025`, the `ε^{1/6}` update, and the `δ[x_i²]<0.01` windowed switch (replace the fixed `lax.scan`).
- **Phase 2:** make the kernel take **N=15** steps/trajectory with `L=1.25·L_full`, and add acceptance **bisection** to 0.7/0.9 with 3% freeze.
- **Init:** seed Phase 1 from the **prior** (or `qz`), not `N(0,I)`; set `ε=0.01√d`; align velocity with gradient; use a **diagonal** preconditioner.

The author's own header TODOs ("EEVPD formula", "make L not frozen after warmup (bisection)") confirm the file is a known-incomplete stub, not a faithful LAPS. Do not trust its outputs as LAPS.
