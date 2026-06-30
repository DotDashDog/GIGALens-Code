# Independent grader audit — LAPS late-adjusted driver

**Auditor role:** independent GRADER (did not author the code). Skeptic stance.
**Date:** 2026-06-29.
**Standard:** `docs/logs/laps_spec.md` (canonical reconciled LAPS spec).
**Artifacts audited:**
`src/gigalens_research/inference/laps_late_adjusted.py` (driver),
`…/laps_core.py` (reductions), composing `…/blackjax_updated_utils.py` kernels.
**In-container checks:** equipartition scaling + Phase-2 freeze latching executed on
CPU/x64 (results inline below). Line numbers refer to the files as read.

---

## Conformance table

| # | Checklist item | Verdict | Key line refs |
|---|---|---|---|
| 1 | Equipartition D̃ (centered, sign, normalization) | **CONFORMS** | core 82–86; driver 280–281 |
| 2 | Phase-1 step law F(C·D̃), C=0.025, ^(1/6), clip, EMAUS flag | **CONFORMS** | core 92–151; driver 239, 286–287 |
| 3 | Ensemble EEVPD = Var[Δ]/d (cross-ensemble) | **CONFORMS** | core 157–180; driver 276, 282 |
| 4 | L = α√(ΣVar), α=2, per-step cadence | **CONFORMS** | core 186–200; driver 283–284 |
| 5 | Switch detector (x_i², window-min guard, threshold) | **CONFORMS** | core 206–274; driver 226–227, 318–331 |
| 6 | Phase boundary (precond 1/Var 2nd-half, integrator/kernel swap) | **CONFORMS** | driver 347–366 |
| 7 | Phase 2 (N=15, bisection 0.7/0.9, 3% freeze, 1 sample/chain) | **DEVIATES** (freeze not latched) | core 285–325; driver 357–419 |
| 8 | Sharding reproduces single-device math | **CONFORMS** | driver 255–293, 378–395 |
| 9 | Control flow (Python outer loop, fixed scans, host switch) | **CONFORMS** | driver 296–333, 403–412 |
| 10 | Diagnostics completeness | **CONFORMS** (minor sparsity) | driver 103–133, 422–433 |
| 11 | Init (warm/cold, dim, dtype) | **CONFORMS** (velocity-init deviation documented) | driver 199–218 |

No **CORRECTNESS-BREAKING** defect found. One FAITHFULNESS-GAP (item 7 freeze)
that touches the ECA-bias-removal validity claim.

---

## Executed evidence

**Equipartition scaling (target N(0,I), grad = −x, ensemble N(0,σ²I)):**
```
σ²=1.0:  mean E_ii=1.001  D̃=0.0000
σ²=2.0:  mean E_ii=1.996  D̃=0.9919
σ²=0.5:  mean E_ii=0.500  D̃=0.2503
F(0)=0   F(0.025)=0.01179   F(1)=1.0
```
→ V_ii→1 ⇔ ρ=p (D̃→0); over-dispersion gives V_ii>1, D̃>0; under-dispersion V_ii<1.
The smoke-test `E_ii≈2` corresponds to a **2× variance** (not 2× std) over-dispersed
ensemble — exactly the spec's intended scaling. F monotone increasing, F(0)=0.

**Phase-2 freeze latching (target 0.7, tol 0.03):**
```
accept=0.71 -> frozen=True
accept=0.69 -> frozen=True
accept=0.55 -> frozen=False     ← UN-FREEZES
accept=0.95 -> frozen=False
```
→ `frozen` is recomputed solely from the current step's acceptance; it is **not** a
one-way latch. See Issue 1.

---

## Issues, ranked by severity

### FAITHFULNESS-GAP

**Issue 1 — Phase-2 freeze is not latched (ECA-bias-removal not implemented).**
`laps_core.bisection_step` (core 323) computes `frozen = |accept − target| ≤ tol`
from the *current* step only and does not accept/AND an incoming flag. The driver
threads a `frozen` carry (driver 400, 392, 406) but `bisection_step` ignores it
(driver 390–391 does not pass it), so the carry is overwritten every step.
Spec Algorithm 1 / B-table / §5 require the **latched** update
`ADAPT ← ADAPT ∧ |a−a_targeted| > 0.03` — once within tol, adaptation is off
*forever*, which is what "freezing removes ECA bias" means: ε becomes a hard
constant, decoupled from the ongoing ensemble. As executed above, the code
un-freezes when acceptance later drifts outside tol; even when the bracket has
collapsed, `ε_next = ½(lo+hi)` with lo/hi still updated from the ensemble's
acceptance each step (core 315–322), so ε remains a (weak) function of the
ensemble — the ECA coupling the freeze was meant to kill persists, and the final
ensemble (the only collected sample, driver 418) may be taken mid-adaptation.
*Not a hard correctness break*: the MH adjustment still targets p at any fixed ε;
the defect is residual ECA coupling bias, exactly the bias the spec's freeze
removes. **Fix:** make freeze sticky. Pass the incoming flag and latch:
`adapt_next = adapt_in & (|a−target| > tol)`; once `~adapt`, hold ε at the frozen
value and stop updating lo/hi. Surface `p2_frozen[-1]` as a hard gate before
trusting Phase-2 expectations.

**Issue 2 — Phase-1 velocity init is random, not gradient-aligned.**
`_single_init` (blackjax_updated_utils 477–481) draws a random unit momentum; spec
Init (Alg. 1 p.16) aligns the initial velocity with ∇log p. Documented in the
driver header (lines 69–71). Phase 1 only "approaches the target", so impact is on
early-transient speed, not the stationary result. **Fix (optional):** seed
momentum = normalize(∇log p) for exact-spec parity; otherwise keep and note.

### EFFICIENCY

**Issue 3 — MAMS partial refresh (L_proposal=1.25·L_full) not applied.**
`_build_adjusted_kernel_shardmap` (blackjax_updated_utils 151–248) is the
*Hamiltonian* MAMS variant: **full** momentum refresh once per trajectory, then
`N` deterministic isokinetic steps with **no** intra-trajectory Maruyama refresh
(≡ L_proposal_factor=∞). Driver records `p2_L_proposal=1.25·L_full` (driver 432)
but it does not influence sampling.
**Correctness reasoning (requested):** This is **not** a correctness break and does
**not** invalidate the Metropolis adjustment or detailed balance. MH validity here
requires only (a) momentum drawn from the isokinetic invariant (uniform on the
sphere — done, line 187), (b) a reversible, volume-preserving deterministic
trajectory (the isokinetic mclachlan/omelyan integrators are reversible), and
(c) accept `min(1,e^Δ)` with `Δ = logp_new − logp_old − Σ kinetic_change`
(lines 213, 221), the standard isokinetic MH ratio (matches
`blackjax.adjusted_mclmc_proposal`). The partial refresh is a *proposal-shaping*
device that improves decorrelation per trajectory; replacing it with full refresh
yields a different but still-exact HMC-style proposal. Cost is efficiency
(per-trajectory mixing) and a possibly slightly-shifted optimal acceptance vs the
70/90% tuned for the partial-refresh variant — not bias. Acceptable for Phase-5,
provided efficiency is not over-claimed.

**Issue 4 — Switch / freeze granularity = `chunk_size` steps.**
The active Phase-1 switch is evaluated only at chunk boundaries and only on the
chunk's last step (driver 321–331); Phase-2 has no early exit. Documented
(driver 43–58). Post-hoc `switch_index_paper/emaus` (driver 344–345, via
`_switch_index_host`, 140–155) recover the per-step switch step exactly, so this
is observability-preserving. Efficiency only.

### COSMETIC / ROBUSTNESS

**Issue 5 — Truncation & empty-Phase-1 edge.** `n_chunks = num_unadjusted_steps //
chunk_size` (driver 302) silently drops a non-divisible remainder; if
`num_unadjusted_steps < chunk_size`, `n_chunks=0` ⇒ `phase1_len=0` ⇒
`np.concatenate([])` / `half=0` crash (driver 334–351). Guard or assert divisibility.

**Issue 6 — `p1_delta_max` is sparse.** Non-NaN only at chunk-boundary steps
(driver 328); the per-dim δ vector is not exposed, only the windowed max. Adequate
for the max-based test but thin for diagnosing which coordinate gates the switch.

**Issue 7 — EEVPD includes NaN-zeroed energy changes.** The unadjusted kernel
zeroes `energy_change` on NaN/Inf reject (blackjax_updated_utils 91); those zeros
enter the cross-ensemble Var[Δ] (driver 276, 282). Faithfully reproduces the
single-device `laps_core.ensemble_eevpd`, so not a sharding error; flagged as a
scientific caveat if reject rates are non-trivial.

**Issue 8 — Full-rank Hutchinson D̃ not implemented.** Diagonal estimator only
(core 51–86); spec App. D states diagonal ≈ full-rank. Documented (driver 72–73).

---

## Per-item notes (spot-checks)

- **Sharding (item 8):** the inline `psum('device')/n_total` reductions yield exact
  ensemble means; `E_ii = −E[xg] + E[x]E[x g]`… i.e. `−s_xg + s_x·s_g`
  (driver 280) is the algebraic identity for the *centered* `E[−(x−x̄)g]`, matching
  `equipartition_diagonal`. EEVPD `(s_dd − s_d²)/d` = population Var/d (ddof=0),
  matches core. L uses `max(s_xx−s_x², 0)` population var, matches core. Acceptance
  `psum(Σ p_accept)/n_total` = expected ensemble acceptance (lower-variance than
  realized-accept mean) — appropriate for bisection. All scalars carried as `P()`
  (replicated, consistent with psum outputs); states `P("device")`; keys
  `P(None,"device")`. No while_loop under shard_map; the adjusted kernel's
  `fori_loop` trip count `N` is static. Dtype `_canon` applied consistently. **No
  P('device') carry-spec or dtype landmine found.**
- **Precond (item 6):** `precond_var = mean_2ndhalf(E[x²]) − mean_2ndhalf(E[x])²`
  (driver 349–351) is exactly the pooled variance over chains × second-half steps;
  `inverse_mass_matrix = diag(Var)` is the standard `M⁻¹ = covariance`
  preconditioning (equiv. `y_i = x_i/√Var_i`). Correct.
- **Step law default (item 2):** default `schedule="paper"`, `C=None→0.025`,
  `F(C·D̃)` — follows the *verified* LAPS law (not the blackjax EMAUS C·D̃^{3/8}),
  resolving spec discrepancy rows 1–2 the right way. EMAUS available behind a flag.
- **Switch (items 4–5):** default paper observable x_i², δ=σ/μ (not squared),
  threshold 0.01 — resolves spec rows 3–4. Window-minimum guard **present**
  (`steps_done ≥ T_window`, driver 321; `_switch_index_host` n<T and t≥T−1 guards).

---

## Overall verdict

**Ready for Phase-5 validation? — WITH FIXES.**

The driver is a faithful, sharding-correct port: equipartition, EEVPD, L, the
verified F(C·D̃) step law (C=0.025), the x_i² δ-switch with window guard, the
phase-boundary preconditioner, and the MH kernel are all correct, and the
cross-device reductions exactly reproduce the single-device `laps_core` math. No
correctness-breaking bug. **Before trusting Phase-2 posterior expectations, fix
Issue 1 (latch the freeze)** — it is the one gap that touches the spec's
bias-removal guarantee; until then, gate results on `p2_frozen[-1] == True` and a
collapsed bracket. The MAMS partial-refresh omission (Issue 3) is an efficiency
choice, not a correctness defect, and is acceptable if not over-claimed. Remaining
items are cosmetic/robustness.
