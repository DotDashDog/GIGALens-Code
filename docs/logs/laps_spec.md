# LAPS — canonical reconciled specification

**Algorithm:** Late-Adjusted Parallel Sampler (LAPS).
**Paper:** Jakob Robnik & Uroš Seljak, *"Faster parallel MCMC: Metropolis adjustment is best served warm"*, arXiv:2601.16696v1 [stat.CO], 23 Jan 2026. PDF: `papers/LAPSRobnik2026.pdf` (21 pp).
**Reconciles:** `docs/logs/laps-spec-paper.md` (Reader A, paper) + `docs/logs/laps-spec-blackjax.md` (Reader B, blackjax 1.5 code).

**Verification method.** Load-bearing constants/formulas were read from the **rendered** PDF pages (ghostscript → PNG → visual read), not from lossy text extraction. Page numbers below are physical PDF pages and were confirmed visually. The earlier paper-spec was extracted with `gs -sDEVICE=txtwrite`; where its reconstruction is now confirmed or corrected, that is noted.

> **Headline finding.** The blackjax `laps` implementation does **not** implement the paper's verified Phase‑1 step-size law. The paper sets `EEVPD_wanted = F(C·D̃)` with `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²` and `C=0.025`; the code computes `EEVPD_wanted = C·D̃^{3/8}` with `C=0.1`, citing "eq (9) of the EMAUS paper" (a *different*, predecessor paper). This is a structural difference in the adaptation, not a tuning constant. See §C rows 1–2.

---

## A. Algorithm (verified)

LAPS evolves `M` independent chains in parallel with **MCLMC** (microcanonical Langevin) dynamics, sharing hyperparameters across chains and updating them each step from cross-chain expectation values (**Ensemble Chain Adaptation**, ECA; Gilks 1994, Fig. 5 p.15). Two phases.

**Output:** posterior **samples / expectations only** — the final ensemble of chain locations `{x^m}`, used as `E_p[f] ≈ (1/M) Σ_m f(x^m)` (Eq. 3). **No evidence / log Z / normalizing constant.** Verified p.15 (Algorithm 1 "Output: samples {x^m}"; Fig. 5 "Output: Samples: final locations of chains"). No annealing / tempering / SMC / resampling / superchains anywhere in the method (those are listed only as *future work* compatible with LAPS, §7). Multimodality is **explicitly out of scope**.

### Dynamics (both phases)
Ideal continuous-time SDE (Eq. 2, p.3):
```
dx = u ds
du = (I − u uᵀ)( ∇log p(x)/(d−1) ds + η dW ),   η = √(2/(L d))  (small-ε limit)
```
Velocity stays on the unit sphere (microcanonical). Discretized via a splitting integrator (App. E): stochastic partial-refresh `Φ^O` (Eq. 23, `c₁=e^{−ε/L}`, `c₂=√(1−c₁²)`, `Z~N(0,I/√d)`), position `Φ^A` (Eq. 24, `Δ=−log p(x+εu)+log p(x)`), velocity `Φ^B` (closed-form rotation keeping ‖u‖=1).

### Phase 1 — Unadjusted initialization (§3, pp.3–5)
Purpose: **not** to converge, only to "approach the target as fast as possible" (p.3).

**Governing principle (§3.1, App. A Thm 1, verified p.3):** keep asymptotic (discretization) bias a fixed factor `C` below total bias:
```
D(p, p_{ε_t}) = C · D(p, ρ_t),   0 < C < 1.
```

**Total-bias proxy — equipartition (verified p.4).** Matrix `V_ij(p,ρ_t)=E_ρt[−(x_i−E_ρt[x_i]) ∂_j log p(x)]` (Eq. 4); `V=I` ⟺ `ρ_t=p` (Eq. 5). Scalar divergence
```
D̃(p,ρ_t) = (1/d) Tr[ (I−V)(I−V)ᵀ ]                     (Eq. 6)
```
Default diagonal form `D̃_diag = (1/d) Σ_i (1−V_ii)²` (Eq. 18). Full-rank via Hutchinson (100 Rademacher z, App. B.1) ≈ diagonal (App. D Table 4).

**Asymptotic-bias proxy — EEVPD (verified p.4).** `EEVPD(ρ|p,ε)=Var_ρ[Δ(x,u|p,ε)]/d` (Eq. 7). Upper-bounds asymptotic bias (Eq. 8, verified p.4):
```
D̃(p,p_ε) ≤ F⁻¹( EEVPD(p_ε|p,ε) ),   F(D̃) = 4 D̃^{3/2} / (1 + D̃^{1/2})²
```
`F` monotonically increasing (verified, exact exponents 3/2 and 1/2). Empirically `EEVPD ∝ ε⁶` (p.4).

**Step-size update (verified p.4 list + Algorithm 1 p.16):**
1. `D̃(p,ρ_t)` from Eq. 6.
2. Desired asymptotic bias `D̃(p,p_{ε_t}) = C·D̃(p,ρ_t)`.
3. **`EEVPD_wanted = F( C·D̃(p,ρ_t) )`**  (Algorithm 1 line: `EEVPD_wanted ← F(C·D̃)`).
4. Estimate current `EEVPD(ρ_t|p,ε_t)` from ensemble variance of `Δ`.
5. `ε_{t+1} = ε_t · ( EEVPD_wanted / EEVPD(ρ_t|p,ε_t) )^{1/6}`.

**Trajectory/decoherence length (verified p.5, Eq. 9):** `L_t = α (Σ_{i=1}^d Var_ρt[x_i])^{1/2}`, **α = 2** (p.5: "here we find that a larger value α=2 works better"; α=1 is the MCLMC default).

**Init (verified Algorithm 1, p.16):** `ε ← 0.01√d`; `L ←` Eq. 9 from initial positions; user-provided initial positions (experiments draw from prior); velocity aligned with gradient, unit norm. Integrator: **Leapfrog** (1 grad/step).

**Switch rule Phase 1 → Phase 2 (§4, verified p.5 + Algorithm 1 p.16).** Relative fluctuation of a windowed summary statistic:
```
δ_t[f] = σ_t[f] / μ_t[f],   μ_t[f]=(1/T)Σ_{s=t−T}^{t} E_ρs[f],   σ_t²=(1/(T−1))Σ(E_ρs[f]−μ_t)²   (Eqs. 10–11)
```
Observable `f(x)=x_i²`. **Terminate Phase 1 when all `δ_t[x_i²] ≤ 0.01`** (or `t ≥ maxiter`); moving window `T = 20% of total sampling time` (verified p.5). At stationarity `δ ~ M^{−1/2}`.
*Algorithm-1 box (p.16) literally prints `until max_i δ[x_i²] > 0.01 and t < maxiter`.* This is **not** an OCR artifact — the typeset text reads exactly that. It is a **loop-continuation predicate** (run while fluctuations large AND below maxiter), consistent with the prose ("terminate when all δ<0.01") and with App. C (p.15) which calls it "the first **while** loop." The keyword "repeat/until" is the typo; the operational meaning (stop when all δ≤0.01 or t≥maxiter) is unambiguous and matches the code's continue-condition.

**End-of-Phase-1 preconditioning (verified p.6).** Diagonal mass-matrix preconditioning by coordinate transform `y_i = x_i / Var_ρt[x_i]^{1/2}`, applied **at the end of the unadjusted phase** (i.e. it takes effect in Phase 2; Phase 1 dynamics are isotropic). Confirmed both paper (p.6) and code.

### Phase 2 — Adjusted sampling (§5, pp.5–6)
Turn on Metropolis adjustment with the **MAMS** kernel (App. F), still ECA-adapting `ε`, then freeze.

**MAMS kernel:** resample `u` on unit sphere; `N` integrator steps (step `ε`) with partial refresh each step; accumulate energy error `Δ`; accept `min(1, e^{−Δ})`. Stationary distribution = exact target.

**Step-size tuning (verified p.6):** tune `ε` to a target average acceptance rate by **bisection** on `a(ε)−a_targeted` (no dual averaging — low noise at large M); double/halve to bracket, then bisect; **freeze when `|a−a_targeted| ≤ 0.03`** (Algorithm 1: `ADAPT ← ADAPT ∧ |a−a_targeted| > 0.03`). Freezing removes ECA bias.

**Target acceptance (verified p.6):** **70% (2nd-order)**, **90% (4th-order)**. Optimal ranges 60–80% (2nd) / 80–87% (4th); slightly-high chosen for numerical stability.

**Trajectory length (verified p.6):** not adapted; **N = 15** steps/trajectory, `L_full = 15ε`, partial-refresh `L = 1.25·L_full` (Riou-Durand & Vogrinc 2023).

**Integrator (verified p.6 §5.1 + App. E p.16):** **MN2** (2nd-order minimal-norm, 2 grad/step) default; **MN4** (4th-order, 5 grad/step) **for d > 200** ("We adopt it for d > 200", App. E). [Code uses McLachlan coefficients for the 2nd-order role and Omelyan for the 4th — same orders/grad-counts.]

**Overall stop:** Phase 2 `while t < maxiter`. `maxiter` numeric value **unspecified in paper**.

---

## B. Hyperparameter table

| Name | Symbol | Controls (mechanism) | Paper value | Code default | **VERIFIED value + page** | Stated rationale |
|---|---|---|---|---|---|---|
| Bias ratio | `C` | Asymptotic/total-bias ratio in step schedule; feeds `EEVPD_wanted` | 0.025 | **0.1** | **0.025 — p.4** ("we will fix it to C = 0.025"); single fixed value, ablation-stable over wide range (Fig. 6 p.16) | Optimal C not known/problem-dependent; value empirical |
| Step-size target law | — | `EEVPD_wanted` from bias | `EEVPD_wanted=F(C·D̃)`, `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²` | **`C·D̃^{3/8}`** | **`F(C·D̃)`, F as above — p.4 (Eq. 8 + step list) & Algorithm 1 p.16** | Eq. 8 bound; ε⁶ scaling for the 1/6 update |
| Step update exponent | — | `ε_{t+1}=ε_t(wanted/current)^{1/6}` | 1/6 | 1/6 | **1/6 — p.4 & Algorithm 1 p.16** | `EEVPD ∝ ε⁶` |
| Decoherence coeff | `α` | `L_t=α(Σ Var[x_i])^{1/2}` | 2 | **1.9** | **2 — p.5** ("α=2 works better") | α=1 is MCLMC default; α=2 suits more deterministic dynamics |
| Init step size | `ε₀` | Phase-1 starting ε | 0.01√d | 0.01√d | **0.01√d — Algorithm 1 p.16** | none (feedback loop corrects) |
| Switch threshold | `δ` / `r_end` | Phase-1 stop on fluctuation | `δ=σ/μ < 0.01` | `r=(σ/μ)² < 0.01` | **δ=σ/μ<0.01 — p.5; Algorithm 1 p.16** | `δ~M^{−1/2}` at stationarity |
| Switch observable | `f` | statistic monitored for the switch | `x_i²` | `x_i` (default `observables_for_bias=λx:x`) | **`f(x)=x_i²` — p.5** | none given |
| Window | `T` | fluctuation window | 20% of total sampling time | `save_frac=0.2`·num_steps1 | **20% — p.5** | none |
| Target accept (2nd) | `a_targeted` | bisection target | 70% | 0.7 (auto, d≤200) | **70% — p.6** | optimal-acceptance theory; +stability |
| Target accept (4th) | `a_targeted` | bisection target | 90% | 0.9 (auto, d>200) | **90% — p.6** | same |
| Bisection tol | — | freeze band | 3% | 0.03 | **3% — p.6** ("up to some tolerance (3% by default)") | none |
| Steps/trajectory | `N` | adjusted-phase integration steps | 15 | `steps_per_sample=15` | **15 — p.6** | phase short, L not adapted |
| Partial-refresh L | — | `L=1.25·L_full` | 1.25·L_full | `L_proposal_factor=1.25` | **1.25 — p.6** | Riou-Durand & Vogrinc 2023 |
| Integrator switch | — | MN4 if high-dim | d>200 | `ndims>200` | **d>200 — App. E p.16** | MN4 4th-order crucial high-dim |
| Hutchinson z | — | full-rank D̃ estimate | 100 Rademacher | (used if `equi_full`) | **100 — App. B.1** | sub-percent error |
| `M` chains | `M` | ensemble size = #samples | 4096 / 256 | — (user) | p.* | accuracy ∝ ensemble var; flat for M>128–256 |

---

## C. Discrepancy table

| # | Quantity | Paper | Code | Verified | Trust & why | **LOAD-BEARING?** | **NEEDS ADJUDICATION?** |
|---|---|---|---|---|---|---|---|
| 1 | `C` | 0.025 | 0.1 | **0.025 (p.4)** | Paper — explicit "we will fix it to C=0.025", default dashed at 0.025 in Fig. 6 (p.16). Code's 0.1 is from the cited EMAUS paper, not LAPS. | **YES** (4× difference) | **YES** |
| 2 | `EEVPD_wanted` functional form | `F(C·D̃)`, `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²` (→ target ∝ D̃^{3/2} small-D̃, ∝D̃^{1/2} large-D̃) | `C·D̃^{3/8}` (target ∝ D̃^{3/8}) | **`F(C·D̃)` (p.4 + Alg.1 p.16)** | Paper — verified in two places (step list p.4, Algorithm 1 box p.16). Code uses a *different paper's* rule (docstring: "eq (9) of EMAUS paper"). Exponent 3/8 lies outside F's range [1/2,3/2]; not an approximation of F. | **YES** (structurally different adaptation) | **YES** |
| 3 | Switch threshold semantics | `δ=σ/μ < 0.01` | `r=(σ/μ)² < 0.01` ⟹ effective `δ<0.1` | **δ<0.01 (p.5)** | Paper. Code's `r=Var/mean²=δ²`, so `r_end=0.01` is **10× looser in δ** → switches ~earlier. Paper warns early switching "dramatically slows convergence" (Fig. 8). | **YES** (switch timing) | **YES** |
| 4 | Switch observable | `f(x)=x_i²` | default `x_i` (identity) | **x_i² (p.5)** | Paper. For mean-zero coords `Var/mean²` on `x_i` is ill-conditioned; caller must pass `observables=square(...)` to match. | YES (couples to #3) | **YES** (with #3) |
| 5 | `α` | 2 | 1.9 | **2 (p.5)** | Paper; 1.9 is ~5% off, inside the flat region of Fig. 6. Likely benign rounding. | no | no (trust paper=2) |
| 6 | `until` line | prose: stop when all δ<0.01 | `while r_max>r_end` continue | **Alg.1 prints `until max δ>0.01 and t<maxiter` (p.16); App.C calls it a "while loop" (p.15)** | Resolved: it is a continue-while predicate mislabeled "until"; semantics agree across paper prose, Algorithm box, and code. NOT inverted by OCR. | no | no (resolved) |
| 7 | `maxiter` / total run length | unspecified | `num_steps1`, `num_steps2//(g·N)` | not in paper | Open ambiguity, not a conflict. | n/a | no (see §E) |
| 8 | `eps_factor` clip [0.3,3.0] | not mentioned | present | not in paper | Code-only safety rail; harmless. | no | no |
| 9 | `num_adaptation_samples` | n/a | passed, **never read** | n/a | Dead code; bisection self-terminates on 3% instead. | no | no |
| 10 | Integrator names | MN2/MN4 (Omelyan) | McLachlan / Omelyan | orders 2/4, grads 2/5 (App.E p.16) | Same order & grad-count; naming only. | no | no |

---

## D. Diagnostics (internals to check our implementation against)

**Internal-health (available at run time):**
- **EEVPD vs target** (Eq. 7): observed `EEVPD` should track `EEVPD_wanted` (Phase 1). Fig. 1 (p.5): orange (observed) tracks grey (target). Only monotonic EEVPD-vs-ε is required, not exact ε⁶.
- **Equipartition `D̃ → 0`** (Eqs. 6/18): total-bias proxy; `V→I`. Fig. 1 left "equipartition (blue dots)".
- **Step-size decay** (Phase 1): smooth decay ⇒ bias decreasing (Fig. 1, p.5, green/orange panels).
- **Acceptance → target** (Phase 2): drive `a → 70%/90%` by bisection to within 3% (Fig. 1).
- **Fluctuation `δ_t[x_i²]`** (Eqs. 10–11): switch when all `δ ≤ 0.01` (paper) / code `r_max ≤ 0.01`. Expected `δ~M^{−1/2}`.
- **split-R̂** (code `R_max`, `R_avg = max/mean(R²−1)`; only if `superchain_size>1`): suggested future switch diagnostic (§7).

**Result-quality (require ground truth; paper's *evaluation* metric, not a run-time certificate):**
- Second-moment bias `b²_t[f] = (E_ρt[f]−E_p[f])²/Var_p[f]` (Eq. 12, p.6). Aggregated `b²_max = max_i b²[x_i²]`, `b²_avg = (1/d)Σ b²[x_i²]` (Eq. 13).
- **Success threshold `b² < 0.01`** (≈100 effective samples) — used throughout for both `b²_max` and `b²_avg`. Verified p.6.

---

## E. Open ambiguities (after PDF verification)

1. **`maxiter` / total run length** numeric value is **not given** in the paper. The Phase‑1 window `T = 20%` of "total sampling time" forward-references this; how `T` is fixed online before the total is known is not stated (presumably `T = 0.2·maxiter`). Code makes `num_steps1`, `num_steps2` user inputs.
2. **Phase-2 stop rule.** Beyond `t < maxiter` and freezing after bisection, the paper gives no rule for how many adjusted steps to collect. Code: `num_samples = num_steps2 // (grad_per_step · steps_per_sample)`, no early stop.
3. **Preconditioner during Phase 2.** Set once at end of Phase 1 (`y_i=x_i/Var^{1/2}`); whether re-estimated in Phase 2 is not stated (App. D implies fixed). Code: fixed (phase-1 `Var_d`).
4. **Per-iteration gradient count in Phase 1.** Whether one Phase-1 iteration = one LF step (1 grad) or several is not spelled out; trajectory length enters only via the refresh `c₁=e^{−ε/L}`.
5. **`C` and the step-size law (the load-bearing conflict).** The paper's verified law is `EEVPD_wanted=F(C·D̃)`, `C=0.025`. The reference code implements `C·D̃^{3/8}`, `C=0.1` from a *different* paper. Which to implement is a **decision for the orchestrator** (recommendation: follow the verified LAPS paper; treat the EMAUS rule as a separate, documented alternative to A/B against).
