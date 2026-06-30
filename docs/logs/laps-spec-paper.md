# LAPS — Paper-only specification (Reader A)

**Source:** Jakob Robnik & Uroš Seljak, *"Faster parallel MCMC: Metropolis adjustment is best served warm"*, arXiv:2601.16696v1 [stat.CO], 23 Jan 2026.
File: `papers/LAPSRobnik2026.pdf` (21 pages incl. supplementary).

**Method of reading:** PDF text extracted with `gs -sDEVICE=txtwrite` (no poppler/python-pdf available). OCR-style artifacts in equations are flagged where they affect interpretation. Equation/section/figure numbers below are the paper's own.

This document is a *literal, paper-only* reconstruction. Where the paper leaves something unspecified I write **[UNSPECIFIED IN PAPER]**; where I infer I say so.

---

## 1. What LAPS is and what it targets

**Name.** LAPS = **Late-Adjusted Parallel Sampler** (Abstract; §2 "the resulting algorithm, termed Late-Adjusted Parallel Sampler (LAPS)").

**Problem it solves.** Parallel/ensemble MCMC ("many short chains," each chain collects a *single* sample) is fast on GPUs/TPUs but is sensitive to initialization: it converges slowly from a *cold start* (chains far from the target) because, after burn-in, the only error left is the finite-ensemble variance (§1, §2). Metropolis–Hastings (MH) adjusted methods converge fast from a **warm start** but suffer from a cold start (low acceptance away from the typical set); unadjusted methods are the opposite — fast from cold start but biased, needing a small step size for fine convergence (§2 "Late adjustment"). The known theoretical optimum is: run **unadjusted** to get from cold→warm, then turn on **adjustment** for fine convergence (Durmus & Eberle 2023; Altschuler & Chewi 2024). The obstacle in practice is *automatically choosing the unadjusted step size*. LAPS is "a first practical parallel algorithm that initially performs unadjusted sampling and then uses MH adjustment to achieve fine convergence" (§2 "Our contributions"), with **all hyperparameters selected automatically** (Abstract).

**What it produces.** **Posterior samples / expectation values only.** Output is the final ensemble of chain locations `{x^m}` used to estimate `E_p[f] ≈ (1/M) Σ_m f(x^m)` (Eq. 3; Algorithm 1 "Output: samples {x^m}"). **It does NOT compute a normalizing constant / evidence / log Z.** There is no mention anywhere of evidence, marginal likelihood, partition function as an *output*, or thermodynamic integration. (The word "partition function" appears only in a *cited reference title*, Röver et al. 2023.) — This is a clean, confident negative: LAPS as defined is a sampler, not an evidence estimator.

**Core idea (3–5 sentences).** Evolve `M` independent chains in parallel with **MCLMC** (Microcanonical Langevin Monte Carlo) dynamics. During an **unadjusted** phase, adapt the step size on the fly so that the *asymptotic (discretization) bias* is held a fixed fraction `C` below the *total bias* (distance to target), estimating total bias via an **equipartition** diagnostic and asymptotic bias via the **energy-error variance (EEVPD)** — so the step size starts large and shrinks as chains approach the target. When ensemble summary statistics stop changing (chains near stationarity), **switch on Metropolis adjustment** (the MAMS kernel) and tune the step size to a target acceptance rate by bisection; then freeze hyperparameters, which removes the ensemble-adaptation bias and lets the ensemble converge exactly to the target. LAPS thus combines unadjusted (fast cold→warm) and adjusted (fast fine convergence) sampling.

**Relation to other methods.**
- **MCLMC / microcanonical Langevin:** LAPS *uses* MCLMC as its underlying kernel (§2 "Choice of the kernel"; SDE Eq. 2). Chosen over HMC/LMC for faster, more deterministic convergence; ablation in App. D shows microcanonical dynamics is *important* (replacing with underdamped Langevin is much slower).
- **MAMS** (Metropolis Adjusted Microcanonical Sampler, Robnik et al. 2025): LAPS's **adjusted phase kernel** (Algorithm 1 "MAMSupdate"; App. F). "Only the adjusted phase" = a parallel MAMS (App. D, Table 3).
- **ECA** (Ensemble Chain Adaptation, Gilks et al. 1994): the hyperparameter-adaptation framework — ensemble averages at iteration `t` inform hyperparameters at `t+1` (§2; Fig. 5). This couples the chains and introduces an asymptotic bias that is small for large `M` and is *eliminated by freezing* hyperparameters after the adjusted phase converges.
- **Annealing / SMC / parallel tempering / path sampling:** LAPS does **NOT** use these. It has **no temperature schedule, no annealing path, no resampling, no superchains.** The paper explicitly lists annealing/SMC/parallel-tempering as *future work* that is "agnostic with respect to the kernel" and "compatible with LAPS" — e.g. "in annealing, one could run LAPS at each temperature level" (§7 Future work). So any annealing/path-sampling layer is *outside* the paper's LAPS.
- Baselines it beats: ChEES-HMC, MEADS, parallel NUTS, Pathfinder (initializer), sequential NUTS and sequential MCLMC.

---

## 2. The algorithm, step by step

Full pseudocode: **Algorithm 1** (App. C, p. 16). Two phases: a `repeat` loop (unadjusted) then a `while` loop (adjusted). ECA schematic: Fig. 5.

### Underlying dynamics (both phases)
Ideal continuous-time MCLMC solves the SDE (Eq. 2):
```
dx = u ds
du = (I − u uᵀ)( ∇log p(x_s)/(d−1) ds + η dW )
```
`W` = Wiener process; `η` = stochasticity strength. The velocity stays on the unit sphere (norm fixed — microcanonical). In discretized form (Eq. 23) stochasticity is reparameterized by a length scale `L` with `η = √(2/(L d))` in the small-step limit, making `L` comparable to an HMC trajectory length (§2). Discrete kernel `φ(·|ε,L)` is a splitting integrator (App. E): `φ = Φ^O_{ε/2,L} ∘ Φ_ε ∘ Φ^O_{ε/2,L}` (Eq. 21), with a deterministic position update `Φ^A` (Eq. 24), velocity update `Φ^B` (Eq. 25, a closed-form rotation keeping ‖u‖=1), and stochastic partial-refresh `Φ^O` (Eq. 23, `c₁=e^{−ε/L}`).

### Phase 1 — Unadjusted initialization (§3)

**Purpose (explicit, §3):** *not* to converge to the target, only to "approach it as fast as possible."

**Governing principle (§3.1, App. A Theorem 1):** the optimal step-size schedule keeps the asymptotic bias `D(p,p_ε)` a fixed factor below the total bias `D(p,ρ_t)`:
```
D(p, p_{ε_t}) = C · D(p, ρ_t)        with 0 < C < 1
```
App. A proves (Theorem 1, Eqs. 14–16) that under Assumption A1 `D(p,p_ε) ≤ c_asym ε^κ` and Assumption A2 (contraction toward biased limit, rate `c`), the schedule `c_asym ε_t^κ = c/((2−c)(κ+1)) · D(ρ_t,p)` maximizes the convergence-rate lower bound and guarantees `D(ρ_{t+1},p) ≤ (1 − cκ/(κ+1)) D(ρ_t,p)`.

Because true distances are intractable, two computable proxies are used:

**(a) Total bias proxy — equipartition (§3.1, App. B).** Equipartition matrix (Eq. 4):
```
V_ij(p,ρ_t) = E_ρt[ −(x_i − E_ρt[x_i]) ∂_j log p(x) ]
```
`V = I` exactly when `ρ_t = p` (Eq. 5, integration by parts). Bias proxy (Eq. 6):
```
D̃(p,ρ_t) = (1/d) Tr[ (I − V)(I − V)ᵀ ]
```
`D̃` is *not* a true metric (only summary-statistic based) but App. B.2 proves it is a **divergence on zero-mean Gaussians** (Eqs. 19–20). Computed via ECA (Eq. 3) cheaply. Naïve cost `O(d²M)`; **Hutchinson's trick** with **100 Rademacher `z`-vectors** reduces full-rank cost to `O(dM)` (Eq. 17, App. B.1), giving sub-percent error. **Default is the diagonal version** `D̃_diag = (1/d) Σ_i (1 − V_ii)²` (Eq. 18) — avoids the memory of storing 100 d-vectors; App. D Table 4 shows full-rank vs diagonal differences are marginal.

**(b) Asymptotic bias proxy — energy error / EEVPD (§3.1).** A step of approximate dynamics changes the conserved "energy" by `∆(x,u|p,ε)` (closed forms: Eq. 26 for velocity update, and `∆ = −log p(x+εu)+log p(x)` for position update, App. E). Define **EEVPD** (Energy Error Variance Per Dimension, Eq. 7):
```
EEVPD(ρ|p,ε) = Var_ρ[∆(x,u|p,ε)] / d
```
It upper-bounds asymptotic bias (Eq. 8):
```
D̃(p,p_ε) ≤ F^{-1}( EEVPD(p_ε|p,ε) ),   F(D̃) = 4 D̃^{3/2} / (1 + D̃^{1/2})²
```
(`F` monotonically increasing; bound rigorous for Gaussians, numerically typical otherwise.) Empirically `EEVPD ∝ ε⁶` (§3.1).

**Step-size update (5 steps, §3.1; Algorithm 1):**
1. Compute `D̃(p,ρ_t)` (Eq. 6 / 18).
2. Desired asymptotic bias `D̃(p,p_{ε_t}) = C·D̃(p,ρ_t)`.
3. Desired `EEVPD_wanted = F( C·D̃(p,ρ_t) )` (Eq. 8).
4. Estimate current `EEVPD(ρ_t|p,ε_t)` from the ensemble variance of `∆` (Eq. 3).
5. Rescale using `EEVPD ∝ ε⁶`:
   ```
   ε_{t+1} = ε_t · ( EEVPD_wanted / EEVPD(ρ_t|p,ε_t) )^{1/6}
   ```
Fig. 1 (right) shows EEVPD tracks its (adaptively moving) target and step size decays → rapid bias decay.

**Trajectory-length / momentum-decoherence scale `L` (§3.2):** retuned each iteration (Eq. 9):
```
L_t = α ( Σ_{i=1}^d Var_ρt[x_i] )^{1/2}
```
i.e. proportional to the typical posterior size. Default **α = 2** here (vs α=1 in MCLMC; larger value suits the more deterministic dynamics; ablation App. D).

**Initialization (Algorithm 1; App. C):**
- Step size `ε ← 0.01 √d`.
- `L ← Eq. (9)` from initial positions.
- Initial **positions provided by user** (experiments draw from the prior).
- Initial **velocity aligned with the gradient**, unit norm.
- Integrator in unadjusted phase: **Leapfrog (LF)**, 1 gradient/step (App. E: "preferred … in the unadjusted phase").

**Stopping rule for Phase 1 (§4):** monitor relative fluctuation of a summary statistic over a moving window of length `T` (Eqs. 10–11):
```
δ_t[f] = σ_t[f] / μ_t[f],   μ_t[f] = (1/T)Σ_{s=t−T}^{t} E_ρs[f],   σ_t² = (1/(T−1)) Σ (E_ρs[f]−μ_t[f])²
```
At true stationarity `δ[f] ~ M^{−1/2}` (small for large ensembles). Observable `f(x) = x_i²`. **Terminate Phase 1 when all `δ_t[x_i²]` fall below threshold 0.01**, with **moving-window size = 20% of total sampling time** (§4). [NOTE/ambiguity: "20% of total sampling time" is forward-referencing a total run length; see §"Open ambiguities".] The Algorithm-1 `until` line OCR-reads `until max_i δ[x²] > 0.01 and t < maxiter`, which is the *negation* of the prose; I trust the prose: stop when all `δ < 0.01` (or `t ≥ maxiter`).

**End-of-Phase-1 preconditioning (§4):** diagonal mass-matrix preconditioning via coordinate transform `y_i(x) = x_i / Var_ρt[x_i]^{1/2}`.

### Phase 2 — Adjusted sampling (§5)
Turn on MH adjustment using the **MAMS** kernel (App. F), still adapting via ECA, then freeze.

**MAMS kernel (App. F):** from `(x,u)`: (1) resample `u` uniformly on the unit sphere (`u = Z/‖Z‖`, `Z~N(0,I)`); (2) apply `N` integrator steps (step `ε`) with partial velocity refreshment each step; (3) accumulate energy error `∆`; (4) accept with prob `min(1, e^{−∆})`, else stay. Stationary distribution is *exactly* the target (MH removes integrator bias).

**Step-size tuning (§5.1):** tune `ε` to a **target average acceptance rate**. Because `M` is large the acceptance estimate is low-noise, so **no dual averaging** — instead **bisection** on `a(ε) − a_targeted`: first double/halve `ε` to bracket the root, then bisect until within tolerance **3%**. Converges in a few steps (Fig. 1). After convergence **hyperparameters are frozen** → eliminates ECA bias (typically "unobservably small"). The `ADAPT` flag (Algorithm 1) stops adapting once `|a − a_targeted| ≤ 0.03`.

**Target acceptance rate (§5.1):** default **70%** for second-order integrators, **90%** for fourth-order (extending Beskos 2013 / Neal 2011 / Betancourt 2015 optimal-acceptance arguments; slightly-high values are numerically more stable). Ablation: marginal impact.

**Trajectory length in Phase 2 (§5.2):** **not adapted** (phase is short). **Fixed `N = 15` integration steps per trajectory**, `L_full = 15 ε`. Partial-refreshment `L` set to `L = 1.25 L_full` (per Riou-Durand & Vogrinc 2023).

**Integrator in Phase 2 (App. E):** **MN2** (2nd-order minimal-norm, 2 grad/step) by default; **MN4** (4th-order, 5 grad/step) **for d > 200** (high-dimensional fine convergence).

**Resampling / superchains:** **None.** No SMC-style resampling. The only inter-chain coupling is ECA averaging.

**Overall stopping:** the `while` loop runs `while t < maxiter`. Output = final chain locations. `maxiter` value is **[UNSPECIFIED IN PAPER]** numerically.

---

## 3. Hyperparameters

| Name | Controls | Default | Paper's stated reason / mechanism |
|---|---|---|---|
| `C` | Ratio of asymptotic bias to total bias in the unadjusted step-size schedule (`D̃(p,p_ε)=C·D̃(p,ρ_t)`) | **0.025** | "Optimal constant `0<C<1` is not known and can depend on the problem" (§3.1). Theoretical basis: App. A Theorem 1 (schedule maximizes convergence rate). Trade-off: small `C` → keep discretization bias well below total bias (safe, but smaller steps/slower); large `C` → bigger steps but risk asymptotic bias dominating. App. D Fig. 6 shows performance stable over a wide range; GC & IRT could improve ~2× with smaller `C`. *Default value itself is empirical (ablation-chosen), not derived.* |
| `α` | Momentum-decoherence / trajectory length scale `L_t = α (Σ Var[x_i])^{1/2}` (Eq. 9) | **2** | Time before momentum decoheres should be "comparable to the typical scale of the posterior." α=1 is the MCLMC default; "a larger value α = 2 works better, corresponding to the more deterministic dynamics" (§3.2). Empirical (ablation App. D Fig. 6). |
| `a_targeted` | Target average acceptance rate in adjusted phase (bisection target) | **70%** (2nd-order), **90%** (4th-order) | Optimal-acceptance theory for product targets / HMC ⇒ 60–80% (2nd-order), 80–87% (4th-order) (Beskos 2013; Neal 2011; Betancourt 2015; applies to MCLMC per Robnik 2025). Slightly larger = numerically more stable (Phan 2019). Ablation: marginal impact (App. D Fig. 7). |
| bisection tolerance | Stop bisection when `|a − a_targeted|` within this | **3%** | Stated as "up to some tolerance (3% by default)" (§5.1). No deeper justification given — **default given without mechanistic justification.** |
| `N` (steps/trajectory, adjusted) | Integration steps per MAMS proposal; `L_full = Nε` | **15** | Phase is "typically quite short" so `L` is not adapted; fixed at 15 (§5.2). Ablation App. D Fig. 7 (steps-per-trajectory) shows stability. *Why 15 specifically is not derived* — chosen empirically. |
| partial-refresh `L` (adjusted) | Velocity refreshment scale within a trajectory | **1.25 · L_full** | "as recommended in Riou-Durand and Vogrinc (2023)" (§5.2). Cited, not re-derived. |
| `δ` threshold | Phase-1 termination: stop when all `δ_t[x_i²] <` this | **0.01** | At stationarity `δ ~ M^{−1/2}`; a small `δ` "indicates we are approaching equilibrium" (§4). Threshold value 0.01 stated as a fixed choice — **no derivation of why 0.01.** |
| moving-window `T` | Window for `δ_t` fluctuation estimate | **20% of total sampling time** | "We fix … the moving window size to 20% of the total sampling time" (§4). No mechanistic justification; and "total sampling time" is forward-referential (see ambiguities). |
| `M` (number of chains) | Ensemble size = number of output samples | **4096** (ensemble-vs-ensemble exps), **256** (vs sequential) | Accuracy ∝ ensemble variance. App. D Fig. 9: performance ~independent of `M` for **M > 128–256**, target-dependent. Lower `M` cannot reach `b²_avg=0.01` (~100 effective samples needed) (§6, App. D). |
| Hutchinson `z` count | Random vectors for full-rank `D̃` estimate | **100** (Rademacher) | "100 realizations … ensures sub-percent error in the D̃ computation" (App. B.1). Only relevant if full-rank `D̃` used; default is diagonal `D̃_diag` (no `z` needed). |
| initial `ε` | Phase-1 starting step size | **0.01 √d** | Algorithm 1 init. **No justification given** beyond being a starting point the feedback loop corrects. |
| integrator choice | LF (unadjusted), MN2 (adjusted, d≤200), MN4 (adjusted, d>200) | as listed | LF: 1 grad/step, low accuracy ok in unadjusted phase. MN2: accuracy at low-D. MN4: 4th-order error ∝ε⁴, "crucial … if the target is high-dimensional"; "adopt it for d > 200" (App. E). |

**Headline claim:** "all the hyperparameters are selected automatically" / "applicable out of the box" (Abstract) — meaning `ε`, `L`, switch time, and adjusted-`ε` are *adapted at run time*; the *meta*-hyperparameters above (`C`, `α`, thresholds, `a_targeted`) are fixed constants the user need not touch.

---

## 4. Diagnostics / convergence criteria

**Internal-health diagnostics (is the sampler mechanically behaving):**
- **EEVPD tracking** (Eq. 7): in Phase 1, observed EEVPD should track its adaptively-set target `EEVPD_wanted = F(C·D̃)` (Fig. 1 upper-right; Fig. 1 caption; §"Robustness" p.8 notes orange tracks grey). The *feedback loop* (reduce `ε` if EEVPD too high, increase if too low) is what matters; exact `EEVPD ∝ ε⁶` need not hold, only monotonic growth with `ε` (§"Robustness").
- **Acceptance rate** (Phase 2): drive `a → a_targeted` (70%/90%) by bisection to within 3% (Fig. 1; §5.1).
- **`D̃` equipartition loss** (Eqs. 6/18): used as the *total-bias proxy* driving the schedule; `D̃ → 0` ⇔ `V → I`. Shown as "blue dots" in Fig. 1 left.
- **Step-size decay** (Phase 1): should decay smoothly (Fig. 1 lower-right) — signals bias decreasing.

**Switch criterion (Phase 1 → Phase 2):**
- Relative fluctuations `δ_t[x_i²]` (Eqs. 10–11): all `< 0.01` over a 20%-window ⇒ chains near stationarity ⇒ switch. This is the operational convergence-detection rule. Expected scale at stationarity `δ ~ M^{−1/2}`.

**Result-quality diagnostics (are the samples right) — used for *evaluation in the paper*, not as internal stopping rules:**
- **Second-moment bias** relative to ground truth (Eq. 12):
  ```
  b²_t[f] = (E_ρt[f] − E_p[f])² / Var_p[f]
  ```
  Aggregated as **`b²_max` (max over params)** and **`b²_avg` (mean over params)** (Eq. 13). Success threshold used throughout: **`b² < 0.01`** ("low bias"), corresponding to ~100 effective samples. Ground truth from analytic values (Ill-Conditioned Gaussian) or long sequential NUTS (others). These require ground truth and are *not available at run time* — they are the paper's validation metric, not LAPS's internal certificate.
- Fig. 1 left panel demonstrates that the *run-time-available* proxies (equipartition loss `D̃`, second-moment fluctuations `δ`) "reflect the actual bias quite well" — i.e. the internal diagnostics are validated against the true bias.

**Distinction made explicit (Fig. 1 caption):** the residual error of the final ensemble is *finite-`M` variance, not bias* ("the final ensemble is not biased, the residual error is caused by the finite number of chains").

**Suggested alternative diagnostic (Future work, §7):** split-R̂ (Margossian et al. 2024) "could also be useful for determining when to switch on adjustment" — i.e. the authors flag the switch criterion as an area with alternatives.

---

## 5. Documented failure modes & difficult posteriors

From §"Robustness" (p. 8) and §7:

1. **`D̃` underestimates distance (pathological "false-converged" case).** "It is possible to construct pathological cases where `D̃(p,ρ_t)` is zero (or very small) but the distributions are not close." ⇒ step size set too short ⇒ slow progress. Root cause: `D̃` is based only on the equipartition summary statistics (and the construction is "largely based on the Gaussian assumption").
2. **Long/heavy tails (e.g. Banana) — `D̃` over-stringent.** For long-tailed distributions `D̃` "is an overly stringent metric and even for exact i.i.d. samples decays only slowly." ⇒ step size driven too large ⇒ "the ensemble gets stuck, limited by the discretization bias."
3. **Common outcome & built-in safety net:** both failure modes give an ensemble that reaches a *stationary distribution ≠ target*. **But** because expectation values settle, the `δ` fluctuation criterion fires and **the adjusted phase (which does not rely on the Gaussian assumptions) is triggered** — "in the worst case, the adjusted phase is triggered," which makes the scheme robust. Demonstrated by the Stochastic Volatility example where the unadjusted phase reaches only `b²_max = 21.5` (Fig. 2; §6) yet final convergence is achieved by the adjusted phase.
4. **EEVPD ∝ ε⁶ not exact** — fine, because only monotonic EEVPD-vs-ε is needed (feedback loop self-corrects).
5. **Adjusted-only (no unadjusted phase) is unreliable** (App. D Table 3): parallel MAMS (with or without ADAM init) fails to converge on Ill-Conditioned Gaussian, IRT, Stochastic Volatility within 2000 grads; cold-start MH acceptance is too low. ⇒ unadjusted warm-up is essential.
6. **Unadjusted-only never reaches very low bias** (App. D Fig. 8): plateaus at a stationary bias; adjusted phase needed for the final decay. Switching too *early* "dramatically slows down convergence."
7. **Too few chains:** `M < 128–256` cannot reach `b²_avg=0.01` because output accuracy is variance-limited at `M` samples (App. D Fig. 9).
8. **Multimodality:** **explicitly out of scope / not solved.** "Several parallel methods such as sequential Monte Carlo, annealing and parallel tempering approach the problem of multi-modal distributions … compatible with LAPS" — i.e. LAPS alone does **not** address multimodality; it would need an annealing/SMC/tempering wrapper (Future work, §7). The Conclusion also lists "heavy tails, multimodality or complex geometry" as needing "extensive testing … to establish the LAPS strategy in the broader context" — i.e. these are *not yet validated*.
9. **High dimensionality:** addressed by switching to the 4th-order MN4 integrator for `d > 200` (App. E); tested up to `d = 2519` (Stochastic Volatility) successfully.

---

## 6. Validation in the paper

**Test problems** (Inference Gym, Sountsov et al. 2020; §6):
| Problem | Dim | Notes |
|---|---|---|
| Banana | 2 | banana-shaped (Fig. 3) |
| Ill-Conditioned Gaussian | 100 | condition number ~10⁵, random orientation, analytic ground truth |
| German Credit (GC) | 51 | sparse Bayesian logistic regression (hierarchical) |
| Brownian Motion | 32 | hierarchical, unknown innovation noise, missing data |
| Item Response Theory (IRT) | 501 | hierarchical |
| Stochastic Volatility (SV) | 2519 | hierarchical, non-Gaussian random walk, S&P500 |

Ground truth: analytic (Gaussian) or long sequential NUTS (others). Metric: gradient evaluations to `b² < 0.01` (grads ∝ wall-clock since gradient dominates).

**Baselines:** NUTS (parallel & sequential), ChEES-HMC, MEADS, sequential MCLMC, Pathfinder (ensemble, Algorithm 2 of Zhang 2022), MAMS (= LAPS adjusted-only, App. D).

**Headline results:**
- **Table 1** (grads to low *max* bias, per chain, `M=4096`): LAPS best on **all** problems, by factors **2–20** over NUTS/ChEES/MEADS. E.g. Banana 17 (vs 264–390), Gaussian 308 (vs 5138–9846), SV 1325 (vs 2860–4173).
- **Table 2** (grads to low *avg* bias, `M=256`): per-chain (wall-clock proxy) LAPS dramatically faster (Banana 17, SV 1100); *total* grads (all chains) sequential methods often cheaper — explicit, honest caveat that LAPS wins on *wall-clock when parallelized*, not on total compute.
- **Fig. 2** (SV): LAPS converges much faster than ensemble baselines; here the unadjusted phase alone does *not* meet accuracy and the adjusted phase is essential.
- **Fig. 3** (Banana): converges to posterior in ~20 grad calls/chain.
- **Pathfinder** failed to converge on most benchmarks (only Banana `b²_max=0.02` in 67 grads, and GC `b²_max=21.5` in 135 grads — i.e. didn't really converge on GC).
- **Wall-clock:** "two orders of magnitude lower wall-clock than … NUTS" (Abstract); "one to two orders faster than sequential MCLMC and two to three than NUTS" (§6).
- **Ablations (App. D):** stable over wide `C`, `α` ranges (Fig. 6, ≤2× variation); adjusted-phase hyperparameters ~10% effect (Fig. 7); both phases necessary (Tables 3, Fig. 8); diagonal ≈ full-rank `D̃` (Table 4); microcanonical dynamics beats underdamped Langevin (App. D "Dynamics"); robust for `M > 128–256` (Fig. 9).

**Hardware:** NVIDIA A100 (40GB). Implementation: JAX / Blackjax (§2).

---

## Open ambiguities / under-specified items (flagged)

1. **`maxiter` / total run length.** Algorithm 1 bounds both loops by `maxiter`, but its numeric value is **[UNSPECIFIED IN PAPER]**. This matters because the Phase-1 window `T` is "20% of total sampling time" — a forward reference to a total length that is itself not given. How `T` is set in a single online run (before total length is known) is **not made concrete**; presumably `T = 0.2 × maxiter`, but the paper does not say so explicitly. *Inference, not stated.*
2. **Phase-2 termination.** Beyond `t < maxiter` and freezing hyperparameters after bisection converges, there is **no stated rule for how many adjusted steps to take / when to stop collecting**. Output is "final locations," but the number of post-freeze iterations is unspecified.
3. **Algorithm-1 `until` condition appears inverted/garbled in OCR** (`until max δ > 0.01 and t < maxiter`) vs the prose (§4: stop when all `δ < 0.01`). I trust the prose. Worth confirming against the actual PDF rendering if precision is needed.
4. **`F` functional form** read from OCR as `F(D̃)=4 D̃^{3/2}/(1+D̃^{1/2})²`. The superscripts are reconstructed from garbled glyphs ("e3/2", "e1/2") — high-confidence given monotonicity claim and the EEVPD↔bias context, but **verify exact exponents** against the rendered PDF.
5. **Unadjusted "MCLMC update (21)" granularity.** Whether one Phase-1 iteration = one integrator step (one gradient) or several is not spelled out in Algorithm 1; LF is 1 grad/step and the per-iteration update appears to be a single step, but the trajectory-length `L` enters only through the refresh `c₁=e^{−ε/L}`. Mild ambiguity in counting "gradient evaluations" per iteration.
6. **Preconditioner update during Phase 2.** Diagonal preconditioning is set *once* at end of Phase 1 (`y_i = x_i/Var^{1/2}`); whether it is re-estimated during Phase 2 is not stated (App. D MAMS comparison says MAMS was *given* the LAPS post-unadjusted preconditioner as a fixed advantage, implying it is fixed).
7. **`C = 0.025` and `α = 2` are empirical**, chosen by the App. D grid search; the paper does not derive them from first principles (Theorem 1 gives the *form* of the schedule, not the constant `C`).
8. **No evidence/log-Z** — confident negative, but worth stating plainly to whoever expected a normalizing-constant output: LAPS does not produce one.
