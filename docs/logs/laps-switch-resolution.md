# LAPS Phase-1 → Phase-2 switch δ: definition, noise floor, and threshold resolution

**Question.** What is the EXACT definition of the Phase-1 switch statistic δ in LAPS
(Robnik & Seljak 2026, arXiv:2601.16696v1), how does its equilibrium noise floor scale
with the ensemble size M, and is the literal threshold `δ < 0.01` consistent with the
paper's M = 4096 (the build flagged it is not)?

**Verification.** Paper pages read from the *rendered* PDF (ghostscript → PNG → visual),
not text extraction. Empirical floor measured in-container (CPU JAX image, numpy Monte
Carlo), script `…/jobs/cf0ab128/tmp/floor.py`.

---

## 1. The EXACT paper definition of δ (quoted, page 5)

§4 "Switch to adjustment", p.5, right column. Verbatim (typeset, confirmed visually):

> "We define a relative fluctuation of a summary statistic E[f] as
>
> &nbsp;&nbsp;&nbsp; **δ_t[f] = σ_t[f] / μ_t[f],**  (10)
>
> where average and standard deviation are computed in a moving window of length T:
>
> &nbsp;&nbsp;&nbsp; **μ_t[f] = (1/T) Σ_{s=t−T}^{t} E_{ρ_s}[f]**  (11)
> &nbsp;&nbsp;&nbsp; **σ_t²[f] = (1/(T−1)) Σ_{s=t−T}^{t} (E_{ρ_s}[f] − μ_t[f])²**
>
> … If chains were stationary and in the target distribution, we would expect
> **δ[f] = Var_p[f]^{1/2} M^{−1/2} / E_p[f] ∼ M^{−1/2}**, so it should be very small if the
> ensemble is large. … For observables, we will take **f(x) = x_i²** and terminate the
> unadjusted phase when all **δ_t[x_i²]** fall below a certain threshold. **We fix the
> threshold to 0.01 and the moving window size to the 20% of the total sampling time.**"

Algorithm 1 (p.16) prints the loop predicate `until max_i δ[x_i²] > 0.01 and t < maxiter`.

**Reading of the math (unambiguous):**
- The per-step inner quantity is the **cross-chain ensemble mean** E_{ρ_s}[x_i²] = (1/M) Σ_m (x_i^m)²
  at step s — one scalar per coordinate i.
- σ_t[f] is the **plain sample standard deviation (ddof = 1, i.e. 1/(T−1)) over the T window
  STEPS** of that per-step ensemble mean. It is a **standard deviation, NOT a standard error**:
  there is **no division by √T**. The paper's own floor formula `δ = Var_p[f]^{1/2} M^{−1/2}/E_p[f]`
  confirms this — it carries **no √T**, only the M^{−1/2} from the M-chain ensemble mean.
- μ_t[f] is the window mean of the same per-step ensemble means.
- **No detrending / differencing** is implied anywhere; the signal is the raw windowed series.
- Window length **T = 20% of total sampling time**; observable **f = x_i²**; fire when **all
  coordinates** have δ ≤ 0.01 (equivalently `max_i δ_i < 0.01`).

The current `phase1_switch` / `ensemble_mean_observable` in `laps_core.py` implement EXACTLY
this (`switch="paper"`): `jnp.std(window, ddof=1) / jnp.mean(window)`, observable `x_i²`,
`max_i`, fire `< threshold`. **The estimator is faithful. No √T, no standard-error reading.**

---

## 2. Empirical noise floor (definition × M), d = 5, T = 100

Exact-equilibrium ensemble: each step draws an independent ensemble whose particles are
marginally N(0, I) (so the per-step ensemble mean of x_i² carries only finite-M Monte Carlo
noise; for x ~ N(0,1), E_p[x²]=1, Var_p[x²]=2 → theory floor √(2/M)). iid-per-step is the
**decorrelated lower bound** on a real run's window content (consecutive steps of a real run
are autocorrelated; see the persistent test below).

| M | theory √(2/M) | (a) paper δ, per-dim mean | (a) paper δ, **max_i over d=5** (the switch) | (b) standard-error variant σ/√T /μ |
|---:|---:|---:|---:|---:|
| 512 | 0.0625 | 0.0624 | 0.0677 | 0.00677 |
| 4096 | 0.0221 | **0.0221** | **0.0238** | 0.00238 |
| 16384 | 0.0110 | 0.0110 | 0.0119 | 0.00119 |

M needed for the switch quantity `max_i δ` to floor below 0.01 (paper estimator, zero-mean
Gaussian): M=16384 → 0.0119; M=32768 → 0.0085; M=65536 → 0.0060. **⇒ M ≳ 3×10⁴.**

**Persistent (autocorrelated) ensemble at equilibrium**, M=4096, T=100, AR(1) lag-1
autocorrelation ρ (each particle marginally N(0,I)):

| ρ_step | τ = −1/ln ρ | τ/T | `max_i δ` floor |
|---:|---:|---:|---:|
| 0.00 | ∞ (iid) | — | 0.0239 |
| 0.50 | 1.4 | 0.01 | 0.0238 |
| 0.90 | 9.5 | 0.10 | 0.0247 |
| 0.95 | 19.5 | 0.20 | 0.0242 |
| 0.99 | 99.5 | 1.0 | 0.0201 |

**Key empirical fact:** at TRUE equilibrium the floor is **invariant to autocorrelation**
(only the ρ=0.99, τ≈T case dips slightly). This refutes the "autocorrelation suppresses the
window std" rescue: the window sample-std estimates the *marginal* std of the ensemble mean,
which is Var_p[f]^{1/2}/√M regardless of ρ. The floor is a marginal property of an M-chain
ensemble, not an autocorrelation artifact.

---

## 3. Resolution of the contradiction

1. **The build's central finding is CONFIRMED, and slightly strengthened.** The paper's δ
   (window std / window mean, no √T) floors at **√(Var_p[f]/M)/E_p[f]**. For a zero-mean
   unit-variance coordinate with f=x² that is **√(2/M)**: 0.0221 (per-dim) / **0.0238 (max
   over d=5)** at M=4096 — **above** 0.01. The switch quantity is `max_i δ`, which sits a
   further ~8% (d=5) to ~20% (d≈25) above the per-dim floor (extreme-value inflation grows
   slowly with d). The literal 0.01 is **not** reachable at the paper's M=4096 for a generic
   zero-mean observable. (Note: the build quoted "0.044 at M=512"; the *correct* √(2/M) floor
   is 0.0625 there — the build's number was actually 1/√M, i.e. it dropped the √2. The true
   floor is LARGER, so the tension is worse, not better.)

2. **It is NOT a standard-error / √T definition.** Variant (b) (σ/√T) would give 0.0024 at
   M=4096 and fire trivially — but Eq. 11 is a plain sample std and the paper's floor formula
   has no √T. Reading δ as a standard error is unsupported by the typeset paper.

3. **It is NOT an autocorrelation effect.** Confirmed: the equilibrium floor is ρ-invariant.

4. **What the paper actually relies on.** The paper writes the floor as "**∼ M^{−1/2}**",
   **dropping the prefactor** `Var_p[f]^{1/2}/E_p[f]` (= √2 ≈ 1.41 for a zero-mean Gaussian
   x²). M^{−1/2} at M=4096 is 0.0156; with the √2 prefactor restored it is 0.022 — exactly the
   gap that puts the true floor above 0.01. The 0.01 threshold is therefore an **implicitly
   M-tuned, observable-dependent** number: it is a "fluctuations are no longer changing
   significantly" heuristic (the paper's words, §4), NOT a value the estimator reaches at its
   asymptotic floor for M=4096. The paper's German-Credit run (Fig. 1) fires at ~210 grad
   evals, i.e. *during the transient* as δ decays through 0.01 from above; for real-posterior
   coordinates with nonzero mean the per-coordinate floor `√(Var_p[x²]/M)/E_p[x²]` is *below*
   √2/√M (factor √(2σ⁴+4μ²σ²)/(μ²+σ²) < √2 for |μ|>0), which helps individual coordinates —
   but the worst (most zero-mean) coordinate still governs `max_i` and keeps the literal 0.01
   marginal at M=4096. **The paper is genuinely under-specified here:** it does not state how
   the switch reconciles 0.01 with its own √2/√M ≈ 0.022 floor at M=4096; the operative
   behaviour is "fire on the transient, else hit maxiter." This is an inference, flagged as such.

**Bottom line:** δ is defined and implemented correctly; the literal `0.01` is **not** a
floor-consistent threshold at M=4096 and is **catastrophically** inconsistent at the smaller M
GIGALens will use (M=512 floor = 0.063 ⇒ Phase 1 can never switch and always burns maxiter).

---

## 4. Recommendation for the GIGALens default

**Do NOT keep the literal `δ < 0.01` for GIGALens.** GIGALens runs at M of order 10²–10³,
where 0.01 is 3–6× below the irreducible floor; Phase 1 would never fire on δ and would always
terminate on `maxiter`, making the switch a silent no-op (a hidden scientific default — exactly
what `docs/project-standards.md` forbids).

**Primary recommendation — self-calibrated, relative-to-floor switch (M-, observable-, and
posterior-aware).** The floor is *computable online* from the current ensemble:
`floor_i = √( Var_{ρ}[x_i²] / M ) / E_{ρ}[x_i²]` (both moments are already cross-chain
reductions you have). Fire when

&nbsp;&nbsp;&nbsp; **max_i ( δ_i / floor_i ) < k**, with **k ≈ 1.5–2.0**.

Mechanism: δ_i decays toward floor_i from above as the ensemble mean stops drifting; firing at
δ_i ≈ k·floor_i means "fluctuations are within k× of their irreducible Monte-Carlo floor" — an
M-invariant, problem-invariant statement of "no longer changing significantly", which is the
paper's stated intent. This reduces *exactly* to the paper's δ∼M^{−1/2} heuristic but with the
prefactor kept. Recommended default **k = 1.5** (fires when within 50% of floor), exposed and
documented.

**Simpler fallback — explicit M-aware threshold.** If a constant is preferred,
**threshold(M) = k · M^{−1/2}** with **k ≈ 2.0–2.1** (= κ·√2 with margin κ≈1.4–1.5). Checks:
M=4096 → 0.033 (≈1.4× the 0.024 measured floor); M=512 → 0.093 (≈1.4× the 0.068 floor). This
keeps a fixed "1.4× above floor" firing margin across M. It ignores per-coordinate
mean/variance structure (slightly conservative); the self-calibrated form above is preferred.

Note for the record: the literal `0.01` corresponds to the M-aware rule at **M ≈ 2×10⁴**
(k=2 ⇒ M=(2/0.01)²=4×10⁴; k=1.4 ⇒ M≈2×10⁴). Keep `0.01` available only as an explicit
`threshold_literal_paper` option for reproducing the paper at large M, never as the default.

**Always pair with the existing `maxiter` fallback** (already present) so a hard zero-mean
coordinate cannot stall Phase 1 indefinitely.

---

## 5. Correction needed to `phase1_switch` (laps_core.py)

- **The estimator is correct — no bug.** `phase1_switch` faithfully implements Eqs. 10–11:
  `mu = mean(window)`, `sigma = std(window, ddof=1)`, `delta = |sigma/mu|`, `max_i`, fire
  `< threshold`. `ensemble_mean_observable(switch="paper")` correctly uses f = x_i². Keep both.
- **Change the default threshold semantics.** `threshold=0.01` is hard-coded as a literal and
  is unreachable for M < ~2×10⁴. Recommended changes:
  1. Add the floor-aware path: accept the ensemble second/fourth moments (or M and the
     per-coordinate `Var_ρ[x²]`, `E_ρ[x²]`) and fire on `max_i(δ_i/floor_i) < k`, default
     k=1.5; OR accept `threshold=k/√M`. Make the literal 0.01 an opt-in
     (`threshold="paper_literal"`), not the default.
  2. **Emit a warning/assert when the configured threshold is below the estimated floor**
     `√(2/M)` (i.e. when M < ~2×10⁴ and threshold=0.01): the switch cannot fire at the floor
     and Phase 1 will silently run to maxiter. This converts a hidden no-op into a visible
     condition (project-standards: no silent scientific defaults).
  3. **Fix the docstring.** It says "At stationarity delta ~ M^{−1/2}". Correct/expand to:
     "floor = √(Var_p[f]/M)/E_p[f] = √(2/M) for a zero-mean unit-variance x² coordinate (≈0.022
     at M=4096, ≈0.063 at M=512); the literal 0.01 is only reachable for M ≳ 2×10⁴, and `max_i`
     inflates the floor a further ~8–20% over d." Keep the spec's `δ<0.01` note tagged as
     "paper-literal, large-M only".

---

### Appendix — what δ would fail to detect (metric blind spot)

δ is a *fluctuation-stalling* diagnostic, not a *bias* diagnostic. It goes small whenever the
windowed ensemble mean of x_i² stops moving — which also happens if the ensemble is **stuck**
(too-small step size, trapped mode) rather than converged, and it is blind to a coordinate's
mean (it monitors x_i², so a drift in E[x_i] that leaves E[x_i²] stationary is invisible).
It is correctly paired in LAPS with the equipartition divergence D̃ (the actual bias proxy);
the switch should never be read as a convergence certificate on its own.
