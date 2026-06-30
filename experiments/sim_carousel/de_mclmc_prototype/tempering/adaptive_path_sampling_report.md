# Adaptive Path Sampling / Continuous Tempering — method report for the carousel MCLMC sampler

**Paper:** Yuling Yao, Collin Cademartori, Aki Vehtari, Andrew Gelman, *"Adaptive Path
Sampling in Metastable Posterior Distributions"* (1 Sep 2020). File
`/global/u1/l/linusu/GIGALens-Code/papers/AdaptivePathSamplingTempering.pdf`.

**Reading method / provenance note.** HPC has no PDF renderer (no poppler) and no
PDF text library in any available Python env. I extracted the text directly by
decompressing the PDF's FlateDecode content streams with the built-in `zlib` and
parsing the LaTeX `TJ`/`Tj` text operators (114/116 streams decoded; full body
recovered, ~106 k chars). The math reads cleanly but loses some sub/superscript
structure, so **every equation below was transcribed by hand and the symbols
reconstructed from context** — I flag any place that reconstruction is uncertain.
Section/figure/equation numbers are the paper's own. The public Stan/R repo was
additionally fetched live (Appendix B URL) and confirmed. Tags used throughout:
**[PAPER §x]** = verified from the paper at that location; **[REPO]** = verified
from the GitHub README I fetched; **[OWN]** = my own knowledge / inference, not in
the paper; **[CAROUSEL]** = from our lab notebook `docs/logs/carousel-mclmc-sampling.md`.

---

## 0. TL;DR

- **Method name:** *Adaptive continuous (simulated) tempering with path sampling*
  (the paper's Algorithm 1). A second algorithm (Algorithm 2, "implicit
  divide-and-conquer") targets *entropic*/funnel barriers and is **not** our case.
- **Does it solve PT's R× replica cost?** **Yes, structurally.** It replaces the
  R-replica ladder with **one** augmented chain that carries a *single continuous
  temperature coordinate* `a` (→ β = f(a)). No replicas. The thing that normally
  makes a single tempered chain fail — not knowing the per-temperature normalizing
  constants z(β) needed to mix in temperature — is **learned adaptively** by path
  sampling (thermodynamic integration) with a Pareto-k̂ stopping rule. **[PAPER §2.1–2.2]**
- **Does it give discovery + tiny-mode draining?** **Discovery: yes** — it
  demonstrably populates all 10 modes of a 100-D 10-component mixture in correct
  (equal) proportions, where plain HMC fails **[PAPER §4.2]**. **Tiny-mode weight:
  in principle yes** — because it estimates a *continuous* z(β) and reweights, the
  cold (β=1) marginal carries the *true* between-mode mass, not a 1/n quantization;
  this is exactly the capability our ensemble hops lack **[CAROUSEL C-16]**. But see
  the hard caveat in §2.
- **Does it fit our UNADJUSTED MCLMC kernel?** **Mostly yes, with one real
  subtlety.** It wraps the base sampler as a black box that only needs `log q − log
  c` and its gradient — MCLMC provides both. The augmented target is a *standard
  density*, not a modified Hamiltonian, so MCLMC samples it natively. The swap/
  detailed-balance machinery of PT is **gone** (no Metropolis swaps), which actually
  *removes* a reliance MCLMC can't satisfy. **The one thing it needs that bare MCLMC
  doesn't expose is the temperature dynamics of the augmented coordinate `a`**, plus
  the log-density values along the trajectory (MCLMC does evaluate these). See §2/§3.
- **The load-bearing risk for us (Appendix C):** the method **fails for sampling**
  when the log-likelihood scale is large and the base↔target KL is huge — *exactly*
  the LDA failure mode, and our carousel lives in that regime (logp ≈ −1.2×10⁵
  minimal / −2.9×10⁵ full). With a prior base distribution this method is predicted
  to **fail the same way** unless we use a tight, MAP-centred base. This is the
  single most important finding in this report.

---

## 1. The method, precisely (so it can be implemented)

### 1.1 General framework — adaptive path sampling **[PAPER §2.1]**

Problem: an unnormalized density `q(θ; λ)` augmented by a scalar `λ ∈ Λ` (for us λ
will be the inverse temperature β); we want the normalizing-constant function
`z(λ) = ∫ q(θ; λ) dθ` (Eq. 1), or its ratio `z̃(λ) = z(λ)/z(λ₀)` (Eq. 2).

Reparameterize λ through a **link function** `λ = f(a)`, `f: A → Λ`,
continuously differentiable, with flat plateaus so that an interval of `a` maps
*exactly* to λ=1 (giving exact target draws) and another to λ=0 (exact base draws).
The general algorithm iterates four steps:

**Step 1 — Joint sampling with invariant conditionals (Eq. 5–6).** Sample S joint
draws `(θ_i, a_i)` from
> `p(θ, a) ∝ (1/c(λ)) · q(θ; λ)`, with `λ = f(a)`,   **(Eq. 5)**

where `c(λ)` is a **parametric pseudo-prior** (the learned weights), parameterized
in log space by regression coefficients over fixed kernels {φ_j}:
> `log c(λ) = β_c0·λ + Σ_{j=1}^{I} β_cj·φ_j(λ)`.   **(Eq. 6)**

Initialize `log c ≡ 0` (c ≡ 1). **Whatever c(λ) is, the conditional θ|a ∝ q(θ; f(a))
is left invariant** — c only changes the *marginal over a (temperature)*. The joint
sampler is "typically dynamic HMC in Stan, which only requires the unnormalized log
density `log q(θ; λ) − log c(λ)` as input." **[PAPER §2.1, Step 1]** ← *this is the
whole black-box contract with the base kernel.*

**Step 2 — Estimate log z by thermodynamic integration (Eq. 7–9).** The TI identity:
> `d/da log z(f(a)) = E_{θ|f(a)}[ ∂/∂a log q(θ; f(a)) ]`.   **(Eq. 7)**

Rank all draws by their `a` coordinate `a_(1) < … < a_(s)`. The **pointwise gradient**
at a distinct value `a_(i)` is the average over draws sharing that `a` (Eq. 8) — when
ties are absent, this is a *one-Monte-Carlo-draw* estimate of the inner expectation
(stochastic approximation). Then **integrate by the trapezoidal rule** with reference
z(f(0)) (Eq. 9):
> `log [z(f(a*)) / z(f(0))] ≈ ½(a_(1)−0)(U_(1)+U_0) + ½ Σ_{j=1}^{i*−1}(a_(j+1)−a_(j))(U_(j+1)+U_(j)) + ½(a*−a_(i*))(U_(i*)+U_{a*})`   **(Eq. 9)**

with U at the ends obtained by extrapolation. **[PAPER §2.1, Step 2]**

**Step 3 — Parametric regularization + adaptive update (Eq. 10).** Smooth the noisy
pointwise log z̃ by least-squares regression onto the kernel basis on a uniform grid
{λ̃_i = i/I}:
> `β̂_z = argmin_β Σ_i ( log z(λ̃_i) − (β_0 λ̃_i + Σ_j β_j φ_j(λ̃_i)) )²`   **(Eq. 10)**

Then **update the pseudo-prior `log c := log ẑ`** (i.e. `c(λ) := z(λ)`). **[PAPER §2.1, Step 3]**

**Step 4 — Diagnostic / stopping / mixing.** The marginal of `a` under Eq. 5 is
`p(a) = z(λ)/c(λ)`. If z were exact, one update `c := z` makes the `a`-marginal
**uniform** — that is the convergence target. Estimate `p(a)` itself by the same TI
formula with a *modified* gradient that subtracts the pseudo-prior (Eq. 11):
> `U_p,(i) = mean over draws at a_(i) of  ∂/∂a [ log q(θ; f(a)) − log c(f(a)) ]`.   **(Eq. 11)**

**Stopping rule:** treat the importance ratio `r_i = 1/p(a_i)` (from joint proposal
`c⁻¹q` to target `z⁻¹q`, which has a uniform a-marginal) as an importance-sampling
problem, fit a **generalized Pareto** to its right tail, and **stop when the Pareto
shape k̂ < 0.7** (PSIS diagnostic, Vehtari et al.). k̂ measures the Rényi divergence
between sampled and uniform-a target; a smaller threshold is more conservative.
**[PAPER §2.1, Step 4]** Crucially, **TI (Eq. 9) is unbiased for log z under *any*
sampling distribution as long as θ|a is invariant**, so **all draws from all past
adaptations are pooled** ("remixing") into the z-estimate each round. **[PAPER §2.1, Step 4]**

### 1.2 Continuous tempering specialization (our case) **[PAPER §2.2, Algorithm 1]**

Target `q(θ) := p(θ, y)` (the unnormalized posterior). Bridge target to a base
density π(θ) by a **geometric path**:
> `p(θ | λ) = (1/z(λ)) · q(θ)^λ · π(θ)^{1−λ}`,  `z(λ) = ∫ π(θ)^{1−λ} q(θ)^λ dθ`.

π is a proper, easy-to-sample base — "typically a simple initial guess or the prior."
λ=1 → target, λ=0 → base; smaller λ "flattens" the target. **[PAPER §2.2]**

The **link function `λ = f(a)`** (Eq. 17, Appendix A.2) is a piecewise cubic on
`a ∈ [0, 2]`, *symmetric* `f(a)=f(2−a)`, flat at both ends:
> `f(a) = 0` for `0 ≤ a < a_min`;
> `f(a) = −2u³ + 3u²` with `u = (a−a_min)/(a_max−a_min)` for `a_min ≤ a < a_max`;
> `f(a) = 1` for `a_max ≤ a < 2−a_max`;
> mirror cubic for `2−a_max ≤ a < 2−a_min`; `f(a)=0` for `2−a_min ≤ a ≤ 2`.
> Defaults **a_min = 0.1, a_max = 0.8**. **[PAPER Appendix A.2, Eq. 17]**

So one `a`-trajectory 0→2 is a full **cooling-then-heating tour** of β, with the flat
plateau (a∈[0.8,1.2], i.e. β=1) producing **exact target draws** and a continuous,
gradient-guided path elsewhere. Joint density actually sampled:
> `p(θ, a) ∝ (1/c(f(a))) · q(θ)^{f(a)} · π(θ)^{1−f(a)}`,  `a ∈ [0,2]`. **[PAPER §2.2]**

Because f is symmetric, **flip all a_s > 1 to 2−a_s** before the log z estimate (Eq.
9). With the symmetric link the pointwise TI gradient (Eq. 8) **simplifies** to
> `U_(i) = mean over draws at a_(i) of  f′(a_(i)) · ( log q(θ_j) − log π(θ_j) )`,

i.e. **if the base π is the model prior, U = f′(a) × (log-likelihood)**. **[PAPER §2.2]**
**Output:** after k̂<0.7, keep `{θ_i : f(a_i)=1}` as the posterior draws; `log z(1)`
is the **log marginal likelihood** when π=prior. **[PAPER §2.2, Algorithm 1]**

**Prior on the a-marginal (Appendix A.5).** Default target is uniform-a (set c:=z),
"conservative…ensures the chain has explored the whole temperature space." Optional
efficiency-optimal a-priors (Jeffreys-type, minimize Var(log z); or constant-KL-gap
`p_opt(a) ∝ (1/f′(a))·√Var_{θ|a}(log π − log q)`) exist but need extra adaptation;
the authors **default to uniform for robustness**. **[PAPER Appendix A.5, Prop 3]**

**Regression kernels (Appendix A.6).** Gaussian + logit kernels, J=10 points, during
adaptation; cubic spline smoothing for the final z estimate. Kernel choice found
**not** to matter much. This is *function approximation*, not KDE. **[PAPER Appendix A.6]**

### 1.3 What is wrapped around the base kernel

The base sampler (HMC in Stan) is used **only** in Step 1: sample the joint `(θ, a)`
from `log q(θ; f(a)) − log c(f(a))`. Everything else (TI integral, regression,
Pareto-k̂, c-update) is host-side post-processing on the draws. **There is no
accept/reject swap, no replica ladder, no Metropolization** — the augmented system
is one ordinary density and the base kernel just samples it. **[PAPER §2.1–2.2]**
*(Algorithm 2 / §2.3 instead adds a bias `log(p_targ(γ)/p(γ))` to a problematic
marginal γ to fight entropic/funnel barriers — irrelevant to our energetic
between-basin barrier; noted for completeness, not used.)*

### 1.4 Relationship to PT / AIS / discrete simulated tempering **[PAPER §3]**

The paper positions continuous path sampling as the **K→∞ dense-ladder limit** of
discrete simulated tempering, annealed importance sampling, and bridge sampling
(Props 2, A.3–A.4). Key scaling argument **[PAPER §3]**: discrete tempering needs the
number of rungs K to grow with dimension — best-case `K = O(d^{1/2})`, in practice
`O(d)` (Woodard 2009), and the temperature random walk relaxation scales `O(K²)` —
"soon unaffordable as K grows." Continuous tempering removes the ladder entirely.
**This is the precise sense in which it beats the R× replica cost.** Graham–Storkey
(2017) continuous tempering is shown to be the special case where z(λ)=λ^β (single
log-linear coefficient); the adaptive parametric z(λ) generalizes it.

---

## 2. Utility for our case (honest)

### 2.1 The R× scaling problem — SOLVED in principle ✔

PT (`parallel_tempering.py`) pays R× by running R replicas every round. Continuous
tempering runs **one** augmented chain (per walker); temperature is a *sampled
coordinate*, not a parallel ladder. Cost per adaptation is "one joint HMC sample"
**[PAPER §2.1, Appendix C]**, and the paper explicitly markets this against discrete
tempering's `O(d)` rungs / `O(K²)` relaxation **[PAPER §3]**. For us this is the
right structural answer to **Gate E (cost)**: no R× replicas, only an
augmentation-by-one-dimension + an adaptation loop. **[OWN, from PAPER §3]**

### 2.2 Discovery — YES, demonstrated ✔ (in the paper's regime)

The 100-D, 10-component separated-Gaussian-mixture experiment populates **all**
modes in correct equal proportions, with mode coverage **not degrading from d=10 to
d=100**, and faster than Rao-Blackwellized discrete tempering **[PAPER §4.2, Fig 8]**.
The flower target (40 petals) mixes where plain HMC visibly fails **[PAPER §4.3–4.4,
Fig 9–10]**. Mechanistically this is the *same* gradient-on-flattened-target crossing
that our `tempered_mclmc.py` uses and that beat the affine ensemble hops on the curved
testbed **[CAROUSEL C-18, Gate D]** — so discovery is expected to carry over. **[OWN]**

### 2.3 Tiny-mode draining + exact comparable-mass weight — YES in principle, and this
is the *advantage over our one-shot tempered burn-in* ✔ (with the §2.5 caveat)

Our tempered-*burn-in* (no replicas) **discovers but freezes the between-mode weight**
at a k/n quantization (`drill_schedule.py`: 0.51 vs truth 0.70) because a frozen
integer-chain cold ensemble cannot represent a continuous weight; we needed full **PT
(R×)** to fix the weight and drain the 1e-3/1e-5 modes **[CAROUSEL C-18, Gates A/B;
NOTES.md]**. Continuous tempering **gets the weight from a different place**: it
*estimates z(β) continuously* and the cold marginal is reweighted by the learned
pseudo-prior, so the comparable-mass weight and the tiny-mode mass come out of the
**path-sampling z-estimate + the time the single chain spends at β=1**, not out of an
integer occupancy. This is precisely the "covering a mode independent of current
occupancy" capability our notebook identified as the missing discovery-class lever
**[CAROUSEL C-16]**. So on paper it delivers PT's robustness (weight + tiny-mode)
**without R× replicas** — the exact thing the task is asking for. **[OWN, inference
from PAPER §2.1–2.2 + CAROUSEL C-16/C-18]**

### 2.4 Wrapping our UNADJUSTED MCLMC kernel

What the method asks of the base kernel, and whether MCLMC supplies it:

| Requirement of the method | HMC (paper) | Our MCLMC | Verdict |
|---|---|---|---|
| Sample θ from `log q − log c` (unnormalized) | ✔ | ✔ — MCLMC needs only `log p` + grad; `β·log q + (1−β)·log π − log c` is just another log-density (the existing `tempered_mclmc._build_kernel` already builds `β·log p`) | **Fits** |
| **No** Metropolis accept / detailed balance / swaps | n/a (TI is the engine) | MCLMC has no MH — **and the method needs none** (no PT swaps, no AIS reweight-on-accept) | **Better fit than PT** |
| Evaluate `log q(θ)` and `log π(θ)` at draws (for U in Eq. 8/9) | ✔ | ✔ — log-density evaluations are available; for π=prior, U=f′·log-likelihood, also available | **Fits** |
| Move in the **temperature coordinate a** (Step 1 samples `(θ,a)` *jointly*) | HMC moves a as an extra continuous dim with its own gradient `∂/∂a[…]` | **This is the open question** — see below | **Needs design** |

**The one genuine adaptation point.** In Stan/HMC, `a` is just another continuous
parameter and HMC propagates it with the joint gradient `∂/∂a [β·log q + (1−β)·log
π − log c]` = `f′(a)·(log q − log π) − (d/da)log c`. To wrap MCLMC faithfully we have
two choices **[OWN]**:
  - **(A) Augmented MCLMC** — append `a` to the state vector and let the *same* MCLMC
    isokinetic dynamics carry it, with the augmented log-density above. This is the
    most literal port and keeps a single kernel; `a`'s gradient is cheap (it's `f′(a)
    × loglik − c′`). Risk: `a` and θ have very different scales/curvature, so the
    shared MCLMC step/mass-matrix must accommodate the extra dim (the c-adaptation is
    designed to make the a-marginal uniform, which *helps* conditioning). **Preferred.**
  - **(B) Gibbs-style** — alternate MCLMC-on-θ-at-fixed-β with a separate
    random-walk/Metropolis update of `a` using the learned `c`. This reintroduces an
    MH step *on the scalar a only* (cheap, 1-D, and exact — not on θ), which MCLMC's
    lack of MH does not preclude (we'd be Metropolizing the auxiliary, not the
    kernel). Closer to *discrete* simulated tempering; loses the "gradient-guided
    subtle jumps near β≈0" advantage the paper highlights **[PAPER §4.1, Fig 5]**.

**No reliance on anything MCLMC lacks.** The method never uses an MH acceptance on θ,
never uses detailed balance for swaps, and uses only log-density/gradient evaluations
that MCLMC provides. The only "new" primitive is propagating the scalar `a`, handled
by (A) or (B). **[OWN]**

### 2.5 Where it might NOT fit — the load-bearing caveat (Appendix C) ⚠⚠

**Appendix C is a documented failure mode and it is *our* regime.** On a high-D LDA
posterior the method **fails for sampling**: the chain explores only a thin slice of
`a`, fails Pareto-k̂, and gets pinned at one temperature. The mechanism **[PAPER
Appendix C, Fig 17–18]**:
- `log z(β)` enters the joint density **additively**, so what matters is the
  **absolute** error of the z-estimate, not its relative error.
- TI estimates the pointwise gradient with **one Monte-Carlo draw** of the
  log-likelihood; that draw has variation ~10³ when the log-likelihood scale is ~10⁵.
- A **1% relative** error in `log z` at a single temperature = `exp(0.01·10⁴)=exp(100)
  ≈ 10⁴³` bump in the a-marginal → effectively a point mass → the chain cannot move in
  temperature. Quote: *"if at one single temperature point a₀ we add a 1% noise … we
  will create an exp(100)=10⁴³ bump in the marginal density p(a): essentially a point
  mass."* **[PAPER Appendix C]**
- General lessons the authors draw: *"tempering imposes dimensional limitations. The
  log likelihood scales linearly with the data input… A generic prior-posterior
  geometric path will essentially fail when we add more and more data."* and *"a weak
  prediction model will amplify the log likelihood explosion in the prior-posterior
  path."* **[PAPER Appendix C]**

**Why this is alarming for the carousel.** Our log-posterior is **−1.2×10⁵ (minimal)
to −2.9×10⁵ (full)** **[CAROUSEL C-9, C-1]**. The lensing likelihood dominates and the
prior↔posterior KL is enormous (same structure as LDA: large data, strong likelihood,
the prior is a *terrible* base). A prior-based geometric path is predicted to land in
**exactly** the LDA failure mode: TI can't pin `log z(β)` to the absolute accuracy
(≪1 in units of nats, i.e. relative error ≪ 10⁻⁵) needed for the temperature to mix.
**[OWN, by direct analogy to PAPER Appendix C using CAROUSEL logp scale]**

**The remedy the paper gives, and what it implies for us.** *"One remedy here is to
start with a better constructed base measurement, such that the log normalizing
constant will be smaller."* **[PAPER Appendix C]** For us that means **do not use the
prior as base** — use a **tight base centred on the MAP** (a Laplace/Gaussian
approximation at the global-basin MAP), so the base↔target KL and the span of log z(β)
collapse from ~10⁵ to O(d). This is *compatible* with our existing finding that a
good global-basin MAP already nearly solves the carousel **[CAROUSEL C-9]**, but it
also means the method's benefit narrows: with a MAP-Gaussian base, continuous
tempering becomes a *local* multi-basin reconciler, not a from-the-prior global
explorer. **Honest bottom line: this method is more fragile on the carousel's
likelihood scale than our gradient-only tempered burn-in, and its z-estimation step
is the fragile part.** **[OWN]**

### 2.6 EEVPD / step-tuning interaction (our non-negotiable)

The paper tunes HMC by Stan's NUTS adaptation; **it says nothing about EEVPD** because
HMC self-tunes via accept-rate. For MCLMC we must keep our discipline **[CAROUSEL
C-15/C-18; NOTES.md]**: the step must be **energy-variance-tuned (EEVPD =
mean(ΔE²)/D = 5e-4), never hand-set**, and on a tempered path the step must satisfy
the **anneal-max EEVPD**, not the β=1 equilibrium EEVPD. In *continuous* tempering the
chain visits **all β in [0,1] within a single trajectory** (the a-tour), so the step
must be faithful at the **worst β it visits** — concretely, tune EEVPD over a *grid of
β* (reusing `tune_step_eevpd`) and take the min step, or use the existing argument that
the β=1 step is conservative for β<1 **only at equilibrium** and verify the realized
EEVPD along the whole a-trajectory stays ≤5e-4 (this is the `curved_discovery.py`
faithfulness sweep, generalized to the continuous coordinate). A coarse step would
fake crossings via numerical heating exactly as flagged in `curved_discovery.py`. **[OWN
+ CAROUSEL]**

---

## 3. Proposed implementation for us

A standalone module mirroring `tempering/{tempered_mclmc, parallel_tempering}.py`:
imports the **real** MCLMC kernel read-only from
`gigalens_research.inference.blackjax_updated_utils`
(`_build_kernel_shardmap`, `isokinetic_mclachlan_smart`, `_single_init`), touches no
shared module, CPU-testbed first. Suggested file:
`tempering/adaptive_path_tempering.py`.

### 3.1 State and configuration

```
State (host-side, numpy where adaptation lives; jax for the kernel scan):
  positions_theta : (n_walkers, D)          # θ
  a               : (n_walkers,)            # continuous temperature coord in [0,2]
  beta_coeffs     : (1+J,) regression coeffs for log c(λ)   # the learned pseudo-prior
  draw_archive    : list of (theta, a, loglik, logprior) over ALL adaptations  # for pooled TI
Config:
  base_logprob_fn(theta)   -> log π(θ)      # MAP-CENTRED GAUSSIAN base, NOT the prior (see §2.5)
  target_logprob_fn(theta) -> log q(θ)      # the carousel logp (read-only)
  link f(a), f'(a)         # Eq.17 cubic, a_min=0.1, a_max=0.8
  kernels {φ_j}            # Gaussian+logit, J=10  (Eq.6/A.6)
  L, step_size             # EEVPD-tuned over a β-grid (anneal-max), see §2.6
  pareto_k_threshold = 0.7
```

### 3.2 Functions (sketch)

```
def make_apt_sampler(target_logprob_fn, base_logprob_fn, D, n_walkers, link, kernels,
                     L, step_size, imm=None, integrator=isokinetic_mclachlan_smart):
    # joint log-density at fixed coeffs:
    #   logc(a)   = beta_coeffs · [f(a), φ_1(f(a)), ..., φ_J(f(a))]
    #   logjoint(theta,a) = f(a)*logq(theta) + (1-f(a))*logpi(theta) - logc(a)
    # OPTION A (preferred): augment state z=(theta,a); build ONE real MCLMC kernel on
    #   logjoint via _build_kernel_shardmap(logdensity_fn=logjoint_aug, imm=block(imm,1)).
    #   a is carried by the same isokinetic dynamics; grad_a = f'(a)*(logq-logpi)-c'(a).
    # OPTION B: alternate scans[β]-style MCLMC-on-theta (reuse tempered_mclmc kernel
    #   builder at the current f(a)) with a cheap 1-D Metropolis move on a using logc.

    def joint_scan(states, keys): ...        # jitted K-step MCLMC scan (as in tempered_mclmc)

    def thermodynamic_integrate(archive):
        # rank draws by a; FLIP a>1 -> 2-a (symmetric link); pointwise
        #   U_(i) = mean_{draws at a_(i)} f'(a_(i)) * (logq - logpi)          # Eq.8 simplified
        # trapezoidal cumulative integral -> log z(λ_grid)                    # Eq.9
        return lz_grid

    def regress_logc(lz_grid):
        # least squares (Eq.10) of log z onto [λ, φ_1..φ_J]  -> beta_coeffs   # Eq.6/Eq.10
        return beta_coeffs

    def marginal_a_and_paretok(archive, beta_coeffs):
        # estimate p(a) via TI with gradient (Eq.11) including -log c;
        # r_i = 1/p(a_i); fit GPD tail -> k_hat                               # Step 4
        return k_hat

    def adapt_loop(init_theta, key, n_adapt, S_per_adapt, K):
        coeffs = zeros(1+J)                       # log c ≡ 0 init
        archive = []
        for it in range(n_adapt):
            # Step 1: joint sample (theta,a) at current coeffs, S draws
            states = init_states(theta, a, key); states,trace = joint_scan(...)
            archive += extract(theta,a,logq,logpi)
            # Step 2-3: TI over POOLED archive -> log z -> regress -> coeffs
            lz   = thermodynamic_integrate(archive)
            coeffs = regress_logc(lz)
            # Step 4: diagnostic + stop
            if marginal_a_and_paretok(archive, coeffs) < 0.7: break
        cold = [theta for (theta,a,...) in last_adapt if f(a)==1]   # f(a_i)=1 plateau
        return cold, coeffs, lz                  # lz(1) = log marginal likelihood

    return dict(adapt_loop=..., joint_scan=..., thermodynamic_integrate=...,
                regress_logc=..., diagnostics=..., config=...)
```

Notes: **(i)** the EEVPD tuner from `tempered_mclmc.py` is reused but swept over a
β-grid and the **anneal-max** step is taken (§2.6). **(ii)** Keep π a **MAP-Gaussian**
by default and make "π=prior" an explicit opt-in that *raises a warning* given the
Appendix-C scale risk (consistent with `docs/project-standards.md` "no silent
scientific defaults"). **(iii)** The TI/regression/Pareto-k are pure NumPy host-side,
exactly like `parallel_tempering.py`'s host-side swap sweep. **(iv)** Start with
Option A (single augmented kernel) — simplest, one kernel, no MH; fall back to B only
if a-mixing is poor.

---

## 4. Test plan (mapped onto our existing gates)

Reuse the existing testbeds verbatim: analytic mixture from `validate_analytic.py`
(`logdensity_fn`, D=10, w=0.7/0.3, m=5), curved bimodal from
`sa_mcmc/curved_testbed.CurvedBimodal` (GATE-1 ≈0.6–1.5% linear-DE acceptance), and
the harness style of `curved_discovery.py` / `pt_drain.py` / `pt_weight.py`. **Add one
new mandatory gate (F) for the Appendix-C scale risk**, since that is the predicted
failure point for the carousel.

| Gate | What it checks | Concrete test (mirroring existing) | Pass criterion (pre-registered) | Contrast / null |
|---|---|---|---|---|
| **A — Discovery** | wrong-basin init → reach dominant basin | All walkers init in the **minor** basin of `validate_analytic` mixture; run `adapt_loop`; cold = f(a)=1 draws. Mirror `curved_discovery.py` init. | cold occ(dominant) > 0.5 and ≫ vanilla | vanilla MCLMC from same init stays ~0 (it's trapped, C-10) |
| **B — Tiny-mode drain** | seed 1e-3/1e-5 mode → cold drains to truth, not 1/n | Reproduce `pt_drain.py`: seed a 1e-3 and a 1e-5 mode; run adaptation; measure cold occ(minor). | occ→ truth within block-bootstrap 3·SE (1e-3→~0.00125; 1e-5→~0) — **NOT** pinned at 1/n_walkers | every ensemble hop pins at 1/n (C-16); one-shot tempered burn-in freezes (C-18) |
| **C — Unbiased cold** | invariance-from-truth | `validate_analytic` V2: init at **exact** mixture draws, run, check no drift; per-mode moments within 0.10; axis-0/axis-1 **KS p>0.05** (thinned); weight \|est−0.70\|<3·SE. | all of V2(a–d) pass | a biased kernel pulls weight/moments off truth |
| **D — Curved-barrier crossing** | follows curved ridge a chord can't | `curved_discovery.py` on `CurvedBimodal` (GATE-1 0.5–1.5%); all-minor init → cold occ_A. | occ_A>0.5 at a **faithful** step | affine DE = 0 round-trips (C-12/C-14); vanilla ~0 |
| **E — Cost vs PT** | the headline claim | Count base-kernel log-density/grad evals to reach a target ESS/weight accuracy, vs `parallel_tempering.py` at R replicas for the same accuracy. | continuous-tempering cost ≪ R× PT cost (no replica multiplier; one augmented chain + adaptation) | PT pays R× per round (NOTES.md) |
| **F — Scale / Appendix-C (NEW, mandatory)** | does z-estimation survive a 10⁵-scale log-likelihood? | Build a mixture target whose log-likelihood scale is inflated to ~10⁵ (carousel-like). Run with (i) **prior base** and (ii) **MAP-Gaussian base**. Monitor Pareto-k̂ and the a-marginal histogram (Fig-17 style). | With prior base: **expected FAIL** (a-marginal collapses, k̂≫0.7) — this *reproduces* the predicted failure and bounds applicability. With MAP base: k̂<0.7, a-marginal ≈ uniform, Gates A–D still pass. | n/a — this gate exists to falsify the method honestly before GPU |

**EEVPD requirement on every gate (non-negotiable, C-15/C-18):** the step must be
**EEVPD-tuned (mean(ΔE²)/D=5e-4)**, and because the continuous chain visits all β in
one a-tour, tuned to the **anneal-max** EEVPD over a β-grid — never hand-set. Record
the realized EEVPD along the whole a-trajectory (generalize `curved_discovery.py`'s
faithfulness sweep); a coarse step that "discovers" via numerical heating is a
**fail**, not a pass. **Plots before metrics** on every gate (a-marginal histogram +
log z(β) curve + cold corner), per method discipline.

**Ordering / discipline.** Run **Gate F first** — it is the cheap falsifier of the
whole approach for the carousel. If F shows the prior-base path collapses at 10⁵
scale (as predicted), the verdict is *"viable only with a MAP-Gaussian base"* before
any GPU spend. Then A–E on the CPU testbeds. **Then GPU-validate on the real carousel
— mandatory, per the C-17 cautionary tale (SA won on CPU, froze on GPU).** Single-seed
CPU success is *not* belief.

---

## 5. Public Stan implementation **[REPO — fetched live + PAPER Appendix B]**

- **Repo:** `https://github.com/yao-yl/path-tempering` (verified live). R package
  **`pathtemp`**, runs inside **Stan**. *(I fetched the README; quotes below are from
  it and from the paper's Appendix B.)*
- **Install:**
  ```r
  devtools::install_github("yao-yl/path-tempering/package/pathtemp", upgrade="never")
  ```
- **Model spec — the `alternative model` block** (the base/π is written like an
  ordinary model):
  ```stan
  model{                    // the original target
    y ~ cauchy(theta, 0.2);
    -y ~ cauchy(theta, 0.2);
  }
  alternative model{        // the base measure (e.g. the prior)
    theta ~ normal(0, 5);
  }
  ```
- **Augment + sample (the two-line workflow):**
  ```r
  library(pathtemp)
  update_model <- stan_model("solve_tempering.stan")
  file_new     <- code_temperature_augment("cauchy.stan")   # -> cauchy_augmented.stan
  sampling_model   <- stan_model(file_new)
  path_sample_fit  <- path_sample(data = list(...), sampling_model = sampling_model)
  ```
- **Documented `path_sample()` arguments** (README): `sampling_model`, `data`,
  `N_loop` (number of adaptation loops), `visualize_progress`, `iter_final` (final
  iteration count). `code_temperature_augment()` primary arg `stan_file`. **[REPO]**
- **Outputs** (paper Appendix B + README): posterior draws from target
  (`lambda==1`) and base (`lambda==0`), the joint `(a, θ)` path of the final
  adaptation, and `log z(λ)` (the log marginal likelihood when base=prior). Extracted
  via `extract(path_sample_fit$fit_main)`; `in_target <- sim$lambda==1`. **[PAPER App B]**

**Transferability of the public code.** It is **Stan/HMC-specific** (it rewrites the
Stan program and drives NUTS); we cannot reuse the code directly. **But the
algorithm** — augment with `a`, geometric path via the Eq.17 link, TI z-estimate (Eq.
8–9), kernel regression (Eq.6/10), Pareto-k̂<0.7 stop, keep f(a)=1 draws — **is fully
specified in the paper and §3 above**, and is what we port onto MCLMC. The README is a
useful reference for the adaptation-loop control flow (`N_loop`, `iter_final`,
`visualize_progress`).

---

## 6. Verdict and confidence

- **Method:** adaptive continuous simulated tempering with path sampling (Yao et al.
  2020, Algorithm 1). **[PAPER, verified]**
- **R× scaling:** **solves it** — one augmented chain with a continuous temperature
  coordinate replaces the R-replica ladder; the per-temperature weights z(β) that a
  single tempered chain normally lacks are learned adaptively by TI. **High
  confidence** this is the right structural answer to Gate E. **[PAPER §3]**
- **Fits MCLMC:** **yes, with one design point** — the augmented `a` coordinate must
  be carried by MCLMC (Option A: append to the state; Option B: 1-D MH on `a` only).
  Crucially it needs **no** Metropolis accept on θ and **no** detailed-balance swaps
  (it uses only log-density/grad evals MCLMC provides), so it is a *better* structural
  fit than PT's swaps. **Medium-high confidence.**
- **The real risk (must test first):** Appendix C is our regime — at the carousel's
  ~10⁵ log-likelihood scale, a **prior** base is predicted to make the z-estimate
  fail (point-mass a-marginal), exactly as it failed on LDA. **A MAP-Gaussian base is
  the required remedy**, which narrows the method to local multi-basin reconciliation
  rather than from-the-prior global discovery. **This is the highest-value thing to
  falsify before any GPU run (new Gate F).** **Medium confidence it transfers; high
  confidence the prior-base variant will fail at our scale.**
- **Overall:** worth prototyping as `tempering/adaptive_path_tempering.py` with a
  MAP-Gaussian base, gated by F→A–E on CPU and then **mandatory GPU validation**
  (C-17 lesson). It is the first candidate that targets PT's robustness *without* the
  R× cost; its fragility is well-characterized and testable up front.

---

*All claims here are PROPOSED / UNCERTIFIED for orchestrator grading. Paper-verified
items are tagged [PAPER §location]; repo items [REPO]; my inferences [OWN]; lab-notebook
facts [CAROUSEL C-n]. No citation, equation, or API was fabricated; equations were
hand-transcribed from a zlib-decompressed text layer (no PDF renderer on HPC) and any
reconstruction uncertainty is flagged in §1.*
