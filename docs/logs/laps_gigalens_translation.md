# Translating LAPS into a production GIGALens sampler — design doc

**Scope.** How to build a paper-faithful LAPS (Late-Adjusted Parallel Sampler,
Robnik & Seljak 2026) as a GIGALens sampler. This is a *design* decision record,
not sampler code. It is derived **only** from the canonical LAPS spec
(`docs/logs/laps_spec.md`) and the MCLMC translation patterns
(`docs/logs/mclmc-translation-patterns.md` + `src/gigalens_research/inference/mclmc.py`).
The blackjax `laps`, `laps.py`, and `laps-COLDSTART3.py` were deliberately **not**
consulted (audited separately).

---

## 1. Build-fork recommendation

**Recommendation: Option B — reimplement LAPS by composing the in-tree MCLMC
reusable APIs. Do not thin-wrap blackjax `adaptation.laps`.**

### The two options

- **(A) Thin-wrap blackjax `adaptation.laps`** (adapt I/O only, as
  `laps_blackjax.py` attempted): take the stock blackjax LAPS adaptation, feed it
  the gigalens `log_prob`, and marshal inputs/outputs.
- **(B) Reimplement LAPS** by composing the gigalens-hardened building blocks
  already shipped for MCLMC: `_build_kernel_shardmap` (Phase-1 unadjusted MCLMC),
  `_build_adjusted_kernel_shardmap` (Phase-2 MAMS), the EEVPD `step_size_adapt`
  controller, the dense-metric isokinetic integrators, `welford_combine`,
  `_ess_shardmap`, dtype-safe `init_multi`/`_single_init`, and the whole
  shard_map / VMA / JAX-0.10 safety layer.

### Rationale (weighed on the required axes)

1. **Faithfulness to the paper.** This is decisive. The spec's headline finding
   (§C rows 1–2) is that blackjax `laps` does **not** implement the paper's
   Phase-1 step law: the paper sets `EEVPD_wanted = F(C·D̃)` with
   `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²`, `C=0.025`; blackjax computes `C·D̃^{3/8}`,
   `C=0.1`, from a *different* (EMAUS) paper. The exponent 3/8 lies outside F's
   range [1/2, 3/2] — it is not even an approximation of F. It also uses a
   10×-looser switch threshold (δ vs δ²) on the wrong observable (`x_i` vs
   `x_i²`). Option A would ship the EMAUS variant as our default and require us to
   *patch the upstream adaptation* to recover the paper — i.e. we end up
   reimplementing the load-bearing parts anyway, but inside someone else's
   control flow. Option B makes the paper the default and the EMAUS rule an
   explicit A/B switch, which is exactly the disposition the spec recommends (§E.5).

2. **Reuse of gigalens-hardened layers.** The expensive, already-paid translation
   tax — shard_map sharding for M chains across devices, the VMA / `while_loop`
   landmines (igamma, `effective_sample_size`, the kernel NaN branch), the
   single-dtype `_canon` invariant against float32-qz/float64-energy mixing, and
   JAX-0.10 reshard/gather strictness — lives in `_build_*_shardmap`,
   `_ess_shardmap`, and `full_mclmc_with_adapt_sharded`. Stock blackjax `laps`
   has **none** of this; a thin wrap would reintroduce every landmine the MCLMC
   port already cleared. Option B inherits all of it for free because LAPS's two
   kernels *are* the two kernels we already sharded.

3. **The EEVPD controller.** `step_size_adapt` already implements the
   `ε ∝ EEVPD^{1/6}` update with `handle_nans` step-size clamping — LAPS's exact
   1/6 exponent. The only structural change LAPS needs is a **dynamic** target
   (`desired_energy_var → F(C·D̃_t)` recomputed each step) instead of a static
   scalar. That is a one-argument generalization in Option B; in Option A it is a
   reach into upstream internals.

4. **Maintainability & risk.** The novel LAPS pieces not already in-tree
   (equipartition `D̃`, the `F(C·D̃)` target, the `x_i²` switch, target-accept
   bisection) are all **ensemble-level reductions and scalar control**, not
   kernel-level numerics. They compose cleanly on top of the existing
   `psum`/`pmin('device')` reduction pattern. Option B's risk is concentrated in
   ~5 small, testable ensemble functions; Option A's risk is diffuse (tracking an
   upstream package whose default *disagrees with the paper*, plus re-clearing the
   sharding/VMA/dtype minefield). blackjax is also already a lazy, unpinned-risk
   dependency (`pipeline.py:1697`); deepening reliance on its adaptation module
   for a method we want paper-faithful is the wrong direction.

**Net:** Option B is more faithful, reuses everything costly, localizes risk to a
few ensemble reductions, and makes the paper-vs-EMAUS choice an explicit flag
rather than an upstream default we must fight. Build LAPS as a sibling driver to
`full_mclmc_with_adapt_sharded`.

---

## 2. Component → implementation mapping

| LAPS spec element | In-tree API | Disposition | Notes / gap |
|---|---|---|---|
| **Phase-1 unadjusted MCLMC kernel** (Eq. 2 dynamics) | `_build_kernel_shardmap(logdensity_fn, inverse_mass_matrix, integrator)` | **Reuse as-is** | This *is* the LAPS Phase-1 kernel. Only change: pass a **Leapfrog/velocity-verlet** integrator (paper: 1 grad/step Phase 1), not the MCLMC-default `isokinetic_mclachlan`. `_velocity_verlet_smart` exists. |
| **Phase-2 MAMS (adjusted) kernel** | `_build_adjusted_kernel_shardmap(logdensity_fn, inverse_mass_matrix, integrator)` → `kernel(rng_key, state, step_size, num_integration_steps)` | **Reuse as-is** | Pass `num_integration_steps=N=15`. Integrator: MN2 (`mclachlan`) default; **MN4 (`omelyan`) for d>200** (App. E). Partial-refresh `L=1.25·L_full` is a kernel/refresh parameter to plumb. |
| **L = α(Σ Varρ[x_i])^{1/2}, α=2** (Eq. 9) | — (in-tree L is **ESS-based**, `Lfactor·num_steps3·ss/min_ess`) | **Build new** | Different mechanism. Reuse the cross-chain Welford/`psum` pattern to get per-coord `Var[x_i]`, then `L=α·sqrt(sum Var)`. Recomputed each Phase-1 step. α=2 (not in-tree 0.4/1.9). |
| **Equipartition D̃** (Eqs. 4/6, diag Eq. 18) | — (no API computes `V_ij`) | **Build new** | `V_ij=E_ρ[−(x_i−E x_i)∂_j log p]`. Gradients are already in `IntegratorState.logdensity_grad` from the kernel. Diagonal `D̃_diag=(1/d)Σ(1−V_ii)²` via a cross-chain `psum` of `−(x_i−x̄_i)·g_i`. Hutchinson full-rank (100 Rademacher, App. B.1) optional later. |
| **EEVPD estimate** (Eq. 7) | `step_size_adapt` computes `xi=Δ²/(d·desired_energy_var)` per chain, **decayed running avg** | **Adapt** | LAPS wants `EEVPD=Var_ρ[Δ]/d` as an **ensemble-instantaneous** cross-chain variance of the integrator energy error Δ, not a per-chain decayed estimate. Reuse `info.energy_change` (=Δ, already NaN-zeroed) but reduce with `psum` cross-chain Var. |
| **Step law `EEVPD_wanted=F(C·D̃)`** + `ε_{t+1}=ε_t(wanted/current)^{1/6}` | `step_size_adapt` has the **1/6 exponent** and `handle_nans` clamp, but a **static** `desired_energy_var` | **Adapt controller + build target** | The 1/6 update is reusable. The target must become **dynamic**: `desired_energy_var ← F(C·D̃_t)`. The in-tree controller's trust-weighted decayed Welford is *not* the paper's direct ratio update — cleanest is a **new lightweight Phase-1 update** (`ε·(wanted/current)^{1/6}`, reusing `handle_nans` for the step_size_max clamp). Provide `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²`, `C=0.025` as default; **EMAUS `C·D̃^{3/8}`, C=0.1 as an A/B flag**. |
| **Switch test on x_i²** (Eqs. 10–11) | — | **Build new** | Windowed `δ_t[f]=σ_t/μ_t` over `f=x_i²`, window `T=20%`, stop when `max_i δ≤0.01`. Needs a ring buffer of ensemble `E_ρ[x_i²]` per step and a cross-step σ/μ. Data-dependent stop → see §3 control-flow. |
| **Target-accept step bisection** (Phase 2, 70%/90%, freeze \|a−a_target\|≤0.03) | — (in-tree uses EEVPD/PSMILE controllers, no bisection) | **Build new** | Ensemble-mean acceptance via `psum` of accept prob; double/halve to bracket then bisect; freeze on 3% band. Sequential → §3. |
| **Diagonal preconditioner at phase boundary** (`y_i=x_i/Var^{1/2}`, set once) | Windowed **dense** IMM via `welford_combine` + `welford_cov` | **Adapt** | LAPS uses a **diagonal** metric set **once** at end of Phase 1 (not 3 expanding dense windows). Reuse the cross-chain Welford Var accumulation but install `diag(Var)` as `inverse_mass_matrix` at the boundary only. The kernels already accept a dense 2-D IMM, so a diagonal matrix drops in. |
| **Ensemble / M-chain handling** (M up to 4096) | Two-level batching: shard `'device'` + inner `vmap`; `psum`/`pmin('device')` | **Reuse as-is** | Identical pattern. No superchains by default (spec: out of scope; split-R̂ only if `superchain_size>1`, future work). |
| **Diagnostics** | `Hist` namedtuple; `MCLMCInfoWithExtras`/`KernelExtras`; `diagnostics_config()` | **Adapt/extend** | Add `D̃`, `EEVPD_wanted` vs observed `EEVPD`, `δ[x_i²]`, Phase-2 acceptance to the history. Reuse the `Hist`/io-callback plotting plumbing. |
| **Init from qz; unconstrained z** | `init_multi` / `_single_init`; `log_prob(z)=prob_model.log_prob(z)[0]` | **Reuse** | See §4. One open point: paper inits velocity **gradient-aligned**; `_single_init` draws a **random** unit momentum. |
| **Single-dtype / x64 safety** | `_canon` cast of state + params | **Reuse as-is** | D̃ uses gradients (energy-dtype), consistent with `_canon`. |

---

## 3. Concrete change list (relative to in-tree MCLMC machinery)

New, paper-faithful pieces to add; everything else is reuse.

1. **Equipartition estimator `D̃` (new ensemble reduction).** From the per-chain
   `IntegratorState` positions and `logdensity_grad`, compute the diagonal
   `V_ii=E_ρ[−(x_i−x̄_i) g_i]` via cross-chain mean (`psum`) and
   `D̃_diag=(1/d)Σ(1−V_ii)²`. Optional Hutchinson full-rank later. No kernel
   change — gradients already flow out of the kernel.

2. **`F(C·D̃)` target + Phase-1 step update (new + controller adapt).** Add
   `F(D̃)=4D̃^{3/2}/(1+D̃^{1/2})²`; default `EEVPD_wanted=F(C·D̃)`, `C=0.025`. Add
   an A/B flag `step_law={"paper","emaus"}` where `emaus` selects `C·D̃^{3/8}`,
   `C=0.1`. Step update `ε←ε·(EEVPD_wanted/EEVPD_obs)^{1/6}`, reusing
   `handle_nans` for the `step_size_max` clamp.

3. **Ensemble EEVPD estimator (new).** Cross-chain `Var[Δ]/d` from
   `info.energy_change`, replacing the per-chain decayed `xi` average for the
   Phase-1 target loop (keep `xi` logging for diagnostics).

4. **L law `α√ΣVar`, α=2 (new).** Cross-chain per-coord variance →
   `L=α·sqrt(Σ_i Var[x_i])`, recomputed each Phase-1 step. Replaces the
   ESS-based L during Phase 1.

5. **`x_i²` windowed switch detector (new).** Ring buffer of ensemble
   `E_ρ[x_i²]`; `δ_t[x_i²]=σ/μ` over the trailing `T=20%` window; switch when
   `max_i δ≤0.01` (or `t≥maxiter`).

6. **Phase-boundary handoff (new, sequencing logic).** At the switch: (a) install
   the **diagonal** preconditioner `diag(Varρ)` as `inverse_mass_matrix`;
   (b) swap integrator Leapfrog→MN2/MN4 (d>200); (c) swap kernel
   unadjusted→MAMS; (d) seed Phase-2 ε from the final Phase-1 ε.

7. **Phase-2 bisection step tuner (new).** Ensemble-mean acceptance (`psum`),
   double/halve bracket then bisect on `a(ε)−a_target`, freeze (stop adapting)
   when `|a−a_target|≤0.03`; targets 70% (MN2) / 90% (MN4).

8. **Control-flow decision (new, see open Q1).** Both the Phase-1 `δ` switch and
   the Phase-2 bisection are **data-dependent stops** — forbidden as
   `while_loop` under shard_map (VMA). Two viable shapes:
   (i) a **Python-level loop of fixed-length jitted scan blocks** at the ensemble
   driver level (recompiled-once, checks the scalar stop between blocks), or
   (ii) a **fixed `maxiter` scan with `where`-masked freeze** (no early exit;
   compute the full budget but stop updating once converged). (i) saves compute
   and is closer to the paper's `while`; (ii) is simplest and fully traceable.

9. **Diagnostics extension (new fields).** Add `D_tilde`, `eevpd_wanted`,
   `eevpd_obs`, `delta_xi2`, `accept_rate` to the history; extend
   `diagnostics_config()` so plotting can mark the Phase-1→2 boundary and the
   bisection freeze.

---

## 4. GIGALens-specific concerns

- **`prob_model` / `qz` interface.** Identical coupling to MCLMC:
  `log_prob(z)=prob_model.log_prob(z)[0]`, with `z` the **unconstrained**
  vector (bijectors + log-det handled inside `BackwardProbModel.log_prob`). The
  sampler never touches bijectors. Chains seed from `qz.sample((M,), seed=...)`,
  `dim=positions.shape[-1]`. (The paper inits from the **prior**; gigalens default
  `qz` is the SVI/`HessianSurrogate` surrogate — see open Q3.)

- **Unconstrained dim & integrator switch.** `dim=state.position.shape[-1]`
  drives the **MN2 vs MN4** choice at the Phase-1→2 boundary (d>200 → MN4) and
  the `init_step_size≈0.01√d` Phase-1 init.

- **Chain/device sharding for M up to ~4096.** Reuse the two-level batching
  verbatim: `num_chains` floored to a multiple of `num_devices`,
  `chains_per_device=M//num_devices`, chains on the `'device'` axis, inner `vmap`,
  all cross-chain reductions via `psum`/`pmin('device')`. The dense (or diagonal)
  IMM is replicated `P()`; at M=4096 the per-chain gradient/position arrays are
  the memory driver, not the `dim×dim` metric. All new ensemble reductions (D̃,
  EEVPD-Var, Var for L, acceptance) follow the existing local-`jnp`-then-`psum`
  shape.

- **Dtype / x64.** Reuse the `_canon` single-dtype invariant: energy/log-density
  dtype is canonical (float64 under `jax_enable_x64`, else float32); cast initial
  state, `svi_mean`, and all params. D̃ consumes `logdensity_grad` (energy dtype),
  so it is consistent by construction. x64 is enabled upstream (session/env), not
  in the sampler.

- **GPU vs CPU.** No explicit switch; `num_devices=len(jax.devices())`, works
  1..N uniformly. Same code path on a CPU testbed (just fewer/slower devices).

- **CPU testbed with a synthetic logdensity.** gigalens is not importable outside
  the Shifter container, so CPU validation substitutes a **synthetic target**: a
  closure `log_prob(z): (dim,)->scalar` (e.g. ill-conditioned Gaussian /
  Rosenbrock / Neal's funnel — known `D̃`, known second moments for the `b²<0.01`
  evaluation) plus a **fake `qz`** object exposing `.sample((M,), seed)`,
  `.mean()`, `.covariance()` (a `tfd.MultivariateNormalTriL` works). Because the
  whole gigalens coupling is the two-line `log_prob` + the qz seeding, the LAPS
  driver runs unchanged on the synthetic target — this is the unit-test/validation
  substrate for D̃, the `F(C·D̃)` law, the switch, and the bisection, independent of
  the lens model. **Per project standards: the synthetic target must be passed
  explicitly; no silent default target.**

---

## 5. Open design questions (for check-in)

1. **Phase-1/Phase-2 control flow under shard_map.** Python-loop-of-jitted-scan-
   blocks (early-exit, closer to the paper `while`) vs fixed-`maxiter` scan with
   `where`-masked freeze (simplest, full budget always run)? Affects compute cost
   and how faithfully the `δ≤0.01` / 3% stops behave.
2. **Velocity initialization.** Paper aligns initial velocity with the gradient;
   `_single_init` draws a **random** unit momentum. Adopt gradient-aligned init,
   or keep random (Phase 1 is only meant to "approach fast", arguably robust)?
3. **Chain init: paper-prior vs gigalens-qz.** Keep seeding from the SVI/Hessian
   `qz` (gigalens default, warm start) or draw from the prior to match the paper?
   A warm `qz` start may make the Phase-1 `δ` switch fire almost immediately —
   does that interact badly with the 20% window?
4. **EEVPD estimator form.** Ensemble-instantaneous `Var[Δ]/d` (paper) vs the
   in-tree per-chain decayed average (already battle-tested)? Both share the 1/6
   update; the estimator differs. Default to paper, keep decayed as fallback?
5. **Preconditioner: diagonal-once (paper) vs dense-windowed (in-tree).** gigalens
   posteriors can be strongly correlated; the in-tree dense windowed IMM may
   precondition better than the paper's diagonal-once. Ship diagonal as the
   faithful default with dense as an option, or evaluate both on the synthetic
   testbed first?
6. **`maxiter` / total run length.** Unspecified in the paper (§E.1–2). Need
   defaults for Phase-1 `maxiter`, the `T=20%` window (presumably `0.2·maxiter`),
   and Phase-2 sample count. Expose as user inputs like `num_steps1/2`?
7. **A/B harness.** Confirm the paper rule (`F(C·D̃)`, `C=0.025`, switch on `x_i²`
   at `δ≤0.01`) is the **default**, with a single flag bundling the EMAUS variant
   (`C·D̃^{3/8}`, `C=0.1`, looser switch) for side-by-side comparison.
8. **Hutchinson full-rank D̃.** Ship diagonal-only first (App. D says diag ≈
   full-rank), add the 100-Rademacher Hutchinson estimator later, or build both now?
9. **Superchains / split-R̂.** Spec marks these out of scope (future work). Confirm
   we omit `superchain_size>1` and the split-R̂ switch diagnostic for v1.
10. **Driver shape.** New sibling module `inference/laps.py` mirroring
    `mclmc.py`'s `LAPS_JIT` / `full_laps_with_adapt_sharded` public surface, or
    fold a `phase` switch into the existing driver? (Recommend a sibling for
    clarity and to keep the MCLMC driver byte-stable.)
