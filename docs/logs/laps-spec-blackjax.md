# LAPS spec — as implemented in blackjax 1.5 (Reader B, code-only)

**Sources read (host filesystem, blackjax 1.5):**
- `blackjax/adaptation/laps.py` (top-level `laps`, phase-2 `Adaptation`)
- `blackjax/adaptation/laps_burn_in.py` (phase-1 `Adaptation`, init, kernel)
- `blackjax/util.py` (`run_eca`, `eca_step`, `while_with_info`, `ensemble_execute_fn`, `add_splitR`)
- `blackjax/adaptation/step_size.py` (`bisection_monotonic_fn`)
- `blackjax/mcmc/mclmc.py` (unadjusted kernel), `blackjax/mcmc/adjusted_mclmc.py` (adjusted kernel)
- `blackjax/mcmc/integrators.py` (isokinetic integrators, coefficients, Maruyama refresh)
- `blackjax/diagnostics.py` (`splitR`)

This document reports what the code does, line-traceably. Surprises / undocumented choices are flagged `[NO STATED RATIONALE]`.

LAPS = "Late Adjusted Parallel Sampler": an **ensemble** (many chains in parallel via `shard_map` over a `mesh`) run in two phases — an **unadjusted MCLMC burn-in** then an **adjusted (Metropolis-corrected) MCLMC refinement**. Adaptation is *ensemble chain adaptation* (ECA): hyperparameters are shared across all chains and updated from cross-chain expectation values each step.

---

## 1. Top-level entry point & signature

`laps(...)` — `laps.py:143-168`. Full argument list with defaults:

| arg | default | feeds into |
|---|---|---|
| `logdensity_fn` | — | both kernels (`laps.py:204`, `267`) |
| `sample_init` | — | `initialize` draws initial positions (`laps_burn_in.py:98`) |
| `ndims` | — | mass-matrix size, `norm_factor=sqrt(ndims)`, integrator choice |
| `num_steps1` | — | max steps of phase 1 (`laps.py:218`) |
| `num_steps2` | — | **gradient budget** of phase 2; `num_samples = num_steps2 // (gradient_calls_per_step * steps_per_sample)` (`laps.py:285`) |
| `num_chains` | — | ensemble size |
| `mesh` | — | device mesh for `shard_map` |
| `rng_key` | — | split into `key_init, key1, key2` (`laps.py:190`) |
| `microcanonical` | `True` | only `True` supported; else raises (`laps_burn_in.py:54`, `laps.py:280`) |
| `alpha` | `1.9` | sets L: `L = alpha * sqrt(mean Var) * sqrt(ndims)` (`laps_burn_in.py:310-314`) |
| `save_frac` | `0.2` | `save_num = round(save_frac*num_steps1)` — history-window length (`laps.py:205`) |
| `C` | `0.1` | constant in `EEVPD_wanted = C * bias^(3/8)` (`laps_burn_in.py:326`) |
| `early_stop` | `True` | enables phase-1 while-loop early stop (`laps.py:228`) |
| `r_end` | `0.01` | phase-1 stop threshold on `r_max` (`laps_burn_in.py:365`) |
| `bias_type` | `3` | selects which scalar drives `EEVPD_wanted` (`laps_burn_in.py:323-325`) |
| `diagonal_preconditioning` | `True` | use estimated per-dim variance as inverse mass matrix in **phase 2** (`laps.py:250-257`) |
| `integrator_coefficients` | `None` | phase-2 integrator; `None` ⇒ auto by `ndims>200` (`laps.py:233-244`) |
| `steps_per_sample` | `15` | phase-2 `num_integration_steps` per MH step (held fixed) |
| `acc_prob` | `None` | phase-2 target accept; `None` ⇒ auto 0.7/0.9 (`laps.py:238-244`) |
| `observables_for_bias` | `lambda x: x` | feeds history → `r_max/r_avg` and the early-stop test; **also** the `bias` driver iff `bias_type∈{0,1}` |
| `all_chains_info` | `None` | optional per-chain per-step summary fn (memory heavy) (`util.py:406-413`) |
| `diagnostics` | `True` | whether to return the info dict |
| `contract` | `lambda x: 0.0` | maps `E[observables_for_bias]` → the **diagnostic** `bias` only (`laps_burn_in.py:318`, `laps.py:93`) |
| `superchain_size` | `1` | split-Rhat grouping + key sharing across chains |

**Returns** (`laps.py:320`): a 4-tuple
`(info, gradient_calls_per_step, _acc_prob, final_state)` where
- `info = {"phase_1": info1, "phase_2": info2}` (or `None` if `diagnostics=False`). Each `infoN` is a pytree of per-step scalar histories (averaged over the ensemble), already trimmed to the actual number of executed steps (`util.py:549`).
- `gradient_calls_per_step = len(coeffs)//2` (`laps.py:246-248`).
- `_acc_prob`: the resolved phase-2 target acceptance.
- `final_state`: phase-2 `HMCState(position, logdensity, logdensity_grad)` — the ensemble of final positions, sharded over chains.

There is **no central temperature/"path" / annealing schedule** anywhere in this code (despite the name). "Adaptive path" reduces, in this implementation, to (a) adaptive step size + L + diagonal mass matrix during burn-in, and (b) a late switch to a Metropolis-adjusted kernel. `[NO STATED RATIONALE]` for the "path"/"adaptive" naming relative to code.

---

## 2. Control flow, step by step

`run_eca` (`util.py:455-551`) is the ensemble driver for **both** phases: it `shard_map`s over chains, and per step (`eca_step`, `util.py:322-366`): `vmap`s the kernel over chains → `vmap`s `summary_statistics_fn` → `lax.psum(... )/num_chains` to form the **cross-chain expectation** `Etheta` → calls `adaptation.update(adaptation_state, Etheta)`. Hyperparameters are thus a single shared object updated from ensemble averages (this is the "ECA" mechanism).

### Initialization — `laps_burn_in.initialize` (`laps_burn_in.py:77-156`)
- Draw each chain's position from `sample_init(key)`; set velocity = `grad logp / |grad logp|` (unit norm, since `microcanonical`) (`laps_burn_in.py:96-106`).
- Compute per-coordinate equipartition `E_ii = -x*g` of the initial point (`summary_statistics_fn`, `:108-112`), ensemble-averaged.
- Flip each velocity component sign: `signs = -2*(flat_equi<1)+1` (i.e. −1 where `E_ii<1`, +1 otherwise) (`:145`). `[NO STATED RATIONALE]` beyond the docstring "velocity along grad if E_ii>1, against if E_ii<1".

### Phase 1 — unadjusted MCLMC burn-in (`laps.py:204-229`)
- Kernel = `laps_burn_in.build_kernel` → `mclmc.build_kernel(integrator=isokinetic_velocity_verlet, inverse_mass_matrix=jnp.ones(ndims))` (`laps_burn_in.py:48-52`). **The mass matrix used in the dynamics is hard-coded to identity** (`laps_burn_in.py:51`); the per-dim variance estimated by `update` is *not* fed back into the phase-1 kernel — it is only carried forward to phase 2. `[NOTABLE]` Phase-1 dynamics are isotropic; preconditioning is estimated, not applied, during burn-in.
- The kernel calls `mclmc` with `L = adap.L`, `step_size = adap.step_size`; wraps with NaN rejection (`laps_burn_in.py:56-72`): on any non-finite leaf it keeps the old state and reports `nans=1`.
- `mclmc.build_kernel` integrates with `with_isokinetic_maruyama` (`mclmc.py:91`): half-step partial momentum refresh → one isokinetic velocity-Verlet step → half-step refresh (`integrators.py:530-556`); decoherence length = `L`. High-energy cutoff is `inf` by default (disabled); NaNs trigger a momentum re-draw (`mclmc.py:202-256`).
- **Per-step adaptation** (`laps_burn_in.py:288-361`), from ensemble means `Etheta`:
  - `L = alpha * sqrt(mean_d(Var_d)) * sqrt(ndims)`, with `Var_d = E[x²]−E[x]²` (`:310-316`).
  - `inverse_mass_matrix = Var_d` (per-dimension variance) (`:316`) — stored for phase 2.
  - `EEVPD = (E[ΔE²]−E[ΔE]²)/ndims` — variance of energy change per dim (`:317`).
  - `bias = [r_max, r_avg, equi_full, equi_diag][bias_type]` (`:323-325`).
  - `EEVPD_wanted = C * bias^(3/8)` (`:326`). (An alternative "phi function" form is present but commented out, `:327-328`.)
  - `eps_factor = clip((EEVPD_wanted/EEVPD)^(1/6), 0.3, 3.0)`; on NaNs forced to `0.5` (`:330-335`).
  - `step_size ← step_size * eps_factor` (`:354`). Step size is set **directly** (no dual averaging in phase 1).
- **Sample collection / convergence history** updated each step (`:293-308`): `update_history` keeps the last `save_num` ensemble-mean observables; `contract_history` → `(r_max, r_avg)` (see §3). `r_max` is logged into `history.stopping` only after `step_count > save_num`.
- **Early stop**: `run_eca(..., early_stop=True)` uses `while_with_info` (`util.py:416-452`) with `while_cond` = `(r_max > r_end) | (counter < save_num)` (`laps_burn_in.py:363-365`). Phase 1 halts when `r_max ≤ r_end` **and** at least `save_num` steps have run, else at `num_steps1`. Initial hyperparameters: `L=inf` (no first-step decoherence), `step_size=0.01*sqrt(ndims)`, `EEVPD=EEVPD_wanted=1e-3` (`laps_burn_in.py:260-268`).

### Between phases (`laps.py:232-292`)
- Integrator: `omelyan_coefficients` if `ndims>200` else `mclachlan_coefficients` (`:234-237`); `gradient_calls_per_step = len(coeffs)//2` (mclachlan→2, omelyan→5) (`:246-248`).
- Target accept `_acc_prob`: 0.9 if high-dim else 0.7 (auto), or 0.9 if user passed coefficients (`:238-244`). `[NO STATED RATIONALE]` for the specific 0.7/0.9 split.
- Diagonal preconditioning (`:250-257`): `inverse_mass_matrix = final_adaptation_state.inverse_mass_matrix` (= phase-1 `Var_d`); rescale `step_size /= sqrt(mean(inverse_mass_matrix))` so the step reflects the average scale change. If disabled, `inverse_mass_matrix = 1.0`.

### Phase 2 — adjusted (Metropolis) MCLMC refinement (`laps.py:262-313`)
- Kernel = `adjusted_mclmc.build_kernel(integrator=generate_isokinetic_integrator(coeffs), logdensity_fn, inverse_mass_matrix)` wrapped to call with `step_size=adap.step_size`, `num_integration_steps=adap.steps_per_sample`, `L_proposal_factor=1.25` (`:265-277`). The adjusted kernel draws a unit-vector momentum, integrates `num_integration_steps` isokinetic-Maruyama steps (partial refresh with decoherence `1.25 * num_steps*step_size`), then a binomial MH accept/reject on `Δenergy` (`adjusted_mclmc.py:79-114, 179-263`).
- Runs `num_samples = num_steps2 // (gradient_calls_per_step * steps_per_sample)` steps (`:285`), **no early stop** (`run_eca` default `early_stop=False` ⇒ plain `lax.scan`).
- Phase-2 adaptation (`laps.py:90-120`): only **step size** is adapted, via `bisection_monotonic_fn(acc_prob_target)` toward `_acc_prob` (see §4). `steps_per_sample` is held fixed; `L = step_size*steps_per_sample` is a *derived diagnostic*, not a control. `num_adaptation_samples = num_samples//2` is passed in but **never read** by `update` — the comment "number of samples after which the stepsize is fixed" is **not enforced here**; the bisection self-terminates on tolerance instead. `[NOTABLE / NO STATED RATIONALE]`.
- `final_state` = phase-2 ensemble `HMCState`.

---

## 3. Bias-correction mechanism

Two distinct "bias" notions — keep them separate:

**(a) The adaptation driver `bias`** (`laps_burn_in.py:323-326`). A scalar selected by `bias_type` from `[r_max, r_avg, equi_full, equi_diag]`:
- `bias_type=0` → `r_max` (max over dims of recent relative fluctuation of `observables_for_bias`)
- `bias_type=1` → `r_avg` (mean over dims of same)
- `bias_type=2` → `equi_full` = full-rank equipartition loss `Tr[(1−E)ᵀ(1−E)]/d²` via Hutchinson (`laps_burn_in.py:200-224`)
- `bias_type=3` → `equi_diag` = `mean_d( (1 − E_ii)² )`, `E_ii = E[−x_i g_i]` (`laps_burn_in.py:217-219`) — **the default**.

This drives the step-size target: `EEVPD_wanted = C·bias^(3/8)`. So with the default `bias_type=3`, step-size adaptation is governed purely by **equipartition** (`−x·g→1`), and `observables_for_bias`/`contract` do **not** affect the dynamics at all. The exponent 3/8 and `C=0.1` are tagged in the docstring only as "eq (9) of EMAUS paper"; `[NO STATED RATIONALE]` for 3/8 within blackjax.

**The fluctuation statistic** `contract_history(theta, weights)` (`laps_burn_in.py:168-174`): over a window of the last `save_num` ensemble-mean observables (shape `(save_num, ndims)`):
```
square_average = (Σ_t w_t θ_t / Σw)²            # per dim
average_square =  Σ_t w_t θ_t² / Σw             # per dim
r = (average_square − square_average) / square_average   # per dim, ≈ Var/mean²
return [max_d r, mean_d r]                       # = (r_max, r_avg)
```
i.e. the squared coefficient of variation of the running ensemble-mean observable across the recent window — a drift/stability measure. `weights` are all 1 once the window fills (`update_history_scalar(1.0, …)`).

**(b) The diagnostic `bias`** (`true_bias`): `self.contract(E[observables_for_bias])` (`laps_burn_in.py:318`, also phase-2 `laps.py:93`). Pure logging; with the default `contract = lambda x: 0.0` it is `0`. The helper `laps.bias(model)` (`laps.py:123-133`) shows the intended use: `observables(x)=square(transform(x))`, `contract(E_x2)=[max, mean] of (E_x2 − model.E_x2)²/Var_x2` — a normalized squared error of the second moment vs. analytic truth (benchmarks only).

**`alpha`** enters only through `L` (`:310-314`); **`save_frac`** only sets `save_num` (window length and the `counter<save_num` guard); **`C`** only scales `EEVPD_wanted`.

---

## 4. Adaptation / schedule internals

- **No temperature/annealing path** is constructed anywhere; there is no schedule object. "Schedule" in this code = the per-step EEVPD-target rule (phase 1) and the bisection (phase 2).
- **`superchain_size` & resampling**: There is **no resampling** of chains anywhere. `superchain_size` only (i) groups chains for split-Rhat in `splitR` (`diagnostics.py:212-230`): chains are reshaped into `num_chains/superchain_size` superchains of `superchain_size`, within-superchain variance `W` vs between-superchain variance `B`, `R = sqrt(1 + B/W)`; and (ii) controls RNG-key sharing in `ensemble_execute_fn` (chains in a superchain share an init key, `util.py:600-606`). `add_splitR` (`util.py:373-403`): `superchain_size==1` ⇒ `R_avg=R_max=0` (split-Rhat disabled); `>1` ⇒ `R_avg=mean(R²−1)`, `R_max=max(R²−1)`.
- **Phase-1 early-stop / `r_end`** (covered §2): statistic = `r_max` (max-over-dims squared CoV of the recent ensemble-mean `observables_for_bias`); threshold `r_end=0.01`; stop when `r_max ≤ r_end` and `counter ≥ save_num`. This is the **only** convergence test that halts a phase.
- **Phase-2 step-size adaptation — `bisection_monotonic_fn`** (`step_size.py:262-304`): a bracket-free bisection of the (monotonically decreasing) accept-rate-vs-log-step-size curve, targeting `acc_prob_wanted`. Per step: if `acc_rate > target` raise lower bound, else lower upper bound; once both bounds finite, bisect their average, else shift by `reduce_shift=log2`. Terminates (freezes step size) when `|acc_rate − target| < tolerance=0.03` (latched). No dual-averaging in either phase.

---

## 5. Every hyperparameter

| name | default | controls | rationale in code? |
|---|---|---|---|
| `microcanonical` | `True` | selects MCLMC; only value supported | none; raises otherwise |
| `alpha` | `1.9` | `L = alpha·sqrt(mean Var)·sqrt(d)` | docstring "L=sqrt(d)·alpha·variances"; value unjustified |
| `save_frac` | `0.2` | history window `save_num=round(save_frac·num_steps1)` | docstring "fraction used to estimate fluctuation"; 0.2 unjustified |
| `C` | `0.1` | `EEVPD_wanted=C·bias^(3/8)` | docstring "eq (9) of EMAUS paper"; not derived here |
| `early_stop` | `True` | enables phase-1 while-loop stop | none |
| `r_end` | `0.01` | phase-1 stop threshold on `r_max` | none; bare number |
| `bias_type` | `3` | picks adaptation driver (see §3) | inline comment lists the 4 options; default choice unjustified |
| `diagonal_preconditioning` | `True` | phase-2 inverse mass matrix = phase-1 `Var_d`; rescales step | comment on the step rescale only |
| `integrator_coefficients` | `None` | phase-2 integrator | `None`⇒omelyan if `d>200` else mclachlan; threshold 200 unjustified |
| `steps_per_sample` | `15` | phase-2 integration steps/MH sample (fixed) | none |
| `acc_prob` | `None` | phase-2 target accept | `None`⇒0.9 (d>200) / 0.7 else; split unjustified |
| `observables_for_bias` | `lambda x:x` | history→`r_max/r_avg`+early-stop; driver iff `bias_type∈{0,1}` | docstring |
| `contract` | `lambda x:0.0` | diagnostic `true_bias` only | — |
| `superchain_size` | `1` | split-Rhat grouping + key sharing; no resampling | — |
| `L_proposal_factor` | `1.25` (hard-coded, `laps.py:276`) | partial-refresh decoherence = `1.25·n·ε` in phase 2 | none; not exposed as an arg |
| `num_adaptation_samples` | `num_samples//2` | passed to phase-2 `Adaptation`, **never used** | comment claims it fixes step size; not enforced |
| phase-1 init `step_size` | `0.01·sqrt(d)` | initial ε | none (`laps_burn_in.py:263`) |
| phase-1 init `EEVPD` | `1e-3` | seed for first ratio | none |
| bisection `tolerance` | `0.03` | phase-2 step-size freeze band | none (`step_size.py:262`) |
| `eps_factor` clip | `[0.3, 3.0]` | per-step step-size change cap | none (`laps_burn_in.py:331`) |
| EEVPD exponent | `3/8` (target), `1/6` (factor) | step-size law | none in-code |

---

## 6. Diagnostics exposed (`info` outputs)

These are the internal-health signals to check our own implementation against. Each is an ensemble-averaged scalar stored per executed step.

**`info["phase_1"]`** (`laps_burn_in.py:337-349`, plus `util.py`):
- `L` — momentum-decoherence length used that step.
- `step_size` — integrator ε that step.
- `EEVPD` — energy-error variance per dim `(E[ΔE²]−E[ΔE]²)/d`. Core health metric (energy conservation quality).
- `EEVPD_wanted` — target `C·bias^(3/8)`; compare to `EEVPD` to see whether ε is being pushed up/down.
- `equi_diag` — diagonal equipartition loss `mean((1−E_ii)²)`; →0 at stationarity.
- `equi_full` — full-rank equipartition loss (Hutchinson); →0 at stationarity.
- `bias` — diagnostic `contract(E[obs])` (0 unless user supplies `contract`/benchmark truth).
- `r_max`, `r_avg` — max / mean recent relative fluctuation of `observables_for_bias`; `r_max` is the **early-stop driver** (target `r_end`).
- `entropy` — `E[−logdensity]` (ensemble mean negative log-density).
- `observables` — `E[observables(position)]` (user diagnostic; default 0).
- `R_avg`, `R_max` — split-Rhat health: `mean/max(R²−1)` if `superchain_size>1`, else 0 (`util.py:373-403`).
- `all_chains_info` — per-chain values of `all_chains_info(position)` if provided (memory heavy).

**`info["phase_2"]`** (`laps.py:95-103`):
- `L` = `step_size·steps_per_sample` (derived trajectory length).
- `steps_per_sample` — fixed integration steps/sample.
- `step_size` — current ε (being bisected toward `acc_prob`).
- `acc_prob` — ensemble-mean MH acceptance rate that step (the bisection target signal).
- `equi_diag` — diagonal equipartition loss.
- `bias` — diagnostic `contract(E[obs])`.
- `observables` — `E[observables(position)]`.
- `R_avg`, `R_max` — split-Rhat as above.

Histories are trimmed to the actually-executed count when `early_stop` short-circuits phase 1 (`util.py:549`).

---

## Notable findings / flags

1. **Preconditioning is estimated in phase 1 but applied only in phase 2.** The phase-1 MCLMC kernel hard-codes `inverse_mass_matrix=jnp.ones(ndims)` (`laps_burn_in.py:51`); the estimated `Var_d` only takes effect after the switch (`laps.py:251`). Phase-1 anisotropy is absorbed into the scalar `L` instead.
2. **`observables_for_bias`/`contract` do not steer the sampler under the default `bias_type=3`.** They influence dynamics only via `r_max/r_avg` (bias_type 0/1); `contract` is purely diagnostic. Default adaptation is equipartition-driven.
3. **`num_adaptation_samples` is dead in phase 2** — passed but never used; the "freeze step size after N samples" behavior its comment describes is not implemented (bisection self-terminates on a 0.03 tolerance instead).
4. **No annealing/temperature/path schedule and no chain resampling** exist in this implementation, despite the "adaptive path sampler" name. `superchain_size` only affects split-Rhat grouping and RNG key sharing.
5. **`L_proposal_factor=1.25` is hard-coded** in the phase-2 kernel wrapper (`laps.py:276`), not exposed.
6. Numerous bare constants (`r_end=0.01`, `C=0.1`, exponent `3/8`, `eps_factor` clip `[0.3,3.0]`, `ndims>200`, accept `0.7/0.9`, init `step_size=0.01√d`, tolerance `0.03`) carry **no in-code derivation**.
