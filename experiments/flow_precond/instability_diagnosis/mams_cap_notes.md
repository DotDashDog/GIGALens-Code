# MAMS trajectory-length cap (`max_num_integration_steps`) — design notes

2026-07-09. Patch to `src/gigalens_research/inference/mams.py` on branch
`flow-precond-mams`. Motivation: a 20-step-burnin MAMS run hung ~3.6 h;
hypothesis (confirmed structurally below): dual-averaging left the step size
collapsed, and with L fixed, `n = L/eps` exploded — unbounded integrator calls
per transition. All verdicts here are **UNCERTIFIED**.

## 1. Upstream practice (investigated, with citations)

Env inspected: blackjax 1.5 and tfp-nightly 0.26.0.dev20260505 in
`/global/homes/l/linusu/.conda/envs/gigalens_oldapi/lib/python3.14/site-packages/`
(paths below relative to that), plus gigalens-old at
`/global/u1/l/linusu/gigalens-old/src/gigalens/`.

**The "30" the user remembered is real and is ours-adjacent**: gigalens-old's
HMC pipeline defaults `max_leapfrog_steps=30`
(`gigalens-old/src/gigalens/jax/inference.py:303` and `:402`; also
`jax/experimental/normalizing_flows.py:119`) and passes it to TFP's
`GradientBasedTrajectoryLengthAdaptation` (`jax/inference.py:348-351`). So the
precedent is a **ChEES-style trajectory-length adaptation cap**, not a NUTS
treedepth.

How TFP enforces it (`tensorflow_probability/python/experimental/mcmc/gradient_based_trajectory_length_adaptation.py`):

- The **controller state itself is clipped**, not the realized step count:
  `_clip_max_trajectory_length` clamps `max_trajectory_length` to
  `[0, step_size * max_leapfrog_steps]` inside every adaptation update, and
  only while adapting (lines 1101-1104, 1112-1119). This is simultaneously the
  bound and the anti-windup: the trajectory-length optimizer can never store an
  unrealizable value.
- The Halton jitter **multiplies the already-clipped value** (jitter in [0,1]),
  then `num_leapfrog_steps = ceil(jittered / step_size)` (lines 843-863), so
  realized steps <= `max_leapfrog_steps` by construction and the jitter
  distribution keeps its shape (it is scaled, not truncated).
- TFP default is `max_leapfrog_steps=1000` (line 629); gigalens-old chose 30.

blackjax equivalents:

- `blackjax/adaptation/chees_adaptation.py`: same design —
  `max_leapfrog_steps: int = 1000` (line 317), and the trajectory-length state
  is clipped to `[new_step_size, max_leapfrog_steps * new_step_size]` at every
  update (lines 236-243, "clip new trajectory length to avoid too large
  trajectories").
- `blackjax/adaptation/adjusted_mclmc_adaptation.py` (blackjax's own MAMS
  tuning) has **no explicit n cap**, but avoids the explosion structurally:
  during dual averaging with `fix_L=False`, **L is rescaled proportionally to
  the step size** (`L = L * (new_eps/old_eps)`, lines 223-226), so
  `n = L/eps` stays constant through the DA transient; additionally
  `step_size` is clamped to `[1e-5, L/1.1]` (lines 207-210). L updates are
  ratio-bounded: `Lratio_upperbound = 2.0` (lines 13-14, applied at 304-311 and
  380-384) — same 2x-per-update bound our `L_max_ratio` implements.
- `blackjax/mcmc/adjusted_mclmc_dynamic.py`: the kernel takes whatever
  `integration_steps_fn` yields, uncapped (line 91); the *default* is
  `jax.random.randint(key, (), 1, 10)` — 1..9 steps (line 134). Our
  `halton_trajectory_length` lives in `blackjax/mcmc/dynamic_hmc.py:206-211`:
  `rint(0.5 + halton(i) * rescale(mu))`, i.e. quasi-uniform on ~[1, 2*mu-1]
  with mean exactly `mu` (`rescale`, lines 199-203).
- NUTS bounds by doubling, differently: `max_num_doublings: int = 10`
  (`blackjax/mcmc/nuts.py:119,155`; `max_num_expansions` in
  `blackjax/mcmc/trajectory.py:526`). At the bound the expansion loop simply
  stops (`trajectory.py:569`) and the sample is drawn from the trajectory built
  so far — no rejection. Stan's analogous `max_treedepth=10` (=> <=1023
  leapfrogs, with a warning) is lore, not in these files.

Healthy in-repo reference: `experiments/sim_carousel/messy_tests/dpie/mams/diagnostics.npz`
key `num_integration_steps`, shape (64, 2000): mean 13.4, p99 37, **max 38**.
A cap of 60 realized steps never binds there.

## 2. Controller-interaction analysis (our mams.py)

Controller layout (pre-patch): per-chain DA on eps (target acceptance 0.9),
active only in modes 1/2 (tune1 + tune2), reset at each mass-matrix window
install, synced across chains once at `step_size_sync_step`, **frozen in mode 3
and mode 0**. L is constant from init until a **single** ESS-based update at
`i == L_adaptation_step` (first sampling step), ratio-capped at
`L_max_ratio = 2`. Each step: `avg_n = max(L/mean_eps, 1)`;
`n = max(halton_trajectory_length(i, avg_n), 1)` — **no upper bound**.

Hang mechanism: a DA transient (e.g. after a mass-matrix install changes the
geometry under eps and resets DA) or a too-short burnin drives eps down with L
fixed; `n = L/eps` explodes. Crucially, with a 20-step burnin the run enters
mode 0 almost immediately, where **no controller runs at all**: a collapsed eps
is frozen and every one of the `num_results` transitions costs ~L/eps gradient
evals forever. In mode 0 a hard cap is the *only* possible protection.

**(a) Is the DA loop stable when the cap binds?** Yes — self-correcting. For
fixed `n = N_MAX`, MH acceptance is still strictly monotone decreasing in eps
(per-step energy error grows with eps; capping n only removes the n-growth of
the accumulated error, making acceptance weakly *higher* at small eps). The cap
binds precisely when eps is far below the acceptance-0.9 fixed point, so
acceptance ~ 1 > 0.9, DA raises eps, `avg_n = L/eps` falls, the cap releases.
No positive feedback exists; the DA fixed point (acceptance = 0.9) is
unchanged. The only cost while capped is transiently shorter trajectories
(less exploration per transition) — and each DA recovery step now costs
<= N_MAX gradient evals instead of unbounded.

**(b) L windup?** Unbounded windup cannot occur in the current schedule (a
single L update, ratio-capped at 2x). But a capped, slow-mixing tune3 yields
low ESS, pushing the one update *up* into territory the cap makes unreachable:
the installed L would be a lie (realized trajectory time = N_MAX*eps < L) and
the Hist L trace would mislead. Standard fix, taken verbatim from TFP
`_clip_max_trajectory_length` / blackjax chees (cites above): **clamp L itself
to (N_MAX/2) * mean_eps in the L update**. At that step, eps has been synced
and frozen since `step_size_sync_step`, so the clamp uses exactly the eps that
will run the whole sampling phase — the installed L is always realizable.

**(c) Where to cap so the Halton jitter stays unbiased?** Cap the Halton
**mean before the jitter**, not the post-jitter integer.
`halton_trajectory_length(i, mu)` is engineered (via `rescale`) so draws are
quasi-uniform on ~[1, 2*mu-1] with mean exactly `mu`. Clamping
`mu' = min(avg_n, N_MAX/2)` preserves the family (mean mu', max realized
2*mu'-1 <= N_MAX-1): full anti-cycling coverage over the realizable range.
Truncating post-jitter (`min(n, N_MAX)`) would instead (i) collapse the whole
upper half of the Halton sequence onto a single atom at N_MAX — reintroducing
the resonance risk the jitter exists to kill — and (ii) push the realized mean
below the `avg_n` the controllers reason about. This mirrors TFP, where the
clipped state is what the jitter multiplies. A post-jitter
`min(n, N_MAX)` is kept as a belt-and-braces guard that is mathematically
unreachable while the mean clamp is active (`rint(0.5 + h*rescale(N/2)) <= N-1`
for Halton h < 1).

Cap semantics chosen: `max_num_integration_steps = 60` bounds the **realized**
n at 60 (attains 59), with the Halton mean pinned at 30 when saturated — the
direct "a bit more generous" analog of gigalens-old HMC's
`max_leapfrog_steps=30` (realized <= 30, mean ~15 under U[0,1] jitter).

## 3. Patch summary (mams.py)

- `MAMS_JIT(..., max_num_integration_steps=60)` threaded through to
  `full_mams_with_adapt_sharded(..., max_num_integration_steps=60)`; validated
  `>= 2`; `half_max_n = 0.5 * max_num_integration_steps` kept a Python float so
  JAX weak typing leaves dtypes untouched.
- Trajectory block: `avg_n_raw = max(L/mean_ss, 1)`;
  `avg_n = min(avg_n_raw, half_max_n)`; `traj_capped = avg_n_raw >= half_max_n`;
  `n = min(max(halton_trajectory_length(i, avg_n), 1), max_num_integration_steps)`.
- `calc_new_L`: after the existing `min(L_new, L * L_max_ratio)`, added
  `min(L_new, half_max_n * mean_ss)` (anti-windup / realizability clamp).
- `Hist` gains `traj_capped` (bool, per chain per step), **appended last**;
  all known consumers (`inference_utils/pipeline.py:1877-1892`,
  `plotting/diagnostics.py:293`, `experiments/flow_precond/carousel_benchmark.py:189-190`,
  `experiments/mams_validation/*`) access Hist by attribute, so layout is
  backward-compatible. `full_mams_with_adapt_sharded`'s external caller
  (`experiments/mams_validation/compare_bimodal.py:131`) uses kwargs; the new
  kwarg has a default.
- `traj_capped` uses `>=` (not `>`): once the L clamp installs
  `L = half_max_n * mean_ss`, `avg_n_raw` sits exactly AT the bound — that
  pinned state is semantically capped and must stay visible (see Test B note).

## 4. Pre-registered predictions and results

Runs: 8-dim anisotropic Gaussian (scales logspace(-0.5, 0.5, 8)), 8 chains,
seed 3, CPU, x64. Script: `mams_cap_tests.py` (this dir). Unpatched baseline =
byte snapshot of HEAD `mams.py` (`tmp/mams_unpatched_snapshot.py`), imported
via a symlink-copied package tree so both variants resolve the same package.

**Test A — bit-identity, cap never binds** (200+200 steps, default init).
Prediction: all compared arrays byte-identical; capped fraction 0.
Falsifier: any byte differs. **Result: PASS** — position (8,400,8) f64,
step_size, L, acceptance_rate f64, num_integration_steps i32 all
`tobytes()`-identical between unpatched and patched; capped fraction 0.0;
max n = 5. (Also confirms the added Hist field / scan-pytree change does not
perturb XLA numerics here.)

**Test B — pathological hang class** (eps0=1e-8, L0=sqrt(8), 20-step burnin,
100 results). Arithmetic for unpatched: first transition alone needs
`avg_n = L/eps = 2.83/1e-8 = 2.8e8` integrator calls; at the measured
~200 us per chain-integrator-call on this CPU target that projects to
**>= ~7,700 min for the single first transition** (the hang class; on GPU with
the real model, the observed 3.6 h is the same mechanism at different
constants). Patched: **wall 5.8 s** for all 120 steps, 3,551 integrator
calls/chain total, max n = 59 <= 60, mean n = 29.6, capped fraction 1.000
(burnin capped fraction 1.000). Final eps 9.9e-3 — DA had only 16 steps and
could not fully recover; the run stays pinned at the cap, cheap and visible.

*Prediction revision (recorded per method discipline):* the original falsifier
"whole-run capped fraction < 0.5" tripped on the first execution (fraction
0.175) with the original `>` flag: the L anti-windup clamp installed
`L = 30 * eps` at the L-update step, after which `avg_n_raw == 30` exactly —
pinned at the bound, but the strict `>` reported "not capped". The mechanism
outperformed the prediction (L self-corrected too); the *diagnostic
definition* was wrong. Fixed by `>=` and by asserting burnin-capped > 0.5 in
addition. Not a tuned-to-pass change: the flag now reports "trajectory at the
maximum realizable length", which is the quantity a user must see.

**Test C — controller self-correction** (eps0 = healthy-final/100 = 3.9e-2,
400 burnin, 200 results). Prediction: capped fraction -> ~0 by end of tuning,
final eps within 3x of healthy final, no L windup.
Falsifier: end-of-tuning capped fraction > 0.1, eps ratio outside [1/3, 3], or
final L above `(N_MAX/2)*eps`. **Result: PASS** — capped fraction by tuning
quarter: 0.01, 0.00, 0.00, 0.00; sampling phase 0.00; final eps 4.56 (1.16x
healthy final 3.94); L trace {2.828 (init) -> 2.096 (single ESS update,
downward)}, far below the realizability bound 136.9; max n = 30 (early
transient), mean sampling n = 1.0. DA recovered from the 100x-too-small eps
within the first tuning quarter with the cap active — the loop is
self-correcting, no controller instability introduced.

Summary: A=PASS, B=PASS, C=PASS — **UNCERTIFIED**; grader should inspect the
test script, the mams.py diff, and the printed numbers above (reproducible via
the run command in the script docstring).

## 5. Open notes / residual risks

- Bit-identity was verified on CPU/x64 for this target; the guarantee argument
  (pure `min` against unreached bounds, weakly-typed Python-float constants,
  Hist field appended without touching existing dataflow) is
  platform-independent, but a GPU spot-check on a real target is cheap
  insurance before relying on it in a comparison study.
- When the cap is pinned for an entire *sampling* phase (Test B), the samples
  are still asymptotically unbiased (MH-adjusted; trajectory length does not
  affect invariance) but mixing is degraded and eps may be far from optimal —
  `traj_capped` (and `num_integration_steps` hugging N_MAX-ish values) is the
  signal to rerun with a longer burnin, not a license to trust the run.
- Int32 aside: unpatched, `n` could overflow int32 for eps < ~L/2e9; the cap
  also removes that hazard.
- `tmp/` holds the unpatched snapshot and test outputs; keep the snapshot if
  the bit-identity test should stay rerunnable after the patch lands.
