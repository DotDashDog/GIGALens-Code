# Plan: Normalizing-flow-preconditioned MAMS (NeuTra-style)

**Status:** approved plan, not started. Handoff doc — written to be executable by an agent
with no other context. Read `docs/agent-operating-card.md` first; the non-negotiables
(design checkpoint before expensive runs, gates, plots-before-metrics, UNCERTIFIED verdicts)
apply to every step here. Log substantive steps in
`docs/logs/carousel-mclmc-sampling.md` (newest-first + claims register).

## 1. Goal and context

Sampling the carousel lens posterior (`experiments/sim_carousel/prelim_sim_carousel.ipynb`,
33 params, dPIE-for-Lf model, July 2026) is limited by **curved narrow ridges** ("bananas"):
worst-parameter integrated autocorrelation time τ ≈ 3,600 draws for MCLMC; MAMS is ~14×
better per draw but the same geometry is its limiter too. A local-Hessian eigenframe analysis
(June 2026, see the carousel log) showed the curvature is physical and rotates 10–40° along
the ridge, so **no constant (affine) mass matrix can fix it** — but a nonlinear change of
variables can. Plan: train a normalizing flow T and run MAMS on the pulled-back target in
the flow's latent space (the NeuTra pattern, Hoffman et al. 2019, arXiv:1903.03704).

The posterior will keep getting harder (more sources, more data), so this is infrastructure,
not a one-off fix. The user has approved this plan including the one deviation from vanilla
NeuTra (Phase-B training, §4).

## 2. The algorithm (all of it)

Everything lives in the existing unconstrained z-space (`ProbModel.log_prob(z)`,
`gigalens/jax/scene_prob_model.py`). Train a diffeomorphism `T: u -> z` (flow) so that
`T#N(0,I) ≈ posterior`. Define the pulled-back target

```
log_pi_tilde(u) = pm.log_prob(T(u))[0] + log|det J_T(u)|
```

and run MAMS **unchanged** on `log_pi_tilde` in u-space; push final draws through `z = T(u)`.
If T is perfect, π̃ = N(0,I) and MAMS mixes in O(1) steps; an imperfect T costs efficiency
only — MAMS's Metropolis correction keeps everything exactly unbiased regardless of flow
quality. The flow is trained once and **frozen** before sampling (no adaptive retraining
in v1).

## 3. Code touchpoints (verified 2026-07-08)

All in `src/gigalens_research/inference_utils/pipeline.py` unless noted:

- `InferenceContext` (line ~203): carries `phys_model` + `prob_model` + `sim_config`. Stages read it.
- `SVIStage` (line ~1422): the training-loop skeleton to copy — optax/adabelief loop via
  `gigalens.jax.inference.SVI(ctx.prob_model, ...)`, produces `qz` (full-rank `MultivariateNormalTriL`) from arrays
  `qz_loc`, `qz_scale_tril`. The dpie run's SVI config: `n_vi=128, num_steps=1000,
  init_scales=1e-3, adabelief_1e-4_b1_0.95_b2_0.99`.
- `MAMSStage` (line ~1777): `requires=("qz",)`; calls
  `gigalens_research.inference.MAMS_JIT(prob_model=ctx.prob_model, qz=..., ...)`. Per its
  docstring, `qz` is used for **chain init, initial mass matrix, and SVI-mean reference** —
  all three need u-space equivalents (§5.3).
- Flow bijectors: use **tfp-jax** (`tensorflow_probability.substrates.jax.bijectors` —
  RealNVP / `RationalQuadraticSpline` / `Chain` / masked coupling). **Do NOT pip-install
  anything** (flowjax, distrax, …) — environment mutations require the user's explicit
  permission (see user CLAUDE.md). tfp-jax is already in the stack.

## 4. Design decisions already made (don't relitigate without new evidence)

1. **Kernel = MAMS** (not HMC as in the NeuTra paper, not unadjusted MCLMC). MAMS's MH step
   makes flow imperfection slow-not-bias, and unadjusted MCLMC's mode weights are known
   untrustworthy here (~3× pocket over-weighting, §6).
2. **Flow = affine whitening layer initialized from the SVI solution** (`qz_loc`,
   `qz_scale_tril` — start where full-rank SVI ends, learn only residual curvature) **+ 6–8
   RQ-spline coupling layers** (8–16 bins, range ±6 in whitened coords, [64,64] MLP
   conditioners, alternating binary masks). ~O(50k) params — negligible vs the ~ms two-band
   300² render+lstsq likelihood, so preconditioning overhead should be <5%.
3. **Float64 flow** to match the x64 pipeline (`JAX_ENABLE_X64=1`); log-det errors enter lp
   directly and the model-identity noise floor is ~0.3 nats (§6).
4. **Two-phase training** — the one deliberate deviation from vanilla NeuTra:
   - **Phase A (= NeuTra):** reverse-KL / ELBO, exactly the SVIStage loss with the flow as
     the family. 128 draws/step, 2–5k steps, monitor ELBO plateau.
   - **Phase B (addition):** forward-KL (maximum likelihood) refinement on real MCMC warmup
     samples. Rationale: reverse KL is mode-seeking; the existing full-rank SVI assigned the
     known secondary mode (~5% mass, §6) probability ≈ 0 (14σ out). A pure-Phase-A flow will
     do the same, making pocket transits *rarer* than vanilla MAMS. Training data exists:
     `messy_tests/dpie/mclmc/arrays.npz` `samples_z` (8×10k draws, 14.6% pocket occupancy).
     Phase B requires the flow inverse `T^{-1}(z)` — cheap for coupling flows (this is why
     coupling, not NeuTra's IAF).
   - **Run the A/B**: benchmark Phase-A-only vs Phase-A+B (same code, Phase B toggled).
     Pre-registered claim to test: A-only fails the pocket-coverage gate below.

## 5. Implementation stages (in order; each has a gate)

### 5.1 `TransformedProbModel` wrapper + identity gate  (~50 lines, zero risk)
Wrapper exposing the ProbModel interface MAMS_JIT actually touches (verify by reading
`gigalens_research/inference.py::MAMS_JIT` — at minimum `log_prob`; check for `log_like`,
`bij`, `num_pixels`, `z_param_names` uses):
- `log_prob(u) = inner.log_prob(T(u))` with `+ log|det J_T(u)|` added to the lp component
  (log_prob returns `(lp, red_chi2)` — Jacobian goes into lp only).
- `bij_forward` for physical decoding composes: `inner.bij.forward(T(u))`.
- **GATE I (identity):** with T = identity flow, a short MAMS run through the wrapper must
  reproduce vanilla MAMS statistics (same seed ⇒ ideally bit-identical trajectories; at
  minimum statistically indistinguishable ESS/R̂/acceptance). A wrapper that fails this is
  wrong regardless of anything downstream.

### 5.2 `FlowStage`
New stage: `requires=("z_best","qz")` (qz for the whitening init), `produces=("flow_params",
"flow_loss_hist")` (+ a `flow` artifact via `derive_artifacts`). Copy the SVIStage
run/hash/manifest pattern. Config-hash all architecture + training hyperparams.
- **GATE F (flow sanity):** (a) ELBO ≥ full-rank SVI's final ELBO (the flow strictly
  generalizes the affine family; final_loss for the dpie SVI run was 291453.1, lower=better
  in their sign convention — verify sign against `svi_loss_hist`); (b) **pocket-coverage
  gate:** `log q_flow(z_pocket_median) - log q_flow(z_main_median)` must be ≥ −8 nats
  (true value ≈ −3; both median vectors are saved in
  `/pscratch/sd/l/linusu/carousel_diag/basin_slice/basin_slice.npz` as `zP`, `zM`).
  Expect Phase-A-only to FAIL (b) — that failure is a result, not a bug; record it.

### 5.3 Latent-space plumbing for MAMSStage
`MAMSStage` consumes `qz` for chain init + initial mass matrix + mean reference. In u-space:
- Chain init: `u0 = T^{-1}(z0)` with `z0 ~ qz`, or directly `u0 ~ N(0, eps*I)` — prefer
  T^{-1}(qz-draws) to keep the stage contract's semantics.
- Initial (inverse) mass matrix: **identity** — the flow whitens by construction. Do not
  feed the z-space qz covariance.
- Cleanest wiring: a `BridgeStage` producing `qz_u` (e.g. `tfd.MultivariateNormalDiag(0, 1e-2)`
  in u-space, mirroring the existing `diag_qz` bridge) + the wrapped model, so MAMSStage
  itself stays untouched.

### 5.4 Benchmark on carousel-dPIE (pre-registered; this IS a consequential run)
Write a design checkpoint to the carousel log **before** running (operating-card rule 1) and
get user approval. The benchmark posterior is calibrated — numbers to beat, all from July
7–8 2026 runs (`messy_tests/dpie`, artifacts under `/pscratch/sd/l/linusu/carousel_diag/`):

| Quantity | Vanilla MAMS (1k draws, 8ch, SVI-seeded) | Vanilla MCLMC (10k, 8ch) |
|---|---|---|
| max rank-split-R̂ | 1.18 | 1.27 |
| min bulk-ESS | 31 | 22 |
| ESS / wall-second | 0.024 | 0.029 |
| worst-param τ | — | ≈3,600 (src5/center_x) |
| pocket occupancy | 4.6% (≈true; Laplace proxy 5.4%) | 14.6% (over-weighted ~3×) |

**Win conditions (flow-MAMS, same 8 chains / 1k results):** min bulk-ESS ≥ 5× vanilla MAMS;
ESS/wall-s ≥ 5×; flow overhead < 5% of step cost; pocket occupancy in [2%, 8%]; acceptance
≥ 0.85 at target 0.9. **Falsifier:** a flow passing GATE F that yields < 2× ESS gain ⇒
escalate architecture once (double layers/bins); if still < 2×, that's a real negative
result about flow-capturability of this geometry — log it, stop, budget goes to
many-chain scaling instead.

## 6. Facts an implementing agent needs (hard-won; do not rediscover)

- **Column→name map:** sampler z columns = alphabetically sorted flattened prior keys;
  verified for the 33-param dPIE model via `pm.z_param_names` (2026-07-07). Col 6 =
  `planes/0/mass/1/center_x` (EPL_Le). Pocket membership test: `z[:, 6] > -22.35`.
- **After the flow, u-columns are NOT parameters.** All per-coordinate diagnostics
  (R̂/ESS/traces/occupancy) must be computed on `z = T(u)`, never raw u. This is the C-8
  trap's flow-era descendant — bake the mapping into whatever the stage saves.
- **Secondary mode is real:** pocket median lp = main + 0.9 nats (equal height), mass ≈ 5%
  (MAMS occupancy 4.6% ≈ Laplace proxy 5.4%); differs only in EPL_Le center (~1.2 px) vs
  θ_E (2.24↔2.37). Straight-line z-slice between basin medians has a smooth ~278-nat gorge
  (upper bound; ridge is curved). Diagnostic script:
  `/global/homes/l/linusu/.claude/jobs/fe3e84e8/tmp/basin_logp.py` (model rebuild + GATE M
  pattern to copy; job tmp is ephemeral — copy what you need before relying on it).
- **Model-identity noise floor:** rebuilding the ProbModel reproduces recorded lp to
  ~±0.3 nats (float32-conv reduction-order noise; red-χ² matches to ~2e-6). Gates comparing
  cross-session lp values need rtol ≥ 3e-6, not 1e-10. `EPL(18)` vs `EPL(50)` for EPL_Le is
  bit-identical.
- **MAP gate:** any pipeline run feeding this work must use a plateaued MAP (4000 steps on
  this posterior; best-lp gain over last 250 steps ≲ 1 nat). A 500-step MAP produced fake
  "many modes" twice.
- **Environment:** run GPU work via shifter image
  `docker:ghcr.io/nvidia/jax:jax-2026-04-13` with
  `PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages`,
  `JAX_ENABLE_X64=1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Own allocation:
  `salloc --no-shell -N1 -C gpu -G 1 -t 30 -q interactive -A deepsrch_g`, then
  `srun --overlap --jobid=<ID> --ntasks=1 --gpus=1 shifter ... /usr/bin/python3 <script>`.
  Never run GPU jobs on the login node; never pip/conda install anything.
- **Known failure modes to watch:** reverse-KL mode collapse (GATE F(b)); spline tail
  under-coverage (check `max |T^{-1}(z)|` over warmup samples ≲ spline range; widen if not);
  flow staleness when the model changes (FlowStage must re-run on ctx_hash change — the
  pipeline's input-hash machinery gives this for free if `qz`/`z_best` are in `requires`).

## 7. Definition of done

1. GATE I passes (identity wrapper ≡ vanilla MAMS).
2. FlowStage trains on carousel-dPIE; GATE F outcomes recorded for Phase-A-only AND
   Phase-A+B (the A/B is part of the deliverable).
3. Benchmark run against the table in §5.4 with a pre-run design checkpoint and
   post-run UNCERTIFIED verdict in `docs/logs/carousel-mclmc-sampling.md`.
4. Playbook (`docs/playbooks/sampling-diagnosis-playbook.md`) gains a fix-ladder entry for
   flow preconditioning with the measured gains (or the negative result).

## 8. References

NeuTra: Hoffman et al. 2019, arXiv:1903.03704. MAMS: arXiv:2503.01707. MCLMC:
Robnik & Seljak, PMLR v253. Ensemble MCLMC (many-chain precedent): MILE, arXiv:2502.06335.
Flow-preconditioned SMC (tier-2 successor to this plan): pocoMC, arXiv:2207.05652.
Flow-complexity guidance: arXiv:2511.02345. Note: no published NeuTra×microcanonical
combination exists as of 2026-07; this plan composes the two.
