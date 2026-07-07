# LAPS stock-take — 2026-07-06 (start of prior-init fix engagement)

Purpose: re-baseline before resuming the goal **prior-seeded LAPS on the real lens, no MAP/SVI warm
start**. Two independent audits were run: (A) evidence-status ledger (every headline claim re-checked
against on-disk artifacts), (B) fresh line-by-line divergence audit of
`src/gigalens_research/inference/laps_late_adjusted.py` (+`mams.py`, `blackjax_updated_utils.py`)
vs blackjax 1.5 (`adaptation/laps.py`, `laps_burn_in.py`, `util.py`, kernels) and the paper spec.

## 1. What is validated (artifact-verified)

- **Warm-init LAPS matches HMC on the demo lens: 0.92–1.04x** on all 8 mass params
  (`handoff/hmc_ref/overlay_summary.json`; note the previously quoted 0.95–1.03 was slightly rounded).
- CPU known-answer beds (T_iso/T_ill/T_corr/T_curve + T_rot cond-1e4 + banana d=12) pass to the finite-M
  floor; M-scaling unbiasedness to M=8192; Phase-2-corrects-Phase-1 (F3 table exact in
  `results/phase2off/summary.json`); x_i^2 switch fix decisive; k=1.5 pinned both ways
  (too-strict k=1 / premature k=3 on T_rot cold); cold N(0,I) start robust on Gaussian beds (0/10 fail).
- Internals validated with equal weight (D-tilde -> floor, EEVPD_obs/wanted -> 1, freeze latch, accept
  0.65–0.74); the Phase-5 adversarial audit corrections are all incorporated.

## 2. What fails (established)

- **Prior-seeded LAPS on the lens over-disperses ~20x median / ~340–370x max (theta_E)** vs HMC.
  Six falsified hypotheses (anisotropy, NaN-freeze, eps, P2-accept, 13x budget, switch+precond
  poisoning) all artifact-backed. Annealing v1 bridges **location** (logp -1.68e5 -> -369) but not
  **scale** (18x/213x unchanged). Mechanism: broad-basin scale-lock — ensemble-adaptive hyperparameters
  computed from a broad ensemble keep it broad; bistable vs the warm basin.
- Self-calibrated switch **false-fires** on broad quasi-stationary ensembles (documented, unfixed).

## 3. Divergences from the references — adjudication

Audit B classified every mechanism. Summary: the deliberate divergences are paper-over-blackjax
choices or documented regression fixes and are in good shape; three items are NOT:

**Justified & documented (spot-checks passed):** C=0.025 + F-law (paper, empirically best in F3);
alpha=2 (paper); centered equipartition (paper; correct far-from-equilibrium — blackjax's uncentered
version is only equal at stationarity); x_i^2 switch observable + delta=sigma/mu (paper; identity-x
never fires); self-calibrated switch floor + k=1.5 (laps-switch-resolution.md, empirically pinned);
chunked host-side early-stop (shard_map constraint, documented); staged Phase-2 bisection on settled
acceptance, growth 2.5, eps clamp, freeze persistence=2, FIX A eps0=L/N (docstring 44–106 + diag_p2accept
artifacts); Phase-2 no intra-trajectory partial refresh (efficiency-only, adjudicated); samples-only
estimator (paper Eq. 3); dropped splitR/superchains (diagnostic-only in bj, never gates stopping).

**Problem items:**
- **F1 (stale justification, cold-start-relevant): dropped gradient-aligned, equipartition-sign-flipped
  velocity init.** blackjax `laps_burn_in.initialize` (lines 77–156) sets every chain's initial velocity
  along grad log p with per-coordinate sign chosen by ensemble equipartition (overdispersed coordinate ->
  velocity toward the mode). This is blackjax's ONE purpose-built cold-start contraction mechanism and it
  targets exactly our failure (scale non-contraction). Our recorded rationale for dropping it predates
  the prior-init finding. It was never tested as a lever. **Untested lever; cheap.**
- **F2 (undocumented divergence): no NaN -> eps-halving in Phase-1.** blackjax halves eps whenever any
  chain NaN-rejects (`laps_burn_in.py:333-335`). Ours zeroes NaN chains' energy change, which biases
  EEVPD_obs LOW and makes the controller GROW eps — no countervailing brake. Not operative on the lens
  (0% non-finite measured) but a real cold-start robustness gap on harder targets; drop is documented
  nowhere.
- **F3 (rationale gap): Phase-boundary preconditioner from second-half pooled Var** vs both references'
  end-state instantaneous Var. Stated in code, never argued. On a still-contracting ensemble pooling
  inflates the metric further. (Known from diag_precond: the correct metric alone does NOT rescue
  prior-init — so this is robustness hygiene, not the root cause.)
- **(c)-minor:** `num_adjusted_steps` counts trajectories; blackjax's `num_steps2` is a gradient budget
  (~30–75x apart at equal numbers) — must be normalized in any compute comparison vs blackjax. RNG key
  reuse for init positions+momenta. Dropped bj diagnostics (entropy stream, per-step NaN rate) — a
  Phase-1 NaN-rate stream would be cheap and cold-start-relevant.
- blackjax has **no tempering/SMC machinery** we could have dropped; its cold-init handling differs from
  ours only via F1, F2, and first-step L=inf (we follow the paper instead; bj's own audit flagged that).

## 4. Evidence-status gaps (audit A)

- **128-chain operation (user goal): zero evidence.** `handoff/compare_runs.py` (warm,prior)x(512,128)
  exists, never run.
- Phase-2 lengths/thinning (200 traj, chunk=8, growth 2.5, thin=5) have mechanism narratives but **no
  empirical sweep** — fails the mechanism+numbers standard.
- Controlled falsification beds (aniso/rotated/banana/boundary) live only in an ephemeral job dir —
  **not preserved in-tree**, so those falsifications are not re-runnable from the repo.
- Core E-A grid single-seed (acknowledged); real-lens failure characterization single-lens/single-seed
  per diagnostic (magnitude >> noise, but unreplicated); HMC reference self-flags global
  `hmc_converged:false` (mass params R-hat<=1.0045 — fine for mass claims).
- Branch has **zero commits** — all code/docs/artifacts uncommitted or untracked; `laps_late_adjusted`
  depends on uncommitted `blackjax_updated_utils.py` additions (+141 lines, the entire Phase-2 kernel).
  Stale-step bug in production `mclmc.py` NOT fixed in this worktree (fix exists only in the aps-mclmc
  worktree; LAPS itself does not read the stale field).

## 5. Recommended next steps (ordered, cheapest-falsifiable first)

1. **Reference-faithfulness cold-start levers (pre-registered, flag-gated, cheap):** implement
   (i) gradient-aligned + equipartition-sign-flipped velocity init [F1], (ii) NaN->eps-halving [F2],
   (iii) end-state-Var preconditioner option [F3]; run prior-init on the lens (+1 controlled bed).
   Hypothesis for F1: basin selection happens in the first ~L/eps steps and contraction-directed initial
   velocities seed the tight basin. Prediction if true: width-ratio collapses from ~20x/340x toward
   O(1-3x). Falsifier: unchanged ratios => velocity init is not the basin selector (plausible — Maruyama
   refresh decorrelates velocity quickly; must be tested, not argued). This closes "does ANY dropped
   reference mechanism explain the failure?" before novel research.
2. **Fix the switch false-fire** (gate firing on equilibrium evidence, e.g. D-tilde near its floor or
   EEVPD_obs/wanted ~ 1, not just delta-small). Needed regardless of init mode.
3. **Annealing v2 / SMC bridge** (adjusted MAMS per temperature, per-beta metric, schedule dense near
   beta=1) — the real research bet, only after 1–2 leave a residual failure.
4. **Housekeeping in parallel:** preserve the controlled beds in-tree; run the 128-vs-512 compare
   (user goal, script ready); commit plan (sampler+tests / handoff / docs / artifacts); backfill docs for
   F2/F3 and the steps-unit semantics; (lower priority) small Phase-2 hyperparameter sweep to meet the
   mechanism+numbers bar.
