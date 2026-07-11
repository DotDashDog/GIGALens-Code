# Lab Notebook — Carousel-lens MCLMC sampling diagnosis

Why MCLMC mixes slowly / fails to converge on the `experiments/sim_carousel`
multi-plane lens, and what is / is not the cause.

**Last updated:** 2026-07-06

> One log per research area (see `../../AGENTS.md` → *The record*). Mode B (one agent
> proposes, the human grades). **Every claim below is `proposed (UNCERTIFIED)`** — the
> producing agent may not self-certify. Conclusions are written conservatively on purpose:
> if one is wrong, it should not silently mislead a future agent. Verify against the cited
> artifact before relying on any of this.

---

## Current state

System: `experiments/sim_carousel/prelim_sim_carousel.ipynb` — 1 deflector plane
(NFW_ELLIPSE + 2×EPL + Shear) lensing two source planes (2 shapelet sources @ z=1.432
seen in band 0; 1 Sérsic @ z=1.506 seen in band 1), `mode="lstsq"`, float64 likelihood,
**`conv_precision="float32"`**, 32 sampled non-linear params. Sampler:
`gigalens_research.inference.MCLMC_JIT` (8 chains).

Diagnostic artifacts (scripts + plots + cached runs) live in
`experiments/sim_carousel/_h1h2_diag/` (model rebuilt exactly via `build_model.py`).
Two durable code changes were made (see Log).

**Headline (proposed):** the original "slow chains / R̂≈70" was *dominated* by an
under-converged MAP, not a sampler defect. With a 4000-step MAP the run is no longer
broken (rank-R̂≈1.7, drift≈0.4σ) but is **ESS-limited** (min bulk-ESS≈12/16000). The
residual inefficiency traces to **smooth, curved, weakly-identified parameter
degeneracies** (a linear mass matrix can't whiten them) — NOT to numerical noise,
multimodality, conditioning, or the NFW profile.

---

## Claims register

### C-1 — Under-converged MAP was the dominant cause of the slow/"non-converged" chains
- **Status:** `proposed (UNCERTIFIED)`
- **Criterion:** if MAP quality is the leading term, raising MAP optimizer steps should
  close the chain↔MAP log-prob gap and drop max R̂ by ~an order of magnitude.
- **Evidence:** 500-step MAP sat ~1938 logp *below* the posterior bulk (chains lived at
  logp≈−292034 vs MAP −293643); |∇logp|≈4e4 at that MAP. Re-running MAP at 4000 steps:
  best_lp −293640→**−291702**, chain−MAP gap **+1600→+82**, drift **1.79σ→0.40σ**,
  max tfp-PSRF **71→11.6**. Scripts: `grade_new.py`, `grade_lp.py`.
- **Scope / caveats:** Establishes MAP-convergence was the *leading* blocker. Does **NOT**
  establish the run is converged (it is not — see C-2). MAP step count is not a principled
  stopping rule; "4000" is empirical, not derived. Particle count (n=64) was *not* varied;
  evidence (single connected basin, see C-5) argues against a particle-count/prior-volume
  cause but did not test it directly.

### C-2 — With a good MAP the run is ESS-limited, not "broken"; most of the apparent R̂ was a metric artifact
- **Status:** `proposed (UNCERTIFIED)`
- **Evidence:** On the 4000-MAP run, the classic tfp PSRF (unrooted, in unconstrained z)
  reads max≈11.6, but **rank-normalized split-R̂ (Vehtari 2021) reads max≈1.73**, while
  **min bulk-ESS≈12, tail-ESS≈23** out of 16000 draws. Rank-R̂ is invariant to the
  monotone bijector; the inflation came from prior-bound→z stretch + a non-robust
  estimator. Verified against ArviZ to 0 diff (`verify_diag.py`, `test1_rank.py`).
- **Scope / caveats:** rank-R̂≈1.7 and ESS≈12 are *both still failing* the usual gates
  (1.01/1.1; ESS≫chains). "Not broken" means the chains sample one distribution slowly —
  not that results are trustworthy yet.

### C-3 — The slow mixing is NOT caused by the posterior condition number (linear anisotropy)
- **Status:** `proposed (UNCERTIFIED)`
- **Criterion:** if cond(Σ)≈1e7 anisotropy were the limiter, MCLMC on a Gaussian with the
  *same* covariance would mix as badly.
- **Evidence:** A cond-matched multivariate-Gaussian reference (same Σ, same sampler
  settings/init) gave **rank-R̂ 1.00, bulk-ESS 7135/16000, xi max 50** — vs the real
  target's ESS 12, xi max 2e4. So the mass-matrix adaptation handles 1e7 conditioning
  fine. Script: `test2_gaussian.py`.
- **Scope / caveats:** Rules out *linear* conditioning as the bottleneck. By extension this
  argues `desired_energy_variance` tuning and per-chain-vs-shared mass matrix (both linear
  levers) are not the fix. The Gaussian used Σ from a not-fully-converged run; the
  conclusion is robust to small Σ error given the 600× ESS gap.

### C-4 — The difficulty is NOT prosaic numerical roughness in logp (the Hamiltonian is smooth)
- **Status:** `proposed (UNCERTIFIED)`
- **Criterion:** a smooth Hamiltonian gives leapfrog single-step |ΔE| ∝ ε³ with no floor;
  numerical noise gives an ε-independent floor; gradient autodiff↔finite-difference should
  agree to ~1e-8 (float64) if smooth.
- **Evidence:** With `conv_precision="float64"`, ΔE follows a clean slope-3 line to a 7e-10
  floor, and FD↔AD agree to ~4e-8. The notebook's **`conv_precision="float32"`** raises the
  logp noise to ~2e-4 (gradient ~3e-6 relative) and creates a ΔE floor at ~2e-4 — **but at
  the operating step size ε≈0.15 the float32 and float64 ΔE curves are identical (~0.05–0.1),
  ~500× above that floor.** So the float32-conv noise is real but inert for sampling.
  Scripts: `test_A1v2.py`, `test_A2.py`; plot `A2_energy_scaling.png`.
- **Scope / caveats:** Tested single-step ΔE at ~10 *typical-set* points (median behaviour).
  Does not characterise the heavy tail (rare high-curvature excursions, xi max 2e4) — those
  are attributed to geometry (C-5), not noise, but that attribution is inferential.
  Switching conv to float64 removes a confound but is **not** expected to fix ESS.

### C-5 — The residual slow mixing is weakly-identified, genuinely CURVED mass-model degeneracies; no multimodality found in the slow subspace.
- **Status:** `proposed (UNCERTIFIED)`. The curvature sub-claim was withdrawn (failed banana
  contour) then **re-established by a confound-free local-Hessian test** (2026-06-26 pm); see
  the new evidence block. Bijector-vs-physical origin of the curvature is still open (C-7).
- **CURVATURE — confound-free confirmation (local-Hessian eigenframe rotation):** at 9
  typical-set samples, the Hessian (autodiff HVPs, conv=float64) eigenframe of the top stiff
  subspace **rotates 10–40° (top-6; up to 85° top-4)** point-to-point, vs **0.0° for the
  cond-matched Gaussian null** and **0.0° float32-vs-float64 precision floor**; Hessian finite
  & exactly symmetric. Pre-registered: rotation ≫ floor + precision-stable ⇒ genuine
  curvature (not a noisy Hessian); met decisively. This test does NOT use `v*`, so it is free
  of the eigenvector-estimation confound that weakened the earlier profile argument.
  Script `hess_rotation.py`; plot `hess_rotation.png`. Caveat: rotation not monotone in Δs
  (expected — points differ in all 32 dims, not just the slow coord).
- **Criterion:** profile logp (re-optimising the other 31 dims) along the slowest
  eigen-direction is single-peaked ⇒ one connected ridge (not multimodal). A *curved* ridge
  would additionally show a banana in the plane of (slow direction, its bend direction).
- **Evidence (supported parts):** Slowest eigen-directions of Σ are *diagonal combinations*
  (slowest, ESS≈19: EPL e1 / shear γ2 / e2 — an ellipticity–shear degeneracy); profile along
  it is single-peaked, no barrier; 2-D sample scatter is one connected winding blob, not
  separated clusters (`test_CB.py`, `CB_ridge.png`). The EPL_Le "two ellipticity modes" are
  **one physical solution** (|e|≈0.30, PA≈34° from both) — a z-space prior-bound-stretch
  artefact (`phys_modes.py`).
- **2-D banana visualisation — only a MILD curve found (curvature is high-D, not a clean
  2-D banana):** logp-contour attempts in a (slow, bend) plane failed (blob; cheap ridge
  estimates unreliable — `banana.png`, `banana3.png`). A systematic search over the
  highest-variance sample directions found the most-curved 2-D *sample* marginal in the **NFW
  halo subspace (alpha_Rs × center_y)**, where the conditional-mean ridge bends only ~0.5× its
  scatter (`banana_found.png`). Interpretation: the 10–40° Hessian rotation and ~0.5σ 2-D bend
  are mutually consistent with *moderate* curvature **distributed** over the ~5–8-dim
  weakly-identified subspace, so no single 2-D marginal concentrates it into a dramatic banana
  (compounded by under-converged samples, ESS≈12–19). The 2-D-sample-banana falsifier assumes
  low-dimensional curvature and is therefore weak here; the local-Hessian rotation test (C-5
  evidence block) is the primary, dimension-agnostic evidence.
  The reliable profile-path ridge needs the lstsq VJP, which OOM'd once the GPU was busy.
- **CONFOUND (important):** the ridge-trace's slow direction `v*` is an eigenvector of the
  *sample* covariance from non-converged chains. For a Gaussian, conditional==profile only
  along a *true* eigenvector, so the 316-vs-30 conditional/profile gap could partly reflect
  `v*` being the wrong axis rather than genuine curvature. So that gap is **not** clean proof
  of curvature on its own.
- **Settled (2026-06-26 pm):** the local-Hessian eigenframe-rotation test (above) confirms
  genuine curvature confound-free, superseding the `v*`-dependent profile argument.
- **Scope / caveats:** Only the single slowest eigen-direction examined for multimodality;
  2nd-slowest (n_sersic-bearing) not traced. Multimodality ruled out only within the sampled
  slow subspace.

### C-7 — The curvature is PHYSICAL (persists in physical space), not a bijector/parametrization artifact
- **Status:** `proposed (UNCERTIFIED)`
- **Criterion:** Hessian eigenframe rotation of the *physical* log-posterior (logp(bij.inverse(x))
  − fldj) at the same points; rotation ≈ z-space ⇒ physical, rotation ≈ 0 ⇒ bijector-induced.
- **Validation gate (passed):** round-trip |inverse(forward(z))−z| = 5e-8; identity
  lp(z) = phys(x)+fldj(z) to 2e-5 (a flipped fldj sign would mis-match by ~2·fldj ≈ 70).
- **Evidence:** physical-space rotation median **14–19°, max 45°** vs z-space 18–46° — same
  order, both ≫ 0. So the non-Gaussian curvature is real in physical parameter space (an
  ellipticity–shear degeneracy bending into NFW position/scale), with the bijector adding
  *some* extra z-space curvature (z runs larger at top-4/8 — the e2 bound-stretch). Script
  `phys_hess.py`.
- **Scope / caveats:** The physical log-posterior includes the physical *prior's* curvature
  near bounds, so "physical" = likelihood degeneracy **+** physical-prior curvature, not the
  lensing likelihood in isolation (a likelihood-only Hessian would separate them — not done).
  9 points; the z-vs-phys ratio is noisy. **Implication:** because the curvature is real (not
  a coordinate artifact), a better bijector / reparameterisation will *not* remove it — the
  prior-robust fix must be geometry-adaptive (flow/transport warmup or position-dependent
  metric), or accept curvature-limited ESS and budget more draws.

### C-6 — NFW_ELLIPSE is not the source of instability; the data path has no bug
- **Status:** `proposed (UNCERTIFIED)`
- **Evidence:** ±6σ logp sweeps in sampling coords: no NaN/Inf, smooth (no kinks → lstsq
  not injecting jitter); NFW Rs/alpha_Rs/center_x are the *smoothest, most weakly-constrained*
  params. Data: STAT is variance (√STAT→error_map correct), MASK True=use, the −73846 outlier
  + 14 others are masked out, extra source9 HDU == MASK. Scripts: `probe_sweeps.py`,
  FITS checks in session.
- **Scope / caveats:** Smoothness shown on axis-aligned 1-D sweeps through typical/MAP points;
  NFW ellipticity (e1, e2) ARE stiff (large curvature) though smooth. Physicality flags
  recorded under Open questions.

### C-8 — Sample-column → parameter-name labeling was REVERSED; the minimal-case "NFW e1 multimodality" was a mislabel (and the full-case named attributions need re-checking)
- **Status:** `proposed (UNCERTIFIED)`. Bijector mapping is deterministically verified; the
  plotting-code impact and the full-case impact are *under verification* (subagents dispatched 2026-06-27).
- **Scope of this work:** new **minimal** system `experiments/sim_carousel/carousel_sampling_minimal_example.ipynb`
  (14 sampled params: NFW_ELLIPSE + Shear deflector, one source plane with src4=Shapelets(n_max=8)
  + src5=Shapelets(n_max=6), lstsq, float64 / conv float32). Mode A (orchestrator + subagents).
- **Test:** AD-grad each *named* physical output of `prob_model.bij.forward` w.r.t. the unconstrained
  input vector at the MAP, accessing outputs by dict key (no reliance on flatten alignment); FD cross-check.
  Correct labeling ⇒ output name at flatten-position `a` is driven by input column `a`.
- **Evidence:** the bijector is exactly **per-coordinate & monotone** (off-diagonal ratio 0.0 for all 14;
  FD: bumping any other column → Δ=0). BUT output `a` is driven by input column **(13−a)** — the sampler
  column order is the exact REVERSE of `param_names = flatten_param_names(forward(probe))`. True map:
  col0=NFW Rs, col1=alpha_Rs, col2=NFW cx, col3=NFW cy, col4=NFW e1, col5=NFW e2, col6=shear g1,
  col7=shear g2, col8=src4 beta, col9=src4 cx, col10=src4 cy, col11=src5 beta, col12=src5 cx, col13=src5 cy.
  Scripts `~/.claude/jobs/09d63727/tmp/verify_bijector{,_v2}.py`; plot `minimal_case_recheck_plots/T3_corrected_labeling.png`.
- **Consequence / WITHDRAWAL:** a within-session re-analysis had called col9 "NFW e1" and proposed
  "the minimal-case multimodality is a z-coordinate / bound-stretch artifact." Col9 is actually
  **src4 center_x**; that proposed claim is **WITHDRAWN**. Correctly labeled, the worst-mixing/bimodal
  param is **src4 center_x** (between/within sep 1.67), which has a `Normal` prior ⇒ **identity** bijector
  ⇒ the bimodality (modes ~4.33″ / ~4.45″, 2 chains vs 6) is **physical**, not a coordinate artifact.
  True NFW e1 (col4) is a middling mixer with a clean monotone z→phys curve.
- **Open / high-impact:** (1) does `PipelineReport`/cornerplot + per-param R̂/ESS labeling inherit this
  reversal? (subagent verifying + robust fix). (2) **Do C-5/C-7 (full 32-param case) name their slow
  directions via the same machinery?** If labels were reversed there too, the *geometry/curvature*
  evidence stands but the *parameter identities* ("ellipticity–shear degeneracy", "NFW position/scale")
  may be wrong — re-verify the full-case column↔name map with the same AD/FD test before relying on them.
  (3) real-multimodality vs unconverged-metastable nature of the src4 center_x modes (subagent).

### C-9 — Minimal-case slow mixing = a genuine but negligible-mass secondary basin in src4 center_x, sampled out of equilibrium because the MAP landed in it
- **Status:** `proposed (UNCERTIFIED)`.
- **Criterion:** profile logp over `src4 cx` (re-optimise the other 13 params at each cx); a real
  second mode ⇔ two interior maxima separated by a saddle with barrier ≥ 3 logp (pre-committed).
- **Evidence:** profile (cold-start AND warm-continuation agree to ~0.1 logp) → maxima at cx≈4.34
  (logp −119514.2, secondary) and cx≈4.46 (−119503.2, **global**), saddle cx≈4.38 (−119518.5) ⇒
  barrier ≈ **4.2–4.3 ≥ 3** ⇒ real second local maximum. Hop/dwell: 6 chains upper / 2 lower; **3
  transitions, all upward, 0 round-trips** ⇒ one-way drain, not equilibrated. Joint structure: split
  is ~7σ in `src4 cx`, all other params shift ≤ 1.9σ ⇒ a curved source-position↔lens/shapelet
  degeneracy, **not** a coherent second lens solution. Caustic: both modes same side, 0.9–1.0″ from
  the caustic; the 0.11″ inter-mode shift does **not** cross ⇒ not an image-multiplicity change.
  Plots `R_profile_cx`, `R_col9_hop_dwell`, `R_mode_joint_structure`, `R_caustic_check`; scripts in
  `minimal_case_recheck_plots/_scripts/`. GPU via `srun --overlap --jobid=55135874`.
- **Independent finding:** the stored 500-step MAP (−119514.93) sits in the **secondary** basin
  (cx=4.339); the profile global beats it by **~11.8 logp**. The `diag_qz` init (`scale_diag=1e-3`
  around z_best) therefore started all 8 chains in the minor basin; 6 escaped, 2 draining ⇒ the 6/2
  split is an out-of-equilibrium snapshot that **over-represents** the secondary mass.
- **Scope / caveats:** mass of the secondary basin not computed via Laplace — inferred small from the
  ~11.8-logp lower peak (the basin width would have to span ~e^11.8 to matter; not checked, but
  implausible). Barrier height carries a minor `conv_precision="float32"` caveat, but the float32 logp
  noise floor (~2e-4, C-4) is ~4 orders below the 4-logp barrier, so the *existence* of the second mode
  is robust (11.8-logp peak separation, reproduced 3 ways); only the precise barrier value would move.
- **Implication:** this is C-1 (under-converged MAP) with the mechanism pinned — **wrong basin, not just
  wrong logp**. Predicted prior-robust fix (PRE-REGISTERED for the rerun): converge the MAP into the
  **global** basin (multi-start / more steps) and start/disperse the chains there ⇒ the ~1e-5-mass
  secondary is rarely visited ⇒ max R̂ → < 1.1 and min ESS rises substantially. Falsifier: if a
  global-MAP start + dispersed init still splits with bad R̂, the shallow barrier is crossed often
  enough to matter (a real mixing problem), not just an init artifact.

### C-10 — Vanilla MCLMC cannot robustly cross/find the separated basin: STRUCTURAL, not a tuning issue
- **Status:** `proposed (UNCERTIFIED)`.
- **Question:** is the stuck-in-secondary behavior a tuning/budget issue or structural? Fixed-knob
  (adaptation OFF) MCLMC on the characterized src4-cx double-well, faithful kernel primitives
  (`_build_kernel_shardmap` + `isokinetic_mclachlan_smart`), chains started **on-ridge at z_best**
  (the secondary mode peak, logp −119514.9 — NOT the off-ridge cluster mean; that confound was caught
  and fixed). 8 chains × 10k steps. Metric: MFPT(secondary→global), escape = col9 crosses 4.40.
- **Pre-registered falsifier of "structural":** some knob setting gives uniform escape < 2000 steps.
- **Result — NOT met (structural confirmed):** pooled MM (best case, already covers both modes):
  L=1×→7/8 escaped (MFPT 378–8348, med 4578, 1 stuck); **larger L is monotonically WORSE**:
  2×→5/8, 4×→2/8, 8×→1/8, while energy error xi-max explodes 26k→364k. No L reaches uniform <2000.
  **MM contrast (L=1×):** within-mode cov →1/8, identity →0/8 (numerically broken, xi~2e10). So the
  *only* thing enabling crossings is the pooled MM's accidental gap-stretch (a preconditioner that
  already encodes both modes — unavailable in a real novel-multimodal case). **Basin-finding:** from a
  global-mode start, **0/8 chains genuinely visit the secondary basin** (2 brief 4.39 threshold grazes
  only) ⇒ MCLMC cannot *discover* a separated mode. Plots `D_escape_fraction`, `D_mfpt_vs_L`,
  `D_traces_by_basin`; data `D_lean_data.npz`; scripts `_scripts/{fixed_knob_mclmc,D_sweep_lean,D_plots}.py`.
- **Observed vs predicted:** predicted escape ≳ run length and weakly knob-dependent → observed escape
  IS run-length-scale & incomplete AND strongly knob-dependent in the *wrong* direction. Hypothesis
  confirmed and strengthened. Faithfulness check: pooled-MM L=1× (7/8, slowest 8348, 1 stuck) matches
  the production run (2 stuck, escaped ~8000).
- **Caveat (blind spot):** L swept at FIXED step size, so part of the large-L degradation is rising
  energy error (in adaptive MCLMC L and step co-tune). A fair L test would co-shrink step (more cost).
  This does not change the conclusion: the knob-independent evidence (within-mode MM 1/8, identity 0/8,
  basin-finding 0/8, no config uniform <2000) already establishes structural.
- **Implication:** robustness to this morphology needs an added mechanism, not a knob. Recommended
  minimal/low-slowdown candidate: **DE-MC / affine-invariant ensemble cross-chain jumps** interleaved
  with MCLMC (exploits the existing parallel chains; lets a stuck minority teleport to the global mode;
  exact with an MH accept). The same escape harness is a ready-made pass/fail testbed. (MAMS alone does
  NOT fix this; normalizing-flow / within-mode-MM do not either, per the contrast above.)

### C-11 — DE-MCLMC composite: validated UNBIASED mode-EQUILIBRATION, but it does NOT provide mode-DISCOVERY (and the production blocker is discovery)
- **Status:** `proposed (UNCERTIFIED)`.
- **Build:** standalone composite = real MCLMC kernel + two-group DE-MC ensemble jumps with a
  **re-randomized partition** each step and **Cholesky-whitened jitter** (`eps ~ N(0,b0²·within-mode-cov)`).
  Existing `MCLMCStage`/`mclmc.py` untouched. Code in `experiments/sim_carousel/de_mclmc_prototype/`.
- **Unbiasedness (analytic, D=10 unequal-weight 0.7/0.3 Gaussian mixture, honest identity MM):** PASS —
  recovers weights 0.69–0.72±SE, **invariant from exact-draw init** (no drift), per-mode moments exact,
  KS p=0.32/0.38; vanilla MCLMC stays trapped at 1.000. Two-group DE-MH is symmetric-proposal Metropolis
  ⇒ exact; re-randomized partition is a mixture of valid block kernels ⇒ still exact (re-validated).
- **Carousel double-well (honest within-mode MM, 16 chains, 500×K=20):**
  - BALANCED init: composite **equilibrates** toward correct ~all-global occupancy (0.50→0.88, 6 visible
    secondary→global jumps), unbiased direction. Controls: **refresh-only = 0 crossings** (momentum
    refresh is NOT the cause), vanilla = 0.56 (honest-MM MCLMC barely crosses) ⇒ crossings are the DE move.
  - ALL-SECONDARY init (= production: all chains start at MAP=secondary): composite **0 crossings,
    occ stays 0.0** ⇒ DE CANNOT DISCOVER a mode the ensemble doesn't already straddle.
  - Efficiency: DE acceptance only **~0.6%** on the carousel's CURVED degeneracy (jumps `z_i+(glob−sec)`
    land off-ridge); 2/16 chains still stuck after 500 rounds. Unbiased but jump-rate-limited.
  - Plots `C_all_configs.png`, `C_occ_trajectory.png`; data `C_carousel_data.npz`.
- **Implication:** DE = mass **equilibration** between *populated* modes (unbiased), **not discovery**. The
  production blocker (C-9/C-10: chains start in the wrong basin) is a **discovery** problem ⇒ DE alone does
  not fix it. Minimal discovery fixes: (a) global-basin MAP via multi-start + start chains there (secondary
  is ~1e-5 mass ⇒ DE then unnecessary for THIS carousel); (b) tempered burn-in to populate modes, then DE
  for equilibration (needed when modes have comparable mass — harder carousels). Open improvement for the
  equilibration role: higher DE jump acceptance on curved degeneracies (snooker / reduced-subspace jump).
- **Caveat:** the production-run partial escape (6/8) seen earlier used the *pooled* MM gap-stretch; with the
  honest within-mode MM neither MCLMC nor DE discovers ⇒ discovery here genuinely needs a dedicated mechanism.

### C-12 — DE jump acceptance is throttled by within-mode CURVATURE (not jitter/p_jump/weight); affine fixes won't help ⇒ DE-equilibration is the wrong tool for these posteriors
- **Status:** `proposed (UNCERTIFIED)`.
- **Carousel-faithful testbed** (two Gaussians fit to the real secondary/global clusters, driven by the real
  MCLMC kernel + honest global MM; structurally identical to the carousel composite, analytic & cheap).
  **GATE-1 FAILED to reproduce:** testbed DE acceptance **8.7% + 8 round-trips** vs real carousel
  **0.6% + 0** ⇒ a Gaussian surrogate is ~14× easier. Per protocol, did NOT tune a non-problem; diagnosed.
- **Weight ruled out (falsified):** sweeping w_sec→1e-5 *raises* acceptance to ~15% (not lower).
- **Curvature confirmed (model-free, kNN off-manifold ratio on real samples):** linear DE proposals land off
  the real ridge **2.3×(global)/7.2×(secondary)** more than off a same-covariance Gaussian (Gaussian control ~1);
  excess **grows with γ** (jump>local) = curved thin ridge; secondary (jump target) most curved; excess-kurtosis
  0.55/0.25. Plots `E_curvature_diag.png`, `E_gate1.png`; scripts `de_mclmc_prototype/{carousel_testbed,E_run,E_weight_diag,E_curvature_diag}.py`.
- **Implication:** jitter/`p_jump` are NOT the dominant killer; the cause is structural non-Gaussian within-mode
  curvature. **Linear/affine jumps (current DE, snooker, mode-matching affine) all walk off curved ridges ⇒
  cannot fix it.** Only NONLINEAR transport (normalizing flow / ridge-following coords) would — which the human
  is wary of (group experience). With C-11 (DE gives no discovery, the production blocker), DE-equilibration is
  doubly limited here. Vindicates the original C-5 "genuinely curved degeneracies" at the jump-proposal level.
- **Recommended pivot:** tempering (tempered burn-in / parallel tempering) crosses barriers via MCLMC's
  *gradient* dynamics (curvature-robust — follows the ridge) AND provides discovery ⇒ sidesteps both
  limitations. DE yields a rigorous negative result (linear ensemble jumps fail on curved-ridge lensing
  posteriors) + a validated unbiased equilibration kernel (only useful for ~Gaussian modes).

### C-13 — Kernel-hop (Normal Kernel Coupler) mode-jump: UNBIASED but knife-edge bandwidth + tiny-mode over-representation; curvature advantage untested; still no discovery
- **Status:** `proposed (UNCERTIFIED)`.
- **Build:** frozen-complement two-group kernel-hop — propose `z'~(1/|C|)ΣN(·;z_j,ε²M)`, independence-MH accept
  `α=π(z')q(z)/[π(z)q(z')]`. `de_mclmc_prototype/kernel_hop.py` (design note from literature: Warnes NKC 2001,
  Tierney 1994 indep-MH, ter Braak–Vrugt 2008 two-group; HPC web PDFs unparseable ⇒ labeled own-knowledge,
  no fabricated cites). Literature-faithful remedy for the knife-edge flagged: self-inclusive **Zhu 2019
  Sample-Adaptive MCMC** (keeps `q(z_i)` large at small ε; needs sequential single-site acceptance).
- **Validation (carousel-fit GAUSSIAN mixture, 16 chains):** UNBIASED — invariance-from-truth (w 0.46/0.41
  bracket 0.40, no drift), KS p=0.405. BUT (i) **knife-edge bandwidth** — accept ~0 for ε<1 and ε>2, only
  ε≈1 (3.8% < linear-DE 8.9%); root cause = self-exclusion Hastings gap (`q(z_i)≈0` at small ε since i∉C).
  (ii) **tiny-mode OVER-representation** (finite runs): wt=0.03→0.067 (~2×), wt=0.001→pinned ~1/16=0.0625 (~60×).
  (iii) **NO discovery** (single-mode init → w=0). (iv) testbed per-mode Gaussian ⇒ the **curvature** advantage that
  motivated kernel-hop is **UNTESTED**. Round-trips 122 vs linear-DE 6 — but from more attempts, not higher
  acceptance, on no-curvature geometry. Plots `F_validation.png`; code `kernel_hop.py`, `F_run.py`, `F_bandwidth_diag.py`.
- **STRATEGIC (anti-rabbit-hole, method-discipline §6):** THREE jump variants (linear-DE C-12, Cholesky-DE,
  kernel-hop) have now hit structural walls, and NONE addresses **discovery** — the production blocker (C-9/C-10/C-11).
  **REFRAME:** if the troublesome modes are negligible-mass (human experience: "real lensing multimodality
  almost always vanishes on convergence"; C-9 secondary ~1e-5), inter-mode **equilibration is not needed** — the
  requirement is **discovery** (reach the dominant basin) + **avoidance** (not get stuck in a minor one), served by
  tempering / multi-start init, NOT by mode-jumps. Mode-jump engineering may be the wrong tool for this problem class.

### C-14 — DE γ=1 teleport + snooker (faithful): UNBIASED, solve benign separated modes, but DO NOT beat curved cross-mode hopping AND pin tiny modes ⇒ REJECTED for the carousel
- **Status:** `proposed (UNCERTIFIED)`. Orchestrator-audited (human asked to revisit DE with periodic γ=1 teleport + snooker, after C-12).
- **Build (`de_mclmc_prototype/de_teleport/`):** γ=1 periodic teleport (ter Braak 2006; ordered pairs ⇒ SYMMETRIC ⇒ plain Metropolis); γ=1 *near*-teleport (partner `z_b`≈`z_i` ⇒ `prop≈z_a`, lands on-manifold; exact computable KDE Hastings ratio); snooker (ter Braak–Vrugt 2008; acceptance `min(1,(‖z'−z_c‖/‖z_i−z_c‖)^{D−1}·π(z')/π(z_i))`, the (D−1) radial Jacobian VERIFIED from DREAM-Suite source `Calc_proposal.m`/`Metropolis_rule.m`, PDFs unparseable on HPC ⇒ source-verified not own-knowledge). Curved banana-warp testbed GATE-1 = 0.68% within-mode linDE acc (carousel-faithful).
- **UNBIASED (empirical+analytic):** V2 invariance-from-truth — γ1 w=0.717, near w=0.696 (vs 0.70), no drift, per-mode moments/KS pass. **Snooker Jacobian necessity DEMONSTRATED:** dropping it biases w 0.66→0.99 (drains the minor mode), within-mode var 0.97→1.80 ⇒ the (D−1) factor is correct and required.
- **Solve benign well-separated modes:** easy-case round-trips γ1=115, near=950 (vanilla MCLMC 0; snooker 0 — it is a within-mode move, not a mode-jumper).
- **FAIL curved cross-mode hop:** at carousel curvature every affine proposal lands 6–700× off the on-manifold off-ridge scale (near cross-mode off-ridge 17 vs on-manifold 3 → 0.55% acc); 0 round-trips. **Orchestrator cross-check:** round-trips floor at 0 for ALL moves incl. the linear-DE baseline across b∈[1,6] (`audit_curvature_sweep.py`) AND on the real carousel (`comp_balanced` 0 rt) ⇒ the round-trip metric is uninformative in this regime (off-ridge geometry is the discriminator); the testbed is faithful, NOT over-hard.
- **FAIL tiny-mode (the human's worry):** γ1/near both PIN — wt=0.03→0.083–0.090, wt=0.001→~1/16=0.0625. Mechanism: a lone minor-mode chain can't be handed a minor→major difference vector (frozen complement has no same-mode partner).
- **Verdict:** well-supported NEGATIVE. The two walls bracket the affine family (residual→0 ⇒ kernel-hop ⇒ pinning; residual>0 ⇒ chord-off-ridge ⇒ curvature wall); no simpler variant avoids both. Code `de_teleport/{curved_testbed,de_teleport,offridge_diag,curved_gates,validate_easy,tiny_mode_T3}.py`; orchestrator audit `de_teleport/audit_curvature_sweep.py`.

### C-15 — Self-inclusive Sample-Adaptive MCMC (Zhu 2019): faithfully UNBIASED; the FIRST mode-hop to equilibrate COMPARABLE-MASS modes on the curved ridge (where every affine move fails) — but pins tiny modes & the COMPOSITE inherits unadjusted-MCLMC curvature bias ⇒ MERITS the GPU carousel
- **Status:** `proposed (UNCERTIFIED)`. Orchestrator-audited; BOTH red flags attributed AWAY from the SA move.
- **Build (`sa_mcmc/sa_move.py`):** Zhu Algorithm 1 VERIFIED from the machine-readable NeurIPS PDF. State = N points S; draw θ_{N+1}~q(·|S); delete j~Categorical(λ), `λ_n=q(θ_n|S_{-n})/p(θ_n)`. **Self-INCLUSIVE** ⇒ no "valley at your own location" (fixes C-13's self-exclusion knife-edge). Prop 1 detailed balance needs ONLY that q depends on the unordered SET ⇒ licenses a curvature-aware self-inclusive KDE *mixture* proposal (propose near an on-ridge point). Orchestrator verified the leave-one-out Gaussian rank-1 downdate and the KDE-LOO (exclude diagonal, 1/N normalizer) faithfully evaluate `q(θ_n|S_{-n})`, and the substitute/reject indexing on the augmented (N+1) set is correct.
- **UNBIASED (analytic Prop 1 + empirical):** C1 invariance-from-truth w=0.607 (truth 0.60), no drift, per-mode base moments exact; 7.6 flips/round on the CURVED ridge (affine moves floor at ~0).
- **Both empirical red flags attributed to the UNADJUSTED MCLMC kernel, NOT SA (orchestrator runs):**
  (a) along-ridge dim0 std COLLAPSE 0.65 vs 3.0 — vanilla MCLMC *alone* collapses to **0.20** at step 0.2 (recovers to 2.5 at step 0.05); **pure-SA (K=0) PRESERVES 2.87–3.00**. ⇒ MCLMC step×curvature artifact (`attrib_ridge.py`).
  (b) cross-mode WEIGHT bias at super-carousel curvature (b≥3, composite drifts to 0.89–0.92 from a truth init) — **pure-SA (K=0) holds 0.53–0.58; the composite drifts.** ⇒ unadjusted-MCLMC bias; the SA deletion step is exact and faithfully equilibrates to whatever (biased) effective distribution the kernel produces.
- **THE WIN:** at carousel-level curvature (b≈1.5, static 0.64% ≈ real 0.6%) SA re-equilibrates a WRONG populated init **0.30→0.608** with ~13× DE's cross-mode throughput (**9.7 vs 0.72 flips/round**); DE undershoots (0.497). First variant to mode-hop on the curved ridge.
- **Caveats:** (i) **PINS tiny modes too** (wt=0.001→1/16 mixture, 0.039 gaussian) — STRUCTURAL to ALL empirical-ensemble moves (`λ_minor=q(θ_minor|S_{-minor})/p→0` when a chain is alone), so it will NOT drain the carousel's ~1e-5 secondary. (ii) The composite "bias on curved targets" was largely an UNTUNED-STEP ARTIFACT. **EEVPD-tuning check (`evar_tune_check.py`, the catch I initially missed — the step_robust/attrib steps were hand-set, NOT energy-variance-tuned):** at b=1.5 the hand-set step 0.2 has EEVPD=mean(ec²)/D ≈ 1.4e5 = **~3e8× the 5e-4 target** (step 0.1 ≈1e6×; 0.05 ≈5e3×) — no adaptation would ever pick these. The EEVPD-tuned step at b=1.5 is **~0.0125** (16× finer), where the weight is ~0.55 (within MC scatter of 0.60) and the ridge dim0 recovers to ~2.6–3.0. So the gross step-0.2 weight drift / ridge collapse are artifacts of catastrophically-untuned coarse steps; with normal energy-variance tuning the composite sits in the (approximately) unbiased regime. The real carousel's adapted step ≈0.048 is its own EEVPD-tuned step and is already in that regime ⇒ no retuning needed; no MAMS. SURVIVING cost note: EEVPD-tuning on strongly-curved targets picks a VERY fine step (≤0.0125 at b=1.5, finer at b=3 — didn't reach target by 0.0125) ⇒ more integrator steps/trajectory (a speed cost, a property of MCLMC-on-curvature, not SA). NOT-YET-CERTIFIED: the residual weight at the tuned step (0.548 vs 0.60) is within run scatter but a longer run at the tuned step would certify it. (The earlier "finer-step-worse at b≥3" was a short-run R=400/800 artifact, gone at R=1500.) (iii) CPU surrogates are dynamically MILDER than the real carousel for cross-mode DE (DE not fully blocked here) ⇒ the true SA-vs-DE magnitude is a GPU question.
- **Verdict:** WINNER of the two; MERITS the GPU carousel for comparable-mass equilibration (mixture proposal, honest within-mode MM, **finer step than 0.2**). Code `sa_mcmc/{sa_move,curved_testbed,validate_sa_analytic,validate_curved,tiny_mode_test}.py`; orchestrator audits `sa_mcmc/{attrib_ridge,audit_equil_sweep}.py`.

### C-16 — Unifying: empirical-ensemble mode-hops UNIVERSALLY pin tiny modes; curved geometry degrades the UNADJUSTED MCLMC kernel itself
- Every cross-chain move built from the other chains' empirical positions (linear-DE C-11, Cholesky-DE, near-teleport/snooker C-14, kernel-hop C-13, SA-MCMC C-15) PINS a lone chain in a negligible-mass mode (over-represents at ~1/n_chains): the reverse/deletion proposal density at an isolated location vanishes. **Tiny-mode DRAINING is a DISCOVERY-class capability** (a proposal covering a mode independent of current occupancy) — NOT deliverable by an ensemble hop. This confirms C-13's reframe for the TINY-mode case, while C-15 shows COMPARABLE-mass equilibration on curved ridges IS solvable (SA, unbiased).
- Curved within-mode geometry defeats affine hops (chord off-ridge, C-12/C-14) AND degrades the UNADJUSTED MCLMC kernel itself (ridge-marginal collapse + cross-mode weight bias at high curvature; C-15 attribution). ⇒ on strongly-curved targets the base sampler needs a finer step or Metropolization (MAMS) independent of any mode-hop.

### C-17 — GPU carousel: SA-MCMC's CPU win did NOT transfer — it FROZE on the real curved posterior (worst of three); CPU surrogates are too mild
- **Status:** `proposed (UNCERTIFIED)`. Orchestrator GPU run (`carousel_sa.py` → `SA_carousel_data.npz`), real minimal-carousel posterior, real adapted (EEVPD-tuned) step ss0=0.0484, L0=10.58, honest upper_cov MM, 16 chains × 500 rounds, balanced init.
- **Result (occ(global), balanced init; truth≈1.0 since secondary ~1e-5):** SA-mixture **0.500** (0 crossings, SA sub-rate 3.8%) — FROZE at the init; vanilla MCLMC 0.563 (2 cross); linear-DE **0.695** (6 cross, 0.6% acc, reproduces C-11's 0.681). ⇒ **SA is the WORST of the three on the real carousel.** The on-ridge KDE proposal (bandwidth 0.2·upper_cov) that hopped freely on the CPU banana (7.6 flips/round, C-15) lands OFF the real curved ridge → 3.8% acceptance, all within-mode, ZERO cross-mode hops. All three give 0 round-trips and none drains the secondary ⇒ confirms the carousel is a DISCOVERY problem (C-9/C-10), not equilibration.
- **Lesson:** the CPU curved surrogates (banana warp) are DYNAMICALLY MILDER than the real lens posterior (which has higher-D correlated curvature + lstsq-amplitude structure); a CPU mode-hop win must be GPU-validated before belief. (The all-secondary discovery config died mid-run — hung srun, not needed: ensemble moves don't discover.)
- **MECHANISM MEASURED (`carousel_sa_diag.py` → `sa_diag.log`; a pre-registered packing/`q-p`-cancellation hypothesis was FALSIFIED and replaced by measurement):** (i) the DELETION is NOT the problem — secondary chains have FAR higher deletion weight λ=q/p (dλ(sec−glob)=+142 nats at bw=0.2; secondary per-point density 19 nats below global, packing ratio only 1.11× so the falsified "tight-packing offset" does not happen) ⇒ the deletion WANTS to drain. (ii) The block is the PROPOSAL: a KDE proposal (kernel bw²·upper_cov) lands catastrophically OFF-manifold — logp drop −19 (bw0.05) / −314 (bw0.2) / −2360 (bw0.5) even WITHIN a mode ⇒ the proposed point has the largest λ and is itself deleted (rejected) ⇒ frozen. upper_cov is far too coarse a kernel metric for the real stiff within-mode geometry. (iii) SA hops NOT by a proposal spanning the gap but by proposing a near-copy of an other-mode chain + deleting a this-mode chain. **Bandwidth sweep (`carousel_sa_bwsweep.py` → `sa_bwsweep.log`, bw 0.01→0.20, balanced init, 200 rounds): FROZEN at occ=0.500 at EVERY bw** — and at bw=0.01 sub-rate is 0.40 (high acceptance) yet ZERO draining ⇒ NOT a width/efficiency problem. **Measured structural mechanism: the proposal y pairs with its source chain θ_m (mutual nearest-neighbor) and that pair DOMINATES the deletion weights** — at large bw y is off-manifold (low p → λ_y max → reject), at small bw y is a near-copy (high mutual q → λ_pair max → y replaces θ_m, same mode). So every accepted substitution is a within-mode near-copy swap; the deletion never reaches the over-populated secondary despite λ_sec being elevated (+142). Worked on the CPU banana because 64 dense chains gave each proposal many same-mode neighbors (no single pair dominates); the carousel's 16 sparse, well-separated chains break that. ⇒ SA-mixture is structurally incapable of rebalancing modes here, at ANY bandwidth (measured, not asserted; the Gaussian-fit variant untested but should fail similarly — fits across the 19-nat gap → off-manifold). Does NOT condemn tempering (gradient-flow on a flattened target needs no cross-mode proposal). **Process note (method discipline):** three successive mechanism hypotheses — q/p packing-cancellation, then too-broad-kernel — were each FALSIFIED by measurement before the proposal-pair-domination mechanism survived the bw sweep.

### C-18 — Tempering (tempered burn-in + parallel tempering): the FIRST method to DISCOVER across curved barriers AND DRAIN tiny modes; unbiased; tempered-burn-in suffices for the carousel's 1e-5 secondary with NO replica cost — PENDING GPU validation
- **Status:** `proposed (UNCERTIFIED)`. Subagent build (`tempering/`), orchestrator-audited. CPU testbeds only.
- **Construction:** tempering log p→β·log p scales the MCLMC force by β ⇒ stationary ∝ p^β (own-knowledge, direct from the verified MCLMC dynamics arXiv:2303.18221). Tempered BURN-IN = anneal β small→1 then sample plain β=1 MCLMC (tempered samples DISCARDED ⇒ no importance weights ⇒ unbiased cold by the same argument as plain MCLMC). PARALLEL TEMPERING = replica ladder, swap accept `min(1,exp((β_i−β_j)(E_i−E_j)))` (detailed-balance-exact for ∏p(x_r)^{β_r}; cold replica marginal = p). **Subtlety (flagged + handled):** the EEVPD-tuned step is β-dependent; the β=1 step is too coarse for the HOT anneal rungs on a curved target (anneal-max EEVPD ≫ target) ⇒ must tune to the ANNEAL-max EEVPD, not equilibrium (curved faithful step 0.05, not the knee 0.19).
- **Gates (CPU):** A discovery — tempered burn-in occ 0→0.51 easy / 0→0.58 curved, vanilla 0 (PASS); comparable-mass WEIGHT — one-shot FREEZES OUT (~0.51 vs 0.70, cold ensemble quantizes at k/n) → PT fixes it **0.6986±0.0122** (PASS). B tiny-drain — one-shot unreliable for 1e-3 (freeze-out) → PT drains **1e-3→0.00125±0.00017, 1e-5→0.0** (PASS — the KEY differentiator; every ensemble move PINS at ~1/n, C-16). C unbiased cold — invariance-from-truth, moments/KS pass. D curved barrier — crosses where affine DE got 0 round-trips (PASS, budget-limited ~0.55 vs 0.6). E cost — tempered burn-in adds NO sampling-time replicas; PT is R× replicas.
- **Mechanism split:** for the CAROUSEL (secondary ~1e-5, a discovery problem), tempered BURN-IN alone gives discovery + drains the 1e-5 mode to ~0 with NO replica multiplier. PT is the robust tool when modes are COMPARABLE-mass / near-1/n (R× cost). Unlike the ensemble hops, tempering crosses curved barriers via MCLMC's GRADIENT flow on the flattened target (follows the ridge) ⇒ mechanism is NOT the chord-off-ridge that defeated affine moves AND not the KDE-off-ridge that froze SA (C-17) ⇒ MORE likely to transfer to the real carousel.
- **CRITICAL caveat (C-17 cautionary tale):** the CPU win MUST be GPU-validated on the real carousel before belief — SA also won on CPU and froze on GPU. Plus: GATE-1 curvature 1.85% (slightly milder than carousel 0.6%); single-seed PT. Code `tempering/{tempered_mclmc,parallel_tempering,curved_discovery,pt_weight,pt_drain,tiny_drain,drill_*}.py`.
- **ADDENDUM 2026-07-10 (archaeology — the PT leg's GPU validation was RUN and FAILED, unlogged):** three GPU PT runs on the real minimal carousel (2026-06-28, `carousel_pt.py`, no design checkpoint, never logged) all show ZERO cold-rung cross-basin transport in 90 rounds — cold occ pinned at init (0.000/0.500/0.000 vs truth ≈1.0) despite healthy AVERAGE swap acceptance (0.38–0.72) and hot-rung kernel crossing. The C-18 PT drain claim currently holds ONLY on CPU toys; on the real MINIMAL carousel the naive geometric power-path (p^β) PT did NOT transport. HUMAN CONTEXT (2026-07-10): the minimal carousel is a more pathological target (1e-5 secondary) than the dPIE production case (~10:1 modes) and the June-28 implementation was by a less capable agent — the failure is NOT read as evidence against PT on the dPIE target. See Log 2026-07-10 archaeology entry.

### C-19 — GPU carousel: tempering's DISCOVERY transfers (vanilla can't), but tempered-burn-in alone FREEZES OUT at ~0.6 occupancy (truth ~1.0) ⇒ full drain needs PT / adaptive tempering
- **Status:** `proposed (UNCERTIFIED)`. Orchestrator GPU runs on the real minimal-carousel posterior, all-secondary init (the bad-MAP production scenario). `carousel_tempering.py` (fixed step, CONFOUNDED) → `carousel_tempering_adapt.py` (clean, uses the USER's `step_size_adapt` per-step).
- **Step-tuning confound found + fixed:** the first run EEVPD-tuned the step (0.069) at the SECONDARY init; it was far too coarse for the annealed/global geometry (per-stage EEVPD blew up to 8.75 ≈10⁴× target for β≥0.17) ⇒ high-β stages biased. FIX: drive the anneal with the project MCLMC step-size adaptation (`mclmc.step_size_adapt`, defaults DEVAR=5e-4/TRUST=1.5/decay=149/151), per-chain per-step. Result: per-stage EEVPD pinned at ~5e-4 at EVERY β (step auto-tightened 0.23→0.029 across the ladder). Confound removed.
- **DISCOVERY TRANSFERS (the key result SA lacked, C-17):** from all-secondary (occ_global=0), tempering crosses the barrier to occ_global **0.628**; vanilla MCLMC from the same init stays **0.000** (trapped, C-10). Tempering's gradient-flow-on-flattened-target crosses where every ensemble move (incl. SA) structurally could not.
- **FREEZE-OUT is the genuine ceiling (NOT the step bug):** at faithful EEVPD the cold occupancy plateaus at ~0.56–0.63 (truth ~1.0; ~6/16 chains stay in the secondary). The occupancy freezes near the decoupling-temperature tempered weight instead of sharpening to ~1.0 as β→1 — exactly the C-18 CPU freeze-out, now reproduced on the real carousel. Tempered BURN-IN alone under-drains.
- **Next:** PT (continuously-mixing cold replica drained tiny modes to truth on CPU, C-18) OR adaptive continuous tempering / path sampling (Yao et al.; `tempering/adaptive_path_sampling_report.md`) which removes PT's R× replica cost — gated by its Appendix-C log-z scale-collapse risk (cheap Gate-F first, given the ~1e5 logp scale).

### C-20 — APS-on-MCLMC (Yao continuous tempering) on the GPU carousel: does NOT collapse once logT carries the Laplace evidence offset; but cold-end mixing is too slow (low ESS, 3/16 chains stuck at base) for the cold draws to characterize the posterior (single-seed diagnostic)
- **Status:** `proposed (UNCERTIFIED)`. Orchestrator build + GPU runs; independent code audit + independent grader (proposer≠grader). Mode B.
- **Build:** Option-A augmented sampler (temperature coord `a` IN the parameter space, ONE MCLMC kernel on the (θ,a) joint), `de_mclmc_prototype/tempering/apt_carousel.py` + `carousel_aps_run.py`. FROZEN θ-block inverse-mass-matrix = cached honest `upper_cov` (NOT re-adapted from identity — re-adapting a linear metric on the curved carousel is the C-3/C-5 bottleneck), a-block = scalar; per-step EEVPD `step_size_adapt` (faithful copy of `mclmc.step_size_adapt`, used GPU-validated in `carousel_tempering_adapt.py`); base B = N(global-MAP, upper_cov) matched-covariance Laplace. Host side (`apt_core.adapt_loop`: TI / 41-basis regression / Pareto-k̂) untouched. Independent audit: faithful (one fix applied — `ss_max` step cap 1.0 not inf).
- **THE decisive bug + fix (Gate-F made concrete):** the gigalens log-posterior is UNNORMALIZED (logp≈−119503 at MAP); a normalized Gaussian base has logB≈+60, so logT−logB≈−1.2e5 = the log-evidence. First full APS run (no offset) = **total Appendix-C collapse** (frac_base≈0.997 every loop, frac_cold=0, ZERO cold draws, log z(1) mis-estimated −1000 vs true −1.2e5, k̂ degenerate): the e^(−1e5) base-pinning starves the pseudo-prior bootstrap. **Matched COVARIANCE does NOT fix this — matched NORMALIZATION does:** subtract C = logp(MAP)−logB(MAP) = **−119562.66** (= the Laplace log-evidence) from logT so logT'(MAP)=logB(MAP) and |log z'(1)|→O(10). A constant offset leaves the λ=1 distribution unchanged ⇒ cold draws still the true posterior; log z'(1) becomes the true-vs-Laplace evidence correction. Toys (T0–T4, Gate D) didn't need this (near-normalized, |log z|~O(10)).
- **Result WITH offset:** NO TOTAL collapse, but a MARGINAL regime (k̂ is sample-size-sensitive — the short run was optimistic). log z'(1) STABLE ≈−4.5 in BOTH runs (NOT −1000); realized EEVPD faithful (worst-bin ~3e-6 ≪ 5e-4 ⇒ NOT numerical heating); EEVPD smoke transfer clean (NaN-frac 0, full a-tour).
  - n_loop=5/250-results: k̂ [0.21,0.03,0.46,0.03,0.03] (looked clean, max 0.46); frac_cold 0→0.225→0.035→0.299→0.122; 244 cold draws.
  - n_loop=6/400-results (LARGER, more reliable): **k̂ [0.65,−0.37,0.70,0.92,0.25,0.57] — SPIKES to 0.92 / 0.70, intermittently EXCEEDS the 0.7 Pareto threshold**; frac_cold 0→0.126→0.185→0.170→0.104→0.069 (oscillates, NOT converging); 221 cold draws. ⇒ the pre-registered falsifier ("PERSISTENT k̂≫0.7") does NOT fire, but this is NOT a clean pass — TI/IS weights are borderline-unreliable on the curved ridge with a single-Gaussian base + linear metric.
- **Grader RED FLAG (plots-before-metrics; scalars were blind):** cold draws are BIMODAL in src4_cx — main ≈4.45 (152 draws) + a **38%-mass cluster ≈4.48** (92 draws, NFW_Rs>2.4), displaced 1–2 base-σ COHERENTLY across ~all 14 params; BOTH >4.40 so occ_global=1.0 / k̂ missed it entirely (NOT the C-9 <4.40 ~1e-5 basin). Cold draws FILAMENTARY (low cold-ESS; threads bridge the clusters; R-hat/ESS ill-defined on APS pooled cold sub-selections).
- **Disambiguation (GPU `cluster_check.py`): cold draws sit at CORRECT density (NOT θ-lag contamination), but the apparent geometry is NOT reliably resolved.** All cold draws within ~13 nats of MAP (= typical set MAP−D/2≈−119510); the two "clusters" at the same density (med −119511.1 vs −119510.8); straight chord between centroids dips 791 nats ⇒ the gap is CURVATURE not a barrier.
- **CORRECTION (GPU `chain_trace.py` + user grading, plots-before-metrics): do NOT read "thin curved ridge" or "two clusters" off the cold draws — they are dominated by SLOW PER-CHAIN WALKS (low ESS).** Coloring the cold draws BY CHAIN shows each "worm" is ONE chain's autocorrelated crawl; src4_cx traces oscillate with ~100–300-step correlation times (a few effective samples/chain); **3/16 chains stay stuck at λ=0 (base) and never reach the cold target**; chains DO overlap (not trapped in disjoint regions) and a single chain's src4_cx span (0.045) EXCEEDS the inter-"cluster" gap ⇒ the 38% "second cluster" / "152:92 split" is within-walk autocorrelation, NOT a robust posterior weight. ROBUST facts only: cold draws are at correct density + a real positive src4_cx–NFW_Rs correlation (a degeneracy direction). NOT resolved: thin-ridge vs broad-band vs mild-bimodal — cold mixing too slow to say. This IS the C-5/C-10 curved-degeneracy slow-crawl (linear metric can't whiten the curvature), now inherited by APS's cold end.
- **Net:** APS-on-MCLMC + the evidence offset AVERTS the TOTAL Appendix-C collapse on the real carousel at a faithful step (the carousel-decisive risk), but sits in a MARGINAL reliability regime (k̂ intermittently >0.7, frac_cold non-converging) because it INHERITS the underlying curved-ridge ESS limitation of linear-metric MCLMC (C-5/C-7/C-19) — it neither creates nor cures it. Trustworthy ridge weights need a curvature-aware metric / better base. log z'(1)≈−4.5 is a bonus marginal-likelihood-vs-Laplace estimate (stable, the most trustworthy single number here).
- **Independent multi-start MAP** (32 starts) corroborated this geometry confound-free: global MAP −119502.93 (beats stored secondary −119514.9 by ~12, matches C-9's global), but only **3/32 starts reached within 5 logp** (median ~24 logp short) — gradient ascent crawling/under-converging on the curved ridge, NOT a clean multimodal landscape.
- **Scope / caveats (do NOT over-read):** single seed, diagnostic scale (5 loops, 244 correlated cold draws); ridge weights unreliable; EEVPD smoke only probed λ≤0.885 so step adequacy at λ→1 (cold region) is UNTESTED; this run inits θ from the global-MAP base so it does NOT test discovery-from-wrong-basin (C-19 all-secondary) — occ=1.0 here = "stays in the dominant basin", a DIFFERENT claim from the C-19 freeze-out; no same-harness vanilla-MCLMC baseline yet (C-10/C-19 establish vanilla on this model). Curvature-aware metric / mixture-or-warped base / multi-seed + more loops are the natural next steps for trustworthy ridge weights.

---

### C-21 — Catastrophic MCLMC energy-error (ξ) spikes on the NEW complex carousel are a QUANTIZED 2-D lattice in the EPL_Lf perturber's (theta_E, gamma); the global step-size suppression is downstream of a few dozen such events

- **Status:** `proposed (UNCERTIFIED)` — awaiting grader inspection of `batchA_diag/` artifacts.
- **Scope:** NEW complex carousel, **NFW_ELLIPSE_SLOPE** parameterization, run
  `experiments/sim_carousel/messy_tests/just_map/mclmc` (8 chains, 10k burn-in + 10k results,
  seed 42, max R̂ 1.027, min bulk-ESS 482). Covers the *carrier* of the ξ spikes; does **not**
  yet establish the *mechanism* (aliasing vs physical caustic — Test 1/3 below pending).
- **Evidence / artifact:** `experiments/sim_carousel/messy_tests/just_map/batchA_diag/` P1–P8;
  analysis scripts under this job's tmp (to be promoted).
  - Tuner is globally suppressed: tuned `eps=0.1255` identical to 1e-17 across all 8 chains.
    Results-phase `max(ξ)=2.75e8`, `frac(ξ>10)=0.0065`. **Top-8 burn-in steps carry 74% and
    top-80 carry 99.8% of Σξ**; `mean(ξ)/mean(ξ|ξ<10)=3.7e4` ⇒ a few dozen events dominate the
    energy-error variance the tune3 adaptation targets, forcing `eps` down for all 80k steps.
  - Spikes are diffuse in the worst-ESS coords (source/lens **positions**) and in the whitened
    eigenbasis, sit **mid-marginal** (not at prior walls — exonerates the ellipticity bounds),
    are **forward-moving** (not reflections), and the inverse mass matrix is frozen in the
    results phase ⇒ NOT a 1-D funnel / curved-valley / init / rotating-metric disease.
  - Spikes **are** quantized: a 2-D **lattice** in EPL_Lf `(theta_E, gamma)` **only** (the other
    EPL is clean; ellipticity/centers diffuse). γ(z) comb centers −5.040/−4.851/−4.663/−4.455,
    spacing ≈0.189 (near-perfectly periodic).
  - Source read of `gigalens/.../mass/epl.py`: the only `niter`-dependent part (the angular
    recurrence) depends on ellipticity `f=(1−q)/(1+q)` and `t=γ−1` but **not on theta_E**; the
    observed banding is in theta_E (present) and ellipticity (absent) — the *opposite* pattern.
    ⇒ series truncation is excluded; the carrier is the radial `(b/R)^{γ−2}` / critical-curve
    geometry, which depends on theta_E (via `b`) and γ and can alias against the render grid.
- **Doubt report (mandatory):**
  (i) *Endpoint-blind.* All localization uses the trajectory endpoint; a stiff region met
  *mid-trajectory* (leapfrog L≈29) is fully consistent and not yet proven the generator — the
  Test-1 dial-scan removes this by probing the likelihood directly.
  (ii) *Aliasing vs physical caustic unresolved.* The lattice is consistent with an
  un-supersampled (`supersample=1`) pixel-aliasing comb (sys60 disease-(i) analog at pixel
  pitch) OR a real caustic crossing bright pixels; Test 1 (supersample collapse) discriminates.
  (iii) `conv_precision="float32"` not yet excluded as a contributor (Test 3 float64 arm).
  (iv) Single seed; the Gaussian-clone (Batch C) that would size the shape-limited ESS floor is
  not yet run — the ESS/eps numbers are one realization.
- **Proposed by / on:** Claude (Batch A) · 2026-07-06   ·   **Grader:** _pending_
- **CORRECTION (2026-07-06, C-8 trap):** Batch A within-component parameter NAMES were wrong — I labeled z-columns in `_unique` *insertion* order, but the sampler uses **sorted-key** (JAX-tree-flatten) order (confirmed empirically by a per-column perturbation test through `pm.bij.forward`; map in `z_names_TRUE.json`). Column *indices* and all numbers are unaffected; only names within each component were permuted. Corrected identifications: **the ξ lattice is in EPL_Lf.center_x / center_y (perturber POSITION), NOT (theta_E, gamma)**; the slowest/widest direction (softest eig, loading 0.998; min bulk-ESS 482) is **src9.e2 (Sérsic source ellipticity)**, not a center; the worst-ESS subspace is a mix of weakly-identified source-shape (src9 e1/e2), source-shapelet centres, and mass shape/ellipticity — the Batch A 'positions dominate' phrasing is withdrawn. At the max-ξ draw EPL_Lf.e1=0.4977 (pinned at the +0.5 prior wall) and the perturber sits ~3σ off its center-x prior. Corrected plots: batchA_diag/P10_EPL_Lf_CORRECTED.png, P11_position_lattice.png.

### C-22 — GATE L: the carousel's main basin is NOT Laplace-approximable at any point reachable by production-scale polish (3 negative Hessian eigenvalues at −1.3e-3·λ_max at the best reachable point; nat-grad 1898 after 3000 whitened steps; independent 1024×4000 production MAP tops out 20.6 nats lower; true-mode PSD status UNTESTED — the stationary point was never reached); pipeline-realistic Laplace evidence weights under-count the pocket 18×; cross-mode Laplace/t jump acceptance ≈ 0 at ANY weight (oracle included); 1024-start multistart MAP finds the pocket 0 times and the main in-basin band 2 times — mode-enumeration + local-Gaussian jump designs are structurally out FOR THIS POSTERIOR; annealing family indicated as mainline (pre-committed M3-falsifier routing; human decision pending)

- **Status:** `proposed (UNCERTIFIED)` — grader result-pass rd-4 CERTIFY-RECOMMENDED (as amended); design pre-approved rd-3; human certification pending.
- **Scope:** Link 1 of the jump-mixture pipeline chain only; the two KNOWN modes of the dPIE carousel; this prior; equilibrium-state acceptance; single seed (0); MAP = production config (AdaBelief 1e-2, 4000 steps).
- **Evidence:** `experiments/flow_precond/carousel_gate_l_out/` (summary JSON, npz, 4 pre-registered PNGs); Log entry "2026-07-09 (carousel GATE L RAN…)"; pre-registered checkpoint "GATE L" with 3 grader rounds.
- **Key numbers:** main H: 3 negative eigs, λ_min = −1.3e-3·λ_max (5 orders above noise), nat-grad 1898 after 3000 polish steps; KL(emp‖Laplace) 157 (main) / 69 (pocket) nats; w̃_P = 0.0052 vs truth 0.0957; cross-mode ᾱ = 0/1024 in all four pass-eligible cells; P3 translation 0.10%/0.035% (inside its pre-registered volume-ratio band — mechanism confirmed); M3 final: 0 pocket / 2 main / 1022 stragglers.
- **Blind spots (named):** third-mode risk unobservable here; pocket Σ_emp rests on chain-segregated draws (benchmark robustness check agrees); acceptance measured at equilibrium states, not within a running sampler; every mode-local measurement is anchored at points produced by THIS polish apparatus from these basin medians (seed-family dependence unobserved); the "18×" is a property of the pipeline-realistic construction (non-stationary anchor + 3 floored axes inflating Σ_M's log-det), not of "Laplace weights" in the abstract.
- **Register caveat (grader rd-4):** the prior-record "Laplace pocket-mass proxy 5.4%" is NO LONGER a usable anchor — GATE L's pipeline-realistic construction gives 0.52% (10× apart; provenance of the old number not reconstructed); NEITHER is validated. Any future Laplace-mass claim must re-derive from scratch.

---

### C-23 — GATE PT-0 (dPIE carousel PT-MCLMC pilot): mechanism diagnosis COMPLETE — no entropic starvation on either tempering path, no cross-basin swap suppression; transport failure localized to swap-back cadence (K ≪ IAT), end-pair ladder disconnection, and ss-cap binding
- **Status:** `proposed (UNCERTIFIED)` — result grader rd-1 amendments applied; CERTIFY-RECOMMENDED as mechanism diagnosis only. 2026-07-11, artifacts `experiments/flow_precond/carousel_gate_pt0_out/`, Log entry "GATE PT-0 RAN".
- **Content:** power-path tempered-mass profile Δ(β) flat within ±0.5 (se ≤ 0.9) nats over β ∈ [0.01, 1] (Gaussian-model −8.4-nat starvation prediction WRONG — hypothesis failure); likelihood path +1.5…+2.9 nats (pocket mildly enhanced hot; m_prior = 0.9944). Cross-basin swap acceptance ≈ same-basin (0.18–0.34) — the June-28 minimal-carousel suppression signature is ABSENT here. Transport failure decomposed: swap-back (K = 10 ≪ measured IAT(u) 11–202; label mobility ~3 rungs/900 rounds vs ~13–15 free-walk), likelihood-ladder end-pair disconnection (hottest pair acc 0.000, coldest 0.005–0.05; measured swap-cost 23 nats over 11 pairs, end-concentrated), ss_max = 1.0 cap binding at hot rungs on BOTH B arms + control (EEVPD below band).
- **Scope:** this posterior only; NO working sampler demonstrated; NO pocket-weight value; W-3 bracketing untested (B3 canceled, op-8); within-basin ESS not certified; IAT transfer confined→pooled metrics is order-of-magnitude; K/(K+IAT) closure is a model.
- **Open findings:** cold pocket weight may be ~0.3–0.4 (hot-end direct occ 0.379 ± 0.046 reconciles with the flat TI profile only under a ~0.3–0.4 cold anchor; alternatives: 1500-step hot-end transient; hottest-rung EEVPD 5.0 dynamics error) — MAMS64's 9.6% doubly suspect; adjudication = a converged PT-0b cold chain.
- **PT-0b measured inputs (NOT validated conclusions):** 21/24-rung equal-cost ladders (`ladder_design_{power,lik}.json`); IAT-derived K; ss-cap re-derivation; β ≥ ~0.36 short-ladder option (kernel crosses basins at β = 0.6 at 17–37%/1500 steps). Fresh design checkpoint + grader required.

## Design checkpoints (criteria awaiting approval)

- **Run: carousel GATE PT-0b — short-ladder power-path PT-MCLMC transport certification
  on the dPIE carousel (routed continuation of GATE PT-0; C-23's three measured knobs
  applied).**
  **Status: grader rd-1 NEEDS-MORE (2026-07-11) — 4 blocking + 4 advisory, ALL APPLIED
  in-place (ss_max justification corrected to artifact values 0.54–0.84 with W-b3/F-b3
  reframed as ladder-health, cap fix MOOTED by the short ladder; every outcome zone
  routed incl. F-b5 ≻ F-b2 precedence and the restored PT-0 flux-limited/2–3σ
  readings; W-b2 statistics pinned incl. the POWER clause se_comb ≤ 0.06 and
  adjudication-only-if-CI-excludes-a-candidate; extension committed + diff audit
  recorded below; IAT 11.4–45.6, ā = 0.53 erfc provenance, rung-0 leakage 25.6/62.6%
  cited, wall margin noted). Grader verified by recomputation: ladder knots + 4.4708-nat
  cost integral, K* derivation, RT/flip/discovery arithmetic, 7.78 s/round.
  rd-2 (2026-07-11): CERTIFY-RECOMMENDED to LAUNCH at 79cdccd — independent B4/env diff
  audit CLEAN (properties a–e verified at file:line); ladder + erfc/flip/wall/headroom
  arithmetic independently reproduced; conditional numeric fix APPLIED (RT bound
  848→755, total lower 170→190; scoring keys off the 300 point prediction, unaffected).
  Scope: this exact 4-arm config only; β < 0.3594 transport and shared-systematic
  unbiasedness remain untested. LAUNCHING under the engagement mandate.** Same script `experiments/flow_precond/carousel_gate_pt0.py`
  (audited lineage; PT-0b config via the recorded env overrides + one small extension:
  a power-path all-main arm `B4` and env overrides for K/NSYS/ROUNDS/ss_max — extension
  audited by diff before launch); outputs tagged `_pt0b` via GATE_PT0_TAG_SUFFIX; fresh
  4 h allocation, 4 arms on 4 GPUs in parallel.
  **Claim under test + classification.** Stochastic-estimator behaviour; the SINGLE link
  this run tests: with the three PT-0-measured knob fixes applied (ladder spacing, swap
  cadence vs IAT, ss cap), power-path PT-MCLMC achieves label transport, cold-rung
  basin mixing, and TWO-SIDED occupancy bracketing on the dPIE carousel within a
  ~16.5k-step/chain budget. Explicitly UNTESTED: absolute weight truth (only bracketing
  + reproducibility); within-basin ESS certification; efficiency frontier; other
  lenses; the likelihood path (deprioritized per C-23).
  **Cause hypothesis.** PT-0's zero label transport was caused by (i) K = 10 ≪ IAT(u)
  (swap-back), (ii) ~2 nats/pair average spacing with end-concentration, (iii) ss_max
  cap binding at hot rungs — NOT by basin-mass starvation (profile flat) or cross-basin
  rejection (cross ≈ same). Fixing (i)–(iii) at FIXED kernel and swap machinery should
  restore transport at the rate of the revised flux model.
  **Config (all values derived from PT-0 measurements):** power path; ladder = the
  equal-cost R = 6 knots over β ∈ [0.3594, 1] from `ladder_design_power.json` restricted
  to the measured crossing-capable range: **[0.3594, 0.4388, 0.5373, 0.6598, 0.8116,
  1.0]** (0.894 nats/pair ⇒ predicted adjacent acceptance ≈ 0.53 via the erfc(s/2)
  Gaussian swap model — provenance: that model reproduces B1's measured pair
  acceptances from measured pair costs, e.g. cold pair 1.87 nats → 0.186 predicted vs
  0.189 measured; β_min = 0.3594 is the
  coldest Arm-A grid point with directly measured class leakage, and leakage at β = 0.6
  was already 17.6/36.7% per 1500 steps, and rung-0's OWN measured leakage at
  β = 0.3594 is 25.6/62.6% per 1500 steps (grader advisory 7 — the β = 0.6 figure is
  the conservative one) ⇒ rung-0 kernel crossing is the discovery channel); K = 10
  (near the wall-optimum K* ≈ √IAT given round-cost ∝ (K+1) and swap-back factor
  f ≈ K/(K+IAT); IAT at the retained rungs = 11.4–45.6 steps, so f ≈ 0.18–0.47 — no
  longer ≪ 1); NSYS = 16 ladders/arm (96-wide fused, the measured 7.8
  s/round width class); ROUNDS = 1500 (16.5k kernel steps/chain ≈ the 10k+10k reference
  scale; wall 1500 × 7.78 s ≈ 3.25 h/arm, inside a 4 h allocation with startup); ss_max
  = 5.0 (ARTIFACT-CORRECTED per grader rd-1: adapted steps at β ≥ 0.36 sat at
  0.54–0.84 — the earlier "~0.05" was ss_init, a memory-for-artifact substitution —
  so cap 5.0 is ≈6–9× headroom; NOTE the short ladder MOOTS rather than TESTS the
  cap-binding mechanism (iii), since PT-0's sub-band EEVPD lived at β ≤ 0.19 which
  this ladder excludes; handle_nans still shrinks the cap on NaN). Arms: **P1** power/balanced seed 10; **P2**
  power/balanced seed 11 (seed replica); **P3** power/ALL-MAIN seed 12 (production
  bad-MAP scenario — new arm type B4: every rung from main pool; discovery must come
  from rung-0 kernel crossing); **P4** power/all-main seed 13 (replica). Init pools,
  metric (pooled MAMS64 cov, positions only), indicator, instrumentation: unchanged
  from PT-0.
  **Predictions (direction + magnitude, from the revised flux model with PT-0-measured
  inputs; the model itself is UNDER TEST via W-b1).** Adjacent-pair acceptance ā ≈ 0.53
  (erfc model, provenance above; W-b3's [0.25, 0.65] band brackets both this and the
  design-band 0.41). Label transport: per-walker round-trip time ≈ 2R²/(ā·f) with
  f ≈ K/(K+IAT_pooled), f ∈ 0.18–0.47 ⇒ 289–755 rounds/walker (grader-corrected:
  2·36/(0.53·0.18) = 755, not 848) ⇒ with 96 walkers/arm ⇒
  **≈ 190–500 total round trips per arm in 1500 rounds (point prediction ≈ 300);
  pocket-classified ≈ 30–120** (pocket fraction between the 0.1-anchor and the 0.3–0.4
  open-finding readings). Cold-rung basin flips: PT-0's B1 cold pair acc 0.189 gave
  19–77 flips/system/900 rounds (npz-confirmed) ⇒ at acc ≈ 0.53 and 1500 rounds
  predict **≈ 90–360 flips/system**. Discovery (P3/P4): at rung-0's own measured
  crossing rate (25.6%/1500 steps at β = 0.3594), 16 systems ⇒ first pocket
  discoveries within ~40–100 rounds, cold-rung arrivals by ~200–500.
  **Win conditions (derived; statistics PINNED per grader rd-1).** (W-b1, transport +
  model coherence) pocket-classified round trips per balanced arm ≥ 10 (floor =
  detection at Poisson 3σ above the PT-0-measured 0) AND total round trips within ×/÷4
  of the 300/arm point prediction. ROUTED ZONES: pocket RTs ∈ [1, 9] with the total-RT
  clause passing ⇒ the PT-0-amendment-(i) reading (mechanism confirmed, FLUX-LIMITED;
  derived ROUNDS scaling for the next gate, no free rerun); total RTs in the (×4, ×10]
  annulus ⇒ flux model MARGINAL — report, no scale-up; outside ×/÷10 ⇒ F-b4.
  (W-b2, THE product test — two-sided bracketing; formulas pinned) per arm: m_a =
  mean over 16 systems of the per-system last-500-round cold-rung occupancy mean;
  sd_a = across-system sd (ddof=1); se_a = sd_a/4; se_comb = √(se_P1P2² + se_P3P4²)
  with each pair pooled (32 systems). CLAUSES: (agreement) |m_balanced − m_allmain| ≤
  2·se_comb; (movement) each arm-pair moved ≥ 3·se_pair from its init (0.5 / 0.0) —
  precedence note below; (POWER) se_comb ≤ 0.06, DERIVED: adjudicating 0.1-vs-0.35
  needs a CI half-width < 0.125 ⇒ 2·se_comb ≤ 0.125; if agreement holds but
  se_comb > 0.06 the verdict is "bracket CONSISTENT but UNDERPOWERED — NOT a pass"
  (routes to a longer run, powered by the measured se scaling). Pocket-weight
  adjudication ONLY if the pooled bracket CI (weighted mean ± 2·se_comb) EXCLUDES one
  of the two candidate readings (0.1 / 0.35); else "bracket passed, weight NOT
  adjudicated" — pre-committed wording. (2σ, 3σ] agreement zone ⇒ "not demonstrated at
  pilot precision" — fails W-b2, does NOT fire F-b2 (restored from PT-0 grading).
  (W-b3, ladder health — NOTE per grader: the short ladder MOOTS the cap-fix test;
  this clause validates SPACING + rung health, not mechanism (iii)) EEVPD ∈
  [1e-4, 2e-3] at every rung (median, last 500 rounds); all 5 pair acceptances ∈
  [0.25, 0.65] (prediction 0.53 erfc / 0.41 design band); NaN reverts = 0.
  (W-b4, reproducibility) seed replicas agree: |m_P1 − m_P2| ≤ 2·√(se_P1² + se_P2²),
  same for P3/P4. (W-b5, health-2) cold-rung indicator split-R̂ across 16 systems ≤
  1.05 per arm; (1.05, 1.2] with W-b1 passing ⇒ budget-limited mixing (report
  occupancy-ESS), not failure — pre-committed. ALL-PASS ROUTING: draft GATE PT-1
  (production-config certification + efficiency accounting + cross-method
  unbiasedness); NO auto-scale-up inside PT-0b.
  **Falsifiers (with precedence).** F-b1: pocket round trips = 0 again in ANY balanced
  arm ⇒ the swap-back/spacing mechanism story is WRONG or incomplete ⇒ STOP, report to
  human (no auto-lever; C-23's remedial content is then falsified). F-b5 (discovery)
  — TAKES PRECEDENCE OVER F-b2: NO pocket-classified state EVER appears at any rung of
  P3/P4 in 1500 rounds ⇒ β_min = 0.3594 insufficient for discovery-from-main ⇒ ladder
  must extend hotter (recorded; NOT a transport failure); in that configuration the
  all-main arms' occupancy trivially disagrees with balanced — F-b2 MUST NOT fire and
  W-b2 is UNSCOREABLE (pre-committed). F-b2: bracketing disagreement > 3σ WITH P3/P4
  discovery having occurred (hysteresis) ⇒ no unbiasedness claim; pocket weight stays
  open; report. F-b3: EEVPD below band at rung 0 despite ss_max = 5 — reachable only
  where PT-0 was already in-band, so a firing implicates the pooled-metric/PT context,
  NOT the (mooted) cap mechanism — open finding. F-b4: total RTs outside ×/÷10 of the
  300/arm prediction ⇒ flux model wrong even with measured inputs.
  **Metric blind spots.** (i) Bracketing agreement is blind to both arms converging to
  the same wrong value via a shared systematic (unadjusted-kernel bias; named in PT-0;
  cross-method adjudication deferred to a later gate). (ii) Round trips certify label
  transport, not within-basin ESS. (iii) The z[6] halfspace remains blind to any third
  mode. (iv) IAT_pooled ≈ 30 is an assumption bridging confined→pooled metrics — W-b1's
  wide ×/÷4 band absorbs it, and its failure is informative (F-b4).
  **Pre-committed plot appearances.** Worms: pocket-colored bands should now TRAVERSE
  hot↔cold repeatedly in balanced arms (vs PT-0's rung-local churn); all-main worms
  should show pocket color first appearing at rung 0 then propagating cold. Cold-occ
  traces: P1/P2 falling from 0.5 and P3/P4 rising from 0.0 into a COMMON band (the
  band's level answers the weight question); F-b2 ⇒ plateaus at different levels.
  EEVPD traces: flat in-band at all 6 rungs (F-b3 ⇒ rung-0 trace below band). Pair-acc
  bars: 5 bars in [0.25, 0.65], same ≈ cross.
  **Cost estimate.** 1 × 4 h interactive node (4 × A100, -A m5362): smoke (full-shape,
  R = 6 × 16, ~10 min incl. compile) then 4 arms × 1500 rounds × 7.8 s ≈ 3.3 h in
  parallel (one arm per GPU) + ~15 min startup/compile per process ⇒ ≈ 3.7 h wall,
  ≈ 15 GPU·h. Incremental saves every 100 rounds; if the allocation clips the tail,
  the op-7 realized-rounds scaling applies (pre-committed here too).
  **Process notes.** Env overrides (GATE_PT0_BETAS_B, GATE_PT0_K_B, GATE_PT0_NSYS_B,
  GATE_PT0_ROUNDS_B, GATE_PT0_SSMAX, GATE_PT0_TAG_SUFFIX=_pt0b) recorded in each model
  card; the B4 arm-type + env-override extension is diff-audited before launch; seeds
  10–13; W-2-statistic machinery (round_trips_pocket) unchanged from the audited code.**

- **Run: carousel GATE PT-0 — tempering-path diagnosis + instrumented PT-MCLMC pilot on
  the dPIE carousel (opening gate of the 2026-07-10 long-horizon engagement: efficient
  accurate sampler for multimodal lensing posteriors; MCLMC kernel per human directive;
  PT first avenue, not locked in).**
  **Status: grader-verified rd-2 (2026-07-10) — all 5 blocking + 6 advisory rd-1
  amendments confirmed applied and faithful (control target verbatim pt_weight.py;
  adapt_one math-exact to lineage; GATE L header byte-identical to 8e3e8bf; audit-fix
  commit contains no unrelated changes); CERTIFY-RECOMMENDED to LAUNCH at 409824a
  (+ rd-2 advisories A1/A2/A4 applied post-verdict: superseded-W-2 tag, smoke
  count/docstring wall fix, thinned walker-id series persisted; A3 = hot-end flag
  omits delta_se, note if it fires; A5 = result grading scores W-2 jointly with the
  calibrated c_rw·ā_B expectation, not the raw 7). LAUNCHING under the engagement's
  free-hand mandate (human validates final product; run outcomes return UNCERTIFIED
  for fresh grading). RAN 2026-07-11; result graded rd-1 NEEDS-MORE → amendments
  APPLIED (see RESULT-GRADER AMENDMENTS block in the Log entry): W-2/W-5 FAIL confirmed
  against artifacts (F-2/F-4 routing); W-4 re-scored FAIL for B1 AND B2 (grader
  recount: hot-rung EEVPD below band on both, cold split-R̂ 1.68/1.60); W-3 not tested
  (B3 canceled — op-8 deviation); Arm A no-starvation + Arm 0 PASS verified against
  artifacts; grader: CERTIFY-RECOMMENDED as mechanism diagnosis ONLY after amendments.
  C-23 registered. PT-0b continuation checkpoint is the routed next step.** Script `experiments/flow_precond/carousel_gate_pt0.py`
  (new, written fresh — the June-28 PT implementation is NOT reused
  per the human's provenance note — and independently code-audited before launch);
  outputs `carousel_gate_pt0_out/`; float64; model via `carousel_model.build()` (D=33);
  pocket indicator z[6] ('planes/0/mass/1/center_x') > −22.35; per-basin position pools =
  MAMS64 draws split by indicator (POSITION/METRIC use only — per the 2026-07-10 human
  directive, MAMS64 weights are NOT trusted and NO win condition scores against 9.57%);
  frozen inverse-mass metrics: pooled MAMS64 empirical cov for ladders, per-basin cov for
  confined profile runs. Seeds: Arm A = 0, Arm B1/B2/B3 = 0/1/2. SMOKE env var (reduced
  config, 1 GPU, ~10 min) must pass before full launch. 1 interactive node, 4 GPUs.
  **Claim under test + classification.** Stochastic-estimator behaviour, two chained
  links. Link 1 (mechanism): the June-28 minimal-carousel PT failure and any dPIE PT
  behaviour are governed by the per-rung equilibrium basin-mass profile of the tempering
  PATH — power path p^β re-weights basin masses along the ladder (peak-height term)
  while a likelihood-tempered path π·L^β anchors the hot end at the PRIOR's basin split;
  replica transport flux is bottlenecked by the min-rung minority mass. Link 2 (transport
  + drain): on a path whose measured min-rung pocket mass is workable, PT-MCLMC with
  per-rung EEVPD adaptation achieves cross-basin round trips and drains an off-balance
  init to the same cold occupancy from both sides, within a ~20k-step/chain pilot budget.
  Explicitly UNTESTED links, named now: absolute unbiasedness against external truth (NO
  trusted truth exists — MAMS64 weights are dead by human directive; only bracketing
  agreement + mechanism coherence are claimed); production init realism beyond the
  all-main arm; efficiency frontier; other lenses/priors; within-basin sampling quality
  (= plain-MCLMC validity, C-2..C-7 territory).
  **Cause hypothesis.** PT transport on multimodal targets fails not through pairwise
  swap acceptance (June-28: healthy 0.38–0.72 average) but through the equilibrium
  starvation of the minority basin at hot rungs: for the power path, log[w_P/w_M](β) ≈
  β·Δlp* + ΔS with Δlp* = lp*P − lp*M = +8.50 (GATE L M1) and ΔS β-independent to
  Gaussian order, so the pocket's relative mass shrinks by e^{(β−1)·8.50} toward hot —
  ≈ e^{−8.4} ≈ 2×10⁻⁴ of its cold value at β = 0.01. A likelihood-tempered path replaces
  the hot-end anchor with the prior's indicator split m_prior (measured in-run from
  prior draws; expected O(0.1–0.5) — the prior does not know about basins), removing the
  starvation. If Link 1 holds, the SAME PT machinery that failed June-28 should work on
  the likelihood path and fail on the power path, with transport counts PREDICTED by the
  measured profiles.
  **Arm A — tempered-mass profile (2 GPUs, ~50–75 min).** For each path (power: u = logp;
  likelihood: u = logL, requires log_prior + log_like = log_prob verified at build time
  to ≤1e-6 — contingency if the API exposes no separable prior: implement logπ directly
  from the prior bijector stack; the validation suite's LOG_PRIOR anchor says it exists),
  measure the RELATIVE profile Δ(β) = log[w_P/w_M](β) − log[w_P/w_M](1) =
  ∫_β^1 (E_{p_β'|M}[u] − E_{p_β'|P}[u]) dβ' by per-basin-confined tempered MCLMC at
  β ∈ geomspace(0.01, 1, 10) (one 320-wide vmap per path: 10β × 2 basins × 16 chains;
  3000 steps: 1500 equilibration discard + 1500 measure; per-chain EEVPD step adaptation,
  per-basin frozen metric). Confinement is monitored (indicator flips per chain per
  config recorded); configs with >10% leaked samples are flagged and the leaked samples
  classified-and-reassigned, not silently dropped. The D/(2β) leading term of E[u]
  CANCELS in the M−P difference (same D), so trapezoid bias on the geometric grid is
  second-order; MC error target ≤2 nats on Δ (per-point se ≈ sd(u)/√ESS with sd(u) ≈
  √(D/2)/β at the hot end — ESS ≥ ~150/config suffices; realized se reported with
  chain-clustered errors, 16 chains as clusters). Internal-consistency check (validate
  internals): at the hottest 1–2 rungs where confinement breaks (barrier off), direct
  unconfined occupancy must agree with the TI-extrapolated relative profile + assumed
  O(10%) cold anchor within 2× its se — disagreement is an open finding, not a pass.
  **Arm A predictions (direction + magnitude).** Power path: Δ(β) falls MONOTONICALLY to
  −8.4 nats at β = 0.01 (Gaussian model; GATE L says the basins are non-Gaussian — KL 157
  nats main — so ±3 nats tolerance; the DIRECTION and ≥4-nat depth are the tested
  content). Likelihood path: |Δ(β)| ≤ 2 nats across the ladder (hot anchor = prior split;
  shallow interpolation), i.e., the two paths separate by ≥3 nats at the hot end.
  **Arm A falsifier F-1:** the likelihood path ALSO shows ≤ −4 nats suppression somewhere
  (both paths starve ⇒ Link 1's path-choice remedy is wrong; PT-as-planned out; routing
  below). Secondary falsifier: power path measures FLAT (|Δ| ≤ 2 nats everywhere) — then
  the June-28 failure cannot be entropic starvation on the dPIE analogue and the
  bug-hypothesis gains weight (recorded; does not block Arm B, which runs regardless).
  **Arm B — instrumented PT-MCLMC pilot (3 arms × 1 GPU each, ~100–120 min,
  runs UNCONDITIONALLY per the human's provenance note).** Fresh implementation:
  R = 12 rungs, β = geomspace(0.01, 1, 12) initial (refined to approximate
  equal-Δβ·sd[u] spacing from Arm A's curves if Arm A completes first; refinement is an
  in-gate amendment, recorded, not a re-approval); NSYS = 8 independent ladders per arm
  (the reproducibility axis); K = 10 MCLMC steps/round; ROUNDS = 2000 (20k kernel
  steps/chain ≈ the user's 10k+10k reference scale); 96 chains vmapped per arm; jitted
  on-device even/odd adjacent swaps, log α = (β_i − β_j)(u(x_j) − u(x_i)) on the path's
  u; per-(rung,chain) EEVPD step adaptation (adapt_one lineage, DEVAR 5e-4); momentum
  refresh per round. Arms: B1 = power path, balanced init (cold rungs half main / half
  pocket positions); B2 = likelihood path, balanced init; B3 = likelihood path, ALL-MAIN
  init (the production bad-MAP scenario: pocket must be discovered via the hot/prior end;
  hot rungs init from prior draws on the likelihood arms). Instrumentation (the June-28
  lesson): CROSS-basin vs same-basin swap acceptance per pair (separately — the average
  is a proven-misleading metric); replica basin-identity worm traces (rung × round);
  pocket-label round-trip counter (hot↔cold); per-rung occupancy time series; per-rung
  EEVPD; cold-rung indicator split-R̂ across the 8 ladders.
  **Arm B thresholds (derived, not invented).** Transport-flux model: a replica label
  random-walks R rungs with per-sweep move probability ā ⇒ round-trip time ≈ R²/ā
  rounds; pocket-label current additionally suppressed by the profile bottleneck factor
  e^{min Δ} = w_min/w_cold. Predicted pocket round trips per ladder = (ROUNDS·ā/R²) ×
  (w_min/w_cold); at ROUNDS = 2000, R = 12, ā ≈ 0.5 (June-28 measured average): ≈ 6.9 ×
  (w_min/w_cold). Likelihood path (|Δ| ≤ 2 ⇒ w_min/w_cold ≥ 0.135): ≥ ~0.9 per ladder,
  ~7–55 total over 8 ladders. Power path (Δ ≈ −8.4): ≈ 1.5×10⁻³ per ladder ⇒ ~0 total.
  (W-2, transport) [SUPERSEDED by amendment i] likelihood-path balanced arm: ≥10 pocket round trips total AND median
  ≥1 per ladder. (W-3, bracketing drain) B2 vs B3 final cold-rung occupancy (last 500
  rounds, per-ladder means, n=8 each): |occ_B2 − occ_B3| ≤ 2·se_comb AND each arm moved
  ≥3× its binomial se from its init value (0.5 and 0.0) — convergence from OPPOSITE
  sides to a common value is the unbiasedness instrument replacing the dead MAMS64
  anchor. (W-4, health) per-rung EEVPD ∈ [1e-4, 2e-3]; cold-rung indicator split-R̂ ≤
  1.05 (8 ladders as chains, per arm); zero NaN chains. (W-5, mechanism coherence —
  validate internals) observed pocket-label cold-arrival counts per path within ×/÷4 of
  the Arm-A-profile prediction; >10× mismatch = mechanism model WRONG even if sampling
  "looks good" (open finding, blocks scale-up). Absolute pocket occupancy: threshold NOT
  derivable — no trusted truth exists (human directive); deliberately not scored.
  **Arm B falsifiers.** F-2: likelihood-path profile viable BUT zero pocket round trips
  (transport machinery/dynamics problem — mundane-first response: replica-trace
  localization of where labels stall; ONE pre-authorized amendment probe = double K to
  20 at halved ROUNDS on one GPU; no other knobs). F-3: B2 and B3 converge to occupancies
  differing >3σ (hysteresis/hidden bias — blocks any scale-up claim). F-4: W-5 mismatch
  >10× in either direction. F-1 (from Arm A): both paths starve.
  **Metric blind spots (named).** (i) The z[6] halfspace indicator is blind to any THIRD
  mode or within-basin substructure — everything here conditions on the two known basins.
  (ii) Confined-TI expectations are biased if mid-β metastability fails asymmetrically
  (leakage is monitored + reassigned, but strong leakage makes Δ(β) a lower-confidence
  band there). (iii) W-3 agreement is blind to BOTH arms converging to the same WRONG
  value through a shared systematic (e.g., unadjusted-kernel curvature bias, C-16);
  flagged for cross-method (SMC or long-reference) adjudication at a later gate, not
  resolvable inside PT-0. (iv) Round-trip counts do not certify within-basin ESS.
  **Pre-committed plot appearances.** Arm A: Δ(β) vs β, two curves + error bands —
  hypothesis holds ⇒ power curve dives monotonically ≥4 nats (to ≈ −8 ± 3 at β = 0.01)
  while likelihood curve stays in a ±2-nat band; F-1 fires ⇒ both curves dive. Arm B
  worm plot (rung index vs round, colored by basin identity): success ⇒ pocket-colored
  worms repeatedly traverse hot↔cold on likelihood arms; F-2 ⇒ pocket worms pinned at
  one end (the June-28 signature). Cold-occupancy traces: B2 falling and B3 rising to a
  COMMON band; F-3 ⇒ plateaus at different levels. EEVPD-per-rung traces flat at 5e-4.
  **Routing (pre-committed).** All of W-2..W-5 pass ⇒ draft GATE PT-1 (production-config
  scale-up + efficiency accounting + cross-method unbiasedness check) — no auto-scale-up
  inside PT-0. F-1 ⇒ tempering-path family on this posterior needs basin-mass
  reweighting: pivot checkpoint choosing between per-basin TI mode-weight stitching (Arm
  A's machinery already produces the estimator) and adaptive-weight ladders
  (multicanonical on the indicator); PT-as-planned stops. F-2 ⇒ diagnostic-first (no
  knob-turning beyond the ONE pre-authorized K probe). F-3/F-4 ⇒ report to human with
  localization evidence; no auto-lever. Any wall overrun ⇒ arms are independently
  checkpointed (per-arm incremental npz saves every ~100 rounds) and resumable.
  **Cost estimate.** 1× interactive GPU node (4× A100-40G, -A m5362), single ~4 h
  allocation ≈ 12–16 GPU·h: smoke ~10 min on 1 GPU; Arm A ≈ 50–75 min on 2 GPUs (one per
  path; 320-wide vmap, 3000 steps, MCLMC ≈ 2 grad evals/step, scaled from the MAP-128
  0.197 s/step measurement); Arm B ≈ 100–120 min, 3 arms on 3 GPUs (96-wide, 20k
  steps/chain + jitted swap sync every 10 steps); slack for the B-ladder refinement
  restart and the single pre-authorized F-2 probe. Wall-clock is the binding constraint
  per the human; GPU-hours are not.
  **GRADER AMENDMENTS (round 1, NEEDS-MORE 2026-07-10; 5 blocking + 6 advisory, all
  adopted):**
  (i) *W-2 reconciled with its own prediction + gap zones routed (BLOCKING).* W-2 is now:
  PASS = ≥7 total pocket round trips over the 8 likelihood-balanced ladders (the derived
  band's lower edge; the per-ladder median clause is DROPPED — at the band edge the
  median sits on the 0/1 knife and tests nothing). Routed zones: 1–6 total WITH W-5
  coherence ⇒ mechanism CONFIRMED, flux-limited — routing is a DERIVED ROUNDS scaling
  for GATE PT-1 (ROUNDS × 10/observed-rate), not a free PT-0 rerun; 0 total ⇒ F-2. W-3
  2σ–3σ zone ⇒ "W-3 not demonstrated at pilot precision" (fails W-3, does NOT fire F-3);
  scale-up blocked pending derived-longer ROUNDS or redesign. Arm A power-path depth
  measured in 2–4 nats ⇒ prediction MISSED in magnitude (hypothesis failure per
  discipline) even with the right direction; W-5 adjudicates on the MEASURED profile.
  Likelihood-path realized hot-end se > 2 nats ⇒ the |Δ| ≤ 2 test widens to
  |Δ| ≤ 2 + se_realized and the path discrimination is flagged INCONCLUSIVE if the
  widened band spans the −2.0 viability floor — no silent pass. (Viability refs
  harmonized: floor Δ_min ≥ −2.0 nats ⟺ w_min/w_cold ≥ 0.135; unworkable ≤ −4.2 nats.)
  (ii) *Known-answer + calibration control, Arm 0 (BLOCKING — severs the "fresh
  implementation confirms the path story even if June-28 was just buggy" confound).*
  Before any dPIE arm, the NEW harness code path must (0a) re-pass the CPU-era
  Gaussian-mixture weight gate (D=10, modes ±5, weights 0.7/0.3, R=10
  β=geomspace(0.03,1), NSYS=16, K=20, 3000 rounds, burn 600): cold occ₊ within
  max(2·se, 0.025) of 0.70 (June-28 lineage passed 0.6986±0.0122); and (0b) calibrate
  the transport constant on the same analytic target, whose tempered profile is EXACT
  (equal-cov mixture ⇒ w₋(β) = 0.3^β/(0.3^β+0.7^β), benign): c_rw :=
  observed-round-trips / (ROUNDS·ā_measured/R²), requiring ≥20 total round trips so
  c_rw carries ≤~25% Poisson error. W-5's prediction then uses c_rw and the in-run
  measured ā_B in place of the June-28 plug-ins (ā=0.5 and the naked R²/ā constant are
  DEMOTED to a-priori estimates). Sanity band c_rw ∈ [0.1, 3]; outside ⇒ the random-walk
  flux model itself is wrong ⇒ W-2/W-5 re-derived from c_rw before Arm B launches
  (recorded as an in-gate amendment). Cost ≈ minutes (analytic target).
  (iii) *Shape-faithful smoke (BLOCKING — the GATE L attempt-1 lesson).* The smoke must
  compile and execute the FULL production shapes at reduced step counts: Arm A per-β
  runner at full 32-wide (all 10 β compiles exercised), Arm B at full R=12 × NSYS=8
  including the jitted swap sync and the incremental-npz save path; only steps/rounds
  are reduced. Width-reduced smokes are insufficient.
  (iv) *Confinement mechanism specified (BLOCKING).* There is NO barrier: "confined"
  runs differ only in INIT (per-basin pools). Every retained sample is classified by
  its CURRENT indicator value and pooled by class; the estimand is therefore the
  halfspace-conditional expectation E_{p_β}[u | class] = d/dβ log Z_class(β) for the
  fixed halfspace partition — well-defined at every β including where metastability
  fails, with no in-run lever. Boundary-interaction rates (indicator flips per chain,
  leak fraction per (β, init-basin)) are recorded per config; split-half stationarity
  of E[u] within the measurement window is reported per config (advisory viii).
  (v) *Record repair (BLOCKING):* the PT-0 insertion had clobbered the GATE L
  design-checkpoint header line — restored verbatim above (same failure class as the
  Fv6/89cf321 clobber; caught by grader before commit this time).
  (vi) *Δlp★ provenance (ADVISORY):* the +8.50 rests on GATE L's NON-stationary main
  anchor (nat-grad 1898, 3 negative eigenvalues; C-22) inside a basin with
  KL(emp‖Laplace) = 157 nats — the −8.4-nat depth is a Gaussian-order estimate around
  an unconverged anchor; if the true main peak is higher the real depth shrinks. Arm A's
  MEASURED profile supersedes the model everywhere downstream (W-5 scores against the
  measurement, never the model).
  (vii) *Likelihood-path hot-end error budget (ADVISORY):* the sd(u) ≈ √(D/2)/β formula
  is power-path-only; near β→0 the likelihood-path sd(logL) under ~the prior is not
  derivable a priori at the ~1e5 evidence scale — the realized se is REPORTED per rung
  and routed via (i)'s widened-band rule.
  (viii) *Hot-rung stationarity evidence (ADVISORY):* the frozen cold-basin metric is
  ~100× variance-mismatched at β = 0.01 on the power path; a split-half E[u] check over
  the 1500-step measurement window is reported alongside Δ(β) per config.
  (ix) *Anchor sourcing (ADVISORY):* the "assumed O(10%) cold anchor" in Arm A's
  internal-consistency check is sourced to the human's "~10:1 modes" statement
  (archaeology HUMAN CONTEXT), NOT to MAMS64, and never enters a pass/fail.
  (x) *Scheduling honesty (ADVISORY):* Arm A (2 GPUs) + Arm B (3 arms) need 5 GPU-slots
  on a 4-GPU node ⇒ partial serialization (realistic wall ≈ 3.5 h incl. smoke + Arm 0);
  the pre-authorized F-2 probe realistically lands in a SECOND allocation via the
  per-arm incremental checkpoint/resume path.
  **IN-GATE OPERATIONAL AMENDMENTS (2026-07-11, during the run; no estimand/threshold
  changes):** (op-1) upstream gigalens API refactor landed mid-gate (linusu-dev-merge
  @698b990, 15:20 Jul 10: Dataset → ABC + concrete ImageData) — carousel_model.py
  adapted (identical ctor signature, fallback import; commit c0c32e7); verify() re-run
  under the merged code: zP/zM anchors reproduce to millinats, red-χ² 1.1618 — the dPIE
  likelihood is UNCHANGED, no re-anchoring needed. (op-2) smoke revealed the B-arm
  round loop is dispatch-fixed-cost dominated (12 sequential 8-wide per-rung calls,
  34 s/round ⇒ ~19 h/arm): rung loop fused into ONE jitted vmap-over-rungs call
  (traced-β kernel construction, 96-wide) with a mandatory 3-round equivalence check
  vs the retained per-rung path (identical inits/keys; max|Δpos|, max|Δu| reported)
  before B arms launch — projected ~2 h/arm. (op-3) smoke also VINDICATED the relative
  u-identity gate: B3's round-0 identity measured 7.9e-6 ABSOLUTE (2.7e-11 relative) —
  the pre-fix absolute 1e-6 gate would have spuriously aborted the arm. (op-4) measured
  m_prior = 0.9944 ± 0.0012 (n=4096): the PRIOR puts 99.4% of its indicator mass on the
  POCKET side — the hot-end asymmetry is the reverse of the design sketch; consequence
  pre-read: on the likelihood path the pocket class (the W-2 species) runs ~0.99 → ~0.10
  along the ladder with NO predicted interior starvation, while the MAIN class is the
  hot-starved one (~0.6%) — main-class round trips may be suppressed without touching
  W-2; Arm A's measured profile adjudicates.
  (op-5, Arm 0 RESULT + pre-registered c_rw routing FIRED) 0a known-answer gate PASS:
  fresh-harness cold occ₊ = 0.6885 ± 0.0112 vs truth 0.70 (|diff| 0.0115 ≤ 0.025), 2480
  round trips, ā = 0.551, wall 450 s — the June-28 implementation-bug hypothesis is now
  UNREPRODUCED by this code path on the benign target. 0b: c_rw = 9.371, OUTSIDE the
  [0.1, 3] sanity band ⇒ per the pre-commitment, W-5's prediction is RE-DERIVED before
  Arm B: diagnosis is bookkeeping, not physics — the naive R²/ā formula prices ONE
  walker's round-trip time while all R walkers per ladder circulate concurrently
  (per-walker rate 15.5/3000 rounds ≈ the naive 1/(R²/ā) = 1/182 within 7%), so c_rw ≈ R
  as it should be under ballistic-ish exchange. Calibrated W-5 expectation for B arms:
  RT_pocket/ladder ≈ c_rw · (ROUNDS·ā_B/R_B²) · (w_min/w_cold) = 9.371 · (2000·ā_B/144)
  · e^{Δ_min}, ā_B measured in-run; at ā_B ≈ 0.5 and likelihood-path Δ_min ≥ −2 nats
  that is ≥ ~9/ladder (~70+ total). W-2's ≥7-total floor is UNCHANGED (conservative);
  W-5 coherence (×/÷4) scores against the calibrated expectation. Control EEVPD sat at
  1e-9–1e-7 (≪ 5e-4): the ss_max=1 cap binds on the trivial target — irrelevant to the
  weight gate; the W-4 EEVPD band applies to the dPIE arms only (noted).
  (op-6, fused-runner equivalence VERDICT) control target: fused vs legacy BITWISE-level
  (max|Δpos| 2.8e-15–5.3e-15 over 3 rounds, all R=10×NSYS=16) — rules out any
  wiring/permutation bug in the fused machinery (a rung misalignment would give O(1)
  diffs). dPIE (B1 smoke config): positions diverge (max|Δpos| 0.41, max|Δu| 4.8 nats
  after one K=10 round) while the u0 identity passes at 5e-11 for BOTH impls —
  attributed to the RECORDED cross-compile FP non-reproducibility (lstsq log_like)
  amplified by chaotic MCLMC dynamics; β-algebra validated end-to-end by u0; shared
  code path validated bitwise on control. Validation chain: fused ≡ legacy (bitwise,
  control) ∧ legacy passes the 0.70 known-answer gate (the full Arm 0 ran pre-fusion,
  i.e. legacy) ⇒ fused inherits the known-answer validation. B arms run FUSED; per-arm
  in-run guards (u0 identity, EEVPD band, swap accounting) cover the dPIE regime.
  (op-7, expected ROUNDS truncation) fused B rounds measure ~5.5 s/round on the dPIE
  target (vs ~3 s projected; still 6× over the legacy 34 s), so 2000 rounds ≈ 3.1 h/arm
  does not fit the remaining allocation for arms launched after the smoke/API delays.
  Pre-commitment for analysis: each arm's realized rounds_done is recorded in its npz;
  last-500-round windows (W-3, W-4) remain defined for any arm reaching ≥1000 rounds;
  the W-5 calibrated expectation scales with the REALIZED ROUNDS·ā_B (not nominal
  2000); W-2's ≥7-total floor is kept but a shortfall with W-5 coherence routes through
  the pre-registered flux-limited zone (amendment i). If any arm ends <1000 rounds, its
  windowed stats are reported at reduced window with the reduction stated. A follow-up
  allocation MAY extend arms only as fresh pre-registered continuation runs (no resume;
  data-salvage only per the audit).**
  (xi) *Audit artifact (ADVISORY):* the independent code audit must leave a record —
  auditor identity, commit hash audited, findings — in the Log before launch; an
  unrecorded audit is indistinguishable from none.
  **INDEPENDENT CODE AUDIT (2026-07-10, per amendment xi):** auditor = dedicated
  read-only subagent (fresh context, no authorship overlap), audited
  `carousel_gate_pt0.py` @ commit 9f1f874 against this checkpoint + first principles,
  14-item mandatory checklist. Verdict: DEFECTS-FOUND → fixed. VERIFIED clean: swap
  math + sign conventions end-to-end (higher-u configs migrate cold; detailed-balance
  derivation traced for both paths, prior terms cancel exactly on the likelihood path),
  adapt_one byte-faithful to the June-28 lineage, rung-resident adaptation (no
  stale-state class), Arm A estimator signs/cluster-se/trapezoid propagation, control
  target verbatim pt_weight.py (axis-0 modes, occ₊ = 0.70 mode), inits/seeds/indicator
  conventions/EEVPD/split-R̂/x64, no OOM-class shapes (positions never scan-traced —
  the GATE L attempt-1 class is absent). BLOCKING defect found + FIXED pre-launch: the
  W-2 statistic (pocket-classified round trips) was label-blind and unreconstructible —
  walker flag now carries arrival class (states 2/3), round_trips_pocket/main counted
  per system, wflag+wid persisted. Minor fixes applied: round-0 u-identity tolerance
  made RELATIVE (1e-6; the absolute 1e-6 on |u|~3e5 across jit compiles was a ~1e-11
  reproducibility demand the recorded non-bitwise lstsq log_like could spuriously trip,
  and a likelihood-path log_prior cross-compile mismatch amplifies as ε/β); Arm A given
  its own per-config u-recovery identity check (build-time separability gate is
  CSE-trivial by construction); empty-class guard (<50 samples ⇒ E missing, Δ(β)
  truncated at the hottest complete β, recorded — a reporting rule, not interpolation);
  hot-end consistency flag auto-computed. Wording corrections to THIS checkpoint from
  the audit: Arm B is implemented as 12 sequential 8-wide per-rung jitted dispatches per
  round + host-side numpy swaps (mathematically identical to the "96-wide / on-device
  swaps" description; wall-relevant only); "resumable" means incremental-npz
  DATA-SALVAGE (diagnostics preserved, sampler state not) — the F-2 probe is a fresh
  run in a second allocation, per amendment x; smoke is shape-faithful at ~60 dPIE
  compiles ⇒ realistically 25–60 min in one process — RUN THE SIX SMOKE SUB-ARMS IN
  PARALLEL ACROSS THE 4 GPUs (~15 min wall), and production arms each pay their own
  ~12-compile startup (~6–12 min/process, included in the wall envelope). ESS proxy is
  2× conservative vs N/IAT (report-only, safe direction — recorded).**

- **Run: carousel GATE L — Laplace jump-proposal feasibility (3 offline diagnostics,
  human-directed 2026-07-09 after the Fv6 escalation; strategic pivot candidate =
  multistart MAP → per-mode Laplace surrogates → MAMS + MH independence-jump mixture).**
  **Status: grader-approved (round 3, conditional; conditions met) — RAN 2026-07-09;
  result in Log ("GATE L RAN"); M2 all-fail + M3 falsifier fired; C-22 registered.** Script `experiments/flow_precond/carousel_gate_l.py`
  (new), outputs `carousel_gate_l_out/`; seed 0 throughout; float64; carousel model via
  `carousel_model.build()`; zM/zP = basin medians from
  `/pscratch/sd/l/linusu/carousel_diag/basin_slice/basin_slice.npz`; MH-exact draws =
  MAMS64 `experiments/sim_carousel/messy_tests/dpie/mams/arrays.npz` (64×1000×33, pocket
  occupancy 9.57%, indicator z[6] > −22.35); benchmark 8×4000 u-space draws
  (`carousel_benchmark_out/benchmark_arrays.npz`) reserved as a pocket-shape robustness
  check ONLY (MH-exact in law ⇒ within-pocket conditional valid; occupancy trap-biased ⇒
  never used for weights).
  **Claim under test + classification.** Link 1 of the jump-mixture pipeline chain:
  "local Laplace surrogates on this posterior are accurate enough that an MH
  independence-jump mixture would mix the two modes within the user's 10k+10k budget."
  M1 is a deterministic computation (curvature spectra at polished stationary points);
  M2 is stochastic-estimator behaviour (MC acceptance with derivable MC error); M3 is a
  finite-sample frequency estimate. Explicitly UNTESTED links, named now: interaction of
  jumps with MCLMC adaptation; behaviour on other lenses/priors; missing-mode risk beyond
  the two known modes. Passing GATE L licenses only a prototype design checkpoint, not
  the pipeline claim.
  **Cause hypothesis.** The flow program failed on cross-mode DENSITY CALIBRATION, which
  the MH proposal role does not need (miscalibration costs acceptance ∝ e^−Δ, not
  correctness). Laplace surrogates avoid SVI's ELBO variance-shrinkage mechanism (the
  measured sd_w 1.7–50× miss); their distinct failure mode — curvature ≠ spread on
  non-Gaussian shapes — is the thing M1/M2 measure. Prior evidence both ways: Laplace
  pocket-mass proxy 5.4% vs truth 9.57% (0.57 nats — encouraging); sibling-config C-20
  found the main basin is a CURVED RIDGE (791-nat straight-chord dip; 3/32 multistart
  within 5 logp) — a mechanism for main-side Gaussian collapse (q_main(z) tiny at ridge
  points enters cross-mode acceptance through the q(z)/q(z′) factor).
  **M1 — polish + Hessian spectra.** Adam-polish zM, zP to local stationary points z*M,
  z*P (lp ascent, ≤3000 steps, lr ladder 1e-3→1e-5 in whitened coords; record lp gain,
  final natural-units gradient ‖Σ_L^{1/2}∇lp‖, pocket membership retained). Then
  H = −∇²lp(z*) by forward-over-reverse (33 HVPs), symmetrize, eigendecompose.
  PREDICTION: both Hessians PSD (λ_min > −1e-8·λ_max, the float64 eig noise scale with
  ~1e2 margin); condition number 1e4–1e9 (lensing degeneracies); polish gain per point
  ≈ d/2 ≈ 16 nats (median-of-basin → peak; predict 5–30). FALSIFIER: λ_min < −1e-6·λ_max
  at a converged point (natural-grad < 0.5) ⇒ saddle/ridge at "mode" ⇒ Laplace-at-mode
  unsound here.
  **M2 — surrogate fidelity + MC jump acceptance (the decision measurement).**
  (a) Split MAMS64 by indicator (≈57.9k main / 6.1k pocket); per-mode empirical moments
  (μ_emp, Σ_emp); Laplace Σ_L = H⁻¹. Report KL(N_emp ‖ N_Laplace) in nats per mode +
  per-direction scale ratios √(gen-eigvals(Σ_emp, Σ_L)) (worst direction). Heuristic
  link (Jensen, stated as such; MC below is ground truth): within-mode independence
  acceptance ᾱ ≳ e^−KL. (b) MC acceptance: mixture proposal q with components at z*M,
  z*P, pipeline-realistic Laplace weights w̃_k ∝ exp(lp(z*_k) + ½ log det 2πΣ_k)
  (recorded vs 5.4% proxy and 9.57% truth; oracle-weight variant as diagnostic only).
  Three pre-registered proposal configs — P1 plain Gaussian Laplace; P2 armored:
  multivariate-t df=5, scale ×1.5 (the point-and-go default candidate); P3 pure
  translation jump z′ = z ± (z*P − z*M), symmetric pair (surrogate-free kernel; tests
  whether the modes are local translates). For each: 1024 proposals against 1024 stored
  equilibrium states per basin (subsample seed 0), α = min(1, exp(lp(z′) − lp(z) +
  log q(z) − log q(z′))) (P3: symmetric, q-terms drop); report mean ᾱ per
  (from-basin → to-basin) cell with binomial se. THRESHOLDS (derived from switch-count
  arithmetic at jump-attempt-per-step, w_P ≈ 0.1, 10k kept: round trips ≈ 900·ᾱ;
  W1a-class ESS_occ 200/10k needs ᾱ ≈ 20%; baseline-parity ESS_occ ≈ 19/10k needs
  ᾱ ≈ 2%): ᾱ_cross ≥ 20% full health; 2–20% clear improvement over plain MAMS;
  < 0.5% unworkable (≤ baseline). PREDICTIONS: pocket-side KL ≤ 3 nats and P2
  ᾱ(main→pocket) ∈ [2%, 30%] (pocket is narrow — peak +5.43 nats, volume e^−7.7 —
  hence most Gaussian); main-side KL 3–15 nats (curved-ridge prior); P3 ᾱ ~ 0.01–0.5%
  (volume-ratio argument: translation of broad-basin states into a e^−7.7-smaller mode
  lands outside support; central estimate e^−7.7 ≈ 0.05%). FALSIFIER for the direction:
  P1 AND P2 cross-mode ᾱ < 0.5% in BOTH directions ⇒ Laplace-mixture jumps unworkable
  on this posterior ⇒ escalate (flow-as-proposal or annealing mainline); P3 ≥ 2% with
  P1/P2 failed ⇒ translation-kernel prototype instead.
  **M3 — multistart enumeration probe.** 1024 prior-drawn starts through the existing
  gigalens MAP machinery (settings recorded from the carousel notebook/manifest; seed 0).
  Outcome: fraction with final z[6] > −22.35 AND lp ≥ lp(z*P) − 33 (typical-set
  half-width 2·d/2 ≈ 33 nats below peak). PREDICTION (low confidence, blocking
  assumption = unknown basin-of-attraction geometry under Adam from prior-width starts):
  1–50 of 1024 find the pocket; sibling-config prior (3/32 near global) predicts a large
  straggler fraction converging to neither peak — distribution recorded. FALSIFIER:
  0/1024 ⇒ enumeration link broken at ≤1024-start budget ⇒ annealing family becomes
  mainline for multimodal lenses regardless of M2.
  **Metric blind spots (named).** M1: curvature at the peak cannot detect ridge
  curvature / heavy tails at 2σ+ (exactly the C-20 warning) — covered by M2(a) empirical
  comparison and M2(b) true-lp MC. M2: equilibrium-state acceptance ignores
  adaptation-interaction (named untested link); pocket Σ_emp rests on chain-segregated
  draws from few chains — robustness check vs benchmark pocket draws (11.7k). M3: this
  prior/this lens only; finding the KNOWN pocket ≠ finding all modes.
  **Expected appearance.** M1: log-eigenvalue spectra, both strictly positive with wide
  spread. M2: scale-ratio scatter near 1 (pocket) / heavier main tail; acceptance
  histograms with non-vanishing cross-mode cells if hypothesis holds; falsified ⇒
  cross-mode cells hug 0 (< 5/1024 accepts). M3: final-lp vs z[6] scatter clustering at
  the two peaks (+ straggler cloud); falsified ⇒ single cluster + stragglers.
  **Cost.** Offline vs stored draws + ~5k fresh lp evals + 33×2 HVPs + one 1024-start
  MAP: ≈ 20–30 min wall on 1 GPU of a 4-GPU interactive allocation (M3 sharded if
  convenient); Slurm cap 60 min. No sampling runs.
  **Decision matrix (pre-committed).** M2 pass (any cross ᾱ ≥ 2%) & M3 ≥ 1 ⇒ draft
  jump-mixture MAMS prototype checkpoint. M2 pass & M3 = 0 ⇒ prototype retains research
  value with oracle modes; pipeline claim demoted; annealing branch opens. M2 fail &
  P3 ≥ 2% ⇒ translation-kernel prototype. All fail ⇒ report to human with options (b)
  flow-as-proposal / (c) annealing mainline; no auto-lever.
  **Grader revision items (round 1, 2026-07-09; all pre-registered before launch):**
  (i) Σ_L eigenvalue-floor rule, DERIVED from "no proposal axis may exceed the measured
  posterior extent": in H's eigenbasis, λ_H,i ← max(λ_H,i, 1/(4·λ_max(Σ_emp,mode))) —
  i.e. no Σ_L axis sd may exceed 2× the longest empirical posterior axis of that mode
  (Σ_emp from the MH-exact MAMS64 split; the eventual pipeline would substitute prior
  widths — noted as an untested substitution). n_floored axes recorded; any floored
  axis flags M1. Gray-zone reading pre-committed: λ_min ∈ [−1e-6, −1e-8]·λ_max ⇒ M1's
  PSD PREDICTION fails (curvature indistinguishable from flat at float64 along ≥1 axis)
  but the saddle FALSIFIER does not fire; Laplace proceeds only via the floor rule,
  carrying the flag. A P1+P2 double-fail is read as "unworkable" ONLY after the M2(a)
  per-direction scale ratios rule out a noise-inflated Σ_L axis as the cause.
  (ii) M2(b) reports a chain-clustered se (per-source-chain mean ᾱ; se = sd of chain
  means/√n_chains, n_chains recorded — pocket states come from few chains) alongside
  the binomial se; DECISIONS route on the clustered se: a cell passes only if ᾱ ≥ 2%
  AND ᾱ − 2·se_clust > 0.5% (distinguishable from the unworkable line); if ᾱ ≥ 2% but
  the margin fails, the pre-committed reading is "insufficient state diversity — widen
  states, do not route" (NEEDS-MORE-data, no matrix branch).
  (iii) The M2(a) benchmark pocket-shape robustness check excludes stuck chains
  (per-chain occupancy > 0.99; catches the documented frozen chain with occ 1.000 and
  ~4k near-duplicate draws).
  (iv) "M2 pass" = P1 or P2 ONLY, pipeline-realistic Laplace weights ONLY, cross-mode
  cells (states-in-M → pocket component; states-in-P → main component); P3 and any
  oracle-weight diagnostic never enter the pass criterion; ᾱ ∈ [0.5%, 2%) routes as
  FAIL in the matrix.
  (v) Polish-gain band restated from basin_slice records (grader derivation; the d/2
  median-vs-mode argument was wrong for a Gaussian): expected gain ≈ (median-draw lp −
  lp(z_med)) + d/2 ≈ 8.9 + 16.5 ≈ 25 nats (main) and ≈ 21 nats (pocket); a small
  pocket gain is NOT anomalous.
  (vi) M3 MAP settings pinned NOW: optax.adabelief(1e-2, b1=0.95, b2=0.99,
  nesterov=True if supported) — the production MAPStage default factory; n_samples
  1024, num_steps 4000 (manifest-matching), seed 0, start=None (prior draws),
  output_type "all". Polish whitening matrix = SVI qz_scale_tril from
  `messy_tests/dpie/svi/arrays.npz` (frozen; measurement coordinates only).
  **Grader revision items (round 2, 2026-07-09):** (vii) polish best-iterate
  off-by-one fixed (best_dw now stores the iterate whose lp is recorded; the final
  iterate's lp is also assessed) — the grader caught that H/Σ_L/w̃/cutoffs were
  anchored at a point provably not the one whose lp was reported. (viii) M3
  classification recomputes lp at final_z (the production MAP "all" output pairs
  post-update params with pre-update lp — one-step offset, recorded as
  lp_recompute_vs_lib_max_abs; not patching the library for this gate). (ix) COST
  amended: M3 pipeline-realistic 1024×4000 ≈ 26 min on 4 GPUs (manifest basis
  128×4000 = 790 s), total ≈ 45 min wall, Slurm request 90 min. (x) The M2 ᾱ cutoffs
  were derived at w̃_P ≈ 0.1 but w̃_P is a run OUTPUT (prior record suggests ~0.054):
  decisions route on the per-step cross-jump rate w̃_P·ᾱ ≥ 19/9000 (parity) /
  ≥ 200/9000 (W1a-class) — implemented as rescaling the 0.5%/2%/20% ᾱ lines by
  0.1/w̃_P when the measured Laplace weight falls outside [0.05, 0.2]
  (threshold_scale recorded). (xi) Matrix pins: "M3 ≥ 1" reads the FINAL-step counts
  (best-step is diagnostic only); the P3 branch requires BOTH P3_MtoP and P3_PtoM
  ≥ 2%; the oracle-weight acceptance variant IS implemented (logq-only recompute on
  the same proposals; recorded per cell as oracle_weight_mean_alpha_diag, excluded
  from all verdicts); the model card records smoke mode so a smoke-run summary can
  never be mistaken for the measurement.
  **Grader round 3 (2026-07-09): CERTIFY-RECOMMENDED, conditional.** All round-2
  items verified in entry and script. Launch conditions: (1) plot-keys fix at the M2b
  acceptance histogram (threshold_scale/cell_verdicts entries crashed the bar plot;
  measurement unaffected — summaries and npz are written before the plot stage; the
  cell_verdicts half predates round 3 and was missed by the grader in round 2) —
  APPLIED, and the plot's threshold lines now scale by threshold_scale; (2)
  GATE_L_SMOKE=1 completes end-to-end with all four PNGs before the real run. Reading
  notes for the result grader: PtoM cells use the conservative scale; M1 natural-grad
  uses floored Σ_L — read with n_floored; cond field meaningless if λ_min ≤ 0
  (recompute from eig_min/eig_max). Scope unchanged: Link 1 only; two known modes;
  this lens/prior; equilibrium-state acceptance.

- **Run: carousel GATE Fv6 — Phase-B ELBO-early-stop (human-directed 2026-07-09 after the
  Fv5 high-side escalation; user's balance question answered and design adjusted
  accordingly: the stop rule and the pass gates use DISJOINT instruments).**
  (script `carousel_gate_f.py`; Phase A reused from the r10b28 cache — no retrain.)
  **Design:** rerun Phase B (28 bins, lr 1e-4, max 4000 steps) from the cached Phase-A
  flow with EARLY STOPPING ON THE DIRECT ELBO: every 250 steps evaluate the 5×128-draw
  neg-ELBO with FIXED keys (common random numbers; measured fixed-key resolution ~0.3
  nats vs ±8 across-key); STOP (grader-revised rule) on TWO CONSECUTIVE checks where
  metric − running-best > max(2·se, 4.0 nats) — the floor is derived from this stack's
  observed benign optimizer transients (~+4 nats) vs the CRN estimator's 2·se ≈ 0.23
  nats at healthy params (hair-trigger) and ≈ 7 nats at damaged params (desensitized);
  patience 2 costs ≤ 250 extra steps. REVERT to the best-ELBO checkpoint's params.
  Rationale (user's concern, addressed): ELBO is mode-seeking — blind to pocket
  UNDER-coverage — so it is NEVER asked to certify coverage; it detects the two observed
  Phase-B failure modes (bulk damage; gross over-weighting, which KL(q‖p) charges in
  proportion to misplaced mass) while being ≈ flat along legitimate fkl progress.
  Coverage is judged ONLY by the unchanged gates: ratio ∈ [+3.43, +7.43] (known truth
  +5.43), pocket-mass band [5%, 13%], held-out sd(lp − log q) ≤ 2 nats over 512 pocket
  draws, pullback-scale. The stop rule never sees any gate statistic (no self-grading);
  the RATIO TRAJECTORY is recorded every 250 steps (2 renders/check) as a DIAGNOSTIC
  ONLY, pre-registered as excluded from the stop decision.
  **Cache-tag fix (pre-registered):** the flow tag now encodes the Phase-B schedule
  (e.g. AB_es250 suffix) — the Fv5 log entry recorded that phase_b_steps was absent from
  the key and would silently reuse the 4000-step cache.
  **Predictions:** the ELBO trace stays ≈ flat (within 2 sd of Fv4's SVI−21 level) for
  an initial stretch and then degrades — the stop fires BEFORE step 4000 (Fv5 proved
  degradation by then); at the stopped checkpoint the ratio lies IN the band (mechanism:
  at 28 bins the ratio provably transits +1 → +13.9, and overfit damage and ratio
  overshoot were coupled in Fv5, so stopping at bulk-health should land mid-transit).
  **Falsifiers + pre-committed readings:** (i) stop fires but ratio still < +3.43 ⇒
  capacity/coverage progress is slower than bulk damage ⇒ the A2 anchor branch becomes
  available (its original arming condition, one pass, all recorded guardrails); (ii)
  ELBO stays healthy to 4000 AND ratio > +7.43 ⇒ overfit ruled out ⇒ prime suspect =
  the TRAINING DATA's pocket profile (chain-segregated 9.6% occupancy, ~2× uncertainty;
  the named data-limited alternative) ⇒ NEGATIVE finding for fkl-on-this-data, human
  escalation — new data, not new knobs; (iii) stop fires in-band on ratio but an
  independence check fails ⇒ warping-style pathology without the anchor ⇒ human
  escalation with artifacts; (iv) ELBO never recovers to ≤ SVI at any checkpoint ⇒
  Phase B at 28 bins is bulk-destructive from step ~0 ⇒ joint evaluation, human
  escalation; (v — grader item) stop fires (or run completes) with the BEST checkpoint's
  ratio > +7.43 while its ELBO is healthy ⇒ same reading as (ii): data-profile suspect,
  human escalation, no auto-lever. A2 note: branch (i) re-arms the SAME single A2
  allowance carried unspent from Fv5 (whose high-side branch excluded it) — a carry-over,
  not a reset. PRE-REGISTERED INSTRUCTION (grader's transit-granularity warning): the
  ratio-diagnostic trace is read BEFORE choosing any branch — if the ratio transits the
  band entirely between two 250-step checks while ELBO stays flat, the finding is that
  250-step granularity cannot resolve the transit, and no branch is entered on a
  mischaracterized endpoint. Winner's-curse note (grader): the stop selects the min-ELBO
  checkpoint of ~16, and the ELBO gate then tests the selected flow — bias bounded by the
  across-key se (~3 nats), stop/gate evaluations use different key families, vs 21 nats
  of gate headroom: acceptable, recorded.
  **Cost (measured basis, grader item):** Fv5 attempt-2 telemetry — 24.5 min total on
  4 GPUs INCLUDING compile (~8), Phase A (11.4), and the full 4000-step Phase B (≈4–5) —
  so Fv6 on 4 GPUs with Phase A cached: compile ~8 + Phase B ≤ 5 + 16 ES checks (each
  5×128 = 640 forward renders; ≈ 8 min total; the ratio diagnostic is spline-only,
  0 renders) + gates ~4 ⇒ **≈ 25 min wall, Slurm-capped 45 min on 4 GPUs** — falsifier
  branch (ii) (reach step 4000) is safely inside the cap. GPU-h not the constraint per
  the human; wall is.**
  **Status: Fv6 approved 2026-07-09 (rigor-grader, second round; ES test executed by
  grader, sacct telemetry independently confirmed). Launch: 4 GPUs, GATE_F_SKIP_PILOT=1,
  45-min cap. Grader's watch point: the first ES check (step 0) exercises es_eval on
  mesh-annotated mid-training params on real hardware for the first time — surfaces
  within minutes if it fails.**

- **Run: carousel GATE Fv5 — the plan-§5.4 ladder's ONE architecture escalation
  (human-directed 2026-07-09: "I'd like to try A1 and A3, with A2 as a fallback if that
  doesn't work"; also "Don't worry too much about GPU hours... I'm more concerned about
  wall-clock time. Go ahead!")** (script `carousel_gate_f.py`, same apparatus as Fv4).
  **Target mechanism (from benchmark attempt 2 + corner diagnosis):** the Fv4 flow's
  pocket MASS is right (8.7% vs 9.6% training data) but its peak DENSITY is ~80× low
  (+1.0 vs +5.43) — density spreading, localized to resolution: the pocket sits at
  ‖w‖∞ ≤ 8.82, in the OUTERMOST cells of the 14-bin grid (spacing 1.43 vs pocket extent
  ~2–3 cells), where uniform-init knots are coarsest. The user's corner reading concurs
  ("covers the main mode very well... doesn't extend completely over the second mode").
  **Changes (A1+A3):** (A3) num_bins = 2× the demo-v3 rule → 28 (spacing 0.71; pocket
  gets ~4–6 cells); layers/range/lr unchanged (R·lr still 0.03 — bins exonerated for
  stability by the CPU diagnosis); (A1) Phase B 1000 → 4000 steps (fkl slope at stop was
  −0.0028 nats/step, still descending). Cache key car_std_r10b28ts0lr0.003.
  **TIGHTENED pocket gate (pre-registered):** the escalation PASSES only if the gate
  ratio matches the KNOWN truth: |ratio − 5.43| ≤ 2 (trap depth ≤ e² ≈ 7×); the old −8
  coverage floor is reported but no longer sufficient. All other Fv4 gates (ELBO ≤ SVI,
  pullback-scale) must hold as before.
  **Predictions:** ratio reaches +5.43 ± 2 (mechanism: mass is already right; 2× local
  resolution + 4× fkl steps lets forward-KL concentrate it); ELBO ≈ SVI−21 ± few
  (unchanged — bulk already fit); pullback gate still passes (sd [0.5,2]).
  **Tolerance derivation (grader item):** ±2 ⇔ trap depth ≤ e² ≈ 7.4×; attempt-2 scaling
  (depth ≈ 84 → 14.6 switches/1000) implies depth ≲ 20 suffices for the W1b 60/1000 bar
  under linear exit-rate scaling (UNCERTIFIED), so the band carries ~3× margin.
  **Falsifiers + pre-committed responses:** ratio < +3.43 after A1+A3 ⇒ A2 fallback,
  ONE pass (ratio-anchoring auxiliary loss λ(log q(zP) − log q(zM) − 5.43)²; λ pinned by
  RULE: λ = |fkl loss at A2 start| / (ratio at A2 start − 5.43)², i.e. anchor term =
  fkl magnitude at init, both recorded; the user is WARY of pre-determined posterior
  information — A2 uses only two likelihood evaluations, recorded as such, the LAST
  flow-side lever). **ANY A2 pass additionally requires INDEPENDENT pocket-wide checks
  (grader item — the anchored gate is A2's own objective, not a test):** (i) flow-sample
  pocket mass (64k draws, fixed seed) ∈ [5%, 13%] (band = estimator spread 4.6–9.6% with
  margin; note the Fv4 mass estimate 8.7% carries ±0.5pp seed sensitivity — grader
  recomputed 8.3%); (ii) sd of (lp − log q) over 512 random pocket draws ≤ 2 nats (all
  pocket draws are held out w.r.t. the 2-point anchor); plus the usual ELBO + pullback
  re-checks. **High-side falsifier (grader item):** ratio > +7.43 after A1+A3 ⇒ read as
  Phase-B OVERFIT to the ~6.1k correlated pocket draws (A1's known risk; no held-out fkl
  monitoring exists) ⇒ evaluate jointly with ELBO/pullback and ESCALATE TO THE HUMAN —
  no auto-lever. **Mechanism-falsification reading (grader item):** ratio unchanged
  (≈ +1) after A1+A3 ⇒ the RESOLUTION mechanism is FALSIFIED and recorded as such,
  independent of A2's subsequent outcome; ratio materially improved but short ⇒
  mechanism supported, capacity short. A2 also fails ⇒ the escalation is spent:
  pre-registered NEGATIVE result for flow-MAMS on this target, budget moves per plan
  §5.4 (many-chain scaling), human informed. ELBO or pullback REGRESSES under A3
  (28 bins hurt the bulk) ⇒ evaluate jointly, no silent knob iteration.
  **Named alternative (grader):** the fkl target is the EMPIRICAL 64k-draw distribution
  (~6.1k correlated, chain-segregated pocket draws, ~2× occupancy uncertainty) — if the
  data's own pocket density profile is off, no resolution reaches +5.43 and a miss is
  data-limited, not capacity-limited; the A2 branch covers the consequence but the cause
  would need new data, not new knobs. **Training parallelism (human-directed):** Phase-A
  chunks and Phase-B chunk-rounds are data-parallel across all visible GPUs (GSPMD
  vmap-over-sharded-chunks inside the existing scan; verified on 4 virtual CPU devices
  to match the sequential path to 9e-16; per-device memory = the validated per-chunk
  footprint; stream matches the sequential chunked path — recorded). Expected wall:
  Phase A ~10–12 min (was 40), Phase B ~6–8 min at 4× steps (was 6 at 1×), gates ~3.
  **Cost: ≤ 45 min wall on a 4-GPU node (≤ 3 GPU-h), Slurm-capped at 60 min. The gate-F
  timing pilot is SKIPPED (GATE_F_SKIP_PILOT=1 in the launch command — the in-script
  90-min pilot budget is stale vs the 45-min wall; grader item).** Deviation note
  (grader): plan-§5.4's ladder wording "double layers/bins" is instantiated as BINS ONLY
  (mechanism-grounded: resolution, not depth; human-directed). n_devices recorded in the
  model card (trained values are device-count-dependent, amendment-v2 precedent); the
  GSPMD equivalence test is committed at
  experiments/flow_precond/instability_diagnosis/gspmd_equiv_test.py (grader reproduced
  1.1e-16 independently). Phase-B 28-bin chunk footprint ≈ 2× the validated 14-bin one
  (~5 GiB est. per 8000-row chunk — in budget on A100-40; per-device memory under GSPMD
  is inferred from code semantics, not measured — a fast OOM inside the Slurm cap is the
  bounded worst case). If the tightened gate passes: the RE-BENCHMARK (same design as
  attempt 2, new flow) returns to the human for explicit go.
  **Status: Fv5 approved 2026-07-09 (rigor-grader, second round; all eight revision items
  verified in the artifacts; equivalence test reproduced by grader). Launch authorized:
  4-GPU node, GATE_F_SKIP_PILOT=1, Slurm-capped 60 min. Scope: flow training + gates only;
  A2 exercisable ONCE under the recorded guardrails; re-benchmark = human go; a gate pass
  is NOT a benchmark prediction beyond the recorded uncertified ~3×-margin scaling
  argument.** ATTEMPT 1 (2026-07-09) CRASHED pre-training: XLA collective Rendezvous
  deadlock + abort (AwaitAndLogIfStuck) at the first Phase-A step on 4 real GPUs — the
  GSPMD auto-sharding variant; the CPU virtual-device test structurally cannot exercise
  NCCL (recorded caveat proved out). Environmental fix, same estimator: parallel branches
  rewritten to the MANUAL shard_map pattern proven on this cluster by the MAMS kernel
  (renderer inside shard_map, psum over the device axis — ran the full 4-GPU benchmark);
  equivalence test re-run as committed: PASS, identical values (8.9e-16 / 5.6e-17).
  Cost of attempt ≈ 1.3 GPU-h (deadlock+abort ~20 min × 4). Relaunch with fail-fast log
  watch.

- **Run: carousel BENCHMARK (§5.4) — flow-MAMS vs vanilla MAMS** (HUMAN approval required
  by standing pre-commitment; grader pre-review first). **Question:** does preconditioning
  MAMS with the Fv4 A+B flow (car_std_r10b14ts0lr0.003 — pocket ratio +1.0, pullback sd
  0.96–1.01, ELBO SVI−21) fix the carousel's between-basin mixing failure?
  **Baseline (no new GPU), MEASURED (grader-corrected, coordinator-verified):** the
  existing MAMS64 run (64×1000): per-chain pocket occupancy spans [0.001, 0.951] with
  sd = 0.214; ALL 64 chains visit both basins (median 12 switches/chain, min 2) — the
  failure is NOT absent transits but DWELL DISEQUILIBRIUM: implied per-chain occupancy-ESS
  ≈ 1.9 (from sd² ≈ p(1−p)/ESS at p = 0.096); pocket-column rank-normalized split-R̂ =
  1.184; worst-param τ ≈ 3600. (The earlier draft's "sd ≈ 0.42" and "most chains never
  switch" were WRONG — corrected against the arrays before human review.)
  **Test arm (HUMAN-AMENDED 2026-07-08, verbatim: "Yeah, 8x4000 is a good idea. That's my
  only modification to the test. Go ahead!" — basis: user's testing experience that 8
  chains behave ≈ as well as 64 on this posterior at ~4× the speed):** MAMS in u-space
  through the Fv4 A+B flow (demo-validated TransformedProbModel + MAMS plumbing, GATE I
  bit-identity heritage), **8 chains × 4000 kept steps** (per-chain length 4× the
  baseline's — mixing is a per-chain-length phenomenon, so this strengthens the W1/W2
  statistics at ≈ the wall of the originally-planned 64×1000), burnin = the baseline's
  actual adaptation length = **2000 steps** (read from diagnostics.npz array shapes; the
  mams run stored (64, 2000) adaptation traces + (64, 1000) kept). Per-chain criteria
  NORMALIZED per-1000-kept-steps for baseline comparability. Gradient evaluations and
  wall time recorded for the test arm; the baseline's results-phase gradient count and
  wall are UNRECOVERABLE from its stored diagnostics (adaptation-phase only) ⇒ the
  pre-committed fallback normalization activates: W4 compares ESS/kept-step (per-1000),
  with the test arm's ESS/wall and flow-eval wall fraction reported alongside (baseline
  wall unknown — recorded as a limitation, not silently dropped). All W1–W3 diagnostics computed on
  DECODED z (= T(u)), never raw u (C-8 descendant); occupancy column is z[:,6] post-decode.
  **Pre-registered win conditions (thresholds DERIVED, not rounded):**
  (W1, THE sharp instrument — between-basin mixing; PREDICTION with direction+magnitude:
  the Fv4 flow maps both basins into the base bulk — pocket pullbacks max|u| = 4.11 — so
  u-space MAMS should transit freely; predicted occupancy-ESS ≥ 20 per chain per 1000
  kept, i.e. ≥ 10× the baseline's 1.9): (W1a) occupancy-ESS ≥ 20/chain/1000-kept — at
  8×4000 this means per-chain-occupancy sd ≤ sqrt(p(1−p)/80) ≈ 0.033 at p = 0.096
  (raw sd threshold rescales with chain length; the per-1000 ESS bar is the invariant;
  baseline: 1.9); (W1b) switches per chain per 1000 kept: median ≥ 60 (5× baseline's 12)
  AND min ≥ 12 (every chain ≥ the baseline's median) — at 4000 kept: median ≥ 240,
  min ≥ 48 raw; NOTE the 8-chain median/min are coarser order statistics than the
  64-chain versions (recorded); (W1c) occupancy-indicator R̂ ≤ 1.05, defined as PLAIN split-R̂ on
  the binary indicator (rank-normalization is a no-op on binary data); baseline =
  **1.719** (the continuous pocket-column rank-R̂ 1.184 is a different, milder metric,
  reported separately above) — the indicator R̂ is the sharpest single diagnostic in this
  packet.
  (W2, plausibility band — RE-DERIVED as pre-committed) pooled occupancy ∈ [2%, 15%]:
  the plan-§5.4 [2%, 8%] presumed truth ≈ 5%; estimators span 4.6–9.6% (Laplace 5.4,
  MAMS8 4.6, MAMS64 9.6 at ~2× uncertainty) — band covers the spread; W1 carries the
  scientific weight. (W3, health) R̂ ≤ 1.02 all params on decoded z; bulk-ESS reported,
  direction flow ≥ baseline (no absolute floor: the baseline's pocket-dim ESS is itself
  the failure under test — a fixed floor would be arbitrary). (W4, efficiency) min
  bulk-ESS per GRADIENT EVALUATION ≥ 3× baseline; gradient counts taken from each run's
  MAMS diagnostics (steps × trajectory lengths); PRE-COMMITTED fallback if the baseline's
  count is unrecoverable from its manifest: normalize by ESS/kept-step AND ESS/wall-s
  (same hardware class, noted) — chosen NOW, not post-hoc.
  **Plan-§5.4 deviations, enumerated (each else silent):** (i) ESS threshold 5×→3× —
  first multimodal target; 5× was calibrated on unimodal expectations; (ii) normalization
  ESS/wall-s → ESS/gradient — isolates kernel efficiency from implementation overhead;
  CAVEAT: this hides flow-eval cost, so (iii) the plan's "flow overhead < 5% of step
  cost" criterion is carried as a MANDATORY REPORT (flow-eval fraction of wall, both
  arms), not a gate — overhead optimization is the human's stated "paring down later"
  phase; (iv) the plan's escalation ladder is CARRIED verbatim: ESS gain < 2× ⇒ escalate
  architecture ONCE (grader-gated); still < 2× ⇒ pre-registered negative result, budget
  moves to many-chain scaling.
  **Falsifiers + pre-committed readings:** W1 fails ⇒ the flow preconditions geometry but
  MAMS still cannot cross in u-space ⇒ NEGATIVE finding for the flow-MAMS mechanism on
  multimodal targets (the flow itself remains validated by GATE Fv4); no retuning
  iteration without a new checkpoint. CAP CAVEAT (pre-committed, amendment v2): the
  W1-fail and W4-fail readings apply as written ONLY if the results-phase trajectory-cap
  binding fraction ≈ 0; a fail with a materially binding cap has an alternative
  explanation (truncated trajectories) ⇒ diagnose before attribution, no
  reclassification. A W4 PASS under a binding cap remains valid (conservative
  direction). Expected: ≈ 0 (baseline max n = 38 — from its stored ADAPTATION-phase
  traces; results-phase n is implied by the frozen final L/ε — vs cap 60). The script
  reports the capped fraction split burnin vs results. W1 passes but W2 fails ⇒ mixing works, occupancy
  disagrees with all estimators ⇒ escalate to human with the PRE-REGISTERED evidence
  standard for "genuine measurement of the pocket mass" (note: BOTH arms are MH-exact in
  law — exactness is not the discriminator, CONVERGENCE is): occupancy-indicator R̂ ≤ 1.02 (plain split-R̂ on
  the indicator, as in W1c) across the 64 flow chains AND first-half/second-half pooled
  occupancy agreement within 1.5 percentage points AND explicit comparison against the Laplace proxy (5.4%); a
  converged mixing sampler's estimate legitimately supersedes segregated chains' — but
  only with that evidence; human review before any claim regardless. W4 fails while W1 passes ⇒ mixing win at
  efficiency cost — report both, no reclassification. **Known adverse signal (carried per
  standing condition):** demo v3 arm C (A+B flow, SAMPLED) failed health at demo scale
  (R̂ 1.076, ESS 232) — argument for difference, not proof: that flow was the
  unstandardized range-35 config with poor conditioning; the Fv4 flow's near-perfect
  pullback (sd 0.96–1.01 vs demo arm C's unmeasured-but-poor geometry) is exactly the
  property that failure implicated. The human should weigh this explicitly.
  **Cost: ≤ 4 GPU-h**, pilot-gated (time 20 steps post-compile, project, abort >4 h);
  cumulative project spend to date ≈ 5.6 GPU-h.
  **Also offered for certification alongside this decision:** (i) the F1 mode-dropping
  result (−108.8/−406/−16.4 across three architectures); (ii) the R·lr spline-instability
  mechanism (diagnosis entry); (iii) the Fv4 gate results.
  **Status: grader pre-review PASSED 2026-07-08 (three rounds; baseline claims
  independently recomputed and corrected, W1 rebuilt against measured reality, W1c
  estimator fixed); human APPROVED 2026-07-08 with the 8×4000 amendment.**
  **ATTEMPT 1 (2026-07-08/09) FAILED — full 4 GPU-h allocation burned, no science:**
  the in-run timing pilot's second leg (20+20 steps) hung ~3.6 h until the Slurm wall
  limit (first leg completed in 149.6 s incl. compile; flow identity check PASSED on GPU,
  1.0029454473771153 vs recorded ...69874). Mechanism hypothesis (UNCERTIFIED, later
  supported by kernel analysis): truncated adaptation schedules can strand step_size
  mid-transient and n = L/ε is UNBOUNDED in the kernel (only a floor). Design lesson:
  short-burnin MAMS pilots are structurally unsound — the pilot meant to protect the
  budget consumed it. A separate 25-min false start (245-min salloc vs 240-min QOS cap)
  cost 0 GPU-h. Cumulative ≈ 9.6 GPU-h.
  **AMENDMENT v2 (2026-07-09, prepared at human direction; grader re-confirmation
  pending):** (i) in-run pilot REMOVED (budget bounded by the Slurm wall limit instead);
  (ii) kernel trajectory-length cap `max_num_integration_steps=60` added to MAMS_JIT at
  the human's direction ("HMC has a similar rule... 60ish steps"), mirroring TFP's
  GradientBasedTrajectoryLengthAdaptation clip (= gigalens-old HMC's max_leapfrog_steps=30
  precedent): Halton MEAN clamped at N_MAX/2 pre-jitter (jitter family preserved) + L
  anti-windup clamp (L ≤ N_MAX/2·mean ε) + Hist.traj_capped diagnostic; CPU tests:
  bit-identical when not binding (healthy 200+200, all Hist fields tobytes()-equal;
  baseline p99 n=37, max 38 vs cap 60 ⇒ never binds when healthy), hang class bounded
  (5.8 s vs ~5-day projection), controller self-correcting (capped fraction → 0 during
  tuning, no L windup); one falsifier revision honestly recorded (>= vs > on the capped
  flag, mams_cap_notes.md §4). (iii) Run SHARDED across 4 GPUs (2 chains/device;
  shard-map path verified on 4 virtual CPU devices: mesh auto-discovery, 8%4=0 exact,
  psum-shared mass matrix; distribution gates pass). RECORDED: the sample stream is
  DEVICE-COUNT-DEPENDENT (reduction order chaos-amplifies through adaptation) — the
  4-GPU run is a different, equally-valid realization; device count recorded in the
  model card alongside seed. (iv) Expected wall ~30-45 min on 4 GPUs; allocation capped
  at 60 min ⇒ worst case 4 GPU-h charge, expected ~2-3.

- **Run: carousel GATE Fv4 — frozen measured scale + data-derived box (diagnosis-grounded
  revision of Fv3)** (script `carousel_gate_f.py`; supersedes Fv3 after its pilot abort;
  same gates, budgets, Phase-B data, and pilot-with-abort as Fv3 unless stated).
  **Config:** whitening L′ = L_SVI · diag(sd_w) with sd_w = per-dim sd of the whitened
  MAMS64 draws, FROZEN (measured, never ELBO-trained — the Fv2 rule; sd_w ∈ [1.70, 50.25]);
  on the standardized coords the pre-registered rules give range = ceil(1.1 × max|w′|)
  = **10**, bins = ceil(range × 48/35) = **14**; R·lr = 10 × 3e-3 = **0.03**, inside the
  CPU-measured stable regime (≤ 0.035; the diagnosis Log entry has the mechanism: knot
  decoder amplifies adam's coherent first-step logit kick by O(R), bins exonerated).
  Containment: all Phase-B data in-box by construction (max|w′| = 8.97 ≤ 10);
  zP/zM at 5.24/4.31. Phase-B chunks back to 8 (14-bin tensors < the validated 24-bin
  footprint). Cache key car_std_r10b14ts0lr0.003. CPU-grid evidence for THIS regime:
  synthetic r11b16 at lr 3e-3 stable (diagnosis run (d)) — CAVEAT: the pre-standardized
  synthetic init was exactly optimal, so (d)/(e) test stability-near-optimum, not
  convergence-from-afar (the real situation; a +4-nat step-10 transient from the same
  mechanism appeared even there). Convergence support comes from R·lr = 0.03 sitting in
  the regime where every real run to date trained stably (demo v3 0.035, v4 0.018,
  Fv2 0.048), and the flow-gate + loss plots will adjudicate.
  **Predictions:** (G1–G4 as Fv3, expectations updated) — G1 A-only fails pocket gate
  (≈ −100s); G2 A+B PASSES (≥ −8; ≈ +5.4 if well-covering); G3 both ≤ SVI — RE-REGISTERED (grader
  correction): with L′ = L·diag(sd_w) the identity-init pushforward is the moment-matched
  OVERDISPERSED Gaussian N(loc, L·diag(sd_w²)·Lᵀ), NOT the SVI Gaussian, so (i) step-0
  loss will be ≫ SVI — EXPECTED, not a bug — and the step-0=SVI nesting check used in
  F/Fv2/Fv3 is UNAVAILABLE in this family; (ii) "≤ SVI" is an ε-approximate FAMILY-CAPACITY
  bound (couplings must LEARN the per-dim compression — ~2–3 knots/dim suffice in
  principle) reached by optimization-from-afar, adjudicated by the loss plots; the
  (SVI−20, SVI] intermediate reading carries; (iii) PRE-COMMITTED Phase-A hard-divergence
  response (overdispersed base draws reach ~50-whitened-sd corners where lp/∇lp finiteness
  is untested — arm-D leg-A precedent): if the train_flow divergence guard fires ⇒ ONE
  retry at lr 1e-3; fires again ⇒ human escalation with the recorded traces, no further
  improvisation. Containment is UNAFFECTED by the init change (data pullbacks at identity
  are exactly w′, max 8.97 in-box); G4 the pullback-scale gate now tests
  SHAPE, not scale (per-dim sd handled by the frozen rescale): A+B required to pass;
  A-only recorded, sd plausibly in-band with |mean| the informative part. All falsifier
  branches and pre-committed responses CARRY OVER from Fv3 verbatim (G2-fail-in-box →
  ONE subsample/early-stop retry then human escalation; G4-fail-while-G2-passes → partial
  win, human go/no-go; G1-wrong → investigate mechanism; G3-A-only → lr fallback;
  G3-only-A+B → Phase-B evidence; pilot projection > 90 min → abort + re-checkpoint).
  Blind spots carry over, plus: (d) sd_w estimated from 64k correlated draws
  (chain-segregated; the ~2× occupancy uncertainty propagates into sd_w of the pocket
  dims) — but range is derived from the SAME standardized data, so containment is
  self-consistent regardless of sd_w estimation error; measured main-basin/pooled sd
  ratio ∈ [0.59, 1.03] (grader), i.e. pooled inflation is bounded and in the conservative
  extent-shrinking direction; the residual exposure is base-shape mismatch, O(1) and
  exactly what in-box splines fix. **Cost: ≤ 90 GPU-min**, pilot-gated as before.
  **Status: Fv4 approved 2026-07-08 (rigor-grader, second round; derivation numbers
  independently reproduced; step-0=SVI prediction error caught and re-registered
  pre-launch); run authorized at ≤90 GPU-min, pilot-gated.**

- **Run: carousel GATE Fv3 — data-derived range/bins (plan-§6 path, v3-validated recipe)**
  (script `carousel_gate_f.py`, same gates/apparatus as F/Fv2; commit follows this entry).
  HUMAN CONCURRENCE obtained 2026-07-08 after the pre-committed Fv2 escalation, verbatim:
  **"Okay, sounds good. Go ahead with the data-derived range/bins, since we should get a
  working version sampling the carousel before we start paring down and going for
  efficiency."** — priority is a WORKING carousel sampler; efficiency later.
  **Config (derivation rule pre-registered, computed+printed in-run):** range =
  ceil(1.1 × max|w|) over the whitened MAMS64 draws = **357** (all Phase-B data in-box BY
  CONSTRUCTION); bins = ceil(range × 48/35) capped at 512 = **490** (keeps demo-v3 knot
  density); trainable_scale **OFF** — the Fv2 lesson elevated to design rule: containment
  comes from measurement, never from an ELBO-trained parameter. Phase A/B budgets
  unchanged (3000×128 reverse-KL, lr 3e-3, 4 chunks; 1000-step full-batch fkl in 32×2000
  chunks — exact). Cache key car_r357b490ts0lr0.003.
  **Predictions:** (G1) A-only fails the pocket gate (mode-dropping is objective-level —
  both boxes proved it; ≈ −100s or worse); (G2) **A+B PASSES the pocket gate** (≥ −8;
  ≈ +5.4 if well-covering): fkl now has spline capacity everywhere its data lives — this
  is the configuration the whole chain argues for; (G3) ELBO gate: both ≤ SVI. Weaker than
  F3′'s "≤ SVI−20": 490 uniform-init knots over ±357 give the bulk only ~1.5-whitened-unit
  initial knot spacing (core dims sd 1.7) — knot positions are trainable but optimization
  quality at this ratio is unknown; PRE-COMMITTED intermediate reading: (SVI−20, SVI] =
  optimization-quality finding, not method failure, no post-hoc reclassification. (G4)
  pullback-scale gate = the REAL capacity question: whitened per-dim sd spans 1.7–50.2, so
  whitening requires up to ~50× per-dim compression learned by fkl through knot
  allocation. Required: A+B main-basin sd ∈ [0.5, 2], |mean| ≤ 1. A-only recorded,
  predicted to FAIL sd (reverse-KL never sees the ridge tails).
  **Falsifiers + pre-committed responses:** G2 fails (with data in-box by construction) ⇒
  genuine fkl fit problem ⇒ the re-armed ONE retry applies (subsampling/early-stopping);
  still fails ⇒ negative finding + human escalation. G4 fails while G2 passes ⇒ flow
  covers the pocket but cannot whiten the ridge ⇒ PARTIAL win — escalate to human with
  the u-space-geometry evidence for a benchmark go/no-go (u-space MAMS may still beat
  z-space even unwhitened). G3 A-only fails ⇒ lr fallback (ONE retrain at 1e-3) — carried
  over. G1 WRONG (A-only passes the pocket gate — newly plausible at range 357 where
  reverse-KL draws have more room to wander) ⇒ GATE F's carried-over response: investigate
  the mechanism before any claim. G3 fails ONLY for A+B ⇒ Fv2's carried-over response:
  evaluate as Phase-B evidence jointly with G2/G4, no lr iteration.
  PRE-COMMITTED TIMING PILOT (after two consecutive budget overruns): before the main run,
  time 4-vs-8-step pilot pairs for Phase A and Phase B at 490 bins (compile cancels in the
  difference), project the total, and ABORT + re-checkpoint if the projection exceeds
  90 min — the budget is a gate, not a hope. Rationale: the Phase-B tensors scale with
  spline pcount (71→1469, ~21×) and Phase B is render-free, so "render-dominated" does not
  cover it; Phase-A step time also grew an unexplained 2× at 8→24 bins. **Blind spots:** (a) knot-allocation dynamics unmeasured at 490 bins (loss plots
  will show); (b) pocket gate still 2-point; (c) demo cross-validation of THIS recipe =
  v3 demo arm B (range 35/bins 48, passed; the derivation rule instantiated on the demo
  reproduces exactly that config) — the recipe, not the constants, is what transfers; no
  new demo run needed for GATE Fv3 (flow gates only). CAVEAT carried visibly to the
  benchmark checkpoint: the A+B-SAMPLING side of this recipe holds a standing demo-scale
  negative signal — demo v3 arm C (A+B flow, sampled) FAILED health (R̂ 1.076, ESS 232) —
  and the §5.4 benchmark samples with the A+B flow; the human benchmark decision must
  weigh that signal (this is the surviving residue of the Fv2 demo-re-validation
  obligation). **Cost: ≤ 90 GPU-min** (Phase A ~60 measured-based,
  render-dominated; Phase B ~20 at 32 chunks; gates ~5).
  **Status: Fv3 approved 2026-07-08 (rigor-grader, second round; all derivation numbers
  independently reproduced); launch authorized at ≤90 GPU-min with the timing pilot's
  projection as a hard in-run gate. Operational note (grader): on any RESUMED run either
  set GATE_F_SKIP_PILOT=1 or delete the PILOT* caches — cached pilot legs would print a
  bogus projection.**

- **Run: carousel GATE F — one-shot flow training + pocket-coverage A/B** (plan §5.2/§4.4;
  script `experiments/flow_precond/carousel_gate_f.py`, branch `flow-precond-mams`; model
  via `carousel_model.py`, identity-verified to 0.01 nats against basin_slice records).
  Trains Phase-A-only and Phase-A+B one-shot flows (demo-v4-validated architecture: range 6
  / 8 bins / trainable DiagScale / lr 3e-3, 3000×128-draw reverse-KL; Phase B 1000-step
  full-batch fkl) on the 33-param carousel-dPIE posterior.
  **DEVIATION from plan §4.4 (pre-registered):** Phase-B data = the fresh 64-chain MAMS run
  (`messy_tests/dpie/mams/arrays.npz`, 64×1000) instead of the plan's named MCLMC file
  (8×10k). Measured occupancies (z[:,6] > −22.35, the plan-§6 pocket test, verified
  2026-07-07): **MAMS64 = 9.57%** (6127 pocket draws; per-chain occupancy range
  [0.001, 0.951] — chains are strongly pocket-SEGREGATED, so this finite-sample estimate
  from correlated draws is itself uncertain, plausibly ~2× — a NEW finding, recorded);
  MCLMC = 14.57% (11660 draws; matches the plan-§5.4 table's 14.6%, over-weighted vs the
  Laplace proxy 5.4% and old-MAMS8 4.6%). MAMS64 is preferred because its stationary law is
  MH-exact and it is the least over-weighted MH-correct set available; forward-KL trains
  toward its data's weights. Lost vs MCLMC: ~6.1k vs ~11.7k pocket draws (less pocket-shape
  information — immaterial to the median-point coverage gate).
  **Cause hypothesis:** reverse-KL from the SVI start assigns the separate ~5% pocket mode
  (14σ from the SVI solution) probability ≈ 0 (mode-dropping — the mechanism the demo could
  not test, having no second mode); forward-KL on MH-exact samples restores it.
  **Pre-registered predictions:** (F1) A-only FAILS the pocket gate: log q(zP) − log q(zM)
  < −8 nats, plausibly ≪ −8 (SVI itself was 14σ ⇒ ratio ~ −100s; the flow starts there and
  reverse-KL has no gradient signal to build density at an unvisited mode) — its failure is
  a RESULT confirming §4.4, not a bug; (F2) A+B PASSES: ratio ≥ −8. Expected magnitude for a
  well-covering flow: **≈ +5.4 nats** — the gate statistic is a DENSITY ratio at the two
  medians, and for perfect q it equals lp(zP) − lp(zM) = −291319.81 − (−291325.24) = +5.43
  (normalization cancels; both values pinned by carousel_model.verify()). CORRECTION to
  plan §5.2, which says "true value ≈ −3": that is the log MASS ratio log(0.05/0.95) ≈ −3
  — a different quantity. The −8 coverage floor is unaffected (mode-dropped flows sit at
  ≈ −100s; the gate's separation is huge either way). Phase-B data contains ~6.1k pocket
  draws. (F3) both flows pass the ELBO gate: direct 5×128-draw
  neg-ELBO ≤ SVI final 291453.1 (family nesting; identity-init starts AT the SVI loss);
  (F4) A+B passes the pullback-scale gate on MAIN-BASIN draws (sd ∈ [0.5,2], |mean| ≤ 1;
  band derivation as in the demo checkpoint); pocket-draw pullback reported separately as a
  diagnostic (A-only SHOULD place pocket draws far out — that is F1 seen from the sample
  side). **Falsifiers + pre-committed responses:** F1 wrong (A-only covers the pocket) ⇒
  surprising positive about reverse-KL here; investigate how (lp evaluations at 128 draws/
  step CAN see the pocket if draws land there) before any claim; F2 wrong (A+B fails) ⇒ ONE
  retry with subsampling/early-stopping (the v4-flagged overfit levers), then if still
  failing: pre-registered NEGATIVE finding "Phase B as designed cannot restore the pocket"
  ⇒ escalate to the human before any benchmark (the plan's §5.4 pocket-occupancy win
  condition would be unreachable); F3 wrong for the A-ONLY flow ⇒ Phase-A optimization
  issue ⇒ apply the pre-registered lr fallback (ONE retrain at 1e-3, no further iteration);
  F3 wrong ONLY for A+B (A-only passes) ⇒ the Phase-B shift moved the flow off the
  reverse-KL optimum — evaluate as Phase-B evidence jointly with F2/F4, NO lr iteration;
  F4 wrong while F2 passes ⇒ scale layer insufficient at 33 dims ⇒
  diagnose (s travel ~ measured mismatch) before benchmark. **Blind spots:** (a) gates
  evaluate the FLOW, not sampling — no MAMS run here; mixing/occupancy claims wait for the
  benchmark; (b) zP/zM are 2 points — the pocket gate tests coverage at the medians, not
  pocket shape; (c) pullback-scale uses MAMS64 draws — fresh but finite (ESS unknown at
  gate time; band has 3.5× slack). **Cost:** ≤ 60 GPU-min, one allocation (Phase A ≈ 3000
  steps × 128 two-band 300² lstsq renders; Phase B render-free; gate evals ≈ 15 renders'
  worth). Per the user-approved plan (2026-07-08, "Go ahead with this updated plan"), GATE F
  is grader-approved; the §5.4 benchmark checkpoint goes to the human.
  **Status: approved 2026-07-08 (rigor-grader, second round; occupancy numbers
  independently reproduced by grader); GATE F authorized at ≤60 GPU-min; §5.4 benchmark
  checkpoint reserved for human approval (win-condition band re-derivation required:
  occupancy estimates span 4.6–9.6%; pocket-occupancy R̂ across chains flagged as the
  sharp benchmark diagnostic).** First attempt OOM'd at Phase-A step 0 (128-draw ELBO
  VJP wants ~30 GiB on a 40GB A100 — basis-gen+VJP dominance, cf. June profiling);
  environmental fix, no design change: the SAME 128-draw estimator evaluated as 4
  gradient-accumulated 32-draw chunks (identical loss/grad in expectation; demo n_chunks=1
  path kept bit-identical). Cost of failed attempt ~0.15 GPU-h. Rerun launched same day.
  RAN 2026-07-08: F1 confirmed (−108.8), F2/F3(A+B)/F4 FAILED — structural diagnosis in the
  Log entry (box ±6 cannot contain the post-scale carousel geometry; |w| to 322, pullbacks
  to 22). **AMENDED v2 (2026-07-08, re-approval required):**
  (Fv2.i) Method-default box: spline_range 6 → **16**, num_bins 8 → **24** (per-bin
  resolution 1.33 ≈ demo's 1.5). ±16 is adopted as a FIXED method-level default (one-shot
  compatible — a constant like NUTS's max_treedepth, not per-problem derived), sized so a
  box must hold the post-scale shape: measured here, main-basin dynamic range ≤ 9.2 and
  pocket offset ≤ 4.2 main-sd units ⇒ ±16 gives ~1.7× margin against the dynamic range
  alone and ~1.2× against the compound worst case (range + pocket offset ≈ 13.4 in one
  dim; the operative check is the predicted |T⁻¹(z_pocket)| ≲ 10, verified by the F4′/box
  gates); per-problem violations are caught by the box-coverage/pullback gates, which is
  the gates' job. Honesty note: the default is chosen in light of this problem's
  measurements — its status as a universal default is a hypothesis future systems test,
  not a validated fact. PRE-COMMITTED demo re-validation: before the §5.4 benchmark runs,
  the demo validation is re-run at ±16/24 (one flow retrain + arm B′, ~15–20 GPU-min with
  caches) so the benchmark config is validated on BOTH systems; a demo failure at ±16/24
  blocks the benchmark. PRE-COMMITTED premise-level reading (grader): this is the SECOND
  post-failure widening (demo 6→35, now 6→16 as default) — if ±16/24 fails the demo
  re-validation or a future system, the honest conclusion is NOT a third widening but
  that the one-shot fixed-box premise is FALSIFIED, and the plan-§6 data-derived-range
  path (v3, which passed) becomes the method.
  (Fv2.ii) The pre-committed F2 retry (subsampling/early-stopping) is SKIPPED: premise
  (overfit) falsified — Phase B trained monotonically; the failure is support, not fit.
  This amendment replaces that retry as the single pre-committed response.
  (Fv2.iii) Predictions, re-registered: (F1′) A-only at ±16 STILL fails the pocket gate
  (mode-dropping is about reverse-KL's objective, not the box — if it now PASSES, the box
  was constraining reverse-KL's view of the pocket, an informative surprise); (F2′) A+B at
  ±16 PASSES (≥ −8; expected ≈ +5.4): pocket draws now pull back in-box (predicted
  |T⁻¹(z_pocket)| ≲ 10 post-scale), so fkl has spline capacity where its data lives;
  (F3′) A-only ELBO ≤ SVI − 20 nats. Derivation (cross-architecture nesting): RQ-spline
  widths/heights/slopes are trainable, so a ±16/24 layer can allocate knots to reproduce
  any ±6/8 configuration inside ±6 and identity on [6,16] — the measured SVI−26 optimum is
  (ε-approximately) representable in the new family; the 6-nat slack covers optimization
  shortfall in the larger parameterization. PRE-COMMITTED intermediate reading: a result
  in (SVI−20, SVI] is an OPTIMIZATION-QUALITY finding (bigger landscape, same representable
  optimum), not a method failure — no post-hoc reclassification either way. A+B ELBO ≤ SVI
  (the ±6 version's F3 failure was the support problem);
  (F4′) A+B passes the main-basin pullback-scale gate; A-only recorded (may still fail on
  |mean| — reverse-KL centering under ridge curvature is exactly what the benchmark
  probes). Falsifier: if F2′ fails WITH pocket pullbacks in-box, Phase B has a genuine
  fit problem ⇒ THEN the subsampling/early-stopping retry applies (ONE pass), else the
  pre-registered negative finding + human escalation stands. The original GATE F falsifier
  responses CARRY OVER mutatis mutandis: F3′ fails for A-only → lr fallback (ONE retrain
  at 1e-3, no iteration); F3′ fails ONLY for A+B → Phase-B evidence evaluated jointly with
  F2′/F4′, no lr iteration; F4′ fails while F2′ passes → diagnose the scale layer
  (s travel vs measured mismatch) before any benchmark. **Cost: ≤ 45 GPU-min**
  (Phase A retrain at 24 bins + Phase B + gates). **Status: Fv2 approved 2026-07-08
  (rigor-grader, second round; box-premise falsification pre-committed); run authorized at
  ≤45 GPU-min; §5.4 benchmark blocked on demo-±16/24 re-validation + human approval.**

- **Run: demo 4-arm flow-preconditioning validation** (plan §5.2/§5.3 dry run before any
  carousel work; script `experiments/flow_precond/demo_validation.py`, branch
  `flow-precond-mams`). Demo lens (22 params, easiest system), identical 8-chain budgets:
  A = vanilla MAMS (300+300, in-family reference; demo posterior itself validated vs HMC in
  `laps_validation`); B = flow-MAMS, whitened-spline flow Phase-A (128-draw reverse-KL,
  3000 steps); C = flow-MAMS, same flow + Phase-B forward-KL (1000 steps on the GATE I
  vanilla-MAMS draws — exercises the fwd-KL path; demo has no secondary mode for it to add);
  D = plain NeuTra, NumPyro-faithful (3-block IAF, 1-sample ELBO Adam 3e-3 10k steps, NUTS
  target 0.8, 1000 warmup + 300 draws). All diagnostics on decoded z = T(u).
  **Cause hypothesis:** the pulled-back targets are the same posterior re-parameterized;
  with MH-corrected kernels every arm is exactly unbiased regardless of flow quality, so on
  an easy unimodal posterior all arms must agree with A. This is a CORRECTNESS gate for the
  implementation (Jacobian, decoding, qz_u plumbing), not an efficiency benchmark.
  **Predictions (direction + magnitude):** (1) flow gate: spline Phase-A neg-ELBO tail ≤ SVI
  final loss (−70.98; family nesting — flow ⊇ affine — makes ≤ derivable; expect ~0–10 nats
  below, demo ≈ Gaussian); (2) agreement gates pass on ALL 22 params for B, C, D vs A: |Δmean|
  ≤ 4·SE_MC and sd-ratio within 4·SE_r (both SEs derived from measured ESS; with (3+1)×22
  width tests + 3×22 mean tests ≈ 154 4σ tests, expected false failures ≈ 0.01 ⇒ ANY failure
  is a finding); (3) C ≈ B via a DIRECT B-vs-C sd-ratio gate (within 4·SE_r; implemented in
  the script as `bc_width_gate`); (4) health booleans: max split-R̂ < 1.05 every arm
  (`rhat_health_pass`); NUTS divergent transitions = 0 exactly (`divergence_gate_pass`;
  any divergence fails and is a recorded finding).
  **Falsifier:** any arm failing (2) on ≥1 param ⇒ a bug in that arm's path (fldj sign/scale,
  decode, latent init/mass, NUTS plumbing) — the demo is easy by construction, so "hard
  posterior" is NOT an admissible explanation; STOP and diagnose before any carousel run.
  **Metric + derived thresholds:** as in (1)–(4); all thresholds derived (family nesting;
  MC error at measured ESS), none tuned. **Blind spots:** (a) agreement at ESS ~O(10²) is
  blind to biases ≲0.2σ; (b) a bug common to BOTH MAMS arms' shared path (wrapper/decode)
  could cancel in B vs A only if it vanishes at identity — arm D uses a different kernel AND
  different flow family, mitigating common-mode failure; (c) u-space split-R̂/ESS computed and
  reported for B/C/D, not gated; (d) agreement SEs assume near-Gaussian marginals and use
  bulk-ESS as the mean/variance-ESS proxy — adequate at 4σ on the demo; re-derive before
  reusing the gate on a non-Gaussian target. **Cost:** ≤ 45 GPU-min one allocation (flows
  ~10 min + 3 sampler runs + NUTS).
  **Status:** graded REVISE 2026-07-08 (rigor-grader: arm-C `num_steps=0` crash in
  `train_flow`; prediction (3) B-vs-C gate unimplemented; prediction (4) thresholds not
  evaluated as booleans; blind-spot (c) wording — design sound, no threshold/budget changes)
  → all four fixed (+ script now imports the tested library losses from flows.py instead of
  inline re-implementations) → approved 2026-07-08 (rigor-grader re-inspection); ran same day,
  TIMEOUT — see Log entry (3 findings, all diagnosed). **AMENDED v2 (2026-07-08, post-diagnosis;
  material design changes, re-approval required):**
  (i) spline out-of-range NaN-gradient bug FIXED in flows.py (double-where; out-of-range =
  bitwise identity, finite grads; suite 15/15 incl. real-reproducer fkl step) ⇒ Phase B now
  expected to train: NEW prediction (5) Phase B loss finite and decreasing (final < initial
  19.94); arm C proceeds. Spline range stays ±6 (identity tails absorb the 7% out-of-range
  coords; hypothesis-driven — widen only if the flow gate or agreement gates fail).
  (ii) NEW arm E = whitened-IAF NeuTra (identical NumPyro recipe with the SVI whitening
  composed under the IAF; documented deviation from vanilla). E is subject to the same
  agreement + health gates as B/C. Gate accounting unchanged at ≈154 tests (E replaces D
  in the 3 gated arms).
  (iii) Arm D (faithful unwhitened NeuTra) re-scoped: EXPECTED to diverge in training within
  ~10 steps and be skipped for sampling — recorded as a finding with verified 3-leg mechanism,
  ALL ARCHIVED: leg A (demo ∇lp non-finite at ordinary z: lp finite 16/16 but grad-finite
  rows 5/16 at scale 1, 1/16 at 3, 0/16 at 5 and 8) + leg B (identical faithful recipe on an
  equally-sharp finite-grad 22-dim Gaussian: 400 steps × 2 seeds, zero non-finite losses,
  params finite, 3.2e5→550 / 2.5e5→823) re-derivable via
  `experiments/flow_precond/armD_mechanism.py` → `armD_mechanism_out/{summary.json,arrays.npz}`;
  leg C (IAF numerically faithful to numpyro source) pinned by
  `test_flows.py::test_iaf_init_reproduces_numpyro_scheme` + Jacobian tests. NEW prediction
  (6): if D does NOT diverge, the mechanism analysis is wrong ⇒ investigate before reporting
  either way; if D completes, its agreement stats are `diagnostic_only` — outside the
  154-test accounting. Non-finite ∇lp at prior-scale z is itself a finding (possible
  connection to the cold-init LAPS pathology — flagged, not claimed).
  (iv) Hardening: divergence-aware training (gate on updated params), NaN-cache
  invalidation, per-arm isolation with arm_status, fail-fast skip of arms with non-finite
  flows (the timeout mechanism — MAMS NaN-guard ε-collapse — cannot recur), line-buffered
  stdout. **Cost: ≤30 GPU-min** (MAP/SVI + spline-A cached; retrain B-phase + 2 IAFs + 4
  sampler arms). Status: v2 approved 2026-07-08 (rigor-grader, third round); v2 RAN — arms
  A/E green, B/C failed agreement+health; diagnosed as range-clipped flow + under-adaptation
  (see v2 Log entry). **AMENDED v3 (2026-07-08, post-v2-diagnosis; re-approval required):**
  (v3.i) spline_range 6→35 (measured max|T⁻¹(z)| = 31.4; plan §6 "widen if not" — derived,
  not tuned), num_bins 8→48 (preserves per-bin resolution); cache keys now encode
  range/bins (stale-cache gap the grader flagged, now binding).
  (v3.ii) Flow-MAMS arms B/C get burnin 1000 (= NUTS warmup budget; v2 showed 300 from
  identity mass cannot learn residual flow-imperfection scales — chains spread 1.1–3.6σ vs
  target 7.2σ). Arm A stays 300 (it converged; R̂ 1.015).
  (v3.iii) NEW pullback-scale gate: per-dim sd(T⁻¹(arm-A draws)) ∈ [0.5, 2] and |mean| ≤ 1.
  Derivation basis: an order-unity fitness band for what identity-init mass adaptation can
  absorb within the burnin budget (mass adaptation corrects O(1) scale mismatches cheaply);
  the measured failure was sd 7.2, 3.5× beyond the band edge, so the verdict is
  threshold-insensitive. Pre-registered prediction (7): the Phase-A+B flow PASSES it (with
  range 35 the training data is finally in-range, so forward-KL can fix scales);
  prediction (8): Phase-A-only FAILS it (mode-seeking reverse KL — the demo analog of the
  carousel §4.4 pocket claim; its failure is a result, not a bug). Candidate GATE F
  addition for the carousel stage. CROSS-CHECK: if (7) FAILS while B/C health passes, a
  green agreement result is budget-carried (MAMS brute-forcing a bad geometry) — record it
  as requiring diagnosis, not as a flow-MAMS pass.
  (v3.iv) Gate semantics fixed: agreement gates are interpretable as bias evidence ONLY
  when the arm's health gate passes (`agreement_interpretable`); an unconverged arm is a
  budget/adaptation finding, not a bias verdict (v2's falsifier conflated these).
  Predictions for the v3 rerun: (2') arm C passes health + agreement (154-accounting
  unchanged, arms B/C/E vs A + B-vs-C); (9) arm B: pullback-scale fails (8) but with 1000
  burnin its health gate may pass — if health passes, agreement must pass (MH exactness);
  if health fails at 1000 burnin, that is PRE-REGISTERED as an approach-level NEGATIVE
  finding for flow-MAMS with identity-init mass on this system (same adaptation budget
  under which NUTS passed, easiest system, range-fixed flow — evidence against its
  viability relative to NeuTra-NUTS here), and NO further burnin escalation happens on the
  demo without an explicit human decision. E expected to repeat its v2 pass.
  **Cost: ≤35 GPU-min** (flows retrain at new arch; MAP/SVI cached; B/C burnin ×3.3).
  Status: v3 approved 2026-07-08 (rigor-grader, fourth round); v3 RAN — arm B FULL PASS
  (flow-MAMS validated at demo scale, 3.0× vanilla ESS), arm E repeat pass, predictions
  (8) and (2') failed informatively, B-vs-C width gate failed non-interpretably (see v3
  Log entry). **AMENDED v4 (2026-07-08,
  user-approved one-shot architecture; re-approval required):**
  (v4.i) Spline config: range 35/48 bins (data-derived) → **fixed range 6 / 8 bins +
  trainable per-dim scale layer** (`trainable_scale=True`; flows.py `DiagScale`,
  T(u) = loc + L·(exp(s)⊙C(u)), s zeros-init so identity-at-init is preserved). Motivation:
  the user's one-shot goal — no posterior samples may be needed to size the box. Unit-pinned:
  sd-30 direction recovered through a ±6 box (pullback sd 1.03); control without the layer
  stays at 29.5 (suite 20/20). Identifiability caveat recorded: exp(s) is a readout only for
  expansion directions; T as a whole is pinned. Phase-A lr 1e-3→3e-3 (s must travel
  ~log(7)≈2 nats; adam travel ≈ lr×steps; headroom as measured: stable at 5e-3 = 1.7×
  above 3e-3, unstable at 2e-2, onset unmeasured between). Cache keys now encode
  range/bins/trainable_scale/lr.
  (v4.ii) Arms: same A/B/C/D/E structure; B′/C′ use the one-shot flow. The v3 pre-commitment
  "no demo re-runs of arm C without an explicit human decision" is discharged: the user,
  presented with the plan "I'd validate this variant on the demo first" + the full carousel
  step list (session 2026-07-08), replied verbatim **"Okay, that's reassuring. Go ahead
  with this updated plan."** — which includes C′.
  (v4.iii) Pre-registered predictions: (10) pullback-scale gate PASSES for the A-only
  one-shot flow (the flows.py unit test is the same mechanism at harder mismatch);
  (11) arm B′ passes health + agreement (as v3 arm B did with the data-derived range).
  FALSIFIER + pre-registered ablation ladder: if B′ fails where B passed, the failure
  attributes to the ONE-SHOT CONFIG AS A PACKAGE (four factors moved at once: range 35→6,
  bins 48→8, +DiagScale, lr 1e-3→3e-3 — the lr hits the couplings too); diagnose in this
  order, cheapest discriminator first, ONE pass each: (i) retrain Phase A at lr 1e-3
  (isolates the moved knob), (ii) compare against the retained v3 r35b48 caches (isolates
  range/bins vs scale layer). No other iteration before a diagnosis is recorded. A B′ PASS
  is unconfounded (all four factors simultaneously acceptable).
  lr fallback (pre-registered): if the FLOW gate fails at lr 3e-3, retrain ONCE at lr 1e-3
  (or 1e-3 × 10000 steps) before drawing any architecture conclusion; no further lr
  iteration. Headroom as measured: stable at 5e-3 (1.7× above 3e-3), unstable at 2e-2,
  onset unmeasured in (5e-3, 2e-2).
  (12) arm C′ (Phase B, full-batch fkl on 2400 correlated draws):
  health outcome RECORDED either way; a repeat health-fail ⇒ Phase-B-as-implemented
  overfits regardless of architecture, and the carousel Phase B design must add
  subsampling/early-stopping BEFORE GATE F rather than after a failure there. D/E
  unchanged (D expected-diverge; E cached flow, NUTS rerun).
  **Cost: ≤35 GPU-min.** **Status: v4 approved 2026-07-08 (rigor-grader, sixth round;
  fixed per grader items 1-2, launch pre-authorized on their application); rerun
  authorized at ≤35 GPU-min.**

- **Run: GATE I — identity-flow wrapper ≡ vanilla MAMS on the demo lens** (flow-preconditioning
  plan `docs/plans/flow-preconditioned-mams.md` §5.1; script
  `experiments/flow_precond/gate_i_identity.py`, branch `flow-precond-mams`). Demo lens
  (22 params, validated vs HMC in `laps_validation/handoff`), MAP→SVI→two MAMS runs
  (8 chains, 300 burnin + 300 results, seed 0): vanilla `MAMS_JIT(model_seq, qz)` vs the same
  through `TransformedProbModel(pm, tfb.Identity())` + `FlowModelSeq`.
  **Cause hypothesis:** the wrapper only adds `lp + fldj` with `fldj ≡ 0.0` and an identity
  `forward()`, so the pulled-back target is the *same computation*; MAMS consumes only
  `prob_model.log_prob` (mams.py:58) and the identical `qz` (:73/:79/:91).
  **Prediction (direction + magnitude):** result samples **bit-identical**
  (`np.array_equal`, max|Δ| = 0.0 exactly; IEEE x+0.0 = x).
  **Falsifier:** ANY nonzero element fails the gate. Diagnosis is two-tier (the two graphs
  compile as separate jaxprs): a max|Δ| at the ULP floor with unstructured scatter points at
  XLA FP-scheduling of the extra (zero) add — diagnose compiler determinism (compare lowered
  HLOs) before touching the wrapper; a structured or large Δ (dtype cast, broadcasting,
  tracing difference) ⇒ fix the wrapper. In neither case relax to a tolerance or to
  "statistically indistinguishable" without diagnosing why bit-identity failed.
  **Metric + derived threshold:** `np.array_equal` on the (8, 300, 22) result-sample arrays
  (dim = demo `num_free_params` = 22, printed at runtime) — exact by construction, hence
  derived, not tuned. **Blind spot:** exercises only the `log_prob` path at identity; says
  nothing about nonzero-Jacobian correctness (to be covered by autodiff-slogdet unit tests in
  `inference/tests/test_flows.py` — **pending**, being written; a hard blocker for any
  nonzero-flow run) or about latent-`qz` plumbing (§5.3, gated later). A green GATE I
  validates only the no-op-at-identity property, nothing more. **Cost:** ~5–10 GPU-min
  (MAP+SVI+2 short MAMS). **Status:** graded REVISE 2026-07-08 (rigor-grader; 4 doc/diagnostic
  fixes, no design change) → fixes applied → approved 2026-07-08 (rigor-grader re-inspection)
  → **RAN 2026-07-08: PASS, bit-identical (see Log entry).** Human approves carousel-scale
  runs.

- **Run: Test 4 — conv_precision=float64 dial-scan arm** (rerun the EPL_Lf center_x dial-scan at
  ss∈{1,5} with conv_precision float32 vs float64, niter=50, other 30 params frozen at the max-ξ draw).
  **Cause hypothesis:** the proliferating likelihood teeth in the perturber's position are *physical*
  near-singular caustic structure, NOT float32-convolution rounding roughness. **Prediction:** at matched
  ss, float64 leaves the comb ~unchanged (prominence ratio f64/f32 ∈ [0.5,2]). **Falsifier:** float64
  *collapses* the teeth (ratio < 0.3) ⇒ it was float32 numerics. **Metric+threshold:** gradnorm comb
  p99.5/median and max, f32 vs f64 at the same ss; physical if f64/f32 ∈ [0.5,2], numerics if < 0.3
  (deterministic float64-vs-float32 render, no sampling noise). **Blind spot:** float64 conv still uses
  the same pixel grid — separates float32 *rounding* from real structure, not physical-caustic from
  grid-discretization of a true singularity. **Cost:** ~10–15 GPU-min. **Status:** RAN 2026-07-06
  (human-approved).
- **Run: Test 2 — spike localization** (at the max-ξ draw, per-pixel sensitivity d(model)/d(center_x) via
  jvp, model/data/residual, both bands). **Cause hypothesis:** the stiffness is localized on the small
  elliptical perturber's caustic/critical region clipping a bright image feature. **Prediction:** |∂model/∂
  center_x| concentrates on a compact region near the perturber / a bright arc, not diffuse.
  **Falsifier:** sensitivity spread broadly and unrelated to the perturber. **Metric:** sensitivity map +
  fraction of |sensitivity| in the top-1% pixels. **Blind spot:** localizes where the *model* moves, a
  proxy for the caustic. **Cost:** a few GPU-min. **Status:** RAN 2026-07-06 (human-approved).

Before a consequential run, the producer logs a checkpoint here and stops for grading
(structural rule 3). Clears once launched — then log observed vs. predicted magnitude.

- **Run: Test 1 — 2-D `(theta_E, gamma)` likelihood + gradient-norm dial-scan of EPL_Lf around a
  spike lattice cell, at `supersample ∈ {1,3,5}` × `niter ∈ {50,200}`** (comb-collapse
  discriminator; other 30 params frozen at the global-max-ξ draw; reuse
  `experiments/why_hard_to_sample/t12_flank_crossing.py` where possible).
  **Cause hypothesis:** the catastrophic stiffness at the EPL_Lf `(theta_E, γ)` lattice is an
  *aliasing artifact* — the perturber's near-singular radial factor `(b/R)^{γ−2}` / critical
  curve, rendered on a `supersample=1` (0.2″) grid, produces a comb as the feature crosses pixel
  centres when theta_E/γ vary. (Numerics/render term, not physics.)
  **Prediction (direction + magnitude):** the gradient-norm comb teeth **collapse** as
  supersample↑ — peak-to-trough(PtT) drops ≥1 order of magnitude ss1→ss3 and ≥2 orders ss1→ss5
  (calibrated from the sys60 analog, T12: ×1400 over ss2→ss4). `niter` 50→200 changes PtT by
  <2× (theta_E-in-band argument).
  **Falsifier:** teeth persist (PtT(ss5)/PtT(ss1) > 0.5) under supersampling — then it is NOT
  pixel aliasing (⇒ physical caustic if the teeth track the critical curve on bright pixels, or
  the series if instead PtT collapses under `niter`↑, contradicting the source read).
  **Metric + derived threshold:** PtT ratio of `‖∇_z logL‖` along a fine γ cut through ≥3 comb
  teeth. Aliasing confirmed if `PtT(ss5)/PtT(ss1) ≤ 0.1`; series excluded if
  `PtT(niter200)/PtT(niter50) ∈ [0.5, 2]`. These are ratios of a **deterministic** float64
  quantity (render reproducibility ~1e-6 rel), so ≥10× collapse is far outside numerical noise;
  the ≥10× bar is the conservative floor of the sys60 comb collapse.
  **Blind spot:** a 1-D γ-cut PtT is blind to teeth that move in the *other* coordinate (mitigate
  with the 2-D map) and to structure finer than the scan step (use step ≤0.2× the per-pixel
  equivalent in theta_E/γ); it cannot say whether a *residual* collapsed comb still matters for
  sampling (that is Test 3).
  **Expected plot:** at ss1, `‖∇logL‖` vs γ = sharp periodic teeth at the measured 0.189 spacing;
  at ss3/ss5 they flatten into a smooth envelope. Falsifier: teeth survive at ss5.
  **Cost:** moderate GPU — fine 1-D γ + 1-D theta_E scans (~200 pts) + a 60×60 2-D map, each ×6
  configs; ss5 renders 1500². ~10–20 GPU-min (A100). **Status: RAN 2026-07-06 (forward-only logp scan, FD gradient).**
  **OBSERVED vs PREDICTED:** predicted comb *collapse* ≥10× under supersampling (aliasing). Observed the OPPOSITE — teeth *proliferate* with supersampling (ss1 mostly smooth + a few sharp teeth; ss3/ss5 a dense forest, peak heights ~constant ~1e4–6e4); prominence ss5/ss1=0.48; **niter 50→200 ratio = 1.000**. ⇒ **pixel-aliasing FALSIFIED and series FALSIFIED.** Hypothesis failed on the aliasing prediction (magnitude+direction), correct that niter is irrelevant. Artifacts: batchA_diag/P9_dialscan_comb.png, test1_dialscan.npz. See 2026-07-06 log entry.

- **Run: Test 3 — short causal MCLMC rerun at `supersample=3`** (8 chains, ~2k burn-in + 2k
  results, seed 42; watch the ξ tail and tuned `eps`). *Pre-registration only — not yet
  authorized to launch.*
  **Cause hypothesis:** the few dozen aliasing teeth generate the catastrophic-ξ events that
  dominate the tuner's energy-error variance and suppress `eps` globally (C-21); removing the
  aliasing (ss=3) removes the catastrophic tail and lets the tuner recover `eps`.
  **Prediction (direction + magnitude):** `max(ξ)` drops below 1e3 and `frac(ξ>10)` toward clone
  level; tuned `eps` recovers from 0.1255 by ~one order into ~1.0–1.5 (from `eps ∝ Var^{1/4..1/6}`
  and the measured 3.7e4 variance domination ⇒ ×8–14); min-ESS rises.
  **Falsifier:** ss=3 leaves `max(ξ) ≫ 1e3` and `eps < 0.3` ⇒ aliasing is not the carrier
  (physical caustic / other). Partial miss: tail shrinks but `eps` barely moves ⇒ the tail was
  not the `eps` driver (would contradict the C-21 variance-domination cut).
  **Metric + derived threshold:** tuned `eps` (results-phase `step_size`) and `max(ξ)` from the
  rerun `diagnostics.npz`. Recovered if `eps ≥ 0.8` (conservative vs the clone-era healthy
  eps~1.0–1.2, mirrors the Route-A eps-recovery causal test) **and** `max(ξ) < 1e3`.
  **Blind spot:** `eps` recovery confirms the tuner-tax link but is blind to whether higher
  render fidelity *changes the posterior* (must check marginal identity — the sys60
  "ss2-on-accurate-data is fast but biased" caveat); and ss=3 may be insufficient (need ss=5),
  so a null at ss=3 does not prove aliasing absent.
  **Expected plot:** ξ trace with catastrophic spikes gone (`max<1e3`); `eps` trajectory settling
  ≥0.8; per-param ESS bars lifting. **Cost:** one short GPU run, ~3–4× the ss=1 wall (~10–15 min
  on a node). **Status:** awaiting approval (pre-registration only).

---

## Log (newest first)

- **2026-07-11 (carousel GATE PT-0 RAN — pre-registered transport falsifiers FIRED
  (W-2 = 0 pocket round trips), but the mechanism decomposition is COMPLETE and
  everything points at three measured, fixable cadence/spacing knobs; NO entropic
  starvation on either path; NO cross-basin swap suppression; the dPIE carousel is
  PT-FRIENDLY at the swap level; PROPOSED (UNCERTIFIED)):** ran 2026-07-11 ~23:10–02:30
  on job 55789329 (1 node, 4×A100), script commit e051c02 (+ in-gate op-1..op-7
  amendments, final code 6e32eea/+override), outputs `carousel_gate_pt0_out/`.
  **Arm 0 (known-answer + calibration):** 0a PASS — fresh harness cold occ₊ 0.6885 ±
  0.0112 vs truth 0.70; the June-28 "implementation bug" hypothesis is unreproduced by
  this code path. 0b c_rw = 9.371 (out of band, pre-registered routing fired; diagnosed
  as per-walker vs per-ladder bookkeeping, c_rw ≈ R).
  **Arm A (tempered-mass profiles, the mechanism instrument):** POWER path Δ(β) is FLAT
  — [−0.42, +0.38] nats across β ∈ [0.01, 1], |Δ| ≤ 1σ everywhere (plot inspected:
  flat band far above the −2.0 viability floor). The Gaussian-model prediction (−8.4
  nats at β = 0.01) is WRONG in magnitude and direction — hypothesis failure as
  routed by amendment (i)/(vi); the June-28 minimal-carousel failure does NOT
  generalize: **the dPIE pocket keeps its relative mass at ALL temperatures.**
  LIKELIHOOD path Δ(β) = +1.45 to +2.92 nats (pocket mildly ENHANCED hot; m_prior =
  0.9944 ± 0.0012 — the prior sits 99.4% pocket-side of the indicator). W-1's predicted
  configuration (power dives, lik flat) is refuted in DIRECTION; measured profiles
  supersede. **Hot-end consistency flag FIRED (open finding, both paths):** direct
  unconfined occupancy at β = 0.01 is 0.379 ± 0.046 (power) / 0.967 ± 0.006 (lik) vs
  0.1-anchor predictions 0.078 / 0.673 — on the power path, direct and TI measurements
  reconcile if the TRUE cold pocket weight is ~0.3–0.4 rather than ~0.1 (alternative:
  1500-step hot-end equilibration transient). MAMS64's 9.6% is now DOUBLY suspect (it
  was already human-flagged unconverged). Adjudication needs a converged cold chain —
  exactly what PT-0b should deliver. Estimator health: per-config u-identity checks all
  passed (rel ≤ 2e-11); leak fractions large at hot β as expected (classification
  handles them); one config (power, M-init, β = 0.046) had EEVPD 0.11 — that point's E
  carries extra unmodeled error (se there is wide, conclusion unchanged).
  **Arm B pilot (fused runner; truncated by allocation expiry per op-7 — B1 901/2000
  rounds, B2 1201/2000, B3 NOT RUN (auto-launch canceled: it would have reproduced B2's
  diagnosed pathology underpowered ⇒ W-3 bracketing UNSCOREABLE this gate)):**
  B1 (power, balanced): all 11 adjacent-pair acceptances healthy (0.17–0.38) and
  **cross-basin ≈ same-basin acceptance (0.18–0.34 vs 0.17–0.33) — the June-28
  cross-basin suppression signature is ABSENT on the dPIE target.** EEVPD in band at
  every rung (~1.1e-4). Worm plots (inspected): basin content churns at ALL rungs —
  the COLD rung flips basin identity repeatedly within 900 rounds in most systems
  (multi-switch cold-chain basin mixing, which vanilla MCLMC cannot do; C-10) — but
  walker-LABEL transport is ~25× slower than free-random-walk diffusion (mean
  displacement 2–4 rungs in 900 rounds vs ~15 expected; 5 down-traverses, 0 completed
  round trips) ⇒ **W-2 = 0, F-2 zone; W-5 mismatch >10× (F-4 zone).** Cause measured,
  not conjectured: SWAP-BACK suppression — K = 10 kernel steps/round vs measured
  within-rung IAT(u) of 11–202 steps (Arm A series), so swapped-in configs don't
  decorrelate between sweeps and swaps reverse; ALSO the c_rw = 9.371 calibration came
  from the control's K ≫ IAT regime and cannot transfer (transport model revised: flux
  needs a decorrelation factor ~K/(K+IAT)). B2 (likelihood, balanced): ladder
  DISCONNECTED at both ends — hottest pair acceptance 0.000, coldest three 0.005–0.048;
  interior rungs pinned ~0.9+ pocket-side; 0 down-traverses. Mechanism: (a) geometric
  spacing mismatched to the measured swap-cost density (total ∫sd(u)dβ = 23.0 nats over
  11 pairs, concentrated at the ends), (b) the hot rung NEVER equilibrated off its
  prior-draw inits — hot-rung EEVPD ~2.6e-8 shows the ss_max = 1.0 step cap BINDS on
  near-prior targets whose z-scale is ≫ 1 (same cap-binding seen on the control).
  **Measured design outputs for PT-0b (the point of the gate):** equal-swap-cost
  ladders computed from Arm A sd(u): 21 rungs (power, total 19.1 nats) / 24 rungs (lik)
  at 1 nat/pair (`ladder_design_{power,lik}.json`); measured IAT(u) per β for K
  calibration; Arm A leak fractions show the KERNEL already crosses basins at β as
  cold as 0.6 (17–37% class leakage per 1500 steps) ⇒ a SHORT power-path ladder
  β ∈ [~0.36, 1] (~6 rungs, ~1 nat/pair) may suffice — discovery does not need β ≪ 0.1
  on this posterior; raise/re-derive the ss cap at hot rungs. Wall data: fused B round
  ≈ 5.5 s (96-wide, K=10); Arm A ≈ 9,500–9,700 s/path; total gate ≈ 14 GPU·h.
  **Scope + verdicts (PROPOSED, UNCERTIFIED):** W-2 FAIL (0 < 7; F-2 routing —
  diagnostic complete in-gate: replica-trace localization + IAT measurement); W-5 FAIL
  (>10×; F-4 routing — flux-model revision identified); W-3 unscoreable (B3 absent);
  W-4 PASS for B1 / FAIL for B2 (end rungs); W-1's predicted path separation refuted in
  direction. NOT claimed: any working sampler yet, any pocket-weight value, any
  generalization beyond this posterior. Falsifier status honestly: the PILOT failed
  as configured; the MECHANISM instrument succeeded and localized every failure to a
  measured knob. Blind spots: hot-end direct occupancies may be transients; per-basin
  TI conditioned on the z[6] halfspace (third modes invisible); B arms truncated
  (901/1201 rounds) per op-7. **Routing (per pre-commitment):** F-2/F-4 ⇒ no scale-up;
  next = GATE PT-0b continuation checkpoint (short measured ladder, K from IAT, ss-cap
  fix, balanced + all-main arms, B3 bracketing restored) — new checkpoint + grader
  before any run.
  **RESULT-GRADER AMENDMENTS (rd-1, 2026-07-11; applied — corrections to THIS entry):**
  (1) W-4 RE-SCORED: **FAIL for B1 AND B2** — B1 hot rungs 0–6 median EEVPD
  4.4e-6–9.3e-5, BELOW the [1e-4, 2e-3] band (ss-cap binding is present on B1 too, not
  only B2/control); cold-rung indicator split-R̂ = 1.68 (B1) / 1.60 (B2) vs ≤ 1.05
  (expected given 0 round trips). The earlier "EEVPD in band at every rung (~1.1e-4)"
  sentence is WITHDRAWN — it generalized a single round-900 print line (operating-card
  rule-3 violation caught by the grader: the full series swings 5.6e-8–2.3e-2).
  (2) Estimator-health CORRECTION: the worst Arm-A config EEVPD is **5.01** at (power,
  P-init, β = 0.01) — the hottest rung, the SAME rung as the 0.379 hot-end occupancy —
  with 0.11 at (power, M-init, β = 0.046) second; the hot-end open finding therefore
  carries under-controlled dynamics error IN ADDITION to the named transient
  alternative. The flatness conclusion is unchanged because the required rescue is ~8
  nats against ~1-nat noise and the independent likelihood path agrees; stated per
  grader. (3) The 2026-07-10 RECORD ARCHAEOLOGY header below was clobbered by this
  entry's insertion — restored verbatim (THIRD instance of this failure class; see the
  standing lesson: an Edit that consumes a following header in its anchor must restore
  it at the end of the insertion). (4) Number fixes: B1 pair acceptance range
  0.17–0.34 (not 0.38); power-path flatness is |Δ| ≤ 1.4σ (β = 0.36 is 1.40σ, β = 0.60
  is 1.27σ), not ≤ 1σ; fused B rounds ran 7.8 s/round steady-state (not 5.5 — feeds
  PT-0b wall planning); run window extended to ~03:26 (B arms killed at the 10:26 UTC
  time limit). (5) op-8 DEVIATION (recorded): canceling B3's auto-launch was an in-run
  producer judgment — op-7 pre-authorized truncation, NOT cancellation; and B1 was
  relaunched via a second same-job srun (`srun_full_B1b.log`) after its queue-waiter
  deadlocked on a self-matching pgrep. (6) Mechanism caveats added: Arm-A IATs were
  measured on CONFINED chains with per-basin metrics — transfer to the B arms'
  pooled-metric chains is order-of-magnitude only, and the K/(K+IAT) closure of the
  0-round-trip observation is a MODEL, not a measurement; also the A3 advisory stands
  (hot-end consistency flag omits delta_se — conservative direction). Grader verdict
  after amendments: CERTIFY-RECOMMENDED as mechanism diagnosis ONLY (no working
  sampler, no pocket weight, no bracketing, within-basin ESS not certified).**

- **2026-07-10 (RECORD ARCHAEOLOGY — three UNLOGGED GPU PT runs on the minimal carousel,
  2026-06-28, all FAILED to transport; C-18's "PENDING GPU validation" was in fact answered
  negatively for the naive form and never recorded):** found on disk while scoping the new
  PT-MCLMC engagement: `de_mclmc_prototype/carousel_pt.py` (committed in WIP snapshot
  4e9f212) + `pt_carousel.log` (R=10, β_min=0.02, all-secondary init),
  `pt_carousel_balanced.log` (R=10, balanced 0.5 init), `pt_carousel_b005.log` (R=12,
  β_min=0.005, all-secondary; the only one matching the committed script), and
  `pt_carousel.npz` (main checkout, mtime Jun 28 16:12). Common config: NSYS=16
  independent ladders, K=10 MCLMC steps/round, 90 rounds, per-(level,chain) EEVPD step
  adaptation (faithful — per-round prints ≈5e-4), swap math verified-faithful to
  `tempering/parallel_tempering.py` (which PASSED the CPU Gaussian-mixture weight gate,
  0.6986 vs truth 0.70). **Result: the cold rung's occ(global) stayed pinned at its INIT
  value for all 90 rounds in ALL THREE runs (0.000 / 0.500 / 0.000; truth ≈1.0) while
  average adjacent-pair swap acceptance was healthy (0.38–0.72) and the hot rung showed
  genuine kernel crossing (b005 hot occ 0.06–0.31).** The balanced run had not a single
  cold-chain basin flip in ~45 sweeps × 16 systems (≥several hundred cross-basin cold-pair
  attempts) ⇒ CROSS-basin swap acceptance ≲1e-2–1e-3 while same-basin ≈0.5 — **pairwise-
  average swap acceptance is a misleading health metric on this posterior.** No design
  checkpoint exists for these runs (illegitimate under the standing rule; recorded now as
  found artifacts, UNCERTIFIED). Mechanism HYPOTHESIS (untested): the power path p^β
  re-weights basin masses along the ladder (entropic starvation / first-order-like
  bottleneck) — configs of the cold-favored basin are equilibrium-disfavored at
  intermediate β, so downward replica transport is exponentially suppressed even though
  same-basin swaps accept freely (consistent with b005's mid-rung occ ≈ 0 between hot ≈0.2
  and cold-truth ≈1.0). Not yet distinguished from a subtle implementation bug (runs 1–2
  used earlier uncommitted script variants). Consequence: C-18 addendum; the next PT gate
  must FIRST measure the per-basin tempered-mass profile vs β (path diagnosis) before any
  expensive dPIE PT run.
  **HUMAN CONTEXT (2026-07-10, same day):** the user notes "the previous PT example was on
  a more pathological posterior and the implementation was done by a less capable agent."
  Weighting adjusted accordingly: the minimal carousel's 1e-5 secondary is an extreme
  drain test (the dPIE carousel's modes are ~10:1), and an implementation defect is a
  live explanation alongside the entropic-bottleneck hypothesis. The June-28 failure is
  therefore NOT treated as evidence that PT fails on the dPIE target; the GATE PT-0
  design runs the dPIE PT pilot unconditionally (fresh implementation, independently
  audited) with the tempered-mass profile arm as the mechanism PREDICTOR the pilot must
  match, not as a gate that can block the pilot.

- **2026-07-10 (HUMAN DIRECTIVE — MAMS64 is NOT converged; its pocket weight must not be
  treated as ground truth):** the user states, verbatim: "the MAMS64 run is explicitly not
  converged, and the pocket weights should not be trusted." This is consistent with the
  record's own measurements (indicator split-R̂ = 1.719, per-chain occupancy-ESS ≈ 1.9,
  per-chain occupancy range [0.001, 0.951] — see the BENCHMARK baseline block), but it
  supersedes every place the record uses "MAMS64 = 9.57%" as *truth* (e.g. the GATE L
  M2b weight comparison and the W2 band derivation). Standing consequence for all future
  gates on the dPIE carousel: **9.57% is an unconverged point estimate, not an anchor.**
  No win condition may be scored against it as a reference value; unbiasedness must be
  established by convergence + reproducibility of the new sampler itself (multi-seed
  agreement, indicator R̂/ESS, round-trip counts) and, where a weight claim is made, by
  cross-method agreement or a purpose-built long reference run — not by matching MAMS64.
  Same session, the user opened a long-horizon engagement: develop an efficient accurate
  sampler for multimodal lensing posteriors, MCLMC kernel (their workhorse), PT first
  avenue but not locked in; wall-clock is the constraint, 4-GPU interactive node,
  reference budget 10k burn-in + 10k kept.

- **2026-07-09 (carousel GATE L RAN — Laplace jump-mixture FALSIFIED, structurally: the
  main basin admits NO PSD quadratic model at its best reachable point; Laplace evidence
  weight under-counts the pocket 18×; cross-mode MH acceptance = 0/1024 in all four
  pass-eligible cells; multistart enumeration 0/1024 — the M3 falsifier FIRED and the
  annealing family becomes the indicated mainline)** `proposed (UNCERTIFIED)`.
  Artifacts: `carousel_gate_l_out/` (summary, arrays npz, 4 pre-registered PNGs);
  script `carousel_gate_l.py`; 4 GPUs; run wall 4246 s (+1 crashed attempt, below);
  ≈ 6.5 GPU-h incl. smoke + crashed attempt; cumulative program ≈ 27 GPU-h.
  **Execution record:** smoke (GATE_L_SMOKE=1) passed end-to-end per the grader's launch
  condition. Full attempt 1 crashed at M3: 1024×4000 `output_type="all"` exceeds XLA
  compile-time buffers (40.4 GB args; 12.5 GiB single-op OOM on A100-40G) — a size class
  the reduced smoke cannot exercise. Operational amendment (recorded): M3 executed as
  8×128-start chunks, seeds 0–7 (the production manifest's proven scale); M1/M2 rerun
  deterministically, numbers identical across attempts (thr_scale 19.360 vs 19.361 =
  psum reduction order). `carousel_gate_l_m3resume.py` written as contingency, unused.
  **M1 (polish + Hessians).** Pocket: gain +20.97 nats (prediction ≈21 — HIT); z*P
  stays in-pocket; H PSD (λ ∈ [0.64, 1.56e8], cond 2.4e8 — inside the predicted
  1e4–1e9), nat-grad 0.90 (spec was < 0.5: near- but not fully-converged, recorded).
  Main: gain +17.91 (band 5–30 — in-band, low side of the ≈25 point estimate); NOT
  stationary after 3000 whitened Adam steps (nat-grad 1898); H has **3 negative
  eigenvalues**, λ_min = −1.0e5 = **−1.3e-3·λ_max** — five orders above the eig-noise
  floor, NOT the gray zone; 3 axes floored by the pre-registered rule. The saddle
  falsifier formally requires convergence first, so it does not fire as written; the
  structural finding stands regardless: after 3000 steps of polish the best reachable
  main-basin point has genuine negative curvature — **"Laplace at the main mode" is not
  merely inaccurate, it is undefined** (C-20's curved ridge, landed on this config).
  Notable: lp*M = −291307.33 is **29.4 nats ABOVE the production pipeline's own MAP
  best** (−291336.70, manifest 128×4000); lp*P − lp*M = +8.50 (medians gave +5.43).
  **M2a (fidelity).** KL(emp‖Laplace): main **157.3 nats** (predicted 3–15 — ×10
  magnitude miss ⇒ the "Laplace beats SVI on the main basin" hypothesis FAILED; driver:
  12.9σ mean offset — polish walks along the ridge away from the mass centroid — plus 3
  collapsed axes at ratio 0.0013–0.042 and a widest-direction ratio 7.8); pocket **69.2
  nats** (predicted ≤3 — ×20 miss; driver: **5.1σ mean offset between the pocket's peak
  and its mass centroid** — the pocket itself is substantially non-Gaussian, its ratio
  tail reaches 8.1). Benchmark-draw robustness (stuck chain occ=1.000 excluded, 7/8
  kept, n=7683): KL 33.5, ratios [0.81, 6.3] — same story, not a chain-segregation
  artifact. Both KL magnitudes are large prediction misses: per discipline, the
  surrogate-fidelity hypothesis failed even where directions were right.
  **M2b (MC acceptance — the decision measurement).** Pipeline-realistic Laplace
  pocket weight **w̃_P = 0.0052** vs truth 0.0957 (18× under; the ridge-inflated Σ_M
  log-det steals the weight) ⇒ pre-registered threshold_scale = 19.36 (pass line
  ᾱ ≥ 38.7%). Measured: **0 accepted cross-mode proposals out of 1024 in ALL FOUR
  P1/P2 cells** (α ~ e^−big; within-main cells ALSO 0.0000 — the main Laplace cannot
  re-enter its own basin's typical set; within-pocket: P1 2.7%, P2 0.7%). Oracle
  weights: no change (mismatch is shape-dominated, not weight-dominated). P3
  translation: MtoP 0.100%, PtoM 0.035% — **inside the pre-registered volume-ratio band
  [0.01%, 0.5%]** (a clean mechanism HIT: the e^−7.7 pocket/main volume ratio governs
  translation acceptance) but far below the 2%-both-directions branch bar; anti-cells
  ~e^−52/e^−26 (as expected). Item-(i) caveat discharged before reading "unworkable":
  the inflated Σ_M axes are REAL negative/flat curvature (5 orders above noise), and
  the pocket side — PSD, zero floored axes, ratios ≥ 0.78 — fails cross-mode anyway.
  **Verdict: Laplace-mixture jumps are UNWORKABLE on the carousel, structurally.**
  **M3 (enumeration probe).** FINAL routing count: **pocket 0/1024** — falsifier
  FIRED: the enumeration link is broken at a 1024-start budget. Deeper: only **2/1024**
  reach even the MAIN in-basin band (lp ≥ lp*M − 33); 1022 stragglers; best of all
  1024×4000 AdaBelief steps = −291328.0, still 20.6 nats below the polished main peak.
  Diagnostic (best-step counts): exactly 1 particle transited the pocket band
  mid-optimization and left. Retro-reading of the production pipeline: MAP-alone
  essentially never reaches ANY peak here; the pipeline works because SVI + sampler
  downstream rescue a straggler MAP point. lp_recompute_vs_lib_max_abs = 504 nats
  (the "all"-output one-step z/lp offset is huge for stragglers — grader item viii
  validated as load-bearing).
  **Predictions scoreboard.** HITS: both polish gains in-band (pocket dead-on);
  pocket-H PSD + cond in range; P3 in its band; main-KL > pocket-KL direction; M3
  straggler-dominance (sibling-config prior). MISSES (hypothesis-relevant): both KL
  magnitudes (×10, ×20); the pocket's 5σ peak-vs-centroid offset (unanticipated:
  narrow ≠ Gaussian); w̃_P = 0.0052 vs the prior record's 5.4% proxy — **10× apart;
  the two "Laplace weight" computations disagree and the discrepancy is UNRESOLVED**
  (different anchor points/Hessians presumed; one line item for any future Laplace
  claim). M3 pocket-find prediction band (1–50) missed low (0).
  **Decision-matrix routing (pre-committed, "all fail" branch):** report to human with
  options (b) flow-as-proposal / (c) annealing mainline; NO auto-lever. With M3 = 0
  ALSO firing, the annealing family is the indicated mainline (flow-as-proposal's
  mode-seeding depends on the same broken enumeration link — post-hoc rationale,
  flagged as such by the grader). Scope: Link 1 only; two known modes; this
  lens/prior; equilibrium-state acceptance; M3 chunk seeds 0–7 (amendment), not
  literally "seed 0 throughout". **Grader result-pass additions (rd 4,
  CERTIFY-RECOMMENDED):** npz recount — 574/1024 M3 finals end pocket-side of the
  INDICATOR but fail the lp band; median final lp = −328,558 (~37k nats below the
  peaks, below the m3 plot's y-window) — the enumeration failure is
  wandering-without-descent, not wrong-basin capture. KL-driver decomposition
  (pocket): ≈55 of 69 nats from the three heavy directions (4.5–8.1×) + ≈13 from the
  5.1σ offset. Robustness: at the prior-record weight (w̃_P=0.054, pass line 3.7%)
  the cross cells still fail by ~6 orders; oracle-weight cells fail identically — no
  conclusion depends on which weight is right. M3 wall 68 min vs ~26 estimated
  (recompile per chunk; inside the 90-min cap). (Record repair: commit 89cf321's
  insertion accidentally clobbered the following Fv6 entry's header line; restored
  verbatim here.)

- **2026-07-09 (carousel GATE Fv6 RAN — ES machinery worked; the TRAJECTORY has no good
  stopping point; coupled-dynamics assumption FALSIFIED; human escalation)**
  `proposed (UNCERTIFIED)`. ES trace (read before branch selection, per the
  pre-registered instruction): step 0 metric 291433.1 / ratio −23.6; step 250 metric
  **+144 nats above best** / ratio +0.07; step 500 metric +105 / ratio +1.6 → two
  consecutive floor violations → stop at 500, reverted to best = STEP 0 (the A-only
  flow; its gates fail as at Fv5, recorded). Joined with Fv5's endpoint (step 4000:
  ELBO +35, ratio +13.9), the full Phase-B trajectory at 28 bins / lr 1e-4 is:
  a LARGE IMMEDIATE bulk-damage transient (+144 by step 250, while the ratio has moved
  only 0.07 of the needed ~5.4), then slow non-monotone bulk recovery (+105 → +35)
  co-occurring with ratio overshoot (+1.6 → +13.9). **No stoppable checkpoint on this
  trajectory has both healthy ELBO and in-band ratio — the Fv6 prediction's
  coupled-dynamics assumption (damage and coverage move together) is FALSIFIED: damage
  LEADS, coverage LAGS.** Branch reading: closest to (iv) (bulk-destructive from ~step 0)
  with the trace's added structure; branch (i)'s condition also technically fires
  (ratio < +3.43 at the stop) and arms the carried A2 allowance, but A2 anchors the
  RATIO and does nothing about the leading damage transient — mechanistically
  unpromising on this evidence (recorded, not exercised). Pre-committed route: HUMAN
  ESCALATION, no auto-lever. Suspects for the transient (named, untested): fresh-adam
  fkl kick at lr 1e-4 through 28-bin couplings (a warmup/smaller-lr schedule would
  discriminate transient-artifact vs intrinsic); or fkl's target (the empirical 64k
  draws) genuinely pulling the bulk away from the posterior bulk (the data-limited
  alternative). Attempt 1 crashed at the step-250 ES check (mid-training mesh-annotated
  params — the grader's named watch point; unshard fix committed; ~0.7 GPU-h). Fv6
  total ≈ 2.4 GPU-h; cumulative ≈ 20. Artifacts: es_trace in the AB_es250x4000 cache,
  gate_f_summary.json (gates of the reverted flow), gate_fv6b.log.

- **2026-07-09 (carousel GATE Fv5 RAN — HIGH-SIDE falsifier fired: resolution mechanism
  SUPPORTED, Phase-B 4000 steps OVERFIT; pre-committed human escalation)**
  `proposed (UNCERTIFIED)`. Observed vs re-registered, A+B flow at 28 bins:
  **pocket ratio +13.89** vs target band [+3.43, +7.43] — the ratio moved +1.0 → +13.9,
  so per the pre-committed mechanism-falsification reading the RESOLUTION mechanism is
  SUPPORTED (capacity was binding at 14 bins) — but training overshot: **A+B ELBO
  291488.4 ± 7.9 FAILS the gate** (SVI+35; Fv4's A+B was SVI−21) while train-fkl
  improved (−88.78 vs −87.93) — textbook overfit to the ~6.1k correlated pocket draws,
  exactly the named A1 risk (no held-out fkl monitoring). Pullback-scale still passes for
  A+B (sd [0.907, 1.021]; |mean| crept 0.095 → 0.33). A-only at 28 bins: ELBO fine
  (291432.6), pocket −23.6 (28 bins let reverse-KL carve the pocket out MORE than 14),
  pullback fails (recorded). Physical reading of +13.9: the trap is now INVERTED — q
  OVERWEIGHTS the pocket by ~8.5 nats relative to truth (u-space pocket image ~e^8.5
  UNDER-dense ⇒ chains would rarely enter), AND the bulk fit is broken; this flow is not
  benchmark-eligible on two counts. **Pre-committed route: high-side ⇒ evaluate jointly
  (done above) ⇒ ESCALATE TO THE HUMAN, no auto-lever; the A2 branch does NOT apply
  (it was armed for ratio < +3.43).** Process notes: 4-GPU shard_map training WORKED
  (Phase A 686 s ≈ 11.4 min vs 40 single-GPU; attempt 1 = NCCL rendezvous deadlock of the
  GSPMD variant, 1.3 GPU-h; attempt 2 trained but crashed at the gate stage on leaked
  mesh annotations — fixed, gates re-run from caches). CACHE-KEY GAP found: the flow tag
  encodes lr but NOT phase_b_steps — any Phase-B-length variant needs the tag extended
  or it silently reuses the 4000-step cache. Fv5 cost ≈ 2.8 GPU-h; cumulative ≈ 17.
  Artifacts: carousel_gate_f_out/* (r10b28 caches, updated summary).

- **2026-07-09 (carousel BENCHMARK ATTEMPT 2 RAN — ALL SIX WIN CONDITIONS FAIL; the
  pre-registered NEGATIVE finding stands)** `proposed (UNCERTIFIED)`. Clean run: 4 GPUs
  (2 chains/device), 51 min wall (≈3.4 GPU-h), kernel cap results-phase binding = 0.0
  (burnin 0.0025) ⇒ the pre-committed W1/W4-fail readings apply as written; flow identity
  check passed on GPU; flow overhead 2.8% of a gradient eval (plan's <5% criterion met —
  the one bright number). **Observed vs predicted:** predicted occupancy-ESS ≥ 20/chain/
  1000 (≥10× baseline); observed **0.84 — WORSE than the baseline's 1.9, direction
  reversed.** (W1a fail). W1b: median switches 14.6/1000 (vs required 60; baseline 12);
  min 0 — one chain never switched. W1c: indicator R̂ 1.412 (vs ≤1.05; baseline 1.719).
  W2: pooled occupancy **0.365** — far outside [0.02, 0.15] and above every estimator.
  W3: max R̂ 1.370 (chains under-converged); min bulk-ESS 126.6 vs baseline 3814.9.
  W4: ESS/kept-step ratio 0.066 — flow-MAMS is 15× LESS efficient per step (avg n_k ≈ 20
  vs baseline ≈13, and much lower ESS). PLOTS FIRST (pocket_traces.png): 7/8 chains
  transit visibly MORE freely than baseline chains but DWELL in the pocket 2–4× too long
  (per-chain occupancies 0.15–0.44); chain 8 sat in the pocket ALL 4000 steps with
  near-zero movement (occ = 1.000 — a stuck/degenerate chain driving the min-switch,
  R̂, and ESS failures on top of the systematic over-dwell).
  **Mechanism hypothesis (UNCERTIFIED, quantitatively consistent):** flow-MAMS is exact
  in law regardless of flow quality, but the Fv4 flow's pocket-density smoothing deficit
  — gate ratio +1.0 observed vs +5.43 ideal, i.e. q underweights the pocket by ≈4.4 nats
  — makes the u-space pullback target OVERDENSE by e^4.4 ≈ 80× at the pocket image: the
  flow turned the pocket from hard-to-reach (z-space) into easy-to-reach but
  hard-to-leave (u-space trap). Predicted-mechanism falsification: "both basins in the
  base bulk ⇒ free transit" is FALSIFIED — proximity in u is not sufficient; DENSITY-
  RATIO fidelity (Phase-B quality) sets the trap depth. The 0.365 occupancy is
  transient over-dwell of under-converged chains, not a mass measurement (W2-fail branch
  evidence standard not met: indicator R̂ 1.41 ≫ 1.02, half-split 14.0pp ≫ 1.5pp).
  **Pre-committed consequences:** the NEGATIVE finding for the flow-MAMS mechanism on
  this multimodal target stands (the flow itself remains validated by GATE Fv4);
  NO retuning iteration without a new checkpoint; the plan-§5.4 ladder's next rung
  (ONE grader-gated architecture escalation) is available but NOT exercised without
  human direction. Cost: 3.4 GPU-h; cumulative ≈ 13 GPU-h. Artifacts:
  carousel_benchmark_out/{benchmark_summary.json, benchmark_arrays.npz, pocket_traces.png,
  per_chain_occupancy.png}.

- **2026-07-08 (carousel GATE Fv4 RAN — ALL FOUR GATES AS PRE-REGISTERED; first working
  carousel flow)** `proposed (UNCERTIFIED)`. Observed vs re-registered predictions:
  **(G1) pass** — A-only pocket ratio −16.4 (< −8, fails the gate as predicted; third and
  softest confirmation of objective-level mode-dropping — the overdispersed base sees some
  pocket mass but reverse-KL still under-weights it 17+ nats). **(G2) PASS** — A+B pocket
  ratio **+1.00** ≥ −8: Phase B on the MH-exact draws RESTORED the pocket (expected ≈ +5.4
  for perfect coverage; +1.0 = mild flow smoothing, decisively inside the gate).
  **(G3) BOTH PASS** — A-only ELBO 291430.8 ± 0.1, A+B 291432.4 ± 6.7, both ≈ SVI−21
  (inside the good band) — optimization-from-afar SUCCEEDED from the +180-nat
  overdispersed init (step-0 291633.1 as re-registered; no divergence; pre-committed
  hard-divergence branch never fired). **(G4) PASS for A+B** — main-basin pullback sd
  [0.956, 1.011], |mean|max 0.095: near-perfect whitening; the fkl learned the ~50×
  compression + shape through 14 in-box knots. A-only fails G4 as predicted (sd to 2.85,
  |mean| to 4.75 — recorded). Diagnostics: everything in-box (A+B max|u| main 5.43, pocket
  4.11). Pilot projected 13 min (its step_a diff hit timer noise — recorded; actuals:
  Phase A 2381 s ≈ 40 min at 0.79 s/step, Phase B 372 s); total run ≈ 55 min ≤ 90 budget.
  **The Fv4 A+B flow (car_std_r10b14ts0lr0.003) is the first flow that covers the pocket,
  whitens the main basin, and beats the SVI ELBO — the §5.4 benchmark's preconditioner
  candidate.** Cumulative ≈ 5.6 GPU-h. Artifacts: carousel_gate_f_out/*.

- **2026-07-08 (CPU diagnosis of the 490-bin instability — mechanism: R·lr, not bins)**
  `proposed (UNCERTIFIED)`. Synthetic 33-dim diagonal Gaussian with the carousel's
  whitened sd profile (1.7–50), same flow/optimizer/loss, CPU, no renders. REPRODUCED:
  range 357 blows up at lr 3e-3 within 10 steps regardless of bins (490 AND 48 —
  **bin count exonerated**); stable at lr 1e-4; range 11 stable at 3e-3 (bins 16 and 48).
  LOCALIZED (param-group probes): the zero-init final conditioner layer's first adam step
  is rank-1-coherent (grad W_ij = h_i·ḡ_j), shifting ALL width/height logits by
  ~lr·Σ|h| ≈ 30–60·lr; the softmax → ×2R → cumsum knot decoder converts any coherent
  logit shift into knot displacement with gain O(R) — measured one-step output kick
  ≈ 0.35·R, BIN-COUNT INDEPENDENT. At R=357 that is ~125 whitened units against target
  sds ≤ 50: lp explodes, adam keeps stepping at fixed per-coordinate rate, settles in a
  noisy equilibrium far above init. Never-hit-bin softmax coupling confirmed but
  secondary; box-edge terms ruled out. **Stability knob: R·lr** (measured: 1.07
  catastrophic, 0.036 marginal, ≤0.035 stable) — retro-consistent with EVERY prior run
  (demo v3 R35·1e-3=0.035 stable; v4 R6·3e-3=0.018 stable; Fv2 R16·3e-3=0.048 stable-ish;
  Fv3 R357·3e-3=1.07 unstable). Caveats recorded: diagonal target (mechanism is
  parameterization-side), pre-standardized runs test stability-near-optimum. Scripts/logs
  archived in-repo at experiments/flow_precond/instability_diagnosis/. Diagnosis only; no
  fixes applied.

- **2026-07-08 (carousel GATE Fv3 PILOT FIRED — run aborted pre-budget; two findings)**
  `proposed (UNCERTIFIED)`. The pre-committed timing pilot caught both problems at
  ~2 GPU-min: (1) **Phase-B OOM at the approved 32 chunks** (21 GiB per 2000-row chunk —
  490-bin spline internals ~20× the 24-bin footprint; would need ~256 chunks, ballooning
  step time); (2) **Phase-A optimization INSTABILITY at 490 bins**: step-0 loss 291453.5
  (identity-init nesting still exact) but +5938 nats after 4 steps, +7370 after 8 —
  monotone INCREASE at the same lr that was stable at 8/24 bins. The Fv3 blind spot (a)
  ("knot-allocation dynamics unmeasured at 490 bins — loss plots will show") materialized
  immediately. Config numbers as pre-registered (range 357, bins 490; zP/zM containment
  180.3/177.4 in-box, matching the grader's independent values). No gates evaluated — this
  is a pilot-stage infeasibility, not a G1–G4 result; the pre-committed abort +
  re-checkpoint path is in force. Cost ≈ 0.1 GPU-h. Next: CPU diagnosis of the instability
  (flow-only, render-free), then a revised checkpoint (Fv4).

- **2026-07-08 (carousel GATE Fv2 RAN — fixed box ±16/24 FAILS; one-shot fixed-box premise
  FALSIFIED per pre-commitment; HUMAN ESCALATION)** `proposed (UNCERTIFIED)`.
  Observed vs re-registered: **(F1′) pass** — A-only pocket ratio −406 (still fails the
  pocket gate as predicted; more extreme than ±6's −108.8). **(F3′) BOTH PASS** — A-only
  ELBO 291427.0 ± 0.1 (= SVI−26, in the derived band; nesting argument confirmed live:
  step-0 loss 291453.7 = SVI), A+B 291433.8 ± 5.5 ≤ SVI (Phase B no longer breaks the ELBO
  — real improvement over ±6). **(F2′) FAILED** — A+B pocket ratio −403, essentially
  unchanged from A-only. **(F4′) FAILED for both** — main-basin pullbacks now reach |u| ≈
  41 (vs 22 at ±6), pocket pullbacks 37.5 ≫ box 16; predicted |T⁻¹(z_pocket)| ≲ 10 was
  WRONG by 4×. **Mechanism (from exp_s):** trained exp(s) ∈ [0.75, 13.5] — SMALLER than at
  ±6 ([1.1, 34.6]). The sizing arithmetic assumed s → main-basin sd; but s is trained by
  reverse-KL, which sets it to fit the BULK and is indifferent to regions it assigns ~0
  mass. The mode-seeking pathology RECURSES AT THE SCALE LEVEL: a containment parameter
  trained by an objective that ignores the tails cannot contain them, for ANY fixed box.
  This explains both GATE F failures and elevates the plan-§6 data-derived range (set by
  DATA quantiles — containment by measurement, not by ELBO; v3-validated on the demo) from
  fallback to necessity. **Pre-committed outcomes now in force:** (i) F2′ failed with
  pocket pullbacks OUT of box ⇒ the subsampling/early-stopping retry does NOT apply; the
  pre-registered negative finding stands ⇒ HUMAN ESCALATION before any benchmark. (ii)
  Premise-level pre-commitment: no third widening — the one-shot fixed-box premise is
  FALSIFIED; the data-derived-range path becomes the method, pending human concurrence.
  The ±16/24 demo re-validation is moot (config dead). Cost: Fv2 ≈ 1.1 GPU-h (45-min
  TIMEOUT with Phase A banked + 20-min completion; overran the approved ≤45 GPU-min —
  Phase-A step time at 24 bins ~2× estimate; recorded). Cumulative ≈ 4.5 GPU-h.
  Artifacts: carousel_gate_f_out/* (r16b24 caches, summary, losses).

- **2026-07-08 (carousel GATE F RAN — F1 CONFIRMED, F2/F3/F4 FAIL with structural
  diagnosis; retry premise falsified)** `proposed (UNCERTIFIED)`. Observed vs predicted:
  **(F1) CONFIRMED precisely** — A-only pocket ratio **−108.8 nats** (predicted ≈ −100s):
  reverse-KL mode-dropping of the 14σ pocket is REAL on the carousel (the plan-§4.4 claim,
  now measured). A-only ELBO 291426.9 ± 0.2 ≤ 291453.1 (F3 pass for A-only; −26 nats below
  SVI; step-0 loss 291453.7 = SVI, nesting confirmed live). **(F2) FAILED, and worse than
  A-only**: A+B pocket ratio −121.6; A+B ELBO 291458.9 ± 8.0 (> SVI ⇒ F3 fails for A+B
  only); both flows FAIL F4 (main-basin pullback sd to 4.5, |mean| to 8.1).
  **DIAGNOSIS (CPU, from caches + arrays):** NOT overfitting — Phase B trained cleanly
  (fkl 131.3→15.5 monotone). STRUCTURAL: the carousel posterior in SVI-whitened coords
  extends to **|w| = 322** (vs demo's 31; both basins — the curved ridge's tails, not the
  pocket), pullbacks reach |u| ≈ 20–22 ≫ the ±6 box, so most training data sits where the
  splines have ZERO capacity and only the shared DiagScale can respond — improving average
  data-likelihood there (what fkl does) degrades the interior (ELBO ↑, pocket density at
  zP ↓). A shared per-dim scale cannot represent bimodality; the box must contain the
  POST-SCALE shape. Measured post-scale requirements (scales ≡ main-basin sd): per-dim
  dynamic range ≤ 9.2 (median 4.4), pocket-mean offset ≤ 4.2 main-sd units ⇒ a fixed
  ±12–16 box suffices with margin. **The pre-committed F2 retry (subsampling/
  early-stopping) is NOT executed: its premise (overfit) is falsified by the diagnosis —
  amendment v2 goes to the grader instead.** Two OOMs en route, both fixed exactly
  (gradient-accumulated chunks; Phase-B equivalence verified to 9e-16). Cost: ~0.7 GPU-h
  across 3 attempts. Artifacts: carousel_gate_f_out/{gate_f_summary.json, flow caches,
  gate_f_losses.png}.

- **2026-07-08 (demo validation v4 RAN — one-shot config CLEAN SWEEP)**
  `proposed (UNCERTIFIED)`. Observed vs predicted, all pre-registered: **(10) PASS** —
  pullback-scale gate passes for BOTH one-shot flows (A-only sd [0.99, 1.14], A+B
  [1.02, 1.13]; |mean|max 0.19/0.06) at FIXED range 6 + trainable DiagScale — nothing
  data-derived. **(11) PASS, unconfounded** — arm B′ R̂ 1.003, worst mean dev 1.48σ, sd
  ratios in gate, min bulk-ESS 1151 (vs vanilla 412); flow gate −76.75 ≤ −70.98 (best flow
  of any config). **(12) health-fail branch NOT triggered** — arm C′ (Phase B) PASSED
  health (R̂ 1.015, ESS 575) + agreement + the B-vs-C width gate (0/22 fail; v3's fail was
  annotated non-interpretable) — Phase B did NOT degrade the one-shot flow (unlike v3's
  range-35 flow; C′ still 4× arm B′'s wall: 339 s vs 85 s). Carousel Phase B therefore
  proceeds as planned (full-batch fkl), per the pre-commitment. (6) arm D diverged
  (expected, recorded); arm E repeat pass with divergence gate FAILED again (7/2400 —
  flag persists, third occurrence). Phase-B gate PASS (−82.9→−86.6, finite decreasing).
  **Demo validation phase COMPLETE**: flow-MAMS and whitened-NeuTra-NUTS both validated at
  demo scale on the one-shot architecture; faithful NeuTra divergence + mechanism archived.
  Cumulative demo-phase cost ≈ 2.7 GPU-h. Artifacts: demo_validation_out/* (cache keys
  r6b8ts1lr0.003).

- **2026-07-08 (demo validation v3 RAN — flow-MAMS VALIDATED on demo; 2 predictions failed,
  both informative)** `proposed (UNCERTIFIED)`. Observed vs predicted, per pre-registered gate:
  **(9) ARM B (flow-MAMS, Phase-A spline, range 35) FULL PASS** — health R̂ 1.009, worst mean
  dev 1.68σ (<4), sd ratios [0.94, 1.03], min bulk-ESS **1256** vs vanilla arm A's 412 at
  equal kept draws (3.0× ESS; wall-clock: 7.2 ESS/s (B, incl. its 1000-step burnin) vs 8.4
  ESS/s (A, 300 burnin) — vanilla ~15% better on this easy system; the geometry win is
  expected to matter on hard targets, not here). The MAMS × nonzero-flow link is now
  validated at demo scale. **(2'') ARM E repeat pass** (R̂ 1.005, worst 2.24σ; divergence
  gate FAILED again: 5/2400 — persistent curvature flag, weigh before carousel NeuTra use).
  **(6) ARM D diverged as predicted** (faithful NeuTra; mechanism archived). **(1)/(5) flow +
  Phase-B-trains gates PASS** (neg-ELBO tail −76.6 ≤ −70.98; fkl −83.0→−91.1).
  **PREDICTION (8) FAILED — informative:** Phase-A-only PASSED the pullback-scale gate
  (sd [0.99, 1.24], |mean|max 0.21). With range 35, reverse-KL learned the scales fine on
  this unimodal target ⇒ the v2 attribution "mode-seeking kept SVI underdispersion" was
  WRONG in its mechanism — the ±6 range clip was binding for Phase A too. CORRECTION to the
  v2 entry recorded here. Scope: this does NOT bear on the carousel §4.4 pocket claim
  (mode-dropping of a separate 14σ mode is a different mechanism from same-basin scale
  learning; the demo has no second mode to drop) — the carousel GATE F A/B remains the test.
  **PREDICTION (2') FAILED for ARM C — negative finding about Phase B on unimodal targets:**
  the A+B flow passed pullback-scale (sd [1.13, 1.46]) but arm C failed health (R̂ 1.076,
  min-ESS 232) and was 7.5× slower than arm B (1306 s vs 174 s) with width ratios to 1.18.
  Reading: forward-KL refinement on 2400 correlated MAMS draws (ESS~400) degraded an
  already-good flow (train fkl kept improving — consistent with overfitting the finite
  sample and roughening the pullback geometry). On the demo Phase B had nothing to fix;
  its intended value (pocket coverage) is untestable here. Implication for carousel: run the
  pre-registered A-only vs A+B GATE F exactly as planned, but treat Phase B as
  needing-evidence, not presumed-better; consider more/less-correlated training data and
  early stopping if it fails there too. Per v3 pre-commitment, no demo re-runs of arm C
  without an explicit human decision. Budget-carried cross-check: NOT triggered (pullback
  gate (7) passed). **Pre-registered B-vs-C width gate (prediction 3): FAILED**
  (`bc_width_gate.pass_=false`, n_fail=2, ratios to 1.19) — NON-INTERPRETABLE because arm C
  failed health (same rule as agreement gates); recorded, not dropped. **Corner overlays
  vs true HMC:** flow-MAMS (arm B), vanilla MAMS (arm A), and the 72k-draw HMC reference
  (`experiments/laps_validation/handoff/hmc_ref/hmc_mass.npy`, produced by the handoff
  `hmc_reference.py` demo-lens run) coincide at 68/95% on all 8 mass params —
  `demo_validation_out/corner_mass_chunk{1,2}.png` (two 4×4 chunks per legibility rule).
  Cost: ~0.6 GPU-h. Artifacts: demo_validation_out/* (summary JSON, arrays,
  traces_worst_param.png, agreement.png, flow_losses.png, corner chunks), branch
  flow-precond-mams.

- **2026-07-08 (demo validation v2 RAN — mixed; B/C failure DIAGNOSED as range-clipped flow
  + under-adaptation, not a bug)** `proposed (UNCERTIFIED)`. Observed vs predicted:
  predictions (1) flow gate PASS (−75.18 ≤ −70.98), (5) Phase B PASS (19.94→7.14),
  (6) faithful NeuTra diverged step 2 as predicted (mechanism archived), and **arm E
  (whitened-IAF NeuTra-NUTS) passed all agreement + health gates** (worst mean dev 1.9σ,
  sd ratios [0.97,1.08], R̂ 1.009); E's pre-registered divergence gate FAILED (6/2400
  divergent transitions vs the 0-exactly rule — recorded finding; a curvature flag whose
  weight is to be assessed later, not graded away now). **Arms B/C (flow-MAMS) FAILED agreement + health** (worst 18σ, sd ratios
  →0.48, R̂≈1.26–1.28, u-R̂ 1.39). Diagnosis (CPU, from committed arrays): the pulled-back
  demo posterior through the spline flow has per-dim sd up to **7.2** and |mean| 6.4 in
  u-space (reverse-KL mode-seeking inherited SVI's underdispersion — the SAME flaw family
  the plan §4.4 predicts for the carousel pocket); MAMS's 300-step burnin from identity
  mass never adapted to those scales (chains spread only 1.1–3.6σ), while arm E's NUTS had
  1000 warmup steps. AND Phase B could not fix the scales (sd 7.18→7.21 unchanged) because
  the offending draws sit at up to **31 whitened-σ — outside the ±6 spline range where the
  flow is exact-identity with zero capacity** (plan §6 "widen if not" realized; range is
  binding). Kernel exactness is not the leading hypothesis (E validates wrapper+decode on
  the NUTS path; GATE I validates MAMS at identity flow) — but the MAMS × nonzero-flow
  combination that B/C exercise is validated by neither and is tested by v3 prediction (9).
  v3 amendments follow in the design checkpoint. Cost: ~0.4 GPU-h (run ~20 min).
  Artifacts: demo_validation_out/{demo_validation_summary.json, demo_validation_arrays.npz,
  traces_worst_param.png, agreement.png, flow_losses.png}.

- **2026-07-08 (demo 4-arm validation RAN — TIMEOUT; diagnosed, 3 findings)**
  `proposed (UNCERTIFIED)`. Run hit the 45-min limit with arms unfinished; stdout lost
  (buffered; use PYTHONUNBUFFERED next time). Observed vs predicted, from cached artifacts:
  (1) **Spline Phase A healthy + flow gate (1) PASSES pre-registered prediction**: neg-ELBO
  starts −71.2 (= SVI −70.98, the identity-init nesting prediction) → tail −75.18, no NaN.
  (2) **Phase B forward-KL NaN at step 1** — the plan §6 "spline tail under-coverage" failure
  mode, realized: demo MAMS draws reach **max |T⁻¹(z)| = 31.4 whitened-σ** (7.1% of coords
  outside the ±6 spline range) ⇒ full-rank SVI (n_vi=128, 1500 steps) is strongly
  underdispersed vs the true demo posterior along several directions. Step-0 loss finite
  (19.9) ⇒ the NaN is the *gradient* at out-of-range points (spline out-of-range should be
  exact-identity with finite grads — implementation hardening bug, fix + unit-test in
  flows.py before any widen-the-range knob).
  (3) **NumPyro-faithful IAF (unwhitened, 1-sample ELBO Adam 3e-3) NaN at step 3** from
  neg-ELBO ~1.5e5 (N(0,I) start vs sharp posterior). Whether "plain NeuTra as-shipped
  diverges on the easiest lensing system" is a real baseline finding or an implementation
  artifact is OPEN until the numerics are audited vs numpyro source (stable log-scale
  handling etc.). Do NOT cite as a result yet.
  (4) **Hang mechanism (why timeout, not crash):** arms C/D sampled with all-NaN flow params;
  MAMS's NaN-guard shrinks step_size_max ×0.8 per event ⇒ ε collapses ⇒ deterministic
  trajectory length n_k = L/ε explodes ⇒ wall-clock black hole. Samplers must FAIL FAST on
  non-finite flow params (guard added to the script).
  Cost: 0.75 GPU-h. Artifacts: demo_validation_out/*.npz (flow caches incl. NaN params,
  MAP/SVI cache). Next: fix + re-gate (grader), rerun with cached flows where valid.

- **2026-07-08 (GATE I RAN — PASS)** — **Identity-flow wrapper ≡ vanilla MAMS, bit-identical.**
  `proposed (UNCERTIFIED)`. Observed vs predicted: predicted max|Δ|=0.0 exactly; observed
  `np.array_equal=True`, max|Δ|=0.0, on (8,300,22) demo-lens result samples (seed 0), plus
  pre-diagnostic `log_prob` bit-identity on 64 qz draws (max|Δ|=0.0). Trace overlay confirms
  (gate_i_traces.png). Model card: worktree `gigalens_research`, jax 0.10.0.dev20260708, x64,
  1×A100. Discharges ONLY the no-op-at-identity property (per approved checkpoint); the
  nonzero-Jacobian path is covered by `inference/tests/test_flows.py` (11 green: fldj vs
  autodiff slogdet at ~1e-15 for IAF + whitened-spline) — unit-level, not yet a sampling gate.
  Two environmental fixes en route, neither touching the wrapper: (a) demo SVI n_vi=1000 OOMs
  a 40GB A100 → 128; (b) `model_seq.SVI`-returned qz carries Explicit device sharding that
  new JAX rejects when MAMS closes over `qz.covariance()` inside shard_map → rebuild qz from
  host arrays (what pipeline SVIStage does anyway). NOTE for anyone running MAMS_JIT outside
  the pipeline: fix (b) is required in this jax version. Artifacts:
  `experiments/flow_precond/gate_i_out/{gate_i_verdict.json,gate_i_arrays.npz,gate_i_traces.png}`,
  branch `flow-precond-mams`. Cost: ~0.5 GPU-h incl. 2 failed attempts.

- **2026-07-06 (Test 4 + Test 2)** — **Catastrophic ξ = PHYSICAL localized caustic stiffness; numerics fully excluded.** `proposed (UNCERTIFIED)`. Test 4 (EPL_Lf.center_x dial-scan, conv_precision float32 vs float64 at ss1 and ss5): **f64/f32 comb ratio = 1.001** at both ss ⇒ the teeth are NOT float32-convolution roughness. Test 2 (per-pixel |∂model/∂center_x| via jvp at the max-ξ draw): sensitivity is highly localized — **top-1% of pixels carry 92% (band0) / 95% (band1)** — and in band0 it **traces the bright lensed arcs, peaking right at the perturber** (peak px→(-15.0,-4.6) vs perturber (-14.78,-4.48)). Synthesis: the small (θ_E≈1.07), maximally-elliptical (e1≈0.50, pinned at the +0.5 wall) EPL_Lf perturber sits among the lensed arcs; its position near-singularly controls those arc pixels (a critical-curve/caustic effect), producing the sharp likelihood teeth that at ss=1 the sampler occasionally hits → the catastrophic ξ that suppress eps globally. Ruled OUT across Tests 1/4: pixel-aliasing, EPL series, float32 numerics. **Fix is NOT render fidelity / conv precision / a variance-stabilizing reparam (this is not a smooth funnel).** Likely levers: regularize the perturber (cap ellipticity below the wall / add a core / question whether the component is justified) or sampler robustness. This unifies the user's two symptoms (ellipticity-at-bounds + ξ spikes) into one cause. Artifacts (on $SCRATCH, home quota full): /pscratch/sd/l/linusu/carousel_diag/{P12_f64_vs_f32_comb,P13_localize_band0_shapelets,P13_localize_band1_sersic}.png, test_f64_loc.npz. NOT yet done: cond(G) of the lstsq Gram at spike vs calm (lstsq-conditioning sub-check).
- **2026-07-06 (later)** — **Test 1 dial-scan RAN + Batch A labeling CORRECTED.** `proposed (UNCERTIFIED)`. (a) C-8 trap caught: Batch A names were insertion-order; true map is sorted-key (`z_names_TRUE.json`, perturbation-verified). The ξ lattice is in **EPL_Lf position (center_x,center_y)**, not (theta_E,gamma). (b) Test 1 (EPL_Lf position dial-scan, supersample{1,3,5}×niter{50,200}, other 30 params frozen at the max-ξ draw; faithful rebuild reproduces run red_chi2=1.1608): **teeth do NOT collapse under supersampling — they proliferate**; **niter has zero effect**. ⇒ pixel-aliasing and the EPL series both FALSIFIED. The perturber (small θ_E≈1.07, e1 at the +0.5 wall) produces near-singular caustic structure in the likelihood-vs-position that finer grids resolve as sharper; at the run's ss=1 it is mostly blurred with a few sharp teeth = the catastrophic ξ events. **Implication: raising supersampling is NOT a fix (would expose more teeth) — Test 3's premise is falsified in advance.** Remaining discriminator (cheap): rerun the SAME dial-scan at conv_precision=float64 to separate physical-caustic from float32-conv roughness that worsens with pixel count. Artifacts: batchA_diag/P9–P11, test1_dialscan.npz.
- **2026-07-06** — **Batch A diagnosis of the NEW complex-carousel run (`messy_tests/just_map`);
  EPL_Lf ξ-lattice found.** `proposed (UNCERTIFIED)` → see **C-21** and the two Design
  checkpoints (Test 1/3). Zero-GPU reads on the saved `diagnostics.npz`: confirmed the user's
  global-`eps`-suppression hypothesis quantitatively (top-8 burn-in steps = 74% of Σξ; `eps`
  pinned 0.1255) and localized the catastrophic ξ spikes to a **quantized 2-D lattice in the
  EPL_Lf perturber's (theta_E, γ)** (γ comb spacing ≈0.189; the other EPL clean). Ruled out the
  1-D funnel / prior-wall / reflection / init / rotating-metric stories and the `niter` angular
  series (theta_E absent from it yet bands). Leading hypothesis: `supersample=1` pixel-aliasing
  comb from the under-resolved perturber (sys60 disease-(i) analog). Artifacts:
  `experiments/sim_carousel/messy_tests/just_map/batchA_diag/` P1–P8.

- **2026-07-02** — **C-8 open item (2) partially resolved: the full 32-param case's cached
  `names.npy` IS reversed.** `proposed (UNCERTIFIED)`. Evidence: (a)
  `experiments/sim_carousel/_h1h2_diag/names.npy` is stored in *exactly* reversed-alphabetical
  key order (checked empirically, 2026-07-02; `sorted(names, reverse=True) == names` is True),
  i.e. it was written via the unsorted `flatten_param_names(bij.forward(...))` path; (b) the
  validated column rule (`plotting/diagnostics.py::_z_space_labels` COLUMN ORDER NOTE) says
  sampler column *i* = alphabetically-*i*-th key. Together: `names.npy[i]` labels column
  `dim−1−i`, so any full-case analysis that read `names.npy` positionally inherits the C-8
  reversal — per C-8, full-case *parameter-identity* claims (e.g. which named directions are
  slow) need re-derivation via sorted keys; geometry/curvature evidence is unaffected. The new
  `experiments/why_hard_to_sample/` harness (see `why-hard-to-sample.md` log) therefore never
  uses `names.npy` positionally: it sorts the key set (T0/T1 `common.py::load_param_names`) or
  zero-probes the bijector with sorted output keys (T2/T5 scripts). NOT yet done: an AD/FD
  cross-check on the 32-param bijector itself (the minimal-case-style direct test); the sorted
  labels are validated only by the C-8 mechanism, not independently.

- **2026-06-29** — **APS-on-MCLMC ran on the GPU carousel (C-20).** Built `tempering/apt_carousel.py` + `carousel_aps_run.py` (Option-A augmented sampler, frozen `upper_cov` metric, per-step EEVPD adapt, matched-cov Laplace base). First run COLLAPSED (Appendix-C); diagnosed the structural cause = unnormalized-posterior evidence offset ~1e5 (matched cov ≠ matched normalization); fixed by subtracting the Laplace log-evidence C=−119562.66. Post-fix: NO collapse (k̂≤0.46, log z'≈−5 stable, frac_cold>0, EEVPD faithful), cold draws recovered. Independent grader found the cold draws BIMODAL/filamentary; GPU `cluster_check.py` disambiguated → REAL curved-ridge mass (both clusters same density; 791-nat straight-chord dip = curvature), cold-ESS-limited — the C-5/C-7 degeneracy, inherited not cured. Independent 32-start MAP (global −119502.93, only 3/32 within 5 logp) corroborates the curved-ridge geometry. Single-seed diagnostic; weights unreliable; λ→1 EEVPD + vanilla same-harness baseline still open. Process: orchestrator ran all GPU harnesses directly (srun --overlap, per-GPU flock) and delegated only non-GPU code-audit + grading to subagents.

- **2026-06-28** — **Two parallel mode-hop variants tested + orchestrator-audited (C-14/C-15/C-16).** Human
  redirected to KEEP pursuing mode-hopping (expects comparable-mass modes in future). Two Opus subagents
  (research→implement→test, simplest-first, on curved CPU testbeds): **DE γ=1 teleport + snooker (C-14)** —
  faithful (snooker (D−1) Jacobian source-verified + drop-Jac bias demo), unbiased, solves benign separated
  modes, but fails curved cross-mode hop (off-ridge 17 vs 3) and pins tiny modes ⇒ REJECTED. **Self-inclusive
  SA-MCMC (Zhu 2019) (C-15)** — faithfully unbiased (Alg 1 + Prop 1 verified from PDF), the FIRST variant to
  equilibrate COMPARABLE-MASS modes on the curved ridge (0.30→0.608, ~13× DE throughput at carousel-level
  curvature). Both its red flags (dim0 ridge collapse; high-curvature weight bias) ATTRIBUTED by orchestrator
  to the UNADJUSTED MCLMC kernel, not SA (pure-SA K=0 preserves the ridge & weight; vanilla MCLMC alone
  collapses). Caveats: pins tiny modes (structural, C-16); composite needs finer step/MAMS on strong curvature;
  CPU surrogates milder than the real carousel ⇒ MERITS the GPU carousel. **Process:** subagents again
  orphaned concurrent detached jobs that got login-node-killed mid-run; orchestrator re-ran the load-bearing
  attributions (`sa_mcmc/{attrib_ridge,audit_equil_sweep}.py`, `de_teleport/audit_curvature_sweep.py`) directly,
  one job at a time. GPU node had timed out — GPU test pending a fresh allocation.

- **2026-06-27** — **Kernel-hop tested + strategic reframe (C-13).** NKC kernel-hop is unbiased but knife-edge
  bandwidth + tiny-mode over-representation + no discovery; curvature advantage untested (Gaussian testbed).
  3 jump variants now failed; none solves discovery. Reframe: negligible-mass modes ⇒ need discovery+avoidance
  (tempering/multi-start), not equilibration jumps. Recommend stepping back from jump-engineering.

- **2026-06-27** — **DE jump efficiency = curvature-limited (C-12).** Carousel-faithful Gaussian testbed
  did NOT reproduce the failure (8.7% vs 0.6%); model-free kNN shows linear DE proposals land off the real
  curved ridge (7.2× secondary). Jitter/p_jump/weight ruled out. Affine fixes (snooker/mode-matching) won't
  help; only nonlinear transport would. ⇒ recommend pivot to tempering (curvature-robust + provides discovery).
- **2026-06-27** — **DE-MCLMC composite prototyped + tested (C-11).** Standalone (MCLMC untouched).
  Unbiased on analytic mixture (weights/invariance/KS pass). On carousel: equilibrates from a balanced
  ensemble (0.50→0.88, DE-attributed via refresh-only=0 control) but CANNOT discover from all-secondary
  (0 crossings) and is jump-rate-limited (~0.6% acc) on the curved degeneracy. ⇒ DE solves equilibration,
  not discovery; production needs a discovery mechanism (global-MAP/tempered init).

- **2026-06-27** — **MCLMC barrier-crossing diagnostic (C-10): STRUCTURAL.** Fixed-knob escape sweep
  on the src4-cx double-well (on-ridge z_best init). No L reaches uniform <2000-step escape; larger L
  is monotonically worse; only the both-mode-covering pooled MM enables crossings (within-mode 1/8,
  identity 0/8); global-init chains never find the secondary basin (0/8). ⇒ vanilla MCLMC can't
  robustly cross/discover separated basins. Next: prototype DE-MC ensemble jumps on the same harness.
  (Process note: subagents kept backgrounding the run and one left orphaned 4-GPU container procs that
  OOM'd the node; ran the final clean sweep directly with a GPU-fraction cap.)

- **2026-06-27** — **src4 cx bimodality characterized (C-9).** Profile-likelihood barrier test:
  genuine secondary local max (barrier ~4 logp) but ~11.8 logp below the global ⇒ negligible mass;
  the 500-step MAP is mislocated *in* the secondary basin, so the run samples it out of equilibrium
  (0 round-trips, one-way drain). Not a caustic crossing. Recommends a global-basin MAP + dispersed
  init rerun (pre-registered prediction in C-9).
- **2026-06-27** — **Minimal case + labeling-bug discovery (C-8).** Mode A (orchestrator).
  Regraded the run's own artifacts (χ²/ν 1.19; rank-R̂≈1.35 still falling; min ESS≈12–20 still rising
  ⇒ slow/under-converged, not stuck). Found the column↔name **reversal** (C-8) via AD+FD on the
  per-coordinate bijector; **withdrew** the "z-coordinate-artifact" reading. Real worst-mixing param =
  **src4 center_x**, physical bimodality. Dispatched two subagents: (1) verify+robustly-fix the
  `PipelineReport` labeling; (2) re-characterize the source-position modes (real vs unconverged) incl.
  a 2-D logp-slice barrier test. Recheck artifacts in `experiments/sim_carousel/minimal_case_recheck_plots/`.

- **2026-06-26** — Code changes (durable):
  (1) `inference/mclmc.py` `skip_adapt`: log the real energy-error ratio `xi` during the
  results phase instead of a −1 sentinel (logged-only, does not affect sampling).
  (2) `inference_utils/posterior.py` `SamplerPosterior._rhat/_ess`: switched to
  rank-normalized split-R̂ = max(rank, folded) and rank-normalized bulk-ESS via ArviZ
  (lazy import; runtime deps are env-managed). Propagates to `PosteriorReport.diagnostics`,
  `running_rhat/ess`, and the `max_rhat`/`min_ess` simtest metrics. Verified == ArviZ.
- **2026-06-26** — Tests A1/A2 (smoothness): ruled out numerical roughness as the bottleneck
  (C-4). Found `conv_precision="float32"` adds an inert 2e-4 noise floor.
- **2026-06-26** — Tests C/B (localize + ridge-trace): slow directions are smooth curved
  degeneracies, not multimodal (C-5).
- **2026-06-26** — Test 2 (Gaussian reference): ruled out conditioning (C-3).
- **2026-06-26** — 4000-step MAP (run by human) closed the gap (C-1, C-2).
- **NEGATIVE RESULTS recorded:** condition number (C-3), numerical noise (C-4),
  multimodality in the slow subspace (C-5), NFW instability (C-6), and — from the session —
  the "boundary-stretch corrupts the mass matrix" hypothesis (the largest mass-matrix
  eigenvalues sit on genuinely-wide weakly-identified params, not boundary params; energy
  error is *anti*-correlated with boundary excursions). `desired_energy_variance` and
  per-chain mass matrix are deprioritised as linear levers (C-3).

---

## Open questions

- **[HIGH-IMPACT] Does the C-8 column↔name reversal also affect the full 32-param case?** C-5/C-7
  attribute the slow/curved directions to specific *named* params ("ellipticity–shear", "NFW
  position/scale"). Re-run the AD/FD column→name check on the full model; if reversed there too, the
  curvature evidence survives but the parameter *labels* on it do not. Verify before citing C-5/C-7 names.
- **Is the EPL_Le ellipticity pinned at its prior bound (|e|≈0.30, e2≈0.29 vs cap 0.30) a
  physical or prior-driven result?** The data appears to want a more elongated mass than the
  `TruncatedNormal(...,−0.3,0.3)` prior allows. Widening the prior is *prior-fiddling* (the
  human wants prior-robust sampling, not per-prior tuning) — but the physicality question
  stands independently.
- **What is the prior-robust fix for the curved-degeneracy slowness (C-5)?** Candidates not
  yet tested: a learned transport/normalizing-flow warmup (NeuTra-style) to Gaussianize the
  target (predicted to recover ~Gaussian ESS per Test 2); position-dependent/Riemannian
  metric; or simply budget to a target ESS (rank-R̂ already ~1.7, ESS scales ~linearly with
  draws → ~30× more results steps for ESS≈400). None certified.
- **Ridge-trace the 2nd-slowest eigen-direction (n_sersic-bearing)** to confirm it too is
  unimodal — not yet done (C-5 caveat).
- **Does flipping `conv_precision` to float64 change ESS at all?** Predicted negligible
  (C-4); not run.
