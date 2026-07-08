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

---

## Design checkpoints (criteria awaiting approval)

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
