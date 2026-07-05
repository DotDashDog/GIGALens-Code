# Sampling-Diagnosis Playbook — why lensing posteriors are hard, and how to tell which way

Status: proposed (UNCERTIFIED) — drafted 2026-07-04 from `docs/logs/why-hard-to-sample.md`;
grader: the human.

This playbook distills a two-arc investigation into *why gigalens lensing posteriors sample so
much slower than their converged cornerplots (all nearly Gaussian) would predict*. Every claim
below carries a **T-number receipt** into `docs/logs/why-hard-to-sample.md`; if you cannot find a
number there, it is not in here. Nothing in this file is certified. It is a map of what we
measured, what the measurements *looked like*, and — for each instrument — an explicit warning
that a shape we did **not** see is a candidate NEW pathology to characterize and report, never to
pattern-match onto this catalog.

The two arcs produced **two different classes of bad posterior**, and the single most important
lesson of the whole investigation is that they are genuinely different diseases that a
cornerplot cannot tell apart:

- **sys60 (a fully synthetic Sérsic system, no lstsq layer).** The *computed* likelihood was
  drastically stiffer than the *intended* one — a supersample-grid aliasing "comb" that the
  posterior locked onto because the data and the model were rendered at the *same* low fidelity
  (the self-consistency trap). This is the **likelihood-model-defect class**: the geometry the
  sampler fights is an artifact of how the likelihood is *computed*, not of the physics.
- **carousel (a real observed system, lstsq amplitudes).** A stack of ordinary-but-real
  diseases: an under-converged MAP sitting in a genuine local maximum (init class), float32
  convolution numerics corrupting every finite-difference check (numerics class), a curved
  degeneracy valley in the old NFW coordinates (parameterization class), and a
  marginal-vs-conditional funnel in `Rs` whose neck the adaptation tuner responded to by
  globally suppressing the step size (funnel / tuner-suppression class).

Neither system's slowness was "banana curvature" in the loose sense agents had repeatedly
invoked (`why-hard-to-sample.md`, O3 / the human's verbatim primary source). Both had
near-Gaussian marginals. The hardness lived in places cornerplots are structurally blind to.

---

## 1. How to use this playbook

**Pre-register before consequential runs.** Anything that will enter the record or inform a
decision goes through `/pre-run-checklist` first: a cause hypothesis, a predicted direction *and
order of magnitude*, and a falsifier you can derive. This is not ritual — half the corrections in
the log below happened because a *threshold* or a *falsifier window* was wrong, not because a
measurement was wrong (T23 wall statistic, T25 window). The governing discipline is
`docs/method-discipline.md`; this playbook is downstream of it and must never contradict it.

**Plots before metrics, and the plot wins.** Every instrument section below leads with *what the
plot looks like*, because in this investigation the plot repeatedly overturned a passing or
failing number: T3's "roughness" table flags were macro-averaging artifacts that the D2(h) *plot*
exonerated; T12's registered 1/px Fourier metric failed while the comb was plainly visible at
1.9/px; T25's profile "flat" number came from a mis-drawn window that the full profile plot
contradicts. If a number passes and the plot disagrees, that is an open finding, not a pass
(`method-discipline.md` §5).

**Producers never self-certify.** Findings are `proposed (UNCERTIFIED)` until the human grader
inspects artifacts — not summaries. Log negative results too; they are the point of a
decision-tree design.

**Every number carries a receipt.** Below, "(T10)" etc. points into
`docs/logs/why-hard-to-sample.md`. Reference implementations live in
`experiments/why_hard_to_sample/`; each instrument names its script. Do not copy code out of
them into analyses — call them.

---

## 2. Symptom triage — "sampling is slow / weird", the first three cheap things

You have a run with low min-ESS, bad R̂, or a posterior that "looks wrong". Before any exotic
hypothesis, compute these three, in order. They are cheap (two are zero extra GPU) and between
them they split the hypothesis space that took two arcs to map.

### (a) The matched Gaussian-clone gap — THE master metric

Fit a full mean+covariance Gaussian to *your own post-burn-in samples in z-space*, define a
synthetic target = that Gaussian's logpdf (no simulator, no bijector — it lives natively in z),
and run the **identical** sampler mechanics on it: same chains, same budget, and critically the
**same qz** as the real run (whatever the real run used — the MAP-centered isotropic ball in the
T1-era runs, the typical-set-init ball in T21/T26 — NOT the fitted clone covariance; qz seeds the
init positions, the initial inverse mass matrix, and the adaptation anchor, so reusing it is what
makes the pipeline "identical"). Reference:
`build_clone.py` → `run_t1_clone.py` → `report_t1.py`.

The clone strips away everything except the *second-moment geometry as the sampler sees it*. The
gap between real min-ESS and clone min-ESS is the load-bearing number:

- **Gap 1–2×** → you are shape-limited; the difficulty is the covariance itself, and you are
  essentially done. (Route-A-cured carousel: real 436 vs clone-in-same-coords 531 = 1.22×, T26.)
- **Gap ~5–10×** → roughly one disease on top of Gaussian geometry. (Carousel new arm at
  typical-set init: 7.2× below its clone, T21 — a single funnel.)
- **Gap 20–70×+** → stacked diseases. (Carousel old arm 70× below clone, T21 — curved valley
  *and* funnel; sys60 self-consistent pairing ~130× below clone, T1.)

Blind spots: the clone is built from a run you *believe* converged — if it was secretly
unconverged the covariance (hence the clone's difficulty) is wrong, so log the source run's max
R̂ / min-ESS (T1 mitigation). It is also Gaussian by construction, hence blind to non-Gaussian
tails — that is intentional; it is the control. A clone that is *itself* slow is not a bug: it is
the positive control proving the pipeline produces slow clones when geometry warrants (vela clone
764–2237, seed-matched to real, T1).

### (b) D3 mode-vs-typical gap — init health, zero extra compute

Compute `logp(z_best) − median(per-sample logp of the chains)`. A converged mode should sit
**above** the typical set by ≈ +dim/2 (Gaussian ballpark). Strongly **negative** means the init
(MAP) is *below* its own posterior samples — a bad start, regardless of what the optimizer
trajectory or a local-curvature certificate says. Reference: `t18_map_quality.py` (D3 is the
workhorse of its trio), `t18b_logp_gap.py`.

In the carousel this caught **all three** bad MAPs at zero extra compute: mode-minus-median logp
= −5.3 / −5.5 / −4.8 nats, with 441–475 of 512 posterior draws *beating* the "mode" (T18b) —
even though a Newton-decrement certificate (D2) passed one of them as a sharp local max. Use the
**logp** form (coordinate-consistent), not just the loglike form; the log reports both and they
fired identically (T18b amendment).

### (c) Seed band — 3 seeds, is the geometry stationary or is this noise?

Run the standard config on 3–4 seeds and report the *full set* of min-ESS / max-R̂, not a
summary. Reference: `run_t0_seed_variance.py` (`--min-seeds`).

- **Huge spread** (order of magnitude) → init / adaptation noise dominates; a single number is
  not thresholdable and any clone bar built on it is unsafe. Carousel new arm at MAP init spread
  17–65 (T16) collapsed to a tight 139.9/144/144 once the init was moved to the typical set (T21)
  — the spread *was* the disease.
- **Tight but slow** → stationary geometry; the hardness is in the target, not the start. sys60
  band 11.2–15.3 (ratio 1.37, T0) — tight and slow, pointing straight at a stationary property.

### Decision flow

```
slow/weird run
   │
   ├─ D3 strongly negative? ───────────► init/MAP disease (Disease ii). Fix init FIRST;
   │                                      everything downstream is diagnosis of a bad start.
   │
   ├─ seed band spans ~10×? ───────────► init/adaptation noise. Stabilize init, re-measure.
   │
   └─ clone gap:
        1–2×   ► shape-limited. Done (or reparameterize only if you want the shape itself easier).
        5–10×  ► one disease. Localize with the instrument catalog (funnel? curved valley?).
        20–70× ► stacked. Peel them: init, then numerics, then coordinates, then funnel.
```

Two same-class fixes that both fail → **stop**, list the assumptions you have not tested, and
design a *diagnostic* that isolates the cause rather than another fix that assumes it
(`method-discipline.md` §6; this rule exists because the prior history of this exact question
rabbit-holed, O6).

---

## 3. Instrument catalog

Each instrument: **what it measures / cost / how to make it / how to read it (the plot) / what we
saw / blind spots / other shapes**. The "how to read it" is the heart. The "other shapes" note is
mandatory and load-bearing: a shape not listed under "what we saw" is a candidate **new pathology
class** — characterize it and report it; do NOT map it onto our catalog or apply our fixes blind.

### 3.1 xi-vs-coordinate hexbin (where in parameter space do the energy errors live?)

- **Measures:** per-step energy-error proxy `xi` against a suspect coordinate (e.g. `z_Rs`),
  pooled over all steps. `xi = energy_change²/(dim·desired_energy_var)+1e-8` (convention traced
  to `mclmc.py:334`; results-phase is sentinel-free — verify before reading, T9).
- **Cost:** zero extra GPU (uses saved chains). **How:** `t23_momentum_gpu.py` produces the
  columns; `t23_t24_analyze.py` draws the hexbin. Plot xi-vs-coordinate over ALL steps *first*
  (plots before metrics — a registered gate, T23).
- **How to read it:** healthy = a structureless cloud, xi low everywhere, no dependence on the
  coordinate. A **funnel** = the `xi>threshold` mass hugging one edge of the coordinate's range
  (a hockey-stick), i.e. the big energy errors are all generated at one *place*.
- **What we saw:** carousel new arm — the xi>10 mass pinned at the **low-`Rs`** edge
  (spike-median signed `z_Rs` +1.86, 5–95% [+1.71,+2.17]; calm +3.16 [+2.35,+5.11]), while the
  region actually pressed against the `Rs=100` prior wall was *quiet* (T23). That inversion
  exonerated the prior wall and located a funnel neck *inside* the support. Old arm: a V-shaped
  hexbin, spikes at *both* ends of the visited range (T23).
- **Blind spots:** a median/quantile summary of "where the spikes are" is blind to a minority
  wall-event sub-population — read the full hexbin and the ECDFs, not the summary (T23 blind-spot
  note). The hexbin is 1-D-in-the-chosen-coordinate; a funnel in a rotated combination will not
  show against any single axis.
- **Other shapes:** xi mass concentrated at a *sharp bound* (true wall reflections), spread
  uniformly (diffuse — points away from a localized geometric cause, toward integrator/momentum
  accounting), or clustered at an *interior* locus in a different coordinate than you guessed —
  each is a distinct story to characterize before naming.

### 3.2 xi tail statistics, real vs clone (is the heavy tail a real-posterior property?)

- **Measures:** `frac(xi>10)`, `p99(xi)`, `max(xi)` for the real run and its matched clone.
- **Cost:** zero extra GPU. **How:** saved chains; `t23_t24_analyze.py`.
- **How to read it:** the *clone* sets the Gaussian baseline. Real ≫ clone means the tail is a
  property of the true posterior's stationary geometry, not of the sampler.
- **What we saw:** carousel real `frac(xi>10)` 0.065–0.17 with p99 ~1200–1900, max 5e4–7e4; the
  matched clones sat at frac ~0.02, p99 ~19, max ~230 (T21). The tail co-occurred with *all*
  remaining hardness (7–70× clone gaps). It was present from step one at typical-set init ⇒
  stationary, not init/adaptation/numerics (T21 synthesis). Route A then drove it to clone level
  (frac 0.0198, T26) — a causal confirmation.
- **Blind spots:** a fraction-above-threshold conflates a slightly-heavier bulk with a few
  catastrophic events. Route B had a *tiny* frac(xi>10) (0.0036) but rare max 8.4e4–2.7e5 blowups
  (T26) — the fraction looked great while the run was worse. Always report p99 and max alongside.
- **Other shapes:** a heavy tail that the clone *also* has (then it is Gaussian-geometry/integrator
  physics, not a disease — see the T24 clone rho note), or a tail that *vanishes* under a numerics
  fix (then numerics contributed — flag loudly; in the carousel it did NOT vanish, consistent
  with T22).

### 3.3 stability-number ECDF: `eps·sqrt(curvature-along-motion)` (spike vs calm)

- **Measures:** the dimensionless integrator-stability number per step, `eps_t·sqrt(max(c_t,0))`,
  where `c_t = Δz'(−H)Δz / (Δz'Σ⁻¹Δz)` is the curvature along the actual direction of motion (HVP
  along the chord; Σ = the sampler's saved preconditioning metric). Compare the spike population
  (xi>10) against calm (xi<median).
- **Cost:** ~one HVP/step; conv float64, chunk ≤16 (`t23_momentum_gpu.py`, `t24_census_gpu.py`).
- **How to read it:** healthy = the whole distribution well below 1 (the integrator is stable).
  A funnel/stiff-encounter disease = the spike population's ECDF **crosses O(1)** while calm does
  not — that is the direct statement that the adapted `eps` is too large for the terrain the
  spikes sit on.
- **What we saw:** carousel spike stability-number median 1.85 / p90 4.13 (new; old 1.46 / 3.14)
  vs calm 0.69 / 0.57 (T23). This descriptive column was *decisive* where the registered ratio
  metric was ambiguous (median `c_t` spike/calm was 6–7×, a "gray zone" between the <2 falsifier
  and the ≥10 bar — the median bar was too crude because the spike set includes marginal xi~10–50
  events that dilute it, T23).
- **Blind spots:** `c_t` is evaluated at the step *endpoint*; a spike generated by curvature
  *varying inside* a step (a third-derivative event) is invisible to it and lands in the "H0"
  category by construction (T23 blind-spot 2). The chord momentum proxy hides within-step
  reversals (T23 blind-spot 3).
- **Other shapes:** stability number high *everywhere* (global step-size mismatch, not a localized
  neck), or spikes with stability number *near calm* (then the energy errors are not local-curvature
  driven — the carrier is elsewhere; do not assume a funnel).

### 3.4 turn-angle histograms (reversals = reflections/bounces)

- **Measures:** the cosine of the turn angle between successive displacement vectors, spike vs
  calm.
- **Cost:** zero extra GPU. **How:** `t23_t24_analyze.py` (C4 column).
- **How to read it:** forward motion (a healthy diffusive traverse) piles up near cos ≈ +1. A
  **reflection/bounce** piles up near cos ≈ −1 — the chain reverses direction.
- **What we saw:** carousel spikes had strong reversal mass at cos ≈ −1 on both incoming and
  outgoing legs (new ~10× calm density at −1; old ~3.4×), while calm was forward (T23). Combined
  with the whitened reflection axis being Rs-dominated and the Rs momentum component flipping sign
  in **100% of reversals both arms** (T23 orientation addendum), this said the chain *rattles*
  across a narrowing channel rather than reflecting off a one-sided end-wall.
- **Blind spots:** a bounce that completes *inside* one step shrinks |Δz| and moves the endpoint,
  so the chord-based angle can miss it (T23 blind-spot 3). Turn angle alone does not say *which*
  coordinate is reflecting — pair it with the whitened reflection-axis loadings.
- **Other shapes:** reversals with a *one-sided* entry direction (a genuine end-wall, unlike the
  carousel's 50/50 rattling), or reversal mass with *no* stability-number excess (a different
  reflection mechanism) — report the geometry, don't assume "funnel".

### 3.5 C3 eigenbasis loadings (which named direction is the chain travelling?)

- **Measures:** the loadings of the motion direction `u_t` (and of a stiff/soft eigenvector) on
  the fixed eigenbasis of `−H(z_ref)`, per named parameter.
- **Cost:** one Hessian at a reference point + projections. **How:** `t23_t24_analyze.py` (C3),
  `e1_fisher_survey.py` for the Fisher/GN survey version.
- **How to read it:** tells you whether the sampler is stuck along an *axis-aligned* coordinate
  (nameable directly) or a *rotated combination* (a degeneracy surface). A large loading of the
  softest eigenvector on a single physical parameter names the slow coordinate.
- **What we saw:** carousel — both spike and calm motion loaded ~0.95 on the **softest**
  eigenvector, i.e. the chain always travels the valley floor; the softest whitened eigenvector was
  **pure Rs** (loading 1.00, both arms) — so after preconditioning everything else is O(1) and Rs
  is the slow coordinate (T23, orientation addendum). sys60 — the stiff GN family was a rotated
  combination: EPL γ (0.59), shear γ2 (0.38), src center_x (0.32), γ1 (0.29), mass e1 (0.26), e2
  (0.17), θ_E (0.14) — the classic slope–shear–source-position degeneracy, invisible to
  cornerplots because it is rotated (E1/T6).
- **Blind spots:** loadings are taken against the eigenbasis at *one* reference point; if the
  stiff subspace *rotates* along the ridge, a single-point basis is only locally meaningful (see
  the encounter-census and the E1 rotation measurement, §3.6, 3.11).
- **Other shapes:** motion loading heavily on a *stiff* eigenvector (not soft) would be a different
  regime; a stiff subspace that rotates fast point-to-point (sys60 bends at 0.2–0.7σ, T6) needs
  its own rotation-vs-separation measurement rather than a single loading vector.

### 3.6 per-parameter ESS bars vs clone, and the tuner-eps comparison (the global-tax readout)

- **Measures:** per-parameter ESS for the real run and its matched clone, plus the tuned `eps`
  (and `L`) of each. The **eps ratio (clone/real) is the global-tax readout**.
- **Cost:** zero extra GPU (arviz ESS + saved tuner outputs). **How:** the T23 Addendum-2
  decomposition — read `step_size`/`L` from the saved npz and per-param ESS via arviz.
- **How to read it:** a flat bar chart across parameters at ~the eps ratio means the disease acts
  **globally through the tuner**: the funnel neck makes the burn-in adaptation shrink `eps` for
  *everyone*, so every parameter slows by the same factor; a few parameters rising above that base
  are the ones *directly* on the degeneracy surface.
- **What we saw:** carousel new arm — tuned `eps` real 0.354 vs clone 1.223 = **3.45× suppression**;
  11/14 parameters sat at a broad base of median 3.74× (tracking the eps ratio, i.e. ESS ∝ eps),
  with an elevated trio {s4.beta 7.40×, Rs 7.06×, s5.center_y 6.11×} carrying only ~2× extra
  (T23 Addendum-2). Old arm: eps 0.133 vs clone 1.268 = 9.5× global, Rs/alpha_Rs 26×. So the
  funnel's *main* tax is global-via-tuner, and its direct blocking adds only ~2× on the degeneracy
  trio. This is also the measured answer to "just lower eps?": `eps` is *already* the adapted
  compromise, so the gap **is** mostly the eps suppression (T23 Addendum-2).
- **Blind spots:** this reads adaptation outcomes, not causes; a low eps could in principle come
  from a source other than a neck (rule that in with §3.1–3.4). ESS ∝ eps is an empirical
  regularity here, not a theorem.
- **Other shapes:** a *non-flat* bar chart with no global suppression (eps ~ clone but one
  parameter starved) points to a purely local, coordinate-specific problem rather than a
  tuner-mediated global one — a different fix (§5).

### 3.7 conditional-precision profile `λ(z)` with a reliability floor

- **Measures:** the diagonal conditional precision along the slow coordinate,
  `λ_Rs(z_Rs) = e_Rs'(−H)e_Rs`, binned over the visited range (and, cautiously, extrapolated below
  the explored floor).
- **Cost:** ~1 HVP/point (~250 total), conv float64. **How:** `t25_profile_gpu.py`.
- **How to read it:** healthy = roughly flat (the conditional width is constant along the
  coordinate). A funnel = a smooth **monotone decay/rise** of many-fold: the conditional narrows
  sharply at one end even though the marginal is wide. This profile is *the* object a
  variance-stabilizing reparameterization integrates.
- **What we saw:** carousel `λ_Rs` bin medians decayed ~exponentially across the visited range:
  3108 (z=2.02) → 1000 (2.64) → 361 (3.21) → 98 (3.87) → 23.8 (4.50) → 7.5 (5.21) → 2.2 (5.86),
  then HVP-noise-level (|0.01–0.14|, including *negatives* on the sigmoid plateau) — reliable-range
  variation ~1400× (T25). The Rs marginal is wide but its local conditional narrows sharply at low
  Rs (T23 orientation addendum): the exact signature of a marginal-vs-conditional funnel.
- **Blind spots:** frozen-others slices *below* the explored floor measure *slice* curvature, which
  overstates the true conditional along the adaptive valley floor — treat below-floor knots as an
  upper envelope (T25 named blind spot). Past the cliff inflection the slice curvature goes
  *negative* (T25) — these knots carry no width information and must be filtered (λ>1 reliability
  floor).
- **Other shapes:** a profile that is genuinely flat (the funnel is *not* 1-D-expressible in this
  coordinate — both 1-D routes are mis-aimed; STOP before building a transform, F-T25a), or a
  profile with *multiple* necks (a more complex reparameterization than a single monotone spline).

### 3.8 encounter census: `ρ(c,xi)`, spike widths, rates (does the chain meet localized encounters?)

- **Measures:** along contiguous chain segments, per-step curvature-along-motion `c_t` paired with
  `xi_t`; census statistics = spike rate (`c_t > 3× segment median`), max/median, spike width
  (consecutive steps above threshold), inter-spike spacing, and Spearman `ρ(c,xi)`.
- **Cost:** ~one HVP/step over segments (`t24_census_gpu.py`; sys60 render-space analogue
  `t10_spike_census.py`). **How:** contiguous results-phase segments per chain.
- **How to read it:** a **localized-encounter** disease = moderate-to-high `ρ(c,xi)`, narrow
  spikes (1–3 steps), and a spike *rate* comparable to the tail fraction — the chain episodically
  *enters* a stiff region, bounces for tens of steps, and exits. A *diffuse* elevation would show
  low ρ and no rate/width structure.
- **What we saw:** carousel pooled `ρ(c,xi)` 0.511 (old) / 0.539 (new), spike widths median 1 /
  p90 3, census rate vs frac(xi>10) within ~2× (new 1.74×, old 2.14× — a marginal miss) — episodic
  funnel-neck bounces (T24). sys60 on-ridge census: spike rate 11.4%, pooled max/median 263 (up to
  562), widths 1–3 with sustained 18/26-step wall encounters, ρ(λ1,xi_s)=0.41 (T10) — the chain
  *does* meet the comb teeth on-ridge.
- **Blind spots:** a segment Spearman is bulk-dominated and blind to a few extreme co-events — read
  it with the rate/width/spacing stats and overlay plots (T23 blind-spot 4). **Within-segment**
  correlation is the *wrong* instrument when `xi_s` sits on sustained plateaus: T14b's per-segment
  ρ(λ1,xi) missed both predictions (OLD −0.085, NEW 0.231) precisely because the smoothed-xi
  plateaus dominate — the *stratified cross-decile* contrast (T9) is the valid version. Also:
  ambient top curvature `λ1` did **not** move at carousel spikes (2.28e7 vs 2.29e7; 1.69e7 vs
  1.82e7, T24) — this is a `c_t`-along-motion field, *not* a `λ1`-spike field (unlike sys60).
- **Other shapes:** high ρ with *wide* spikes (a broad stiff basin, not a neck); a `λ1`-spike field
  where the ambient top eigenvalue *does* move (sys60's comb, §3.10 — a different disease); no
  spike structure at all with a heavy tail (trajectory-level energy accumulation, which closes
  per-step local probing and points to multi-step accounting).

### 3.9 D1/D2/D3 MAP-convergence trio (is the start actually on the mode?)

- **Measures, no ground truth needed:** D1 trajectory-slope (is best-so-far lp still improving in
  the final 10% of steps?); D2 Newton decrement `λ = sqrt(gᵀH⁻¹g)` at z_best (expected logp gain
  of a Newton step); D3 mode-vs-typical gap (§2b).
- **Cost:** D1/D3 ≈ free; D2 = one exact dim-Hessian. **How:** `t18_map_quality.py`,
  `t18b_logp_gap.py`.
- **How to read it — with the honest calibration the log forced:** **D3 is the workhorse** — it
  caught all three carousel bad MAPs (loglike gaps −6.7/−12.3/−6.0; logp gaps −5.3/−5.5/−4.8,
  T18b). **D1 is one-sided** — it *passes* bad MAPs that plateau (both carousel reference MAPs
  plateaued short). **D2 is one-sided the other way** — a D2 *FAIL* (large predicted Newton gain)
  cheaply confirms badness *before* sampling, but a D2 *PASS proves nothing global*: it passed a
  point that was a sharp local max yet still 5 nats below the typical set (T18b). Use the logp form
  of D3 (coordinate-consistent), reported alongside loglike.
- **What we saw:** carousel — "run MAP longer" was structurally useless: 5000 steps (10× the
  reference) bought +0.32 nats and moved z_best 0.63σ (T18); the mode is a *genuine* local maximum
  (−H positive definite, 441–475/512 draws above it, T18b) that traps both AdaBelief-with-64-particles
  and Newton locally.
- **Blind spots:** D2's local certificate is meaningless on a rugged landscape; on a
  gradient-noise-limited system the Newton certificate is itself noise-scale-local (T20 synthesis).
  All three are init diagnostics — they say nothing about stationary geometry.
- **Other shapes:** D3 *positive* but sampling still slow (init is fine; look downstream — carousel
  post-T21); D3 negative *and* D2 pass (a genuine sub-typical local max, as here — the fix is a
  better *init construction*, e.g. a typical-set draw, not more optimizer steps).

### 3.10 xi/λ1 comb scan — flank-crossing / subgrid aliasing (sys60's disease)

- **Measures:** the top GN/Fisher eigenvalue `λ1` (or a blob-window λ1) as a function of a measured
  sub-pixel displacement of a compact image feature, scanned across ~±2 px; the dominant spatial
  frequency of the resulting oscillation; and the same scan with the model rebuilt at higher
  supersampling.
- **Cost:** ~hundreds of Jacobians + renders. **How:** `t12_flank_crossing.py` (dial scan +
  supersample control + census cross-check); localization via `t11_spike_pixels.py`.
- **How to read it:** a **rendering-fidelity comb** = `λ1` vs measured displacement is a train of
  narrow teeth at the *supersample-subgrid* pitch, riding a smooth envelope; rebuilding at higher
  supersampling *collapses* the teeth while leaving the envelope. A *physical* magnification response
  would instead be carried by the smooth envelope (the blob flux term) and would be stable under
  supersampling.
- **What we saw:** sys60 — teeth at pitch ≈0.52 px = the ss=2 subgrid (recovered 1.90–1.91/px),
  peak-to-trough 30,645; at ss=4 the top tooth collapsed 6.0e7 → 4.2e4 (×1400), peak-to-trough
  30645 → 4.7 (T12). Census `ΔR²(phase|flux) = 0.563` confirmed the phase (aliasing) term carries
  the variance, not flux (magnification). The stiffness was localized (L=0.999) on a single
  ~4×4-px hotspot = the observed counter-image (peak 15.3 = ~75σ, 13% of the main peak, T11).
- **Blind spots:** the registered 1/px Fourier metric **failed** (0.64/0.24/2.22 vs ≥5) because it
  was aimed at the *pixel* grid, not the *supersample* grid — a metric pinned to the wrong frequency
  reports a null while the comb is plainly visible (T12; a canonical "the plot wins" case). The
  ss4/ss2 fourier-amp ratio landed in a "mixed" band for the same reason; the ×1400 collapse and the
  log-amplitude ratio 0.150 were the decisive reads.
- **Other shapes:** a comb whose teeth *survive* supersampling (then it carries physical
  pixel-integration information, not an artifact — reparameterization can't remove it); a smooth
  envelope with no teeth (ordinary caustic-proximity stiffness); teeth at a pitch matching neither
  grid (characterize before naming).

### 3.11 C3-family Fisher/GN eigen survey + rotation-vs-separation (curved-valley geometry)

- **Measures:** across typical-set points and same-chain lag pairs, the local GN metric
  `M(z)=JᵀWJ`, its top eigenvalue(s), the named stiff family, and the principal angles between stiff
  subspaces vs z-separation (rotation), in both z and θ coordinates.
- **Cost:** thousands of forward evals + a few dim-Hessians. **How:** `e1_fisher_survey.py`.
- **How to read it:** a **curved stiff valley** = the top GN eigenvalue *breathes* (varies
  many-fold point-to-point) and its eigenvector *rotates* along the ridge; the z-vs-θ rotation ratio
  says whether the curvature is manufactured by the bijector (ratio ≫1) or is intrinsic physics
  (ratio ≈1).
- **What we saw:** sys60 — stiffest GN eigenvalue ×22 across 32 typical-set points (ranks 2–3 only
  ×1.9); local spectra = the global precision at all ranks *except* the top 1–2 (10–200× above);
  bending turns on at 0.2–0.7σ (15–70° over a 2σ traverse), and the z=θ rotation ratio was 1.00 at
  all scales ⇒ **bijector innocent, curvature intrinsic** (T6/T7). One razor-thin breathing,
  slowly-bending crease in an otherwise-Gaussian 22-dim landscape; a single global `eps` must fit
  the worst wall anywhere on the ridge (~√22 ≈ 4.7× too small for the median region).
- **Blind spots:** point Hessians are honestly *unrepresentative* when the smooth Hessian field
  varies ~100× over ~0.07σ (this re-explains the human's O3 "rough, unstable Hessians" *without*
  micro-roughness — the field is smooth but fast-varying, T3/C-2 withdrawal). A breathing/bending
  attribution via `eᵀMe = Σλ_k(v_k·e)²` is entangled off-ridge (v1/v2 identity swaps at peaks); the
  ≥70%-breathing sub-claim was too coarse (T8).
- **Other shapes:** a stiff subspace that rotates *fast* at h*-scale (sys60's did not — 5° vs the
  predicted 15°, T6), a bijector-manufactured curvature (ratio ≫1 — would make reparameterization
  the immediate fix), or breathing with no bending (a different geometry).

### 3.12 conditional-vs-marginal multi-scale transects + D2(h) roughness ladder

- **Measures:** 1-D log-density transects through a typical-set point along random / stiff / soft /
  worst-ESS directions, at geometrically decreasing spacings h; second-difference curvature D2(h);
  and the *marginal-vs-conditional* width ratio via a clone null overlaid on each transect.
- **Cost:** hundreds–thousands of likelihood evals, no sampling. **How:** `t3_transects.py`
  (`replot_macro_with_clone.py` for the null overlay).
- **How to read it — two readings from one instrument.** (1) *Roughness (H1):* D2(h) on log axes —
  a flat plateau through the sampler band [ε/100, ε] = smooth; a blow-up below some h* = roughness
  (the float32 arm is the positive control and *must* blow up). (2) *Curved valley:* overlay the
  matched-clone transect as the Gaussian null; the nat-drop magnitude *is* the null, so **deviation
  from the dashed null is the finding** — a real transect sitting *above* the null at ±2σ (locally
  stiffer, globally shallower) is the line re-entering a curved ridge.
- **What we saw:** sys60 — float64 D2(h) flat to ≪ε/100 on all 7 directions (FD-AD 1e-9–5e-8);
  float32 control blew up at h≈1.2e-5 (FD-AD floor 2.7e-3) ⇒ **H1 roughness dead, C-2 withdrawn**
  (T3). The positive finding: local D2 ~5e8 vs σ-scale average ~5e6 (~100×), curvature correlation
  length ~0.07σ ≈ 3× per-step move; macro transects dropped 15,000–30,000 nats over ±2σ and
  `random_2` was **non-monotonic** (cliff to −31,500 at +1.5σ, then recovery — a curved ridge). The
  clone-null width ratios were 124/126/184 along random directions, 25.7/34.6 along worst-ESS axes,
  1 along eigendirections (T3) ⇒ the structure lives in *rotated joint* directions, invisible to
  cornerplots.
- **Blind spots:** the summary table's `h_D2_dev` flags were **macro-plateau deviations (averaging),
  not small-h blow-up** — read the plots, not the table (T3 caveat; a "plots before metrics" trap).
  Transects sample a handful of directions out of ~22–32; roughness confined to unvisited directions
  escapes (include the worst-ESS direction). A means-of-the-cloud evaluation is off-support for a
  curved posterior — never evaluate logp at cloud means (T14/T15b).
- **Other shapes:** a genuine blow-up *inside* the sampler band (real micro-roughness — the carousel
  had value texture but it sat *below* the band, T20; sys60 had none); a plateau that is flat *and*
  the clone null matches the real transect (truly Gaussian along that line).

### 3.13 FD-vs-AD gradient ladder (numerics injection)

- **Measures:** relative disagreement between a central finite-difference gradient and the autodiff
  gradient, at the error-minimizing h, along several directions; with a float32 arm as positive
  control.
- **Cost:** part of `t3_transects.py`; run before trusting any optimizer/HMC on a system with
  a linear-solve or nonstandard-precision layer.
- **How to read it:** a clean smooth double-precision system floors at ~1e-8; `fd_ad_min ≳ 1e-5`
  marks a value-noise-injecting layer. The float32 control must fire (validates the detector).
- **What we saw:** carousel FD-AD 1.7e-5–3.0e-3 on 7/7 directions vs sys60 baseline 1.4e-9–5.0e-8 —
  a 1,000–100,000× gap (T20); switching `conv_precision` to float64 collapsed it to 3.3e-9–1.3e-6
  (5/7 inside the sys60-clean band, T22). The injector was the **float32 PSF convolution**, and the
  lstsq layer was fully exonerated (cond(G) ~3e3, branchless clean-VJP solve, `simulator.py:90`).
- **Blind spots — the big one:** `fd_ad` measures `|FD − AD|`; it **cannot say which leg is
  corrupted**. The log first attributed the gap to "noisy gradients through lstsq"; a user challenge
  forced the correction (T20 correction / T22): the *value* noise corrupts the *FD* leg (differencing
  a bumpy function), while the AD gradient was likely clean all along. So a large `fd_ad` is a flag,
  not a diagnosis of the gradient the sampler consumes — attribute the corrupted leg with an
  independent numerics discriminator (one config field, T22).
- **Other shapes:** `fd_ad` large *and* the sampler visibly degraded (then the gradient itself may
  be corrupted — but prove it, don't assume); `fd_ad` large but value-noise crossover *below* the
  sampler band (harmless to sampling, fatal to optimizers whose signal shrinks to zero at
  stationarity — carousel, T20).

### 3.14 R̂ + marginal comparison for truncation bias

- **Measures:** rank-R̂ per parameter, plus a *pre/post marginal overlay* of any reparameterized run
  against baseline.
- **Cost:** zero extra GPU. **How:** standard diagnostics; overlay Route-A vs baseline marginals.
- **How to read it:** clean R̂ (≈1.01–1.03) on a reparameterized run whose marginal *opens* a
  previously-unexplored tail means the baseline runs were **truncating** that tail — a reflecting
  boundary, not a real edge. The reparameterization did not change the posterior (proven by family
  gates, §6); it let the sampler reach mass that was always there.
- **What we saw:** carousel Route A — Rs marginal p01 79.2, min 71.9, vs baseline chains that never
  went below Rs 86.3, at clean R̂ 1.012–1.028 (T26). Every baseline-coordinate run — *including the
  user's 10k reference* — truncated the low-Rs tail; the funnel neck acted as a reflecting boundary.
  A funnel fix can therefore **UNBIAS marginals, not just speed them up** — so downstream Rs (and
  correlated-parameter) summaries are biased until re-derived.
- **Blind spots:** R̂ alone will *not* flag this — the truncated runs had structured R̂ but the bias
  is only visible against the un-truncated marginal. All sampling metrics are blind to a silently
  *changed* posterior; that is what the family/identity gates (§6) are for, not R̂.
- **Other shapes:** a reparameterization that *shifts* a marginal without a family-gate pass (then
  you changed the posterior — a bug or a prior change, not an unbiasing); an opened tail that makes
  the physics *implausible* (then reconsider the prior, not just the coordinate).

---

## 4. Disease catalog (validated classes only)

Each entry: **Signature / Discriminating test / Validated fix / Receipts.** These are the classes
we actually reached ground on. A run that does not match one of these is not thereby "one of these"
— see every §3 "other shapes".

### (i) Computed-likelihood-stiffer-than-intended (the self-consistency / subgrid-comb trap)

- **Signature:** near-Gaussian marginals, a huge clone gap (~130×, T1), a stiffness comb at the
  supersample-subgrid pitch localized on a compact image feature (T11/T12), the chain in sustained
  energy-error crisis with the posterior *locked* sub-pixel onto the comb (T14b: x_c range 0.054 px,
  ON teeth). The pathology appears only in the **self-consistent** pairing — synthetic data rendered
  and fit at the *same* low fidelity.
- **Discriminating test:** the fidelity 2×2 payoff test (T13′) — regenerate the data at *higher*
  fidelity (ss=128 after a convergence ladder) and run the standard config with the low- and
  high-fidelity models. If accurate data restores fast sampling, the disease was the
  self-consistent aliased pair, not the physics. Corroborate with the ss4 comb-collapse (T12) and
  the own-chain census showing the accurate-data posterior never meets a tooth (T14b: NEW max/med
  2.75 spike rate 0.0% vs OLD 25.4 / 6.2%; experienced Hessian 428× calmer; ξ at target).
- **Validated fix:** render the data at higher fidelity than the model is fit at (or raise model
  supersampling): {d′, ss2} min-ESS 1031–1253 and {d′, ss4} 889–1164, both ~80–110× the old
  self-consistent band (11–15) and within ~1.5–2× of the clone (T13′). Caveat: the ss2-on-accurate-data
  arm is *fast but biased* (13.9σ misfit) — fidelity is a modeling choice to grade, not a sampler
  trick.
- **Receipts:** T1, T3, T6–T12, T13′, T14/T14b. **Scope:** this explains *synthetic self-consistent*
  slowness; it does NOT explain real-data slowness — the carousel is real, so its hardness needed a
  different mechanism (C-4 scope amendment, T13′).

### (ii) Init trap: under-converged MAP + genuine local maximum

- **Signature:** a cornerplot "second mode" that is actually the MAP-init region; chains parked
  several σ out for ~1e4 steps then migrating (T15); D3 mode-vs-typical gap strongly negative with
  most draws beating the "mode" (T18b).
- **Discriminating test:** D3 (§3.9) catches it at zero compute; confirm the "second mode" is the
  init by comparing the suspect cluster to z_best (|z_best−basin| 1.84σ vs |z_best−bulk| 6.19σ, T15).
  Distinguish "run MAP longer" (useless — 5000 steps bought 0.32 nats, T18) from a genuine local max
  (−H PD, T18b) with a Newton certificate *plus* D3.
- **Validated fix:** typical-set init — start at the **median-logp posterior draw**, not any
  optimizer output. Carousel: the entire displaced-chain phenomenology vanished (0/9 chains, T21),
  new-arm seed spread collapsed 17–65 → 139.9/144/144. (This does *not* cure stationary hardness — it
  peels the init disease off the stack.)
- **Receipts:** T15–T18, T21.

### (iii) Numerics injection (float32 convolution)

- **Signature:** FD-vs-AD gradient disagreement 1e-5–1e-3 where a clean system floors at 1e-8
  (T20); value noise ~1e-4 nats.
- **Discriminating test:** the one-config-field discriminator — rerun with `conv_precision=float64`
  and watch `fd_ad_min` collapse (5.3e-5..3.0e-3 → 3.3e-9..1.3e-6, T22). This also *reassigns* the
  corrupted leg: the noise was in FD, not the AD gradient the sampler uses.
- **Validated fix:** `conv_precision=float64` for inference-grade runs (memory cost ~2× conv
  buffers). **Lesson attached:** the original attribution ("gradient noise through lstsq") was wrong
  on *both* nouns; a **user challenge** triggered the correction and T22 then went 2/2 — challenge
  your own attribution before writing a mechanism claim. Note the numerics never actually affected
  *sampling* here (the value-noise crossover sat below the sampler band, T20/T22) — it corrupted the
  optimizer and every FD-based check.
- **Receipts:** T20, T20-correction, T22.

### (iv) Curved-valley parameterization

- **Signature:** a bent stiff valley in the sampling coordinates that a global mass matrix cannot
  align with; R̂ 1.4–1.7, min-ESS ~15; a clone gap multiplier of ~10× *on top of* other diseases
  (old-arm 70× vs new-arm 7.2× below clone, T21; measured eps cost old/new 0.133/0.354 = 2.7×,
  T23 Addendum-2).
- **Discriminating test:** compare parameterizations directly — ridge-spine sagitta/transverse in
  each coordinate system (old (Rs,alpha_Rs) 1.71 vs new (Rs,θ_E) 0.436 = ~4× straighter, T15); or
  the Fisher rotation-vs-separation survey (§3.11). Confirm the reparameterization does not merely
  *move* the posterior via a prior change (T15 found real prior leakage: src5 β 6.3σ — so check
  pushforward overlap before crediting geometry).
- **Validated fix:** the straighter physical coordinates (NFW_ELLIPSE_EINSTEIN over
  NFW_ELLIPSE) — but only after confirming it is a coordinate change, not a prior change.
- **Receipts:** T15, T21, E1/T6.

### (v) Marginal-vs-conditional funnel + tuner suppression

- **Signature:** a wide marginal whose *local conditional* narrows sharply at one end (T23
  addendum); xi tail = a census of reflections at that neck (§3.1–3.4, T23/T24); the tuner responds
  by suppressing `eps` **globally**, so all parameters slow by the eps ratio with a small extra tax
  on the degeneracy trio (§3.6, T23 Addendum-2).
- **Discriminating test:** measure the conditional-precision profile λ(z) (§3.7, T25) and confirm
  it is a strong monotone variation over the visited range with a reliability floor (1422×, T25).
- **Validated fix:** a **measured variance-stabilizing transform** — Route A,
  `u(z)=∫sqrt(λ̂(z'))dz'` as a monotone spline bijector (`reparam_bijector.py`, `t25_transforms.py`).
  The **causal acceptance test** is the eps-recovery prediction: if the funnel drives the tuner, a
  successful fix makes tuned `eps` recover to ~1.0–1.2. Route A: eps 0.354 → 1.167 (predicted
  1.0–1.2, HIT), clone gap 7.2× → 1.22×, frac(xi>10) 0.08 → 0.02 = clone level (T26). It also
  **unbiased** the Rs marginal (opened the truncated low-Rs tail, §3.14).
  - **Route B cautionary tale:** an observable-anchored coordinate (slope-at-fixed-arc-radius) whose
    observable *saturates at the prior-box edge* over-compresses the plateau into a near-wall and
    fails the battery (eps 0.372 < 0.5; rare catastrophic max 8.4e4–2.7e5, T26). It was flagged
    *before* the run by the **mandatory pre-battery gate**: the Jacobian-vs-profile cross-check
    (ds/dz vs sqrt(λ̂) varied 55×, F-T25c fired, T25) — an observable coordinate is trustworthy only
    *after* that check passes.
  - **Family-preservation gates** (hard, §6) proved both routes sample the same NFW family.
  - **Gap-based acceptance bars, not absolute-ESS bars:** the min-ESS 500 bar was mis-derived
    because the clone *refit in the new coordinates* itself floors at 531 — a coordinate change moves
    the clone's own difficulty, so acceptance must be gap-vs-clone-in-same-coordinates (Route A gap
    1.22×), not an absolute ESS (T26 lesson).
- **Receipts:** T23–T26.

---

## 5. The fix ladder

Escalate in this order. Each rung is cheaper, more transferable, and less likely to add its own
failure mode than the next. Sampler knobs are **last** and mostly do not help (the measured "just
lower eps?" answer, T23 Addendum-2: `eps` is already the adapted compromise, so a global-tax funnel
gap *is* the eps suppression — sliding along the adapted trade-off curve buys little).

1. **Coordinates.**
   - *Default:* a **measured 1-D variance-stabilizing** transform from the conditional-precision
     profile (Route A, §4v). Cheap, mechanical, provably family-preserving, and it cured the
     carousel funnel outright (T26).
   - *Observable-anchored only after the Jacobian-vs-profile check* (§4v Route B) — physics-anchored
     coordinates are appealing but fail when the observable saturates at a prior edge; the
     cross-check is mandatory before the battery.
   - *Escalation if 1-D is insufficient:* 2-D / low-rank measured transform, then a learned flow,
     then Riemannian / local-metric methods **last** — global-structure tools (flows, tempering)
     address diseases these posteriors do not have and add their own failure modes (the human's O4;
     `anti-patterns.md` AP-6).
2. **Priors.**
   - Impose *physics* priors through a Jacobian, **decoupled from the sampling coordinates** — e.g.
     a prior on (M200, c) applied via pushforward while sampling in the variance-stabilized
     coordinate (the T27 direction). This keeps the geometry fix and the physics separate.
   - Prefer **soft tails over hard bounds** on weakly-constrained parameters — a hard `Rs=100` box
     did not itself cause the funnel (it was *exonerated*, T23), but hard bounds interact badly with
     reflecting necks and truncate tails (§3.14).
3. **Sampler knobs — LAST.** Only after coordinates and priors. They slide along the adapted
   trade-off curve; a lower `desired_energy_variance` (smaller eps) or an Rs clip is a candidate
   *intervention test* (watch the clone gap), not a cure. If a knob "helps", state how it could be
   an artifact before accepting it (`method-discipline.md` §6).

---

## 6. Verification discipline

Any reparameterization is a claim that you changed the *coordinates*, not the *posterior*. Prove
it before trusting any downstream number.

- **Family / identity gates (hard, block the battery on failure).** For a remap of one coordinate:
  - bijector **round-trip** |x − x''| — carousel achieved 1.4e-14 (T25);
  - **prior identity** in θ-space (logp identity) — achieved 0.0 (T25);
  - **rendered-likelihood identity** (max|image diff| on random θ points) — achieved ≤8.7e-11 (T25);
  - **mapped-init identity** (the typical-set init maps to the same physical point) — achieved 0.0
    (T26).
  These are *deterministic identities* and must be tested to solver tolerance, not with a loose
  percentage band on a stochastic sample (`method-discipline.md` §2).
- **Same-coordinates clone refits.** Acceptance is *gap vs a clone refit in the new coordinates*,
  because a coordinate change moves the clone's own difficulty (min-ESS bar mis-derivation, T26).
- **Pre-registered acceptance bars with derivations.** State the bars and *why those numbers*
  before the run (T25/T26 battery: eps recovery ≥0.8 derived from the clone eps 1.22 and the log-gap;
  frac(xi>10) ≤0.04 as the geometric midpoint; clone gap ≤2×). A bar you cannot derive is not a test.
- **The honest-miss ledger habit.** Every results entry in the log carries an "honest misses"
  block (mis-derived bars, fired-then-corrected windows, engineering false starts, predictions
  falsified). Keep it — it is what lets the next agent trust the rest of the entry.

---

## 7. Registered-metric design lessons (meta-lessons, each with a receipt)

These are the ways our *instruments*, not our systems, fooled us. Every one cost a correction.

- **A wall/saturation statistic needs an absolute scale.** The `|z|`-quantile "wall" statistic
  false-positived on old-arm alpha_Rs (q=0.889) at |z|≤0.2 — nowhere near bijector saturation
  (O(2–3)). Quantile-within-calm has no absolute saturation scale; add one (T23).
- **Falsifier windows for a PROFILE claim must span the full instrumented range, with a reliability
  floor.** F-T25a fired on a mis-registered [1.7,3] sub-window (3.11× < 5) inferred from a *different*
  instrument's contrast populations; the full profile varied ~1400×. Corrected to full-range/30×
  with a λ>1 floor *before* relaunch (T25 correction).
- **Never evaluate logp at cloud means.** Means of curved posteriors are off-support: the carousel
  basin looked +1284 nats *better* at the cloud mean but was 12 nats *worse* on-support per-sample
  (T15/T15b); the T14 dial "falsifier" fired for a design reason because the dial exited the typical
  set within 0.02 px (T14b post-mortem). Use per-sample distributions.
- **`fd_ad` cannot say which leg is corrupted.** `|FD − AD|` flags value noise but not its
  location; the log's first attribution blamed the gradient (AD leg) when the noise was in FD
  (T20-correction / T22). Attribute the corrupted leg with an independent discriminator.
- **Clone sanity criteria must not bundle amplitude with correlation.** The clone "sanity" clause
  bundled positional cleanliness with a c–xi correlation bar and tripped on universal integrator
  physics (a Gaussian *should* show c–xi correlation at trivial amplitude — clone rho 0.495, ratio
  ~2; real max c ~1740 vs clone ~15). Separate the two (T23/T24).
- **Within-segment correlation is the wrong instrument when xi sits on plateaus.** T14b's
  per-segment ρ(λ1,xi) missed both predictions because smoothed-xi plateaus dominate; the
  *stratified cross-decile* contrast (T9) is the valid one (T14b).
- **A metric pinned to the wrong frequency reports a null while the effect is plainly visible.**
  The registered 1/px Fourier metric failed on sys60's comb because the real grid is the
  *supersample* grid at 1.9/px (T12) — the plot won.
- **Read the plot, not the summary-table flag.** T3's `h_D2_dev` table flags were macro-averaging
  deviations, not small-h roughness; the D2(h) *plot* was smooth through the sampler band (T3).
- **Acceptance bars must be gap-based, not absolute.** A coordinate change moves the clone's own
  floor; an absolute-ESS bar bakes in a false clone-invariance assumption (T26).
- **Structure can be confirmed while magnitudes are inverted.** The gap decomposition confirmed the
  funnel's *structure* (global-via-tuner + small degeneracy-trio extra) while the *predicted split
  magnitudes were wrong* (T23 Addendum-2) — a right-shape/wrong-number outcome is still a partial
  miss to log, per `method-discipline.md` §6 ("movement in the desired direction is not success").

---

*End. Status: proposed (UNCERTIFIED) — every number above is a `proposed (UNCERTIFIED)` finding in
`docs/logs/why-hard-to-sample.md`; the human is the grader. Reference implementations in
`experiments/why_hard_to_sample/`.*
