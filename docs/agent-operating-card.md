# GIGALens Agent Operating Card

> **Auto-injected into every session and subagent** by `.claude/hooks/inject-operating-card.sh`.
> This card is the always-in-context core: the *most-violated, most-consequential* rules in
> trigger → action form, plus the decision tables agents actually need mid-task. Depth lives in
> the linked docs (one hop away). **Rotation rule:** this card holds the head of the violation
> distribution — when a rule stops being violated, demote it to its depth doc; when a post-mortem
> surfaces a new failure mode, it earns at most one slot here. Edit the depth doc first, then
> distill here.

This is **scientific-research** work: the goal is a *defensible answer to a scientific
question*, not the shortest path to "something runs." Governing idea — Feynman's
[*Cargo Cult Science*](https://calteches.library.caltech.edu/51/2/CargoCult.htm): **you are the
easiest person to fool**, and "the first principle is that you must not fool yourself." Every
rule below is a specific instance of that principle; when a rule's letter and its spirit
conflict, the spirit wins.

If you spawn subagents, tune the model choice to the task. Don't use expensive models for rote tasks.
But make sure the model's ability is up to the task.

---

## Non-negotiables (trigger → required action)

1. **Before any consequential or expensive run** (Slurm job, long sampling run, anything
   entering the record or informing a decision) → write a **design checkpoint** to the area's
   lab-notebook log *before* running: cause hypothesis, predicted direction **and** order of
   magnitude, falsifier, and a **derived** threshold (a threshold you cannot derive is not a
   test). Then **stop for grader approval**. Use the `/pre-run-checklist` skill. A run without
   a prior checkpoint entry is illegitimate regardless of its result.

2. **After two failed fixes of the same class** → STOP editing. List the assumptions you have
   *not* tested; propose a **diagnostic** that isolates the cause, not another fix that assumes
   it. (Full rule: `docs/method-discipline.md` §6.)

3. **Before reporting any aggregate metric** → look at the plots (residuals, distributions,
   best/worst/median examples). **A number that "passes" while the plot disagrees is an open
   finding, not a pass** — resolve toward investigating, not toward the number.

4. **When a metric moves toward the target** → that is *not* success. Report target, current
   value, and gap in meaningful units; classify **structurally wrong vs. fine-tuning**
   (χ²/ν of 26 vs. 38 are both structurally wrong when the target is ~1 — hunt for a bug or
   mis-specification, not knobs). State how the improvement could be an **artifact** before
   accepting it.

5. **When you produce a result** → you may propose a verdict, marked **UNCERTIFIED**; you may
   never certify it. The grader (human, orchestrator, or the `rigor-grader` agent) inspects the
   **artifact** — the actual plot / code / numbers — never the producer's summary.

6. **When a model input is missing** (PSF, noise model, mask, units, pixel grid, priors,
   `n_max`) → it must **raise, never default**. Before trusting any run, check the printed
   **model card** (PSF / noise / grid / precision). *Real instance: a run silently modeled with
   `psf=None` because a path was wrong; χ²/ν≈1.25 hid it completely.*

7. **Before any strong claim about a model or posterior** → converged MCMC samples is required (this project's
   gold standard). Point estimates and variational surrogates can be badly biased. Convergence
   means the thresholds in the decision table below, on the **worst** parameter.

8. **When uncertain about a source, value, or step** → never fabricate a citation, path,
   number, or quote; read the primary source, not a summary; never swap in a silent
   approximation for a principled step. A blocked question beats a wrong assumption — **"I
   don't know" and "this points to a structural problem" are correct, encouraged answers.**

9. **When a claim is made, changed, or killed** → update the area's lab-notebook log
   (`docs/logs/`): scope what it does *and does not* cover, record negative results, withdraw
   wrong claims explicitly. The record is the source of truth; agent memory is not.

---

## Diagnostics decision table

Always evaluate the **worst** parameter: max(R̂), min(ESS) — never means. Depth and plot-reading
guidance: `docs/inference-diagnostics.md` (canonical for thresholds; keep this table in sync).

| Signal | Reading | Required first action |
|---|---|---|
| χ²/ν ≳ 3 (any stage) | Pathological: NaNs / frozen particles / model mismatch | Check NaNs + whether particles move; residual plot. Do **not** tune optimizer knobs. |
| χ²/ν in ~1.2–3 | Imperfect fit | Normalized-residual plot; look for structure (dipoles, point-in-ring, 4σ+ clumps). |
| χ²/ν ≈ 1 | Necessary, **not sufficient** | Check model card (PSF/noise/grid/precision); verify against ground truth where available — flexible sources absorb misspecification. |
| max R̂ > 1.1 | Not converged; claims invalid | Trace plots; check MAP/init convergence **before** touching sampler knobs (see AP-1). |
| max R̂ < 1.01 (and ideally rank-R̂) | Converged — necessary only | Still check min ESS and residuals. |
| min ESS ≈ n_chains | Not sampling | Trace plots: frozen vs. slow mixing. |
| Bad R̂/ESS but each mode looks well-sampled | Test for multimodality | Cornerplot 2-D contours; per-chain histograms; look for segregation or discrete hops in traces. If this is visible, invoke `/diagnose-sampling skill` |
| SVI/surrogate multiple σ from MCMC | Surrogate is poor | Trust the MCMC; don't reuse the surrogate for downstream claims. |

**Cornerplot legibility (agents):** never create or interpret a cornerplot larger than
**4×4 panels** — vision-input downsampling makes denser ones unreadable and past agents have
confidently misread them. Select ≤4 parameters (worst R̂ / worst ESS / parameters of interest)
and chunk. Reading guide: `docs/inference-diagnostics.md`. If handed an illegible plot, say so
and request chunks — do not guess.

For a full ordered walkthrough, invoke the `/diagnose-sampling` skill or dispatch the
`inference-diagnostician` agent.

## Anti-pattern headlines (full playbook: `docs/anti-patterns.md`)

- **AP-1 — Sampler-knobs-first on slow MCLMC.** When mixing is slow, do not reach for step
  size / L / integrator first. MCLMC is a very robust sampler, and the defaults are well-validated.
- **AP-2 — Trusting a good χ² as correct specification.** A flexible model fits a
  misspecified forward model to χ²/ν≈1 (the `psf=None` incident). Verify the model card and,
  where possible, parameter recovery.
- **AP-3 — Exotic explanations for poor sampling performance.** the exotic causes are *sometimes* real (see AP-4 for a genuine noise
  floor) but they are rarer than the mundane ones, are often very difficult to rule out, and each costs an expensive run to chase. 
  Look to the mundane knobs (initialization, preconditioning, model correctness) first.

---

## Mechanisms and depth docs

- **Skills:** `/pre-run-checklist` (writes the design checkpoint, stops for approval) ·
  `/diagnose-sampling` (ordered sampling-diagnosis workflow).
- **Agents:** `rigor-grader` (adversarial grading of a proposed claim against this discipline —
  cheap to run; use it before asking the human to certify) · `inference-diagnostician`
  (sampling diagnosis with the domain workflow baked in).
- **Depth (read before consequential work in the area):**
  - `AGENTS.md` — operating modes; the structural rules in full; the record.
  - `docs/method-discipline.md` — the general discipline in full (identifiability, match the
    test to the claim, pre-registration, metric blind spots).
  - `docs/project-standards.md` — domain standards: physicality, reproducibility, float64,
    failure modes.
  - `docs/inference-diagnostics.md` — what each diagnostic means and how to read the plots.
  - `docs/anti-patterns.md` — the negative-knowledge playbook (tempting-but-wrong moves).
  - `docs/env_setup.md` — canonical environment; how to run code.
  - `docs/logs/<area>.md` — the lab notebook for your area. Read it before starting; update it
    after any substantive step.
