# Agent Protocol & Rigor Control — GIGALens Inference and Modeling

This is a **scientific-research** project: the goal is a *defensible answer to a scientific question*, not the shortest path to "something runs." This file is the entry point for any agent — and the human — working here. Keep it short; depth lives in the linked docs.

## Read first (required)

0. `docs/agent-operating-card.md` — **auto-injected at session/subagent start**, so you should already have it: the distilled most-violated rules, the diagnostics decision table, and pointers. The items below are the depth behind it.
1. `docs/method-discipline.md` — general anti-cargo-cult method discipline (how not to fool yourself). High priority. Grounding: Feynman, [*Cargo Cult Science*](https://calteches.library.caltech.edu/51/2/CargoCult.htm).
2. **This file** — operating modes + the structural rules that make the discipline actually bind.
3. `docs/project-standards.md` — project-specific standards (controls/baselines, validation, failure-modes-to-watch, domain conventions).
4. `docs/anti-patterns.md` — the negative-knowledge playbook: tempting-but-wrong moves, why they fail here, and the correct first move.
5. `README.md` - project structure, code conventions.
6. The lab-notebook log for the area you're working in (see *The record*, below).
7. `env_setup.md` - python environment standards for the project (the canonical environment)

## Mechanisms (structural enforcement — prefer these over prose rules)

The discipline is enforced by artifacts and roles, not only by reading:

- **`/pre-run-checklist` skill** — produces the mandatory pre-registration **design checkpoint**
  in the lab log *before* a consequential run, then stops for approval (structural rule 3 as an
  artifact, not a promise).
- **`/diagnose-sampling` skill** — the ordered diagnostic workflow for misbehaving inference
  runs (cheap mundane checks first; produces UNCERTIFIED findings, not fixes).
- **`rigor-grader` agent** — adversarial grading of a proposed claim against this discipline;
  cheap to run, so run it before asking the human to certify (structural rules 1–2 as a role).
- **`inference-diagnostician` agent** — sampling diagnosis with the domain workflow baked in.
- **Lints/tests** — `tools/lint_silent_defaults.py`, `tests/test_no_silent_scientific_defaults.py`,
  and the `Pipeline.run` model card guard against silent scientific defaults.

When a post-mortem yields a new rule (structural rule 4), first ask whether it can be a
mechanism instead — a lint, a template field, a grader-checklist item, a skill step. Only if
not, add prose.

## Operating modes

Two modes. **The rules below are identical in both; only *who grades* changes.** State which mode you're in at the start of a session.

- **Mode A — planning agent + subagents.** A human + a planning/orchestrator agent that dispatches execution subagents. Subagents *propose*; the orchestrator + human *dispose*.
- **Mode B — single agent.** A human + one agent. The agent *proposes*; the *human* is the grader.

In both modes the load-bearing invariant is the same: **the party that produced a result never certifies it.**

## Structural rules (this is the part that makes the discipline work)

Discipline documents get ritualized — followed in letter, not spirit. These structural rules exist to stop "going through the motions."

1. **Proposer ≠ grader. No self-certified passes.** Whoever ran an analysis may *propose* a verdict, marked **UNCERTIFIED**, but may not write a PASS / CONFIRMED into the durable record. The verdict is rendered by the grader — in **Mode A** the orchestrator agent (with human concurrence on consequential claims); in **Mode B** the human.
2. **Grade the artifact, not the summary.** The grader opens the actual plot / code / numbers and verifies independently — never certifies from the producer's self-description. "The curves match" is not evidence; the plot is.
3. **Surface the criterion before a consequential or expensive run.** State the metric, the *derived* threshold (why that number would falsify the claim), and the pre-committed expected appearance of the result — and stop for approval — *before* computing. A threshold you cannot derive is not a test. (Mode A: this is the subagent's design checkpoint. Mode B: the agent surfaces it to the human.) This gate is for runs that enter the record or inform a substantial decision — not every micro-step.
4. **Rules are earned, and kept lean.** A failure → a short post-mortem → extract the *general* failure mode → add at most one rule. Reject candidate rules that are too case-specific; rule-bloat is itself a failure mode (long docs get parts overlooked).
5. **Scope and withdraw honestly.** Scope every certified claim precisely — what it does *and does not* cover. Record negative results. If a claim turns out wrong, withdraw it *explicitly in the record*. Don't mark anything "done" while something is outstanding.
6. **The durable record is the source of truth; agent memory is not.** Persist state to the lab-notebook logs continuously, so work survives interrupts and context loss. Recovery = read the record, not resume an agent. (Mode A: subagents are disk-checkpointed batch jobs, not resumable threads — anything not written to disk is lost on interrupt.)

## Carried-over rigor rules

- **No fabrication.** Never invent a citation, file path, numerical value, or quote. If something is uncertain, locate it (search the repo, read call sites, check upstream) or ask. A blocked question beats a wrong assumption.
- **Primary sources over summaries.** When a task depends on a specific claim, equation, number, or figure, read the primary source. Do not substitute an abstract, summary, or web reconstruction, and do not present such a reconstruction as if verified against the original. If you cannot access the source, ask for it and state the limitation explicitly rather than guessing.
- **No silent approximations in mathematical or statistical steps.** Do not swap an unjustified approximation, heuristic, or quick fix in for a principled step. If a shortcut seems necessary, *stop and ask* whether it is scientifically acceptable before relying on it.

## The record: lab-notebook logs

The durable record is split into **one lab-notebook log per rough research area**, so each stays short and relevant. Each log holds: current state, what's been tried (including **negative results**), and a **claims register** — every claim with a status of `proposed` / `certified (scope: …)` / `withdrawn`.

Areas → logs (lab-notebook logs live in `docs/logs/`, one per area):
- Compute / likelihood-gradient profiling → `docs/logs/compute-profiling.md`
- Carousel-lens MCLMC sampling diagnosis → `docs/logs/carousel-mclmc-sampling.md`
- Why lensing posteriors are hard to sample (standing question, cross-system) → `docs/logs/why-hard-to-sample.md`

Parked ideas (tabled, not active — do not start without the human re-opening them) live in
`docs/logs/open-topics.md`.

Update the relevant log after any substantive step. Stale state is worse than none.

## Project-specific standards

The project-specific standards are in `docs/project-standards.md`.
It contains this project's domain conventions, controls/baselines, validation discipline, and failure-modes-to-watch. Keep domain content here or in the standards file — out of `docs/method-discipline.md`, which stays general and portable.

## When unsure

Ask. Surface ambiguities in scientific intent, missing context, or a decision that is genuinely the human's — rather than guessing. "I don't know" and "this points to a structural problem" are correct, encouraged answers.
