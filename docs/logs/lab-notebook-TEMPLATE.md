# Lab Notebook — [FILL IN: research-area name]

[FILL IN: one-line description of this research area.]

**Last updated:** [FILL IN: date]

> One log per rough research area (see `../../AGENTS.md` → *The record*). This is the durable source of truth for the area — agent memory is not. It holds the current state, the **claims register**, a chronological log, and open items. Update it after any substantive step; stale state is worse than none.

---

## Current state

[FILL IN: a 2–4 line snapshot — what's active, what's blocked, and where the live artifacts (code/data/plots) live.]

---

## Claims register

Every scientific claim in this area, with a status of `proposed (UNCERTIFIED)` / `certified (scope: …)` / `withdrawn`. The producer logs a claim as **proposed**; the grader (the orchestrator agent + human in Mode A, the human in Mode B) certifies or withdraws it **after inspecting the artifact itself** (the plot / code / numbers), never from the producer's summary. Certified claims carry an explicit **scope** (what they do *and do not* cover).

Every proposed claim must include a **Doubt report** (Feynman's bending-over-backwards, supplied by the *producer*, unprompted): alternative explanations considered and how each was eliminated, plus any detail that could throw doubt on the interpretation. A proposal without one is incomplete and cannot be graded.

> _The three entries below are worked examples showing each status — **delete them** and start your own._

### C-1 — [example, DELETE] Estimator `X` recovers parameter `θ` on synthetic data
- **Status:** `certified (scope: regime A, well-sampled; NOT regime B / small-sample)`
- **Criterion (pre-registered):** relative error of `θ̂` vs. truth, falsifier > 5%; threshold derived from [the solver tolerance / the noise floor / …].
- **Evidence / artifact:** `path/to/recovery_plot.png`, `path/to/results.json`, code `path/to/run.py`
- **Proposed by / on:** [producer] · [date]   ·   **Graded by / on:** [grader] · [date] (plot + code inspected; non-circularity checked)
- **Caveats / noted sub-findings:** [e.g. an anomaly in one sub-case, recorded as benign]

### C-2 — [example, DELETE] Method `Y` converges faster than `Z`
- **Status:** `withdrawn`
- **Why withdrawn:** the comparison metric was [phase-blind / result-dependent — the passing variant was promoted post-hoc]; the plot contradicted the number. Replaced by [C-4] using a principled criterion.
- **Evidence / artifact:** `path/to/withdrawn_run.png` (retained for history)
- **Proposed by / on:** [producer] · [date]   ·   **Withdrawn by / on:** [grader] · [date]

### C-3 — [example, DELETE] New sampler improves ESS/sec by ~2×
- **Status:** `proposed (UNCERTIFIED)` — awaiting grader inspection of trace plots + R̂ across chains
- **Criterion (pre-registered):** ESS/sec on [benchmark]; falsifier: no improvement beyond run-to-run variance (state the variance).
- **Evidence / artifact:** `path/to/proposed_run/` (producer's report marked UNCERTIFIED)
- **Doubt report (mandatory):** [alternative explanations considered and how eliminated — e.g. "could be the shorter burn-in, not the sampler: ruled out by X"; details that could throw doubt — e.g. "baseline was run under float32; not yet rerun under current config"]
- **Proposed by / on:** [producer] · [date]   ·   **Grader:** _pending_

---

## Design checkpoints (criteria awaiting approval)

Before a consequential or expensive run, the producer logs a checkpoint here and stops; the grader approves or revises *before* the run (structural rule 3). The `/pre-run-checklist` skill walks through producing one. A run without a prior checkpoint entry is illegitimate regardless of its result. Clears once the run is launched — then log observed vs. predicted (a badly-missed magnitude means the hypothesis failed even if the direction was right).

Each checkpoint carries: **cause hypothesis** (model/algorithm/data terms, not "try X") · **prediction** (direction *and* order of magnitude) · **falsifier** · **metric + derived threshold** (why that number, in those units — "not derivable because [reason]" is a legal and reportable answer) · **metric's blind spot** (one sentence) · **pre-committed expected appearance** of the key plot · **cost estimate**.

> _Example below — **delete it**._

- **[example, DELETE] Run: convergence study of `X` as N→∞.** Hypothesis: [what is structurally true/wrong, in model terms]. Prediction: [direction + order of magnitude]. Falsifier: [plateaus / wrong sign]. Metric + derived threshold: [discrepancy vs. reference; expected scaling — derive it, and note that the mean of many samples scales differently from a single sample]. Blind spot: [what disagreement this metric can't see]. Expected plot: [what it should look like if the hypothesis holds]. Cost estimate: [~X]. **Status:** awaiting approval.

---

## Log (newest first)

Dated entries: substantive steps, surprises, dead ends, and **negative results** (a baseline beating your method, or an approach abandoned — record *why*).

- **[date]** — [FILL IN: what you did, what you found, links to artifacts. Newest at top.]
- **[date]** — [example, DELETE] Abandoned approach `W`: it [failed in this specific way]; cause traced to [structural reason]. Not to be retried without [what would have to change].

---

## Open questions

- [FILL IN: questions surfaced but not yet resolved — don't silently answer them; log here, with where they'll be resolved.]
