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
- **Proposed by / on:** [producer] · [date]   ·   **Grader:** _pending_

---

## Design checkpoints (criteria awaiting approval)

Before a consequential or expensive run, the producer logs the proposed **metric + derived threshold + pre-committed expected appearance** here and stops; the grader approves or revises *before* the run (structural rule 3). Clears once the run is launched.

> _Example below — **delete it**._

- **[example, DELETE] Run: convergence study of `X` as N→∞.** Proposed metric: [discrepancy vs. reference]; expected scaling [derive it — and note that the mean of many samples scales differently from a single sample]; falsifier: [plateaus / wrong sign]. Cost estimate: [~X]. **Status:** awaiting approval.

---

## Log (newest first)

Dated entries: substantive steps, surprises, dead ends, and **negative results** (a baseline beating your method, or an approach abandoned — record *why*).

- **[date]** — [FILL IN: what you did, what you found, links to artifacts. Newest at top.]
- **[date]** — [example, DELETE] Abandoned approach `W`: it [failed in this specific way]; cause traced to [structural reason]. Not to be retried without [what would have to change].

---

## Open questions

- [FILL IN: questions surfaced but not yet resolved — don't silently answer them; log here, with where they'll be resolved.]
