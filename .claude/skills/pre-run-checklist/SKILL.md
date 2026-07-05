---
name: pre-run-checklist
description: Write the mandatory pre-registration design checkpoint to the lab-notebook log before a consequential or expensive run (Slurm job, long sampling run, any run entering the record or informing a decision), then stop for grader approval. Use BEFORE launching such a run, or when the user asks to pre-register / design-checkpoint a run. Not needed for cheap exploratory micro-steps that won't enter the record.
---

# Pre-run checklist (design checkpoint)

You are about to make a run legitimate. A consequential run without a design checkpoint written
*before* launch is illegitimate regardless of its result (operating card, non-negotiable 1).
Your output is a **filled checkpoint entry in the lab-notebook log**, followed by a **full
stop for approval**. Producing the entry and immediately launching the run defeats the purpose.

## Steps

1. **Locate the record.** Identify the lab-notebook log for this research area in `docs/logs/`
   (see `AGENTS.md` → *The record*). If no log exists for the area, create one from
   `docs/logs/lab-notebook-TEMPLATE.md` first.

2. **State the claim and classify it** (`docs/method-discipline.md` §2): deterministic
   identity / asymptotic limit, distributional claim, or stochastic-estimator behaviour. If the
   claim is a *chain* of links, name explicitly which link this run tests and which remain
   untested. A mismatch between claim type and test type is a category error no threshold can fix.

3. **Fill the pre-registration triplet:**
   - **Cause hypothesis** — what is actually wrong or being tested, in model/algorithm/data
     terms. "Let's try X and see" is not a hypothesis; if that's all you have, say so and
     propose a *diagnostic* run instead, labeled as such.
   - **Prediction** — direction **and** order of magnitude of the effect. If you cannot
     predict a magnitude, state the assumption that blocks you.
   - **Falsifier** — the concrete result that would prove the hypothesis wrong. If no result
     could, the run is not a test; redesign it.

4. **Derive the threshold.** State the metric, the threshold, and *why that number in those
   units* falsifies the claim (solver tolerance, noise floor, run-to-run variance you have
   measured, …). **If you cannot derive it, do not invent one** — write "threshold not
   derivable because [reason]" in the checkpoint and surface that to the grader; an underived
   threshold is itself a finding.

5. **Name the metric's blind spot** in one sentence (what real disagreement would this metric
   fail to detect?). If the disagreement you care about lives in the blind spot, change the
   metric before running.

6. **Pre-commit the expected appearance:** what should the key plot look like if the
   hypothesis holds — and what would it look like if the falsifier fires?

7. **Estimate the cost** (node-hours / wall-clock / queue) so the grader can weigh it.

8. **Write the entry** into the log's **Design checkpoints** section (format follows the
   template's example), status `awaiting approval`. Record seeds, config path, code version.

9. **STOP.** In Mode B, tell the human the checkpoint is ready and wait. In Mode A, return the
   checkpoint to the orchestrator for grading. Do not launch the run in the same action as
   writing the checkpoint. After the run, compare observed vs. predicted **magnitude** — a
   large miss means the hypothesis *failed* even if the direction was right; log that outcome
   either way, and clear the checkpoint.
