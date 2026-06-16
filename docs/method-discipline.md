# Method Discipline (general / anti-cargo-cult)

**Required reading, high priority.** This file is the project's *general* research-method discipline — the standards for not fooling yourself when evaluating a method, metric, model, or inference procedure. It is domain-independent. The *project-specific* standards (controls/baselines, validation, failure modes, domain conventions) live in the project-standards file and are equally required reading.

The governing idea (Feynman, *Cargo Cult Science*): **you are the easiest person to fool.** Following the letter of a rigor checklist — even performing the pre-registration ritual below — does not protect you if the test itself is mis-specified. A confident "PASS" certified against an arbitrary threshold, while a plot of the same result shows a structured disagreement plainly, is the canonical failure these rules exist to prevent.

---

## 1. Identifiability is a precondition for any parameter claim

- Before claiming a method recovers or predicts a parameter well, confirm that parameter is *identifiable* from the data being fit. A degenerate parameter (or combination) must not be reported as a target.
- If a parameter is not identifiable on its own but an identifiable combination is, report the combination only.

## 2. Match the test to the claim

Before testing, **classify the claim** — each type has an appropriate test:

- **Deterministic identity / asymptotic limit** → test the limit object **deterministically, to numerical/solver tolerance**. Do *not* test an identity with a loose percentage band on a finite stochastic sample — that conflates the identity with finite-size and sampling artifacts.
- **Distributional claim** → compare distributions (quantiles, coverage), not just means.
- **Stochastic-estimator behaviour** → a separate claim from the quantity it approximates; validate it on its own terms (large-sample limit, conditioning, alignment), never as a stand-in for the analytic proof.

**Name the link you are testing.** When a claim is a *chain* — an analytic identity, *and* a finite-sample estimator or process converging to it — state explicitly which link your test covers and which remain untested. Confirming that a limiting equation has a property does not show that a finite-sample estimator converges to it; those are different links needing different tests. Conflating links is how a test can "pass" while the claim it appears to support is still open.

A mismatch between claim type and test type is a *category error*, deeper than any threshold choice and unfixable by tuning the threshold.

## 3. Pre-registration must be principled, not ritual

- **Before any change or comparison, state three things:** a **cause hypothesis** (what is actually wrong, in model/algorithm/data terms — not "let's try X"), a **prediction** (the effect's direction *and* order of magnitude), and a **falsifier** (what result would prove the hypothesis wrong). Then run, and compare observed vs. predicted. If the magnitude is far off, the hypothesis **failed** — re-examine assumptions; do not reach for a similar tweak.
- **The ritual is worthless if the threshold is arbitrary.** A threshold you cannot *derive* is not a test. State why *that* number, in *those* units, would falsify the claim.
- **No result-dependent metric selection.** Fix the primary metric in advance. If you compute several variants, report them all with equal standing; promoting the one that happens to pass — after seeing the results — is fishing.

## 4. Name the metric's blind spot

For any diagnostic metric, state in one sentence **what real disagreement it would fail to detect**. (E.g. RMSE is phase-blind — a structured time-shift partly cancels; normalizing by a peak hides sub-peak error; an aggregate goodness-of-fit hides a model that nails one subset and ignores the rest.) If the disagreement you care about lives in the metric's blind spot, choose a different metric.

## 5. Plots before metrics — and the plot wins

- Before reporting any aggregate metric, **look**: the distribution of the target, the residuals (any structure?), and a handful of concrete examples (best, worst, median).
- **A number that "passes" while the plot disagrees is an open finding, not a pass.** Reconcile them explicitly; resolve the conflict toward *investigating*, not toward the number.

## 6. Debugging & method-evaluation discipline (anti-rabbit-hole, metric-sanity)

**Scope:** applies when debugging/evaluating an algorithm, method, model, or inference procedure on a metric (chi-squared, ESS, log-likelihood, $R^2$, posterior quality, identifiability, etc.). Does **not** apply to pure software-engineering work — syntax/import/typing bugs, environment/dependency issues, build/config/CI, refactors with no scientific content. Handle those directly without this overhead.

- **Anti-rabbit-hole stop rule.** After two failed changes of the same class, stop editing. Before trying anything else, list the assumptions you have *not* tested, and propose a **diagnostic** that isolates the true cause — as opposed to another **fix** that assumes the cause.
- **Metric sanity (anti-Millikan-anchoring).** Always report target value, current value, and the gap in meaningful units (orders of magnitude, "Nx from target"). Classify the regime: *fine-tuning territory* vs. *structurally wrong* (a reduced chi-squared of 26 vs. 38 are both structurally wrong when the target is ~1 — treat the difference as noise and hunt for a bug/mis-specification, not optimizer knobs). **Movement in the desired direction is not success.**
- **General discipline.** Preserve the original form as a named baseline; keep changes reversible and isolated; revert before compounding patches. "I don't know" and "this points to a structural problem" are correct, encouraged answers — prefer them over a confident-sounding guess. When a change improves a metric, state how it could be an **artifact** (overfitting to the metric, a masked region, a wrong noise model/units) *before* accepting it.

## 7. Surface assumptions

When making any inference, make assumptions explicit — a comment in code or a note in the log. A statement like "I am treating [X] as proportional to [Y]" is the kind of thing that should be visible, not buried.
