# Existing LAPS implementations — audit synthesis (Phase 3)

Standard = `laps_spec.md` (paper-verified). Per-file detail: `audit-laps-py.md`, `audit-laps-coldstart3.md`.
Both impls were judged against the same axes; neither was treated as correct.

## Cross-impl conformance (H1–H4)

| Axis | Paper | `laps.py` | `laps-COLDSTART3.py` |
|---|---|---|---|
| H1 step law | `EEVPD_wanted=F(C·D̃)` | **EMAUS** `C·D̃^{3/8}` (l.585) | **NEITHER** — bang-bang P-ctrl to fixed `5e-4`; no `D̃` at all |
| H2 switch | `δ=σ/μ<0.01` on **x_i²** | **EMAUS** `(σ/μ)²` on **x_i** (l.255-263,572,590) | **ABSENT** — fixed-length scan, no switch |
| H3 `C` | 0.025 | **0.1** (l.431) | **ABSENT** |
| H4 precond | diagonal `1/Var`, Phase-2 only | **NEITHER** — dense SVI metric in Phase-1 from step 0 | timing ok, but full shrunk cov (not diagonal) |

Conforming in `laps.py`: LF/MN2-MN4 integrators, N=15, L=1.25·L_full, init ε=0.01√d, accept 0.7/0.9,
3% bisection freeze, samples-only (no evidence/tempering). `laps-COLDSTART3.py` conforms only on α=2 `L`
update and samples-only output; everything else is missing (author's own TODOs confirm it's a stub).

## Ranked root causes

**`laps.py` (the "sort of samples but mostly fails to converge" one):**
1. **Switch never fires (DOMINANT).** `(σ/μ)²` on identity `x_i` → divide-by-mean² explodes for
   near-zero-mean coords (shear, centroids — ubiquitous in lensing) → `r_max~1e12` → Phase-1 never
   switches → returns biased UNADJUSTED samples, never reaches the Metropolis-adjusted phase. This is
   the mechanism behind the user's symptom. **Falsifier:** `switch_index == num_unadjusted_steps`,
   `active≡1` on any near-zero-mean target. CONFIRMS pre-reg D1 (the x_i²-vs-x_i + threshold axis).
2. **Dense ill-conditioned SVI metric in Phase 1** (l.92-146,466-469): Cholesky of VI cov with ~1e-6
   jitter → NaN substeps, depressed Phase-2 acceptance. (mclmc.py already solved this via `_regularize_cov`.)
3. **Wrong step law `C·D̃^{3/8}`, C=0.1** (l.585,431): near convergence `D̃^{3/8} ≫ D̃^{3/2}` keeps ε
   too large → D̃ stalls at a floor instead of →0.
   Other bugs: `LAPS_JIT` dead (`raise NotImplementedError` l.976; user must call `full_laps_sharded`);
   first Phase-1 step `L=inf`, `init_L` discarded (l.449,713); no finite-checking in Phase 1 (one NaN
   grad corrupts the global `psum`); `equi_full_loss=equi_diag_loss` (l.575) → full-rank branch no-op.

**`laps-COLDSTART3.py`:** not LAPS — vanilla MCLMC warmup + single-step (N=1) Metropolis; cold `N(0,I)`
init breaks the warm-start premise; no `D̃`/`δ`/`F`/bisection. Incompleteness, not subtlety.

## Salvage vs rewrite
- `laps.py`: **salvageable skeleton** (MAMS kernel, ECA reductions, sharding, diagnostics, bisection are
  sound). But it is a STANDALONE parallel implementation — it does NOT build on the hardened `mclmc.py`
  APIs, so it re-solves (and in places mis-solves: NaN handling, dense-metric stability, L init) problems
  `mclmc.py` already handles.
- `laps-COLDSTART3.py`: **rewrite** (salvage only integrator plumbing + gigalens wrapper + α=2 L update).

## Reconciliation with the Phase-2 design (Option B)
The design recommends building paper-faithful LAPS on the in-tree `mclmc.py` kernels (single hardened
substrate). The audit shows `laps.py` is patchable in ~4 edits but on a parallel un-hardened base.
→ Architecture is a real fork for the user (Option B fresh-build vs salvage `laps.py`). Independent of
that choice, `laps.py` is a cheap **diagnosis-confirmation vehicle**: patch only its switch
(x_i² + δ=σ/μ<0.01) and run on a CPU Gaussian; if it then switches and converges, root-cause #1 / D1 is
empirically confirmed before any build investment (a real diagnostic, not a fix that assumes the cause).
