# why_hard_to_sample — carousel minimal-case sampling investigation (2026-06/07)

Pre-registered investigation into why the carousel minimal case sampled badly, ending
in a validated fix (observable-slope reparameterization) and a transferable diagnosis
playbook. **The complete register — every hypothesis, prediction, falsifier, result,
and honest miss — is `docs/logs/why-hard-to-sample.md`.** Read that first; this README
is only a map.

Start here if you have a NEW misbehaving system:
- `docs/playbooks/sampling-diagnosis-playbook.md` — triage (clone gap), 14 instruments
  with reading guides, 5 validated disease classes, fix ladder. System-agnostic.
- `.claude/skills/diagnose-sampling` — the workflow skill; points at the playbook.

## Phase map (T-number -> scripts -> log entry)

| Phase | What it established | Key scripts |
|---|---|---|
| T0/T1 | Seed band + Gaussian-clone gap (master metric) | `run_t0.py`, `run_t1_clone.py`, `build_clone.py`, `exp_config.py` |
| T15 | (Rs,theta_E) EINSTEIN reparam; prior != coordinate change | `t15_carousel_decompose.py` |
| T18-T20 | MAP init trap; FD-vs-AD gradient-noise ladder | `t18_map_arm.py`, `t20_*.py` |
| T21 | Typical-set init standard | `t21_typical_init.py` |
| T22 | conv float64 standard | (config; see log) |
| T23/T24 | xi tail = funnel-neck reflections; momentum accounting + encounter census | `t23_momentum_gpu.py`, `t24_census_gpu.py`, `t23_t24_common.py` |
| T25 | Conditional-width (HVP) profile; Route A/B transforms; Jacobian-vs-profile cross-check | `t25_profile_gpu.py`, `t25_transforms.py`, `reparam_bijector.py` |
| T26 | Gap-based acceptance battery; Route A cures, Route B fails as predicted | `t26_battery_gpu.py`, `t26_analyze.py`, `systems/carousel_min_newA,B/` |
| T27 | (M200,c) pushforward sanity | `t27_pushforward.py` |
| T28 | Prior set natively in observable slope s; likelihood-limited posterior | `t28_common.py`, `t28_sprior_transform.py`, `t28_run_gpu.py`, `t28_analyze.py`, `systems/carousel_min_sprior/` |
| T29 | NFW_ELLIPSE_SLOPE profile class (native (theta_E, s_E)); gates | `nfw_ellipse_slope.py` (shim; class lives in gigalens), `t29_slope_class_gpu.py` |

`slurm/` holds the launchers. `run_t28_payload.sh` documents the srun-into-existing-
allocation pattern; `run_t29.sh` the sbatch pattern (note: BASH_SOURCE is the spool
copy under sbatch — paths are hardcoded).

## Portability to a new system

The harness talks to its target through `systems/<case>/system.py` exposing a
`load_target()` 5-tuple `(prob_model, qz, z_center, dim, param_names)` — write that one
shim for your system and point it at a reference run. Carousel-specific constants live
in script headers (output dirs, `T21_ARMS`, REF_DIR paths): copy the script, edit the
header. The measured artifacts (lambda profiles, transform npz, seeds) are per-system
by design — regenerate via the same scripts; sha256 of registered artifacts are in the
log. Raw chain npz are gitignored (regenerable; seeds + configs logged).

## Discipline

Everything here ran under pre-registration (hypothesis / predicted magnitude /
falsifier BEFORE each run) with gate-first GPU scripts that abort before sampling.
All findings are proposed (UNCERTIFIED) until graded by the human. See
`docs/method-discipline.md` and the operating card.
