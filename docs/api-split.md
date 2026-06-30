# GIGALens two-API split: old-API (paper) vs new-API (real systems)

The GIGALens work is split into **two self-consistent "worlds"**, each a pinned
pair of (gigalens package branch + GIGALens-Code branch) checked out as parallel
**git worktrees** so both can be live at once — you never branch-switch to move
between paper work and real-systems work.

## Why the split exists

- `gigalens` was refactored to fold all multi-plane (lens **and** source) plus
  multi-band functionality into one unified **scene API**. It is more intuitive
  and is the long-term target, but it has **not yet been validated**.
- The **paper** (testing new sampling algorithms for lensing) is therefore done
  entirely on the **old API**, together with the old-API wrapper/helper
  functions the tests depend on.
- **Real-systems inference** uses the **new API**, since it is much more
  intuitive.
- Both streams progress simultaneously, which is why both worlds stay checked
  out side by side rather than living on branches you swap between.

## The two worlds

Each row is internally consistent. Do not mix a row's pieces across rows.

| World | gigalens (package) | GIGALens-Code (research) | conda env |
|---|---|---|---|
| **New API** (real systems) | `/global/u1/l/linusu/gigalens` @ `linusu-dev-merge` | `/global/u1/l/linusu/GIGALens-Code` @ `main` | `gigalens_multinode_env` (Shifter / JAX 0.10) — see `docs/env_setup.md` on `main` |
| **Old API** (paper) | `/global/u1/l/linusu/gigalens-old` @ `dev` | `/global/u1/l/linusu/GIGALens-Code-paper` @ `paper-oldapi` | `gigalens_oldapi` (conda, JAX 0.9.1) — see `docs/env_setup.md` on `paper-oldapi` |

`paper-oldapi` was branched from `ea265d9`, the last GIGALens-Code commit before
the new-API migration (`05218ec`, *"Migrate gigalens_research to the new gigalens
dev API"*), so it carries the old-API wrappers untouched by the migration.

## Telling the two APIs apart

The new (scene) API ships `gigalens.jax.scene_simulator`; the old API does not
(it uses `gigalens.jax.simulator`). Quick check:

```bash
python -c "import importlib.util as u; \
print('NEW API' if u.find_spec('gigalens.jax.scene_simulator') else 'OLD API')"
```

The checkout folder name (`gigalens-old`) is independent of the import name:
both worlds still `import gigalens` (src layout, package name `gigalens`). Which
world you get is set by which one is editable-installed into the active env.

## Recreating the worktrees

```bash
# gigalens package: old-API checkout next to the new-API one
cd /global/u1/l/linusu/gigalens && git worktree add /global/u1/l/linusu/gigalens-old dev

# GIGALens-Code: paper worktree on a branch off the last old-API commit
cd /global/u1/l/linusu/GIGALens-Code && \
  git worktree add -b paper-oldapi /global/u1/l/linusu/GIGALens-Code-paper ea265d9
```

`git worktree list` (run in either GIGALens-Code or gigalens) shows the live set.

## Rules that keep the worlds from re-tangling

1. **One conda env per world.** Never editable-install *both* gigalens worktrees
   into the same env — they both expose the top-level name `gigalens` and would
   collide (last install wins).
2. **Editable-install with `--no-deps`.** Each env is built with a pinned stack;
   `gigalens-old` hard-pins old `jax`/`tensorflow`/`numpy` in its metadata, so a
   plain `pip install -e` would clobber the env's GPU JAX. See the per-world
   `env_setup.md`.
3. **Expect the wrappers to drift.** The `main` (new-API) and `paper-oldapi`
   (old-API) wrapper sets will diverge — that is intended. Cherry-pick only
   genuinely API-agnostic fixes between them; do not try to make one wrapper set
   serve both APIs.
4. **Record the gigalens commit per result.** For reproducibility (this is a
   paper), note which `gigalens` commit produced each result in the lab-notebook
   logs, since the importable API depends on the worktree, not on this repo alone.

## Open validation

`paper-oldapi` wrappers are pinned at `ea265d9`, but the old-API `gigalens` `dev`
branch has advanced several commits since the split (old-API maintenance:
precision pipeline, JAX-0.10 compat work, profile cleanups). Smoke-test the
old-API wrapper ↔ `gigalens` signatures before trusting paper runs.
