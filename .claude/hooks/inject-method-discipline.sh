#!/usr/bin/env bash
# SessionStart hook — inject the project's method-discipline rules into context
# so every agent (and the human) gets them without having to take a Read action.
#
# Rationale: only CLAUDE.md is guaranteed in context. method-discipline.md used to
# sit two soft-link hops away (CLAUDE.md -> AGENTS.md -> method-discipline.md) and
# agents routinely skipped it. This hook removes the indirection: the canonical
# discipline doc is *present*, not something to go fetch. The pointer block below
# keeps the rest of the doc tree one hop away.
set -euo pipefail

# Self-locate the repo root from this script's path (.claude/hooks/ -> repo root),
# so it works from any cwd and in worktrees without depending on $CLAUDE_PROJECT_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOC="$REPO_ROOT/docs/method-discipline.md"

# If the canonical doc is missing, do nothing rather than block session start.
[ -f "$DOC" ] || exit 0

# Echo back the firing event (SessionStart or SubagentStart) so additionalContext
# is attributed to the correct event; default to SessionStart if unreadable.
RAW="$(cat || true)"
EVENT="$(printf '%s' "$RAW" | jq -r '.hook_event_name // empty' 2>/dev/null || true)"
[ -n "$EVENT" ] || EVENT="SessionStart"

POINTER='# GIGALens — required reading (auto-injected at session start)

This is **scientific-research** work, not just-make-it-run software work. The
canonical *general* method discipline is inlined below. Before any consequential
or expensive run, also read — each is now one hop away, no longer gated behind
reading AGENTS.md first:

- `AGENTS.md` — operating modes + the structural rules that make the discipline
  bind (proposer != grader; grade the artifact not the summary; surface the
  derived threshold before a run; scope and withdraw claims honestly).
- `docs/project-standards.md` — domain standards, controls/baselines, and
  failure-modes-to-watch (esp. **no silent scientific defaults**: a missing
  PSF / noise model / mask / units / prior must raise, never default).
- `docs/env_setup.md` — the canonical python environment and how to run code.
- `docs/inference-diagnostics.md` — what the diagnostics mean (reduced chi^2,
  R-hat / ESS, residual / source-plane plots).
- The lab-notebook log for your area (see AGENTS.md -> "The record").

This injection is the always-on core; the linked docs are the depth.

---

'

# Slurp the doc and emit JSON with the combined text as SessionStart additionalContext.
jq -n \
  --arg event "$EVENT" \
  --arg pointer "$POINTER" \
  --rawfile doc "$DOC" \
  '{hookSpecificOutput: {hookEventName: $event, additionalContext: ($pointer + $doc)}}'
