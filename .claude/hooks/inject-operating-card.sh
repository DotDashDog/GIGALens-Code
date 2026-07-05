#!/usr/bin/env bash
# SessionStart / SubagentStart hook — inject the GIGALens agent operating card
# (docs/agent-operating-card.md) into context, so every agent and subagent starts
# with the distilled, most-violated rules + decision tables without taking a Read.
#
# Design: the card is the always-on core (kept lean, rotated by violation
# frequency); the depth docs (AGENTS.md, method-discipline, project-standards,
# inference-diagnostics, anti-patterns) are one hop away and the card points to
# them. Registration lives in the repo's .claude/settings.json (project-scoped),
# NOT in the user's global settings — this hook should fire only for sessions in
# this repo and its worktrees.
set -euo pipefail

# Self-locate the repo root from this script's path (.claude/hooks/ -> repo root),
# so it works from any cwd and in worktrees without depending on $CLAUDE_PROJECT_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOC="$REPO_ROOT/docs/agent-operating-card.md"

# If the card is missing, do nothing rather than block session start.
[ -f "$DOC" ] || exit 0

# Echo back the firing event (SessionStart or SubagentStart) so additionalContext
# is attributed to the correct event; default to SessionStart if unreadable.
RAW="$(cat || true)"
EVENT="$(printf '%s' "$RAW" | jq -r '.hook_event_name // empty' 2>/dev/null || true)"
[ -n "$EVENT" ] || EVENT="SessionStart"

# Emit the card verbatim as additionalContext (it carries its own pointers to depth docs).
jq -n \
  --arg event "$EVENT" \
  --rawfile doc "$DOC" \
  '{hookSpecificOutput: {hookEventName: $event, additionalContext: $doc}}'
