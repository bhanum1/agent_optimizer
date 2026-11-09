#!/usr/bin/env bash
# Install the agentoptimizer skill for Claude Code and/or Codex by linking this folder
# into their skills directories. Edits here then take effect immediately.
#
#   ./install.sh            both CLIs
#   ./install.sh claude     Claude Code only (~/.claude/skills)
#   ./install.sh codex      Codex only (~/.codex/skills)
#   ./install.sh --uninstall [claude|codex]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
uninstall=0
if [[ "${1:-}" == "--uninstall" ]]; then uninstall=1; shift; fi
targets=("${@:-claude codex}")
targets=(${targets[@]})

for t in "${targets[@]}"; do
  case "$t" in
    claude) dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills" ;;
    codex)  dir="${CODEX_HOME:-$HOME/.codex}/skills" ;;
    *) echo "unknown target: $t (use claude or codex)" >&2; exit 2 ;;
  esac
  link="$dir/agentoptimizer"
  if (( uninstall )); then
    if [[ -L "$link" ]]; then rm "$link"; echo "removed $link"; fi
    continue
  fi
  mkdir -p "$dir"
  if [[ -e "$link" && ! -L "$link" ]]; then
    echo "$link exists and is not a link; leaving it alone" >&2; exit 1
  fi
  ln -sfn "$here" "$link"
  echo "linked $link -> $here"
done

if (( ! uninstall )); then
  command -v uv >/dev/null || echo "note: the skill needs uv (https://docs.astral.sh/uv/)" >&2
  echo "Building the skill's Python environment (first time can take a few minutes)..."
  uv sync --quiet --project "$here" --no-dev
  echo "Done. In Claude Code or Codex, ask to optimize something, or name the agentoptimizer skill."
fi
