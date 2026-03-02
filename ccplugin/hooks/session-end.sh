#!/bin/bash
# SessionEnd hook - Cleanup

# Resolve script directory and project root so sourcing works from any CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$CLAUDE_PLUGIN_ROOT" ] && [ -d "$CLAUDE_PLUGIN_ROOT" ]; then
    PROJECT_ROOT="$(dirname "$CLAUDE_PLUGIN_ROOT")"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT" || true
source "${SCRIPT_DIR}/common.sh"

# Stop watch process
stop_watch

echo "{}"
