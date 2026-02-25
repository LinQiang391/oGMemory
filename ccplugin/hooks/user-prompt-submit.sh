#!/bin/bash
# UserPromptSubmit hook - Append user prompt to today's memory md and show hint

# Current directory when hook runs (user/Claude workspace), used for memory file
HOOK_CWD=$(pwd)
# Resolve script directory for sourcing common.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root only for run_ogmemory later
if [ -n "$CLAUDE_PLUGIN_ROOT" ] && [ -d "$CLAUDE_PLUGIN_ROOT" ]; then
    PROJECT_ROOT="$(dirname "$CLAUDE_PLUGIN_ROOT")"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT" || true
source "${SCRIPT_DIR}/common.sh"
cd "$HOOK_CWD" || true

# Read hook input
INPUT=$(cat)

# Extract prompt text
PROMPT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('prompt', ''))
except Exception:
    print('')
" 2>/dev/null)

# Skip short prompts for hint, but still try to append if non-empty
if [ -n "$PROMPT" ] && [ ${#PROMPT} -ge 1 ]; then
    TODAY=$(date +%Y-%m-%d)
    SESSION_TIME=$(date +%H:%M)
    # Use current directory (hook cwd), not script/project root
    case "$MEMORY_DIR" in
        /*) MEMORY_FILE="$MEMORY_DIR/$TODAY.md" ;;
        *)  MEMORY_FILE="$HOOK_CWD/$MEMORY_DIR/$TODAY.md" ;;
    esac
    mkdir -p "$(dirname "$MEMORY_FILE")"
    if [ ! -f "$MEMORY_FILE" ]; then
        echo "# $TODAY" > "$MEMORY_FILE"
    fi
    # One-line preview for md (newlines -> space, limit length)
    PREVIEW=$(printf '%s' "$PROMPT" | tr '\n' ' ' | head -c 400)
    # Append user prompt line (printf to avoid % interpretation)
    echo "" >> "$MEMORY_FILE" && \
    printf -- '- **%s** (user): ' "$SESSION_TIME" >> "$MEMORY_FILE" && \
    printf '%s\n' "$PREVIEW" >> "$MEMORY_FILE"
    # Index so it's searchable
    run_ogmemory index "$MEMORY_FILE" 2>/dev/null || true
fi

echo '{"systemMessage": "[ogmemory] Memory available"}'
