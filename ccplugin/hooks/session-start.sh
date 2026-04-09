#!/bin/bash
# SessionStart hook - Initialize oG-Memory for Claude Code session

# Resolve script directory and project root so sourcing works from any CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$CLAUDE_PLUGIN_ROOT" ] && [ -d "$CLAUDE_PLUGIN_ROOT" ]; then
    PROJECT_ROOT="$(dirname "$CLAUDE_PLUGIN_ROOT")"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT" || true
source "${SCRIPT_DIR}/common.sh"

# Check for API key
OG_VER="v0.1.0"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "{\"systemMessage\": \"[ogmemory $OG_VER] ERROR: OPENAI_API_KEY not set — memory search disabled\"}"
    exit 0
fi

# Start watch process
start_watch "$MEMORY_DIR"

# Write session heading to today's memory file
TODAY=$(date +%Y-%m-%d)
SESSION_TIME=$(date +%H:%M)
MEMORY_FILE="$MEMORY_DIR/$TODAY.md"

# Create file if doesn't exist
if [ ! -f "$MEMORY_FILE" ]; then
    echo "# $TODAY" > "$MEMORY_FILE"
fi
echo "" >> "$MEMORY_FILE"
echo "## Session $SESSION_TIME" >> "$MEMORY_FILE"

# Inject recent memories as additionalContext
ADDITIONAL_CONTEXT=""
RECENT_FILES=$(ls -t "$MEMORY_DIR"/*.md 2>/dev/null | head -2)

for file in $RECENT_FILES; do
    if [ -f "$file" ]; then
        RECENT_CONTENT=$(tail -30 "$file" 2>/dev/null)
        if [ -n "$RECENT_CONTENT" ]; then
            ADDITIONAL_CONTEXT="$ADDITIONAL_CONTEXT

Recent memories from $(basename $file):
$RECENT_CONTENT"
        fi
    fi
done

# Return JSON response - status line follows OG_* / OPENAI_* env (same as config.py / MemoryEngine)
DB_HOST="${OG_DB_HOST:-localhost}"
DB_PORT="${OG_DB_PORT:-5432}"
EMB_PROVIDER="${OG_EMBEDDING_PROVIDER:-openai}"
EMB_MODEL="${OG_EMBEDDING_MODEL:-${OPENAI_EMBEDDING_MODEL:-text-embedding-3-small}}"
STATUS_LINE="[ogmemory $OG_VER] embedding: ${EMB_PROVIDER}/${EMB_MODEL} | opengauss: ${DB_HOST}:${DB_PORT}"

if [ -n "$ADDITIONAL_CONTEXT" ]; then
    # Escape for JSON string: backslash, double quote, newline
    ESC_ADDITIONAL_CONTEXT=$(printf '%s' "$ADDITIONAL_CONTEXT" | sed ':a;N;$!ba;s/\\/\\\\/g;s/\"/\\\"/g;s/\n/\\n/g')
    echo "{
  \"systemMessage\": \"$STATUS_LINE\",
  \"additionalContext\": \"$ESC_ADDITIONAL_CONTEXT\"
}"
else
    echo "{
  \"systemMessage\": \"$STATUS_LINE\"
}"
fi
