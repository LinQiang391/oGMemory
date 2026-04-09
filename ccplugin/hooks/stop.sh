#!/bin/bash
# Stop hook - Summarize session and append to memory

# Guard against recursion
if [ "$STOP_HOOK_ACTIVE" = "1" ]; then
    echo "{}"
    exit 0
fi
export STOP_HOOK_ACTIVE=1

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
if [ -z "$OPENAI_API_KEY" ]; then
    echo "{}"
    exit 0
fi

# Get transcript path from environment
TRANSCRIPT_FILE="${CLAUDE_TRANSCRIPT_FILE:-}"

# Validate transcript
if [ ! -f "$TRANSCRIPT_FILE" ] || [ $(wc -l < "$TRANSCRIPT_FILE") -lt 3 ]; then
    echo "{}"
    exit 0
fi

# Parse transcript and summarize
TODAY=$(date +%Y-%m-%d)
SESSION_TIME=$(date +%H:%M)
MEMORY_FILE="$MEMORY_DIR/$TODAY.md"

# Generate session ID from transcript path
SESSION_ID=$(basename "$TRANSCRIPT_FILE" .jsonl)

# Parse last 200 lines of transcript
SUMMARY=$(tail -200 "$TRANSCRIPT_FILE" | python3 -c "
import sys
import json

lines = sys.stdin.readlines()
summary_lines = []

for line in lines:
    try:
        data = json.loads(line)
        if 'content' in data:
            content = data['content'][:200]
            summary_lines.append(f'- {content}')
    except:
        continue

print('\n'.join(summary_lines[:8]))
")

# Write summary to memory file
if [ -n "$SUMMARY" ]; then
    echo "" >> "$MEMORY_FILE"
    echo "### $SESSION_TIME" >> "$MEMORY_FILE"
    echo "<!-- session:$SESSION_ID transcript:$TRANSCRIPT_FILE -->" >> "$MEMORY_FILE"
    echo "$SUMMARY" >> "$MEMORY_FILE"
    
    # Index the updated file
    run_ogmemory index "$MEMORY_FILE" 2>/dev/null
fi

echo "{}"
