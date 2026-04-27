#!/usr/bin/env bash
# Stop/PreCompact wrapper — detaches the Python child so Claude Code's `-p`
# mode shutdown does not kill it. Background:
#
#   Plugin-loaded Stop hooks (--plugin-dir) are async-backgrounded by Claude
#   Code. In -p mode Claude exits ~130ms after spawning the hook subprocess —
#   before python (~50-100ms cold start) can run, leaving `outcome: cancelled`
#   in the stream JSONL and never invoking call_after_turn.py.
#
#   This wrapper exits in ~5ms (so the stream records `outcome: success`) and
#   launches call_after_turn.py inside a fresh session via `setsid`, which
#   moves it out of Claude's process group and immunizes it from teardown.
#
# stdin (hook JSON) is buffered in memory and piped to the detached child.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT="$(cat)"

# `setsid ... &` + `disown` fully detach grandchild from this process group.
# inner bash receives HERE as $1, INPUT as $2 (avoids quoting hell).
setsid bash -c '
  printf "%s" "$2" | python3 "$1/call_after_turn.py" >/dev/null 2>&1
' _ "$HERE" "$INPUT" </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true

exit 0
