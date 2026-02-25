#!/bin/bash
# Common setup for oG-Memory Claude Code plugin hooks

# Ensure oG-Memory is available
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    else
        echo "{}"
        exit 0
    fi
else
    PYTHON_CMD=python
fi

# 解析 oG-Memory 项目根目录（common.sh 位于 ccplugin/hooks/，上两级为项目根）
_COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OGMEMORY_PROJECT_ROOT="$(cd "$_COMMON_SH_DIR/../.." && pwd)"

# Add src to Python path for development mode (when running from project root)
export PYTHONPATH="${PYTHONPATH}:${OGMEMORY_PROJECT_ROOT}/src"

# Memory directory
MEMORY_DIR=".ogmemory/memory"
WATCH_PID_FILE=".ogmemory/.watch.pid"

# Create memory directory if needed
mkdir -p "$MEMORY_DIR"

# Get collection name (based on project path)
COLLECTION_NAME=$(python3 -c "import hashlib; import os; print(hashlib.sha256(os.getcwd().encode()).hexdigest()[:16])" 2>/dev/null || echo "ogmemory_chunks")

# Start watch process
start_watch() {
    local memory_path="$1"
    
    # Stop existing watch if running
    stop_watch
    
    # Start new watch process
    nohup python3 -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from ogmemory import MemoryEngine

async def watch_memory():
    from pathlib import Path

    engine = MemoryEngine()
    memory_dir = Path('$memory_path')
    
    print(f'Starting watch on {memory_dir}', flush=True)
    
    # Simple file watching (polling); auto-index on change
    last_mtimes = {}
    
    while True:
        try:
            for md_file in memory_dir.glob('**/*.md'):
                key = str(md_file)
                try:
                    current_mtime = md_file.stat().st_mtime
                except OSError:
                    continue
                if key not in last_mtimes or last_mtimes[key] != current_mtime:
                    print(f'File changed: {md_file}', flush=True)
                    try:
                        engine.delete_by_source(key)
                        await engine.index_file(key)
                        print(f'Indexed: {md_file}', flush=True)
                    except Exception as idx_err:
                        print(f'Index error: {idx_err}', flush=True)
                    last_mtimes[key] = current_mtime
        except Exception as e:
            print(f'Watch error: {e}', flush=True)
        
        await asyncio.sleep(5)
        
asyncio.run(watch_memory())
" > /dev/null 2>&1 &
    
    local pid=$!
    echo "$pid" > "$WATCH_PID_FILE"
    echo "Started watch process (PID: $pid)"
}

# Stop watch process
stop_watch() {
    if [ -f "$WATCH_PID_FILE" ]; then
        local pid=$(cat "$WATCH_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "Stopped watch process (PID: $pid)"
        fi
        rm -f "$WATCH_PID_FILE"
    fi
}

# Run oG-Memory command（在项目根目录执行，保证能找到 ogmemory_cli.py）
run_ogmemory() {
    (cd "$OGMEMORY_PROJECT_ROOT" && $PYTHON_CMD ogmemory_cli.py "$@" 2>/dev/null)
}
