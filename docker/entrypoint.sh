#!/bin/sh
set -e

# --- 1. Prepare openclaw.json if not mounted ---
OPENCLAW_HOME="${HOME}/.openclaw"
mkdir -p "$OPENCLAW_HOME"
if [ ! -f "$OPENCLAW_HOME/openclaw.json" ]; then
  cp /opt/defaults/openclaw.json "$OPENCLAW_HOME/openclaw.json"
  echo "[entrypoint] Copied default openclaw.json to $OPENCLAW_HOME/"
fi

# --- 2. Inject API key into openclaw.json if OPENAI_API_KEY is set ---
if [ -n "$OPENAI_API_KEY" ] && grep -q '__OPENAI_API_KEY__' "$OPENCLAW_HOME/openclaw.json"; then
  sed -i "s|__OPENAI_API_KEY__|${OPENAI_API_KEY}|g" "$OPENCLAW_HOME/openclaw.json"
  echo "[entrypoint] Injected OPENAI_API_KEY into openclaw.json"
fi
if [ -n "$OPENAI_LLM_MODEL" ] && grep -q '__LLM_MODEL__' "$OPENCLAW_HOME/openclaw.json"; then
  sed -i "s|__LLM_MODEL__|${OPENAI_LLM_MODEL}|g" "$OPENCLAW_HOME/openclaw.json"
  echo "[entrypoint] Injected LLM model=${OPENAI_LLM_MODEL} into openclaw.json"
fi

# --- 3. Start AGFS Server (background) ---
mkdir -p /data/agfs
echo "[entrypoint] Starting AGFS server..."
/opt/agfs/agfs-server -c /opt/agfs/config.yaml &
AGFS_PID=$!

for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:1833/serverinfo/version >/dev/null 2>&1; then
    echo "[entrypoint] AGFS server ready"
    break
  fi
  sleep 0.5
done

# --- 4. Start IndexService (background) ---
echo "[entrypoint] Starting IndexService..."
python3 /opt/ogmem/scripts/run_index_service.py &
INDEX_PID=$!

# --- 5. Trap signals for graceful shutdown ---
cleanup() {
  echo "[entrypoint] Shutting down..."
  kill $INDEX_PID $AGFS_PID 2>/dev/null
  wait $INDEX_PID $AGFS_PID 2>/dev/null
  exit 0
}
trap cleanup TERM INT

# --- 6. Start OpenClaw (foreground, main process) ---
echo "[entrypoint] Starting OpenClaw gateway..."
exec node /app/openclaw.mjs gateway --allow-unconfigured "$@"
