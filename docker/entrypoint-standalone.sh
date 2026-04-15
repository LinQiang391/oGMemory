#!/bin/sh
set -e

# --- 0. Load YAML config into shell env (port / workers) ---
eval $(python3 /opt/ogmem/scripts/yaml_to_env.py 2>/dev/null) || true

# --- 1. Start AGFS Server (background) ---
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

# --- 2. Start IndexService (background) ---
echo "[entrypoint] Starting IndexService..."
python3 /opt/ogmem/scripts/run_index_service.py &
INDEX_PID=$!

# --- 3. Trap signals for graceful shutdown ---
cleanup() {
  echo "[entrypoint] Shutting down..."
  kill $INDEX_PID $AGFS_PID 2>/dev/null
  wait $INDEX_PID $AGFS_PID 2>/dev/null
  exit 0
}
trap cleanup TERM INT

# --- 4. Start oG-Memory HTTP API (foreground, main process) ---
OGMEM_PORT=${OGMEM_HTTP_PORT:-8090}
echo "[entrypoint] Starting oG-Memory HTTP server on :${OGMEM_PORT}..."
exec python3 -m gunicorn \
  --bind "0.0.0.0:${OGMEM_PORT}" \
  --workers "${OGMEM_WORKERS:-2}" \
  --timeout "${OGMEM_GUNICORN_TIMEOUT:-900}" \
  --access-logfile - \
  server.app:app
