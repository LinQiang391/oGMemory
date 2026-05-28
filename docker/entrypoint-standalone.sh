#!/bin/sh
set -e

# --- 0. Resolve ${ENV_VAR} references in config and patch connection_string ---
CONFIG_SRC="${OGMEM_CONFIG:-/etc/ogmem/config.yaml}"
CONFIG_WORK="/tmp/ogmem.resolved.yaml"

python3 << 'PYEOF'
import re, os, sys

path_src = os.environ.get("OGMEM_CONFIG", "/etc/ogmem/config.yaml")
path_dst = "/tmp/ogmem.resolved.yaml"

with open(path_src) as f:
    content = f.read()

def replace_env(m):
    var = m.group(1)
    val = os.environ.get(var)
    if val is not None:
        return val
    return m.group(0)

content = re.sub(r'\$\{(\w+)\}', replace_env, content)

conn_parts = []
for key in ("OPENGAUSS_HOST", "OPENGAUSS_PORT", "OPENGAUSS_DBNAME",
            "OPENGAUSS_USER", "OPENGAUSS_PASSWORD"):
    val = os.environ.get(key)
    if val:
        conn_parts.append(val)

if conn_parts:
    host = os.environ.get("OPENGAUSS_HOST", "127.0.0.1")
    port = os.environ.get("OPENGAUSS_PORT", "5432")
    dbname = os.environ.get("OPENGAUSS_DBNAME", "postgres")
    user = os.environ.get("OPENGAUSS_USER", "gaussdb")
    password = os.environ.get("OPENGAUSS_PASSWORD", "")
    conn = f"host={host} port={port} dbname={dbname} user={user} password={password}"
    content = re.sub(
        r'connection_string:\s*".*?"',
        f'connection_string: "{conn}"',
        content,
    )
    print(f"[entrypoint] Patched vector_db.connection_string: {host}:{port}/{dbname}")

with open(path_dst, "w") as f:
    f.write(content)

print(f"[entrypoint] Config resolved: {path_src} -> {path_dst}")
PYEOF

export OGMEM_CONFIG="${CONFIG_WORK}"

# --- 1. Load resolved YAML config into shell env (port / workers) ---
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
# cd ensures perf_logs/ and other relative paths resolve under /opt/ogmem (writable by ogmem user)
cd /opt/ogmem
OGMEM_PORT=${OGMEM_HTTP_PORT:-8090}
echo "[entrypoint] Starting oGMemory HTTP server on :${OGMEM_PORT}..."
exec python3 -m gunicorn \
  --bind "0.0.0.0:${OGMEM_PORT}" \
  --workers "${OGMEM_WORKERS:-2}" \
  --timeout "${OGMEM_GUNICORN_TIMEOUT:-900}" \
  --access-logfile - \
  server.app:app
