#!/bin/sh
set -e

OC_HOME="/home/node/.openclaw"
OC_CONFIG="$OC_HOME/openclaw.json"

mkdir -p "$OC_HOME"

# 如果用户没有挂载自己的 openclaw.json，则使用内置默认配置
if [ ! -f "$OC_CONFIG" ]; then
  echo "[entrypoint] No openclaw.json found, generating from defaults..."
  cp /opt/defaults/openclaw.json "$OC_CONFIG"

  # 注入环境变量（带默认值）
  LLM_PROVIDER="${LLM_PROVIDER:-dashscope}"
  LLM_MODEL="${LLM_MODEL:-glm-5}"
  LLM_BASE_URL="${LLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
  LLM_API_KEY="${LLM_API_KEY:-}"
  OGMEM_URL="${OGMEM_URL:-http://127.0.0.1:8090}"
  GATEWAY_PORT="${GATEWAY_PORT:-18789}"

  sed -i "s|__LLM_PROVIDER__|${LLM_PROVIDER}|g" "$OC_CONFIG"
  sed -i "s|__LLM_MODEL__|${LLM_MODEL}|g" "$OC_CONFIG"
  sed -i "s|__LLM_BASE_URL__|${LLM_BASE_URL}|g" "$OC_CONFIG"
  sed -i "s|__LLM_API_KEY__|${LLM_API_KEY}|g" "$OC_CONFIG"
  sed -i "s|__OGMEM_URL__|${OGMEM_URL}|g" "$OC_CONFIG"
  sed -i "s|__GATEWAY_PORT__|${GATEWAY_PORT}|g" "$OC_CONFIG"

  echo "[entrypoint] Config generated: provider=${LLM_PROVIDER}, model=${LLM_MODEL}, ogmem=${OGMEM_URL}, port=${GATEWAY_PORT}"
else
  echo "[entrypoint] Using existing openclaw.json"
fi

# 启动 OpenClaw Gateway
exec node /app/openclaw.mjs gateway --allow-unconfigured
