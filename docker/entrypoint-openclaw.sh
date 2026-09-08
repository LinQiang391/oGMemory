#!/bin/sh
set -e

# Disable Bonjour / mDNS by default. With network_mode: host the openclaw
# container shares the host's interface list (eth0, wlan0, docker0, ...);
# the ciao probe packet that the responder fires at startup then includes
# an A/AAAA record per interface and blows past the 1440-byte UDP MTU
# (`AssertionError: Probe query packet exceeds the mtu size (2115>1440)`),
# which crashes the gateway. mDNS over a host-network container also serves
# no real purpose — the multicast stays on the host loopback and isn't
# discoverable from other devices. Operators who actually want LAN
# discovery can `unset OPENCLAW_DISABLE_BONJOUR` or run on bridge network.
export OPENCLAW_DISABLE_BONJOUR="${OPENCLAW_DISABLE_BONJOUR:-1}"

OC_HOME="/home/node/.openclaw"
BACKUP="/opt/ogmem-openclaw-defaults"
TEMPLATE="$BACKUP/openclaw.template.json"
CONFIG="$OC_HOME/openclaw.json"
BACKUP_KEEP_COUNT=3

backup_config() {
  if [ ! -f "$CONFIG" ]; then
    return 0
  fi

  timestamp=$(date +"%Y%m%d-%H%M%S")
  backup_path="${CONFIG}.bak.${timestamp}"
  cp "$CONFIG" "$backup_path"
  echo "Backed up existing config to $backup_path"
}

cleanup_old_backups() {
  old_backups=$(ls -1t "${CONFIG}.bak".* 2>/dev/null | tail -n +"$((BACKUP_KEEP_COUNT + 1))")
  if [ -n "$old_backups" ]; then
    echo "$old_backups" | while IFS= read -r backup_file; do
      [ -n "$backup_file" ] && rm -f "$backup_file"
    done
  fi
}

config_needs_refresh() {
  if [ ! -f "$CONFIG" ]; then
    return 0
  fi

  if grep -q "__LLM_PROVIDER__\|__LLM_MODEL__\|__LLM_BASE_URL__\|__LLM_API_KEY__\|__GATEWAY_PORT__\|__OGMEM_URL__\|__OG_AUTH_API_KEY__\|__OG_AUTH_ACCOUNT_ID__" "$CONFIG"; then
    return 0
  fi

  if ! grep -q '"authApiKey"' "$CONFIG"; then
    return 0
  fi

  if ! grep -q '"authAccountId"' "$CONFIG"; then
    return 0
  fi

  return 1
}

render_config() {
  local provider model base_url api_key gateway_port ogmem_url escaped_base_url escaped_api_key escaped_ogmem_url temp_config

  # Verify template exists before proceeding
  if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Config template missing at $TEMPLATE" >&2
    exit 1
  fi

  provider="${LLM_PROVIDER:-openai}"
  model="${LLM_MODEL:-gpt-4o-mini}"
  base_url="${LLM_BASE_URL:-https://api.openai.com/v1}"
  api_key="${LLM_API_KEY:-}"
  gateway_port="${GATEWAY_PORT:-18789}"
  ogmem_url="${OGMEM_URL:-${OGMEM_API_URL:-http://127.0.0.1:8090}}"
  og_auth_api_key="${OG_AUTH_API_KEY:-}"
  og_auth_account_id="${OG_AUTH_ACCOUNT_ID:-}"

  escaped_base_url=$(printf '%s' "$base_url" | sed 's/[\/&]/\\&/g')
  escaped_api_key=$(printf '%s' "$api_key" | sed 's/[\/&]/\\&/g')
  escaped_ogmem_url=$(printf '%s' "$ogmem_url" | sed 's/[\/&]/\\&/g')
  escaped_og_auth_api_key=$(printf '%s' "$og_auth_api_key" | sed 's/[\/&]/\\&/g')
  escaped_og_auth_account_id=$(printf '%s' "$og_auth_account_id" | sed 's/[\/&]/\\&/g')

  # Write to temp file first, then atomically rename
  temp_config="${CONFIG}.tmp"
  if ! sed \
    -e "s/__LLM_PROVIDER__/${provider}/g" \
    -e "s/__LLM_MODEL__/${model}/g" \
    -e "s/__LLM_BASE_URL__/${escaped_base_url}/g" \
    -e "s/__LLM_API_KEY__/${escaped_api_key}/g" \
    -e "s/\"__GATEWAY_PORT__\"/${gateway_port}/g" \
    -e "s/__OGMEM_URL__/${escaped_ogmem_url}/g" \
    -e "s/__OG_AUTH_API_KEY__/${escaped_og_auth_api_key}/g" \
    -e "s/__OG_AUTH_ACCOUNT_ID__/${escaped_og_auth_account_id}/g" \
    "$TEMPLATE" > "$temp_config"; then
    echo "ERROR: Failed to render config template" >&2
    rm -f "$temp_config" 2>/dev/null
    exit 1
  fi
  # Verify the rendered file is not empty
  if [ ! -s "$temp_config" ]; then
    echo "ERROR: Rendered config is empty" >&2
    rm -f "$temp_config" 2>/dev/null
    exit 1
  fi

  # Atomic rename to avoid partial configs
  mv "$temp_config" "$CONFIG"
  echo "Rendered config at $CONFIG"
}

if [ ! -f "$CONFIG" ]; then
  render_config
elif config_needs_refresh; then
  backup_config
  render_config
  cleanup_old_backups
fi

mkdir -p "$OC_HOME/extensions"
if [ ! -d "$OC_HOME/extensions/og-memory-context-engine" ]; then
  cp -r "$BACKUP/extensions/og-memory-context-engine" "$OC_HOME/extensions/"
fi

chown -R node:node "$OC_HOME"
chmod 755 "$OC_HOME"
chmod -R 755 "$OC_HOME/extensions"
chmod -R a+rwX "$OC_HOME/agents" 2>/dev/null || true
chmod a+rw "$CONFIG" 2>/dev/null || true

mkdir -p /tmp/ogmem_ws
chmod 777 /tmp/ogmem_ws

exec su -s /bin/sh node -c "umask 0000; exec openclaw $*"
