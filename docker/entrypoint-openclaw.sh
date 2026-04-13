#!/bin/sh
set -e

OC_HOME="/home/node/.openclaw"
BACKUP="/opt/ogmem-openclaw-defaults"

if [ ! -f "$OC_HOME/openclaw.json" ]; then
  cp "$BACKUP/openclaw.json" "$OC_HOME/openclaw.json"
fi

mkdir -p "$OC_HOME/extensions"
if [ ! -d "$OC_HOME/extensions/og-memory-context-engine" ]; then
  cp -r "$BACKUP/extensions/og-memory-context-engine" "$OC_HOME/extensions/"
fi

chown -R node:node "$OC_HOME"
chmod 755 "$OC_HOME"
chmod -R 755 "$OC_HOME/extensions"
chmod -R a+rwX "$OC_HOME/agents" 2>/dev/null || true
chmod a+rw "$OC_HOME/openclaw.json" 2>/dev/null || true

exec su -s /bin/sh node -c "umask 0000; exec openclaw $*"
