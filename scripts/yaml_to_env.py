#!/usr/bin/env python3
"""Read ogmemory YAML config and print shell export statements.

Used by entrypoint-standalone.sh to make service.http_port and service
concurrency available before gunicorn starts.

Usage (in shell):
    eval $(python3 /opt/ogmem/scripts/yaml_to_env.py)
"""

import os
import sys
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from providers.unified_config import OgMemConfig
except Exception:
    print("Cannot import standalone configuration loader", file=sys.stderr)
    sys.exit(1)

try:
    cfg = OgMemConfig.load()
except Exception:
    print("Cannot load standalone configuration; refusing to start", file=sys.stderr)
    sys.exit(1)

print(f'export OGMEM_HTTP_PORT="{cfg.http_port}"')
# Only the legacy local default is managed by this image. External AGFS
# endpoints and SQL deployments must not spawn an unrelated local server.
agfs_url = urlsplit(cfg.agfs_base_url)
bundled_agfs = (
    cfg.storage_backend == "agfs"
    and agfs_url.scheme == "http"
    and agfs_url.hostname in {"127.0.0.1", "localhost"}
    and agfs_url.port == 1833
)
print(f'export OGMEM_START_BUNDLED_AGFS="{str(bundled_agfs).lower()}"')
print(f'export OGMEM_WORKERS="{cfg.workers}"')
print(f'export OGMEM_THREADS="{cfg.threads}"')
print(f'export OGMEM_MCP_ENABLED="{str(cfg.mcp_enabled).lower()}"')
print(f'export OGMEM_MCP_HOST="{cfg.mcp_host}"')
print(f'export OGMEM_MCP_PORT="{cfg.mcp_port}"')
print(f'export OGMEM_MCP_PATH="{cfg.mcp_path}"')
print(f'export OGMEM_MCP_ENABLE_CODE_EDIT_TOOLS="{str(cfg.mcp_enable_code_edit_tools).lower()}"')
