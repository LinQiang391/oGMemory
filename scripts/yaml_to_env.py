#!/usr/bin/env python3
"""Read ogmemory YAML config and print shell export statements.

Used by entrypoint-standalone.sh to make service.http_port and service
concurrency available before gunicorn starts.

Usage (in shell):
    eval $(python3 /opt/ogmem/scripts/yaml_to_env.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from providers.unified_config import OgMemConfig
except Exception:
    sys.exit(0)

try:
    cfg = OgMemConfig.load()
except Exception:
    sys.exit(0)

print(f'export OGMEM_HTTP_PORT="{cfg.http_port}"')
print(f'export OGMEM_WORKERS="{cfg.workers}"')
print(f'export OGMEM_THREADS="{cfg.threads}"')
print(f'export OGMEM_MCP_ENABLED="{str(cfg.mcp_enabled).lower()}"')
print(f'export OGMEM_MCP_HOST="{cfg.mcp_host}"')
print(f'export OGMEM_MCP_PORT="{cfg.mcp_port}"')
print(f'export OGMEM_MCP_PATH="{cfg.mcp_path}"')
