"""ogmem config command - manage configuration."""

from __future__ import annotations

import json
from pathlib import Path

from cli.lib.colors import CYAN, GREEN, NC, YELLOW, fail, ok
from cli.lib.process import find_project_root


def run(args) -> int:
    """Handle config subcommands."""
    cmd = getattr(args, "config_action", "show")
    if cmd == "show":
        from providers.unified_config import get_config
        try:
            cfg = get_config()
            print(f"\n{GREEN}oG-Memory Configuration{NC}\n")
            print(json.dumps(cfg.dump_summary(), indent=2, default=str))
            print()
            return 0
        except Exception as e:
            print(fail(f"Failed: {e}"))
            return 1
    elif cmd == "init":
        root = find_project_root()
        cfg_path = root / "ogmem.yaml"
        if cfg_path.exists():
            print(fail(f"Config exists: {cfg_path}"))
            return 1
        tmpl = """# oG-Memory Configuration
# Env vars (OGMEM_*) override YAML values

paths:
  data_root: .ogmem_data

llm:
  provider: openai
  api_key: "YOUR_KEY"  # Or OGMEM_API_KEY
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
  temperature: 0.1

embedding:
  provider: st
  st_model: "BAAI/bge-m3"

vector_db:
  type: chroma
  chroma_persist_dir: .ogmem_data/chroma

service:
  http_port: 8090
  workers: 2

agfs:
  base_url: "http://127.0.0.1:1833"
"""
        try:
            cfg_path.write_text(tmpl)
            print(ok(f"Created: {CYAN}{cfg_path}{NC}\n  {YELLOW}Set your API keys{NC}\n"))
            return 0
        except IOError as e:
            print(fail(f"Failed: {e}"))
            return 1
    print(fail(f"Unknown subcommand: {cmd}"))
    return 1
