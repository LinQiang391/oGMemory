from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_example_deploy_enables_auth_when_mcp_http_is_enabled():
    cfg = yaml.safe_load((ROOT / "deploy" / "ogmemory.example.yaml").read_text(encoding="utf-8"))

    assert cfg["mcp"]["enabled"] is True
    assert cfg["auth"]["role_control_enabled"] is True
    assert cfg["auth"]["root_api_key"] == "${OG_ROOT_API_KEY}"
