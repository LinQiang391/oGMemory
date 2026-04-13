"""Tests for V1 auth and sharing config in OgMemConfig."""

from __future__ import annotations

from providers.unified_config import OgMemConfig


def test_load_reads_v1_auth_and_sharing_from_env(monkeypatch):
    monkeypatch.setenv("OG_ROLE_CONTROL_ENABLED", "true")
    monkeypatch.setenv("OG_ROOT_API_KEY", "root-key")
    monkeypatch.setenv("OG_ADMIN_API_KEYS", "admin-a,admin-b")
    monkeypatch.setenv("OG_AGENT_SHARED_MODE", "user")
    monkeypatch.setenv("OG_AGENT_SHARED_LIST", "agent-1,agent-2")

    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.role_control_enabled is True
    assert cfg.root_api_key == "root-key"
    assert cfg.admin_api_keys == ["admin-a", "admin-b"]
    assert cfg.agent_shared_mode == "user"
    assert cfg.agent_shared_list == ["agent-1", "agent-2"]


def test_load_uses_safe_defaults_for_v1_auth_and_sharing():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.role_control_enabled is False
    assert cfg.root_api_key is None
    assert cfg.admin_api_keys == []
    assert cfg.agent_shared_mode == "off"
    assert cfg.agent_shared_list == []
