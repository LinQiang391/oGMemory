"""Tests for V1 auth and sharing config in OgMemConfig."""

from __future__ import annotations

from providers.unified_config import OgMemConfig


def test_load_reads_v1_auth_and_sharing_from_env(monkeypatch):
    monkeypatch.setenv("OG_ROLE_CONTROL_ENABLED", "true")
    monkeypatch.setenv("OG_ROOT_API_KEY", "root-key")
    monkeypatch.setenv("OG_ADMIN_API_KEYS", "admin-a,admin-b")
    monkeypatch.setenv("OG_AGENT_SHARED_MODE", "user")
    monkeypatch.setenv("OG_AGENT_SHARED_LIST", "agent-1,agent-2")
    monkeypatch.setenv("OG_HTTP_IP_ALLOWLIST", "127.0.0.1,10.0.0.0/8")
    monkeypatch.setenv("OG_HTTP_IP_ALLOWLIST_TRUST_PROXY", "true")

    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.role_control_enabled is True
    assert cfg.root_api_key == "root-key"
    assert cfg.admin_api_keys == ["admin-a", "admin-b"]
    assert cfg.agent_shared_mode == "user"
    assert cfg.agent_shared_list == ["agent-1", "agent-2"]
    assert cfg.http_ip_allowlist == ["127.0.0.1", "10.0.0.0/8"]
    assert cfg.http_ip_allowlist_trust_proxy is True


def test_load_uses_safe_defaults_for_v1_auth_and_sharing():
    cfg = OgMemConfig.load("/tmp/non-existent-ogmem.yaml")

    assert cfg.role_control_enabled is False
    assert cfg.root_api_key is None
    assert cfg.admin_api_keys == []
    assert cfg.agent_shared_mode == "off"
    assert cfg.agent_shared_list == []
    assert cfg.http_ip_allowlist == []
    assert cfg.http_ip_allowlist_trust_proxy is False
