"""Tests for V1 auth wiring on HTTP entrypoints."""

from __future__ import annotations

from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthService
from server.control_plane_store import ControlPlaneStore
from server.memory_service import MemoryService
from server.tenant_admin import TenantAdminService
import server.app as app_module


def _make_service(tmp_path):
    cfg = OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
    )
    service = MemoryService(config=cfg)
    service._control_store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    service._key_manager = APIKeyManager(service._control_store)
    service._audit = AuditService(service._control_store)
    service._tenant_admin = TenantAdminService(service._key_manager, service._control_store, service._audit)
    service._auth = AuthService(cfg, service._key_manager)
    service.get_key_manager().create_account("acct-1", "alice")
    member_key = service.get_key_manager().register_user("acct-1", "bob", "user")
    service.compose = lambda params: {
        "ok": True,
        "account": params["_ctx"].account_id,
        "user": params["_ctx"].user_id,
        "agent": params["_ctx"].agent_id,
    }
    return service, member_key


def test_data_route_requires_api_key_when_role_control_enabled(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post("/api/v1/compose", json={"messages": []})

    assert resp.status_code == 401


def test_data_route_uses_trusted_identity_over_request_fields(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={
            "messages": [],
            "accountId": "evil",
            "userId": "mallory",
            "agentId": "agent-x",
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["account"] == "acct-1"
    assert payload["user"] == "bob"
    assert payload["agent"] == "agent-x"
