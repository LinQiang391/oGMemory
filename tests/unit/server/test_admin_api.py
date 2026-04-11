"""Tests for V1 admin HTTP routes."""

from __future__ import annotations

from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.control_plane_store import ControlPlaneStore
from server.memory_service import MemoryService
from server.tenant_admin import TenantAdminService
import server.app as app_module


def _make_service(tmp_path):
    cfg = OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        admin_api_keys=[],
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
    )
    service = MemoryService(config=cfg)
    service._control_store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    service._key_manager = APIKeyManager(service._control_store)
    service._audit = AuditService(service._control_store)
    service._tenant_admin = TenantAdminService(service._key_manager, service._control_store, service._audit)
    service._auth = service.get_auth_service().__class__(cfg, service._key_manager)
    service.get_key_manager().create_account("acct-1", "alice")
    member_key = service.get_key_manager().register_user("acct-1", "bob", "user")
    return service, member_key


def test_admin_routes_enforce_member_forbidden(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.patch(
        "/api/v1/admin/accounts/acct-1/users/alice/role",
        json={"role": "admin"},
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 403


def test_admin_roles_route_for_root(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.get(
        "/api/v1/admin/accounts/acct-1/roles",
        headers={"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"},
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert any(row["user_id"] == "alice" for row in payload["roles"])
