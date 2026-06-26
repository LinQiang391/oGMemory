"""Tests for trusted RequestContext construction in MemoryService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

import server.memory_service as memory_service_module
from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthService
from server.control_plane_store import ControlPlaneStore
from server.memory_service import MemoryService
from server.tenant_admin import TenantAdminService


def test_build_context_uses_resolved_identity(tmp_path: Path):
    cfg = OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
        agfs_mount_prefix="",
    )
    service = MemoryService(config=cfg)
    service._control_store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    service._key_manager = APIKeyManager(service._control_store)
    service._auth = AuthService(cfg, service._key_manager)
    service._audit = AuditService(service._control_store)
    service._tenant_admin = TenantAdminService(service._key_manager, service._control_store, service._audit)
    service.get_key_manager().create_account("acct-1", "owner")
    service.get_tenant_admin_service().create_agent(
        service.get_auth_service().build_request_context(
            service.get_auth_service().resolve_identity({"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"}),
            account_id="acct-1",
            user_id="root",
            agent_id="agent-x",
            session_id="s0",
        ),
        "acct-1",
        "agent-x",
        owner_user_id="owner",
    )
    member_key = service.get_key_manager().register_user("acct-1", "bob", "user")
    identity = service.get_auth_service().resolve_identity({"X-API-Key": member_key})

    ctx = service.build_context(
        {"accountId": "evil", "userId": "mallory", "sessionId": "s1"},
        identity=identity,
    )

    assert ctx.account_id == "acct-1"
    assert ctx.user_id == "bob"
    assert ctx.agent_id == ""
    assert ctx.session_id == "s1"
    assert "user:bob" in ctx.visible_owner_spaces


def test_build_context_root_identity_does_not_require_visible_agent_lookup():
    cfg = OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
    )
    service = MemoryService(config=cfg)
    service._tenant_admin = Mock()
    service._tenant_admin.list_visible_agent_ids.side_effect = AssertionError(
        "root context should not enumerate visible agents"
    )
    identity = service.get_auth_service().resolve_identity(
        {"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"}
    )

    ctx = service.build_context({"sessionId": "s1"}, identity=identity)

    assert ctx.account_id == "acct-1"
    assert ctx.user_id == "root"
    assert ctx.visible_owner_spaces == ()


def test_build_context_includes_shared_agents_in_visible_spaces(tmp_path: Path):
    cfg = OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
        agent_shared_mode="user",
        agent_shared_list=["shared-agent"],
        agfs_mount_prefix="",
    )
    service = MemoryService(config=cfg)
    service._control_store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    service._key_manager = APIKeyManager(service._control_store)
    service._auth = AuthService(cfg, service._key_manager)
    service._audit = AuditService(service._control_store)
    service._tenant_admin = TenantAdminService(service._key_manager, service._control_store, service._audit)
    service.get_key_manager().create_account("acct-1", "owner")
    root_identity = service.get_auth_service().resolve_identity({"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"})
    root_ctx = service.get_auth_service().build_request_context(
        root_identity,
        account_id="acct-1",
        user_id="root",
        agent_id="shared-agent",
        session_id="s0",
    )
    service.get_tenant_admin_service().create_agent(root_ctx, "acct-1", "shared-agent", owner_user_id="owner")
    member_key = service.get_key_manager().register_user("acct-1", "bob", "user")
    identity = service.get_auth_service().resolve_identity({"X-API-Key": member_key})

    ctx = service.build_context({"sessionId": "s1"}, identity=identity)

    assert "user:bob" in ctx.visible_owner_spaces
    assert "agent:shared-agent" in ctx.visible_owner_spaces


def test_build_context_compat_mode_uses_request_defaults():
    cfg = OgMemConfig(
        role_control_enabled=False,
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
    )
    service = MemoryService(config=cfg)

    ctx = service.build_context({"accountId": "acct-1", "userId": "bob", "agentId": "agent-x", "sessionId": "s1"})

    assert ctx.account_id == "acct-1"
    assert ctx.user_id == "bob"
    assert ctx.agent_id == "agent-x"
    assert ctx.session_id == "s1"


def test_build_context_rejects_explicit_unowned_agent(tmp_path: Path):
    cfg = OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
        agfs_mount_prefix="",
    )
    service = MemoryService(config=cfg)
    service._control_store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    service._key_manager = APIKeyManager(service._control_store)
    service._auth = AuthService(cfg, service._key_manager)
    service._audit = AuditService(service._control_store)
    service._tenant_admin = TenantAdminService(service._key_manager, service._control_store, service._audit)
    service.get_key_manager().create_account("acct-1", "owner")
    service.get_tenant_admin_service().create_agent(
        service.get_auth_service().build_request_context(
            service.get_auth_service().resolve_identity({"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "root"}),
            account_id="acct-1",
            user_id="root",
            agent_id="agent-private",
            session_id="s0",
        ),
        "acct-1",
        "agent-private",
        owner_user_id="owner",
    )
    member_key = service.get_key_manager().register_user("acct-1", "bob", "user")
    identity = service.get_auth_service().resolve_identity({"X-API-Key": member_key})

    with pytest.raises(PermissionError, match="agent access denied"):
        service.build_context(
            {"agentId": "agent-private", "sessionId": "s1"},
            identity=identity,
        )


def test_get_control_plane_store_falls_back_to_local_filesystem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    service = MemoryService(
        config=OgMemConfig(
            role_control_enabled=False,
            agfs_mount_prefix="",
        )
    )
    monkeypatch.setattr(memory_service_module, "_HAS_AGFS", False)
    monkeypatch.setattr(memory_service_module, "_HAS_SQL", False)
    monkeypatch.chdir(tmp_path)

    store = service.get_control_plane_store()

    assert isinstance(store, ControlPlaneStore)
    assert store._local_root == tmp_path / ".ogmem_control"
