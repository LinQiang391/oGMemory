"""Tests for V1 tenant admin service."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.models import RequestContext, Role
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.control_plane_store import ControlPlaneStore
from server.tenant_admin import TenantAdminService


@pytest.fixture()
def services(tmp_path: Path):
    store = ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))
    key_manager = APIKeyManager(store)
    audit = AuditService(store)
    admin = TenantAdminService(key_manager, store, audit)
    key_manager.create_account("acct-1", "alice")
    root_ctx = RequestContext("acct-1", "root", "agent", "s", "t", role=Role.ROOT)
    admin_ctx = RequestContext("acct-1", "alice", "agent", "s", "t", role=Role.ADMIN)
    member_ctx = RequestContext("acct-1", "bob", "agent", "s", "t", role=Role.MEMBER)
    return key_manager, audit, admin, root_ctx, admin_ctx, member_ctx


def test_list_accounts_for_root(services):
    _, _, admin, root_ctx, _, _ = services

    result = admin.list_accounts(root_ctx)

    assert result["accounts"][0]["account_id"] == "acct-1"


def test_create_user_and_list_roles(services):
    _, _, admin, _, admin_ctx, _ = services

    created = admin.create_user(admin_ctx, "acct-1", "bob", "user")
    roles = admin.list_roles(admin_ctx, "acct-1")

    assert created["user_id"] == "bob"
    assert {"user_id": "bob", "role": "MEMBER"} in roles["roles"]


def test_set_role_requires_root(services):
    _, _, admin, _, _, member_ctx = services

    with pytest.raises(PermissionError):
        admin.set_role(member_ctx, "acct-1", "alice", "admin")


def test_create_and_update_agent_writes_audit(services):
    _, audit, admin, _, admin_ctx, _ = services

    created = admin.create_agent(admin_ctx, "acct-1", "agent-x", owner_user_id="alice")
    updated = admin.update_agent(admin_ctx, "acct-1", "agent-x", owner_user_id="bob")
    logs = audit.list_logs("acct-1")

    assert created["owner_user_id"] == "alice"
    assert updated["owner_user_id"] == "bob"
    assert any(row["action"] == "agent_created" for row in logs)
    assert any(row["action"] == "agent_updated" for row in logs)
