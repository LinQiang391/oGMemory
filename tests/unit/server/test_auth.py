"""Tests for V1 auth service and key manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.models import Role
from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.auth import AuthService, AuthenticationError
from server.control_plane_store import ControlPlaneStore


@pytest.fixture()
def store(tmp_path: Path):
    return ControlPlaneStore(mount_prefix="", local_root=str(tmp_path))


@pytest.fixture()
def cfg():
    return OgMemConfig(
        role_control_enabled=True,
        root_api_key="root-key",
        admin_api_keys=["admin-key"],
        account_id="acct-default",
        user_id="u-default",
        agent_id="a-default",
    )


def test_resolve_root_key(cfg, store):
    auth = AuthService(cfg, APIKeyManager(store))

    identity = auth.resolve_identity({"X-API-Key": "root-key", "X-Account-ID": "acct-1", "X-User-ID": "rooter"})

    assert identity is not None
    assert identity.role == Role.ROOT
    assert identity.account_id == "acct-1"
    assert identity.user_id == "rooter"


def test_resolve_admin_key_from_config(cfg, store):
    auth = AuthService(cfg, APIKeyManager(store))

    identity = auth.resolve_identity({"Authorization": "Bearer admin-key", "X-Account-ID": "acct-2"})

    assert identity is not None
    assert identity.role == Role.ADMIN
    assert identity.account_id == "acct-2"


def test_resolve_member_key_from_users_json(cfg, store):
    manager = APIKeyManager(store)
    manager.create_account("acct-1", "alice")
    member_key = manager.register_user("acct-1", "bob", "user")
    auth = AuthService(cfg, manager)

    identity = auth.resolve_identity({"X-API-Key": member_key})

    assert identity is not None
    assert identity.role == Role.MEMBER
    assert identity.account_id == "acct-1"
    assert identity.user_id == "bob"


def test_invalid_key_raises(cfg, store):
    auth = AuthService(cfg, APIKeyManager(store))

    with pytest.raises(AuthenticationError):
        auth.resolve_identity({"X-API-Key": "missing"})
