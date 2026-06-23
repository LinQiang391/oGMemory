from __future__ import annotations

import pytest

from core.models import RequestContext, Role
from server.auth import AuthorizationError
from server.mcp_security import MCPVisibilityError
from server.mcp_tools import delete_memory


class _FakeAuth:
    def role_control_active(self):
        return False


class _FakeService:
    def __init__(self):
        self.forget_call = None
        self.raise_on_forget = None

    def get_auth_service(self):
        return _FakeAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="alice",
            agent_id=params.get("agentId") or "bot",
            session_id=params.get("sessionId") or "s1",
            trace_id="t1",
            role=Role.MEMBER,
        )

    def forget_memory(self, uri, ctx, **kwargs):
        self.forget_call = {"uri": uri, "ctx": ctx, **kwargs}
        if self.raise_on_forget:
            raise self.raise_on_forget
        return {"ok": True, "deleted": True, "mode": "archive", "uri": kwargs.get("external_uri")}


def test_delete_memory_maps_external_uri_and_forgets():
    service = _FakeService()

    payload = delete_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        reason="stale",
        session_id="s1",
        client_request_id="req-1",
    )

    assert payload == {
        "ok": True,
        "deleted": True,
        "mode": "archive",
        "uri": "ogmemory://memories/preference/coding_style",
        "type": "preference",
        "id": "coding_style",
        "client_request_id": "req-1",
    }
    assert service.forget_call["uri"] == "ctx://acct-1/users/alice/memories/preferences/coding_style"
    assert service.forget_call["external_uri"] == "ogmemory://memories/preference/coding_style"
    assert service.forget_call["reason"] == "stale"


def test_delete_memory_requires_item_uri():
    with pytest.raises(ValueError, match="single ogmemory"):
        delete_memory(_FakeService(), uri="ogmemory://memories/preference")


def test_delete_memory_maps_permission_errors_to_not_found_or_not_visible():
    service = _FakeService()
    service.raise_on_forget = AuthorizationError("denied")

    with pytest.raises(MCPVisibilityError, match="not_found_or_not_visible"):
        delete_memory(service, uri="ogmemory://memories/preference/coding_style")
