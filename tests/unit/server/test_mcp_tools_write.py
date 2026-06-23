from __future__ import annotations

import pytest

from core.models import RequestContext, Role
from providers.unified_config import OgMemConfig
from server.mcp_tools import write_memory


class _FakeAuth:
    def role_control_active(self):
        return False


class _FakeAudit:
    def __init__(self):
        self.records = []

    def record(self, account_id, **kwargs):
        self.records.append({"account_id": account_id, **kwargs})


class _FakeService:
    def __init__(self):
        self._cfg = OgMemConfig()
        self.after_turn_params = None
        self.audit = _FakeAudit()
        self.after_turn_result = {"ok": True, "status": "accumulating", "pending_tokens": 12}

    def get_auth_service(self):
        return _FakeAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="alice",
            agent_id=params.get("agentId") or "bot",
            session_id=params.get("sessionId") or "unknown",
            trace_id="t1",
            role=Role.MEMBER,
        )

    def after_turn(self, params):
        self.after_turn_params = params
        return self.after_turn_result

    def get_audit_service(self):
        return self.audit


def test_write_memory_submits_text_to_after_turn_flow():
    service = _FakeService()

    payload = write_memory(
        service,
        text="Remember that MCP writes should use the server-side flow.",
        session_id="s1",
        agent_id="codex",
        client_request_id="req-1",
    )

    assert payload["ok"] is True
    assert payload["status"] == "accumulating"
    assert payload["accepted_messages"] == 1
    assert service.after_turn_params["sessionId"] == "s1"
    assert service.after_turn_params["agentId"] == "codex"
    assert service.after_turn_params["clientRequestId"] == "req-1"
    assert service.after_turn_params["messages"] == [
        {"role": "user", "content": "Remember that MCP writes should use the server-side flow."}
    ]
    assert service.after_turn_params["_ctx"].user_id == "alice"
    assert service.audit.records[0]["action"] == "mcp.write_memory"


def test_write_memory_submits_messages_to_after_turn_flow():
    service = _FakeService()

    write_memory(
        service,
        messages=[
            {"role": "user", "content": "Use resources for memory type browsing.", "id": "m1"},
            {"role": "assistant", "content": "Acknowledged.", "created_at": "2026-06-23T00:00:00Z"},
        ],
        session_id="s1",
    )

    assert service.after_turn_params["messages"] == [
        {"role": "user", "content": "Use resources for memory type browsing.", "id": "m1"},
        {"role": "assistant", "content": "Acknowledged.", "created_at": "2026-06-23T00:00:00Z"},
    ]


def test_write_memory_does_not_return_internal_write_result_uris():
    service = _FakeService()
    service.after_turn_result = {
        "ok": True,
        "status": "completed",
        "plans": [{"target_uri": "ctx://acct-1/users/alice/memories/preferences/style"}],
        "write_result": {"target_uri": "ctx://acct-1/users/alice/memories/preferences/style"},
        "pending_tokens": 0,
    }

    payload = write_memory(service, text="Remember this safely.", session_id="s1")

    assert payload["status"] == "completed"
    assert payload["stats"]["pending_tokens"] == 0
    assert "ctx://" not in str(payload)


def test_write_memory_requires_session_id():
    with pytest.raises(ValueError, match="session_id is required"):
        write_memory(_FakeService(), text="hello")


def test_write_memory_requires_exactly_one_material_source():
    with pytest.raises(ValueError, match="either text or messages"):
        write_memory(
            _FakeService(),
            text="hello",
            messages=[{"role": "user", "content": "hello"}],
            session_id="s1",
        )


@pytest.mark.parametrize("field", ["type", "memory_id", "abstract", "overview", "uri"])
def test_write_memory_rejects_atomic_memory_fields(field):
    with pytest.raises(ValueError, match="cannot set atomic memory fields"):
        write_memory(
            _FakeService(),
            messages=[{"role": "user", "content": "hello", field: "forbidden"}],
            session_id="s1",
        )
