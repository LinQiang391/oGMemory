"""Focused MemoryService wiring for the tool-output filter facade."""

from __future__ import annotations

from types import SimpleNamespace

from core.models import RequestContext
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


class _FilterService:
    def __init__(self):
        self.calls: list[tuple[str, dict, RequestContext]] = []

    def filter_read(self, params, ctx):
        self.calls.append(("read", params, ctx))
        return {"ok": True, "filtered_output": "short"}

    def filter_bash(self, params, ctx):
        self.calls.append(("bash", params, ctx))
        return {"ok": True, "filtered_output": "short"}

    def recover_original(self, params, ctx):
        self.calls.append(("recover", params, ctx))
        return {"ok": True, "original": "full"}


def test_memory_service_delegates_tool_output_operations_with_bound_context(monkeypatch):
    service = MemoryService(
        config=OgMemConfig(
            role_control_enabled=False,
            account_id="acct",
            user_id="user",
            agent_id="agent",
        )
    )
    filter_service = _FilterService()
    monkeypatch.setattr(service, "_get_tool_output_filter_service", lambda: filter_service)
    monkeypatch.setattr(
        service,
        "get_auth_service",
        lambda: SimpleNamespace(role_control_active=lambda: False),
    )

    params = {"accountId": "acct", "userId": "user", "agentId": "agent"}
    assert service.filter_read({**params, "content": "source"})["ok"] is True
    assert service.filter_bash({**params, "output": "source"})["ok"] is True
    assert service.read_tool_output_original({**params, "memory_uri": "ctx://..."})["ok"] is True

    assert [call[0] for call in filter_service.calls] == ["read", "bash", "recover"]
    assert all(call[2].account_id == "acct" for call in filter_service.calls)
    assert all(call[2].agent_id == "agent" for call in filter_service.calls)
