from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("flask")

import server.app as app_module
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def _make_service() -> MemoryService:
    return MemoryService(
        config=OgMemConfig(
            role_control_enabled=False,
            account_id="acct-test",
            user_id="user-test",
            agent_id="agent-test",
        )
    )


def test_prepare_compaction_http_route_returns_prepare_token(monkeypatch):
    service = _make_service()
    service.prepare_compaction = Mock(return_value={
        "prepareToken": "prepare-123",
        "messages": [{"role": "user", "content": "hello"}],
    })
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post("/api/v1/prepare_compaction", json={"sessionId": "sess-prepare"})

    assert resp.status_code == 200
    assert resp.get_json() == {
        "ok": True,
        "prepared": True,
        "prepareToken": "prepare-123",
        "messagesPrepared": 1,
    }


def test_compact_http_route_rejects_prepared_without_token(monkeypatch):
    service = _make_service()
    service.get_session_manager().add_message(
        "sess-compact",
        "user",
        "pending content",
        service.build_context({"sessionId": "sess-compact"}),
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compact",
        json={"sessionId": "sess-compact", "prepared": True},
    )

    assert resp.status_code == 400
    assert "prepare token required" in resp.get_json()["error"]
