from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("flask")

import server.app as app_module
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def _make_service() -> MemoryService:
    service = MemoryService(
        config=OgMemConfig(
            role_control_enabled=False,
            account_id="acct-test",
            user_id="user-test",
            agent_id="agent-test",
        )
    )
    service._get_context_fs = Mock(return_value=None)
    return service


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


def test_session_working_set_http_route_lists_access_stats(monkeypatch):
    service = _make_service()
    mgr = service.get_session_manager()
    ctx = service.build_context({"sessionId": "sess-ws"})
    buf = mgr.get_or_create("sess-ws", ctx)
    buf.window_state.last_accessed_at = "2026-05-20T02:00:00+00:00"
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.get("/api/v1/session_working_set")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["sessions"][0]["session_id"] == "sess-ws"
    assert body["sessions"][0]["last_accessed_at"] == "2026-05-20T02:00:00+00:00"


def test_evict_idle_sessions_http_route_uses_last_accessed_at(monkeypatch):
    service = _make_service()
    mgr = service.get_session_manager()
    ctx = service.build_context({"sessionId": "active"})
    active = mgr.get_or_create("active", ctx)
    idle = mgr.get_or_create("idle", ctx)
    active.window_state.last_accessed_at = "2026-05-20T02:00:00+00:00"
    idle.window_state.last_accessed_at = "2026-05-20T01:00:00+00:00"
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/evict_idle_sessions",
        json={"maxIdleSeconds": 1800, "nowIso": "2026-05-20T02:00:01+00:00"},
    )

    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "evicted": ["idle"]}
    assert mgr.has_session("active") is True
    assert mgr.has_session("idle") is False
