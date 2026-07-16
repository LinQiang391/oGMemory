"""Tests for V1 auth wiring on HTTP entrypoints."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_FLASK = importlib.util.find_spec("flask") is not None
pytestmark = pytest.mark.skipif(not _HAS_FLASK, reason="flask is not installed")

from providers.unified_config import OgMemConfig
from server.api_keys import APIKeyManager
from server.audit import AuditService
from server.auth import AuthService
from server.control_plane_store import ControlPlaneStore
from server.memory_service import MemoryService
from server.tenant_admin import TenantAdminService

if _HAS_FLASK:
    import server.app as app_module
else:
    app_module = None


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


def _make_service_with_allowlist(tmp_path, allowlist, *, trust_proxy=False, trusted_proxies=None):
    cfg = OgMemConfig(
        role_control_enabled=False,
        account_id="acct-default",
        user_id="user-default",
        agent_id="agent-default",
        http_ip_allowlist=list(allowlist),
        http_ip_allowlist_trust_proxy=trust_proxy,
        http_trusted_proxies=list(trusted_proxies or []),
    )
    service = MemoryService(config=cfg)
    service.compose = lambda params: {"ok": True}
    return service


def test_data_route_requires_api_key_when_role_control_enabled(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post("/api/v1/compose", json={"messages": []})

    assert resp.status_code == 401


def test_session_working_set_route_requires_api_key_when_role_control_enabled(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.get("/api/v1/session_working_set")

    assert resp.status_code == 401


def test_evict_idle_sessions_route_requires_api_key_when_role_control_enabled(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post("/api/v1/evict_idle_sessions", json={"maxIdleSeconds": 1})

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
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["account"] == "acct-1"
    assert payload["user"] == "bob"


def test_data_route_rejects_explicit_unowned_agent(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={
            "messages": [],
            "agentId": "agent-x",
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 403


def test_trace_route_persists_canonical_non_boundary_event_without_finalizing(
    monkeypatch, tmp_path
):
    service, member_key = _make_service(tmp_path)
    calls = []
    service.add_session_trace = lambda params: calls.append(params) or {
        "ok": True,
        "event_id": "evt-1",
        "event_index": 4,
    }
    service.finalize_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("non-boundary events must not finalize a turn")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "timestamp": "2026-07-15T08:00:00Z",
            "event": {"event_type": "tool_start"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    assert calls[0]["_ctx"].account_id == "acct-1"
    assert calls[0]["eventId"] == "evt-1"
    assert calls[0]["hookType"] == "tool_start"


def test_trace_route_requires_api_key(monkeypatch, tmp_path):
    service, _ = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("unauthenticated trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "event": {"hook_type": "PreToolUse"},
        },
    )

    assert resp.status_code == 401


def test_trace_route_finalizes_stop_event(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    calls = []
    service.add_session_trace = lambda params: calls.append(params) or {
        "ok": True,
        "event_id": "evt-stop",
        "event_index": 12,
    }
    finalized = []
    service.finalize_session_trace = lambda params: finalized.append(params) or {
        "ok": True,
        "turn_id": "turn_sess_12",
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-stop",
            "session_id": "sess",
            "event": {"hook_type": "Stop", "event_type": "assistant_message"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    assert resp.get_json()["turn"]["turn_id"] == "turn_sess_12"
    assert calls[0]["hookType"] == "turn_end"
    assert calls[0]["event"] == {
        "event_type": "turn_end",
    }
    assert finalized[0]["stopEventIndex"] == 12


def test_trace_route_finalizes_canonical_turn_end_event(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    calls = []
    service.add_session_trace = lambda params: calls.append(params) or {
        "ok": True,
        "event_id": "evt-turn-end",
        "event_index": 13,
    }
    finalized = []
    service.finalize_session_trace = lambda params: finalized.append(params) or {
        "ok": True,
        "turn_id": "turn_sess_13",
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-turn-end",
            "session_id": "sess",
            "event": {"event_type": "turn_end"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 200
    assert calls[0]["hookType"] == "turn_end"
    assert resp.get_json()["turn"]["turn_id"] == "turn_sess_13"
    assert finalized[0]["stopEventIndex"] == 13


@pytest.mark.parametrize(
    ("retryable", "expected_status"), [(False, 400), (True, 503)]
)
def test_trace_route_maps_finalize_failure_retryability(
    monkeypatch, tmp_path, retryable, expected_status
):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: {
        "ok": True,
        "event_id": "evt-stop",
        "event_index": 12,
    }
    service.finalize_session_trace = lambda params: {
        "ok": False,
        "error": "finalize failed",
        "retryable": retryable,
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-stop",
            "session_id": "sess",
            "event": {"hook_type": "Stop"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == expected_status


def test_trace_route_returns_retryable_failure_when_store_is_unavailable(
    monkeypatch, tmp_path
):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: {
        "ok": False,
        "error": "trace store unavailable (sql backend required)",
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "event": {"hook_type": "PreToolUse"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 503


def test_trace_route_rejects_unowned_agent(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("unauthorized trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "agent_id": "unowned-agent",
            "event": {"hook_type": "PreToolUse"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 403


def test_trace_route_rejects_oversized_body_before_persistence(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: (_ for _ in ()).throw(
        AssertionError("oversized trace must not reach persistence")
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        data=b"x" * (64 * 1024 + 1),
        content_type="application/json",
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 413


def test_trace_route_maps_permanent_validation_to_400(monkeypatch, tmp_path):
    service, member_key = _make_service(tmp_path)
    service.add_session_trace = lambda params: {
        "ok": False,
        "error": "invalid event_time",
        "retryable": False,
    }
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/trace/events",
        json={
            "request_id": "evt-1",
            "session_id": "sess",
            "event": {"hook_type": "PreToolUse"},
        },
        headers={"X-API-Key": member_key},
    )

    assert resp.status_code == 400


def test_ip_allowlist_blocks_non_whitelisted_remote_addr(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(tmp_path, ["127.0.0.1"])
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 403
    assert resp.get_json()["error"] == "IP not allowed"


def test_ip_allowlist_allows_whitelisted_remote_addr(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(tmp_path, ["127.0.0.1", "10.0.0.0/8"])
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 200


def test_ip_allowlist_uses_forwarded_for_only_when_trusted(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(
        tmp_path, ["203.0.113.7"], trust_proxy=True, trusted_proxies=["10.10.10.10"]
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 200


def test_ip_allowlist_ignores_forwarded_for_from_untrusted_proxy(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(
        tmp_path, ["203.0.113.7"], trust_proxy=True, trusted_proxies=["10.0.0.0/8"]
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        headers={"X-Forwarded-For": "203.0.113.7"},
        environ_base={"REMOTE_ADDR": "198.51.100.20"},
    )

    assert resp.status_code == 403


def test_ip_allowlist_ignores_forwarded_headers_when_proxy_not_configured(monkeypatch, tmp_path):
    service = _make_service_with_allowlist(
        tmp_path, ["10.10.10.10"], trust_proxy=True, trusted_proxies=[]
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        headers={"X-Forwarded-For": "203.0.113.7"},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 200


def test_ip_allowlist_rejects_invalid_client_ip_with_log(monkeypatch, tmp_path, caplog):
    service = _make_service_with_allowlist(tmp_path, ["127.0.0.1"])
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        environ_base={"REMOTE_ADDR": "not-an-ip"},
    )

    assert resp.status_code == 403
    assert "Invalid client IP format" in caplog.text


def test_allowlist_rejection_does_not_initialize_service(monkeypatch, tmp_path):
    app_module._service = None
    monkeypatch.setattr(
        app_module,
        "get_config",
        lambda: OgMemConfig(http_ip_allowlist=["127.0.0.1"]),
    )

    def _boom():
        raise AssertionError("_get_service should not be called for rejected IPs")

    monkeypatch.setattr(app_module, "_get_service", _boom)
    client = app_module.app.test_client()

    resp = client.post(
        "/api/v1/compose",
        json={"messages": []},
        environ_base={"REMOTE_ADDR": "10.10.10.10"},
    )

    assert resp.status_code == 403
