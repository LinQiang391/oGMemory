"""HTTP coverage for the public code RESOURCE lifecycle routes."""

from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest

from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from service.code_workspace_sync import ResolvedCodeWorkspace

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("flask") is None,
    reason="flask is not installed",
)


def test_http_public_lifecycle_routes_delegate_invalidate_cache_and_wake_worker(monkeypatch, tmp_path):
    import server.app as app_module
    import server.memory_service as memory_service_module

    root = tmp_path / "repo"
    root.mkdir()
    service = MemoryService(
        config=OgMemConfig(
            provider="mock",
            vector_db_type="memory",
            role_control_enabled=True,
            root_api_key="root-key",
            code_resources_enabled=True,
            index_workers=0,
            account_id="acct",
            user_id="user",
            agent_id="agent",
        )
    )
    lifecycle = Mock()
    lifecycle.refresh.return_value = {"status": "PENDING", "uri": "ctx://acct/resources/file"}
    service._get_code_resource_lifecycle_service = Mock(return_value=lifecycle)
    service.invalidate_code_retrieval_cache = Mock()
    service._start_outbox_worker = Mock()
    resolver = Mock()
    resolver.resolve_registered.return_value = ResolvedCodeWorkspace(
        root_id="root-1",
        root_slug="repo",
        root_fingerprint="f" * 64,
        physical_root=root,
        max_files=100,
        language_hints={},
    )
    monkeypatch.setattr(
        memory_service_module,
        "CodeWorkspaceRegistryResolver",
        lambda *args, **kwargs: resolver,
    )
    monkeypatch.setattr(app_module, "_service", service)
    client = app_module.app.test_client()

    refresh = client.post(
        "/api/v1/resources/code/refresh",
        json={"path": "a.py", "root_slug": "repo"},
        headers={"X-API-Key": "root-key"},
    )
    assert refresh.status_code == 200
    assert refresh.get_json()["status"] == "PENDING"
    assert lifecycle.refresh.call_args.kwargs["ctx"].account_id == "acct"
    assert lifecycle.refresh.call_args.kwargs["workspace_root"] == str(root)
    assert lifecycle.refresh.call_args.kwargs["name"] == "repo"
    assert service.invalidate_code_retrieval_cache.call_count == 1
    assert service._start_outbox_worker.call_count == 1


def test_code_bootstrap_is_not_a_public_route(monkeypatch):
    import server.app as app_module

    service = MemoryService(
        config=OgMemConfig(
            provider="mock",
            vector_db_type="memory",
            role_control_enabled=False,
        )
    )
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post(
        "/api/v1/resources/code/bootstrap",
        json={"path": "/srv/repo"},
    )

    assert response.status_code == 404


def test_existing_bootstrap_placeholder_route_is_unchanged(monkeypatch):
    import server.app as app_module

    service = MemoryService(
        config=OgMemConfig(
            provider="mock",
            vector_db_type="memory",
            role_control_enabled=False,
        )
    )
    monkeypatch.setattr(app_module, "_service", service)

    response = app_module.app.test_client().post("/api/v1/bootstrap", json={})

    assert response.status_code == 200
    assert response.get_json() == {"bootstrapped": True}
