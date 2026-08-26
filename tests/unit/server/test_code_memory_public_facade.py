"""Unit tests for thin public code-memory facade delegation.

The recording doubles replace the canonical sync, lifecycle, and retrieval
services.  These assertions prove request mapping only; they are not storage,
index, HTTP, or MCP integration evidence.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.models import RequestContext, Role
from providers.unified_config import OgMemConfig
from retrieval.code_retrieval import (
    ActiveCodeSearchOutcome,
    CodeEvidence,
)
from server.auth import AuthorizationError, ControlPlaneDisabledError
from server.memory_service import MemoryService
from service.code_workspace_sync import ResolvedCodeWorkspace


def _ctx(role: Role = Role.ADMIN) -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
        role=role,
    )


def _service() -> MemoryService:
    return MemoryService(
        config=OgMemConfig(
            provider="mock",
            vector_db_type="memory",
            code_resources_enabled=True,
            code_sync_enabled=True,
            code_retrieval_enabled=True,
            role_control_enabled=True,
            root_api_key="root-key",
        )
    )


def test_workspace_facades_delegate_once_with_the_trusted_context(monkeypatch):
    service = _service()
    calls = []

    class SyncService:
        def register(self, **kwargs):
            calls.append(("register", kwargs))
            return {"root_id": "root-1", "accepted": True, "dirty": True}

        def sync(self, **kwargs):
            calls.append(("sync", kwargs))
            return {"root_id": "root-1", "accepted": True, "dirty": False}

    monkeypatch.setattr(service, "_get_code_workspace_sync_service", lambda: SyncService())
    ctx = _ctx()

    registration = service.register_code_workspace(
        {
            "path": "/protected/workspace",
            "name": "repo",
            "trigger_mode": "manual",
            "wait_mode": "accepted",
            "language_hints": {"Dockerfile": "shell"},
            "_ctx": ctx,
        }
    )
    synced = service.sync_code_workspace({"root_id": "root-1", "_ctx": ctx})

    assert registration == {"root_id": "root-1", "accepted": True, "dirty": True}
    assert synced == {"root_id": "root-1", "accepted": True, "dirty": False}
    assert calls[0][0] == "register"
    assert calls[0][1]["ctx"] is ctx
    assert calls[0][1]["path"] == "/protected/workspace"
    assert calls[1] == (
        "sync",
        {"root_id": "root-1", "ctx": ctx, "wait_mode": None},
    )


def test_workspace_facades_require_a_mapping_result(monkeypatch):
    service = _service()

    class SyncService:
        def register(self, **kwargs):
            del kwargs
            return []

    monkeypatch.setattr(service, "_get_code_workspace_sync_service", lambda: SyncService())

    with pytest.raises(RuntimeError, match="code workspace registration returned an invalid result"):
        service.register_code_workspace({"path": "/protected/workspace", "_ctx": _ctx()})


def test_refresh_facade_resolves_registered_root_and_relative_path(monkeypatch, tmp_path):
    import server.memory_service as memory_service_module

    service = _service()
    root = tmp_path / "repo"
    root.mkdir()
    workspace = ResolvedCodeWorkspace(
        root_id="root-1",
        root_slug="repo",
        root_fingerprint="f" * 64,
        physical_root=root,
        max_files=100,
        language_hints={},
    )
    resolver_calls = []

    class Resolver:
        def resolve_registered(self, **kwargs):
            resolver_calls.append(kwargs)
            return workspace

    lifecycle_calls = []
    worker_starts = []
    monkeypatch.setattr(
        memory_service_module,
        "CodeWorkspaceRegistryResolver",
        lambda *args, **kwargs: Resolver(),
    )
    monkeypatch.setattr(
        service,
        "_get_code_resource_lifecycle_service",
        lambda: SimpleNamespace(
            refresh=lambda **kwargs: lifecycle_calls.append(kwargs) or {"status": "PENDING"}
        ),
    )
    monkeypatch.setattr(service, "_start_outbox_worker", lambda: worker_starts.append(True))

    result = service.refresh_code_resource(
        {
            "root_slug": "repo",
            "path": "src/service.py",
            "tags": ["code"],
            "description": "source",
            "_ctx": _ctx(),
        }
    )

    assert result == {"status": "PENDING"}
    assert resolver_calls == [{"ctx": lifecycle_calls[0]["ctx"], "root_slugs": ("repo",)}]
    assert lifecycle_calls == [
        {
            "path": "src/service.py",
            "workspace_root": str(root),
            "ctx": lifecycle_calls[0]["ctx"],
            "name": "repo",
            "tags": ["code"],
            "description": "source",
        }
    ]
    assert lifecycle_calls[0]["ctx"].role == Role.ADMIN
    assert worker_starts == [True]


def test_refresh_facade_rejects_caller_controlled_root_before_resolution(monkeypatch):
    import server.memory_service as memory_service_module

    service = _service()
    monkeypatch.setattr(
        memory_service_module,
        "CodeWorkspaceRegistryResolver",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected resolver")),
    )

    with pytest.raises(ValueError, match="workspace_root is not accepted"):
        service.refresh_code_resource(
            {
                "root_slug": "repo",
                "path": "src/service.py",
                "workspace_root": "/",
                "_ctx": _ctx(),
            }
        )

    with pytest.raises(ValueError, match="relative path"):
        service.refresh_code_resource(
            {"root_slug": "repo", "path": "/etc/passwd", "_ctx": _ctx()}
        )


def test_workspace_mutation_facades_require_active_admin_auth(monkeypatch):
    service = _service()
    monkeypatch.setattr(
        service,
        "_get_code_workspace_sync_service",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected sync service")),
    )

    with pytest.raises(AuthorizationError, match="Operation requires role"):
        service.register_code_workspace(
            {"path": "/protected/workspace", "_ctx": _ctx(Role.MEMBER)}
        )

    inactive = MemoryService(
        config=OgMemConfig(provider="mock", vector_db_type="memory", code_resources_enabled=True)
    )
    with pytest.raises(ControlPlaneDisabledError, match="Code workspace APIs require"):
        inactive.refresh_code_resource(
            {"root_slug": "repo", "path": "src/service.py", "_ctx": _ctx()}
        )


def test_edit_facade_delegates_once_with_trusted_context(monkeypatch):
    service = _service()
    calls = []

    class EditService:
        def edit(self, **kwargs):
            calls.append(kwargs)
            return {
                "ok": True,
                "relative_path": "src/service.py",
                "occurrences": 1,
                "edit_state": "modified",
                "refresh_state": "accepted",
            }

    monkeypatch.setattr(service, "_get_code_file_edit_service", lambda: EditService())
    ctx = _ctx()
    result = service.edit_code_file(
        {
            "file_path": "src/service.py",
            "old_string": "return 1",
            "new_string": "return 2",
            "replace_all": False,
            "root_slug": "repo",
            "accountId": "foreign",
            "_ctx": ctx,
        }
    )

    assert result["relative_path"] == "src/service.py"
    assert calls == [
        {
            "file_path": "src/service.py",
            "old_string": "return 1",
            "new_string": "return 2",
            "replace_all": False,
            "root_slug": "repo",
            "ctx": ctx,
        }
    ]


def test_active_search_and_snippet_facades_only_map_to_canonical_services(monkeypatch):
    service = _service()
    ctx = _ctx()
    captured = {}
    canonical_diagnostics = {
        "duration_ms": 7,
        "routes": {"symbolic": {"state": "zero_result", "count": 0, "duration_ms": 0}},
        "budget_clamps": {"result_limit": {"requested": 3, "effective": 2}},
    }
    outcome = ActiveCodeSearchOutcome(status="zero_result", diagnostics=canonical_diagnostics)

    def search(request, request_ctx):
        captured["request"] = request
        captured["search_ctx"] = request_ctx
        return outcome

    evidence = CodeEvidence(
        uri="ctx://acct/resources/code-node",
        identity="ctx://acct/resources/code-node",
        locator="repo/src/service.py:1-2",
        relative_path="src/service.py",
        root_slug="repo",
        excerpt="def service():\n",
    )

    class RetrievalService:
        def read_focused_snippet(self, **kwargs):
            captured["snippet"] = kwargs
            return evidence

    monkeypatch.setattr(service, "search_code_active", search)
    monkeypatch.setattr(service, "_get_code_retrieval_service", lambda: RetrievalService())

    search_payload = service.search_code_resources(
        {
            "query": "service",
            "scope": {"root_slugs": ["repo"]},
            "budget": {"snippet_chars": 24},
            "accountId": "foreign-request-value",
            "_ctx": ctx,
        }
    )
    snippet_payload = service.read_focused_code_snippet(
        {"uri": evidence.uri, "max_chars": 24, "_ctx": ctx}
    )

    assert search_payload["status"] == "zero_result"
    assert captured["search_ctx"] is ctx
    assert captured["request"].query == "service"
    assert captured["request"].scope.root_slugs == ("repo",)
    assert captured["request"].budget.snippet_chars == 24
    assert search_payload["diagnostics"] == canonical_diagnostics
    assert captured["snippet"] == {"uri": evidence.uri, "ctx": ctx, "max_chars": 24}
    assert snippet_payload["locator"] == "repo/src/service.py:1-2"
    assert "/protected/workspace" not in str(snippet_payload)


def test_disabled_active_search_short_circuits_before_backend_initialization(monkeypatch):
    service = MemoryService(
        config=OgMemConfig(
            provider="mock",
            vector_db_type="memory",
            code_retrieval_enabled=False,
        )
    )
    ctx = _ctx()
    def fail_getter(name):
        return lambda: (_ for _ in ()).throw(
            AssertionError(f"disabled search initialized {name}")
        )

    for name in (
        "_get_context_fs",
        "_get_shared_vector_index",
        "_get_shared_embedder",
        "_get_shared_bm25_index",
    ):
        monkeypatch.setattr(
            service,
            name,
            fail_getter(name),
        )
    monkeypatch.setattr(
        service,
        "_get_candidate_code_embedder",
        fail_getter("candidate embedder"),
    )

    result = service.search_code_resources({"query": "service", "_ctx": ctx})

    assert result["status"] == "disabled"
    assert result["diagnostics"]["duration_ms"] >= 0


@pytest.mark.parametrize("query", ["", " ", "\t\n"])
def test_active_search_rejects_blank_query_before_delegation(monkeypatch, query):
    service = _service()
    delegated = False

    def search(*args, **kwargs):
        nonlocal delegated
        delegated = True
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(service, "search_code_active", search)

    with pytest.raises(ValueError, match="query must not be empty"):
        service.search_code_resources({"query": query, "_ctx": _ctx()})

    assert delegated is False


def test_sync_facade_rejects_blank_root_before_the_core_service(monkeypatch):
    service = _service()
    monkeypatch.setattr(
        service,
        "_get_code_workspace_sync_service",
        lambda: SimpleNamespace(sync=lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs))),
    )

    try:
        service.sync_code_workspace({"root_id": " ", "_ctx": _ctx()})
    except ValueError as exc:
        assert str(exc) == "root_id is required"
    else:
        raise AssertionError("blank root_id must fail before canonical sync")
