from __future__ import annotations

from core.models import ContextNode, RequestContext, RetrievedBlock, SearchMemoryResult
from providers.unified_config import OgMemConfig
from server.mcp_tools import query_memory, read_memory


class _FakeAuth:
    def role_control_active(self):
        return False


class _FakeReadAPI:
    def __init__(self, hits=None):
        self.search_kwargs = None
        self.hits = hits or [
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/preferences/coding_style",
                score=0.9,
                category="preference",
                abstract="Prefers vitest for TypeScript tests.",
                has_content=True,
            )
        ]

    def search_memory(self, query, ctx, **kwargs):
        self.search_kwargs = {"query": query, "ctx": ctx, **kwargs}
        return SearchMemoryResult(
            request_id="req-1",
            query=query,
            hits=self.hits,
        )


class _FakeFS:
    def read_node(self, uri, ctx):
        return ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user:alice",
            abstract="Prefers vitest.",
            overview="Testing framework preference.",
            content="Use vitest by default for TypeScript projects.",
            metadata={"version": 2, "updated_at": "2026-06-23T00:00:00Z"},
        )


class _FallbackPatternFS:
    def read_node(self, uri, ctx):
        if uri == "ctx://acct-1/users/alice/memories/patterns/review":
            raise FileNotFoundError(uri)
        if uri != "ctx://acct-1/agents/bot/memories/patterns/review":
            raise AssertionError(f"unexpected URI: {uri}")
        return ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="pattern",
            level=0,
            owner_space="agent:bot",
            abstract="Legacy agent review pattern.",
            overview="Review pattern overview.",
            content="Use the legacy agent review pattern.",
            metadata={"version": 2, "updated_at": "2026-06-23T00:00:00Z"},
        )


class _FakeService:
    def __init__(self, cfg=None, hits=None):
        self._cfg = cfg or OgMemConfig()
        self.read_api = _FakeReadAPI(hits)
        self.fs = _FakeFS()

    def get_auth_service(self):
        return _FakeAuth()

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="alice",
            agent_id=params.get("agentId") or "bot",
            session_id=params.get("sessionId") or "s1",
            trace_id="t1",
        )

    def get_read_api(self):
        return self.read_api

    def _get_context_fs(self):
        return self.fs


def test_query_memory_returns_external_uri_only():
    service = _FakeService()

    payload = query_memory(service, query="testing", types=["preference"], top_k=3)

    assert payload["hits"][0]["uri"] == "ogmemory://memories/preference/coding_style"
    assert "ctx://" not in str(payload)
    assert service.read_api.search_kwargs["categories"] == ["preference"]
    assert service.read_api.search_kwargs["top_k"] == 3


def test_query_memory_returns_session_summary_hits():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/session_summaries/summary_1/content.md",
                score=0.8,
                category="session_summary",
                abstract="Caroline attended an LGBTQ support group.",
                has_content=True,
            )
        ],
    )

    payload = query_memory(service, query="Caroline support group", top_k=3)

    assert payload["hits"][0]["uri"] == "ogmemory://memories/session_summary/summary_1"
    assert payload["hits"][0]["type"] == "session_summary"


def test_query_memory_returns_scoped_uri_for_legacy_agent_pattern_hit():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/agents/bot/memories/patterns/review",
                score=0.8,
                category="pattern",
                abstract="Legacy agent review pattern.",
                has_content=True,
            )
        ],
    )

    payload = query_memory(service, query="review", types=["pattern"], top_k=3)

    assert payload["hits"][0]["uri"] == "ogmemory://memories/pattern/agent/review"
    assert payload["hits"][0]["type"] == "pattern"
    assert payload["hits"][0]["id"] == "review"


def test_query_memory_rejects_request_larger_than_configured_limit():
    service = _FakeService(OgMemConfig(mcp_max_request_bytes=16))

    try:
        query_memory(service, query="x" * 64)
    except ValueError as exc:
        assert "mcp_max_request_bytes" in str(exc)
    else:
        raise AssertionError("query_memory should reject oversized requests")


def test_query_memory_applies_configured_response_cap_to_hit_abstracts():
    service = _FakeService(OgMemConfig(mcp_max_response_chars=10))

    payload = query_memory(service, query="testing", types=["preference"], top_k=3)

    assert payload["hits"][0]["abstract"] == "Prefers vi"
    assert payload["truncated"] is True


def test_query_memory_rejects_target_uri_style_direct_reads():
    service = _FakeService()

    payload = query_memory(service, query="testing", types=None)

    assert "target_uri" not in service.read_api.search_kwargs
    assert payload["hits"][0]["has_content"] is True


def test_read_memory_reads_external_uri():
    service = _FakeService()

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        level="abstract",
    )

    assert payload["uri"] == "ogmemory://memories/preference/coding_style"
    assert payload["abstract"] == "Prefers vitest."
    assert "content" not in payload
    assert payload["truncated"] is False


def test_read_memory_returns_canonical_uri_after_legacy_pattern_fallback():
    service = _FakeService()
    service.fs = _FallbackPatternFS()

    payload = read_memory(
        service,
        uri="ogmemory://memories/pattern/review",
        level="content",
    )

    assert payload["uri"] == "ogmemory://memories/pattern/agent/review"
    assert payload["type"] == "pattern"
    assert payload["id"] == "review"
    assert payload["content"] == "Use the legacy agent review pattern."


def test_read_memory_uses_configured_response_cap():
    service = _FakeService(OgMemConfig(mcp_max_response_chars=5))

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        level="content",
        max_chars=100,
    )

    assert payload["content"] == "Use v"
    assert payload["truncated"] is True


def test_read_memory_rejects_internal_uri():
    service = _FakeService()

    try:
        read_memory(service, uri="ctx://acct-1/users/alice/memories/preferences/coding")
    except ValueError as exc:
        assert "ctx://" in str(exc) or "ogmemory" in str(exc)
    else:
        raise AssertionError("read_memory should reject internal ctx:// URIs")


def test_read_memory_truncates_large_fields():
    service = _FakeService()

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        level="content",
        max_chars=10,
    )

    assert payload["content"] == "Use vitest"
    assert payload["truncated"] is True
