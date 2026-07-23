from __future__ import annotations

from core.models import (
    ContextNode,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    SearchMemoryResult,
)
from providers.unified_config import OgMemConfig
from server.mcp_tools import (
    delete_memory,
    list_skill_candidates,
    list_skills,
    query_memory,
    read_memory,
    read_skill,
    search_skills,
)
from skills.artifact import internal_skill_uri, is_governed_skill


class _FakeAuth:
    def role_control_active(self):
        return False


class _FakeReadAPI:
    def __init__(self, hits=None, cfg=None):
        self.search_kwargs = None
        self._cfg = cfg or RetrievalConfig()
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


class _FieldAwareFS(_FakeFS):
    def __init__(self):
        self.read_node_calls = []
        self.read_node_fields_calls = []

    def read_node(self, uri, ctx):
        self.read_node_calls.append(uri)
        return super().read_node(uri, ctx)

    def read_node_fields(self, uri, ctx, level):
        self.read_node_fields_calls.append((uri, level))
        node = super().read_node(uri, ctx)
        return ContextNode(
            uri=node.uri,
            context_type=node.context_type,
            category=node.category,
            level=node.level,
            owner_space=node.owner_space,
            abstract=node.abstract if level in {"abstract", "all"} else "",
            overview=node.overview if level in {"overview", "all"} else "",
            content=node.content if level in {"content", "all"} else "",
            metadata=node.metadata,
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
        self.read_api = _FakeReadAPI(
            hits,
            RetrievalConfig(max_top_k=getattr(self._cfg, "retrieval_max_top_k", 50)),
        )
        self.fs = _FakeFS()

    def get_auth_service(self):
        return _FakeAuth()

    @property
    def default_agent_id(self):
        return self._cfg.agent_id

    @property
    def skill_search_scan_limit(self):
        return self._cfg.skill_search_scan_limit

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


class _CollectionFS(_FakeFS):
    def list_children(self, uri, ctx):
        return ["ctx://acct-1/users/alice/memories/preferences/coding_style"]

    def exists(self, uri, ctx):
        return True


class _CollectionService(_FakeService):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.fs = _CollectionFS()


class _SkillFS(_FakeFS):
    def list_children(self, uri, ctx):
        assert uri == "ctx://acct-1/agents/bot/skills"
        return ["ctx://acct-1/agents/bot/skills/review-code"]

    def read_node(self, uri, ctx):
        return ContextNode(
            uri="ctx://acct-1/agents/bot/skills/review-code",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:bot",
            abstract="Review code before production use.",
            overview="Agent skill: review-code",
            content="---\nname: review-code\ndescription: Review code.\n---\n# Review code\n",
            metadata={"version": 1, "ogmemory_generated": True,
                      "publication_status": "published", "review_score": 0.95},
        )


class _SkillService(_FakeService):
    def __init__(self, hits=None):
        super().__init__(
            hits=hits
            or [
                RetrievedBlock(
                    uri="ctx://acct-1/agents/bot/skills/review-code",
                    score=0.95,
                    category="skill",
                    abstract="Review code before production use.",
                )
            ]
        )
        self.fs = _SkillFS()

    def list_published_skills(self, ctx, *, limit=50, cursor=None):
        base = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills"
        after = internal_skill_uri(ctx.account_id, ctx.agent_id, cursor) if cursor else None
        if hasattr(self.fs, "list_node_summaries_page"):
            nodes = self.fs.list_node_summaries_page(base, ctx, limit=limit + 1, after_uri=after)
        else:
            uris = sorted(self.fs.list_children(base, ctx))
            nodes = [self.fs.read_node(uri, ctx) for uri in uris if after is None or uri > after]
        rows = [
            {"name": node.uri.rsplit("/", 1)[-1], "description": node.abstract,
             "updated_at": node.metadata.get("updated_at", "")}
            for node in nodes if is_governed_skill(node)
        ]
        payload = {"skills": rows[:limit], "truncated": len(rows) > limit}
        if payload["truncated"]:
            payload["next_cursor"] = rows[limit - 1]["name"]
        return payload

    def read_published_skill(self, ctx, name):
        node = self.fs.read_node(internal_skill_uri(ctx.account_id, ctx.agent_id, name), ctx)
        if not is_governed_skill(node):
            raise FileNotFoundError(name)
        return node

    def list_skill_candidates(self, ctx, state=None):
        return [
            {
                "capability_key": "review-code",
                "state": state or "accumulating",
                "evidence_count": 4,
                "distinct_session_count": 2,
                "summary": "Review changes safely.",
                "reason": "Awaiting more evidence.",
            }
        ]


def test_query_memory_returns_external_uri_only():
    service = _FakeService()

    payload = query_memory(service, query="testing", types=["preference"], top_k=3)

    assert payload["hits"][0]["uri"] == "ogmemory://memories/preference/coding_style"
    assert "ctx://" not in str(payload)
    assert service.read_api.search_kwargs["categories"] == ["preference"]
    assert service.read_api.search_kwargs["top_k"] == 3


def test_query_memory_requests_content_fill_for_top_three_hits():
    service = _FakeService()

    query_memory(service, query="testing", top_k=5)

    assert service.read_api.search_kwargs["fill_content_for_top_k"] == 3


def test_query_memory_returns_filled_overview_and_content_excerpt():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/preferences/coding_style",
                score=0.9,
                category="preference",
                abstract="Prefers vitest for TypeScript tests.",
                overview="Testing framework preference.",
                content_excerpt="Use vitest by default for TypeScript projects.",
                has_content=True,
            )
        ],
    )

    payload = query_memory(service, query="testing", top_k=3)

    assert payload["hits"][0]["overview"] == "Testing framework preference."
    assert payload["hits"][0]["content_excerpt"] == "Use vitest by default for TypeScript projects."


def test_query_memory_returns_public_temporal_metadata_fields():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/events/support_group",
                score=0.9,
                category="event",
                abstract="Caroline joined the LGBTQ support group.",
                has_content=True,
                metadata={
                    "when": "2023-05-07",
                    "who": ["Caroline"],
                    "where": "community center",
                    "routing_key": "internal",
                },
            )
        ],
    )

    payload = query_memory(service, query="When did Caroline join the support group?", top_k=3)

    assert payload["hits"][0]["when"] == "2023-05-07"
    assert payload["hits"][0]["who"] == ["Caroline"]
    assert payload["hits"][0]["where"] == "community center"
    assert "routing_key" not in payload["hits"][0]


def test_query_memory_does_not_expose_internal_ranking_score():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/events/support_group",
                score=0.7,
                ranking_score=0.95,
                category="event",
                abstract="Support group note",
            )
        ],
    )

    payload = query_memory(service, query="support group", top_k=1)

    assert "ranking_score" not in payload["hits"][0]


def test_query_memory_temporal_query_overfetches_but_returns_requested_top_k():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri=f"ctx://acct-1/users/alice/memories/events/event_{i}",
                score=1.0 - i / 100,
                category="event",
                abstract=f"Event {i}",
                has_content=True,
            )
            for i in range(15)
        ],
    )

    payload = query_memory(service, query="When did Caroline join the support group?", top_k=5)

    assert service.read_api.search_kwargs["top_k"] == 15
    assert service.read_api.search_kwargs["fill_content_for_top_k"] == 3
    assert len(payload["hits"]) == 5


def test_query_memory_temporal_overfetch_respects_configured_max_top_k():
    service = _FakeService(cfg=OgMemConfig(retrieval_max_top_k=20))

    query_memory(service, query="When did Caroline join the support group?", top_k=20)

    assert service.read_api.search_kwargs["top_k"] == 20


def test_query_memory_non_temporal_query_keeps_requested_top_k():
    service = _FakeService()

    query_memory(service, query="Caroline support group notes", top_k=5)

    assert service.read_api.search_kwargs["top_k"] == 5


def test_skill_tools_use_progressive_disclosure_and_no_trace_payloads():
    service = _SkillService()

    listing = list_skills(service, agent_id="bot")
    detail = read_skill(service, name="review-code", agent_id="bot")
    candidates = list_skill_candidates(service, state="accumulating", agent_id="bot")

    assert listing["skills"] == [
        {
            "name": "review-code",
            "description": "Review code before production use.",
            "uri": "ogmemory://skills/review-code",
            "updated_at": "",
        }
    ]
    assert "content" not in listing["skills"][0]
    assert detail["content"].startswith("---\nname: review-code")
    assert detail["review_status"] == "published"
    assert "trace" not in str(detail).lower()
    assert candidates["candidates"][0]["evidence_count"] == 4
    assert "session_id" not in str(candidates)


def test_search_skills_uses_only_reviewed_catalog():
    service = _SkillService()

    payload = search_skills(service, query="review changes", top_k=3, agent_id="bot")

    assert payload["skills"][0]["uri"] == "ogmemory://skills/review-code"
    assert service.read_api.search_kwargs is None
    assert "request_id" not in payload


def test_skill_catalog_hides_legacy_unreviewed_skill_nodes():
    service = _SkillService()
    service.fs.read_node = lambda uri, ctx: ContextNode(
        uri="ctx://acct-1/agents/bot/skills/legacy",
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:bot",
        abstract="Legacy generic extraction.",
        overview="Legacy generic extraction.",
        content="Do something.",
    )

    assert list_skills(service, agent_id="bot")["skills"] == []


def test_skill_catalog_cursor_paginates_without_loss_or_duplication():
    service = _SkillService()
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/agents/bot/skills/skill-{index:03d}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:bot", abstract=f"Skill {index}", overview="",
            content="skill", metadata={"ogmemory_generated": True,
                                       "publication_status": "published", "review_score": .95},
        )
        for index in range(120)
    ]

    class _PagedFS:
        def list_children(self, uri, ctx):
            return [node.uri for node in nodes]

        def read_node(self, uri, ctx):
            return next(node for node in nodes if node.uri == uri)

        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            page = [node for node in nodes if after_uri is None or node.uri > after_uri]
            return page[:limit]

    service.fs = _PagedFS()
    first = list_skills(service, agent_id="bot", limit=100)
    second = list_skills(
        service, agent_id="bot", limit=100, cursor=first["next_cursor"]
    )

    names = [row["name"] for row in first["skills"] + second["skills"]]
    assert len(names) == 120
    assert len(set(names)) == 120
    assert first["truncated"] is True
    assert second["truncated"] is False


def test_skill_catalog_exact_page_is_not_truncated_without_more_rows():
    service = _SkillService()
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/agents/bot/skills/skill-{index:03d}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:bot", abstract=f"Skill {index}", overview="",
            content="skill", metadata={"ogmemory_generated": True,
                                       "publication_status": "published", "review_score": .95},
        )
        for index in range(100)
    ]

    class _PagedFS:
        def list_children(self, uri, ctx):
            return [node.uri for node in nodes]

        def read_node(self, uri, ctx):
            return next(node for node in nodes if node.uri == uri)

        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            page = [node for node in nodes if after_uri is None or node.uri > after_uri]
            return page[:limit]

    service.fs = _PagedFS()
    payload = list_skills(service, agent_id="bot", limit=100)

    assert len(payload["skills"]) == 100
    assert payload["truncated"] is False
    assert "next_cursor" not in payload


def test_skill_search_enforces_configured_catalog_scan_limit():
    service = _SkillService()
    service._cfg.skill_search_scan_limit = 1
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/agents/bot/skills/review-{index}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:bot", abstract=f"Review {index}", overview="",
            content="skill", metadata={"ogmemory_generated": True,
                                       "publication_status": "published", "review_score": .95},
        )
        for index in range(2)
    ]

    class _FS:
        def list_children(self, uri, ctx):
            return [node.uri for node in nodes]

        def read_node(self, uri, ctx):
            return next(node for node in nodes if node.uri == uri)

    service.fs = _FS()

    payload = search_skills(service, query="review", top_k=2, agent_id="bot")

    assert len(payload["skills"]) == 1
    assert payload["truncated"] is True


def test_skill_search_ranks_chinese_description_for_chinese_query():
    service = _SkillService()
    nodes = [
        ContextNode(
            uri="ctx://acct-1/agents/bot/skills/code-review", context_type="SKILL",
            category="skill", level=0, owner_space="agent:bot",
            abstract="用于代码审查和发布前检查", overview="", content="skill",
            metadata={"ogmemory_generated": True, "publication_status": "published",
                      "review_score": .95},
        ),
        ContextNode(
            uri="ctx://acct-1/agents/bot/skills/data-export", context_type="SKILL",
            category="skill", level=0, owner_space="agent:bot",
            abstract="用于导出数据报表", overview="", content="skill",
            metadata={"ogmemory_generated": True, "publication_status": "published",
                      "review_score": .95},
        ),
    ]

    class _FS:
        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            page = [node for node in nodes if after_uri is None or node.uri > after_uri]
            return page[:limit]

        def read_node(self, uri, ctx):
            return next(node for node in nodes if node.uri == uri)

    service.fs = _FS()

    payload = search_skills(service, query="代码审查", top_k=1, agent_id="bot")

    assert payload["skills"][0]["name"] == "code-review"
    assert payload["skills"][0]["score"] > 0


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


def test_query_memory_stops_adding_hits_after_response_budget_is_exhausted():
    service = _FakeService(
        OgMemConfig(mcp_max_response_chars=1),
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/preferences/first",
                score=0.9,
                category="preference",
                abstract="first",
                has_content=True,
            ),
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/preferences/second",
                score=0.8,
                category="preference",
                abstract="second",
                has_content=True,
            ),
        ],
    )

    payload = query_memory(service, query="testing", top_k=2)

    assert [hit["id"] for hit in payload["hits"]] == ["first"]
    assert payload["hits"][0]["abstract"] == "f"
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


def test_read_memory_uses_field_level_reader_when_available():
    service = _FakeService()
    service.fs = _FieldAwareFS()

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        level="abstract",
    )

    assert payload["abstract"] == "Prefers vitest."
    assert "content" not in payload
    assert service.fs.read_node_fields_calls == [
        ("ctx://acct-1/users/alice/memories/preferences/coding_style", "abstract")
    ]
    assert service.fs.read_node_calls == []


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


def test_read_memory_routes_skill_uri_through_governed_catalog():
    service = _SkillService()

    payload = read_memory(
        service,
        uri="ogmemory://skills/review-code",
        level="content",
        agent_id="bot",
    )

    assert payload["uri"] == "ogmemory://skills/review-code"
    assert payload["type"] == "skill"
    assert payload["content"].startswith("---\nname: review-code")


def test_read_memory_rejects_legacy_ungoverned_skill_uri():
    service = _SkillService()
    service.fs.read_node = lambda uri, ctx: ContextNode(
        uri="ctx://acct-1/agents/bot/skills/legacy",
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:bot",
        abstract="Legacy generic extraction.",
        overview="Legacy generic extraction.",
        content="Do something.",
    )

    try:
        read_memory(
            service,
            uri="ogmemory://memories/skill/legacy",
            level="content",
            agent_id="bot",
        )
    except Exception as exc:
        assert "not found" in str(exc).lower() or "visible" in str(exc).lower()
    else:
        raise AssertionError("read_memory should reject ungoverned skill nodes")


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


def test_delete_memory_rejects_skill_uri():
    service = _SkillService()

    try:
        delete_memory(
            service,
            uri="ogmemory://skills/review-code",
            agent_id="bot",
        )
    except ValueError as exc:
        assert "skills cannot be deleted" in str(exc)
    else:
        raise AssertionError("delete_memory should reject generated skill URIs")


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


def test_read_memory_preserves_collection_truncation_flag():
    service = _CollectionService(OgMemConfig(mcp_max_response_chars=5))

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference",
        max_chars=100,
    )

    assert payload["memories"][0]["abstract"] == "Prefe"
    assert payload["truncated"] is True
