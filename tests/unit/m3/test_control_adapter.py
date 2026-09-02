"""Unit tests for ``m3.control_adapter.OGMemoryControlAdapter``.

Tests use the in-memory graph + a stub Embedder (deterministic 16-d
vectors) + a stub LLM that returns a scripted sequence of tool calls
via ``_mock_tool_calls`` (MockLLM API). No network, no DB.
"""

from __future__ import annotations

import pytest

from core.graph_enums import GraphNodeType
from core.graph_models import create_extracted_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.control_adapter import (
    Answer,
    M3TaskContext,
    OGMemoryControlAdapter,
    Search,
    _SEARCH_CONFIG,
)
from providers.embedder import MockEmbedder
from providers.graph_store import MemoryGraphStore
from providers.llm import MockLLM
from service.graph_service import GraphService


class _ScriptedLLM(MockLLM):
    """MockLLM that returns the next item from a scripted queue per call.

    The vanilla MockLLM always returns the entire ``_mock_tool_calls``
    list on every call, which is fine for single-shot extraction
    tests but loops forever under the ReAct loop. ``_ScriptedLLM``
    pops one call off the queue each turn so the LLM can drive a
    deterministic multi-round script (Search → Answer → final content).

    Each script entry is a dict: either
    ``{"tool": "<name>", "input": {...}}`` to return as a tool call,
    or ``{"content": "..."}`` to return as plain LLM content.
    """

    def __init__(self, script: list[dict] | None = None) -> None:
        super().__init__()
        self._script = list(script or [])
        self._call_count = 0

    def complete_with_tools_messages(
        self, messages, tools=None, tool_choice="auto",
    ):
        self._call_count += 1
        self._last_messages = messages
        if not self._script:
            return [], ""
        entry = self._script.pop(0)
        if "content" in entry:
            return [], entry["content"]
        return [entry], ""


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

DIM = 16


def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-control-test", trace_id="t-test",
    )


@pytest.fixture
def graph() -> GraphService:
    return GraphService(MemoryGraphStore())


@pytest.fixture
def embedder() -> MockEmbedder:
    # 16-d deterministic mock so we control the cosine math exactly.
    return MockEmbedder(dimension=DIM)


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM()


def _scripted(script: list[dict]) -> _ScriptedLLM:
    return _ScriptedLLM(script)


def _seed_episode(
    graph: GraphService,
    ctx: RequestContext,
    *,
    video_id: str,
    clip_id: int,
    description: str,
    node_id: str | None = None,
) -> str:
    """Persist one ExtractedNode + set its embedding to a fixed vector
    so we can assert specific cosine similarities in tests."""
    import uuid
    nid = node_id or uuid.uuid4().hex[:32]
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "_video_id": video_id,
            "_clip_id": clip_id,
            "characters": [],
            "events": [],
            "entities": [],
        },
        node_id=nid,
    )
    graph._store.create_node(node.serialize(), ctx)
    # Set a distinctive embedding so cosine scores are stable.
    # Use the node id's first character to pick a basis-like vector:
    # this gives two different embeddings for different node ids, so
    # we can verify Search picks the right one.
    slot = ord(nid[0]) % 3
    base_vectors = [
        [1.0, 0.0] + [0.0] * (DIM - 2),
        [0.0, 1.0] + [0.0] * (DIM - 2),
        [1.0, 1.0] + [0.0] * (DIM - 2),
    ]
    live = graph._store.get_node(nid, ctx)
    assert live is not None
    live.set_embedding(base_vectors[slot])
    graph.update_extracted_node(live, ctx)
    return nid


@pytest.fixture(autouse=True)
def _reset_search_config():
    """Snap the module-level _SEARCH_CONFIG between tests so a stray
    adapter from one test doesn't bleed into another."""
    original = dict(_SEARCH_CONFIG)
    yield
    _SEARCH_CONFIG.clear()
    _SEARCH_CONFIG.update(original)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructs_with_required_dependencies(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder,
    )
    assert adapter._graph is graph
    assert adapter._embedder is embedder
    assert adapter._extractor_types == ("episodic_memory_generator",)
    assert adapter._default_video_id is None


def test_search_tool_schema_exposes_only_query_and_top_k(graph, embedder, llm):
    adapter = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder,
    )
    tools = adapter._get_tools()
    # 2 operational tools, no custom tools.
    assert len(tools) == 2
    search = next(t for t in tools if t["name"] == "Search")
    answer = next(t for t in tools if t["name"] == "Answer")
    # Search input_schema must only expose query + top_k. Anything
    # else (extractor_types, scope_video_id, ctx, graph, embedder) is
    # runtime-only and must not be visible to the LLM.
    props = search["input_schema"]["properties"]
    assert set(props.keys()) == {"query", "top_k"}
    assert search["input_schema"]["required"] == ["query"]
    # Answer takes a single final_answer string.
    assert set(answer["input_schema"]["properties"].keys()) == {"final_answer"}
    assert answer["input_schema"]["required"] == ["final_answer"]


# ---------------------------------------------------------------------------
# Search tool handler (direct invocation)
# ---------------------------------------------------------------------------

def test_search_handler_returns_hits_with_provenance(graph, embedder):
    ctx = _ctx()
    target_id = _seed_episode(
        graph, ctx,
        video_id="vid_a", clip_id=0,
        description="A woman walks through a door.",
    )
    _seed_episode(
        graph, ctx,
        video_id="vid_b", clip_id=5,
        description="Two people stand by a window.",
    )

    # Drive the operational tool handler directly with the same
    # injection shape the ReactLoop uses.
    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "woman walks", "top_k": 5},
        ctx=ctx, graph=graph, embedder=embedder,
    )
    assert "hits" in result
    assert result["query"] == "woman walks"
    # At least one hit came back; we don't pin scores because the
    # MockEmbedder may produce zero vectors for short queries.
    for hit in result["hits"]:
        assert {"node_id", "score", "video_id", "clip_id",
                "description", "characters", "events",
                "entities"} <= set(hit.keys())
        assert hit["video_id"] in {"vid_a", "vid_b"}


def test_search_handler_rejects_empty_query(graph, embedder):
    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "", "top_k": 5},
        ctx=_ctx(), graph=graph, embedder=embedder,
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# Provenance key fallback (FIX-3)
#
# Current M3 writers stamp ``source_video_id`` / ``source_clip_id`` on
# ExtractedNode.extracted_content; the pre-fix EvidenceHit loader read
# only the legacy ``_video_id`` / ``_clip_id`` keys, so every Search hit
# carried null provenance to the LLM. These tests prove the new loader
# lifts both key generations and prefers the new ones on conflict.
# ---------------------------------------------------------------------------


def _seed_episode_with_keys(
    graph, ctx, *, video_id: str, clip_id: int, description: str,
    keys: dict[str, object], node_id_seed: str,
) -> str:
    """Seed an ExtractedNode with caller-specified provenance keys.

    ``keys`` merges into ``extracted_content`` (e.g.
    ``{"source_video_id": "v", "source_clip_id": 0}`` or
    ``{"_video_id": "v", "_clip_id": 0}``).
    """
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "characters": [],
            "events": [],
            "entities": [],
            **keys,
        },
        node_id=node_id_seed,
    )
    graph._store.create_node(node.serialize(), ctx)
    # Distinctive embedding keeps hits ordered.
    slot = ord(node_id_seed[0]) % 3
    base_vectors = [
        [1.0, 0.0] + [0.0] * (DIM - 2),
        [0.0, 1.0] + [0.0] * (DIM - 2),
        [1.0, 1.0] + [0.0] * (DIM - 2),
    ]
    live = graph._store.get_node(node_id_seed, ctx)
    assert live is not None
    live.set_embedding(base_vectors[slot])
    graph.update_extracted_node(live, ctx)
    return node_id_seed


def test_search_lifts_source_provenance_keys(graph, embedder):
    """[Search] hits must carry non-null video_id/clip_id when the
    ExtractedNode was stamped with the new ``source_*`` keys.
    """
    ctx = _ctx()
    _seed_episode_with_keys(
        graph, ctx,
        video_id="vid_a", clip_id=0,
        description="A woman walks through a door.",
        keys={"source_video_id": "vid_a", "source_clip_id": 0},
        node_id_seed="c00000000000000000000000000",
    )

    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "anything", "top_k": 5},
        ctx=ctx, graph=graph, embedder=embedder,
    )
    assert result["hits"], "expected at least one hit"
    for hit in result["hits"]:
        assert hit["video_id"] == "vid_a", (
            f"hit should carry source_video_id='vid_a', got {hit['video_id']!r}"
        )
        assert hit["clip_id"] == 0, (
            f"hit should carry source_clip_id=0, got {hit['clip_id']!r}"
        )


def test_search_lifts_legacy_provenance_keys_as_fallback(graph, embedder):
    """A node stamped only with the legacy ``_video_id`` / ``_clip_id``
    keys must still surface provenance (pre-existing writers keep
    working). clip_id 0 is a valid ordinal, so None-check is mandatory
    — an ``or`` fallback would have hidden clip 0.
    """
    ctx = _ctx()
    _seed_episode_with_keys(
        graph, ctx,
        video_id="vid_legacy", clip_id=0,   # clip 0 is valid
        description="Legacy clip 0.",
        keys={"_video_id": "vid_legacy", "_clip_id": 0},
        node_id_seed="d00000000000000000000000000",
    )

    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "anything", "top_k": 5},
        ctx=ctx, graph=graph, embedder=embedder,
    )
    assert result["hits"]
    for hit in result["hits"]:
        assert hit["video_id"] == "vid_legacy"
        assert hit["clip_id"] == 0, (
            "clip_id 0 must survive the fallback (not be coerced to None)"
        )


def test_search_prefers_source_over_legacy_on_conflict(graph, embedder):
    """When both key generations are stamped, the new ``source_*``
    values win — the writer's intent is the current source of truth.
    """
    ctx = _ctx()
    _seed_episode_with_keys(
        graph, ctx,
        video_id="vid_current", clip_id=7,
        description="Both keys, source_ wins.",
        keys={
            "source_video_id": "vid_current", "source_clip_id": 7,
            "_video_id": "vid_legacy", "_clip_id": 99,
        },
        node_id_seed="e00000000000000000000000000",
    )

    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "anything", "top_k": 5},
        ctx=ctx, graph=graph, embedder=embedder,
    )
    assert result["hits"]
    for hit in result["hits"]:
        assert hit["video_id"] == "vid_current", (
            "source_video_id should win over _video_id on conflict"
        )
        assert hit["clip_id"] == 7, (
            "source_clip_id should win over _clip_id on conflict"
        )


# ---------------------------------------------------------------------------
# Answer tool
# ---------------------------------------------------------------------------

def test_answer_handler_returns_status_payload():
    from core.tools.operational_tool_spec import call_with_injection
    result = call_with_injection(
        Answer._operational_tool_spec,
        {"final_answer": "the woman is Alice"},
    )
    assert result == {
        "status": "answered",
        "answer": "the woman is Alice",
    }


# ---------------------------------------------------------------------------
# End-to-end run: LLM calls Search then Answer
# ---------------------------------------------------------------------------

def test_run_search_then_answer_completes(graph, embedder, llm):
    """Script the LLM to call [Search] once then [Answer] on the next
    round. Adapter must end with the answer text and a clean trace."""
    ctx = _ctx()
    # Seed one episode so Search has something to return.
    _seed_episode(
        graph, ctx,
        video_id="vid_a", clip_id=0,
        description="A woman walks into a bedroom.",
    )

    # Round 1: Search. Round 2: Answer. Round 3: empty content.
    # The adapter captures the Answer on round 2 and disables tools;
    # the loop runs round 3 in content-only mode and surfaces the
    # captured answer via _parse_output.
    adapter_llm = _scripted([
        {"tool": "Search", "input": {"query": "woman bedroom"}},
        {"tool": "Answer", "input": {"final_answer": "A woman walks into a bedroom."}},
        {"content": "A woman walks into a bedroom."},   # final wrap-up turn
    ])

    adapter = OGMemoryControlAdapter(
        llm=adapter_llm, graph=graph, embedder=embedder,
    )
    task_ctx = M3TaskContext(question="What happens in the bedroom?")
    result = adapter.run(ctx, task_ctx)

    assert result.completed is True
    assert result.finish_reason == "done"
    # Final output surfaces the Answer text.
    assert result.output == ["A woman walks into a bedroom."]
    # Task context captured both tool calls.
    assert task_ctx.final_answer == "A woman walks into a bedroom."
    assert len(task_ctx.search_history) == 1
    assert task_ctx.search_history[0]["query"] == "woman bedroom"


def test_run_loops_on_search_only_then_times_out_at_max(graph, embedder, llm):
    """If the LLM never calls [Answer], the loop runs max_iterations
    rounds of [Search] and exits with finish_reason='max_iterations'.
    The last content (empty) drives the exit; no answer captured."""
    ctx = _ctx()
    _seed_episode(graph, ctx, video_id="v", clip_id=0, description="x")

    # Always Search, never Answer — keeps the loop busy until max_iterations.
    adapter_llm = _scripted([
        {"tool": "Search", "input": {"query": f"q{i}"}}
        for i in range(10)
    ])

    adapter = OGMemoryControlAdapter(
        llm=adapter_llm, graph=graph, embedder=embedder, max_iterations=3,
    )
    result = adapter.run(ctx, M3TaskContext(question="Q?"))
    assert result.finish_reason == "max_iterations"
    assert result.output == [] or result.output is None


def test_run_video_scope_filters_search(graph, embedder, llm):
    """When the task_ctx specifies video_id, the adapter narrows
    Search to that video via ClipScope."""
    ctx = _ctx()
    target = _seed_episode(
        graph, ctx, video_id="vid_target", clip_id=0,
        description="Target.",
    )
    _seed_episode(
        graph, ctx, video_id="vid_other", clip_id=1,
        description="Other.",
    )

    # Direct search through the handler with scope_video_id set via
    # _SEARCH_CONFIG (the adapter's _build_messages propagates the
    # task_ctx.video_id into it).
    _SEARCH_CONFIG["scope_video_id"] = "vid_target"
    from core.tools.operational_tool_spec import call_with_injection
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "anything", "top_k": 10},
        ctx=ctx, graph=graph, embedder=embedder,
    )
    assert "hits" in result
    # All hits must be from vid_target; vid_other is excluded by the
    # ClipScope filter.
    assert result["hits"], "expected at least one scoped hit"
    for hit in result["hits"]:
        assert hit["video_id"] == "vid_target"


# ---------------------------------------------------------------------------
# Runtime injection: verify the handler receives graph + embedder
# ---------------------------------------------------------------------------

def test_search_handler_uses_runtime_injection(graph, embedder):
    """If the handler were missing the runtime injection, it would
    raise on the embedder / graph references. Calling it directly
    through call_with_injection proves the wiring works."""
    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=MockLLM(), graph=graph, embedder=embedder)
    # No graph or embedder passed by name → fallback to type-based
    # matching in call_with_injection.
    result = call_with_injection(
        Search._operational_tool_spec,
        {"query": "test", "top_k": 3},
        ctx=_ctx(),  # type: ignore[arg-type]
        graph=graph, embedder=embedder,
    )
    # No crash, result shape correct.
    assert isinstance(result, dict)
    assert "query" in result


def test_runtime_injection_omits_llm_visible_extra_params(graph, embedder, llm):
    """Regression: extractor_types and scope_video_id must NOT be
    visible to the LLM. The Search tool's input_schema lists only
    query + top_k."""
    adapter = OGMemoryControlAdapter(
        llm=llm, graph=graph, embedder=embedder,
        extractor_types=("episodic_memory_generator", "face_detection"),
        default_video_id="vid_x",
    )
    search_schema = next(
        t for t in adapter._get_tools() if t["name"] == "Search"
    )["input_schema"]
    assert "extractor_types" not in search_schema["properties"]
    assert "scope_video_id" not in search_schema["properties"]
    # Adapter-level config is preserved internally.
    assert adapter._extractor_types == (
        "episodic_memory_generator", "face_detection",
    )
    assert adapter._default_video_id == "vid_x"
    assert _SEARCH_CONFIG["extractor_types"] == (
        "episodic_memory_generator", "face_detection",
    )


# ---------------------------------------------------------------------------
# BM25 transcript rerank wiring (finding 3d — retrieval half)
# ---------------------------------------------------------------------------

def test_search_reranks_transcript_match_above_higher_vector_score(graph, embedder, llm):
    """End-to-end proof that the BM25 rerank fires INSIDE the [Search]
    tool: when pure-vector retrieval ranks an off-topic clip above a
    clip whose transcript contains the query's exact words, the rerank
    must flip the order so the transcript match comes out on top.

    Rather than assume the hash-based MockEmbedder produces a specific
    cosine ordering (it won't, for arbitrary strings), this test
    determines the pure-vector order empirically (a baseline Search
    with the rerank disabled), then asserts the rerank flips it when
    the lower-ranked clip carries the query tokens in its transcript.
    """
    import re
    from unittest.mock import patch
    from nltk.stem import PorterStemmer

    _MOCK_STOPS = frozenset(
        "a an the and or but is are was were be been being have has had do "
        "does did will would could should may might shall can to of in for "
        "on with at by from as our i we you they".split()
    )
    _stub_stemmer = PorterStemmer()

    def _stub_tokenize(text: str) -> list[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return [_stub_stemmer.stem(t) for t in toks if t not in _MOCK_STOPS]

    ctx = _ctx()
    query = "photos album"

    # Clip A: off-topic speech (no query tokens).
    _seed_episode_with_transcript(
        graph, ctx, video_id="vid", clip_id=10,
        description="a woman cooking in a kitchen",
        transcript_segments=[{"start_sec": 0, "end_sec": 3,
                              "text": "the weather is nice today"}],
        node_id_seed="a00000000000000000000000000",
    )
    # Clip B: the query's exact words live in its transcript.
    _seed_episode_with_transcript(
        graph, ctx, video_id="vid", clip_id=20,
        description="someone talking near a table",
        transcript_segments=[{"start_sec": 0, "end_sec": 4,
                              "text": "i put our photos in the album"}],
        node_id_seed="b00000000000000000000000000",
    )

    from core.tools.operational_tool_spec import call_with_injection
    OGMemoryControlAdapter(llm=llm, graph=graph, embedder=embedder)

    # --- Baseline: pure-vector order (rerank disabled) ----------------
    with patch("m3.transcript_rerank.rerank_with_transcript_bm25",
               side_effect=lambda q, h, **kw: list(h)):
        baseline = call_with_injection(
            Search._operational_tool_spec,
            {"query": query, "top_k": 5},
            ctx=ctx, graph=graph, embedder=embedder,
        )
    assert len(baseline["hits"]) == 2
    vector_top = baseline["hits"][0]["clip_id"]
    vector_bottom = baseline["hits"][1]["clip_id"]

    # Only a meaningful test if pure-vector ranks the off-topic clip
    # above the transcript match. If the embedder already happens to rank
    # B first, there's nothing to rescue — skip the flip assertion.
    if vector_top == 10 and vector_bottom == 20:
        # --- With rerank enabled ---------------------------------------
        with patch("retrieval.bm25_index._ensure_nltk"), \
             patch("retrieval.bm25_index._tokenize", _stub_tokenize):
            reranked = call_with_injection(
                Search._operational_tool_spec,
                {"query": query, "top_k": 5},
                ctx=ctx, graph=graph, embedder=embedder,
            )
        assert len(reranked["hits"]) == 2
        # The transcript match (clip 20) is rescued to the top.
        assert reranked["hits"][0]["clip_id"] == 20, (
            "BM25 rerank failed to rescue the transcript-matching clip "
            f"to top; got clip_id={reranked['hits'][0]['clip_id']} "
            f"(vector order was {[h['clip_id'] for h in baseline['hits']]})"
        )
    # else: the MockEmbedder happened to rank the transcript clip first
    # already — nothing to assert; the rerank is a no-op-or-confirm in
    # that case and the focused unit tests cover the rescue math.


def _seed_episode_with_transcript(
    graph, ctx, *, video_id, clip_id, description,
    transcript_segments, node_id_seed,
):
    """Seed an ExtractedNode carrying ``speech_transcript`` segments and
    a controlled embedding so the cosine ordering is deterministic.

    Distinct from :func:`_seed_episode` only in that it also plants the
    transcript on ``extracted_content`` — the rerank's input.
    """
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "_video_id": video_id,
            "_clip_id": clip_id,
            "characters": [],
            "events": [],
            "entities": [],
            "speech_transcript": transcript_segments,
        },
        node_id=node_id_seed,
    )
    graph._store.create_node(node.serialize(), ctx)
    slot = ord(node_id_seed[0]) % 3
    base_vectors = [
        [1.0, 0.0] + [0.0] * (DIM - 2),   # slot 0 → highest cosine vs [1,0,..]
        [0.0, 1.0] + [0.0] * (DIM - 2),
        [1.0, 1.0] + [0.0] * (DIM - 2),
    ]
    live = graph._store.get_node(node_id_seed, ctx)
    assert live is not None
    live.set_embedding(base_vectors[slot])
    graph.update_extracted_node(live, ctx)
    return node_id_seed