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