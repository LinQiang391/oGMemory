"""Tests for the ``m3.qa`` end-to-end pipeline stage and the
``run_react_qa`` driver it wraps.

Three layers:

* YAML wiring — the qa stage is registered, declared in the
  video_episodic_qa.yaml, and positioned after build_index so
  EpisractedNode embeddings are in the graph before the adapter
  runs [Search].
* Stage function — given a PipelineContext with a seeded graph +
  a real annotation file, the stage writes m3_qa_results.jsonl with
  one row per question, surfacing the OGMemoryControlAdapter's
  answer + search history.
* run_react_qa driver — direct invocation with the injected
  graph/embedder/llm (bypasses the pipeline) produces the same
  per-question shape and respects --resume.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.graph_models import create_extracted_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.qa import (
    iter_annotation_questions,
    load_questions,
    parse_react_qa_args,
    run_react_qa,
)
from m3.pipeline.runner import PipelineRunner
from m3.pipeline.spec import load_pipeline_spec
from providers.embedder import MockEmbedder
from providers.graph_store import MemoryGraphStore
from providers.llm import MockLLM
from service.graph_service import GraphService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FFMPEG = shutil.which("ffmpeg")
_requires_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None, reason="ffmpeg not available on PATH",
)


def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-qa-test", trace_id="t-test",
    )


def _write_annotations(path: Path, questions: list[dict], video_id: str = "vid_q") -> Path:
    annotations = {video_id: {"video_path": "/tmp/vid_q.mp4",
                              "qa_list": questions}}
    path.write_text(json.dumps(annotations), encoding="utf-8")
    return path


def _seed_episode(
    graph: GraphService, ctx: RequestContext, *, video_id: str,
    clip_id: int, description: str, nid: str | None = None,
) -> str:
    """Persist one ExtractedNode + set its embedding to a fixed vector."""
    import uuid
    nid = nid or uuid.uuid4().hex[:32]
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={"description": description, "_video_id": video_id,
                           "_clip_id": clip_id, "characters": [],
                           "events": [], "entities": []},
        node_id=nid,
    )
    graph._store.create_node(node.serialize(), ctx)
    live = graph._store.get_node(nid, ctx)
    live.set_embedding([1.0, 0.0, 0.5])
    graph.update_extracted_node(live, ctx)
    return nid


class _ScriptedAdapterLLM(MockLLM):
    """Same per-question cycle as the eval harness tests: Search →
    Answer → final content. Resets state on question boundary so the
    same adapter can drive multiple questions.
    """
    def __init__(self, answer_text: str = "the woman walks into a bedroom"):
        super().__init__()
        self._answer_text = answer_text
        self._per_question_round = 0
        self._last_question_prefix = None

    def complete_with_tools_messages(self, messages, tools=None, tool_choice="auto"):
        question_prefix = None
        for m in messages:
            if m.get("role") == "user" and "Question:" in (m.get("content") or ""):
                question_prefix = m["content"][:32]
                break
        if question_prefix and question_prefix != self._last_question_prefix:
            self._per_question_round = 0
            self._last_question_prefix = question_prefix
        self._per_question_round += 1
        self._last_messages = messages
        self._call_count += 1
        if self._per_question_round == 1:
            return ([{"tool": "Search", "input": {"query": "anything"}}], "")
        if self._per_question_round == 2:
            return ([{"tool": "Answer",
                     "input": {"final_answer": self._answer_text}}], "")
        return [], self._answer_text


# ---------------------------------------------------------------------------
# YAML wiring
# ---------------------------------------------------------------------------

def test_video_episodic_qa_yaml_declares_qa_stage_after_build_index():
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic_qa.yaml")
    stage_ids = [s.id for s in spec.stages]
    assert "qa" in stage_ids
    qa = next(s for s in spec.stages if s.id == "qa")
    assert qa.type == "m3.qa"
    assert qa.enabled is True
    # qa must run after build_index so EpisrictedNode embeddings are
    # in the graph before [Search] fires.
    assert stage_ids.index("qa") > stage_ids.index("build_index")
    # qa must depend on build_index.
    assert "build_index" in qa.needs
    # The qa_results artifact is wired so the stage function can
    # produce the JSONL without an explicit --output-jsonl override.
    assert "qa_results" in spec.artifacts


def test_qa_stage_is_registered_in_default_registry():
    from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
    assert "m3.qa" in DEFAULT_STAGE_REGISTRY


# ---------------------------------------------------------------------------
# Driver (run_react_qa) — direct invocation
# ---------------------------------------------------------------------------

def test_run_react_qa_writes_results_with_injected_graph(tmp_path: Path):
    """Driver contract: when graph + embedder + LLM are injected, the
    driver skips the CLI-side construction and uses them directly."""
    annotations = _write_annotations(tmp_path / "ann.json", [
        {"question_id": "q1", "question": "What happens?",
         "answer": "stuff", "type": "object"},
        {"question_id": "q2", "question": "When?",
         "answer": "now", "type": "temporal"},
    ])
    output_jsonl = tmp_path / "results.jsonl"

    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_episode(graph, ctx, video_id="vid_q", clip_id=0, description="x")

    args = parse_react_qa_args([
        "--annotations", str(annotations),
        "--output-jsonl", str(output_jsonl),
        "--graph-store", "memory",
        "--max-iterations", "5",
    ])
    rc = run_react_qa(
        args,
        graph=graph,
        embedder=MockEmbedder(dimension=8),
        llm=_ScriptedAdapterLLM("the answer is 42"),
    )
    assert rc == 0

    records = [json.loads(l) for l in output_jsonl.read_text().splitlines() if l]
    assert len(records) == 2
    by_qid = {r["question_id"]: r for r in records}
    assert by_qid["q1"]["predicted_answer"] == "the answer is 42"
    assert by_qid["q2"]["predicted_answer"] == "the answer is 42"
    # Search history captured per question.
    for r in records:
        assert r["search_rounds"] >= 1
        assert any(
            h.get("query") == "anything" for h in r["search_history"]
        )


def test_run_react_qa_resume_skips_already_answered(tmp_path: Path):
    annotations = _write_annotations(tmp_path / "ann.json", [
        {"question_id": "q1", "question": "Q1?", "answer": "A1",
         "type": "object"},
        {"question_id": "q2", "question": "Q2?", "answer": "A2",
         "type": "temporal"},
    ])
    output_jsonl = tmp_path / "results.jsonl"
    # Pre-populate q1.
    output_jsonl.write_text(json.dumps({
        "video_id": "vid_q", "question_id": "q1",
        "question_index": 0, "question": "Q1?",
        "expected_answer": "A1", "predicted_answer": "A1",
        "type": "object", "timestamp": None, "before_clip": None,
        "iterations": 2, "finish_reason": "done",
        "search_rounds": 1, "search_history": [],
        "error": "",
    }) + "\n", encoding="utf-8")

    graph = GraphService(MemoryGraphStore())
    _seed_episode(graph, _ctx(), video_id="vid_q", clip_id=0, description="x")
    args = parse_react_qa_args([
        "--annotations", str(annotations),
        "--output-jsonl", str(output_jsonl),
        "--graph-store", "memory",
        "--resume",
    ])
    rc = run_react_qa(
        args, graph=graph,
        embedder=MockEmbedder(dimension=8),
        llm=_ScriptedAdapterLLM("A2"),
    )
    assert rc == 0
    records = [json.loads(l) for l in output_jsonl.read_text().splitlines() if l]
    assert len(records) == 2
    assert {r["question_id"] for r in records} == {"q1", "q2"}


def test_run_react_qa_video_filter_narrows_questions(tmp_path: Path):
    # Two videos, one question each. --video vid_a must keep only
    # the vid_a question in the output. Note the video_path must
    # derive to vid_a via video_id_from_path (stem before extension)
    # — iter_annotation_questions uses the path, not the dict key,
    # when video_path is present.
    annotations = tmp_path / "ann.json"
    annotations.write_text(json.dumps({
        "vid_a": {"video_path": "/tmp/vid_a.mp4", "qa_list": [
            {"question_id": "a_q", "question": "A?", "answer": "x", "type": "object"},
        ]},
        "vid_b": {"video_path": "/tmp/vid_b.mp4", "qa_list": [
            {"question_id": "b_q", "question": "B?", "answer": "y", "type": "temporal"},
        ]},
    }), encoding="utf-8")
    output_jsonl = tmp_path / "results.jsonl"

    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_episode(graph, ctx, video_id="vid_a", clip_id=0, description="x")

    args = parse_react_qa_args([
        "--annotations", str(annotations),
        "--output-jsonl", str(output_jsonl),
        "--graph-store", "memory",
        "--video", "vid_a",
    ])
    rc = run_react_qa(
        args, graph=graph,
        embedder=MockEmbedder(dimension=8),
        llm=_ScriptedAdapterLLM("A"),
    )
    assert rc == 0
    records = [json.loads(l) for l in output_jsonl.read_text().splitlines() if l]
    assert len(records) == 1, f"expected 1 record, got {len(records)}: {records}"
    assert records[0]["question_id"] == "a_q"
    assert records[0]["video_id"] == "vid_a"


# ---------------------------------------------------------------------------
# Stage function (qa_stage) — through the pipeline runner
# ---------------------------------------------------------------------------

@_requires_ffmpeg
def test_qa_stage_runs_through_pipeline_runner(tmp_path: Path):
    """End-to-end: build a tiny video, run the full pipeline with
    qa enabled, assert one QA row lands in m3_qa_results.jsonl."""
    # 30-second synthetic clip. Filename stem matters: m3.preprocess
    # derives video_id from the source filename, so we name it
    # vid_q.mp4 to match the annotation key.
    video = tmp_path / "vid_q.mp4"
    subprocess.run([
        _FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=30:r=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(video),
    ], check=True)

    # Real annotation file at the path the pipeline reads from
    # (work_dir/annotations/one_video.json, the same artifact that
    # write_single_annotation would write). We disable that stage
    # in the spec below and inject our own file instead.
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    annotations_dir = work_dir / "annotations"
    annotations_dir.mkdir()
    annotations = annotations_dir / "one_video.json"
    annotations.write_text(json.dumps({
        "vid_q": {"video_path": str(video),
                  "qa_list": [{"question_id": "q1",
                               "question": "What colour is the wall?",
                               "answer": "blue", "type": "object"}]},
    }), encoding="utf-8")

    # Patch the answer LLM factory (via m3.eval, the path run_react_qa
    # uses for build_llm) so we can drive the adapter with a scripted
    # mock. The runner still builds the graph + op_service normally;
    # only the LLM is mocked.
    spec = load_pipeline_spec("config/m3/pipelines/video_episodic_qa.yaml")
    # Detect faces is disabled by default; build_index we enable
    # so qa can run after it. qa is enabled.
    # write_single_annotation generates an empty-qa_list one_video.json
    # — disable it so we can inject our real annotation file before
    # the pipeline runs (qa reads the same path).
    for stage in spec.stages:
        if stage.id == "build_index":
            stage.enabled = True
            stage.params["embedder"] = "mock"
            stage.params["embedding_dimension"] = 8
        elif stage.id == "episodic_memory":
            stage.params["llm"] = "mock"
        elif stage.id == "qa":
            stage.params["embedder"] = "mock"
            stage.params["embedding_dimension"] = 8
        elif stage.id == "annotate":
            stage.enabled = False

    runner = PipelineRunner()
    ctx = runner.build_context(
        spec,
        video=video,
        video_id="vid_q",
        seconds=30,
        work_dir=work_dir,
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory",
        connection_string=None,
        account_id=M3BENCH_ACCOUNT_ID,
    )

    from providers.vector_index.in_memory_index import InMemoryVectorIndex
    ctx.vector_index = InMemoryVectorIndex(dimension=8)

    # Patch the answer LLM factory — run_react_qa lazy-imports
    # build_llm from m3.qa, so the patch must be on m3.qa, not
    # m3.eval.
    import m3.qa as qa_mod
    scripted = _ScriptedAdapterLLM("blue")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(qa_mod, "build_llm", lambda *a, **kw: scripted)
    try:
        summary = runner.run(spec, ctx)
    finally:
        monkey.undo()

    qa_stage_result = summary["stages"]["qa"]
    assert qa_stage_result["ok"] is True
    qa_results_path = work_dir / "m3_qa_results.jsonl"
    assert qa_results_path.exists()
    records = [json.loads(l) for l in qa_results_path.read_text().splitlines() if l]
    assert len(records) == 1
    assert records[0]["question_id"] == "q1"
    assert records[0]["predicted_answer"] == "blue"


def test_qa_stage_propagates_rc_two_as_warning(tmp_path: Path):
    """The stage must NOT abort the pipeline when run_react_qa
    returns rc=2 (partial failures). It should surface the failure
    count as a warning so the rest of the pipeline can continue."""
    from m3.pipeline.context import PipelineContext
    from m3.pipeline.spec import StageSpec
    from m3.pipeline import stages as stages_mod

    annotations = tmp_path / "ann.json"
    _write_annotations(annotations, [
        {"question_id": "q1", "question": "Q?", "answer": "A", "type": "object"},
    ])

    ctx = PipelineContext(
        video=tmp_path / "src.mp4",
        video_id="vid_q", seconds=30, work_dir=tmp_path / "run",
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory", connection_string=None,
        account_id=M3BENCH_ACCOUNT_ID,
        artifacts={
            "annotations": annotations,
            "qa_results": tmp_path / "out.jsonl",
        },
        variables={},
    )
    stage = StageSpec(
        id="qa", type="m3.qa",
        needs=["build_index"], enabled=True,
        params={"embedder": "mock", "llm": "mock"},
    )

    # Patch the underlying run_react_qa to return rc=2 + write one
    # row with error set so the per-question summary reflects the
    # partial failure.
    import m3.qa as qa_mod
    def fake_run_react_qa(args, graph=None, embedder=None, llm=None):
        from pathlib import Path
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "video_id": "vid_q", "question_id": "q1",
            "question_index": 0, "question": "Q?",
            "expected_answer": "A", "predicted_answer": "",
            "type": "object", "timestamp": None, "before_clip": None,
            "iterations": 0, "finish_reason": "",
            "search_rounds": 0, "search_history": [],
            "error": "boom",
        }) + "\n", encoding="utf-8")
        return 2
    monkey = pytest.MonkeyPatch()
    monkey.setattr(qa_mod, "run_react_qa", fake_run_react_qa)
    try:
        result = stages_mod.qa_stage(ctx, stage)
    finally:
        monkey.undo()

    assert result.ok is True
    assert result.counts["records"] == 1
    assert result.counts["failed"] == 1
    assert result.counts["invalid"] == 0
    assert any("partial failures" in w for w in result.warnings)


def test_qa_stage_raises_on_unexpected_rc(tmp_path: Path):
    """Anything other than 0 / 2 (e.g. rc=1 from missing annotations)
    must raise so the runner marks the stage failed."""
    from m3.pipeline.context import PipelineContext
    from m3.pipeline.spec import StageSpec
    from m3.pipeline import stages as stages_mod

    ctx = PipelineContext(
        video=tmp_path / "src.mp4",
        video_id="vid_q", seconds=30, work_dir=tmp_path,
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory", connection_string=None,
        account_id=M3BENCH_ACCOUNT_ID,
        artifacts={
            "annotations": tmp_path / "ann.json",
            "qa_results": tmp_path / "out.jsonl",
        },
        variables={},
    )
    stage = StageSpec(
        id="qa", type="m3.qa",
        needs=["build_index"], enabled=True,
        params={"embedder": "mock", "llm": "mock"},
    )

    import m3.qa as qa_mod
    monkey = pytest.MonkeyPatch()
    monkey.setattr(qa_mod, "run_react_qa", lambda *a, **kw: 1)
    try:
        with pytest.raises(RuntimeError, match="m3.qa.run_react_qa returned 1"):
            stages_mod.qa_stage(ctx, stage)
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# Hardening: detection of LLM transport failures and [Search]/[Answer] leaks
# ---------------------------------------------------------------------------

class _ScaffoldingLLM(MockLLM):
    """LLM mock that emits the ``[Search]`` / ``[Answer]`` protocol text
    in its content stream instead of using the tool_calls channel.

    Used to verify that ``run_react_qa`` flags the prediction as
    failed (the LLM forgot to call the tool and instead echoed the
    protocol text — the answer lands in ``predicted_answer`` verbatim
    via ``_parse_output``).
    """
    def __init__(self, content: str = "[Search] for X [Answer] X is the answer"):
        super().__init__()
        self._mock_content = content
        self._call_count = 0


class _RaisingLLM(MockLLM):
    """LLM mock whose ``complete_with_tools_messages`` raises
    :class:`LLMCallError` on every call — simulates a 400 / 500 from
    the upstream provider.
    """
    def __init__(self, error: Exception):
        super().__init__()
        self._error = error
        self._call_count = 0

    def complete_with_tools_messages(self, messages, tools=None, tool_choice="auto"):
        self._call_count += 1
        raise self._error


def test_run_react_qa_marks_protocol_scaffolding_as_failed(tmp_path: Path):
    """When the LLM emits ``[Search]`` / ``[Answer]`` literal text in
    its content (instead of using the tool_calls channel), the driver
    must mark the row as failed and write ``predicted_answer=""`` —
    ONLY if no answer is recoverable. A pure ``[Search]`` leak (no
    ``[Answer]`` tag) has nothing to salvage and stays a failure.
    """
    annotations = _write_annotations(tmp_path / "ann.json", [
        {"question_id": "q1", "question": "Where is X?",
         "answer": "kitchen", "type": "object"},
    ])
    output_jsonl = tmp_path / "results.jsonl"

    graph = GraphService(MemoryGraphStore())
    _seed_episode(graph, _ctx(), video_id="vid_q", clip_id=0, description="x")

    args = parse_react_qa_args([
        "--annotations", str(annotations),
        "--output-jsonl", str(output_jsonl),
        "--graph-store", "memory",
    ])
    rc = run_react_qa(
        args, graph=graph,
        embedder=MockEmbedder(dimension=8),
        llm=_ScaffoldingLLM("[Search] where is X"),
    )
    assert rc == 2  # partial failures — failed > 0 and answered == 0

    records = [json.loads(l) for l in output_jsonl.read_text().splitlines() if l]
    assert len(records) == 1
    assert records[0]["question_id"] == "q1"
    assert records[0]["predicted_answer"] == ""
    assert "scaffolding" in records[0]["error"]
    assert records[0]["finish_reason"] == "done"  # loop completed; the
    # leak was caught by the driver, not by the loop. The point is
    # that the row is marked failed so the stage summary doesn't
    # count it as answered.


def test_run_react_qa_salvages_answer_tag_leak(tmp_path: Path):
    """When the LLM emits ``[Answer] <real answer text>`` as content
    (in lieu of calling the tool), the salvage helper recovers it and
    the row is recorded as answered, NOT failed (FIX-4).
    """
    annotations = _write_annotations(tmp_path / "ann.json", [
        {"question_id": "q1", "question": "Where is X?",
         "answer": "kitchen", "type": "object"},
    ])
    output_jsonl = tmp_path / "results.jsonl"

    graph = GraphService(MemoryGraphStore())
    _seed_episode(graph, _ctx(), video_id="vid_q", clip_id=0, description="x")

    args = parse_react_qa_args([
        "--annotations", str(annotations),
        "--output-jsonl", str(output_jsonl),
        "--graph-store", "memory",
    ])
    rc = run_react_qa(
        args, graph=graph,
        embedder=MockEmbedder(dimension=8),
        llm=_ScaffoldingLLM("[Answer] The man is in the kitchen"),
    )
    assert rc == 0  # answered > 0; no failures

    records = [json.loads(l) for l in output_jsonl.read_text().splitlines() if l]
    assert len(records) == 1
    assert records[0]["question_id"] == "q1"
    # Salvaged text (trailing period stripped by the helper's trim set).
    assert records[0]["predicted_answer"] == "The man is in the kitchen"
    # Salvage is successful; audit information must not mark it failed.
    assert not records[0]["error"]
    assert "scaffolding" in records[0]["warning"]
    assert records[0]["finish_reason"] == "done"


def test_run_react_qa_marks_llm_error_as_failed(tmp_path: Path):
    """When the LLM raises ``LLMCallError`` (e.g. 400), the driver
    must mark the row as ``error="llm_error: ..."`` instead of
    emitting whatever the loop eventually produces."""
    from core.errors import LLMCallError

    annotations = _write_annotations(tmp_path / "ann.json", [
        {"question_id": "q1", "question": "Q?",
         "answer": "A", "type": "object"},
    ])
    output_jsonl = tmp_path / "results.jsonl"

    graph = GraphService(MemoryGraphStore())
    _seed_episode(graph, _ctx(), video_id="vid_q", clip_id=0, description="x")

    args = parse_react_qa_args([
        "--annotations", str(annotations),
        "--output-jsonl", str(output_jsonl),
        "--graph-store", "memory",
    ])
    rc = run_react_qa(
        args, graph=graph,
        embedder=MockEmbedder(dimension=8),
        llm=_RaisingLLM(LLMCallError("OpenAI BadRequestError (400): bad tool",
                                     status_code=400)),
    )
    assert rc == 2

    records = [json.loads(l) for l in output_jsonl.read_text().splitlines() if l]
    assert len(records) == 1
    assert records[0]["question_id"] == "q1"
    assert records[0]["predicted_answer"] == ""
    assert "llm_error" in records[0]["error"]
    assert "400" in records[0]["error"]


def test_qa_stage_counts_invalid_separately(tmp_path: Path):
    """The qa_stage flags rows with ``[Search]`` / ``[Answer]`` in
    their ``predicted_answer`` as ``invalid`` (defense in depth) —
    the count is excluded from ``answered`` and surfaced as a warning.
    """
    from m3.pipeline.context import PipelineContext
    from m3.pipeline.spec import StageSpec
    from m3.pipeline import stages as stages_mod

    output_jsonl = tmp_path / "results.jsonl"
    # Two rows: one clean, one with scaffolding. Neither has ``error``
    # set — so the driver-level ``run_react_qa`` "scaffolding" branch
    # wasn't engaged. The stage-level check should still catch the
    # scaffolding row when counting.
    output_jsonl.write_text(
        json.dumps({
            "video_id": "vid_q", "question_id": "q1",
            "question_index": 0, "question": "Q1?",
            "expected_answer": "A1", "predicted_answer": "the woman is in the kitchen",
            "type": "object", "timestamp": None, "before_clip": None,
            "iterations": 2, "finish_reason": "done",
            "search_rounds": 1, "search_history": [],
            "error": "",
        }) + "\n" +
        json.dumps({
            "video_id": "vid_q", "question_id": "q2",
            "question_index": 1, "question": "Q2?",
            "expected_answer": "A2",
            "predicted_answer": "[Search] for blue car [Answer] blue car",
            "type": "object", "timestamp": None, "before_clip": None,
            "iterations": 2, "finish_reason": "done",
            "search_rounds": 1, "search_history": [],
            "error": "",
        }) + "\n",
        encoding="utf-8",
    )

    annotations = _write_annotations(tmp_path / "ann.json", [
        {"question_id": "q1", "question": "Q1?", "answer": "A1", "type": "object"},
        {"question_id": "q2", "question": "Q2?", "answer": "A2", "type": "object"},
    ])

    ctx = PipelineContext(
        video=tmp_path / "src.mp4",
        video_id="vid_q", seconds=30, work_dir=tmp_path / "run",
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory", connection_string=None,
        account_id=M3BENCH_ACCOUNT_ID,
        artifacts={
            "annotations": annotations,
            "qa_results": output_jsonl,
        },
        variables={},
    )
    stage = StageSpec(
        id="qa", type="m3.qa",
        needs=["build_index"], enabled=True,
        params={"embedder": "mock", "llm": "mock",
                # Point the stage at our pre-seeded JSONL so its
                # counting logic actually reads the rows.
                "output_jsonl": str(output_jsonl)},
    )

    # Patch run_react_qa so it doesn't re-run the adapter — we just
    # want to exercise the stage's counting logic on the pre-seeded
    # JSONL.
    import m3.qa as qa_mod

    def fake_run_react_qa(args, graph=None, embedder=None, llm=None):
        return 0

    monkey = pytest.MonkeyPatch()
    monkey.setattr(qa_mod, "run_react_qa", fake_run_react_qa)
    try:
        result = stages_mod.qa_stage(ctx, stage)
    finally:
        monkey.undo()

    assert result.ok is True
    assert result.counts["records"] == 2
    assert result.counts["answered"] == 1  # q1 only; q2 excluded
    assert result.counts["invalid"] == 1
    assert result.counts["failed"] == 0
    assert any("scaffolding" in w for w in result.warnings)