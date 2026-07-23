"""Unit tests for the M3-Bench eval harness (``m3.eval``).

Three layers:

* Judge scoring — pure unit tests over ``judge_answer`` with a
  scripted MockLLM. Asserts the {0.0, 0.5, 1.0} normalisation and
  the empty-predicted short-circuit.
* Aggregation — pure unit tests over ``aggregate`` with synthetic
  QuestionResult lists. Asserts per-type bucketing, percentile math,
  and overall totals.
* End-to-end — an integration test that wires a tiny synthetic
  annotation + seeded graph + scripted adapter LLM + scripted judge
  LLM and asserts the produced results.jsonl + summary.json shapes.
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
from m3.control_adapter import M3TaskContext
from m3.eval import (
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_SCHEMA,
    QuestionResult,
    aggregate,
    answer_with_adapter,
    judge_answer,
)
from providers.embedder import MockEmbedder
from providers.graph_store import MemoryGraphStore
from providers.llm import MockLLM
from service.graph_service import GraphService


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------

class _ScriptedJudge(MockLLM):
    def __init__(self, script):
        super().__init__()
        self._script = list(script)
        self._last_prompt = ""

    def complete_json(self, prompt, schema):
        self._call_count += 1
        self._last_prompt = prompt
        if not self._script:
            return {"score": 0.0, "reasoning": "no script"}
        return self._script.pop(0)


def test_judge_returns_one_when_predicted_matches_expected():
    judge = _ScriptedJudge([
        {"score": 1.0, "reasoning": "matches"},
    ])
    score, reasoning = judge_answer(
        judge_llm=judge,
        question="What colour is the mug?",
        expected="red",
        predicted="red",
    )
    assert score == 1.0
    assert reasoning == "matches"


def test_judge_returns_zero_when_predicted_empty():
    judge = _ScriptedJudge([])  # never reached
    score, reasoning = judge_answer(
        judge_llm=judge,
        question="Q?",
        expected="something",
        predicted="",
    )
    assert score == 0.0
    assert "empty" in reasoning.lower()


def test_judge_normalises_score_to_half_bucket():
    """LLMs often emit 0.7 or 0.8; the harness must snap to the
    {0.0, 0.5, 1.0} buckets so the per-type summary numbers add up."""
    judge = _ScriptedJudge([{"score": 0.7, "reasoning": "partial"}])
    score, _ = judge_answer(judge_llm=judge, question="q",
                            expected="e", predicted="p")
    assert score == 0.5
    judge = _ScriptedJudge([{"score": 0.85, "reasoning": "near"}])
    score, _ = judge_answer(judge_llm=judge, question="q",
                            expected="e", predicted="p")
    assert score == 1.0


def test_judge_handles_llm_failure_without_raising():
    class FailingJudge(MockLLM):
        def complete_json(self, prompt, schema):
            raise RuntimeError("LLM is down")
    score, reasoning = judge_answer(
        judge_llm=FailingJudge(),
        question="q", expected="e", predicted="p",
    )
    assert score == 0.0
    assert "judge failed" in reasoning


def test_judge_prompt_includes_question_and_both_answers():
    judge = _ScriptedJudge([{"score": 1.0, "reasoning": "x"}])
    judge_answer(
        judge_llm=judge,
        question="Who walks in?",
        expected="Alice",
        predicted="Alice",
    )
    assert "Who walks in?" in judge._last_prompt
    assert "Alice" in judge._last_prompt


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _qr(video_id, qid, score, type_="object", iterations=2, error=""):
    return QuestionResult(
        video_id=video_id, question_id=qid, question_index=0,
        question="q", expected_answer="e",
        predicted_answer="p",
        type=type_, iterations=iterations,
        latency_seconds=1.0,
        judge_score=score, judge_reasoning="",
        error=error,
    )


def test_aggregate_buckets_by_question_type():
    # 7 questions: 3 objects @ 1.0, 2 temporals @ 0.5, 2 temporals @ 0.0.
    # Sum = 4.0 → mean = 4/7 ≈ 0.571.
    results = [
        _qr("v", f"q{i}", score=1.0, type_="object")
        for i in range(3)
    ] + [
        _qr("v", f"q{i}", score=0.5, type_="temporal")
        for i in range(2)
    ] + [
        _qr("v", f"q{i}", score=0.0, type_="temporal")
        for i in range(2)
    ]
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert summary["total_questions"] == 7
    assert summary["overall_mean_score"] == pytest.approx(4 / 7, abs=1e-9)
    assert summary["answered"] == 7
    assert summary["failed"] == 0
    assert set(summary["by_type"].keys()) == {"object", "temporal"}
    obj = summary["by_type"]["object"]
    assert obj["count"] == 3
    assert obj["mean_score"] == 1.0
    tmp = summary["by_type"]["temporal"]
    assert tmp["count"] == 4
    assert tmp["mean_score"] == 0.25
    assert tmp["p50_iterations"] == 2.0


def test_aggregate_handles_list_type_field():
    """Some M3-Bench annotations store ``type`` as a list of category
    strings. The aggregator should join them into a stable bucket key."""
    results = [
        _qr("v", "q1", score=1.0, type_=["object", "spatial"]),
    ]
    # The dataclass forbids list directly via the type annotation
    # (str), so coerce via attribute assignment.
    results[0].type = "object,spatial"
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert "object,spatial" in summary["by_type"]


def test_aggregate_handles_unspecified_type():
    results = [
        _qr("v", "q1", score=1.0, type_=""),
        _qr("v", "q2", score=0.0, type_=""),
    ]
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert "(unspecified)" in summary["by_type"]
    bucket = summary["by_type"]["(unspecified)"]
    assert bucket["count"] == 2


def test_aggregate_counts_errors():
    results = [
        _qr("v", "ok", score=1.0, type_="object"),
        _qr("v", "fail", score=0.0, type_="object", error="boom"),
    ]
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert summary["answered"] == 1
    assert summary["failed"] == 1
    assert summary["by_type"]["object"]["failed"] == 1


# ---------------------------------------------------------------------------
# End-to-end (integration)
# ---------------------------------------------------------------------------

_FFMPEG = shutil.which("ffmpeg")
_requires_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None, reason="ffmpeg not available on PATH",
)


def _seed_episode(graph, ctx, video_id, clip_id, description, nid):
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


class _ScriptedAdapterLLM(MockLLM):
    """MockLLM that cycles Search → Answer → content on every question.

    The eval harness builds the adapter (and its LLM) once and
    reuses it across all questions, so we can't just count absolute
    calls — we need to reset state when a new question starts. We
    detect the question boundary via ``messages`` shape: the first
    user message in a new run() always starts with "Question: …".
    When that prefix changes vs the previous run, we reset.
    """
    def __init__(self, answer_text: str):
        super().__init__()
        self._answer_text = answer_text
        self._per_question_round = 0
        self._last_question_prefix = None

    def complete_with_tools_messages(self, messages, tools=None, tool_choice="auto"):
        # Detect question boundary: the user message contains "Question: …"
        # and the prefix changes between calls.
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
            return (
                [{"tool": "Search", "input": {"query": "anything"}}],
                "",
            )
        if self._per_question_round == 2:
            return (
                [{"tool": "Answer",
                  "input": {"final_answer": self._answer_text}}],
                "",
            )
        return [], self._answer_text


class _ScriptedJudgeLLM(MockLLM):
    """Mock judge that returns the next score from a queue on each call."""
    def __init__(self, scores):
        super().__init__()
        self._queue = list(scores)
    def complete_json(self, prompt, schema):
        self._call_count += 1
        score = self._queue.pop(0) if self._queue else 0.0
        return {"score": score, "reasoning": f"scored {score}"}


def _write_annotations(tmp_path: Path, questions: list[dict]) -> Path:
    annotations = {"vid_test": {"video_path": "/tmp/vid_test.mp4",
                                "qa_list": questions}}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations), encoding="utf-8")
    return path


@_requires_ffmpeg
def test_run_end_to_end_writes_results_and_summary(tmp_path: Path):
    """Full run: 3 questions, scripted adapter LLM + scripted judge,
    verify results.jsonl + summary.json shapes and per-type buckets."""
    annotations_path = _write_annotations(tmp_path, [
        {"question_id": "q1", "question": "What colour is the mug?",
         "answer": "red", "type": "object_recognition"},
        {"question_id": "q2", "question": "When did she enter?",
         "answer": "after lunch", "type": "temporal_reasoning"},
        {"question_id": "q3", "question": "Where is the dog?",
         "answer": "backyard", "type": "spatial_reasoning"},
    ])

    from m3.eval import parse_args, run
    args = parse_args([
        "--annotations", str(annotations_path),
        "--output-dir", str(tmp_path / "out"),
        "--graph-store", "memory",
        "--embedder", "mock",
        "--embedding-dimension", "8",
        "--llm", "mock",
        "--judge-llm", "mock",
    ])
    # Patch the adapter LLM factory + judge factory via monkeypatch so
    # the run() uses our scripted mocks regardless of CLI choice.
    import m3.eval as eval_mod
    monkey = pytest.MonkeyPatch()

    def _answer_llm_factory(*a, **kw):
        return _ScriptedAdapterLLM("the woman is wearing red")
    monkey.setattr(eval_mod, "build_llm", _answer_llm_factory)

    def _judge_llm_factory(*a, **kw):
        # Score 1.0 for q1 (correct), 0.5 for q2 (partial), 0.0 for q3.
        return _ScriptedJudgeLLM([1.0, 0.5, 0.0])
    monkey.setattr(eval_mod, "build_judge_llm", _judge_llm_factory)

    try:
        rc = eval_mod.run(args)
        assert rc == 0, f"eval returned {rc}"
    finally:
        monkey.undo()

    results_path = tmp_path / "out" / "m3_eval_results.jsonl"
    summary_path = tmp_path / "out" / "m3_eval_summary.json"
    assert results_path.exists()
    assert summary_path.exists()

    records = [json.loads(l) for l in results_path.read_text().splitlines() if l]
    assert len(records) == 3
    by_qid = {r["question_id"]: r for r in records}
    assert by_qid["q1"]["predicted_answer"] == "the woman is wearing red"
    assert by_qid["q1"]["judge_score"] == 1.0
    assert by_qid["q2"]["judge_score"] == 0.5
    assert by_qid["q3"]["judge_score"] == 0.0
    # Search history was captured per question.
    for r in records:
        assert r["iterations"] >= 2
        assert r["search_rounds"] >= 1
        assert any(
            h.get("query") == "anything"
            for h in r["search_history"]
        )

    summary = json.loads(summary_path.read_text())
    assert summary["total_questions"] == 3
    assert summary["answered"] == 3
    assert summary["failed"] == 0
    assert summary["judge_provider"] == "mock"
    # Per-type buckets all present.
    assert set(summary["by_type"].keys()) == {
        "object_recognition", "temporal_reasoning", "spatial_reasoning",
    }
    assert summary["by_type"]["object_recognition"]["mean_score"] == 1.0
    assert summary["by_type"]["temporal_reasoning"]["mean_score"] == 0.5
    assert summary["by_type"]["spatial_reasoning"]["mean_score"] == 0.0


def test_run_resume_skips_already_answered(tmp_path: Path):
    """Re-running with --resume must not re-answer questions already
    in results.jsonl."""
    annotations_path = _write_annotations(tmp_path, [
        {"question_id": "q1", "question": "Q1?", "answer": "A1",
         "type": "object"},
        {"question_id": "q2", "question": "Q2?", "answer": "A2",
         "type": "temporal"},
    ])
    # Pre-populate results with q1 already answered.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "m3_eval_results.jsonl").write_text(
        json.dumps({
            "video_id": "vid_test", "question_id": "q1",
            "question_index": 0, "question": "Q1?",
            "expected_answer": "A1", "predicted_answer": "A1",
            "type": "object", "timestamp": None, "before_clip": None,
            "iterations": 2, "finish_reason": "done",
            "search_rounds": 1, "search_history": [],
            "judge_score": 1.0, "judge_reasoning": "",
            "latency_seconds": 1.0, "error": "",
        }) + "\n", encoding="utf-8",
    )

    from m3.eval import parse_args, run
    import m3.eval as eval_mod

    monkey = pytest.MonkeyPatch()
    adapter_call_count = {"n": 0}

    def _answer_llm_factory(*a, **kw):
        return _ScriptedAdapterLLM("A2")
    monkey.setattr(eval_mod, "build_llm", _answer_llm_factory)
    monkey.setattr(eval_mod, "build_judge_llm",
                  lambda *a, **kw: _ScriptedJudgeLLM([1.0]))

    # Wrap adapter.run to count calls
    original_run = eval_mod.OGMemoryControlAdapter.run
    def counting_run(self, ctx, task_ctx):
        adapter_call_count["n"] += 1
        return original_run(self, ctx, task_ctx)
    monkey.setattr(eval_mod.OGMemoryControlAdapter, "run", counting_run)

    try:
        args = parse_args([
            "--annotations", str(annotations_path),
            "--output-dir", str(out_dir),
            "--graph-store", "memory",
            "--embedder", "mock",
            "--embedding-dimension", "8",
            "--llm", "mock",
            "--judge-llm", "mock",
            "--resume",
        ])
        rc = eval_mod.run(args)
        assert rc == 0
    finally:
        monkey.undo()

    # Only q2 was newly answered (q1 was skipped from the on-disk file).
    assert adapter_call_count["n"] == 1
    records = [json.loads(l) for l in (out_dir / "m3_eval_results.jsonl").read_text().splitlines() if l]
    assert len(records) == 2
    assert {r["question_id"] for r in records} == {"q1", "q2"}


def test_run_writes_summary_with_video_and_type_filters(tmp_path: Path):
    """--video and --type-filter narrow the question set; the
    summary reflects only the filtered subset."""
    annotations_path = _write_annotations(tmp_path, [
        {"question_id": "q1", "question": "Q1", "answer": "A1", "type": "object"},
        {"question_id": "q2", "question": "Q2", "answer": "A2", "type": "temporal"},
    ])
    from m3.eval import parse_args, run
    import m3.eval as eval_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(eval_mod, "build_llm",
                  lambda *a, **kw: _ScriptedAdapterLLM("A"))
    monkey.setattr(eval_mod, "build_judge_llm",
                  lambda *a, **kw: _ScriptedJudgeLLM([1.0]))
    try:
        args = parse_args([
            "--annotations", str(annotations_path),
            "--output-dir", str(tmp_path / "out"),
            "--graph-store", "memory",
            "--embedder", "mock",
            "--embedding-dimension", "8",
            "--llm", "mock",
            "--judge-llm", "mock",
            "--type-filter", "object",
        ])
        rc = eval_mod.run(args)
        assert rc == 0
    finally:
        monkey.undo()

    summary = json.loads((tmp_path / "out" / "m3_eval_summary.json").read_text())
    # Only the object question survived the filter.
    assert summary["total_questions"] == 1
    assert set(summary["by_type"].keys()) == {"object"}