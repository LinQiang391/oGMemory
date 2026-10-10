"""Unit tests for the ``m3.eval`` pipeline stage wrapper.

The wrapper exists so a declarative pipeline can run the judge pass
as a first-class stage. These tests cover the argv-assembly surface
(the contract the rest of the pipeline depends on) without spinning
up the full judge LLM — that's covered by ``test_eval_harness.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from m3.pipeline.context import PipelineContext
from m3.pipeline.spec import StageSpec
from m3.pipeline.stages import eval_stage


def _ctx(tmp_path: Path, *, annotations: Path) -> PipelineContext:
    return PipelineContext(
        video=tmp_path / "video.mp4",
        video_id="vid_001",
        seconds=60,
        work_dir=tmp_path,
        config_path=tmp_path / "ogmemory.yaml",
        graph_store="memory",
        connection_string=None,
        account_id="m3-bench",
        artifacts={"annotations": annotations},
        variables={},
        graph=SimpleNamespace(name="graph-fixture"),
    )


def _stage(**params) -> StageSpec:
    return StageSpec(id="eval", type="m3.eval", params=params)


def _fake_args(argv):
    """Minimal stand-in for ``m3.eval.parse_args`` that extracts just
    enough of the argv for the stage's downstream fake_run to write a
    summary.json into the right directory. Ignores everything else."""
    out = SimpleNamespace(
        annotations=None, output_dir=None, graph_store=None,
        connection_string=None, config=None, video=None, type_filter=None,
        limit=None, resume=False, max_iterations=None, timeout_seconds=None,
        default_video_id=None, embedder=None, embedding_model=None,
        embedding_base_url=None, embedding_api_key=None,
        embedding_dimension=None, llm=None, llm_model=None,
        llm_base_url=None, llm_api_key=None, temperature=None,
        judge_base_url=None, judge_model=None,
        judge_api_key=None, input_qa_results=None, verbose=False,
    )
    it = iter(argv)
    for tok in it:
        nxt = next(it, None)
        if tok == "--annotations":
            out.annotations = nxt
        elif tok == "--output-dir":
            out.output_dir = nxt
        elif tok == "--graph-store":
            out.graph_store = nxt
        elif tok == "--config":
            out.config = nxt
        # All other flags ignored — the fake stage doesn't exercise them.
    return out


def test_eval_stage_forwards_required_args(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_parse_args(argv):
        captured["argv"] = argv
        return _fake_args(argv)

    def fake_run(args):
        # Write a fake summary the stage can read back.
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "m3_eval_summary.json").write_text(json.dumps({
            "total_questions": 3,
            "answered": 3,
            "failed": 0,
            "judge_error_count": 0,
            "overall_mean_score": 1.0,
        }))
        (out / "m3_eval_results.jsonl").write_text("{}\n{}\n{}\n")
        return 0

    monkeypatch.setattr("m3.eval.parse_args", fake_parse_args)
    monkeypatch.setattr("m3.eval.run", fake_run)

    annotations = tmp_path / "annotations.json"
    annotations.write_text("[]")
    result = eval_stage(_ctx(tmp_path, annotations=annotations), _stage())

    argv = captured["argv"]
    assert "--annotations" in argv
    assert str(annotations) in argv
    assert "--graph-store" in argv
    assert "memory" in argv
    assert "--config" in argv

    assert result.counts["total_questions"] == 3
    assert result.counts["answered"] == 3
    assert result.counts["judge_error_count"] == 0
    assert result.counts["mean_score"] == 1.0
    assert result.counts["records"] == 3


def test_eval_stage_forwards_optional_args(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_parse_args(argv):
        captured["argv"] = argv
        return _fake_args(argv)

    def fake_run(args):
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "m3_eval_summary.json").write_text("{}")
        (out / "m3_eval_results.jsonl").write_text("")
        return 0

    monkeypatch.setattr("m3.eval.parse_args", fake_parse_args)
    monkeypatch.setattr("m3.eval.run", fake_run)

    annotations = tmp_path / "annotations.json"
    annotations.write_text("[]")
    result = eval_stage(
        _ctx(tmp_path, annotations=annotations),
        _stage(
            embedder="openai",
            embedding_dimension=1024,
            llm="minimax",
            temperature=0.2,
            judge_base_url="https://api.example.com/v1",
            judge_model="some-judge-model",
            judge_api_key="judge-key-xyz",
            max_iterations=8,
            timeout_seconds=120.0,
            default_video_id="vid_001",
            video="vid_001,vid_002",
            type_filter="object,temporal",
            limit=10,
            resume=True,
        ),
    )
    argv = captured["argv"]

    # Every optional knob must round-trip into the CLI.
    must_have = [
        "--embedder", "openai",
        "--embedding-dimension", "1024",
        "--llm", "minimax",
        "--temperature", "0.2",
        "--judge-base-url", "https://api.example.com/v1",
        "--judge-model", "some-judge-model",
        "--judge-api-key", "judge-key-xyz",
        "--max-iterations", "8",
        "--timeout-seconds", "120.0",
        "--default-video-id", "vid_001",
        "--video", "vid_001,vid_002",
        "--type-filter", "object,temporal",
        "--limit", "10",
        "--resume",
    ]
    for tok in must_have:
        assert tok in argv, f"missing {tok} in {argv}"

    # Resume goes through as a bare flag (no value).
    assert "--resume" in argv and argv.count("--resume") == 1

    # No records on disk → counts fall back to zeros, no exception.
    assert result.counts["records"] == 0


def test_eval_stage_surfaces_partial_failure(monkeypatch, tmp_path):
    def fake_parse_args(argv):
        return _fake_args(argv)

    def fake_run(args):
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "m3_eval_summary.json").write_text(json.dumps({
            "total_questions": 5,
            "answered": 2,
            "failed": 3,
            "judge_error_count": 0,
            "overall_mean_score": 0.4,
        }))
        return 2  # rc=2 = partial failures, not fatal

    monkeypatch.setattr("m3.eval.parse_args", fake_parse_args)
    monkeypatch.setattr("m3.eval.run", fake_run)

    annotations = tmp_path / "annotations.json"
    annotations.write_text("[]")
    result = eval_stage(_ctx(tmp_path, annotations=annotations), _stage())
    assert result.counts["failed"] == 3
    assert result.counts["mean_score"] == pytest.approx(0.4)
    assert any("partial failures" in w for w in result.warnings)


def test_eval_stage_raises_on_fatal(monkeypatch, tmp_path):
    def fake_parse_args(argv):
        return _fake_args(argv)

    monkeypatch.setattr("m3.eval.parse_args", fake_parse_args)
    monkeypatch.setattr("m3.eval.run", lambda args: 7)

    annotations = tmp_path / "annotations.json"
    annotations.write_text("[]")
    with pytest.raises(RuntimeError, match="returned 7"):
        eval_stage(_ctx(tmp_path, annotations=annotations), _stage())


def test_eval_stage_forwards_input_qa_results(monkeypatch, tmp_path):
    """``input_qa_results`` must round-trip into the CLI as
    ``--input-qa-results <path>`` so the eval stage can be run in
    judge-only mode after a pipeline QA pass has already finished."""
    captured: dict = {}

    def fake_parse_args(argv):
        captured["argv"] = argv
        return _fake_args(argv)

    def fake_run(args):
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "m3_eval_summary.json").write_text(
            json.dumps({
                "total_questions": 1,
                "answered": 1,
                "failed": 0,
                "judge_error_count": 0,
                "overall_mean_score": 1.0,
                "missing_from_input": 0,
                "input_qa_results": "ignored",
            })
        )
        (out / "m3_eval_results.jsonl").write_text("{}\n")
        return 0

    monkeypatch.setattr("m3.eval.parse_args", fake_parse_args)
    monkeypatch.setattr("m3.eval.run", fake_run)

    annotations = tmp_path / "annotations.json"
    annotations.write_text("[]")
    existing = tmp_path / "m3_qa_results.jsonl"
    existing.write_text("")
    result = eval_stage(
        _ctx(tmp_path, annotations=annotations),
        _stage(input_qa_results=str(existing)),
    )
    argv = captured["argv"]
    assert "--input-qa-results" in argv
    assert str(existing) in argv
    assert result.counts["missing_from_input"] == 0
    assert result.counts["answered"] == 1