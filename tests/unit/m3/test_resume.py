"""Tests for :mod:`m3.resume` — graph-aware resume helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from m3 import resume
from m3.episodic_memory import extracted_node_id


# ---------------------------------------------------------------------------
# video_fully_indexed — probe logic
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(self, node_id: str, video_id: str | None = None,
                 extraction_status: str = "completed",
                 embedding: list[float] | None = None):
        self.node_id = node_id
        self.metadata = {"video_id": video_id} if video_id is not None else {}
        self.extraction_status = extraction_status
        if embedding is not None:
            self.metadata["embedding"] = embedding


def _fake_graph(video_nodes: list[Any], extracted_nodes: dict[str, Any]):
    """Build a mock GraphService that returns the given VIDEO nodes and ExtractedNodes."""

    def list_raw_nodes(raw_type, ctx, limit=1000, offset=0):
        return video_nodes[offset:offset + limit]

    def get_extracted_node(ext_id, ctx):
        return extracted_nodes.get(ext_id)

    graph = MagicMock()
    graph.list_raw_nodes.side_effect = list_raw_nodes
    graph.get_extracted_node.side_effect = get_extracted_node
    return graph


def test_video_fully_indexed_returns_complete_when_all_extracted():
    """All VIDEO clips have completed ExtractedNodes with embeddings → complete=True."""
    video_nodes = [
        _FakeNode("vnode_1", video_id="bedroom_03"),
        _FakeNode("vnode_2", video_id="bedroom_03"),
    ]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", embedding=[0.1, 0.2, 0.3]
        ),
        extracted_node_id("episodic_memory_generator", "vnode_2"): _FakeNode(
            "vnode_2", embedding=[0.4, 0.5, 0.6]
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["complete"] is True
    assert result["video_clip_count"] == 2
    assert result["missing_extracted"] == []
    assert result["unfinished_extracted"] == []
    assert result["missing_embedding"] == []


def test_video_fully_indexed_reports_missing_extracted():
    """ExtractedNode not in graph → reported as missing, complete=False."""
    video_nodes = [_FakeNode("vnode_1", video_id="bedroom_03")]
    graph = _fake_graph(video_nodes, extracted_nodes={})
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["complete"] is False
    assert result["video_clip_count"] == 1
    assert result["missing_extracted"] == [
        extracted_node_id("episodic_memory_generator", "vnode_1")
    ]


def test_video_fully_indexed_reports_unfinished_extractions():
    """Pending/failed ExtractedNodes are not reused → reported as unfinished."""
    video_nodes = [_FakeNode("vnode_1", video_id="bedroom_03")]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", extraction_status="pending"
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["complete"] is False
    assert len(result["unfinished_extracted"]) == 1


def test_video_fully_indexed_reports_missing_embedding():
    """Completed ExtractedNode without embedding → reported as missing_embedding."""
    video_nodes = [_FakeNode("vnode_1", video_id="bedroom_03")]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", extraction_status="completed", embedding=None
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["complete"] is False
    assert len(result["missing_embedding"]) == 1


def test_video_fully_indexed_no_video_nodes_returns_incomplete():
    """No VIDEO RawNodes for this video → complete=False, no crash."""
    graph = _fake_graph([], extracted_nodes={})
    result = resume.video_fully_indexed(graph, "m3bench", "missing_video")
    assert result["complete"] is False
    assert result["video_clip_count"] == 0


def test_video_fully_indexed_filters_other_videos():
    """VIDEO RawNodes from other videos must be ignored."""
    video_nodes = [
        _FakeNode("vnode_1", video_id="bedroom_03"),
        _FakeNode("vnode_2", video_id="living_room_06"),  # different video
    ]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", embedding=[0.1]
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["video_clip_count"] == 1
    assert result["complete"] is True


def test_video_fully_indexed_can_skip_embedding_requirement():
    """require_embeddings=False bypasses the embedding check."""
    video_nodes = [_FakeNode("vnode_1", video_id="bedroom_03")]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", extraction_status="completed", embedding=None
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(
        graph, "m3bench", "bedroom_03", require_embeddings=False
    )
    assert result["complete"] is True


# ---------------------------------------------------------------------------
# write_resume_summary
# ---------------------------------------------------------------------------


def test_write_resume_summary_creates_minimal_summary(tmp_path: Path):
    """write_resume_summary emits a summary JSON the single-video CLI can be discovered by."""
    jsonl = tmp_path / "m3_qa_results.jsonl"
    jsonl.write_text("", encoding="utf-8")
    summary_path = resume.write_resume_summary(
        work_dir=tmp_path,
        video_id="bedroom_03",
        pipeline_name="video_episodic_full",
        pipeline_version=1,
        jsonl_path=jsonl,
        duration_sec=12.3,
        rc=0,
    )
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["video_id"] == "bedroom_03"
    assert summary["mode"] == "qa_only"
    assert summary["stages"]["qa_only_resume"]["ok"] is True
    assert summary["artifacts"]["qa_results"] == str(jsonl)


# ---------------------------------------------------------------------------
# build_graph_service — cycle workaround
# ---------------------------------------------------------------------------


def test_build_graph_service_memory_returns_memory_store():
    """In-memory store path returns a GraphService backed by MemoryGraphStore."""
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService

    graph, kind = resume.build_graph_service("memory", None)
    assert kind == "memory"
    assert isinstance(graph, GraphService)
    # Underlying store is MemoryGraphStore (or its subclass).
    store = graph._store  # type: ignore[attr-defined]
    assert isinstance(store, MemoryGraphStore)


def test_build_graph_service_sql_without_dsn_raises():
    """SQL store requested but no DSN → RuntimeError."""
    import os
    os.environ.pop("OGMEM_GRAPH_CONNECTION_STRING", None)
    with pytest.raises(RuntimeError, match="no DSN provided"):
        resume.build_graph_service("sql", None)


def test_build_graph_service_sql_with_connection_string(monkeypatch):
    """SQL store path returns a GraphService backed by SQLGraphStore.

    We patch ``SQLGraphStore`` so the test doesn't try to open a real
    database connection (the real constructor eagerly runs
    ``_ensure_schema``).
    """
    from service.graph_service import GraphService

    sentinel = MagicMock()
    monkeypatch.setattr(
        "providers.graph_store.SQLGraphStore", lambda connection_string: sentinel
    )

    graph, kind = resume.build_graph_service(
        "sql", "postgresql://user:pw@host:5432/db"
    )
    assert kind == "sql"
    assert isinstance(graph, GraphService)
    assert graph._store is sentinel  # type: ignore[attr-defined]


def test_build_graph_service_unknown_kind_raises():
    """Unknown graph_store value → ValueError."""
    with pytest.raises(ValueError, match="Unknown graph store"):
        resume.build_graph_service("redis", None)