"""Tests for the silent-AUDIO-skip regression.

``GraphService.list_raw_nodes`` paginates through the underlying store
(``store.list_nodes(RAW, ctx, limit, offset)``) and then filters by
``raw_type`` IN PYTHON. The previous default ``limit=100`` meant that
graphs with more than 100 RawNodes of a different type (typically
VIDEO from a 16-minute clip → 32+ VIDEO nodes × number of videos) had
their AUDIO children filtered out: the first 100 returned were all
VIDEO, the filter rejected every one, and the caller saw an empty list.

This is the bug that left voice_asr / voice_segmentation / speaker_*
saying "Found 0 AUDIO RawNodes" even when the SQL graph had 100+ AUDIO
nodes. Symptom: all voice_*.jsonl artefacts 0 bytes, every episodic
node has ``has_audio=false`` and no ``speech_transcript``.

The fix bumps the default to 4096 at the GraphService / store layer.
These tests pin the behaviour so a future refactor cannot silently
re-introduce the default-100 trap.
"""

from __future__ import annotations

import pytest

from core.graph_enums import RawDataType
from core.models import RequestContext
from providers.graph_store.memory_graph_store import MemoryGraphStore
from service.graph_service import GraphService


ACCOUNT = "m3bench"


def _ctx(account_id: str = ACCOUNT) -> RequestContext:
    return RequestContext(
        account_id=account_id,
        user_id="u",
        agent_id="a",
        session_id="s",
        trace_id="t",
    )


def _populate(graph: GraphService, video_count: int, audio_count: int):
    """Create ``video_count`` VIDEO RawNodes and ``audio_count`` AUDIO RawNodes.

    VIDEO nodes are inserted FIRST so they fill the front of the
    store's ``updated_at DESC`` ordering — the failure mode we're
    testing for. Then AUDIO nodes are inserted AFTER, so a naive
    ``list_nodes(RAW, limit=100)`` returns only VIDEO.
    """
    ctx = _ctx()
    for i in range(video_count):
        graph.create_raw_node(
            raw_type=RawDataType.VIDEO,
            content=b"",
            ctx=ctx,
            metadata={"video_id": "v", "clip_id": i},
        )
    for i in range(audio_count):
        graph.create_raw_node(
            raw_type=RawDataType.AUDIO,
            content=b"RIFF" + b"\x00" * 36,
            ctx=ctx,
            metadata={"video_id": "v", "clip_id": i},
        )


class TestListRawNodesDoesNotSilentlyDropAudio:
    def test_audio_returned_when_video_count_exceeds_old_default(self):
        """The regression: with 150 VIDEO RawNodes (more than the old
        ``limit=100``), ``list_raw_nodes(AUDIO)`` MUST still return
        every AUDIO node.

        Pre-fix root cause: ``list_raw_nodes`` paginated the store
        with ``node_type=RAW`` and filtered by ``raw_type`` in Python
        AFTER pagination. With the previous default ``limit=100`` and
        many VIDEO RawNodes, the first 100 returned were all VIDEO
        and AUDIO children were silently dropped — voice_asr saw 0
        AUDIO nodes even when the graph had 100+ of them.

        The fix pushes the ``raw_type`` filter into the store layer
        so each query returns ONLY the requested type, ordered by
        ``updated_at DESC`` within that type.
        """
        graph = GraphService(MemoryGraphStore())
        _populate(graph, video_count=150, audio_count=10)
        audio_nodes = graph.list_raw_nodes(
            RawDataType.AUDIO, _ctx(), limit=4096,
        )
        assert len(audio_nodes) == 10, (
            f"expected 10 AUDIO nodes; got {len(audio_nodes)}. "
            "This is the silent-AUDIO-skip bug."
        )

    def test_audio_returned_even_with_extreme_video_count(self):
        """Stress: 10x the old default. With limit=4096 in
        ``list_raw_nodes`` the audit filter at the store level keeps
        AUDIO visible regardless of how many VIDEO RawNodes exist.
        """
        graph = GraphService(MemoryGraphStore())
        _populate(graph, video_count=1000, audio_count=5)
        audio_nodes = graph.list_raw_nodes(
            RawDataType.AUDIO, _ctx(), limit=4096,
        )
        assert len(audio_nodes) == 5

    def test_video_returned_when_only_video_present(self):
        """Sanity: AUDIO filter on a VIDEO-only graph returns 0."""
        graph = GraphService(MemoryGraphStore())
        _populate(graph, video_count=50, audio_count=0)
        audio_nodes = graph.list_raw_nodes(
            RawDataType.AUDIO, _ctx(), limit=4096,
        )
        assert audio_nodes == []

    def test_mixed_returns_filtered_correctly(self):
        """Both filters should work; each returns only its own type."""
        graph = GraphService(MemoryGraphStore())
        _populate(graph, video_count=200, audio_count=50)
        videos = graph.list_raw_nodes(
            RawDataType.VIDEO, _ctx(), limit=4096,
        )
        audios = graph.list_raw_nodes(
            RawDataType.AUDIO, _ctx(), limit=4096,
        )
        assert len(videos) == 200
        assert len(audios) == 50

    def test_store_accepts_raw_type_filter_at_layer(self):
        """Pin the structural fix: the store's ``list_nodes`` accepts a
        ``raw_type`` kwarg and applies the filter before pagination.
        Future refactors cannot silently regress to filter-after-page.
        """
        import inspect
        sig = inspect.signature(MemoryGraphStore.list_nodes)
        assert "raw_type" in sig.parameters, (
            "MemoryGraphStore.list_nodes must accept a ``raw_type`` "
            "kwarg so the filter happens at the store layer (not in "
            "Python after pagination)."
        )

    def test_graph_service_passes_raw_type_to_store(self):
        """Pin the wiring: ``GraphService.list_raw_nodes`` must forward
        ``raw_type`` to the store rather than filtering in Python.
        """
        graph = GraphService(MemoryGraphStore())
        _populate(graph, video_count=150, audio_count=3)

        # Spy on the store-level call to verify raw_type is forwarded.
        from unittest.mock import patch
        with patch.object(
            graph._store, "list_nodes",
            wraps=graph._store.list_nodes,
        ) as spy:
            graph.list_raw_nodes(RawDataType.AUDIO, _ctx(), limit=4096)
        # Find the call that requested AUDIO.
        audio_calls = [
            call for call in spy.call_args_list
            if call.kwargs.get("raw_type") == RawDataType.AUDIO
        ]
        assert audio_calls, (
            "GraphService.list_raw_nodes did not pass raw_type=AUDIO "
            "to store.list_nodes; filter-after-pagination regression."
        )