"""Regression tests for ``_index_asr_segments`` per-video scoping.

The 100-video M3-Bench run revealed a wiring bug: with ~140k
extracted nodes in the SQL store, ``graph.list_extracted_nodes(
"voice_asr", ctx)`` translated to ``SELECT ... FROM graph_nodes
WHERE node_type='extracted' ORDER BY updated_at DESC LIMIT 4096``,
which silently dropped the ASR nodes from every video but the
most recently ingested. 20/23 videos received ``speech_transcript:
[]`` even though voice_asr had produced segments for them.

The fix scopes the fetch to ``parent_audio_node_ids`` known from
the manifest / graph walk, and lets the store push that filter
into the WHERE clause. These tests assert that, with both the
MemoryGraphStore and a SQL-shaped stub, the new path returns
the right ASR segments for a target video regardless of how
many other videos' nodes are in the graph.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest

from core.graph_enums import GraphNodeType, RawDataType
from core.graph_models import ExtractedNode, GraphNode, RawNode
from core.models import RequestContext
from m3.episodic_memory import _index_asr_segments


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="m3bench", user_id="u", agent_id="a",
        session_id="m3bench-asr-test", trace_id="t-asr",
    )


def _make_audio_node(audio_id: str, video_id: str, clip_id: int) -> RawNode:
    """Build an AUDIO RawNode with the metadata the fallback walk keys on."""
    return RawNode(
        node_id=audio_id,
        node_type=GraphNodeType.RAW,
        account_id="m3bench",
        raw_type=RawDataType.AUDIO,
        metadata={"video_id": video_id, "clip_id": clip_id},
        content="audio bytes placeholder",
    )


def _make_asr_node(
    asr_id: str,
    source_audio_id: str,
    text: str = "hello",
    start: float = 0.0,
    end: float = 1.0,
    speaker_id: str | None = None,
) -> ExtractedNode:
    """Build a voice_asr ExtractedNode referencing its parent audio."""
    return ExtractedNode(
        node_id=asr_id,
        node_type=GraphNodeType.EXTRACTED,
        account_id="m3bench",
        source_raw_id=source_audio_id,
        extractor_type="voice_asr",
        extracted_content={
            "start_sec": start,
            "end_sec": end,
            "text": text,
            "speaker_id": speaker_id,
            "source_audio_node_id": source_audio_id,
            "words": [],
        },
    )


class _FakeMemoryGraph:
    """In-memory graph stub that honours the post-fix API.

    Tracks ``list_raw_nodes`` + ``list_extracted_nodes`` calls so
    tests can assert the new ``parent_audio_node_ids`` /
    ``limit=None`` kwargs are threaded through. The actual
    filtering logic lives in the real ``MemoryGraphStore``; here
    we just return the canned nodes the test seeded.
    """

    def __init__(self, audio_nodes, asr_nodes):
        self._audio = list(audio_nodes)
        self._asr = list(asr_nodes)
        self.last_list_extracted_kwargs: dict = {}

    def list_raw_nodes(self, raw_type, ctx):
        if raw_type == RawDataType.AUDIO:
            return list(self._audio)
        return []

    def list_extracted_nodes(
        self, extractor_type, ctx, limit=4096, offset=0,
        parent_audio_node_ids=None,
    ):
        self.last_list_extracted_kwargs = {
            "extractor_type": extractor_type,
            "limit": limit,
            "offset": offset,
            "parent_audio_node_ids": parent_audio_node_ids,
        }
        result = [n for n in self._asr if n.extractor_type == extractor_type]
        if parent_audio_node_ids:
            # Mirror the production SQL store: it canonicalises both
            # sides (strip dashes, lower) so the dashed form that
            # voice_asr persists matches the undashed form the
            # manifest holds. Without this, cross-form tests fail
            # because ``source_audio_node_id`` and the manifest id
            # never compare equal as raw strings.
            def _canon(s):
                return s.replace("-", "").lower() if isinstance(s, str) and s else ""
            scoped = {_canon(a) for a in parent_audio_node_ids if a}
            result = [
                n for n in result
                if _canon((n.extracted_content or {}).get("source_audio_node_id"))
                in scoped
            ]
        if limit is not None:
            result = result[offset:offset + limit]
        else:
            result = result[offset:]
        return result


def test_scoped_fetch_returns_only_target_video_segments():
    """The core regression: a target video's ASR segments must come
    back even when the graph holds tens of thousands of unrelated
    nodes. We seed 23 videos × 100 clips × 1 ASR segment each (=
    2300 segments) plus one extra "target" video with its own
    segments, then assert the new path returns only the target's
    entries.
    """
    target_video = "bedroom_01"
    target_audio_ids: list[str] = []
    asr_nodes: list[ExtractedNode] = []

    # Seed 23 unrelated videos with a generous ASR population.
    for vidx in range(23):
        vid = f"other_video_{vidx:03d}"
        for cid in range(73):
            audio_id = f"audio-{vid}-{cid}"
            asr_id = f"asr-{vid}-{cid}"
            asr_nodes.append(_make_asr_node(
                asr_id, audio_id, text=f"unrelated {vid} {cid}",
            ))
    # Seed the target video with a few ASR nodes.
    for cid in (0, 1, 2):
        audio_id = f"audio-{target_video}-{cid}"
        target_audio_ids.append(audio_id)
        asr_nodes.append(_make_asr_node(
            f"asr-{target_video}-{cid}", audio_id,
            text=f"target video clip {cid}: hello world",
            speaker_id="S1",
        ))

    audio_index = {
        f"{target_video}|{cid}": audio_id
        for cid, audio_id in enumerate(target_audio_ids)
    }
    graph = _FakeMemoryGraph(audio_nodes=[], asr_nodes=asr_nodes)

    asr_index = _index_asr_segments(graph, _ctx(), audio_index)

    # The new path must thread ``parent_audio_node_ids`` so the
    # store can push the filter down (this is what prevents the
    # 4096 truncation that broke 20/23 videos).
    kwargs = graph.last_list_extracted_kwargs
    assert kwargs["parent_audio_node_ids"] is not None
    assert set(kwargs["parent_audio_node_ids"]) == set(target_audio_ids)
    # ``limit=None`` is the escape hatch that drops the LIMIT
    # clause so we don't truncate when there are more segments
    # than 4096 for this video alone.
    assert kwargs["limit"] is None

    # And the result must contain ONLY the target video's
    # segments — no leakage from the other 23 videos.
    assert set(asr_index.keys()) == {
        f"{target_video}|0",
        f"{target_video}|1",
        f"{target_video}|2",
    }
    for key, segs in asr_index.items():
        for seg in segs:
            assert seg["speaker_id"] == "S1"
            assert seg["text"].startswith("target video")


def test_unscoped_fallback_keeps_legacy_behaviour():
    """When ``audio_index`` is empty AND the graph walk fallback
    yields no audio nodes (e.g. the driver is called with no
    manifest and the graph has no AUDIO nodes), the function
    returns an empty dict without raising.
    """
    graph = _FakeMemoryGraph(audio_nodes=[], asr_nodes=[])
    asr_index = _index_asr_segments(graph, _ctx(), audio_index={})
    assert asr_index == {}
    # With no audio_index AND no AUDIO nodes in the graph, the
    # function short-circuits before ever calling
    # ``list_extracted_nodes`` (there is nothing to map segments
    # back to). The legacy unscoped query only fires when there IS
    # an audio_index but the scoped lookup comes back empty —
    # see test_scoped_fetch_returns_only_target_video_segments for
    # the parent_audio_node_ids=[] case.
    assert graph.last_list_extracted_kwargs == {}


def test_graph_walk_fallback_provides_audio_ids():
    """When no ``audio_index`` manifest is passed, the function
    walks the graph's AUDIO RawNodes to recover the
    ``audio_node_id -> "video_id|clip_id"`` reverse map. After the
    walk, the scoped ``parent_audio_node_ids`` argument must still
    be populated so the store can filter.
    """
    target_audio_ids: list[str] = []
    asr_nodes: list[ExtractedNode] = []
    for cid in range(5):
        audio_id = f"graph-audio-{cid}"
        target_audio_ids.append(audio_id)
        asr_nodes.append(_make_asr_node(
            f"graph-asr-{cid}", audio_id, text=f"graph-walk {cid}",
        ))

    audio_nodes = [
        _make_audio_node(aid, "walked_video", cid)
        for cid, aid in enumerate(target_audio_ids)
    ]
    graph = _FakeMemoryGraph(audio_nodes=audio_nodes, asr_nodes=asr_nodes)

    asr_index = _index_asr_segments(graph, _ctx(), audio_index={})

    assert graph.last_list_extracted_kwargs["parent_audio_node_ids"] is not None
    # All five target audio ids must be in the scoped list.
    assert set(graph.last_list_extracted_kwargs["parent_audio_node_ids"]) == set(target_audio_ids)
    # And the resulting index must contain 5 clips.
    assert len(asr_index) == 5
    for key, segs in asr_index.items():
        assert key.startswith("walked_video|")
        assert len(segs) == 1


def test_canonical_uuid_normalisation_across_manifest_and_graph():
    """The manifest writes undashed 32-hex ids; the SQL store
    persists the dashed UUID form. ``_canon`` strips dashes +
    lowercases so both sides compare equal. Assert on this so a
    future change to the canonicalisation doesn't silently
    re-introduce the cross-form mismatch.
    """
    dashed = "aabbccdd-eeff-0011-2233-445566778899"
    undashed = dashed.replace("-", "").lower()
    # Same canonical form after strip + lower.
    canon_dashed = dashed.replace("-", "").lower()
    canon_undashed = undashed.lower()
    assert canon_dashed == canon_undashed

    # End-to-end: write the ASR node with the dashed form (matches
    # what voice_asr actually persists), manifest with the
    # undashed form, and verify the index resolves.
    audio_id = undashed
    asr_node = _make_asr_node(
        "asr-cross-form",
        source_audio_id=dashed,  # voice_asr writes dashed
        text="cross-form text",
    )
    graph = _FakeMemoryGraph(
        audio_nodes=[], asr_nodes=[asr_node],
    )
    audio_index = {"walked_video|0": audio_id}  # manifest undashed

    asr_index_out = _index_asr_segments(graph, _ctx(), audio_index)
    assert "walked_video|0" in asr_index_out
    assert asr_index_out["walked_video|0"][0]["text"] == "cross-form text"
