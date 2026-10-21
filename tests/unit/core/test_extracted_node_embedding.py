"""Unit tests for ExtractedNode.embedding accessors.

The embedding lives inside ``extracted_content["embedding"]`` so it round-trips
through the existing serialize / deserialize path used by every graph store
backend. These tests pin both the in-memory behaviour and the payload shape.
"""

from __future__ import annotations

import pytest

from core.graph_enums import ExtractionStatus, GraphNodeType
from core.graph_models import ExtractedNode, create_extracted_node, node_from_payload
from core.models import RequestContext


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acc1",
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        trace_id="t1",
    )


def _make_node(**overrides) -> ExtractedNode:
    """Build an ExtractedNode via the canonical factory.

    Defaults match a typical M3 episodic-memory-generator output so the tests
    can assert behaviour against a realistic node. ``account_id`` is omitted
    from defaults because it is a positional arg on ``create_extracted_node``.
    """
    defaults: dict = dict(
        source_raw_id="raw-1",
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        extracted_content={"description": "hello"},
    )
    defaults.update(overrides)
    account_id = defaults.pop("account_id", "acc1")
    return create_extracted_node(account_id=account_id, **defaults)


class TestExtractedNodeEmbedding:
    def test_default_embedding_is_none(self):
        node = _make_node()
        assert node.get_embedding() is None

    def test_set_embedding_persists_and_returns_same_value(self):
        node = _make_node()
        vec = [0.1, 0.2, 0.3]
        node.set_embedding(vec)
        assert node.get_embedding() == vec

    def test_set_embedding_replaces_previous_value(self):
        node = _make_node()
        node.set_embedding([1.0, 1.0])
        node.set_embedding([2.0, 2.0])
        assert node.get_embedding() == [2.0, 2.0]

    def test_set_embedding_copies_input_list(self):
        # Defensive copy: mutating the caller's list must not change the
        # stored embedding.
        node = _make_node()
        original = [0.5, 0.6]
        node.set_embedding(original)
        original.append(0.7)
        assert node.get_embedding() == [0.5, 0.6]

    def test_set_embedding_touches_updated_at(self):
        node = _make_node()
        before = node.updated_at
        import time as _time

        _time.sleep(0.001)
        node.set_embedding([0.0])
        assert node.updated_at != before

    def test_get_embedding_returns_none_when_missing_key(self):
        # Slot present but empty / weird type — be defensive so a stray write
        # to the same key by some other code path doesn't crash search.
        node = _make_node()
        node.extracted_content["embedding"] = None
        assert node.get_embedding() is None

    def test_get_embedding_returns_none_for_non_list_value(self):
        node = _make_node()
        node.extracted_content["embedding"] = "not-a-list"
        assert node.get_embedding() is None

    def test_get_embedding_returns_list_for_valid_value(self):
        node = _make_node()
        node.extracted_content["embedding"] = [0.4, 0.5]
        assert node.get_embedding() == [0.4, 0.5]

    def test_serialize_round_trips_embedding(self):
        # Storage path: serialize() -> node_from_payload must preserve the
        # embedding because we live inside extracted_content (the existing
        # JSON-safe column).
        node = _make_node()
        node.set_embedding([0.1, 0.2, 0.3, 0.4])
        payload = node.serialize()
        assert payload["extracted_content"]["embedding"] == [0.1, 0.2, 0.3, 0.4]
        # Payload also keeps extractor_type / source_raw_id etc. intact.
        assert payload["extractor_type"] == "episodic_memory_generator"
        assert payload["source_raw_id"] == "raw-1"

    def test_deserialize_via_node_from_payload(self):
        payload = {
            "node_id": "n-1",
            "node_type": "extracted",
            "status": "active",
            "pagerank": 0.0,
            "created_at": "2026-07-16T00:00:00+00:00",
            "updated_at": "2026-07-16T00:00:00+00:00",
            "account_id": "acc1",
            "source_raw_id": "raw-1",
            "extractor_type": "episodic_memory_generator",
            "extraction_schema": "m3_bench_v1",
            "extracted_content": {
                "description": "clip summary",
                "embedding": [0.9, 0.8, 0.7],
            },
            "extraction_status": "completed",
            "confidence": 0.42,
        }
        restored = node_from_payload(payload)
        assert isinstance(restored, ExtractedNode)
        assert restored.node_type == GraphNodeType.EXTRACTED
        assert restored.get_embedding() == [0.9, 0.8, 0.7]
        assert restored.extraction_status == ExtractionStatus.COMPLETED
        assert restored.confidence == 0.42

    def test_mark_completed_overwrites_embedding_via_extracted_content(self):
        # mark_completed replaces extracted_content entirely — embedding
        # accessors must reflect whatever dict is currently set.
        node = _make_node()
        node.set_embedding([0.1])
        node.mark_completed({"description": "new", "embedding": [0.99]}, 0.5)
        assert node.get_embedding() == [0.99]
        assert node.extraction_status == ExtractionStatus.COMPLETED
        assert node.confidence == 0.5

    def test_mark_failed_keeps_existing_embedding(self):
        node = _make_node()
        node.set_embedding([0.42])
        node.mark_failed("llm error")
        # mark_failed writes to extracted_content["error"]; the embedding
        # key is preserved because the dict is mutated in place.
        assert node.get_embedding() == [0.42]
        assert node.extracted_content.get("error") == "llm error"
