"""Unit tests for ``m3.index_episodic._upsert_to_vector_index``.

The helper is the boundary between the M3 JSONL producer
(``build_index``) and the canonical retrieval store
(``OpenGaussVectorIndex``). We mock the VectorIndex so the test
exercises only the IndexRecord-shape contract — the OpenGauss SQL
behaviour is covered separately by
``tests/unit/providers/vector_index/test_opengauss_filter_extensions.py``
and ``tests/integration/m3/test_vector_index_path.py``.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock

from core.models import IndexRecord
from m3.index_episodic import _upsert_to_vector_index


def _args(account_id: str = "m3bench") -> argparse.Namespace:
    return argparse.Namespace(account_id=account_id)


def _rec(video_id: str, clip_id: int, ext_id: str, embedding: list[float]):
    """Build the in-memory record shape ``build_index`` produces."""
    return {
        "id": f"rec-{video_id}-{clip_id}",
        "uri": f"m3://{video_id}/clips/{clip_id}/episodic",
        "video_id": video_id,
        "clip_id": clip_id,
        "text": f"description for {video_id} {clip_id}",
        "embedding": embedding,
        "source": "episodic_memory_generator",
        "source_extracted_node_id": ext_id,
        "video_raw_node_id": f"raw-{video_id}-{clip_id}",
    }


def test_upsert_builds_correct_index_record_shape():
    """One record → one IndexRecord with the expected fields."""
    vi = MagicMock()
    records = [_rec("vid_a", 0, "ext-0", [0.1, 0.2, 0.3])]
    written = _upsert_to_vector_index(records, vi, _args())
    assert written == 1

    vi.upsert.assert_called_once()
    payload = vi.upsert.call_args.args[0]
    assert len(payload) == 1
    ir = payload[0]
    assert isinstance(ir, IndexRecord)
    # ID is sha256(uri+level)[:16]; we generated the URI from the
    # record fields so we can re-derive the expected id and assert
    # exact equality.
    expected_uri = "m3://vid_a/clips/0/episodic"
    assert ir.id == IndexRecord.generate_id(expected_uri, level=2)
    assert ir.uri == expected_uri
    assert ir.level == 2
    assert ir.text == "description for vid_a 0"
    # Filters carry the tenant + M3 scope columns. The VectorIndex
    # _build_where clause reads these directly.
    assert ir.filters == {
        "account_id": "m3bench",
        "video_id": "vid_a",
        "extractor_type": "episodic_memory_generator",
    }
    # Metadata carries the forward-channel back to the graph node +
    # the scope columns the WHERE clause filters on, plus the
    # embedding itself (so the upsert has it without an extra join).
    assert ir.metadata["extracted_node_id"] == "ext-0"
    assert ir.metadata["video_id"] == "vid_a"
    assert ir.metadata["clip_id"] == 0
    assert ir.metadata["before_clip"] == 0
    assert ir.metadata["source"] == "episodic_memory_generator"
    assert ir.metadata["_embedding"] == [0.1, 0.2, 0.3]


def test_upsert_skips_records_missing_extracted_node_id():
    """Records without ``source_extracted_node_id`` can't reverse-
    resolve to a graph node — drop them silently. The IndexRecord
    payload the index sees only contains the well-formed records."""
    vi = MagicMock()
    records = [
        _rec("vid_a", 0, "ext-0", [0.1]),
        # missing extracted_node_id
        {**_rec("vid_a", 1, "", [0.2]), "source_extracted_node_id": ""},
        # missing embedding
        {**_rec("vid_a", 2, "ext-2", []), "embedding": []},
    ]
    written = _upsert_to_vector_index(records, vi, _args())
    assert written == 1
    vi.upsert.assert_called_once()
    payload = vi.upsert.call_args.args[0]
    assert len(payload) == 1
    assert payload[0].metadata["extracted_node_id"] == "ext-0"


def test_upsert_with_none_vector_index_returns_zero():
    """``vector_index is None`` is the legacy-CLI escape hatch — the
    function returns 0 without raising so the JSONL write still
    happens."""
    records = [_rec("vid_a", 0, "ext-0", [0.1])]
    written = _upsert_to_vector_index(records, None, _args())
    assert written == 0


def test_upsert_coerces_clip_id_to_int_for_before_clip():
    """The OpenGauss ``before_clip`` column is INTEGER. clip_id may
    arrive as a string from upstream JSONL producers; coerce so
    the MERGE statement doesn't reject it."""
    vi = MagicMock()
    records = [_rec("vid_a", "7", "ext-7", [0.1])]
    _upsert_to_vector_index(records, vi, _args())
    payload = vi.upsert.call_args.args[0]
    assert payload[0].metadata["before_clip"] == 7
    assert payload[0].metadata["clip_id"] == "7"


def test_upsert_passes_account_id_through():
    """The per-run account_id is the tenant isolation boundary; a
    typo here would silently leak cross-tenant embeddings."""
    vi = MagicMock()
    records = [_rec("vid_a", 0, "ext-0", [0.1])]
    _upsert_to_vector_index(records, vi, _args(account_id="alt-tenant"))
    payload = vi.upsert.call_args.args[0]
    assert payload[0].filters["account_id"] == "alt-tenant"


def test_upsert_handles_empty_input():
    vi = MagicMock()
    written = _upsert_to_vector_index([], vi, _args())
    assert written == 0
    vi.upsert.assert_not_called()
