"""Unit tests for the M3 scope-filter extensions on OpenGaussVectorIndex.

The OpenGaussVectorIndex._build_where method now recognises three M3
scope keys in addition to the original tenant / level filters:

- ``video_id`` — exact-match equality.
- ``extractor_type`` — scalar or list (uses ``= ANY(...)``).
- ``before_clip`` — ``<= N`` semantics with unknown clips excluded
  so legacy records (no column value) remain findable until
  re-indexed.

These tests exercise ``_build_where`` directly so we don't need a
live Postgres. The companion integration test
``tests/integration/m3/test_vector_index_path.py`` covers the same
behaviour end-to-end against a real DB.
"""

from __future__ import annotations

from providers.vector_index.opengauss_index import OpenGaussVectorIndex


def test_build_where_video_id_scalar():
    where, params = OpenGaussVectorIndex._build_where({"video_id": "vid_42"})
    assert "video_id = %(f_video_id)s" in where
    assert params["f_video_id"] == "vid_42"


def test_build_where_extractor_type_scalar():
    where, params = OpenGaussVectorIndex._build_where(
        {"extractor_type": "episodic_memory_generator"}
    )
    assert "extractor_type = %(f_extractor_type)s" in where
    assert params["f_extractor_type"] == "episodic_memory_generator"


def test_build_where_extractor_type_list():
    where, params = OpenGaussVectorIndex._build_where(
        {"extractor_type": ["episodic_memory_generator", "face_clustering"]}
    )
    assert "extractor_type = ANY(%(f_extractor_type)s)" in where
    assert params["f_extractor_type"] == [
        "episodic_memory_generator",
        "face_clustering",
    ]


def test_build_where_before_clip_excludes_unknown():
    """Unknown clip provenance cannot satisfy a bounded search."""
    where, params = OpenGaussVectorIndex._build_where({"before_clip": 5})
    assert "before_clip IS NULL" not in where
    assert "before_clip <= %(f_before_clip)s" in where
    assert params["f_before_clip"] == 5


def test_build_where_combined_m3_scope():
    """All three M3 keys together with account_id, mirroring the
    GraphService.search_by_embedding path. The clause must include
    each piece so a regression that drops one silently opens up
    cross-clip or cross-video bleed."""
    where, params = OpenGaussVectorIndex._build_where({
        "account_id": "m3bench",
        "video_id": "living_room_06",
        "extractor_type": "episodic_memory_generator",
        "before_clip": 3,
    })
    for expected in (
        "COALESCE(account_id, filters->>'account_id') = %(f_account_id)s",
        "video_id = %(f_video_id)s",
        "extractor_type = %(f_extractor_type)s",
        "before_clip <= %(f_before_clip)s",
    ):
        assert expected in where, f"missing clause fragment: {expected}"
    assert params["f_account_id"] == "m3bench"
    assert params["f_video_id"] == "living_room_06"
    assert params["f_extractor_type"] == "episodic_memory_generator"
    assert params["f_before_clip"] == 3


def test_build_where_empty_filters_returns_true():
    """Edge case: an empty filter dict must still produce a valid
    WHERE clause (we use ``TRUE`` so the caller doesn't have to
    special-case "no filters"). A regression to ``AND `` (empty
    join) would emit a SQL syntax error on the first SELECT."""
    where, params = OpenGaussVectorIndex._build_where({})
    assert where == "TRUE"
    assert params == {}


def test_legacy_ownership_fallback_preserves_explicit_grants():
    where, params = OpenGaussVectorIndex._build_where({
        "account_id": "acme",
        "owner_space": ["user:alice", "agent:main"],
        "space_id": ["private-space"],
        "video_id": "video-1",
        "before_clip": 3,
    })
    assert "COALESCE(account_id, filters->>'account_id') = %(f_account_id)s" in where
    assert "COALESCE(owner_space, filters->>'owner_space') = ANY(%(f_owner_space)s)" in where
    assert "space_id = ANY(%(f_space_id)s)" in where
    assert "space_id IS NULL" not in where
    assert "video_id = %(f_video_id)s" in where
    assert "before_clip <= %(f_before_clip)s" in where
    assert params["f_owner_space"] == ["user:alice", "agent:main"]


def test_build_where_ignores_unknown_keys():
    """Unknown filter keys fall through silently — the legacy
    behaviour we kept for back-compat. Documenting it as a test so
    a future refactor that switches to ``return False on unknown``
    (i.e. fail-closed) trips this and forces an explicit decision."""
    where, params = OpenGaussVectorIndex._build_where(
        {"video_id": "vid_1", "totally_unknown_key": "value"}
    )
    assert "video_id = %(f_video_id)s" in where
    assert "totally_unknown_key" not in where
    assert "totally_unknown_key" not in params
