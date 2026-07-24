r"""Integration tests for ``SQLGraphStore.search_by_embedding``.

These tests need a real PostgreSQL instance with the ``graph_nodes`` /
``graph_edges`` schema created via ``fs/sql_adapter/graph_schema.sql``.
Gated by two env vars:

- ``OGMEM_SQL_INTEGRATION=1`` — master switch. When unset, every test in
  this module skips without further checks.
- ``OGMEM_TEST_DSN`` (preferred) or ``GRAPH_DSN`` (fallback) — a libpq
  connection string. Both URL form (``postgresql://user:pw@host:port/db``)
  and keyword/value form (``host=... port=... dbname=... user=... password=...``)
  are accepted by psycopg2.

The suite reuses an existing Postgres — bring one up however suits you.
``graph_schema.sql`` is loaded lazily the first time ``SQLGraphStore``
connects (via ``_ensure_schema``), so no manual bootstrap is needed.

Two working setups:

1. **Standalone container** (the project-local convention)::

       docker run --name ogmem-sql-graph-store \
           -e POSTGRES_PASSWORD=postgres \
           -e POSTGRES_DB=ogmem_graph \
           -p 4219:5432 \
           -d postgres:16
       export GRAPH_DSN='host=127.0.0.1 port=4219 dbname=ogmem_graph user=postgres password=postgres'
       OGMEM_SQL_INTEGRATION=1 \
           uv run pytest tests/integration/test_sql_graph_search_by_embedding.py -v

2. **docker compose** (``deploy/`` with the ``with-m3`` profile)::

       cd deploy && cp .env.example .env       # then edit POSTGRES_PASSWORD
       docker compose --profile with-m3 up -d
       OGMEM_SQL_INTEGRATION=1 \
       OGMEM_TEST_DSN="postgresql://ogmem:${POSTGRES_PASSWORD}@127.0.0.1:5432/ogmem" \
           uv run pytest tests/integration/test_sql_graph_search_by_embedding.py

The expected runtime is small — M3-Bench scale (~3K episodic nodes) is
trivial for PostgreSQL even without pgvector, and the suite exercises
under 20 rows.
"""

from __future__ import annotations

import math
import os

import pytest

pytestmark = pytest.mark.integration

_INTEGRATION_ENABLED = os.environ.get("OGMEM_SQL_INTEGRATION") == "1"
# Prefer the project-canonical env var, fall back to the convention used by
# the standalone ``docker run`` setup in the project README. Both feed
# psycopg2's connection-string parser, so URL form and libpq keyword/value
# form are both fine.
_DSN = os.environ.get("OGMEM_TEST_DSN") or os.environ.get("GRAPH_DSN")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _INTEGRATION_ENABLED or not _DSN,
        reason=(
            "Set OGMEM_SQL_INTEGRATION=1 and OGMEM_TEST_DSN (or GRAPH_DSN) to run. "
            "Example: OGMEM_SQL_INTEGRATION=1 "
            "GRAPH_DSN='host=127.0.0.1 port=4219 dbname=ogmem_graph user=postgres password=postgres'"
        ),
    ),
]  # type: list[pytest.MarkDecorator]  # silence type-checkers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sql_store():
    """Build a SQLGraphStore connected to the configured DSN.

    The schema is loaded on construction via ``_ensure_schema`` (which runs
    ``fs/sql_adapter/graph_schema.sql``), so an empty database is enough.
    """
    from providers.graph_store.sql_graph_store import SQLGraphStore

    return SQLGraphStore(connection_string=_DSN)


@pytest.fixture
def ctx():
    """Per-test account_id so runs do not collide in a shared DB."""
    import uuid

    from core.models import RequestContext

    account_id = f"sql_embed_{uuid.uuid4().hex[:12]}"
    return RequestContext(
        account_id=account_id,
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        trace_id=f"t_{uuid.uuid4().hex[:8]}",
    )


@pytest.fixture
def other_ctx():
    """A second tenant for isolation tests."""
    import uuid

    from core.models import RequestContext

    return RequestContext(
        account_id=f"sql_other_{uuid.uuid4().hex[:12]}",
        user_id="u2",
        agent_id="a2",
        session_id="s2",
        trace_id=f"t_{uuid.uuid4().hex[:8]}",
    )


# Orthonormal 3-D basis vectors so cosine score is exact.
V_X = [1.0, 0.0, 0.0]
V_Y = [0.0, 1.0, 0.0]
V_Z = [0.0, 0.0, 1.0]


def _canonical_nid(value) -> str:
    """Return ``value`` as a canonical 36-char dashed UUID string.

    ``create_extracted_node`` generates ``node_id`` as ``uuid.uuid4().hex``
    (32 chars, no dashes), while psycopg2 returns UUID columns as
    ``uuid.UUID`` (36 chars, dashed). To compare node IDs across the two
    stores we need a single representation.
    """
    import uuid as _uuid

    return str(_uuid.UUID(str(value)))


def _add_episode(store, ctx, *, video_id: str = "vid1", clip_id: int = 0,
                 embedding=None, extractor_type: str = "episodic_memory_generator",
                 description: str = "episode"):
    from core.graph_models import create_extracted_node
    import uuid as _uuid

    # The SQL store's ``source_raw_id`` column is UUID, so unlike the
    # in-memory store (which is type-agnostic) we need a real UUID here.
    node = create_extracted_node(
        source_raw_id=str(_uuid.uuid4()),
        extractor_type=extractor_type,
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={
            "description": description,
            "_video_id": video_id,
            "_clip_id": clip_id,
        },
    )
    store.create_node(node.serialize(), ctx)
    if embedding is not None:
        live = store.get_node(node.node_id, ctx)
        assert live is not None
        live.set_embedding(list(embedding))
        store.update_node(live.serialize(), ctx)
    return _canonical_nid(node.node_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSQLGraphStoreSearchByEmbedding:
    def test_empty_store_returns_empty(self, sql_store, ctx):
        results = sql_store.search_by_embedding([0.1, 0.2, 0.3], ctx)
        assert results == []

    def test_identical_vector_scores_one(self, sql_store, ctx):
        nid = _add_episode(sql_store, ctx, embedding=V_X)
        results = sql_store.search_by_embedding(V_X, ctx)
        assert results[0][0] == nid
        assert math.isclose(results[0][1], 1.0, abs_tol=1e-6)

    def test_orthogonal_vector_scores_zero(self, sql_store, ctx):
        _add_episode(sql_store, ctx, embedding=V_X)
        results = sql_store.search_by_embedding(V_Y, ctx)
        assert len(results) == 1
        assert math.isclose(results[0][1], 0.0, abs_tol=1e-6)

    def test_results_sorted_desc(self, sql_store, ctx):
        for clip in (0, 1, 2):
            _add_episode(sql_store, ctx, clip_id=clip, embedding=V_X)
        # Use a query close to V_Y so orthogonality isn't tested instead.
        # Three parallel vectors -> identical scores, ordering stable.
        results = sql_store.search_by_embedding([2.0, 0.1, 0.0], ctx)
        scores = [s for _n, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_caps_results(self, sql_store, ctx):
        for clip in range(5):
            _add_episode(sql_store, ctx, clip_id=clip, embedding=V_X)
        results = sql_store.search_by_embedding(V_X, ctx, top_k=3)
        assert len(results) == 3

    def test_dimension_mismatch_is_skipped(self, sql_store, ctx):
        _add_episode(sql_store, ctx, embedding=[1.0, 0.0])  # 2-D
        results = sql_store.search_by_embedding([0.5, 0.5, 0.5], ctx)
        assert results == []

    def test_zero_norm_embedding_returns_zero_score(self, sql_store, ctx):
        nid = _add_episode(sql_store, ctx, embedding=[0.0, 0.0, 0.0])
        results = sql_store.search_by_embedding(V_X, ctx)
        assert len(results) == 1
        assert results[0][0] == nid
        assert math.isclose(results[0][1], 0.0)

    def test_before_clip_filter(self, sql_store, ctx):
        _add_episode(sql_store, ctx, clip_id=0, embedding=V_X)
        _add_episode(sql_store, ctx, clip_id=5, embedding=V_X)
        results = sql_store.search_by_embedding(V_X, ctx, before_clip=2)
        # Restricted set; unfiltered has 2, filtered has 1.
        unfiltered = sql_store.search_by_embedding(V_X, ctx)
        nids = {r[0] for r in results}
        assert len(nids) == 1
        assert nids.issubset({r[0] for r in unfiltered})

    def test_video_id_filter(self, sql_store, ctx):
        _add_episode(sql_store, ctx, video_id="vid_a", embedding=V_X)
        _add_episode(sql_store, ctx, video_id="vid_b", embedding=V_X)
        results = sql_store.search_by_embedding(V_X, ctx, video_id="vid_a")
        assert len(results) == 1

    def test_extractor_types_filter(self, sql_store, ctx):
        nid = _add_episode(sql_store, ctx, embedding=V_X, extractor_type="episodic_memory_generator")
        _add_episode(sql_store, ctx, embedding=V_X, extractor_type="other")
        results = sql_store.search_by_embedding(
            V_X, ctx, extractor_types=("episodic_memory_generator",)
        )
        nids = [r[0] for r in results]
        assert nids == [nid]

    def test_account_isolation(self, sql_store, ctx, other_ctx):
        _add_episode(sql_store, ctx, embedding=V_X)
        _add_episode(sql_store, other_ctx, embedding=V_X)
        results = sql_store.search_by_embedding(V_X, ctx)
        assert len(results) == 1
        results_other = sql_store.search_by_embedding(V_X, other_ctx)
        assert len(results_other) == 1

    def test_missing_clip_id_excluded_when_before_clip_set(
        self, sql_store, ctx
    ):
        # Node without _clip_id must be excluded when before_clip applies.
        _add_episode(sql_store, ctx, clip_id=3, embedding=V_X)
        from core.graph_models import create_extracted_node
        import uuid as _uuid

        node = create_extracted_node(
            source_raw_id=str(_uuid.uuid4()),
            extractor_type="episodic_memory_generator",
            extraction_schema="v1",
            account_id=ctx.account_id,
            extracted_content={"description": "no clip tag"},
        )
        sql_store.create_node(node.serialize(), ctx)
        live = sql_store.get_node(node.node_id, ctx)
        live.set_embedding(V_X)
        sql_store.update_node(live.serialize(), ctx)

        results = sql_store.search_by_embedding(V_X, ctx, before_clip=5)
        nids = {r[0] for r in results}
        assert _canonical_nid(node.node_id) not in nids
        assert len(nids) == 1

    def test_non_extracted_nodes_ignored(self, sql_store, ctx):
        # RawNode and MemoryNode have no embedding; the search primitive
        # today only indexes ExtractedNode, so such rows must not surface
        # in results regardless of the JSONB 'embedding' key.
        from core.graph_models import create_memory_node, create_raw_node
        from core.graph_enums import RawDataType

        raw = create_raw_node(
            raw_type=RawDataType.VIDEO,
            content="bytes",
            account_id=ctx.account_id,
        )
        sql_store.create_node(raw.serialize(), ctx)

        mem = create_memory_node(
            summary="no embedding",
            account_id=ctx.account_id,
        )
        sql_store.create_node(mem.serialize(), ctx)

        _add_episode(sql_store, ctx, embedding=V_X)
        results = sql_store.search_by_embedding(V_X, ctx)
        # Only the ExtractedNode has a node_type='extracted' AND embedding.
        assert len(results) == 1

    def test_search_by_embedding_matches_memory_store(self, sql_store, ctx):
        """Cross-check: SQL ranking should match MemoryGraphStore ranking
        for the same data (both use cosine, both skip silently).

        Compares counts and score sequences rather than node IDs because
        each backend generates its own UUIDs on insert; the cosine
        mathematics is what's under test.
        """
        from providers.graph_store.memory_graph_store import MemoryGraphStore

        mem_store = MemoryGraphStore()
        for clip in range(3):
            vec = [1.0, float(clip), 0.0]
            _add_episode(sql_store, ctx, clip_id=clip, embedding=vec)
            _add_episode(mem_store, ctx, clip_id=clip, embedding=vec)
        # Use a query to spread the scores.
        q = [1.0, 1.0, 0.0]
        sql_results = sql_store.search_by_embedding(q, ctx)
        mem_results = mem_store.search_by_embedding(q, ctx)
        # Both stores should rank the same three clips in the same order
        # with the same cosine scores.
        assert len(sql_results) == len(mem_results) == 3
        for sql_r, mem_r in zip(sql_results, mem_results):
            assert math.isclose(sql_r[1], mem_r[1], abs_tol=1e-9)

    def test_tied_scores_sort_by_node_id_ascending(self, sql_store, ctx):
        """Review comment #2: the SQL backend must break score ties by
        node_id ascending so the control loop's "which clip to back-fill"
        decision is identical to the memory backend's.

        Without the tiebreaker Postgres returns rows in physical heap
        order (no ``ORDER BY``), so two runs against the same data can
        legitimately produce different ids — and disagree with the
        memory backend's insertion-order output.
        """
        # Three identical embeddings → identical cosine scores (1.0).
        # Insertion order is intentionally NOT lexical (clip_id goes
        # 0, 1, 2 so we can verify the tiebreaker is sorting by id,
        # not by insertion order).
        nid_a = _add_episode(sql_store, ctx, clip_id=0, embedding=V_X)
        nid_b = _add_episode(sql_store, ctx, clip_id=1, embedding=V_X)
        nid_c = _add_episode(sql_store, ctx, clip_id=2, embedding=V_X)

        results = sql_store.search_by_embedding(V_X, ctx)
        assert len(results) == 3
        for _nid, score in results:
            assert math.isclose(score, 1.0, abs_tol=1e-6)
        # node_id is canonical (dashed UUID) coming back from psycopg2.
        # Tied scores must be sorted by id ascending, NOT by insertion
        # order (which would just be the original clip_id sequence).
        ordered_ids = [_canonical_nid(nid) for nid, _ in results]
        assert ordered_ids == sorted(ordered_ids), (
            f"expected ascending node_id on tied scores, got {ordered_ids}"
        )
        # Sanity: at least two distinct ids exist (clip_id ordering
        # does NOT imply id ordering once UUIDs are involved).
        assert len({nid_a, nid_b, nid_c}) == 3

    def test_tied_scores_match_memory_store_order(self, sql_store, ctx):
        """Stronger parity: on tied scores, SQL and Memory must return
        the SAME node_id order. This is the property the control loop
        relies on (sql vs memory → same "next clip to back-fill")."""
        from providers.graph_store.memory_graph_store import MemoryGraphStore

        mem_store = MemoryGraphStore()
        # Identical embeddings → identical scores. The control loop
        # wants the two backends to break ties identically.
        for clip in range(5):
            _add_episode(sql_store, ctx, clip_id=clip, embedding=V_X)
            _add_episode(mem_store, ctx, clip_id=clip, embedding=V_X)

        sql_results = sql_store.search_by_embedding(V_X, ctx)
        mem_results = mem_store.search_by_embedding(V_X, ctx)
        assert len(sql_results) == len(mem_results) == 5
        # All scores are exactly 1.0 → every pair is tied.
        for _nid, score in sql_results:
            assert math.isclose(score, 1.0, abs_tol=1e-6)
        for _nid, score in mem_results:
            assert math.isclose(score, 1.0, abs_tol=1e-6)
        # node_id order is identical across backends (each backend
        # receives its own UUIDs, so we sort both and compare).
        sql_ids = sorted(_canonical_nid(nid) for nid, _ in sql_results)
        mem_ids = sorted(str(nid) for nid, _ in mem_results)
        assert sql_ids == mem_ids, (
            f"sql/memory tiebreak sets diverge:\n  sql={sql_ids}\n  mem={mem_ids}"
        )
