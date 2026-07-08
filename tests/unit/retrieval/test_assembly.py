"""Unit tests for retrieval.result_ranker.ResultRanker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.models import (
    LeafHit,
    RetrievalConfig,
    RetrievedBlock,
    SeedResult,
    TypedQuery,
    SeedHit,
)
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.result_ranker import ResultRanker


@pytest.fixture()
def svc():
    return ResultRanker(RetrievalConfig())


@pytest.fixture()
def tq():
    return TypedQuery(text="test", context_type="MEMORY", categories=[], top_k=3)


def _leaf(
    uri: str,
    score: float,
    level: int = 2,
    metadata: dict | None = None,
    category: str = "",
) -> LeafHit:
    return LeafHit(
        uri=uri,
        score=score,
        level=level,
        category=category,
        abstract=f"abs-{uri}",
        metadata=metadata or {},
    )


class TestResultRanker:
    def test_filters_to_l2_only(self, svc, tq):
        leaves = [
            _leaf("u1", 0.9, level=2),
            _leaf("u2", 0.8, level=0),
            _leaf("u3", 0.7, level=2),
        ]
        blocks = svc.assemble(tq, leaves, None)
        assert all(b.level_hit == "L2" for b in blocks)
        assert len(blocks) == 2

    def test_sorts_descending(self, svc, tq):
        leaves = [_leaf("a", 0.5), _leaf("b", 0.9), _leaf("c", 0.7)]
        blocks = svc.assemble(tq, leaves, None)
        scores = [b.score for b in blocks]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self, svc):
        small_tq = TypedQuery(text="x", context_type="", categories=[], top_k=1)
        leaves = [_leaf("a", 0.5), _leaf("b", 0.9)]
        blocks = svc.assemble(small_tq, leaves, None)
        assert len(blocks) == 1
        assert blocks[0].uri == "b"

    def test_fallback_from_seed(self, svc, tq):
        seed = SeedResult(
            initial_candidates=[
                SeedHit(
                    uri="x",
                    score=0.6,
                    level=2,
                    abstract="from-seed",
                    metadata={"when": "2023-05-07", "who": ["Caroline"], "where": "Boston"},
                ),
            ],
        )
        blocks = svc.assemble(tq, [], seed)
        assert len(blocks) == 1
        assert blocks[0].uri == "x"
        assert blocks[0].metadata == {
            "when": "2023-05-07",
            "who": ["Caroline"],
            "where": "Boston",
        }

    def test_temporal_query_reranks_candidates_with_date_metadata(self, svc):
        temporal_tq = TypedQuery(
            text="When did Caroline join the LGBTQ support group on 2023-05-07?",
            context_type="MEMORY",
            categories=[],
            top_k=1,
        )
        leaves = [
            _leaf("wrong-date", 0.9, metadata={"when": "2023-05-08"}),
            _leaf("right-date", 0.86, metadata={"when": "2023-05-07"}),
        ]

        blocks = svc.assemble(temporal_tq, leaves, None)

        assert [b.uri for b in blocks] == ["right-date"]

    def test_non_temporal_query_preserves_score_order(self, svc):
        normal_tq = TypedQuery(
            text="Caroline support group notes",
            context_type="MEMORY",
            categories=[],
            top_k=1,
        )
        leaves = [
            _leaf("higher-score", 0.9, metadata={"when": "2023-05-08"}),
            _leaf("lower-score", 0.86, metadata={"when": "2023-05-07"}),
        ]

        blocks = svc.assemble(normal_tq, leaves, None)

        assert [b.uri for b in blocks] == ["higher-score"]

    def test_default_ranker_ignores_metadata_for_non_temporal_query(self):
        ranker = ResultRanker(RetrievalConfig(result_ranker_strategy="semantic"))
        normal_tq = TypedQuery(
            text="Caroline support group notes",
            context_type="MEMORY",
            categories=[],
            top_k=1,
        )
        leaves = [
            _leaf("higher-score", 0.9),
            _leaf(
                "metadata-match",
                0.86,
                metadata={
                    "when": "2023-05-07",
                    "who": ["Caroline"],
                    "where": "Boston",
                },
            ),
        ]

        blocks = ranker.assemble(normal_tq, leaves, None)

        assert [b.uri for b in blocks] == ["higher-score"]

    def test_metadata_aware_ranker_reorders_close_candidates(self):
        ranker = ResultRanker(RetrievalConfig(result_ranker_strategy="metadata_aware"))
        query = TypedQuery(
            text="What happened with Caroline in Boston on 2023-05-07?",
            context_type="MEMORY",
            categories=[],
            top_k=1,
        )
        leaves = [
            _leaf("semantic-only", 0.9),
            _leaf(
                "metadata-match",
                0.86,
                category="event",
                metadata={
                    "when": "2023-05-07",
                    "who": ["Caroline"],
                    "where": "Boston",
                },
            ),
        ]

        blocks = ranker.assemble(query, leaves, None)

        assert [b.uri for b in blocks] == ["metadata-match"]
        assert blocks[0].score == 0.86
        assert "metadata-aware" in blocks[0].match_reason

    def test_metadata_aware_ranker_matches_list_items_individually(self):
        ranker = ResultRanker(RetrievalConfig(result_ranker_strategy="metadata_aware"))
        query = TypedQuery(
            text="What happened with Caroline?",
            context_type="MEMORY",
            categories=[],
            top_k=1,
        )
        leaves = [
            _leaf("semantic-only", 0.88),
            _leaf("list-match", 0.86, metadata={"who": ["Caroline", "Sam"]}),
        ]

        blocks = ranker.assemble(query, leaves, None)

        assert [b.uri for b in blocks] == ["list-match"]

    def test_metadata_aware_ranker_does_not_match_word_fragments(self):
        ranker = ResultRanker(RetrievalConfig(result_ranker_strategy="metadata_aware"))
        query = TypedQuery(
            text="Who came to the party?",
            context_type="MEMORY",
            categories=[],
            top_k=1,
        )
        leaves = [
            _leaf("higher-score", 0.88),
            _leaf("fragment-only", 0.86, metadata={"where": "art"}),
        ]

        blocks = ranker.assemble(query, leaves, None)

        assert [b.uri for b in blocks] == ["higher-score"]

    def test_preserves_content_flags_from_leaf_hits(self, svc, tq):
        leaves = [
            LeafHit(
                uri="u1",
                score=0.9,
                level=2,
                abstract="abs-u1",
                has_content=True,
                has_overview=True,
            )
        ]

        blocks = svc.assemble(tq, leaves, None)

        assert len(blocks) == 1
        assert blocks[0].has_content is True
        assert blocks[0].has_overview is True

    def test_hierarchical_conversion_preserves_temporal_metadata(self):
        seed_hit = SeedHit(
            uri="ctx://acct/users/alice/memories/events/support_group",
            score=0.9,
            level=2,
            category="event",
            abstract="Caroline joined the LGBTQ support group.",
            metadata={
                "when": "2023-05-07",
                "who": ["Caroline"],
                "where": "Boston",
            },
        )
        searcher = HierarchicalSearcher(vector_index=object(), config=RetrievalConfig())

        leaves = searcher._convert_results([searcher._hit_to_dict(seed_hit, seed_hit.score)], limit=1)

        assert leaves[0].metadata == {
            "when": "2023-05-07",
            "who": ["Caroline"],
            "where": "Boston",
        }

    def test_empty_returns_empty(self, svc, tq):
        assert svc.assemble(tq, [], None) == []
