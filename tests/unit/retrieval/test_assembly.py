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
from retrieval.result_ranker import ResultRanker


@pytest.fixture()
def svc():
    return ResultRanker(RetrievalConfig())


@pytest.fixture()
def tq():
    return TypedQuery(text="test", context_type="MEMORY", categories=[], top_k=3)


def _leaf(uri: str, score: float, level: int = 2) -> LeafHit:
    return LeafHit(uri=uri, score=score, level=level, abstract=f"abs-{uri}")


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
                SeedHit(uri="x", score=0.6, level=2, abstract="from-seed"),
            ],
        )
        blocks = svc.assemble(tq, [], seed)
        assert len(blocks) == 1
        assert blocks[0].uri == "x"

    def test_empty_returns_empty(self, svc, tq):
        assert svc.assemble(tq, [], None) == []
