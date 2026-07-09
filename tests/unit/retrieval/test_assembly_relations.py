"""Tests for ResultRanker with relation store integration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.models import (
    LeafHit,
    RelationEdge,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    SeedResult,
    TypedQuery,
)
from retrieval.result_ranker import ResultRanker
from tests.unit.retrieval.conftest import make_ctx


def _leaf(uri: str, score: float, level: int = 2) -> LeafHit:
    return LeafHit(uri=uri, score=score, level=level, abstract=f"abs-{uri}")


def _edge(to_uri: str, weight: float = 0.8) -> RelationEdge:
    return RelationEdge(
        from_uri="source", to_uri=to_uri,
        relation_type="related_to", weight=weight, reason="test",
    )


class TestAssemblyWithRelations:
    def test_fills_relations_from_store(self):
        """When relation_store is provided, blocks get their relations filled."""
        mock_store = MagicMock()
        mock_store.get_one_hop.return_value = [_edge("rel-a"), _edge("rel-b")]

        svc = ResultRanker(RetrievalConfig(), relation_store=mock_store)
        tq = TypedQuery(text="test", context_type="MEMORY", categories=[], top_k=3)
        ctx = make_ctx()
        leaves = [_leaf("u1", 0.9)]

        blocks = svc.assemble(tq, leaves, None, ctx=ctx)

        assert len(blocks) == 1
        assert len(blocks[0].relations) == 2
        assert blocks[0].relations[0].to_uri == "rel-a"
        mock_store.get_one_hop.assert_called_once_with("u1", ctx, limit=3)

    def test_no_relation_store_graceful(self):
        """Without relation_store, relations list stays empty."""
        svc = ResultRanker(RetrievalConfig(), relation_store=None)
        tq = TypedQuery(text="test", context_type="MEMORY", categories=[], top_k=3)
        leaves = [_leaf("u1", 0.9)]

        blocks = svc.assemble(tq, leaves, None, ctx=make_ctx())

        assert len(blocks) == 1
        assert blocks[0].relations == []

    def test_relation_store_failure_graceful(self):
        """When relation_store raises, blocks still returned with empty relations."""
        mock_store = MagicMock()
        mock_store.get_one_hop.side_effect = Exception("connection error")

        svc = ResultRanker(RetrievalConfig(), relation_store=mock_store)
        tq = TypedQuery(text="test", context_type="MEMORY", categories=[], top_k=3)
        leaves = [_leaf("u1", 0.9)]

        blocks = svc.assemble(tq, leaves, None, ctx=make_ctx())

        assert len(blocks) == 1
        assert blocks[0].relations == []
