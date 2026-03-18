"""Retrieval layer for ContextEngine.

This package implements the retrieval chain:
1. Planner - infers query type and categories
2. L0Retriever - vector similarity search
3. MemoryReadService - L1/L2 reading with URI validation
4. AssemblyService - assembles RetrievedBlock
5. HierarchicalSearcher - expands results with relation edges (Phase 2)

See CLAUDE.md §7 for retrieval chain rules.
"""

from retrieval.planner import Planner
from retrieval.l0_retriever import L0Retriever
from retrieval.memory_read_service import MemoryReadService, L1, L2
from retrieval.assembly_service import AssemblyService, RetrievedBlock, RetrievalResult
from retrieval.hierarchical_searcher import HierarchicalSearcher, ExpandedHit

__all__ = [
    "Planner",
    "L0Retriever",
    "MemoryReadService",
    "L1",
    "L2",
    "AssemblyService",
    "RetrievedBlock",
    "RetrievalResult",
    "HierarchicalSearcher",
    "ExpandedHit",
]
