"""Retrieval layer for ContextEngine.

This package implements the retrieval chain:
1. Planner - infers query type and categories
2. L0Retriever - vector similarity search
3. MemoryReadService - L1/L2 reading with URI validation
4. AssemblyService - assembles RetrievedBlock
5. HierarchicalSearcher - expands results with relation edges (Phase 2, optional)

See CLAUDE.md §7 for retrieval chain rules.
"""

import importlib.util

from retrieval.planner import Planner
from retrieval.l0_retriever import L0Retriever
from retrieval.memory_read_service import MemoryReadService, L1, L2
from retrieval.assembly_service import AssemblyService, RetrievedBlock, RetrievalResult

# Phase 2: HierarchicalSearcher is optional for graph-enhanced retrieval
_HIERARCHICAL_AVAILABLE = importlib.util.find_spec("retrieval.hierarchical_searcher") is not None
if _HIERARCHICAL_AVAILABLE:
    from retrieval.hierarchical_searcher import HierarchicalSearcher, ExpandedHit
else:
    HierarchicalSearcher = None  # type: ignore
    ExpandedHit = None  # type: ignore

__all__ = [
    "Planner",
    "L0Retriever",
    "MemoryReadService",
    "L1",
    "L2",
    "AssemblyService",
    "RetrievedBlock",
    "RetrievalResult",
    "HierarchicalSearcher",  # Phase 2, may be None
    "ExpandedHit",  # Phase 2, may be None
]
