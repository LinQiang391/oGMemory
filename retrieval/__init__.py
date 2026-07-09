"""Retrieval pipeline — 4-stage search_memory chain.

Stages:
  1. QueryPlanner    — classify query into TypedQuery[]
  2. SeedRetriever   — global L0/L1/L2 mixed vector search
  3. HierarchicalSearcher — auto-expand L0/L1 to L2 leaves
  4. ResultRanker — filter to L2, sort, top-k → RetrievedBlock[]
"""
