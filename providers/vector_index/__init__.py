"""Vector Index providers.

Implementations of the VectorIndex protocol for different backends.
"""

from providers.vector_index.in_memory_index import InMemoryVectorIndex

__all__ = ["InMemoryVectorIndex"]
