"""Embedder providers for text embedding.

Includes mock implementations for testing and production implementations.
"""

from core.interfaces import Embedder
from .mock_embedder import MockEmbedder, ZeroEmbedder, MOCK_EMBEDDING_DIM
from .openai_embedder import OpenAIEmbedder, CachedOpenAIEmbedder

__all__ = [
    "Embedder",
    # Mock implementations
    "MockEmbedder",
    "ZeroEmbedder",
    "MOCK_EMBEDDING_DIM",
    # Production implementations
    "OpenAIEmbedder",
    "CachedOpenAIEmbedder",
]
