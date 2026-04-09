"""Embedder providers for text embedding.

Includes mock implementations for testing and production implementations.
"""

from core.interfaces import Embedder
from .mock_embedder import MockEmbedder, ZeroEmbedder, MOCK_EMBEDDING_DIM

def get_openai_embedder():
    """Lazy-load OpenAIEmbedder. Requires: pip install openai"""
    from .openai_embedder import OpenAIEmbedder, CachedOpenAIEmbedder
    return OpenAIEmbedder, CachedOpenAIEmbedder

def get_st_embedder():
    """Lazy-load SentenceTransformerEmbedder. Requires: pip install sentence-transformers"""
    from .st_embedder import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder

__all__ = [
    "Embedder",
    # Mock implementations
    "MockEmbedder",
    "ZeroEmbedder",
    "MOCK_EMBEDDING_DIM",
    # Lazy loader for OpenAI
    "get_openai_embedder",
    # Lazy loader for sentence-transformers
    "get_st_embedder",
]
