"""Sentence-Transformer Embedder for local semantic embeddings.

Uses sentence-transformers library for real semantic similarity without
requiring external API calls. Falls back to MockEmbedder if the library
is not installed.
"""

import logging
from typing import Final

from core.interfaces import Embedder

logger = logging.getLogger(__name__)

# Default dimension for common models
MINILM_DIM: Final[int] = 384


class SentenceTransformerEmbedder(Embedder):
    """Local embedder using sentence-transformers.

    Supports any sentence-transformers model. Default is all-MiniLM-L6-v2
    (fast, 384-dim, good quality for English).

    Example:
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        vectors = embedder.embed_texts(["hello", "world"])
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", dimension: int | None = None):
        """Initialize sentence-transformer embedder.

        Args:
            model: Model name from sentence-transformers (e.g. 'all-MiniLM-L6-v2').
            dimension: Embedding dimension. If None, inferred from model.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model
        self._st = SentenceTransformer(model)
        self._dimension = dimension or self._st.get_sentence_embedding_dimension()
        logger.info(
            "SentenceTransformerEmbedder loaded: model=%s, dim=%d",
            model, self._dimension,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self._st.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model_name
