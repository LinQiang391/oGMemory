"""Mock Embedder for testing.

Returns fixed-dimension vectors for reproducible testing.
Each unique text gets a deterministic vector based on its hash.

MockEmbedder is purely local (no API, no tokens), but it still
participates in the Embedder Protocol — specifically the
``model_hint`` keyword introduced in VECTORIZATION_PLAN §1. Accepting
and ignoring the hint keeps contract tests green across providers.

For consistency with production providers we also dedupe inputs before
hashing so identical strings in the same batch share one hash call.
"""

import hashlib
from typing import Final

from core.interfaces import Embedder
from providers.token_tracker import TokenTracker


# Fixed embedding dimension for mock (matches common small models)
MOCK_EMBEDDING_DIM: Final[int] = 384


class MockEmbedder(Embedder):
    """Mock embedder for testing.

    Returns deterministic vectors based on text hash.
    Each unique text produces a consistent vector across runs.
    """

    def __init__(self, dimension: int = MOCK_EMBEDDING_DIM) -> None:
        """Initialize mock embedder.

        Args:
            dimension: Vector dimension (default: 384).
        """
        self._dimension = dimension
        self._model = "mock-embedding"
        self.token_tracker = TokenTracker(model=self._model)
        # Mocks don't call a real SDK and therefore have no usage block. Tests
        # that care about embedding token attribution can stub this attribute.
        self._mock_usage: dict | None = None

    @property
    def model(self) -> str:
        """Stable identifier used by perf attribution / rate cards."""
        return self._model

    def embed_texts(
        self,
        texts: list[str],
        *,
        model_hint: str | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of input texts.
            model_hint: Accepted for Protocol parity; ignored (mock embedder
                is single-model).

        Returns:
            List of vectors, one per input text.
            Vectors are deterministic based on text content.
        """
        del model_hint  # single-model embedder

        if not texts:
            return []

        # Dedup: hash each unique text once, then broadcast by position.
        cache: dict[str, list[float]] = {}
        results: list[list[float]] = []
        for text in texts:
            vec = cache.get(text)
            if vec is None:
                vec = self._hash_to_vector(text)
                cache[text] = vec
            results.append(vec)
        self.token_tracker.record_embed_from_usage(self._mock_usage)
        return results

    def _hash_to_vector(self, text: str) -> list[float]:
        """Convert text to deterministic vector via hash.

        Uses SHA256 hash of text to seed a pseudo-random vector.
        Ensures same text always produces same vector.
        """
        # Hash the text
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Expand hash to full dimension
        vector: list[float] = []
        for i in range(self._dimension):
            # Use hash bytes cyclically
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1] range
            normalized = (byte_val - 128) / 128.0
            vector.append(normalized)

        return vector


class ZeroEmbedder(Embedder):
    """Mock embedder that returns zero vectors.

    Useful for testing isolation of vector effects.
    All texts get identical zero vectors.
    """

    def __init__(self, dimension: int = MOCK_EMBEDDING_DIM) -> None:
        """Initialize zero embedder.

        Args:
            dimension: Vector dimension (default: 384).
        """
        self._dimension = dimension

    def embed_texts(
        self,
        texts: list[str],
        *,
        model_hint: str | None = None,
    ) -> list[list[float]]:
        """Return zero vectors for all texts.

        Args:
            texts: List of input texts (ignored).
            model_hint: Accepted for Protocol parity; ignored.

        Returns:
            List of zero vectors, one per input text.
        """
        del model_hint
        return [[0.0] * self._dimension for _ in texts]