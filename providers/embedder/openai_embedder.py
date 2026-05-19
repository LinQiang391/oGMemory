"""OpenAI Embedder implementation.

Production-ready embedder using OpenAI-compatible API.
Supports custom base_url for compatible services.
"""

import logging
from typing import Final

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI package is required for OpenAIEmbedder. "
        "Install it with: pip install openai"
    )

from core.interfaces import Embedder
from providers.token_tracker import TokenTracker


# Default embedding dimensions
ADA_002_DIM: Final[int] = 1536
EMBEDDING_3_SMALL_DIM: Final[int] = 1536
EMBEDDING_3_LARGE_DIM: Final[int] = 3072


class OpenAIEmbedder(Embedder):
    """OpenAI embedding model for production use.

    Supports OpenAI API and compatible services (via base_url).

    Models:
        - text-embedding-ada-002: 1536 dimensions
        - text-embedding-3-small: 1536 dimensions, faster/cheaper
        - text-embedding-3-large: 3072 dimensions, best quality

    Example:
        embedder = OpenAIEmbedder(
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="text-embedding-ada-002"
        )
        vectors = embedder.embed_texts(["hello", "world"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "text-embedding-ada-002",
        dimension: int | None = None,
        multimodal: bool = False,
    ):
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key.
            base_url: Custom base URL for compatible APIs.
            model: Embedding model name.
            dimension: Expected dimension (for validation). If None, inferred from model.
            multimodal: Use the ``/embeddings/multimodal`` endpoint.
        """
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Pass api_key parameter or set it via config / env."
            )

        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._dimension = dimension or self._get_model_dimension(model)
        self._multimodal = multimodal

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        # Increase connection pool limits to avoid PoolTimeout when
        # background drain thread and main request thread compete for
        # the same embedding API connection pool.
        import httpx
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
            ),
            timeout=httpx.Timeout(120.0, pool=60.0),
        )
        self._client = OpenAI(http_client=http_client, **client_kwargs)
        self.token_tracker = TokenTracker(model=self._model)

    # Max chars per text for embedding (~7500 tokens at 4 chars/token, under 8191 limit)
    _MAX_EMBED_CHARS: Final[int] = 30000

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI API.

        Args:
            texts: List of input texts (max 2048 texts per request).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            OpenAIError: If the API request fails.
            ValueError: If texts list is empty or too large.
        """
        if not texts:
            return []

        if len(texts) > 2048:
            raise ValueError(
                f"Batch size too large: {len(texts)}. "
                "OpenAI API supports max 2048 texts per request."
            )

        # Truncate texts exceeding token limit to avoid 429 errors
        truncated = []
        for t in texts:
            if len(t) > self._MAX_EMBED_CHARS:
                logger.warning(
                    "Truncating text for embedding: %d -> %d chars",
                    len(t), self._MAX_EMBED_CHARS,
                )
                truncated.append(t[:self._MAX_EMBED_CHARS])
            else:
                truncated.append(t)

        try:
            if self._multimodal:
                embeddings = self._embed_multimodal(truncated)
            else:
                response = self._client.embeddings.create(
                    input=truncated,
                    model=self._model
                )
                embeddings = [item.embedding for item in response.data]
                self.token_tracker.record_embed_from_usage(getattr(response, "usage", None))
        except Exception as e:
            logger.error("embed_texts FAILED: %s", e)
            raise
        for emb in embeddings:
            if len(emb) != self._dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._dimension}, "
                    f"got {len(emb)}. Check model configuration."
                )

        return embeddings

    def _embed_multimodal(self, texts: list[str]) -> list[list[float]]:
        """Call the /embeddings/multimodal endpoint for vision embedding models.

        The multimodal endpoint treats `input` as parts of ONE embedding
        (e.g. text + image), so each text must be a separate request.
        """
        import httpx

        base = (self._base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/embeddings/multimodal"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        results: list[list[float]] = []
        with httpx.Client(timeout=120) as client:
            for text in texts:
                payload = {
                    "model": self._model,
                    "input": [{"type": "text", "text": text}],
                    "dimensions": self._dimension,
                }
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                embedding = data.get("data", {}).get("embedding", [])
                if not embedding:
                    raise ValueError(f"No embedding from multimodal endpoint: {data}")
                results.append(embedding)
        return results

    def _get_model_dimension(self, model: str) -> int:
        """Get expected dimension for a model."""
        dimensions = {
            "text-embedding-ada-002": ADA_002_DIM,
            "text-embedding-3-small": EMBEDDING_3_SMALL_DIM,
            "text-embedding-3-large": EMBEDDING_3_LARGE_DIM,
        }
        return dimensions.get(model, ADA_002_DIM)  # Default to ada-002

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model


class CachedOpenAIEmbedder(OpenAIEmbedder):
    """OpenAI embedder with simple in-memory caching.

    Caches embeddings for repeated texts to reduce API calls.
    Useful for repeated queries or indexing duplicate content.

    Note: Cache is in-memory and not persisted across restarts.
    Consider using Redis for distributed caching in production.
    """

    def __init__(self, *args, cache_max_size: int = 1000, **kwargs):
        """Initialize cached embedder.

        Args:
            cache_max_size: Maximum number of cached embeddings.
        """
        # Extract cache_max_size from kwargs before passing to parent
        self._cache_max_size = cache_max_size
        super().__init__(*args, **kwargs)
        self._cache: dict[str, list[float]] = {}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with caching.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if text in self._cache:
                results.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Fetch uncached texts from API
        if uncached_texts:
            new_embeddings = super().embed_texts(uncached_texts)
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._cache[text] = embedding

            # Evict oldest entries if cache exceeds max size
            while len(self._cache) > self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            # Add to results in correct order
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results.append((idx, embedding))

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._cache)
