"""Volcengine Ark Embedder — same model as OpenViking.

Uses ``volcenginesdkarkruntime`` to call doubao-embedding-vision-250615
via the multimodal embeddings endpoint.

The vendor endpoint treats ``input`` as parts of ONE document, so there
is no real batched mode at the API level (confirmed by the prior
comment in this file). What we can do — and what VECTORIZATION_PLAN §2
prescribes — is:

  * deduplicate identical texts within a batch (1 API call per unique
    canonical key, not per raw occurrence),
  * run the remaining unique requests in parallel under a bounded
    thread pool (``embed_concurrency``),
  * back the result with the shared ``CacheBackend`` so repeat queries
    and unchanged nodes cost zero tokens.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from core.interfaces import Embedder
from providers.cache_backend import CacheBackend, LRUCacheBackend, make_embed_key
from providers.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "doubao-embedding-vision-250615"
DEFAULT_DIMENSION: Final[int] = 1024
DEFAULT_BASE_URL: Final[str] = "https://ark.cn-beijing.volces.com/api/coding/v3"


class VolcengineEmbedder(Embedder):
    """Dense embedder backed by Volcengine Ark API.

    Uses the multimodal embeddings endpoint with text input. The endpoint
    requires one HTTP request per text, so speed gains come exclusively
    from deduplication + concurrency + caching.
    """

    # Provider prefix baked into cache keys to prevent cross-vendor
    # collisions when caches are shared across embedders.
    _CACHE_PROVIDER_PREFIX: Final[str] = "volcengine"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dimension: int = DEFAULT_DIMENSION,
        *,
        enable_cache: bool = False,
        cache_max_size: int = 1000,
        cache_backend: CacheBackend | None = None,
        embed_concurrency: int = 8,
    ):
        try:
            import volcenginesdkarkruntime  # noqa: F401
        except ImportError:
            raise ImportError(
                "volcenginesdkarkruntime is required for VolcengineEmbedder. "
                "Install with: pip install volcenginesdkarkruntime"
            )

        self._model_name = model
        self._dimension = dimension
        self._client = volcenginesdkarkruntime.Ark(
            api_key=api_key,
            base_url=base_url,
        )
        self._embed_concurrency = max(1, embed_concurrency)
        self.token_tracker = TokenTracker()

        if cache_backend is not None:
            self._cache: CacheBackend | None = cache_backend
        elif enable_cache:
            self._cache = LRUCacheBackend(max_size=cache_max_size)
        else:
            self._cache = None

        logger.info(
            "VolcengineEmbedder loaded: model=%s, dim=%d, cache=%s, concurrency=%d",
            model,
            dimension,
            "on" if self._cache is not None else "off",
            self._embed_concurrency,
        )

    def embed_texts(
        self,
        texts: list[str],
        *,
        model_hint: str | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts via Ark multimodal embeddings API.

        Args:
            texts: Batch of input strings; identical entries are deduped.
            model_hint: Reserved for tiered-model selection. Ignored here
                (single-model embedder). Keyword-only per Embedder Protocol.

        Raises:
            Any network/ API error — no silent fallback (CLAUDE.md §工程原则).
        """
        del model_hint  # single-model embedder

        if not texts:
            return []

        keys = [
            make_embed_key(t, model=self._cache_model_name(), dimension=self._dimension)
            for t in texts
        ]

        cached_vectors: dict[str, list[float]] = {}
        if self._cache is not None:
            for key in set(keys):
                hit = self._cache.get(key)
                if hit is not None:
                    cached_vectors[key] = hit

        uncached_order: list[str] = []
        uncached_texts: list[str] = []
        seen_pending: set[str] = set()
        for key, text in zip(keys, texts):
            if key in cached_vectors or key in seen_pending:
                continue
            seen_pending.add(key)
            uncached_order.append(key)
            uncached_texts.append(text)

        if uncached_texts:
            new_vectors = self._call_api(uncached_texts)
            for key, vector in zip(uncached_order, new_vectors):
                cached_vectors[key] = vector
                if self._cache is not None:
                    self._cache.put(key, vector)

        saved_count = len(keys) - len(uncached_texts)
        if saved_count > 0:
            avg_tokens_per_text = max(
                1,
                sum(len(t) for t in texts) // (4 * max(1, len(texts))),
            )
            self.token_tracker.record_local_cache_hit(
                tokens_saved=saved_count * avg_tokens_per_text,
            )

        return [cached_vectors[key] for key in keys]

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Issue one API request per text, parallelized under a thread pool.

        Previously this was a serial ``for`` loop that issued N sequential
        HTTP calls for a batch of N; the fix — prescribed by the plan — is
        concurrent fan-out. We keep the request body identical to what
        worked before so the vendor contract is unchanged.
        """
        if not texts:
            return []

        def _one(text: str) -> list[float]:
            response = self._client.multimodal_embeddings.create(
                input=[{"type": "text", "text": text}],
                model=self._model_name,
                dimensions=self._dimension,
            )
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                if isinstance(usage, dict):
                    total = usage.get("total_tokens", 0) or 0
                else:
                    total = getattr(usage, "total_tokens", 0) or 0
                if total:
                    self.token_tracker.record_embed(total)
            data = response.data
            if hasattr(data, "embedding"):
                return data.embedding
            if isinstance(data, list) and len(data) > 0:
                return data[0].embedding
            raise ValueError(
                f"Unexpected embedding response for text prefix: {text[:50]!r}"
            )

        concurrency = min(self._embed_concurrency, len(texts))
        results: list[list[float] | None] = [None] * len(texts)
        try:
            if concurrency == 1:
                for idx, text in enumerate(texts):
                    results[idx] = _one(text)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = {
                        pool.submit(_one, text): idx
                        for idx, text in enumerate(texts)
                    }
                    for fut in futures:
                        idx = futures[fut]
                        # Let exceptions propagate — no silent fallback.
                        results[idx] = fut.result()
        except Exception as exc:
            logger.error(
                "VolcengineEmbedder._call_api FAILED: %s", exc, exc_info=True,
            )
            raise

        return [vec for vec in results if vec is not None]

    def _cache_model_name(self) -> str:
        return f"{self._CACHE_PROVIDER_PREFIX}/{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def cache_enabled(self) -> bool:
        return self._cache is not None

    @property
    def cache_size(self) -> int:
        if self._cache is None:
            return 0
        return self._cache.stats().get("size", 0)

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()