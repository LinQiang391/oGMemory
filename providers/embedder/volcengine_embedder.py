"""Volcengine Ark Embedder.

Uses ``volcenginesdkarkruntime`` to call doubao-embedding-vision-250615
via the multimodal embeddings endpoint.
"""

import logging
from typing import Final

from core.interfaces import Embedder
from providers.token_tracker import TokenTracker, embedding_input_scope, embedding_usage_tokens

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "doubao-embedding-vision-250615"
DEFAULT_DIMENSION: Final[int] = 1024
DEFAULT_BASE_URL: Final[str] = "https://ark.cn-beijing.volces.com/api/coding/v3"


class VolcengineEmbedder(Embedder):
    """Dense embedder backed by Volcengine Ark API.

    Uses the multimodal embeddings endpoint with text input, which is
    using text input via the multimodal embeddings endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dimension: int = DEFAULT_DIMENSION,
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
        self.token_tracker = TokenTracker(model=self._model_name)
        logger.info(
            "VolcengineEmbedder loaded: model=%s, dim=%d", model, dimension,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via Ark multimodal embeddings API.

        The multimodal endpoint treats the input list as parts of ONE document,
        so each text must be embedded in a separate API call.
        """
        if not texts:
            return []

        results: list[list[float]] = []
        for index, text in enumerate(texts):
            try:
                response = self._client.multimodal_embeddings.create(
                    input=[{"type": "text", "text": text}],
                    model=self._model_name,
                    dimensions=self._dimension,
                )
            except Exception:
                with embedding_input_scope([index]):
                    self.token_tracker.record_failure('embedding')
                raise
            usage = getattr(response, 'usage', None)
            total = embedding_usage_tokens(usage)
            with embedding_input_scope([index]):
                self.token_tracker.record_embed(total or 0, usage_known=total is not None)
            data = response.data
            if hasattr(data, "embedding"):
                results.append(data.embedding)
            elif isinstance(data, list) and len(data) > 0:
                results.append(data[0].embedding)
            else:
                raise ValueError(f"Unexpected embedding response for text: {text[:50]}")
        return results

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model_name
