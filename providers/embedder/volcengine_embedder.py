"""Volcengine Ark Embedder — same model as OpenViking.

Uses ``volcenginesdkarkruntime`` to call doubao-embedding-vision-250615
via the multimodal embeddings endpoint (identical to OpenViking's
VolcengineDenseEmbedder).
"""

import logging
from typing import Final

from core.interfaces import Embedder

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "doubao-embedding-vision-250615"
DEFAULT_DIMENSION: Final[int] = 1024
DEFAULT_BASE_URL: Final[str] = "https://ark.cn-beijing.volces.com/api/v3"


class VolcengineEmbedder(Embedder):
    """Dense embedder backed by Volcengine Ark API.

    Uses the multimodal embeddings endpoint with text input, which is
    exactly how OpenViking's ``VolcengineDenseEmbedder`` works.
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
        logger.info(
            "VolcengineEmbedder loaded: model=%s, dim=%d", model, dimension,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via Ark multimodal embeddings API."""
        if not texts:
            return []

        # Batch call — multimodal endpoint accepts a list of input items
        multimodal_inputs = [{"type": "text", "text": t} for t in texts]
        response = self._client.multimodal_embeddings.create(
            input=multimodal_inputs,
            model=self._model_name,
        )

        # response.data is a single item for multimodal, or a list
        # Handle both cases
        data = response.data
        if hasattr(data, "embedding"):
            # Single item returned
            return [data.embedding]
        elif isinstance(data, list):
            return [item.embedding for item in data]
        else:
            # Batch multimodal returns .data as list of items
            embeddings = []
            for item in data:
                emb = item.embedding
                embeddings.append(emb)
            return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model_name
