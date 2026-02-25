"""OpenAI 嵌入提供者"""

import os
from typing import List, Optional
from .base import EmbeddingBase, EmbeddingFactory


class OpenAIEmbedding(EmbeddingBase):
    """OpenAI文本嵌入实现"""

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "text-embedding-v4": 1024,
    }

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        dimension: Optional[int] = None,
    ):
        self.model = model or os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # 仅当模型也来自环境变量时才使用环境变量中的维度，否则按模型检测
        dim_env = os.environ.get("OPENAI_EMBEDDING_DIMENSION")
        model_from_env = os.environ.get("OPENAI_EMBEDDING_MODEL")
        if dimension is not None:
            self._dimension = dimension
        elif dim_env and (model_from_env == self.model):
            self._dimension = int(dim_env)
        else:
            self._dimension = self._get_model_dimension()
        self._client = None

    def _get_client(self):
        """延迟初始化客户端"""
        if self._client is None:
            try:
                import openai

                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = openai.AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError("Please install: pip install openai")
        return self._client

    def _get_model_dimension(self) -> int:
        """获取模型维度"""
        if self.model in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.model]

        try:
            import openai

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            client = openai.OpenAI(**kwargs)
            response = client.embeddings.create(input=["test"], model=self.model)
            return len(response.data[0].embedding)
        except Exception as e:
            raise ValueError(f"Failed to detect dimension for model {self.model}: {e}")

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self._dimension

    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model

    # 部分 API（如兼容接口）单次请求最多 10 条，统一按批处理
    ENCODE_BATCH_SIZE = 10

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本（按批请求，每批最多 ENCODE_BATCH_SIZE 条）"""
        if not texts:
            return []

        client = self._get_client()
        results: List[List[float]] = []

        try:
            for i in range(0, len(texts), self.ENCODE_BATCH_SIZE):
                batch = texts[i : i + self.ENCODE_BATCH_SIZE]
                response = await client.embeddings.create(input=batch, model=self.model)
                results.extend([item.embedding for item in response.data])
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts: {e}")

    async def encode_single(self, text: str) -> List[float]:
        """编码单个文本"""
        results = await self.encode([text])
        return results[0] if results else []


EmbeddingFactory.register("openai", OpenAIEmbedding)
