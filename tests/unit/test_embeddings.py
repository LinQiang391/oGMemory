"""嵌入提供者单元测试"""

import os
import pytest
from embeddings.base import EmbeddingBase, EmbeddingFactory

# 注册 openai 提供者（执行 register 逻辑）
import embeddings.openai  # noqa: F401


class TestEmbeddingFactory:
    """嵌入工厂测试"""
    
    def test_list_providers(self):
        """测试列出提供者"""
        providers = EmbeddingFactory.list_providers()
        
        assert "openai" in providers
        assert isinstance(providers, list)
    
    @pytest.mark.requires_openai
    def test_create_openai_provider(self):
        """测试创建 OpenAI 提供者"""
        embedder = EmbeddingFactory.create("openai")
        assert isinstance(embedder, EmbeddingBase)
        assert embedder.get_model_name()
        assert embedder.get_dimension() > 0
    
    @pytest.mark.requires_openai
    def test_create_with_model_override(self):
        """测试使用模型覆盖创建提供者"""
        embedder = EmbeddingFactory.create(
            "openai",
            model="text-embedding-3-large"
        )
        
        assert embedder.get_model_name() == "text-embedding-3-large"
        assert embedder.get_dimension() == 3072
    
    def test_invalid_provider(self):
        """测试无效的提供者"""
        with pytest.raises(ValueError):
            EmbeddingFactory.create("invalid_provider")

    @pytest.mark.requires_openai
    def test_create_with_explicit_dimension(self):
        """测试创建时显式传入 dimension 参数"""
        embedder = EmbeddingFactory.create(
            "openai",
            model="text-embedding-v4",
            dimension=1024,
        )
        assert embedder.get_model_name() == "text-embedding-v4"
        assert embedder.get_dimension() == 1024


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIEmbedding:
    """OpenAI 嵌入集成测试"""
    
    @pytest.fixture
    def embedder(self):
        """嵌入器 fixture"""
        from embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding()
    
    @pytest.mark.asyncio
    async def test_encode_single(self, embedder):
        """测试单个文本编码"""
        vector = await embedder.encode_single("test text")
        dim = embedder.get_dimension()
        assert len(vector) == dim
        assert all(isinstance(v, float) for v in vector)
        assert all(0.0 <= abs(v) <= 1.0 for v in vector)

    @pytest.mark.asyncio
    async def test_encode_batch(self, embedder):
        """测试批量编码"""
        texts = ["text1", "text2", "text3"]
        vectors = await embedder.encode(texts)
        dim = embedder.get_dimension()
        assert len(vectors) == 3
        assert all(len(v) == dim for v in vectors)
        assert all(all(isinstance(x, float) for x in v) for v in vectors)
    
    @pytest.mark.asyncio
    async def test_encode_empty_list(self, embedder):
        """测试空列表编码"""
        vectors = await embedder.encode([])
        
        assert vectors == []
    
    @pytest.mark.asyncio
    async def test_encode_special_characters(self, embedder):
        """测试特殊字符编码"""
        vector = await embedder.encode_single("测试中文 🎉")
        assert len(vector) == embedder.get_dimension()
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_encode_single_empty_string(self, embedder):
        """测试空字符串编码（应返回与维度长度一致的向量）"""
        vector = await embedder.encode_single("")
        assert len(vector) == embedder.get_dimension()
        assert all(isinstance(v, float) for v in vector)


class TestOpenAIEmbeddingInit:
    """OpenAI 嵌入初始化与错误路径"""

    def test_init_without_api_key_raises(self, monkeypatch):
        """未设置 API key 时应抛出 ValueError"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from embeddings.openai import OpenAIEmbedding
        with pytest.raises(ValueError) as exc_info:
            OpenAIEmbedding(api_key="")
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_init_with_explicit_api_key_ok(self, monkeypatch):
        """显式传入 api_key 和 model 时不依赖环境变量"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_EMBEDDING_DIMENSION", raising=False)
        from embeddings.openai import OpenAIEmbedding
        embedder = OpenAIEmbedding(
            api_key="sk-test-fake", model="text-embedding-3-small", dimension=1536
        )
        assert embedder.get_dimension() == 1536
        assert embedder.get_model_name() == "text-embedding-3-small"
