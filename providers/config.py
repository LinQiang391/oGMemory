"""Provider configuration for ContextEngine.

Centralized configuration for LLM and Embedder providers.
Now delegates to ``OgMemConfig`` for unified YAML + env loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from providers.embedder import Embedder, MockEmbedder, get_openai_embedder
from providers.llm import LLM, MockLLM, get_openai_llm
from providers.vector_index import InMemoryVectorIndex, OpenGaussVectorIndex
from providers.vector_index.chroma_index import ChromaVectorIndex
from core.interfaces import VectorIndex

if TYPE_CHECKING:
    from providers.unified_config import OgMemConfig


ProviderType = Literal["mock", "openai", "openai-cached", "st", "volcengine"]
VectorDbType = Literal["memory", "opengauss", "chroma"]


@dataclass
class ProviderConfig:
    """Configuration for LLM and Embedder providers."""

    # Provider type
    provider: ProviderType = "mock"

    # OpenAI configuration
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_llm_model: str = "gpt-4o-mini"

    # Separate URL/key overrides for embedding (if different from LLM)
    openai_embedding_base_url: str | None = None
    openai_embedding_api_key: str | None = None

    # Cache configuration (for cached providers)
    enable_cache: bool = True
    cache_max_size: int = 1000

    # Model parameters
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # Vector database configuration
    vector_db_type: VectorDbType = "memory"
    opengauss_connection_string: str | None = None
    opengauss_dimension: int = 1536
    opengauss_table_name: str = "vector_index"
    opengauss_pool_size: int = 5

    # ChromaDB configuration
    chroma_persist_directory: str = ".chroma_data"
    chroma_collection_name: str = "contextengine"

    # Embedding multimodal flag
    embedding_multimodal: bool = False

    @classmethod
    def from_env(cls) -> ProviderConfig:
        """Create configuration from the unified config loader.

        Environment variables:
            CONTEXTENGINE_PROVIDER: Provider type (mock/openai/openai-cached)
            OPENAI_API_KEY: OpenAI API key
            OPENAI_BASE_URL: Custom base URL for OpenAI-compatible API (auto-appends /v1 if needed)
            OPENAI_EMBEDDING_MODEL: Embedding model name
            OPENAI_LLM_MODEL: LLM model name
            VECTOR_DB_TYPE: Vector database type (memory/opengauss)
            OPENGAUSS_CONNECTION_STRING: PostgreSQL connection string for opengauss
            OPENGAUSS_DIMENSION: Embedding dimension (default 1536)
            OPENGAUSS_TABLE_NAME: Table name for vector storage
            OPENGAUSS_POOL_SIZE: Connection pool size
        """
        from providers.unified_config import get_config
        return get_config().to_provider_config()

    @classmethod
    def from_ogmem_config(cls, cfg: OgMemConfig) -> ProviderConfig:
        """Create from an existing OgMemConfig instance."""
        return cls(
            provider=os.environ.get("CONTEXTENGINE_PROVIDER", "mock"),  # type: ignore
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=cfg.openai_base_url,
            openai_embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            openai_llm_model=os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            vector_db_type=os.environ.get("VECTOR_DB_TYPE", "memory"),  # type: ignore
            opengauss_connection_string=os.environ.get("OPENGAUSS_CONNECTION_STRING"),
            opengauss_dimension=int(os.environ.get("OPENGAUSS_DIMENSION", "1536")),
            opengauss_table_name=os.environ.get("OPENGAUSS_TABLE_NAME", "vector_index"),
            opengauss_pool_size=int(os.environ.get("OPENGAUSS_POOL_SIZE", "5")),
            chroma_persist_directory=os.environ.get("CHROMA_PERSIST_DIR", ".chroma_data"),
            chroma_collection_name=os.environ.get("CHROMA_COLLECTION", "contextengine"),
        )

    def create_embedder(self) -> Embedder:
        """Create an Embedder instance based on configuration."""
        if self.provider == "mock":
            return MockEmbedder(dimension=self.opengauss_dimension)

        if self.provider == "st":
            from providers.embedder.st_embedder import SentenceTransformerEmbedder
            return SentenceTransformerEmbedder(model="BAAI/bge-large-en-v1.5")

        if self.provider == "volcengine":
            from providers.embedder.volcengine_embedder import VolcengineEmbedder
            api_key = os.environ.get("VOLC_API_KEY", "")
            base_url = os.environ.get("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            model = os.environ.get("VOLC_EMBEDDING_MODEL", "doubao-embedding-vision-250615")
            return VolcengineEmbedder(
                api_key=api_key,
                model=model,
                base_url=base_url,
                dimension=1024,
            )

        if self.provider in ("openai", "openai-cached"):
            emb_url = self.openai_embedding_base_url or self.openai_base_url
            emb_key = self.openai_embedding_api_key or self.openai_api_key
            OpenAIEmbedder, CachedOpenAIEmbedder = get_openai_embedder()
            if self.provider == "openai-cached" and self.enable_cache:
                return CachedOpenAIEmbedder(
                    api_key=emb_key,
                    base_url=emb_url,
                    model=self.openai_embedding_model,
                    dimension=self.opengauss_dimension,
                    cache_max_size=self.cache_max_size,
                    multimodal=self.embedding_multimodal,
                )
            return OpenAIEmbedder(
                api_key=emb_key,
                base_url=emb_url,
                model=self.openai_embedding_model,
                dimension=self.opengauss_dimension,
                multimodal=self.embedding_multimodal,
            )

        raise ValueError(f"Unknown provider type: {self.provider}")

    def create_vector_index(self) -> VectorIndex:
        """Create a VectorIndex instance based on configuration."""
        if self.vector_db_type == "memory":
            return InMemoryVectorIndex(dimension=self.opengauss_dimension)

        if self.vector_db_type == "opengauss":
            if not self.opengauss_connection_string:
                raise ValueError(
                    "OPENGAUSS_CONNECTION_STRING is required when vector_db_type='opengauss'"
                )
            return OpenGaussVectorIndex(
                connection_string=self.opengauss_connection_string,
                dimension=self.opengauss_dimension,
                table_name=self.opengauss_table_name,
                pool_size=self.opengauss_pool_size,
            )

        if self.vector_db_type == "chroma":
            return ChromaVectorIndex(
                collection_name=self.chroma_collection_name,
                persist_directory=self.chroma_persist_directory,
                dimension=self.opengauss_dimension,
            )

        raise ValueError(f"Unknown vector_db_type: {self.vector_db_type}")

    def create_llm(self) -> LLM:
        """Create an LLM instance based on configuration."""
        if self.provider == "mock":
            return MockLLM()

        if self.provider in ("openai", "openai-cached"):
            OpenAILLM, CachedOpenAILLM = get_openai_llm()
            if self.provider == "openai-cached" and self.enable_cache:
                return CachedOpenAILLM(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    model=self.openai_llm_model,
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                    cache_max_size=self.cache_max_size,
                )
            return OpenAILLM(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                model=self.openai_llm_model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )

        raise ValueError(f"Unknown provider type: {self.provider}")

    # Helper: convert this ProviderConfig back to an OgMemConfig-compatible form
    def to_ogmem_config(self) -> OgMemConfig:
        from providers.unified_config import OgMemConfig
        return OgMemConfig(
            provider=self.provider,
            openai_api_key=self.openai_api_key,
            openai_base_url=self.openai_base_url,
            openai_llm_model=self.openai_llm_model,
            llm_temperature=self.llm_temperature,
            llm_max_tokens=self.llm_max_tokens,
            openai_embedding_model=self.openai_embedding_model,
            openai_embedding_base_url=self.openai_embedding_base_url,
            openai_embedding_api_key=self.openai_embedding_api_key,
            embedding_multimodal=self.embedding_multimodal,
            vector_db_type=self.vector_db_type,
            opengauss_connection_string=self.opengauss_connection_string,
            opengauss_dimension=self.opengauss_dimension,
            opengauss_table_name=self.opengauss_table_name,
            opengauss_pool_size=self.opengauss_pool_size,
            enable_cache=self.enable_cache,
            cache_max_size=self.cache_max_size,
        )


# Default configuration — loaded lazily via the unified config system
DEFAULT_CONFIG = ProviderConfig.from_env()


def get_embedder(config: ProviderConfig | None = None) -> Embedder:
    """Get configured Embedder instance."""
    config = config or DEFAULT_CONFIG
    return config.create_embedder()


def get_llm(config: ProviderConfig | None = None) -> LLM:
    """Get configured LLM instance."""
    config = config or DEFAULT_CONFIG
    return config.create_llm()


def get_vector_index(config: ProviderConfig | None = None) -> VectorIndex:
    """Get configured VectorIndex instance."""
    config = config or DEFAULT_CONFIG
    return config.create_vector_index()
