"""Provider configuration for ContextEngine.

Centralized configuration for LLM and Embedder providers.
Now delegates to ``OgMemConfig`` for unified YAML + env loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from providers.embedder import Embedder, MockEmbedder, get_openai_embedder
from providers.llm import LLM, MockLLM, get_openai_llm
from providers.vector_index import InMemoryVectorIndex, OpenGaussVectorIndex
from core.interfaces import VectorIndex

if TYPE_CHECKING:
    from providers.unified_config import OgMemConfig


ProviderType = Literal["mock", "openai", "openai-cached"]
VectorDbType = Literal["memory", "opengauss"]


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

    # Embedding multimodal flag
    embedding_multimodal: bool = False

    @classmethod
    def from_env(cls) -> ProviderConfig:
        """Create configuration from the unified config loader.

        Reads YAML first (if present), then fills gaps from env vars.
        Fully backward-compatible: without a YAML file this behaves
        identically to the old pure-env approach.
        """
        from providers.unified_config import get_config
        return get_config().to_provider_config()

    @classmethod
    def from_ogmem_config(cls, cfg: OgMemConfig) -> ProviderConfig:
        """Create from an existing OgMemConfig instance."""
        return cls(
            provider=cfg.provider,  # type: ignore[arg-type]
            openai_api_key=cfg.openai_api_key,
            openai_base_url=cfg.openai_base_url,
            openai_embedding_model=cfg.openai_embedding_model,
            openai_llm_model=cfg.openai_llm_model,
            openai_embedding_base_url=cfg.openai_embedding_base_url,
            openai_embedding_api_key=cfg.openai_embedding_api_key,
            enable_cache=cfg.enable_cache,
            cache_max_size=cfg.cache_max_size,
            llm_temperature=cfg.llm_temperature,
            llm_max_tokens=cfg.llm_max_tokens,
            vector_db_type=cfg.vector_db_type,  # type: ignore[arg-type]
            opengauss_connection_string=cfg.opengauss_connection_string,
            opengauss_dimension=cfg.opengauss_dimension,
            opengauss_table_name=cfg.opengauss_table_name,
            opengauss_pool_size=cfg.opengauss_pool_size,
            embedding_multimodal=cfg.embedding_multimodal,
        )

    def create_embedder(self) -> Embedder:
        """Create an Embedder instance based on configuration."""
        if self.provider == "mock":
            return MockEmbedder(dimension=self.opengauss_dimension)

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
