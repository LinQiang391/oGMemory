"""Provider configuration for ContextEngine.

Centralized configuration for LLM and Embedder providers.
Supports environment-based configuration for different deployments.
"""

import os
from dataclasses import dataclass
from typing import Literal

from providers.embedder import Embedder, MockEmbedder, get_openai_embedder
from providers.llm import LLM, MockLLM, get_openai_llm


ProviderType = Literal["mock", "openai", "openai-cached"]


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

    # Cache configuration (for cached providers)
    enable_cache: bool = True
    cache_max_size: int = 1000

    # Model parameters
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        """Create configuration from environment variables.

        Environment variables:
            CONTEXTENGINE_PROVIDER: Provider type (mock/openai/openai-cached)
            OPENAI_API_KEY: OpenAI API key
            OPENAI_BASE_URL: Custom base URL for OpenAI-compatible API (auto-appends /v1 if needed)
            OPENAI_EMBEDDING_MODEL: Embedding model name
            OPENAI_LLM_MODEL: LLM model name
        """
        base_url = os.environ.get("OPENAI_BASE_URL")
        # Auto-append /v1 if needed (for custom OpenAI-compatible endpoints)
        if base_url and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        return cls(
            provider=os.environ.get("CONTEXTENGINE_PROVIDER", "mock"),  # type: ignore
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=base_url,
            openai_embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            openai_llm_model=os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        )

    def create_embedder(self) -> Embedder:
        """Create an Embedder instance based on configuration.

        Returns:
            Configured Embedder instance.
        """
        if self.provider == "mock":
            return MockEmbedder()

        if self.provider in ("openai", "openai-cached"):
            # Lazy load OpenAI embedder
            OpenAIEmbedder, CachedOpenAIEmbedder = get_openai_embedder()
            if self.provider == "openai-cached" and self.enable_cache:
                return CachedOpenAIEmbedder(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    model=self.openai_embedding_model,
                    cache_max_size=self.cache_max_size,
                )
            return OpenAIEmbedder(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                model=self.openai_embedding_model,
            )

        raise ValueError(f"Unknown provider type: {self.provider}")

    def create_llm(self) -> LLM:
        """Create an LLM instance based on configuration.

        Returns:
            Configured LLM instance.
        """
        if self.provider == "mock":
            return MockLLM()

        if self.provider in ("openai", "openai-cached"):
            # Lazy load OpenAI LLM
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


# Default configuration
DEFAULT_CONFIG = ProviderConfig.from_env()


def get_embedder(config: ProviderConfig | None = None) -> Embedder:
    """Get configured Embedder instance.

    Args:
        config: Provider configuration. If None, uses environment variables.

    Returns:
        Configured Embedder.
    """
    config = config or DEFAULT_CONFIG
    return config.create_embedder()


def get_llm(config: ProviderConfig | None = None) -> LLM:
    """Get configured LLM instance.

    Args:
        config: Provider configuration. If None, uses environment variables.

    Returns:
        Configured LLM.
    """
    config = config or DEFAULT_CONFIG
    return config.create_llm()
