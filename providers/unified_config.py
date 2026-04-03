"""Unified configuration for oG-Memory.

Loads all parameters from a single YAML file, falling back to environment
variables for any values not specified.  When no YAML file exists the
behaviour is identical to the old env-only mode (full backward compat).

Priority:  YAML value  >  environment variable  >  hard-coded default
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger("ogmem.config")

DEFAULT_CONFIG_PATH = "/etc/ogmem/config.yaml"


def _load_yaml(path: str) -> dict:
    """Read a YAML file and return the top-level dict (empty on failure)."""
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed – YAML config disabled")
        return {}

    p = Path(path)
    if not p.is_file():
        logger.info("Config file %s not found, using env / defaults", path)
        return {}

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        logger.warning("Config file %s is not a YAML mapping, ignored", path)
        return {}

    logger.info("Loaded config from %s", path)
    return data


def _resolve(yaml_val: Any, env_name: str, default: Any, cast: type = str) -> Any:
    """Pick the first non-None source: YAML > env var > default, then cast."""
    if yaml_val is not None:
        return cast(yaml_val)
    env = os.environ.get(env_name)
    if env is not None:
        return cast(env)
    return default


def _resolve_bool(yaml_val: Any, env_name: str, default: bool) -> bool:
    if yaml_val is not None:
        if isinstance(yaml_val, bool):
            return yaml_val
        return str(yaml_val).lower() in ("1", "true", "yes")
    env = os.environ.get(env_name)
    if env is not None:
        return env.lower() in ("1", "true", "yes")
    return default


def _normalize_url(url: str | None) -> str | None:
    """Append /v1 if the URL does not already end with a version path."""
    if not url:
        return url
    url = url.rstrip("/")
    if url.endswith("/v1") or url.endswith("/v2") or url.endswith("/v3"):
        return url
    return url + "/v1"


@dataclass
class OgMemConfig:
    """All ogmemory configuration in one place."""

    # -- LLM ------------------------------------------------------------------
    provider: str = "mock"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # -- Embedding -------------------------------------------------------------
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_embedding_base_url: str | None = None
    openai_embedding_api_key: str | None = None
    embedding_multimodal: bool = False

    # -- Vector DB -------------------------------------------------------------
    vector_db_type: str = "memory"
    opengauss_connection_string: str | None = None
    opengauss_dimension: int = 1536
    opengauss_table_name: str = "vector_index"
    opengauss_pool_size: int = 5

    # -- Service ---------------------------------------------------------------
    http_port: int = 8090
    workers: int = 2

    # -- AGFS ------------------------------------------------------------------
    agfs_base_url: str = "http://127.0.0.1:1833"
    agfs_mount_prefix: str = "/local/plugin"

    # -- Index service ---------------------------------------------------------
    index_interval: int = 30
    index_workers: int = 1

    # -- Identity --------------------------------------------------------------
    account_id: str = "acct-demo"
    user_id: str = "u-alice"
    agent_id: str = "main"

    # -- Cache -----------------------------------------------------------------
    enable_cache: bool = True
    cache_max_size: int = 1000

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: str | None = None) -> OgMemConfig:
        """Load config: YAML first, then fill gaps from env vars.

        Args:
            config_path: Explicit path to YAML.  Falls back to ``OGMEM_CONFIG``
                         env var, then ``/etc/ogmem/config.yaml``.
        """
        path = (
            config_path
            or os.environ.get("OGMEM_CONFIG")
            or DEFAULT_CONFIG_PATH
        )
        raw = _load_yaml(path)

        llm = raw.get("llm") or {}
        emb = raw.get("embedding") or {}
        vdb = raw.get("vector_db") or {}
        svc = raw.get("service") or {}
        agfs = raw.get("agfs") or {}
        idx = raw.get("index") or {}
        ident = raw.get("identity") or {}
        cache = raw.get("cache") or {}

        base_url = _normalize_url(
            _resolve(llm.get("base_url"), "OPENAI_BASE_URL", None)
        )
        emb_base_url = _normalize_url(
            _resolve(emb.get("base_url"), "OPENAI_EMBEDDING_BASE_URL", None)
        )

        return cls(
            # LLM
            provider=_resolve(llm.get("provider"), "CONTEXTENGINE_PROVIDER", "mock"),
            openai_api_key=_resolve(llm.get("api_key"), "OPENAI_API_KEY", None),
            openai_base_url=base_url,
            openai_llm_model=_resolve(llm.get("model"), "OPENAI_LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=_resolve(llm.get("temperature"), "LLM_TEMPERATURE", 0.7, float),
            llm_max_tokens=_resolve(llm.get("max_tokens"), "LLM_MAX_TOKENS", 4096, int),
            # Embedding
            openai_embedding_model=_resolve(
                emb.get("model"), "OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002",
            ),
            openai_embedding_base_url=emb_base_url,
            openai_embedding_api_key=_resolve(
                emb.get("api_key"), "OPENAI_EMBEDDING_API_KEY", None,
            ),
            embedding_multimodal=_resolve_bool(
                emb.get("multimodal"), "OPENAI_EMBEDDING_MULTIMODAL", False,
            ),
            # Vector DB
            vector_db_type=_resolve(vdb.get("type"), "VECTOR_DB_TYPE", "memory"),
            opengauss_connection_string=_resolve(
                vdb.get("connection_string"), "OPENGAUSS_CONNECTION_STRING", None,
            ),
            opengauss_dimension=_resolve(
                vdb.get("dimension"), "OPENGAUSS_DIMENSION", 1536, int,
            ),
            opengauss_table_name=_resolve(
                vdb.get("table_name"), "OPENGAUSS_TABLE_NAME", "vector_index",
            ),
            opengauss_pool_size=_resolve(
                vdb.get("pool_size"), "OPENGAUSS_POOL_SIZE", 5, int,
            ),
            # Service
            http_port=_resolve(svc.get("http_port"), "OGMEM_HTTP_PORT", 8090, int),
            workers=_resolve(svc.get("workers"), "OGMEM_WORKERS", 2, int),
            # AGFS
            agfs_base_url=_resolve(
                agfs.get("base_url"), "AGFS_BASE_URL", "http://127.0.0.1:1833",
            ),
            agfs_mount_prefix=_resolve(
                agfs.get("mount_prefix"), "AGFS_MOUNT_PREFIX", "/local/plugin",
            ),
            # Index
            index_interval=_resolve(idx.get("interval"), "INDEX_INTERVAL", 30, int),
            index_workers=_resolve(idx.get("workers"), "INDEX_WORKERS", 1, int),
            # Identity
            account_id=_resolve(ident.get("account_id"), "OG_ACCOUNT_ID", "acct-demo"),
            user_id=_resolve(ident.get("user_id"), "OG_USER_ID", "u-alice"),
            agent_id=_resolve(ident.get("agent_id"), "OG_AGENT_ID", "main"),
            # Cache
            enable_cache=_resolve_bool(cache.get("enabled"), "OGMEM_CACHE_ENABLED", True),
            cache_max_size=_resolve(
                cache.get("max_size"), "OGMEM_CACHE_MAX_SIZE", 1000, int,
            ),
        )

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def effective_embedding_base_url(self) -> str | None:
        """Embedding base URL, falling back to the LLM base URL."""
        return self.openai_embedding_base_url or self.openai_base_url

    def effective_embedding_api_key(self) -> str | None:
        """Embedding API key, falling back to the LLM API key."""
        return self.openai_embedding_api_key or self.openai_api_key

    def to_provider_config(self):
        """Convert to a legacy ``ProviderConfig`` instance."""
        from providers.config import ProviderConfig
        return ProviderConfig.from_ogmem_config(self)

    def dump_summary(self) -> dict[str, Any]:
        """Return a sanitised dict suitable for logging (keys masked)."""
        d: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if "key" in f.name.lower() and val:
                val = val[:8] + "..."
            d[f.name] = val
        return d


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_global_config: OgMemConfig | None = None


def get_config(config_path: str | None = None) -> OgMemConfig:
    """Return the module-level singleton, creating it on first call."""
    global _global_config
    if _global_config is None:
        _global_config = OgMemConfig.load(config_path)
        logger.info("Global config loaded: %s", _global_config.dump_summary())
    return _global_config


def reset_config() -> None:
    """Reset the singleton (useful for tests)."""
    global _global_config
    _global_config = None
