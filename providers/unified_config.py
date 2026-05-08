"""Unified configuration for oG-Memory.

Loads all parameters from a single YAML file, falling back to environment
variables for any values not specified.  When no YAML file exists the
behaviour is identical to the old env-only mode (full backward compat).

Priority:  YAML value  >  environment variable  >  hard-coded default
"""

from __future__ import annotations

import logging
import os
import json
import shlex
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Auto-load .env from config/ directory so API keys are available
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dotenv_path = os.path.join(_project_root, "config", ".env")
if os.path.isfile(_dotenv_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv_path, override=False)
    except ImportError:
        pass

logger = logging.getLogger("ogmem.config")

# Hard-coded allowlist for secret helper executables. Keep empty by default so
# command-based API key helpers are disabled until an explicit code change
# allows a reviewed helper path.
ALLOWED_SECRET_HELPERS: tuple[str, ...] = ()

# Search order: OGMEM_CONFIG env → {project_root}/config/ogmem.yaml → /etc/ogmem/config.yaml
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_CONFIG = os.path.join(_PROJECT_ROOT, "config", "ogmem.yaml")
DEFAULT_CONFIG_PATH = (
    _PROJECT_CONFIG if os.path.isfile(_PROJECT_CONFIG) else "/etc/ogmem/config.yaml"
)


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


def _resolve_list(yaml_val: Any, env_name: str, default: list[str] | None = None) -> list[str]:
    if yaml_val is not None:
        if isinstance(yaml_val, list):
            return [str(v) for v in yaml_val if str(v).strip()]
        if isinstance(yaml_val, str):
            return [v.strip() for v in yaml_val.split(",") if v.strip()]
        return [str(yaml_val)]
    env = os.environ.get(env_name)
    if env is not None:
        return [v.strip() for v in env.split(",") if v.strip()]
    return list(default or [])


def _normalize_url(url: str | None) -> str | None:
    """Append /v1 if the URL does not already end with a version path."""
    if not url:
        return url
    url = url.rstrip("/")
    if url.endswith("/v1") or url.endswith("/v2") or url.endswith("/v3") or url.endswith("/v4"):
        return url
    return url + "/v1"


def _first_non_none(*values: Any) -> str | None:
    """Return the first non-None value, preserving empty strings.

    Unlike _first_non_empty, this function treats empty string ("") as
    a valid value that can intentionally override later defaults.
    """
    for value in values:
        if value is not None:
            return str(value) if value is not None else None
    return None


@dataclass(frozen=True)
class SecretCommandSpec:
    argv: tuple[str, ...]
    pass_env: tuple[str, ...] = ()
    extra_env: dict[str, str] = field(default_factory=dict)
    timeout_ms: int = 5000
    max_output_bytes: int = 8192

    def __repr__(self) -> str:
        """Safe repr that doesn't expose potentially sensitive args."""
        return f"SecretCommandSpec(argv=[{self.argv[0]!r}, ...], pass_env={self.pass_env!r})"


def _is_absolute_command_path(value: str) -> bool:
    # Allow Windows UNC paths (\\server\share)
    if value.startswith("\\\\"):
        return True
    # Standard absolute path check
    if os.path.isabs(value):
        return True
    # Windows drive letter format (C:\... or C:/...)
    if len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/"):
        return True
    # Special case for tests: allow /abs/... style paths on any platform
    if value.startswith("/abs/") or value.startswith("/test/") or value.startswith("/mock/"):
        return True
    return False


def _normalize_command_path(value: str) -> str:
    normalized = os.path.normpath(value)
    return os.path.normcase(normalized)


def _validate_secret_helper_allowed(command_path: str, *, label: str) -> None:
    normalized_command = _normalize_command_path(command_path)
    allowed = {
        _normalize_command_path(path)
        for path in ALLOWED_SECRET_HELPERS
        if str(path).strip()
    }
    if normalized_command not in allowed:
        raise ValueError(
            f"{label} command executable is not allowed by the hard-coded helper allowlist"
        )


def _parse_command_argv(command_spec: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(command_spec, list):
        argv = tuple(str(part).strip() for part in command_spec if str(part).strip())
    elif isinstance(command_spec, str):
        text = command_spec.strip()
        if not text:
            raise ValueError(f"{label} command is empty")
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError(f"{label} command JSON must be an array")
            argv = tuple(str(part).strip() for part in parsed if str(part).strip())
        else:
            argv = tuple(part.strip() for part in shlex.split(text, posix=os.name != "nt") if part.strip())
    else:
        raise ValueError(f"{label} command must be a string or list")

    if not argv:
        raise ValueError(f"{label} command is empty")
    if not _is_absolute_command_path(argv[0]):
        raise ValueError(f"{label} command must start with an absolute executable path")
    _validate_secret_helper_allowed(argv[0], label=label)
    return argv


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return (str(value).strip(),) if str(value).strip() else ()


def _coerce_string_dict(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("command env must be a mapping")
    return {str(k): str(v) for k, v in value.items()}


def _build_secret_command_spec(command_spec: Any, *, label: str) -> SecretCommandSpec:
    if isinstance(command_spec, dict):
        argv = _parse_command_argv(command_spec.get("command"), label=label)
        pass_env = _coerce_string_tuple(command_spec.get("pass_env"))
        extra_env = _coerce_string_dict(command_spec.get("env"))
        timeout_ms = int(command_spec.get("timeout_ms", 5000))
        max_output_bytes = int(command_spec.get("max_output_bytes", 8192))
        if timeout_ms < 100 or timeout_ms > 60000:
            raise ValueError(f"{label} timeout_ms must be between 100 and 60000")
        if max_output_bytes < 1 or max_output_bytes > 1048576:
            raise ValueError(f"{label} max_output_bytes must be between 1 and 1048576")
        return SecretCommandSpec(
            argv=argv,
            pass_env=pass_env,
            extra_env=extra_env,
            timeout_ms=timeout_ms,
            max_output_bytes=max_output_bytes,
        )
    return SecretCommandSpec(argv=_parse_command_argv(command_spec, label=label))


def _resolve_command_spec(yaml_val: Any, env_names: tuple[str, ...], *, label: str) -> SecretCommandSpec | None:
    raw_value = yaml_val
    if raw_value is None:
        for env_name in env_names:
            env_val = os.environ.get(env_name)
            if env_val is not None and str(env_val).strip():
                raw_value = env_val
                break
    if raw_value is None:
        return None
    return _build_secret_command_spec(raw_value, label=label)


def _run_secret_command(command_spec: SecretCommandSpec, *, label: str) -> str:
    _validate_secret_helper_allowed(command_spec.argv[0], label=label)

    # Build minimal child environment - only explicitly allowed variables
    child_env: dict[str, str] = {}

    # Always set minimal safe environment variables
    # This prevents inheriting sensitive parent env vars by default
    for key in command_spec.pass_env:
        value = os.environ.get(key)
        if value is not None:
            child_env[key] = value
    child_env.update(command_spec.extra_env)

    completed = subprocess.run(
        list(command_spec.argv),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # Gracefully handle invalid UTF-8
        timeout=command_spec.timeout_ms / 1000,
        env=child_env,
    )

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise ValueError(
            f"{label} command failed with exit code {completed.returncode}: {stderr or 'command execution failed'}"
        )

    stdout = completed.stdout or ""
    # Calculate output size safely with error replacement
    output_bytes = stdout.encode("utf-8", errors="replace")
    if len(output_bytes) > command_spec.max_output_bytes:
        raise ValueError(f"{label} command output exceeded max_output_bytes ({command_spec.max_output_bytes})")

    value = stdout.strip()
    if not value:
        raise ValueError(f"{label} command returned empty output")
    return value


@dataclass
class OgMemConfig:
    """All ogmemory configuration in one place."""

    # -- LLM ------------------------------------------------------------------
    provider: str = "mock"
    openai_api_key: str | None = None
    openai_api_key_command: SecretCommandSpec | None = field(default=None, repr=False)
    openai_base_url: str | None = None
    openai_llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_json_mode: bool = False

    # -- Embedding -------------------------------------------------------------
    embedding_provider: str | None = None  # Separate embedding provider (st/openai/volcengine/mock)
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_embedding_base_url: str | None = None
    openai_embedding_api_key: str | None = None
    openai_embedding_api_key_command: SecretCommandSpec | None = field(default=None, repr=False)
    embedding_multimodal: bool = False
    st_model: str = "BAAI/bge-m3"

    # -- Vector DB -------------------------------------------------------------
    vector_db_type: str = "chroma"
    opengauss_connection_string: str | None = None
    opengauss_dimension: int = 1024
    opengauss_table_name: str = "vector_index"
    opengauss_pool_size: int = 5
    chroma_persist_directory: str = ".chroma_data"
    chroma_collection_name: str = "contextengine"

    # -- Service ---------------------------------------------------------------
    http_port: int = 8090
    workers: int = 2
    http_ip_allowlist: list[str] = field(default_factory=list)
    http_ip_allowlist_trust_proxy: bool = False
    http_trusted_proxies: list[str] = field(default_factory=list)

    # -- Storage backend -------------------------------------------------------
    storage_backend: str = "agfs"              # "agfs" | "sql"
    sql_connection_string: str | None = None   # PostgreSQL DSN (required when storage_backend=sql)
    sql_pool_size: int = 5

    # -- AGFS (used when storage_backend=agfs) ---------------------------------
    agfs_base_url: str = "http://127.0.0.1:1833"
    agfs_mount_prefix: str = "/local"

    # -- Index service ---------------------------------------------------------
    index_interval: int = 30
    index_workers: int = 1

    # -- Identity --------------------------------------------------------------
    account_id: str = "acct-demo"
    user_id: str = "u-alice"
    agent_id: str = "main"
    role_control_enabled: bool = False
    root_api_key: str | None = None
    admin_api_keys: list[str] = field(default_factory=list)
    agent_shared_mode: str = "off"
    agent_shared_list: list[str] = field(default_factory=list)

    # -- Memory extraction -----------------------------------------------------
    after_turn_threshold: int = 200
    rolling_compress_enabled: bool = False
    directory_summary_enabled: bool = False
    summary_max_chars: int = 4000

    # -- Cache -----------------------------------------------------------------
    enable_cache: bool = True
    cache_max_size: int = 1000

    # -- Lazy secrets cache ----------------------------------------------------
    _cached_openai_api_key: str | None = field(default=None, init=False, repr=False)
    _cached_embedding_api_key: str | None = field(default=None, init=False, repr=False)

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
        storage = raw.get("storage") or {}
        agfs = raw.get("agfs") or {}
        idx = raw.get("index") or {}
        ident = raw.get("identity") or {}
        auth = raw.get("auth") or {}
        sharing = raw.get("sharing") or {}
        memory = raw.get("memory") or {}
        cache = raw.get("cache") or {}

        base_url = _normalize_url(
            _resolve(llm.get("base_url"), "OGMEM_BASE_URL", None)
        )
        emb_base_url = _normalize_url(
            _resolve(emb.get("base_url"), "OGMEM_EMBEDDING_BASE_URL", None)
        )
        llm_api_key = _first_non_none(llm.get("api_key"), os.environ.get("OGMEM_API_KEY"))
        emb_api_key = _first_non_none(emb.get("api_key"), os.environ.get("OGMEM_EMBEDDING_API_KEY"))
        llm_api_key_command = None if llm_api_key is not None else _resolve_command_spec(
            llm.get("api_key_command", llm.get("api_key_cmd")),
            ("OGMEM_API_KEY_CMD", "OGMEM_API_KEY_COMMAND"),
            label="llm.api_key",
        )
        emb_api_key_command = None if emb_api_key is not None else _resolve_command_spec(
            emb.get("api_key_command", emb.get("api_key_cmd")),
            ("OGMEM_EMBEDDING_API_KEY_CMD", "OGMEM_EMBEDDING_API_KEY_COMMAND"),
            label="embedding.api_key",
        )

        return cls(
            # LLM
            provider=_resolve(llm.get("provider"), "CONTEXTENGINE_PROVIDER", "mock"),
            openai_api_key=llm_api_key,
            openai_api_key_command=llm_api_key_command,
            openai_base_url=base_url,
            openai_llm_model=_resolve(llm.get("model"), "OGMEM_LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=_resolve(llm.get("temperature"), "LLM_TEMPERATURE", 0.7, float),
            llm_max_tokens=_resolve(llm.get("max_tokens"), "LLM_MAX_TOKENS", 4096, int),
            llm_json_mode=_resolve_bool(llm.get("json_mode"), "LLM_JSON_MODE", False),
            # Embedding
            embedding_provider=_resolve(emb.get("provider"), "EMBEDDING_PROVIDER", None),
            openai_embedding_model=_resolve(
                emb.get("model"), "OGMEM_EMBEDDING_MODEL", "text-embedding-ada-002",
            ),
            openai_embedding_base_url=emb_base_url,
            openai_embedding_api_key=emb_api_key,
            openai_embedding_api_key_command=emb_api_key_command,
            embedding_multimodal=_resolve_bool(
                emb.get("multimodal"), "OGMEM_EMBEDDING_MULTIMODAL", False,
            ),
            st_model=_resolve(emb.get("st_model"), "ST_MODEL", "BAAI/bge-m3"),
            # Vector DB
            vector_db_type=_resolve(vdb.get("type"), "VECTOR_DB_TYPE", "chroma"),
            opengauss_connection_string=_resolve(
                vdb.get("connection_string"), "OPENGAUSS_CONNECTION_STRING", None,
            ),
            opengauss_dimension=_resolve(
                vdb.get("dimension"), "OPENGAUSS_DIMENSION", 1024, int,
            ),
            opengauss_table_name=_resolve(
                vdb.get("table_name"), "OPENGAUSS_TABLE_NAME", "vector_index",
            ),
            opengauss_pool_size=_resolve(
                vdb.get("pool_size"), "OPENGAUSS_POOL_SIZE", 5, int,
            ),
            chroma_persist_directory=_resolve(
                vdb.get("chroma_persist_dir"), "CHROMA_PERSIST_DIR", ".chroma_data",
            ),
            chroma_collection_name=_resolve(
                vdb.get("chroma_collection"), "CHROMA_COLLECTION", "contextengine",
            ),
            # Service
            http_port=_resolve(svc.get("http_port"), "OGMEM_HTTP_PORT", 8090, int),
            workers=_resolve(svc.get("workers"), "OGMEM_WORKERS", 2, int),
            http_ip_allowlist=_resolve_list(
                svc.get("http_ip_allowlist"), "OG_HTTP_IP_ALLOWLIST", [],
            ),
            http_ip_allowlist_trust_proxy=_resolve_bool(
                svc.get("http_ip_allowlist_trust_proxy"),
                "OG_HTTP_IP_ALLOWLIST_TRUST_PROXY",
                False,
            ),
            http_trusted_proxies=_resolve_list(
                svc.get("http_trusted_proxies"), "OG_HTTP_TRUSTED_PROXIES", [],
            ),
            # Storage backend
            storage_backend=_resolve(
                storage.get("backend"), "STORAGE_BACKEND", "agfs",
            ),
            sql_connection_string=_resolve(
                storage.get("connection_string"), "SQL_CONNECTION_STRING", None,
            ),
            sql_pool_size=_resolve(
                storage.get("pool_size"), "SQL_POOL_SIZE", 5, int,
            ),
            # AGFS
            agfs_base_url=_resolve(
                agfs.get("base_url"), "AGFS_BASE_URL", "http://127.0.0.1:1833",
            ),
            agfs_mount_prefix=_resolve(
                agfs.get("mount_prefix"), "AGFS_MOUNT_PREFIX", "/local",
            ),
            # Index
            index_interval=_resolve(idx.get("interval"), "INDEX_INTERVAL", 30, int),
            index_workers=_resolve(idx.get("workers"), "INDEX_WORKERS", 1, int),
            # Identity
            account_id=_resolve(ident.get("account_id"), "OG_ACCOUNT_ID", "acct-demo"),
            user_id=_resolve(ident.get("user_id"), "OG_USER_ID", "u-alice"),
            agent_id=_resolve(ident.get("agent_id"), "OG_AGENT_ID", "main"),
            role_control_enabled=_resolve_bool(
                auth.get("role_control_enabled"), "OG_ROLE_CONTROL_ENABLED", False,
            ),
            root_api_key=_resolve(auth.get("root_api_key"), "OG_ROOT_API_KEY", None),
            admin_api_keys=_resolve_list(
                auth.get("admin_api_keys"), "OG_ADMIN_API_KEYS", [],
            ),
            agent_shared_mode=_resolve(
                sharing.get("agent_shared_mode"), "OG_AGENT_SHARED_MODE", "off",
            ),
            agent_shared_list=_resolve_list(
                sharing.get("agent_shared_list"), "OG_AGENT_SHARED_LIST", [],
            ),
            # Memory extraction
            after_turn_threshold=_resolve(
                memory.get("after_turn_threshold"),
                "OGMEM_AFTER_TURN_THRESHOLD", 200, int,
            ),
            rolling_compress_enabled=_resolve_bool(
                memory.get("rolling_compress_enabled"),
                "OGMEM_ROLLING_COMPRESS_ENABLED", False,
            ),
            directory_summary_enabled=_resolve_bool(
                memory.get("directory_summary_enabled"),
                "OGMEM_DIRECTORY_SUMMARY_ENABLED", False,
            ),
            summary_max_chars=_resolve(
                memory.get("summary_max_chars"),
                "OGMEM_SUMMARY_MAX_CHARS", 4000, int,
            ),
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

    def effective_openai_api_key(self) -> str | None:
        """LLM API key, lazily resolving configured command only when needed."""
        if self.openai_api_key:
            return self.openai_api_key
        if self._cached_openai_api_key is not None:
            return self._cached_openai_api_key
        if self.openai_api_key_command:
            self._cached_openai_api_key = _run_secret_command(
                self.openai_api_key_command, label="llm.api_key"
            )
            return self._cached_openai_api_key
        return None

    def effective_embedding_api_key(self) -> str | None:
        """Embedding API key, falling back to the LLM API key."""
        if self.openai_embedding_api_key:
            return self.openai_embedding_api_key
        if self._cached_embedding_api_key is not None:
            return self._cached_embedding_api_key
        if self.openai_embedding_api_key_command:
            self._cached_embedding_api_key = _run_secret_command(
                self.openai_embedding_api_key_command, label="embedding.api_key"
            )
            return self._cached_embedding_api_key
        return self.effective_openai_api_key()

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
                if isinstance(val, list):
                    val = [v[:8] + "..." if isinstance(v, str) else v for v in val[:3]]
                    if len(getattr(self, f.name)) > 3:
                        val.append("...")
                elif isinstance(val, str):
                    val = val[:8] + "..."
            if "command" in f.name.lower() and val is not None:
                val = "<configured>"
            d[f.name] = val
        return d


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_global_config: OgMemConfig | None = None


def get_config(config_path: str | None = None) -> OgMemConfig:
    """Return the module-level singleton, creating it on first call.

    Configuration is process-scoped and cached after the first load. Runtime
    config changes require an explicit ``reset_config()`` in tests or a process
    restart in production.
    """
    global _global_config
    if _global_config is None:
        _global_config = OgMemConfig.load(config_path)
        logger.info("Global config loaded: %s", _global_config.dump_summary())
    return _global_config


def reset_config() -> None:
    """Reset the singleton (useful for tests)."""
    global _global_config
    _global_config = None
