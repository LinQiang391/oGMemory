"""Unified configuration for oG-Memory.

Loads all parameters from a single YAML file, falling back to environment
variables for any values not specified.  When no YAML file exists the
behaviour is identical to the old env-only mode (full backward compat).

Priority:  YAML value  >  environment variable  >  hard-coded default
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex

# Secret helpers use an allowlisted executable and never invoke a shell.
import subprocess  # nosec B404
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


_MODEL_CONFIG_OVERRIDE_ENV = "OGMEM_MODEL_CONFIG_OVERRIDES"
_MODEL_CONFIG_OVERRIDE_FIELDS = {
    "llmProvider": ("llm", "provider"),
    "llmApiKey": ("llm", "api_key"),
    "llmBaseUrl": ("llm", "base_url"),
    "llmModel": ("llm", "model"),
    "llmReasoningEffort": ("llm", "reasoning_effort"),
    "llmIsReasoningModel": ("llm", "is_reasoning_model"),
    "llmExtraBody": ("llm", "extra_body"),
    "embeddingProvider": ("embedding", "provider"),
    "embeddingApiKey": ("embedding", "api_key"),
    "embeddingBaseUrl": ("embedding", "base_url"),
    "embeddingModel": ("embedding", "model"),
}


def _apply_model_config_overrides(raw: dict) -> dict:
    """Apply explicit launcher model settings after YAML without exposing secrets."""
    encoded = os.environ.get(_MODEL_CONFIG_OVERRIDE_ENV)
    if not encoded:
        return raw
    try:
        overrides = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{_MODEL_CONFIG_OVERRIDE_ENV} must be valid JSON") from exc
    if not isinstance(overrides, dict):
        raise ValueError(f"{_MODEL_CONFIG_OVERRIDE_ENV} must be a JSON object")

    unknown = set(overrides) - set(_MODEL_CONFIG_OVERRIDE_FIELDS)
    if unknown:
        raise ValueError(
            f"{_MODEL_CONFIG_OVERRIDE_ENV} contains unsupported fields: "
            + ", ".join(sorted(unknown))
        )

    merged = dict(raw)
    for config_field, value in overrides.items():
        expected_type = (
            dict if config_field == "llmExtraBody"
            else bool if config_field == "llmIsReasoningModel"
            else str
        )
        if not isinstance(value, expected_type):
            raise ValueError(
                f"{_MODEL_CONFIG_OVERRIDE_ENV}.{config_field} must be "
                + ("an object" if expected_type is dict else "a boolean" if expected_type is bool else "a string")
            )
        section_name, yaml_key = _MODEL_CONFIG_OVERRIDE_FIELDS[config_field]
        section = dict(merged.get(section_name) or {})
        section[yaml_key] = value
        merged[section_name] = section
    return merged


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


def _resolve_optional_bool(yaml_val: Any, env_name: str) -> bool | None:
    value = yaml_val if yaml_val is not None else os.environ.get(env_name)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if not lowered:
        return None
    if lowered in ("1", "true", "yes"):
        return True
    if lowered in ("0", "false", "no"):
        return False
    raise ValueError(f"{env_name} must be a boolean")


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


def _resolve_json_object(
    yaml_val: Any, env_name: str, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve a mapping from YAML or a JSON-encoded environment variable."""
    value = yaml_val if yaml_val is not None else os.environ.get(env_name)
    if value is None:
        return dict(default or {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{env_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{env_name} must be a JSON object")
    return dict(value)


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
        _normalize_command_path(path) for path in ALLOWED_SECRET_HELPERS if str(path).strip()
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
            argv = tuple(
                part.strip() for part in shlex.split(text, posix=os.name != "nt") if part.strip()
            )
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


def _resolve_command_spec(
    yaml_val: Any, env_names: tuple[str, ...], *, label: str
) -> SecretCommandSpec | None:
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

    # argv[0] is allowlisted above, and shell execution is explicitly disabled.
    completed = subprocess.run(  # nosec B603
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
        raise ValueError(
            f"{label} command output exceeded max_output_bytes ({command_spec.max_output_bytes})"
        )

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
    llm_reasoning_effort: str | None = None
    llm_is_reasoning_model: bool | None = None
    llm_extra_body: dict[str, Any] = field(default_factory=dict)

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
    threads: int = 2
    http_ip_allowlist: list[str] = field(default_factory=list)
    http_ip_allowlist_trust_proxy: bool = False
    http_trusted_proxies: list[str] = field(default_factory=list)

    # -- MCP ------------------------------------------------------------------
    mcp_enabled: bool = True
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8091
    mcp_path: str = "/mcp"
    mcp_default_toolset: str = "read-only"
    mcp_enable_write_tools: bool = False
    mcp_enable_delete_tools: bool = False
    mcp_enable_runtime_tools: bool = False
    mcp_enable_diagnostic_tools: bool = False
    mcp_max_request_bytes: int = 1048576
    mcp_max_sse_connections: int = 64
    mcp_max_response_chars: int = 50000
    mcp_collection_limit: int = 50
    mcp_allowed_origins: list[str] = field(default_factory=list)
    mcp_allowed_hosts: list[str] = field(default_factory=lambda: ["127.0.0.1", "localhost"])

    # -- Storage backend -------------------------------------------------------
    storage_backend: str = "agfs"  # "agfs" | "sql"
    sql_connection_string: str | None = None  # PostgreSQL DSN (required when storage_backend=sql)
    sql_pool_size: int = 5
    trace_max_turn_events: int = 128

    # -- Agent skill sedimentation --------------------------------------------
    skill_sedimentation_enabled: bool = False
    skill_worked_examples_enabled: bool = False
    skill_min_evidence_turns: int = 1
    skill_min_distinct_sessions: int = 1
    skill_turn_score_threshold: float = 0.85
    skill_publish_score_threshold: float = 0.90
    skill_semantic_update_threshold: float = 0.92
    skill_semantic_hold_threshold: float = 0.80
    skill_scan_interval_seconds: int = 60
    skill_lease_seconds: int = 300
    skill_max_attempts: int = 5
    skill_candidate_limit: int = 200
    skill_catalog_scan_limit: int = 10000
    skill_max_evidence_turns: int = 20
    skill_max_trace_chars: int = 32768
    skill_max_prompt_chars: int = 65536
    skill_retry_base_seconds: int = 30
    skill_retry_cap_seconds: int = 3600
    skill_batch_size: int = 20
    skill_account_workers: int = 2
    skill_search_scan_limit: int = 1000
    skill_reviewer_provider: str | None = None
    skill_reviewer_model: str | None = None
    skill_sync_target: str | None = None

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
    extraction_mode: str = "eager"  # "eager" (prefetch + single-shot) or "lazy" (ReAct loop)
    after_turn_threshold: int = 200
    rolling_compress_enabled: bool = True
    rolling_compress_fallback_enabled: bool = False
    compact_prepare_token_ttl: int = 300
    directory_summary_enabled: bool = False
    summary_max_chars: int = 4000
    archive_max_count: int = 10
    archive_merge_threshold: int = 10
    prefetch_enabled: bool = False
    prefetch_top_k: int = 5
    session_state_bridge_enabled: bool = True
    session_state_sync_interval_turns: int = 1
    topic_detection_enabled: bool = False
    compression_quality_enabled: bool = False
    compression_quality_persist_metadata: bool = False

    # -- Large tool-result externalization (opt-in) ---------------------------
    tool_result_externalization_enabled: bool = False
    tool_result_first_view_enabled: bool = True
    tool_result_semantic_turn_mode: str = "user_message"
    tool_result_single_chars: int = 20_000
    tool_result_synopsis_max_chars: int = 2_000
    tool_result_stub_summary_turn_chars: int = 12_000
    tool_result_write_timeout_ms: int = 3_000
    tool_result_retry_attempts: int = 2
    tool_result_search_chunk_bytes: int = 16_384
    tool_result_search_overlap_bytes: int = 1_024
    tool_result_search_top_k: int = 5
    tool_result_search_max_artifacts: int = 20
    tool_result_search_deadline_ms: int = 2_000
    tool_result_read_default_bytes: int = 32_768
    tool_result_artifact_temp_grace_seconds: int = 600
    tool_result_http_max_response_chars: int = 100_000

    # -- Retrieval ------------------------------------------------------------
    retrieval_max_top_k: int = 50
    retrieval_default_top_k: int = 10
    retrieval_global_search_topk: int = 30
    retrieval_candidate_pool_enabled: bool = False
    retrieval_candidate_pool_multiplier: int = 3
    retrieval_min_candidate_pool_size: int = 15
    retrieval_result_ranker_strategy: str = "semantic"
    retrieval_global_result_rerank_enabled: bool = False
    retrieval_owner_scope_filter_enabled: bool = False
    retrieval_hybrid_enabled: bool = False
    retrieval_lexical_provider: str = "opengauss_native"
    retrieval_lexical_options: dict[str, Any] = field(default_factory=dict, repr=False)
    retrieval_vector_top_k: int = 100
    retrieval_lexical_top_k: int = 50
    retrieval_rrf_k: int = 60
    retrieval_semantic_weight: float = 1.0
    retrieval_lexical_weight: float = 1.0
    retrieval_backfill_page_size: int = 500

    # -- Cache -----------------------------------------------------------------
    enable_cache: bool = True
    cache_max_size: int = 1000

    # -- Perf monitoring -------------------------------------------------------
    # The legacy ``perf_*`` fields remain for backward compatibility with the
    # offline ``perf/`` analyzer driven from ``scripts/run_e2e_long_context_real.py``
    # — they are *not* required by anything inside the docker image.
    perf_enabled: bool = False
    perf_file_path: str | None = None
    perf_http_url: str | None = None
    perf_run_id: str | None = None
    perf_rate_card: str | None = None

    # -- Lazy secrets cache ----------------------------------------------------
    _cached_openai_api_key: str | None = field(default=None, init=False, repr=False)
    _cached_embedding_api_key: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 2 <= self.trace_max_turn_events <= 1024:
            raise ValueError("trace_max_turn_events must be between 2 and 1024")
        score_fields = {
            "skill_turn_score_threshold": self.skill_turn_score_threshold,
            "skill_publish_score_threshold": self.skill_publish_score_threshold,
            "skill_semantic_update_threshold": self.skill_semantic_update_threshold,
            "skill_semantic_hold_threshold": self.skill_semantic_hold_threshold,
        }
        for name, value in score_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.skill_semantic_hold_threshold >= self.skill_semantic_update_threshold:
            raise ValueError(
                "skill_semantic_hold_threshold must be lower than "
                "skill_semantic_update_threshold"
            )
        for name in (
            "skill_min_evidence_turns",
            "skill_min_distinct_sessions",
            "skill_scan_interval_seconds",
            "skill_lease_seconds",
            "skill_max_attempts",
            "skill_candidate_limit",
            "skill_catalog_scan_limit",
            "skill_max_evidence_turns",
            "skill_max_prompt_chars",
            "skill_retry_base_seconds",
            "skill_retry_cap_seconds",
            "skill_batch_size",
            "skill_account_workers",
            "skill_search_scan_limit",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.skill_max_trace_chars < 1024:
            raise ValueError("skill_max_trace_chars must be at least 1024")
        if self.skill_min_distinct_sessions > self.skill_min_evidence_turns:
            raise ValueError(
                "skill_min_distinct_sessions cannot exceed skill_min_evidence_turns"
            )
        if self.skill_candidate_limit > 1000:
            raise ValueError("skill_candidate_limit must not exceed 1000")
        if self.skill_catalog_scan_limit > 100000:
            raise ValueError("skill_catalog_scan_limit must not exceed 100000")
        if self.skill_catalog_scan_limit < self.skill_candidate_limit:
            raise ValueError("skill_catalog_scan_limit cannot be below skill_candidate_limit")
        if self.skill_account_workers > 32:
            raise ValueError("skill_account_workers must not exceed 32")
        if self.skill_search_scan_limit > 10000:
            raise ValueError("skill_search_scan_limit must not exceed 10000")
        if self.skill_sedimentation_enabled and self.skill_lease_seconds < 180:
            raise ValueError(
                "skill_lease_seconds must be at least 180 when sedimentation is enabled"
            )
        if self.skill_max_evidence_turns < self.skill_min_evidence_turns:
            raise ValueError("skill_max_evidence_turns cannot be below skill_min_evidence_turns")
        if self.skill_sedimentation_enabled and self.skill_max_evidence_turns < 2:
            raise ValueError(
                "skill_max_evidence_turns must be at least 2 when sedimentation is enabled"
            )
        if self.skill_retry_cap_seconds < self.skill_retry_base_seconds:
            raise ValueError("skill_retry_cap_seconds cannot be below skill_retry_base_seconds")
        if (self.skill_sedimentation_enabled
                and self.storage_backend not in {"agfs", "sql"}):
            raise ValueError(
                "skill sedimentation requires storage.backend=agfs or sql"
            )
        if (self.skill_sedimentation_enabled and self.storage_backend == "sql"
                and not self.sql_connection_string):
            raise ValueError("skill sedimentation requires storage.sql.connection_string")
        if (self.skill_sedimentation_enabled and self.storage_backend == "agfs"
                and self.workers != 1):
            raise ValueError(
                "AGFS skill sedimentation requires service.workers=1"
            )
        if self.retrieval_hybrid_enabled and self.storage_backend != "sql":
            raise ValueError(
                "hybrid retrieval requires storage.backend=sql for canonical backfill"
            )
        if self.retrieval_hybrid_enabled and not self.sql_connection_string:
            raise ValueError(
                "hybrid retrieval requires storage.connection_string for canonical backfill"
            )
        if not 1 <= self.retrieval_vector_top_k <= 1_000:
            raise ValueError("retrieval_vector_top_k must be between 1 and 1000")
        if not 1 <= self.retrieval_lexical_top_k <= 200:
            raise ValueError("retrieval_lexical_top_k must be between 1 and 200")
        if not 1 <= self.retrieval_rrf_k <= 10_000:
            raise ValueError("retrieval_rrf_k must be between 1 and 10000")
        for name in ("retrieval_semantic_weight", "retrieval_lexical_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"{name} must be finite and between 0 and 100")
        if self.retrieval_semantic_weight + self.retrieval_lexical_weight <= 0:
            raise ValueError("at least one retrieval fusion weight must be positive")
        if not 1 <= self.retrieval_backfill_page_size <= 10_000:
            raise ValueError("retrieval_backfill_page_size must be between 1 and 10000")
        if self.tool_result_externalization_enabled:
            for name in (
                "tool_result_single_chars",
                "tool_result_synopsis_max_chars",
                "tool_result_stub_summary_turn_chars",
                "tool_result_write_timeout_ms",
                "tool_result_retry_attempts",
                "tool_result_search_chunk_bytes",
                "tool_result_search_overlap_bytes",
                "tool_result_search_top_k",
                "tool_result_search_max_artifacts",
                "tool_result_search_deadline_ms",
                "tool_result_read_default_bytes",
                "tool_result_artifact_temp_grace_seconds",
                "tool_result_http_max_response_chars",
            ):
                if getattr(self, name) < 1:
                    raise ValueError(f"{name} must be positive")
            if self.tool_result_search_overlap_bytes >= self.tool_result_search_chunk_bytes:
                raise ValueError(
                    "tool_result_search_overlap_bytes must be below tool_result_search_chunk_bytes"
                )
        if self.tool_result_semantic_turn_mode != "user_message":
            raise ValueError(
                "tool_result_semantic_turn_mode currently supports only "
                "'user_message'"
            )
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
        path = config_path or os.environ.get("OGMEM_CONFIG") or DEFAULT_CONFIG_PATH
        raw = _apply_model_config_overrides(_load_yaml(path))

        llm = raw.get("llm") or {}
        emb = raw.get("embedding") or {}
        vdb = raw.get("vector_db") or {}
        svc = raw.get("service") or {}
        mcp = raw.get("mcp") or {}
        storage = raw.get("storage") or {}
        agfs = raw.get("agfs") or {}
        idx = raw.get("index") or {}
        ident = raw.get("identity") or {}
        auth = raw.get("auth") or {}
        sharing = raw.get("sharing") or {}
        memory = raw.get("memory") or {}
        retrieval = raw.get("retrieval") or {}
        cache = raw.get("cache") or {}
        perf = raw.get("perf") or {}
        sedimentation = (raw.get("skills") or {}).get("sedimentation") or {}
        tool_results = raw.get("tool_result_externalization") or {}

        base_url = _normalize_url(_resolve(llm.get("base_url"), "OGMEM_BASE_URL", None))
        emb_base_url = _normalize_url(
            _resolve(emb.get("base_url"), "OGMEM_EMBEDDING_BASE_URL", None)
        )
        llm_api_key = _first_non_none(llm.get("api_key"), os.environ.get("OGMEM_API_KEY"))
        emb_api_key = _first_non_none(emb.get("api_key"), os.environ.get("OGMEM_EMBEDDING_API_KEY"))
        llm_api_key_command = (
            None
            if llm_api_key is not None
            else _resolve_command_spec(
                llm.get("api_key_command", llm.get("api_key_cmd")),
                ("OGMEM_API_KEY_CMD", "OGMEM_API_KEY_COMMAND"),
                label="llm.api_key",
            )
        )
        emb_api_key_command = (
            None
            if emb_api_key is not None
            else _resolve_command_spec(
                emb.get("api_key_command", emb.get("api_key_cmd")),
                ("OGMEM_EMBEDDING_API_KEY_CMD", "OGMEM_EMBEDDING_API_KEY_COMMAND"),
                label="embedding.api_key",
            )
        )

        cfg = cls(
            # LLM
            provider=_resolve(llm.get("provider"), "CONTEXTENGINE_PROVIDER", "mock"),
            openai_api_key=llm_api_key,
            openai_api_key_command=llm_api_key_command,
            openai_base_url=base_url,
            openai_llm_model=_resolve(llm.get("model"), "OGMEM_LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=_resolve(llm.get("temperature"), "LLM_TEMPERATURE", 0.7, float),
            llm_max_tokens=_resolve(llm.get("max_tokens"), "LLM_MAX_TOKENS", 4096, int),
            llm_json_mode=_resolve_bool(llm.get("json_mode"), "LLM_JSON_MODE", False),
            llm_reasoning_effort=_resolve(
                llm.get("reasoning_effort"), "LLM_REASONING_EFFORT", None
            ),
            llm_is_reasoning_model=_resolve_optional_bool(
                llm.get("is_reasoning_model"), "LLM_IS_REASONING_MODEL"
            ),
            llm_extra_body=_resolve_json_object(
                llm.get("extra_body"), "LLM_EXTRA_BODY", {}
            ),
            # Embedding
            embedding_provider=_resolve(emb.get("provider"), "EMBEDDING_PROVIDER", None),
            openai_embedding_model=_resolve(
                emb.get("model"),
                "OGMEM_EMBEDDING_MODEL",
                "text-embedding-ada-002",
            ),
            openai_embedding_base_url=emb_base_url,
            openai_embedding_api_key=emb_api_key,
            openai_embedding_api_key_command=emb_api_key_command,
            embedding_multimodal=_resolve_bool(
                emb.get("multimodal"),
                "OGMEM_EMBEDDING_MULTIMODAL",
                False,
            ),
            st_model=_resolve(emb.get("st_model"), "ST_MODEL", "BAAI/bge-m3"),
            # Vector DB
            vector_db_type=_resolve(vdb.get("type"), "VECTOR_DB_TYPE", "chroma"),
            opengauss_connection_string=_resolve(
                vdb.get("connection_string"),
                "OPENGAUSS_CONNECTION_STRING",
                None,
            ),
            opengauss_dimension=_resolve(
                vdb.get("dimension"),
                "OPENGAUSS_DIMENSION",
                1024,
                int,
            ),
            opengauss_table_name=_resolve(
                vdb.get("table_name"),
                "OPENGAUSS_TABLE_NAME",
                "vector_index",
            ),
            opengauss_pool_size=_resolve(
                vdb.get("pool_size"),
                "OPENGAUSS_POOL_SIZE",
                5,
                int,
            ),
            chroma_persist_directory=_resolve(
                vdb.get("chroma_persist_dir"),
                "CHROMA_PERSIST_DIR",
                ".chroma_data",
            ),
            chroma_collection_name=_resolve(
                vdb.get("chroma_collection"),
                "CHROMA_COLLECTION",
                "contextengine",
            ),
            # Service
            http_port=_resolve(svc.get("http_port"), "OGMEM_HTTP_PORT", 8090, int),
            workers=_resolve(svc.get("workers"), "OGMEM_WORKERS", 2, int),
            threads=_resolve(
                svc.get("threads", svc.get("workers")),
                "OGMEM_THREADS",
                2,
                int,
            ),
            http_ip_allowlist=_resolve_list(
                svc.get("http_ip_allowlist"),
                "OG_HTTP_IP_ALLOWLIST",
                [],
            ),
            http_ip_allowlist_trust_proxy=_resolve_bool(
                svc.get("http_ip_allowlist_trust_proxy"),
                "OG_HTTP_IP_ALLOWLIST_TRUST_PROXY",
                False,
            ),
            http_trusted_proxies=_resolve_list(
                svc.get("http_trusted_proxies"),
                "OG_HTTP_TRUSTED_PROXIES",
                [],
            ),
            # MCP
            mcp_enabled=_resolve_bool(mcp.get("enabled"), "OGMEM_MCP_ENABLED", True),
            mcp_host=_resolve(mcp.get("host"), "OGMEM_MCP_HOST", "127.0.0.1"),
            mcp_port=_resolve(mcp.get("port"), "OGMEM_MCP_PORT", 8091, int),
            mcp_path=_resolve(mcp.get("path"), "OGMEM_MCP_PATH", "/mcp"),
            mcp_default_toolset=_resolve(
                mcp.get("default_toolset"),
                "OGMEM_MCP_DEFAULT_TOOLSET",
                "read-only",
            ),
            mcp_enable_write_tools=_resolve_bool(
                mcp.get("enable_write_tools"),
                "OGMEM_MCP_ENABLE_WRITE_TOOLS",
                False,
            ),
            mcp_enable_delete_tools=_resolve_bool(
                mcp.get("enable_delete_tools"),
                "OGMEM_MCP_ENABLE_DELETE_TOOLS",
                False,
            ),
            mcp_enable_runtime_tools=_resolve_bool(
                mcp.get("enable_runtime_tools"),
                "OGMEM_MCP_ENABLE_RUNTIME_TOOLS",
                False,
            ),
            mcp_enable_diagnostic_tools=_resolve_bool(
                mcp.get("enable_diagnostic_tools"),
                "OGMEM_MCP_ENABLE_DIAGNOSTIC_TOOLS",
                False,
            ),
            mcp_max_request_bytes=_resolve(
                mcp.get("max_request_bytes"),
                "OGMEM_MCP_MAX_REQUEST_BYTES",
                1048576,
                int,
            ),
            mcp_max_sse_connections=_resolve(
                mcp.get("max_sse_connections"),
                "OGMEM_MCP_MAX_SSE_CONNECTIONS",
                64,
                int,
            ),
            mcp_max_response_chars=_resolve(
                mcp.get("max_response_chars"),
                "OGMEM_MCP_MAX_RESPONSE_CHARS",
                50000,
                int,
            ),
            mcp_collection_limit=_resolve(
                mcp.get("collection_limit"),
                "OGMEM_MCP_COLLECTION_LIMIT",
                50,
                int,
            ),
            mcp_allowed_origins=_resolve_list(
                mcp.get("allowed_origins"),
                "OGMEM_MCP_ALLOWED_ORIGINS",
                [],
            ),
            mcp_allowed_hosts=_resolve_list(
                mcp.get("allowed_hosts"),
                "OGMEM_MCP_ALLOWED_HOSTS",
                ["127.0.0.1", "localhost"],
            ),
            # Storage backend
            storage_backend=_resolve(
                storage.get("backend"),
                "STORAGE_BACKEND",
                "agfs",
            ),
            sql_connection_string=_resolve(
                storage.get("connection_string"),
                "SQL_CONNECTION_STRING",
                None,
            ),
            sql_pool_size=_resolve(
                storage.get("pool_size"),
                "SQL_POOL_SIZE",
                5,
                int,
            ),
            trace_max_turn_events=_resolve(
                storage.get("trace_max_turn_events"),
                "OGMEM_TRACE_MAX_TURN_EVENTS",
                128,
                int,
            ),
            # Agent skill sedimentation
            skill_sedimentation_enabled=_resolve_bool(
                sedimentation.get("enabled"), "OGMEM_SKILL_SEDIMENTATION_ENABLED", False,
            ),
            skill_worked_examples_enabled=_resolve_bool(
                sedimentation.get("worked_examples_enabled"),
                "OGMEM_SKILL_WORKED_EXAMPLES_ENABLED",
                False,
            ),
            skill_min_evidence_turns=_resolve(
                sedimentation.get("min_evidence_turns"), "OGMEM_SKILL_MIN_EVIDENCE_TURNS", 1, int,
            ),
            skill_min_distinct_sessions=_resolve(
                sedimentation.get("min_distinct_sessions"), "OGMEM_SKILL_MIN_DISTINCT_SESSIONS", 1, int,
            ),
            skill_turn_score_threshold=_resolve(
                sedimentation.get("turn_score_threshold"), "OGMEM_SKILL_TURN_SCORE_THRESHOLD", 0.85, float,
            ),
            skill_publish_score_threshold=_resolve(
                sedimentation.get("publish_score_threshold"), "OGMEM_SKILL_PUBLISH_SCORE_THRESHOLD", 0.90, float,
            ),
            skill_semantic_update_threshold=_resolve(
                sedimentation.get("semantic_update_threshold"), "OGMEM_SKILL_SEMANTIC_UPDATE_THRESHOLD", 0.92, float,
            ),
            skill_semantic_hold_threshold=_resolve(
                sedimentation.get("semantic_hold_threshold"), "OGMEM_SKILL_SEMANTIC_HOLD_THRESHOLD", 0.80, float,
            ),
            skill_scan_interval_seconds=_resolve(
                sedimentation.get("scan_interval_seconds"), "OGMEM_SKILL_SCAN_INTERVAL_SECONDS", 60, int,
            ),
            skill_lease_seconds=_resolve(
                sedimentation.get("lease_seconds"), "OGMEM_SKILL_LEASE_SECONDS", 300, int,
            ),
            skill_max_attempts=_resolve(
                sedimentation.get("max_attempts"), "OGMEM_SKILL_MAX_ATTEMPTS", 5, int,
            ),
            skill_candidate_limit=_resolve(
                sedimentation.get("candidate_limit"), "OGMEM_SKILL_CANDIDATE_LIMIT", 200, int,
            ),
            skill_catalog_scan_limit=_resolve(
                sedimentation.get("catalog_scan_limit"),
                "OGMEM_SKILL_CATALOG_SCAN_LIMIT",
                10000,
                int,
            ),
            skill_max_evidence_turns=_resolve(
                sedimentation.get("max_evidence_turns"), "OGMEM_SKILL_MAX_EVIDENCE_TURNS", 20, int,
            ),
            skill_max_trace_chars=_resolve(
                sedimentation.get("max_trace_chars"), "OGMEM_SKILL_MAX_TRACE_CHARS", 32768, int,
            ),
            skill_max_prompt_chars=_resolve(
                sedimentation.get("max_prompt_chars"), "OGMEM_SKILL_MAX_PROMPT_CHARS", 65536, int,
            ),
            skill_retry_base_seconds=_resolve(
                sedimentation.get("retry_base_seconds"), "OGMEM_SKILL_RETRY_BASE_SECONDS", 30, int,
            ),
            skill_retry_cap_seconds=_resolve(
                sedimentation.get("retry_cap_seconds"), "OGMEM_SKILL_RETRY_CAP_SECONDS", 3600, int,
            ),
            skill_batch_size=_resolve(
                sedimentation.get("batch_size"), "OGMEM_SKILL_BATCH_SIZE", 20, int,
            ),
            skill_account_workers=_resolve(
                sedimentation.get("account_workers"), "OGMEM_SKILL_ACCOUNT_WORKERS", 2, int,
            ),
            skill_search_scan_limit=_resolve(
                sedimentation.get("search_scan_limit"), "OGMEM_SKILL_SEARCH_SCAN_LIMIT", 1000, int,
            ),
            skill_reviewer_provider=_resolve(
                sedimentation.get("reviewer_provider"), "OGMEM_SKILL_REVIEWER_PROVIDER", None,
            ),
            skill_reviewer_model=_resolve(
                sedimentation.get("reviewer_model"), "OGMEM_SKILL_REVIEWER_MODEL", None,
            ),
            skill_sync_target=_resolve(
                sedimentation.get("sync_target"), "OGMEM_SKILL_SYNC_TARGET", None,
            ),
            # AGFS
            agfs_base_url=_resolve(
                agfs.get("base_url"),
                "AGFS_BASE_URL",
                "http://127.0.0.1:1833",
            ),
            agfs_mount_prefix=_resolve(
                agfs.get("mount_prefix"),
                "AGFS_MOUNT_PREFIX",
                "/local",
            ),
            # Index
            index_interval=_resolve(idx.get("interval"), "INDEX_INTERVAL", 30, int),
            index_workers=_resolve(idx.get("workers"), "INDEX_WORKERS", 1, int),
            # Identity
            account_id=_resolve(ident.get("account_id"), "OG_ACCOUNT_ID", "acct-demo"),
            user_id=_resolve(ident.get("user_id"), "OG_USER_ID", "u-alice"),
            agent_id=_resolve(ident.get("agent_id"), "OG_AGENT_ID", "main"),
            role_control_enabled=_resolve_bool(
                auth.get("role_control_enabled"),
                "OG_ROLE_CONTROL_ENABLED",
                False,
            ),
            root_api_key=_resolve(auth.get("root_api_key"), "OG_ROOT_API_KEY", None),
            admin_api_keys=_resolve_list(
                auth.get("admin_api_keys"),
                "OG_ADMIN_API_KEYS",
                [],
            ),
            agent_shared_mode=_resolve(
                sharing.get("agent_shared_mode"),
                "OG_AGENT_SHARED_MODE",
                "off",
            ),
            agent_shared_list=_resolve_list(
                sharing.get("agent_shared_list"),
                "OG_AGENT_SHARED_LIST",
                [],
            ),
            # Memory extraction
            extraction_mode=_resolve(
                memory.get("extraction", {}).get("mode"),
                "OGMEM_EXTRACTION_MODE",
                "eager",
            ),
            after_turn_threshold=_resolve(
                memory.get("after_turn_threshold"),
                "OGMEM_AFTER_TURN_THRESHOLD",
                200,
                int,
            ),
            rolling_compress_enabled=_resolve_bool(
                memory.get("rolling_compress_enabled"),
                "OGMEM_ROLLING_COMPRESS_ENABLED",
                True,
            ),
            rolling_compress_fallback_enabled=_resolve_bool(
                memory.get("rolling_compress_fallback_enabled"),
                "OGMEM_ROLLING_COMPRESS_FALLBACK_ENABLED",
                False,
            ),
            compact_prepare_token_ttl=_resolve(
                memory.get("compact_prepare_token_ttl"),
                "OGMEM_COMPACT_PREPARE_TOKEN_TTL",
                300,
                int,
            ),
            directory_summary_enabled=_resolve_bool(
                memory.get("directory_summary_enabled"),
                "OGMEM_DIRECTORY_SUMMARY_ENABLED",
                False,
            ),
            summary_max_chars=_resolve(
                memory.get("summary_max_chars"),
                "OGMEM_SUMMARY_MAX_CHARS",
                4000,
                int,
            ),
            archive_max_count=_resolve(
                memory.get("archive_max_count"),
                "OGMEM_ARCHIVE_MAX_COUNT",
                10,
                int,
            ),
            archive_merge_threshold=_resolve(
                memory.get("archive_merge_threshold"),
                "OGMEM_ARCHIVE_MERGE_THRESHOLD",
                10,
                int,
            ),
            prefetch_enabled=_resolve_bool(
                memory.get("prefetch_enabled"),
                "OGMEM_PREFETCH_ENABLED",
                False,
            ),
            prefetch_top_k=_resolve(
                memory.get("prefetch_top_k"),
                "OGMEM_PREFETCH_TOP_K",
                5,
                int,
            ),
            session_state_bridge_enabled=_resolve_bool(
                memory.get("session_state_bridge_enabled"),
                "OGMEM_SESSION_STATE_BRIDGE_ENABLED",
                True,
            ),
            session_state_sync_interval_turns=_resolve(
                memory.get("session_state_sync_interval_turns"),
                "OGMEM_SESSION_STATE_SYNC_INTERVAL_TURNS",
                1,
                int,
            ),
            topic_detection_enabled=_resolve_bool(
                memory.get("topic_detection_enabled"),
                "OGMEM_TOPIC_DETECTION_ENABLED",
                False,
            ),
            compression_quality_enabled=_resolve_bool(
                memory.get("compression_quality_enabled"),
                "OGMEM_COMPRESSION_QUALITY_ENABLED",
                False,
            ),
            compression_quality_persist_metadata=_resolve_bool(
                memory.get("compression_quality_persist_metadata"),
                "OGMEM_COMPRESSION_QUALITY_PERSIST_METADATA",
                False,
            ),
            # Large tool-result externalization
            tool_result_externalization_enabled=_resolve_bool(
                tool_results.get("enabled"), "OGMEM_TOOL_RESULT_EXTERNALIZATION_ENABLED", False,
            ),
            tool_result_first_view_enabled=_resolve_bool(
                tool_results.get("first_view_enabled"),
                "OGMEM_TOOL_RESULT_FIRST_VIEW_ENABLED",
                True,
            ),
            tool_result_semantic_turn_mode=_resolve(
                tool_results.get("semantic_turn_mode"),
                "OGMEM_TOOL_RESULT_SEMANTIC_TURN_MODE",
                "user_message",
            ),
            tool_result_single_chars=_resolve(
                tool_results.get("single_result_chars"), "OGMEM_TOOL_RESULT_SINGLE_CHARS", 20_000, int,
            ),
            tool_result_synopsis_max_chars=_resolve(
                tool_results.get("synopsis_max_chars"), "OGMEM_TOOL_RESULT_SYNOPSIS_MAX_CHARS", 2_000, int,
            ),
            tool_result_stub_summary_turn_chars=_resolve(
                tool_results.get("stub_summary_turn_chars"),
                "OGMEM_TOOL_RESULT_STUB_SUMMARY_TURN_CHARS",
                12_000,
                int,
            ),
            tool_result_write_timeout_ms=_resolve(
                tool_results.get("write_timeout_ms"), "OGMEM_TOOL_RESULT_WRITE_TIMEOUT_MS", 3_000, int,
            ),
            tool_result_retry_attempts=_resolve(
                tool_results.get("retry_attempts"), "OGMEM_TOOL_RESULT_RETRY_ATTEMPTS", 2, int,
            ),
            tool_result_search_chunk_bytes=_resolve(
                tool_results.get("search_chunk_bytes"), "OGMEM_TOOL_RESULT_SEARCH_CHUNK_BYTES", 16_384, int,
            ),
            tool_result_search_overlap_bytes=_resolve(
                tool_results.get("search_overlap_bytes"), "OGMEM_TOOL_RESULT_SEARCH_OVERLAP_BYTES", 1_024, int,
            ),
            tool_result_search_top_k=_resolve(
                tool_results.get("search_top_k"), "OGMEM_TOOL_RESULT_SEARCH_TOP_K", 5, int,
            ),
            tool_result_search_max_artifacts=_resolve(
                tool_results.get("search_max_artifacts"), "OGMEM_TOOL_RESULT_SEARCH_MAX_ARTIFACTS", 20, int,
            ),
            tool_result_search_deadline_ms=_resolve(
                tool_results.get("search_deadline_ms"), "OGMEM_TOOL_RESULT_SEARCH_DEADLINE_MS", 2_000, int,
            ),
            tool_result_read_default_bytes=_resolve(
                tool_results.get("read_default_bytes"), "OGMEM_TOOL_RESULT_READ_DEFAULT_BYTES", 32_768, int,
            ),
            tool_result_artifact_temp_grace_seconds=_resolve(
                tool_results.get("artifact_temp_grace_seconds"), "OGMEM_TOOL_RESULT_TEMP_GRACE_SECONDS", 600, int,
            ),
            tool_result_http_max_response_chars=_resolve(
                tool_results.get("http_max_response_chars"),
                "OGMEM_TOOL_RESULT_HTTP_MAX_RESPONSE_CHARS",
                100_000,
                int,
            ),
            # Retrieval
            retrieval_max_top_k=_resolve(
                retrieval.get("max_top_k"),
                "OGMEM_RETRIEVAL_MAX_TOP_K",
                50,
                int,
            ),
            retrieval_default_top_k=_resolve(
                retrieval.get("default_top_k"),
                "OGMEM_RETRIEVAL_DEFAULT_TOP_K",
                10,
                int,
            ),
            retrieval_global_search_topk=_resolve(
                retrieval.get("global_search_topk"),
                "OGMEM_RETRIEVAL_GLOBAL_SEARCH_TOPK",
                30,
                int,
            ),
            retrieval_candidate_pool_enabled=_resolve_bool(
                retrieval.get("candidate_pool_enabled"),
                "OGMEM_RETRIEVAL_CANDIDATE_POOL_ENABLED",
                False,
            ),
            retrieval_candidate_pool_multiplier=_resolve(
                retrieval.get("candidate_pool_multiplier"),
                "OGMEM_RETRIEVAL_CANDIDATE_POOL_MULTIPLIER",
                3,
                int,
            ),
            retrieval_min_candidate_pool_size=_resolve(
                retrieval.get("min_candidate_pool_size"),
                "OGMEM_RETRIEVAL_MIN_CANDIDATE_POOL_SIZE",
                15,
                int,
            ),
            retrieval_result_ranker_strategy=_resolve(
                retrieval.get("result_ranker_strategy"),
                "OGMEM_RETRIEVAL_RESULT_RANKER_STRATEGY",
                "semantic",
            ),
            retrieval_global_result_rerank_enabled=_resolve_bool(
                retrieval.get("global_result_rerank_enabled"),
                "OGMEM_RETRIEVAL_GLOBAL_RESULT_RERANK_ENABLED",
                False,
            ),
            retrieval_owner_scope_filter_enabled=_resolve_bool(
                retrieval.get("owner_scope_filter_enabled"),
                "OGMEM_RETRIEVAL_OWNER_SCOPE_FILTER_ENABLED",
                False,
            ),
            retrieval_hybrid_enabled=_resolve_bool(
                retrieval.get("hybrid_enabled"),
                "OGMEM_RETRIEVAL_HYBRID_ENABLED",
                False,
            ),
            retrieval_lexical_provider=_resolve(
                (retrieval.get("lexical") or {}).get("provider")
                if isinstance(retrieval.get("lexical"), dict)
                else retrieval.get("lexical_provider"),
                "OGMEM_RETRIEVAL_LEXICAL_PROVIDER",
                "opengauss_native",
            ),
            retrieval_lexical_options=(
                dict((retrieval.get("lexical") or {}).get("options") or {})
                if isinstance(retrieval.get("lexical"), dict)
                else {}
            ),
            retrieval_vector_top_k=_resolve(
                retrieval.get("vector_top_k"),
                "OGMEM_RETRIEVAL_VECTOR_TOP_K",
                100,
                int,
            ),
            retrieval_lexical_top_k=_resolve(
                retrieval.get("lexical_top_k"),
                "OGMEM_RETRIEVAL_LEXICAL_TOP_K",
                50,
                int,
            ),
            retrieval_rrf_k=_resolve(
                retrieval.get("rrf_k"), "OGMEM_RETRIEVAL_RRF_K", 60, int
            ),
            retrieval_semantic_weight=_resolve(
                retrieval.get("semantic_weight"),
                "OGMEM_RETRIEVAL_SEMANTIC_WEIGHT",
                1.0,
                float,
            ),
            retrieval_lexical_weight=_resolve(
                retrieval.get("lexical_weight"),
                "OGMEM_RETRIEVAL_LEXICAL_WEIGHT",
                1.0,
                float,
            ),
            retrieval_backfill_page_size=_resolve(
                retrieval.get("backfill_page_size"),
                "OGMEM_RETRIEVAL_BACKFILL_PAGE_SIZE",
                500,
                int,
            ),
            # Cache
            enable_cache=_resolve_bool(cache.get("enabled"), "OGMEM_CACHE_ENABLED", True),
            cache_max_size=_resolve(
                cache.get("max_size"),
                "OGMEM_CACHE_MAX_SIZE",
                1000,
                int,
            ),
            # Perf (retained for offline analyzer / back-compat)
            perf_enabled=_resolve_bool(perf.get("enabled"), "OGMEM_PERF_ENABLED", False),
            perf_file_path=_resolve(perf.get("file_path"), "OGMEM_PERF_OUT", None),
            perf_http_url=_resolve(perf.get("http_url"), "OGMEM_PERF_HTTP_URL", None),
            perf_run_id=_resolve(perf.get("run_id"), "OGMEM_PERF_RUN_ID", None),
            perf_rate_card=_resolve(perf.get("rate_card"), "OGMEM_PERF_RATE_CARD", None),
        )
        cfg._export_perf_env()
        return cfg

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def _export_perf_env(self) -> None:
        """Propagate YAML-sourced perf settings to OGMEM_PERF_* env vars.

        ``perf.get_recorder()`` auto-enables based on env vars only, so YAML
        configuration must be bridged here. We never overwrite a variable the
        caller already set — explicit env > YAML.
        """
        if not self.perf_enabled:
            return
        os.environ.setdefault("OGMEM_PERF_ENABLED", "1")
        if self.perf_file_path:
            os.environ.setdefault("OGMEM_PERF_OUT", self.perf_file_path)
        if self.perf_http_url:
            os.environ.setdefault("OGMEM_PERF_HTTP_URL", self.perf_http_url)
        if self.perf_run_id:
            os.environ.setdefault("OGMEM_PERF_RUN_ID", self.perf_run_id)
        if self.perf_rate_card:
            os.environ.setdefault("OGMEM_PERF_RATE_CARD", self.perf_rate_card)

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
        def sanitize(name: str, value: Any) -> Any:
            lowered = name.casefold()
            credential_name = lowered.endswith("token") or any(
                marker in lowered
                for marker in ("key", "password", "secret", "connection_string")
            )
            if value and credential_name:
                return "<configured>"
            if "command" in lowered and value is not None:
                return "<configured>"
            if isinstance(value, dict):
                return {str(key): sanitize(str(key), item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [sanitize(name, item) for item in value]
            return value

        d: dict[str, Any] = {}
        for f in fields(self):
            d[f.name] = sanitize(f.name, getattr(self, f.name))
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
