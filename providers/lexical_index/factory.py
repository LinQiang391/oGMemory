"""Construction for the database-native openGauss lexical provider."""

from __future__ import annotations

from typing import Any

from core.lexical import DEFAULT_MAX_QUERY_TERMS, HybridLexicalIndex

_OPENGAUSS_NATIVE_OPTIONS = frozenset(
    {
        "connection_string",
        "table_name",
        "pool_size",
        "max_write_batch_size",
        "max_query_terms",
        "max_query_chars",
        "rebuild_lease_seconds",
        "max_document_chars",
        "bm25_topk",
        "bm25_k1",
        "bm25_b",
        "bm25_taat",
        "query_timeout_ms",
        "connect_timeout_seconds",
        "operation_timeout_ms",
    }
)


def _validated_options(config) -> dict[str, Any]:
    options = dict(getattr(config, "retrieval_lexical_options", {}) or {})
    unknown = set(options) - _OPENGAUSS_NATIVE_OPTIONS
    if unknown:
        raise ValueError(f"unknown openGauss lexical provider options: {sorted(unknown)}")
    return options


def create_lexical_index(config) -> HybridLexicalIndex:
    """Create the explicitly configured openGauss-native lexical index."""

    name = str(config.retrieval_lexical_provider).strip().casefold()
    if name != "opengauss_native":
        raise ValueError(
            f"unsupported lexical provider {name!r}; this integration supports "
            "only 'opengauss_native'"
        )
    options = _validated_options(config)
    options["connection_string"] = (
        options.get("connection_string")
        or getattr(config, "opengauss_connection_string", None)
        or getattr(config, "sql_connection_string", None)
        or ""
    )
    if not options["connection_string"]:
        raise ValueError("openGauss lexical retrieval requires an openGauss connection")

    from providers.lexical_index.native_sql_index import (
        DEFAULT_MAX_QUERY_CHARS,
        OpenGaussLexicalIndex,
    )
    from providers.lexical_index.sql_index import (
        DEFAULT_MAX_DOCUMENT_CHARS,
        DEFAULT_MAX_WRITE_BATCH_SIZE,
        DEFAULT_POOL_SIZE,
        DEFAULT_REBUILD_LEASE_SECONDS,
    )

    return OpenGaussLexicalIndex(
        connection_string=str(options.get("connection_string") or ""),
        table_name=str(options.get("table_name", "lexical_index_opengauss_native")),
        provider_name=name,
        pool_size=int(options.get("pool_size", DEFAULT_POOL_SIZE)),
        max_write_batch_size=int(
            options.get("max_write_batch_size", DEFAULT_MAX_WRITE_BATCH_SIZE)
        ),
        max_query_terms=int(options.get("max_query_terms", DEFAULT_MAX_QUERY_TERMS)),
        max_query_chars=int(options.get("max_query_chars", DEFAULT_MAX_QUERY_CHARS)),
        rebuild_lease_seconds=int(
            options.get("rebuild_lease_seconds", DEFAULT_REBUILD_LEASE_SECONDS)
        ),
        max_document_chars=int(
            options.get("max_document_chars", DEFAULT_MAX_DOCUMENT_CHARS)
        ),
        query_timeout_ms=int(options.get("query_timeout_ms", 5000)),
        connect_timeout_seconds=int(options.get("connect_timeout_seconds", 10)),
        operation_timeout_ms=int(options.get("operation_timeout_ms", 300000)),
        bm25_topk=int(options["bm25_topk"]) if "bm25_topk" in options else None,
        bm25_k1=float(options["bm25_k1"]) if "bm25_k1" in options else None,
        bm25_b=float(options["bm25_b"]) if "bm25_b" in options else None,
        bm25_taat=options.get("bm25_taat"),
    )
