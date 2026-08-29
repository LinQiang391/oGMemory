"""Canonical schema application — single source of truth.

All SQL DDL lives in ``db_schema.sql`` (next to this file).  Every store
calls :func:`ensure_schema` from its ``_ensure_table`` method so that
the schema is applied exactly once, idempotently, regardless of which
store is instantiated first.
"""

import logging
from pathlib import Path

from fs.sql_adapter.dialect import SQLDialect, resolve_sql_dialect

logger = logging.getLogger(__name__)

_SCHEMA_SQL = (Path(__file__).parent / "db_schema.sql").read_text(encoding="utf-8")

# Process-level guard: ensure_schema only needs to run once.
# Without this, concurrent SQLContextFS instantiations all run DDL
# simultaneously, causing PG deadlocks (ACCESS EXCLUSIVE lock contention).
_schema_ensured: set[str] = set()


def ensure_schema(conn, dialect: str | SQLDialect = "postgresql") -> None:
    """Apply the canonical schema (idempotent).

    Runs the full ``db_schema.sql`` within a single psycopg2 transaction.
    Uses libpq's ``PQexec`` via ``autocommit`` mode to correctly handle
    multi-statement SQL with ``DO $$ ... $$`` blocks.

    Guarded by a process-level flag so DDL only executes once, even when
    multiple SQLContextFS instances are created concurrently.
    """
    adapter = resolve_sql_dialect(dialect)
    if adapter.name in _schema_ensured:
        return
    was_autocommit = conn.autocommit
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            adapter.validate_connection(cur)
            cur.execute(adapter.prepare_schema(_SCHEMA_SQL))
        _schema_ensured.add(adapter.name)
        logger.info("Schema ensured from db_schema.sql (dialect=%s)", adapter.name)
    finally:
        conn.autocommit = was_autocommit
