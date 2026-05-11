"""Canonical schema application — single source of truth.

All SQL DDL lives in ``db_schema.sql`` (next to this file).  Every store
calls :func:`ensure_schema` from its ``_ensure_table`` method so that
the schema is applied exactly once, idempotently, regardless of which
store is instantiated first.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_SQL = (Path(__file__).parent / "db_schema.sql").read_text()

# Process-level guard: ensure_schema only needs to run once.
# Without this, concurrent SQLContextFS instantiations all run DDL
# simultaneously, causing PG deadlocks (ACCESS EXCLUSIVE lock contention).
_schema_ensured = False


def ensure_schema(conn) -> None:
    """Apply the canonical schema (idempotent).

    Runs the full ``db_schema.sql`` within a single psycopg2 transaction.
    Uses libpq's ``PQexec`` via ``autocommit`` mode to correctly handle
    multi-statement SQL with ``DO $$ ... $$`` blocks.

    Guarded by a process-level flag so DDL only executes once, even when
    multiple SQLContextFS instances are created concurrently.
    """
    global _schema_ensured
    if _schema_ensured:
        return
    was_autocommit = conn.autocommit
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        _schema_ensured = True
        logger.info("Schema ensured from db_schema.sql")
    finally:
        conn.autocommit = was_autocommit
