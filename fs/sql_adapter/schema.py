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


def ensure_schema(conn) -> None:
    """Apply the canonical schema (idempotent).

    Runs the full ``db_schema.sql`` within a single psycopg2 transaction.
    The SQL file has no explicit ``BEGIN``/``COMMIT`` — this function
    commits the transaction.  All statements are idempotent
    (``IF NOT EXISTS`` / ``DO $$ EXCEPTION``).
    """
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()
    logger.info("Schema ensured from db_schema.sql")
