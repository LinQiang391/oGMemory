"""Database setup helpers for ogmem onboard."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_postgresql(connection_string: str) -> bool:
    """Check if PostgreSQL is reachable with the given connection string."""
    try:
        import psycopg2
        conn = psycopg2.connect(connection_string)
        conn.close()
        return True
    except Exception as e:
        logger.debug("PostgreSQL check failed: %s", e)
        return False


def check_opengauss_compatibility(connection_string: str) -> bool:
    """Validate SQL compatibility before schema initialization or startup."""
    try:
        import psycopg2
        from fs.sql_adapter.opengauss_dialect import OpenGaussDialect

        conn = psycopg2.connect(connection_string)
        try:
            with conn.cursor() as cursor:
                OpenGaussDialect().validate_connection(cursor)
        finally:
            conn.close()
        return True
    except Exception as exc:
        logger.error("openGauss compatibility check failed: %s", exc)
        return False


def init_schema(connection_string: str, dialect: str = "postgresql") -> bool:
    """Initialize the oG-Memory schema in the database.

    Uses the canonical ensure_schema from fs/sql_adapter/schema.py.
    Idempotent — safe to run multiple times.
    """
    try:
        import psycopg2

        from fs.sql_adapter.schema import ensure_schema

        conn = psycopg2.connect(connection_string)
        try:
            ensure_schema(conn, dialect)
            conn.commit()
        finally:
            conn.close()
        return True
    except ImportError as e:
        logger.error("Missing dependency: %s (install psycopg2-binary)", e)
        return False
    except Exception as e:
        logger.error("Schema init failed: %s", e)
        return False
