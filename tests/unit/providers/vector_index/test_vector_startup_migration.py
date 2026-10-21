"""Startup is schema-only; legacy row updates require an explicit migration."""
from unittest.mock import MagicMock

import pytest

from providers.vector_index.opengauss_index import OpenGaussVectorIndex


@pytest.fixture
def index():
    conn = MagicMock()
    vi = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
    vi._table_name, vi._dimension = 'vector_index', 3
    vi._get_connection = lambda: conn
    vi._return_connection = MagicMock()
    return vi, conn, conn.cursor.return_value.__enter__.return_value


def test_repeated_startup_never_backfills_or_reads_context_nodes(index):
    vi, conn, cur = index
    for _ in range(3):
        vi._ensure_table()
    statements = [call.args[0].strip().upper() for call in cur.execute.call_args_list]
    assert not any(sql.startswith('UPDATE') for sql in statements)
    assert not any('CONTEXT_NODES' in sql for sql in statements)


def test_completed_migration_does_not_scan_rows(index):
    vi, conn, cur = index
    cur.fetchone.return_value = (1,)
    assert vi.migrate_legacy_rows() is False
    assert not any(call.args[0].strip().startswith('UPDATE') for call in cur.execute.call_args_list)
    conn.commit.assert_called_once()


def test_migration_rolls_back_marker_on_failure(index):
    vi, conn, cur = index
    cur.fetchone.return_value = None
    def execute(sql, *args):
        if sql.strip().startswith('UPDATE'):
            raise RuntimeError('interrupted')
    cur.execute.side_effect = execute
    with pytest.raises(RuntimeError, match='interrupted'):
        vi.migrate_legacy_rows()
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()
    assert not any('INSERT INTO ogmem_vector_migrations' in call.args[0]
                   for call in cur.execute.call_args_list)
