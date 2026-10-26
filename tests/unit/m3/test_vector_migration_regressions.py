"""Regression coverage for the September vector-storage migration."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from m3.index_episodic import build_index, _upsert_to_vector_index
from m3.pipeline.runner import _build_vector_index
from providers.vector_index.opengauss_index import OpenGaussVectorIndex


def test_built_embeddings_reach_sql_scope_columns():
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1, 0.2]]
    records = build_index([{
        'video_id': 'bedroom_03', 'clip_id': 0,
        'extracted_node_id': 'episode-0', 'description': 'In the bedroom.',
    }], embedder=embedder)
    index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
    index._table_name = 'test_vectors'
    conn = MagicMock()
    index._get_connection = lambda: conn
    index._return_connection = lambda c: None
    assert _upsert_to_vector_index(records, index, SimpleNamespace(account_id='m3bench')) == 1
    args = conn.cursor.return_value.__enter__.return_value.execute.call_args.args[1]
    # Both UPDATE and INSERT must persist the SQL columns used by QA filters.
    assert args[13:16] == ('bedroom_03', 'episodic_memory_generator', 0)
    assert args[29:32] == ('bedroom_03', 'episodic_memory_generator', 0)
    conn.commit.assert_called_once()


@pytest.mark.parametrize('vector_dsn', ['postgresql://vector-db/vector', None])
def test_pipeline_never_uses_graph_dsn_for_vectors(monkeypatch, vector_dsn):
    runtime = SimpleNamespace(vector_index_connection_string=vector_dsn)
    monkeypatch.setattr('m3.config.load_m3_runtime_config', lambda _: runtime)
    builder = MagicMock()
    monkeypatch.setattr('m3.vector_index.build_m3_vector_index', builder)
    ctx = SimpleNamespace(config_path='config.yaml', connection_string='postgresql://storage-db/graph')
    result = _build_vector_index(ctx)
    if vector_dsn:
        builder.assert_called_once_with(runtime)
        assert result is builder.return_value
    else:
        builder.assert_not_called()
        assert result is None
