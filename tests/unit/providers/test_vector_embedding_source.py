"""Contract tests for reading vectors already stored by an index backend."""

from unittest.mock import Mock

from core.models import IndexRecord
from providers.vector_index.chroma_index import ChromaVectorIndex
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.vector_index.opengauss_index import OpenGaussVectorIndex


def test_in_memory_get_embeddings_applies_scope_and_level() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.upsert(
        [
            IndexRecord(
                id="wanted",
                uri="ctx://tenant/memory/wanted",
                level=0,
                text="wanted",
                filters={"account_id": "tenant"},
                metadata={"_embedding": [0.25, 0.75]},
            ),
            IndexRecord(
                id="other-account",
                uri="ctx://other/memory/hidden",
                level=0,
                text="hidden",
                filters={"account_id": "other"},
                metadata={"_embedding": [1.0, 0.0]},
            ),
            IndexRecord(
                id="other-level",
                uri="ctx://tenant/memory/overview",
                level=1,
                text="overview",
                filters={"account_id": "tenant"},
                metadata={"_embedding": [0.0, 1.0]},
            ),
        ]
    )

    result = index.get_embeddings(
        [
            "ctx://tenant/memory/wanted",
            "ctx://other/memory/hidden",
            "ctx://tenant/memory/overview",
        ],
        {"account_id": "tenant"},
    )

    assert result == {"ctx://tenant/memory/wanted": [0.25, 0.75]}


def test_chroma_get_embeddings_applies_scope_and_requested_uris() -> None:
    index = ChromaVectorIndex.__new__(ChromaVectorIndex)
    index._collection = Mock()
    index._collection.get.return_value = {
        "embeddings": [[0.1, 0.9], [0.8, 0.2]],
        "metadatas": [
            {"uri": "ctx://tenant/memory/wanted"},
            {"uri": "ctx://tenant/memory/not-requested"},
        ],
    }

    result = index.get_embeddings(
        ["ctx://tenant/memory/wanted"],
        {"account_id": "tenant"},
    )

    assert result == {"ctx://tenant/memory/wanted": [0.1, 0.9]}
    index._collection.get.assert_called_once_with(
        where={
            "$and": [
                {"account_id": "tenant"},
                {"level": 0},
            ]
        },
        include=["embeddings", "metadatas"],
    )


def test_opengauss_get_embeddings_applies_scope_and_parses_vectors() -> None:
    cursor = Mock()
    cursor.__enter__ = Mock(return_value=cursor)
    cursor.__exit__ = Mock(return_value=None)
    cursor.fetchall.return_value = [
        ("ctx://tenant/memory/one", "[0.125,0.875]"),
    ]
    connection = Mock()
    connection.cursor.return_value = cursor
    index = OpenGaussVectorIndex.__new__(OpenGaussVectorIndex)
    index._table_name = "vector_index"
    index._get_connection = Mock(return_value=connection)
    index._return_connection = Mock()

    result = index.get_embeddings(
        ["ctx://tenant/memory/one"],
        {"account_id": "tenant", "owner_space": "user:one"},
    )

    assert result == {"ctx://tenant/memory/one": [0.125, 0.875]}
    sql, params = cursor.execute.call_args.args
    assert "uri=ANY(%(uris)s)" in sql
    assert params == {
        "f_account_id": "tenant",
        "f_owner_space": "user:one",
        "f_level": 0,
        "uris": ["ctx://tenant/memory/one"],
    }
    index._return_connection.assert_called_once_with(connection)
