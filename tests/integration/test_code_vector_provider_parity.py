from __future__ import annotations

import os
import uuid
from collections.abc import Callable

import pytest

from core.models import ContextNode, IndexRecord, TypedQuery
from index.index_record_builder import build_index_records
from providers.vector_index.chroma_index import ChromaVectorIndex
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from providers.vector_index.opengauss_index import OpenGaussVectorIndex


def _node(account_id: str, slug: str, *, resource_type: str) -> ContextNode:
    return ContextNode(
        uri=f"ctx://{account_id}/resources/{slug}",
        context_type="RESOURCE",
        category="code" if resource_type == "code" else "document",
        level=0,
        owner_space=f"resource:{account_id}",
        abstract=f"{resource_type} abstract",
        overview="",
        content=f"{resource_type} content",
        metadata={
            "resource_type": resource_type,
            "resource_root_slug": "project",
            "root_fingerprint": "f" * 64,
            "file_key": uuid.uuid4().hex * 2,
            "file_slug": uuid.uuid4().hex,
            "relative_path": f"src/{slug}.py",
            "source_extension": ".py",
            "language": "python",
            "start_line": 1,
            "end_line": 1,
            "symbol": slug,
            "symbol_kind": "function",
            "signature": f"{slug}()",
            "chunk_hash": uuid.uuid4().hex * 2,
            "chunk_key": uuid.uuid4().hex * 2,
            "graph": {"calls": ["must_not_reach_vector_metadata"]},
        },
    )


def _records(account_id: str) -> tuple[list[IndexRecord], str]:
    code = build_index_records(_node(account_id, "code-hit", resource_type="code"))
    non_code = build_index_records(_node(account_id, "document-hit", resource_type="document"))
    for record in code:
        record.metadata["_embedding"] = [0.8, 0.2, 0.0]
    for record in non_code:
        record.metadata["_embedding"] = [1.0, 0.0, 0.0]
    return code + non_code, code[0].metadata["chunk_key"]


def _assert_provider_parity(index, account_id: str, expected_chunk_key: str) -> None:
    query = TypedQuery(
        text="code",
        context_type="RESOURCE",
        categories=[],
        account_id=account_id,
        resource_type="code",
        top_k=4,
    )
    if isinstance(index, ChromaVectorIndex):
        hits = index.search_by_vector(
            [1.0, 0.0, 0.0],
            {
                "account_id": account_id,
                "context_type": "RESOURCE",
                "resource_type": "code",
            },
            4,
        )
    else:
        hits = index.search(query, top_k=4)

    assert hits
    assert all(hit.metadata["resource_type"] == "code" for hit in hits)
    assert all("document-hit" not in hit.uri for hit in hits)
    assert any(hit.metadata["chunk_key"] == expected_chunk_key for hit in hits)
    assert all("graph" not in hit.metadata for hit in hits)


@pytest.mark.integration
@pytest.mark.parametrize("factory", [InMemoryVectorIndex])
def test_in_memory_code_prefilter_parity(
    factory: Callable[..., InMemoryVectorIndex],
):
    account_id = f"ogmemory-code-memory-{uuid.uuid4().hex[:12]}"
    records, chunk_key = _records(account_id)
    index = factory(dimension=3)
    index.upsert(records)

    _assert_provider_parity(index, account_id, chunk_key)


@pytest.mark.integration
def test_chroma_code_prefilter_parity(tmp_path):
    pytest.importorskip("chromadb", reason="chromadb is required for real Chroma parity")
    account_id = f"ogmemory-code-chroma-{uuid.uuid4().hex[:12]}"
    records, chunk_key = _records(account_id)
    index = ChromaVectorIndex(
        collection_name=f"ogmemory_code_{uuid.uuid4().hex}",
        persist_directory=str(tmp_path / "chroma"),
        dimension=3,
    )
    index.upsert(records)

    _assert_provider_parity(index, account_id, chunk_key)


@pytest.mark.integration
def test_opengauss_code_prefilter_parity():
    dsn = os.environ.get("OGMEM_TEST_OPENGAUSS_DSN") or os.environ.get(
        "OPENGAUSS_CONNECTION_STRING"
    )
    if not dsn:
        pytest.skip("OGMEM_TEST_OPENGAUSS_DSN is required for real provider parity")

    account_id = f"ogmemory-code-opengauss-{uuid.uuid4().hex[:12]}"
    records, chunk_key = _records(account_id)
    index = OpenGaussVectorIndex(
        dsn,
        dimension=3,
        table_name="ogmemory_code_vector_parity",
        pool_size=1,
    )
    try:
        index.upsert(records)
        _assert_provider_parity(index, account_id, chunk_key)
    finally:
        index.delete([record.id for record in records])
        index.close()
