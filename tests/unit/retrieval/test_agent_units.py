"""Agent units retain their identity from indexing through public consumption."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.models import ContextNode, RequestContext
from extraction.agent_memory import render_agent_memory
from index.index_record_builder import build_index_records


@pytest.fixture
def unit_memory():
    ctx = RequestContext(account_id="acct", user_id="user", agent_id="agent",
                         session_id="session", trace_id="trace")
    uri = "ctx://acct/agents/agent/memories/tools/search"
    units = [
        {"key": "batch", "title": "Search batch limit", "content": "Use batches of 20 requests.",
         "grounding": [{"uri": "trace://session/event-1"}]},
        {"key": "cursor", "title": "Search pagination", "content": "Reuse the returned cursor for the next page.",
         "grounding": [{"uri": "trace://session/event-2"}]},
    ]
    fields = {"tool_name": "search", "units": units}
    node = ContextNode(
        uri, "MEMORY", "tool", 4, "agent:agent", "Search limits and pagination", "Batch limits; cursor pagination.",
        render_agent_memory("tool", fields),
        {"agent_memory_version": 1, "agent_memory_content_version": 3,
         "routing_key": "search", "structured_fields": fields,
         "agent_memory": {"units": {item["key"]: {"revision": i + 1, "grounding": item["grounding"]}
                                    for i, item in enumerate(units)}}},
    )
    user = ContextNode("ctx://acct/users/user/memories/preferences/research", "MEMORY",
                       "preference", 2, "user:user", "Research preference", "",
                       "Prefer primary sources for research.")
    experience_fields = {"key": "rollback", "title": "Preserve the previous page",
                         "content": "Keep the previous cursor until the current page is processed.",
                         "grounding": [{"uri": "trace://session/event-3"}]}
    experience = ContextNode(
        "ctx://acct/agents/agent/memories/experiences/rollback", "MEMORY", "experience", 4,
        "agent:agent", experience_fields["title"], "", render_agent_memory("experience", experience_fields),
        {"agent_memory_version": 1, "agent_memory_content_version": 3, "routing_key": "rollback",
         "structured_fields": experience_fields,
         "agent_memory": {"units": {"rollback": {"revision": 1}}}},
    )
    nodes = {node.uri: node, user.uri: user, experience.uri: experience}
    fs = SimpleNamespace(read_node=lambda uri, ctx: nodes[uri], nodes=nodes)
    return ctx, node, user, fs


def test_short_unit_indexes_l0_and_canonical_l2(unit_memory):
    _, node, _, _ = unit_memory
    fields = {"tool_name": "search", "units": node.metadata["structured_fields"]["units"][:1]}
    node = replace(node, overview="", content=render_agent_memory("tool", fields),
                   metadata={**node.metadata, "structured_fields": fields})
    records = build_index_records(node)
    assert [record.level for record in records] == [0, 2]
    assert [(record.metadata["unit_key"], record.metadata["unit_revision"])
            for record in records if record.level == 2] == [("batch", 1)]
    assert {(record.filters["account_id"], record.filters["owner_space"]) for record in records} == {
        ("acct", "agent:agent")}


def test_chroma_unit_identity_and_grounding_round_trip(unit_memory, tmp_path):
    pytest.importorskip("chromadb")
    from providers.vector_index.chroma_index import ChromaVectorIndex

    ctx, node, _, _ = unit_memory
    index = ChromaVectorIndex(collection_name="agent_units", persist_directory=str(tmp_path), dimension=2)
    records = [record for record in build_index_records(node) if record.level == 2]
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    index.upsert(records)
    hits = index.search_by_vector([1.0, 0.0],
                                  {"account_id": ctx.account_id, "owner_space": "agent:agent"}, top_k=5)
    assert {hit.metadata["unit_key"]: (hit.metadata["unit_revision"], hit.metadata["grounding"])
            for hit in hits} == {
        unit["key"]: (i + 1, unit["grounding"])
        for i, unit in enumerate(node.metadata["structured_fields"]["units"])}


def test_chroma_adapter_and_bm25_reload_preserve_unit_metadata(unit_memory, monkeypatch):
    from providers.vector_index.chroma_index import ChromaVectorIndex
    from retrieval.bm25_index import Bm25Index
    from tests.unit.retrieval.test_bm25 import _mock_tokenize

    monkeypatch.setattr("retrieval.bm25_index._tokenize", _mock_tokenize)
    ctx, node, _, _ = unit_memory
    records = [record for record in build_index_records(node) if record.level == 2]
    for record in records:
        record.metadata["_embedding"] = [1.0, 0.0]
    index = ChromaVectorIndex.__new__(ChromaVectorIndex)
    index._collection = Mock()
    index._client = SimpleNamespace(get_max_batch_size=lambda: 100)
    index._locked = nullcontext
    index.upsert(records)
    saved = index._collection.upsert.call_args.kwargs
    assert all(isinstance(value, (str, int, float, bool))
               for metadata in saved["metadatas"] for value in metadata.values())
    index._collection.count.return_value = 2
    index._collection.query.return_value = {
        "ids": [saved["ids"]], "metadatas": [saved["metadatas"]],
        "documents": [saved["documents"]], "distances": [[0, 0]],
    }
    hits = index.search_by_vector([1, 0], {"account_id": ctx.account_id}, top_k=2)
    expected = {unit["key"]: (i + 1, unit["grounding"])
                for i, unit in enumerate(node.metadata["structured_fields"]["units"])}
    assert {hit.metadata["unit_key"]: (hit.metadata["unit_revision"], hit.metadata["grounding"])
            for hit in hits} == expected
    index._collection.count.return_value = 3
    index._collection.get.return_value = {
        "ids": saved["ids"] + ["preference"],
        "documents": saved["documents"] + ["Prefer primary sources."],
        "metadatas": saved["metadatas"] + [{"category": "preference"}],
    }
    bm25 = Bm25Index()
    bm25.build_from_chroma(index._collection)
    lexical_hits = bm25.search("batch cursor", top_k=2)
    assert {hit.metadata["unit_key"]: (hit.metadata["unit_revision"], hit.metadata["grounding"])
            for hit in lexical_hits} == expected
