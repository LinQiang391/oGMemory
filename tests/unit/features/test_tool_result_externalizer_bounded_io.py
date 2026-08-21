from __future__ import annotations

from core.models import RequestContext
from features.tool_result_externalization.externalizer import (
    ToolResultExternalizer,
    render_safe_message_content,
    search_artifacts,
)
from features.tool_result_externalization.normalizer import ToolResultNormalizer
from features.tool_result_externalization.policy import ToolResultPolicy
from features.tool_result_externalization.store_local import (
    LocalToolResultArtifactStore,
)
from features.tool_result_externalization.synopsis import ToolResultSynopsisBuilder


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )


class _BoundedIOStore(LocalToolResultArtifactStore):
    def __init__(self, root):
        super().__init__(root)
        self.allow_full_reads = True
        self.full_read_calls = 0
        self.metadata_read_calls = 0
        self.range_read_calls: list[tuple[int, int]] = []

    def read_all(self, artifact_id, session_id, ctx):
        self.full_read_calls += 1
        if not self.allow_full_reads:
            raise AssertionError("validation and search must not read full artifacts")
        return super().read_all(artifact_id, session_id, ctx)

    def read_metadata(self, artifact_id, session_id, ctx):
        self.metadata_read_calls += 1
        return super().read_metadata(artifact_id, session_id, ctx)

    def read_range(self, artifact_id, session_id, ctx, offset_bytes, length_bytes):
        self.range_read_calls.append((offset_bytes, length_bytes))
        return super().read_range(
            artifact_id,
            session_id,
            ctx,
            offset_bytes,
            length_bytes,
        )


class _FailSecondPartStore(LocalToolResultArtifactStore):
    def write(self, metadata, output_text, ctx):
        if "FAIL-SECOND-PART" in output_text:
            raise OSError("second part store failure")
        return super().write(metadata, output_text, ctx)


def _externalizer(store) -> ToolResultExternalizer:
    return ToolResultExternalizer(
        ToolResultNormalizer(),
        ToolResultPolicy(1),
        ToolResultSynopsisBuilder(300),
        store,
        retry_attempts=1,
        summary_per_artifact_chars=2_000,
        summary_per_turn_chars=12_000,
    )


def _externalize(store, payload: str) -> tuple[ToolResultExternalizer, str]:
    externalizer = _externalizer(store)
    governed = externalizer.govern_messages(
        [{
            "role": "toolResult",
            "toolCallId": "call-1",
            "toolName": "read_logs",
            "content": payload,
        }],
        ctx=_ctx(),
        session_id="session",
        persistence=True,
    )
    assert governed.externalized_count == 1
    return externalizer, render_safe_message_content(governed.messages[0])


def test_stub_validation_reads_metadata_without_loading_artifact(tmp_path):
    store = _BoundedIOStore(tmp_path)
    externalizer, stub = _externalize(
        store,
        "ERROR request_id=req-123 backend unavailable\n" * 2_000,
    )
    store.full_read_calls = 0
    store.metadata_read_calls = 0
    store.allow_full_reads = False

    assert externalizer.stub_sequence_status(stub, _ctx(), "session") == "valid"
    assert (
        externalizer.embedded_stub_projection_status(stub, _ctx(), "session")
        == "valid"
    )
    assert store.full_read_calls == 0
    assert store.metadata_read_calls == 2


def test_search_materializes_hits_with_bounded_ranges_only(tmp_path):
    store = _BoundedIOStore(tmp_path)
    payload = (
        "prefix\n"
        + "A" * 80_000
        + "\nERROR needle-unique request_id=req-789\n"
        + "B" * 80_000
    )
    _externalize(store, payload)
    store.full_read_calls = 0
    store.range_read_calls.clear()
    store.allow_full_reads = False

    result = search_artifacts(
        store,
        session_id="session",
        query="needle-unique",
        ctx=_ctx(),
        top_k=3,
        max_chars=2_000,
        max_artifacts=10,
        chunk_bytes=4_096,
        overlap_bytes=256,
        deadline_ms=5_000,
    )

    assert result["results"]
    assert any(
        "needle-unique" in hit["content"]
        for hit in result["results"]
    )
    assert store.full_read_calls == 0
    assert store.range_read_calls
    assert max(length for _offset, length in store.range_read_calls) <= 64 * 1_024


def test_multi_part_failure_never_orphans_new_artifact_or_overcounts(tmp_path):
    store = _FailSecondPartStore(tmp_path)
    externalizer = _externalizer(store)
    successful_payload = "SUCCESS-FIRST-PART" * 20
    failing_payload = "FAIL-SECOND-PART" * 20
    message = {
        "role": "assistant",
        "content": [{
            "type": "tool_result",
            "tool_call_id": "call-1",
            "output": successful_payload,
            "text": failing_payload,
        }],
        "_ogmem_source": {
            "source_message_id": "message-1",
            "turn_id": "turn-1",
        },
    }

    first = externalizer.govern_messages(
        [message],
        ctx=_ctx(),
        session_id="session",
        persistence=True,
    )

    assert first.externalized_count == 1
    assert first.failed_count == 1
    assert first.preserved_message_ids == []
    assert len(first.messages) == 1
    rendered = render_safe_message_content(first.messages[0])
    artifacts = store.list("session", _ctx())["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["ref"] in rendered
    assert successful_payload not in rendered
    assert failing_payload not in rendered

    retried = externalizer.govern_messages(
        [message],
        ctx=_ctx(),
        session_id="session",
        persistence=True,
    )

    assert retried.messages == []
    assert retried.externalized_count == 0
    assert retried.failed_count == 1
    assert retried.preserved_message_ids == ["message-1"]
    assert len(store.list("session", _ctx())["artifacts"]) == 1
