from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime

import pytest

from core.models import RequestContext
from features.tool_result_externalization.externalizer import (
    BoundedDaemonIORunner,
    ToolResultExternalizer,
    parse_stub,
    render_safe_message_content,
    search_artifacts,
)
from features.tool_result_externalization.models import ArtifactMetadata
from features.tool_result_externalization.normalizer import ToolResultNormalizer
from features.tool_result_externalization.policy import ToolResultPolicy
from features.tool_result_externalization.store_agfs import (
    AGFSToolResultArtifactStore,
)
from features.tool_result_externalization.store_base import (
    ArtifactNotVisibleError,
    ArtifactStoreUnavailableError,
    ToolResultArtifactStore,
)
from features.tool_result_externalization.store_local import (
    LocalToolResultArtifactStore,
)
from features.tool_result_externalization import store_sql as store_sql_module
from features.tool_result_externalization.store_sql import (
    SQLToolResultArtifactStore,
)
from features.tool_result_externalization.synopsis import ToolResultSynopsisBuilder


def _ctx(**overrides) -> RequestContext:
    values = {
        "account_id": "acct",
        "user_id": "user",
        "agent_id": "agent",
        "session_id": "session",
        "trace_id": "trace",
    }
    values.update(overrides)
    return RequestContext(**values)


def _foundation_metadata(payload: str) -> ArtifactMetadata:
    encoded = payload.encode("utf-8")
    return ArtifactMetadata(
        schema_version=2,
        artifact_id="tr_0123456789abcdef",
        account_id="acct",
        session_id="session",
        owner_user_id="user",
        owner_agent_id="agent",
        turn_id=None,
        source_message_id=None,
        tool_call_id="call-1",
        tool_name="read_file",
        part_path="/content",
        occurrence_index=0,
        mime_type="text/plain",
        detected_type="text",
        serialization="raw_string",
        original_chars=len(payload),
        original_bytes=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        synopsis="foundation payload",
        externalization_reason="single_result_threshold",
        identity_strength="fallback",
        created_at=datetime.now(UTC).isoformat(),
    )


def _externalizer(
    store,
    *,
    single=10,
    artifact_budget=2_000,
    turn_budget=12_000,
):
    return ToolResultExternalizer(
        ToolResultNormalizer(),
        ToolResultPolicy(single),
        ToolResultSynopsisBuilder(300),
        store,
        retry_attempts=1,
        summary_per_artifact_chars=artifact_budget,
        summary_per_turn_chars=turn_budget,
    )


class _FailingStore(ToolResultArtifactStore):
    def write(self, metadata, output_text, ctx):
        raise OSError("unavailable")

    def read_all(self, artifact_id, session_id, ctx):
        raise ArtifactNotVisibleError("not_found_or_not_visible")

    def list(self, session_id, ctx, limit=50, cursor=None):
        return {"artifacts": [], "truncated": False, "next_cursor": None}

    def delete_session_artifacts(self, session_id, ctx):
        return 0


class _BlockingStore(ToolResultArtifactStore):
    def __init__(self, *, block_on: str):
        self.block_on = block_on
        self.entered = threading.Event()
        self.release = threading.Event()
        self.rows = {}
        self.write_calls = 0

    def _block(self, stage):
        if self.block_on == stage:
            self.entered.set()
            self.release.wait(timeout=5)

    def write(self, metadata, output_text, ctx):
        self.write_calls += 1
        self._block("write")
        self.rows[metadata.artifact_id] = (metadata, output_text)
        return False

    def read_all(self, artifact_id, session_id, ctx):
        self._block("read")
        return self.rows[artifact_id]

    def list(self, session_id, ctx, limit=50, cursor=None):
        self._block("list")
        return {"artifacts": [], "truncated": False, "next_cursor": None}

    def delete_session_artifacts(self, session_id, ctx):
        return 0


class _MemoryAGFSClient:
    def __init__(self):
        self.files = {}
        self.read_paths = []

    def mkdir(self, path):
        return None

    def write(self, path, data):
        self.files[path] = bytes(data)

    def read(self, path, offset=0, size=-1):
        self.read_paths.append(path)
        data = self.files[path]
        return data[offset:] if size < 0 else data[offset:offset + size]

    def rm(self, path, recursive=False):
        if recursive:
            for key in list(self.files):
                if key.startswith(path.rstrip("/") + "/"):
                    self.files.pop(key)
        else:
            self.files.pop(path, None)

    def ls(self, path):
        return []


class _OrderedMetadataStore(ToolResultArtifactStore):
    def __init__(self, rows):
        self.rows = rows
        self.outputs = {row.artifact_id: row.synopsis for row in rows}

    def write(self, metadata, output_text, ctx):
        raise NotImplementedError

    def read_all(self, artifact_id, session_id, ctx):
        metadata = next(row for row in self.rows if row.artifact_id == artifact_id)
        return metadata, self.outputs[artifact_id]

    def list(self, session_id, ctx, limit=50, cursor=None):
        ids = [row.artifact_id for row in self.rows]
        start = ids.index(cursor) + 1 if cursor else 0
        page = self.rows[start:start + limit]
        truncated = start + limit < len(self.rows)
        return {
            "artifacts": [row.to_dict() | {"ref": row.ref} for row in page],
            "truncated": truncated,
            "next_cursor": page[-1].artifact_id if truncated and page else None,
        }

    def delete_session_artifacts(self, session_id, ctx):
        return 0


def _metadata(artifact_id: str, synopsis: str) -> ArtifactMetadata:
    encoded = synopsis.encode("utf-8")
    return ArtifactMetadata(
        schema_version=2,
        artifact_id=artifact_id,
        account_id="acct",
        session_id="session",
        owner_user_id="user",
        owner_agent_id="agent",
        turn_id=None,
        source_message_id=None,
        tool_call_id=None,
        tool_name="tool",
        part_path="/content",
        occurrence_index=0,
        mime_type="text/plain",
        detected_type="text",
        serialization="raw_string",
        original_chars=len(synopsis),
        original_bytes=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        synopsis=synopsis,
        externalization_reason="test",
        identity_strength="fallback",
        created_at=datetime.now(UTC).isoformat(),
        coverage={
            "candidate_pool_truncated": False,
            "scan_complete": True,
            "scanned_end_bytes": len(encoded),
            "scanned_start_bytes": 0,
            "uncovered_candidates": 0,
        },
    )


def test_normalizer_associates_openclaw_tool_call_and_preserves_ordinary_text():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ordinary"},
                {"type": "toolCall", "id": "call-1", "name": "read_file"},
            ],
        },
        {"role": "toolResult", "content": [{"type": "text", "text": "payload"}]},
    ]
    normalized = ToolResultNormalizer().normalize(messages)
    assert normalized[0].ordinary_text_parts == ("ordinary",)
    part = normalized[1].tool_results[0]
    assert (part.tool_call_id, part.tool_name, part.output_text) == (
        "call-1", "read_file", "payload",
    )
    assert messages[1]["content"][0]["text"] == "payload"
@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "tool_call_id": "c", "content": {"rows": [1, 2]}},
        {
            "role": "assistant",
            "content": [{"type": "tool_result", "toolCallId": "c", "output": [1, 2]}],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_call", "id": "c", "tool_output": "done"}],
        },
    ],
)
def test_normalizer_supported_schemas(message):
    part = ToolResultNormalizer().normalize([message])[0].tool_results[0]
    assert part.tool_call_id == "c"
    assert part.output_text
def test_policy_externalizes_at_or_above_single_result_threshold():
    normalized = ToolResultNormalizer().normalize([
        {"role": "tool", "tool_call_id": "below", "content": "a" * 9},
        {"role": "tool", "tool_call_id": "exact", "content": "b" * 10},
        {"role": "tool", "tool_call_id": "above", "content": "c" * 11},
    ])

    selected = ToolResultPolicy(single_result_chars=10).select(normalized)

    assert [(item.part.tool_call_id, item.reason) for item in selected] == [
        ("exact", "single_result_threshold"),
        ("above", "single_result_threshold"),
    ]
def test_local_store_round_trips_owner_bound_artifact(tmp_path):
    payload = "foundation payload"
    store = LocalToolResultArtifactStore(tmp_path)
    metadata = _foundation_metadata(payload)

    assert store.write(metadata, payload, _ctx()) is False
    restored_metadata, restored_payload = store.read_all(
        metadata.artifact_id,
        "session",
        _ctx(),
    )

    assert restored_metadata.artifact_id == metadata.artifact_id
    assert restored_payload == payload
    assert store.list("session", _ctx())["artifacts"][0]["artifact_id"] == (
        metadata.artifact_id
    )


def test_local_store_reuses_cross_process_publication_winner(
    tmp_path,
    monkeypatch,
):
    payload = "foundation payload"
    store = LocalToolResultArtifactStore(tmp_path)
    metadata = _foundation_metadata(payload)
    destination = store._dir(metadata.artifact_id, "session", _ctx())

    def publish_winner(_source, target):
        assert target == destination
        destination.mkdir(parents=True)
        (destination / "output.txt").write_text(payload, encoding="utf-8")
        (destination / "metadata.json").write_text(
            json.dumps(metadata.to_dict()),
            encoding="utf-8",
        )
        raise FileExistsError("publication race lost")

    monkeypatch.setattr(os, "replace", publish_winner)

    assert store.write(metadata, payload, _ctx()) is True
    assert not list(destination.parent.glob(f".{metadata.artifact_id}.*"))


class _AtomicAGFSClient:
    """Small shared client with conflict-preserving mkdir semantics."""

    def __init__(self):
        self.directories = {"/", "/local"}
        self.files = {}
        self.lock = threading.RLock()
        self.metadata_write_entered = threading.Event()
        self.allow_metadata_write = threading.Event()
        self.claim_conflict = threading.Event()
        self.block_metadata_write = False
        self.fail_marker_write = False

    def mkdir(self, path):
        path = path.rstrip("/") or "/"
        with self.lock:
            if path in self.directories:
                self.claim_conflict.set()
                raise FileExistsError(path)
            parent = path.rpartition("/")[0] or "/"
            if parent not in self.directories:
                raise FileNotFoundError(parent)
            self.directories.add(path)

    def write(self, path, data):
        if self.fail_marker_write and path.endswith("/_writing.json"):
            raise RuntimeError("marker write failed")
        if self.block_metadata_write and path.endswith("/metadata.json"):
            self.metadata_write_entered.set()
            assert self.allow_metadata_write.wait(timeout=2)
        with self.lock:
            self.files[path] = bytes(data)

    def read(self, path, offset=0, size=-1):
        with self.lock:
            if path not in self.files:
                raise FileNotFoundError(path)
            data = self.files[path]
        return data[offset:] if size < 0 else data[offset:offset + size]

    def rm(self, path, recursive=False):
        with self.lock:
            if recursive:
                prefix = path.rstrip("/") + "/"
                self.files = {
                    key: value
                    for key, value in self.files.items()
                    if not key.startswith(prefix)
                }
                self.directories = {
                    key for key in self.directories
                    if key != path and not key.startswith(prefix)
                }
                return
            if path not in self.files:
                raise FileNotFoundError(path)
            self.files.pop(path)

    def ls(self, path):
        prefix = path.rstrip("/") + "/"
        with self.lock:
            names = {
                directory[len(prefix):]
                for directory in self.directories
                if directory.startswith(prefix)
                and directory[len(prefix):]
                and "/" not in directory[len(prefix):]
            }
        return [{"name": name, "isDir": True} for name in sorted(names)]


def test_agfs_concurrent_identical_write_reuses_committed_winner():
    payload = "foundation payload"
    metadata = _foundation_metadata(payload)
    client = _AtomicAGFSClient()
    client.block_metadata_write = True
    owner = AGFSToolResultArtifactStore(client)
    contender = AGFSToolResultArtifactStore(client)
    results = {}
    errors = []

    def run(name, store):
        try:
            results[name] = store.write(metadata, payload, _ctx())
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    owner_thread = threading.Thread(target=run, args=("owner", owner))
    owner_thread.start()
    assert client.metadata_write_entered.wait(timeout=1)

    contender_thread = threading.Thread(
        target=run,
        args=("contender", contender),
    )
    contender_thread.start()
    assert client.claim_conflict.wait(timeout=1)
    client.allow_metadata_write.set()
    owner_thread.join(timeout=2)
    contender_thread.join(timeout=2)

    assert not owner_thread.is_alive()
    assert not contender_thread.is_alive()
    assert errors == []
    assert results == {"owner": False, "contender": True}
    assert owner.read_all(metadata.artifact_id, "session", _ctx())[1] == payload


def test_agfs_incomplete_concurrent_claim_is_retryable_and_not_overwritten():
    payload = "foundation payload"
    metadata = _foundation_metadata(payload)
    client = _AtomicAGFSClient()
    store = AGFSToolResultArtifactStore(client)
    directory = store._dir(metadata.artifact_id, "session", _ctx())
    store._mkdir_p(directory)
    marker_path = f"{directory}/_writing.json"
    client.files[marker_path] = b'{"writer_token":"other"}'
    store._concurrent_commit_wait_seconds = 0

    with pytest.raises(
        ArtifactStoreUnavailableError,
        match="already in progress",
    ):
        store.write(metadata, payload, _ctx())

    assert client.files == {marker_path: b'{"writer_token":"other"}'}


def test_agfs_marker_failure_releases_exclusive_empty_claim():
    payload = "foundation payload"
    metadata = _foundation_metadata(payload)
    client = _AtomicAGFSClient()
    client.fail_marker_write = True
    store = AGFSToolResultArtifactStore(client)
    directory = store._dir(metadata.artifact_id, "session", _ctx())

    with pytest.raises(ArtifactStoreUnavailableError):
        store.write(metadata, payload, _ctx())

    assert directory not in client.directories
    assert not any(path.startswith(directory + "/") for path in client.files)


def test_agfs_committed_identity_conflict_is_not_overwritten():
    first_payload = "foundation payload"
    second_payload = "different payload"
    client = _AtomicAGFSClient()
    store = AGFSToolResultArtifactStore(client)
    first = _foundation_metadata(first_payload)
    second = _foundation_metadata(second_payload)

    assert store.write(first, first_payload, _ctx()) is False
    with pytest.raises(FileExistsError, match="identity conflict"):
        store.write(second, second_payload, _ctx())
    assert store.read_all(first.artifact_id, "session", _ctx())[1] == first_payload


class _AGFSClient:
    def __init__(
        self,
        *,
        ls_result=None,
        ls_error=None,
        read_error=None,
        rm_error=None,
    ):
        self.ls_result = list(ls_result or [])
        self.ls_error = ls_error
        self.read_error = read_error
        self.rm_error = rm_error
        self.removed = []
        self.ls_by_path = {}

    def ls(self, path):
        if self.ls_error is not None:
            raise self.ls_error
        if path in self.ls_by_path:
            return self.ls_by_path[path]
        return self.ls_result

    def read(self, _path, **_kwargs):
        if self.read_error is not None:
            raise self.read_error
        raise AssertionError("incomplete directories must not require metadata reads")

    def rm(self, path, recursive=False):
        if self.rm_error is not None:
            raise self.rm_error
        self.removed.append((path, recursive))


def test_agfs_read_propagates_backend_failure():
    store = AGFSToolResultArtifactStore(
        _AGFSClient(read_error=RuntimeError("AGFS unavailable")),
    )

    with pytest.raises(ArtifactStoreUnavailableError):
        store.read_metadata("tr_0123456789abcdef", "session", _ctx())


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("connection reset"),
        RuntimeError("backend index not found during timeout"),
        PermissionError("denied"),
    ],
)
def test_agfs_session_delete_propagates_backend_failures(error):
    store = AGFSToolResultArtifactStore(_AGFSClient(ls_error=error))

    with pytest.raises(ArtifactStoreUnavailableError):
        store.delete_session_artifacts("session", _ctx())


def test_agfs_session_delete_propagates_remove_backend_failure():
    client = _AGFSClient(
        ls_result=[{"name": "tr_0123456789abcdef", "isDir": True}],
        rm_error=RuntimeError("backend index not found during timeout"),
    )
    store = AGFSToolResultArtifactStore(client)

    with pytest.raises(ArtifactStoreUnavailableError):
        store.delete_session_artifacts("session", _ctx())


def test_agfs_session_delete_treats_only_explicit_not_found_as_complete():
    missing = AGFSToolResultArtifactStore(
        _AGFSClient(ls_error=FileNotFoundError("missing")),
    )
    raced = AGFSToolResultArtifactStore(_AGFSClient(
        ls_result=[{"name": "tr_0123456789abcdef", "isDir": True}],
        rm_error=FileNotFoundError("already removed"),
    ))

    assert missing.delete_session_artifacts("session", _ctx()) == 0
    assert raced.delete_session_artifacts("session", _ctx()) == 1


def test_agfs_stale_cleanup_propagates_listing_backend_failure():
    store = AGFSToolResultArtifactStore(_AGFSClient(
        ls_error=RuntimeError("backend index not found during timeout"),
    ))

    with pytest.raises(ArtifactStoreUnavailableError):
        store.cleanup_stale_writes("session", _ctx(), 60)


def test_agfs_stale_cleanup_does_not_delete_committed_artifact_on_read_failure():
    artifact_id = "tr_0123456789abcdef"
    now = datetime(2026, 1, 1, tzinfo=UTC)

    class _CleanupClient(_AGFSClient):
        def __init__(self):
            super().__init__(ls_result=[{"name": artifact_id, "isDir": True}])

        def read(self, path, **_kwargs):
            if path.endswith("/metadata.json"):
                raise RuntimeError("backend index not found during timeout")
            if path.endswith("/_writing.json"):
                return json.dumps({
                    "started_at": "2025-12-31T23:00:00+00:00",
                }).encode()
            raise AssertionError(path)

    client = _CleanupClient()
    store = AGFSToolResultArtifactStore(client)

    with pytest.raises(ArtifactStoreUnavailableError):
        store.cleanup_stale_writes("session", _ctx(), 60, now=now)
    assert client.removed == []


def test_agfs_stale_cleanup_removes_only_explicitly_uncommitted_artifact():
    artifact_id = "tr_0123456789abcdef"
    now = datetime(2026, 1, 1, tzinfo=UTC)

    class _CleanupClient(_AGFSClient):
        def __init__(self):
            super().__init__(ls_result=[{"name": artifact_id, "isDir": True}])

        def read(self, path, **_kwargs):
            if path.endswith("/metadata.json"):
                raise FileNotFoundError("metadata missing")
            if path.endswith("/_writing.json"):
                return json.dumps({
                    "started_at": "2025-12-31T23:00:00+00:00",
                }).encode()
            raise AssertionError(path)

    client = _CleanupClient()
    store = AGFSToolResultArtifactStore(client)

    assert store.cleanup_stale_writes("session", _ctx(), 60, now=now) == 1
    assert len(client.removed) == 1


def test_local_session_delete_removes_incomplete_artifact_directories(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    base = store._base("session", _ctx())
    (base / "tr_0123456789abcdef").mkdir(parents=True)

    assert store.delete_session_artifacts("session", _ctx()) == 1
    assert not base.exists()


def test_local_session_delete_isolated_by_agent_path(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    payload = "agent-owned payload"
    metadata = _foundation_metadata(payload)
    owner_ctx = _ctx()
    other_ctx = _ctx(agent_id="other-agent")
    store.write(metadata, payload, owner_ctx)

    assert store._base("session", owner_ctx) != store._base("session", other_ctx)
    assert store.delete_session_artifacts("session", other_ctx) == 0
    assert store.read_all(metadata.artifact_id, "session", owner_ctx)[1] == payload


def test_agfs_session_delete_isolated_by_agent_path():
    client = _AGFSClient()
    store = AGFSToolResultArtifactStore(client)
    owner_ctx = _ctx()
    other_ctx = _ctx(agent_id="other-agent")
    owner_base = store._base("session", owner_ctx)
    other_base = store._base("session", other_ctx)
    client.ls_by_path[owner_base] = [{
        "name": "tr_0123456789abcdef",
        "isDir": True,
    }]

    assert owner_base != other_base
    assert store.delete_session_artifacts("session", other_ctx) == 0
    assert client.removed == [(other_base, True)]
    assert store.delete_session_artifacts("session", owner_ctx) == 1
    assert client.removed == [(other_base, True), (owner_base, True)]


def test_store_rejects_unsafe_agent_path_component(tmp_path):
    with pytest.raises(ArtifactNotVisibleError):
        LocalToolResultArtifactStore(tmp_path)._base(
            "session",
            _ctx(agent_id="../other-agent"),
        )
    with pytest.raises(ArtifactNotVisibleError):
        AGFSToolResultArtifactStore(_AGFSClient())._base(
            "session",
            _ctx(agent_id="../other-agent"),
        )


class _SQLWriteCursor:
    def __init__(self, insert_row, existing_row=None):
        self.insert_row = insert_row
        self.existing_row = existing_row
        self.current_row = None
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.queries.append((query, params))
        self.current_row = (
            self.insert_row if "INSERT INTO" in query else self.existing_row
        )

    def fetchone(self):
        return self.current_row


class _SQLWriteConnection:
    closed = 0

    def __init__(self, insert_row, existing_row=None):
        self.cursor_value = _SQLWriteCursor(insert_row, existing_row)
        self.commit_calls = 0
        self.rollback_calls = 0

    def cursor(self):
        return self.cursor_value

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1


def _sql_write_store(monkeypatch, insert_row, existing_row=None):
    connection = _SQLWriteConnection(insert_row, existing_row)
    store = object.__new__(SQLToolResultArtifactStore)
    store._get_connection = lambda: connection
    store._return_connection = lambda _conn: None
    store.bind_tenant = lambda _conn, _account_id: None
    monkeypatch.setattr(store_sql_module, "Json", lambda value: value)
    return store, connection


def test_sql_write_uses_atomic_insert_for_new_artifact(monkeypatch):
    payload = "foundation payload"
    metadata = _foundation_metadata(payload)
    store, connection = _sql_write_store(
        monkeypatch,
        (metadata.artifact_id,),
    )

    assert store.write(metadata, payload, _ctx()) is False
    query = connection.cursor_value.queries[0][0]
    assert "ON CONFLICT" in query
    assert "DO NOTHING" in query
    assert "RETURNING artifact_id" in query
    assert connection.commit_calls == 1
    assert connection.rollback_calls == 0


def test_sql_concurrent_identical_write_reuses_winner(monkeypatch):
    payload = "foundation payload"
    metadata = _foundation_metadata(payload)
    store, connection = _sql_write_store(
        monkeypatch,
        None,
        (metadata.to_dict(), payload),
    )

    assert store.write(metadata, payload, _ctx()) is True
    assert len(connection.cursor_value.queries) == 2
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 1


def test_sql_concurrent_conflicting_write_is_not_overwritten(monkeypatch):
    payload = "foundation payload"
    metadata = _foundation_metadata(payload)
    store, connection = _sql_write_store(
        monkeypatch,
        None,
        (metadata.to_dict(), "different payload"),
    )

    with pytest.raises(FileExistsError, match="identity conflict"):
        store.write(metadata, payload, _ctx())

    assert connection.commit_calls == 0
    assert connection.rollback_calls == 1


def test_sql_concurrent_winner_disappearing_is_retryable(monkeypatch):
    payload = "foundation payload"
    metadata = _foundation_metadata(payload)
    store, connection = _sql_write_store(monkeypatch, None, None)

    with pytest.raises(
        ArtifactStoreUnavailableError,
        match="could not be verified",
    ):
        store.write(metadata, payload, _ctx())

    assert connection.commit_calls == 0
    assert connection.rollback_calls == 1


def test_local_read_range_does_not_load_the_complete_output(tmp_path, monkeypatch):
    payload = "prefix-工具结果-suffix"
    store = LocalToolResultArtifactStore(tmp_path)
    metadata = _foundation_metadata(payload)
    store.write(metadata, payload, _ctx())
    monkeypatch.setattr(
        store,
        "read_all",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("read_range must not call read_all")
        ),
    )

    page = store.read_range(metadata.artifact_id, "session", _ctx(), 7, 12)

    assert page["content"]
    assert page["actual_length_bytes"] <= 15


def test_externalization_writes_verified_stub_and_reuses_strong_identity(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store)
    message = {
        "role": "toolResult",
        "content": "结果" * 20,
        "_ogmem_source": {"source_message_id": "m-1", "turn_id": "t-1"},
    }
    first = externalizer.govern_messages(
        [message], ctx=_ctx(), session_id="session", persistence=True,
    )
    second = externalizer.govern_messages(
        [message], ctx=_ctx(), session_id="session", persistence=True,
    )
    assert first.messages == second.messages
    stub = render_safe_message_content(first.messages[0])
    fields = parse_stub(stub)
    assert fields is not None
    artifact_id = fields["ref"].rsplit("/", 1)[-1]
    metadata, output = store.read_all(artifact_id, "session", _ctx())
    assert output == message["content"]
    assert metadata.identity_strength == "strong"
    assert len(store.list("session", _ctx())["artifacts"]) == 1


@pytest.mark.parametrize("block_on", ["write", "read"])
def test_externalization_deadline_bounds_blocking_write_and_verify(block_on):
    store = _BlockingStore(block_on=block_on)
    externalizer = ToolResultExternalizer(
        ToolResultNormalizer(),
        ToolResultPolicy(1),
        ToolResultSynopsisBuilder(100),
        store,
        retry_attempts=2,
        write_timeout_ms=20,
        io_runner=BoundedDaemonIORunner(max_inflight=1),
    )
    started = time.monotonic()
    try:
        governed = externalizer.govern_messages(
            [{"role": "toolResult", "content": "payload"}],
            ctx=_ctx(), session_id="session", persistence=True,
        )
        elapsed = time.monotonic() - started
        assert store.entered.wait(timeout=1)
        assert elapsed < 0.15
        assert governed.messages == []
        assert governed.failed_count == 1
        assert store.write_calls == 1
    finally:
        store.release.set()
def test_local_store_uses_account_user_agent_session_tool_result_path(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    metadata = _metadata("tr_" + "a" * 32, "plain-session-path")

    store.write(metadata, metadata.synopsis, _ctx())

    artifact_dir = (
        tmp_path
        / "accounts"
        / "acct"
        / "user"
        / "user"
        / "agents"
        / "agent"
        / "sessions"
        / "session"
        / "tool-results"
        / metadata.artifact_id
    )
    assert (artifact_dir / "output.txt").read_text(encoding="utf-8") == metadata.synopsis
    assert (artifact_dir / "metadata.json").is_file()


@pytest.mark.parametrize("unsafe", [".", "..", "../other", "/absolute", "a\\b", "C:drive"])
def test_local_store_rejects_unsafe_plain_session_path_components(tmp_path, unsafe):
    store = LocalToolResultArtifactStore(tmp_path)
    with pytest.raises(ArtifactNotVisibleError, match="not_found_or_not_visible"):
        store._base(unsafe, _ctx(session_id=unsafe))
    with pytest.raises(ArtifactNotVisibleError, match="not_found_or_not_visible"):
        store._base("session", _ctx(account_id=unsafe))
    with pytest.raises(ArtifactNotVisibleError, match="not_found_or_not_visible"):
        store._base("session", _ctx(user_id=unsafe))
    with pytest.raises(ArtifactNotVisibleError, match="not_found_or_not_visible"):
        store._base("session", _ctx(agent_id=unsafe))
def test_v2_stub_is_the_only_canonical_format(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    metadata = _metadata("tr_" + "a" * 32, "partial synopsis")
    store.write(metadata, "partial synopsis", _ctx())
    externalizer = _externalizer(store)

    stub = externalizer.build_stub(metadata)
    fields = parse_stub(stub)
    assert fields is not None
    assert fields["version"] == "2"
    assert fields["session_id"] == "session"
    assert fields["bytes"] == str(metadata.original_bytes)
    assert json.loads(fields["retrieval_hint"]) == externalizer._retrieval_hint()
    assert externalizer.validates_stub_sequence(stub, _ctx(), "session")
    legacy = stub.replace("v=2", "v=1", 1)
    assert parse_stub(legacy) is None
    assert not externalizer.validates_stub_sequence(legacy, _ctx(), "session")
def test_verified_stub_replay_is_idempotent_and_does_not_create_artifact(
    tmp_path,
):
    store = LocalToolResultArtifactStore(tmp_path)
    metadata = _metadata("tr_" + "e" * 32, "persisted payload")
    store.write(metadata, "persisted payload", _ctx())
    externalizer = _externalizer(store, single=1)
    stub = externalizer.build_stub(metadata)

    governed = externalizer.govern_messages(
        [{"role": "toolResult", "content": stub}],
        ctx=_ctx(),
        session_id="session",
        persistence=True,
    )

    assert governed.externalized_count == 0
    assert governed.failed_count == 0
    assert render_safe_message_content(governed.messages[0]) == stub
    assert len(store.list("session", _ctx())["artifacts"]) == 1


@pytest.mark.parametrize("field", ["session_id", "bytes", "retrieval_hint"])
def test_enriched_stub_rejects_tampered_guidance_fields(tmp_path, field):
    store = LocalToolResultArtifactStore(tmp_path)
    metadata = _metadata("tr_" + "b" * 32, "partial synopsis")
    store.write(metadata, "partial synopsis", _ctx())
    externalizer = _externalizer(store)
    stub = externalizer.build_stub(metadata)
    line = next(value for value in stub.splitlines() if value.startswith(field + ": "))
    tampered = stub.replace(line, field + ": forged", 1)

    assert not externalizer.validates_stub_sequence(tampered, _ctx(), "session")
def test_utf8_byte_range_pages_reconstruct_complete_output(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, single=1)
    output = "甲乙🙂丙丁" * 30
    governed = externalizer.govern_messages(
        [{
            "role": "tool",
            "tool_call_id": "c",
            "content": output,
            "_ogmem_source": {"source_message_id": "m"},
        }],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    artifact_id = parse_stub(render_safe_message_content(governed.messages[0]))["ref"].rsplit("/", 1)[-1]
    pages = []
    offset = 0
    while True:
        page = store.read_range(artifact_id, "session", _ctx(), offset, 7)
        pages.append(page["content"])
        assert page["content"].encode("utf-8").decode("utf-8") == page["content"]
        if page["eof"]:
            break
        assert page["next_offset_bytes"] > offset
        offset = page["next_offset_bytes"]
    assert "".join(pages) == output
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Request timeout after 30s"),
        RuntimeError("Connection refused - server not running"),
        RuntimeError("Internal server error"),
        RuntimeError("Bad Gateway - backend service unavailable"),
        RuntimeError("HTTP error 503"),
        RuntimeError("read: backend index not found during timeout"),
        PermissionError("denied"),
    ],
)
def test_agfs_list_propagates_backend_failures(error):
    client = _MemoryAGFSClient()
    client.ls = lambda _path: (_ for _ in ()).throw(error)
    store = AGFSToolResultArtifactStore(client, mount_prefix="/local")

    with pytest.raises(
        ArtifactStoreUnavailableError,
        match="artifact store is unavailable",
    ):
        store.list("session", _ctx())


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing"),
        RuntimeError("No such file or directory"),
        RuntimeError("no such directory: /accounts/acct/session/tool-results"),
        RuntimeError("no such file: /accounts/acct/session/metadata.json"),
        RuntimeError("file not found: output.txt"),
        RuntimeError("directory not found: tool-results"),
        RuntimeError("read: /accounts/acct/session/metadata.json: not found"),
        RuntimeError("readdir: /accounts/acct/session/tool-results: not found"),
        RuntimeError("not found"),
    ],
)
def test_agfs_list_treats_only_explicit_not_found_as_empty(error):
    client = _MemoryAGFSClient()
    client.ls = lambda _path: (_ for _ in ()).throw(error)
    store = AGFSToolResultArtifactStore(client, mount_prefix="/local")

    assert store.list("session", _ctx()) == {
        "artifacts": [],
        "truncated": False,
        "next_cursor": None,
    }
def test_agfs_session_delete_removes_incomplete_artifact_directories():
    client = _MemoryAGFSClient()
    store = AGFSToolResultArtifactStore(client, mount_prefix="/local")
    artifact_id = "tr_" + "c" * 32
    base = store._base("session", _ctx())
    client.files[f"{base}/{artifact_id}/_writing.json"] = b"{}"
    client.files[f"{base}/{artifact_id}/output.txt"] = b"partial"
    client.ls = lambda path: (
        [{"name": artifact_id, "isDir": True}]
        if path == base else []
    )

    assert store.delete_session_artifacts("session", _ctx()) == 1
    assert not any(path.startswith(base + "/") for path in client.files)
def test_owner_tuple_isolation_and_tamper_detection(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, single=1)
    governed = externalizer.govern_messages(
        [{"role": "tool", "content": "large"}],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    artifact_id = parse_stub(render_safe_message_content(governed.messages[0]))["ref"].rsplit("/", 1)[-1]
    with pytest.raises(ArtifactNotVisibleError):
        store.read_all(artifact_id, "session", _ctx(user_id="other"))
    metadata, _ = store.read_all(artifact_id, "session", _ctx())
    directory = next(tmp_path.rglob(artifact_id))
    (directory / "output.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactNotVisibleError):
        store.read_all(metadata.artifact_id, "session", _ctx())


def test_search_returns_bounded_consistent_byte_metadata(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, single=1)
    externalizer.govern_messages(
        [{"role": "tool", "content": "prefix needle suffix " * 30}],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    result = search_artifacts(
        store, session_id="session", query="needle", ctx=_ctx(), top_k=2,
        max_chars=17, max_artifacts=10, chunk_bytes=32, overlap_bytes=4,
        deadline_ms=1000,
    )
    assert result["results"]
    hit = result["results"][0]
    encoded = hit["content"].encode("utf-8")
    assert hit["actual_length_bytes"] == len(encoded)
    assert hit["next_offset_bytes"] == hit["offset_bytes"] + len(encoded)
    assert hit["chunk_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert result["search_complete"] is True
    assert result["no_match_confirmed"] is False
    assert result["scanned_artifacts"] == 1
    assert result["scanned_bytes"] == len(("prefix needle suffix " * 30).encode("utf-8"))
def test_search_keeps_session_and_owner_isolation(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, single=1)
    externalizer.govern_messages(
        [{"role": "tool", "content": "private needle"}],
        ctx=_ctx(),
        session_id="session",
        persistence=True,
    )

    other_session = search_artifacts(
        store,
        session_id="other-session",
        query="needle",
        ctx=_ctx(session_id="other-session"),
        top_k=5,
        max_chars=1_000,
        max_artifacts=5,
        chunk_bytes=256,
        overlap_bytes=32,
        deadline_ms=1_000,
    )
    other_owner = search_artifacts(
        store,
        session_id="session",
        query="needle",
        ctx=_ctx(user_id="other"),
        top_k=5,
        max_chars=1_000,
        max_artifacts=5,
        chunk_bytes=256,
        overlap_bytes=32,
        deadline_ms=1_000,
    )

    assert other_session["results"] == []
    assert other_owner["results"] == []


def test_source_aware_envelope_tamper_is_not_trusted(tmp_path):
    store = LocalToolResultArtifactStore(tmp_path)
    externalizer = _externalizer(store, single=1)
    original = externalizer.govern_messages(
        [{
            "role": "tool",
            "tool_call_id": "original-call",
            "content": "original output",
            "_ogmem_source": {"source_message_id": "m1"},
        }],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    ref = parse_stub(render_safe_message_content(original.messages[0]))["ref"]
    metadata, output = store.read_all(ref.rsplit("/", 1)[-1], "session", _ctx())
    envelope = {
        "_ogmemory_artifact_source": True,
        "schema_version": 1,
        "source_tool": "read_tool_result_artifact",
        "source_tool_call_id": "read-call",
        "artifact_ref": ref,
        "offset_bytes": 0,
        "actual_length_bytes": len(output.encode()),
        "artifact_sha256": metadata.sha256,
        "content_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "content": output + "tampered",
    }
    governed = externalizer.govern_messages(
        [{
            "role": "assistant",
            "content": [{
                "type": "tool_result", "tool_call_id": "read-call", "output": envelope,
            }],
        }],
        ctx=_ctx(), session_id="session", persistence=True,
    )
    # A recognized but invalid source envelope must never become an Artifact
    # of an Artifact. Persistence fails closed.
    assert governed.messages == []
    assert governed.preserved_message_ids == []
    assert len(store.list("session", _ctx())["artifacts"]) == 1
