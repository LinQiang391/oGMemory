from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime

import pytest

from core.models import RequestContext
from features.tool_result_externalization.externalizer import (
    ToolResultExternalizer,
    parse_stub,
    render_safe_message_content,
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


def _externalizer(store, *, threshold: int = 10) -> ToolResultExternalizer:
    return ToolResultExternalizer(
        ToolResultNormalizer(),
        ToolResultPolicy(threshold),
        ToolResultSynopsisBuilder(300),
        store,
        retry_attempts=1,
        summary_per_artifact_chars=2_000,
        summary_per_turn_chars=12_000,
    )


def test_normalizer_associates_tool_call_without_mutating_input():
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
        "call-1",
        "read_file",
        "payload",
    )
    assert messages[1]["content"][0]["text"] == "payload"


def test_policy_externalizes_at_the_single_result_threshold():
    normalized = ToolResultNormalizer().normalize([
        {"role": "toolResult", "content": "X" * 20},
    ])

    below = ToolResultPolicy(21).select(normalized)
    at_threshold = ToolResultPolicy(20).select(normalized)

    assert below == []
    assert len(at_threshold) == 1


def test_synopsis_is_bounded_and_deterministic():
    builder = ToolResultSynopsisBuilder(80)
    payload = '{"status":503,"error":"backend unavailable","items":[1,2,3]}'

    first = builder.build(payload)
    second = builder.build(payload)

    assert first == second
    assert first[0] == "json"
    assert len(first[1]) <= 80


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


@pytest.mark.parametrize(
    "error",
    [RuntimeError("connection reset"), PermissionError("denied")],
)
def test_agfs_list_propagates_backend_failures(error):
    store = AGFSToolResultArtifactStore(_AGFSClient(ls_error=error))

    with pytest.raises(ArtifactStoreUnavailableError):
        store.list("session", _ctx())


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(), RuntimeError("directory not found")],
)
def test_agfs_list_treats_only_explicit_not_found_as_empty(error):
    store = AGFSToolResultArtifactStore(_AGFSClient(ls_error=error))

    assert store.list("session", _ctx()) == {
        "artifacts": [],
        "truncated": False,
        "next_cursor": None,
    }


def test_agfs_read_propagates_backend_failure():
    store = AGFSToolResultArtifactStore(
        _AGFSClient(read_error=RuntimeError("AGFS unavailable")),
    )

    with pytest.raises(ArtifactStoreUnavailableError):
        store.read_metadata("tr_0123456789abcdef", "session", _ctx())


def test_agfs_session_delete_removes_incomplete_artifact_directories():
    client = _AGFSClient(ls_result=[{
        "name": "tr_0123456789abcdef",
        "isDir": True,
    }])
    store = AGFSToolResultArtifactStore(client)

    assert store.delete_session_artifacts("session", _ctx()) == 1
    assert client.removed == [(
        "/local/accounts/acct/user/user/agents/agent/sessions/session/tool-results",
        True,
    )]


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


def test_externalizer_writes_artifact_and_replaces_large_result_with_multires_stub(
    tmp_path,
):
    store = LocalToolResultArtifactStore(tmp_path)
    payload = "ERROR request_id=req-123 backend unavailable\n" * 20

    governed = _externalizer(store).govern_messages(
        [{"role": "toolResult", "content": payload}],
        ctx=_ctx(),
        session_id="session",
        persistence=True,
    )

    assert governed.externalized_count == 1
    assert governed.failed_count == 0
    fields = parse_stub(render_safe_message_content(governed.messages[0]))
    assert fields is not None and fields["version"] == "2"
    assert fields["retained_records"]
    assert fields["coverage"]
    artifact = store.list("session", _ctx())["artifacts"][0]
    _metadata_row, restored = store.read_all(
        artifact["artifact_id"],
        "session",
        _ctx(),
    )
    assert restored == payload
