from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import RequestContext
from service.tool_output import ToolOutputFilterService, ToolOutputRepository


class FakeContextFS:
    """Unit double for ContextFS; no Outbox or index implementation is present."""

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}
        self.write_count = 0
        self.delete_count = 0
        self.list_count = 0
        self.fail_list_on_call: int | None = None

    def exists(self, uri, ctx):
        del ctx
        return uri in self.nodes

    def write_node(self, node, ctx):
        del ctx
        self.write_count += 1
        self.nodes[node.uri] = node

    def read_node(self, uri, ctx):
        del ctx
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        return self.nodes[uri]

    def list_children(self, uri, ctx):
        del ctx
        self.list_count += 1
        if self.fail_list_on_call == self.list_count:
            raise OSError("retention listing failed")
        prefix = uri.rstrip("/") + "/"
        return [
            child_uri
            for child_uri in self.nodes
            if child_uri.startswith(prefix) and "/" not in child_uri[len(prefix) :]
        ]

    def delete_node(self, uri, ctx):
        del ctx
        if uri not in self.nodes:
            raise NodeNotFoundError(uri)
        self.delete_count += 1
        del self.nodes[uri]


def _ctx(*, account: str = "acct", agent: str = "agent") -> RequestContext:
    return RequestContext(
        account_id=account,
        user_id="user",
        agent_id=agent,
        session_id="session",
        trace_id="trace",
    )


def _cfg(**overrides):
    values = {
        "tool_output_filter_enabled": True,
        "tool_output_filter_read_enabled": True,
        "tool_output_filter_bash_enabled": True,
        "tool_output_recovery_max_chars": 20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_repository_persists_agent_scoped_no_index_node_without_outbox():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    digest = "a" * 64

    uri = repository.store_original(
        _ctx(),
        slug="tool_output_read_" + "a" * 32,
        content_digest=digest,
        original="complete original",
        filtered="short",
        tool_name="Read",
        strategy="read_head_tail",
        metadata={"original_lines": 2},
    )

    node = fs.nodes[uri]
    assert uri == (
        "ctx://acct/agents/agent/memories/tool_outputs/"
        "tool_output_read_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    assert node.context_type == "SKILL"
    assert node.category == "tool_outputs"
    assert node.level == 4
    assert node.owner_space == "agent:agent"
    assert node.content == "complete original"
    assert node.metadata["content_digest"] == digest
    assert node.metadata["indexing_disabled"] is True
    assert node.metadata["original_bytes"] == len(b"complete original")
    assert node.metadata["stored_at"]
    assert fs.write_count == 1


def test_repository_is_idempotent_but_rejects_digest_collision():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    ctx = _ctx()
    kwargs = {
        "slug": "tool_output_bash_" + "b" * 32,
        "content_digest": "b" * 64,
        "original": "same original",
        "filtered": "same",
        "tool_name": "Bash",
        "strategy": "bash_passthrough_below_min_size",
    }

    first = repository.store_original(ctx, **kwargs)
    second = repository.store_original(ctx, **kwargs)

    assert first == second
    assert fs.write_count == 2
    with pytest.raises(ValueError, match="identity collision"):
        repository.store_original(
            ctx,
            **{**kwargs, "content_digest": "c" * 64, "original": "different"},
        )


def test_duplicate_store_refreshes_retention_timestamp_before_next_write():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs, max_retained_nodes=2)
    ctx = _ctx()
    first_kwargs = {
        "slug": "tool_output_read_" + "a" * 32,
        "content_digest": "a" * 64,
        "original": "first original",
        "filtered": "short",
        "tool_name": "Read",
        "strategy": "read_head_tail",
    }
    second_kwargs = {
        "slug": "tool_output_read_" + "b" * 32,
        "content_digest": "b" * 64,
        "original": "second original",
        "filtered": "short",
        "tool_name": "Read",
        "strategy": "read_head_tail",
    }

    first_uri = repository.store_original(ctx, **first_kwargs)
    second_uri = repository.store_original(ctx, **second_kwargs)
    fs.nodes[first_uri].metadata["stored_at"] = "2020-01-01T00:00:00+00:00"
    fs.nodes[second_uri].metadata["stored_at"] = "2021-01-01T00:00:00+00:00"

    repository.store_original(ctx, **first_kwargs)
    refreshed_at = fs.nodes[first_uri].metadata["stored_at"]
    assert refreshed_at != "2020-01-01T00:00:00+00:00"

    third_uri = repository.store_original(
        ctx,
        slug="tool_output_read_" + "c" * 32,
        content_digest="c" * 64,
        original="third original",
        filtered="short",
        tool_name="Read",
        strategy="read_head_tail",
    )

    assert first_uri in fs.nodes
    assert second_uri not in fs.nodes
    assert third_uri in fs.nodes


def test_repository_rejects_oversized_multibyte_original_before_write():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    original = "界" * ((1024 * 1024 // 3) + 1)

    with pytest.raises(ValueError, match="1048576-byte limit"):
        repository.store_original(
            _ctx(),
            slug="tool_output_read_" + "9" * 32,
            content_digest="9" * 64,
            original=original,
            filtered="short",
            tool_name="Read",
            strategy="read_head_tail",
        )

    assert fs.write_count == 0
    assert not fs.nodes


def test_repository_evicts_oldest_per_agent_and_keeps_default_bound():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    ctx = _ctx()
    other_ctx = _ctx(agent="other")
    other_uri = repository.store_original(
        other_ctx,
        slug="tool_output_read_" + "f" * 32,
        content_digest="f" * 64,
        original="other agent",
        filtered="short",
        tool_name="Read",
        strategy="read_head_tail",
    )

    uris = []
    for index in range(65):
        digest = f"{index:064x}"
        uris.append(
            repository.store_original(
                ctx,
                slug=f"tool_output_read_{index:032x}",
                content_digest=digest,
                original=f"original {index}",
                filtered="short",
                tool_name="Read",
                strategy="read_head_tail",
            )
        )

    prefix = "ctx://acct/agents/agent/memories/tool_outputs/"
    retained = [uri for uri in fs.nodes if uri.startswith(prefix)]
    assert len(retained) == 64
    assert uris[0] not in fs.nodes
    assert uris[-1] in fs.nodes
    assert other_uri in fs.nodes


def test_repository_retention_does_not_evict_node_written_after_snapshot(monkeypatch):
    fs = FakeContextFS()
    ctx = _ctx()
    repository_a = ToolOutputRepository(fs, max_retained_nodes=2)
    repository_b = ToolOutputRepository(fs, max_retained_nodes=2)

    first_uri = repository_a.store_original(
        ctx,
        slug="tool_output_read_" + "1" * 32,
        content_digest="1" * 64,
        original="first original",
        filtered="short",
        tool_name="Read",
        strategy="read_head_tail",
    )
    second_uri = repository_a.store_original(
        ctx,
        slug="tool_output_read_" + "2" * 32,
        content_digest="2" * 64,
        original="second original",
        filtered="short",
        tool_name="Read",
        strategy="read_head_tail",
    )
    # Make the seeded records unambiguously older than either interleaved write.
    fs.nodes[first_uri].metadata["stored_at"] = "2020-01-01T00:00:00+00:00"
    fs.nodes[second_uri].metadata["stored_at"] = "2020-01-01T00:00:01+00:00"

    concurrent: dict[str, str] = {}

    snapshot = repository_a._retention_keep_set

    def snapshot_then_write(*args, **kwargs):
        result = snapshot(*args, **kwargs)
        concurrent["uri"] = repository_b.store_original(
            ctx,
            slug="tool_output_read_" + "b" * 32,
            content_digest="b" * 64,
            original="concurrent original",
            filtered="short",
            tool_name="Read",
            strategy="read_head_tail",
        )
        return result

    # Pause repository A immediately after its stale snapshot, let repository
    # B finish a complete write/retention cycle, then resume A.  A must delete
    # only URIs from its snapshot, never B's newly returned URI.
    monkeypatch.setattr(repository_a, "_retention_keep_set", snapshot_then_write)
    repository_a.store_original(
        ctx,
        slug="tool_output_read_" + "a" * 32,
        content_digest="a" * 64,
        original="primary original",
        filtered="short",
        tool_name="Read",
        strategy="read_head_tail",
    )

    concurrent_uri = concurrent["uri"]
    assert concurrent_uri in fs.nodes
    assert repository_a.read_original(ctx, concurrent_uri) == "concurrent original"


def test_repository_retention_serializes_refresh_against_stale_snapshot(monkeypatch):
    """A refresh that starts during pruning must not lose its returned URI."""

    fs = FakeContextFS()
    ctx = _ctx()
    repository_a = ToolOutputRepository(fs, max_retained_nodes=1)
    repository_b = ToolOutputRepository(fs, max_retained_nodes=1)

    original_kwargs = {
        "slug": "tool_output_read_" + "1" * 32,
        "content_digest": "1" * 64,
        "original": "first original",
        "filtered": "short",
        "tool_name": "Read",
        "strategy": "read_head_tail",
    }
    first_uri = repository_a.store_original(ctx, **original_kwargs)
    fs.nodes[first_uri].metadata["stored_at"] = "2020-01-01T00:00:00+00:00"

    snapshot_ready = threading.Event()
    release_snapshot = threading.Event()
    refresh_attempted = threading.Event()
    errors: list[BaseException] = []

    original_keep_set = repository_b._retention_keep_set

    def paused_keep_set(*args, **kwargs):
        result = original_keep_set(*args, **kwargs)
        snapshot_ready.set()
        if not release_snapshot.wait(timeout=5):
            raise AssertionError("timed out waiting to release retention snapshot")
        return result

    monkeypatch.setattr(repository_b, "_retention_keep_set", paused_keep_set)
    original_scope = repository_a._tool_output_mutation_scope

    @contextmanager
    def tracked_scope(scope_ctx):
        refresh_attempted.set()
        with original_scope(scope_ctx) as ensure_owned:
            yield ensure_owned

    monkeypatch.setattr(repository_a, "_tool_output_mutation_scope", tracked_scope)

    def store_concurrent() -> None:
        try:
            repository_b.store_original(
                ctx,
                slug="tool_output_read_" + "2" * 32,
                content_digest="2" * 64,
                original="second original",
                filtered="short",
                tool_name="Read",
                strategy="read_head_tail",
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def refresh_existing() -> None:
        try:
            repository_a.store_original(ctx, **original_kwargs)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    writer = threading.Thread(target=store_concurrent)
    refresher = threading.Thread(target=refresh_existing)
    writer.start()
    assert snapshot_ready.wait(timeout=5)
    refresher.start()
    assert refresh_attempted.wait(timeout=5)
    # B holds the shared account/Agent mutation scope while its stale snapshot
    # is paused.  A must therefore run only after B finishes its deletion.
    release_snapshot.set()
    writer.join(timeout=5)
    refresher.join(timeout=5)

    assert not writer.is_alive()
    assert not refresher.is_alive()
    assert not errors
    assert first_uri in fs.nodes
    assert repository_a.read_original(ctx, first_uri) == "first original"


def test_repository_rolls_back_new_node_when_post_write_retention_fails():
    fs = FakeContextFS()
    fs.fail_list_on_call = 1
    repository = ToolOutputRepository(fs)

    with pytest.raises(OSError, match="retention listing failed"):
        repository.store_original(
            _ctx(),
            slug="tool_output_read_" + "8" * 32,
            content_digest="8" * 64,
            original="original",
            filtered="short",
            tool_name="Read",
            strategy="read_head_tail",
        )

    assert fs.write_count == 1
    assert fs.delete_count == 1
    assert not fs.nodes


def test_recovery_requires_exact_current_agent_and_account():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    owner_ctx = _ctx()
    uri = repository.store_original(
        owner_ctx,
        slug="tool_output_read_" + "d" * 32,
        content_digest="d" * 64,
        original="original",
        filtered="short",
        tool_name="Read",
        strategy="read_head_tail",
    )

    assert repository.read_original(owner_ctx, uri) == "original"
    with pytest.raises(AccessDeniedError, match="agent mismatch"):
        repository.read_original(_ctx(agent="other"), uri)
    with pytest.raises(AccessDeniedError, match="account mismatch"):
        repository.read_original(_ctx(account="other"), uri)


@pytest.mark.parametrize("requested", [None, "bad", 0, -1, False])
def test_recovery_invalid_or_nonpositive_budget_uses_server_cap(requested):
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    service = ToolOutputFilterService(repository, _cfg())
    ctx = _ctx()
    uri = repository.store_original(
        ctx,
        slug="tool_output_bash_" + "e" * 32,
        content_digest="e" * 64,
        original="0123456789" * 10,
        filtered="short",
        tool_name="Bash",
        strategy="bash_test_runner",
    )

    result = service.recover_original(
        {"memory_uri": uri, "max_chars": requested},
        ctx,
    )

    assert result["ok"] is True
    assert result["max_chars"] == 20
    assert result["truncated"] is True
    assert result["original_chars"] == 100
    assert result["returned_chars"] <= 20
    assert result["diagnostics"] == {
        "state": "recovered",
        "reason": None,
        "input_chars": 100,
        "output_chars": result["returned_chars"],
        "truncated": True,
        "budget": {"configured_max_chars": 20, "effective_max_chars": 20},
    }


def test_recovery_client_budget_is_bounded_by_server_cap():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    service = ToolOutputFilterService(repository, _cfg())
    ctx = _ctx()
    uri = repository.store_original(
        ctx,
        slug="tool_output_bash_" + "f" * 32,
        content_digest="f" * 64,
        original="x" * 100,
        filtered="short",
        tool_name="Bash",
        strategy="bash_test_runner",
    )

    small = service.recover_original({"memory_uri": uri, "max_chars": 8}, ctx)
    oversized = service.recover_original({"memory_uri": uri, "max_chars": 1000}, ctx)

    assert small["max_chars"] == 8
    assert oversized["max_chars"] == 20
    assert small["returned_chars"] <= 8
    assert oversized["returned_chars"] <= 20


def test_recovery_cap_includes_omission_marker_within_server_budget():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    service = ToolOutputFilterService(repository, _cfg(tool_output_recovery_max_chars=80))
    ctx = _ctx()
    uri = repository.store_original(
        ctx,
        slug="tool_output_bash_" + "1" * 32,
        content_digest="1" * 64,
        original="x" * 200,
        filtered="short",
        tool_name="Bash",
        strategy="bash_test_runner",
    )

    result = service.recover_original({"memory_uri": uri}, ctx)

    assert result["truncated"] is True
    assert result["returned_chars"] <= 80
    assert "original output clipped" in result["original"]


def test_missing_original_is_a_business_result():
    service = ToolOutputFilterService(ToolOutputRepository(FakeContextFS()), _cfg())
    uri = (
        "ctx://acct/agents/agent/memories/tool_outputs/"
        "tool_output_read_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )

    result = service.recover_original({"memory_uri": uri}, _ctx())

    assert result == {
        "ok": False,
        "error": "original output not found",
        "memory_uri": uri,
        "diagnostics": {
            "state": "recovery_failed",
            "reason": "not_found_or_not_visible",
            "input_chars": None,
            "output_chars": None,
            "truncated": False,
            "budget": {"configured_max_chars": 20, "effective_max_chars": None},
        },
    }
    diagnostics = json.dumps(result["diagnostics"], sort_keys=True)
    assert uri not in diagnostics
    assert "original output not found" not in diagnostics


def test_recovery_keeps_foreign_owner_failure_for_the_public_visibility_adapter():
    fs = FakeContextFS()
    repository = ToolOutputRepository(fs)
    service = ToolOutputFilterService(repository, _cfg())
    uri = repository.store_original(
        _ctx(),
        slug="tool_output_read_" + "c" * 32,
        content_digest="c" * 64,
        original="private original content",
        filtered="short",
        tool_name="Read",
        strategy="read_head_tail",
    )

    with pytest.raises(AccessDeniedError):
        service.recover_original({"memory_uri": uri}, _ctx(account="other"))
