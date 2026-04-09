"""Tests for retrieval.memory_read_service."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from core.errors import AccessDeniedError
from core.models import ContextNode
from retrieval.memory_read_service import MemoryReadService
from tests.unit.retrieval.conftest import make_ctx


def test_read_returns_placeholder_block_when_fs_missing(caplog: pytest.LogCaptureFixture):
    service = MemoryReadService(fs=None)

    block = service.read("ctx://acct-1/users/u1/memories/profile", make_ctx())

    assert block.uri == "ctx://acct-1/users/u1/memories/profile"
    assert block.level_hit == "L2"
    assert block.content_excerpt is None
    assert "no ContextFS configured" in caplog.text


def test_read_returns_placeholder_block_when_read_node_fails(caplog: pytest.LogCaptureFixture):
    fs = Mock()
    fs.read_node.side_effect = RuntimeError("boom")
    service = MemoryReadService(fs=fs)

    block = service.read("ctx://acct-1/users/u1/memories/profile", make_ctx())

    assert block.content_excerpt is None
    assert "read_node failed" in caplog.text


def test_read_populates_fields_from_node():
    fs = Mock()
    fs.read_node.return_value = ContextNode(
        uri="ctx://acct-1/users/u1/memories/profile",
        context_type="MEMORY",
        category="profile",
        level=2,
        owner_space="user:u1",
        abstract="summary",
        overview="overview",
        content="full content",
    )
    service = MemoryReadService(fs=fs)
    ctx = make_ctx()

    block = service.read("ctx://acct-1/users/u1/memories/profile", ctx)

    assert block.category == "profile"
    assert block.owner_space == "user:u1"
    assert block.content_excerpt == "full content"
    fs.read_node.assert_called_once_with("ctx://acct-1/users/u1/memories/profile", ctx)


def test_check_acl_rejects_cross_account_uri():
    with pytest.raises(AccessDeniedError, match="account mismatch"):
        MemoryReadService._check_acl(
            "ctx://other/users/u1/memories/profile",
            make_ctx(account_id="acct-1"),
        )


def test_check_acl_allows_none_context():
    MemoryReadService._check_acl("ctx://other/users/u1/memories/profile", None)
