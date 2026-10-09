"""Lifecycle recovery uses public entries, deterministic candidates and no model."""

import json
import threading
from copy import deepcopy
from unittest.mock import Mock

import pytest

from commit.write_receipts import WriteNotStartedError
from core.models import RequestContext
from service.extraction_checkpoint import EXTENSION_KEY, RecoveryNeedsConfirmationError
from session.archive_store import SessionArchiveStore
from session.session_manager import SessionManager, SessionOperationInProgress
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import (
    CommitThenFailFS,
    candidate,
    nodes,
    params,
    setup_service,
)


@pytest.fixture
def ctx():
    return RequestContext("lifecycle-test", "reader", "agent", "session", "proof")


def lifecycle_service(fs, ctx, **kwargs):
    service, writer, manager = setup_service(fs, ctx, **kwargs)
    service._get_context_fs = Mock(return_value=fs)
    service._get_archive_store = Mock(return_value=None)
    manager._get_llm = lambda: None
    manager._get_agfs = lambda: fs
    manager._get_archive_store = lambda: None
    return service, writer, manager


def reject_entity_once(writer):
    original = writer._writer.write_candidate
    rejected = False

    def write(entry, ctx):
        nonlocal rejected
        if entry.category == "entity" and not rejected:
            rejected = True
            raise WriteNotStartedError("known rejection before storage")
        return original(entry, ctx)

    writer._writer.write_candidate = write


def seed(manager, ctx, content="I repaired the receiver. " * 40):
    manager.add_message(ctx.session_id, "user", content, ctx, message_id="msg-one")


class InlineThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)






def real_archives(fs, ctx, **kwargs):
    service, writer, manager = lifecycle_service(fs, ctx, **kwargs)
    store = SessionArchiveStore(fs)
    service._get_archive_store = Mock(return_value=store)
    manager._get_archive_store = lambda: store
    manager._archive_store_required = True
    manager.commit_snapshot = SessionManager.commit_snapshot.__get__(manager)
    return service, writer, manager






def configure_inline_merge(manager):
    manager._archive_max_count = 1
    manager._archive_merge_threshold = 2
    original = manager._maybe_merge_archives
    manager._maybe_merge_archives = lambda session_id, ctx: original(session_id, ctx, wait=True)
















class ArchiveCommitThenFailFS(MemoryFS):
    once = True

    def write_node(self, node, ctx):
        super().write_node(node, ctx)
        if node.category == "history" and self.once:
            self.once = False
            raise RuntimeError("archive commit acknowledged only by durable data")












class FailFinalStateOnceFS(MemoryFS):
    failed = False

    def write_node(self, node, ctx):
        if (node.category == "state" and not self.failed and nodes(self, "event")
                and EXTENSION_KEY not in json.loads(node.content).get("runtime_extensions", {})):
            self.failed = True
            raise RuntimeError("final state acknowledgement unavailable")
        super().write_node(node, ctx)
