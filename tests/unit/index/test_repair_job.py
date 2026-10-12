"""Unit tests for RepairJob.

Tests verify all branch logic paths corresponding to AGFS write order:
- .meta.json missing + content.md exists → Repair
- .meta.json missing + content.md missing → Skip
- status=PENDING + files complete → Activate
- status=PENDING + files missing → Mark BROKEN
- status=ACTIVE → Skip
- status=BROKEN → Skip
"""

import json

import pytest

from index.repair_job import (
    RepairJob,
    RepairReport,
    RepairJobSummary,
    MemoryAGFSSnapshot,
    MemoryOutboxEventSink,
    _REQUIRED_FILES,
)


class TestMemoryAGFSSnapshot:
    """Tests for MemoryAGFSSnapshot test double."""

    def test_add_and_list_nodes(self):
        snapshot = MemoryAGFSSnapshot()
        snapshot.add_node("ctx://test/users/u1/memories/profile", {})

        nodes = snapshot.list_nodes_under_account("test")
        assert len(nodes) == 1
        assert nodes[0] == "ctx://test/users/u1/memories/profile"

    def test_add_and_read_file(self):
        snapshot = MemoryAGFSSnapshot()
        snapshot.add_node("ctx://test/users/u1/memories/profile", {
            "content.md": "test content",
        })

        content = snapshot.read_file("ctx://test/users/u1/memories/profile", "content.md")
        assert content == "test content"

    def test_read_nonexistent_file_returns_none(self):
        snapshot = MemoryAGFSSnapshot()
        snapshot.add_node("ctx://test/users/u1/memories/profile", {})

        content = snapshot.read_file("ctx://test/users/u1/memories/profile", "missing.txt")
        assert content is None

    def test_write_file(self):
        snapshot = MemoryAGFSSnapshot()
        snapshot.add_node("ctx://test/users/u1/memories/profile", {})
        snapshot.write_file("ctx://test/users/u1/memories/profile", "test.txt", "hello")

        content = snapshot.read_file("ctx://test/users/u1/memories/profile", "test.txt")
        assert content == "hello"

    def test_list_files(self):
        snapshot = MemoryAGFSSnapshot()
        snapshot.add_node("ctx://test/users/u1/memories/profile", {
            "content.md": "content",
            ".meta.json": "{}",
        })

        files = snapshot.list_files("ctx://test/users/u1/memories/profile")
        assert set(files) == {"content.md", ".meta.json"}


class TestMemoryOutboxEventSink:
    """Tests for MemoryOutboxEventSink test double."""

    def test_create_and_retrieve_event(self):
        sink = MemoryOutboxEventSink()

        from core.models import OutboxEvent
        event = OutboxEvent(
            event_id="test-event",
            event_type="UPSERT_CONTEXT",
            uri="ctx://test/users/u1/memories/profile",
            payload={},
            status="PENDING",
        )

        sink.create_event(event)
        assert len(sink.events) == 1
        assert sink.events[0].event_id == "test-event"


class TestRepairJobMissingMetadata:
    """Tests for .meta.json missing branch."""

    def test_meta_missing_with_content_repaired(self):
        """Node with content.md but missing .meta.json should be repaired."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        # Add node with content.md but no .meta.json (step ① complete, ④ not)
        uri = "ctx://acme/users/alice/memories/profile"
        snapshot.add_node(uri, {
            "content.md": "full content",
            ".abstract.md": "abstract",
            ".overview.md": "overview",
            ".relations.json": "[]",
            # .meta.json is missing
        })

        summary = job.scan_account("acme")

        assert summary.nodes_repaired == 1
        assert summary.nodes_activated == 0
        assert summary.nodes_marked_broken == 0

        # Check .meta.json was created
        meta_content = snapshot.read_file(uri, ".meta.json")
        assert meta_content is not None
        meta = json.loads(meta_content)
        assert meta["status"] == "ACTIVE"

        # Check OutboxEvent was created
        assert len(sink.events) == 1
        assert sink.events[0].event_type == "UPSERT_CONTEXT"

    def test_meta_missing_without_content_skipped(self):
        """Node without content.md or .meta.json should be skipped."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        # Add node with no content.md and no .meta.json (step ① not complete)
        uri = "ctx://acme/users/alice/memories/profile"
        snapshot.add_node(uri, {
            ".abstract.md": "abstract",
            # content.md is missing
            # .meta.json is missing
        })

        summary = job.scan_account("acme")

        assert summary.nodes_repaired == 0
        assert summary.nodes_skipped == 1

        # No .meta.json should be created
        meta_content = snapshot.read_file(uri, ".meta.json")
        assert meta_content is None

        # No OutboxEvent should be created
        assert len(sink.events) == 0

    def test_meta_missing_empty_directory_skipped(self):
        """Completely empty directory should be skipped."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        snapshot.add_node(uri, {})

        summary = job.scan_account("acme")

        assert summary.nodes_skipped == 1
        assert len(sink.events) == 0


class TestRepairJobPendingNode:
    """Tests for status=PENDING branch."""

    def test_pending_with_all_files_activated(self):
        """PENDING node with all files should be activated."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        meta = {
            "uri": uri,
            "status": "PENDING",
            "category": "profile",
        }
        snapshot.add_node(uri, {
            "content.md": "content",
            ".relations.json": "[]",
            ".abstract.md": "abstract",
            ".overview.md": "overview",
            ".meta.json": json.dumps(meta),
        })

        summary = job.scan_account("acme")

        assert summary.nodes_activated == 1
        assert summary.nodes_marked_broken == 0

        # Check status was updated
        meta_content = snapshot.read_file(uri, ".meta.json")
        updated_meta = json.loads(meta_content)
        assert updated_meta["status"] == "ACTIVE"

        # Check OutboxEvent was created
        assert len(sink.events) == 1

    def test_pending_missing_files_marked_broken(self):
        """PENDING node with missing files should be marked BROKEN."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        meta = {
            "uri": uri,
            "status": "PENDING",
            "category": "profile",
        }
        snapshot.add_node(uri, {
            "content.md": "content",
            ".relations.json": "[]",
            ".abstract.md": "abstract",
            # .overview.md is missing
            ".meta.json": json.dumps(meta),
        })

        summary = job.scan_account("acme")

        assert summary.nodes_activated == 0
        assert summary.nodes_marked_broken == 1

        # Check status was updated to BROKEN
        meta_content = snapshot.read_file(uri, ".meta.json")
        updated_meta = json.loads(meta_content)
        assert updated_meta["status"] == "BROKEN"

        # No OutboxEvent for broken nodes
        assert len(sink.events) == 0

    def test_pending_missing_multiple_files(self):
        """PENDING node missing multiple files should report all missing."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        meta = {
            "uri": uri,
            "status": "PENDING",
            "category": "profile",
        }
        snapshot.add_node(uri, {
            ".meta.json": json.dumps(meta),
            # Only .meta.json exists, all others missing
        })

        summary = job.scan_account("acme")

        assert summary.nodes_marked_broken == 1

        # Get the repair report by checking node status
        meta_content = snapshot.read_file(uri, ".meta.json")
        updated_meta = json.loads(meta_content)
        assert updated_meta["status"] == "BROKEN"


class TestRepairJobActiveNode:
    """Tests for status=ACTIVE branch."""

    def test_active_node_skipped(self):
        """ACTIVE node should be skipped (normal case)."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        meta = {
            "uri": uri,
            "status": "ACTIVE",
            "category": "profile",
        }
        snapshot.add_node(uri, {
            "content.md": "content",
            ".relations.json": "[]",
            ".abstract.md": "abstract",
            ".overview.md": "overview",
            ".meta.json": json.dumps(meta),
        })

        summary = job.scan_account("acme")

        assert summary.nodes_skipped == 1
        assert summary.nodes_repaired == 0
        assert summary.nodes_activated == 0

        # No modifications
        meta_content = snapshot.read_file(uri, ".meta.json")
        assert json.loads(meta_content)["status"] == "ACTIVE"

        # No OutboxEvent
        assert len(sink.events) == 0


class TestRepairJobBrokenNode:
    """Tests for status=BROKEN branch."""

    def test_broken_node_skipped(self):
        """BROKEN node should be skipped (already marked)."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        meta = {
            "uri": uri,
            "status": "BROKEN",
            "category": "profile",
        }
        snapshot.add_node(uri, {
            ".meta.json": json.dumps(meta),
        })

        summary = job.scan_account("acme")

        assert summary.nodes_already_broken == 1
        assert summary.nodes_repaired == 0
        assert summary.nodes_activated == 0

        # No modifications
        meta_content = snapshot.read_file(uri, ".meta.json")
        assert json.loads(meta_content)["status"] == "BROKEN"

        # No OutboxEvent
        assert len(sink.events) == 0


class TestRepairJobEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_invalid_meta_json_marked_broken(self):
        """Node with invalid JSON in .meta.json should be marked BROKEN."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        snapshot.add_node(uri, {
            ".meta.json": "not valid json {{{",
        })

        summary = job.scan_account("acme")

        assert summary.nodes_marked_broken == 1

    def test_unreadable_meta_creates_new(self):
        """Node with missing .meta.json but all required files should be marked ACTIVE."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        # SECURITY: All required files from steps 1-3 must be present for ACTIVE
        snapshot.add_node(uri, {
            "content.md": "content",
            ".relations.json": "[]",
            ".abstract.md": "abstract",
            ".overview.md": "overview",
            # .meta.json doesn't exist
        })

        job.scan_account("acme")

        meta_content = snapshot.read_file(uri, ".meta.json")
        assert meta_content is not None

        meta = json.loads(meta_content)
        assert "uri" in meta
        assert "status" in meta
        assert meta["status"] == "ACTIVE"

    def test_unknown_status_marked_broken(self):
        """Node with unknown status should be marked BROKEN."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        uri = "ctx://acme/users/alice/memories/profile"
        meta = {
            "uri": uri,
            "status": "UNKNOWN_STATUS",
            "category": "profile",
        }
        snapshot.add_node(uri, {
            ".meta.json": json.dumps(meta),
        })

        summary = job.scan_account("acme")

        assert summary.nodes_marked_broken == 1

    def test_multiple_nodes_mixed_states(self):
        """Test scan with multiple nodes in different states."""
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        # Node 1: Active (skip)
        uri1 = "ctx://acme/users/alice/memories/profile"
        snapshot.add_node(uri1, {
            "content.md": "content",
            ".relations.json": "[]",
            ".abstract.md": "abstract",
            ".overview.md": "overview",
            ".meta.json": json.dumps({"uri": uri1, "status": "ACTIVE"}),
        })

        # Node 2: Missing meta with all required files (repair -> ACTIVE)
        uri2 = "ctx://acme/users/bob/memories/profile"
        snapshot.add_node(uri2, {
            "content.md": "content",
            ".relations.json": "[]",
            ".abstract.md": "abstract",
            ".overview.md": "overview",
        })

        # Node 3: Pending with all files (activate)
        uri3 = "ctx://acme/users/charlie/memories/profile"
        snapshot.add_node(uri3, {
            "content.md": "content",
            ".relations.json": "[]",
            ".abstract.md": "abstract",
            ".overview.md": "overview",
            ".meta.json": json.dumps({"uri": uri3, "status": "PENDING"}),
        })

        summary = job.scan_account("acme")

        assert summary.total_nodes_scanned == 3
        assert summary.nodes_repaired == 1
        assert summary.nodes_activated == 1
        assert summary.nodes_skipped == 1

        # Two OutboxEvents (repaired + activated)
        assert len(sink.events) == 2


class TestRepairJobSummary:
    """Tests for RepairJobSummary."""

    def test_summary_counts_all_actions(self):
        snapshot = MemoryAGFSSnapshot()
        sink = MemoryOutboxEventSink()
        job = RepairJob(snapshot, sink)

        # Add one node of each type
        # 1. Active
        uri1 = "ctx://test/users/u1/memories/profile"
        snapshot.add_node(uri1, {
            "content.md": "c",
            ".relations.json": "[]",
            ".abstract.md": "a",
            ".overview.md": "o",
            ".meta.json": json.dumps({"uri": uri1, "status": "ACTIVE"}),
        })

        # 2. Broken
        uri2 = "ctx://test/users/u2/memories/profile"
        snapshot.add_node(uri2, {
            ".meta.json": json.dumps({"uri": uri2, "status": "BROKEN"}),
        })

        # 3. Pending incomplete
        uri3 = "ctx://test/users/u3/memories/profile"
        snapshot.add_node(uri3, {
            ".meta.json": json.dumps({"uri": uri3, "status": "PENDING"}),
        })

        # 4. Missing meta with all required files (repair -> ACTIVE)
        uri4 = "ctx://test/users/u4/memories/profile"
        snapshot.add_node(uri4, {
            "content.md": "content",
            ".relations.json": "[]",
            ".abstract.md": "abstract",
            ".overview.md": "overview",
        })

        # 5. Pending complete
        uri5 = "ctx://test/users/u5/memories/profile"
        snapshot.add_node(uri5, {
            "content.md": "c",
            ".relations.json": "[]",
            ".abstract.md": "a",
            ".overview.md": "o",
            ".meta.json": json.dumps({"uri": uri5, "status": "PENDING"}),
        })

        summary = job.scan_account("test")

        assert summary.total_nodes_scanned == 5
        assert summary.nodes_skipped == 1  # Active
        assert summary.nodes_already_broken == 1  # Broken
        assert summary.nodes_marked_broken == 1  # Pending incomplete
        assert summary.nodes_repaired == 1  # Missing meta
        assert summary.nodes_activated == 1  # Pending complete


class TestRequiredFiles:
    """Tests for _REQUIRED_FILES constant."""

    def test_required_files_contains_all_expected(self):
        """Verify _REQUIRED_FILES contains all expected files."""
        expected = {"content.md", ".relations.json", ".abstract.md", ".overview.md", ".meta.json"}
        assert set(_REQUIRED_FILES) == expected
