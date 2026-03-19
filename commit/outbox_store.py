"""Outbox event storage for async index synchronization.

Persists OutboxEvents to AGFS for processing by OutboxWorker.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Optional

from pyagfs import AGFSClient

from core.models import RequestContext, OutboxEvent, ContextNode
from core.enums import EventType
from core.interfaces import ContextFS
from core.logging_config import get_logger

logger = get_logger(__name__)


class OutboxStore:
    """Persists OutboxEvents to AGFS .outbox/ directories.

    OutboxEvents are written alongside the node directory:
    /{mount_prefix}/accounts/{...}/node/.outbox/{event_id}.json

    These are processed by OutboxWorker (index package) to sync
    changes to the vector index.
    """

    def __init__(self, client: AGFSClient, fs: ContextFS, mount_prefix: str = "/local"):
        """Initialize OutboxStore.

        Args:
            client: AGFS client for file operations
            fs: ContextFS for URI to path conversion (must be AGFSContextFS)
            mount_prefix: AGFS mount point prefix (default: "/local")
                         Must match the mount_prefix used by AGFSContextFS

        Raises:
            ValueError: If mount_prefix contains path traversal patterns
        """
        # Security: validate mount_prefix to prevent path traversal attacks
        if ".." in mount_prefix or not mount_prefix.startswith("/"):
            raise ValueError(f"Invalid mount_prefix: must be absolute path without '..': {mount_prefix}")

        self._client = client
        self._fs = fs
        self._mount_prefix = mount_prefix.rstrip("/")

    def register_write(
        self,
        node: ContextNode,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register a node write for index synchronization.

        Called after successful ContextFS.write_node().

        Args:
            node: ContextNode that was written
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        from index.index_record_builder import build_index_records

        # Build IndexRecords from ContextNode (L0/L1/L2)
        index_records = build_index_records(node)

        # Serialize IndexRecords to payload format expected by OutboxWorker
        records_data = [
            {
                "id": r.id,
                "uri": r.uri,
                "level": r.level,
                "text": r.text,
                "filters": r.filters,
                "metadata": r.metadata,
            }
            for r in index_records
        ]

        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.UPSERT_CONTEXT.value,
            uri=node.uri,
            payload={"records": records_data},
            status="PENDING",
        )

        self._write_event(event, node.uri)
        return event

    def register_delete(
        self,
        uri: str,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register a node deletion for index synchronization.

        Args:
            uri: URI of deleted node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.DELETE_CONTEXT.value,
            uri=uri,
            payload={},
            status="PENDING",
        )

        self._write_event(event, uri)
        return event

    def register_relation(
        self,
        edges: list,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register relation edge updates for index synchronization.

        Args:
            edges: List of RelationEdge objects
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        # Use the first edge's URI as the base location
        base_uri = edges[0].from_uri if edges else ""

        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.UPSERT_RELATION.value,
            uri=base_uri,
            payload={
                "edges": [
                    {
                        "from_uri": e.from_uri,
                        "to_uri": e.to_uri,
                        "relation_type": e.relation_type,
                        "weight": e.weight,
                    }
                    for e in edges
                ],
            },
            status="PENDING",
        )

        if base_uri:
            self._write_event(event, base_uri)

        return event

    def _uri_to_agfs_path(self, uri: str) -> str:
        """Convert ContextEngine URI to AGFS path with mount prefix.

        Args:
            uri: ContextEngine URI

        Returns:
            Full AGFS path including mount prefix
        """
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        base_path = uri_to_path(uri)
        if self._mount_prefix:
            return self._mount_prefix + base_path
        return base_path

    def _write_event(self, event: OutboxEvent, node_uri: str) -> None:
        """Write OutboxEvent to .outbox/ directory using atomic write pattern.

        Uses write-to-temp + rename for atomicity:
        1. Write to temporary file .tmp_{event_id}.json
        2. Rename to {event_id}.json (atomic operation)

        Args:
            event: OutboxEvent to write
            node_uri: URI of the associated node (for path calculation)
        """
        # Get node path and construct outbox path (with mount_prefix)
        node_path = self._uri_to_agfs_path(node_uri)
        outbox_dir = node_path + ".outbox"
        event_path = outbox_dir + "/" + event.event_id + ".json"
        temp_path = outbox_dir + "/.tmp_" + event.event_id + ".json"

        # Ensure .outbox directory exists
        try:
            self._client.mkdir(outbox_dir)
        except Exception:
            pass  # Directory may already exist

        # Serialize event
        event_json = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
        }, ensure_ascii=False, indent=2)

        # Write to temporary file first
        try:
            self._client.write(temp_path, event_json.encode('utf-8'))
        except Exception as e:
            # Temp write failed - no state change
            logger.error("Failed to write temp outbox event %s: %s", event.event_id, e)
            raise

        # Rename to final path (atomic if AGFS supports it)
        try:
            # AGFS rename is atomic for same-backend operations
            self._client.rename(temp_path, event_path)
        except Exception:
            # If rename fails, attempt direct write (last resort)
            try:
                self._client.rm(temp_path)
            except Exception:
                pass
            self._client.write(event_path, event_json.encode('utf-8'))

    def mark_processing(self, event: OutboxEvent, node_uri: str) -> None:
        """Mark an event as being processed.

        Args:
            event: OutboxEvent to update
            node_uri: URI of the associated node
        """
        node_path = self._uri_to_agfs_path(node_uri)
        event_path = node_path + ".outbox/" + event.event_id + ".json"

        # Update event status
        event.status = "PROCESSING"
        event_json = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
        }, ensure_ascii=False, indent=2)

        self._client.write(event_path, event_json.encode('utf-8'))

    def mark_done(self, event: OutboxEvent, node_uri: str) -> None:
        """Mark an event as completed and delete it.

        Args:
            event: OutboxEvent to complete
            node_uri: URI of the associated node
        """
        node_path = self._uri_to_agfs_path(node_uri)
        event_path = node_path + ".outbox/" + event.event_id + ".json"

        # Delete the event file
        try:
            self._client.rm(event_path)
        except Exception:
            pass  # May have been processed already

    def list_pending(self, account_id: str) -> list[tuple[str, "OutboxEvent"]]:
        """Scan AGFS for all pending OutboxEvents under an account.

        Scans /accounts/{account_id}/**/.outbox/*.json (excluding dlq/).
        Returns list of (node_uri, OutboxEvent) tuples.
        """
        results = []
        prefix = self._mount_prefix + f"/accounts/{account_id}/"
        try:
            self._scan_outbox_recursive(prefix, results)
        except Exception:
            pass
        return results

    def _scan_outbox_recursive(self, path: str, results: list) -> None:
        """Recursively scan directory for .outbox/ event files."""
        try:
            entries = self._client.ls(path)
        except Exception:
            return
        for entry in entries:
            name = entry.get("name", "")
            is_dir = entry.get("is_dir", False)
            child_path = path.rstrip("/") + "/" + name
            if is_dir and name == ".outbox":
                try:
                    ev_entries = self._client.ls(child_path + "/")
                except Exception:
                    continue
                for ev in ev_entries:
                    ev_name = ev.get("name", "")
                    if ev.get("is_dir") or not ev_name.endswith(".json"):
                        continue
                    if ev_name.startswith("."):
                        continue
                    ev_path = child_path + "/" + ev_name
                    try:
                        raw = self._client.read(ev_path)
                        data = json.loads(raw.decode("utf-8"))
                        if "/dlq/" in ev_path:
                            continue
                        event = OutboxEvent(
                            event_id=data["event_id"],
                            event_type=data["event_type"],
                            uri=data["uri"],
                            payload=data.get("payload", {}),
                            status=data.get("status", "PENDING"),
                            retry_count=data.get("retry_count", 0),
                            created_at=data.get("created_at", ""),
                        )
                        results.append((event.uri, event))
                    except Exception:
                        pass
            elif is_dir and name not in (".outbox", "dlq"):
                self._scan_outbox_recursive(child_path + "/", results)

    def move_to_dlq(self, event: "OutboxEvent", node_uri: str) -> None:
        """Move a failed event to the dead letter queue."""
        import json
        node_path = self._uri_to_agfs_path(node_uri)
        src_path = node_path + ".outbox/" + event.event_id + ".json"
        dlq_dir = node_path + ".outbox/dlq/"
        dlq_path = dlq_dir + event.event_id + ".json"
        try:
            self._client.mkdir(dlq_dir.rstrip("/"))
        except Exception:
            pass
        event.status = "FAILED"
        data = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
        }, ensure_ascii=False, indent=2)
        self._client.write(dlq_path, data.encode("utf-8"))
        try:
            self._client.rm(src_path)
        except Exception:
            pass

    def increment_retry(self, event: "OutboxEvent", node_uri: str) -> None:
        """Increment retry_count and re-write event file."""
        import json
        node_path = self._uri_to_agfs_path(node_uri)
        event_path = node_path + ".outbox/" + event.event_id + ".json"
        event.retry_count += 1
        event.status = "PENDING"
        data = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
        }, ensure_ascii=False, indent=2)
        self._client.write(event_path, data.encode("utf-8"))
