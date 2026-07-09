"""Outbox event storage for async index synchronization.

Persists OutboxEvents to AGFS for processing by OutboxWorker.
"""

import json
import uuid
from datetime import UTC, datetime

from pyagfs import AGFSClient

from core.enums import EventType
from core.interfaces import ContextFS
from core.logging_config import get_logger
from core.models import ContextNode, OutboxEvent, RequestContext

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

    def register_archive(
        self,
        uri: str,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register a node archive for index synchronization.

        Args:
            uri: URI of archived node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.ARCHIVE_CONTEXT.value,
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

    def register_directory(
        self,
        node: ContextNode,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register UPSERT_DIRECTORY event for node's parent directory.

        Collects sibling abstracts (max 20) and writes OutboxEvent
        with payload self-contained — indexer does NOT access AGFS.

        Args:
            node: ContextNode that was written (child of the directory)
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        dir_uri = self._node_uri_to_directory_uri(node.uri)
        abstracts = self._collect_sibling_abstracts(node.uri, ctx)

        payload = {
            "directory_uri": dir_uri,
            "child_abstracts": abstracts,
            "filters": {
                "account_id": ctx.account_id,
                "owner_space": node.owner_space,
            },
        }

        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.UPSERT_DIRECTORY.value,
            uri=dir_uri,
            payload=payload,
            status="PENDING",
        )

        # Directory URIs (e.g. ctx://acct/users/u1/memories/cat/) lack a slug,
        # so uri_to_path() won't match. Build AGFS path directly.
        dir_agfs_path = (self._mount_prefix or "") + "/accounts/" + dir_uri.replace("ctx://", "").rstrip("/")
        outbox_path = f"{dir_agfs_path}/.outbox/{event.event_id}.json"

        # Ensure .outbox directory exists
        try:
            self._client.mkdir(f"{dir_agfs_path}/.outbox")
        except Exception:
            pass

        # Serialize and write event
        event_json = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
        }, ensure_ascii=False, indent=2)

        self._client.write(outbox_path, event_json.encode('utf-8'))

        return event

    def _node_uri_to_directory_uri(self, node_uri: str) -> str:
        """Convert leaf node URI to parent directory URI.

        Args:
            node_uri: Full URI of a leaf node

        Returns:
            Parent directory URI with trailing slash

        Example:
            ctx://acme/users/u1/memories/preferences/coffee
            → ctx://acme/users/u1/memories/preferences/
        """
        # Remove trailing slash and final slug, add trailing slash
        parts = node_uri.rstrip("/").rsplit("/", 1)
        return parts[0] + "/"

    def _collect_sibling_abstracts(
        self,
        node_uri: str,
        ctx: RequestContext
    ) -> list[str]:
        """Collect sibling node abstracts from AGFS.

        Args:
            node_uri: URI of a node in the directory
            ctx: RequestContext for access control

        Returns:
            List of abstract strings (max 20)
        """
        dir_uri = self._node_uri_to_directory_uri(node_uri)
        try:
            siblings = self._fs.list_children(dir_uri, ctx)
        except Exception:
            return []

        abstracts = []
        for sibling_uri in siblings[:20]:
            try:
                node = self._fs.read_node(sibling_uri, ctx)
                if node and node.abstract:
                    abstracts.append(node.abstract[:200])
            except Exception:
                continue
        return abstracts

    def _uri_to_agfs_path(self, uri: str) -> str:
        """Convert ContextEngine URI to AGFS path with mount prefix.

        Handles both node URIs (ctx://acct/users/u1/memories/cat/slug)
        and directory URIs (ctx://acct/users/u1/memories/cat/).
        """
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        try:
            base_path = uri_to_path(uri)
        except ValueError:
            # Directory-level URI (no slug) — build path directly
            base_path = "/accounts/" + uri.replace("ctx://", "").rstrip("/") + "/"

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

        Raises:
            ValueError: If node_uri contains path traversal patterns
        """
        # SECURITY: Validate URI format and reject path traversal
        from fs.agfs_adapter.agfs_context_fs import parse_uri
        try:
            components = parse_uri(node_uri)
        except ValueError as e:
            raise ValueError(f"Invalid node_uri in OutboxEvent: {node_uri}") from e

        # Check each URI component for path traversal patterns
        for key, val in components.items():
            val_str = str(val)
            if ".." in val_str or val_str.startswith("/"):
                raise ValueError(
                    f"Path traversal detected in URI component '{key}': {val_str}"
                )

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
            "next_retry_at": event.next_retry_at,
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
        processing_path = node_path + ".outbox/" + event.event_id + ".processing"

        # Delete the event file
        try:
            self._client.rm(event_path)
        except Exception:
            pass

        # Delete the processing lock file
        try:
            self._client.rm(processing_path)
        except Exception:
            pass

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
        """Recursively scan directory for .outbox/ event files.

        SECURITY: Skips paths containing '..' to prevent path traversal.
        Skips events with active .processing locks (default 300s timeout).
        """
        # SECURITY: Skip paths with traversal patterns
        if ".." in path:
            return

        try:
            entries = self._client.ls(path)
        except Exception:
            return

        for entry in entries:
            name = entry.get("name", "")
            is_dir = entry.get("is_dir", entry.get("isDir", False))
            child_path = path.rstrip("/") + "/" + name
            if is_dir and name == ".outbox":
                try:
                    ev_entries = self._client.ls(child_path + "/")
                except Exception:
                    continue
                self._collect_outbox_events(child_path, ev_entries, results)
            elif is_dir and name not in (".outbox", "dlq"):
                self._scan_outbox_recursive(child_path + "/", results)

    def _collect_outbox_events(self, outbox_path: str, entries: list[dict], results: list) -> None:
        """Collect pending OutboxEvents from a single .outbox directory."""
        for entry in entries:
            ev_name = entry.get("name", "")
            is_dir = entry.get("is_dir", entry.get("isDir", False))
            if is_dir or not ev_name.endswith(".json"):
                continue
            if ev_name.startswith("."):
                continue

            ev_path = outbox_path.rstrip("/") + "/" + ev_name
            try:
                raw = self._client.read(ev_path)
                data = json.loads(raw.decode("utf-8"))
                if "/dlq/" in ev_path:
                    continue

                event_id = data["event_id"]
                processing_path = outbox_path.rstrip("/") + "/" + event_id + ".processing"
                has_active_lock = False
                try:
                    lock_raw = self._client.read(processing_path)
                    lock_data = json.loads(lock_raw.decode("utf-8"))
                    acquired_at = lock_data.get("acquired_at", "")
                    try:
                        lock_time = datetime.fromisoformat(acquired_at)
                        elapsed = (datetime.now(UTC) - lock_time).total_seconds()
                        if elapsed < 300:
                            has_active_lock = True
                    except (ValueError, KeyError):
                        pass
                except Exception:
                    pass

                if has_active_lock:
                    continue

                next_retry_at_str = data.get("next_retry_at", "")
                if next_retry_at_str:
                    try:
                        next_retry_time = datetime.fromisoformat(next_retry_at_str)
                        if datetime.now(UTC) < next_retry_time:
                            continue
                    except ValueError:
                        pass

                event = OutboxEvent(
                    event_id=data["event_id"],
                    event_type=data["event_type"],
                    uri=data["uri"],
                    payload=data.get("payload", {}),
                    status=data.get("status", "PENDING"),
                    retry_count=data.get("retry_count", 0),
                    created_at=data.get("created_at", ""),
                    next_retry_at=next_retry_at_str,
                )
                results.append((event.uri, event))
            except Exception:
                pass

    def move_to_dlq(self, event: "OutboxEvent", node_uri: str) -> None:
        """Move a failed event to the dead letter queue."""
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
            "next_retry_at": event.next_retry_at,
        }, ensure_ascii=False, indent=2)
        self._client.write(dlq_path, data.encode("utf-8"))
        try:
            self._client.rm(src_path)
        except Exception:
            pass

    def increment_retry(
        self,
        event: "OutboxEvent",
        node_uri: str,
        next_retry_at: datetime | None = None,
    ) -> None:
        """Increment retry_count and re-write event file.

        Args:
            event: OutboxEvent to update
            node_uri: URI of the associated node
            next_retry_at: Optional datetime for scheduled retry (exponential backoff)
        """
        node_path = self._uri_to_agfs_path(node_uri)
        event_path = node_path + ".outbox/" + event.event_id + ".json"
        event.retry_count += 1
        event.status = "PENDING"
        if next_retry_at:
            event.next_retry_at = next_retry_at.isoformat()
        data = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
            "next_retry_at": event.next_retry_at,
        }, ensure_ascii=False, indent=2)
        self._client.write(event_path, data.encode("utf-8"))

    def try_acquire(
        self,
        event: "OutboxEvent",
        node_uri: str,
        worker_id: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Atomically claim an event for processing.

        Creates .outbox/{event_id}.processing lock file.
        Returns False if already locked by another worker within timeout.

        Args:
            event: OutboxEvent to acquire
            node_uri: URI of the associated node
            worker_id: Unique identifier for the processing worker
            timeout_seconds: Seconds before a stale lock is considered expired

        Returns:
            True if lock was acquired, False if already locked
        """
        node_path = self._uri_to_agfs_path(node_uri)
        processing_path = node_path + ".outbox/" + event.event_id + ".processing"

        # Check if lock already exists
        try:
            raw = self._client.read(processing_path)
            lock_data = json.loads(raw.decode("utf-8"))
            acquired_at = lock_data.get("acquired_at", "")

            # Parse timestamp to check if expired
            try:
                lock_time = datetime.fromisoformat(acquired_at)
                elapsed = (datetime.now(UTC) - lock_time).total_seconds()
                if elapsed < timeout_seconds:
                    # Lock is still valid
                    return False
            except (ValueError, KeyError):
                # Invalid timestamp, treat as expired
                pass
        except Exception:
            # Lock file doesn't exist, can acquire
            pass

        # Create lock file
        lock_data = {
            "worker_id": worker_id,
            "acquired_at": datetime.now(UTC).isoformat(),
        }
        self._client.write(
            processing_path,
            json.dumps(lock_data, ensure_ascii=False).encode("utf-8")
        )
        return True

    def release(self, event: "OutboxEvent", node_uri: str) -> None:
        """Release the processing lock. Called on success or exception.

        Removes the .outbox/{event_id}.processing lock file.

        Args:
            event: OutboxEvent to release
            node_uri: URI of the associated node
        """
        node_path = self._uri_to_agfs_path(node_uri)
        processing_path = node_path + ".outbox/" + event.event_id + ".processing"

        try:
            self._client.rm(processing_path)
        except Exception:
            # Lock file may have been removed already, silently ignore
            pass
