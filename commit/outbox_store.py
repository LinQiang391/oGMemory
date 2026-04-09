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


class OutboxStore:
    """Persists OutboxEvents to AGFS .outbox/ directories.

    OutboxEvents are written alongside the node directory:
    /accounts/{...}/node/.outbox/{event_id}.json

    These are processed by OutboxWorker (index package) to sync
    changes to the vector index.
    """

    def __init__(self, client: AGFSClient, fs: ContextFS):
        """Initialize OutboxStore.

        Args:
            client: AGFS client for file operations
            fs: ContextFS for URI to path conversion
        """
        self._client = client
        self._fs = fs

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
        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.UPSERT_CONTEXT.value,
            uri=node.uri,
            payload={
                "context_type": node.context_type,
                "category": node.category,
                "level": node.level,
                "owner_space": node.owner_space,
            },
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

    def _write_event(self, event: OutboxEvent, node_uri: str) -> None:
        """Write OutboxEvent to .outbox/ directory.

        Args:
            event: OutboxEvent to write
            node_uri: URI of the associated node (for path calculation)
        """
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        # Get node path and construct outbox path
        node_path = uri_to_path(node_uri)
        outbox_path = node_path + ".outbox/" + event.event_id + ".json"

        # Ensure .outbox directory exists
        try:
            self._client.mkdir(node_path + ".outbox")
        except Exception:
            pass  # Directory may already exist

        # Write event file
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

    def mark_processing(self, event: OutboxEvent, node_uri: str) -> None:
        """Mark an event as being processed.

        Args:
            event: OutboxEvent to update
            node_uri: URI of the associated node
        """
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        node_path = uri_to_path(node_uri)
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
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        node_path = uri_to_path(node_uri)
        event_path = node_path + ".outbox/" + event.event_id + ".json"

        # Delete the event file
        try:
            self._client.rm(event_path)
        except Exception:
            pass  # May have been processed already
