"""Pure Outbox event builders for durable code RESOURCE mutations.

The builders have no storage/client dependency.  Both SQL and the later AGFS
adapter can use the same payload shape without importing one another.
"""

from __future__ import annotations

import uuid

from core.enums import EventType
from core.models import ContextNode, IndexRecord, OutboxEvent


def build_code_resource_write_event(node: ContextNode) -> OutboxEvent:
    """Build a complete UPSERT event for one code RESOURCE node."""
    from index.index_record_builder import build_index_records

    records_data = [
        {
            "id": record.id,
            "uri": record.uri,
            "level": record.level,
            "text": record.text,
            "filters": record.filters,
            "metadata": record.metadata,
        }
        for record in build_index_records(node)
    ]
    return OutboxEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.UPSERT_CONTEXT.value,
        uri=node.uri,
        payload={"records": records_data},
        status="PENDING",
    )


def build_code_resource_delete_event(uri: str) -> OutboxEvent:
    """Build a self-contained DELETE event for one code RESOURCE URI."""
    normalized_uri = str(uri).rstrip("/")
    l2_uri = normalized_uri + "/content.md"
    ids_to_delete = [
        IndexRecord.generate_id(normalized_uri, 0),
        IndexRecord.generate_id(normalized_uri, 1),
        IndexRecord.generate_id(l2_uri, 2),
    ]
    return OutboxEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.DELETE_CONTEXT.value,
        uri=normalized_uri,
        payload={"ids_to_delete": ids_to_delete},
        status="PENDING",
    )


__all__ = [
    "build_code_resource_delete_event",
    "build_code_resource_write_event",
]
