"""Core enumerations for ContextEngine.

Phase 0 frozen — these enums are used across all packages.
"""

from enum import Enum


class NodeStatus(str, Enum):
    """Status of a ContextNode in AGFS.

    Corresponds to .meta.json status field.
    Used by Repair Job to determine node health and recovery actions.

    See CLAUDE.md §2 AGFS Directory Spec for write order semantics.
    """
    PENDING = "PENDING"  # Node directory created, write incomplete (steps ①-③ done, ④ not done)
    ACTIVE = "ACTIVE"    # Step ④ complete, node visible for retrieval
    BROKEN = "BROKEN"    # Repair Job detected missing/corrupt files, needs manual intervention


class ContextType(str, Enum):
    """High-level classification of context content.

    Used by TypedQuery for index-level filtering and by Planner for query routing.
    """
    MEMORY = "MEMORY"    # User memories: profile, preferences, entities, events
    SKILL = "SKILL"      # Agent capabilities: skills, tools, procedures
    RESOURCE = "RESOURCE"  # Reference materials, documentation, external resources


class EventType(str, Enum):
    """Types of outbox events for async processing.

    Events are written to .outbox/{event_id}.json and processed by OutboxWorker.
    """
    UPSERT_CONTEXT = "UPSERT_CONTEXT"      # Create or update a ContextNode
    UPSERT_DIRECTORY = "UPSERT_DIRECTORY"  # Update directory summary from children
    DELETE_CONTEXT = "DELETE_CONTEXT"      # Remove a ContextNode
    MOVE_CONTEXT = "MOVE_CONTEXT"          # Rename/relocate a ContextNode
    UPSERT_RELATION = "UPSERT_RELATION"    # Add or update relation edges


class OwnerScope(str, Enum):
    """Scope of node ownership within an account.

    Used for access control and namespace isolation.
    """
    USER = "user"       # User-owned memory: ctx://{account}/users/...
    AGENT = "agent"     # Agent-owned memory: ctx://{account}/agents/...
