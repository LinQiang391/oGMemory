"""Session archive storage using AGFS.

Persists compressed session snapshots to AGFS with the following structure:
ctx://{account}/sessions/{session_id}/history/{archive_id}/
  ├── overview.md
  ├── abstract.md
  ├── messages.json
  └── .meta.json
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Optional

from core.interfaces import ContextFS
from core.models import RequestContext, ContextNode
from session.models import ArchiveEntry, ArchiveWriteResult


def session_uri_to_path(account_id: str, session_id: str, archive_id: str) -> str:
    """Convert session archive identifiers to AGFS physical path.

    Args:
        account_id: Account identifier
        session_id: Session identifier
        archive_id: Archive identifier

    Returns:
        AGFS path: /accounts/{account_id}/sessions/{session_id}/history/{archive_id}/
    """
    path = str(
        PurePosixPath("/accounts") / account_id / "sessions" / session_id / "history" / archive_id
    )
    return path if path.endswith("/") else path + "/"


def _is_merged_metadata(metadata: dict | None) -> bool:
    return str((metadata or {}).get("status", "")).upper() == "MERGED"


class SessionArchiveStore:
    """AGFS-backed storage for session archives.

    Provides methods to write, list, and read session archives.
    Uses ContextFS interface for abstraction over the storage layer.
    """

    def __init__(self, fs: ContextFS):
        """Initialize the archive store.

        Args:
            fs: ContextFS implementation for storage operations
        """
        self.fs = fs
        self._metadata_lock = threading.RLock()

    def _archive_uri(self, account_id: str, session_id: str, archive_id: str) -> str:
        return f"ctx://{account_id}/sessions/{session_id}/history/{archive_id}"

    def write_archive(
        self,
        session_id: str,
        overview: str,
        abstract: str,
        messages: list[dict],
        ctx: RequestContext,
        archive_id: str | None = None,
        metadata: dict | None = None,
    ) -> ArchiveWriteResult:
        """Write a session archive to AGFS.

        Creates a new archive entry with the following structure:
        ctx://{account}/sessions/{session_id}/history/{archive_id}/
          ├── overview.md
          ├── abstract.md
          ├── messages.json
          └── .meta.json

        Args:
            session_id: Session identifier
            overview: Structured overview of the session
            abstract: Brief summary (≤100 chars)
            messages: Full message history
            ctx: Request context for access control
            archive_id: Optional archive ID (auto-generated if None)

        Returns:
            ArchiveWriteResult with success status and URI
        """
        if archive_id is None:
            # Generate timestamp-based archive ID
            archive_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Build URI for this archive
        uri = f"ctx://{ctx.account_id}/sessions/{session_id}/history/{archive_id}"

        try:
            # Build metadata
            now = datetime.now(timezone.utc).isoformat()
            archive_metadata = {
                "archive_id": archive_id,
                "session_id": session_id,
                "created_at": now,
                "message_count": len(messages),
            }
            if metadata:
                archive_metadata.update(metadata)

            # Build a ContextNode for the archive
            # Store messages as JSON in content field
            node = ContextNode(
                uri=uri,
                context_type="RESOURCE",
                category="history",
                level=0,
                owner_space=f"session:{session_id}",
                abstract=abstract,
                overview=overview,
                content=json.dumps(messages, ensure_ascii=False, indent=2),
                metadata=archive_metadata,
            )

            # Write to AGFS via ContextFS
            self.fs.write_node(node, ctx)

            return ArchiveWriteResult(
                archive_id=archive_id,
                session_id=session_id,
                uri=uri,
                success=True,
                created_at=now,
            )

        except Exception as e:
            return ArchiveWriteResult(
                archive_id=archive_id,
                session_id=session_id,
                uri=uri,
                success=False,
                error=str(e),
            )

    def list_archives(self, session_id: str, ctx: RequestContext) -> list[ArchiveEntry]:
        """List all archives for a session.

        Args:
            session_id: Session identifier
            ctx: Request context for access control

        Returns:
            List of ArchiveEntry objects (without full message content)
        """
        # Build the session history URI
        session_uri = f"ctx://{ctx.account_id}/sessions/{session_id}/history/"

        try:
            # List children using ContextFS
            archive_uris = self.fs.list_children(session_uri, ctx)

            entries = []
            for archive_uri in archive_uris:
                # Extract archive_id from URI
                # Format: ctx://.../sessions/{session_id}/history/{archive_id}
                parts = archive_uri.rstrip("/").split("/")
                if parts:
                    archive_id = parts[-1]

                    # Read the archive node to get abstract and overview
                    try:
                        node = self.fs.read_node(archive_uri, ctx)
                        metadata = node.metadata or {}
                        if _is_merged_metadata(metadata):
                            continue

                        # Extract metadata
                        created_at = metadata.get("created_at", "")

                        entry = ArchiveEntry(
                            archive_id=archive_id,
                            session_id=session_id,
                            overview=node.overview,
                            abstract=node.abstract,
                            messages=[],  # Don't include full messages in list
                            created_at=created_at,
                            metadata=metadata,
                        )
                        entries.append(entry)
                    except Exception:
                        # If we can't read this archive, skip it
                        continue

            return entries

        except Exception:
            # If session doesn't exist or access denied, return empty list
            return []

    def read_archive(
        self, session_id: str, archive_id: str, ctx: RequestContext
    ) -> Optional[ArchiveEntry]:
        """Read a full session archive.

        Args:
            session_id: Session identifier
            archive_id: Archive identifier
            ctx: Request context for access control

        Returns:
            ArchiveEntry with full message content, or None if not found
        """
        uri = f"ctx://{ctx.account_id}/sessions/{session_id}/history/{archive_id}"

        try:
            # Check if archive exists
            if not self.fs.exists(uri, ctx):
                return None

            # Read the node from AGFS
            node = self.fs.read_node(uri, ctx)
            metadata = node.metadata or {}

            # Parse messages from content field (stored as JSON)
            try:
                messages = json.loads(node.content) if node.content else []
            except json.JSONDecodeError:
                messages = []

            # Extract metadata
            created_at = metadata.get("created_at", "")

            return ArchiveEntry(
                archive_id=archive_id,
                session_id=session_id,
                overview=node.overview,
                abstract=node.abstract,
                messages=messages,
                created_at=created_at,
                metadata=metadata,
            )

        except Exception:
            return None

    def read_archive_abstract(
        self, session_id: str, archive_id: str, ctx: RequestContext
    ) -> Optional[str]:
        """Read only the abstract from an archive.

        Args:
            session_id: Session identifier
            archive_id: Archive identifier
            ctx: Request context for access control

        Returns:
            Abstract string, or None if not found
        """
        entry = self.read_archive(session_id, archive_id, ctx)
        if entry:
            return entry.abstract
        return None

    def delete_archive(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
    ) -> bool:
        """Delete a session archive."""
        uri = self._archive_uri(ctx.account_id, session_id, archive_id)
        try:
            self.fs.delete_node(uri, ctx)
            return True
        except Exception:
            return False

    def unmark_archive_merged(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
        merged_into: str,
    ) -> bool:
        """Undo a merge marker when a merge operation rolls back."""
        with self._metadata_lock:
            uri = self._archive_uri(ctx.account_id, session_id, archive_id)
            try:
                if hasattr(self.fs, "exists") and not self.fs.exists(uri, ctx):
                    return False
                node = self.fs.read_node(uri, ctx)
                metadata = dict(node.metadata or {})
                if metadata.get("merged_into") != merged_into:
                    return False
                metadata.pop("status", None)
                metadata.pop("merged_into", None)
                metadata.pop("merged_at", None)
                self.fs.write_node(
                    ContextNode(
                        uri=node.uri,
                        context_type=node.context_type,
                        category=node.category,
                        level=node.level,
                        owner_space=node.owner_space,
                        abstract=node.abstract,
                        overview=node.overview,
                        content=node.content,
                        metadata=metadata,
                    ),
                    ctx,
                )
                return True
            except Exception:
                return False

    def mark_archive_merged(
        self,
        session_id: str,
        archive_id: str,
        ctx: RequestContext,
        merged_into: str,
    ) -> bool:
        """Mark an archive as merged into another archive."""
        with self._metadata_lock:
            uri = self._archive_uri(ctx.account_id, session_id, archive_id)
            try:
                if hasattr(self.fs, "exists") and not self.fs.exists(uri, ctx):
                    return False
                node = self.fs.read_node(uri, ctx)
                metadata = dict(node.metadata or {})
                metadata.setdefault("archive_id", archive_id)
                metadata.setdefault("session_id", session_id)
                metadata["status"] = "MERGED"
                metadata["merged_into"] = merged_into
                metadata["merged_at"] = datetime.now(timezone.utc).isoformat()
                self.fs.write_node(
                    ContextNode(
                        uri=node.uri,
                        context_type=node.context_type,
                        category=node.category,
                        level=node.level,
                        owner_space=node.owner_space,
                        abstract=node.abstract,
                        overview=node.overview,
                        content=node.content,
                        metadata=metadata,
                    ),
                    ctx,
                )
                return True
            except Exception:
                return False
