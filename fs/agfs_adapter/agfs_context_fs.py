"""AGFS adapter for ContextFS interface.

Implements ContextFS protocol using AGFS (Adaptive Globally-addressable File System)
as the physical storage layer.

Phase 1 implementation:
- P1-W1: URI to AGFS path mapping
- P1-W2: Access control via _is_accessible()
- P1-W3: write_node with atomic 4-step order
- P1-W4: read_node/exists/list_children/delete_node
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import unquote

logger = logging.getLogger(__name__)

from pyagfs import AGFSClient
from pyagfs.exceptions import AGFSClientError

from core.models import RequestContext, ContextNode, RelationEdge
from core.interfaces import ContextFS
from core.errors import AccessDeniedError, NodeNotFoundError, NodeBrokenError, ConcurrentModificationError
from core.enums import NodeStatus


# URI patterns for different memory types
# Pattern 1: Skills (different path structure)
# ctx://{account}/agents/{agent}/skills/{skill_name}
_SKILL_URI_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/agents/(?P<owner_id>[^/]+)/skills/(?P<skill_name>[^/]+)$'
)

# Pattern 2: Profile (no slug)
# ctx://{account}/users/{user}/memories/profile
_PROFILE_URI_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/users/(?P<owner_id>[^/]+)/memories/profile$'
)

# Pattern 3: Standard memories with slug
# ctx://{account}/{owner_type}/{owner_id}/memories/{category}/{slug}
_MEMORY_URI_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/'
    r'memories/(?P<category>[^/]+)/(?P<slug>[^/]+)$'
)

# Pattern 4: Category-level directory (no slug)
# ctx://{account}/{owner_type}/{owner_id}/memories/{category}
# Also matches trailing slash: ctx://…/memories/{category}/
_CATEGORY_DIR_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/'
    r'memories/(?P<category>[^/]+)/?$'
)

# Pattern 5: Memories root directory
# ctx://{account}/{owner_type}/{owner_id}/memories
_MEMORIES_ROOT_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/(?P<owner_type>users|agents)/(?P<owner_id>[^/]+)/memories/?$'
)

# Pattern 6: Session archive (specific archive)
# ctx://{account}/sessions/{session_id}/history/{archive_id}
_SESSION_ARCHIVE_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/history/(?P<archive_id>[^/]+)$'
)

# Pattern 7: Session history directory
# ctx://{account}/sessions/{session_id}/history/
_SESSION_HISTORY_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/history/?$'
)


def uri_to_path(uri: str) -> str:
    """Convert a ContextEngine URI to AGFS physical path.

    Mapping rules (from CLAUDE.md §1 URI Spec and §2 AGFS Directory Spec):
    - ctx://{account}/users/{user}/memories/profile
      → /accounts/{account}/users/{user}/memories/profile/
    - ctx://{account}/users/{user}/memories/{category}/{slug}
      → /accounts/{account}/users/{user}/memories/{category}/{slug}/
    - ctx://{account}/agents/{agent}/memories/{category}/{slug}
      → /accounts/{account}/agents/{agent}/memories/{category}/{slug}/
    - ctx://{account}/agents/{agent}/skills/{skill_name}
      → /accounts/{account}/agents/{agent}/skills/{skill_name}/

    Args:
        uri: ContextEngine URI (e.g., "ctx://acme/users/alice/memories/profile")

    Returns:
        AGFS physical path (e.g., "/accounts/acme/users/alice/memories/profile/")

    Raises:
        ValueError: If URI format is invalid
    """
    # Try skill pattern first (different path structure)
    match = _SKILL_URI_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        account = groups['account']
        owner_id = groups['owner_id']
        skill_name = unquote(groups['skill_name'])
        path = PurePosixPath('/accounts') / account / 'agents' / owner_id / 'skills' / skill_name
        return str(path) + '/'

    # Try profile pattern (no slug)
    match = _PROFILE_URI_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        account = groups['account']
        owner_id = groups['owner_id']
        path = PurePosixPath('/accounts') / account / 'users' / owner_id / 'memories' / 'profile'
        return str(path) + '/'

    # Try standard memory pattern with slug
    match = _MEMORY_URI_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        account = groups['account']
        owner_type = groups['owner_type']  # 'users' or 'agents'
        owner_id = groups['owner_id']
        category = groups['category']
        slug = unquote(groups['slug'])
        path = PurePosixPath('/accounts') / account / owner_type / owner_id / 'memories' / category / slug
        return str(path) + '/'

    # Try category-level directory (no slug)
    match = _CATEGORY_DIR_PATTERN.match(uri.rstrip('/'))
    if match:
        groups = match.groupdict()
        path = PurePosixPath('/accounts') / groups['account'] / groups['owner_type'] / groups['owner_id'] / 'memories' / groups['category']
        return str(path) + '/'

    # Try memories root directory
    match = _MEMORIES_ROOT_PATTERN.match(uri.rstrip('/'))
    if match:
        groups = match.groupdict()
        path = PurePosixPath('/accounts') / groups['account'] / groups['owner_type'] / groups['owner_id'] / 'memories'
        return str(path) + '/'

    # Try session archive pattern
    match = _SESSION_ARCHIVE_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        path = PurePosixPath('/accounts') / groups['account'] / 'sessions' / groups['session_id'] / 'history' / groups['archive_id']
        return str(path) + '/'

    # Try session history directory
    match = _SESSION_HISTORY_PATTERN.match(uri.rstrip('/'))
    if match:
        groups = match.groupdict()
        path = PurePosixPath('/accounts') / groups['account'] / 'sessions' / groups['session_id'] / 'history'
        return str(path) + '/'

    raise ValueError(f"Invalid URI format: {uri}")


def parse_uri(uri: str) -> dict:
    """Parse a ContextEngine URI into its components.

    Args:
        uri: ContextEngine URI

    Returns:
        Dict with keys: account, owner_type, owner_id, category, slug
        For skills: category='skills', slug=skill_name
        For profile: category='profile', slug='profile'

    Raises:
        ValueError: If URI format is invalid
    """
    # Try skill pattern first
    match = _SKILL_URI_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': 'agents',
            'owner_id': groups['owner_id'],
            'category': 'skills',
            'slug': unquote(groups['skill_name']),
        }

    # Try profile pattern
    match = _PROFILE_URI_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': 'users',
            'owner_id': groups['owner_id'],
            'category': 'profile',
            'slug': 'profile',
        }

    # Try standard memory pattern
    match = _MEMORY_URI_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': groups['owner_type'],
            'owner_id': groups['owner_id'],
            'category': groups['category'],
            'slug': unquote(groups['slug']),
        }

    # Try category-level directory (no slug)
    stripped = uri.rstrip('/')
    match = _CATEGORY_DIR_PATTERN.match(stripped)
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': groups['owner_type'],
            'owner_id': groups['owner_id'],
            'category': groups['category'],
            'slug': '',
        }

    # Try memories root directory
    match = _MEMORIES_ROOT_PATTERN.match(stripped)
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': groups['owner_type'],
            'owner_id': groups['owner_id'],
            'category': '',
            'slug': '',
        }

    # Try session archive pattern (specific archive)
    match = _SESSION_ARCHIVE_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': 'sessions',
            'owner_id': groups['session_id'],
            'category': 'history',
            'slug': groups['archive_id'],
        }

    # Try session history directory
    match = _SESSION_HISTORY_PATTERN.match(uri.rstrip('/'))
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': 'sessions',
            'owner_id': groups['session_id'],
            'category': 'history',
            'slug': '',
        }

    raise ValueError(f"Invalid URI format: {uri}")


def build_uri(
    account: str,
    owner_type: str,
    owner_id: str,
    category: str,
    slug: str
) -> str:
    """Build a ContextEngine URI from components.

    Args:
        account: Account ID
        owner_type: 'users' or 'agents'
        owner_id: User or agent ID
        category: Memory/skill category (e.g., 'profile', 'preferences', 'skills')
        slug: URL-safe identifier (for profile, use 'profile'; for skills, use skill name)

    Returns:
        ContextEngine URI
    """
    # Ensure owner_type is plural
    if owner_type in ('user', 'agent'):
        owner_type = owner_type + 's'

    # Skills have a different path structure
    if category == 'skills':
        return f"ctx://{account}/{owner_type}/{owner_id}/skills/{slug}"

    # Profile has no slug component
    if category == 'profile':
        return f"ctx://{account}/{owner_type}/{owner_id}/memories/profile"

    # Standard memory URI
    return f"ctx://{account}/{owner_type}/{owner_id}/memories/{category}/{slug}"


class AGFSContextFS:
    """AGFS implementation of ContextFS protocol.

    This adapter provides:
    - Multi-tenant isolation via account-based path separation
    - Access control via RequestContext validation
    - Atomic write order for node durability
    - PENDING node visibility control

    Usage:
        client = AGFSClient(api_base_url="http://localhost:1833")
        fs = AGFSContextFS(client, mount_prefix="/local")
        fs.write_node(node, context)
    """

    # File names within a node directory
    FILE_CONTENT = "content.md"
    FILE_RELATIONS = ".relations.json"
    FILE_ABSTRACT = ".abstract.md"
    FILE_OVERVIEW = ".overview.md"
    FILE_META = ".meta.json"
    DIR_OUTBOX = ".outbox"

    def __init__(self, client: AGFSClient, mount_prefix: str = "/local"):
        """Initialize AGFS context filesystem adapter.

        Args:
            client: AGFS Python SDK client instance
            mount_prefix: AGFS mount point prefix (default: "/local")
                         Use "" for root mount, "/local" for localfs, etc.
        """
        self._client = client
        self._mount_prefix = mount_prefix.rstrip("/")
        self._ensure_mount_prefix()

    def _ensure_mount_prefix(self) -> None:
        """Ensure the full mount_prefix directory exists in AGFS.

        AGFS only auto-creates the top-level mount point (e.g. /local).
        Subdirectories within the mount point (e.g. /local/plugin) must
        be created explicitly.
        """
        if not self._mount_prefix:
            return
        parts = [p for p in self._mount_prefix.split("/") if p]
        if len(parts) <= 1:
            return
        current = "/" + parts[0]
        for part in parts[1:]:
            current += "/" + part
            try:
                self._client.mkdir(current)
            except AGFSClientError as e:
                if "exists" not in str(e).lower():
                    logger.warning("Failed to ensure mount prefix dir %s: %s", current, e)

    def _uri_to_agfs_path(self, uri: str) -> str:
        """Convert ContextEngine URI to AGFS path with mount prefix.

        Args:
            uri: ContextEngine URI

        Returns:
            Full AGFS path including mount prefix
        """
        base_path = uri_to_path(uri)
        if self._mount_prefix:
            return self._mount_prefix + base_path
        return base_path

    def _is_accessible(self, uri: str, ctx: RequestContext) -> bool:
        """Check if the given RequestContext can access the URI.

        Access control rules (CLAUDE.md §8):
        - URI account prefix must match RequestContext.account_id
        - Cross-account access is ALWAYS DENIED (v1)

        Args:
            uri: ContextEngine URI to check
            ctx: Request context containing account_id

        Returns:
            True if accessible, False otherwise
        """
        try:
            components = parse_uri(uri)
            if components['account'] != ctx.account_id:
                return False
            # Owner space isolation
            if 'owner_id' in components:
                owner_id = components['owner_id']
                owner_type = components.get('owner_type', '')
                if owner_type == 'users' and owner_id != ctx.user_id:
                    return False
                if owner_type == 'agents' and owner_id != ctx.agent_id:
                    return False
            return True
        except ValueError:
            # Invalid URI is not accessible
            return False

    def _ensure_accessible(self, uri: str, ctx: RequestContext) -> None:
        """Raise AccessDeniedError if URI is not accessible.

        Args:
            uri: ContextEngine URI to check
            ctx: Request context

        Raises:
            AccessDeniedError: If account_id doesn't match URI prefix
        """
        if not self._is_accessible(uri, ctx):
            try:
                components = parse_uri(uri)
                raise AccessDeniedError(
                    uri,
                    ctx.account_id,
                    f"URI belongs to account '{components['account']}'"
                )
            except ValueError:
                raise AccessDeniedError(uri, ctx.account_id, "Invalid URI format")

    def _mkdir_p(self, path: str) -> None:
        """Create a directory recursively, ignoring if it exists.

        Args:
            path: Directory path to create
        """
        # Remove trailing slash for processing
        path = path.rstrip("/")

        # If path is just the mount prefix, it already exists
        if path == self._mount_prefix:
            return

        # Build path components and create recursively.
        # Skip any component that falls within the mount prefix —
        # the mount point already exists and AGFS forbids mkdir on it.
        prefix_parts = self._mount_prefix.strip("/").split("/") if self._mount_prefix else []
        parts = [p for p in path.split("/") if p]  # skip empty strings
        current_path = ""

        for idx, part in enumerate(parts):
            current_path += "/" + part
            # Skip mkdir for components within the mount prefix
            if idx < len(prefix_parts):
                continue
            try:
                self._client.mkdir(current_path)
            except AGFSClientError as e:
                # Ignore "already exists" errors
                if "exists" not in str(e).lower():
                    # Re-raise if it's not an "exists" error
                    raise

    def _write_file(self, path: str, content: str) -> None:
        """Write content to a file.

        Args:
            path: File path to write
            content: String content to write
        """
        self._client.write(path, content.encode('utf-8'))

    def _read_file(self, path: str) -> str:
        """Read content from a file.

        Args:
            path: File path to read

        Returns:
            File content as string

        Raises:
            NodeNotFoundError: If file doesn't exist
        """
        try:
            content = self._client.read(path)
            return content.decode('utf-8')
        except AGFSClientError as e:
            if "no such file" in str(e).lower() or "not found" in str(e).lower():
                raise NodeNotFoundError(path)
            raise

    def _relation_edge_to_dict(self, edge: RelationEdge) -> dict:
        """Convert RelationEdge to dict for JSON serialization."""
        return {
            "from_uri": edge.from_uri,
            "to_uri": edge.to_uri,
            "relation_type": edge.relation_type,
            "weight": edge.weight,
            "reason": edge.reason,
        }

    def _dict_to_relation_edge(self, d: dict) -> RelationEdge:
        """Convert dict to RelationEdge."""
        return RelationEdge(
            from_uri=d["from_uri"],
            to_uri=d["to_uri"],
            relation_type=d["relation_type"],
            weight=d["weight"],
            reason=d["reason"],
        )

    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        """Write a node to AGFS with atomic 4-step write order.

        Write order (strictly enforced, Repair Job depends on this):
        ① content.md (largest file, write first)
        ② .relations.json
        ③ .abstract.md, .overview.md (parallel safe)
        ④ .meta.json (commit point: status=ACTIVE)

        Args:
            node: ContextNode to write
            ctx: RequestContext for access control

        Raises:
            AccessDeniedError: If account_id doesn't match URI prefix
        """
        # Step 0: Access control check
        self._ensure_accessible(node.uri, ctx)

        # Step 0.5: Optimistic locking check for merge operations
        expected_version = node.metadata.get("expected_version")
        if expected_version is not None:
            # This is a merge operation - verify version hasn't changed
            node_path = self._uri_to_agfs_path(node.uri)
            meta_path = node_path + self.FILE_META

            try:
                meta_content = self._read_file(meta_path)
                existing_meta = json.loads(meta_content)
                actual_version = existing_meta.get("version", 0)

                if actual_version != expected_version:
                    raise ConcurrentModificationError(
                        node.uri, expected_version, actual_version
                    )
            except NodeNotFoundError:
                # Node doesn't exist yet, proceed with create
                pass

        # Get the physical path for the node
        node_path = self._uri_to_agfs_path(node.uri)

        # Create node directory
        self._mkdir_p(node_path)

        # Prepare file paths
        content_path = node_path + self.FILE_CONTENT
        relations_path = node_path + self.FILE_RELATIONS
        abstract_path = node_path + self.FILE_ABSTRACT
        overview_path = node_path + self.FILE_OVERVIEW
        meta_path = node_path + self.FILE_META

        # Step ①: Write content.md (largest file, do first)
        # If this fails, no .meta.json exists, node doesn't exist → skip in repair
        self._write_file(content_path, node.content)

        # Step ②: Write .relations.json
        # Get relations from node metadata if available
        relations = node.metadata.get('_relations', [])
        relations_json = json.dumps(
            [self._relation_edge_to_dict(r) for r in relations],
            ensure_ascii=False,
            indent=2
        )
        self._write_file(relations_path, relations_json)

        # Step ③: Write .abstract.md and .overview.md (parallel safe)
        self._write_file(abstract_path, node.abstract)
        self._write_file(overview_path, node.overview)

        # Step ④: Write .meta.json with status=ACTIVE ★ COMMIT POINT
        # After this step, the node is visible to retrieval
        now = datetime.now(timezone.utc).isoformat()

        # Determine version: increment for merge, default to 1 for create
        if expected_version is not None:
            # Merge operation: increment from expected version
            new_version = expected_version + 1
        else:
            # Create operation: use existing version or default to 1
            new_version = node.metadata.get("version", 1)

        meta = {
            "uri": node.uri,
            "context_type": node.context_type,
            "category": node.category,
            "level": node.level,
            "owner_space": node.owner_space,
            "status": NodeStatus.ACTIVE.value,
            "created_at": node.metadata.get("created_at", now),
            "updated_at": now,
            "version": new_version,
        }

        # Add any additional metadata (excluding internal _relations)
        extra_meta = {k: v for k, v in node.metadata.items() if not k.startswith('_')}
        meta.update(extra_meta)

        meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
        self._write_file(meta_path, meta_json)

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        """Read a node from AGFS.

        Args:
            uri: ContextEngine URI to read
            ctx: RequestContext for access control

        Returns:
            ContextNode with all fields populated

        Raises:
            AccessDeniedError: If account_id doesn't match URI prefix
            NodeNotFoundError: If node doesn't exist or status != ACTIVE
            NodeBrokenError: If node is in BROKEN state
        """
        # Access control check
        self._ensure_accessible(uri, ctx)

        node_path = self._uri_to_agfs_path(uri)
        meta_path = node_path + self.FILE_META

        # Read and parse .meta.json
        try:
            meta_content = self._read_file(meta_path)
        except NodeNotFoundError:
            raise NodeNotFoundError(uri)

        meta = json.loads(meta_content)
        status = meta.get("status")

        # Check node status
        if status == NodeStatus.BROKEN.value:
            raise NodeBrokenError(uri)
        if status != NodeStatus.ACTIVE.value:
            raise NodeNotFoundError(uri)

        # Read content files
        content_path = node_path + self.FILE_CONTENT
        abstract_path = node_path + self.FILE_ABSTRACT
        overview_path = node_path + self.FILE_OVERVIEW
        relations_path = node_path + self.FILE_RELATIONS

        content = self._read_file(content_path)
        abstract = self._read_file(abstract_path)
        overview = self._read_file(overview_path)

        # Read relations
        try:
            relations_content = self._read_file(relations_path)
            relations_data = json.loads(relations_content) if relations_content else []
            relations = [self._dict_to_relation_edge(r) for r in relations_data]
        except (NodeNotFoundError, json.JSONDecodeError):
            relations = []

        # Build metadata dict
        metadata = {
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "version": meta.get("version", 1),
            "_relations": relations,
        }
        # Add any extra metadata from meta.json
        for key in ["tags", "status"]:
            if key in meta:
                metadata[key] = meta[key]

        return ContextNode(
            uri=meta["uri"],
            context_type=meta["context_type"],
            category=meta["category"],
            level=meta["level"],
            owner_space=meta["owner_space"],
            abstract=abstract,
            overview=overview,
            content=content,
            metadata=metadata,
        )

    def exists(self, uri: str, ctx: RequestContext) -> bool:
        """Check if node exists and is ACTIVE.

        PENDING nodes are NOT visible to upper layers (CLAUDE.md §2).

        Args:
            uri: ContextEngine URI to check
            ctx: RequestContext for access control

        Returns:
            True if node exists and status == ACTIVE, False otherwise
        """
        # Access control check
        if not self._is_accessible(uri, ctx):
            return False

        try:
            node_path = self._uri_to_agfs_path(uri)
            meta_path = node_path + self.FILE_META
            meta_content = self._read_file(meta_path)
            meta = json.loads(meta_content)
            return meta.get("status") == NodeStatus.ACTIVE.value
        except (ValueError, NodeNotFoundError, json.JSONDecodeError):
            return False

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        """List child URIs under a given URI.

        Args:
            uri: Parent ContextEngine URI
            ctx: RequestContext for access control

        Returns:
            List of child URIs that are ACTIVE

        Raises:
            AccessDeniedError: If account_id doesn't match URI prefix
        """
        # Access control check
        self._ensure_accessible(uri, ctx)

        # Convert URI to AGFS path and list directory
        try:
            parent_path = self._uri_to_agfs_path(uri)
            files = self._client.ls(parent_path)
        except AGFSClientError:
            # Directory doesn't exist
            return []

        # Filter for ACTIVE nodes only
        children = []
        for file_info in files:
            if file_info.get("isDir", False):
                name = file_info["name"]
                # Skip hidden directories (.outbox)
                if name.startswith("."):
                    continue

                # Build child URI by appending child name to parent URI
                # Need to handle the URI structure properly
                if uri.endswith("/"):
                    child_uri = uri + name
                else:
                    child_uri = uri + "/" + name

                # Check if child exists and is ACTIVE
                if self.exists(child_uri, ctx):
                    children.append(child_uri)

        return children

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        """Delete a node from AGFS.

        Args:
            uri: ContextEngine URI to delete
            ctx: RequestContext for access control

        Raises:
            AccessDeniedError: If account_id doesn't match URI prefix
            NodeNotFoundError: If node doesn't exist
        """
        # Access control check
        self._ensure_accessible(uri, ctx)

        node_path = self._uri_to_agfs_path(uri)

        # Delete recursively (removes all files and subdirectories)
        try:
            self._client.rm(node_path, recursive=True)
        except AGFSClientError as e:
            if "no such file" in str(e).lower() or "not found" in str(e).lower():
                raise NodeNotFoundError(uri)
            raise

    def move_node(self, from_uri: str, to_uri: str, ctx: RequestContext) -> None:
        """Move/rename a node in AGFS.

        Args:
            from_uri: Source ContextEngine URI
            to_uri: Destination ContextEngine URI
            ctx: RequestContext for access control

        Raises:
            AccessDeniedError: If either URI account prefix doesn't match
            NodeNotFoundError: If source node doesn't exist
        """
        # Access control check for both URIs
        self._ensure_accessible(from_uri, ctx)
        self._ensure_accessible(to_uri, ctx)

        from_path = self._uri_to_agfs_path(from_uri)
        to_path = self._uri_to_agfs_path(to_uri)

        # Ensure source exists
        if not self.exists(from_uri, ctx):
            raise NodeNotFoundError(from_uri)

        # Move the directory
        self._client.mv(from_path, to_path)
