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
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from threading import Event, Thread
from urllib.parse import unquote

logger = logging.getLogger(__name__)

from pyagfs import AGFSClient
from pyagfs.exceptions import AGFSClientError

_UPDATE_LOCK_STALE_AFTER = timedelta(minutes=1)
_UPDATE_LOCK_RENEW_INTERVAL_SECONDS = 20


def _is_not_found_error(exc: AGFSClientError) -> bool:
    message = str(exc).lower()
    return "no such file" in message or "not found" in message

from core.models import RequestContext, ContextNode, RelationEdge
from core.interfaces import ContextFS
from core.errors import AccessDeniedError, NodeNotFoundError, NodeBrokenError, ConcurrentModificationError
from core.enums import NodeStatus
from fs.access_control import check_uri_access


# URI patterns for different memory types
# Pattern 1: Skills (different path structure)
# ctx://{account}/agents/{agent}/skills/{generated|imported}/{skill_name}
_SKILL_URI_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/agents/(?P<owner_id>[^/]+)/'
    r'(?P<skill_root>skills)/(?P<skill_source>generated|imported)/'
    r'(?P<skill_name>[^/]+)$'
)
_LEGACY_SKILL_URI_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/agents/(?P<owner_id>[^/]+)/'
    r'(?P<skill_root>skills)/(?P<skill_name>[^/]+)$'
)
_SKILLS_DIR_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/agents/(?P<owner_id>[^/]+)/'
    r'(?P<skill_root>skills)(?:/(?P<skill_source>generated|imported))?/?$'
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

# Pattern 8: Session state
# ctx://{account}/sessions/{session_id}/state
_SESSION_STATE_PATTERN = re.compile(
    r'^ctx://(?P<account>[^/]+)/sessions/(?P<session_id>[^/]+)/state(?:\.json)?$'
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
        path = (
            PurePosixPath('/accounts') / account / 'agents' / owner_id
            / groups['skill_root'] / groups['skill_source'] / skill_name
        )
        return str(path) + '/'

    match = _LEGACY_SKILL_URI_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        path = (
            PurePosixPath('/accounts') / groups['account'] / 'agents'
            / groups['owner_id'] / groups['skill_root']
            / unquote(groups['skill_name'])
        )
        return str(path) + '/'

    match = _SKILLS_DIR_PATTERN.match(uri.rstrip('/'))
    if match:
        groups = match.groupdict()
        account = groups['account']
        owner_id = groups['owner_id']
        path = PurePosixPath('/accounts') / account / 'agents' / owner_id / groups['skill_root']
        if groups.get('skill_source'):
            path /= groups['skill_source']
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

    # Try session state pattern
    match = _SESSION_STATE_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        state_name = 'state.json' if uri.endswith('state.json') else 'state'
        path = PurePosixPath('/accounts') / groups['account'] / 'sessions' / groups['session_id'] / state_name
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
            'category': f"{groups['skill_root']}/{groups['skill_source']}",
            'slug': unquote(groups['skill_name']),
        }

    match = _LEGACY_SKILL_URI_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': 'agents',
            'owner_id': groups['owner_id'],
            'category': groups['skill_root'],
            'slug': unquote(groups['skill_name']),
        }

    match = _SKILLS_DIR_PATTERN.match(uri.rstrip('/'))
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': 'agents',
            'owner_id': groups['owner_id'],
            'category': (
                f"{groups['skill_root']}/{groups['skill_source']}"
                if groups.get('skill_source')
                else groups['skill_root']
            ),
            'slug': '',
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

    # Try session state pattern
    match = _SESSION_STATE_PATTERN.match(uri)
    if match:
        groups = match.groupdict()
        return {
            'account': groups['account'],
            'owner_type': 'sessions',
            'owner_id': groups['session_id'],
            'category': 'state',
            'slug': 'state',
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
    if category == 'skills' or category.startswith('skills/'):
        return f"ctx://{account}/{owner_type}/{owner_id}/{category}/{slug}"

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
        - AGFS uses strict mode (ignores visible_owner_spaces)

        Args:
            uri: ContextEngine URI to check
            ctx: Request context containing account_id

        Returns:
            True if accessible, False otherwise
        """
        try:
            components = parse_uri(uri)
            return check_uri_access(ctx, components, strict_mode=True)
        except ValueError:
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
            if _is_not_found_error(e):
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

        expected_version = node.metadata.get(
            "_expected_version", node.metadata.get("expected_version")
        )
        final_node_path = self._uri_to_agfs_path(node.uri)
        create_only = expected_version == 0
        if expected_version is not None and not create_only:
            self._replace_node(final_node_path, node, expected_version)
            return

        staging_path = None
        if create_only:
            final_path = PurePosixPath(final_node_path.rstrip("/"))
            parent_path = str(final_path.parent) + "/"
            self._mkdir_p(parent_path)
            staging_path = (
                f"{parent_path}.{final_path.name}.create-{uuid.uuid4().hex}/"
            )
            node_path = staging_path
        else:
            node_path = final_node_path

        try:
            self._mkdir_p(node_path)
            self._write_node_files(node_path, node, expected_version)
            if create_only:
                try:
                    self._client.mv(
                        node_path.rstrip("/"), final_node_path.rstrip("/")
                    )
                except AGFSClientError as exc:
                    try:
                        existing = json.loads(
                            self._read_file(final_node_path + self.FILE_META)
                        )
                    except (NodeNotFoundError, json.JSONDecodeError):
                        raise exc
                    raise ConcurrentModificationError(
                        node.uri, 0, existing.get("version", 1)
                    ) from exc
        finally:
            if staging_path:
                self._remove_tree_if_present(staging_path)

    def _replace_node(
        self,
        final_node_path: str,
        node: ContextNode,
        expected_version: int,
    ) -> None:
        """Publish a version-checked replacement through an atomic metadata swap."""

        final_path = PurePosixPath(final_node_path.rstrip("/"))
        parent_path = str(final_path.parent) + "/"
        token = uuid.uuid4().hex
        lock_path = f"{parent_path}.{final_path.name}.update-lock"
        candidate_lock_path = f"{lock_path}-{token}"
        marker_path = lock_path + "/transaction.json"
        temporary_meta_path = final_node_path + f".meta.update-{token}.json"
        marker = {
            "owner_token": token,
            "expected_version": expected_version,
            "temporary_meta_path": temporary_meta_path,
        }
        lock_acquired = False
        lease_stop = Event()
        lease_lost = Event()
        lease_thread = None

        try:
            self._client.mkdir(candidate_lock_path)
            self._write_update_marker(candidate_lock_path, marker, token)
            self._write_update_lease(candidate_lock_path, token)
            try:
                self._client.mv(candidate_lock_path, lock_path)
                lock_acquired = True
            except AGFSClientError as exc:
                if "exist" not in str(exc).lower():
                    raise
                if not self._recover_update_lock(
                    final_node_path, lock_path, marker_path
                ):
                    raise ConcurrentModificationError(
                        node.uri, expected_version, expected_version + 1
                    ) from exc
                try:
                    self._client.mv(candidate_lock_path, lock_path)
                except AGFSClientError as retry_exc:
                    raise ConcurrentModificationError(
                        node.uri, expected_version, expected_version + 1
                    ) from retry_exc
                lock_acquired = True

            lease_thread = Thread(
                target=self._renew_update_lease,
                args=(lock_path, token, lease_stop, lease_lost),
                daemon=True,
            )
            lease_thread.start()

            actual_version = self._read_node_version(final_node_path)
            if actual_version != expected_version:
                raise ConcurrentModificationError(
                    node.uri, expected_version, actual_version
                )

            self._write_node_files(
                final_node_path,
                node,
                expected_version,
                meta_path=temporary_meta_path,
            )
            if lease_lost.is_set() or not self._owns_update_lock(lock_path, token):
                raise ConcurrentModificationError(
                    node.uri,
                    expected_version,
                    self._read_node_version(final_node_path),
                )
            self._client.mv(temporary_meta_path, final_node_path + self.FILE_META)
        finally:
            lease_stop.set()
            if lease_thread is not None:
                lease_thread.join()
            self._remove_file_if_present(temporary_meta_path)
            self._remove_tree_if_present(candidate_lock_path)
            if lock_acquired and self._owns_update_lock(lock_path, token):
                try:
                    self._client.rm(lock_path, recursive=True)
                except AGFSClientError:
                    logger.warning("Failed to release AGFS update lock: %s", lock_path)

    def _write_update_marker(
        self,
        lock_path: str,
        marker: dict,
        owner_token: str,
    ) -> None:
        marker_path = lock_path + "/transaction.json"
        temporary_path = lock_path + f"/.transaction-{owner_token}.json"
        self._write_file(temporary_path, json.dumps(marker))
        self._client.mv(temporary_path, marker_path)

    def _write_update_lease(self, lock_path: str, owner_token: str) -> None:
        lease_path = lock_path + f"/lease-{owner_token}.json"
        temporary_path = lock_path + f"/.lease-{owner_token}.json"
        lease = {
            "owner_token": owner_token,
            "lease_updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_file(temporary_path, json.dumps(lease))
        self._client.mv(temporary_path, lease_path)

    def _renew_update_lease(
        self,
        lock_path: str,
        owner_token: str,
        stop: Event,
        lost: Event,
    ) -> None:
        while not stop.wait(_UPDATE_LOCK_RENEW_INTERVAL_SECONDS):
            try:
                if not self._owns_update_lock(lock_path, owner_token):
                    lost.set()
                    return
                self._write_update_lease(lock_path, owner_token)
            except AGFSClientError:
                lost.set()
                return

    def _read_update_marker(self, marker_path: str) -> dict | None:
        try:
            marker = json.loads(self._read_file(marker_path))
        except (json.JSONDecodeError, NodeNotFoundError):
            return None
        return marker if isinstance(marker, dict) else None

    def _owns_update_lock(self, lock_path: str, owner_token: str) -> bool:
        marker = self._read_update_marker(lock_path + "/transaction.json")
        return marker is not None and marker.get("owner_token") == owner_token

    def _read_update_lease(
        self, lock_path: str, owner_token: str
    ) -> datetime | None:
        lease = self._read_update_marker(
            lock_path + f"/lease-{owner_token}.json"
        )
        if lease is None or lease.get("owner_token") != owner_token:
            return None
        try:
            renewed_at = datetime.fromisoformat(lease["lease_updated_at"])
        except (KeyError, TypeError, ValueError):
            return None
        return renewed_at if renewed_at.tzinfo is not None else None

    def _read_node_version(self, node_path: str) -> int:
        try:
            metadata = json.loads(self._read_file(node_path + self.FILE_META))
        except NodeNotFoundError:
            return 0
        return metadata.get("version", 0)

    def _recover_update_lock(
        self,
        node_path: str,
        lock_path: str,
        marker_path: str,
    ) -> bool:
        marker = self._read_update_marker(marker_path)
        if marker is None:
            return False
        try:
            owner_token = marker["owner_token"]
            expected_version = int(marker["expected_version"])
        except (KeyError, TypeError, ValueError):
            return False

        if not isinstance(owner_token, str) or not owner_token:
            return False
        lease_updated_at = self._read_update_lease(lock_path, owner_token)
        if lease_updated_at is None:
            return False

        published = self._read_node_version(node_path) > expected_version
        stale = (
            datetime.now(timezone.utc) - lease_updated_at
            > _UPDATE_LOCK_STALE_AFTER
        )
        if not published and not stale:
            return False

        confirmed = self._read_update_marker(marker_path)
        if confirmed is None or confirmed.get("owner_token") != owner_token:
            return False
        confirmed_lease = self._read_update_lease(lock_path, owner_token)
        if confirmed_lease is None:
            return False
        if (
            not published
            and datetime.now(timezone.utc) - confirmed_lease
            <= _UPDATE_LOCK_STALE_AFTER
        ):
            return False

        temporary_meta_path = confirmed.get("temporary_meta_path")
        temporary_prefix = node_path + ".meta.update-"
        if (
            isinstance(temporary_meta_path, str)
            and temporary_meta_path.startswith(temporary_prefix)
            and temporary_meta_path.endswith(".json")
        ):
            # Invalidate the retired writer before freeing the lock. Even if
            # it passed its owner check, it cannot publish without this file.
            try:
                self._client.rm(temporary_meta_path)
            except AGFSClientError as exc:
                if not _is_not_found_error(exc):
                    return False
        self._remove_tree_if_present(lock_path)
        return True

    def _remove_tree_if_present(self, path: str) -> None:
        try:
            self._client.rm(path.rstrip("/"), recursive=True)
        except AGFSClientError:
            pass

    def _remove_file_if_present(self, path: str) -> None:
        try:
            self._client.rm(path)
        except AGFSClientError:
            pass

    def _write_node_files(
        self,
        node_path: str,
        node: ContextNode,
        expected_version: int | None,
        *,
        meta_path: str | None = None,
    ) -> None:
        """Write a complete node into an already-created directory."""

        # Prepare file paths
        content_path = node_path + self.FILE_CONTENT
        relations_path = node_path + self.FILE_RELATIONS
        abstract_path = node_path + self.FILE_ABSTRACT
        overview_path = node_path + self.FILE_OVERVIEW
        meta_path = meta_path or node_path + self.FILE_META

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

        # Step ④: Write status=ACTIVE metadata. Versioned updates pass a
        # temporary path and publish it with one atomic file rename.
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

        if meta.get("ogmemory_managed"):
            content = meta.get("package_skill_md")
            abstract = meta.get("package_description")
            if content is None:
                content = self._read_file(content_path)
            if abstract is None:
                abstract = self._read_file(abstract_path)
        else:
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
        # Preserve extension metadata such as complete managed Skill packages.
        core_keys = {
            "uri", "context_type", "category", "level", "owner_space",
            "created_at", "updated_at", "version",
        }
        metadata.update({key: value for key, value in meta.items() if key not in core_keys})

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

    def read_node_fields(self, uri: str, ctx: RequestContext, level: str) -> ContextNode:
        """Read only the MCP-requested detail fields for a node."""

        if level not in {"abstract", "overview", "content", "all"}:
            raise ValueError("level must be one of: abstract, overview, content, all")
        self._ensure_accessible(uri, ctx)

        node_path = self._uri_to_agfs_path(uri)
        meta_path = node_path + self.FILE_META
        try:
            meta = json.loads(self._read_file(meta_path))
        except NodeNotFoundError:
            raise NodeNotFoundError(uri)

        status = meta.get("status")
        if status == NodeStatus.BROKEN.value:
            raise NodeBrokenError(uri)
        if status != NodeStatus.ACTIVE.value:
            raise NodeNotFoundError(uri)

        if level in {"abstract", "all"}:
            abstract = meta.get("package_description") if meta.get("ogmemory_managed") else None
            if abstract is None:
                abstract = self._read_file(node_path + self.FILE_ABSTRACT)
        else:
            abstract = ""
        overview = self._read_file(node_path + self.FILE_OVERVIEW) if level in {"overview", "all"} else ""
        if level in {"content", "all"}:
            content = meta.get("package_skill_md") if meta.get("ogmemory_managed") else None
            if content is None:
                content = self._read_file(node_path + self.FILE_CONTENT)
        else:
            content = ""

        metadata = {
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "version": meta.get("version", 1),
            "has_content": True,
        }
        core_keys = {
            "uri", "context_type", "category", "level", "owner_space",
            "created_at", "updated_at", "version",
        }
        metadata.update({key: value for key, value in meta.items() if key not in core_keys})

        return ContextNode(
            uri=meta["uri"],
            context_type=meta["context_type"],
            category=meta["category"],
            level=meta["level"],
            owner_space=meta.get("owner_space", ""),
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

    def list_node_summaries(self, uri: str, ctx: RequestContext,
                            limit: int = 50) -> list[ContextNode]:
        """List the first stable page of child node summaries."""
        return self.list_node_summaries_page(
            uri, ctx, limit=limit, after_uri=None,
        )

    def list_node_summaries_page(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        limit: int,
        after_uri: str | None = None,
    ) -> list[ContextNode]:
        """List one URI-ordered page without reading full content/overview."""

        self._ensure_accessible(uri, ctx)
        if limit <= 0:
            return []

        try:
            parent_path = self._uri_to_agfs_path(uri)
            files = self._client.ls(parent_path)
        except AGFSClientError:
            return []

        summaries: list[ContextNode] = []
        for file_info in sorted(files, key=lambda item: str(item.get("name", ""))):
            if len(summaries) >= limit:
                break
            if not file_info.get("isDir", False):
                continue
            name = file_info["name"]
            if name.startswith("."):
                continue
            child_uri = uri.rstrip("/") + "/" + name
            if after_uri is not None and child_uri <= after_uri:
                continue
            try:
                node_path = self._uri_to_agfs_path(child_uri)
                meta = json.loads(self._read_file(node_path + self.FILE_META))
                status = meta.get("status")
                if status == NodeStatus.BROKEN.value:
                    raise NodeBrokenError(child_uri)
                if status != NodeStatus.ACTIVE.value:
                    continue
                abstract = (
                    meta.get("package_description")
                    if meta.get("ogmemory_managed")
                    else None
                )
                if abstract is None:
                    abstract = self._read_file(node_path + self.FILE_ABSTRACT)
            except NodeNotFoundError:
                continue
            metadata = {
                key: meta[key]
                for key in (
                    "ogmemory_generated",
                    "ogmemory_managed",
                    "publication_status",
                    "content_hash",
                    "package_description",
                )
                if key in meta
            } | {
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", ""),
                "version": meta.get("version", 1),
                "has_content": True,
            }
            summaries.append(
                ContextNode(
                    uri=meta["uri"],
                    context_type=meta.get("context_type", "MEMORY"),
                    category=meta.get("category", ""),
                    level=meta.get("level", 0),
                    owner_space=meta.get("owner_space", ""),
                    abstract=abstract,
                    overview="",
                    content="",
                    metadata=metadata,
                )
            )

        return summaries

    def archive_node(self, uri: str, ctx: RequestContext) -> None:
        """Archive a node from AGFS by marking its metadata status ARCHIVED."""
        self._ensure_accessible(uri, ctx)

        node_path = self._uri_to_agfs_path(uri)
        meta_path = node_path + self.FILE_META

        try:
            meta = json.loads(self._read_file(meta_path))
        except NodeNotFoundError:
            raise NodeNotFoundError(uri)

        if meta.get("status") != NodeStatus.ACTIVE.value:
            raise NodeNotFoundError(uri)

        now = datetime.now(timezone.utc).isoformat()
        meta["status"] = NodeStatus.ARCHIVED.value
        meta["archived_at"] = now
        meta["updated_at"] = now
        self._write_file(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))

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
            if _is_not_found_error(e):
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
