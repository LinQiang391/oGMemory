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
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Optional
from urllib.parse import unquote

from pyagfs import AGFSClient
from pyagfs.exceptions import AGFSClientError

from core.models import RequestContext, ContextNode, RelationEdge
from core.interfaces import ContextFS
from core.errors import AccessDeniedError, NodeNotFoundError, NodeBrokenError
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
        client = AGFSClient(api_base_url="http://localhost:8080")
        fs = AGFSContextFS(client)
        fs.write_node(node, context)
    """

    # File names within a node directory
    FILE_CONTENT = "content.md"
    FILE_RELATIONS = ".relations.json"
    FILE_ABSTRACT = ".abstract.md"
    FILE_OVERVIEW = ".overview.md"
    FILE_META = ".meta.json"
    DIR_OUTBOX = ".outbox"

    def __init__(self, client: AGFSClient):
        """Initialize AGFS context filesystem adapter.

        Args:
            client: AGFS Python SDK client instance
        """
        self._client = client

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
            return components['account'] == ctx.account_id
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

    # TODO: P1-W3 - Implement write_node with atomic 4-step order
    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        """Write a node to AGFS with atomic 4-step write order.

        NOT YET IMPLEMENTED - See P1-W3 in tasks.md
        """
        raise NotImplementedError("P1-W3: write_node with atomic order")

    # TODO: P1-W4 - Implement read_node
    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        """Read a node from AGFS.

        NOT YET IMPLEMENTED - See P1-W4 in tasks.md
        """
        raise NotImplementedError("P1-W4: read_node")

    # TODO: P1-W4 - Implement exists
    def exists(self, uri: str, ctx: RequestContext) -> bool:
        """Check if node exists and is ACTIVE.

        NOT YET IMPLEMENTED - See P1-W4 in tasks.md
        """
        raise NotImplementedError("P1-W4: exists")

    # TODO: P1-W4 - Implement list_children
    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        """List child URIs under a given URI.

        NOT YET IMPLEMENTED - See P1-W4 in tasks.md
        """
        raise NotImplementedError("P1-W4: list_children")

    # TODO: P1-W4 - Implement delete_node
    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        """Delete a node from AGFS.

        NOT YET IMPLEMENTED - See P1-W4 in tasks.md
        """
        raise NotImplementedError("P1-W4: delete_node")

    # TODO: P1-W4 - Implement move_node
    def move_node(self, from_uri: str, to_uri: str, ctx: RequestContext) -> None:
        """Move/rename a node in AGFS.

        NOT YET IMPLEMENTED - See P1-W4 in tasks.md
        """
        raise NotImplementedError("P1-W4: move_node")
