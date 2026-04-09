"""Memory Read Service - L1/L2 reading with URI validation.

Provides structured reading of ContextNode content at different levels.
Validates URI account prefix against RequestContext for multi-tenant isolation.

See CLAUDE.md §7 for MemoryReadService trigger conditions and §8 for access control.
"""

from core.errors import AccessDeniedError, NodeNotFoundError
from core.interfaces import ContextFS
from core.models import ContextNode, RequestContext


# Level constants for API clarity
L0 = "L0"  # abstract only (from VectorIndex, not this service)
L1 = "L1"  # overview
L2 = "L2"  # full content


class MemoryReadService:
    """Service for reading ContextNode content at different granularity levels.

    L1 returns overview.md - topic + scope + applicability.
    L2 returns content.md - full original text.

    All reads validate URI account prefix against RequestContext.account_id.
    """

    def __init__(self, fs: ContextFS):
        """Initialize MemoryReadService with a ContextFS implementation.

        Args:
            fs: ContextFS implementation for reading nodes
        """
        self._fs = fs

    def read(self, uri: str, level: str, ctx: RequestContext) -> str:
        """Read a memory node at the specified level.

        Args:
            uri: ContextNode URI (e.g., ctx://{account}/users/{user}/memories/profile)
            level: "L1" for overview, "L2" for full content
            ctx: RequestContext containing account_id for validation

        Returns:
            String content at the requested level:
            - L1: overview.md content (structured summary)
            - L2: content.md content (full original text)

        Raises:
            AccessDeniedError: If URI account prefix != ctx.account_id
            NodeNotFoundError: If node does not exist or is not ACTIVE
            ValueError: If level is not "L1" or "L2"
        """
        # Validate level
        if level not in (L1, L2):
            raise ValueError(f"Invalid level: {level}. Must be L1 or L2")

        # Extract and validate account from URI
        uri_account = self._extract_account_from_uri(uri)
        if uri_account != ctx.account_id:
            raise AccessDeniedError(
                f"URI account '{uri_account}' does not match context account '{ctx.account_id}'"
            )

        # Read the node (this also checks access control)
        try:
            node = self._fs.read_node(uri, ctx)
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                raise NodeNotFoundError(f"Node not found: {uri}")
            raise

        # Return content at requested level
        if level == L1:
            return node.overview
        else:  # L2
            return node.content

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        """Read the full ContextNode.

        Args:
            uri: ContextNode URI
            ctx: RequestContext for validation

        Returns:
            Complete ContextNode with all fields

        Raises:
            AccessDeniedError: If URI account prefix != ctx.account_id
            NodeNotFoundError: If node does not exist
        """
        # Validate account prefix
        uri_account = self._extract_account_from_uri(uri)
        if uri_account != ctx.account_id:
            raise AccessDeniedError(
                f"URI account '{uri_account}' does not match context account '{ctx.account_id}'"
            )

        return self._fs.read_node(uri, ctx)

    def _extract_account_from_uri(self, uri: str) -> str:
        """Extract account ID from a ContextNode URI.

        URI format: ctx://{account}/users/{user}/memories/...
                    or ctx://{account}/agents/{agent}/...

        Args:
            uri: ContextNode URI

        Returns:
            Account ID string

        Raises:
            ValueError: If URI format is invalid
        """
        # Expected format: ctx://{account}/...
        if not uri.startswith("ctx://"):
            raise ValueError(f"Invalid URI format: {uri}")

        # Remove scheme and split
        path = uri[6:]  # Remove "ctx://"
        parts = path.split("/")

        if len(parts) < 2:
            raise ValueError(f"Invalid URI format: {uri}")

        return parts[0]

    def read_batch(self, uris: list[str], level: str, ctx: RequestContext) -> dict[str, str]:
        """Read multiple nodes at once.

        Args:
            uris: List of ContextNode URIs
            level: "L1" or "L2"
            ctx: RequestContext for validation

        Returns:
            Dict mapping URI to content at requested level.
            URIs that raise AccessDeniedError are omitted.
            URIs that raise NodeNotFoundError have None as value.

        Note:
            This is a convenience method for batch reads.
            Errors are handled per-URI rather than failing the entire batch.
        """
        results = {}

        for uri in uris:
            try:
                results[uri] = self.read(uri, level, ctx)
            except AccessDeniedError:
                # Skip URIs the user doesn't have access to
                continue
            except NodeNotFoundError:
                # Mark as not found
                results[uri] = None

        return results
