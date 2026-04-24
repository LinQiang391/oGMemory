"""ContextReader — URI-based full-content read.

Powers the read_memory tool interface.  Since search_memory already
returns L2 results with abstract, read_memory only reads the actual
md file content.  Enforces account-level ACL.
"""

from __future__ import annotations

import logging

from core.errors import AccessDeniedError, ValidationError
from core.interfaces import ContextFS
from core.models import RequestContext, RetrievedBlock

logger = logging.getLogger(__name__)


class ContextReader:
    def __init__(self, fs: ContextFS | None = None) -> None:
        self._fs = fs

    def read(
        self,
        uri: str,
        ctx: RequestContext,
    ) -> RetrievedBlock:
        self._check_acl(uri, ctx)

        block = RetrievedBlock(uri=uri, level_hit="L2")

        if self._fs is None:
            logger.warning("[ContextReader] no ContextFS configured")
            return block

        try:
            # Strip file suffix from L2 URIs (e.g. "/content.md") — read_node
            # expects the directory URI, not the file URI stored in the vector index.
            read_uri = uri
            if read_uri.endswith("/content.md"):
                read_uri = read_uri[: -len("/content.md")]
            node = self._fs.read_node(read_uri, ctx)
        except Exception as exc:
            logger.warning("[ContextReader] read_node failed for %s: %s", uri, exc, exc_info=True)
            return block

        block.category = node.category
        block.owner_space = node.owner_space
        block.content_excerpt = node.content

        return block

    @staticmethod
    def _check_acl(uri: str, ctx: RequestContext) -> None:
        # SECURITY: RequestContext is mandatory for all read operations
        # Per CLAUDE.md §8: "唯一强制注入 RequestContext 的位置是 service 层"
        if ctx is None:
            raise ValidationError("ctx", "RequestContext is required for read operation")

        if ctx.account_id is None or not ctx.account_id.strip():
            raise ValidationError("account_id", "account_id is required for read operation")

        prefix = f"ctx://{ctx.account_id}/"
        if uri.startswith("ctx://") and not uri.startswith(prefix):
            raise AccessDeniedError(uri, ctx.account_id, "account mismatch")

        # Owner space isolation: prevent same-account cross-user/cross-agent reads
        if uri.startswith(f"ctx://{ctx.account_id}/users/"):
            user_prefix = f"ctx://{ctx.account_id}/users/{ctx.user_id}/"
            if not uri.startswith(user_prefix):
                raise AccessDeniedError(uri, ctx.account_id, "owner_space mismatch")
        if ctx.agent_id and uri.startswith(f"ctx://{ctx.account_id}/agents/"):
            agent_prefix = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/"
            if not uri.startswith(agent_prefix):
                raise AccessDeniedError(uri, ctx.account_id, "owner_space mismatch")
