"""MemoryReadService — URI-based full-content read.

Powers the read_memory tool interface.  Since search_memory already
returns L2 results with abstract, read_memory only reads the actual
md file content.  Enforces account-level ACL.
"""

from __future__ import annotations

import logging

from core.errors import AccessDeniedError
from core.interfaces import ContextFS
from core.models import RequestContext, RetrievedBlock

logger = logging.getLogger(__name__)


class MemoryReadService:
    def __init__(self, fs: ContextFS | None = None) -> None:
        self._fs = fs

    def read(
        self,
        uri: str,
        ctx: RequestContext | None = None,
    ) -> RetrievedBlock:
        self._check_acl(uri, ctx)

        block = RetrievedBlock(uri=uri, level_hit="L2")

        if self._fs is None:
            logger.warning("[MemoryReadService] no ContextFS configured")
            return block

        try:
            node = self._fs.read_node(uri, ctx)
        except Exception:
            logger.warning("[MemoryReadService] read_node failed for %s", uri)
            return block

        block.category = node.category
        block.owner_space = node.owner_space
        block.content_excerpt = node.content

        return block

    @staticmethod
    def _check_acl(uri: str, ctx: RequestContext | None) -> None:
        if ctx is None:
            return
        prefix = f"ctx://{ctx.account_id}/"
        if uri.startswith("ctx://") and not uri.startswith(prefix):
            raise AccessDeniedError(uri, ctx.account_id, "account mismatch")
