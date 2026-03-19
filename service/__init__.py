"""Service layer - external API for ContextEngine.

This is the ONLY layer where RequestContext is mandatory.
All external calls must provide a RequestContext for multi-tenant isolation.

For write operations, see the commit package:
- ContextWriter: write ContextNodes to storage
- OutboxStore: register events for async indexing

For read/search operations, see the retrieval package (not included in this branch).
"""

from service.api import (
    MemoryWriteAPI,
    init_write_api,
    get_write_api,
)

__all__ = [
    "MemoryWriteAPI",
    "init_write_api",
    "get_write_api",
]
