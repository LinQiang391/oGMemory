"""Service layer - external API for ContextEngine.

This is the ONLY layer where RequestContext is mandatory.
All external calls must provide a RequestContext for multi-tenant isolation.

For write operations, see the commit package:
- ContextWriter: write ContextNodes to storage
- OutboxStore: register events for async indexing

For read/search operations, see the retrieval package (not included in this branch).
"""

<<<<<<< HEAD
__all__ = []
=======
from service.api import (
    MemoryReadAPI,
    MemoryWriteAPI,
    init_read_api,
    init_write_api,
    init_api,
    get_read_api,
    get_write_api,
    get_api,
    Level,
)

__all__ = [
    "MemoryReadAPI",
    "MemoryWriteAPI",
    "init_read_api",
    "init_write_api",
    "init_api",
    "get_read_api",
    "get_write_api",
    "get_api",
    "Level",
]
>>>>>>> 16716e1 (feat(service): add MemoryWriteAPI with commit_session interface)
