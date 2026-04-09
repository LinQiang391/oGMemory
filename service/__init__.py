"""Service layer - external API for ContextEngine.

This is the ONLY layer where RequestContext is mandatory.
All external calls must provide a RequestContext for multi-tenant isolation.

For write operations:
- MemoryWriteAPI: commit_session for writing memories

For browse operations:
- MemoryFS: file system-like memory browsing

For search/read operations:
- ReadAPI: search_memory, read_memory with vector search

For async index synchronization:
- IndexService: manages OutboxScheduler for background index updates
"""

from service.api import (
    MemoryWriteAPI,
    ReadAPI,
    init_write_api,
    get_write_api,
)
from service.memory_fs import MemoryFS
from service.index_service import (
    IndexService,
    init_index_service,
    get_index_service,
    setup_signal_handlers,
)

__all__ = [
    "MemoryWriteAPI",
    "ReadAPI",
    "MemoryFS",
    "init_write_api",
    "get_write_api",
    "IndexService",
    "init_index_service",
    "get_index_service",
    "setup_signal_handlers",
]
