"""Service layer - external API for ContextEngine.

This is the ONLY layer where RequestContext is mandatory.
All external calls must provide a RequestContext for multi-tenant isolation.

For write operations:
- MemoryWriteAPI: commit_session for writing memories

For browse operations (dev branch):
- MemoryFS: file system-like memory browsing

For full read/search operations (phase1 branch):
- MemoryReadAPI: search_memory, read_memory with vector search
"""

from service.api import (
    MemoryWriteAPI,
    init_write_api,
    get_write_api,
)
from service.memory_fs import MemoryFS

__all__ = [
    "MemoryWriteAPI",
    "MemoryFS",
    "init_write_api",
    "get_write_api",
]
