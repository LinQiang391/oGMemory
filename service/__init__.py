"""Service layer - external API for ContextEngine.

This is the ONLY layer where RequestContext is mandatory.
All external calls must provide a RequestContext for multi-tenant isolation.
"""

from service.api import MemoryReadAPI, init_api, get_api, Level

__all__ = ["MemoryReadAPI", "init_api", "get_api", "Level"]
