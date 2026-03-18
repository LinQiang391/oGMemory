"""Service layer - external API for ContextEngine.

This is the ONLY layer where RequestContext is mandatory.
All external calls must provide a RequestContext for multi-tenant isolation.

For write operations, see the commit package:
- ContextWriter: write ContextNodes to storage
- OutboxStore: register events for async indexing

For read/search operations, see the retrieval package (not included in this branch).
"""

__all__ = []
