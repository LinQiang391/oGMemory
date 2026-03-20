"""OpenClaw plugin bridge for ContextEngine memory writes.

This module provides the write_memory function that integrates with OpenClaw
to store conversation memories in ContextEngine.
"""

import uuid
from typing import Dict, Any, Optional

from service.api import MemoryWriteAPI, get_write_api
from core.models import RequestContext, CandidateMemory


# Category to owner_scope mapping per CLAUDE.md §1 URI Spec
# profile/preference/entity/event → user_space
# case/pattern/skill → agent_space
CATEGORY_OWNER_SCOPE = {
    "profile": "user",
    "preference": "user",
    "entity": "user",
    "event": "user",
    "case": "agent",
    "pattern": "agent",
    "skill": "agent",
}

DEFAULT_OWNER_SCOPE = "user"


def write_memory(
    content: str,
    category: str,
    account_id: str,
    user_id: str,
    agent_id: str,
    session_id: str,
    trace_id: Optional[str] = None,
    routing_key: Optional[str] = None,
    abstract: Optional[str] = None,
    overview: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a memory to ContextEngine via OpenClaw plugin.

    Args:
        content: Full memory content
        category: Memory category (profile, preference, entity, event, case, pattern, skill)
        account_id: Tenant account ID
        user_id: User ID
        agent_id: Agent ID
        session_id: Session ID
        trace_id: Optional trace ID for observability
        routing_key: Stable identifier for this memory topic (e.g., 'coffee', 'python_style')
        abstract: Optional pre-computed abstract (auto-generated if not provided)
        overview: Optional pre-computed overview (auto-generated if not provided)

    Returns:
        Dict with write result:
        {
            "action": str,  # "create" | "merge" | "skip"
            "target_uri": str,
            "merged_fields": dict,
        }
    """
    # Get or create the write API
    api = get_write_api()
    if api is None:
        raise RuntimeError(
            "MemoryWriteAPI not initialized. "
            "Call init_write_api(fs, llm, outbox_store) first."
        )

    # Determine owner_scope from category (Bug 2 fix)
    owner_scope = CATEGORY_OWNER_SCOPE.get(category, DEFAULT_OWNER_SCOPE)

    # Bug 1 fix: Use routing_key from input, default to category
    if not routing_key:
        routing_key = category

    # Auto-generate abstract and overview if not provided (Bug 3 fix)
    if not abstract:
        # Extract first sentence from content for better L0 retrieval
        first_sentence = content.split("。")[0].split(".")[0].strip()
        abstract = first_sentence[:100] if first_sentence else content[:100]
        if len(content) > 100 and len(abstract) == 100:
            abstract += "..."

    if not overview:
        # Create structured overview from content
        overview = f"## {category.capitalize()}\n\n{content[:500]}"
        if len(content) > 500:
            overview += "..."

    # Build RequestContext
    ctx = RequestContext(
        account_id=account_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        trace_id=trace_id or f"openclaw_{uuid.uuid4().hex[:16]}",
    )

    # Create CandidateMemory
    candidate = CandidateMemory(
        category=category,
        owner_scope=owner_scope,
        routing_key=routing_key,
        abstract=abstract,
        overview=overview,
        content=content,
        confidence=0.9,  # Default confidence for explicit writes
    )

    # Write to ContextEngine
    result = api.write_memory(candidate, ctx)

    return result


def init_context_engine(
    fs: Any,
    llm: Any,
    outbox_store: Optional[Any] = None,
) -> MemoryWriteAPI:
    """Initialize ContextEngine for OpenClaw plugin.

    Args:
        fs: ContextFS implementation (AGFS client)
        llm: LLM instance for extraction
        outbox_store: Optional OutboxStore for async indexing

    Returns:
        Configured MemoryWriteAPI instance
    """
    from service.api import init_write_api
    return init_write_api(fs, llm, outbox_store)
