# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ReAct loop for memory extraction with tool use.

Adapted from OpenViking's extract_loop.py for ContextEngine.
Implements iterative tool-assisted extraction with safety checks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

from core.interfaces import ContextFS, LLM
from core.models import CandidateMemory, RequestContext
from core.uri_resolver import URIResolver
from extraction.prefetch import MemoryPrefetcher, PrefetchResult
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import _ROUTING_KEY_FIELDS, build_extraction_tools, parse_tool_call

logger = getLogger(__name__)


# Built-in tool definitions for reading memory nodes
_READ_TOOL = {
    "name": "read",
    "description": "Read a memory node by URI.",
    "input_schema": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "URI of the memory node to read.",
            }
        },
        "required": ["uri"],
    },
}

_LIST_TOOL = {
    "name": "list",
    "description": "List child nodes under a directory URI.",
    "input_schema": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "URI of the directory to list.",
            }
        },
        "required": ["uri"],
    },
}

_RELATIONS_TOOL = {
    "name": "get_relations",
    "description": "Get related nodes for a URI. Returns one-hop neighbors.",
    "input_schema": {
        "type": "object",
        "properties": {"uri": {"type": "string"}},
        "required": ["uri"],
    },
}

_ACCESS_STATS_TOOL = {
    "name": "get_access_stats",
    "description": "Get access stats for a node: last access time, 30d hit count.",
    "input_schema": {
        "type": "object",
        "properties": {"uri": {"type": "string"}},
        "required": ["uri"],
    },
}


@dataclass
class ReActIterationTrace:
    """Trace data for a single ReAct iteration."""
    iteration: int
    latency_seconds: float
    tool_calls_count: int = 0
    tool_call_names: list = field(default_factory=list)
    content_length: int = 0
    safety_check_triggered: bool = False
    refetch_uris: list = field(default_factory=list)


@dataclass
class ReActTrace:
    """Full trace for a ReAct extraction run."""
    total_iterations: int = 0
    total_latency_seconds: float = 0.0
    iterations: list = field(default_factory=list)
    final_candidate_count: int = 0
    total_read_uris: int = 0


@dataclass
class ReActResult:
    """Result of a ReAct extraction loop."""

    candidates: list[CandidateMemory] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)
    iterations: int = 0
    read_uris: set[str] = field(default_factory=set)
    trace: Optional[ReActTrace] = None


class ExtractionReActLoop:
    """ReAct loop for memory extraction with tool use.

    Workflow:
    1. Start with system prompt + prefetch context + conversation
    2. While iteration < max_iterations:
       a. Call LLM with tools (read, list, extract_*)
       b. If tool calls: execute and continue
       c. If content: parse operations → safety check → return
       d. If neither: disable tools and continue
    3. Return extracted candidates with metadata

    Safety features:
    - Refetch check: re-read existing files before write operations
    - URI validation: ensure all URIs are within allowed namespace
    - Single safety re-read: prevent infinite loops
    """

    def __init__(
        self,
        llm: LLM,
        fs: ContextFS,
        registry: SchemaRegistry,
        uri_resolver: URIResolver,
        prefetcher: MemoryPrefetcher | None = None,
        max_iterations: int = 3,
        timeout_seconds: float = 30.0,
    ):
        """Initialize the ReAct loop.

        Args:
            llm: LLM instance for completion with tools
            fs: ContextFS for memory node operations
            registry: SchemaRegistry for schema definitions
            uri_resolver: URIResolver for URI generation and validation
            prefetcher: Optional MemoryPrefetcher for context loading
            max_iterations: Maximum number of ReAct iterations
            timeout_seconds: Maximum seconds to run before graceful degradation
        """
        self._llm = llm
        self._fs = fs
        self._registry = registry
        self._uri_resolver = uri_resolver
        self._prefetcher = prefetcher
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds

        # Build extraction tools from registry
        self._extraction_tools = build_extraction_tools(registry)

        # All available tools: read, list, relations, access_stats, and extraction tools
        self._all_tools = [_READ_TOOL, _LIST_TOOL, _RELATIONS_TOOL, _ACCESS_STATS_TOOL] + self._extraction_tools

        # Runtime state (reset per run())
        self._read_files: set[str]
        self._disable_tools_for_iteration: bool
        self._did_safety_reread: bool

    def run(
        self,
        conversation_text: str,
        ctx: RequestContext,
        prefetch_result: PrefetchResult | None = None,
    ) -> ReActResult:
        """Run the ReAct loop for memory extraction.

        Args:
            conversation_text: Text content of the conversation to extract from
            ctx: RequestContext for multi-tenant isolation
            prefetch_result: Optional prefetch result with pre-loaded context

        Returns:
            ReActResult with extracted candidates and metadata
        """
        # Reset runtime state
        self._read_files = set()
        self._disable_tools_for_iteration = False
        self._did_safety_reread = False

        # Track prefetch URIs if provided
        if prefetch_result:
            self._read_files.update(prefetch_result.read_uris)

        # Build initial messages
        messages = self._build_initial_messages(conversation_text, prefetch_result, ctx)

        # Initialize trace and timing
        trace = ReActTrace()
        run_start = time.monotonic()
        self._run_start_time = run_start

        iteration = 0
        max_iterations = self._max_iterations
        final_candidates = []
        tools_used = []

        while iteration < max_iterations:
            iteration += 1
            iter_start = time.monotonic()
            logger.info(f"ReAct iteration {iteration}/{max_iterations}")

            # Timeout check with graceful degradation
            elapsed = time.monotonic() - self._run_start_time
            if elapsed > self._timeout_seconds and iteration > 1:
                logger.warning(
                    f"ReAct timeout ({elapsed:.1f}s > {self._timeout_seconds}s) "
                    f"at iteration {iteration}, returning current results"
                )
                # Use whatever we have so far
                break

            # Check if this is the last iteration
            is_last_iteration = iteration >= max_iterations

            # If last iteration, add termination message
            if is_last_iteration:
                messages.append({
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of tool call iterations. "
                        "Do not call any more tools - return your final result directly now."
                    ),
                })

            # Call LLM with tools
            tool_calls, content = self._call_llm(messages)
            iter_latency = time.monotonic() - iter_start

            # Track iteration data
            iter_trace = ReActIterationTrace(
                iteration=iteration,
                latency_seconds=iter_latency,
                tool_calls_count=len(tool_calls) if tool_calls else 0,
                tool_call_names=[tc.get("name", "") for tc in tool_calls] if tool_calls else [],
                content_length=len(content) if content else 0,
            )

            # Case 1: Model made tool calls
            if tool_calls:
                has_unknown = self._execute_tool_calls(
                    messages, tool_calls, tools_used, ctx
                )
                if has_unknown:
                    self._disable_tools_for_iteration = True
                    logger.info("Unknown tool called, disabling tools for next iteration")

                # Extend max_iterations if needed
                if iteration >= max_iterations:
                    max_iterations += 1
                    self._disable_tools_for_iteration = True
                    logger.info(f"Extended max_iterations to {max_iterations} for tool call")

                trace.iterations.append(iter_trace)
                continue

            # Case 2: Model returned content (operations)
            if content:
                candidates = self._parse_operations(content)
                if candidates:
                    # Safety check: refetch unread existing files
                    refetch_uris = self._check_unread_existing_files(candidates, ctx)
                    if refetch_uris:
                        logger.info(f"Found unread existing files: {refetch_uris}, refetching...")
                        iter_trace.safety_check_triggered = True
                        iter_trace.refetch_uris = refetch_uris
                        self._add_refetch_results(messages, refetch_uris, ctx)

                        # Extend max_iterations for refetch
                        if iteration >= max_iterations:
                            max_iterations += 1
                            logger.info(f"Extended max_iterations to {max_iterations} for refetch")

                        trace.iterations.append(iter_trace)
                        continue

                    # Validate URIs before returning
                    errors = self._validate_operations_uris(candidates, ctx)
                    if errors:
                        logger.error(f"URI validation errors: {errors}")
                        # Still return candidates but log errors

                    trace.iterations.append(iter_trace)
                    final_candidates = candidates
                    break

            # Case 3: Neither tool calls nor content
            trace.iterations.append(iter_trace)
            if is_last_iteration:
                logger.warning("Last iteration: returning empty candidates")
                break

            logger.warning(
                f"LLM returned neither tool calls nor content (iteration {iteration}/{max_iterations})"
            )
            self._disable_tools_for_iteration = True

        # Finalize trace
        trace.total_iterations = iteration
        trace.total_latency_seconds = time.monotonic() - run_start
        trace.final_candidate_count = len(final_candidates)
        trace.total_read_uris = len(self._read_files)

        result = ReActResult(
            candidates=final_candidates,
            tools_used=tools_used,
            iterations=iteration,
            read_uris=self._read_files.copy(),
            trace=trace,
        )
        return result

    def _build_initial_messages(
        self,
        conversation_text: str,
        prefetch_result: PrefetchResult | None,
        ctx: RequestContext,
    ) -> list[dict]:
        """Build initial messages for the LLM.

        Args:
            conversation_text: Conversation text to extract from
            prefetch_result: Optional prefetch result
            ctx: RequestContext

        Returns:
            List of messages in OpenAI format
        """
        messages = []

        # System prompt with instruction
        system_prompt = self._build_system_prompt(ctx)
        messages.append({"role": "system", "content": system_prompt})

        # Add prefetch context if available
        if prefetch_result and prefetch_result.messages:
            for msg in prefetch_result.messages:
                messages.append({"role": "system", "content": msg})

        # Add conversation text
        messages.append({
            "role": "user",
            "content": f"## Conversation\n\n{conversation_text}",
        })

        return messages

    def _build_system_prompt(self, ctx: RequestContext) -> str:
        """Build system prompt for extraction.

        Args:
            ctx: RequestContext

        Returns:
            System prompt string
        """
        # Build tool descriptions
        tool_descriptions = []
        for tool in self._all_tools:
            desc = f"- {tool['name']}: {tool['description']}"
            if tool.get("input_schema", {}).get("properties"):
                params = ", ".join(tool["input_schema"]["properties"].keys())
                desc += f" (params: {params})"
            tool_descriptions.append(desc)

        prompt = f"""# Memory Extraction Task

You are a memory extraction assistant. Your task is to extract structured memories from the conversation.

## Available Tools

{chr(10).join(tool_descriptions)}

## Instructions

1. Use `read` and `list` tools to explore existing memories before making decisions.
2. Use `extract_*` tools to create new memories or update existing ones.
3. Always read existing content before deciding to merge or create new entries.
4. Return your final result using the extraction tool calls in your response.

## Context

Account: {ctx.account_id}
User: {ctx.user_id}
Agent: {ctx.agent_id}
"""
        return prompt

    def _call_llm(
        self, messages: list[dict]
    ) -> tuple[list[dict] | None, str | None]:
        """Call LLM with tools.

        Args:
            messages: Current message list

        Returns:
            Tuple of (tool_calls, content) - one will be None
        """
        # Determine if tools should be provided
        tools = None
        tool_choice = "none"
        if not self._disable_tools_for_iteration:
            tools = self._all_tools
            tool_choice = "auto"

        try:
            result = self._llm.complete_with_tools_messages(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

            # complete_with_tools_messages should return (tool_calls, content)
            if isinstance(result, tuple) and len(result) == 2:
                return result

            # If single list, it's tool calls
            if isinstance(result, list):
                return result, None

            # If string, it's content
            if isinstance(result, str):
                return None, result

            return None, None

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None, None

    def _execute_tool_calls(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        tools_used: list[dict],
        ctx: RequestContext,
    ) -> bool:
        """Execute tool calls and add results to messages.

        Args:
            messages: Message list to append to
            tool_calls: List of tool call dicts
            tools_used: List to track tool usage
            ctx: RequestContext

        Returns:
            True if any unknown tool was encountered
        """
        has_unknown = False

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_input = tool_call.get("input", {})
            call_id = tool_call.get("id", "unknown")

            # Execute tool
            result = self._execute_single_tool(tool_name, tool_input, ctx)

            # Track unknown tools
            if isinstance(result, dict) and result.get("error", "").startswith("Unknown tool"):
                has_unknown = True

            # Track read operations
            if tool_name == "read" and tool_input.get("uri"):
                self._read_files.add(tool_input["uri"])

            # Record tool usage
            tools_used.append({
                "tool_name": tool_name,
                "params": tool_input,
                "result": result,
            })

            # Add to messages in OpenAI format
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_input),
                    },
                }],
            })

            # Add tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result) if isinstance(result, dict) else str(result),
            })

        return has_unknown

    def _execute_single_tool(
        self, tool_name: str, tool_input: dict, ctx: RequestContext
    ) -> dict | str:
        """Execute a single tool call.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            ctx: RequestContext

        Returns:
            Tool result as dict or string
        """
        try:
            if tool_name == "read":
                uri = tool_input.get("uri", "")
                if not uri:
                    return {"error": "Missing required parameter: uri"}

                node = self._fs.read_node(uri, ctx)
                return {
                    "uri": uri,
                    "abstract": node.abstract,
                    "overview": node.overview,
                    "content": node.content,
                    "metadata": node.metadata,
                }

            elif tool_name == "list":
                uri = tool_input.get("uri", "")
                if not uri:
                    return {"error": "Missing required parameter: uri"}

                children = self._fs.list_children(uri, ctx)
                return {
                    "uri": uri,
                    "children": children,
                }

            elif tool_name == "get_relations":
                uri = tool_input.get("uri", "")
                return self._execute_get_relations(uri, ctx)

            elif tool_name == "get_access_stats":
                uri = tool_input.get("uri", "")
                return self._execute_get_stats(uri, ctx)

            elif tool_name.startswith("extract_"):
                # Extraction tools are parsed later, just record
                return {
                    "tool": tool_name,
                    "input": tool_input,
                    "status": "recorded",
                }

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {"error": str(e)}

    def _parse_operations(self, content: str) -> list[CandidateMemory] | None:
        """Parse extraction operations from LLM content.

        Args:
            content: LLM response content

        Returns:
            List of CandidateMemory or None if parsing fails
        """
        try:
            # Try to parse as JSON
            data = json.loads(content)

            # Look for extract_* tool calls in the content
            candidates = []

            # Handle different JSON structures
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # Check for tool_calls key
                if "tool_calls" in data:
                    items = data["tool_calls"]
                # Check for operations key
                elif "operations" in data:
                    items = data["operations"]
                else:
                    # Treat dict itself as single operation
                    items = [data]
            else:
                logger.warning(f"Unexpected JSON structure: {type(data)}")
                return None

            # Parse each item
            for item in items:
                if not isinstance(item, dict):
                    continue

                # Find tool name and input
                tool_name = item.get("name", "")
                tool_input = item.get("input", item)

                if not tool_name.startswith("extract_"):
                    # Maybe input has the tool name
                    for key in item.keys():
                        if key.startswith("extract_"):
                            tool_name = key
                            tool_input = item[key]
                            break

                if not tool_name.startswith("extract_"):
                    continue

                # Parse tool call using schema
                parsed = parse_tool_call(tool_name, tool_input, self._registry)
                if parsed:
                    memory_type, owner_scope, candidate = parsed
                    candidates.append(candidate)
                    logger.info(f"Parsed candidate: {memory_type}/{candidate.routing_key}")

            return candidates if candidates else None

        except json.JSONDecodeError:
            logger.warning("Failed to parse content as JSON")
            return None
        except Exception as e:
            logger.error(f"Error parsing operations: {e}")
            return None

    def _check_unread_existing_files(
        self, candidates: list[CandidateMemory], ctx: RequestContext
    ) -> list[str]:
        """Check if candidates target existing files that weren't read.

        Args:
            candidates: List of CandidateMemory to check
            ctx: RequestContext

        Returns:
            List of URIs that need to be refetched
        """
        if self._did_safety_reread:
            return []

        refetch_uris = []

        for candidate in candidates:
            # Get schema for this candidate
            schema = self._registry.get(candidate.category)
            if schema is None:
                continue

            # Skip add-only schemas (no conflict risk)
            if schema.is_add_only:
                continue

            # Resolve URI
            fields = self._candidate_to_fields(candidate)
            try:
                uri = self._uri_resolver.resolve(candidate.category, fields, ctx)
            except Exception as e:
                logger.warning(f"Failed to resolve URI for {candidate.category}: {e}")
                continue

            # Check if URI was already read
            if uri in self._read_files:
                continue

            # Check if file exists
            if self._fs.exists(uri, ctx):
                refetch_uris.append(uri)

        return refetch_uris

    def _add_refetch_results(
        self, messages: list[dict], refetch_uris: list[str], ctx: RequestContext
    ) -> None:
        """Add refetched file contents to messages.

        Args:
            messages: Message list to append to
            refetch_uris: URIs to refetch
            ctx: RequestContext
        """
        self._did_safety_reread = True

        for uri in refetch_uris:
            try:
                node = self._fs.read_node(uri, ctx)
                self._read_files.add(uri)

                # Add as read tool result
                result = {
                    "uri": uri,
                    "abstract": node.abstract,
                    "overview": node.overview,
                    "content": node.content,
                    "metadata": node.metadata,
                }

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"refetch_{uri}",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"uri": uri}),
                        },
                    }],
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": f"refetch_{uri}",
                    "content": json.dumps(result),
                })

            except Exception as e:
                logger.warning(f"Failed to refetch {uri}: {e}")

        # Add safety reminder
        messages.append({
            "role": "user",
            "content": (
                "SAFETY REMINDER: The files above were automatically read because they "
                "exist and you didn't read them before deciding to write. Please consider "
                "the existing content when making write decisions. You can now output "
                "updated operations."
            ),
        })

    def _validate_operations_uris(
        self, candidates: list[CandidateMemory], ctx: RequestContext
    ) -> list[str]:
        """Validate that all candidate URIs are within allowed namespace.

        Args:
            candidates: List of CandidateMemory to validate
            ctx: RequestContext

        Returns:
            List of error strings (empty if all valid)
        """
        errors = []

        for candidate in candidates:
            fields = self._candidate_to_fields(candidate)
            try:
                uri = self._uri_resolver.resolve(candidate.category, fields, ctx)
                if not self._uri_resolver.validate_uri(uri, ctx):
                    errors.append(
                        f"Invalid URI for {candidate.category}/{candidate.routing_key}: {uri}"
                    )
            except Exception as e:
                errors.append(
                    f"Failed to validate URI for {candidate.category}/{candidate.routing_key}: {e}"
                )

        return errors

    def _candidate_to_fields(self, candidate: CandidateMemory) -> dict:
        """Convert CandidateMemory to fields dict for URI resolution.

        Args:
            candidate: CandidateMemory to convert

        Returns:
            Dict of field names to values
        """
        # Get the routing key field name for this category
        routing_field = _ROUTING_KEY_FIELDS.get(candidate.category, "routing_key")

        fields = {
            routing_field: candidate.routing_key,
            "abstract": candidate.abstract,
            "overview": candidate.overview,
            "content": candidate.content,
            "confidence": candidate.confidence,
        }

        # Add optional fields if present
        if candidate.when:
            fields["when"] = candidate.when
        if candidate.who:
            fields["who"] = candidate.who
        if candidate.where:
            fields["where"] = candidate.where
        if candidate.tool_stats:
            fields.update(candidate.tool_stats)

        return fields

    def _execute_get_relations(self, uri: str, ctx: RequestContext) -> dict:
        """Execute get_relations tool call.

        Args:
            uri: URI to get relations for
            ctx: RequestContext

        Returns:
            Dict with uri and relations list
        """
        try:
            if hasattr(self._fs, 'get_relations'):
                edges = self._fs.get_relations(uri, ctx)
                return {"uri": uri, "relations": [
                    {"target": e.target_uri, "weight": e.weight, "reason": e.reason}
                    for e in edges
                ]}
            return {"uri": uri, "relations": [], "note": "Not available"}
        except Exception as e:
            return {"error": str(e)}

    def _execute_get_stats(self, uri: str, ctx: RequestContext) -> dict:
        """Execute get_access_stats tool call.

        Args:
            uri: URI to get access stats for
            ctx: RequestContext

        Returns:
            Dict with uri, last_accessed_at, and hit_count_30d
        """
        try:
            node = self._fs.read_node(uri, ctx)
            metadata = node.metadata or {}
            return {
                "uri": uri,
                "last_accessed_at": metadata.get("last_accessed_at"),
                "hit_count_30d": metadata.get("hit_count_30d", 0),
            }
        except Exception as e:
            return {"error": str(e)}
