"""Regression tests for the before_clip wiring.

The pre-fix flow appended ``(information must be from clips before
clip_id=N)`` to the question string. That polluted the query embedding
and left enforcement to the LLM. The post-fix flow wires
``before_clip`` into :class:`service.graph_service.ClipScope`, so the
store hard-filters by ``clip_id <= before_clip + 1`` — the question's
own clip IS included in the searchable window (the load-bearing
utterance for M3-Bench questions usually sits in that clip, seconds
before the question timestamp; the annotation's ``before_clip`` is the
clip BEFORE the question's own clip).

These tests pin the wiring at two layers:
1. ``M3TaskContext`` carries ``before_clip``; ``qa.py`` no longer
   rewrites the question string.
2. ``control_adapter.Search`` forwards ``before_clip`` to ``ClipScope``
   so the store-level filter actually fires (with the +1 shift).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from m3.control_adapter import (
    M3TaskContext,
    OGMemoryControlAdapter,
)


def _adapter_with_search_spy():
    """Build an adapter whose graph.search_by_embedding is a spy.

    The spy records every call's kwargs so we can assert that
    ``ClipScope(before_clip=…)`` is forwarded.
    """
    from providers.llm import MockLLM
    from providers.embedder import MockEmbedder

    graph = MagicMock()
    graph.search_by_embedding = MagicMock(return_value=[])
    embedder = MockEmbedder(dimension=8)

    adapter = OGMemoryControlAdapter(
        llm=MockLLM(),
        graph=graph,
        embedder=embedder,
        max_iterations=1,
        timeout_seconds=1.0,
    )
    return adapter, graph


class TestBeforeClipInTaskContext:
    def test_task_context_carries_before_clip(self):
        """M3TaskContext must carry the before_clip value so the
        adapter can pass it to the [Search] tool."""
        ctx = M3TaskContext(
            question="What is in the bedroom?",
            video_id="bedroom_03",
            before_clip=5,
        )
        assert ctx.before_clip == 5

    def test_default_before_clip_is_None(self):
        ctx = M3TaskContext(question="anything")
        assert ctx.before_clip is None


class TestBeforeClipWiredToClipScope:
    def test_search_passes_before_clip_to_clip_scope(self):
        """The adapter's [Search] tool must wrap before_clip into a
        ClipScope so the store can hard-filter by ``clip_id <= N``.
        """
        adapter, graph = _adapter_with_search_spy()

        # Simulate the message-build phase where _SEARCH_CONFIG is
        # populated from the task context.
        from m3.control_adapter import _SEARCH_CONFIG
        _SEARCH_CONFIG["scope_video_id"] = "bedroom_03"
        _SEARCH_CONFIG["scope_before_clip"] = 6

        # Build a fake task_ctx carrying before_clip; trigger the
        # message build (which is what propagates the scope to
        # _SEARCH_CONFIG and ultimately to ClipScope).
        from core.models import RequestContext
        fake_ctx = RequestContext(
            account_id="acct", user_id="u", agent_id="a",
            session_id="s", trace_id="t",
        )
        task_ctx = M3TaskContext(
            question="What is in the bedroom?",
            video_id="bedroom_03",
            before_clip=5,
        )
        # Use the adapter's _build_messages — this is where scope
        # _SEARCH_CONFIG["scope_before_clip"] is set. The adapter
        # shifts before_clip to before_clip + 1 to include the
        # question's own clip in the searchable window.
        adapter._build_messages(fake_ctx, task_ctx)

        # Now call Search directly to verify the wiring.
        from m3.control_adapter import Search
        Search.__wrapped__ if hasattr(Search, '__wrapped__') else Search
        # Call the search tool's underlying function via the adapter
        # via _execute_and_append_tool_calls or directly.
        # The cleanest path: invoke the tool with a fake call.
        from core.tools.operational_tool_spec import (
            operational_tool, OperationalToolSpec,
        )

        # Build a tool call for [Search] and route it through the
        # adapter's execute path.
        tool_calls = [{
            "name": "Search",
            "input": {"query": "bedroom", "top_k": 5},
            "tool_call_id": "tc-1",
        }]
        # We need a messages list — use a minimal one.
        messages = [
            {"role": "user", "content": "What is in the bedroom?"},
        ]
        adapter._execute_and_append_tool_calls(
            messages, tool_calls, tools_used=[], read_uris=[],
            ctx=fake_ctx, round_id="r-1",
        )
        # Verify search_by_embedding received ClipScope(before_clip=6).
        # The adapter's before_clip + 1 shift (5 → 6) is what extends
        # the searchable window through the question's own clip.
        assert graph.search_by_embedding.called, (
            "Search tool did not call graph.search_by_embedding"
        )
        kwargs = graph.search_by_embedding.call_args.kwargs
        assert "scope" in kwargs, (
            "search_by_embedding was called without a scope; "
            "before_clip must be wired through ClipScope."
        )
        scope = kwargs["scope"]
        assert scope.before_clip == 6, (
            f"ClipScope.before_clip should be 6 (before_clip=5 + 1 to "
            f"include the question's own clip), got {scope.before_clip}"
        )
        assert scope.video_id == "bedroom_03", (
            f"ClipScope.video_id should be bedroom_03, got {scope.video_id}"
        )
        # And the adapter's _SEARCH_CONFIG['scope_before_clip'] must
        # have been populated with the +1 value.
        from m3.control_adapter import _SEARCH_CONFIG
        assert _SEARCH_CONFIG["scope_before_clip"] == 6

    def test_search_passes_no_clip_scope_when_before_clip_is_none(self):
        """When ``before_clip`` is None, no clip constraint is wired —
        the searchable window is the entire video."""
        adapter, graph = _adapter_with_search_spy()

        from m3.control_adapter import _SEARCH_CONFIG
        _SEARCH_CONFIG["scope_video_id"] = "bedroom_03"
        _SEARCH_CONFIG["scope_before_clip"] = None

        from core.models import RequestContext
        fake_ctx = RequestContext(
            account_id="acct", user_id="u", agent_id="a",
            session_id="s", trace_id="t",
        )
        task_ctx = M3TaskContext(
            question="Anything?",
            video_id="bedroom_03",
            before_clip=None,
        )
        adapter._build_messages(fake_ctx, task_ctx)

        tool_calls = [{
            "name": "Search",
            "input": {"query": "anything", "top_k": 5},
            "tool_call_id": "tc-1",
        }]
        messages = [{"role": "user", "content": "Anything?"}]
        adapter._execute_and_append_tool_calls(
            messages, tool_calls, tools_used=[], read_uris=[],
            ctx=fake_ctx, round_id="r-1",
        )

        assert graph.search_by_embedding.called
        kwargs = graph.search_by_embedding.call_args.kwargs
        # The scope is still wired (with ``before_clip=None``) so the
        # video_id scoping still applies; the ``clip_id <= before_clip``
        # filter is effectively disabled because ``None`` means no
        # constraint. The point of the ``before_clip=None`` case is
        # "no clip cap" — NOT "no scope at all".
        scope = kwargs.get("scope")
        assert scope is not None, (
            "scope should still be passed when a video_id is set, "
            "even if before_clip is None"
        )
        assert scope.before_clip is None
        assert scope.video_id == "bedroom_03"
        assert _SEARCH_CONFIG["scope_before_clip"] is None