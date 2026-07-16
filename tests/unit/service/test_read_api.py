"""Unit tests for the public memory read API."""

from types import SimpleNamespace

import pytest

from core.models import RequestContext
from service.api import ReadAPI


class _Pipeline:
    def __init__(self, result):
        self.result = result

    def run(self, *_args, **_kwargs):
        return self.result


@pytest.mark.parametrize("include_debug", [False, True])
def test_search_memory_controls_debug_trace(include_debug):
    trace = object()
    result = SimpleNamespace(trace=trace, request_id="request-1")
    api = ReadAPI(pipeline=_Pipeline(result))
    ctx = RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
    )

    returned = api.search_memory("query", ctx, include_debug=include_debug)

    assert returned.request_id == "request-1"
    assert returned.trace is (trace if include_debug else None)
