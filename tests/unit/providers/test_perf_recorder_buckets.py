"""End-to-end smoke tests for the bucket-based perf recorder.

Verifies that ``Recorder.snapshot_tokens`` / ``Recorder.finalize_tokens``
correctly attribute token usage to each span when concurrent threads
record into the same global ``TokenTracker``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from perf.recorder import Recorder
from perf.sinks import MemorySink
from providers.token_tracker import TokenTracker


@pytest.fixture()
def recorder() -> Recorder:
    rec = Recorder()
    rec.enable(sink=MemorySink(), run_id="test-run")
    return rec


def _emit_span(rec: Recorder, tracker: TokenTracker, input_tok: int, output_tok: int) -> dict:
    """Open a perf scope, record one LLM call, and return the finalised totals."""
    before = rec.snapshot_tokens()
    tracker.record_llm(input_tok, output_tok)
    after = rec.snapshot_tokens()
    tokens, cost, source = rec.finalize_tokens(after, before)
    return {"tokens": tokens, "cost": cost, "source": source}


@pytest.mark.unit
def test_recorder_attributes_only_within_scope(recorder: Recorder) -> None:
    """Tokens recorded outside ``snapshot_tokens..finalize_tokens`` must NOT appear."""
    tracker = TokenTracker(model="gpt-4o-mini")

    tracker.record_llm(999, 999)  # outside any perf scope
    result = _emit_span(recorder, tracker, 100, 50)
    tracker.record_llm(999, 999)  # outside again

    llm = result["tokens"]["llm"]
    assert llm["input_tokens"] == 100
    assert llm["output_tokens"] == 50
    assert result["tokens"]["llm_model"] == "gpt-4o-mini"
    assert result["source"] == "tracker"


@pytest.mark.unit
def test_recorder_concurrent_threads_no_pollution(recorder: Recorder) -> None:
    """N threads each open their own scope and see ONLY their own tokens.

    This is the regression test for the original bug — under the old
    snapshot-diff model, late-finishing threads picked up siblings'
    tokens. With buckets, every thread's scope is isolated.
    """
    n_threads = 32
    barrier = threading.Barrier(n_threads)
    tracker = TokenTracker(model="gpt-4o-mini")
    results: list[dict] = [None] * n_threads  # type: ignore[list-item]

    def _worker(idx: int) -> None:
        my_in = (idx + 1) * 1000
        my_out = (idx + 1) * 500
        before = recorder.snapshot_tokens()
        barrier.wait()
        tracker.record_llm(my_in, my_out)
        after = recorder.snapshot_tokens()
        tokens, _cost, _source = recorder.finalize_tokens(after, before)
        results[idx] = {
            "got_in": tokens["llm"]["input_tokens"],
            "got_out": tokens["llm"]["output_tokens"],
            "expected_in": my_in,
            "expected_out": my_out,
        }

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(_worker, range(n_threads)))

    for idx, r in enumerate(results):
        assert r is not None
        assert r["got_in"] == r["expected_in"], (
            f"thread {idx}: got {r['got_in']} input tokens, expected {r['expected_in']} "
            "(bucket isolation broke under concurrency!)"
        )
        assert r["got_out"] == r["expected_out"]

    # The global tracker should have the cumulative total of every thread
    total_expected_in = sum((i + 1) * 1000 for i in range(n_threads))
    snap = tracker.snapshot()
    assert snap.input_tokens == total_expected_in


@pytest.mark.unit
def test_recorder_disabled_returns_empty_handle() -> None:
    """When the recorder is disabled, snapshot/finalize are no-ops."""
    rec = Recorder()  # not enabled
    before = rec.snapshot_tokens()
    after = rec.snapshot_tokens()
    tokens, cost, source = rec.finalize_tokens(after, before)
    assert before == {}
    assert after == {}
    assert tokens == {}
    assert source == "missing"
    assert cost["total"] == 0.0
