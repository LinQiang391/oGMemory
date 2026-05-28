"""Concurrency tests for the per-context TokenBucket attribution.

These tests guard the property that motivated the bucket redesign:
multiple threads can run :class:`providers.token_tracker.TokenTracker`
recordings simultaneously and each thread's bucket sees only its own
calls, with no cross-thread pollution.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from providers.token_tracker import (
    TokenBucket,
    TokenTracker,
    current_buckets,
    pop_bucket,
    push_bucket,
)


@pytest.mark.unit
def test_push_pop_isolated_per_thread() -> None:
    """Buckets pushed in one thread MUST NOT appear in another thread."""
    main_bucket, main_token = push_bucket()
    seen_in_other_thread: list[int] = []

    def _worker() -> None:
        seen_in_other_thread.append(len(current_buckets()))

    t = threading.Thread(target=_worker)
    t.start()
    t.join()

    assert seen_in_other_thread == [0], (
        "ContextVar bucket stack must NOT leak from main thread to spawned thread"
    )
    assert len(current_buckets()) == 1
    pop_bucket(main_token)
    assert len(current_buckets()) == 0


@pytest.mark.unit
def test_concurrent_threads_no_token_pollution() -> None:
    """Each thread's bucket reflects ONLY its own recordings.

    Spawn N threads, each with its own bucket and its own TokenTracker
    (simulating distinct providers). Have each thread record a unique
    token count, then verify:
      * its bucket equals exactly what it recorded
      * no bucket contains any other thread's tokens
    """
    n_threads = 16
    barrier = threading.Barrier(n_threads)
    results: dict[int, dict[str, int]] = {}
    lock = threading.Lock()

    def _worker(idx: int) -> None:
        # Distinct token counts so cross-pollution is detectable
        my_input = (idx + 1) * 100
        my_output = (idx + 1) * 50
        my_embed = (idx + 1) * 10

        tracker_llm = TokenTracker(model=f"llm-{idx}")
        tracker_embed = TokenTracker(model=f"embed-{idx}")

        bucket, token = push_bucket()
        try:
            barrier.wait()  # release all threads simultaneously
            tracker_llm.record_llm(my_input, my_output)
            tracker_embed.record_embed(my_embed)
        finally:
            data = bucket.to_attribution_dict()
            pop_bucket(token)

        with lock:
            results[idx] = {
                "llm_input": data["llm"]["input_tokens"],
                "llm_output": data["llm"]["output_tokens"],
                "embed": data["embed"]["embed_tokens"],
                "expected_input": my_input,
                "expected_output": my_output,
                "expected_embed": my_embed,
                "llm_model": data.get("llm_model"),
                "embed_model": data.get("embed_model"),
            }

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(ex.map(_worker, range(n_threads)))

    assert len(results) == n_threads
    for idx, r in results.items():
        assert r["llm_input"] == r["expected_input"], (
            f"thread {idx}: LLM input contaminated — got {r['llm_input']}, "
            f"expected {r['expected_input']}"
        )
        assert r["llm_output"] == r["expected_output"], (
            f"thread {idx}: LLM output contaminated"
        )
        assert r["embed"] == r["expected_embed"], (
            f"thread {idx}: embed contaminated"
        )
        assert r["llm_model"] == f"llm-{idx}"
        assert r["embed_model"] == f"embed-{idx}"


@pytest.mark.unit
def test_nested_buckets_both_receive_recording() -> None:
    """When buckets nest, every active bucket receives each recording."""
    tracker = TokenTracker(model="m1")

    outer, outer_token = push_bucket()
    tracker.record_llm(100, 50)

    inner, inner_token = push_bucket()
    tracker.record_llm(200, 80)
    tracker.record_embed(15)
    pop_bucket(inner_token)

    tracker.record_llm(50, 20)
    pop_bucket(outer_token)

    inner_data = inner.to_attribution_dict()
    outer_data = outer.to_attribution_dict()

    # Inner saw only what happened during its scope
    assert inner_data["llm"]["input_tokens"] == 200
    assert inner_data["llm"]["output_tokens"] == 80
    assert inner_data["embed"]["embed_tokens"] == 15

    # Outer saw everything (parent semantics: total includes children)
    assert outer_data["llm"]["input_tokens"] == 350  # 100 + 200 + 50
    assert outer_data["llm"]["output_tokens"] == 150  # 50 + 80 + 20
    assert outer_data["embed"]["embed_tokens"] == 15


@pytest.mark.unit
def test_global_tracker_unaffected_by_buckets() -> None:
    """Global cumulative counters keep working alongside buckets."""
    tracker = TokenTracker(model="m1")

    bucket, token = push_bucket()
    tracker.record_llm(100, 50)
    pop_bucket(token)

    tracker.record_llm(7, 3)  # outside any bucket — global only

    snap = tracker.snapshot()
    assert snap.input_tokens == 107
    assert snap.output_tokens == 53


@pytest.mark.unit
def test_recording_outside_any_bucket_is_safe() -> None:
    """No active bucket = no-op for the bucket layer; no exceptions."""
    tracker = TokenTracker(model="m1")
    assert current_buckets() == ()
    tracker.record_llm(10, 5)
    tracker.record_embed(20)
    tracker.record_local_cache_hit(tokens_saved=3)
    tracker.record_local_cache_miss()
    snap = tracker.snapshot()
    assert snap.input_tokens == 10
    assert snap.embed_tokens == 20
    assert snap.local_cache_hits == 1


@pytest.mark.unit
def test_cache_hit_propagates_to_bucket() -> None:
    tracker = TokenTracker(model="m1")
    bucket, token = push_bucket()
    tracker.record_local_cache_hit(tokens_saved=42)
    tracker.record_local_cache_miss()
    pop_bucket(token)
    data = bucket.to_attribution_dict()
    assert data["llm"]["local_cache_hits"] == 1
    assert data["llm"]["local_cache_misses"] == 1
    assert data["llm"]["local_cache_saved_tokens"] == 42
