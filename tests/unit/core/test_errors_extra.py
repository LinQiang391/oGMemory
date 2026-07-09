"""Additional tests for core.errors."""

from __future__ import annotations

import pytest

from core import errors as errors_module


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: errors_module.OutboxError("evt-1", "bad payload"), "Outbox event evt-1 failed: bad payload"),
        (lambda: errors_module.MergePolicyError("ctx://x", "no route"), "Merge policy failed for ctx://x: no route"),
        (lambda: errors_module.EmbeddingError("hello world", "boom"), "Embedding failed for 'hello world...'"),
        (lambda: errors_module.VectorIndexError("upsert", "boom"), "Vector index upsert failed: boom"),
        (lambda: errors_module.ValidationError("field", "required"), "Validation failed for field: required"),
        (lambda: errors_module.RetrievalError("stage", "boom", retryable=True), "Retrieval [stage] failed: boom"),
        (lambda: errors_module.RetrievalPlannerError("no match"), "Retrieval [planner] failed: no match"),
        (lambda: errors_module.RetrievalSearchError("timeout"), "Retrieval [search] failed: timeout"),
    ],
)
def test_error_messages_and_attributes(factory, expected):
    error = factory()

    assert str(error).startswith(expected)


def test_embedding_error_truncates_text_preview():
    error = errors_module.EmbeddingError("x" * 120, "boom")

    assert len(error.text_preview) == 100


def test_retrieval_search_error_defaults_retryable_true():
    error = errors_module.RetrievalSearchError("timeout")

    assert error.retryable is True
