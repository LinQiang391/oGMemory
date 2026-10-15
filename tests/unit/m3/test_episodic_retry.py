import json
from types import SimpleNamespace

import pytest

from m3 import episodic_memory as module


@pytest.mark.parametrize("message,retryable", [
    ("HTTP 529 overloaded", True),
    ("Error code: 429 - rate limit", True),
    ("{'http_code': 503}", True),
    ("HTTP/1.1 502 Bad Gateway", True),
    ("HTTP 401 unauthorized", False),
    ("HTTP 400 bad request", False),
    ("invalid schema at clip 529", False),
])
def test_retryable_error_classification(message, retryable):
    assert module._retryable_generation_error(RuntimeError(message)) == retryable


@pytest.mark.parametrize("recover", [True, False])
def test_bounded_exponential_backoff(monkeypatch, recover):
    attempts, sleeps = [], []
    def generate(**kwargs):
        attempts.append(1)
        if recover and len(attempts) == 4:
            return {"description": "ok"}
        raise RuntimeError("Error code: 529")
    monkeypatch.setattr(module, "generate_episodic_for_clip", generate)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module.random, "uniform", lambda a, b: 1)
    args = dict(max_retries=3, retry_base_seconds=10, retry_max_seconds=30,
                video_raw_node=SimpleNamespace(node_id="clip"))
    if recover:
        result, retries = module._generate_with_retry(**args)
        assert result["description"] == "ok"
        assert retries == 3
    else:
        with pytest.raises(RuntimeError, match="529"):
            module._generate_with_retry(**args)
    assert len(attempts) == 4
    assert sleeps == [10, 20, 30]


def test_permanent_failure_is_not_retried(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("HTTP 401")
    monkeypatch.setattr(module, "generate_episodic_for_clip", fail)
    monkeypatch.setattr(module.time, "sleep", lambda _: pytest.fail("must not sleep"))
    with pytest.raises(RuntimeError, match="401"):
        module._generate_with_retry(
            max_retries=3, retry_base_seconds=10, retry_max_seconds=60,
            video_raw_node=SimpleNamespace(node_id="clip"),
        )


@pytest.mark.parametrize("options", [
    ["--max-retries", "-1"], ["--max-retries", "11"],
    ["--retry-base-seconds", "61", "--retry-max-seconds", "60"],
    ["--retry-base-seconds", "nan"], ["--retry-max-seconds", "301"],
])
def test_retry_settings_are_bounded(options):
    with pytest.raises(SystemExit):
        module.parse_args([
            "--video-records-manifest", "v.jsonl", "--frames-manifest", "f.jsonl",
            *options,
        ])


@pytest.mark.parametrize("receipt", [{}, {"account_id": "another-account"}])
def test_retry_account_mismatch_preserves_diagnostics(tmp_path, receipt):
    output = tmp_path / "episodic_memory.jsonl"
    summary = output.with_suffix(".summary.json")
    failures = tmp_path / "episodic_memory_failures.json"
    summary.write_text(json.dumps(receipt))
    failures.write_text('[{"video_id": "v", "clip_id": 0}]')
    original_summary, original_failures = summary.read_bytes(), failures.read_bytes()
    args = module.parse_args([
        "--video-records-manifest", "v.jsonl", "--frames-manifest", "f.jsonl",
        "--output-jsonl", str(output), "--account-id", "experiment",
        "--retry-failed",
    ])
    with pytest.raises(ValueError, match="same account_id"):
        module.run(args)
    assert summary.read_bytes() == original_summary
    assert failures.read_bytes() == original_failures
