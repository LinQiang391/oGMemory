"""Structured stage failures must reach the batch driver's process status."""

import json

import pytest

from m3 import run_video_pipeline


@pytest.mark.parametrize("ok,warnings,exit_code", [
    (True, [], 0),
    (True, ["degraded but usable"], 0),
    (False, [], 2),
])
def test_cli_returns_nonzero_for_failed_stage(monkeypatch, capsys, ok, warnings, exit_code):
    summary = {"stages": {
        "semantic": {"ok": ok, "warnings": warnings, "errors": []},
    }}
    monkeypatch.setattr(run_video_pipeline, "run_pipeline", lambda args: summary)
    assert run_video_pipeline.main(["--video", "unused.mp4"]) == exit_code
    assert json.loads(capsys.readouterr().out) == summary
