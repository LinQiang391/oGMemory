"""Shared fixtures for M3-Bench integration tests.

All tests in this directory are pure-Python (no real ffmpeg, no real
PostgreSQL). External tools are mocked via ``monkeypatch`` so the tests
run in CI without system dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Replace ``subprocess.run`` with a stub that pretends to slice videos.

    The stub records every ffmpeg / ffprobe call so tests can assert on
    them. ``ffprobe_duration`` returns a fixed value (60.0 seconds) so
    test fixtures can rely on deterministic clip counts.
    """
    import m3.preprocess as pp

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output=True, text=True, **kwargs):
        calls.append(cmd)
        binary = cmd[0]
        # ffprobe ... duration -> return a fixed value
        if binary.endswith("ffprobe") or binary == "ffprobe":
            class _Result:
                returncode = 0
                stdout = "60.0\n"
                stderr = ""
            return _Result()
        # ffmpeg ... -> create the output file with a tiny dummy payload
        # The last argument is the destination path in our call shape.
        dst = Path(cmd[-1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"\x00\x00\x00\x00ftypmp42")
        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Result()

    import subprocess
    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def sample_annotation(tmp_path: Path) -> Path:
    """Write a 2-video annotation fixture at ``tmp_path/data/annotations/annotations.json``.

    The fixture mirrors the real M3 layout: ``data/annotations/annotations.json``
    with paths declared as ``data/videos/robot/<id>.mp4``.
    """
    annotations = {
        "living_room_06": {
            "video_path": "data/videos/robot/living_room_06.mp4",
            "mem_path": "data/memory_graphs/robot/living_room_06.pkl",
            "qa_list": [
                {
                    "question": "What are the drinks?",
                    "answer": "Coffee.",
                    "question_id": "living_room_06_Q01",
                    "timestamp": "00:43",
                    "type": ["Cross-Modal Reasoning"],
                    "before_clip": 0,
                }
            ],
        },
        "kitchen_22": {
            "video_path": "data/videos/robot/kitchen_22.mp4",
            "mem_path": "data/memory_graphs/robot/kitchen_22.pkl",
            "qa_list": [
                {
                    "question": "What fruit is on the counter?",
                    "answer": "Apple.",
                    "question_id": "kitchen_22_Q01",
                    "timestamp": "02:10",
                    "type": ["Person Understanding"],
                    "before_clip": 4,
                }
            ],
        },
    }
    annotations_dir = tmp_path / "data" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    out = annotations_dir / "annotations.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(annotations, f)
    return out


@pytest.fixture
def sample_video_layout(tmp_path: Path, sample_annotation: Path) -> dict[str, Path]:
    """Create the M3 directory layout under ``tmp_path``:

    ``tmp_path/data/annotations/annotations.json`` -> sample_annotation
    ``tmp_path/data/videos/robot/<id>.mp4`` -> empty mp4 files

    Mirrors the real M3-Bench layout so ``list_video_paths`` can resolve
    declared paths via ``videos_root.parent / declared``.
    """
    annotations = json.loads(sample_annotation.read_text())
    data_root = sample_annotation.parent.parent  # tmp_path/data
    videos_dir = data_root / "videos" / "robot"
    videos_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {"data_root": data_root}
    for vid in annotations.keys():
        vp = videos_dir / f"{vid}.mp4"
        vp.write_bytes(b"\x00\x00\x00\x00ftypmp42")
        paths[vid] = vp
    return paths
