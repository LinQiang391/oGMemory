"""Benchmark-scope and resume regressions from the September 17 smoke run."""
import json

import pytest

from scripts import run_m3_bench_subset as driver


def question(**overrides):
    return {
        "question_id": "v_Q01", "question": "Where is the person?",
        "answer": "Bedroom.", "timestamp": "01:18", "before_clip": 1,
        **overrides,
    }


@pytest.mark.parametrize("timestamp", ["02:01", "37:14", "01:00:00", 121])
def test_trim_rejects_future_questions(timestamp):
    annotations = {"v": {"qa_list": [question(timestamp=timestamp)]}}
    assert len(driver.validate_annotation_duration(annotations, ["v"], 120)) == 1


@pytest.mark.parametrize("timestamp", ["00:05", "01:58", "02:00", None, ""])
def test_smoke_questions_within_window(timestamp):
    annotations = {"v": {"qa_list": [question(timestamp=timestamp)]}}
    assert driver.validate_annotation_duration(annotations, ["v"], 120) == []


def test_full_video_and_unselected_videos_are_not_rejected():
    annotations = {
        "v": {"qa_list": [question(timestamp="37:14")]},
        "other": {"qa_list": [question()]},
    }
    assert driver.validate_annotation_duration(annotations, ["v"], 0) == []
    assert driver.validate_annotation_duration(annotations, ["other"], 120) == []


def write_result(tmp_path, **overrides):
    record = {
        "video_id": "v", "question_id": "v_Q01",
        "question": "Where is the person?", "expected_answer": "Bedroom.",
        "timestamp": "01:18", "before_clip": 1,
        "predicted_answer": "In the bedroom.", "error": "", **overrides,
    }
    path = tmp_path / "v" / "m3_qa_results.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record) + "\n")
    return path


def test_matching_annotation_can_resume(tmp_path):
    write_result(tmp_path)
    assert driver.already_done(tmp_path, "v", {"v_Q01"}, [question()])


@pytest.mark.parametrize("overrides", [
    {"question": "Where is the bag?"},
    {"expected_answer": "Wardrobe."},
    {"timestamp": "37:14"},
    {"before_clip": 73},
    {"video_id": "other"},
    {"predicted_answer": ""},
    {"predicted_answer": None},
    {"error": "timeout"},
])
def test_reused_ids_are_not_proof_of_matching_results(tmp_path, overrides):
    write_result(tmp_path, **overrides)
    assert not driver.already_done(tmp_path, "v", {"v_Q01"}, [question()])


def test_extra_questions_and_corrupt_results_cannot_resume(tmp_path):
    path = write_result(tmp_path)
    original = path.read_text()
    for extra in ['{"video_id":"v","question_id":"v_Q02"}\n', "{broken\n"]:
        path.write_text(original + extra)
        assert not driver.already_done(tmp_path, "v", {"v_Q01"}, [question()])


def test_preflight_stops_before_pipeline_or_judge(tmp_path, monkeypatch, caplog):
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({"v": {"qa_list": [question(timestamp="37:14")]}}))
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    monkeypatch.setattr(driver, "run_one_video", lambda **kw: pytest.fail("pipeline ran"))
    monkeypatch.setattr(driver, "run_judge", lambda **kw: pytest.fail("judge ran"))
    assert driver.main([
        "--annotations", str(annotation), "--config", str(config),
        "--videos", "v", "--seconds", "120",
        "--work-dir-root", str(tmp_path / "run"), "--judge",
    ]) == 1
    assert "--allow-out-of-window-questions" in caplog.text


def test_override_is_explicit():
    assert not driver.parse_args([]).allow_out_of_window_questions
    assert driver.parse_args(["--allow-out-of-window-questions"]).allow_out_of_window_questions
