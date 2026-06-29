"""Tests for M3 episodic index/search/QA MVP."""

from __future__ import annotations

import json
from pathlib import Path

from m3.index_episodic import (
    build_index,
    build_retrieval_text,
    index_record_from_episodic,
)
from m3.qa import answer_question, iter_annotation_questions, parse_args as parse_qa_args, run_qa
from m3.search import EpisodicSearcher
from providers.embedder import MockEmbedder
from providers.llm import MockLLM


def _episodic_record(video_id="living_room_06", clip_id=0, description="A mug of coffee is on the table."):
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "video_raw_node_id": f"raw-{video_id}-{clip_id}",
        "extracted_node_id": f"ext-{video_id}-{clip_id}",
        "description": description,
        "characters": [{"name": "robot", "role": "observer"}],
        "events": [{"type": "observation", "summary": "The camera sees the table."}],
        "entities": [{"name": "coffee", "type": "drink"}],
        "frame_count": 2,
        "has_audio": False,
    }


class TestIndexEpisodic:
    def test_build_retrieval_text_includes_structured_fields(self):
        text = build_retrieval_text(_episodic_record())
        assert "Description:" in text
        assert "mug of coffee" in text
        assert "Events:" in text
        assert "Entities:" in text
        assert "Characters:" in text

    def test_index_record_shape(self):
        rec = index_record_from_episodic(_episodic_record(), [0.1, 0.2])
        assert rec["video_id"] == "living_room_06"
        assert rec["clip_id"] == 0
        assert rec["embedding"] == [0.1, 0.2]
        assert rec["uri"].startswith("m3://living_room_06")
        assert rec["source"] == "episodic_memory_generator"

    def test_build_index_skips_empty_records(self):
        embedder = MockEmbedder(dimension=8)
        records = [_episodic_record(), {"video_id": "v", "clip_id": 1}]
        index = build_index(records, embedder=embedder)
        # The sparse second record still has Source text, so both are kept.
        assert len(index) == 2
        assert all(len(r["embedding"]) == 8 for r in index)


class TestSearch:
    def test_search_filters_by_video_and_before_clip(self):
        embedder = MockEmbedder(dimension=8)
        records = [
            _episodic_record("living_room_06", 0, "Coffee is on the table."),
            _episodic_record("living_room_06", 5, "A future clip shows tea."),
            _episodic_record("kitchen_22", 0, "An apple is on the counter."),
        ]
        index = build_index(records, embedder=embedder)
        searcher = EpisodicSearcher(index, embedder=embedder)
        hits = searcher.search(
            "What drink is on the table?",
            video_id="living_room_06",
            before_clip=0,
            top_k=5,
        )
        assert len(hits) == 1
        assert hits[0].video_id == "living_room_06"
        assert hits[0].clip_id == 0
        assert "Coffee" in hits[0].text

    def test_include_neighbors_adds_adjacent_clip(self):
        embedder = MockEmbedder(dimension=8)
        index = build_index([
            _episodic_record("v", 0, "The robot enters the kitchen."),
            _episodic_record("v", 1, "The robot sees a red apple."),
        ], embedder=embedder)
        searcher = EpisodicSearcher(index, embedder=embedder)
        hits = searcher.search("red apple", video_id="v", top_k=2, include_neighbors=1)
        assert {h.clip_id for h in hits} == {0, 1}


class TestQAMVP:
    def test_iter_annotation_questions_normalizes_fixture(self):
        annotations = {
            "vid": {
                "video_path": "data/videos/robot/vid.mp4",
                "qa_list": [{"question": "Q?", "answer": "A", "type": ["T"], "before_clip": 1}],
            }
        }
        qs = list(iter_annotation_questions(annotations))
        assert qs[0]["video_id"] == "vid"
        assert qs[0]["question"] == "Q?"
        assert qs[0]["answer"] == "A"
        assert qs[0]["before_clip"] == 1

    def test_answer_question_uses_llm_schema(self):
        llm = MockLLM()
        llm._fixtures = {"coffee": {"answer": "Coffee.", "reasoning": "Evidence mentions coffee.", "confidence": 0.9}}
        evidence = []
        result = answer_question(
            llm=llm,
            question="What drink is on the table? coffee",
            evidence=evidence,
            video_id="living_room_06",
        )
        assert result["answer"] == "Coffee."
        assert result["confidence"] == 0.9

    def test_run_qa_writes_jsonl(self, tmp_path: Path):
        annotations = {
            "living_room_06": {
                "video_path": "data/videos/robot/living_room_06.mp4",
                "qa_list": [{
                    "question": "What drink is on the table? coffee",
                    "answer": "Coffee.",
                    "question_id": "q1",
                    "type": ["Object Recognition"],
                    "before_clip": 0,
                }],
            }
        }
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(annotations), encoding="utf-8")

        embedder = MockEmbedder(dimension=8)
        index = build_index([_episodic_record()], embedder=embedder)
        index_path = tmp_path / "index.jsonl"
        index_path.write_text("\n".join(json.dumps(r) for r in index) + "\n", encoding="utf-8")

        out = tmp_path / "qa.jsonl"
        args = parse_qa_args([
            "--annotations", str(ann_path),
            "--index", str(index_path),
            "--output-jsonl", str(out),
            "--embedder", "mock",
            "--embedding-dimension", "8",
            "--llm", "mock",
        ])
        rc = run_qa(args)
        assert rc == 0
        rows = [json.loads(line) for line in out.read_text().splitlines() if line]
        assert len(rows) == 1
        assert rows[0]["question_id"] == "q1"
        assert rows[0]["predicted_answer"]
        assert rows[0]["evidence"]
