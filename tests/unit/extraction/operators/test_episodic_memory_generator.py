"""Tests for :class:`EpisodicMemoryGenerator`."""

from __future__ import annotations

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.builtin.episodic_memory_generator import (
    EPISODIC_MEMORY_SCHEMA,
    EpisodicMemoryGenerator,
    _evenly_spaced_indices,
    _image_part_from_node,
)
from providers.llm import MockLLM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acct", user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )


@pytest.fixture
def llm():
    return MockLLM()


@pytest.fixture
def video_node():
    return create_raw_node(
        raw_type=RawDataType.VIDEO,
        content=b"",
        account_id="acct",
        metadata={
            "video_id": "vid_a", "clip_id": 0,
            "duration_sec": 30.0, "start_sec": 0.0,
            "source_path": "/tmp/vid_a/0.mp4",
        },
    )


@pytest.fixture
def frame_nodes():
    """5 fake JPEG frames (one byte payload each)."""
    return [
        create_raw_node(
            raw_type=RawDataType.IMAGE, content=b"\xff\xd8\xff\xe0j%d" % i,
            account_id="acct",
            metadata={"frame_idx": i, "ts_sec": float(i)},
            node_id=f"frame-{i:03d}",
        )
        for i in range(5)
    ]


@pytest.fixture
def audio_node():
    # WAV magic + a few bytes; format detection will pick "wav".
    return create_raw_node(
        raw_type=RawDataType.AUDIO,
        content=b"RIFF" + b"\x00" * 36,
        account_id="acct",
        metadata={"duration_sec": 30.0},
        node_id="audio-000",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_zero_max_frames_rejected(self, llm):
        with pytest.raises(ValueError, match="max_frames"):
            EpisodicMemoryGenerator(llm=llm, max_frames=0)

    def test_negative_max_frames_rejected(self, llm):
        with pytest.raises(ValueError, match="max_frames"):
            EpisodicMemoryGenerator(llm=llm, max_frames=-1)

    def test_invalid_image_detail_rejected(self, llm):
        with pytest.raises(ValueError, match="image_detail"):
            EpisodicMemoryGenerator(llm=llm, image_detail="ultra")

    def test_supported_types(self, llm):
        op = EpisodicMemoryGenerator(llm=llm)
        assert RawDataType.VIDEO in op.supported_types

    def test_extractor_type(self, llm):
        op = EpisodicMemoryGenerator(llm=llm)
        assert op.extractor_type == "episodic_memory_generator"

    def test_schema_name(self, llm):
        op = EpisodicMemoryGenerator(llm=llm)
        assert op.schema_name == "episodic_memory_v1"

    def test_info_has_config_schema(self, llm):
        info = EpisodicMemoryGenerator(llm=llm).info
        assert "max_frames" in info.config_schema["properties"]
        assert "include_audio" in info.config_schema["properties"]

    def test_info_no_materialization(self, llm):
        """episodic_memory_generator outputs a single ExtractedNode per
        clip — no graph children, so materialization stays None."""
        info = EpisodicMemoryGenerator(llm=llm).info
        assert info.materialization is None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_description_not_provider_required(self):
        """Provider-side schema is permissive because real multimodal
        models may drift field names (summary/text/memory instead of
        description). The operator normalizes after the call."""
        assert "description" not in EPISODIC_MEMORY_SCHEMA.get("required", [])

    def test_top_level_fields(self):
        props = EPISODIC_MEMORY_SCHEMA["properties"]
        for key in ("description", "characters", "events", "entities"):
            assert key in props


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------

class TestExtract:
    def test_calls_llm_with_multimodal_messages(
        self, llm, ctx, video_node, frame_nodes,
    ):
        op = EpisodicMemoryGenerator(llm=llm, max_frames=8)
        op.extract(video_node, ctx, frame_nodes=frame_nodes)
        # MockLLM records _last_multimodal_messages.
        msgs = llm._last_multimodal_messages
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        # User content is a list of parts (text + image_url for each frame).
        user_content = msgs[1]["content"]
        assert isinstance(user_content, list)
        # 1 text part + N image_url parts.
        image_parts = [p for p in user_content if p["type"] == "image_url"]
        text_parts = [p for p in user_content if p["type"] == "text"]
        assert len(text_parts) == 1
        assert len(image_parts) == len(frame_nodes)

    def test_subsamples_frames_to_max(
        self, llm, ctx, video_node, frame_nodes,
    ):
        # 5 frames, max=3 -> 3 image_url parts.
        op = EpisodicMemoryGenerator(llm=llm, max_frames=3)
        op.extract(video_node, ctx, frame_nodes=frame_nodes)
        image_parts = [
            p for p in llm._last_multimodal_messages[1]["content"]
            if p["type"] == "image_url"
        ]
        assert len(image_parts) == 3

    def test_includes_audio_when_provided(
        self, llm, ctx, video_node, frame_nodes, audio_node,
    ):
        op = EpisodicMemoryGenerator(
            llm=llm, max_frames=8, include_audio=True,
        )
        op.extract(
            video_node, ctx,
            frame_nodes=frame_nodes, audio_node=audio_node,
        )
        user_content = llm._last_multimodal_messages[1]["content"]
        audio_parts = [p for p in user_content if p["type"] == "input_audio"]
        assert len(audio_parts) == 1
        # WAV magic -> format=wav.
        assert audio_parts[0]["input_audio"]["format"] == "wav"

    def test_audio_skipped_when_disabled(
        self, llm, ctx, video_node, frame_nodes, audio_node,
    ):
        op = EpisodicMemoryGenerator(
            llm=llm, max_frames=8, include_audio=False,
        )
        op.extract(
            video_node, ctx,
            frame_nodes=frame_nodes, audio_node=audio_node,
        )
        user_content = llm._last_multimodal_messages[1]["content"]
        audio_parts = [p for p in user_content if p["type"] == "input_audio"]
        assert len(audio_parts) == 0

    def test_returns_extraction_result_with_provenance(
        self, llm, ctx, video_node, frame_nodes,
    ):
        # Pin the canned LLM response so the assertion is independent
        # of MockLLM's deterministic-stub text pick.
        llm._mock_multimodal_response = {
            "description": "The robot picks up a red cup.",
            "characters": [{"name": "robot_arm", "role": "agent"}],
            "events": [{"type": "manipulation", "summary": "Pick and place."}],
            "entities": [{"name": "red cup", "type": "object"}],
        }
        op = EpisodicMemoryGenerator(llm=llm, max_frames=8)
        result = op.extract(video_node, ctx, frame_nodes=frame_nodes)
        assert result.error is None
        content = result.extracted_content
        # LLM response fields land in extracted_content.
        assert content["description"] == "The robot picks up a red cup."
        assert content["characters"] == [{"name": "robot_arm", "role": "agent"}]
        assert content["events"] == [{"type": "manipulation", "summary": "Pick and place."}]
        assert content["entities"] == [{"name": "red cup", "type": "object"}]
        # Provenance + run metadata merged in.
        assert content["frame_count"] == len(frame_nodes)
        assert content["has_audio"] is False
        assert content["source_video_id"] == "vid_a"
        assert content["source_clip_id"] == 0
        assert content["source_duration_sec"] == 30.0

    def test_returns_error_when_no_frames(self, llm, ctx, video_node):
        op = EpisodicMemoryGenerator(llm=llm)
        result = op.extract(video_node, ctx, frame_nodes=[])
        assert result.error is not None
        assert "frame_nodes" in result.error

    def test_returns_error_on_llm_failure(self, ctx, video_node, frame_nodes):
        class FailingLLM:
            def complete_multimodal(self, messages, schema):
                raise RuntimeError("api timeout")
        op = EpisodicMemoryGenerator(llm=FailingLLM(), max_frames=8)
        result = op.extract(video_node, ctx, frame_nodes=frame_nodes)
        assert result.error == "api timeout"
        assert result.confidence == 0.0

    def test_normalizes_summary_to_description(self, llm, ctx, video_node, frame_nodes):
        llm._mock_multimodal_response = {
            "summary": "A concise clip summary.",
            "characters": "not-a-list",
        }
        op = EpisodicMemoryGenerator(llm=llm, max_frames=8)
        result = op.extract(video_node, ctx, frame_nodes=frame_nodes)
        assert result.error is None
        assert result.extracted_content["description"] == "A concise clip summary."
        assert result.extracted_content["characters"] == []
        assert result.extracted_content["events"] == []
        assert result.extracted_content["entities"] == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestEvenlySpacedIndices:
    def test_k_equals_n_returns_full_range(self):
        assert _evenly_spaced_indices(5, 5) == [0, 1, 2, 3, 4]

    def test_k_greater_than_n_clamps(self):
        assert _evenly_spaced_indices(3, 10) == [0, 1, 2]

    def test_k_zero_returns_empty(self):
        assert _evenly_spaced_indices(5, 0) == []

    def test_typical_subsample(self):
        # 10 items, pick 3 -> roughly [0, 3, 6] / [0, 3, 7] depending on step
        indices = _evenly_spaced_indices(10, 3)
        assert len(indices) == 3
        assert indices[0] == 0
        # Should span the range roughly evenly.
        assert indices[-1] >= 6


class TestImagePartFromNode:
    def test_image_node_with_bytes_produces_data_uri(self, frame_nodes):
        part = _image_part_from_node(frame_nodes[0], detail="low")
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert part["image_url"]["detail"] == "low"

    def test_image_node_with_empty_content_returns_none(self):
        from core.graph_models import create_raw_node
        node = create_raw_node(
            raw_type=RawDataType.IMAGE, content=b"", account_id="acct",
        )
        assert _image_part_from_node(node, detail="auto") is None
