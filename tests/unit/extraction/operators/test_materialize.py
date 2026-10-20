"""Tests for :mod:`extraction.operators.materialize`."""

from __future__ import annotations

import base64

import pytest

from core.graph_enums import ExtractionStatus, GraphEdgeType, RawDataType
from extraction.operators.materialize import (
    FACE_DETECTION_MATERIALIZATION,
    VIDEO_FRAME_MATERIALIZATION,
    MaterializationPolicy,
    MaterializedChild,
    lightweight_summary,
    materialize_children,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_graph():
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService
    return GraphService(MemoryGraphStore())


@pytest.fixture
def ctx():
    from core.models import RequestContext
    return RequestContext(
        account_id="t-acct", user_id="u", agent_id="a",
        session_id="t-sess", trace_id="t-trace",
    )


@pytest.fixture
def source_node(memory_graph, ctx):
    from core.graph_enums import RawDataType
    from core.graph_models import create_raw_node
    node = create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",
        account_id=ctx.account_id,
        metadata={"video_id": "v1", "clip_id": 0, "source_path": "/tmp/v.mp4"},
        node_id="src-node-1",
    )
    memory_graph.create_raw_node.__self__ if False else None  # noop; just for clarity
    return node


# Use a tiny static policy so we don't need to mock opencv. The
# materialization helper itself does not care what operator declared it.

def _static_child_id(item, source_raw_id, source_metadata):
    import hashlib
    seed = f"static/{source_raw_id}/{item['i']}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def _static_payload(item, source_raw_id, source_metadata):
    md = {"i": item["i"], "label": item.get("label", "")}
    if source_metadata is not None:
        md["source_video_id"] = source_metadata.get("video_id")
    return md


SIMPLE_POLICY = MaterializationPolicy(
    child_raw_type=RawDataType.IMAGE,
    collection_field="items",
    child_id_fn=_static_child_id,
    child_payload_fn=_static_payload,
    edge_type=GraphEdgeType.REFERENCES,
    edge_properties_fn=lambda item, src, meta: {"relation": "has_item", "i": item["i"]},
)


@pytest.fixture
def source_raw_node(memory_graph, ctx):
    """Create and persist a source VIDEO RawNode in the graph."""
    from core.graph_enums import RawDataType
    memory_graph.create_raw_node(
        raw_type=RawDataType.VIDEO,
        content="",
        ctx=ctx,
        metadata={"video_id": "v1", "clip_id": 0, "source_path": "/tmp/v.mp4"},
        node_id="src-node-1",
    )
    return memory_graph.get_raw_node("src-node-1", ctx)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class TestMaterializationPolicy:
    def test_video_frame_policy_targets_image(self):
        assert VIDEO_FRAME_MATERIALIZATION.child_raw_type == RawDataType.IMAGE
        assert VIDEO_FRAME_MATERIALIZATION.collection_field == "frames"
        assert VIDEO_FRAME_MATERIALIZATION.edge_type == GraphEdgeType.REFERENCES
        assert VIDEO_FRAME_MATERIALIZATION.bytes_field == "image_bytes_b64"
        assert VIDEO_FRAME_MATERIALIZATION.bytes_metadata_key == "source_path"

    def test_video_frame_child_id_uses_source_raw_id(self):
        a = VIDEO_FRAME_MATERIALIZATION.child_id_fn(
            {"frame_idx": 5}, "src-A", None,
        )
        b = VIDEO_FRAME_MATERIALIZATION.child_id_fn(
            {"frame_idx": 5}, "src-B", None,
        )
        assert a != b, "different source_raw_id must yield different ids"
        assert len(a) == 32

    def test_video_frame_child_id_is_deterministic(self):
        a = VIDEO_FRAME_MATERIALIZATION.child_id_fn(
            {"frame_idx": 7}, "src-X", {"video_id": "v1"},
        )
        b = VIDEO_FRAME_MATERIALIZATION.child_id_fn(
            {"frame_idx": 7}, "src-X", {"video_id": "v1"},
        )
        assert a == b

    def test_video_frame_child_id_varies_with_frame_idx(self):
        a = VIDEO_FRAME_MATERIALIZATION.child_id_fn({"frame_idx": 1}, "src", None)
        b = VIDEO_FRAME_MATERIALIZATION.child_id_fn({"frame_idx": 2}, "src", None)
        assert a != b


# ---------------------------------------------------------------------------
# materialize_children
# ---------------------------------------------------------------------------

class TestMaterializeChildren:
    def test_creates_one_node_per_item(self, memory_graph, ctx, source_raw_node):
        content = {"items": [{"i": 0}, {"i": 1}, {"i": 2}]}
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=SIMPLE_POLICY,
        )
        assert len(children) == 3
        for i, c in enumerate(children):
            assert isinstance(c, MaterializedChild)
            assert c.child_node_id == _static_child_id(
                {"i": i}, source_raw_node.node_id, source_raw_node.metadata,
            )
            assert c.resumed is False
            assert c.edge_id is not None
            node = memory_graph.get_raw_node(c.child_node_id, ctx)
            assert node is not None
            assert node.raw_type == RawDataType.IMAGE
            assert node.metadata["i"] == i

    def test_creates_references_edge(self, memory_graph, ctx, source_raw_node):
        content = {"items": [{"i": 0}]}
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=SIMPLE_POLICY,
        )
        neighbors = memory_graph.get_neighbors(
            source_raw_node.node_id, ctx,
            edge_types=[GraphEdgeType.REFERENCES],
        )
        assert len(neighbors) == 1
        # neighbors is a list of (node, edge, direction)
        target_node, edge, direction = neighbors[0]
        assert target_node.node_id == children[0].child_node_id
        assert edge.edge_type == GraphEdgeType.REFERENCES
        assert direction == "outgoing"

    def test_empty_collection_returns_empty_list(self, memory_graph, ctx, source_raw_node):
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content={"items": []},
            policy=SIMPLE_POLICY,
        )
        assert children == []
        assert report.attempted == 0

    def test_missing_collection_field_is_noop(self, memory_graph, ctx, source_raw_node):
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content={"other_key": "x"},
            policy=SIMPLE_POLICY,
        )
        assert children == []
        assert report.attempted == 0

    def test_skips_non_dict_items_without_aborting(self, memory_graph, ctx, source_raw_node):
        content = {"items": [{"i": 0}, "not-a-dict", {"i": 1}]}
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=SIMPLE_POLICY,
        )
        assert len(children) == 2  # only the two dicts
        assert report.attempted == 3
        assert report.failed == 1
        assert report.errors[0]["reason"] == "non_dict_item"
        assert children[0].payload["i"] == 0
        assert children[1].payload["i"] == 1

    def test_idempotency_skip_existing(self, memory_graph, ctx, source_raw_node):
        content = {"items": [{"i": 0}, {"i": 1}]}
        first, first_report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=SIMPLE_POLICY,
        )
        assert len(first) == 2
        assert first_report.created == 2
        # Second run should resume both
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=SIMPLE_POLICY,
        )
        assert len(children) == 2
        assert report.resumed == 2
        assert all(c.resumed for c in children)
        assert all(c.edge_id is None for c in children)

    def test_source_metadata_propagates_to_child(self, memory_graph, ctx, source_raw_node):
        content = {"items": [{"i": 0}]}
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=SIMPLE_POLICY,
        )
        assert children[0].payload["source_video_id"] == "v1"

    def test_bytes_field_decoded_and_written(self, memory_graph, ctx, source_raw_node, tmp_path):
        def writer(item, raw_bytes):
            out = tmp_path / f"{item['i']}.bin"
            out.write_bytes(raw_bytes)
            return str(out)

        def payload_with_disk(item, source_raw_id, source_metadata):
            md = {"i": item["i"]}
            if "on_disk" in item:
                md["on_disk"] = item["on_disk"]
            return md

        policy = MaterializationPolicy(
            child_raw_type=RawDataType.IMAGE,
            collection_field="items",
            child_id_fn=_static_child_id,
            child_payload_fn=payload_with_disk,
            bytes_field="payload_b64",
            bytes_metadata_key="on_disk",
        )
        payload = b"\x00\x01\x02hello"
        content = {"items": [
            {"i": 0, "payload_b64": base64.b64encode(payload).decode()},
        ]}
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=policy,
            bytes_writer=writer,
        )
        assert len(children) == 1
        node = memory_graph.get_raw_node(children[0].child_node_id, ctx)
        # The b64 field is NOT persisted; the writer's path is.
        assert "payload_b64" not in node.metadata
        assert node.metadata["on_disk"].endswith("/0.bin")
        assert (tmp_path / "0.bin").read_bytes() == payload

    def test_no_bytes_writer_skips_bytes_persistence(
        self, memory_graph, ctx, source_raw_node,
    ):
        """bytes_field set but no writer passed → b64 stays in metadata as-is."""
        policy = MaterializationPolicy(
            child_raw_type=RawDataType.IMAGE,
            collection_field="items",
            child_id_fn=_static_child_id,
            child_payload_fn=lambda item, src, meta: {
                "i": item["i"], "payload_b64": item.get("payload_b64", ""),
            },
            bytes_field="payload_b64",
            bytes_metadata_key="on_disk",
        )
        content = {"items": [{"i": 0, "payload_b64": "abc"}]}
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=policy,
        )
        node = memory_graph.get_raw_node(children[0].child_node_id, ctx)
        # No bytes_writer → b64 field passes through (operator's choice).
        assert node.metadata["payload_b64"] == "abc"


# ---------------------------------------------------------------------------
# lightweight_summary
# ---------------------------------------------------------------------------

class TestLightweightSummary:
    def test_strips_collection_field_and_adds_ids(self):
        original = {
            "video_path": "/tmp/v.mp4",
            "fps_actual": 30.0,
            "frames": [{"frame_idx": 0, "image_bytes_b64": "AAA"}, {"frame_idx": 1}],
            "frame_count": 2,
        }
        summary = lightweight_summary(
            original, VIDEO_FRAME_MATERIALIZATION, ["c1", "c2"],
        )
        assert "frames" not in summary
        assert summary["child_node_ids"] == ["c1", "c2"]
        assert summary["child_count"] == 2
        assert summary["video_path"] == "/tmp/v.mp4"
        assert summary["fps_actual"] == 30.0
        assert summary["frame_count"] == 2

    def test_no_op_when_collection_already_absent(self):
        original = {"video_path": "/x", "frame_count": 5}
        summary = lightweight_summary(original, VIDEO_FRAME_MATERIALIZATION, [])
        assert summary == {
            "video_path": "/x",
            "frame_count": 5,
            "child_node_ids": [],
            "child_count": 0,
        }


# ---------------------------------------------------------------------------
# Extracted-kind materialization (face_detection, etc.)
# ---------------------------------------------------------------------------

def _face_payload(item, source_raw_id, source_metadata):
    """Minimal payload fn for face-detection-style items."""
    md = {
        "bbox": list(item["bbox"]),
        "confidence": float(item["confidence"]),
    }
    face_meta = item.get("metadata", {}) or {}
    if "embedding" in face_meta:
        md["embedding"] = face_meta["embedding"]
    md["source_frame_node_id"] = source_raw_id
    return md


EXTRACTED_POLICY = MaterializationPolicy(
    child_node_kind="extracted",
    child_extractor_type="face_detection",
    child_extraction_schema="face_detection_v1",
    collection_field="faces",
    child_id_fn=lambda item, src, meta: (
        __import__("hashlib").sha1(
            f"face/{src}/{item['face_id']}".encode(),
        ).hexdigest()[:32]
    ),
    child_payload_fn=_face_payload,
    # edge_type left unset -> defaults to EXTRACTED_FROM for kind="extracted"
)


class TestExtractedKindMaterialization:
    """When ``child_node_kind='extracted'`` the materializer creates
    ``ExtractedNode`` children (carrying the operator's structured output
    like bbox + embedding) rather than ``RawNode`` children. This keeps
    the boundary clean: raw data flows through ``RawNode``, operator
    outputs flow through ``ExtractedNode``.
    """

    def test_face_detection_policy_uses_extracted_kind(self):
        assert FACE_DETECTION_MATERIALIZATION.child_node_kind == "extracted"
        assert FACE_DETECTION_MATERIALIZATION.collection_field == "faces"
        assert FACE_DETECTION_MATERIALIZATION.child_extractor_type == "face_detection"
        assert FACE_DETECTION_MATERIALIZATION.child_extraction_schema == "face_detection_v1"
        assert FACE_DETECTION_MATERIALIZATION.bytes_field is None

    def test_default_edge_type_is_extracted_from(self):
        # Not explicitly set on the policy — defaults by kind.
        assert FACE_DETECTION_MATERIALIZATION.edge_type is None
        assert FACE_DETECTION_MATERIALIZATION.effective_edge_type() == GraphEdgeType.EXTRACTED_FROM

    def test_face_child_id_uses_face_id_when_present(self):
        item = {"bbox": [0.1, 0.2, 0.3, 0.4], "metadata": {"face_id": 7}}
        cid = FACE_DETECTION_MATERIALIZATION.child_id_fn(item, "src-X", None)
        # Deterministic + unique per (source, face_id)
        again = FACE_DETECTION_MATERIALIZATION.child_id_fn(item, "src-X", None)
        assert cid == again
        other = FACE_DETECTION_MATERIALIZATION.child_id_fn(item, "src-Y", None)
        assert cid != other

    def test_creates_extracted_node_not_raw_node(
        self, memory_graph, ctx, source_raw_node,
    ):
        content = {"faces": [
            {
                "face_id": 0,
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "confidence": 0.9,
                "metadata": {"detector": "insightface", "face_id": 0,
                             "embedding": [0.1, 0.2, 0.3]},
            },
        ]}
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=EXTRACTED_POLICY,
        )
        assert len(children) == 1
        c = children[0]
        assert c.child_node_kind == "extracted"
        assert c.resumed is False
        # Critical: it's an ExtractedNode, NOT a RawNode.
        assert memory_graph.get_raw_node(c.child_node_id, ctx) is None
        extracted = memory_graph.get_extracted_node(c.child_node_id, ctx)
        assert extracted is not None
        assert extracted.extraction_status == ExtractionStatus.COMPLETED
        assert extracted.confidence == 1.0
        assert extracted.extractor_type == "face_detection"
        assert extracted.extraction_schema == "face_detection_v1"
        assert extracted.source_raw_id == source_raw_node.node_id
        # The payload landed in extracted_content.
        assert extracted.extracted_content["bbox"] == [0.1, 0.2, 0.3, 0.4]
        assert extracted.extracted_content["confidence"] == 0.9
        assert extracted.extracted_content["embedding"] == [0.1, 0.2, 0.3]
        assert extracted.extracted_content["source_frame_node_id"] == source_raw_node.node_id

    def test_creates_extracted_from_edge(
        self, memory_graph, ctx, source_raw_node,
    ):
        """create_extracted_node auto-wires EXTRACTED_FROM from the
        child ExtractedNode back to the source. The materialization
        path does NOT add a second edge in the opposite direction."""
        content = {"faces": [
            {"face_id": 0, "bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9,
             "metadata": {"face_id": 0}},
        ]}
        children, report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=EXTRACTED_POLICY,
        )
        # The auto-edge points FROM the child ExtractedNode TO the
        # source, so it's incoming to source_raw_node.
        neighbors = memory_graph.get_neighbors(
            source_raw_node.node_id, ctx,
            edge_types=[GraphEdgeType.EXTRACTED_FROM],
            direction="incoming",
        )
        assert len(neighbors) == 1
        target, edge, direction = neighbors[0]
        assert target.node_id == children[0].child_node_id
        assert edge.edge_type == GraphEdgeType.EXTRACTED_FROM
        assert direction == "incoming"
        # No duplicate outgoing edge from source to child.
        outgoing = memory_graph.get_neighbors(
            source_raw_node.node_id, ctx,
            edge_types=[GraphEdgeType.EXTRACTED_FROM],
            direction="outgoing",
        )
        assert outgoing == []  # parent ExtractedNode creation didn't add one

    def test_idempotent_resume(self, memory_graph, ctx, source_raw_node):
        content = {"faces": [
            {"face_id": 0, "bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9,
             "metadata": {"face_id": 0}},
        ]}
        first, first_report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=EXTRACTED_POLICY,
        )
        second, second_report = materialize_children(
            memory_graph, ctx,
            source_raw_id=source_raw_node.node_id,
            source_metadata=source_raw_node.metadata,
            extracted_content=content,
            policy=EXTRACTED_POLICY,
        )
        assert first[0].child_node_id == second[0].child_node_id
        assert second[0].resumed is True
        # No duplicate child ExtractedNodes — the graph stays clean.
        face_records = memory_graph.list_extracted_nodes(
            extractor_type="face_detection", ctx=ctx,
        )
        assert len(face_records) == 1


class TestPolicyValidation:
    """The policy validates its own shape via __post_init__ so callers
    get a clear error instead of a downstream AttributeError later."""

    def test_raw_kind_requires_child_raw_type(self):
        with pytest.raises(ValueError, match="child_raw_type"):
            MaterializationPolicy(
                child_node_kind="raw",
                collection_field="items",
                child_id_fn=lambda i, s, m: "x",
                child_payload_fn=lambda i, s, m: {},
            )

    def test_extracted_kind_requires_extractor_type(self):
        with pytest.raises(ValueError, match="child_extractor_type"):
            MaterializationPolicy(
                child_node_kind="extracted",
                collection_field="items",
                child_id_fn=lambda i, s, m: "x",
                child_payload_fn=lambda i, s, m: {},
                child_extraction_schema="v1",
            )

    def test_extracted_kind_requires_extraction_schema(self):
        with pytest.raises(ValueError, match="child_extraction_schema"):
            MaterializationPolicy(
                child_node_kind="extracted",
                collection_field="items",
                child_id_fn=lambda i, s, m: "x",
                child_payload_fn=lambda i, s, m: {},
                child_extractor_type="op",
            )

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="child_node_kind"):
            MaterializationPolicy(
                child_node_kind="magic",
                collection_field="items",
                child_id_fn=lambda i, s, m: "x",
                child_payload_fn=lambda i, s, m: {},
                child_raw_type=RawDataType.IMAGE,
            )
