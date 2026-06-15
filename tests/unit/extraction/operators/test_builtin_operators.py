"""Unit tests for built-in extraction operators."""

from __future__ import annotations

import base64

import pytest

from core.graph_enums import RawDataType
from core.graph_models import create_raw_node
from core.models import RequestContext
from extraction.operators.backends import MockBackend
from extraction.operators.builtin import (
    DocumentExtractor,
    FaceDetectionExtractor,
    ImageCaptionExtractor,
    ObjectDetectionExtractor,
    OCRExtractor,
    PoseDetectionExtractor,
)
from extraction.operators.builtin.face_detection import FaceDetectionExtractor as FDE


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acct", user_id="u", agent_id="a",
        session_id="s", trace_id="t",
    )


@pytest.fixture
def image_raw_node():
    return create_raw_node(
        raw_type=RawDataType.IMAGE,
        content=b"fake-image-bytes",
        account_id="acct",
    )


@pytest.fixture
def document_raw_node():
    return create_raw_node(
        raw_type=RawDataType.DOCUMENT,
        content=(
            "# Introduction\n"
            "This is the introduction section with enough text to be retained.\n"
            "\n"
            "Methods:\n"
            "We did some things in this section. The body must be at least 20 chars.\n"
            "\n"
            "RESULTS\n"
            "Here are the results, summarised for the reader's convenience. "
            "Adding more text so the section passes the minimum length filter.\n"
        ),
        account_id="acct",
    )


# ---------------------------------------------------------------------------
# FaceDetectionExtractor
# ---------------------------------------------------------------------------

class TestFaceDetectionExtractor:
    def test_basic_extraction(self, image_raw_node, ctx):
        backend = MockBackend(face_count=2)
        op = FaceDetectionExtractor(backend=backend)
        result = op.extract(image_raw_node, ctx)
        assert result.error is None
        assert result.extracted_content["face_count"] == 2
        assert len(result.extracted_content["faces"]) == 2
        assert 0.0 < result.confidence <= 1.0

    def test_min_confidence_filter(self, image_raw_node, ctx):
        backend = MockBackend(face_count=2, default_confidence=0.5)
        op = FaceDetectionExtractor(backend=backend, min_confidence=0.9)
        result = op.extract(image_raw_node, ctx)
        assert result.extracted_content["face_count"] == 0
        assert result.confidence == 0.0

    def test_max_faces(self, image_raw_node, ctx):
        backend = MockBackend(face_count=10)
        op = FaceDetectionExtractor(backend=backend, max_faces=3)
        result = op.extract(image_raw_node, ctx)
        assert result.extracted_content["face_count"] == 3

    def test_backend_error(self, image_raw_node, ctx):
        backend = MockBackend(fail_on_image_prefix="fake-image")
        op = FaceDetectionExtractor(backend=backend)
        result = op.extract(image_raw_node, ctx)
        assert result.error is not None
        assert "Forced failure" in result.error

    def test_empty_content(self, ctx):
        node = create_raw_node(
            raw_type=RawDataType.IMAGE, content=b"", account_id="acct",
        )
        op = FaceDetectionExtractor(backend=MockBackend())
        result = op.extract(node, ctx)
        assert result.extracted_content["face_count"] == 0
        assert result.confidence == 0.0

    def test_info_has_config_schema(self):
        op = FaceDetectionExtractor()
        info = op.info
        assert info.config_schema is not None
        assert "min_confidence" in info.config_schema["properties"]

    def test_supported_types(self):
        op = FaceDetectionExtractor()
        assert RawDataType.IMAGE in op.supported_types


# ---------------------------------------------------------------------------
# ObjectDetectionExtractor
# ---------------------------------------------------------------------------

class TestObjectDetectionExtractor:
    def test_basic(self, image_raw_node, ctx):
        op = ObjectDetectionExtractor(backend=MockBackend())
        result = op.extract(image_raw_node, ctx)
        assert result.error is None
        assert "object_count" in result.extracted_content
        assert "labels" in result.extracted_content
        assert "objects" in result.extracted_content
        # Default MockBackend has 4 labels; default max_objects=50 covers all
        assert result.extracted_content["object_count"] == 4

    def test_allowed_labels_filter(self, image_raw_node, ctx):
        backend = MockBackend(object_labels=("cat", "dog", "car"))
        op = ObjectDetectionExtractor(backend=backend, allowed_labels=["cat"])
        result = op.extract(image_raw_node, ctx)
        for obj in result.extracted_content["objects"]:
            assert obj["label"] == "cat"

    def test_max_objects(self, image_raw_node, ctx):
        backend = MockBackend(object_labels=("a", "b", "c", "d", "e"))
        op = ObjectDetectionExtractor(backend=backend, max_objects=2)
        result = op.extract(image_raw_node, ctx)
        assert result.extracted_content["object_count"] <= 2

    def test_label_summary(self, image_raw_node, ctx):
        backend = MockBackend(object_labels=("cat", "cat", "dog"))
        op = ObjectDetectionExtractor(backend=backend, max_objects=10)
        result = op.extract(image_raw_node, ctx)
        # Summary may include only labels that actually appear in the
        # deterministic mock output
        for label, count in result.extracted_content["labels"].items():
            assert isinstance(count, int) and count > 0


# ---------------------------------------------------------------------------
# PoseDetectionExtractor
# ---------------------------------------------------------------------------

class TestPoseDetectionExtractor:
    def test_basic(self, image_raw_node, ctx):
        op = PoseDetectionExtractor(backend=MockBackend(person_count=2))
        result = op.extract(image_raw_node, ctx)
        assert result.error is None
        assert result.extracted_content["person_count"] == 2
        for p in result.extracted_content["persons"]:
            assert "keypoints" in p
            assert len(p["keypoints"]) >= 10

    def test_min_confidence_filter(self, image_raw_node, ctx):
        op = PoseDetectionExtractor(
            backend=MockBackend(default_confidence=0.5),
            min_confidence=0.9,
        )
        result = op.extract(image_raw_node, ctx)
        assert result.extracted_content["person_count"] == 0


# ---------------------------------------------------------------------------
# ImageCaptionExtractor
# ---------------------------------------------------------------------------

class TestImageCaptionExtractor:
    def test_stub_backend(self, image_raw_node, ctx):
        op = ImageCaptionExtractor()
        result = op.extract(image_raw_node, ctx)
        assert result.error is None
        assert "caption" in result.extracted_content
        assert "16 bytes" in result.extracted_content["caption"]

    def test_custom_backend(self, image_raw_node, ctx):
        op = ImageCaptionExtractor(backend=lambda b: f"a {len(b)}-byte image")
        result = op.extract(image_raw_node, ctx)
        assert result.extracted_content["caption"] == "a 16-byte image"
        assert result.confidence == 0.9  # real backend

    def test_max_length_truncates(self, ctx):
        op = ImageCaptionExtractor(
            backend=lambda b: "x" * 1000,
            max_caption_length=50,
        )
        node = create_raw_node(
            raw_type=RawDataType.IMAGE, content=b"abc", account_id="acct",
        )
        result = op.extract(node, ctx)
        assert result.extracted_content["caption"].endswith("...")
        assert len(result.extracted_content["caption"]) <= 53  # 50 + "..."

    def test_backend_exception_captured(self, image_raw_node, ctx):
        def bad_backend(b):
            raise RuntimeError("upstream down")
        op = ImageCaptionExtractor(backend=bad_backend)
        result = op.extract(image_raw_node, ctx)
        assert result.error is not None
        assert "upstream down" in result.error

    def test_base64_content_format(self, ctx):
        node = create_raw_node(
            raw_type=RawDataType.IMAGE,
            content=base64.b64encode(b"hello").decode("ascii"),
            account_id="acct",
        )
        op = ImageCaptionExtractor(
            backend=lambda b: f"len={len(b)}",
            content_format="base64",
        )
        result = op.extract(node, ctx)
        assert result.extracted_content["caption"] == "len=5"


# ---------------------------------------------------------------------------
# OCRExtractor
# ---------------------------------------------------------------------------

class TestOCRExtractor:
    def test_no_text_detections(self, image_raw_node, ctx):
        op = OCRExtractor(backend=MockBackend())
        result = op.extract(image_raw_node, ctx)
        # Mock backend's object detection doesn't set metadata['text'],
        # so OCR sees no recognisable text
        assert result.extracted_content["text"] == ""
        assert result.extracted_content["line_count"] == 0

    def test_with_text_metadata(self, ctx):
        from extraction.operators.backends.base import Detection

        class _TextBackend(MockBackend):
            def detect_objects(self, image, **kwargs):
                # Pretend each detection is a word/line
                return [
                    Detection(
                        label="word", confidence=0.95,
                        bbox=(0.1, 0.1, 0.4, 0.15),
                        metadata={"text": "Hello"},
                    ),
                    Detection(
                        label="word", confidence=0.9,
                        bbox=(0.1, 0.2, 0.5, 0.25),
                        metadata={"text": "world"},
                    ),
                ]

        node = create_raw_node(
            raw_type=RawDataType.IMAGE, content=b"img", account_id="acct",
        )
        op = OCRExtractor(backend=_TextBackend())
        result = op.extract(node, ctx)
        assert "Hello" in result.extracted_content["text"]
        assert "world" in result.extracted_content["text"]
        assert result.extracted_content["line_count"] == 2
        assert result.confidence > 0


# ---------------------------------------------------------------------------
# DocumentExtractor
# ---------------------------------------------------------------------------

class TestDocumentExtractor:
    def test_markdown_headings(self, document_raw_node, ctx):
        op = DocumentExtractor(min_section_length=10)
        result = op.extract(document_raw_node, ctx)
        assert result.error is None
        assert result.extracted_content["section_count"] >= 3
        titles = [s["title"] for s in result.extracted_content["sections"]]
        assert "Introduction" in titles
        assert "Methods" in titles
        assert "RESULTS" in titles

    def test_empty_document(self, ctx):
        node = create_raw_node(
            raw_type=RawDataType.DOCUMENT, content="", account_id="acct",
        )
        op = DocumentExtractor()
        result = op.extract(node, ctx)
        assert result.extracted_content["section_count"] == 0
        assert result.confidence == 1.0  # empty is "successfully empty"

    def test_preamble_only(self, ctx):
        node = create_raw_node(
            raw_type=RawDataType.DOCUMENT,
            content="Just some text without any heading. " * 5,
            account_id="acct",
        )
        op = DocumentExtractor()
        result = op.extract(node, ctx)
        assert result.extracted_content["section_count"] == 1
        assert result.extracted_content["sections"][0]["title"] == "(preamble)"

    def test_min_section_length_drops_short_bodies(self, ctx):
        node = create_raw_node(
            raw_type=RawDataType.DOCUMENT,
            content="Title:\nshort\n\nBody:\n" + ("x" * 50),
            account_id="acct",
        )
        op = DocumentExtractor(min_section_length=30)
        result = op.extract(node, ctx)
        # "short" body is dropped, but the "Title:" section is still
        # recorded because the section record is never dropped when it
        # has a non-preamble title.
        titles = [s["title"] for s in result.extracted_content["sections"]]
        assert "Title" in titles
        assert "Body" in titles

    def test_max_sections(self, ctx):
        text = "\n\n".join(f"H{i}\n" + ("body " * 10) for i in range(20))
        node = create_raw_node(
            raw_type=RawDataType.DOCUMENT, content=text, account_id="acct",
        )
        op = DocumentExtractor(max_sections=5, min_section_length=10)
        result = op.extract(node, ctx)
        assert result.extracted_content["section_count"] <= 5


# ---------------------------------------------------------------------------
# register_all_defaults
# ---------------------------------------------------------------------------

class TestRegisterAllDefaults:
    def test_registers_built_ins(self):
        from extraction.operators import OperatorRegistry
        from extraction.operators.builtin import register_all_defaults
        reg = OperatorRegistry()
        n = register_all_defaults(reg)
        assert n >= 6
        assert reg.has("prompt_analyzer")
        assert reg.has("image_caption")
        assert reg.has("object_detection")
        assert reg.has("face_detection")
        assert reg.has("pose_detection")
        assert reg.has("ocr")
        assert reg.has("document_structure")
