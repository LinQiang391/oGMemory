"""Unit tests for CV backends."""

from __future__ import annotations

import json

import pytest

from extraction.operators.backends import (
    CVBackend,
    CVBackendError,
    Detection,
    HTTPBackend,
    MockBackend,
)
from extraction.operators.backends.base import require_image_bytes


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class TestMockBackend:
    def test_name(self):
        assert MockBackend().name == "mock"

    def test_health(self):
        assert MockBackend().health_check() is True

    def test_empty_image_returns_empty(self):
        b = MockBackend()
        assert b.detect_objects(b"") == []
        assert b.detect_faces(b"") == []
        assert b.detect_pose(b"") == []

    def test_objects_deterministic(self):
        b = MockBackend(object_labels=("cat", "dog"))
        out1 = b.detect_objects(b"image-1")
        out2 = b.detect_objects(b"image-1")
        assert out1 == out2
        assert all(isinstance(d, Detection) for d in out1)
        assert all(0.0 <= d.confidence <= 1.0 for d in out1)
        # Bbox is normalised
        for d in out1:
            assert 0.0 <= d.bbox[0] < d.bbox[2] <= 1.0
            assert 0.0 <= d.bbox[1] < d.bbox[3] <= 1.0

    def test_objects_filter_by_threshold(self):
        b = MockBackend(default_confidence=0.7)
        out = b.detect_objects(b"abc", threshold=0.9)
        # All detections have confidence 0.7 < 0.9 → filtered out
        assert out == []

    def test_objects_respect_max_items(self):
        b = MockBackend(object_labels=("a", "b", "c", "d"))
        out = b.detect_objects(b"abc", max_items=2)
        assert len(out) <= 2

    def test_faces_count(self):
        b = MockBackend(face_count=3)
        out = b.detect_faces(b"image-1")
        assert len(out) == 3
        assert all(d.label == "face" for d in out)

    def test_pose_returns_keypoints(self):
        b = MockBackend(person_count=1)
        out = b.detect_pose(b"img")
        assert len(out) == 1
        kps = out[0].metadata["keypoints"]
        assert len(kps) >= 10
        assert all({"label", "x", "y", "confidence"} <= set(kp.keys()) for kp in kps)

    def test_forced_failure(self):
        b = MockBackend(fail_on_image_prefix="FAIL:")
        with pytest.raises(CVBackendError, match="Forced failure"):
            b.detect_objects(b"FAIL:this is a fake image")
        with pytest.raises(CVBackendError, match="Forced failure"):
            b.detect_faces(b"FAIL:x")
        with pytest.raises(CVBackendError, match="Forced failure"):
            b.detect_pose(b"FAIL:x")

    def test_set_healthy(self):
        b = MockBackend()
        b.set_healthy(False)
        assert b.health_check() is False
        b.set_healthy(True)
        assert b.health_check() is True

    def test_conforms_to_protocol(self):
        assert isinstance(MockBackend(), CVBackend)


# ---------------------------------------------------------------------------
# require_image_bytes helper
# ---------------------------------------------------------------------------

class TestRequireImageBytes:
    def test_passes_bytes(self):
        assert require_image_bytes(b"\x00\x01") == b"\x00\x01"

    def test_encodes_string(self):
        assert require_image_bytes("hello") == b"hello"

    def test_none_returns_empty(self):
        assert require_image_bytes(None) == b""

    def test_unsupported_type_raises(self):
        with pytest.raises(CVBackendError, match="Unsupported image type"):
            require_image_bytes(12345)


# ---------------------------------------------------------------------------
# HTTP backend
# ---------------------------------------------------------------------------

class _StubResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


class _StubSession:
    """Mimics ``requests.Session`` enough for HTTPBackend._post()."""

    def __init__(self, responses):
        # responses: dict path -> _StubResponse or list of them
        self.responses = responses
        self.calls = []

    def post(self, url, data, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "params": params, "headers": headers})
        # url looks like "http://x/path" — match by suffix
        for path_suffix, resp in self.responses.items():
            if url.endswith(path_suffix):
                return resp
        return _StubResponse({"error": "no stub"}, status=404)


class TestHTTPBackend:
    def _factory(self, session):
        return lambda: session

    def test_health(self):
        b = HTTPBackend(base_url="http://cv.local", session_factory=self._factory(_StubSession({})))
        assert b.health_check() is True

    def test_constructor_requires_base_url(self):
        with pytest.raises(CVBackendError, match="base_url"):
            HTTPBackend(base_url="")

    def test_empty_image_returns_empty(self):
        b = HTTPBackend(base_url="http://x", session_factory=self._factory(_StubSession({})))
        assert b.detect_objects(b"") == []

    def test_detect_objects_parses_bare_list(self):
        payload = [
            {"label": "person", "confidence": 0.9, "bbox": [0.1, 0.2, 0.3, 0.5]},
            {"label": "cat", "confidence": 0.8, "bbox": [0.4, 0.4, 0.6, 0.7]},
        ]
        session = _StubSession({"/detect/objects": _StubResponse(payload)})
        b = HTTPBackend(base_url="http://cv", session_factory=self._factory(session))
        out = b.detect_objects(b"img-bytes")
        assert len(out) == 2
        assert out[0].label == "person"
        assert out[0].bbox == (0.1, 0.2, 0.3, 0.5)
        # Verify request was made with image bytes and threshold param
        assert session.calls[0]["data"] == b"img-bytes"
        assert session.calls[0]["params"]["threshold"] == 0.0

    def test_detect_objects_parses_wrapped(self):
        payload = {"detections": [{"label": "x", "score": 0.7, "box": [0, 0, 0.5, 0.5]}]}
        session = _StubSession({"/detect/objects": _StubResponse(payload)})
        b = HTTPBackend(base_url="http://cv", session_factory=self._factory(session))
        out = b.detect_objects(b"img")
        assert len(out) == 1
        assert out[0].label == "x"
        assert out[0].confidence == 0.7
        assert out[0].bbox == (0, 0, 0.5, 0.5)

    def test_detect_faces(self):
        payload = [{"label": "face", "confidence": 0.95, "bbox": [0.1, 0.2, 0.3, 0.4]}]
        session = _StubSession({"/detect/faces": _StubResponse(payload)})
        b = HTTPBackend(base_url="http://cv", session_factory=self._factory(session))
        out = b.detect_faces(b"img")
        assert len(out) == 1
        assert out[0].label == "face"

    def test_detect_pose(self):
        payload = [{"label": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1], "keypoints": []}]
        session = _StubSession({"/detect/pose": _StubResponse(payload)})
        b = HTTPBackend(base_url="http://cv", session_factory=self._factory(session))
        out = b.detect_pose(b"img")
        assert len(out) == 1
        assert out[0].metadata["keypoints"] == []

    def test_auth_token_in_header(self):
        session = _StubSession({"/detect/objects": _StubResponse([])})
        b = HTTPBackend(
            base_url="http://cv",
            auth_token="secret",
            session_factory=self._factory(session),
        )
        b.detect_objects(b"img")
        assert session.calls[0]["headers"]["Authorization"] == "Bearer secret"

    def test_request_failure_raises(self):
        session = _StubSession({"/detect/objects": _StubResponse({}, status=500)})
        b = HTTPBackend(base_url="http://cv", session_factory=self._factory(session))
        with pytest.raises(CVBackendError, match="request to .* failed"):
            b.detect_objects(b"img")

    def test_malformed_response_raises(self):
        session = _StubSession({"/detect/objects": _StubResponse("not a dict or list")})
        b = HTTPBackend(base_url="http://cv", session_factory=self._factory(session))
        with pytest.raises(CVBackendError, match="parse .* response"):
            b.detect_objects(b"img")

    def test_custom_path(self):
        session = _StubSession({"/custom/obj": _StubResponse([])})
        b = HTTPBackend(
            base_url="http://cv",
            object_path="/custom/obj",
            session_factory=self._factory(session),
        )
        out = b.detect_objects(b"img")
        assert out == []
        assert session.calls[0]["url"] == "http://cv/custom/obj"
